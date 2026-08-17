"""Most Recent Changepoint для независимого поиска текущего GMV-режима.

Модуль получает готовую полную панель ``segment_id x cal_date`` и не читает
файлы, не строит календарь и не участвует в anomaly score. Для одного активного
ряда он один раз оценивает общий scale, кэширует стоимости всех допустимых
независимых линейных сегментов, выполняет exact dynamic programming по этой
матрице стоимостей и строит полный профиль последнего changepoint ``G(tau)``.

Outer DP точен для рассчитанных segment costs. Для невыпуклого capped cost
используется детерминированная multi-start оптимизация; глобальный оптимум самой
bounded regression этим не заявляется.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .trend_analysis import (
    NO_DIRECTION,
    TrendModelConfig,
    TrendThresholds,
    _coerce_finite_values,
    _effectively_zero,
    _ordered_frame,
    _segment_metadata,
    evaluate_trend,
    trim_leading_zero_history,
)


DIFF_MAD = "DIFF_MAD"
OLS_RESIDUAL_MAD = "OLS_RESIDUAL_MAD"
MAE_FALLBACK = "MAE_FALLBACK"
PERFECT_FIT = "PERFECT_FIT"


@dataclass(frozen=True)
class SigmaEstimate:
    """Хранить единый scale активного временного ряда.

    Args:
        sigma: Оценка масштаба в исходных единицах GMV.
        source: Источник оценки или ``PERFECT_FIT``.

    Returns:
        Неизменяемую диагностику scale.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> estimate_series_sigma([100, 110, 120, 130]).source
        'PERFECT_FIT'
    """

    sigma: float
    source: str


@dataclass(frozen=True)
class SegmentCostResult:
    """Хранить стоимость и fitted line одного half-open сегмента.

    Args:
        start: Первый индекс сегмента, включительно.
        end: Последний индекс сегмента, не включительно.
        points: Число наблюдений.
        intercept: Свободный член в глобальной временной координате.
        slope: Наклон на один период.
        rss: Обычная сумма квадратов residuals выбранной линии.
        cost: Стандартизованная стоимость выбранного типа.
        cost_type: ``ols``, ``capped`` или ``huber``.
        optimizer_status: Диагностика оценивания коэффициентов.

    Returns:
        Неизменяемую запись кэша segment costs.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> result = calculate_segment_cost([1, 2, 3, 4], 0, 4, 1.0, 'ols')
        >>> result.points
        4
    """

    start: int
    end: int
    points: int
    intercept: float
    slope: float
    rss: float
    cost: float
    cost_type: str
    optimizer_status: str


@dataclass(frozen=True)
class MostRecentCPAnalysis:
    """Объединить summary и три QA-диагностики одного сегмента.

    Args:
        summary: Итоговая строка бизнес- и математического результата.
        cp_profile: Полный профиль допустимых ``G(tau)``.
        segment_diagnostics: Все рассчитанные и закэшированные ``C(s,e)``.
        segmentation: Сегменты выбранного оптимального решения.

    Returns:
        Неизменяемый контейнер результата одного сегмента.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> analyze_most_recent_cp_series([1, 2, 3, 4], [1, 2, 3, 4]).summary['last_cp_index'] is None
        True
    """

    summary: Dict[str, object]
    cp_profile: Tuple[Dict[str, object], ...]
    segment_diagnostics: Tuple[Dict[str, object], ...]
    segmentation: Tuple[Dict[str, object], ...]


def _objective_tolerance(left: float, right: float) -> float:
    """Рассчитать scale-aware machine tolerance для двух objective.

    Args:
        left: Первое конечное значение objective.
        right: Второе конечное значение objective.

    Returns:
        Абсолютный допуск только уровня арифметики float.

    Raises:
        ValueError: Если передано нечисловое значение.

    Examples:
        >>> _objective_tolerance(1.0, 1.0) > 0.0
        True
    """

    return float(
        256.0 * np.finfo(float).eps * max(
            1.0,
            abs(float(left)),
            abs(float(right)),
        )
    )


def _objectives_tied(left: float, right: float) -> bool:
    """Проверить численную ничью objective без бизнес-порога.

    Args:
        left: Первое значение.
        right: Второе значение.

    Returns:
        True только для конечных значений в machine tolerance.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _objectives_tied(1.0, 1.0 + np.finfo(float).eps)
        True
    """

    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return False
    return bool(
        abs(float(left) - float(right))
        <= _objective_tolerance(left, right)
    )


def _strictly_better(candidate: float, incumbent: float) -> bool:
    """Проверить материальное численное улучшение objective.

    Args:
        candidate: Новое значение.
        incumbent: Текущее лучшее значение.

    Returns:
        True, если candidate меньше вне machine tolerance.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _strictly_better(1.0, 2.0)
        True
    """

    if not math.isfinite(float(candidate)):
        return False
    if not math.isfinite(float(incumbent)):
        return True
    return bool(
        float(candidate) < float(incumbent) - _objective_tolerance(
            candidate,
            incumbent,
        )
    )


def _fit_ols_line(
    time: np.ndarray,
    values: np.ndarray,
) -> Tuple[float, float, np.ndarray, float]:
    """Оценить независимую OLS-линию в глобальной временной координате.

    Args:
        time: Глобальные позиции наблюдений сегмента.
        values: Значения GMV сегмента.

    Returns:
        ``intercept, slope, residuals, rss``.

    Raises:
        ValueError: Если длины не совпадают или точек меньше двух.

    Examples:
        >>> fit = _fit_ols_line(np.arange(4.0), np.array([1., 2., 3., 4.]))
        >>> round(fit[1], 12)
        1.0
    """

    if time.ndim != 1 or values.ndim != 1 or len(time) != len(values):
        raise ValueError("time и values должны быть одномерными массивами одной длины")
    if len(values) < 2:
        raise ValueError("Для линейной модели нужно минимум две точки")
    time_center = float(np.mean(time))
    value_center = float(np.mean(values))
    centered_time = time - time_center
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        raise ValueError("Временная координата должна содержать разные позиции")
    slope = float(np.dot(centered_time, values - value_center) / denominator)
    intercept = float(value_center - slope * time_center)
    residuals = values - (intercept + slope * time)
    rss = float(np.dot(residuals, residuals))
    return intercept, slope, residuals, rss


def estimate_series_sigma(values: Sequence[float]) -> SigmaEstimate:
    """Оценить единый scale по заданной fallback-цепочке.

    Args:
        values: Активная история после удаления ведущих нулей.

    Returns:
        Sigma и источник ``DIFF_MAD``, ``OLS_RESIDUAL_MAD``,
        ``MAE_FALLBACK`` или ``PERFECT_FIT``.

    Raises:
        ValueError: Если точек меньше двух или значения не конечны.

    Examples:
        >>> estimate_series_sigma([100, 110, 120, 130]).sigma
        0.0
    """

    array = _coerce_finite_values(values)
    if len(array) < 2:
        raise ValueError("Для оценки sigma нужно минимум две точки")

    differences = np.diff(array)
    difference_median = float(np.median(differences))
    difference_mad = float(
        np.median(np.abs(differences - difference_median))
    )
    sigma = difference_mad / (0.67448975 * math.sqrt(2.0))
    if not _effectively_zero(np.asarray([sigma], dtype=float), array):
        return SigmaEstimate(float(sigma), DIFF_MAD)

    time = np.arange(len(array), dtype=float)
    _, _, residuals, _ = _fit_ols_line(time, array)
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    sigma = 1.4826 * residual_mad
    if not _effectively_zero(np.asarray([sigma], dtype=float), array):
        return SigmaEstimate(float(sigma), OLS_RESIDUAL_MAD)

    if not _effectively_zero(residuals, array):
        sigma = float(np.mean(np.abs(residuals)))
        if sigma > 0.0:
            return SigmaEstimate(sigma, MAE_FALLBACK)

    return SigmaEstimate(0.0, PERFECT_FIT)


def _standardized_problem(
    time: np.ndarray,
    values: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Центрировать segment regression и стандартизовать GMV.

    Args:
        time: Глобальная временная координата сегмента.
        values: GMV сегмента.
        sigma: Общий положительный scale ряда.

    Returns:
        ``design, response, time_center, value_center``.

    Raises:
        ValueError: Если sigma неположительна.

    Examples:
        >>> problem = _standardized_problem(np.arange(4.0), np.arange(4.0), 1.0)
        >>> problem[0].shape
        (4, 2)
    """

    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma должна быть конечной и положительной")
    time_center = float(np.mean(time))
    value_center = float(np.median(values))
    centered_time = time - time_center
    response = (values - value_center) / float(sigma)
    design = np.column_stack(
        (np.ones(len(values), dtype=float), centered_time)
    )
    return design, response, time_center, value_center


def _restore_original_parameters(
    standardized_parameters: np.ndarray,
    sigma: float,
    time_center: float,
    value_center: float,
) -> Tuple[float, float]:
    """Вернуть intercept/slope из центрированной стандартизованной модели.

    Args:
        standardized_parameters: ``alpha, gamma`` в standardized problem.
        sigma: Общий scale.
        time_center: Центр глобальной временной координаты.
        value_center: Центр GMV.

    Returns:
        Intercept и slope в исходных единицах GMV.

    Raises:
        ValueError: Если передано не два коэффициента.

    Examples:
        >>> _restore_original_parameters(np.array([0., 1.]), 2., 1.5, 10.)
        (7.0, 2.0)
    """

    parameters = np.asarray(standardized_parameters, dtype=float)
    if parameters.shape != (2,):
        raise ValueError("Ожидались два стандартизованных коэффициента")
    slope = float(parameters[1] * sigma)
    intercept = float(
        value_center + parameters[0] * sigma - slope * time_center
    )
    return intercept, slope


def _deterministic_regression_starts(
    design: np.ndarray,
    response: np.ndarray,
) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Построить четыре детерминированных старта robust regression.

    Args:
        design: Матрица ``[1, centered_time]``.
        response: Центрированный стандартизованный GMV.

    Returns:
        OLS, Theil–Sen, endpoints и median-level старты.

    Raises:
        ValueError: Если точек меньше двух.

    Examples:
        >>> len(_deterministic_regression_starts(np.column_stack((np.ones(4), np.arange(4.))), np.arange(4.)))
        4
    """

    if len(response) < 2:
        raise ValueError("Для regression starts нужно минимум две точки")
    ols, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    x = design[:, 1]
    slopes = np.asarray(
        [
            (float(response[j]) - float(response[i]))
            / (float(x[j]) - float(x[i]))
            for i in range(len(response) - 1)
            for j in range(i + 1, len(response))
        ],
        dtype=float,
    )
    theil_slope = float(np.median(slopes))
    theil_intercept = float(np.median(response - theil_slope * x))
    endpoint_slope = float(
        (response[-1] - response[0]) / (x[-1] - x[0])
    )
    endpoint_intercept = float(response[0] - endpoint_slope * x[0])
    median_level = float(np.median(response))
    return (
        ("OLS", np.asarray(ols, dtype=float)),
        ("THEIL_SEN", np.asarray([theil_intercept, theil_slope], dtype=float)),
        ("ENDPOINTS", np.asarray([endpoint_intercept, endpoint_slope], dtype=float)),
        ("MEDIAN_LEVEL", np.asarray([median_level, 0.0], dtype=float)),
    )


def _capped_objective(
    parameters: Sequence[float],
    design: np.ndarray,
    response: np.ndarray,
    capped_k: float,
) -> float:
    """Рассчитать bounded quadratic objective для заданной линии.

    Args:
        parameters: Два standardized коэффициента.
        design: Матрица standardized regression.
        response: Standardized GMV.
        capped_k: Положительный порог K.

    Returns:
        ``sum(min(u^2, K^2))`` без предварительного OLS-fit.

    Raises:
        ValueError: Если параметры имеют неверную форму.

    Examples:
        >>> _capped_objective([0., 0.], np.ones((2, 2)), np.array([0., 3.]), 2.)
        4.0
    """

    vector = np.asarray(parameters, dtype=float)
    if vector.shape != (2,):
        raise ValueError("Capped objective ожидает два коэффициента")
    residuals = response - design @ vector
    bounded = np.minimum(np.abs(residuals), float(capped_k))
    return float(np.dot(bounded, bounded))


def _fit_capped_model(
    time: np.ndarray,
    values: np.ndarray,
    sigma: float,
    capped_k: float,
) -> Tuple[float, float, np.ndarray, float, str]:
    """Минимизировать C2 детерминированным multi-start active-set методом.

    Args:
        time: Глобальные позиции сегмента.
        values: GMV сегмента.
        sigma: Общий положительный scale.
        capped_k: Порог K.

    Returns:
        ``intercept, slope, residuals, objective, optimizer_status``.

    Raises:
        ValueError: Если optimizer не вернул ни одного конечного кандидата.

    Examples:
        >>> fit = _fit_capped_model(np.arange(4.0), np.array([1., 2., 3., 4.]), 1., 2.)
        >>> round(fit[3], 12)
        0.0
    """

    design, response, time_center, value_center = _standardized_problem(
        time,
        values,
        sigma,
    )
    candidates: List[Tuple[float, int, int, str, np.ndarray]] = []
    for start_index, (start_name, start) in enumerate(
        _deterministic_regression_starts(design, response)
    ):
        parameters = np.asarray(start, dtype=float)
        visited_active_sets: set[Tuple[bool, ...]] = set()
        for iteration in range(101):
            objective = _capped_objective(
                parameters,
                design,
                response,
                capped_k,
            )
            if np.isfinite(parameters).all() and math.isfinite(objective):
                candidates.append(
                    (
                        objective,
                        start_index,
                        iteration,
                        start_name,
                        parameters.copy(),
                    )
                )
            residuals = response - design @ parameters
            active_mask = np.abs(residuals) <= float(capped_k)
            active_key = tuple(bool(value) for value in active_mask)
            if active_key in visited_active_sets:
                break
            visited_active_sets.add(active_key)
            active_design = design[active_mask]
            active_response = response[active_mask]
            if (
                len(active_response) < 2
                or np.linalg.matrix_rank(active_design) < 2
            ):
                break
            updated, _, _, _ = np.linalg.lstsq(
                active_design,
                active_response,
                rcond=None,
            )
            updated = np.asarray(updated, dtype=float)
            parameter_tolerance = 256.0 * np.finfo(float).eps * max(
                1.0,
                float(np.linalg.norm(parameters)),
                float(np.linalg.norm(updated)),
            )
            if float(np.linalg.norm(updated - parameters)) <= parameter_tolerance:
                parameters = updated
                final_objective = _capped_objective(
                    parameters,
                    design,
                    response,
                    capped_k,
                )
                candidates.append(
                    (
                        final_objective,
                        start_index,
                        iteration + 1,
                        start_name,
                        parameters.copy(),
                    )
                )
                break
            parameters = updated
    if not candidates:
        raise ValueError("Capped optimizer не вернул конечного кандидата")
    best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    intercept, slope = _restore_original_parameters(
        best[4],
        sigma,
        time_center,
        value_center,
    )
    residuals = values - (intercept + slope * time)
    status = (
        "DETERMINISTIC_ACTIVE_SET_MULTI_START:"
        f"best_start={best[3]}:iterations={best[2]}:starts=4"
    )
    return intercept, slope, residuals, float(best[0]), status


def _huber_rho(
    standardized_residuals: Sequence[float],
    delta: float,
) -> np.ndarray:
    """Рассчитать Huber loss с квадратичной частью ``u^2``.

    Args:
        standardized_residuals: Значения u.
        delta: Положительный порог Huber.

    Returns:
        Поэлементные ``u^2`` либо ``2*delta*|u|-delta^2``.

    Raises:
        ValueError: Если delta неположительна.

    Examples:
        >>> _huber_rho([0.5, 1.0, 2.0], 1.0).tolist()
        [0.25, 1.0, 3.0]
    """

    if not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise ValueError("delta должна быть конечной и положительной")
    residuals = np.asarray(standardized_residuals, dtype=float)
    absolute = np.abs(residuals)
    return np.where(
        absolute <= float(delta),
        residuals * residuals,
        2.0 * float(delta) * absolute - float(delta) ** 2,
    )


def _fit_huber_model(
    time: np.ndarray,
    values: np.ndarray,
    sigma: float,
    delta: float,
) -> Tuple[float, float, np.ndarray, float, str]:
    """Оценить выпуклую C3-модель детерминированным Huber IRLS.

    Args:
        time: Глобальные позиции сегмента.
        values: GMV сегмента.
        sigma: Общий положительный scale.
        delta: Huber threshold.

    Returns:
        ``intercept, slope, residuals, objective, optimizer_status``.

    Raises:
        ValueError: Если входная задача вырождена.

    Examples:
        >>> fit = _fit_huber_model(np.arange(4.0), np.array([1., 2., 3., 4.]), 1., 1.345)
        >>> round(fit[3], 12)
        0.0
    """

    design, response, time_center, value_center = _standardized_problem(
        time,
        values,
        sigma,
    )
    parameters, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    parameters = np.asarray(parameters, dtype=float)
    best_parameters = parameters.copy()
    best_objective = float(
        np.sum(_huber_rho(response - design @ parameters, delta))
    )
    converged = False
    iterations = 0
    for iterations in range(1, 501):
        standardized_residuals = response - design @ parameters
        absolute = np.abs(standardized_residuals)
        weights = np.ones_like(absolute)
        outer = absolute > float(delta)
        weights[outer] = float(delta) / absolute[outer]
        root_weights = np.sqrt(weights)
        weighted_design = design * root_weights[:, None]
        weighted_response = response * root_weights
        updated, _, _, _ = np.linalg.lstsq(
            weighted_design,
            weighted_response,
            rcond=None,
        )
        updated = np.asarray(updated, dtype=float)
        objective = float(
            np.sum(_huber_rho(response - design @ updated, delta))
        )
        if _strictly_better(objective, best_objective):
            best_objective = objective
            best_parameters = updated.copy()
        parameter_tolerance = 256.0 * np.finfo(float).eps * max(
            1.0,
            float(np.linalg.norm(parameters)),
            float(np.linalg.norm(updated)),
        )
        if float(np.linalg.norm(updated - parameters)) <= parameter_tolerance:
            parameters = updated
            converged = True
            if objective <= best_objective + _objective_tolerance(
                objective,
                best_objective,
            ):
                best_parameters = updated.copy()
                best_objective = objective
            break
        parameters = updated

    intercept, slope = _restore_original_parameters(
        best_parameters,
        sigma,
        time_center,
        value_center,
    )
    residuals = values - (intercept + slope * time)
    status = f"HUBER_IRLS:converged={converged}:iterations={iterations}"
    return intercept, slope, residuals, best_objective, status


def calculate_segment_cost(
    values: Sequence[float],
    start: int,
    end: int,
    sigma: float,
    cost_type: str,
    *,
    capped_k: float = 2.0,
    huber_delta: float = 1.345,
) -> SegmentCostResult:
    """Рассчитать один ``C(s,e)`` по единому dispatcher стоимости.

    Args:
        values: Полная активная история.
        start: Первый индекс сегмента, включительно.
        end: Последний индекс сегмента, не включительно.
        sigma: Единый scale полной активной истории.
        cost_type: ``ols``, ``capped`` или ``huber``.
        capped_k: Порог K для C2.
        huber_delta: Порог delta для C3.

    Returns:
        Стоимость, коэффициенты и optimizer diagnostics сегмента.

    Raises:
        ValueError: Если границы, cost или параметры невалидны.
        ValueError: Если cost optimization не возвращает конечный результат.

    Examples:
        >>> calculate_segment_cost([1, 2, 3, 4], 0, 4, 1.0, 'ols').cost < 1e-20
        True
    """

    array = _coerce_finite_values(values)
    if cost_type not in {"ols", "capped", "huber"}:
        raise ValueError("cost_type должен быть одним из: capped, huber, ols")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, np.integer))
        or not isinstance(end, (int, np.integer))
        or start < 0
        or end > len(array)
        or end <= start
    ):
        raise ValueError("Некорректные half-open границы сегмента")
    if int(end) - int(start) < 4:
        raise ValueError("Каждый Most Recent CP сегмент должен содержать минимум 4 точки")
    if not math.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("sigma должна быть конечной и неотрицательной")
    if not math.isfinite(float(capped_k)) or float(capped_k) <= 0.0:
        raise ValueError("capped_k должен быть конечным положительным числом")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
        raise ValueError("huber_delta должен быть конечным положительным числом")

    segment_values = array[int(start) : int(end)]
    time = np.arange(int(start), int(end), dtype=float)
    ols_intercept, ols_slope, ols_residuals, ols_rss = _fit_ols_line(
        time,
        segment_values,
    )
    if float(sigma) == 0.0:
        if _effectively_zero(ols_residuals, segment_values):
            return SegmentCostResult(
                start=int(start),
                end=int(end),
                points=int(end) - int(start),
                intercept=ols_intercept,
                slope=ols_slope,
                rss=ols_rss,
                cost=0.0,
                cost_type=cost_type,
                optimizer_status="PERFECT_FIT_ZERO_COST",
            )
        return SegmentCostResult(
            start=int(start),
            end=int(end),
            points=int(end) - int(start),
            intercept=ols_intercept,
            slope=ols_slope,
            rss=ols_rss,
            cost=math.inf,
            cost_type=cost_type,
            optimizer_status="ZERO_SIGMA_NONPERFECT_INFINITE_COST",
        )

    if cost_type == "ols":
        standardized = ols_residuals / float(sigma)
        cost = float(np.dot(standardized, standardized))
        intercept = ols_intercept
        slope = ols_slope
        residuals = ols_residuals
        optimizer_status = "CLOSED_FORM_OLS"
    elif cost_type == "capped":
        intercept, slope, residuals, cost, optimizer_status = _fit_capped_model(
            time,
            segment_values,
            float(sigma),
            float(capped_k),
        )
    else:
        intercept, slope, residuals, cost, optimizer_status = _fit_huber_model(
            time,
            segment_values,
            float(sigma),
            float(huber_delta),
        )
    rss = float(np.dot(residuals, residuals))
    if not math.isfinite(float(cost)):
        raise ValueError("Segment cost должен быть конечным при положительной sigma")
    return SegmentCostResult(
        start=int(start),
        end=int(end),
        points=int(end) - int(start),
        intercept=float(intercept),
        slope=float(slope),
        rss=rss,
        cost=float(cost),
        cost_type=cost_type,
        optimizer_status=optimizer_status,
    )


def _build_segment_cost_cache(
    values: np.ndarray,
    sigma: float,
    model_config: TrendModelConfig,
) -> Dict[Tuple[int, int], SegmentCostResult]:
    """Один раз рассчитать все допустимые ``C(s,e)``.

    Args:
        values: Активная история.
        sigma: Единый scale ряда.
        model_config: Параметры cost и минимальной длины.

    Returns:
        Словарь по half-open ключам ``(s, e)``.

    Raises:
        ValueError: Если segment cost не может быть рассчитан.

    Examples:
        >>> cfg = TrendModelConfig(trend_search_method='most_recent_cp')
        >>> len(_build_segment_cost_cache(np.arange(8.0), 1.0, cfg))
        15
    """

    minimum = int(model_config.most_recent_cp_min_segment_points)
    cache: Dict[Tuple[int, int], SegmentCostResult] = {}
    for start in range(0, len(values) - minimum + 1):
        for end in range(start + minimum, len(values) + 1):
            cache[(start, end)] = calculate_segment_cost(
                values,
                start,
                end,
                sigma,
                model_config.most_recent_cp_cost,
                capped_k=model_config.most_recent_cp_capped_k,
                huber_delta=model_config.most_recent_cp_huber_delta,
            )
    return cache


def _build_prefix_dp(
    points: int,
    minimum: int,
    beta: float,
    cache: Dict[Tuple[int, int], SegmentCostResult],
) -> Tuple[List[float], List[Optional[int]]]:
    """Выполнить exact DP prefix segmentation по рассчитанным costs.

    Args:
        points: Длина активной истории.
        minimum: Минимальная длина любого сегмента.
        beta: Penalty за каждый changepoint.
        cache: Полная матрица допустимых segment costs.

    Returns:
        Массивы ``F(t)`` и predecessor для ``t=0..n``.

    Raises:
        ValueError: Если в кэше отсутствует обязательный сегмент.

    Examples:
        >>> cfg = TrendModelConfig(trend_search_method='most_recent_cp')
        >>> cache = _build_segment_cost_cache(np.arange(8.0), 1.0, cfg)
        >>> len(_build_prefix_dp(8, 4, 3 * math.log(8), cache)[0])
        9
    """

    prefix_cost = [math.inf] * (points + 1)
    predecessor: List[Optional[int]] = [None] * (points + 1)
    prefix_cost[0] = 0.0
    predecessor[0] = 0
    for end in range(minimum, points + 1):
        if (0, end) not in cache:
            raise ValueError(f"В segment cost cache отсутствует (0, {end})")
        best_cost = float(cache[(0, end)].cost)
        best_start = 0
        for start in range(minimum, end - minimum + 1):
            if not math.isfinite(prefix_cost[start]):
                continue
            segment = cache.get((start, end))
            if segment is None or not math.isfinite(segment.cost):
                continue
            candidate = prefix_cost[start] + segment.cost + float(beta)
            if _strictly_better(candidate, best_cost) or (
                _objectives_tied(candidate, best_cost) and start > best_start
            ):
                best_cost = float(candidate)
                best_start = int(start)
        prefix_cost[end] = best_cost
        predecessor[end] = best_start
    return prefix_cost, predecessor


def _restore_prefix_segmentation(
    end: int,
    predecessor: Sequence[Optional[int]],
) -> List[Tuple[int, int]]:
    """Восстановить оптимальные сегменты prefix ``values[:end]``.

    Args:
        end: Правая half-open граница prefix.
        predecessor: Массив predecessor из DP.

    Returns:
        Упорядоченные ``(start, end)`` сегменты.

    Raises:
        ValueError: Если predecessor не позволяет восстановление.

    Examples:
        >>> _restore_prefix_segmentation(4, [0, None, None, None, 0])
        [(0, 4)]
    """

    segments: List[Tuple[int, int]] = []
    current_end = int(end)
    while current_end > 0:
        if current_end >= len(predecessor) or predecessor[current_end] is None:
            raise ValueError("Не удалось восстановить DP segmentation")
        current_start = int(predecessor[current_end])
        if current_start < 0 or current_start >= current_end:
            raise ValueError("DP predecessor содержит некорректную границу")
        segments.append((current_start, current_end))
        current_end = current_start
    return list(reversed(segments))


def _select_most_recent_profile_candidate(
    profile_candidates: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], bool, float]:
    """Выбрать минимальный G и более поздний tau только при machine tie.

    Args:
        profile_candidates: Строки с числовыми полями ``tau`` и ``G_tau``.

    Returns:
        Выбранную строку, флаг численной ничьей и точный минимальный G.

    Raises:
        ValueError: Если профиль пуст или не содержит конечного objective.

    Examples:
        >>> eps = np.finfo(float).eps
        >>> _select_most_recent_profile_candidate([{'tau': 4, 'G_tau': 1.0}, {'tau': 5, 'G_tau': 1.0 + eps}])[0]['tau']
        5
    """

    if not profile_candidates:
        raise ValueError("Most Recent CP profile не должен быть пустым")
    finite_objectives = [
        float(candidate["G_tau"])
        for candidate in profile_candidates
        if math.isfinite(float(candidate["G_tau"]))
    ]
    if not finite_objectives:
        raise ValueError("Most Recent CP profile не содержит конечных objective")
    exact_best = min(finite_objectives)
    tied_candidates = [
        candidate
        for candidate in profile_candidates
        if _objectives_tied(float(candidate["G_tau"]), exact_best)
        or float(candidate["G_tau"]) == exact_best
    ]
    selected = max(
        tied_candidates,
        key=lambda candidate: int(candidate["tau"]),
    )
    return dict(selected), len(tied_candidates) > 1, exact_best


def _empty_summary(
    raw_points: int,
    used_points: int,
    status: str,
    model_config: TrendModelConfig,
) -> Dict[str, object]:
    """Создать summary для отсутствующей или слишком короткой истории.

    Args:
        raw_points: Длина календарной истории.
        used_points: Длина после trim ведущих нулей.
        status: ``NO_ACTIVE_HISTORY`` или ``INSUFFICIENT_HISTORY``.
        model_config: Конфигурация метода.

    Returns:
        Полную пустую строку Most Recent CP.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _empty_summary(3, 3, 'INSUFFICIENT_HISTORY', TrendModelConfig(trend_search_method='most_recent_cp'))['current_trend_direction']
        'NONE'
    """

    return {
        "status": status,
        "history_points_raw": int(raw_points),
        "history_points_used": int(used_points),
        "leading_zero_points_trimmed": int(raw_points - used_points),
        "trend_search_method": model_config.trend_search_method,
        "cost_type": model_config.most_recent_cp_cost,
        "min_segment_points": model_config.most_recent_cp_min_segment_points,
        "capped_k": model_config.most_recent_cp_capped_k,
        "huber_delta": model_config.most_recent_cp_huber_delta,
        "sigma": math.nan,
        "sigma_source": "NOT_EVALUATED",
        "beta": math.nan,
        "structural_change_detected": False,
        "last_cp_index": None,
        "last_cp_left_end_date": None,
        "last_cp_right_start_date": None,
        "current_regime_length": int(used_points),
        "current_regime_start_date": None,
        "current_regime_end_date": None,
        "objective_no_change": math.nan,
        "objective_selected": math.nan,
        "objective_improvement": math.nan,
        "numeric_tie_break_used": False,
        "optimal_breakpoints_json": "[]",
        "current_trend_exists": False,
        "current_trend_direction": NO_DIRECTION,
        "current_trend_length": 0,
        "current_trend_start_date": None,
        "current_trend_end_date": None,
        "trend_slope_abs": math.nan,
        "trend_slope_relative": math.nan,
        "trend_total_change": math.nan,
        "direction_count_share": math.nan,
        "direction_movement_share": math.nan,
        "trend_to_noise": math.nan,
        "current_regime_trend_status": "NOT_EVALUATED",
        "previous_regime_slope": math.nan,
        "current_regime_slope": math.nan,
        "delta_slope": math.nan,
        "level_shift": math.nan,
    }


def analyze_most_recent_cp_series(
    values: Sequence[float],
    dates: Optional[Sequence[object]] = None,
    thresholds: Optional[TrendThresholds] = None,
    model_config: Optional[TrendModelConfig] = None,
) -> MostRecentCPAnalysis:
    """Выполнить Most Recent CP для одного полного временного ряда.

    Args:
        values: Полная история GMV до trim ведущих нулей.
        dates: Синхронные даты; по умолчанию позиции ``0..N-1``.
        thresholds: Существующие бизнес-пороги ``evaluate_trend``.
        model_config: Параметры Most Recent CP.

    Returns:
        Summary, полный профиль, cache diagnostics и выбранную segmentation.

    Raises:
        ValueError: Если вход или конфигурация невалидны.
        ImportError: Если для C2 недоступен scipy.

    Examples:
        >>> result = analyze_most_recent_cp_series([100, 101, 100, 102, 300, 302, 301, 303])
        >>> result.summary['last_cp_index']
        4
    """

    thresholds = thresholds or TrendThresholds()
    model_config = model_config or TrendModelConfig(
        trend_search_method="most_recent_cp"
    )
    if model_config.trend_search_method != "most_recent_cp":
        raise ValueError(
            "analyze_most_recent_cp_series требует trend_search_method='most_recent_cp'"
        )
    raw_values = _coerce_finite_values(values)
    raw_dates = list(range(len(raw_values))) if dates is None else list(dates)
    if len(raw_dates) != len(raw_values):
        raise ValueError("Количество дат должно совпадать с количеством GMV-точек")
    used_values_list, used_dates = trim_leading_zero_history(
        raw_values,
        raw_dates,
    )
    raw_points = len(raw_values)
    used_points = len(used_values_list)
    minimum = int(model_config.most_recent_cp_min_segment_points)
    if used_points == 0:
        return MostRecentCPAnalysis(
            summary=_empty_summary(
                raw_points,
                0,
                "NO_ACTIVE_HISTORY",
                model_config,
            ),
            cp_profile=tuple(),
            segment_diagnostics=tuple(),
            segmentation=tuple(),
        )
    if used_points < minimum:
        return MostRecentCPAnalysis(
            summary=_empty_summary(
                raw_points,
                used_points,
                "INSUFFICIENT_HISTORY",
                model_config,
            ),
            cp_profile=tuple(),
            segment_diagnostics=tuple(),
            segmentation=tuple(),
        )

    used_values = np.asarray(used_values_list, dtype=float)
    sigma_estimate = estimate_series_sigma(used_values)
    beta = 3.0 * math.log(float(used_points)) * 1.5
    cache = _build_segment_cost_cache(
        used_values,
        sigma_estimate.sigma,
        model_config,
    )
    prefix_cost, predecessor = _build_prefix_dp(
        used_points,
        minimum,
        beta,
        cache,
    )

    profile_candidates: List[Dict[str, object]] = [
        {
            "tau": 0,
            "prefix_optimal_cost": 0.0,
            "current_segment_cost": cache[(0, used_points)].cost,
            "G_tau": cache[(0, used_points)].cost,
        }
    ]
    for tau in range(minimum, used_points - minimum + 1):
        current_cost = cache[(tau, used_points)].cost
        profile_candidates.append(
            {
                "tau": int(tau),
                "prefix_optimal_cost": float(prefix_cost[tau]),
                "current_segment_cost": float(current_cost),
                "G_tau": float(prefix_cost[tau] + current_cost + beta),
            }
        )
    (
        selected_candidate,
        numeric_tie_break_used,
        exact_best,
    ) = _select_most_recent_profile_candidate(
        profile_candidates,
    )
    tau_star = int(selected_candidate["tau"])
    if tau_star == 0:
        selected_segments = [(0, used_points)]
    else:
        selected_segments = _restore_prefix_segmentation(
            tau_star,
            predecessor,
        )
        selected_segments.append((tau_star, used_points))
    selected_segment_set = set(selected_segments)
    breakpoints = [start for start, _ in selected_segments[1:]]

    no_change_objective = float(cache[(0, used_points)].cost)
    selected_objective = float(selected_candidate["G_tau"])
    current_start = tau_star
    current_cost = cache[(current_start, used_points)]
    current_evaluation = evaluate_trend(
        used_values[current_start:],
        thresholds,
    )
    current_exists = bool(current_evaluation.trend_exists)
    previous_slope = math.nan
    delta_slope = math.nan
    level_shift = math.nan
    if tau_star > 0:
        previous_start, previous_end = selected_segments[-2]
        previous_cost = cache[(previous_start, previous_end)]
        previous_slope = previous_cost.slope
        delta_slope = current_cost.slope - previous_cost.slope
        level_shift = (
            current_cost.intercept + current_cost.slope * float(tau_star)
        ) - (
            previous_cost.intercept + previous_cost.slope * float(tau_star)
        )

    summary = {
        "status": "TREND" if current_exists else "NO_TREND",
        "history_points_raw": raw_points,
        "history_points_used": used_points,
        "leading_zero_points_trimmed": raw_points - used_points,
        "trend_search_method": model_config.trend_search_method,
        "cost_type": model_config.most_recent_cp_cost,
        "min_segment_points": minimum,
        "capped_k": model_config.most_recent_cp_capped_k,
        "huber_delta": model_config.most_recent_cp_huber_delta,
        "sigma": sigma_estimate.sigma,
        "sigma_source": sigma_estimate.source,
        "beta": beta,
        "structural_change_detected": tau_star > 0,
        "last_cp_index": tau_star if tau_star > 0 else None,
        "last_cp_left_end_date": (
            used_dates[tau_star - 1] if tau_star > 0 else None
        ),
        "last_cp_right_start_date": (
            used_dates[tau_star] if tau_star > 0 else None
        ),
        "current_regime_length": used_points - tau_star,
        "current_regime_start_date": used_dates[tau_star],
        "current_regime_end_date": used_dates[-1],
        "objective_no_change": no_change_objective,
        "objective_selected": selected_objective,
        "objective_improvement": no_change_objective - selected_objective,
        "numeric_tie_break_used": numeric_tie_break_used,
        "optimal_breakpoints_json": json.dumps(breakpoints),
        "current_trend_exists": current_exists,
        "current_trend_direction": (
            current_evaluation.direction if current_exists else NO_DIRECTION
        ),
        "current_trend_length": (
            used_points - tau_star if current_exists else 0
        ),
        "current_trend_start_date": (
            used_dates[tau_star] if current_exists else None
        ),
        "current_trend_end_date": used_dates[-1] if current_exists else None,
        "trend_slope_abs": current_evaluation.slope if current_exists else math.nan,
        "trend_slope_relative": (
            current_evaluation.relative_slope if current_exists else math.nan
        ),
        "trend_total_change": (
            current_evaluation.total_change if current_exists else math.nan
        ),
        "direction_count_share": (
            current_evaluation.direction_count_share if current_exists else math.nan
        ),
        "direction_movement_share": (
            current_evaluation.direction_movement_share if current_exists else math.nan
        ),
        "trend_to_noise": (
            current_evaluation.trend_to_noise if current_exists else math.nan
        ),
        "current_regime_trend_status": current_evaluation.status,
        "previous_regime_slope": previous_slope,
        "current_regime_slope": current_cost.slope,
        "delta_slope": delta_slope,
        "level_shift": level_shift,
    }

    profile_rows: List[Dict[str, object]] = []
    for candidate_id, candidate in enumerate(profile_candidates):
        tau = int(candidate["tau"])
        objective = float(candidate["G_tau"])
        profile_rows.append(
            {
                "profile_candidate_id": candidate_id,
                "tau": tau,
                "left_end_date": used_dates[tau - 1] if tau > 0 else None,
                "right_start_date": used_dates[tau] if tau > 0 else used_dates[0],
                "right_points": used_points - tau,
                "prefix_optimal_cost": float(candidate["prefix_optimal_cost"]),
                "current_segment_cost": float(candidate["current_segment_cost"]),
                "beta": beta,
                "G_tau": objective,
                "delta_vs_no_change": no_change_objective - objective,
                "selected": tau == tau_star,
                "numerical_tie_with_best": (
                    _objectives_tied(objective, exact_best)
                    or objective == exact_best
                ),
                "cost_type": model_config.most_recent_cp_cost,
                "sigma": sigma_estimate.sigma,
                "sigma_source": sigma_estimate.source,
            }
        )

    segment_order = {
        segment: index for index, segment in enumerate(selected_segments)
    }
    segment_rows: List[Dict[str, object]] = []
    for segment in sorted(cache):
        cost_result = cache[segment]
        row = asdict(cost_result)
        row.update(
            {
                "start_date": used_dates[cost_result.start],
                "end_date": used_dates[cost_result.end - 1],
                "selected_segment": segment in selected_segment_set,
                "selected_segment_order": segment_order.get(segment),
                "sigma": sigma_estimate.sigma,
                "sigma_source": sigma_estimate.source,
            }
        )
        segment_rows.append(row)

    segmentation_rows: List[Dict[str, object]] = []
    cumulative_objective = 0.0
    for segment_index, segment in enumerate(selected_segments):
        cost_result = cache[segment]
        cumulative_objective += cost_result.cost
        if segment_index > 0:
            cumulative_objective += beta
        segmentation_rows.append(
            {
                "segment_index": segment_index,
                **asdict(cost_result),
                "start_date": used_dates[cost_result.start],
                "end_date": used_dates[cost_result.end - 1],
                "changepoint_penalty": beta if segment_index > 0 else 0.0,
                "cumulative_objective": cumulative_objective,
            }
        )

    return MostRecentCPAnalysis(
        summary=summary,
        cp_profile=tuple(profile_rows),
        segment_diagnostics=tuple(segment_rows),
        segmentation=tuple(segmentation_rows),
    )


def analyze_segment_most_recent_cp(
    segment_panel: pd.DataFrame,
    dates: Sequence[int],
    thresholds: Optional[TrendThresholds] = None,
    model_config: Optional[TrendModelConfig] = None,
) -> MostRecentCPAnalysis:
    """Применить Most Recent CP к готовой панели одного сегмента.

    Args:
        segment_panel: Полная панель ровно одного ``segment_id``.
        dates: Полная total-календарная ось.
        thresholds: Существующие пороги подтверждения тренда.
        model_config: Параметры Most Recent CP.

    Returns:
        Результат с добавленной metadata сегмента во все строки.

    Raises:
        ValueError: Если панель не соответствует контракту полной сетки.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [1, 2, 3, 4]})
        >>> analyze_segment_most_recent_cp(panel, [1, 2, 3, 4]).summary['segment_id']
        's'
    """

    model_config = model_config or TrendModelConfig(
        trend_search_method="most_recent_cp"
    )
    required = {"segment_id", "cal_date", "gmv"}
    missing = sorted(required - set(segment_panel.columns))
    if missing:
        raise ValueError(f"Для trend analysis не хватает колонок: {missing}")
    normalized_dates = [int(date) for date in dates]
    if not normalized_dates:
        raise ValueError("dates не должен быть пустым")
    if (
        len(set(normalized_dates)) != len(normalized_dates)
        or normalized_dates != sorted(normalized_dates)
    ):
        raise ValueError("dates должен быть строго возрастающей уникальной осью")
    metadata = _segment_metadata(segment_panel)
    if segment_panel.duplicated(subset=["cal_date"]).any():
        raise ValueError("Панель сегмента содержит дубли cal_date")
    indexed = segment_panel.copy()
    indexed["cal_date"] = pd.to_numeric(indexed["cal_date"], errors="coerce")
    if indexed["cal_date"].isna().any():
        raise ValueError("cal_date должен быть числовым")
    indexed["cal_date"] = indexed["cal_date"].astype(int)
    if set(indexed["cal_date"].tolist()) != set(normalized_dates):
        raise ValueError(
            "segment_panel должен содержать ровно одну строку на каждую дату "
            "total-календаря"
        )
    ordered = indexed.set_index("cal_date").reindex(normalized_dates)
    raw_values = _coerce_finite_values(ordered["gmv"].tolist())
    result = analyze_most_recent_cp_series(
        raw_values,
        normalized_dates,
        thresholds,
        model_config,
    )
    return MostRecentCPAnalysis(
        summary={**metadata, **result.summary},
        cp_profile=tuple({**metadata, **row} for row in result.cp_profile),
        segment_diagnostics=tuple(
            {**metadata, **row} for row in result.segment_diagnostics
        ),
        segmentation=tuple(
            {**metadata, **row} for row in result.segmentation
        ),
    )


MOST_RECENT_CP_SUMMARY_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "status",
    "history_points_raw",
    "history_points_used",
    "leading_zero_points_trimmed",
    "trend_search_method",
    "cost_type",
    "min_segment_points",
    "capped_k",
    "huber_delta",
    "sigma",
    "sigma_source",
    "beta",
    "structural_change_detected",
    "last_cp_index",
    "last_cp_left_end_date",
    "last_cp_right_start_date",
    "current_regime_length",
    "current_regime_start_date",
    "current_regime_end_date",
    "objective_no_change",
    "objective_selected",
    "objective_improvement",
    "numeric_tie_break_used",
    "optimal_breakpoints_json",
    "current_trend_exists",
    "current_trend_direction",
    "current_trend_length",
    "current_trend_start_date",
    "current_trend_end_date",
    "trend_slope_abs",
    "trend_slope_relative",
    "trend_total_change",
    "direction_count_share",
    "direction_movement_share",
    "trend_to_noise",
    "current_regime_trend_status",
    "previous_regime_slope",
    "current_regime_slope",
    "delta_slope",
    "level_shift",
)

MOST_RECENT_CP_PROFILE_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "profile_candidate_id",
    "tau",
    "left_end_date",
    "right_start_date",
    "right_points",
    "prefix_optimal_cost",
    "current_segment_cost",
    "beta",
    "G_tau",
    "delta_vs_no_change",
    "selected",
    "numerical_tie_with_best",
    "cost_type",
    "sigma",
    "sigma_source",
)

MOST_RECENT_CP_SEGMENT_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "start",
    "end",
    "start_date",
    "end_date",
    "points",
    "intercept",
    "slope",
    "rss",
    "cost",
    "cost_type",
    "optimizer_status",
    "selected_segment",
    "selected_segment_order",
    "sigma",
    "sigma_source",
)

MOST_RECENT_CP_SEGMENTATION_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "segment_index",
    "start",
    "end",
    "start_date",
    "end_date",
    "points",
    "intercept",
    "slope",
    "rss",
    "cost",
    "cost_type",
    "optimizer_status",
    "changepoint_penalty",
    "cumulative_objective",
)


def build_most_recent_cp_trend_analysis(
    panel_df: pd.DataFrame,
    dates: Sequence[int],
    thresholds: Optional[TrendThresholds] = None,
    model_config: Optional[TrendModelConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Построить Most Recent CP для всех сегментов полной панели.

    Args:
        panel_df: Готовая полная панель из ``build_full_week_grid``.
        dates: Полная total-календарная ось.
        thresholds: Неизменённые пороги ``evaluate_trend``.
        model_config: Параметры нового метода.

    Returns:
        ``trend_summary``, ``trend_cp_profile``,
        ``trend_segment_diagnostics`` и ``trend_segmentation``.

    Raises:
        ValueError: Если панель пуста или конфигурация не Most Recent CP.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [1, 2, 3, 4]})
        >>> sorted(build_most_recent_cp_trend_analysis(panel, [1, 2, 3, 4]))
        ['trend_cp_profile', 'trend_segment_diagnostics', 'trend_segmentation', 'trend_summary']
    """

    thresholds = thresholds or TrendThresholds()
    model_config = model_config or TrendModelConfig(
        trend_search_method="most_recent_cp"
    )
    if model_config.trend_search_method != "most_recent_cp":
        raise ValueError(
            "build_most_recent_cp_trend_analysis требует "
            "trend_search_method='most_recent_cp'"
        )
    if panel_df.empty:
        raise ValueError("Нельзя анализировать пустую panel_df")
    required = {"segment_id", "cal_date", "gmv"}
    missing = sorted(required - set(panel_df.columns))
    if missing:
        raise ValueError(f"Для trend analysis не хватает колонок: {missing}")

    summaries: List[Dict[str, object]] = []
    profile_rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    segmentation_rows: List[Dict[str, object]] = []
    for _, segment_panel in panel_df.groupby(
        "segment_id",
        sort=True,
        dropna=False,
    ):
        result = analyze_segment_most_recent_cp(
            segment_panel,
            dates,
            thresholds,
            model_config,
        )
        summaries.append(result.summary)
        profile_rows.extend(result.cp_profile)
        segment_rows.extend(result.segment_diagnostics)
        segmentation_rows.extend(result.segmentation)
    return {
        "trend_summary": _ordered_frame(
            summaries,
            MOST_RECENT_CP_SUMMARY_COLUMNS,
        ),
        "trend_cp_profile": _ordered_frame(
            profile_rows,
            MOST_RECENT_CP_PROFILE_COLUMNS,
        ),
        "trend_segment_diagnostics": _ordered_frame(
            segment_rows,
            MOST_RECENT_CP_SEGMENT_COLUMNS,
        ),
        "trend_segmentation": _ordered_frame(
            segmentation_rows,
            MOST_RECENT_CP_SEGMENTATION_COLUMNS,
        ),
    }


__all__ = [
    "MAE_FALLBACK",
    "OLS_RESIDUAL_MAD",
    "PERFECT_FIT",
    "DIFF_MAD",
    "MostRecentCPAnalysis",
    "SegmentCostResult",
    "SigmaEstimate",
    "analyze_most_recent_cp_series",
    "analyze_segment_most_recent_cp",
    "build_most_recent_cp_trend_analysis",
    "calculate_segment_cost",
    "estimate_series_sigma",
]
