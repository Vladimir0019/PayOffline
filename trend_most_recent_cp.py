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
    DECLINE,
    GROWTH,
    NO_DIRECTION,
    TrendEvaluation,
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

LEVEL_SHIFT = "LEVEL_SHIFT"
SLOPE_CHANGE = "SLOPE_CHANGE"
LEVEL_AND_SLOPE = "LEVEL_AND_SLOPE"
WEAK_OR_UNCLASSIFIED = "WEAK_OR_UNCLASSIFIED"
NO_STRUCTURAL_CHANGE = "NONE"

LOCAL_FLAT = "FLAT"
LOCAL_UNCONFIRMED = "UNCONFIRMED"
GLOBAL_DIRECTIONAL = "DIRECTIONAL"
GLOBAL_FLAT_BRIDGE = "FLAT_BRIDGE"
GLOBAL_NOT_INCLUDED = "NOT_INCLUDED"


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
class OLSRegimeDiagnostic:
    """Хранить OLS fit и геометрию выбранного half-open режима.

    Fit использует глобальную временную координату активного ряда и служит
    только post-classification последнего уже выбранного changepoint.

    Args:
        start: Первый глобальный индекс режима, включительно.
        end: Последний глобальный индекс режима, не включительно.
        points: Число наблюдений режима.
        time_mean: Средняя глобальная временная координата режима.
        sxx: Сумма квадратов отклонений времени от ``time_mean``.
        intercept: OLS intercept в глобальной временной координате.
        slope: OLS slope в единицах GMV за период.

    Returns:
        Неизменяемую OLS-диагностику выбранного режима.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> diagnostic = _fit_ols_regime_diagnostic(np.arange(8.0), 0, 4)
        >>> diagnostic.sxx
        5.0
    """

    start: int
    end: int
    points: int
    time_mean: float
    sxx: float
    intercept: float
    slope: float


@dataclass(frozen=True)
class MostRecentCPAnalysis:
    """Объединить summary и четыре QA-диагностики одного сегмента.

    Args:
        summary: Итоговая строка бизнес- и математического результата.
        cp_profile: Полный профиль допустимых ``G(tau)``.
        segment_diagnostics: Все рассчитанные и закэшированные ``C(s,e)``.
        segmentation: Сегменты выбранного оптимального решения.
        changepoints: Диагностика всех границ выбранной сегментации.

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
    changepoints: Tuple[Dict[str, object], ...]


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


def _fit_ols_regime_diagnostic(
    values: Sequence[float],
    start: int,
    end: int,
) -> OLSRegimeDiagnostic:
    """Отдельно оценить OLS-линию и геометрию выбранного режима.

    Args:
        values: Полная активная GMV-история.
        start: Первый глобальный индекс режима, включительно.
        end: Последний глобальный индекс режима, не включительно.

    Returns:
        OLS fit и ``Sxx`` на глобальной оси ``t=0,...,n-1``.

    Raises:
        ValueError: Если half-open границы или временная геометрия некорректны.

    Examples:
        >>> fit = _fit_ols_regime_diagnostic([1, 2, 3, 4, 5], 1, 5)
        >>> (fit.time_mean, fit.sxx, fit.slope)
        (2.5, 5.0, 1.0)
    """

    array = _coerce_finite_values(values)
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, np.integer))
        or not isinstance(end, (int, np.integer))
        or int(start) < 0
        or int(end) > len(array)
        or int(end) - int(start) < 2
    ):
        raise ValueError("Некорректные half-open границы OLS diagnostic режима")
    time = np.arange(int(start), int(end), dtype=float)
    segment_values = array[int(start) : int(end)]
    time_mean = float(np.mean(time))
    centered_time = time - time_mean
    sxx = float(np.dot(centered_time, centered_time))
    if not math.isfinite(sxx) or sxx <= 0.0:
        raise ValueError("Некорректная временная геометрия OLS diagnostic режима")
    intercept, slope, _, _ = _fit_ols_line(time, segment_values)
    return OLSRegimeDiagnostic(
        start=int(start),
        end=int(end),
        points=int(end) - int(start),
        time_mean=time_mean,
        sxx=sxx,
        intercept=intercept,
        slope=slope,
    )


def _standardized_effect_score(
    effect: float,
    standard_error: float,
    reference_values: Sequence[float],
) -> float:
    """Рассчитать устойчивый standardized effect score ``|effect| / SE``.

    Args:
        effect: Signed OLS diagnostic effect.
        standard_error: Неотрицательная standard error на общей sigma ряда.
        reference_values: Активный ряд для scale-aware проверки машинного нуля.

    Returns:
        Конечный неотрицательный score либо ``+inf`` при ненулевом effect и
        машинно нулевой SE.

    Raises:
        ValueError: Если effect, SE или reference некорректны.

    Examples:
        >>> _standardized_effect_score(2.0, 1.0, [1.0, 2.0])
        2.0
    """

    reference = _coerce_finite_values(reference_values)
    if not math.isfinite(float(effect)):
        raise ValueError("Diagnostic effect должен быть конечным")
    if (
        not math.isfinite(float(standard_error))
        or float(standard_error) < 0.0
    ):
        raise ValueError("Diagnostic standard error должна быть конечной и неотрицательной")
    se_is_zero = _effectively_zero(
        np.asarray([float(standard_error)], dtype=float),
        reference,
    )
    if se_is_zero:
        effect_is_zero = _effectively_zero(
            np.asarray([float(effect)], dtype=float),
            reference,
        )
        return 0.0 if effect_is_zero else math.inf
    return abs(float(effect)) / float(standard_error)


def _calculate_slope_change_statistics(
    previous: OLSRegimeDiagnostic,
    current: OLSRegimeDiagnostic,
    sigma: float,
    reference_values: Sequence[float],
) -> Tuple[float, float, float]:
    """Рассчитать OLS diagnostic slope effect, SE и standardized score.

    Args:
        previous: OLS diagnostic предыдущего выбранного режима.
        current: OLS diagnostic текущего выбранного режима.
        sigma: Существующая общая sigma активного ряда.
        reference_values: Активный ряд для обработки машинного нуля.

    Returns:
        ``delta_slope_ols, slope_change_se, slope_change_z``.

    Raises:
        ValueError: Если sigma или геометрия режимов некорректны.

    Examples:
        >>> left = _fit_ols_regime_diagnostic(np.arange(8.0), 0, 4)
        >>> right = _fit_ols_regime_diagnostic(np.arange(8.0), 4, 8)
        >>> round(_calculate_slope_change_statistics(left, right, 2.0, np.arange(8.0))[1], 12)
        1.264911064067
    """

    if not math.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("Diagnostic sigma должна быть конечной и неотрицательной")
    if previous.sxx <= 0.0 or current.sxx <= 0.0:
        raise ValueError("Sxx OLS diagnostic режима должен быть положительным")
    delta_slope = float(current.slope - previous.slope)
    standard_error = float(
        float(sigma)
        * math.sqrt((1.0 / previous.sxx) + (1.0 / current.sxx))
    )
    score = _standardized_effect_score(
        delta_slope,
        standard_error,
        reference_values,
    )
    return delta_slope, standard_error, score


def _calculate_level_shift_statistics(
    previous: OLSRegimeDiagnostic,
    current: OLSRegimeDiagnostic,
    tau: int,
    sigma: float,
    reference_values: Sequence[float],
) -> Tuple[float, float, float]:
    """Рассчитать fitted-mean jump на ``t=tau``, его SE и score.

    Args:
        previous: OLS diagnostic режима ``[s,tau)``.
        current: OLS diagnostic режима ``[tau,n)``.
        tau: Глобальная граница и первый индекс текущего режима.
        sigma: Существующая общая sigma активного ряда.
        reference_values: Активный ряд для обработки машинного нуля.

    Returns:
        ``level_shift_ols, level_shift_se, level_shift_z`` без prediction
        variance нового наблюдения.

    Raises:
        ValueError: Если граница, sigma или геометрия режимов некорректны.

    Examples:
        >>> left = _fit_ols_regime_diagnostic(np.arange(8.0), 0, 4)
        >>> right = _fit_ols_regime_diagnostic(np.arange(8.0), 4, 8)
        >>> round(_calculate_level_shift_statistics(left, right, 4, 2.0, np.arange(8.0))[1], 12)
        2.966479394838
    """

    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, np.integer))
        or previous.end != int(tau)
        or current.start != int(tau)
    ):
        raise ValueError("OLS diagnostic режимы должны стыковаться в глобальном t=tau")
    if not math.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("Diagnostic sigma должна быть конечной и неотрицательной")
    if previous.sxx <= 0.0 or current.sxx <= 0.0:
        raise ValueError("Sxx OLS diagnostic режима должен быть положительным")
    boundary = float(tau)
    fitted_previous = previous.intercept + previous.slope * boundary
    fitted_current = current.intercept + current.slope * boundary
    level_shift = float(fitted_current - fitted_previous)
    previous_leverage = (
        1.0 / float(previous.points)
        + (boundary - previous.time_mean) ** 2 / previous.sxx
    )
    current_leverage = (
        1.0 / float(current.points)
        + (boundary - current.time_mean) ** 2 / current.sxx
    )
    leverage_sum = float(previous_leverage + current_leverage)
    if not math.isfinite(leverage_sum) or leverage_sum <= 0.0:
        raise ValueError("Leverage OLS diagnostic границы должен быть положительным")
    standard_error = float(float(sigma) * math.sqrt(leverage_sum))
    score = _standardized_effect_score(
        level_shift,
        standard_error,
        reference_values,
    )
    return level_shift, standard_error, score


def _classify_structural_change(
    structural_change_detected: bool,
    level_shift_z: float,
    slope_change_z: float,
    threshold: float,
) -> str:
    """Классифицировать ортогональные level/slope свойства последнего CP.

    Args:
        structural_change_detected: Был ли выбран ``tau > 0``.
        level_shift_z: Standardized OLS fitted-level effect.
        slope_change_z: Standardized OLS slope effect.
        threshold: Единый положительный диагностический порог.

    Returns:
        ``NONE``, ``LEVEL_SHIFT``, ``SLOPE_CHANGE``, ``LEVEL_AND_SLOPE`` или
        ``WEAK_OR_UNCLASSIFIED``. Равенство порогу считается подтверждением.

    Raises:
        ValueError: Если для найденного CP scores или threshold некорректны.

    Examples:
        >>> _classify_structural_change(True, 2.0, 1.0, 2.0)
        'LEVEL_SHIFT'
    """

    if not structural_change_detected:
        return NO_STRUCTURAL_CHANGE
    if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError("Diagnostic change threshold должен быть конечным и положительным")
    for name, value in (
        ("level_shift_z", level_shift_z),
        ("slope_change_z", slope_change_z),
    ):
        if math.isnan(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} должен быть неотрицательным diagnostic score")
    has_level_change = bool(float(level_shift_z) >= float(threshold))
    has_slope_change = bool(float(slope_change_z) >= float(threshold))
    if has_level_change and has_slope_change:
        return LEVEL_AND_SLOPE
    if has_level_change:
        return LEVEL_SHIFT
    if has_slope_change:
        return SLOPE_CHANGE
    return WEAK_OR_UNCLASSIFIED


def _classify_direction_change(
    structural_change_detected: bool,
    previous_trend: Optional[TrendEvaluation],
    current_trend: TrendEvaluation,
) -> bool:
    """Определить смену бизнес-направления независимо от structural type.

    Args:
        structural_change_detected: Был ли выбран structural CP.
        previous_trend: Результат ``evaluate_trend`` предыдущего режима.
        current_trend: Результат ``evaluate_trend`` текущего режима.

    Returns:
        True только для подтверждённых ``GROWTH <-> DECLINE``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _classify_direction_change(False, None, evaluate_trend([1, 2, 3, 4]))
        False
    """

    allowed_directions = {GROWTH, DECLINE}
    return bool(
        structural_change_detected
        and previous_trend is not None
        and previous_trend.trend_exists
        and current_trend.trend_exists
        and previous_trend.direction in allowed_directions
        and current_trend.direction in allowed_directions
        and previous_trend.direction != current_trend.direction
    )


def _local_noise_ratio(evaluation: TrendEvaluation) -> float:
    """Нормировать noise scale локального режима на его типичный GMV.

    Args:
        evaluation: Каноническая диагностика ``evaluate_trend``.

    Returns:
        Конечное неотрицательное отношение либо ``+inf``, если scale режима
        не позволяет безопасную нормировку.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _local_noise_ratio(evaluate_trend([100, 100, 100, 100]))
        0.0
    """

    scale = float(evaluation.typical_scale)
    noise = float(evaluation.noise_scale)
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(noise)
        or noise < 0.0
    ):
        return math.inf
    return float(noise / scale)


def _classify_local_regime(
    evaluation: TrendEvaluation,
    model_config: TrendModelConfig,
) -> str:
    """Классифицировать выбранный режим как direction, FLAT или barrier.

    Args:
        evaluation: Результат существующего ``evaluate_trend``.
        model_config: Пороговый контракт глобальной надстройки.

    Returns:
        ``GROWTH``, ``DECLINE``, ``FLAT`` либо ``UNCONFIRMED``.

    Raises:
        ValueError: Не выбрасывается для валидной конфигурации.

    Examples:
        >>> _classify_local_regime(
        ...     evaluate_trend([100, 100, 100, 100]), TrendModelConfig()
        ... )
        'FLAT'
    """

    if evaluation.trend_exists and evaluation.direction in {GROWTH, DECLINE}:
        return evaluation.direction
    metrics = (
        float(evaluation.relative_slope),
        float(evaluation.total_change),
        _local_noise_ratio(evaluation),
    )
    if not all(math.isfinite(metric) for metric in metrics):
        return LOCAL_UNCONFIRMED
    relative_slope, total_change, noise_ratio = metrics
    if (
        abs(relative_slope)
        <= float(model_config.global_flat_max_relative_slope)
        and abs(total_change)
        <= float(model_config.global_flat_max_total_change)
        and noise_ratio
        <= float(model_config.global_flat_max_noise_ratio)
    ):
        return LOCAL_FLAT
    return LOCAL_UNCONFIRMED


def _evaluate_selected_regimes(
    values: np.ndarray,
    dates: Sequence[object],
    selected_segments: Sequence[Tuple[int, int]],
    thresholds: TrendThresholds,
    model_config: TrendModelConfig,
) -> Tuple[List[Dict[str, object]], List[TrendEvaluation]]:
    """Оценить каждый локальный режим выбранной DP-сегментации.

    Args:
        values: Активная GMV-история.
        dates: Синхронная активная календарная ось.
        selected_segments: Упорядоченные half-open режимы.
        thresholds: Неизменённый бизнес-контракт тренда.
        model_config: Параметры FLAT-классификации.

    Returns:
        Строки локальной диагностики и соответствующие ``TrendEvaluation``.

    Raises:
        ValueError: Если границы выбранного режима некорректны.

    Examples:
        >>> rows, _ = _evaluate_selected_regimes(
        ...     np.asarray([1, 2, 3, 4], dtype=float), [0, 1, 2, 3],
        ...     [(0, 4)], TrendThresholds(), TrendModelConfig()
        ... )
        >>> rows[0]['local_regime_class']
        'GROWTH'
    """

    rows: List[Dict[str, object]] = []
    evaluations: List[TrendEvaluation] = []
    for segment_index, (start, end) in enumerate(selected_segments):
        if start < 0 or end > len(values) or start >= end:
            raise ValueError("Некорректные границы выбранного локального режима")
        evaluation = evaluate_trend(values[start:end], thresholds)
        evaluations.append(evaluation)
        start_gmv = float(values[start])
        end_gmv = float(values[end - 1])
        gmv_change = float(end_gmv - start_gmv)
        gmv_change_relative = (
            float(gmv_change / abs(start_gmv))
            if not _effectively_zero(
                np.asarray([start_gmv], dtype=float),
                values[start:end],
            )
            else math.nan
        )
        local_class = _classify_local_regime(evaluation, model_config)
        rows.append(
            {
                "segment_index": int(segment_index),
                "start": int(start),
                "end": int(end),
                "start_date": dates[start],
                "end_date": dates[end - 1],
                "points": int(end - start),
                "local_start_gmv": start_gmv,
                "local_end_gmv": end_gmv,
                "local_gmv_change_abs": gmv_change,
                "local_gmv_change_relative": gmv_change_relative,
                "local_trend_exists": bool(evaluation.trend_exists),
                "local_trend_direction": (
                    evaluation.direction
                    if evaluation.trend_exists
                    else NO_DIRECTION
                ),
                "local_trend_status": evaluation.status,
                "local_trend_slope_abs": evaluation.slope,
                "local_trend_slope_relative": evaluation.relative_slope,
                "local_trend_total_change": evaluation.total_change,
                "local_direction_count_share": (
                    evaluation.direction_count_share
                ),
                "local_direction_movement_share": (
                    evaluation.direction_movement_share
                ),
                "local_trend_to_noise": evaluation.trend_to_noise,
                "local_typical_scale": evaluation.typical_scale,
                "local_noise_scale": evaluation.noise_scale,
                "local_noise_scale_source": evaluation.noise_scale_source,
                "local_noise_ratio": _local_noise_ratio(evaluation),
                "local_regime_class": local_class,
                "in_global_trend": False,
                "global_trend_regime_role": GLOBAL_NOT_INCLUDED,
            }
        )
    return rows, evaluations


def _select_global_regime_indices(
    values: np.ndarray,
    local_rows: Sequence[Dict[str, object]],
    local_evaluations: Sequence[TrendEvaluation],
    thresholds: TrendThresholds,
    model_config: TrendModelConfig,
) -> Tuple[Tuple[int, ...], Optional[TrendEvaluation]]:
    """Расширить последний тренд назад через совместимые режимы и один FLAT.

    Args:
        values: Полная активная GMV-история.
        local_rows: Диагностика выбранных режимов в хронологическом порядке.
        local_evaluations: Синхронные результаты ``evaluate_trend``.
        thresholds: Действующий бизнес-контракт тренда.
        model_config: Ограничение числа FLAT-мостов.

    Returns:
        Индексы режимов глобального тренда и диагностику всего объединённого
        окна. Пустой tuple означает, что последний режим не является трендом.

    Raises:
        ValueError: Если локальные таблицы не синхронны.

    Examples:
        >>> rows, evaluations = _evaluate_selected_regimes(
        ...     np.asarray([1,2,3,4,5,6,7,8], dtype=float), list(range(8)),
        ...     [(0,4),(4,8)], TrendThresholds(), TrendModelConfig()
        ... )
        >>> _select_global_regime_indices(
        ...     np.asarray([1,2,3,4,5,6,7,8], dtype=float), rows,
        ...     evaluations, TrendThresholds(), TrendModelConfig()
        ... )[0]
        (0, 1)
    """

    if len(local_rows) != len(local_evaluations):
        raise ValueError("Локальные режимы и evaluations должны быть синхронны")
    if not local_rows:
        return tuple(), None
    last_index = len(local_rows) - 1
    direction = str(local_rows[last_index]["local_regime_class"])
    if direction not in {GROWTH, DECLINE}:
        return tuple(), None

    committed_start = last_index
    global_evaluation = local_evaluations[last_index]
    pending_flat_count = 0
    maximum_flats = int(model_config.global_max_flat_bridge_regimes)
    global_end = int(local_rows[last_index]["end"])

    for regime_index in range(last_index - 1, -1, -1):
        local_class = str(local_rows[regime_index]["local_regime_class"])
        if local_class == LOCAL_FLAT:
            pending_flat_count += 1
            if pending_flat_count > maximum_flats:
                break
            continue
        if local_class != direction:
            break

        candidate_start = int(local_rows[regime_index]["start"])
        candidate_evaluation = evaluate_trend(
            values[candidate_start:global_end],
            thresholds,
        )
        if not (
            candidate_evaluation.trend_exists
            and candidate_evaluation.direction == direction
        ):
            break
        committed_start = regime_index
        global_evaluation = candidate_evaluation
        pending_flat_count = 0

    included = tuple(range(committed_start, last_index + 1))
    return included, global_evaluation


def _build_selected_changepoint_rows(
    values: np.ndarray,
    dates: Sequence[object],
    selected_segments: Sequence[Tuple[int, int]],
    cache: Dict[Tuple[int, int], SegmentCostResult],
    local_rows: Sequence[Dict[str, object]],
    local_evaluations: Sequence[TrendEvaluation],
    included_global_indices: Sequence[int],
    sigma_estimate: SigmaEstimate,
    beta: float,
    model_config: TrendModelConfig,
) -> List[Dict[str, object]]:
    """Построить post-classification каждой выбранной structural CP.

    Args:
        values: Активная GMV-история.
        dates: Активная календарная ось.
        selected_segments: Выбранные режимы DP.
        cache: Неизменённые segment costs.
        local_rows: Локальная классификация режимов.
        local_evaluations: Результаты ``evaluate_trend`` локальных режимов.
        included_global_indices: Индексы режимов глобального тренда.
        sigma_estimate: Единая sigma ряда.
        beta: Неизменённый penalty за CP.
        model_config: Порог structural post-classification.

    Returns:
        Одну строку на границу соседних выбранных режимов.

    Raises:
        ValueError: Если выбранные сегменты и диагностики не синхронны.

    Examples:
        >>> # Вызывается после восстановления selected_segments.
    """

    if not (
        len(selected_segments) == len(local_rows) == len(local_evaluations)
    ):
        raise ValueError("Выбранные режимы и локальная диагностика не синхронны")
    included = set(int(index) for index in included_global_indices)
    global_start = min(included) if included else None
    global_direction = (
        str(local_rows[max(included)]["local_regime_class"])
        if included
        else NO_DIRECTION
    )
    rows: List[Dict[str, object]] = []
    cp_count = max(0, len(selected_segments) - 1)
    threshold = float(model_config.most_recent_cp_change_z_threshold)
    for current_index in range(1, len(selected_segments)):
        previous_index = current_index - 1
        previous_start, previous_end = selected_segments[previous_index]
        current_start, current_end = selected_segments[current_index]
        if previous_end != current_start:
            raise ValueError("Соседние выбранные режимы должны стыковаться")
        tau = int(current_start)
        previous_cost = cache[(previous_start, previous_end)]
        current_cost = cache[(current_start, current_end)]
        delta_slope = float(current_cost.slope - previous_cost.slope)
        # [FIXED] Сохраняем прежний порядок float-операций последнего CP
        # побитово; global layer не должен менять существующую диагностику.
        level_shift = float(
            (
                current_cost.intercept
                + current_cost.slope * float(tau)
            )
            - (
                previous_cost.intercept
                + previous_cost.slope * float(tau)
            )
        )
        previous_diagnostic = _fit_ols_regime_diagnostic(
            values,
            previous_start,
            previous_end,
        )
        current_diagnostic = _fit_ols_regime_diagnostic(
            values,
            current_start,
            current_end,
        )
        (
            classification_delta_slope,
            slope_change_se,
            slope_change_z,
        ) = _calculate_slope_change_statistics(
            previous_diagnostic,
            current_diagnostic,
            sigma_estimate.sigma,
            values,
        )
        (
            classification_level_shift,
            level_shift_se,
            level_shift_z,
        ) = _calculate_level_shift_statistics(
            previous_diagnostic,
            current_diagnostic,
            tau,
            sigma_estimate.sigma,
            values,
        )
        structural_type = _classify_structural_change(
            True,
            level_shift_z,
            slope_change_z,
            threshold,
        )
        previous_evaluation = local_evaluations[previous_index]
        current_evaluation = local_evaluations[current_index]
        inside_global = previous_index in included and current_index in included
        start_cp = bool(
            global_start is not None
            and current_index == global_start
            and previous_index not in included
        )
        adverse_level_shift = bool(
            global_direction in {GROWTH, DECLINE}
            and level_shift_z >= threshold
            and (
                (global_direction == GROWTH and classification_level_shift < 0.0)
                or (
                    global_direction == DECLINE
                    and classification_level_shift > 0.0
                )
            )
        )
        rows.append(
            {
                "changepoint_order": int(current_index),
                "changepoint_count": int(cp_count),
                "is_last_changepoint": current_index == cp_count,
                "cp_index": tau,
                "left_end_date": dates[tau - 1],
                "right_start_date": dates[tau],
                "previous_regime_index": int(previous_index),
                "current_regime_index": int(current_index),
                "previous_regime_start": int(previous_start),
                "previous_regime_end": int(previous_end),
                "current_regime_start": int(current_start),
                "current_regime_end": int(current_end),
                "previous_regime_start_date": dates[previous_start],
                "previous_regime_end_date": dates[previous_end - 1],
                "current_regime_start_date": dates[current_start],
                "current_regime_end_date": dates[current_end - 1],
                "previous_regime_points": int(previous_end - previous_start),
                "current_regime_points": int(current_end - current_start),
                "previous_regime_class": local_rows[previous_index][
                    "local_regime_class"
                ],
                "current_regime_class": local_rows[current_index][
                    "local_regime_class"
                ],
                "connects_flat_regime": bool(
                    local_rows[previous_index]["local_regime_class"]
                    == LOCAL_FLAT
                    or local_rows[current_index]["local_regime_class"]
                    == LOCAL_FLAT
                ),
                "cp_inside_global_trend": inside_global,
                "global_trend_start_cp": start_cp,
                "global_trend_direction": global_direction,
                "adverse_significant_level_shift": adverse_level_shift,
                "cost_type": model_config.most_recent_cp_cost,
                "sigma": sigma_estimate.sigma,
                "sigma_source": sigma_estimate.source,
                "beta": float(beta),
                "change_z_threshold": threshold,
                "previous_regime_slope": previous_cost.slope,
                "current_regime_slope": current_cost.slope,
                "delta_slope": delta_slope,
                "level_shift": level_shift,
                "structural_change_type": structural_type,
                "classification_previous_slope_ols": (
                    previous_diagnostic.slope
                ),
                "classification_current_slope_ols": current_diagnostic.slope,
                "classification_delta_slope_ols": (
                    classification_delta_slope
                ),
                "classification_level_shift_ols": (
                    classification_level_shift
                ),
                "slope_change_se": slope_change_se,
                "level_shift_se": level_shift_se,
                "slope_change_z": slope_change_z,
                "level_shift_z": level_shift_z,
                "previous_regime_trend_exists": bool(
                    previous_evaluation.trend_exists
                ),
                "previous_regime_trend_direction": (
                    previous_evaluation.direction
                    if previous_evaluation.trend_exists
                    else NO_DIRECTION
                ),
                "previous_regime_trend_status": previous_evaluation.status,
                "current_regime_trend_exists": bool(
                    current_evaluation.trend_exists
                ),
                "current_regime_trend_direction": (
                    current_evaluation.direction
                    if current_evaluation.trend_exists
                    else NO_DIRECTION
                ),
                "current_regime_trend_status": current_evaluation.status,
                "direction_change": _classify_direction_change(
                    True,
                    previous_evaluation,
                    current_evaluation,
                ),
            }
        )
    return rows


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
        "change_z_threshold": model_config.most_recent_cp_change_z_threshold,
        "global_flat_max_relative_slope": (
            model_config.global_flat_max_relative_slope
        ),
        "global_flat_max_total_change": (
            model_config.global_flat_max_total_change
        ),
        "global_flat_max_noise_ratio": (
            model_config.global_flat_max_noise_ratio
        ),
        "global_max_flat_bridge_regimes": (
            model_config.global_max_flat_bridge_regimes
        ),
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
        # [ADDED] OLS-only post-classification не вычисляется без structural CP.
        "structural_change_type": NO_STRUCTURAL_CHANGE,
        "classification_previous_slope_ols": math.nan,
        "classification_current_slope_ols": math.nan,
        "classification_delta_slope_ols": math.nan,
        "classification_level_shift_ols": math.nan,
        "slope_change_se": math.nan,
        "level_shift_se": math.nan,
        "slope_change_z": math.nan,
        "level_shift_z": math.nan,
        "previous_regime_trend_exists": False,
        "previous_regime_trend_direction": NO_DIRECTION,
        "previous_regime_trend_status": "NOT_EVALUATED",
        "direction_change": False,
        # [ADDED] Глобальный тренд — отдельная надстройка над current trend.
        "global_trend_exists": False,
        "global_trend_direction": NO_DIRECTION,
        "global_trend_length": 0,
        "global_trend_start_date": None,
        "global_trend_end_date": None,
        "global_trend_regime_count": 0,
        "global_trend_changepoint_count": 0,
        "global_trend_flat_regime_count": 0,
        "global_trend_structure": "",
        "global_trend_structure_json": "[]",
        "global_trend_start_gmv": math.nan,
        "global_trend_end_gmv": math.nan,
        "global_trend_gmv_change_abs": math.nan,
        "global_trend_gmv_change_relative": math.nan,
        "global_trend_slope_abs": math.nan,
        "global_trend_slope_relative": math.nan,
        "global_trend_total_change": math.nan,
        "global_direction_count_share": math.nan,
        "global_direction_movement_share": math.nan,
        "global_trend_to_noise": math.nan,
        "global_trend_status": "NOT_EVALUATED",
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
            changepoints=tuple(),
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
            changepoints=tuple(),
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
    local_rows, local_evaluations = _evaluate_selected_regimes(
        used_values,
        used_dates,
        selected_segments,
        thresholds,
        model_config,
    )
    current_evaluation = local_evaluations[-1]
    current_exists = bool(current_evaluation.trend_exists)
    included_global_indices, global_evaluation = _select_global_regime_indices(
        used_values,
        local_rows,
        local_evaluations,
        thresholds,
        model_config,
    )
    included_global_set = set(included_global_indices)
    for regime_index, local_row in enumerate(local_rows):
        in_global = regime_index in included_global_set
        local_row["in_global_trend"] = in_global
        local_row["global_trend_regime_role"] = (
            GLOBAL_FLAT_BRIDGE
            if in_global and local_row["local_regime_class"] == LOCAL_FLAT
            else GLOBAL_DIRECTIONAL
            if in_global
            else GLOBAL_NOT_INCLUDED
        )

    changepoint_rows = _build_selected_changepoint_rows(
        used_values,
        used_dates,
        selected_segments,
        cache,
        local_rows,
        local_evaluations,
        included_global_indices,
        sigma_estimate,
        beta,
        model_config,
    )
    last_changepoint = changepoint_rows[-1] if changepoint_rows else None
    previous_evaluation = (
        local_evaluations[-2] if len(local_evaluations) > 1 else None
    )

    global_exists = bool(included_global_indices)
    global_start_index = (
        int(local_rows[included_global_indices[0]]["start"])
        if global_exists
        else None
    )
    global_end_index = (
        int(local_rows[included_global_indices[-1]]["end"])
        if global_exists
        else None
    )
    global_structure_rows = (
        [local_rows[index] for index in included_global_indices]
        if global_exists
        else []
    )
    global_structure = " -> ".join(
        f"{row['local_regime_class']}({int(row['points'])})"
        for row in global_structure_rows
    )
    global_structure_json = json.dumps(
        [
            {
                "segment_index": int(row["segment_index"]),
                "type": row["local_regime_class"],
                "role": row["global_trend_regime_role"],
                "points": int(row["points"]),
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "start_gmv": row["local_start_gmv"],
                "end_gmv": row["local_end_gmv"],
                "gmv_change_abs": row["local_gmv_change_abs"],
                "gmv_change_relative": row["local_gmv_change_relative"],
                "slope_abs": row["local_trend_slope_abs"],
                "slope_relative": row["local_trend_slope_relative"],
                "total_change": row["local_trend_total_change"],
                "trend_to_noise": row["local_trend_to_noise"],
            }
            for row in global_structure_rows
        ],
        ensure_ascii=False,
        default=str,
    )
    global_start_gmv = (
        float(used_values[global_start_index]) if global_exists else math.nan
    )
    global_end_gmv = (
        float(used_values[global_end_index - 1]) if global_exists else math.nan
    )
    global_gmv_change = (
        float(global_end_gmv - global_start_gmv)
        if global_exists
        else math.nan
    )
    global_gmv_change_relative = (
        float(global_gmv_change / abs(global_start_gmv))
        if global_exists
        and not _effectively_zero(
            np.asarray([global_start_gmv], dtype=float),
            used_values[global_start_index:global_end_index],
        )
        else math.nan
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
        "change_z_threshold": model_config.most_recent_cp_change_z_threshold,
        "global_flat_max_relative_slope": (
            model_config.global_flat_max_relative_slope
        ),
        "global_flat_max_total_change": (
            model_config.global_flat_max_total_change
        ),
        "global_flat_max_noise_ratio": (
            model_config.global_flat_max_noise_ratio
        ),
        "global_max_flat_bridge_regimes": (
            model_config.global_max_flat_bridge_regimes
        ),
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
        "previous_regime_slope": (
            last_changepoint["previous_regime_slope"]
            if last_changepoint is not None
            else math.nan
        ),
        "current_regime_slope": current_cost.slope,
        "delta_slope": (
            last_changepoint["delta_slope"]
            if last_changepoint is not None
            else math.nan
        ),
        "level_shift": (
            last_changepoint["level_shift"]
            if last_changepoint is not None
            else math.nan
        ),
        "structural_change_type": (
            last_changepoint["structural_change_type"]
            if last_changepoint is not None
            else NO_STRUCTURAL_CHANGE
        ),
        "classification_previous_slope_ols": (
            last_changepoint["classification_previous_slope_ols"]
            if last_changepoint is not None
            else math.nan
        ),
        "classification_current_slope_ols": (
            last_changepoint["classification_current_slope_ols"]
            if last_changepoint is not None
            else math.nan
        ),
        "classification_delta_slope_ols": (
            last_changepoint["classification_delta_slope_ols"]
            if last_changepoint is not None
            else math.nan
        ),
        "classification_level_shift_ols": (
            last_changepoint["classification_level_shift_ols"]
            if last_changepoint is not None
            else math.nan
        ),
        "slope_change_se": (
            last_changepoint["slope_change_se"]
            if last_changepoint is not None
            else math.nan
        ),
        "level_shift_se": (
            last_changepoint["level_shift_se"]
            if last_changepoint is not None
            else math.nan
        ),
        "slope_change_z": (
            last_changepoint["slope_change_z"]
            if last_changepoint is not None
            else math.nan
        ),
        "level_shift_z": (
            last_changepoint["level_shift_z"]
            if last_changepoint is not None
            else math.nan
        ),
        "previous_regime_trend_exists": bool(
            last_changepoint is not None
            and last_changepoint["previous_regime_trend_exists"]
        ),
        "previous_regime_trend_direction": (
            last_changepoint["previous_regime_trend_direction"]
            if last_changepoint is not None
            else NO_DIRECTION
        ),
        "previous_regime_trend_status": (
            last_changepoint["previous_regime_trend_status"]
            if last_changepoint is not None
            else "NOT_EVALUATED"
        ),
        "direction_change": (
            bool(last_changepoint["direction_change"])
            if last_changepoint is not None
            else False
        ),
        "global_trend_exists": global_exists,
        "global_trend_direction": (
            global_evaluation.direction
            if global_exists and global_evaluation is not None
            else NO_DIRECTION
        ),
        "global_trend_length": (
            int(global_end_index - global_start_index)
            if global_exists
            else 0
        ),
        "global_trend_start_date": (
            used_dates[global_start_index] if global_exists else None
        ),
        "global_trend_end_date": (
            used_dates[global_end_index - 1] if global_exists else None
        ),
        "global_trend_regime_count": len(included_global_indices),
        "global_trend_changepoint_count": max(
            0,
            len(included_global_indices) - 1,
        ),
        "global_trend_flat_regime_count": sum(
            row["local_regime_class"] == LOCAL_FLAT
            for row in global_structure_rows
        ),
        "global_trend_structure": global_structure,
        "global_trend_structure_json": global_structure_json,
        "global_trend_start_gmv": global_start_gmv,
        "global_trend_end_gmv": global_end_gmv,
        "global_trend_gmv_change_abs": global_gmv_change,
        "global_trend_gmv_change_relative": global_gmv_change_relative,
        "global_trend_slope_abs": (
            global_evaluation.slope
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_trend_slope_relative": (
            global_evaluation.relative_slope
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_trend_total_change": (
            global_evaluation.total_change
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_direction_count_share": (
            global_evaluation.direction_count_share
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_direction_movement_share": (
            global_evaluation.direction_movement_share
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_trend_to_noise": (
            global_evaluation.trend_to_noise
            if global_exists and global_evaluation is not None
            else math.nan
        ),
        "global_trend_status": (
            global_evaluation.status
            if global_exists and global_evaluation is not None
            else "NOT_EVALUATED"
        ),
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
        segmentation_row = {
            "segment_index": segment_index,
            **asdict(cost_result),
            "start_date": used_dates[cost_result.start],
            "end_date": used_dates[cost_result.end - 1],
            "changepoint_penalty": beta if segment_index > 0 else 0.0,
            "cumulative_objective": cumulative_objective,
        }
        segmentation_row.update(
            {
                key: value
                for key, value in local_rows[segment_index].items()
                if key
                not in {
                    "segment_index",
                    "start",
                    "end",
                    "start_date",
                    "end_date",
                    "points",
                }
            }
        )
        segmentation_rows.append(segmentation_row)

    return MostRecentCPAnalysis(
        summary=summary,
        cp_profile=tuple(profile_rows),
        segment_diagnostics=tuple(segment_rows),
        segmentation=tuple(segmentation_rows),
        changepoints=tuple(changepoint_rows),
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
        changepoints=tuple(
            {**metadata, **row} for row in result.changepoints
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
    "change_z_threshold",
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
    "structural_change_type",
    "classification_previous_slope_ols",
    "classification_current_slope_ols",
    "classification_delta_slope_ols",
    "classification_level_shift_ols",
    "slope_change_se",
    "level_shift_se",
    "slope_change_z",
    "level_shift_z",
    "previous_regime_trend_exists",
    "previous_regime_trend_direction",
    "previous_regime_trend_status",
    "direction_change",
    "global_flat_max_relative_slope",
    "global_flat_max_total_change",
    "global_flat_max_noise_ratio",
    "global_max_flat_bridge_regimes",
    "global_trend_exists",
    "global_trend_direction",
    "global_trend_length",
    "global_trend_start_date",
    "global_trend_end_date",
    "global_trend_regime_count",
    "global_trend_changepoint_count",
    "global_trend_flat_regime_count",
    "global_trend_structure",
    "global_trend_structure_json",
    "global_trend_start_gmv",
    "global_trend_end_gmv",
    "global_trend_gmv_change_abs",
    "global_trend_gmv_change_relative",
    "global_trend_slope_abs",
    "global_trend_slope_relative",
    "global_trend_total_change",
    "global_direction_count_share",
    "global_direction_movement_share",
    "global_trend_to_noise",
    "global_trend_status",
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
    "local_start_gmv",
    "local_end_gmv",
    "local_gmv_change_abs",
    "local_gmv_change_relative",
    "local_trend_exists",
    "local_trend_direction",
    "local_trend_status",
    "local_trend_slope_abs",
    "local_trend_slope_relative",
    "local_trend_total_change",
    "local_direction_count_share",
    "local_direction_movement_share",
    "local_trend_to_noise",
    "local_typical_scale",
    "local_noise_scale",
    "local_noise_scale_source",
    "local_noise_ratio",
    "local_regime_class",
    "in_global_trend",
    "global_trend_regime_role",
)

MOST_RECENT_CP_CHANGEPOINT_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "changepoint_order",
    "changepoint_count",
    "is_last_changepoint",
    "cp_index",
    "left_end_date",
    "right_start_date",
    "previous_regime_index",
    "current_regime_index",
    "previous_regime_start",
    "previous_regime_end",
    "current_regime_start",
    "current_regime_end",
    "previous_regime_start_date",
    "previous_regime_end_date",
    "current_regime_start_date",
    "current_regime_end_date",
    "previous_regime_points",
    "current_regime_points",
    "previous_regime_class",
    "current_regime_class",
    "connects_flat_regime",
    "cp_inside_global_trend",
    "global_trend_start_cp",
    "global_trend_direction",
    "adverse_significant_level_shift",
    "cost_type",
    "sigma",
    "sigma_source",
    "beta",
    "change_z_threshold",
    "previous_regime_slope",
    "current_regime_slope",
    "delta_slope",
    "level_shift",
    "structural_change_type",
    "classification_previous_slope_ols",
    "classification_current_slope_ols",
    "classification_delta_slope_ols",
    "classification_level_shift_ols",
    "slope_change_se",
    "level_shift_se",
    "slope_change_z",
    "level_shift_z",
    "previous_regime_trend_exists",
    "previous_regime_trend_direction",
    "previous_regime_trend_status",
    "current_regime_trend_exists",
    "current_regime_trend_direction",
    "current_regime_trend_status",
    "direction_change",
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
        ``trend_segment_diagnostics``, ``trend_segmentation`` и
        ``trend_changepoints``.

    Raises:
        ValueError: Если панель пуста или конфигурация не Most Recent CP.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [1, 2, 3, 4]})
        >>> sorted(build_most_recent_cp_trend_analysis(panel, [1, 2, 3, 4]))
        ['trend_changepoints', 'trend_cp_profile', 'trend_segment_diagnostics', 'trend_segmentation', 'trend_summary']
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
    changepoint_rows: List[Dict[str, object]] = []
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
        changepoint_rows.extend(result.changepoints)
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
        "trend_changepoints": _ordered_frame(
            changepoint_rows,
            MOST_RECENT_CP_CHANGEPOINT_COLUMNS,
        ),
    }


__all__ = [
    "LEVEL_AND_SLOPE",
    "LEVEL_SHIFT",
    "LOCAL_FLAT",
    "LOCAL_UNCONFIRMED",
    "MAE_FALLBACK",
    "NO_STRUCTURAL_CHANGE",
    "OLS_RESIDUAL_MAD",
    "OLSRegimeDiagnostic",
    "PERFECT_FIT",
    "DIFF_MAD",
    "SLOPE_CHANGE",
    "WEAK_OR_UNCLASSIFIED",
    "MostRecentCPAnalysis",
    "SegmentCostResult",
    "SigmaEstimate",
    "analyze_most_recent_cp_series",
    "analyze_segment_most_recent_cp",
    "build_most_recent_cp_trend_analysis",
    "calculate_segment_cost",
    "estimate_series_sigma",
]
