"""Независимый анализ текущего GMV-тренда и одной смены направления.

Модуль работает только поверх полной панели ``segment_id x cal_date``, уже
построенной :func:`gmv_anomaly.data_preparation.build_full_week_grid`. Он не
участвует в anomaly score, hierarchy adjustment или Set Packing.

Текущий MVP поддерживает произвольную длину последовательной истории, но
возвращает не более одной подтверждённой смены направления. Ожидаемая история
на практике короткая (около 13 точек), поэтому прозрачная реализация
Тейла–Сена с вычислительной сложностью ``O(m^2)`` предпочтительнее тяжёлой
дополнительной зависимости.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


GROWTH = "GROWTH"
DECLINE = "DECLINE"
NO_DIRECTION = "NONE"

TREND_SEARCH_METHODS = frozenset({"legacy", "most_recent_cp"})
MOST_RECENT_CP_COSTS = frozenset({"ols", "capped", "huber"})


# [ADDED] Пороги тренда отделены от конфигурации anomaly detection.
@dataclass(frozen=True)
class TrendThresholds:
    """Задать единый контракт подтверждения тренда и смены направления.

    Args:
        min_trend_points: Минимальная длина одного трендового режима.
        min_total_trend_change: Минимальный модуль трендового изменения окна.
        min_direction_count_share: Минимальная доля ненулевых переходов по тренду.
        min_direction_movement_share: Минимальная доля движения по тренду.
        min_trend_to_noise: Минимальное отношение трендового движения к шуму.
        min_change_f: Инженерный порог F для смены наклона.
        near_best_change_ratio: Доля лучшего F для выбора более позднего k.

    Returns:
        Неизменяемую конфигурацию независимого трендового анализа.

    Raises:
        ValueError: Если пороги выходят за допустимые диапазоны.

    Examples:
        >>> TrendThresholds().min_trend_points
        4
    """

    min_trend_points: int = 4
    min_total_trend_change: float = 0.10
    min_direction_count_share: float = 0.75
    min_direction_movement_share: float = 0.75
    min_trend_to_noise: float = 2.0
    min_change_f: float = 8.0
    near_best_change_ratio: float = 0.95

    def __post_init__(self) -> None:
        """Проверить параметры сразу при создании конфигурации.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            ValueError: Если параметр не соответствует контракту.

        Examples:
            >>> TrendThresholds(min_trend_points=4)
            TrendThresholds(min_trend_points=4, min_total_trend_change=0.1, min_direction_count_share=0.75, min_direction_movement_share=0.75, min_trend_to_noise=2.0, min_change_f=8.0, near_best_change_ratio=0.95)
        """

        if (
            isinstance(self.min_trend_points, bool)
            or not isinstance(self.min_trend_points, (int, np.integer))
            or self.min_trend_points < 4
        ):
            raise ValueError("min_trend_points должен быть целым числом не меньше 4")
        numeric_nonnegative = {
            "min_total_trend_change": self.min_total_trend_change,
            "min_trend_to_noise": self.min_trend_to_noise,
            "min_change_f": self.min_change_f,
        }
        for name, value in numeric_nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} должен быть конечным неотрицательным числом")
        shares = {
            "min_direction_count_share": self.min_direction_count_share,
            "min_direction_movement_share": self.min_direction_movement_share,
        }
        for name, value in shares.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} должен находиться в диапазоне [0, 1]")
        if (
            not math.isfinite(float(self.near_best_change_ratio))
            or not 0.0 < float(self.near_best_change_ratio) <= 1.0
        ):
            raise ValueError("near_best_change_ratio должен находиться в диапазоне (0, 1]")


# [ADDED] Выбор модели тренда отделён от бизнес-порогов подтверждения тренда.
@dataclass(frozen=True)
class TrendModelConfig:
    """Настроить способ поиска начала текущего тренда.

    Args:
        trend_search_method: ``legacy`` или ``most_recent_cp``.
        most_recent_cp_cost: ``ols``, ``capped`` или ``huber``.
        most_recent_cp_min_segment_points: Минимальная длина каждого режима.
        most_recent_cp_capped_k: Порог K capped quadratic objective.
        most_recent_cp_huber_delta: Порог delta нормированного Huber objective.

    Returns:
        Неизменяемую конфигурацию selector и Most Recent CP.

    Raises:
        ValueError: Если метод, cost или числовой параметр нарушает контракт.

    Examples:
        >>> TrendModelConfig().trend_search_method
        'legacy'
    """

    trend_search_method: str = "most_recent_cp"
    most_recent_cp_cost: str = "ols"
    most_recent_cp_min_segment_points: int = 4
    most_recent_cp_capped_k: float = 2.0
    most_recent_cp_huber_delta: float = 1.345

    def __post_init__(self) -> None:
        """Проверить конфигурацию до запуска расчёта.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            ValueError: Если значение не поддерживается.

        Examples:
            >>> TrendModelConfig(most_recent_cp_cost="huber").most_recent_cp_cost
            'huber'
        """

        if self.trend_search_method not in TREND_SEARCH_METHODS:
            raise ValueError(
                "trend_search_method должен быть одним из: "
                + ", ".join(sorted(TREND_SEARCH_METHODS))
            )
        if self.most_recent_cp_cost not in MOST_RECENT_CP_COSTS:
            raise ValueError(
                "most_recent_cp_cost должен быть одним из: "
                + ", ".join(sorted(MOST_RECENT_CP_COSTS))
            )
        if (
            isinstance(self.most_recent_cp_min_segment_points, bool)
            or not isinstance(
                self.most_recent_cp_min_segment_points,
                (int, np.integer),
            )
            or self.most_recent_cp_min_segment_points < 4
        ):
            raise ValueError(
                "most_recent_cp_min_segment_points должен быть целым числом "
                "не меньше 4"
            )
        positive_values = {
            "most_recent_cp_capped_k": self.most_recent_cp_capped_k,
            "most_recent_cp_huber_delta": self.most_recent_cp_huber_delta,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} должен быть конечным положительным числом")


@dataclass(frozen=True)
class TrendEvaluation:
    """Хранить раздельную диагностику тренда одного окна.

    Args:
        points: Число точек окна.
        trend_exists: Прошло ли окно все критерии тренда.
        direction: ``GROWTH``, ``DECLINE`` или ``NONE``.
        slope: Наклон Тейла–Сена в единицах GMV за период.
        relative_slope: Наклон относительно медианного GMV.
        total_change: Трендовое изменение на полной длине окна.
        direction_count_share: Доля ненулевых переходов по направлению тренда.
        direction_movement_share: Доля абсолютного движения по направлению.
        trend_to_noise: Отношение трендового движения к остаточному шуму.
        typical_scale: Медианный GMV окна.
        intercept: Устойчивый свободный член линии.
        residual_mad: MAD остатков устойчивой линии.
        noise_scale: Масштаб шума в знаменателе ``trend_to_noise``.
        noise_scale_source: ``MAD``, ``MAE_FALLBACK`` или ``PERFECT_FIT``.
        status: Результат проверки окна.

    Returns:
        Неизменяемую диагностику окна.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> evaluate_trend([100, 110, 120, 130]).direction
        'GROWTH'
    """

    points: int
    trend_exists: bool
    direction: str
    slope: float
    relative_slope: float
    total_change: float
    direction_count_share: float
    direction_movement_share: float
    trend_to_noise: float
    typical_scale: float
    intercept: float
    residual_mad: float
    noise_scale: float
    noise_scale_source: str
    status: str
    # [ADDED] Диагностика бизнес-правила о направлении первого перехода.
    first_change: float = math.nan
    passes_initial_direction: bool = False


@dataclass(frozen=True)
class TrendWindowEvaluation:
    """Связать диагностику тренда с длиной последнего окна.

    Args:
        window_length: Число последних точек в окне.
        evaluation: Результат канонического ``evaluate_trend``.

    Returns:
        Неизменяемую запись suffix-window.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> evaluate_suffix_trends([100, 110, 120, 130])[0].window_length
        4
    """

    window_length: int
    evaluation: TrendEvaluation


@dataclass(frozen=True)
class LinearModelFit:
    """Хранить коэффициенты, fitted values и RSS линейной модели.

    Args:
        coefficients: Оценённые коэффициенты МНК.
        fitted_values: Значения модели на исходной временной сетке.
        rss: Сумма квадратов остатков.

    Returns:
        Неизменяемый результат МНК.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> round(fit_single_linear_model([1, 2, 3, 4]).rss, 12)
        0.0
    """

    coefficients: Tuple[float, ...]
    fitted_values: Tuple[float, ...]
    rss: float


@dataclass(frozen=True)
class TrendChangeEvaluation:
    """Хранить полную диагностику одного допустимого breakpoint ``k``.

    Args:
        k: Число наблюдений в левой части ``values[:k]``.
        left_points: Длина левого режима.
        right_points: Длина правого режима.
        rss_single: RSS одной OLS-линии.
        rss_piecewise: RSS непрерывной кусочно-линейной OLS-модели.
        f_statistic: Инженерная F-статистика изменения наклона.
        left_trend: Устойчивая оценка тренда левой части.
        right_trend: Устойчивая оценка тренда правой части.
        passes_f_threshold: Выполнен ли инженерный порог F.
        passes_direction_change: Подтверждены ли противоположные направления.
        valid_change: Выполнены ли оба условия настоящей смены.
        selected: Выбрана ли эта точка итоговым near-best правилом.

    Returns:
        Неизменяемую диагностику точки разделения.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> len(evaluate_change_candidates([100, 110, 120, 130, 120, 110, 100, 90]))
        1
    """

    k: int
    left_points: int
    right_points: int
    rss_single: float
    rss_piecewise: float
    f_statistic: float
    left_trend: TrendEvaluation
    right_trend: TrendEvaluation
    passes_f_threshold: bool
    passes_direction_change: bool
    valid_change: bool
    selected: bool = False


@dataclass(frozen=True)
class SegmentTrendAnalysis:
    """Объединить итог сегмента и две подробные диагностики.

    Args:
        summary: Одна итоговая запись сегмента.
        window_diagnostics: Записи последних окон от ``min_trend_points`` до N.
        change_diagnostics: Записи всех допустимых ``k``.

    Returns:
        Неизменяемый контейнер анализа сегмента.

    Raises:
        ValueError: Не выбрасывается при создании результата.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [100, 110, 120, 130]})
        >>> analyze_segment_trend(panel, [1, 2, 3, 4]).summary['current_trend_direction']
        'GROWTH'
    """

    summary: Dict[str, object]
    window_diagnostics: Tuple[Dict[str, object], ...]
    change_diagnostics: Tuple[Dict[str, object], ...]


def _coerce_finite_values(values: Sequence[float]) -> np.ndarray:
    """Преобразовать одномерную последовательность в конечный float-массив.

    Args:
        values: Числовая последовательность GMV.

    Returns:
        Одномерный NumPy-массив ``float``.

    Raises:
        ValueError: Если значения не одномерны, нечисловы или не конечны.

    Examples:
        >>> _coerce_finite_values([1, 2]).tolist()
        [1.0, 2.0]
    """

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("GMV-история должна состоять из числовых значений") from exc
    if array.ndim != 1:
        raise ValueError("GMV-история должна быть одномерной")
    if not np.isfinite(array).all():
        raise ValueError("GMV-история должна содержать только конечные значения")
    return array


def _effectively_zero(values: np.ndarray, reference: np.ndarray) -> bool:
    """Проверить машинный ноль относительно масштаба исходного ряда.

    Args:
        values: Проверяемые остатки.
        reference: Исходные значения, задающие числовой масштаб.

    Returns:
        True, если все остатки отличаются от нуля только на уровне float error.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _effectively_zero(np.array([1e-15]), np.array([100.0]))
        True
    """

    scale = max(
        float(np.max(np.abs(reference))) if reference.size else 0.0,
        np.finfo(float).tiny,
    )
    tolerance = 64.0 * np.finfo(float).eps * scale
    return bool(np.all(np.abs(values) <= tolerance))


def _rss_effectively_zero(rss: float, values: np.ndarray) -> bool:
    """Проверить RSS на машинный ноль с сохранением масштабной инвариантности.

    Args:
        rss: Сумма квадратов остатков.
        values: Исходный ряд модели.

    Returns:
        True, если RSS объясняется только арифметикой float.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _rss_effectively_zero(0.0, np.array([1.0, 2.0]))
        True
    """

    squared_scale = max(float(np.dot(values, values)), np.finfo(float).tiny)
    tolerance = 256.0 * np.finfo(float).eps * squared_scale
    return bool(float(rss) <= tolerance)


# [ADDED] Ведущие нули удаляются только на полном ряду, до построения окон.
def trim_leading_zero_history(
    values: Sequence[float],
    dates: Optional[Sequence[object]] = None,
) -> Tuple[List[float], List[object]]:
    """Удалить точки до первого строго положительного GMV.

    Args:
        values: Полная последовательная история сегмента.
        dates: Соответствующие даты; если None, используются позиции ``0..N-1``.

    Returns:
        Два синхронно обрезанных списка: GMV и даты. Внутренние и конечные
        нули после начала активности сохраняются.

    Raises:
        ValueError: Если даты не соответствуют значениям или GMV невалиден.

    Examples:
        >>> trim_leading_zero_history([0, 0, 100, 0], [1, 2, 3, 4])
        ([100.0, 0.0], [3, 4])
    """

    array = _coerce_finite_values(values)
    normalized_dates = list(range(len(array))) if dates is None else list(dates)
    if len(normalized_dates) != len(array):
        raise ValueError("Количество дат должно совпадать с количеством GMV-точек")
    positive_positions = np.flatnonzero(array > 0.0)
    if positive_positions.size == 0:
        return [], []
    first_active = int(positive_positions[0])
    return array[first_active:].astype(float).tolist(), normalized_dates[first_active:]


def theil_sen_slope(values: Sequence[float]) -> float:
    """Рассчитать медиану всех попарных наклонов Тейла–Сена.

    Args:
        values: Последовательность минимум из двух конечных значений.

    Returns:
        Устойчивый наклон в единицах значения за один период.

    Raises:
        ValueError: Если передано меньше двух точек или невалидные значения.

    Examples:
        >>> theil_sen_slope([100, 110, 120, 130])
        10.0
    """

    array = _coerce_finite_values(values)
    if len(array) < 2:
        raise ValueError("Для наклона Тейла–Сена нужно минимум две точки")
    slopes = [
        (float(array[j]) - float(array[i])) / float(j - i)
        for i in range(len(array) - 1)
        for j in range(i + 1, len(array))
    ]
    return float(np.median(np.asarray(slopes, dtype=float)))


def evaluate_trend(
    values: Sequence[float],
    thresholds: Optional[TrendThresholds] = None,
) -> TrendEvaluation:
    """Канонически оценить устойчивый тренд одного окна.

    Args:
        values: GMV-точки окна без повторной обрезки ведущих нулей.
        thresholds: Пороги тренда; если None, используются значения MVP.

    Returns:
        Полную раздельную диагностику наклона, направленности, шума и первого
        изменения GMV.

    Raises:
        ValueError: Если GMV-значения нечисловые или не конечны.

    Examples:
        >>> evaluate_trend([100, 110, 120, 130]).trend_exists
        True
    """

    thresholds = thresholds or TrendThresholds()
    array = _coerce_finite_values(values)
    points = len(array)
    empty_metric = math.nan
    if points < thresholds.min_trend_points:
        return TrendEvaluation(
            points=points,
            trend_exists=False,
            direction=NO_DIRECTION,
            slope=empty_metric,
            relative_slope=empty_metric,
            total_change=empty_metric,
            direction_count_share=0.0,
            direction_movement_share=0.0,
            trend_to_noise=empty_metric,
            typical_scale=float(np.median(array)) if points else empty_metric,
            intercept=empty_metric,
            residual_mad=empty_metric,
            noise_scale=empty_metric,
            noise_scale_source="NOT_EVALUATED",
            status="INSUFFICIENT_HISTORY",
        )

    typical_scale = float(np.median(array))
    if typical_scale <= 0.0:
        return TrendEvaluation(
            points=points,
            trend_exists=False,
            direction=NO_DIRECTION,
            slope=empty_metric,
            relative_slope=empty_metric,
            total_change=empty_metric,
            direction_count_share=0.0,
            direction_movement_share=0.0,
            trend_to_noise=empty_metric,
            typical_scale=typical_scale,
            intercept=empty_metric,
            residual_mad=empty_metric,
            noise_scale=empty_metric,
            noise_scale_source="NOT_EVALUATED",
            status="ZERO_SCALE",
        )

    slope = theil_sen_slope(array)
    relative_slope = slope / typical_scale
    total_change = relative_slope * float(points - 1)
    deltas = np.diff(array)
    nonzero_deltas = deltas[deltas != 0.0]
    if slope == 0.0 or nonzero_deltas.size == 0:
        direction_count_share = 0.0
    else:
        direction_count_share = float(
            np.count_nonzero(nonzero_deltas * slope > 0.0) / nonzero_deltas.size
        )
    total_movement = float(np.abs(deltas).sum())
    if slope == 0.0 or total_movement == 0.0:
        direction_movement_share = 0.0
    else:
        direction_movement_share = float(
            np.abs(deltas[deltas * slope > 0.0]).sum() / total_movement
        )

    time = np.arange(points, dtype=float)
    intercept = float(np.median(array - slope * time))
    residuals = array - (intercept + slope * time)
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    robust_noise = 1.4826 * residual_mad
    trend_movement = abs(slope) * float(points - 1)
    if robust_noise > 0.0:
        noise_scale = robust_noise
        noise_scale_source = "MAD"
        trend_to_noise = trend_movement / noise_scale
    elif _effectively_zero(residuals, array):
        noise_scale = 0.0
        noise_scale_source = "PERFECT_FIT"
        trend_to_noise = math.inf if slope != 0.0 else 0.0
    else:
        noise_scale = float(np.mean(np.abs(residuals)))
        noise_scale_source = "MAE_FALLBACK"
        trend_to_noise = trend_movement / noise_scale if noise_scale > 0.0 else 0.0

    first_change = float(deltas[0])
    passes_initial_direction = bool(
        (slope > 0.0 and first_change > 0.0)
        or (slope < 0.0 and first_change < 0.0)
    )
    passes_core_trend = bool(
        slope != 0.0
        and abs(total_change) >= thresholds.min_total_trend_change
        and direction_count_share >= thresholds.min_direction_count_share
        and direction_movement_share >= thresholds.min_direction_movement_share
        and trend_to_noise >= thresholds.min_trend_to_noise
    )
    # [ADDED] Первый переход обязан совпасть с итоговым знаком устойчивого тренда.
    # Остальные переходы по-прежнему оцениваются агрегированными критериями выше.
    trend_exists = bool(passes_core_trend and passes_initial_direction)
    direction = (
        GROWTH
        if trend_exists and slope > 0.0
        else DECLINE
        if trend_exists and slope < 0.0
        else NO_DIRECTION
    )
    return TrendEvaluation(
        points=points,
        trend_exists=trend_exists,
        direction=direction,
        slope=slope,
        relative_slope=relative_slope,
        total_change=total_change,
        direction_count_share=direction_count_share,
        direction_movement_share=direction_movement_share,
        trend_to_noise=float(trend_to_noise),
        typical_scale=typical_scale,
        intercept=intercept,
        residual_mad=residual_mad,
        noise_scale=noise_scale,
        noise_scale_source=noise_scale_source,
        status=(
            "TREND"
            if trend_exists
            else "INITIAL_DIRECTION_MISMATCH"
            if passes_core_trend
            else "NO_TREND"
        ),
        first_change=first_change,
        passes_initial_direction=passes_initial_direction,
    )


def evaluate_suffix_trends(
    values: Sequence[float],
    thresholds: Optional[TrendThresholds] = None,
) -> List[TrendWindowEvaluation]:
    """Оценить все вложенные последние окна от минимального до полного.

    Args:
        values: Уже один раз подготовленная активная история сегмента.
        thresholds: Единые пороги всех окон.

    Returns:
        Диагностики окон в порядке ``min_trend_points, ..., N``.

    Raises:
        ValueError: Если значения невалидны.

    Examples:
        >>> [item.window_length for item in evaluate_suffix_trends([1, 2, 3, 4, 5])]
        [4, 5]
    """

    thresholds = thresholds or TrendThresholds()
    array = _coerce_finite_values(values)
    return [
        TrendWindowEvaluation(
            window_length=window_length,
            evaluation=evaluate_trend(array[-window_length:], thresholds),
        )
        for window_length in range(thresholds.min_trend_points, len(array) + 1)
    ]


def fit_single_linear_model(values: Sequence[float]) -> LinearModelFit:
    """Оценить нулевую OLS-модель ``y = a + b*t``.

    Args:
        values: Конечная последовательность минимум из двух точек.

    Returns:
        Коэффициенты ``(a, b)``, fitted values и RSS.

    Raises:
        ValueError: Если точек меньше двух или значения невалидны.

    Examples:
        >>> tuple(round(value, 12) for value in fit_single_linear_model([1, 2, 3]).coefficients)
        (1.0, 1.0)
    """

    array = _coerce_finite_values(values)
    if len(array) < 2:
        raise ValueError("Для линейной модели нужно минимум две точки")
    time = np.arange(len(array), dtype=float)
    design = np.column_stack((np.ones(len(array), dtype=float), time))
    coefficients, _, _, _ = np.linalg.lstsq(design, array, rcond=None)
    fitted = design @ coefficients
    residuals = array - fitted
    return LinearModelFit(
        coefficients=tuple(float(value) for value in coefficients),
        fitted_values=tuple(float(value) for value in fitted),
        rss=float(np.dot(residuals, residuals)),
    )


def fit_continuous_piecewise_model(
    values: Sequence[float],
    k: int,
    min_regime_points: int = 4,
) -> LinearModelFit:
    """Оценить непрерывную кусочно-линейную OLS-модель.

    Математическая модель использует наблюдения ``t=1..N`` и hinge
    ``max(0, t-k)``. Для Python-индекса ``t=0..N-1`` это строго эквивалентно
    ``max(0, t-(k-1))``: ``values[k-1]`` — последняя левая точка, а
    ``values[k]`` — первая правая.

    Args:
        values: Полная последовательность значений.
        k: Число точек слева; разбиение строго ``values[:k]`` и ``values[k:]``.
        min_regime_points: Минимальное число точек с каждой стороны.

    Returns:
        Коэффициенты ``(a, b, c)``, fitted values и RSS.

    Raises:
        ValueError: Если k не оставляет требуемое число точек с обеих сторон.

    Examples:
        >>> fit_continuous_piecewise_model([1, 2, 3, 4, 3, 2, 1, 0], 4).rss >= 0
        True
    """

    array = _coerce_finite_values(values)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k должен быть целым числом")
    if k < min_regime_points or len(array) - k < min_regime_points:
        raise ValueError(
            "k должен оставлять минимум min_regime_points точек с каждой стороны"
        )
    time = np.arange(len(array), dtype=float)
    # [FIXED] Перенос 1-based формулы на Python без сдвига breakpoint вправо.
    hinge = np.maximum(0.0, time - float(k - 1))
    design = np.column_stack((np.ones(len(array), dtype=float), time, hinge))
    coefficients, _, _, _ = np.linalg.lstsq(design, array, rcond=None)
    fitted = design @ coefficients
    residuals = array - fitted
    return LinearModelFit(
        coefficients=tuple(float(value) for value in coefficients),
        fitted_values=tuple(float(value) for value in fitted),
        rss=float(np.dot(residuals, residuals)),
    )


def _change_f_statistic(
    rss_single: float,
    rss_piecewise: float,
    values: np.ndarray,
) -> float:
    """Рассчитать F изменения наклона с явной обработкой нулевого RSS.

    Args:
        rss_single: RSS нулевой модели с двумя коэффициентами.
        rss_piecewise: RSS альтернативной модели с тремя коэффициентами.
        values: Полный ряд, задающий N и числовой масштаб.

    Returns:
        Неотрицательную F-статистику, ``0`` или ``+inf``.

    Raises:
        ValueError: Если N не позволяет оценить модель с тремя коэффициентами.

    Examples:
        >>> _change_f_statistic(1.0, 0.0, np.arange(8.0))
        inf
    """

    points = len(values)
    if points <= 3:
        raise ValueError("Для F-критерия требуется N > 3")
    piecewise_zero = _rss_effectively_zero(rss_piecewise, values)
    single_zero = _rss_effectively_zero(rss_single, values)
    if piecewise_zero:
        return 0.0 if single_zero else math.inf
    improvement = max(float(rss_single) - float(rss_piecewise), 0.0)
    if _rss_effectively_zero(improvement, values):
        return 0.0
    return improvement / (float(rss_piecewise) / float(points - 3))


def evaluate_change_candidates(
    values: Sequence[float],
    thresholds: Optional[TrendThresholds] = None,
) -> List[TrendChangeEvaluation]:
    """Проверить все динамически допустимые точки одной смены направления.

    Args:
        values: Уже один раз обрезанная активная история сегмента.
        thresholds: Единые пороги трендов и F-критерия.

    Returns:
        Диагностики для каждого ``k=min_points,...,N-min_points``.

    Raises:
        ValueError: Если значения невалидны.

    Examples:
        >>> [item.k for item in evaluate_change_candidates(list(range(9)))]
        [4, 5]
    """

    thresholds = thresholds or TrendThresholds()
    array = _coerce_finite_values(values)
    minimum = thresholds.min_trend_points
    if len(array) < 2 * minimum:
        return []
    single_fit = fit_single_linear_model(array)
    results: List[TrendChangeEvaluation] = []
    for k in range(minimum, len(array) - minimum + 1):
        piecewise_fit = fit_continuous_piecewise_model(array, k, minimum)
        f_statistic = _change_f_statistic(
            single_fit.rss,
            piecewise_fit.rss,
            array,
        )
        left_trend = evaluate_trend(array[:k], thresholds)
        right_trend = evaluate_trend(array[k:], thresholds)
        passes_direction_change = bool(
            left_trend.trend_exists
            and right_trend.trend_exists
            and left_trend.direction != right_trend.direction
        )
        passes_f_threshold = bool(f_statistic >= thresholds.min_change_f)
        results.append(
            TrendChangeEvaluation(
                k=k,
                left_points=k,
                right_points=len(array) - k,
                rss_single=single_fit.rss,
                rss_piecewise=piecewise_fit.rss,
                f_statistic=f_statistic,
                left_trend=left_trend,
                right_trend=right_trend,
                passes_f_threshold=passes_f_threshold,
                passes_direction_change=passes_direction_change,
                valid_change=passes_f_threshold and passes_direction_change,
            )
        )
    return results


def select_change_point(
    candidates: Sequence[TrendChangeEvaluation],
    thresholds: Optional[TrendThresholds] = None,
) -> Optional[TrendChangeEvaluation]:
    """Выбрать более поздний breakpoint только среди near-best решений.

    Args:
        candidates: Полные диагностики допустимых k.
        thresholds: Конфигурация near-best правила.

    Returns:
        Выбранный кандидат с ``selected=True`` либо None.

    Raises:
        ValueError: Если конфигурация порогов невалидна.

    Examples:
        >>> select_change_point([], TrendThresholds()) is None
        True
    """

    thresholds = thresholds or TrendThresholds()
    valid = [candidate for candidate in candidates if candidate.valid_change]
    if not valid:
        return None
    best_f = max(candidate.f_statistic for candidate in valid)
    if math.isinf(best_f):
        near_best = [candidate for candidate in valid if math.isinf(candidate.f_statistic)]
    else:
        minimum_near_best = thresholds.near_best_change_ratio * best_f
        near_best = [
            candidate
            for candidate in valid
            if candidate.f_statistic >= minimum_near_best
        ]
    return replace(max(near_best, key=lambda candidate: candidate.k), selected=True)


def _select_current_suffix_trend(
    windows: Sequence[TrendWindowEvaluation],
) -> Tuple[Optional[int], Optional[TrendEvaluation]]:
    """Найти актуальное направление и максимальное подтверждённое suffix-окно.

    Args:
        windows: Окна в порядке от самого короткого к полному.

    Returns:
        Длину и диагностику текущего тренда либо ``(None, None)``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _select_current_suffix_trend([])
        (None, None)
    """

    first_index = next(
        (
            index
            for index, window in enumerate(windows)
            if window.evaluation.trend_exists
        ),
        None,
    )
    if first_index is None:
        return None, None
    current_direction = windows[first_index].evaluation.direction
    current_length = windows[first_index].window_length
    current_evaluation = windows[first_index].evaluation
    for window in windows[first_index + 1 :]:
        evaluation = window.evaluation
        if evaluation.trend_exists and evaluation.direction != current_direction:
            break
        if evaluation.trend_exists and evaluation.direction == current_direction:
            current_length = window.window_length
            current_evaluation = evaluation
    return current_length, current_evaluation


def _empty_summary(
    metadata: Dict[str, object],
    raw_points: int,
    used_points: int,
    status: str,
) -> Dict[str, object]:
    """Создать типизированную итоговую строку без подтверждённого тренда.

    Args:
        metadata: Идентификаторы сегмента.
        raw_points: Число календарных точек до обрезки.
        used_points: Число точек активной истории.
        status: ``NO_ACTIVE_HISTORY`` или ``INSUFFICIENT_HISTORY``.

    Returns:
        Полную итоговую запись с пустыми трендовыми полями.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _empty_summary({'segment_id': 's'}, 4, 0, 'NO_ACTIVE_HISTORY')['change_type']
        'NONE'
    """

    return {
        **metadata,
        "status": status,
        "history_points_raw": raw_points,
        "history_points_used": used_points,
        "leading_zero_points_trimmed": raw_points - used_points,
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
        "change_detected": False,
        "change_type": NO_DIRECTION,
        "change_k": None,
        "change_left_end_date": None,
        "change_right_start_date": None,
        "trend_before_direction": NO_DIRECTION,
        "trend_after_direction": NO_DIRECTION,
        "trend_before_slope": math.nan,
        "trend_after_slope": math.nan,
        "change_f": math.nan,
        "max_f": math.nan,
    }


def _segment_metadata(segment_panel: pd.DataFrame) -> Dict[str, object]:
    """Извлечь стандартные неизменные идентификаторы готовой панели.

    Args:
        segment_panel: Строки ровно одного сегмента.

    Returns:
        Словарь доступных ``segment_id/key/level/slice_depth``.

    Raises:
        ValueError: Если ``segment_id`` не единственный или metadata меняется.

    Examples:
        >>> _segment_metadata(pd.DataFrame({'segment_id': ['s', 's']}))
        {'segment_id': 's'}
    """

    if "segment_id" not in segment_panel.columns or segment_panel.empty:
        raise ValueError("segment_panel должен содержать непустую колонку segment_id")
    segment_ids = segment_panel["segment_id"].astype(str).unique().tolist()
    if len(segment_ids) != 1:
        raise ValueError("analyze_segment_trend принимает ровно один segment_id")
    metadata: Dict[str, object] = {"segment_id": segment_ids[0]}
    for column in ("segment_key", "segment_level", "slice_depth"):
        if column not in segment_panel.columns:
            continue
        unique_values = segment_panel[column].drop_duplicates()
        if len(unique_values) != 1:
            raise ValueError(f"Metadata {column!r} меняется внутри одного segment_id")
        metadata[column] = unique_values.iloc[0]
    return metadata


# [ADDED] Анализ сегмента потребляет готовую полную панель и не строит календарь.
def analyze_segment_trend(
    segment_panel: pd.DataFrame,
    dates: Sequence[int],
    thresholds: Optional[TrendThresholds] = None,
) -> SegmentTrendAnalysis:
    """Проанализировать текущий тренд и максимум одну смену одного сегмента.

    Args:
        segment_panel: Полная панель одного ``segment_id`` после preprocessing.
        dates: Последовательная временная ось из total-слоя.
        thresholds: Независимые пороги тренда.

    Returns:
        Итоговую запись, suffix-window диагностику и диагностику всех k.

    Raises:
        ValueError: Если панель не соответствует готовой сетке ``segment x date``.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [100, 110, 120, 130]})
        >>> analyze_segment_trend(panel, [1, 2, 3, 4]).summary['current_trend_exists']
        True
    """

    thresholds = thresholds or TrendThresholds()
    required = {"segment_id", "cal_date", "gmv"}
    missing = sorted(required - set(segment_panel.columns))
    if missing:
        raise ValueError(f"Для trend analysis не хватает колонок: {missing}")
    normalized_dates = [int(date) for date in dates]
    if not normalized_dates:
        raise ValueError("dates не должен быть пустым")
    if len(set(normalized_dates)) != len(normalized_dates) or normalized_dates != sorted(normalized_dates):
        raise ValueError("dates должен быть строго возрастающей уникальной осью")
    metadata = _segment_metadata(segment_panel)
    if segment_panel.duplicated(subset=["cal_date"]).any():
        raise ValueError("Панель сегмента содержит дубли cal_date")
    indexed = segment_panel.copy()
    indexed["cal_date"] = pd.to_numeric(indexed["cal_date"], errors="coerce")
    if indexed["cal_date"].isna().any():
        raise ValueError("cal_date должен быть числовым")
    indexed["cal_date"] = indexed["cal_date"].astype(int)
    actual_dates = set(indexed["cal_date"].tolist())
    if actual_dates != set(normalized_dates):
        raise ValueError("segment_panel должен содержать ровно одну строку на каждую дату total-календаря")
    ordered = indexed.set_index("cal_date").reindex(normalized_dates)
    raw_values = _coerce_finite_values(ordered["gmv"].tolist())
    used_values, used_dates = trim_leading_zero_history(raw_values, normalized_dates)
    raw_points = len(raw_values)
    used_points = len(used_values)
    if used_points == 0:
        return SegmentTrendAnalysis(
            summary=_empty_summary(metadata, raw_points, 0, "NO_ACTIVE_HISTORY"),
            window_diagnostics=tuple(),
            change_diagnostics=tuple(),
        )
    if used_points < thresholds.min_trend_points:
        return SegmentTrendAnalysis(
            summary=_empty_summary(
                metadata,
                raw_points,
                used_points,
                "INSUFFICIENT_HISTORY",
            ),
            window_diagnostics=tuple(),
            change_diagnostics=tuple(),
        )

    windows = evaluate_suffix_trends(used_values, thresholds)
    current_length, current_evaluation = _select_current_suffix_trend(windows)
    change_candidates = evaluate_change_candidates(used_values, thresholds)
    selected_change = select_change_point(change_candidates, thresholds)
    if selected_change is not None:
        change_candidates = [
            replace(candidate, selected=candidate.k == selected_change.k)
            for candidate in change_candidates
        ]
        current_length = selected_change.right_points
        current_evaluation = selected_change.right_trend

    current_exists = current_evaluation is not None and current_evaluation.trend_exists
    current_length_value = int(current_length) if current_length is not None else 0
    current_start_date = (
        used_dates[used_points - current_length_value] if current_exists else None
    )
    current_end_date = used_dates[-1] if current_exists else None
    max_f = (
        max(candidate.f_statistic for candidate in change_candidates)
        if change_candidates
        else math.nan
    )
    if selected_change is None:
        change_fields = {
            "change_detected": False,
            "change_type": NO_DIRECTION,
            "change_k": None,
            "change_left_end_date": None,
            "change_right_start_date": None,
            "trend_before_direction": NO_DIRECTION,
            "trend_after_direction": NO_DIRECTION,
            "trend_before_slope": math.nan,
            "trend_after_slope": math.nan,
            "change_f": math.nan,
        }
    else:
        change_type = (
            "GROWTH_TO_DECLINE"
            if selected_change.left_trend.direction == GROWTH
            else "DECLINE_TO_GROWTH"
        )
        change_fields = {
            "change_detected": True,
            "change_type": change_type,
            "change_k": selected_change.k,
            "change_left_end_date": used_dates[selected_change.k - 1],
            "change_right_start_date": used_dates[selected_change.k],
            "trend_before_direction": selected_change.left_trend.direction,
            "trend_after_direction": selected_change.right_trend.direction,
            "trend_before_slope": selected_change.left_trend.slope,
            "trend_after_slope": selected_change.right_trend.slope,
            "change_f": selected_change.f_statistic,
        }

    summary = {
        **metadata,
        "status": "TREND" if current_exists else "NO_TREND",
        "history_points_raw": raw_points,
        "history_points_used": used_points,
        "leading_zero_points_trimmed": raw_points - used_points,
        "current_trend_exists": bool(current_exists),
        "current_trend_direction": (
            current_evaluation.direction if current_exists else NO_DIRECTION
        ),
        "current_trend_length": current_length_value,
        "current_trend_start_date": current_start_date,
        "current_trend_end_date": current_end_date,
        # ``abs`` означает абсолютные единицы GMV, знак наклона сохраняется.
        "trend_slope_abs": (
            current_evaluation.slope if current_exists else math.nan
        ),
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
        **change_fields,
        "max_f": max_f,
    }

    window_rows: List[Dict[str, object]] = []
    for window in windows:
        evaluation = window.evaluation
        window_rows.append(
            {
                **metadata,
                "window_length": window.window_length,
                "window_start_date": used_dates[-window.window_length],
                "window_end_date": used_dates[-1],
                **asdict(evaluation),
            }
        )

    change_rows: List[Dict[str, object]] = []
    for candidate in change_candidates:
        change_rows.append(
            {
                **metadata,
                "k": candidate.k,
                "left_points": candidate.left_points,
                "right_points": candidate.right_points,
                "change_left_end_date": used_dates[candidate.k - 1],
                "change_right_start_date": used_dates[candidate.k],
                "rss_single": candidate.rss_single,
                "rss_piecewise": candidate.rss_piecewise,
                "F": candidate.f_statistic,
                "left_trend_exists": candidate.left_trend.trend_exists,
                "left_direction": candidate.left_trend.direction,
                "left_slope": candidate.left_trend.slope,
                "right_trend_exists": candidate.right_trend.trend_exists,
                "right_direction": candidate.right_trend.direction,
                "right_slope": candidate.right_trend.slope,
                "passes_f_threshold": candidate.passes_f_threshold,
                "passes_direction_change": candidate.passes_direction_change,
                "valid_change": candidate.valid_change,
                "selected": candidate.selected,
            }
        )
    return SegmentTrendAnalysis(
        summary=summary,
        window_diagnostics=tuple(window_rows),
        change_diagnostics=tuple(change_rows),
    )


TREND_SUMMARY_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "status",
    "history_points_raw",
    "history_points_used",
    "leading_zero_points_trimmed",
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
    "first_change",
    "passes_initial_direction",
    "trend_to_noise",
    "change_detected",
    "change_type",
    "change_k",
    "change_left_end_date",
    "change_right_start_date",
    "trend_before_direction",
    "trend_after_direction",
    "trend_before_slope",
    "trend_after_slope",
    "change_f",
    "max_f",
)

TREND_WINDOW_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "window_length",
    "window_start_date",
    "window_end_date",
    "points",
    "trend_exists",
    "direction",
    "slope",
    "relative_slope",
    "total_change",
    "direction_count_share",
    "direction_movement_share",
    "trend_to_noise",
    "typical_scale",
    "intercept",
    "residual_mad",
    "noise_scale",
    "noise_scale_source",
    "status",
)

TREND_CHANGE_COLUMNS = (
    "segment_id",
    "segment_key",
    "segment_level",
    "slice_depth",
    "k",
    "left_points",
    "right_points",
    "change_left_end_date",
    "change_right_start_date",
    "rss_single",
    "rss_piecewise",
    "F",
    "left_trend_exists",
    "left_direction",
    "left_slope",
    "right_trend_exists",
    "right_direction",
    "right_slope",
    "passes_f_threshold",
    "passes_direction_change",
    "valid_change",
    "selected",
)


def _ordered_frame(
    rows: Sequence[Dict[str, object]],
    leading_columns: Sequence[str],
) -> pd.DataFrame:
    """Собрать DataFrame с фиксированным началом схемы и без потери полей.

    Args:
        rows: Итоговые словари.
        leading_columns: Колонки, которые должны идти первыми.

    Returns:
        DataFrame с устойчивым порядком колонок, включая пустой результат.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _ordered_frame([], ['segment_id']).columns.tolist()
        ['segment_id']
    """

    frame = pd.DataFrame(list(rows))
    for column in leading_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    remaining = [column for column in frame.columns if column not in leading_columns]
    return frame[[*leading_columns, *remaining]]


# [ADDED] Публичная batch-функция возвращает три независимые QA-таблицы.
def build_trend_analysis(
    panel_df: pd.DataFrame,
    dates: Sequence[int],
    thresholds: Optional[TrendThresholds] = None,
) -> Dict[str, pd.DataFrame]:
    """Построить независимый трендовый анализ для всех сегментов панели.

    Args:
        panel_df: Готовая полная панель из ``build_full_week_grid``.
        dates: Полная упорядоченная ось из total-слоя.
        thresholds: Пороги трендового MVP.

    Returns:
        Словарь из ``trend_summary``, ``trend_window_diagnostics`` и
        ``trend_change_diagnostics``.

    Raises:
        ValueError: Если панель пуста или нарушает контракт полной сетки.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [100, 110, 120, 130]})
        >>> len(build_trend_analysis(panel, [1, 2, 3, 4])['trend_summary'])
        1
    """

    thresholds = thresholds or TrendThresholds()
    if panel_df.empty:
        raise ValueError("Нельзя анализировать пустую panel_df")
    required = {"segment_id", "cal_date", "gmv"}
    missing = sorted(required - set(panel_df.columns))
    if missing:
        raise ValueError(f"Для trend analysis не хватает колонок: {missing}")
    summaries: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    change_rows: List[Dict[str, object]] = []
    for _, segment_panel in panel_df.groupby("segment_id", sort=True, dropna=False):
        result = analyze_segment_trend(segment_panel, dates, thresholds)
        summaries.append(result.summary)
        window_rows.extend(result.window_diagnostics)
        change_rows.extend(result.change_diagnostics)
    return {
        "trend_summary": _ordered_frame(summaries, TREND_SUMMARY_COLUMNS),
        "trend_window_diagnostics": _ordered_frame(
            window_rows,
            TREND_WINDOW_COLUMNS,
        ),
        "trend_change_diagnostics": _ordered_frame(
            change_rows,
            TREND_CHANGE_COLUMNS,
        ),
    }


# [ADDED] Общая точка выбора сохраняет legacy-функцию отдельным публичным API.
def build_configured_trend_analysis(
    panel_df: pd.DataFrame,
    dates: Sequence[int],
    thresholds: Optional[TrendThresholds] = None,
    model_config: Optional[TrendModelConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Запустить выбранный пользователем способ поиска текущего тренда.

    Args:
        panel_df: Готовая полная панель из ``build_full_week_grid``.
        dates: Полная упорядоченная ось из total-слоя.
        thresholds: Неизменённые бизнес-пороги ``evaluate_trend``.
        model_config: Selector модели и параметры Most Recent CP.

    Returns:
        Legacy-таблицы либо собственные диагностические таблицы Most Recent CP.

    Raises:
        ValueError: Если конфигурация или панель нарушает контракт.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['s'] * 4, 'cal_date': [1, 2, 3, 4], 'gmv': [100, 110, 120, 130]})
        >>> build_configured_trend_analysis(panel, [1, 2, 3, 4])['trend_summary'].iloc[0]['current_trend_direction']
        'GROWTH'
    """

    thresholds = thresholds or TrendThresholds()
    model_config = model_config or TrendModelConfig()
    if model_config.trend_search_method == "legacy":
        return build_trend_analysis(panel_df, dates, thresholds)

    # Локальный импорт разрывает цикл: новый математический модуль переиспользует
    # TrendThresholds, trim_leading_zero_history и evaluate_trend отсюда.
    from .trend_most_recent_cp import build_most_recent_cp_trend_analysis

    return build_most_recent_cp_trend_analysis(
        panel_df,
        dates,
        thresholds,
        model_config,
    )


__all__ = [
    "DECLINE",
    "GROWTH",
    "MOST_RECENT_CP_COSTS",
    "NO_DIRECTION",
    "TREND_SEARCH_METHODS",
    "LinearModelFit",
    "SegmentTrendAnalysis",
    "TrendChangeEvaluation",
    "TrendEvaluation",
    "TrendModelConfig",
    "TrendThresholds",
    "TrendWindowEvaluation",
    "analyze_segment_trend",
    "build_configured_trend_analysis",
    "build_trend_analysis",
    "evaluate_change_candidates",
    "evaluate_suffix_trends",
    "evaluate_trend",
    "fit_continuous_piecewise_model",
    "fit_single_linear_model",
    "select_change_point",
    "theil_sen_slope",
    "trim_leading_zero_history",
]
