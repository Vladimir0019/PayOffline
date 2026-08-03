"""Расчёт необычности GMV-сегментов и подготовка веса для оптимизации.

Модуль считает robust z-score относительного WoW-изменения GMV, материальность
сегмента, lifecycle-состояние, атомарное покрытие и итоговый ``anomaly_score``,
который затем максимизирует Set Packing.

Формулы и их обоснование: ``docs/anomaly_scoring.py.md``.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import AnomalyThresholds, METRIC_COLUMNS, RatioMetricSpec
from .data_preparation import candidate_covers_atomic


def _safe_float(value: object, default: float = 0.0) -> float:
    """Преобразовать значение в float с безопасным fallback.

    Args:
        value: Исходное значение.
        default: Значение, возвращаемое при ошибке.

    Returns:
        Число float.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _safe_float("10")
        10.0
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _history_reliability(nonzero_weeks: int) -> float:
    """Оценить надёжность истории сегмента.

    Чем меньше у сегмента ненулевых исторических недель, тем менее устойчивы
    его baseline и MAD, поэтому score домножается на понижающий коэффициент.

    Args:
        nonzero_weeks: Количество ненулевых исторических недель до текущей.

    Returns:
        Коэффициент надёжности истории в диапазоне (0, 1].

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _history_reliability(8)
        1.0
        >>> _history_reliability(1)
        0.4
        >>> _history_reliability(0)
        0.1
    """

    if nonzero_weeks >= 8:
        return 1.0
    if nonzero_weeks >= 4:
        return 0.7
    if nonzero_weeks >= 1:
        return 0.4
    # FIXED: Ранее ветка возвращала те же 0.4, что и «1-3 недели», поэтому
    # сегмент вообще без истории считался таким же надёжным, как сегмент с
    # тремя неделями. Для полностью новой истории baseline и MAD не определены,
    # поэтому коэффициент понижен до 0.1: такой сегмент должен попадать в
    # аномалии только при действительно большом движении GMV.
    return 0.1


def calculate_segment_anomaly(
    segment_panel: pd.DataFrame,
    total_by_date: pd.Series,
    dates: Sequence[int],
    current_cal_date: int,
    thresholds: AnomalyThresholds,
) -> Dict[str, object]:
    """Посчитать метрики необычности одного сегмента.

    Args:
        segment_panel: Недельная панель одного сегмента.
        total_by_date: Total GMV по неделям.
        dates: Полный список недель.
        current_cal_date: Анализируемая текущая неделя.
        thresholds: Пороги алгоритма.

    Returns:
        Словарь метрик сегмента.

    Raises:
        ValueError: Если текущая неделя первая в истории.

    Examples:
        >>> # metrics = calculate_segment_anomaly(panel_one_segment, total, dates, dates[-1], AnomalyThresholds())
    """

    # ADDED: Техническое предусловие формулы robust z-score.
    if not math.isfinite(float(thresholds.sigma_floor)) or thresholds.sigma_floor <= 0:
        raise ValueError("sigma_floor должен быть конечным положительным числом")

    current_idx = list(dates).index(int(current_cal_date))
    if current_idx == 0:
        raise ValueError("Текущая неделя не может быть первой неделей истории")
    previous_cal_date = int(dates[current_idx - 1])

    segment_panel = segment_panel.sort_values("cal_date")
    gmv = segment_panel.set_index("cal_date")["gmv"].astype(float).reindex(dates, fill_value=0.0)
    total = total_by_date.reindex(dates).astype(float)
    share = gmv / total
    share_delta = share.diff()
    gmv_delta = gmv.diff()
    relative_growth = gmv_delta / gmv.shift(1)
    relative_growth = relative_growth.where(gmv.shift(1) > 0)

    # FIXED: z-score считаем по истории относительных приростов GMV сегмента, а не по изменению доли.
    history_growth = relative_growth.reindex(dates[1:current_idx]).dropna()
    baseline_growth = _safe_float(history_growth.median(), 0.0) if not history_growth.empty else 0.0
    mad = _safe_float((history_growth - baseline_growth).abs().median(), 0.0) if not history_growth.empty else 0.0
    robust_sigma = 1.4826 * mad
    z_uses_sigma_floor = robust_sigma < thresholds.sigma_floor
    sigma = max(robust_sigma, thresholds.sigma_floor)
    # ADDED: Явно показываем, каким источником определён масштаб z-score.
    z_scale_source = "SIGMA_FLOOR" if z_uses_sigma_floor else "MAD"

    previous_share = _safe_float(share.loc[previous_cal_date], 0.0)
    current_gmv = _safe_float(gmv.loc[current_cal_date], 0.0)
    previous_gmv = _safe_float(gmv.loc[previous_cal_date], 0.0)
    wow_delta_gmv = current_gmv - previous_gmv

    history_nonzero_weeks = int((gmv.reindex(dates[:current_idx]) > 0).sum())
    reliability = _history_reliability(history_nonzero_weeks)

    if current_gmv > 0 and history_nonzero_weeks == 0:
        state = "новый сегмент"
    elif current_gmv > 0 and previous_gmv == 0 and history_nonzero_weeks > 0:
        state = "возобновившийся сегмент"
    elif current_gmv == 0 and previous_gmv > 0:
        state = "исчезнувший сегмент"
    else:
        state = "обычный"

    relative_wow = None
    if previous_gmv > 0:
        relative_wow = wow_delta_gmv / previous_gmv

    # FIXED: WoW ratio-метрик остаётся неопределённым, если хотя бы одна сторона
    # сравнения NULL. Для обязательных аддитивных метрик значения конечны после
    # входной валидации.
    metric_values: Dict[str, object] = {}
    ratio_metrics = {"aov", "tpm", "freq"}
    for metric in METRIC_COLUMNS:
        if metric not in segment_panel.columns:
            continue
        metric_series = segment_panel.set_index("cal_date")[metric].astype(float).reindex(dates)
        previous_raw = metric_series.loc[previous_cal_date]
        current_raw = metric_series.loc[current_cal_date]
        if metric in ratio_metrics and (pd.isna(previous_raw) or pd.isna(current_raw)):
            previous_metric = math.nan if pd.isna(previous_raw) else float(previous_raw)
            current_metric = math.nan if pd.isna(current_raw) else float(current_raw)
            metric_wow = math.nan
        else:
            previous_metric = _safe_float(previous_raw, 0.0)
            current_metric = _safe_float(current_raw, 0.0)
            metric_wow = (
                math.nan
                if previous_metric == 0
                else (current_metric - previous_metric) / previous_metric
            )
        metric_values[f"{metric}_previous"] = previous_metric
        metric_values[f"{metric}_current"] = current_metric
        metric_values[f"{metric}_wow_pct"] = metric_wow

    if relative_wow is None:
        robust_z = (
            thresholds.lifecycle_z_score
            if state != "обычный" and wow_delta_gmv != 0
            else 0.0
        )
    else:
        robust_z = (relative_wow - baseline_growth) / sigma

    return {
        "current_cal_date": int(current_cal_date),
        "previous_cal_date": int(previous_cal_date),
        "gmv_current": current_gmv,
        "gmv_previous": previous_gmv,
        "wow_delta_gmv": wow_delta_gmv,
        "relative_wow": relative_wow,
        "share_current": _safe_float(share.loc[current_cal_date], 0.0),
        "share_previous": previous_share,
        "share_delta_current": _safe_float(share_delta.loc[current_cal_date], 0.0),
        "baseline_relative_growth": baseline_growth,
        "mad_relative_growth": mad,
        "sigma_relative_growth": sigma,
        "z_scale_source": z_scale_source,
        "z_uses_sigma_floor": z_uses_sigma_floor,
        "robust_z": robust_z,
        "abs_robust_z": abs(robust_z),
        "history_nonzero_weeks": history_nonzero_weeks,
        "history_points": int(len(history_growth)),
        "reliability_factor": reliability,
        "state": state,
        **metric_values,
    }


def build_anomaly_candidates(
    panel_df: pd.DataFrame,
    dim_cols: Sequence[str],
    dates: Sequence[int],
    thresholds: AnomalyThresholds,
    current_cal_date: Optional[int] = None,
    coverage: Optional[Dict[str, frozenset[str]]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Посчитать аномальность независимо для каждого среза.

    Args:
        panel_df: Полная недельная панель segment x week.
        dim_cols: Список признаков.
        dates: Список недель.
        thresholds: Пороги алгоритма.
        current_cal_date: Текущая неделя. Если None, берётся последняя.
        coverage: Уже рассчитанное атомарное покрытие ``segment_id -> атомы``.
            Если None, покрытие считается внутри сверки иерархии. 

    Returns:
        Кортеж: таблица кандидатов и total GMV по неделям.

    Raises:
        ValueError: Если текущая неделя отсутствует.

    Examples:
        >>> # candidates, total = build_anomaly_candidates(panel, dims, dates, AnomalyThresholds())
    """

    current = int(current_cal_date) if current_cal_date is not None else int(dates[-1])
    if current not in dates:
        raise ValueError(f"Текущая неделя {current} отсутствует в total-слое")

    # ADDED: Останавливаем расчёт до scoring, если иерархическая витрина не
    # согласуется с физическим атомарным слоем хотя бы на одной дате.
    validate_hierarchy_reconciliation(
        panel_df,
        dim_cols,
        dates,
        thresholds.hierarchy_reconciliation_abs_tolerance,
        coverage=coverage,
    )

    total_panel = panel_df[panel_df["slice_depth"].astype(int) == 0].copy()
    total_by_date = total_panel.groupby("cal_date")["gmv"].sum().reindex(dates).astype(float)
    if total_by_date.isna().any() or (total_by_date <= 0).any():
        raise ValueError("Total GMV должен присутствовать и быть положительным на каждой неделе")

    rows: List[Dict[str, object]] = []
    meta_cols = ["segment_id", "segment_key", "segment_level", "slice_depth", *dim_cols]
    for segment_id, segment_panel in panel_df.groupby("segment_id", sort=False):
        meta = segment_panel.iloc[0][meta_cols].to_dict()
        metrics = calculate_segment_anomaly(segment_panel, total_by_date, dates, current, thresholds)
        rows.append({**meta, **metrics})

    candidates = pd.DataFrame(rows)
    candidates["slice_depth"] = candidates["slice_depth"].astype(int)
    # ADDED: Материальность считаем как долю изменения сегмента в gross movement атомарного слоя.
    max_depth = int(candidates["slice_depth"].max())
    atomic_candidates = candidates[candidates["slice_depth"] == max_depth].copy()
    gross_atomic_movement = float(atomic_candidates["wow_delta_gmv"].astype(float).abs().sum())
    candidates["gross_atomic_movement"] = gross_atomic_movement
    candidates["materiality_share"] = 0.0
    if gross_atomic_movement > 0:
        candidates["materiality_share"] = candidates["wow_delta_gmv"].astype(float).abs() / gross_atomic_movement
    # ADDED: Предварительный фильтр аномальности. Неаномальные или нематериальные сегменты
    # остаются в диагностике, но не участвуют в оптимизационном выборе.
    candidates["passes_initial_anomaly_filter"] = (
        (candidates["slice_depth"].astype(int) > 0)
        & (candidates["abs_robust_z"].astype(float) >= thresholds.min_z_score)
        & (candidates["materiality_share"].astype(float) >= thresholds.min_materiality_share)
        & (candidates["wow_delta_gmv"].astype(float).abs() >= thresholds.min_anomaly_abs)
    )
    return candidates.sort_values(["slice_depth", "segment_key"]).reset_index(drop=True), total_by_date


def calculate_ratio_segment_anomaly(
    segment_panel: pd.DataFrame,
    dates: Sequence[int],
    current_cal_date: int,
    thresholds: AnomalyThresholds,
    spec: RatioMetricSpec,
) -> Dict[str, object]:
    """ADDED: Посчитать аномальность одной доли для одного сегмента.

    Args:
        segment_panel: Недельная панель одного сегмента.
        dates: Полная календарная ось.
        current_cal_date: Анализируемая неделя.
        thresholds: Пороги алгоритма.
        spec: Контракт долевой метрики.

    Returns:
        Словарь метрик для scoring и диагностики.

    Raises:
        ValueError: Если не поддержан режим изменения или текущая неделя первая.

    Examples:
        >>> # result = calculate_ratio_segment_anomaly(panel, dates, dates[-1], thresholds, spec)
    """

    if spec.change_mode != "absolute_delta":
        raise ValueError(
            f"Режим change_mode={spec.change_mode!r} пока не поддерживается"
        )
    current_idx = list(dates).index(int(current_cal_date))
    if current_idx == 0:
        raise ValueError("Текущая неделя не может быть первой неделей истории")
    previous_cal_date = int(dates[current_idx - 1])
    indexed = segment_panel.sort_values("cal_date").set_index("cal_date")
    metric_value = indexed[spec.value_column].astype(float).reindex(dates)
    numerator = indexed[spec.numerator_column].astype(float).reindex(dates, fill_value=0.0)
    denominator = indexed[spec.denominator_column].astype(float).reindex(dates, fill_value=0.0)
    gmv = indexed["gmv"].astype(float).reindex(dates, fill_value=0.0)
    metric_change = metric_value.diff()

    history_changes = metric_change.reindex(dates[1:current_idx]).dropna()
    baseline = _safe_float(history_changes.median(), 0.0) if not history_changes.empty else 0.0
    mad = _safe_float((history_changes - baseline).abs().median(), 0.0) if not history_changes.empty else 0.0
    robust_sigma = 1.4826 * mad
    z_uses_sigma_floor = robust_sigma < thresholds.sigma_floor
    sigma = max(robust_sigma, thresholds.sigma_floor)

    current_value_raw = metric_value.loc[current_cal_date]
    previous_value_raw = metric_value.loc[previous_cal_date]
    metric_valid_for_scoring = bool(
        pd.notna(current_value_raw)
        and pd.notna(previous_value_raw)
        and math.isfinite(float(current_value_raw))
        and math.isfinite(float(previous_value_raw))
    )
    current_delta = (
        float(current_value_raw) - float(previous_value_raw)
        if metric_valid_for_scoring
        else math.nan
    )
    robust_z = (
        (current_delta - baseline) / sigma
        if metric_valid_for_scoring
        else 0.0
    )

    numerator_current = _safe_float(numerator.loc[current_cal_date], 0.0)
    numerator_previous = _safe_float(numerator.loc[previous_cal_date], 0.0)
    denominator_current = _safe_float(denominator.loc[current_cal_date], 0.0)
    denominator_previous = _safe_float(denominator.loc[previous_cal_date], 0.0)
    earlier_nonzero_numerator_weeks = int(
        (numerator.reindex(dates[: current_idx - 1]) > 0.0).sum()
    )
    if numerator_current > 0.0 and int((numerator.reindex(dates[:current_idx]) > 0.0).sum()) == 0:
        state = "новый"
    elif numerator_current > 0.0 and numerator_previous == 0.0 and earlier_nonzero_numerator_weeks > 0:
        state = "возобновившийся"
    elif numerator_current == 0.0 and numerator_previous > 0.0:
        state = "исчезнувший"
    else:
        state = "обычный"

    history_nonzero_weeks = int((gmv.reindex(dates[:current_idx]) > 0.0).sum())
    mean_denominator = (denominator_current + denominator_previous) / 2.0
    hierarchy_movement = (
        current_delta * mean_denominator
        if metric_valid_for_scoring
        else math.nan
    )
    row_missing_current = bool(
        indexed.get(
            "row_missing_in_source",
            pd.Series(False, index=indexed.index),
        ).reindex(dates, fill_value=False).loc[current_cal_date]
    )
    if row_missing_current:
        metric_status = "METRIC_ROW_MISSING_IN_SOURCE"
    elif denominator_current == 0.0:
        metric_status = "METRIC_UNDEFINED_ZERO_DENOMINATOR"
    elif not metric_valid_for_scoring:
        metric_status = "METRIC_UNDEFINED_COMPARISON"
    else:
        metric_status = "METRIC_VALID"
    return {
        "metric_name": spec.name,
        "change_mode": spec.change_mode,
        "current_cal_date": int(current_cal_date),
        "previous_cal_date": previous_cal_date,
        "metric_value_current": float(current_value_raw) if pd.notna(current_value_raw) else math.nan,
        "metric_value_previous": float(previous_value_raw) if pd.notna(previous_value_raw) else math.nan,
        "metric_delta": current_delta,
        "metric_delta_pp": current_delta * 100.0 if metric_valid_for_scoring else math.nan,
        "numerator_current": numerator_current,
        "numerator_previous": numerator_previous,
        "denominator_current": denominator_current,
        "denominator_previous": denominator_previous,
        "mean_denominator": mean_denominator,
        "hierarchy_movement": hierarchy_movement,
        # ADDED: Технический alias сохраняет контракт Set Packing; это не GMV.
        "wow_delta_gmv": hierarchy_movement,
        "baseline_metric_delta": baseline,
        "mad_metric_delta": mad,
        "sigma_metric_delta": sigma,
        "z_scale_source": "SIGMA_FLOOR" if z_uses_sigma_floor else "MAD",
        "z_uses_sigma_floor": z_uses_sigma_floor,
        "robust_z": robust_z,
        "abs_robust_z": abs(robust_z),
        "metric_valid_for_scoring": metric_valid_for_scoring,
        "metric_status": metric_status,
        "history_points": int(len(history_changes)),
        "history_nonzero_weeks": history_nonzero_weeks,
        "reliability_factor": _history_reliability(history_nonzero_weeks),
        "state": state,
    }


def build_ratio_anomaly_candidates(
    panel_df: pd.DataFrame,
    dim_cols: Sequence[str],
    dates: Sequence[int],
    thresholds: AnomalyThresholds,
    spec: RatioMetricSpec,
    current_cal_date: Optional[int] = None,
    coverage: Optional[Dict[str, frozenset[str]]] = None,
) -> pd.DataFrame:
    """ADDED: Построить кандидатов одной долевой метрики.

    Args:
        panel_df: Полная недельная панель.
        dim_cols: Иерархические признаки.
        dates: Полная календарная ось.
        thresholds: Пороги алгоритма.
        spec: Контракт одной метрики.
        current_cal_date: Анализируемая неделя.
        coverage: Готовое атомарное покрытие.

    Returns:
        Все сегменты со scoring, materiality и структурным статусом.

    Raises:
        ValueError: Если нет колонок метрики или нарушена аддитивная иерархия.

    Examples:
        >>> # candidates = build_ratio_anomaly_candidates(panel, dims, dates, thresholds, spec)
    """

    required = {spec.value_column, spec.numerator_column, spec.denominator_column, "gmv"}
    missing = sorted(required - set(panel_df.columns))
    if missing:
        raise ValueError(f"Для метрики {spec.name!r} не хватает колонок: {missing}")
    current = int(current_cal_date) if current_cal_date is not None else int(dates[-1])
    for additive_column in (spec.numerator_column, spec.denominator_column):
        validate_hierarchy_reconciliation(
            panel_df,
            dim_cols,
            dates,
            thresholds.hierarchy_reconciliation_abs_tolerance,
            coverage=coverage,
            value_column=additive_column,
        )

    rows: List[Dict[str, object]] = []
    meta_cols = ["segment_id", "segment_key", "segment_level", "slice_depth", *dim_cols]
    for _, segment_panel in panel_df.groupby("segment_id", sort=False):
        meta = segment_panel.iloc[0][meta_cols].to_dict()
        metrics = calculate_ratio_segment_anomaly(
            segment_panel, dates, current, thresholds, spec
        )
        rows.append({**meta, **metrics})
    candidates = pd.DataFrame(rows)
    candidates["slice_depth"] = candidates["slice_depth"].astype(int)
    max_depth = int(candidates["slice_depth"].max())
    atomic = candidates[candidates["slice_depth"].eq(max_depth)]
    atomic_numerator_total = float(atomic["numerator_current"].astype(float).sum())
    candidates["atomic_numerator_total"] = atomic_numerator_total
    candidates["materiality_share"] = 0.0
    if atomic_numerator_total > 0.0:
        candidates["materiality_share"] = (
            candidates["numerator_current"].astype(float) / atomic_numerator_total
        )
    candidates["passes_initial_anomaly_filter"] = (
        candidates["slice_depth"].gt(0)
        & (atomic_numerator_total > 0.0)
        & candidates["metric_valid_for_scoring"].eq(True)
        & candidates["abs_robust_z"].astype(float).ge(thresholds.min_z_score)
        & candidates["materiality_share"].astype(float).ge(
            thresholds.min_materiality_share
        )
    )
    return candidates.sort_values(["slice_depth", "segment_key"]).reset_index(drop=True)


def build_atomic_coverage(candidates: pd.DataFrame, dim_cols: Sequence[str]) -> Dict[str, frozenset[str]]:
    """Построить покрытие каждого кандидата атомарными сегментами.

    Args:
        candidates: Таблица кандидатов.
        dim_cols: Список признаков.

    Returns:
        Словарь segment_id -> frozenset атомарных segment_id.

    Raises:
        ValueError: Если атомарный слой пустой.

    Examples:
        >>> # coverage = build_atomic_coverage(candidates, ['geo', 'products'])
    """

    max_depth = int(candidates["slice_depth"].max())
    atomic_df = candidates[candidates["slice_depth"] == max_depth].copy()
    if atomic_df.empty:
        raise ValueError("Атомарный слой пустой")

    coverage: Dict[str, frozenset[str]] = {}
    for _, candidate in candidates.iterrows():
        atoms = []
        for _, atom in atomic_df.iterrows():
            if candidate_covers_atomic(candidate, atom, dim_cols):
                atoms.append(str(atom["segment_id"]))
        coverage[str(candidate["segment_id"])] = frozenset(atoms)
    return coverage


def validate_hierarchy_reconciliation(
    panel_df: pd.DataFrame,
    dim_cols: Sequence[str],
    dates: Sequence[int],
    absolute_tolerance: float = 1e-4,
    coverage: Optional[Dict[str, frozenset[str]]] = None,
    value_column: str = "gmv",
) -> None:
    """ADDED: Сверить аддитивную метрику родителя с покрытыми атомами.

    Args:
        panel_df: Полная панель ``segment_id x cal_date``.
        dim_cols: Иерархические признаки сегмента.
        dates: Полный список анализируемых дат.
        absolute_tolerance: Допустимая абсолютная ошибка сверки GMV.
        coverage: Готовое атомарное покрытие. Если None, считается здесь.
        value_column: Имя сверяемой аддитивной колонки.

    Returns:
        None.

    Raises:
        ValueError: Если контракт панели нарушен, coverage пусто или метрика
            родителя не совпадает с суммой атомов хотя бы на одной дате.

    Examples:
        >>> # validate_hierarchy_reconciliation(panel, ['geo'], [1, 8], 1e-4)
    """

    tolerance = float(absolute_tolerance)
    value_label = "GMV" if value_column == "gmv" else value_column
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "hierarchy_reconciliation_abs_tolerance должен быть конечным "
            "неотрицательным числом"
        )

    required_columns = {
        "segment_id",
        "segment_key",
        "slice_depth",
        "cal_date",
        value_column,
        *dim_cols,
    }
    missing_columns = sorted(required_columns - set(panel_df.columns))
    if missing_columns:
        raise ValueError(
            "Для сверки иерархии не хватает колонок: "
            f"{missing_columns}"
        )
    if panel_df.empty:
        raise ValueError("Нельзя сверить пустую иерархическую панель")

    metadata_columns = [
        "segment_id",
        "segment_key",
        "slice_depth",
        *dim_cols,
    ]
    segment_metadata = (
        panel_df[metadata_columns]
        .drop_duplicates(subset=["segment_id"])
        .reset_index(drop=True)
    )
    # FIXED: Покрытие зависит только от metadata сегментов, поэтому повторный
    # расчёт не нужен, если вызывающий код уже его посчитал.
    if coverage is None:
        coverage = build_atomic_coverage(segment_metadata, dim_cols)
    max_depth = int(segment_metadata["slice_depth"].astype(int).max())
    parent_rows = segment_metadata[
        segment_metadata["slice_depth"].astype(int) < max_depth
    ]

    try:
        value_by_segment_date = panel_df.pivot(
            index="segment_id",
            columns="cal_date",
            values=value_column,
        ).reindex(columns=list(dates))
    except ValueError as exc:
        raise ValueError(
            "Панель содержит дубли segment_id x cal_date; сверка иерархии невозможна"
        ) from exc
    value_by_segment_date.index = value_by_segment_date.index.astype(str)
    value_by_segment_date = value_by_segment_date.apply(
        pd.to_numeric,
        errors="coerce",
    )
    if value_by_segment_date.isna().any().any():
        raise ValueError(
            f"Панель содержит пропущенный или нечисловой {value_label} "
            "после построения полной сетки"
        )

    for _, parent in parent_rows.iterrows():
        parent_id = str(parent["segment_id"])
        atomic_ids = sorted(coverage.get(parent_id, frozenset()))
        if not atomic_ids:
            raise ValueError(
                "Иерархическая сверка не найдена: родитель не покрывает атомы; "
                f"segment_id={parent_id}, segment_key={parent['segment_key']}"
            )
        missing_atomic_ids = sorted(
            set(atomic_ids) - set(value_by_segment_date.index)
        )
        if missing_atomic_ids:
            raise ValueError(
                "Иерархическая сверка невозможна: атомы отсутствуют в панели; "
                f"segment_id={parent_id}, atomic_ids={missing_atomic_ids[:10]}"
            )

        parent_value = value_by_segment_date.loc[parent_id]
        atomic_value = value_by_segment_date.loc[atomic_ids].sum(axis=0)
        absolute_error = (parent_value - atomic_value).abs()
        failed_dates = absolute_error[absolute_error > tolerance]
        if not failed_dates.empty:
            failed_date = failed_dates.index[0]
            raise ValueError(
                f"Нарушена сверка {value_label} родителя с атомами максимальной глубины: "
                f"segment_id={parent_id}, segment_key={parent['segment_key']}, "
                f"cal_date={failed_date}, parent_value={float(parent_value.loc[failed_date])}, "
                f"atomic_value_sum={float(atomic_value.loc[failed_date])}, "
                f"absolute_error={float(absolute_error.loc[failed_date])}, "
                f"tolerance={tolerance}"
            )


def _enumerate_disjoint_descendant_groups(
    descendant_ids: Sequence[str],
    coverage: Dict[str, frozenset[str]],
    max_descendants: int = 25,
) -> List[Tuple[str, ...]]:
    """ADDED: Физически перечислить все непустые непересекающиеся группы потомков.

    Сложность перебора экспоненциальна,поэтому число потомков ограничено:
    превышение лимита останавливает расчёт с диагностикой.

    Args:
        descendant_ids: Идентификаторы eligible-потомков одного родителя.
        coverage: Атомарное покрытие eligible-сегментов.
        max_descendants: Максимальное допустимое число потомков.

    Returns:
        Полный список непустых групп. Внутри каждой группы покрытия попарно
        не пересекаются; полное покрытие родителя не требуется.

    Raises:
        KeyError: Если для потомка отсутствует покрытие.
        ValueError: Если число потомков превышает ``max_descendants``.

    Examples:
        >>> groups = _enumerate_disjoint_descendant_groups(
        ...     ['a', 'b'], {'a': frozenset({'x'}), 'b': frozenset({'y'})}
        ... )
        >>> groups
        [('b',), ('a',), ('a', 'b')]
    """

    ordered_ids = tuple(sorted(str(segment_id) for segment_id in descendant_ids))
    # Fail-fast предохранитель экспоненциального перебора.
    if len(ordered_ids) > int(max_descendants):
        raise ValueError(
            "Слишком много eligible-потомков для полного перечисления групп: "
            f"{len(ordered_ids)} > max_hierarchy_descendants="
            f"{int(max_descendants)}. Перебор растёт как 2^n − 1; уточните "
            "первичный фильтр аномальности или повысьте лимит осознанно."
        )
    groups: List[Tuple[str, ...]] = []

    def enumerate_from(
        position: int,
        selected_ids: Tuple[str, ...],
        occupied_atoms: frozenset[str],
    ) -> None:
        """ADDED: Выполнить полный include/exclude-перебор без pruning.

        Args:
            position: Позиция в упорядоченном списке потомков.
            selected_ids: Уже выбранные идентификаторы группы.
            occupied_atoms: Уже занятые атомы.

        Returns:
            None.

        Raises:
            KeyError: Если для рассматриваемого потомка отсутствует покрытие.

        Examples:
            >>> # Вызывается рекурсивно из внешней функции.
        """

        if position == len(ordered_ids):
            if selected_ids:
                groups.append(selected_ids)
            return

        segment_id = ordered_ids[position]
        enumerate_from(position + 1, selected_ids, occupied_atoms)
        segment_atoms = coverage[segment_id]
        if segment_atoms.isdisjoint(occupied_atoms):
            enumerate_from(
                position + 1,
                (*selected_ids, segment_id),
                occupied_atoms.union(segment_atoms),
            )

    enumerate_from(0, tuple(), frozenset())
    return groups


def apply_hierarchy_score_adjustment(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    aggregation_bonus_lambda: float = 0.3,
    single_child_factor: float = 0.85,
    dominant_child_capture_threshold: float = 0.80,
    dominant_child_score_margin: float = 0.02,
    max_hierarchy_descendants: int = 25,
    movement_column: str = "wow_delta_gmv",
    allow_zero_movement: bool = False,
) -> pd.DataFrame:
    """ADDED: Скорректировать score по coherence сильнейшей группы потомков.

    Для каждого eligible-родителя физически перечисляются все непустые
    попарно непересекающиеся группы eligible-потомков любых более глубоких
    уровней. Неполное покрытие родителя разрешено. Сильнейшая группа
    максимизирует сумму уже скорректированных score потомков; расчёт идёт
    снизу вверх по глубине.

    Args:
        candidates: Таблица всех кандидатов до оптимизационного отбора.
        coverage: Атомарное покрытие `segment_id -> set[atomic_segment_id]`.
        aggregation_bonus_lambda: Наклон линейной корректировки coherence.
        single_child_factor: Коэффициент родителя при одном потомке в
            сильнейшей группе.
        dominant_child_capture_threshold: Минимальная доля абсолютного
            атомарного движения родителя, объяснённая единственным сильным
            потомком, для применения dominance cap.
        dominant_child_score_margin: Относительный запас, на который cap
            удерживает score родителя ниже score доминирующего потомка.
        max_hierarchy_descendants: Максимальное число eligible-потомков одного
            родителя для полного перечисления групп.
        movement_column: Аддитивное движение для hierarchy coherence.
        allow_zero_movement: Разрешить нулевое движение; нужно для долей,
            у которых z-score может быть аномальным при нулевой текущей дельте.

    Returns:
        Копия `candidates` с базовым и итоговым score, параметрами coherence,
        числом физически перечисленных групп и составом сильнейшей группы.

    Raises:
        ValueError: Если не хватает обязательных колонок, коэффициенты
            некорректны или eligible-сегмент имеет невалидные score,
            изменение GMV либо атомарное покрытие.

    Examples:
        >>> df = pd.DataFrame([
        ...     {'segment_id': 'p', 'slice_depth': 1, 'passes_initial_anomaly_filter': True,
        ...      'abs_robust_z': 2.0, 'materiality_share': 0.5,
        ...      'reliability_factor': 1.0, 'wow_delta_gmv': 100.0},
        ... ])
        >>> apply_hierarchy_score_adjustment(
        ...     df, {'p': frozenset({'a'})}
        ... )['hierarchy_score_factor'].iloc[0]
        np.float64(1.0)
    """

    required_columns = {
        "segment_id",
        "slice_depth",
        "passes_initial_anomaly_filter",
        "abs_robust_z",
        "materiality_share",
        "reliability_factor",
        movement_column,
    }
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(
            "Для apply_hierarchy_score_adjustment не хватает колонок: "
            f"{missing_columns}"
        )
    if not 0.0 <= float(aggregation_bonus_lambda) < 2.0:
        raise ValueError("aggregation_bonus_lambda должен быть в интервале [0, 2)")
    if not 0.0 < float(single_child_factor) <= 1.0:
        raise ValueError("single_child_factor должен быть в интервале (0, 1]")
    if not 0.0 <= float(dominant_child_capture_threshold) <= 1.0:
        raise ValueError(
            "dominant_child_capture_threshold должен быть в интервале [0, 1]"
        )
    if not 0.0 <= float(dominant_child_score_margin) < 1.0:
        raise ValueError(
            "dominant_child_score_margin должен быть в интервале [0, 1)"
        )

    result = candidates.copy()
    result["segment_id"] = result["segment_id"].astype(str)
    result["slice_depth"] = result["slice_depth"].astype(int)
    result["base_anomaly_score"] = (
        pd.to_numeric(result["abs_robust_z"], errors="coerce")
        * pd.to_numeric(result["materiality_share"], errors="coerce")
        * pd.to_numeric(result["reliability_factor"], errors="coerce")
    )
    result["hierarchy_eligible_descendant_count"] = 0
    result["hierarchy_group_count"] = 0
    result["hierarchy_best_group_size"] = 0
    # FIXED: Список технических ID хранится в однозначном JSON-массиве.
    result["hierarchy_best_group_ids_json"] = "[]"
    # ADDED: Однозначный состав сильнейшей группы для отчётного графа.
    result["hierarchy_best_group_segment_keys"] = "[]"
    result["hierarchy_best_group_score"] = 0.0
    result["hierarchy_direction_unity"] = math.nan
    result["hierarchy_dominant_share"] = math.nan
    result["hierarchy_balance_max"] = math.nan
    result["hierarchy_balance_effective"] = math.nan
    result["hierarchy_balance"] = math.nan
    result["hierarchy_coherence"] = math.nan
    # ADDED: Диагностика dominance cap для единственного сильного потомка.
    result["hierarchy_single_child_capture"] = math.nan
    result["hierarchy_single_child_direction_match"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )
    result["hierarchy_single_child_uncapped_score"] = math.nan
    result["hierarchy_dominance_cap_score"] = math.nan
    result["hierarchy_dominance_cap_applied"] = False
    result["hierarchy_score_factor"] = 1.0
    result["anomaly_score"] = result["base_anomaly_score"].astype(float)

    normalized_coverage = {
        str(segment_id): frozenset(str(atom_id) for atom_id in (atoms or frozenset()))
        for segment_id, atoms in coverage.items()
    }
    eligible_mask = (
        result["passes_initial_anomaly_filter"].eq(True)
        & result["slice_depth"].gt(0)
    )
    eligible = result[eligible_mask].copy()
    fatal_issues: List[str] = []
    coverage_by_id: Dict[str, frozenset[str]] = {}
    depth_by_id: Dict[str, int] = {}
    delta_by_id: Dict[str, float] = {}
    index_by_id: Dict[str, int] = {}
    all_index_by_id = {
        str(row["segment_id"]): int(index)
        for index, row in result.iterrows()
    }

    for index, row in eligible.iterrows():
        segment_id = str(row["segment_id"])
        segment_atoms = normalized_coverage.get(segment_id)
        if segment_atoms is None:
            fatal_issues.append(
                f"{segment_id} ({row.get('segment_key', '')}): "
                "отсутствует атомарное покрытие"
            )
            continue
        if not segment_atoms:
            fatal_issues.append(
                f"{segment_id} ({row.get('segment_key', '')}): "
                "пустое атомарное покрытие"
            )
            continue
        base_score = _safe_float(row.get("base_anomaly_score"), math.nan)
        movement = _safe_float(row.get(movement_column), math.nan)
        if not math.isfinite(base_score):
            fatal_issues.append(
                f"{segment_id} ({row.get('segment_key', '')}): "
                "некорректный base_anomaly_score"
            )
            continue
        if not math.isfinite(movement) or (
            movement == 0.0 and not allow_zero_movement
        ):
            fatal_issues.append(
                f"{segment_id} ({row.get('segment_key', '')}): "
                f"некорректный {movement_column}"
            )
            continue
        coverage_by_id[segment_id] = segment_atoms
        depth_by_id[segment_id] = int(row["slice_depth"])
        delta_by_id[segment_id] = float(movement)
        index_by_id[segment_id] = int(index)

    if fatal_issues:
        raise ValueError(
            "Нельзя рассчитать hierarchy score для eligible-сегментов: "
            + "; ".join(fatal_issues[:10])
        )

    if not coverage_by_id:
        return result

    atomic_depth = int(result["slice_depth"].max())
    parent_depths = sorted(
        {depth for depth in depth_by_id.values() if depth < atomic_depth},
        reverse=True,
    )
    for parent_depth in parent_depths:
        parent_ids = sorted(
            segment_id
            for segment_id, depth in depth_by_id.items()
            if depth == parent_depth
        )
        for parent_id in parent_ids:
            parent_atoms = coverage_by_id[parent_id]
            descendant_ids = sorted(
                child_id
                for child_id, child_atoms in coverage_by_id.items()
                if depth_by_id[child_id] > parent_depth
                and child_atoms.issubset(parent_atoms)
            )
            parent_index = index_by_id[parent_id]
            result.at[
                parent_index, "hierarchy_eligible_descendant_count"
            ] = int(len(descendant_ids))
            if not descendant_ids:
                continue

            groups = _enumerate_disjoint_descendant_groups(
                descendant_ids,
                coverage_by_id,
                max_descendants=max_hierarchy_descendants,
            )
            result.at[parent_index, "hierarchy_group_count"] = int(len(groups))
            group_records = []
            for group in groups:
                group_score = float(
                    sum(
                        float(result.at[index_by_id[child_id], "anomaly_score"])
                        for child_id in group
                    )
                )
                group_records.append((group_score, len(group), group))
            best_group_score, best_group_size, best_group = sorted(
                group_records,
                key=lambda item: (-item[0], item[1], item[2]),
            )[0]

            result.at[parent_index, "hierarchy_best_group_size"] = int(
                best_group_size
            )
            result.at[parent_index, "hierarchy_best_group_ids_json"] = (
                json.dumps(list(best_group), ensure_ascii=False)
            )
            result.at[parent_index, "hierarchy_best_group_segment_keys"] = (
                json.dumps(
                    [
                        str(result.at[index_by_id[child_id], "segment_key"])
                        for child_id in best_group
                    ],
                    ensure_ascii=False,
                )
            )
            result.at[parent_index, "hierarchy_best_group_score"] = float(
                best_group_score
            )

            base_parent_score = float(
                result.at[parent_index, "base_anomaly_score"]
            )
            if best_group_size == 1:
                child_id = best_group[0]
                uncapped_score = base_parent_score * float(
                    single_child_factor
                )
                cap_score = float(
                    result.at[index_by_id[child_id], "anomaly_score"]
                ) * (1.0 - float(dominant_child_score_margin))

                missing_atom_ids = sorted(
                    atom_id
                    for atom_id in parent_atoms
                    if atom_id not in all_index_by_id
                )
                if missing_atom_ids:
                    raise ValueError(
                        f"Для dominance cap родителя {parent_id} отсутствуют "
                        "строки атомов: "
                        + ", ".join(missing_atom_ids[:10])
                    )
                atomic_delta_by_id = {
                    atom_id: _safe_float(
                        result.at[
                            all_index_by_id[atom_id],
                            movement_column,
                        ],
                        math.nan,
                    )
                    for atom_id in parent_atoms
                }
                invalid_atom_ids = sorted(
                    atom_id
                    for atom_id, atom_delta in atomic_delta_by_id.items()
                    if not math.isfinite(atom_delta)
                )
                if invalid_atom_ids:
                    raise ValueError(
                        f"Для dominance cap родителя {parent_id} "
                        f"некорректный {movement_column} атомов: "
                        + ", ".join(invalid_atom_ids[:10])
                    )

                parent_gross_atomic_movement = float(
                    sum(abs(delta) for delta in atomic_delta_by_id.values())
                )
                if parent_gross_atomic_movement <= 0.0 and not allow_zero_movement:
                    raise ValueError(
                        f"Для dominance cap родителя {parent_id} абсолютное "
                        "движение атомов должно быть положительным"
                    )
                child_gross_atomic_movement = float(
                    sum(
                        abs(atomic_delta_by_id[atom_id])
                        for atom_id in coverage_by_id[child_id]
                    )
                )
                capture = (
                    min(
                        1.0,
                        child_gross_atomic_movement
                        / parent_gross_atomic_movement,
                    )
                    if parent_gross_atomic_movement > 0.0
                    else math.nan
                )
                direction_match = (
                    delta_by_id[parent_id] * delta_by_id[child_id] > 0.0
                )
                dominance_rule_matches = (
                    direction_match
                    and math.isfinite(capture)
                    and capture >= float(dominant_child_capture_threshold)
                )
                adjusted_score = (
                    min(uncapped_score, cap_score)
                    if dominance_rule_matches
                    else uncapped_score
                )
                cap_applied = (
                    dominance_rule_matches
                    and adjusted_score < uncapped_score
                )
                factor = (
                    adjusted_score / base_parent_score
                    if base_parent_score != 0.0
                    else float(single_child_factor)
                )

                result.at[
                    parent_index,
                    "hierarchy_single_child_capture",
                ] = capture
                result.at[
                    parent_index,
                    "hierarchy_single_child_direction_match",
                ] = direction_match
                result.at[
                    parent_index,
                    "hierarchy_single_child_uncapped_score",
                ] = uncapped_score
                result.at[
                    parent_index,
                    "hierarchy_dominance_cap_score",
                ] = cap_score
                result.at[
                    parent_index,
                    "hierarchy_dominance_cap_applied",
                ] = cap_applied
            else:
                deltas = [delta_by_id[child_id] for child_id in best_group]
                gross_delta = float(sum(abs(delta) for delta in deltas))
                if not math.isfinite(gross_delta) or (
                    gross_delta <= 0.0 and not allow_zero_movement
                ):
                    raise ValueError(
                        f"Для сильнейшей группы родителя {parent_id} "
                        "gross-сумма изменений GMV должна быть положительной"
                    )
                if gross_delta == 0.0:
                    result.at[parent_index, "hierarchy_score_factor"] = 1.0
                    result.at[parent_index, "anomaly_score"] = base_parent_score
                    continue
                direction_unity = min(1.0, max(0.0, abs(sum(deltas)) / gross_delta))
                shares = [abs(delta) / gross_delta for delta in deltas]
                dominant_share = max(shares)
                balance_max = min(
                    1.0,
                    max(
                        0.0,
                        (1.0 - dominant_share)
                        / (1.0 - 1.0 / best_group_size),
                    ),
                )
                concentration = float(sum(share * share for share in shares))
                effective_count = 1.0 / concentration
                balance_effective = min(
                    1.0,
                    max(
                        0.0,
                        (effective_count - 1.0) / (best_group_size - 1.0),
                    ),
                )
                balance = min(balance_max, balance_effective)
                coherence = direction_unity * balance
                factor = 1.0 + float(aggregation_bonus_lambda) * (
                    coherence - 0.5
                )

                result.at[
                    parent_index, "hierarchy_direction_unity"
                ] = direction_unity
                result.at[
                    parent_index, "hierarchy_dominant_share"
                ] = dominant_share
                result.at[parent_index, "hierarchy_balance_max"] = balance_max
                result.at[
                    parent_index, "hierarchy_balance_effective"
                ] = balance_effective
                result.at[parent_index, "hierarchy_balance"] = balance
                result.at[parent_index, "hierarchy_coherence"] = coherence

            result.at[parent_index, "hierarchy_score_factor"] = float(factor)
            result.at[parent_index, "anomaly_score"] = (
                base_parent_score * float(factor)
            )

    return result
