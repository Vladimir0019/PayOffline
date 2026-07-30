"""?????????????? ?????? ???????????? GMV-?????????."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import AnomalyThresholds, METRIC_COLUMNS
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

    Args:
        nonzero_weeks: Количество ненулевых исторических недель до текущей.

    Returns:
        Коэффициент надёжности истории.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _history_reliability(8)
        1.0
    """

    if nonzero_weeks >= 8:
        return 1.0
    if nonzero_weeks >= 4:
        return 0.7
    if nonzero_weeks >= 1:
        return 0.4
    return 0.4


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
    sigma = max(1.4826 * mad, thresholds.sigma_floor)

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

    # ADDED: WoW-проценты по операционным метрикам для менеджерского вывода.
    metric_values: Dict[str, object] = {}
    for metric in METRIC_COLUMNS:
        if metric not in segment_panel.columns:
            continue
        metric_series = segment_panel.set_index("cal_date")[metric].astype(float).reindex(dates, fill_value=0.0)
        previous_metric = _safe_float(metric_series.loc[previous_cal_date], 0.0)
        current_metric = _safe_float(metric_series.loc[current_cal_date], 0.0)
        metric_values[f"{metric}_previous"] = previous_metric
        metric_values[f"{metric}_current"] = current_metric
        metric_values[f"{metric}_wow_pct"] = None if previous_metric == 0 else (current_metric - previous_metric) / previous_metric

    if relative_wow is None:
        robust_z = thresholds.z_cap if state != "обычный" and wow_delta_gmv != 0 else 0.0
    else:
        robust_z = (relative_wow - baseline_growth) / sigma
    # Ограничение на z снято специально
    # robust_z_capped = max(-thresholds.z_cap, min(thresholds.z_cap, robust_z))
    # abs_z_capped = min(abs(robust_z), thresholds.z_cap)
    robust_z_capped = robust_z
    abs_z_capped = abs(robust_z)

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
        "robust_z": robust_z,
        "robust_z_capped": robust_z_capped,
        "abs_z_capped": abs_z_capped,
        "abnormal_gmv": wow_delta_gmv,
        "abs_abnormal_gmv": abs(wow_delta_gmv),
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
) -> Tuple[pd.DataFrame, pd.Series]:
    """Посчитать аномальность независимо для каждого среза.

    Args:
        panel_df: Полная недельная панель segment x week.
        dim_cols: Список признаков.
        dates: Список недель.
        thresholds: Пороги алгоритма.
        current_cal_date: Текущая неделя. Если None, берётся последняя.

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
        & (candidates["abs_z_capped"].astype(float) >= thresholds.min_z_score)
        & (candidates["materiality_share"].astype(float) >= thresholds.min_materiality_share)
        & (candidates["wow_delta_gmv"].astype(float).abs() >= thresholds.min_anomaly_abs)
    )
    return candidates.sort_values(["slice_depth", "segment_key"]).reset_index(drop=True), total_by_date


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


def apply_local_depth_penalty(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    depth_factor: float = 0.9,
) -> pd.DataFrame:
    """ADDED: Рассчитать локальный штраф за глубину относительно ветки сегмента.

    Args:
        candidates: Таблица всех кандидатов до оптимизационного отбора.
        coverage: Атомарное покрытие `segment_id -> set[atomic_segment_id]`, построенное по всем сегментам.
        depth_factor: Коэффициент штрафа за один уровень до максимальной eligible-глубины ветки.

    Returns:
        Копия `candidates` с колонками `base_anomaly_score`, `local_max_eligible_depth`,
        `local_depth_gap`, `eligible_descendant_count`, `depth_score_weight`, `anomaly_score`.

    Raises:
        ValueError: Если не хватает обязательных колонок, depth_factor некорректен,
            или eligible-сегмент имеет отсутствующее/пустое покрытие либо некорректный base score.

    Examples:
        >>> df = pd.DataFrame([
        ...     {'segment_id': 'p', 'slice_depth': 1, 'passes_initial_anomaly_filter': True,
        ...      'abs_z_capped': 2.0, 'materiality_share': 0.5, 'reliability_factor': 1.0},
        ... ])
        >>> apply_local_depth_penalty(df, {'p': frozenset({'a'})})['depth_score_weight'].iloc[0]
        np.float64(1.0)
    """

    required_columns = {
        "segment_id",
        "slice_depth",
        "passes_initial_anomaly_filter",
        "abs_z_capped",
        "materiality_share",
        "reliability_factor",
    }
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Для apply_local_depth_penalty не хватает колонок: {missing_columns}")
    if not 0.0 < float(depth_factor) <= 1.0:
        raise ValueError("depth_factor должен быть в интервале (0, 1]")

    result = candidates.copy()
    result["segment_id"] = result["segment_id"].astype(str)
    result["slice_depth"] = result["slice_depth"].astype(int)
    result["base_anomaly_score"] = (
        pd.to_numeric(result["abs_z_capped"], errors="coerce")
        * pd.to_numeric(result["materiality_share"], errors="coerce")
        * pd.to_numeric(result["reliability_factor"], errors="coerce")
    )
    result["local_max_eligible_depth"] = result["slice_depth"].astype(int)
    result["local_depth_gap"] = 0
    result["eligible_descendant_count"] = 0
    result["depth_score_weight"] = 1.0
    result["anomaly_score"] = result["base_anomaly_score"] * result["depth_score_weight"]

    normalized_coverage = {
        str(segment_id): frozenset(str(atom_id) for atom_id in (atoms or frozenset()))
        for segment_id, atoms in coverage.items()
    }
    eligible_mask = result["passes_initial_anomaly_filter"].eq(True) & result["slice_depth"].gt(0)
    eligible = result[eligible_mask].copy()
    fatal_issues: List[str] = []
    coverage_by_id: Dict[str, frozenset[str]] = {}
    depth_by_id: Dict[str, int] = {}
    for _, row in eligible.iterrows():
        segment_id = str(row["segment_id"])
        segment_atoms = normalized_coverage.get(segment_id)
        if segment_atoms is None:
            fatal_issues.append(f"{segment_id} ({row.get('segment_key', '')}): отсутствует атомарное покрытие")
            continue
        if not segment_atoms:
            fatal_issues.append(f"{segment_id} ({row.get('segment_key', '')}): пустое атомарное покрытие")
            continue
        base_score = _safe_float(row.get("base_anomaly_score"), math.nan)
        if not math.isfinite(base_score):
            fatal_issues.append(f"{segment_id} ({row.get('segment_key', '')}): некорректный base_anomaly_score")
            continue
        coverage_by_id[segment_id] = segment_atoms
        depth_by_id[segment_id] = int(row["slice_depth"])
    if fatal_issues:
        raise ValueError(
            "Нельзя рассчитать локальный depth penalty для eligible-сегментов: "
            + "; ".join(fatal_issues[:10])
        )

    descendant_depths_by_id: Dict[str, List[int]] = {segment_id: [] for segment_id in coverage_by_id}
    for parent_id, parent_atoms in coverage_by_id.items():
        parent_depth = depth_by_id[parent_id]
        for child_id, child_atoms in coverage_by_id.items():
            if child_id == parent_id:
                continue
            child_depth = depth_by_id[child_id]
            if child_depth > parent_depth and child_atoms.issubset(parent_atoms):
                descendant_depths_by_id[parent_id].append(child_depth)

    index_by_id = {
        str(segment_id): index
        for index, segment_id in result["segment_id"].items()
    }
    for segment_id, descendant_depths in descendant_depths_by_id.items():
        index = index_by_id[segment_id]
        current_depth = depth_by_id[segment_id]
        local_max_depth = max([current_depth, *descendant_depths])
        local_gap = local_max_depth - current_depth
        result.at[index, "local_max_eligible_depth"] = int(local_max_depth)
        result.at[index, "local_depth_gap"] = int(local_gap)
        result.at[index, "eligible_descendant_count"] = int(len(descendant_depths))
        result.at[index, "depth_score_weight"] = float(depth_factor) ** int(local_gap)

    result["anomaly_score"] = result["base_anomaly_score"].astype(float) * result["depth_score_weight"].astype(float)
    return result
