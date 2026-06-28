"""
Поиск необычных сегментов GMV на истории нескольких недель.

Модуль считает статистическую необычность отдельно для каждого среза, а затем
выбирает компактный набор непересекающихся управленческих сегментов.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from main2 import (
    build_segment_key_and_level,
    candidate_covers_atomic,
    normalize_dim_value,
    segment_id_from_row,
)


DEFAULT_INPUT_PATH = Path(__file__).with_name("payoffline_pulse_hier_4_13w.xlsx")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("gmv_anomaly_report.xlsx")

# ADDED: Для исторического anomaly-файла исключаем все технические и метрические колонки.
ANOMALY_TECH_COLUMNS = {
    "period",
    "cal_date",
    "calendar_date",
    "slice_depth",
    "gmv",
    "tx",
    "au",
    "am",
    "aov",
    "tpm",
    "freq",
    "share_in_total_gmv",
    "segment_id",
    "segment_key",
    "segment_level",
}


@dataclass(frozen=True)
class AnomalyThresholds:
    """Пороги алгоритма поиска необычных сегментов.

    Args:
        min_anomaly_abs: Минимальная абсолютная величина аномального вклада в рублях.
        min_z_score: Минимальный robust z-score для обычного сегмента.
        sigma_floor: Нижняя граница масштаба колебаний доли.
        z_cap: Верхняя граница z-score, участвующего в score.
        dominance_threshold: Доля доминирующего ребёнка в движении детей.
        compensation_threshold: Минимальная доля взаимной компенсации детей.
        max_manager_facts: Максимальное число фактов в менеджерском выводе.

    Returns:
        Экземпляр с порогами.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> AnomalyThresholds().min_z_score
        2.0
    """

    min_anomaly_abs: float = 200_000.0
    min_z_score: float = 2.0
    sigma_floor: float = 0.00001
    z_cap: float = 6.0
    dominance_threshold: float = 0.80
    compensation_threshold: float = 0.60
    max_manager_facts: int = 10


def infer_anomaly_dimension_columns(df: pd.DataFrame, explicit_dims: Optional[Sequence[str]] = None) -> List[str]:
    """Определить бизнес-признаки для anomaly-анализа.

    Args:
        df: Входная таблица.
        explicit_dims: Явно заданные признаки. Если переданы, используются они.

    Returns:
        Список бизнес-признаков.

    Raises:
        ValueError: Если явно заданный признак отсутствует.

    Examples:
        >>> infer_anomaly_dimension_columns(pd.DataFrame(columns=['period', 'gmv', 'geo', 'tx']))
        ['geo']
    """

    if explicit_dims:
        missing = [col for col in explicit_dims if col not in df.columns]
        if missing:
            raise ValueError(f"В таблице отсутствуют явно заданные признаки: {missing}")
        return list(explicit_dims)
    return [col for col in df.columns if col not in ANOMALY_TECH_COLUMNS]


def _format_rub(value: float) -> str:
    """Отформатировать рублёвое значение без дробной части.

    Args:
        value: Числовое значение.

    Returns:
        Строка с разделителем групп разрядов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _format_rub(1234567.89)
        '1 234 568'
    """

    return f"{value:,.0f}".replace(",", " ")


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


def load_history_table(
    input_path: str | Path,
    sheet_name: int | str = 0,
    period: Optional[str] = "1W",
    dim_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Загрузить историческую таблицу срезов GMV.

    Args:
        input_path: Путь к Excel- или CSV-файлу.
        sheet_name: Имя или номер листа Excel.
        period: Значение периода для фильтрации. Если None, фильтр не применяется.
        dim_cols: Явно заданные признаки. Если None, признаки определяются автоматически.

    Returns:
        Кортеж: очищенная таблица, список признаков, список недель cal_date из total-слоя.

    Raises:
        ValueError: Если входной файл не поддерживается, нет обязательных колонок или total-слой некорректен.

    Examples:
        >>> # df, dims, dates = load_history_table('payoffline_pulse_hier_4_13w.xlsx')
    """

    input_path = Path(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Поддерживаются только .xlsx, .xls и .csv")

    required = {"cal_date", "slice_depth", "gmv"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Не хватает обязательных колонок: {missing_required}")

    # ADDED: Удаляем служебную строку типов из Excel-выгрузки, если она присутствует.
    if "period" in df.columns:
        df = df[df["period"].astype(str) != "string"].copy()
        if period is not None:
            df = df[df["period"].astype(str) == str(period)].copy()

    df["cal_date"] = pd.to_numeric(df["cal_date"], errors="coerce")
    df["slice_depth"] = pd.to_numeric(df["slice_depth"], errors="coerce")
    df["gmv"] = pd.to_numeric(df["gmv"], errors="coerce")
    df = df.dropna(subset=["cal_date", "slice_depth", "gmv"]).copy()
    df["cal_date"] = df["cal_date"].astype(int)
    df["slice_depth"] = df["slice_depth"].astype(int)
    df["gmv"] = df["gmv"].astype(float)

    dims = infer_anomaly_dimension_columns(df, dim_cols)
    for col in dims:
        df[col] = df[col].map(normalize_dim_value)

    df["segment_id"] = df.apply(lambda row: segment_id_from_row(row, dims), axis=1)
    key_level_depth = df.apply(lambda row: build_segment_key_and_level(row, dims), axis=1)
    df["segment_key"] = [item[0] for item in key_level_depth]
    df["segment_level"] = [item[1] for item in key_level_depth]

    total_df = df[df["slice_depth"] == 0].copy()
    if total_df.empty:
        raise ValueError("Не найден total-слой slice_depth = 0")
    total_by_date = total_df.groupby("cal_date", as_index=False)["gmv"].sum()
    dates = sorted(total_by_date["cal_date"].astype(int).tolist())
    if len(dates) < 4:
        raise ValueError("Для robust z-score нужно минимум 4 недели в total-слое")

    date_diffs = pd.Series(dates).diff().dropna().astype(int).tolist()
    if date_diffs and any(diff != 7 for diff in date_diffs):
        raise ValueError("В total-слое есть пропуск календарной недели; без total GMV неделю нельзя восстановить")
    if (total_by_date["gmv"].astype(float) <= 0).any():
        raise ValueError("Total GMV должен быть положительным на каждой неделе")

    return df.reset_index(drop=True), dims, dates


def build_full_week_grid(history_df: pd.DataFrame, dim_cols: Sequence[str], dates: Sequence[int]) -> pd.DataFrame:
    """Построить полную сетку segment x week с GMV = 0 для отсутствующих строк.

    Args:
        history_df: Очищенная историческая таблица.
        dim_cols: Список признаков.
        dates: Полный список недель из total-слоя.

    Returns:
        Панель сегментов по неделям.

    Raises:
        ValueError: Если нет сегментов.

    Examples:
        >>> # panel = build_full_week_grid(history_df, ['geo'], [1, 8, 15])
    """

    meta_cols = ["segment_id", "segment_key", "segment_level", "slice_depth", *dim_cols]
    meta = history_df[meta_cols].drop_duplicates("segment_id").copy()
    if meta.empty:
        raise ValueError("Не найдены сегменты для построения недельной сетки")

    weekly = (
        history_df.groupby(["segment_id", "cal_date"], as_index=False)
        .agg(gmv=("gmv", "sum"), source_row_count=("gmv", "size"))
    )
    full_index = pd.MultiIndex.from_product(
        [meta["segment_id"].tolist(), list(dates)],
        names=["segment_id", "cal_date"],
    )
    panel = full_index.to_frame(index=False).merge(weekly, how="left", on=["segment_id", "cal_date"])
    panel["gmv"] = panel["gmv"].fillna(0.0).astype(float)
    panel["source_row_count"] = panel["source_row_count"].fillna(0).astype(int)
    panel = panel.merge(meta, how="left", on="segment_id")
    panel["row_missing_in_source"] = panel["source_row_count"] == 0
    return panel.sort_values(["segment_id", "cal_date"]).reset_index(drop=True)


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

    if relative_wow is None:
        robust_z = thresholds.z_cap if state != "обычный" and wow_delta_gmv != 0 else 0.0
    else:
        robust_z = (relative_wow - baseline_growth) / sigma
    robust_z_capped = max(-thresholds.z_cap, min(thresholds.z_cap, robust_z))
    abs_z_capped = min(abs(robust_z), thresholds.z_cap)

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
    candidates["anomaly_score"] = (
        candidates["abs_z_capped"].astype(float)
        * candidates["materiality_share"].astype(float)
        * candidates["reliability_factor"].astype(float)
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


def add_child_context(candidates: pd.DataFrame, coverage: Dict[str, frozenset[str]]) -> pd.DataFrame:
    """Добавить метрики детей для оценки родителя.

    Args:
        candidates: Таблица кандидатов.
        coverage: Покрытие кандидатов атомами.

    Returns:
        Таблица кандидатов с детскими метриками.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # candidates = add_child_context(candidates, coverage)
    """

    result = candidates.copy()
    rows: List[Dict[str, object]] = []
    max_depth = int(result["slice_depth"].max())

    for _, parent in result.iterrows():
        parent_id = str(parent["segment_id"])
        parent_depth = int(parent["slice_depth"])
        parent_atoms = coverage.get(parent_id, frozenset())
        child_rows = []
        if parent_depth < max_depth:
            possible_children = result[result["slice_depth"].astype(int) == parent_depth + 1]
            for _, child in possible_children.iterrows():
                child_atoms = coverage.get(str(child["segment_id"]), frozenset())
                if child_atoms and child_atoms.issubset(parent_atoms):
                    child_rows.append(child)

        if not child_rows:
            rows.append(
                {
                    "segment_id": parent_id,
                    "direct_child_count": 0,
                    "child_net_abnormal": 0.0,
                    "child_gross_abnormal": 0.0,
                    "child_cancellation_ratio": 0.0,
                    "child_same_direction_share": 1.0,
                    "child_dominance_share": 0.0,
                    "dominant_child_id": "",
                    "dominant_child_key": "",
                    "dominant_child_abs_abnormal": 0.0,
                }
            )
            continue

        child_df = pd.DataFrame(child_rows)
        deltas = child_df["abnormal_gmv"].astype(float)
        gross = float(deltas.abs().sum())
        net = float(deltas.sum())
        positive = float(deltas[deltas > 0].sum())
        negative = float(deltas[deltas < 0].sum())
        dominant_idx = deltas.abs().idxmax()
        dominant = child_df.loc[dominant_idx]
        dominant_abs = abs(float(dominant["abnormal_gmv"]))

        rows.append(
            {
                "segment_id": parent_id,
                "direct_child_count": int(len(child_df)),
                "child_net_abnormal": net,
                "child_gross_abnormal": gross,
                "child_cancellation_ratio": 0.0 if gross == 0 else 1.0 - abs(net) / gross,
                "child_same_direction_share": 1.0 if gross == 0 else max(positive, abs(negative)) / gross,
                "child_dominance_share": 0.0 if gross == 0 else dominant_abs / gross,
                "dominant_child_id": str(dominant["segment_id"]),
                "dominant_child_key": str(dominant["segment_key"]),
                "dominant_child_abs_abnormal": dominant_abs,
            }
        )

    context = pd.DataFrame(rows)
    return result.merge(context, how="left", on="segment_id")


def classify_anomaly_candidates(candidates: pd.DataFrame, thresholds: AnomalyThresholds) -> pd.DataFrame:
    """Классифицировать кандидатов по управленческой роли.

    Args:
        candidates: Таблица кандидатов с метриками аномальности и детей.
        thresholds: Пороги алгоритма.

    Returns:
        Таблица кандидатов с полями action, output_block, reason и selection_score.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # candidates = classify_anomaly_candidates(candidates, AnomalyThresholds())
    """

    rows: List[Dict[str, object]] = []
    max_depth = int(candidates["slice_depth"].max())

    for _, row in candidates.iterrows():
        depth = int(row["slice_depth"])
        state = str(row["state"])
        abs_abnormal = abs(float(row["abnormal_gmv"]))
        abs_z = abs(float(row["robust_z_capped"]))
        base_score = float(row["anomaly_score"])
        child_gross = float(row.get("child_gross_abnormal", 0.0))
        gross_atomic = float(row.get("gross_atomic_movement", 0.0))
        child_gross_materiality = 0.0 if gross_atomic == 0 else child_gross / gross_atomic
        child_cancel = float(row.get("child_cancellation_ratio", 0.0))
        child_dom = float(row.get("child_dominance_share", 0.0))
        child_dom_abs = float(row.get("dominant_child_abs_abnormal", 0.0))
        child_count = int(row.get("direct_child_count", 0))

        action = "исключён"
        output_block = "исключён"
        reason = "не прошёл пороги необычности и материальности"
        eligible = False
        selection_score = base_score

        if depth == 0:
            reason = "total не выбирается как аномальный сегмент"
        elif child_dom >= thresholds.dominance_threshold and child_dom_abs >= thresholds.min_anomaly_abs:
            action = "пропустить_из-за_доминирующего_ребёнка"
            output_block = "доминирующий ребёнок"
            reason = f"аномальность родителя в основном сосредоточена в ребёнке: {row.get('dominant_child_key', '')}"
        elif (
            child_count >= 2
            and child_gross >= thresholds.min_anomaly_abs
            and child_cancel >= thresholds.compensation_threshold
        ):
            action = "блок_аномальной_компенсации"
            output_block = "блок аномальной компенсации"
            reason = "внутри родителя есть встречные аномальные движения детей"
            eligible = True
            selection_score = max(
                base_score,
                child_gross_materiality * child_cancel * float(row.get("reliability_factor", 1.0)),
            )
        elif state != "обычный" and abs_abnormal >= thresholds.min_anomaly_abs:
            action = "аномалия_статуса"
            output_block = state
            reason = "сегмент изменил статус присутствия относительно истории"
            eligible = True
        elif abs_abnormal >= thresholds.min_anomaly_abs and abs_z >= thresholds.min_z_score:
            action = "основная_аномалия" if depth < max_depth else "атомарная_аномалия"
            output_block = "основная аномалия" if depth < max_depth else "атомарная аномалия"
            reason = "сегмент материально отклонился от собственной исторической нормы"
            eligible = True

        rows.append(
            {
                "segment_id": str(row["segment_id"]),
                "action": action,
                "output_block": output_block,
                "reason": reason,
                "is_eligible": eligible,
                "selection_score": float(selection_score),
            }
        )

    classified = candidates.merge(pd.DataFrame(rows), how="left", on="segment_id")
    classified["covered_atomic_count"] = 0
    return classified


def select_hierarchical_anomalies(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Выбрать непересекающиеся управленческие аномалии.

    Args:
        candidates: Таблица классифицированных кандидатов.
        coverage: Покрытие кандидатов атомами.

    Returns:
        Кортеж: итоговые выбранные аномалии и обновлённая диагностика кандидатов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # final_df, diagnostics = select_hierarchical_anomalies(candidates, coverage)
    """

    diagnostics = candidates.copy()
    diagnostics["covered_atomic_count"] = diagnostics["segment_id"].map(lambda sid: len(coverage.get(str(sid), frozenset())))
    diagnostics["selected"] = False
    diagnostics["selection_exclusion_reason"] = ""

    eligible = diagnostics[diagnostics["is_eligible"].astype(bool)].copy()
    eligible = eligible.sort_values(
        by=[
            "selection_score",
            "abs_z_capped",
            "materiality_share",
            "abs_abnormal_gmv",
            "reliability_factor",
            "covered_atomic_count",
            "segment_key",
        ],
        ascending=[False, False, False, False, False, False, True],
    )

    selected_ids: List[str] = []
    used_atoms: set[str] = set()
    for _, candidate in eligible.iterrows():
        segment_id = str(candidate["segment_id"])
        atoms = set(coverage.get(segment_id, frozenset()))
        if not atoms:
            diagnostics.loc[diagnostics["segment_id"] == segment_id, "selection_exclusion_reason"] = "нет атомарного покрытия"
            continue
        if used_atoms.intersection(atoms):
            diagnostics.loc[diagnostics["segment_id"] == segment_id, "selection_exclusion_reason"] = "пересекается с уже выбранным сегментом"
            continue
        selected_ids.append(segment_id)
        used_atoms.update(atoms)
        diagnostics.loc[diagnostics["segment_id"] == segment_id, "selected"] = True

    diagnostics.loc[
        (~diagnostics["selected"]) & (diagnostics["selection_exclusion_reason"] == "") & diagnostics["is_eligible"].astype(bool),
        "selection_exclusion_reason",
    ] = "уступил более сильному непересекающемуся кандидату"
    diagnostics.loc[
        (~diagnostics["selected"]) & (diagnostics["selection_exclusion_reason"] == "") & (~diagnostics["is_eligible"].astype(bool)),
        "selection_exclusion_reason",
    ] = diagnostics["reason"]

    final_df = diagnostics[diagnostics["selected"]].copy()
    final_df = final_df.sort_values(
        by=[
            "selection_score",
            "abs_z_capped",
            "materiality_share",
            "abs_abnormal_gmv",
            "reliability_factor",
            "segment_key",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    final_df.insert(0, "rank", range(1, len(final_df) + 1))
    return final_df, diagnostics


def build_manager_summary(final_df: pd.DataFrame, thresholds: AnomalyThresholds, current_total_gmv: float) -> pd.DataFrame:
    """Сформировать менеджерский вывод по необычным сегментам.

    Args:
        final_df: Итоговые выбранные аномалии.
        thresholds: Пороги алгоритма.
        current_total_gmv: Total GMV текущей недели.

    Returns:
        Таблица менеджерского вывода.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # manager_df = build_manager_summary(final_df, AnomalyThresholds(), 1000)
    """

    rows: List[Dict[str, object]] = [
        {
            "раздел": "Заголовок",
            "тип": "",
            "сегмент": "Менеджерский вывод по необычным сегментам GMV",
            "аномальный_вклад": "",
            "внутреннее_аномальное_движение": "",
            "wow_изменение": "",
            "z_score": "",
            "интерпретация": f"Total GMV текущей недели: {_format_rub(current_total_gmv)}.",
        }
    ]

    if final_df.empty:
        rows.append(
            {
                "раздел": "Краткий вывод",
                "тип": "",
                "сегмент": "",
                "аномальный_вклад": "",
                "внутреннее_аномальное_движение": "",
                "wow_изменение": "",
                "z_score": "",
                "интерпретация": "Материальные необычные сегменты по заданным порогам не найдены.",
            }
        )
        return pd.DataFrame(rows)

    top = final_df.head(thresholds.max_manager_facts)
    main = top.iloc[0]
    rows.append(
        {
            "раздел": "Краткий вывод",
            "тип": str(main["output_block"]),
            "сегмент": str(main["segment_key"]),
            "аномальный_вклад": _format_rub(float(main["abnormal_gmv"])),
            "внутреннее_аномальное_движение": _format_rub(float(main.get("child_gross_abnormal", 0.0))),
            "wow_изменение": _format_rub(float(main["wow_delta_gmv"])),
            "z_score": round(float(main["robust_z"]), 2),
            "интерпретация": "Самый сильный выбранный блок по score; для компенсационных блоков score определяется внутренним движением детей.",
        }
    )

    for _, row in top.iterrows():
        direction = "выше" if float(row["abnormal_gmv"]) > 0 else "ниже"
        if str(row["output_block"]) == "блок аномальной компенсации":
            interpretation = (
                "Net-эффект родителя может быть небольшим, но внутри детей есть крупные встречные аномальные движения. "
                f"Причина отбора: {row['reason']}."
            )
        else:
            interpretation = (
                f"Фактический GMV сегмента {direction} ожидаемого уровня. "
                f"Причина отбора: {row['reason']}."
            )
        rows.append(
            {
                "раздел": "Таблица факторов",
                "тип": str(row["output_block"]),
                "сегмент": str(row["segment_key"]),
                "аномальный_вклад": _format_rub(float(row["abnormal_gmv"])),
                "внутреннее_аномальное_движение": _format_rub(float(row.get("child_gross_abnormal", 0.0))),
                "wow_изменение": _format_rub(float(row["wow_delta_gmv"])),
                "z_score": round(float(row["robust_z"]), 2),
                "интерпретация": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_history_for_selected(panel_df: pd.DataFrame, final_df: pd.DataFrame, total_by_date: pd.Series) -> pd.DataFrame:
    """Подготовить историю недель для выбранных аномалий.

    Args:
        panel_df: Полная недельная панель.
        final_df: Итоговые выбранные аномалии.
        total_by_date: Total GMV по неделям.

    Returns:
        Таблица истории выбранных сегментов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # history = build_history_for_selected(panel, final_df, total)
    """

    if final_df.empty:
        return pd.DataFrame()

    selected_ids = final_df["segment_id"].astype(str).tolist()
    history = panel_df[panel_df["segment_id"].astype(str).isin(selected_ids)].copy()
    history["total_gmv"] = history["cal_date"].map(total_by_date.to_dict()).astype(float)
    history["share_in_total"] = history["gmv"] / history["total_gmv"]
    history = history.sort_values(["segment_key", "cal_date"])
    history["share_delta"] = history.groupby("segment_id")["share_in_total"].diff()
    return history[
        [
            "segment_id",
            "segment_key",
            "slice_depth",
            "cal_date",
            "gmv",
            "total_gmv",
            "share_in_total",
            "share_delta",
            "row_missing_in_source",
        ]
    ].reset_index(drop=True)


def build_missing_zero_report(panel_df: pd.DataFrame, dates: Sequence[int], current_cal_date: int) -> pd.DataFrame:
    """Сформировать отчёт по пропускам и нулевому GMV.

    Args:
        panel_df: Полная недельная панель.
        dates: Список недель.
        current_cal_date: Текущая неделя.

    Returns:
        Таблица сегментов с пропусками или нулевыми неделями.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # missing_df = build_missing_zero_report(panel, dates, dates[-1])
    """

    previous_cal_date = int(dates[list(dates).index(int(current_cal_date)) - 1])
    grouped = panel_df.groupby(["segment_id", "segment_key", "segment_level", "slice_depth"], as_index=False).agg(
        missing_week_count=("row_missing_in_source", "sum"),
        zero_week_count=("gmv", lambda s: int((s == 0).sum())),
        nonzero_week_count=("gmv", lambda s: int((s > 0).sum())),
    )
    current = panel_df[panel_df["cal_date"].astype(int) == int(current_cal_date)][["segment_id", "gmv", "row_missing_in_source"]]
    previous = panel_df[panel_df["cal_date"].astype(int) == previous_cal_date][["segment_id", "gmv", "row_missing_in_source"]]
    current = current.rename(columns={"gmv": "gmv_current", "row_missing_in_source": "current_row_missing"})
    previous = previous.rename(columns={"gmv": "gmv_previous", "row_missing_in_source": "previous_row_missing"})
    report = grouped.merge(previous, how="left", on="segment_id").merge(current, how="left", on="segment_id")
    report = report[(report["missing_week_count"] > 0) | (report["zero_week_count"] > 0)].copy()
    return report.sort_values(["missing_week_count", "zero_week_count", "segment_key"], ascending=[False, False, True]).reset_index(drop=True)


def build_control_table(
    history_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    candidates: pd.DataFrame,
    final_df: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    dates: Sequence[int],
    current_cal_date: int,
    total_by_date: pd.Series,
) -> pd.DataFrame:
    """Сформировать контрольные показатели результата.

    Args:
        history_df: Исходная очищенная таблица.
        panel_df: Полная недельная панель.
        candidates: Диагностика кандидатов.
        final_df: Итоговые выбранные аномалии.
        coverage: Покрытие кандидатов атомами.
        dates: Список недель.
        current_cal_date: Текущая неделя.
        total_by_date: Total GMV по неделям.

    Returns:
        Таблица контрольных показателей.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # control = build_control_table(history_df, panel, candidates, final_df, coverage, dates, dates[-1], total)
    """

    selected_atoms: List[str] = []
    for segment_id in final_df["segment_id"].astype(str).tolist() if not final_df.empty else []:
        selected_atoms.extend(list(coverage.get(segment_id, frozenset())))
    selected_atom_unique_count = len(set(selected_atoms))
    double_count_violation_count = len(selected_atoms) - selected_atom_unique_count
    max_depth = int(candidates["slice_depth"].max()) if not candidates.empty else 0
    atomic_count = int((candidates["slice_depth"].astype(int) == max_depth).sum()) if not candidates.empty else 0

    rows = [
        ("source_rows", len(history_df)),
        ("panel_rows", len(panel_df)),
        ("week_count", len(dates)),
        ("previous_cal_date", int(dates[list(dates).index(int(current_cal_date)) - 1])),
        ("current_cal_date", int(current_cal_date)),
        ("current_total_gmv", float(total_by_date.loc[current_cal_date])),
        ("candidate_count", len(candidates)),
        ("eligible_candidate_count", int(candidates["is_eligible"].astype(bool).sum()) if "is_eligible" in candidates else 0),
        ("selected_count", len(final_df)),
        ("atomic_count", atomic_count),
        ("selected_atomic_unique_count", selected_atom_unique_count),
        ("double_count_violation_count", double_count_violation_count),
        ("filled_missing_rows", int(panel_df["row_missing_in_source"].sum())),
    ]
    return pd.DataFrame(rows, columns=["показатель", "значение"])


def write_anomaly_excel(
    output_path: str | Path,
    thresholds: AnomalyThresholds,
    dim_cols: Sequence[str],
    history_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    candidates: pd.DataFrame,
    final_df: pd.DataFrame,
    total_by_date: pd.Series,
    dates: Sequence[int],
    current_cal_date: int,
    coverage: Dict[str, frozenset[str]],
) -> None:
    """Записать результат поиска аномалий в Excel.

    Args:
        output_path: Путь к итоговому Excel-файлу.
        thresholds: Пороги алгоритма.
        dim_cols: Список признаков.
        history_df: Исходная очищенная таблица.
        panel_df: Полная недельная панель.
        candidates: Диагностика кандидатов.
        final_df: Итоговые выбранные аномалии.
        total_by_date: Total GMV по неделям.
        dates: Список недель.
        current_cal_date: Текущая неделя.
        coverage: Покрытие кандидатов атомами.

    Returns:
        None.

    Raises:
        OSError: Если Excel-файл невозможно записать.

    Examples:
        >>> # write_anomaly_excel('gmv_anomaly_report.xlsx', thresholds, dims, history_df, panel, candidates, final_df, total, dates, dates[-1], coverage)
    """

    output_path = Path(output_path)
    params = pd.DataFrame(
        [
            ("признаки", " × ".join(dim_cols)),
            ("min_anomaly_abs", thresholds.min_anomaly_abs),
            ("min_z_score", thresholds.min_z_score),
            ("sigma_floor", thresholds.sigma_floor),
            ("z_cap", thresholds.z_cap),
            ("dominance_threshold", thresholds.dominance_threshold),
            ("compensation_threshold", thresholds.compensation_threshold),
            ("current_cal_date", int(current_cal_date)),
        ],
        columns=["показатель", "значение"],
    )
    control = build_control_table(history_df, panel_df, candidates, final_df, coverage, dates, current_cal_date, total_by_date)
    params.insert(0, "раздел", "Параметры")
    control.insert(0, "раздел", "Контроль")
    params_and_control = pd.concat([params, control], ignore_index=True)

    manager_df = build_manager_summary(final_df, thresholds, float(total_by_date.loc[current_cal_date]))
    history_top = build_history_for_selected(panel_df, final_df, total_by_date)
    missing_zero = build_missing_zero_report(panel_df, dates, current_cal_date)

    final_cols = [
        "rank",
        "output_block",
        "segment_key",
        "slice_depth",
        "gmv_current",
        "gmv_previous",
        "wow_delta_gmv",
        "relative_wow",
        "share_current",
        "share_delta_current",
        "baseline_relative_growth",
        "robust_z",
        "robust_z_capped",
        "materiality_share",
        "gross_atomic_movement",
        "abnormal_gmv",
        "abs_abnormal_gmv",
        "selection_score",
        "reliability_factor",
        "child_gross_abnormal",
        "child_cancellation_ratio",
        "child_dominance_share",
        "history_nonzero_weeks",
        "state",
        "reason",
        "covered_atomic_count",
        "dominant_child_key",
    ]
    final_export = final_df[[col for col in final_cols if col in final_df.columns]].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        params_and_control.to_excel(writer, sheet_name="00_Параметры_и_контроль", index=False)
        manager_df.to_excel(writer, sheet_name="01_Менеджерский_вывод", index=False)
        final_export.to_excel(writer, sheet_name="02_Итог_аномалий", index=False)
        history_top.to_excel(writer, sheet_name="03_История_top", index=False)
        candidates.to_excel(writer, sheet_name="04_Диагностика_кандидатов", index=False)
        missing_zero.to_excel(writer, sheet_name="05_Пропуски_и_нули", index=False)
        control.to_excel(writer, sheet_name="06_Контроль", index=False)

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for col_cells in worksheet.columns:
                max_length = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                worksheet.column_dimensions[col_letter].width = min(max(max_length + 2, 10), 48)


def run_anomaly_analysis(
    input_path: str | Path,
    output_path: str | Path,
    sheet_name: int | str = 0,
    period: Optional[str] = "1W",
    dim_cols: Optional[Sequence[str]] = None,
    current_cal_date: Optional[int] = None,
    thresholds: Optional[AnomalyThresholds] = None,
) -> Dict[str, pd.DataFrame]:
    """Запустить полный анализ необычных сегментов.

    Args:
        input_path: Путь к входному файлу.
        output_path: Путь к итоговому Excel-файлу.
        sheet_name: Имя или номер листа Excel.
        period: Период для фильтрации.
        dim_cols: Явно заданные признаки.
        current_cal_date: Анализируемая неделя. Если None, берётся последняя.
        thresholds: Пороги алгоритма. Если None, используются значения по умолчанию.

    Returns:
        Словарь таблиц результата.

    Raises:
        ValueError: Если входные данные некорректны.

    Examples:
        >>> # result = run_anomaly_analysis('input.xlsx', 'output.xlsx')
    """

    thresholds = thresholds or AnomalyThresholds()
    history_df, dims, dates = load_history_table(input_path, sheet_name=sheet_name, period=period, dim_cols=dim_cols)
    current = int(current_cal_date) if current_cal_date is not None else int(dates[-1])
    panel_df = build_full_week_grid(history_df, dims, dates)
    candidates, total_by_date = build_anomaly_candidates(panel_df, dims, dates, thresholds, current)
    coverage = build_atomic_coverage(candidates, dims)
    candidates = add_child_context(candidates, coverage)
    candidates = classify_anomaly_candidates(candidates, thresholds)
    final_df, diagnostics = select_hierarchical_anomalies(candidates, coverage)
    write_anomaly_excel(output_path, thresholds, dims, history_df, panel_df, diagnostics, final_df, total_by_date, dates, current, coverage)

    return {
        "history": history_df,
        "panel": panel_df,
        "candidates": diagnostics,
        "final": final_df,
        "control": build_control_table(history_df, panel_df, diagnostics, final_df, coverage, dates, current, total_by_date),
    }


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        Нет аргументов.

    Returns:
        Объект argparse.Namespace.

    Raises:
        SystemExit: Если аргументы некорректны.

    Examples:
        >>> # args = parse_args()
    """

    parser = argparse.ArgumentParser(description="Поиск необычных сегментов GMV на истории недель.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Путь к входному .xlsx/.xls/.csv файлу.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Путь к итоговому Excel-файлу.")
    parser.add_argument("--sheet-name", default=0, help="Имя или номер листа Excel.")
    parser.add_argument("--period", default="1W", help="Период для фильтрации.")
    parser.add_argument("--dims", nargs="*", default=None, help="Явный список признаков.")
    parser.add_argument("--current-cal-date", type=int, default=None, help="Текущая анализируемая неделя cal_date.")
    parser.add_argument("--min-anomaly-abs", type=float, default=AnomalyThresholds.min_anomaly_abs)
    parser.add_argument("--min-z-score", type=float, default=AnomalyThresholds.min_z_score)
    parser.add_argument("--sigma-floor", type=float, default=AnomalyThresholds.sigma_floor)
    parser.add_argument("--z-cap", type=float, default=AnomalyThresholds.z_cap)
    parser.add_argument("--dominance-threshold", type=float, default=AnomalyThresholds.dominance_threshold)
    parser.add_argument("--compensation-threshold", type=float, default=AnomalyThresholds.compensation_threshold)
    parser.add_argument("--max-manager-facts", type=int, default=AnomalyThresholds.max_manager_facts)
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI.

    Args:
        Нет аргументов.

    Returns:
        None.

    Raises:
        ValueError: Если входные данные некорректны.

    Examples:
        >>> # python gmv_anomaly_analysis.py --input payoffline_pulse_hier_4_13w.xlsx
    """

    args = parse_args()
    try:
        sheet_name: int | str = int(args.sheet_name)
    except (TypeError, ValueError):
        sheet_name = args.sheet_name

    thresholds = AnomalyThresholds(
        min_anomaly_abs=args.min_anomaly_abs,
        min_z_score=args.min_z_score,
        sigma_floor=args.sigma_floor,
        z_cap=args.z_cap,
        dominance_threshold=args.dominance_threshold,
        compensation_threshold=args.compensation_threshold,
        max_manager_facts=args.max_manager_facts,
    )
    result = run_anomaly_analysis(
        input_path=args.input,
        output_path=args.output,
        sheet_name=sheet_name,
        period=args.period,
        dim_cols=args.dims,
        current_cal_date=args.current_cal_date,
        thresholds=thresholds,
    )

    control = result["control"].set_index("показатель")["значение"].to_dict()
    print("Готово.")
    print(f"Итоговый файл: {args.output}")
    print(f"Кандидатов: {control.get('candidate_count')}")
    print(f"Выбрано аномалий: {control.get('selected_count')}")
    print(f"Нарушения пересечения атомов: {control.get('double_count_violation_count')}")


if __name__ == "__main__":
    main()
