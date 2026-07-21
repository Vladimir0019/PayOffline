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
from openpyxl.styles import PatternFill

from main2 import (
    build_segment_key_and_level,
    candidate_covers_atomic,
    normalize_dim_value,
    segment_id_from_row,
)


DEFAULT_INPUT_PATH = Path(__file__).with_name("payoffline_pulse_hier_4_13w.xlsx")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("gmv_anomaly_report_2.xlsx")
DEFAULT_TREE_OUTPUT_PATH = Path(__file__).with_name("Граф.png")

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

# ADDED: Операционные метрики, для которых в менеджерском выводе нужен WoW-процент.
METRIC_COLUMNS = ["tx", "au", "am", "aov", "tpm", "freq"]
MANAGER_METRIC_PCT_COLUMNS = {
    "relative_wow": "GMV WoW %",
    "tx_wow_pct": "TX WoW %",
    "au_wow_pct": "AU WoW %",
    "am_wow_pct": "AM WoW %",
    "aov_wow_pct": "AOV WoW %",
    "tpm_wow_pct": "TPM WoW %",
    "freq_wow_pct": "Freq WoW %",
}


@dataclass(frozen=True)
class AnomalyThresholds:
    """Пороги алгоритма поиска необычных сегментов.

    Args:
        min_anomaly_abs: Минимальная абсолютная величина аномального вклада в рублях.
        min_z_score: Минимальный robust z-score для обычного сегмента.
        min_materiality_share: Минимальная доля изменения GMV в gross movement атомарного слоя.
        sigma_floor: Нижняя граница масштаба колебаний доли.
        z_cap: Верхняя граница z-score, участвующего в score.
        dominance_threshold: Доля доминирующего ребёнка в движении детей.
        compensation_threshold: Минимальная доля взаимной компенсации детей.
        single_child_z_multiplier: Множитель Z ребёнка для правила активного разбиения.
        single_child_gross_share_threshold: Порог gross_share единственного активного ребёнка.
        parent_child_absorption_k_threshold: Порог k для поглощения ребёнка родителем.
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
    min_materiality_share: float = 0.0001
    sigma_floor: float = 0.00001
    z_cap: float = 6.0
    dominance_threshold: float = 0.80
    compensation_threshold: float = 0.60
    anomaly_gross_move_threshold: float = 0.50
    single_child_z_multiplier: float = 1.20
    single_child_gross_share_threshold: float = 0.70
    parent_child_absorption_k_threshold: float = 1.35
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


def _format_pct(value: object) -> object:
    """Отформатировать относительное изменение как процент для менеджерского вывода.

    Args:
        value: Числовая доля, например 0.15 для 15%.

    Returns:
        Строка с процентом или пустая строка, если процент нельзя корректно посчитать.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _format_pct(0.1234)
        '12.34%'
    """

    if value is None or pd.isna(value):
        return ""
    numeric_value = float(value)
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return ""
    return f"{numeric_value * 100:.2f}%"


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


def _segment_key_parts(segment_key: object) -> List[Tuple[str, str]]:
    """Разобрать ключ сегмента на пары «признак — значение».

    Args:
        segment_key: Человекочитаемый ключ сегмента.

    Returns:
        Список пар `(dimension, value)` в порядке исходного ключа.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _segment_key_parts("products=FULLPAYMENT × merchants_type=SMB")
        [('products', 'FULLPAYMENT'), ('merchants_type', 'SMB')]
    """

    parts: List[Tuple[str, str]] = []
    for raw_part in str(segment_key).replace(" x ", " × ").split(" × "):
        part = raw_part.strip()
        if "=" not in part:
            continue
        dimension, value = part.split("=", 1)
        parts.append((dimension.strip(), value.strip()))
    return parts


def _relative_child_segment_name(parent_key: object, child_key: object) -> str:
    """Оставить в имени ребёнка только признаки, отсутствующие у родителя.

    Args:
        parent_key: Ключ родительского сегмента.
        child_key: Ключ дочернего сегмента.

    Returns:
        Сокращённый ключ ребёнка относительно родителя.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _relative_child_segment_name(
        ...     "products=FULLPAYMENT × merchants_type=SMB",
        ...     "geo=РФ × products=FULLPAYMENT × merchants_type=SMB",
        ... )
        'geo=РФ'
    """

    parent_parts = dict(_segment_key_parts(parent_key))
    child_only_parts = [
        f"{dimension}={value}"
        for dimension, value in _segment_key_parts(child_key)
        if parent_parts.get(dimension) != value
    ]
    return " × ".join(child_only_parts) if child_only_parts else str(child_key)


def _manager_segment_name(row: pd.Series) -> str:
    """Добавить к имени родителя краткую отметку о поглощении.

    Args:
        row: Строка итогового сегмента.

    Returns:
        Имя сегмента для менеджерского вывода.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _manager_segment_name(pd.Series({'segment_key': 'products=A', 'absorbed_child_labels': 'geo=РФ'}))
        'products=A (поглощён: geo=РФ)'
    """

    segment_key = str(row.get("segment_key", ""))
    absorbed_labels = str(row.get("absorbed_child_labels", "")).strip()
    if not absorbed_labels or absorbed_labels.lower() == "nan":
        return segment_key
    return f"{segment_key} (поглощён: {absorbed_labels})"


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
    # ADDED: Метрики не являются признаками; приводим доступные колонки к числам для расчёта WoW-процентов.
    for metric in METRIC_COLUMNS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

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

    # ADDED: Сохраняем операционные метрики в недельной панели. Вход считается предагрегированным;
    # для счётчиков берём сумму, для ratio-метрик берём первое значение строки среза.
    available_metric_cols = [metric for metric in METRIC_COLUMNS if metric in history_df.columns]
    metric_agg = {
        metric: (metric, "sum" if metric in {"tx", "au", "am"} else "first")
        for metric in available_metric_cols
    }
    weekly = (
        history_df.groupby(["segment_id", "cal_date"], as_index=False)
        .agg(gmv=("gmv", "sum"), source_row_count=("gmv", "size"), **metric_agg)
    )
    full_index = pd.MultiIndex.from_product(
        [meta["segment_id"].tolist(), list(dates)],
        names=["segment_id", "cal_date"],
    )
    panel = full_index.to_frame(index=False).merge(weekly, how="left", on=["segment_id", "cal_date"])
    panel["gmv"] = panel["gmv"].fillna(0.0).astype(float)
    for metric in available_metric_cols:
        panel[metric] = panel[metric].fillna(0.0).astype(float)
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
    # ADDED: Depth discount keeps the deepest segments at weight 1.0 and mildly lowers broader segments.
    candidates["depth_score_weight"] = 0.9 ** (max_depth - candidates["slice_depth"].astype(int))
    candidates["anomaly_score"] = (
        candidates["abs_z_capped"].astype(float)
        * candidates["materiality_share"].astype(float)
        * candidates["reliability_factor"].astype(float)
        * candidates["depth_score_weight"].astype(float)
    )
    # ADDED: Предварительный фильтр аномальности. Неаномальные или нематериальные сегменты
    # остаются в диагностике, но не участвуют в поглощении, компенсации и выборе.
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
            possible_children = result[
                (result["slice_depth"].astype(int) == parent_depth + 1)
                & (result["passes_initial_anomaly_filter"].astype(bool))
            ]
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
        elif not bool(row.get("passes_initial_anomaly_filter", False)):
            reason = "сегмент не прошёл предварительный фильтр: z-score или доля изменения GMV в gross movement ниже порога"
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


def _segment_feature_set(row: pd.Series, dim_cols: Sequence[str]) -> frozenset[Tuple[str, str]]:
    """Построить множество заполненных признаков сегмента.

    Args:
        row: Строка сегмента.
        dim_cols: Упорядоченный список измерений.

    Returns:
        Множество пар `(dimension, value)`.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _segment_feature_set(pd.Series({'geo': 'РФ', 'products': None}), ['geo', 'products'])
        frozenset({('geo', 'РФ')})
    """

    return frozenset(
        (dimension, value)
        for dimension in dim_cols
        if (value := normalize_dim_value(row.get(dimension))) is not None
    )


def _nearest_active_child_map(
    active_ids: Sequence[str],
    feature_sets: Dict[str, frozenset[Tuple[str, str]]],
) -> Dict[str, List[str]]:
    """Найти ближайших активных потомков каждого активного сегмента.

    Args:
        active_ids: Идентификаторы активных аномальных сегментов.
        feature_sets: Наборы признаков сегментов.

    Returns:
        Словарь `parent_id -> [nearest_child_id, ...]`.

    Raises:
        KeyError: Если для активного сегмента отсутствует набор признаков.

    Examples:
        >>> features = {'p': frozenset({('a', '1')}), 'c': frozenset({('a', '1'), ('b', '2')})}
        >>> _nearest_active_child_map(['p', 'c'], features)
        {'c': [], 'p': ['c']}
    """

    ordered_ids = sorted(set(str(segment_id) for segment_id in active_ids))
    result: Dict[str, List[str]] = {}
    for parent_id in ordered_ids:
        parent_features = feature_sets[parent_id]
        descendants = [
            child_id
            for child_id in ordered_ids
            if parent_features < feature_sets[child_id]
        ]
        nearest_children = []
        for child_id in descendants:
            child_features = feature_sets[child_id]
            has_active_middle = any(
                parent_features < feature_sets[middle_id] < child_features
                for middle_id in ordered_ids
                if middle_id not in {parent_id, child_id}
            )
            if not has_active_middle:
                nearest_children.append(child_id)
        result[parent_id] = sorted(
            nearest_children,
            key=lambda segment_id: (len(feature_sets[segment_id]), segment_id),
        )
    return result


def _single_child_gross_share(
    child_delta_gmv: float,
    parent_residual_atoms: frozenset[str],
    atomic_deltas: Dict[str, float],
) -> float:
    """Посчитать долю движения ребёнка в residual gross movement родителя.

    Args:
        child_delta_gmv: Изменение GMV единственного активного ребёнка.
        parent_residual_atoms: Текущее остаточное атомарное покрытие родителя.
        atomic_deltas: Изменения GMV физических атомов.

    Returns:
        Неотрицательная доля gross movement; ноль при нулевом знаменателе.

    Raises:
        KeyError: Если для атома отсутствует изменение GMV.

    Examples:
        >>> _single_child_gross_share(-70.0, frozenset({'a', 'b'}), {'a': -70.0, 'b': 30.0})
        0.7
    """

    residual_gross_movement = float(
        sum(abs(float(atomic_deltas[atom_id])) for atom_id in parent_residual_atoms)
    )
    if residual_gross_movement == 0.0:
        return 0.0
    return abs(float(child_delta_gmv)) / residual_gross_movement


def _single_child_decision(
    parent_abs_z: float,
    child_abs_z: float,
    gross_share: float,
    thresholds: AnomalyThresholds,
) -> str:
    """Применить правило выбора между родителем и единственным ребёнком.

    Args:
        parent_abs_z: Абсолютный z-score родителя.
        child_abs_z: Абсолютный z-score ребёнка.
        gross_share: Неотрицательная доля движения ребёнка.
        thresholds: Порог доли и множитель Z.

    Returns:
        `CHILD_WINS` или `PARENT_WINS`.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _single_child_decision(2.0, 3.0, 0.8, AnomalyThresholds())
        'CHILD_WINS'
        >>> _single_child_decision(3.6, 3.0, 0.8, AnomalyThresholds())
        'PARENT_WINS'
    """

    if (
        gross_share > thresholds.single_child_gross_share_threshold
        and child_abs_z >= parent_abs_z * thresholds.single_child_z_multiplier
    ):
        return "CHILD_WINS"
    return "PARENT_WINS"


def _active_child_group_key(
    parent_features: frozenset[Tuple[str, str]],
    child_features: frozenset[Tuple[str, str]],
) -> Tuple[str, ...]:
    """Определить родственную группу ребёнка относительно родителя.

    Args:
        parent_features: Набор признаков родительского сегмента.
        child_features: Набор признаков дочернего сегмента.

    Returns:
        Кортеж добавленных измерений; дети с одинаковым набором измерений считаются одной родственной группой.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _active_child_group_key(frozenset({('merchant', 'SMB')}), frozenset({('merchant', 'SMB'), ('geo', 'RF')}))
        ('geo',)
    """

    return tuple(sorted(dimension for dimension, _ in child_features - parent_features))


def _segments_overlap_by_atoms(segment_ids: Sequence[str], coverage: Dict[str, frozenset[str]]) -> bool:
    """Проверить, пересекаются ли сегменты по атомарному покрытию.

    Args:
        segment_ids: Идентификаторы сегментов.
        coverage: Словарь атомарного покрытия.

    Returns:
        True, если хотя бы два сегмента имеют общий атом.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _segments_overlap_by_atoms(['a', 'b'], {'a': frozenset({'1'}), 'b': frozenset({'2'})})
        False
    """

    used_atoms: set[str] = set()
    for segment_id in segment_ids:
        atoms = set(coverage.get(str(segment_id), frozenset()))
        if used_atoms & atoms:
            return True
        used_atoms.update(atoms)
    return False


def _split_sibling_groups(
    parent_id: str,
    child_ids: Sequence[str],
    feature_sets: Dict[str, frozenset[Tuple[str, str]]],
    coverage: Dict[str, frozenset[str]],
) -> List[List[str]]:
    """Построить родственные группы детей для правила 1:N.

    Args:
        parent_id: Идентификатор родителя.
        child_ids: Ближайшие активные дети родителя.
        feature_sets: Наборы признаков сегментов.
        coverage: Словарь атомарного покрытия.

    Returns:
        Список групп детей. Если внутри группы остаётся пересечение, группа дробится до одиночных сегментов.

    Raises:
        KeyError: Если для сегмента нет набора признаков.

    Examples:
        >>> features = {'p': frozenset({('m', 'SMB')}), 'c': frozenset({('m', 'SMB'), ('geo', 'RF')})}
        >>> _split_sibling_groups('p', ['c'], features, {'c': frozenset({'1'})})
        [['c']]
    """

    parent_features = feature_sets[parent_id]
    grouped: Dict[Tuple[str, ...], List[str]] = {}
    for child_id in child_ids:
        group_key = _active_child_group_key(parent_features, feature_sets[child_id])
        grouped.setdefault(group_key, []).append(str(child_id))

    result: List[List[str]] = []
    for _, group_ids in sorted(grouped.items(), key=lambda item: (item[0], item[1])):
        ordered_group = sorted(group_ids)
        if _segments_overlap_by_atoms(ordered_group, coverage):
            result.extend([[child_id] for child_id in ordered_group])
        else:
            result.append(ordered_group)
    return result


def _sum_abs_delta(segment_ids: Sequence[str], lookup: Dict[str, pd.Series]) -> float:
    """Посчитать сумму модулей delta GMV по сегментам.

    Args:
        segment_ids: Идентификаторы сегментов.
        lookup: Словарь `segment_id -> строка кандидата`.

    Returns:
        Сумма `abs(wow_delta_gmv)`.

    Raises:
        KeyError: Если сегмент отсутствует в lookup.

    Examples:
        >>> _sum_abs_delta(['a'], {'a': pd.Series({'wow_delta_gmv': -10})})
        10.0
    """

    return float(sum(abs(_safe_float(lookup[str(segment_id)].get("wow_delta_gmv"))) for segment_id in segment_ids))


def _nearest_parent_map(child_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Инвертировать карту ближайших детей в карту ближайших родителей.

    Args:
        child_map: Словарь `parent_id -> child_ids`.

    Returns:
        Словарь `child_id -> parent_ids`.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _nearest_parent_map({'p': ['c']})
        {'c': ['p']}
    """

    result: Dict[str, List[str]] = {}
    for parent_id, child_ids in child_map.items():
        result.setdefault(str(parent_id), result.get(str(parent_id), []))
        for child_id in child_ids:
            result.setdefault(str(child_id), []).append(str(parent_id))
    return {segment_id: sorted(parent_ids) for segment_id, parent_ids in result.items()}


def _segment_feature_set_from_key(segment_key: object) -> frozenset[Tuple[str, str]]:
    """ADDED: Восстановить набор признаков сегмента из человекочитаемого ключа.

    Args:
        segment_key: Значение колонки `segment_key`.

    Returns:
        Набор пар `(dimension, value)`.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _segment_feature_set_from_key('geo=RF × product=QR') == frozenset({('geo', 'RF'), ('product', 'QR')})
        True
    """

    return frozenset(_segment_key_parts(segment_key))


def _build_coverage_from_segment_keys(candidates: pd.DataFrame) -> Dict[str, frozenset[str]]:
    """ADDED: Построить атомарное покрытие по `segment_key` без внешнего `dim_cols`.

    Args:
        candidates: Таблица кандидатов с `segment_id`, `segment_key`, `slice_depth`.

    Returns:
        Словарь `segment_id -> frozenset(atomic_segment_id)`.

    Raises:
        ValueError: Если таблица не содержит физический атомарный слой.

    Examples:
        >>> df = pd.DataFrame([
        ...     {'segment_id': 'p', 'segment_key': 'a=1', 'slice_depth': 1},
        ...     {'segment_id': 'c', 'segment_key': 'a=1 × b=2', 'slice_depth': 2},
        ... ])
        >>> _build_coverage_from_segment_keys(df)['p']
        frozenset({'c'})
    """

    if candidates.empty:
        return {}
    max_depth = int(candidates["slice_depth"].astype(int).max())
    atomic_df = candidates[candidates["slice_depth"].astype(int).eq(max_depth)].copy()
    if atomic_df.empty:
        raise ValueError("Физический атомарный слой search_anomal пуст")

    feature_sets = {
        str(row["segment_id"]): _segment_feature_set_from_key(row.get("segment_key", ""))
        for _, row in candidates.iterrows()
    }
    atomic_features = {
        str(row["segment_id"]): feature_sets[str(row["segment_id"])]
        for _, row in atomic_df.iterrows()
    }

    coverage: Dict[str, frozenset[str]] = {}
    for _, row in candidates.iterrows():
        segment_id = str(row["segment_id"])
        features = feature_sets.get(segment_id, frozenset())
        depth = int(row["slice_depth"])
        if not features and depth == max_depth:
            coverage[segment_id] = frozenset({segment_id})
            continue
        if not features:
            coverage[segment_id] = frozenset()
            continue
        coverage[segment_id] = frozenset(
            atom_id
            for atom_id, atom_features in atomic_features.items()
            if features.issubset(atom_features)
        )
    return coverage


def _same_nonzero_direction(left: float, right: float) -> bool:
    """ADDED: Проверить, что два изменения GMV ненулевые и направлены одинаково.

    Args:
        left: Первое изменение GMV.
        right: Второе изменение GMV.

    Returns:
        True, если оба значения одного знака и не равны нулю.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _same_nonzero_direction(-10, -3)
        True
    """

    return float(left) * float(right) > 0.0


def _parent_child_absorption_k(parent: pd.Series, child: pd.Series) -> float:
    """ADDED: Посчитать коэффициент поглощения родителем дочернего сегмента.

    Args:
        parent: Строка родительского сегмента.
        child: Строка дочернего сегмента.

    Returns:
        Неотрицательный коэффициент `k` или NaN, если расчёт некорректен.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _parent_child_absorption_k(
        ...     pd.Series({'robust_z_capped': 4, 'wow_delta_gmv': 200}),
        ...     pd.Series({'robust_z_capped': 2, 'wow_delta_gmv': 100}),
        ... )
        4.0
    """

    parent_z = abs(_safe_float(parent.get("robust_z_capped")))
    child_z = abs(_safe_float(child.get("robust_z_capped")))
    parent_delta = abs(_safe_float(parent.get("wow_delta_gmv")))
    child_delta = abs(_safe_float(child.get("wow_delta_gmv")))
    if child_z == 0.0 or child_delta == 0.0:
        return math.nan
    return (parent_z / child_z) * (parent_delta / child_delta)


def search_anomal(candidates: pd.DataFrame, thresholds: AnomalyThresholds) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ADDED: Найти итоговые аномалии через активное разбиение снизу вверх.

    Args:
        candidates: Таблица кандидатов после `build_anomaly_candidates` и `add_child_context`.
        thresholds: Пороги алгоритма, включая `parent_child_absorption_k_threshold`.

    Returns:
        Кортеж: итоговые аномалии, диагностика кандидатов, журнал решений.

    Raises:
        ValueError: Если отсутствуют обязательные колонки или физический атомарный слой.

    Examples:
        >>> # final_df, diagnostics, log = search_anomal(candidates, AnomalyThresholds())
    """

    required_columns = {
        "segment_id",
        "segment_key",
        "slice_depth",
        "passes_initial_anomaly_filter",
        "robust_z_capped",
        "wow_delta_gmv",
        "anomaly_score",
    }
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Для search_anomal не хватает колонок: {missing_columns}")

    diagnostics = candidates.copy()
    if diagnostics.empty:
        return diagnostics.copy(), diagnostics, pd.DataFrame()

    diagnostics["segment_id"] = diagnostics["segment_id"].astype(str)
    diagnostics["slice_depth"] = diagnostics["slice_depth"].astype(int)
    max_depth = int(diagnostics["slice_depth"].max())
    coverage = _build_coverage_from_segment_keys(diagnostics)
    feature_sets = {
        str(row["segment_id"]): _segment_feature_set_from_key(row.get("segment_key", ""))
        for _, row in diagnostics.iterrows()
    }

    string_columns = [
        "action",
        "output_block",
        "reason",
        "active_child_ids",
        "active_child_keys",
        "active_parent_ids",
        "active_parent_keys",
        "active_partition_action",
        "active_partition_reason",
        "active_partition_status",
        "active_partition_status_reason",
        "active_graph_status",
        "active_decision_code",
        "active_decision_reason",
        "active_decision_history",
        "conflict_parent_ids",
        "conflict_parent_keys",
        "covered_by_segment",
        "covered_by_segment_key",
        "absorbed_child_ids",
        "absorbed_child_keys",
        "absorbed_child_labels",
        "absorbed_by_parent_ids",
        "absorbed_by_parent_keys",
        "original_atomic_descendants",
        "active_atomic_descendants",
        "residual_atomic_descendants",
        "connection_break_reason",
        "active_relationship_type",
        "absorbed_by_role",
        "absorbed_by_segment_key",
        "absorption_reason",
    ]
    for column in string_columns:
        diagnostics[column] = ""
    diagnostics["is_eligible"] = False
    diagnostics["selection_score"] = pd.to_numeric(diagnostics["anomaly_score"], errors="coerce").fillna(0.0)
    diagnostics["covered_atomic_count"] = diagnostics["segment_id"].map(lambda segment_id: len(coverage.get(str(segment_id), frozenset())))
    diagnostics["technical_child_count"] = 0
    diagnostics["active_child_count"] = 0
    diagnostics["original_atomic_count"] = 0
    diagnostics["active_atomic_count"] = 0
    diagnostics["residual_atomic_count"] = 0
    diagnostics["residual_gross_movement"] = 0.0
    diagnostics["single_child_gross_share"] = math.nan
    diagnostics["anomaly_gross_move"] = math.nan
    diagnostics["active_parent_z"] = math.nan
    diagnostics["active_child_z"] = math.nan
    diagnostics["parent_child_absorption_k"] = math.nan
    diagnostics["active_decision_iteration"] = pd.Series(pd.NA, index=diagnostics.index, dtype="Int64")
    diagnostics["connection_break_flag"] = False
    diagnostics["is_active"] = False
    diagnostics["is_resolved"] = False
    diagnostics["is_natural_terminal"] = False
    diagnostics["is_terminal_current"] = False
    diagnostics["selected"] = False
    diagnostics["selection_exclusion_reason"] = ""

    anomaly_mask = diagnostics["slice_depth"].gt(0) & diagnostics["passes_initial_anomaly_filter"].astype(bool)
    parseable_mask = diagnostics["segment_id"].map(lambda segment_id: bool(feature_sets.get(str(segment_id), frozenset())))
    anomaly_df = diagnostics[anomaly_mask & parseable_mask].copy()

    for index, row in diagnostics.iterrows():
        if int(row["slice_depth"]) == 0:
            diagnostics.at[index, "action"] = "исключён"
            diagnostics.at[index, "output_block"] = "исключён"
            diagnostics.at[index, "reason"] = "total не выбирается как аномальный сегмент"
            diagnostics.at[index, "active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
            continue
        if not bool(row.get("passes_initial_anomaly_filter", False)):
            diagnostics.at[index, "action"] = "исключён"
            diagnostics.at[index, "output_block"] = "исключён"
            diagnostics.at[index, "reason"] = "сегмент не прошёл предварительный фильтр аномальности"
            diagnostics.at[index, "active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
            continue
        if not bool(feature_sets.get(str(row["segment_id"]), frozenset())):
            diagnostics.at[index, "action"] = "исключён"
            diagnostics.at[index, "output_block"] = "исключён"
            diagnostics.at[index, "reason"] = "не удалось восстановить признаки сегмента из segment_key"
            diagnostics.at[index, "active_graph_status"] = "UNPARSEABLE_SEGMENT_KEY"
            diagnostics.at[index, "selection_exclusion_reason"] = diagnostics.at[index, "reason"]
            continue
        diagnostics.at[index, "action"] = "кандидат_active-разбиения"
        diagnostics.at[index, "output_block"] = "кандидат active-разбиения"
        diagnostics.at[index, "reason"] = "сегмент прошёл первичный фильтр и участвует в active-разбиении"
        diagnostics.at[index, "is_eligible"] = True

    if anomaly_df.empty:
        diagnostics.loc[diagnostics["selection_exclusion_reason"].eq(""), "selection_exclusion_reason"] = diagnostics["reason"]
        return diagnostics.iloc[0:0].copy(), diagnostics, pd.DataFrame()

    lookup = {
        str(row["segment_id"]): row.copy()
        for _, row in anomaly_df.iterrows()
    }
    anomaly_ids = sorted(lookup)
    original_child_map = _nearest_active_child_map(anomaly_ids, feature_sets)
    original_parent_map = _nearest_parent_map(original_child_map)
    natural_terminals = {
        segment_id for segment_id in anomaly_ids if not original_child_map.get(segment_id, [])
    }

    active_ids = set(anomaly_ids)
    selected_ids: set[str] = set()
    removed_ids: set[str] = set()
    graph_status = {segment_id: "ACTIVE_PENDING" for segment_id in anomaly_ids}
    status_reason = {segment_id: "ожидает решения active-разбиения" for segment_id in anomaly_ids}
    last_children: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    last_parents: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    last_decision: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    last_reason: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    last_iteration: Dict[str, Optional[int]] = {segment_id: None for segment_id in anomaly_ids}
    last_k: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_parent_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_child_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_relationship_type: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    absorbed_by_role: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    absorbed_by_segment_key: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    absorption_reason: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    conflict_parents: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    connection_break_flag = {segment_id: False for segment_id in anomaly_ids}
    connection_break_reason = {segment_id: "" for segment_id in anomaly_ids}
    covered_by: Dict[str, str] = {}
    absorbed_lineage: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    decision_history: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    selection_scores = {
        segment_id: _safe_float(lookup[segment_id].get("selection_score"))
        for segment_id in anomaly_ids
    }
    decision_log_rows: List[Dict[str, object]] = []

    def log_decision(
        iteration: int,
        event_type: str,
        parent_id: str,
        child_id: str,
        relationship_type: str,
        decision_code: str,
        applied: bool,
        reason: str,
        k_value: float = math.nan,
        winner_parent_id: str = "",
    ) -> None:
        """ADDED: Записать решение active-разбиения в журнал."""

        parent_row = lookup.get(parent_id)
        child_row = lookup.get(child_id)
        decision_log_rows.append(
            {
                "iteration": iteration,
                "event_type": event_type,
                "relationship_type": relationship_type,
                "parent_id": parent_id,
                "parent_key": "" if parent_row is None else str(parent_row["segment_key"]),
                "parent_depth": pd.NA if parent_row is None else int(parent_row["slice_depth"]),
                "child_id": child_id,
                "child_key": "" if child_row is None else str(child_row["segment_key"]),
                "child_depth": pd.NA if child_row is None else int(child_row["slice_depth"]),
                "parent_abs_z": math.nan if parent_row is None else abs(_safe_float(parent_row.get("robust_z_capped"))),
                "child_abs_z": math.nan if child_row is None else abs(_safe_float(child_row.get("robust_z_capped"))),
                "parent_delta_gmv": math.nan if parent_row is None else _safe_float(parent_row.get("wow_delta_gmv")),
                "child_delta_gmv": math.nan if child_row is None else _safe_float(child_row.get("wow_delta_gmv")),
                "k_value": k_value,
                "k_threshold": thresholds.parent_child_absorption_k_threshold,
                "pair_decision": decision_code,
                "conflict_code": decision_code,
                "winner_parent_id": winner_parent_id,
                "winner_parent_key": "" if not winner_parent_id else str(lookup[winner_parent_id]["segment_key"]),
                "applied": applied,
                "reason": reason,
            }
        )

    def mark_removed(segment_id: str, status: str, reason: str, covered_by_id: str = "") -> None:
        """REMOVED: Вывести сегмент из активного графа."""

        active_ids.discard(segment_id)
        removed_ids.add(segment_id)
        graph_status[segment_id] = status
        status_reason[segment_id] = reason
        last_decision[segment_id] = status
        last_reason[segment_id] = reason
        if covered_by_id:
            covered_by[segment_id] = covered_by_id
            absorbed_by_segment_key[segment_id] = str(lookup[covered_by_id]["segment_key"]) if covered_by_id in lookup else ""
            absorption_reason[segment_id] = reason

    max_iterations = max(2, len(anomaly_ids) * 4 + 4)
    for iteration in range(1, max_iterations + 1):
        if not active_ids:
            break

        child_map = _nearest_active_child_map(sorted(active_ids), feature_sets)
        parent_map = _nearest_parent_map(child_map)
        for segment_id in anomaly_ids:
            last_children[segment_id] = child_map.get(segment_id, last_children[segment_id])
            last_parents[segment_id] = parent_map.get(segment_id, [])

        current_depth = max(int(lookup[segment_id]["slice_depth"]) for segment_id in active_ids)
        deepest_ids = sorted(
            segment_id
            for segment_id in active_ids
            if int(lookup[segment_id]["slice_depth"]) == current_depth
        )
        if not deepest_ids:
            break

        deepest_parent_ids = sorted(
            {
                parent_id
                for child_id in deepest_ids
                for parent_id in parent_map.get(child_id, [])
                if parent_id in active_ids
            }
        )
        if not deepest_parent_ids:
            for segment_id in deepest_ids:
                active_ids.discard(segment_id)
                selected_ids.add(segment_id)
                if graph_status[segment_id] != "ACTIVE_PENDING":
                    status_reason[segment_id] = status_reason[segment_id]
                elif original_parent_map.get(segment_id):
                    graph_status[segment_id] = "ORPHAN_AFTER_REWIRE"
                    status_reason[segment_id] = "сегмент остался без активных родителей после перестроения связей"
                    connection_break_flag[segment_id] = True
                    connection_break_reason[segment_id] = status_reason[segment_id]
                else:
                    graph_status[segment_id] = "INITIALLY_UNLINKED_TERMINAL"
                    status_reason[segment_id] = "сегмент изначально не имел аномальных родителей в active-графе"
                last_decision[segment_id] = graph_status[segment_id]
                last_reason[segment_id] = status_reason[segment_id]
                last_iteration[segment_id] = iteration
                decision_history[segment_id].append(f"iteration={iteration}: {graph_status[segment_id]}")
            continue

        changed = False
        processed_children: set[str] = set()
        processed_parents: set[str] = set()

        parent_children = {
            parent_id: [child_id for child_id in child_map.get(parent_id, []) if child_id in active_ids]
            for parent_id in deepest_parent_ids
        }
        child_parents = {
            child_id: [parent_id for parent_id in parent_map.get(child_id, []) if parent_id in active_ids]
            for child_id in deepest_ids
        }

        mn_children = sorted(
            child_id
            for child_id, parent_ids in child_parents.items()
            if len(parent_ids) > 1 and any(len(parent_children.get(parent_id, [])) > 1 for parent_id in parent_ids)
        )
        if mn_children:
            mn_parents = sorted({parent_id for child_id in mn_children for parent_id in child_parents[child_id]})
            reason = "M:N связь временно не разрешается; родители удалены, наиболее глубокие дочерние сегменты оставлены активными"
            for parent_id in mn_parents:
                mark_removed(parent_id, "DEFER_M_N_STUB_PARENT_REMOVED", reason)
                last_relationship_type[parent_id] = "M:N"
                conflict_parents[parent_id] = mn_parents
                decision_history[parent_id].append(f"iteration={iteration}: DEFER_M_N_STUB")
                processed_parents.add(parent_id)
            for child_id in mn_children:
                graph_status[child_id] = "DEFER_M_N_STUB"
                status_reason[child_id] = reason
                last_decision[child_id] = "DEFER_M_N_STUB"
                last_reason[child_id] = reason
                last_iteration[child_id] = iteration
                last_relationship_type[child_id] = "M:N"
                conflict_parents[child_id] = mn_parents
                decision_history[child_id].append(f"iteration={iteration}: DEFER_M_N_STUB")
                processed_children.add(child_id)
                for parent_id in child_parents[child_id]:
                    log_decision(iteration, "RELATIONSHIP_STUB", parent_id, child_id, "M:N", "DEFER_M_N_STUB", True, reason)
            changed = True

        for child_id in deepest_ids:
            if child_id not in active_ids or child_id in processed_children:
                continue
            parent_ids = [parent_id for parent_id in child_parents.get(child_id, []) if parent_id in active_ids]
            if not parent_ids:
                continue
            if not all(len(parent_children.get(parent_id, [])) == 1 for parent_id in parent_ids):
                continue

            child_row = lookup[child_id]
            child_delta = _safe_float(child_row.get("wow_delta_gmv"))
            evaluations: List[Dict[str, object]] = []
            for parent_id in parent_ids:
                parent_row = lookup[parent_id]
                parent_delta = _safe_float(parent_row.get("wow_delta_gmv"))
                k_value = _parent_child_absorption_k(parent_row, child_row)
                same_direction = _same_nonzero_direction(parent_delta, child_delta)
                can_absorb = (
                    same_direction
                    and not math.isnan(k_value)
                    and k_value >= thresholds.parent_child_absorption_k_threshold
                )
                evaluations.append(
                    {
                        "parent_id": parent_id,
                        "k": k_value,
                        "can_absorb": can_absorb,
                        "same_direction": same_direction,
                    }
                )

            winners = [evaluation for evaluation in evaluations if bool(evaluation["can_absorb"])]
            relationship_type = "N:1" if len(parent_ids) > 1 else "1:1"
            if winners:
                winner = sorted(
                    winners,
                    key=lambda evaluation: (
                        _safe_float(evaluation.get("k"), math.nan),
                        abs(_safe_float(lookup[str(evaluation["parent_id"])].get("robust_z_capped"))),
                        str(evaluation["parent_id"]),
                    ),
                    reverse=True,
                )[0]
                winner_parent_id = str(winner["parent_id"])
                reason = (
                    f"родитель поглотил ребёнка по k={float(winner['k']):.6f} "
                    f">= {thresholds.parent_child_absorption_k_threshold:.2f}"
                )
                mark_removed(child_id, "ABSORBED_BY_PARENT_K", reason, winner_parent_id)
                last_relationship_type[child_id] = relationship_type
                absorbed_by_role[child_id] = "родитель"
                absorbed_lineage[winner_parent_id] = list(
                    dict.fromkeys([*absorbed_lineage.get(winner_parent_id, []), child_id, *absorbed_lineage.get(child_id, [])])
                )
                selection_scores[winner_parent_id] = max(selection_scores[winner_parent_id], selection_scores.get(child_id, 0.0))
                graph_status[winner_parent_id] = "PARENT_ABSORBS_CHILD_BY_K"
                status_reason[winner_parent_id] = reason
                last_decision[winner_parent_id] = "PARENT_ABSORBS_CHILD_BY_K"
                last_reason[winner_parent_id] = reason
                last_iteration[winner_parent_id] = iteration
                last_relationship_type[winner_parent_id] = relationship_type
                last_k[winner_parent_id] = float(winner["k"])
                last_parent_z[winner_parent_id] = abs(_safe_float(lookup[winner_parent_id].get("robust_z_capped")))
                last_child_z[winner_parent_id] = abs(_safe_float(child_row.get("robust_z_capped")))
                for evaluation in evaluations:
                    parent_id = str(evaluation["parent_id"])
                    k_value = _safe_float(evaluation.get("k"), math.nan)
                    if parent_id != winner_parent_id:
                        mark_removed(parent_id, "REMOVED_BY_STRONGER_PARENT_K", f"уступил родителю {lookup[winner_parent_id]['segment_key']} с максимальным k", winner_parent_id)
                        absorbed_by_role[parent_id] = "родитель-конкурент"
                    last_relationship_type[parent_id] = relationship_type
                    last_k[parent_id] = k_value
                    log_decision(
                        iteration,
                        "RULE_PARENT_CHILD_K",
                        parent_id,
                        child_id,
                        "N_PARENTS_TO_1_CHILD" if len(parent_ids) > 1 else "1_PARENT_TO_1_CHILD",
                        "PARENT_ABSORBS_CHILD_BY_K" if parent_id == winner_parent_id else "REMOVED_BY_STRONGER_PARENT_K",
                        parent_id == winner_parent_id,
                        reason if parent_id == winner_parent_id else status_reason[parent_id],
                        k_value,
                        winner_parent_id,
                    )
                processed_children.add(child_id)
                processed_parents.update(parent_ids)
                changed = True
                continue

            reason = (
                f"ни один родитель не достиг k >= {thresholds.parent_child_absorption_k_threshold:.2f} "
                "при одинаковом направлении Delta GMV; ребёнок оставлен активным"
            )
            for evaluation in evaluations:
                parent_id = str(evaluation["parent_id"])
                k_value = _safe_float(evaluation.get("k"), math.nan)
                mark_removed(parent_id, "CHILD_KEPT_PARENTS_REMOVED", reason, child_id)
                last_relationship_type[parent_id] = relationship_type
                absorbed_by_role[parent_id] = "ребёнок"
                last_k[parent_id] = k_value
                log_decision(
                    iteration,
                    "RULE_PARENT_CHILD_K",
                    parent_id,
                    child_id,
                    "N_PARENTS_TO_1_CHILD" if len(parent_ids) > 1 else "1_PARENT_TO_1_CHILD",
                    "CHILD_KEPT_PARENTS_REMOVED",
                    True,
                    reason,
                    k_value,
                    "",
                )
            processed_children.add(child_id)
            processed_parents.update(parent_ids)
            changed = True

        for parent_id in deepest_parent_ids:
            if parent_id not in active_ids or parent_id in processed_parents:
                continue
            active_children = [child_id for child_id in child_map.get(parent_id, []) if child_id in active_ids]
            if len(active_children) <= 1:
                continue
            if not all(len([active_parent for active_parent in parent_map.get(child_id, []) if active_parent in active_ids]) == 1 for child_id in active_children):
                continue

            reason = "временное правило: родитель явно поглощает несколько дочерних сегментов"
            graph_status[parent_id] = "TEMP_PARENT_ABSORBS_CHILDREN"
            status_reason[parent_id] = reason
            last_decision[parent_id] = "TEMP_PARENT_ABSORBS_CHILDREN"
            last_reason[parent_id] = reason
            last_iteration[parent_id] = iteration
            last_relationship_type[parent_id] = "1:N"
            for child_id in active_children:
                mark_removed(child_id, "ABSORBED_BY_TEMP_PARENT", reason, parent_id)
                last_relationship_type[child_id] = "1:N"
                absorbed_by_role[child_id] = "родитель"
                absorbed_lineage[parent_id] = list(
                    dict.fromkeys([*absorbed_lineage.get(parent_id, []), child_id, *absorbed_lineage.get(child_id, [])])
                )
                selection_scores[parent_id] = max(selection_scores[parent_id], selection_scores.get(child_id, 0.0))
                log_decision(iteration, "TEMP_RULE", parent_id, child_id, "1_PARENT_TO_N_CHILDREN", "TEMP_PARENT_ABSORBS_CHILDREN", True, reason, math.nan, parent_id)
                processed_children.add(child_id)
            decision_history[parent_id].append(f"iteration={iteration}: TEMP_PARENT_ABSORBS_CHILDREN {len(active_children)} children")
            processed_parents.add(parent_id)
            changed = True

        if changed:
            continue

        reason = "сложная связь не попала под правила k или временного поглощения; применена M:N заглушка"
        fallback_parents = sorted({parent_id for child_id in deepest_ids for parent_id in child_parents.get(child_id, []) if parent_id in active_ids})
        if fallback_parents:
            for parent_id in fallback_parents:
                mark_removed(parent_id, "DEFER_M_N_STUB_PARENT_REMOVED", reason)
                last_relationship_type[parent_id] = "M:N"
                decision_history[parent_id].append(f"iteration={iteration}: DEFER_M_N_STUB_FALLBACK")
            for child_id in deepest_ids:
                if child_id not in active_ids:
                    continue
                graph_status[child_id] = "DEFER_M_N_STUB"
                status_reason[child_id] = reason
                last_decision[child_id] = "DEFER_M_N_STUB"
                last_reason[child_id] = reason
                last_iteration[child_id] = iteration
                last_relationship_type[child_id] = "M:N"
                for parent_id in child_parents.get(child_id, []):
                    log_decision(iteration, "RELATIONSHIP_STUB", parent_id, child_id, "M:N_FALLBACK", "DEFER_M_N_STUB", True, reason)
            continue

        for segment_id in deepest_ids:
            active_ids.discard(segment_id)
            selected_ids.add(segment_id)
            graph_status[segment_id] = "UNRESOLVED_TERMINAL_FALLBACK"
            status_reason[segment_id] = "сегмент выбран как терминальный fallback без активных применимых правил"

    if active_ids:
        for segment_id in sorted(active_ids):
            selected_ids.add(segment_id)
            graph_status[segment_id] = "ITERATION_LIMIT_TERMINAL"
            status_reason[segment_id] = "достигнут безопасный лимит итераций; сегмент оставлен активным для диагностики"
            last_decision[segment_id] = graph_status[segment_id]
            last_reason[segment_id] = status_reason[segment_id]

    ordered_selected = sorted(
        selected_ids,
        key=lambda segment_id: (
            int(lookup[segment_id]["slice_depth"]),
            selection_scores.get(segment_id, 0.0),
            abs(_safe_float(lookup[segment_id].get("robust_z_capped"))),
            abs(_safe_float(lookup[segment_id].get("wow_delta_gmv"))),
            str(lookup[segment_id]["segment_key"]),
        ),
        reverse=True,
    )
    final_selected_ids: set[str] = set()
    used_atoms: set[str] = set()
    for segment_id in ordered_selected:
        atoms = set(coverage.get(segment_id, frozenset()))
        if not atoms:
            graph_status[segment_id] = "FINAL_NO_ATOMIC_COVERAGE_SUPPRESSED"
            status_reason[segment_id] = "нет атомарного покрытия для итогового выбора"
            continue
        if used_atoms & atoms:
            graph_status[segment_id] = "FINAL_ATOMIC_OVERLAP_SUPPRESSED"
            status_reason[segment_id] = "исключён на финальной проверке непересечения атомарного покрытия"
            continue
        final_selected_ids.add(segment_id)
        used_atoms.update(atoms)

    final_child_map = _nearest_active_child_map(sorted(final_selected_ids), feature_sets) if final_selected_ids else {}
    final_parent_map = _nearest_parent_map(final_child_map)
    index_by_id = {
        str(segment_id): index
        for index, segment_id in diagnostics["segment_id"].items()
    }

    for segment_id in anomaly_ids:
        index = index_by_id[segment_id]
        children = final_child_map.get(segment_id, [])
        parents = final_parent_map.get(segment_id, [])
        covered_by_id = covered_by.get(segment_id, "")
        original_atom_ids = sorted(coverage.get(segment_id, frozenset()))
        active_atom_ids = original_atom_ids if segment_id in final_selected_ids else []
        lineage = absorbed_lineage.get(segment_id, [])
        lineage_keys = [str(lookup[child_id]["segment_key"]) for child_id in lineage if child_id in lookup]
        lineage_labels = [
            _relative_child_segment_name(lookup[segment_id]["segment_key"], lookup[child_id]["segment_key"])
            for child_id in lineage
            if child_id in lookup
        ]
        conflict_ids = conflict_parents.get(segment_id, [])

        diagnostics.at[index, "active_child_ids"] = " || ".join(children)
        diagnostics.at[index, "active_child_keys"] = " || ".join(str(lookup[child_id]["segment_key"]) for child_id in children)
        diagnostics.at[index, "active_parent_ids"] = " || ".join(parents)
        diagnostics.at[index, "active_parent_keys"] = " || ".join(str(lookup[parent_id]["segment_key"]) for parent_id in parents)
        diagnostics.at[index, "technical_child_count"] = len(children)
        diagnostics.at[index, "active_child_count"] = len(children)
        diagnostics.at[index, "active_partition_action"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_partition_reason"] = last_reason[segment_id] or status_reason[segment_id]
        diagnostics.at[index, "active_partition_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_partition_status_reason"] = status_reason[segment_id]
        diagnostics.at[index, "active_graph_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_decision_code"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_decision_reason"] = last_reason[segment_id] or status_reason[segment_id]
        diagnostics.at[index, "active_decision_history"] = " || ".join(decision_history[segment_id])
        diagnostics.at[index, "active_decision_iteration"] = last_iteration[segment_id]
        diagnostics.at[index, "active_relationship_type"] = last_relationship_type[segment_id]
        diagnostics.at[index, "absorbed_by_role"] = absorbed_by_role[segment_id]
        diagnostics.at[index, "absorbed_by_segment_key"] = absorbed_by_segment_key[segment_id]
        diagnostics.at[index, "absorption_reason"] = absorption_reason[segment_id]
        diagnostics.at[index, "parent_child_absorption_k"] = last_k[segment_id]
        diagnostics.at[index, "active_parent_z"] = last_parent_z[segment_id]
        diagnostics.at[index, "active_child_z"] = last_child_z[segment_id]
        diagnostics.at[index, "conflict_parent_ids"] = " || ".join(conflict_ids)
        diagnostics.at[index, "conflict_parent_keys"] = " || ".join(str(lookup[parent_id]["segment_key"]) for parent_id in conflict_ids if parent_id in lookup)
        diagnostics.at[index, "covered_by_segment"] = covered_by_id
        diagnostics.at[index, "covered_by_segment_key"] = str(lookup[covered_by_id]["segment_key"]) if covered_by_id in lookup else ""
        diagnostics.at[index, "absorbed_child_ids"] = " || ".join(lineage)
        diagnostics.at[index, "absorbed_child_keys"] = " || ".join(lineage_keys)
        diagnostics.at[index, "absorbed_child_labels"] = "; ".join(lineage_labels)
        diagnostics.at[index, "absorbed_by_parent_ids"] = covered_by_id
        diagnostics.at[index, "absorbed_by_parent_keys"] = str(lookup[covered_by_id]["segment_key"]) if covered_by_id in lookup else ""
        diagnostics.at[index, "original_atomic_descendants"] = " || ".join(original_atom_ids)
        diagnostics.at[index, "active_atomic_descendants"] = " || ".join(active_atom_ids)
        diagnostics.at[index, "residual_atomic_descendants"] = " || ".join(active_atom_ids)
        diagnostics.at[index, "original_atomic_count"] = len(original_atom_ids)
        diagnostics.at[index, "active_atomic_count"] = len(active_atom_ids)
        diagnostics.at[index, "residual_atomic_count"] = len(active_atom_ids)
        diagnostics.at[index, "residual_gross_movement"] = _sum_abs_delta(active_atom_ids, {str(row["segment_id"]): row for _, row in diagnostics.iterrows()}) if active_atom_ids else 0.0
        diagnostics.at[index, "connection_break_flag"] = connection_break_flag[segment_id]
        diagnostics.at[index, "connection_break_reason"] = connection_break_reason[segment_id]
        diagnostics.at[index, "is_active"] = segment_id in final_selected_ids
        diagnostics.at[index, "is_resolved"] = segment_id in final_selected_ids
        diagnostics.at[index, "is_natural_terminal"] = segment_id in natural_terminals
        diagnostics.at[index, "is_terminal_current"] = segment_id in final_selected_ids and not children
        diagnostics.at[index, "selection_score"] = selection_scores.get(segment_id, diagnostics.at[index, "selection_score"])

        if segment_id in final_selected_ids:
            if graph_status[segment_id] == "DEFER_M_N_STUB":
                diagnostics.at[index, "action"] = "DEFER_M_N_STUB"
                diagnostics.at[index, "output_block"] = "M:N заглушка"
            elif graph_status[segment_id] in {"PARENT_ABSORBS_CHILD_BY_K", "TEMP_PARENT_ABSORBS_CHILDREN"}:
                diagnostics.at[index, "action"] = graph_status[segment_id]
                diagnostics.at[index, "output_block"] = "поглощение родителем"
            else:
                diagnostics.at[index, "action"] = graph_status[segment_id]
                diagnostics.at[index, "output_block"] = "терминальная аномалия"
            diagnostics.at[index, "reason"] = status_reason[segment_id]

    diagnostics["selected"] = diagnostics["segment_id"].astype(str).isin(final_selected_ids)
    for index, row in diagnostics.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id in final_selected_ids:
            diagnostics.at[index, "selection_exclusion_reason"] = ""
        elif segment_id in lookup:
            diagnostics.at[index, "selection_exclusion_reason"] = (
                f"не выбран search_anomal: {graph_status[segment_id]} - {status_reason[segment_id]}"
            )
        elif not str(row.get("selection_exclusion_reason", "")).strip():
            diagnostics.at[index, "selection_exclusion_reason"] = str(row.get("reason", ""))

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

    decision_log = pd.DataFrame(decision_log_rows)
    if not decision_log.empty:
        decision_log = decision_log.sort_values(
            ["iteration", "event_type", "parent_key", "child_key"],
            kind="stable",
        ).reset_index(drop=True)
    return final_df, diagnostics, decision_log


def _evaluate_1n_group(
    parent_id: str,
    group_child_ids: Sequence[str],
    direct_all_children: Dict[str, List[str]],
    feature_sets: Dict[str, frozenset[Tuple[str, str]]],
    lookup: Dict[str, pd.Series],
    all_lookup: Dict[str, pd.Series],
    thresholds: AnomalyThresholds,
) -> Dict[str, object]:
    """Применить правило 1:N к одной непересекающейся родственной группе.

    Args:
        parent_id: Идентификатор родителя.
        group_child_ids: Дети активной родственной группы.
        direct_all_children: Ближайшие дети по всем входным сегментам.
        feature_sets: Наборы признаков сегментов.
        lookup: Аномальные сегменты активного графа.
        all_lookup: Все сегменты из результата build_anomaly_candidates.
        thresholds: Пороги `agm`, `gm`, `z_c_greater`.

    Returns:
        Словарь с решением `PARENT_ABSORBS` или `CHILD_DOMINATES`.

    Raises:
        KeyError: Если сегмент отсутствует в lookup.

    Examples:
        >>> # decision = _evaluate_1n_group('p', ['c'], {}, {}, {}, {}, AnomalyThresholds())
    """

    ordered_children = sorted(str(child_id) for child_id in group_child_ids)
    active_gross = _sum_abs_delta(ordered_children, lookup)
    parent_features = feature_sets[parent_id]
    group_keys = {
        _active_child_group_key(parent_features, feature_sets[child_id])
        for child_id in ordered_children
    }
    all_group_children = [
        child_id
        for child_id in direct_all_children.get(parent_id, [])
        if child_id in all_lookup
        and _active_child_group_key(parent_features, feature_sets[child_id]) in group_keys
    ]
    all_gross = _sum_abs_delta(all_group_children, all_lookup) if all_group_children else active_gross
    anomaly_gross_move = 0.0 if all_gross == 0.0 else active_gross / all_gross
    parent_abs_z = abs(_safe_float(lookup[parent_id].get("robust_z_capped")))

    child_rows: List[Dict[str, object]] = []
    for child_id in ordered_children:
        gross_move = 0.0 if active_gross == 0.0 else abs(_safe_float(lookup[child_id].get("wow_delta_gmv"))) / active_gross
        child_abs_z = abs(_safe_float(lookup[child_id].get("robust_z_capped")))
        child_rows.append(
            {
                "child_id": child_id,
                "gross_move": gross_move,
                "child_abs_z": child_abs_z,
                "dominates": (
                    child_abs_z >= parent_abs_z * thresholds.single_child_z_multiplier
                    and gross_move > thresholds.single_child_gross_share_threshold
                ),
            }
        )

    if anomaly_gross_move < thresholds.anomaly_gross_move_threshold:
        return {
            "decision": "PARENT_ABSORBS",
            "dominant_child_id": "",
            "anomaly_gross_move": anomaly_gross_move,
            "gross_move": math.nan,
            "child_abs_z": math.nan,
            "parent_abs_z": parent_abs_z,
            "reason": (
                f"anomaly_gross_move={anomaly_gross_move:.6f} < "
                f"agm={thresholds.anomaly_gross_move_threshold:.2f}"
            ),
        }

    dominant_rows = [row for row in child_rows if bool(row["dominates"])]
    if dominant_rows:
        dominant = sorted(
            dominant_rows,
            key=lambda row: (
                float(row["gross_move"]),
                float(row["child_abs_z"]),
                _safe_float(lookup[str(row["child_id"])].get("selection_score")),
                str(row["child_id"]),
            ),
            reverse=True,
        )[0]
        return {
            "decision": "CHILD_DOMINATES",
            "dominant_child_id": str(dominant["child_id"]),
            "anomaly_gross_move": anomaly_gross_move,
            "gross_move": float(dominant["gross_move"]),
            "child_abs_z": float(dominant["child_abs_z"]),
            "parent_abs_z": parent_abs_z,
            "reason": (
                f"child dominates: anomaly_gross_move={anomaly_gross_move:.6f}; "
                f"gross_move={float(dominant['gross_move']):.6f}; "
                f"|z_child|={float(dominant['child_abs_z']):.6f}; "
                f"|z_parent|={parent_abs_z:.6f}"
            ),
        }

    return {
        "decision": "PARENT_ABSORBS",
        "dominant_child_id": "",
        "anomaly_gross_move": anomaly_gross_move,
        "gross_move": max((float(row["gross_move"]) for row in child_rows), default=math.nan),
        "child_abs_z": max((float(row["child_abs_z"]) for row in child_rows), default=math.nan),
        "parent_abs_z": parent_abs_z,
        "reason": (
            f"no dominant child: anomaly_gross_move={anomaly_gross_move:.6f}; "
            f"required gross_move>{thresholds.single_child_gross_share_threshold:.2f} "
            f"and |z_child|>=|z_parent|*{thresholds.single_child_z_multiplier:.2f}"
        ),
    }


def _select_active_partition_anomalies_v2(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    thresholds: AnomalyThresholds,
    dim_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выбрать аномалии по новому алгоритму active-разбиения снизу вверх.

    Args:
        candidates: Кандидаты после build_anomaly_candidates и классификации.
        coverage: Атомарное покрытие сегментов.
        thresholds: Пороги `agm`, `gm`, `z_c_greater`.
        dim_cols: Измерения сегментации.

    Returns:
        Итоговые аномалии, диагностика кандидатов и журнал решений.

    Raises:
        ValueError: Если отсутствуют измерения или атомарный слой.
        RuntimeError: Если итерационный проход не сходится.

    Examples:
        >>> # final_df, diagnostics, log = _select_active_partition_anomalies_v2(candidates, coverage, AnomalyThresholds(), dims)
    """

    diagnostics = candidates.copy()
    if diagnostics.empty:
        return diagnostics.copy(), diagnostics, pd.DataFrame()
    if not dim_cols:
        raise ValueError("Для active-разбиения нужен хотя бы один признак")

    max_physical_depth = int(diagnostics["slice_depth"].astype(int).max())
    atomic_df = diagnostics[diagnostics["slice_depth"].astype(int) == max_physical_depth].copy()
    if atomic_df.empty:
        raise ValueError("Физический атомарный слой active-разбиения пуст")

    anomaly_mask = (
        diagnostics["slice_depth"].astype(int).gt(0)
        & diagnostics["passes_initial_anomaly_filter"].astype(bool)
    )
    anomaly_df = diagnostics[anomaly_mask].copy()

    string_columns = [
        "active_child_ids",
        "active_child_keys",
        "active_parent_ids",
        "active_parent_keys",
        "active_partition_action",
        "active_partition_reason",
        "active_partition_status",
        "active_partition_status_reason",
        "active_graph_status",
        "active_decision_code",
        "active_decision_reason",
        "active_decision_history",
        "conflict_parent_ids",
        "conflict_parent_keys",
        "covered_by_segment",
        "covered_by_segment_key",
        "absorbed_child_ids",
        "absorbed_child_keys",
        "absorbed_child_labels",
        "absorbed_by_parent_ids",
        "absorbed_by_parent_keys",
        "original_atomic_descendants",
        "active_atomic_descendants",
        "residual_atomic_descendants",
        "connection_break_reason",
    ]
    for column in string_columns:
        diagnostics[column] = ""
    diagnostics["technical_child_count"] = 0
    diagnostics["active_child_count"] = 0
    diagnostics["original_atomic_count"] = 0
    diagnostics["active_atomic_count"] = 0
    diagnostics["residual_atomic_count"] = 0
    diagnostics["residual_gross_movement"] = 0.0
    diagnostics["single_child_gross_share"] = math.nan
    diagnostics["anomaly_gross_move"] = math.nan
    diagnostics["active_parent_z"] = math.nan
    diagnostics["active_child_z"] = math.nan
    diagnostics["active_decision_iteration"] = pd.Series(pd.NA, index=diagnostics.index, dtype="Int64")
    diagnostics["connection_break_flag"] = False
    diagnostics["is_active"] = False
    diagnostics["is_resolved"] = False
    diagnostics["is_natural_terminal"] = False
    diagnostics["is_terminal_current"] = False
    diagnostics["covered_atomic_count"] = diagnostics["segment_id"].map(
        lambda segment_id: len(coverage.get(str(segment_id), frozenset()))
    )
    diagnostics["selected"] = False
    diagnostics["selection_exclusion_reason"] = ""

    if anomaly_df.empty:
        diagnostics["active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
        diagnostics["selection_exclusion_reason"] = diagnostics["reason"].astype(str)
        return diagnostics.iloc[0:0].copy(), diagnostics, pd.DataFrame()

    all_lookup = {
        str(row["segment_id"]): row.copy()
        for _, row in diagnostics[diagnostics["slice_depth"].astype(int).gt(0)].iterrows()
    }
    lookup = {
        str(row["segment_id"]): row.copy()
        for _, row in anomaly_df.iterrows()
    }
    anomaly_ids = sorted(lookup)
    all_ids = sorted(all_lookup)
    feature_sets = {
        segment_id: _segment_feature_set(row, dim_cols)
        for segment_id, row in {**all_lookup, **lookup}.items()
    }
    direct_all_children = _nearest_active_child_map(all_ids, feature_sets)
    original_anomaly_children = _nearest_active_child_map(anomaly_ids, feature_sets)
    original_anomaly_parents = _nearest_parent_map(original_anomaly_children)
    natural_terminals = {
        segment_id for segment_id in anomaly_ids if not original_anomaly_children.get(segment_id, [])
    }

    active_ids = set(anomaly_ids)
    selected_ids: set[str] = set()
    removed_ids: set[str] = set()
    graph_status = {segment_id: "ACTIVE_PENDING" for segment_id in anomaly_ids}
    status_reason = {segment_id: "waiting for bottom-up active partition decision" for segment_id in anomaly_ids}
    last_children: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    last_parents: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    last_decision: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    last_reason: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    last_iteration: Dict[str, Optional[int]] = {segment_id: None for segment_id in anomaly_ids}
    last_gross_share: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_anomaly_gross_move: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_parent_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_child_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    connection_break_flag = {segment_id: False for segment_id in anomaly_ids}
    connection_break_reason = {segment_id: "" for segment_id in anomaly_ids}
    covered_by: Dict[str, str] = {}
    absorbed_lineage: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    absorbed_all: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    decision_history: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    selection_scores = {
        segment_id: _safe_float(lookup[segment_id].get("selection_score"))
        for segment_id in anomaly_ids
    }
    decision_log_rows: List[Dict[str, object]] = []

    def mark_removed(segment_id: str, status: str, reason: str, covered_by_id: str = "") -> None:
        """REMOVED: вывести сегмент из активного графа с диагностикой."""

        active_ids.discard(segment_id)
        removed_ids.add(segment_id)
        graph_status[segment_id] = status
        status_reason[segment_id] = reason
        last_decision[segment_id] = status
        last_reason[segment_id] = reason
        if covered_by_id:
            covered_by[segment_id] = covered_by_id

    max_iterations = max(2, len(anomaly_ids) * 3 + 3)
    for iteration in range(1, max_iterations + 1):
        if not active_ids:
            break

        child_map = _nearest_active_child_map(sorted(active_ids), feature_sets)
        parent_map = _nearest_parent_map(child_map)
        for segment_id in anomaly_ids:
            last_children[segment_id] = child_map.get(segment_id, last_children[segment_id])
            last_parents[segment_id] = parent_map.get(segment_id, [])

        current_depth = max(int(lookup[segment_id]["slice_depth"]) for segment_id in active_ids)
        deepest_ids = sorted(
            segment_id
            for segment_id in active_ids
            if int(lookup[segment_id]["slice_depth"]) == current_depth
        )
        actionable_parents = sorted(
            {
                parent_id
                for child_id in deepest_ids
                for parent_id in parent_map.get(child_id, [])
                if parent_id in active_ids
            }
        )

        if not actionable_parents:
            for segment_id in deepest_ids:
                active_ids.remove(segment_id)
                selected_ids.add(segment_id)
                if original_anomaly_parents.get(segment_id):
                    graph_status[segment_id] = "ORPHAN_AFTER_REWIRE"
                    status_reason[segment_id] = "segment lost active parents after absorption/removal decisions"
                    connection_break_flag[segment_id] = True
                    connection_break_reason[segment_id] = status_reason[segment_id]
                else:
                    graph_status[segment_id] = "INITIALLY_UNLINKED_TERMINAL"
                    status_reason[segment_id] = "segment initially had no anomalous parent in active graph"
                last_decision[segment_id] = graph_status[segment_id]
                last_reason[segment_id] = status_reason[segment_id]
                last_iteration[segment_id] = iteration
            continue

        changed = False
        for parent_id in actionable_parents:
            if parent_id not in active_ids:
                continue
            active_children = sorted(child_map.get(parent_id, []))
            if not active_children:
                continue

            child_parent_counts = {child_id: len(parent_map.get(child_id, [])) for child_id in active_children}
            if len(active_children) == 1 and child_parent_counts[active_children[0]] == 1:
                child_id = active_children[0]
                reason = "1:1 relationship stub; lower-level child is kept, parent is excluded"
                mark_removed(parent_id, "STUB_1_1_PARENT_SUPPRESSED", reason)
                decision_history[parent_id].append(f"iteration={iteration}: STUB_1_1 -> keep {child_id}")
                decision_log_rows.append(
                    {
                        "iteration": iteration,
                        "event_type": "RELATIONSHIP_STUB",
                        "parent_id": parent_id,
                        "parent_key": str(lookup[parent_id]["segment_key"]),
                        "parent_depth": int(lookup[parent_id]["slice_depth"]),
                        "child_id": child_id,
                        "child_key": str(lookup[child_id]["segment_key"]),
                        "child_depth": int(lookup[child_id]["slice_depth"]),
                        "active_child_count": 1,
                        "parent_abs_z": abs(_safe_float(lookup[parent_id].get("robust_z_capped"))),
                        "child_abs_z": abs(_safe_float(lookup[child_id].get("robust_z_capped"))),
                        "gross_share": math.nan,
                        "pair_decision": "STUB_1_1",
                        "conflict_code": "STUB_1_1",
                        "winner_parent_id": "",
                        "winner_parent_key": "",
                        "applied": True,
                        "reason": reason,
                    }
                )
                changed = True
                continue

            multi_parent_children = [child_id for child_id, count in child_parent_counts.items() if count > 1]
            if multi_parent_children:
                relationship = "STUB_N_N" if len(active_children) > 1 else "STUB_N_1"
                reason = f"{relationship} relationship stub; parent is excluded and child level is kept"
                mark_removed(parent_id, f"{relationship}_PARENT_SUPPRESSED", reason)
                decision_history[parent_id].append(f"iteration={iteration}: {relationship}")
                for child_id in active_children:
                    decision_log_rows.append(
                        {
                            "iteration": iteration,
                            "event_type": "RELATIONSHIP_STUB",
                            "parent_id": parent_id,
                            "parent_key": str(lookup[parent_id]["segment_key"]),
                            "parent_depth": int(lookup[parent_id]["slice_depth"]),
                            "child_id": child_id,
                            "child_key": str(lookup[child_id]["segment_key"]),
                            "child_depth": int(lookup[child_id]["slice_depth"]),
                            "active_child_count": len(active_children),
                            "parent_abs_z": abs(_safe_float(lookup[parent_id].get("robust_z_capped"))),
                            "child_abs_z": abs(_safe_float(lookup[child_id].get("robust_z_capped"))),
                            "gross_share": math.nan,
                            "pair_decision": relationship,
                            "conflict_code": relationship,
                            "winner_parent_id": "",
                            "winner_parent_key": "",
                            "applied": True,
                            "reason": reason,
                        }
                    )
                changed = True
                continue

            # FIXED: Новое правило 1:N работает с группами непересекающихся детей и приоритетом доминирующего ребёнка.
            sibling_groups = _split_sibling_groups(parent_id, active_children, feature_sets, coverage)
            group_decisions = [
                _evaluate_1n_group(
                    parent_id,
                    group_ids,
                    direct_all_children,
                    feature_sets,
                    lookup,
                    all_lookup,
                    thresholds,
                )
                for group_ids in sibling_groups
            ]
            dominant_decisions = [
                decision for decision in group_decisions if decision["decision"] == "CHILD_DOMINATES"
            ]

            if dominant_decisions:
                winner_decision = sorted(
                    dominant_decisions,
                    key=lambda decision: (
                        _safe_float(decision.get("gross_move")),
                        _safe_float(decision.get("child_abs_z")),
                        str(decision.get("dominant_child_id", "")),
                    ),
                    reverse=True,
                )[0]
                winner_child_id = str(winner_decision["dominant_child_id"])
                reason = (
                    "dominant child has priority over parent and sibling groups; "
                    f"{winner_decision['reason']}"
                )
                mark_removed(parent_id, "CHILD_DOMINATES_PARENT_REMOVED", reason, winner_child_id)
                for child_id in active_children:
                    if child_id == winner_child_id:
                        continue
                    mark_removed(
                        child_id,
                        "REMOVED_BY_DOMINANT_SIBLING",
                        f"removed because sibling {lookup[winner_child_id]['segment_key']} dominated parent",
                        winner_child_id,
                    )
                last_gross_share[parent_id] = _safe_float(winner_decision.get("gross_move"), math.nan)
                last_anomaly_gross_move[parent_id] = _safe_float(winner_decision.get("anomaly_gross_move"), math.nan)
                last_parent_z[parent_id] = _safe_float(winner_decision.get("parent_abs_z"), math.nan)
                last_child_z[parent_id] = _safe_float(winner_decision.get("child_abs_z"), math.nan)
                decision_history[parent_id].append(f"iteration={iteration}: CHILD_DOMINATES -> {winner_child_id}")
                for child_id in active_children:
                    decision_log_rows.append(
                        {
                            "iteration": iteration,
                            "event_type": "RULE_1_N",
                            "parent_id": parent_id,
                            "parent_key": str(lookup[parent_id]["segment_key"]),
                            "parent_depth": int(lookup[parent_id]["slice_depth"]),
                            "child_id": child_id,
                            "child_key": str(lookup[child_id]["segment_key"]),
                            "child_depth": int(lookup[child_id]["slice_depth"]),
                            "active_child_count": len(active_children),
                            "parent_abs_z": _safe_float(winner_decision.get("parent_abs_z"), math.nan),
                            "child_abs_z": abs(_safe_float(lookup[child_id].get("robust_z_capped"))),
                            "gross_share": (
                                abs(_safe_float(lookup[child_id].get("wow_delta_gmv"))) / _sum_abs_delta(active_children, lookup)
                                if _sum_abs_delta(active_children, lookup) != 0.0
                                else math.nan
                            ),
                            "pair_decision": "CHILD_DOMINATES" if child_id == winner_child_id else "REMOVED_BY_DOMINANT_SIBLING",
                            "conflict_code": "RULE_1_N_DOMINANT_CHILD_PRIORITY",
                            "winner_parent_id": "",
                            "winner_parent_key": "",
                            "applied": child_id == winner_child_id,
                            "reason": reason,
                        }
                    )
                changed = True
                continue

            reason = "parent absorbs active child groups; " + " | ".join(str(decision["reason"]) for decision in group_decisions)
            graph_status[parent_id] = "PARENT_ABSORBS_CHILDREN"
            status_reason[parent_id] = reason
            last_decision[parent_id] = "PARENT_ABSORBS_CHILDREN"
            last_reason[parent_id] = reason
            last_iteration[parent_id] = iteration
            last_anomaly_gross_move[parent_id] = max(
                (_safe_float(decision.get("anomaly_gross_move"), math.nan) for decision in group_decisions),
                default=math.nan,
            )
            last_parent_z[parent_id] = abs(_safe_float(lookup[parent_id].get("robust_z_capped")))
            parent_abs_z = abs(_safe_float(lookup[parent_id].get("robust_z_capped")))
            for child_id in active_children:
                active_ids.discard(child_id)
                removed_ids.add(child_id)
                covered_by[child_id] = parent_id
                graph_status[child_id] = "ABSORBED_BY_PARENT"
                status_reason[child_id] = f"absorbed by parent {lookup[parent_id]['segment_key']}"
                last_decision[child_id] = "ABSORBED_BY_PARENT"
                last_reason[child_id] = status_reason[child_id]
                last_iteration[child_id] = iteration
                absorbed_all[parent_id] = list(dict.fromkeys([*absorbed_all.get(parent_id, []), child_id]))
                if abs(_safe_float(lookup[child_id].get("robust_z_capped"))) > parent_abs_z:
                    absorbed_lineage[parent_id] = list(dict.fromkeys([*absorbed_lineage.get(parent_id, []), child_id]))
                selection_scores[parent_id] = max(selection_scores[parent_id], selection_scores.get(child_id, 0.0))
                decision_log_rows.append(
                    {
                        "iteration": iteration,
                        "event_type": "RULE_1_N",
                        "parent_id": parent_id,
                        "parent_key": str(lookup[parent_id]["segment_key"]),
                        "parent_depth": int(lookup[parent_id]["slice_depth"]),
                        "child_id": child_id,
                        "child_key": str(lookup[child_id]["segment_key"]),
                        "child_depth": int(lookup[child_id]["slice_depth"]),
                        "active_child_count": len(active_children),
                        "parent_abs_z": parent_abs_z,
                        "child_abs_z": abs(_safe_float(lookup[child_id].get("robust_z_capped"))),
                        "gross_share": (
                            abs(_safe_float(lookup[child_id].get("wow_delta_gmv"))) / _sum_abs_delta(active_children, lookup)
                            if _sum_abs_delta(active_children, lookup) != 0.0
                            else math.nan
                        ),
                        "pair_decision": "PARENT_ABSORBS_CHILDREN",
                        "conflict_code": "RULE_1_N_PARENT_ABSORBS",
                        "winner_parent_id": parent_id,
                        "winner_parent_key": str(lookup[parent_id]["segment_key"]),
                        "applied": True,
                        "reason": reason,
                    }
                )
            decision_history[parent_id].append(f"iteration={iteration}: PARENT_ABSORBS {len(active_children)} children")
            changed = True

        if not changed:
            raise RuntimeError("Active partition did not progress; relationship graph is unresolved")

    else:
        raise RuntimeError("Active partition did not converge within safe iteration limit")

    selected_ids.update(active_ids)
    final_graph_ids = set(selected_ids)
    final_child_map = _nearest_active_child_map(sorted(final_graph_ids), feature_sets) if final_graph_ids else {}
    final_parent_map = _nearest_parent_map(final_child_map)
    index_by_id = {
        str(segment_id): index
        for index, segment_id in diagnostics["segment_id"].items()
    }

    for segment_id in anomaly_ids:
        index = index_by_id[segment_id]
        children = final_child_map.get(segment_id, [])
        parents = final_parent_map.get(segment_id, [])
        original_atom_ids = sorted(coverage.get(segment_id, frozenset()))
        current_atom_ids = original_atom_ids if segment_id in selected_ids else []
        residual_atom_ids = current_atom_ids
        lineage = absorbed_lineage.get(segment_id, [])
        lineage_keys = [str(lookup[child_id]["segment_key"]) for child_id in lineage]
        lineage_labels = [
            _relative_child_segment_name(lookup[segment_id]["segment_key"], lookup[child_id]["segment_key"])
            for child_id in lineage
        ]
        covered_by_id = covered_by.get(segment_id, "")

        diagnostics.at[index, "active_child_ids"] = " || ".join(children)
        diagnostics.at[index, "active_child_keys"] = " || ".join(str(lookup[child_id]["segment_key"]) for child_id in children)
        diagnostics.at[index, "active_parent_ids"] = " || ".join(parents)
        diagnostics.at[index, "active_parent_keys"] = " || ".join(str(lookup[parent_id]["segment_key"]) for parent_id in parents)
        diagnostics.at[index, "technical_child_count"] = len(children)
        diagnostics.at[index, "active_child_count"] = len(children)
        diagnostics.at[index, "active_partition_action"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_partition_reason"] = last_reason[segment_id] or status_reason[segment_id]
        diagnostics.at[index, "active_partition_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_partition_status_reason"] = status_reason[segment_id]
        diagnostics.at[index, "active_graph_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_decision_code"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_decision_reason"] = last_reason[segment_id] or status_reason[segment_id]
        diagnostics.at[index, "active_decision_history"] = " || ".join(decision_history[segment_id])
        diagnostics.at[index, "active_decision_iteration"] = last_iteration[segment_id]
        diagnostics.at[index, "single_child_gross_share"] = last_gross_share[segment_id]
        diagnostics.at[index, "anomaly_gross_move"] = last_anomaly_gross_move[segment_id]
        diagnostics.at[index, "active_parent_z"] = last_parent_z[segment_id]
        diagnostics.at[index, "active_child_z"] = last_child_z[segment_id]
        diagnostics.at[index, "covered_by_segment"] = covered_by_id
        diagnostics.at[index, "covered_by_segment_key"] = str(lookup[covered_by_id]["segment_key"]) if covered_by_id else ""
        diagnostics.at[index, "absorbed_child_ids"] = " || ".join(lineage)
        diagnostics.at[index, "absorbed_child_keys"] = " || ".join(lineage_keys)
        diagnostics.at[index, "absorbed_child_labels"] = "; ".join(lineage_labels)
        diagnostics.at[index, "absorbed_by_parent_ids"] = covered_by_id
        diagnostics.at[index, "absorbed_by_parent_keys"] = str(lookup[covered_by_id]["segment_key"]) if covered_by_id else ""
        diagnostics.at[index, "original_atomic_descendants"] = " || ".join(original_atom_ids)
        diagnostics.at[index, "active_atomic_descendants"] = " || ".join(current_atom_ids)
        diagnostics.at[index, "residual_atomic_descendants"] = " || ".join(residual_atom_ids)
        diagnostics.at[index, "original_atomic_count"] = len(original_atom_ids)
        diagnostics.at[index, "active_atomic_count"] = len(current_atom_ids)
        diagnostics.at[index, "residual_atomic_count"] = len(residual_atom_ids)
        diagnostics.at[index, "residual_gross_movement"] = _sum_abs_delta(residual_atom_ids, all_lookup) if residual_atom_ids else 0.0
        diagnostics.at[index, "connection_break_flag"] = connection_break_flag[segment_id]
        diagnostics.at[index, "connection_break_reason"] = connection_break_reason[segment_id]
        diagnostics.at[index, "is_active"] = segment_id in selected_ids
        diagnostics.at[index, "is_resolved"] = segment_id in selected_ids
        diagnostics.at[index, "is_natural_terminal"] = segment_id in natural_terminals
        diagnostics.at[index, "is_terminal_current"] = segment_id in selected_ids and not children
        diagnostics.at[index, "selection_score"] = selection_scores[segment_id]

        if segment_id in selected_ids:
            if connection_break_flag[segment_id]:
                diagnostics.at[index, "action"] = "CONNECTION_BREAK_TERMINAL"
                diagnostics.at[index, "output_block"] = "аномалия с флагом обрыва связи"
                diagnostics.at[index, "reason"] = status_reason[segment_id]
            elif graph_status[segment_id] == "PARENT_ABSORBS_CHILDREN":
                diagnostics.at[index, "action"] = "PARENT_ABSORBS_CHILDREN"
                diagnostics.at[index, "output_block"] = "поглощение родителем"
                diagnostics.at[index, "reason"] = status_reason[segment_id]
            elif graph_status[segment_id] == "INITIALLY_UNLINKED_TERMINAL":
                diagnostics.at[index, "action"] = "INITIALLY_UNLINKED_TERMINAL"
                diagnostics.at[index, "output_block"] = "изначально несвязанный аномальный сегмент"
                diagnostics.at[index, "reason"] = status_reason[segment_id]
            elif segment_id in natural_terminals:
                diagnostics.at[index, "action"] = "NATURAL_TERMINAL"
                diagnostics.at[index, "output_block"] = "терминальная аномалия"
                diagnostics.at[index, "reason"] = "natural terminal segment of active graph"

    diagnostics["selected"] = diagnostics["segment_id"].astype(str).isin(selected_ids)
    for index, row in diagnostics.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id in selected_ids:
            diagnostics.at[index, "selection_exclusion_reason"] = ""
        elif segment_id in lookup:
            diagnostics.at[index, "selection_exclusion_reason"] = (
                f"не выбран active-разбиением: {graph_status[segment_id]} - {status_reason[segment_id]}"
            )
        else:
            diagnostics.at[index, "active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
            diagnostics.at[index, "selection_exclusion_reason"] = str(row.get("reason", ""))

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

    decision_log_columns = [
        "iteration",
        "event_type",
        "parent_id",
        "parent_key",
        "parent_depth",
        "child_id",
        "child_key",
        "child_depth",
        "active_child_count",
        "parent_abs_z",
        "child_abs_z",
        "gross_share",
        "pair_decision",
        "conflict_code",
        "winner_parent_id",
        "winner_parent_key",
        "applied",
        "reason",
    ]
    decision_log = pd.DataFrame(decision_log_rows, columns=decision_log_columns)
    if not decision_log.empty:
        decision_log = decision_log.sort_values(
            ["iteration", "event_type", "parent_key", "child_key"],
            kind="stable",
        ).reset_index(drop=True)
    return final_df, diagnostics, decision_log


def select_active_partition_anomalies(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    thresholds: AnomalyThresholds,
    dim_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Построить fixed-point active-граф с локальными терминальными ветками.

    Граф содержит только сегменты, прошедшие первичный anomaly-фильтр.
    Терминальный фронт определяется локально по отсутствию более глубокого
    аномального потомка. Все решения одной итерации вычисляются по снимку,
    разрешаются по ребёнку и применяются одним пакетом.

    Args:
        candidates: Классифицированные кандидаты со статистиками аномальности.
        coverage: Исходное атомарное покрытие каждого кандидата.
        thresholds: Порог gross_share и множитель сравнения Z.
        dim_cols: Измерения, задающие строгие связи предок — потомок.

    Returns:
        Кортеж из итоговых разрешённых аномалий, полной диагностики и журнала решений.

    Raises:
        ValueError: Если отсутствует физический атомарный слой или измерения.
        RuntimeError: Если монотонный fixed-point цикл не сошёлся.

    Examples:
        >>> # final_df, diagnostics, log = select_active_partition_anomalies(candidates, coverage, AnomalyThresholds(), dims)
    """

    # FIXED: Финальный active-граф теперь строится по алгоритму agm/gm/z_c_greater
    # после результата build_anomaly_candidates.
    return _select_active_partition_anomalies_v2(candidates, coverage, thresholds, dim_cols)

    diagnostics = candidates.copy()
    if diagnostics.empty:
        return diagnostics.copy(), diagnostics, pd.DataFrame()
    if not dim_cols:
        raise ValueError("Для active-графа нужен хотя бы один признак")

    max_physical_depth = int(diagnostics["slice_depth"].astype(int).max())
    atomic_df = diagnostics[
        diagnostics["slice_depth"].astype(int) == max_physical_depth
    ].copy()
    if atomic_df.empty:
        raise ValueError("Физический атомарный слой active-разбиения пуст")
    atomic_deltas = {
        str(row["segment_id"]): float(row["wow_delta_gmv"])
        for _, row in atomic_df.iterrows()
    }

    anomaly_mask = (
        diagnostics["slice_depth"].astype(int).gt(0)
        & diagnostics["passes_initial_anomaly_filter"].astype(bool)
    )
    anomaly_df = diagnostics[anomaly_mask].copy()

    string_columns = [
        "active_child_ids",
        "active_child_keys",
        "active_parent_ids",
        "active_parent_keys",
        "active_partition_action",
        "active_partition_reason",
        "active_partition_status",
        "active_partition_status_reason",
        "active_graph_status",
        "active_decision_code",
        "active_decision_reason",
        "active_decision_history",
        "conflict_parent_ids",
        "conflict_parent_keys",
        "covered_by_segment",
        "covered_by_segment_key",
        "absorbed_child_ids",
        "absorbed_child_keys",
        "absorbed_child_labels",
        "absorbed_by_parent_ids",
        "absorbed_by_parent_keys",
        "original_atomic_descendants",
        "active_atomic_descendants",
        "residual_atomic_descendants",
    ]
    for column in string_columns:
        diagnostics[column] = ""
    diagnostics["technical_child_count"] = 0
    diagnostics["active_child_count"] = 0
    diagnostics["original_atomic_count"] = 0
    diagnostics["active_atomic_count"] = 0
    diagnostics["residual_atomic_count"] = 0
    diagnostics["residual_gross_movement"] = 0.0
    diagnostics["single_child_gross_share"] = math.nan
    diagnostics["active_parent_z"] = math.nan
    diagnostics["active_child_z"] = math.nan
    diagnostics["active_decision_iteration"] = pd.Series(pd.NA, index=diagnostics.index, dtype="Int64")
    diagnostics["is_active"] = False
    diagnostics["is_resolved"] = False
    diagnostics["is_natural_terminal"] = False
    diagnostics["is_terminal_current"] = False
    diagnostics["covered_atomic_count"] = diagnostics["segment_id"].map(
        lambda segment_id: len(coverage.get(str(segment_id), frozenset()))
    )
    diagnostics["selected"] = False
    diagnostics["selection_exclusion_reason"] = ""

    if anomaly_df.empty:
        diagnostics["active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
        diagnostics["selection_exclusion_reason"] = diagnostics["reason"].astype(str)
        return diagnostics.iloc[0:0].copy(), diagnostics, pd.DataFrame()

    lookup = {
        str(row["segment_id"]): row.copy()
        for _, row in anomaly_df.iterrows()
    }
    anomaly_ids = sorted(lookup)
    feature_sets = {
        segment_id: _segment_feature_set(row, dim_cols)
        for segment_id, row in lookup.items()
    }
    original_atoms = {
        segment_id: frozenset(coverage.get(segment_id, frozenset()))
        for segment_id in anomaly_ids
    }
    coverage_is_known = {
        segment_id: bool(original_atoms[segment_id])
        and original_atoms[segment_id].issubset(atomic_deltas)
        for segment_id in anomaly_ids
    }
    original_descendants = {
        parent_id: {
            child_id
            for child_id in anomaly_ids
            if feature_sets[parent_id] < feature_sets[child_id]
        }
        for parent_id in anomaly_ids
    }
    original_parents = {
        child_id: {
            parent_id
            for parent_id in anomaly_ids
            if feature_sets[parent_id] < feature_sets[child_id]
        }
        for child_id in anomaly_ids
    }
    natural_terminals = {
        segment_id
        for segment_id in anomaly_ids
        if not original_descendants[segment_id]
    }

    active_ids = set(anomaly_ids)
    is_resolved = {
        segment_id: segment_id in natural_terminals
        for segment_id in anomaly_ids
    }
    graph_status = {
        segment_id: (
            "NATURAL_TERMINAL"
            if segment_id in natural_terminals
            else "ACTIVE_PENDING"
        )
        for segment_id in anomaly_ids
    }
    status_reason = {
        segment_id: (
            "изначально отсутствуют более глубокие аномальные потомки"
            if segment_id in natural_terminals
            else "ожидает разрешения нижней ветки"
        )
        for segment_id in anomaly_ids
    }
    active_atoms = {
        segment_id: set(original_atoms[segment_id])
        for segment_id in anomaly_ids
    }
    residual_atoms_for_diagnostics = {
        segment_id: set(original_atoms[segment_id])
        for segment_id in anomaly_ids
    }
    covered_by: Dict[str, str] = {}
    absorbed_lineage: Dict[str, List[str]] = {
        segment_id: [] for segment_id in anomaly_ids
    }
    decision_history: Dict[str, List[str]] = {
        segment_id: [] for segment_id in anomaly_ids
    }
    selection_scores = {
        segment_id: float(lookup[segment_id].get("selection_score", 0.0))
        for segment_id in anomaly_ids
    }
    last_children: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    last_decision: Dict[str, str] = {segment_id: "" for segment_id in anomaly_ids}
    last_reason: Dict[str, str] = dict(status_reason)
    last_iteration: Dict[str, Optional[int]] = {segment_id: None for segment_id in anomaly_ids}
    last_gross_share: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_parent_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_child_z: Dict[str, float] = {segment_id: math.nan for segment_id in anomaly_ids}
    last_conflict_parents: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    decision_log_rows: List[Dict[str, object]] = []

    def gross_movement(atom_ids: Sequence[str]) -> float:
        """Посчитать gross movement набора физических атомов.

        Args:
            atom_ids: Идентификаторы физических атомов.

        Returns:
            Сумма модулей изменений GMV.

        Raises:
            KeyError: Если атом отсутствует в физическом слое.

        Examples:
            >>> # gross_movement(['atom_1'])
        """

        return float(sum(abs(float(atomic_deltas[atom_id])) for atom_id in atom_ids))

    converged = False
    max_iterations = max(2, len(anomaly_ids) * 2 + 2)
    for iteration in range(1, max_iterations + 1):
        snapshot_active = set(active_ids)
        snapshot_resolved = dict(is_resolved)
        snapshot_status = dict(graph_status)
        snapshot_atoms = {
            segment_id: set(active_atoms[segment_id])
            for segment_id in anomaly_ids
        }
        nearest_children = _nearest_active_child_map(
            sorted(snapshot_active), feature_sets
        )
        last_children.update(nearest_children)

        next_status = dict(graph_status)
        next_reason = dict(status_reason)
        next_resolved = dict(is_resolved)
        empty_after_transfer: set[str] = set()
        proposals: List[Dict[str, object]] = []

        for parent_id in sorted(
            snapshot_active,
            key=lambda segment_id: (
                int(lookup[segment_id]["slice_depth"]),
                str(lookup[segment_id]["segment_key"]),
            ),
        ):
            children = nearest_children[parent_id]
            if not coverage_is_known[parent_id]:
                next_status[parent_id] = "DEFER_UNKNOWN_COVERAGE"
                next_reason[parent_id] = "исходное атомарное покрытие отсутствует или неполно"
                next_resolved[parent_id] = False
                last_decision[parent_id] = "DEFER_UNKNOWN_COVERAGE"
                last_reason[parent_id] = next_reason[parent_id]
                last_iteration[parent_id] = iteration
                continue

            if snapshot_status[parent_id] in {
                "DEFER_RESIDUAL_COVERAGE",
                "DEFER_COMPLEX_ATOMIC_CONFLICT",
                "DEFER_EQUAL_Z_CONFLICT",
            }:
                next_resolved[parent_id] = False
                continue

            if not children:
                if parent_id in natural_terminals:
                    next_status[parent_id] = "NATURAL_TERMINAL"
                    next_reason[parent_id] = "изначально отсутствуют более глубокие аномальные потомки"
                    next_resolved[parent_id] = True
                elif snapshot_resolved[parent_id]:
                    next_status[parent_id] = (
                        snapshot_status[parent_id]
                        if snapshot_status[parent_id] == "PARENT_WINS"
                        else "RESOLVED_TERMINAL"
                    )
                    next_reason[parent_id] = "нижняя ветка разрешена; сегмент доступен верхнему уровню"
                    next_resolved[parent_id] = True
                elif gross_movement(snapshot_atoms[parent_id]) == 0.0:
                    next_status[parent_id] = "EMPTY_AFTER_TRANSFER"
                    next_reason[parent_id] = "после передачи покрытия не осталось самостоятельного движения"
                    next_resolved[parent_id] = False
                    empty_after_transfer.add(parent_id)
                elif snapshot_atoms[parent_id] != set(original_atoms[parent_id]):
                    next_status[parent_id] = "DEFER_RESIDUAL_COVERAGE"
                    next_reason[parent_id] = "осталось частичное покрытие; исходный Z нельзя применять к остатку"
                    next_resolved[parent_id] = False
                else:
                    next_status[parent_id] = "DEFER_NO_ACTIVE_CHILD"
                    next_reason[parent_id] = "аномальные потомки существовали, но остаточное покрытие не разрешено"
                    next_resolved[parent_id] = False
                last_decision[parent_id] = next_status[parent_id]
                last_reason[parent_id] = next_reason[parent_id]
                last_iteration[parent_id] = iteration
                continue

            if len(children) > 1:
                next_status[parent_id] = "DEFER_MULTIPLE_CHILDREN"
                next_reason[parent_id] = "у родителя более одного ближайшего активного ребёнка"
                next_resolved[parent_id] = False
                last_decision[parent_id] = "DEFER_MULTIPLE_CHILDREN"
                last_reason[parent_id] = next_reason[parent_id]
                last_iteration[parent_id] = iteration
                decision_history[parent_id].append(
                    f"iteration={iteration}: DEFER_MULTIPLE_CHILDREN ({len(children)})"
                )
                decision_log_rows.append(
                    {
                        "iteration": iteration,
                        "event_type": "CLASSIFICATION",
                        "parent_id": parent_id,
                        "parent_key": str(lookup[parent_id]["segment_key"]),
                        "parent_depth": int(lookup[parent_id]["slice_depth"]),
                        "child_id": "",
                        "child_key": "",
                        "child_depth": pd.NA,
                        "active_child_count": len(children),
                        "parent_abs_z": abs(float(lookup[parent_id]["robust_z_capped"])),
                        "child_abs_z": math.nan,
                        "gross_share": math.nan,
                        "pair_decision": "DEFER_MULTIPLE_CHILDREN",
                        "conflict_code": "",
                        "winner_parent_id": "",
                        "winner_parent_key": "",
                        "applied": False,
                        "reason": next_reason[parent_id],
                    }
                )
                continue

            child_id = children[0]
            if not snapshot_resolved[child_id]:
                next_status[parent_id] = "BLOCKED_BY_UNRESOLVED_DESCENDANT"
                next_reason[parent_id] = "ближайший активный ребёнок ещё не разрешён"
                next_resolved[parent_id] = False
                last_decision[parent_id] = "BLOCKED_BY_UNRESOLVED_DESCENDANT"
                last_reason[parent_id] = next_reason[parent_id]
                last_iteration[parent_id] = iteration
                continue

            parent_abs_z = abs(float(lookup[parent_id]["robust_z_capped"]))
            child_abs_z = abs(float(lookup[child_id]["robust_z_capped"]))
            parent_residual_atoms = frozenset(snapshot_atoms[parent_id])
            if gross_movement(parent_residual_atoms) == 0.0:
                next_status[parent_id] = "DEFER_ZERO_GROSS"
                next_reason[parent_id] = "residual gross movement родителя равен нулю"
                next_resolved[parent_id] = False
                last_decision[parent_id] = "DEFER_ZERO_GROSS"
                last_reason[parent_id] = next_reason[parent_id]
                last_iteration[parent_id] = iteration
                continue
            gross_share = _single_child_gross_share(
                float(lookup[child_id]["wow_delta_gmv"]),
                parent_residual_atoms,
                atomic_deltas,
            )
            pair_decision = _single_child_decision(
                parent_abs_z,
                child_abs_z,
                gross_share,
                thresholds,
            )
            comparison = (
                f"gross_share={gross_share:.6f}; |Z_parent|={parent_abs_z:.6f}; "
                f"|Z_child|={child_abs_z:.6f}; пороги: gross_share > "
                f"{thresholds.single_child_gross_share_threshold:.2f} и |Z_parent| < "
                f"{thresholds.single_child_z_multiplier:.2f} × |Z_child|"
            )
            reason = f"{pair_decision}: {comparison}"
            proposals.append(
                {
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "parent_abs_z": parent_abs_z,
                    "child_abs_z": child_abs_z,
                    "gross_share": gross_share,
                    "pair_decision": pair_decision,
                    "reason": reason,
                }
            )
            next_status[parent_id] = "PAIR_DECISION_PENDING"
            next_reason[parent_id] = reason
            next_resolved[parent_id] = False
            last_decision[parent_id] = pair_decision
            last_reason[parent_id] = reason
            last_iteration[parent_id] = iteration
            last_gross_share[parent_id] = gross_share
            last_parent_z[parent_id] = parent_abs_z
            last_child_z[parent_id] = child_abs_z

        proposals_by_child: Dict[str, List[Dict[str, object]]] = {}
        for proposal in proposals:
            proposals_by_child.setdefault(str(proposal["child_id"]), []).append(proposal)

        to_remove: set[str] = set(empty_after_transfer)
        next_atoms = {
            segment_id: set(snapshot_atoms[segment_id])
            for segment_id in anomaly_ids
        }
        loss_atoms: Dict[str, set[str]] = {}
        winners_this_iteration: set[str] = set()
        structural_events = bool(empty_after_transfer)

        for child_id in sorted(proposals_by_child):
            child_proposals = proposals_by_child[child_id]
            conflict_parent_ids = sorted(
                str(proposal["parent_id"])
                for proposal in child_proposals
            )
            for parent_id in conflict_parent_ids:
                last_conflict_parents[parent_id] = conflict_parent_ids
            parent_win_proposals = [
                proposal
                for proposal in child_proposals
                if proposal["pair_decision"] == "PARENT_WINS"
            ]

            if not parent_win_proposals:
                for proposal in child_proposals:
                    parent_id = str(proposal["parent_id"])
                    to_remove.add(parent_id)
                    next_status[parent_id] = "CHILD_WINS_PARENT_REMOVED"
                    next_reason[parent_id] = "все конкурирующие родители получили CHILD_WINS"
                    next_resolved[parent_id] = False
                    residual_atoms_for_diagnostics[parent_id] = (
                        set(snapshot_atoms[parent_id]) - set(snapshot_atoms[child_id])
                    )
                    decision_history[parent_id].append(
                        f"iteration={iteration}: CHILD_WINS -> {child_id}"
                    )
                    decision_log_rows.append(
                        {
                            "iteration": iteration,
                            "event_type": "PAIR_DECISION",
                            "parent_id": parent_id,
                            "parent_key": str(lookup[parent_id]["segment_key"]),
                            "parent_depth": int(lookup[parent_id]["slice_depth"]),
                            "child_id": child_id,
                            "child_key": str(lookup[child_id]["segment_key"]),
                            "child_depth": int(lookup[child_id]["slice_depth"]),
                            "active_child_count": 1,
                            "parent_abs_z": float(proposal["parent_abs_z"]),
                            "child_abs_z": float(proposal["child_abs_z"]),
                            "gross_share": float(proposal["gross_share"]),
                            "pair_decision": "CHILD_WINS",
                            "conflict_code": "ALL_PARENTS_CHILD_WINS",
                            "winner_parent_id": "",
                            "winner_parent_key": "",
                            "applied": True,
                            "reason": next_reason[parent_id],
                        }
                    )
                structural_events = True
                continue

            max_parent_z = max(
                float(proposal["parent_abs_z"])
                for proposal in parent_win_proposals
            )
            strongest_parent_proposals = [
                proposal
                for proposal in parent_win_proposals
                if math.isclose(
                    float(proposal["parent_abs_z"]),
                    max_parent_z,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
            if len(strongest_parent_proposals) > 1:
                for proposal in child_proposals:
                    parent_id = str(proposal["parent_id"])
                    next_status[parent_id] = "DEFER_EQUAL_Z_CONFLICT"
                    next_reason[parent_id] = "несколько PARENT_WINS имеют одинаковый максимальный |Z|"
                    next_resolved[parent_id] = False
                    decision_history[parent_id].append(
                        f"iteration={iteration}: DEFER_EQUAL_Z_CONFLICT"
                    )
                    decision_log_rows.append(
                        {
                            "iteration": iteration,
                            "event_type": "PAIR_DECISION",
                            "parent_id": parent_id,
                            "parent_key": str(lookup[parent_id]["segment_key"]),
                            "parent_depth": int(lookup[parent_id]["slice_depth"]),
                            "child_id": child_id,
                            "child_key": str(lookup[child_id]["segment_key"]),
                            "child_depth": int(lookup[child_id]["slice_depth"]),
                            "active_child_count": 1,
                            "parent_abs_z": float(proposal["parent_abs_z"]),
                            "child_abs_z": float(proposal["child_abs_z"]),
                            "gross_share": float(proposal["gross_share"]),
                            "pair_decision": str(proposal["pair_decision"]),
                            "conflict_code": "DEFER_EQUAL_Z_CONFLICT",
                            "winner_parent_id": "",
                            "winner_parent_key": "",
                            "applied": False,
                            "reason": next_reason[parent_id],
                        }
                    )
                continue

            winner_proposal = strongest_parent_proposals[0]
            winner_id = str(winner_proposal["parent_id"])
            external_atomic_conflicts = sorted(
                other_id
                for other_id in snapshot_active
                if other_id not in {*conflict_parent_ids, child_id}
                and snapshot_resolved[other_id]
                and bool(snapshot_atoms[winner_id] & snapshot_atoms[other_id])
                and not (
                    feature_sets[winner_id] < feature_sets[other_id]
                    or feature_sets[other_id] < feature_sets[winner_id]
                )
            )
            if external_atomic_conflicts:
                conflict_keys = [
                    str(lookup[segment_id]["segment_key"])
                    for segment_id in external_atomic_conflicts
                ]
                for proposal in child_proposals:
                    parent_id = str(proposal["parent_id"])
                    next_status[parent_id] = "DEFER_COMPLEX_ATOMIC_CONFLICT"
                    next_reason[parent_id] = (
                        "потенциальный победитель пересекается по атомам с разрешённым "
                        f"несопоставимым сегментом: {'; '.join(conflict_keys)}"
                    )
                    next_resolved[parent_id] = False
                    last_conflict_parents[parent_id] = sorted(
                        {*conflict_parent_ids, *external_atomic_conflicts}
                    )
                    decision_history[parent_id].append(
                        f"iteration={iteration}: DEFER_COMPLEX_ATOMIC_CONFLICT"
                    )
                    decision_log_rows.append(
                        {
                            "iteration": iteration,
                            "event_type": "PAIR_DECISION",
                            "parent_id": parent_id,
                            "parent_key": str(lookup[parent_id]["segment_key"]),
                            "parent_depth": int(lookup[parent_id]["slice_depth"]),
                            "child_id": child_id,
                            "child_key": str(lookup[child_id]["segment_key"]),
                            "child_depth": int(lookup[child_id]["slice_depth"]),
                            "active_child_count": 1,
                            "parent_abs_z": float(proposal["parent_abs_z"]),
                            "child_abs_z": float(proposal["child_abs_z"]),
                            "gross_share": float(proposal["gross_share"]),
                            "pair_decision": str(proposal["pair_decision"]),
                            "conflict_code": "DEFER_COMPLEX_ATOMIC_CONFLICT",
                            "winner_parent_id": "",
                            "winner_parent_key": "",
                            "applied": False,
                            "reason": next_reason[parent_id],
                        }
                    )
                continue
            winners_this_iteration.add(winner_id)
            to_remove.add(child_id)
            next_status[winner_id] = "PARENT_WINS"
            next_reason[winner_id] = f"родитель поглотил единственного ребёнка {lookup[child_id]['segment_key']}"
            next_resolved[winner_id] = True
            next_status[child_id] = "ABSORBED_BY_PARENT"
            next_reason[child_id] = f"поглощён родителем {lookup[winner_id]['segment_key']}"
            next_resolved[child_id] = False
            selection_scores[winner_id] = max(
                selection_scores[winner_id],
                selection_scores[child_id],
            )
            lineage = [child_id, *absorbed_lineage.get(child_id, [])]
            absorbed_lineage[winner_id] = list(
                dict.fromkeys([*absorbed_lineage.get(winner_id, []), *lineage])
            )
            for absorbed_id in lineage:
                covered_by[absorbed_id] = winner_id
                residual_atoms_for_diagnostics[absorbed_id] = set()
            decision_history[winner_id].append(
                f"iteration={iteration}: PARENT_WINS -> {child_id}"
            )

            transferred_atoms = set(snapshot_atoms[child_id])
            for other_id in snapshot_active:
                if other_id in {winner_id, child_id}:
                    continue
                if not feature_sets[other_id] < feature_sets[child_id]:
                    continue
                winner_is_descendant = feature_sets[other_id] < feature_sets[winner_id]
                if not winner_is_descendant:
                    loss_atoms.setdefault(other_id, set()).update(transferred_atoms)

            for proposal in child_proposals:
                parent_id = str(proposal["parent_id"])
                is_winner = parent_id == winner_id
                decision_log_rows.append(
                    {
                        "iteration": iteration,
                        "event_type": "PAIR_DECISION",
                        "parent_id": parent_id,
                        "parent_key": str(lookup[parent_id]["segment_key"]),
                        "parent_depth": int(lookup[parent_id]["slice_depth"]),
                        "child_id": child_id,
                        "child_key": str(lookup[child_id]["segment_key"]),
                        "child_depth": int(lookup[child_id]["slice_depth"]),
                        "active_child_count": 1,
                        "parent_abs_z": float(proposal["parent_abs_z"]),
                        "child_abs_z": float(proposal["child_abs_z"]),
                        "gross_share": float(proposal["gross_share"]),
                        "pair_decision": str(proposal["pair_decision"]),
                        "conflict_code": "PARENT_WINS_PRIORITY",
                        "winner_parent_id": winner_id,
                        "winner_parent_key": str(lookup[winner_id]["segment_key"]),
                        "applied": is_winner,
                        "reason": (
                            next_reason[winner_id]
                            if is_winner
                            else "уступил родителю с максимальным |Z| или приоритету PARENT_WINS"
                        ),
                    }
                )
            structural_events = True

        for segment_id, atoms_to_remove in loss_atoms.items():
            if segment_id in to_remove or segment_id in winners_this_iteration:
                continue
            residual_atoms = set(snapshot_atoms[segment_id]) - atoms_to_remove
            next_atoms[segment_id] = residual_atoms
            residual_atoms_for_diagnostics[segment_id] = set(residual_atoms)
            if not residual_atoms or gross_movement(residual_atoms) == 0.0:
                to_remove.add(segment_id)
                next_status[segment_id] = "EMPTY_AFTER_TRANSFER"
                next_reason[segment_id] = "всё самостоятельное покрытие передано внешнему победителю"
                next_resolved[segment_id] = False
            else:
                next_status[segment_id] = "DEFER_RESIDUAL_COVERAGE"
                next_reason[segment_id] = "частичный остаток требует отдельного пересчёта Z"
                next_resolved[segment_id] = False
            structural_events = True

        for segment_id in empty_after_transfer:
            residual_atoms_for_diagnostics[segment_id] = set()
        for segment_id in to_remove:
            next_atoms[segment_id] = set()
            next_resolved[segment_id] = False

        active_ids = snapshot_active - to_remove
        active_atoms = next_atoms
        graph_status = next_status
        status_reason = next_reason
        is_resolved = next_resolved

        state_changed = (
            active_ids != snapshot_active
            or any(
                active_atoms[segment_id] != snapshot_atoms[segment_id]
                for segment_id in anomaly_ids
            )
            or any(
                is_resolved[segment_id] != snapshot_resolved[segment_id]
                for segment_id in anomaly_ids
            )
        )
        if not state_changed and not structural_events:
            converged = True
            break
        if not state_changed:
            converged = True
            break

    if not converged:
        raise RuntimeError("Fixed-point active-граф не сошёлся за безопасное число итераций")

    final_nearest_children = _nearest_active_child_map(
        sorted(active_ids), feature_sets
    )
    final_parent_map: Dict[str, List[str]] = {segment_id: [] for segment_id in anomaly_ids}
    for parent_id, children in final_nearest_children.items():
        for child_id in children:
            final_parent_map[child_id].append(parent_id)

    index_by_id = {
        str(segment_id): index
        for index, segment_id in diagnostics["segment_id"].items()
    }
    for segment_id in anomaly_ids:
        index = index_by_id[segment_id]
        children = final_nearest_children.get(segment_id, [])
        parents = sorted(final_parent_map.get(segment_id, []))
        original_atom_ids = sorted(original_atoms[segment_id])
        current_atom_ids = sorted(active_atoms[segment_id]) if segment_id in active_ids else []
        residual_atom_ids = sorted(residual_atoms_for_diagnostics[segment_id])
        lineage = absorbed_lineage.get(segment_id, [])
        lineage_keys = [str(lookup[child_id]["segment_key"]) for child_id in lineage]
        lineage_labels = [
            _relative_child_segment_name(
                lookup[segment_id]["segment_key"],
                lookup[child_id]["segment_key"],
            )
            for child_id in lineage
        ]
        covered_by_id = covered_by.get(segment_id, "")
        diagnostics.at[index, "active_child_ids"] = " || ".join(children)
        diagnostics.at[index, "active_child_keys"] = " || ".join(
            str(lookup[child_id]["segment_key"]) for child_id in children
        )
        diagnostics.at[index, "active_parent_ids"] = " || ".join(parents)
        diagnostics.at[index, "active_parent_keys"] = " || ".join(
            str(lookup[parent_id]["segment_key"]) for parent_id in parents
        )
        diagnostics.at[index, "technical_child_count"] = len(children)
        diagnostics.at[index, "active_child_count"] = len(children)
        diagnostics.at[index, "active_partition_action"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_partition_reason"] = last_reason[segment_id]
        diagnostics.at[index, "active_partition_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_partition_status_reason"] = status_reason[segment_id]
        diagnostics.at[index, "active_graph_status"] = graph_status[segment_id]
        diagnostics.at[index, "active_decision_code"] = last_decision[segment_id] or graph_status[segment_id]
        diagnostics.at[index, "active_decision_reason"] = last_reason[segment_id]
        diagnostics.at[index, "active_decision_history"] = " || ".join(decision_history[segment_id])
        diagnostics.at[index, "active_decision_iteration"] = last_iteration[segment_id]
        diagnostics.at[index, "single_child_gross_share"] = last_gross_share[segment_id]
        diagnostics.at[index, "active_parent_z"] = last_parent_z[segment_id]
        diagnostics.at[index, "active_child_z"] = last_child_z[segment_id]
        diagnostics.at[index, "conflict_parent_ids"] = " || ".join(last_conflict_parents[segment_id])
        diagnostics.at[index, "conflict_parent_keys"] = " || ".join(
            str(lookup[parent_id]["segment_key"])
            for parent_id in last_conflict_parents[segment_id]
        )
        diagnostics.at[index, "covered_by_segment"] = covered_by_id
        diagnostics.at[index, "covered_by_segment_key"] = (
            str(lookup[covered_by_id]["segment_key"]) if covered_by_id else ""
        )
        diagnostics.at[index, "absorbed_child_ids"] = " || ".join(lineage)
        diagnostics.at[index, "absorbed_child_keys"] = " || ".join(lineage_keys)
        diagnostics.at[index, "absorbed_child_labels"] = "; ".join(lineage_labels)
        diagnostics.at[index, "absorbed_by_parent_ids"] = covered_by_id
        diagnostics.at[index, "absorbed_by_parent_keys"] = (
            str(lookup[covered_by_id]["segment_key"]) if covered_by_id else ""
        )
        diagnostics.at[index, "original_atomic_descendants"] = " || ".join(original_atom_ids)
        diagnostics.at[index, "active_atomic_descendants"] = " || ".join(current_atom_ids)
        diagnostics.at[index, "residual_atomic_descendants"] = " || ".join(residual_atom_ids)
        diagnostics.at[index, "original_atomic_count"] = len(original_atom_ids)
        diagnostics.at[index, "active_atomic_count"] = len(current_atom_ids)
        diagnostics.at[index, "residual_atomic_count"] = len(residual_atom_ids)
        diagnostics.at[index, "residual_gross_movement"] = gross_movement(residual_atom_ids)
        diagnostics.at[index, "is_active"] = segment_id in active_ids
        diagnostics.at[index, "is_resolved"] = bool(is_resolved[segment_id])
        diagnostics.at[index, "is_natural_terminal"] = segment_id in natural_terminals
        diagnostics.at[index, "is_terminal_current"] = segment_id in active_ids and not children
        diagnostics.at[index, "selection_score"] = selection_scores[segment_id]

        if segment_id in natural_terminals and segment_id in active_ids:
            diagnostics.at[index, "action"] = "NATURAL_TERMINAL"
            diagnostics.at[index, "output_block"] = "терминальная аномалия"
            diagnostics.at[index, "reason"] = "естественный терминальный сегмент локальной ветки"
        elif graph_status[segment_id] == "PARENT_WINS" and segment_id in active_ids:
            diagnostics.at[index, "action"] = "PARENT_WINS"
            diagnostics.at[index, "output_block"] = "поглощение родителем"
            diagnostics.at[index, "reason"] = status_reason[segment_id]

    selected_ids = {
        segment_id
        for segment_id in active_ids
        if bool(is_resolved[segment_id])
    }
    diagnostics["selected"] = diagnostics["segment_id"].astype(str).isin(selected_ids)
    for index, row in diagnostics.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id in selected_ids:
            diagnostics.at[index, "selection_exclusion_reason"] = ""
        elif segment_id in lookup:
            diagnostics.at[index, "selection_exclusion_reason"] = (
                f"не выбран active-графом: {graph_status[segment_id]} — {status_reason[segment_id]}"
            )
        else:
            diagnostics.at[index, "active_graph_status"] = "NOT_IN_ANOMALY_GRAPH"
            diagnostics.at[index, "selection_exclusion_reason"] = str(row.get("reason", ""))

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

    decision_log_columns = [
        "iteration",
        "event_type",
        "parent_id",
        "parent_key",
        "parent_depth",
        "child_id",
        "child_key",
        "child_depth",
        "active_child_count",
        "parent_abs_z",
        "child_abs_z",
        "gross_share",
        "pair_decision",
        "conflict_code",
        "winner_parent_id",
        "winner_parent_key",
        "applied",
        "reason",
    ]
    decision_log = pd.DataFrame(decision_log_rows, columns=decision_log_columns)
    if not decision_log.empty:
        decision_log = decision_log.sort_values(
            ["iteration", "event_type", "parent_key", "child_key"],
            kind="stable",
        ).reset_index(drop=True)
    return final_df, diagnostics, decision_log


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

    def metric_pct_output(row: Optional[pd.Series] = None) -> Dict[str, object]:
        """Вернуть отформатированные WoW-проценты для менеджерской строки.

        Args:
            row: Строка выбранного сегмента или None для служебных строк.

        Returns:
            Словарь с колонками процентов по GMV и операционным метрикам.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> metric_pct_output(None)["GMV WoW %"]
            ''
        """

        if row is None:
            return {display_col: "" for display_col in MANAGER_METRIC_PCT_COLUMNS.values()}
        return {
            display_col: _format_pct(row.get(source_col))
            for source_col, display_col in MANAGER_METRIC_PCT_COLUMNS.items()
        }

    rows: List[Dict[str, object]] = [
        {
            "раздел": "Заголовок",
            "тип": "",
            "сегмент": "Менеджерский вывод по необычным сегментам GMV",
            "Delta GMV": "",
            **metric_pct_output(None),
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
                "Delta GMV": "",
                **metric_pct_output(None),
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
            "сегмент": _manager_segment_name(main),
            "Delta GMV": _format_rub(float(main["wow_delta_gmv"])),
            **metric_pct_output(main),
            "z_score": round(float(main["robust_z"]), 2),
            "интерпретация": "Самый сильный выбранный блок по score; для компенсационных блоков score определяется внутренним движением детей.",
        }
    )

    for _, row in top.iterrows():
        direction = "выше" if float(row["abnormal_gmv"]) > 0 else "ниже"
        absorbed_labels = str(row.get("absorbed_child_labels", "")).strip()
        absorption_text = (
            f" Поглощён дочерний сегмент: {absorbed_labels}."
            if absorbed_labels and absorbed_labels.lower() != "nan"
            else ""
        )
        if str(row["output_block"]) == "блок аномальной компенсации":
            interpretation = (
                "Net-эффект родителя может быть небольшим, но внутри детей есть крупные встречные аномальные движения. "
                f"Причина отбора: {row['reason']}.{absorption_text}"
            )
        else:
            interpretation = (
                f"Фактический GMV сегмента {direction} ожидаемого уровня. "
                f"Причина отбора: {row['reason']}.{absorption_text}"
            )
        rows.append(
            {
                "раздел": "Таблица факторов",
                "тип": str(row["output_block"]),
                "сегмент": _manager_segment_name(row),
                "Delta GMV": _format_rub(float(row["wow_delta_gmv"])),
                **metric_pct_output(row),
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
    if not final_df.empty and "active_atomic_descendants" in final_df.columns:
        for value in final_df["active_atomic_descendants"].fillna("").astype(str):
            selected_atoms.extend(
                atom_id.strip()
                for atom_id in value.split(" || ")
                if atom_id.strip()
            )
    else:
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
        (
            "active_unresolved_count",
            int((candidates.get("is_active", False) & ~candidates.get("is_resolved", False)).sum())
            if "is_active" in candidates and "is_resolved" in candidates
            else 0,
        ),
        (
            "natural_terminal_count",
            int(candidates.get("is_natural_terminal", pd.Series(False, index=candidates.index)).astype(bool).sum()),
        ),
        (
            "defer_multiple_children_count",
            int(candidates.get("active_graph_status", pd.Series("", index=candidates.index)).eq("DEFER_MULTIPLE_CHILDREN").sum()),
        ),
        (
            "defer_equal_z_conflict_count",
            int(candidates.get("active_graph_status", pd.Series("", index=candidates.index)).eq("DEFER_EQUAL_Z_CONFLICT").sum()),
        ),
        (
            "empty_after_transfer_count",
            int(candidates.get("active_graph_status", pd.Series("", index=candidates.index)).eq("EMPTY_AFTER_TRANSFER").sum()),
        ),
        ("atomic_count", atomic_count),
        ("selected_atomic_unique_count", selected_atom_unique_count),
        ("double_count_violation_count", double_count_violation_count),
        ("filled_missing_rows", int(panel_df["row_missing_in_source"].sum())),
    ]
    return pd.DataFrame(rows, columns=["показатель", "значение"])


def build_anomaly_analysis_sheet(candidates: pd.DataFrame, final_df: pd.DataFrame, thresholds: AnomalyThresholds) -> pd.DataFrame:
    """Сформировать лист анализа аномалий, прошедших первичный фильтр.

    Args:
        candidates: Диагностика кандидатов после классификации и отбора.
        final_df: Итоговые выбранные аномалии.
        thresholds: Пороги алгоритма, включая лимит фактов менеджерского вывода.

    Returns:
        Таблица аномалий, прошедших первичный фильтр, с причиной отсутствия в менеджерском выводе.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # analysis_df = build_anomaly_analysis_sheet(candidates, final_df, AnomalyThresholds())
    """

    columns = [
        "сегмент",
        "глубина",
        "z_scope",
        "anomaly_score",
        "решение active-разбиения",
        "статус решения active-разбиения",
        "активен",
        "разрешён",
        "естественный терминальный",
        "число ближайших активных детей",
        "gross_share единственного ребёнка",
        "residual gross movement",
        "тип связи",
        "поглощён кем",
        "поглощён сегментом",
        "причина поглощения",
        "покрыт сегментом",
        "поглощённые сегменты",
        "номер добавления в менеджерский вывод",
        "причина не попадания в менеджерский вывод",
        "Delta GMV",
    ]
    if candidates.empty or "passes_initial_anomaly_filter" not in candidates.columns:
        return pd.DataFrame(columns=columns)

    initial_anomaly_mask = (
        candidates["passes_initial_anomaly_filter"].eq(True)
        & candidates["abs_abnormal_gmv"].astype(float).ge(thresholds.min_anomaly_abs)
    )
    active_decision_mask = (
        candidates["active_partition_action"].astype(str).ne("")
        if "active_partition_action" in candidates.columns
        else False
    )
    analysis_mask = initial_anomaly_mask | active_decision_mask
    analysis = candidates[analysis_mask].copy()
    if analysis.empty:
        return pd.DataFrame(columns=columns)

    selected_ids = set(final_df["segment_id"].astype(str).tolist()) if not final_df.empty else set()
    manager_ids = (
        set(final_df.head(thresholds.max_manager_facts)["segment_id"].astype(str).tolist())
        if not final_df.empty and "segment_id" in final_df.columns
        else set()
    )
    manager_order_by_id = (
        {
            str(segment_id): order
            for order, segment_id in enumerate(
                final_df.head(thresholds.max_manager_facts)["segment_id"].astype(str).tolist(),
                start=1,
            )
        }
        if not final_df.empty and "segment_id" in final_df.columns
        else {}
    )

    def manager_exclusion_reason(row: pd.Series) -> str:
        """Вернуть причину отсутствия кандидата в менеджерском выводе.

        Args:
            row: Строка кандидата из диагностики.

        Returns:
            Текст причины для листа анализа аномалий.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> manager_exclusion_reason(pd.Series({'segment_id': 'x'}))
            'не выбран в итоговый набор аномалий'
        """

        segment_id = str(row.get("segment_id", ""))
        if segment_id in manager_ids:
            return "попал в итоговый менеджерский вывод"
        if segment_id in selected_ids:
            return f"выбран в итоговый набор, но не попал в топ-{thresholds.max_manager_facts} менеджерского вывода"
        selection_reason = str(row.get("selection_exclusion_reason", "")).strip()
        if selection_reason:
            return selection_reason
        classifier_reason = str(row.get("reason", "")).strip()
        return classifier_reason if classifier_reason else "не выбран в итоговый набор аномалий"

    analysis["причина не попадания в менеджерский вывод"] = analysis.apply(manager_exclusion_reason, axis=1)
    export = pd.DataFrame(
        {
            "сегмент": analysis["segment_key"].astype(str),
            "глубина": analysis["slice_depth"].astype(int),
            "z_scope": analysis["robust_z_capped"].astype(float).round(2),
            "anomaly_score": analysis["anomaly_score"].astype(float),
            "решение active-разбиения": analysis.get(
                "active_partition_action", pd.Series("", index=analysis.index)
            ).astype(str),
            "статус решения active-разбиения": analysis.get(
                "active_partition_status", pd.Series("", index=analysis.index)
            ).astype(str),
            "активен": analysis.get(
                "is_active", pd.Series(False, index=analysis.index)
            ).astype(bool),
            "разрешён": analysis.get(
                "is_resolved", pd.Series(False, index=analysis.index)
            ).astype(bool),
            "естественный терминальный": analysis.get(
                "is_natural_terminal", pd.Series(False, index=analysis.index)
            ).astype(bool),
            "число ближайших активных детей": pd.to_numeric(
                analysis.get("active_child_count", pd.Series(0, index=analysis.index)),
                errors="coerce",
            ).fillna(0).astype(int),
            "gross_share единственного ребёнка": pd.to_numeric(
                analysis.get("single_child_gross_share", pd.Series(math.nan, index=analysis.index)),
                errors="coerce",
            ),
            "residual gross movement": pd.to_numeric(
                analysis.get("residual_gross_movement", pd.Series(0.0, index=analysis.index)),
                errors="coerce",
            ),
            "тип связи": analysis.get(
                "active_relationship_type", pd.Series("", index=analysis.index)
            ).astype(str),
            "поглощён кем": analysis.get(
                "absorbed_by_role", pd.Series("", index=analysis.index)
            ).astype(str),
            "поглощён сегментом": analysis.get(
                "absorbed_by_segment_key", pd.Series("", index=analysis.index)
            ).astype(str),
            "причина поглощения": analysis.get(
                "absorption_reason", pd.Series("", index=analysis.index)
            ).astype(str),
            "покрыт сегментом": analysis.get(
                "covered_by_segment_key", pd.Series("", index=analysis.index)
            ).astype(str),
            "поглощённые сегменты": analysis.get(
                "absorbed_child_labels", pd.Series("", index=analysis.index)
            ).astype(str),
            "номер добавления в менеджерский вывод": analysis["segment_id"].astype(str).map(manager_order_by_id).astype("Int64"),
            "причина не попадания в менеджерский вывод": analysis["причина не попадания в менеджерский вывод"],
            "Delta GMV": analysis["wow_delta_gmv"].astype(float).round(0).astype("Int64"),
        }
    )
    return export.sort_values(
        by=["глубина", "anomaly_score", "Delta GMV", "сегмент"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def highlight_manager_rows_on_anomaly_analysis(worksheet) -> None:
    """Выделить зелёным строки листа анализа аномалий, попавшие в менеджерский вывод.

    Args:
        worksheet: Лист openpyxl `Анализ аномалий`.

    Returns:
        None.

    Raises:
        AttributeError: Если объект не похож на лист openpyxl.

    Examples:
        >>> # highlight_manager_rows_on_anomaly_analysis(worksheet)
    """

    if worksheet.max_row < 2:
        return

    target_header = "причина не попадания в менеджерский вывод"
    target_value = "попал в итоговый менеджерский вывод"
    green_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")
    header_to_col = {
        str(cell.value): cell.column
        for cell in worksheet[1]
        if cell.value is not None
    }
    reason_col = header_to_col.get(target_header)
    if reason_col is None:
        return

    for row_idx in range(2, worksheet.max_row + 1):
        if worksheet.cell(row=row_idx, column=reason_col).value != target_value:
            continue
        for col_idx in range(1, worksheet.max_column + 1):
            worksheet.cell(row=row_idx, column=col_idx).fill = green_fill


def _order_tree_layers_by_shared_children(
    node_ids_by_depth: Dict[int, List[str]],
    edges: Sequence[Tuple[str, str]],
    records: Dict[str, Dict[str, object]],
    sweep_count: int = 5,
) -> Dict[int, List[str]]:
    """Упорядочить уровни дерева с группировкой родителей общих детей.

    Сначала выполняется top-down barycenter-проход для сближения детей с
    родителями. Затем bottom-up проход объединяет в непрерывные компоненты
    родителей, связанных хотя бы с одним общим ребёнком. Последний проход
    всегда bottom-up, поэтому требование близости общих родителей имеет
    приоритет над косметическим выравниванием детей.

    Args:
        node_ids_by_depth: Узлы, сгруппированные по глубине.
        edges: Рёбра ``(parent_id, child_id)`` соседних глубин.
        records: Атрибуты узлов, включая человекочитаемый сегмент.
        sweep_count: Число двунаправленных проходов минимизации пересечений.

    Returns:
        Упорядоченные идентификаторы узлов для каждой глубины.

    Raises:
        ValueError: Если sweep_count меньше единицы.

    Examples:
        >>> records = {'p1': {'сегмент': 'p1'}, 'p2': {'сегмент': 'p2'}, 'c': {'сегмент': 'c'}}
        >>> _order_tree_layers_by_shared_children({1: ['p2', 'p1'], 2: ['c']}, [('p1', 'c'), ('p2', 'c')], records)[1]
        ['p1', 'p2']
    """

    if sweep_count < 1:
        raise ValueError("sweep_count должен быть не меньше единицы")

    depth_values = sorted(node_ids_by_depth)
    ordered = {
        depth: sorted(
            node_ids,
            key=lambda node_id: str(records[node_id]["сегмент"]),
        )
        for depth, node_ids in node_ids_by_depth.items()
    }
    children_by_parent: Dict[str, List[str]] = {
        node_id: []
        for node_ids in node_ids_by_depth.values()
        for node_id in node_ids
    }
    parents_by_child: Dict[str, List[str]] = {
        node_id: []
        for node_ids in node_ids_by_depth.values()
        for node_id in node_ids
    }
    for parent_id, child_id in edges:
        children_by_parent[parent_id].append(child_id)
        parents_by_child[child_id].append(parent_id)

    def top_down_sweep() -> None:
        """Сблизить детей со средними позициями их родителей.

        Args:
            Аргументы отсутствуют.

        Returns:
            None.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> # top_down_sweep()
        """

        for depth in depth_values[1:]:
            parent_depth = int(depth) - 1
            if parent_depth not in ordered:
                continue
            parent_positions = {
                node_id: position
                for position, node_id in enumerate(ordered[parent_depth])
            }
            sort_keys: Dict[str, Tuple[float, Tuple[int, ...], str]] = {}
            for node_id in ordered[int(depth)]:
                positions = sorted(
                    parent_positions[parent_id]
                    for parent_id in parents_by_child[node_id]
                    if parent_id in parent_positions
                )
                barycenter = (
                    sum(positions) / len(positions)
                    if positions
                    else float("inf")
                )
                sort_keys[node_id] = (
                    barycenter,
                    tuple(positions),
                    str(records[node_id]["сегмент"]),
                )
            ordered[int(depth)] = sorted(
                ordered[int(depth)],
                key=lambda node_id: sort_keys[node_id],
            )

    def bottom_up_shared_child_sweep() -> None:
        """Сделать компоненты родителей общих детей непрерывными.

        Args:
            Аргументы отсутствуют.

        Returns:
            None.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> # bottom_up_shared_child_sweep()
        """

        for depth in reversed(depth_values[:-1]):
            child_depth = int(depth) + 1
            if child_depth not in ordered:
                continue
            child_positions = {
                node_id: position
                for position, node_id in enumerate(ordered[child_depth])
            }
            parent_ids = list(ordered[int(depth)])
            parent_id_set = set(parent_ids)
            shared_child_neighbors = {
                parent_id: set()
                for parent_id in parent_ids
            }
            for child_id in ordered[child_depth]:
                linked_parents = [
                    parent_id
                    for parent_id in parents_by_child[child_id]
                    if parent_id in parent_id_set
                ]
                for parent_id in linked_parents:
                    shared_child_neighbors[parent_id].update(
                        other_parent_id
                        for other_parent_id in linked_parents
                        if other_parent_id != parent_id
                    )

            components: List[List[str]] = []
            remaining = set(parent_ids)
            for seed_id in parent_ids:
                if seed_id not in remaining:
                    continue
                stack = [seed_id]
                component: List[str] = []
                remaining.remove(seed_id)
                while stack:
                    parent_id = stack.pop()
                    component.append(parent_id)
                    for neighbor_id in sorted(shared_child_neighbors[parent_id]):
                        if neighbor_id in remaining:
                            remaining.remove(neighbor_id)
                            stack.append(neighbor_id)
                components.append(component)

            parent_sort_keys: Dict[str, Tuple[float, Tuple[int, ...], str]] = {}
            for parent_id in parent_ids:
                positions = sorted(
                    child_positions[child_id]
                    for child_id in children_by_parent[parent_id]
                    if child_id in child_positions
                )
                barycenter = (
                    sum(positions) / len(positions)
                    if positions
                    else float("inf")
                )
                parent_sort_keys[parent_id] = (
                    barycenter,
                    tuple(positions),
                    str(records[parent_id]["сегмент"]),
                )

            component_rows: List[Tuple[Tuple[float, float, str], List[str]]] = []
            for component in components:
                component_order = sorted(
                    component,
                    key=lambda node_id: parent_sort_keys[node_id],
                )
                finite_barycenters = [
                    parent_sort_keys[node_id][0]
                    for node_id in component_order
                    if not math.isinf(parent_sort_keys[node_id][0])
                ]
                component_key = (
                    min(finite_barycenters) if finite_barycenters else float("inf"),
                    (
                        sum(finite_barycenters) / len(finite_barycenters)
                        if finite_barycenters
                        else float("inf")
                    ),
                    min(str(records[node_id]["сегмент"]) for node_id in component_order),
                )
                component_rows.append((component_key, component_order))

            component_rows.sort(key=lambda row: row[0])
            ordered[int(depth)] = [
                node_id
                for _, component in component_rows
                for node_id in component
            ]

    def refine_shared_child_adjacency() -> None:
        """Локально минимизировать разрывы между родителями общего ребёнка.

        Перестановки оцениваются лексикографически: сначала уменьшается число
        разорванных родительских групп, затем суммарный и максимальный разрыв.

        Args:
            Аргументы отсутствуют.

        Returns:
            None.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> # refine_shared_child_adjacency()
        """

        for depth in depth_values[:-1]:
            child_depth = int(depth) + 1
            if child_depth not in ordered:
                continue
            parent_groups = [
                [
                    parent_id
                    for parent_id in parents_by_child[child_id]
                    if parent_id in ordered[int(depth)]
                ]
                for child_id in ordered[child_depth]
            ]
            parent_groups = [
                parent_ids
                for parent_ids in parent_groups
                if len(parent_ids) > 1
            ]
            if not parent_groups:
                continue

            current_order = list(ordered[int(depth)])

            def gap_objective(candidate_order: Sequence[str]) -> Tuple[int, int, int]:
                """Оценить разрывы родительских групп для одного уровня.

                Args:
                    candidate_order: Проверяемый порядок родителей.

                Returns:
                    Число разорванных групп, сумма разрывов и максимальный разрыв.

                Raises:
                    KeyError: Если родитель отсутствует в порядке.

                Examples:
                    >>> # gap_objective(['p1', 'p2'])
                """

                positions = {
                    node_id: position
                    for position, node_id in enumerate(candidate_order)
                }
                gaps = [
                    max(positions[parent_id] for parent_id in parent_ids)
                    - min(positions[parent_id] for parent_id in parent_ids)
                    + 1
                    - len(parent_ids)
                    for parent_ids in parent_groups
                ]
                return (
                    sum(gap > 0 for gap in gaps),
                    sum(gaps),
                    max(gaps, default=0),
                )

            while True:
                best_objective = gap_objective(current_order)
                best_order: Optional[List[str]] = None
                for left_index in range(len(current_order)):
                    for right_index in range(left_index + 1, len(current_order)):
                        candidate_order = list(current_order)
                        candidate_order[left_index], candidate_order[right_index] = (
                            candidate_order[right_index],
                            candidate_order[left_index],
                        )
                        candidate_objective = gap_objective(candidate_order)
                        if candidate_objective < best_objective:
                            best_objective = candidate_objective
                            best_order = candidate_order
                if best_order is None:
                    break
                current_order = best_order
            ordered[int(depth)] = current_order

    for _ in range(sweep_count):
        top_down_sweep()
        bottom_up_shared_child_sweep()
    bottom_up_shared_child_sweep()
    refine_shared_child_adjacency()
    return ordered


def _tree_edge_port_offset(edge_index: int, edge_count: int, node_width: float) -> float:
    """Рассчитать отдельный горизонтальный порт ребра на границе узла.

    Args:
        edge_index: Индекс ребра среди рёбер узла, начиная с нуля.
        edge_count: Общее число рёбер на соответствующей стороне узла.
        node_width: Ширина узла в координатах графика.

    Returns:
        Смещение порта относительно центра узла.

    Raises:
        ValueError: Если индекс или количество рёбер некорректны.

    Examples:
        >>> round(_tree_edge_port_offset(0, 2, 3.0), 2)
        -0.54
        >>> round(_tree_edge_port_offset(1, 2, 3.0), 2)
        0.54
    """

    if edge_count < 1 or edge_index < 0 or edge_index >= edge_count:
        raise ValueError("Некорректный индекс порта ребра")
    usable_width = node_width * 0.72
    return usable_width * ((edge_index + 0.5) / edge_count - 0.5)


def build_anomaly_tree_from_excel(
    report_path: str | Path,
    output_path: str | Path | None = None,
    sheet_name: str = "Анализ аномалий",
    dpi: int = 160,
) -> Path:
    """Построить дерево аномальных сегментов по листу итогового Excel-файла.

    Узлы группируются по глубине. Ребро проводится только между соседними
    глубинами ``d`` и ``d + 1``, если признаки родителя являются строгим
    подмножеством признаков ребёнка. Один ребёнок может иметь несколько
    родителей, поэтому визуализация фактически является ориентированным DAG.
    Родители общих детей образуют непрерывные группы. Каждому ребру выделяются
    отдельные порты и отдельная трасса; цвет ребра кодирует дочерний сегмент.

    Args:
        report_path: Путь к Excel-отчёту с листом анализа аномалий.
        output_path: Путь к PNG, SVG или PDF. Если None, рядом с отчётом
            создаётся файл ``<имя_отчёта>_tree.png``.
        sheet_name: Имя листа-источника.
        dpi: Разрешение растрового изображения.

    Returns:
        Путь к созданному файлу визуализации.

    Raises:
        FileNotFoundError: Если Excel-отчёт отсутствует.
        ImportError: Если matplotlib недоступен.
        ValueError: Если лист пуст, не содержит обязательных колонок или
            указан неподдерживаемый формат результата.
        OSError: Если изображение невозможно записать.

    Examples:
        >>> # build_anomaly_tree_from_excel('gmv_anomaly_report_1.xlsx', 'gmv_tree.png')
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.path import Path as MatplotlibPath
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
    except ImportError as exc:
        raise ImportError("Для построения дерева требуется matplotlib") from exc

    report = Path(report_path)
    if not report.exists():
        raise FileNotFoundError(f"Excel-отчёт не найден: {report}")

    tree_path = (
        Path(output_path)
        if output_path is not None
        else report.with_name(f"{report.stem}_tree.png")
    )
    if not tree_path.suffix:
        tree_path = tree_path.with_suffix(".png")
    if tree_path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Дерево можно сохранить только в PNG, SVG или PDF")
    if dpi <= 0:
        raise ValueError("dpi должен быть положительным целым числом")

    analysis = pd.read_excel(report, sheet_name=sheet_name)
    required_columns = {"сегмент", "глубина", "z_scope", "anomaly_score", "Delta GMV"}
    missing_columns = sorted(required_columns - set(analysis.columns))
    if missing_columns:
        raise ValueError(
            f"На листе {sheet_name!r} отсутствуют обязательные колонки: {missing_columns}"
        )

    nodes = analysis.copy()
    optional_tree_columns = [
        "тип связи",
        "поглощён кем",
        "поглощён сегментом",
        "причина поглощения",
        "номер добавления в менеджерский вывод",
    ]
    for column in optional_tree_columns:
        if column not in nodes.columns:
            nodes[column] = ""
    nodes["сегмент"] = nodes["сегмент"].astype(str).str.strip()
    nodes["глубина"] = pd.to_numeric(nodes["глубина"], errors="coerce")
    nodes["z_scope"] = pd.to_numeric(nodes["z_scope"], errors="coerce")
    nodes["anomaly_score"] = pd.to_numeric(nodes["anomaly_score"], errors="coerce")
    nodes["Delta GMV"] = pd.to_numeric(nodes["Delta GMV"], errors="coerce")
    nodes["номер добавления в менеджерский вывод"] = pd.to_numeric(
        nodes["номер добавления в менеджерский вывод"],
        errors="coerce",
    )
    nodes = nodes[
        nodes["глубина"].notna()
        & nodes["глубина"].gt(0)
        & nodes["сегмент"].ne("")
        & nodes["сегмент"].ne("nan")
    ].copy()
    if nodes.empty:
        raise ValueError(f"Лист {sheet_name!r} не содержит сегментов глубины выше нуля")

    nodes["глубина"] = nodes["глубина"].astype(int)
    nodes = nodes.drop_duplicates(subset=["глубина", "сегмент"], keep="first")
    nodes = nodes.sort_values(["глубина", "сегмент"], kind="stable").reset_index(drop=True)
    nodes["node_id"] = [
        f"depth={int(depth)}|segment={segment}"
        for depth, segment in zip(nodes["глубина"], nodes["сегмент"])
    ]
    nodes["parts"] = nodes["сегмент"].map(_segment_key_parts)
    nodes["feature_set"] = nodes["parts"].map(frozenset)

    def clean_tree_value(value: object) -> str:
        """Вернуть безопасный текст для подписи узла дерева.

        Args:
            value: Исходное значение из Excel.

        Returns:
            Строка без `nan` и лишних пробелов.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> clean_tree_value(float('nan'))
            ''
        """

        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "<na>", "none"} else text

    def short_part_value(part: Tuple[str, str]) -> str:
        """Вернуть короткое значение отличающейся части сегмента.

        Args:
            part: Пара `(dimension, value)`.

        Returns:
            Значение признака без имени измерения.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> short_part_value(('geo', 'РФ'))
            'РФ'
        """

        return str(part[1]).strip()

    def relative_absorber_delta_lines(row: pd.Series, absorbed_role: str) -> List[str]:
        """Показать только отличие текущего сегмента от сегмента-поглотителя.

        Args:
            row: Строка узла дерева.
            absorbed_role: Роль поглотителя: родитель, ребёнок или конкурент.

        Returns:
            Строки вида `+РФ` или `-РФ`.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> relative_absorber_delta_lines(
            ...     pd.Series({'сегмент': 'a=1 × b=2', 'поглощён сегментом': 'a=1'}),
            ...     'родитель',
            ... )
            ['-2']
        """

        current_parts = set(_segment_key_parts(row.get("сегмент", "")))
        absorber_parts = set(_segment_key_parts(row.get("поглощён сегментом", "")))
        if not current_parts or not absorber_parts:
            return []

        if absorbed_role == "родитель":
            return [
                f"-{short_part_value(part)}"
                for part in sorted(current_parts - absorber_parts)
            ]
        if absorbed_role == "ребёнок":
            return [
                f"+{short_part_value(part)}"
                for part in sorted(absorber_parts - current_parts)
            ]

        removed = [f"-{short_part_value(part)}" for part in sorted(current_parts - absorber_parts)]
        added = [f"+{short_part_value(part)}" for part in sorted(absorber_parts - current_parts)]
        return [*removed, *added]

    def compact_absorption_reason(row: pd.Series) -> str:
        """Сжать техническую причину поглощения до короткой строки дерева.

        Args:
            row: Строка узла дерева.

        Returns:
            Короткая причина для подписи узла.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> compact_absorption_reason(pd.Series({'причина поглощения': 'родитель поглотил ребёнка по k=2.0 >= 1.35'}))
            'k родителя>=1.35'
        """

        reason = clean_tree_value(row.get("причина поглощения"))
        relationship_type = clean_tree_value(row.get("тип связи"))
        absorbed_role = clean_tree_value(row.get("поглощён кем"))
        k_threshold = ""
        if ">=" in reason:
            k_threshold = reason.split(">=", 1)[1].strip().split()[0].rstrip(";,.")
        if not reason:
            return ""
        if "k=" in reason and ">=" in reason:
            return f"k родителя>={k_threshold}" if k_threshold else "k родителя>=порог"
        if "ни один родитель не достиг k" in reason:
            return f"k родителя<{k_threshold}" if k_threshold else "k родителя<порог"
        if absorbed_role == "родитель-конкурент":
            return "k конкурента выше"
        if "временное правило" in reason:
            return "временное правило"
        if relationship_type == "M:N" or "M:N" in reason:
            return "заглушка M:N"
        if "одинаковом направлении Delta GMV" in reason:
            return "направление/порог k"
        return reason[:45].rstrip() + ("…" if len(reason) > 45 else "")

    def display_absorbed_role(absorbed_role: str) -> str:
        """Преобразовать техническую роль в форму для карточки дерева.

        Args:
            absorbed_role: Техническое значение роли поглотителя.

        Returns:
            Человекочитаемая форма в творительном падеже.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> display_absorbed_role('родитель')
            'родителем'
        """

        role_map = {
            "родитель": "родителем",
            "ребёнок": "ребенком",
            "ребенок": "ребенком",
            "родитель-конкурент": "родителем-конкурентом",
        }
        return role_map.get(absorbed_role, absorbed_role)

    def build_tree_detail_lines(row: pd.Series) -> List[str]:
        """Собрать дополнительные строки для не-менеджерского узла.

        Args:
            row: Строка узла из листа анализа аномалий.

        Returns:
            Список строк: связь, кем поглощён, причина.

        Raises:
            ValueError: Не выбрасывается.

        Examples:
            >>> build_tree_detail_lines(pd.Series({'номер добавления в менеджерский вывод': 1}))
            []
        """

        if not pd.isna(row.get("номер добавления в менеджерский вывод")):
            return []
        lines: List[str] = []
        relationship_type = clean_tree_value(row.get("тип связи"))
        absorbed_role = clean_tree_value(row.get("поглощён кем"))
        absorption_reason = compact_absorption_reason(row)
        if absorbed_role:
            lines.append(f"Поглощен {display_absorbed_role(absorbed_role)}")
            lines.extend(relative_absorber_delta_lines(row, absorbed_role))
        if relationship_type:
            lines.append(relationship_type)
        if absorption_reason:
            lines.append(absorption_reason)
        if not absorbed_role and not absorption_reason and relationship_type == "M:N":
            lines.append("заглушка M:N")
        return lines[:5]

    nodes["is_manager_output"] = nodes["номер добавления в менеджерский вывод"].notna()
    nodes["tree_detail_lines"] = nodes.apply(build_tree_detail_lines, axis=1)

    records = {
        str(row["node_id"]): row.to_dict()
        for _, row in nodes.iterrows()
    }
    depth_values = sorted(nodes["глубина"].unique().tolist())
    node_ids_by_depth = {
        int(depth): nodes.loc[nodes["глубина"].eq(depth), "node_id"].astype(str).tolist()
        for depth in depth_values
    }

    edges: List[Tuple[str, str]] = []
    children_by_parent: Dict[str, List[str]] = {
        node_id: [] for node_id in records
    }
    parents_by_child: Dict[str, List[str]] = {
        node_id: [] for node_id in records
    }
    for depth in depth_values:
        parent_ids = node_ids_by_depth[int(depth)]
        child_ids = node_ids_by_depth.get(int(depth) + 1, [])
        for parent_id in parent_ids:
            parent_features = records[parent_id]["feature_set"]
            for child_id in child_ids:
                if parent_features < records[child_id]["feature_set"]:
                    edges.append((parent_id, child_id))
                    children_by_parent[parent_id].append(child_id)
                    parents_by_child[child_id].append(parent_id)

    ordered_by_depth = _order_tree_layers_by_shared_children(
        node_ids_by_depth,
        edges,
        records,
    )

    node_width = 3.15
    horizontal_gap = 0.22
    cluster_padding_x = 0.28
    cluster_padding_y = 0.24
    max_nodes_in_layer = max(len(node_ids) for node_ids in ordered_by_depth.values())
    total_width = max_nodes_in_layer * node_width + max(0, max_nodes_in_layer - 1) * horizontal_gap
    node_heights = {
        depth: 1.24
        + 0.24
        * max(
            len(records[node_id]["parts"])
            for node_id in node_ids
        )
        + 0.18
        * max(
            len(records[node_id].get("tree_detail_lines", []))
            for node_id in node_ids
        )
        for depth, node_ids in ordered_by_depth.items()
    }
    edges_by_parent_depth = {
        int(depth): [
            edge
            for edge in edges
            if int(records[edge[0]]["глубина"]) == int(depth)
        ]
        for depth in depth_values
    }
    gap_after_depth = {
        int(depth): max(
            1.05,
            min(3.0, 0.07 * len(edges_by_parent_depth[int(depth)]) + 0.72),
        )
        for depth in depth_values
    }
    total_height = sum(
        node_heights[depth] + 2 * cluster_padding_y
        for depth in depth_values
    ) + sum(
        gap_after_depth[int(depth)]
        for depth in depth_values[:-1]
    )

    positions: Dict[str, Tuple[float, float]] = {}
    layer_bounds: Dict[int, Tuple[float, float, float, float]] = {}
    current_top = total_height
    for depth_index, depth in enumerate(depth_values):
        node_ids = ordered_by_depth[int(depth)]
        node_height = node_heights[int(depth)]
        cluster_height = node_height + 2 * cluster_padding_y
        center_y = current_top - cluster_height / 2
        row_width = len(node_ids) * node_width + max(0, len(node_ids) - 1) * horizontal_gap
        row_left = (total_width - row_width) / 2
        for position, node_id in enumerate(node_ids):
            center_x = row_left + node_width / 2 + position * (node_width + horizontal_gap)
            positions[node_id] = (center_x, center_y)
        layer_bounds[int(depth)] = (
            row_left - cluster_padding_x,
            center_y - cluster_height / 2,
            row_width + 2 * cluster_padding_x,
            cluster_height,
        )
        current_top -= cluster_height
        if depth_index < len(depth_values) - 1:
            current_top -= gap_after_depth[int(depth)]

    figure_width = max(14.0, min(42.0, total_width * 0.78))
    figure_height = max(8.0, total_height * 1.15 + 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.set_xlim(-0.55, total_width + 0.55)
    axis.set_ylim(-1.15, total_height + 1.15)
    axis.axis("off")

    for depth in depth_values:
        left, bottom, width, height = layer_bounds[int(depth)]
        axis.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.04,rounding_size=0.22",
                linewidth=0.9,
                edgecolor="#c7daf7",
                facecolor="#f8fbff",
                zorder=0,
            )
        )
        axis.text(
            left + 0.14,
            bottom + height - 0.12,
            f"Сегменты глубины {int(depth)}",
            color="#315a9f",
            fontsize=9,
            ha="left",
            va="top",
            zorder=4,
        )

    sorted_children_by_parent = {
        parent_id: sorted(
            child_ids,
            key=lambda child_id: (positions[child_id][0], child_id),
        )
        for parent_id, child_ids in children_by_parent.items()
    }
    sorted_parents_by_child = {
        child_id: sorted(
            parent_ids,
            key=lambda parent_id: (positions[parent_id][0], parent_id),
        )
        for child_id, parent_ids in parents_by_child.items()
    }
    edge_colors = [
        "#2563eb",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#4f46e5",
        "#0f766e",
        "#c2410c",
        "#0369a1",
    ]
    ordered_child_ids = sorted(
        {child_id for _, child_id in edges},
        key=lambda child_id: (
            int(records[child_id]["глубина"]),
            positions[child_id][0],
            child_id,
        ),
    )
    edge_color_by_child = {
        child_id: edge_colors[index % len(edge_colors)]
        for index, child_id in enumerate(ordered_child_ids)
    }

    for parent_depth in depth_values[:-1]:
        layer_edges = sorted(
            edges_by_parent_depth[int(parent_depth)],
            key=lambda edge: (
                positions[edge[0]][0],
                positions[edge[1]][0],
                edge[0],
                edge[1],
            ),
        )
        if not layer_edges:
            continue
        lane_count = len(layer_edges)
        for lane_index, (parent_id, child_id) in enumerate(layer_edges):
            parent_x, parent_y = positions[parent_id]
            child_x, child_y = positions[child_id]
            outgoing_ids = sorted_children_by_parent[parent_id]
            incoming_ids = sorted_parents_by_child[child_id]
            start_x = parent_x + _tree_edge_port_offset(
                outgoing_ids.index(child_id),
                len(outgoing_ids),
                node_width,
            )
            end_x = child_x + _tree_edge_port_offset(
                incoming_ids.index(parent_id),
                len(incoming_ids),
                node_width,
            )
            start_y = parent_y - node_heights[int(records[parent_id]["глубина"])] / 2
            end_y = child_y + node_heights[int(records[child_id]["глубина"])] / 2
            lane_top = start_y - 0.13
            lane_bottom = end_y + 0.13
            lane_y = lane_top - (lane_top - lane_bottom) * (lane_index + 1) / (lane_count + 1)
            path = MatplotlibPath(
                [
                    (start_x, start_y),
                    (start_x, lane_y),
                    (end_x, lane_y),
                    (end_x, end_y),
                ],
                [
                    MatplotlibPath.MOVETO,
                    MatplotlibPath.LINETO,
                    MatplotlibPath.LINETO,
                    MatplotlibPath.LINETO,
                ],
            )
            axis.add_patch(
                FancyArrowPatch(
                    path=path,
                    arrowstyle="-",
                    linewidth=2.2,
                    color="white",
                    alpha=0.96,
                    zorder=1,
                )
            )
            axis.add_patch(
                FancyArrowPatch(
                    path=path,
                    arrowstyle="-|>",
                    mutation_scale=6.5,
                    linewidth=0.82,
                    color=edge_color_by_child[child_id],
                    alpha=0.88,
                    zorder=1.1,
                )
            )

    connected_node_ids = {
        node_id
        for edge in edges
        for node_id in edge
    }
    sequence = 1
    for depth in depth_values:
        for node_id in ordered_by_depth[int(depth)]:
            node = records[node_id]
            center_x, center_y = positions[node_id]
            node_height = node_heights[int(depth)]
            delta_gmv = _safe_float(node["Delta GMV"])
            positive = delta_gmv >= 0
            facecolor = "#eff9f1" if positive else "#fff0f0"
            edgecolor = "#afd8b8" if positive else "#f0b2b2"
            accent = "#49a15d" if positive else "#df4d4d"
            isolated = node_id not in connected_node_ids
            if isolated:
                edgecolor = "#8b9bb2"
            is_manager_output = bool(node.get("is_manager_output", False))
            node_edgecolor = "#172554" if is_manager_output else edgecolor
            node_linewidth = 2.85 if is_manager_output else 1.0

            left = center_x - node_width / 2
            bottom = center_y - node_height / 2
            axis.add_patch(
                Rectangle(
                    (left, bottom),
                    node_width,
                    node_height,
                    linewidth=node_linewidth,
                    linestyle="-" if is_manager_output else ("--" if isolated else "-"),
                    edgecolor=node_edgecolor,
                    facecolor=facecolor,
                    zorder=2,
                )
            )
            badge_width = 0.52
            badge_height = 0.27
            badge_left = left + 0.13
            badge_bottom = bottom + node_height - 0.39
            axis.add_patch(
                Rectangle(
                    (badge_left, badge_bottom),
                    badge_width,
                    badge_height,
                    linewidth=0,
                    facecolor=accent,
                    zorder=3,
                )
            )
            axis.text(
                badge_left + badge_width / 2,
                badge_bottom + badge_height / 2,
                str(sequence),
                color="white",
                fontsize=6.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
            axis.text(
                left + 0.83,
                badge_bottom + badge_height / 2,
                f"Глубина {int(depth)}",
                color="#263342",
                fontsize=6.2,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=4,
            )

            feature_lines = [
                f"{dimension}={value}"
                for dimension, value in node["parts"]
            ]
            body_top = badge_bottom - 0.11
            axis.text(
                left + 0.14,
                body_top,
                "\n".join(feature_lines),
                color="#222222",
                fontsize=5.8,
                linespacing=1.35,
                ha="left",
                va="top",
                zorder=4,
            )
            detail_lines = [
                str(line)
                for line in node.get("tree_detail_lines", [])
                if str(line).strip()
            ]
            if detail_lines:
                detail_top = body_top - 0.215 * max(1, len(feature_lines)) - 0.07
                axis.text(
                    left + 0.14,
                    detail_top,
                    "\n".join(detail_lines),
                    color="#334155",
                    fontsize=4.75,
                    linespacing=1.22,
                    ha="left",
                    va="top",
                    zorder=4,
                )
            delta_y = bottom + 0.35
            axis.text(
                left + 0.14,
                delta_y,
                f"ΔGMV: {'+' if delta_gmv > 0 else ''}{_format_rub(delta_gmv)}",
                color="#171717",
                fontsize=6.3,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=4,
            )
            z_value = _safe_float(node["z_scope"], math.nan)
            score_value = _safe_float(node["anomaly_score"], math.nan)
            z_text = "nan" if math.isnan(z_value) else f"{z_value:.2f}"
            score_text = "nan" if math.isnan(score_value) else f"{score_value:.3f}"
            axis.text(
                left + 0.14,
                bottom + 0.12,
                f"z={z_text} · score={score_text}",
                color="#7f8b99",
                fontsize=5.3,
                ha="left",
                va="bottom",
                zorder=4,
            )
            sequence += 1

    figure.suptitle(
        "Граф",
        fontsize=15,
        y=0.985,
    )
    legend_handles = [
        Patch(facecolor="#eff9f1", edgecolor="#afd8b8", label="Положительный ΔGMV"),
        Patch(facecolor="#fff0f0", edgecolor="#f0b2b2", label="Отрицательный ΔGMV"),
        Patch(facecolor="white", edgecolor="#172554", linewidth=2.4, label="Попал в менеджерский вывод"),
        Patch(facecolor="white", edgecolor="#8b9bb2", linestyle="--", label="Изолированный сегмент"),
        Line2D([0], [0], color="#2563eb", linewidth=1.4, label="Одинаковый цвет рёбер — один ребёнок"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.01),
        frameon=False,
        ncol=5,
        fontsize=8,
    )
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(tree_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return tree_path.resolve()


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
    active_decision_log: pd.DataFrame,
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
        active_decision_log: Итерационный журнал решений active-графа.

    Returns:
        None.

    Raises:
        OSError: Если Excel-файл невозможно записать.

    Examples:
        >>> # write_anomaly_excel('gmv_anomaly_report.xlsx', thresholds, dims, history_df, panel, candidates, final_df, total, dates, dates[-1], coverage, log)
    """

    output_path = Path(output_path)
    params = pd.DataFrame(
        [
            ("признаки", " × ".join(dim_cols)),
            ("min_anomaly_abs", thresholds.min_anomaly_abs),
            ("min_z_score", thresholds.min_z_score),
            ("min_materiality_share", thresholds.min_materiality_share),
            ("sigma_floor", thresholds.sigma_floor),
            ("z_cap", thresholds.z_cap),
            ("dominance_threshold", thresholds.dominance_threshold),
            ("compensation_threshold", thresholds.compensation_threshold),
            ("anomaly_gross_move_threshold", thresholds.anomaly_gross_move_threshold),
            ("single_child_z_multiplier", thresholds.single_child_z_multiplier),
            ("single_child_gross_share_threshold", thresholds.single_child_gross_share_threshold),
            ("parent_child_absorption_k_threshold", thresholds.parent_child_absorption_k_threshold),
            ("current_cal_date", int(current_cal_date)),
        ],
        columns=["показатель", "значение"],
    )
    control = build_control_table(history_df, panel_df, candidates, final_df, coverage, dates, current_cal_date, total_by_date)
    params.insert(0, "раздел", "Параметры")
    control.insert(0, "раздел", "Контроль")
    params_and_control = pd.concat([params, control], ignore_index=True)

    manager_df = build_manager_summary(final_df, thresholds, float(total_by_date.loc[current_cal_date]))
    anomaly_analysis = build_anomaly_analysis_sheet(candidates, final_df, thresholds)
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
        "technical_child_count",
        "active_child_count",
        "single_child_gross_share",
        "anomaly_gross_move",
        "parent_child_absorption_k",
        "active_parent_z",
        "active_child_z",
        "active_partition_action",
        "active_partition_reason",
        "active_partition_status",
        "active_partition_status_reason",
        "active_graph_status",
        "is_active",
        "is_resolved",
        "is_natural_terminal",
        "is_terminal_current",
        "original_atomic_count",
        "active_atomic_count",
        "residual_atomic_count",
        "original_atomic_descendants",
        "active_atomic_descendants",
        "residual_atomic_descendants",
        "residual_gross_movement",
        "connection_break_flag",
        "connection_break_reason",
        "covered_by_segment_key",
        "absorbed_child_keys",
        "absorbed_child_labels",
    ]
    final_export = final_df[[col for col in final_cols if col in final_df.columns]].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        params_and_control.to_excel(writer, sheet_name="00_Параметры_и_контроль", index=False)
        manager_df.to_excel(writer, sheet_name="01_Менеджерский_вывод", index=False)
        final_export.to_excel(writer, sheet_name="02_Итог_аномалий", index=False)
        anomaly_analysis.to_excel(writer, sheet_name="Анализ аномалий", index=False)
        history_top.to_excel(writer, sheet_name="03_История_top", index=False)
        candidates.to_excel(writer, sheet_name="04_Диагностика_кандидатов", index=False)
        missing_zero.to_excel(writer, sheet_name="05_Пропуски_и_нули", index=False)
        control.to_excel(writer, sheet_name="06_Контроль", index=False)
        active_decision_log.to_excel(writer, sheet_name="07_Журнал_active_решений", index=False)

        highlight_manager_rows_on_anomaly_analysis(writer.sheets["Анализ аномалий"])

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
    tree_output_path: str | Path | None = None,
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
        tree_output_path: Необязательный путь к PNG, SVG или PDF с деревом
            листа «Анализ аномалий».

    Returns:
        Словарь таблиц результата.

    Raises:
        ValueError: Если входные данные некорректны.
        ImportError: Если запрошено дерево, но matplotlib недоступен.
        OSError: Если отчёт или дерево невозможно записать.

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
    final_df, diagnostics, active_decision_log = search_anomal(candidates, thresholds)
    write_anomaly_excel(
        output_path,
        thresholds,
        dims,
        history_df,
        panel_df,
        diagnostics,
        final_df,
        total_by_date,
        dates,
        current,
        coverage,
        active_decision_log,
    )
    if tree_output_path is not None:
        build_anomaly_tree_from_excel(output_path, tree_output_path)

    return {
        "history": history_df,
        "panel": panel_df,
        "candidates": diagnostics,
        "final": final_df,
        "active_decision_log": active_decision_log,
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
    parser.add_argument(
        "--tree-output",
        default=str(DEFAULT_TREE_OUTPUT_PATH),
        help="Путь к PNG, SVG или PDF с деревом листа «Анализ аномалий». По умолчанию создаётся «Граф.png».",
    )
    parser.add_argument("--sheet-name", default=0, help="Имя или номер листа Excel.")
    parser.add_argument("--period", default="1W", help="Период для фильтрации.")
    parser.add_argument("--dims", nargs="*", default=None, help="Явный список признаков.")
    parser.add_argument("--current-cal-date", type=int, default=None, help="Текущая анализируемая неделя cal_date.")
    parser.add_argument("--min-anomaly-abs", type=float, default=AnomalyThresholds.min_anomaly_abs)
    parser.add_argument("--min-z-score", type=float, default=AnomalyThresholds.min_z_score)
    parser.add_argument("--min-materiality-share", type=float, default=AnomalyThresholds.min_materiality_share)
    parser.add_argument("--sigma-floor", type=float, default=AnomalyThresholds.sigma_floor)
    parser.add_argument("--z-cap", type=float, default=AnomalyThresholds.z_cap)
    parser.add_argument("--dominance-threshold", type=float, default=AnomalyThresholds.dominance_threshold)
    parser.add_argument("--compensation-threshold", type=float, default=AnomalyThresholds.compensation_threshold)
    parser.add_argument(
        "--anomaly-gross-move-threshold",
        "--agm",
        dest="anomaly_gross_move_threshold",
        type=float,
        default=AnomalyThresholds.anomaly_gross_move_threshold,
        help="Threshold agm for anomalous gross movement share in a child group.",
    )
    parser.add_argument(
        "--single-child-z-multiplier",
        "--z-c-greater",
        type=float,
        default=AnomalyThresholds.single_child_z_multiplier,
        help="Множитель Z ребёнка в правиле active-разбиения.",
    )
    parser.add_argument(
        "--single-child-gross-share-threshold",
        "--single-child-gross-move-threshold",
        "--gm",
        dest="single_child_gross_share_threshold",
        type=float,
        default=AnomalyThresholds.single_child_gross_share_threshold,
        help="Порог неотрицательной gross_share единственного активного ребёнка.",
    )
    parser.add_argument(
        "--parent-child-absorption-k-threshold",
        "--k-threshold",
        dest="parent_child_absorption_k_threshold",
        type=float,
        default=AnomalyThresholds.parent_child_absorption_k_threshold,
        help="Порог k для поглощения дочернего сегмента родителем в search_anomal.",
    )
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
        min_materiality_share=args.min_materiality_share,
        sigma_floor=args.sigma_floor,
        z_cap=args.z_cap,
        dominance_threshold=args.dominance_threshold,
        compensation_threshold=args.compensation_threshold,
        anomaly_gross_move_threshold=args.anomaly_gross_move_threshold,
        single_child_z_multiplier=args.single_child_z_multiplier,
        single_child_gross_share_threshold=args.single_child_gross_share_threshold,
        parent_child_absorption_k_threshold=args.parent_child_absorption_k_threshold,
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
        tree_output_path=args.tree_output,
    )

    control = result["control"].set_index("показатель")["значение"].to_dict()
    print("Готово.")
    print(f"Итоговый файл: {args.output}")
    if args.tree_output:
        tree_path = Path(args.tree_output)
        if not tree_path.suffix:
            tree_path = tree_path.with_suffix(".png")
        print(f"Дерево аномалий: {tree_path.resolve()}")
    print(f"Кандидатов: {control.get('candidate_count')}")
    print(f"Выбрано аномалий: {control.get('selected_count')}")
    print(f"Нарушения пересечения атомов: {control.get('double_count_violation_count')}")


if __name__ == "__main__":
    main()
