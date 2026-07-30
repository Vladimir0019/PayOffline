"""??????, ???????????? ? ?????????? ????????? ?????? GMV."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .config import ANOMALY_TECH_COLUMNS, DIM_COLUMNS, METRIC_COLUMNS


def _is_missing(value: object) -> bool:
    """REMOVED import dependency: проверить, является ли значение отсутствующим.

    Args:
        value: Исходное значение признака.

    Returns:
        True, если значение нужно считать отсутствующим.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _is_missing(float('nan'))
        True
        >>> _is_missing('QR')
        False
    """

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalize_dim_value(value: object) -> Optional[str]:
    """ADDED: Нормализовать значение признака.

    Args:
        value: Исходное значение признака.

    Returns:
        Строковое значение признака или None, если признак не входит в срез.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> normalize_dim_value(' QR ')
        'QR'
        >>> normalize_dim_value(None) is None
        True
    """

    if _is_missing(value):
        return None
    return str(value).strip()


def segment_id_from_row(row: pd.Series, dim_cols: Sequence[str]) -> str:
    """ADDED: Создать технический идентификатор сегмента.

    Args:
        row: Строка таблицы.
        dim_cols: Список признаков.

    Returns:
        Идентификатор сегмента.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> segment_id_from_row(pd.Series({'geo': 'РФ', 'channel': None}), ['geo', 'channel'])
        'РФ|∅'
    """

    parts = []
    for col in dim_cols:
        value = normalize_dim_value(row.get(col))
        parts.append(value if value is not None else "∅")
    return "|".join(parts)


def build_segment_key_and_level(row: pd.Series, dim_cols: Sequence[str]) -> Tuple[str, str, int]:
    """ADDED: Построить человекочитаемый ключ и уровень сегмента.

    Args:
        row: Строка таблицы.
        dim_cols: Список признаков.

    Returns:
        Кортеж: ключ сегмента, уровень сегмента, глубина.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> s = pd.Series({'geo': 'РФ', 'channel': None})
        >>> build_segment_key_and_level(s, ['geo', 'channel'])
        ('geo=РФ', 'geo', 1)
    """

    used = []
    key_parts = []
    for col in dim_cols:
        value = normalize_dim_value(row.get(col))
        if value is not None:
            used.append(col)
            key_parts.append(f"{col}={value}")
    if not used:
        return "ИТОГО", "ИТОГО", 0
    return " × ".join(key_parts), " × ".join(used), len(used)


def _segment_key_parts(segment_key: object) -> List[Tuple[str, str]]:
    """ADDED: Разобрать ключ сегмента для валидации входного контракта.

    Args:
        segment_key: Человекочитаемый ключ сегмента.

    Returns:
        Список пар ``(dimension, value)``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _segment_key_parts("geo=РФ × products=QR")
        [('geo', 'РФ'), ('products', 'QR')]
    """

    parts: List[Tuple[str, str]] = []
    for raw_part in re.split(r"\s+(?:x|\u00d7|\u0413\u2014)\s+", str(segment_key)):
        part = raw_part.strip()
        if "=" not in part:
            continue
        dimension, value = part.split("=", 1)
        parts.append((dimension.strip(), value.strip()))
    return parts


def _validate_segment_metadata(df: pd.DataFrame, dim_cols: Sequence[str]) -> None:
    """ADDED: Проверить согласованность иерархии сегментов до построения панели.

    Args:
        df: Подготовленная входная таблица с вычисленными сегментными полями.
        dim_cols: Разрешённый упорядоченный список dimensions.

    Returns:
        None.

    Raises:
        ValueError: Если ключ, глубина или metadata сегмента противоречат контракту.

    Examples:
        >>> frame = pd.DataFrame([{'segment_id': 'РФ|∅', 'segment_key': 'geo=РФ', 'segment_level': 'geo', 'slice_depth': 1, 'geo': 'РФ', 'products': None}])
        >>> _validate_segment_metadata(frame, ['geo', 'products'])
    """

    known_dims = set(dim_cols)
    issues: List[str] = []
    for row in df.itertuples(index=False):
        segment_id = str(row.segment_id)
        depth = int(row.slice_depth)
        key = str(row.segment_key)
        parts = _segment_key_parts(key)
        part_dims = [dimension for dimension, _ in parts]
        expected_key, expected_level, expected_depth = build_segment_key_and_level(
            pd.Series({column: getattr(row, column) for column in dim_cols}),
            dim_cols,
        )

        if depth != expected_depth:
            issues.append(f"{segment_id}: slice_depth={depth}, вычисленная глубина={expected_depth}")
        elif key != expected_key or str(row.segment_level) != expected_level:
            issues.append(f"{segment_id}: segment_key или segment_level не соответствует dimension values")
        elif depth > 0 and not parts:
            issues.append(f"{segment_id}: segment_key не разбирается")
        elif len(set(part_dims)) != len(part_dims):
            issues.append(f"{segment_id}: segment_key содержит повторяющиеся dimensions")
        elif unknown_dims := sorted(set(part_dims) - known_dims):
            issues.append(f"{segment_id}: segment_key содержит неизвестные dimensions {unknown_dims}")
        elif len(parts) != depth:
            issues.append(f"{segment_id}: в segment_key {len(parts)} dimensions, в slice_depth {depth}")

    metadata_cols = ["segment_key", "segment_level", "slice_depth", *dim_cols]
    conflicts = df.groupby("segment_id", dropna=False)[metadata_cols].nunique(dropna=False)
    inconsistent_ids = conflicts.index[conflicts.gt(1).any(axis=1)].astype(str).tolist()
    if inconsistent_ids:
        issues.append("несогласованная metadata между неделями: " + ", ".join(inconsistent_ids[:5]))

    if issues:
        suffix = "" if len(issues) <= 5 else f" (и ещё {len(issues) - 5})"
        raise ValueError("Нарушен контракт сегментной иерархии: " + "; ".join(issues[:5]) + suffix)


def candidate_covers_atomic(candidate: pd.Series, atomic: pd.Series, dim_cols: Sequence[str]) -> bool:
    """ADDED: Проверить, покрывает ли кандидат атомарный сегмент.

    Args:
        candidate: Строка кандидата-родителя.
        atomic: Строка атомарного сегмента.
        dim_cols: Список признаков.

    Returns:
        True, если кандидат покрывает атом.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> candidate_covers_atomic(pd.Series({'x': None}), pd.Series({'x': 'a'}), ['x'])
        True
    """

    for col in dim_cols:
        value = normalize_dim_value(candidate.get(col))
        if value is not None and value != normalize_dim_value(atomic.get(col)):
            return False
    return True


def infer_anomaly_dimension_columns(df: pd.DataFrame, explicit_dims: Optional[Sequence[str]] = None) -> List[str]:
    """Определить бизнес-признаки для anomaly-анализа.

    Args:
        df: Входная таблица.
        explicit_dims: Явно заданные признаки. Если не переданы, используются
            фиксированные значения из ``config.DIM_COLUMNS``.

    Returns:
        Список бизнес-признаков.

    Raises:
        ValueError: Если явно заданный признак отсутствует.

    Examples:
        >>> infer_anomaly_dimension_columns(pd.DataFrame(columns=['geo', 'products', 'merchants_type', 'is_terminal_or_cpqr', 'source']))
        ['geo', 'products', 'merchants_type', 'is_terminal_or_cpqr']
    """

    # FIXED: Состав dimensions определяется контрактом, а не случайными колонками Excel.
    dimensions = DIM_COLUMNS if explicit_dims is None else explicit_dims
    missing = [col for col in dimensions if col not in df.columns]
    if missing:
        raise ValueError(f"В таблице отсутствуют явно заданные признаки: {missing}")
    # ADDED: Все остальные поля конкретной Excel-выгрузки фиксируются как технические.
    ANOMALY_TECH_COLUMNS.update(col for col in df.columns if col not in dimensions)
    return list(dimensions)


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
    # FIXED: Валидация структуры сегментов выполняется до построения недельной панели.
    _validate_segment_metadata(df, dims)

    # ADDED: Upstream YQL формирует ровно одну строку на сегмент и дату внутри выбранного period.
    # Дубли означают нарушение входного контракта; их нельзя молча агрегировать в Python.
    duplicate_mask = df.duplicated(subset=["segment_id", "cal_date"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            df.loc[duplicate_mask, ["segment_id", "cal_date"]]
            .drop_duplicates()
            .sort_values(["segment_id", "cal_date"])
        )
        examples = [
            f"{row.segment_id} × {int(row.cal_date)}"
            for row in duplicate_keys.head(5).itertuples(index=False)
        ]
        suffix = "" if len(duplicate_keys) <= 5 else f" (и ещё {len(duplicate_keys) - 5})"
        raise ValueError(
            "Нарушена уникальность ключа segment_id × cal_date: "
            + "; ".join(examples)
            + suffix
        )

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

    # FIXED: Вход уже агрегирован upstream YQL и проверен на уникальность segment_id × cal_date.
    # Поэтому метрики переносим напрямую, без повторной агрегации в Python.
    available_metric_cols = [metric for metric in METRIC_COLUMNS if metric in history_df.columns]
    value_cols = ["segment_id", "cal_date", "gmv", *available_metric_cols]
    weekly = history_df[value_cols].copy()
    full_index = pd.MultiIndex.from_product(
        [meta["segment_id"].tolist(), list(dates)],
        names=["segment_id", "cal_date"],
    )
    panel = full_index.to_frame(index=False).merge(
        weekly,
        how="left",
        on=["segment_id", "cal_date"],
        indicator="_source_merge",
        validate="one_to_one",
    )
    panel["row_missing_in_source"] = panel["_source_merge"] == "left_only"
    panel = panel.drop(columns="_source_merge")
    panel["gmv"] = panel["gmv"].fillna(0.0).astype(float)
    for metric in available_metric_cols:
        panel[metric] = panel[metric].fillna(0.0).astype(float)
    panel = panel.merge(meta, how="left", on="segment_id")
    return panel.sort_values(["segment_id", "cal_date"]).reset_index(drop=True)
