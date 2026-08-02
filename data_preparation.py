"""Загрузка, валидация и нормализация исторической витрины GMV.

Модуль отвечает за единственную границу доверия к внешним данным
(boundary layer). После ``load_history_table`` внутренний код работает с
гарантированными типами: ``cal_date`` и ``slice_depth`` — ``int``,
``gmv``/``tx``/``au``/``am`` — конечные ``float``, dimension values — ``str``
либо ``None``. Ratio-метрики ``aov``/``tpm``/``freq`` могут быть ``NaN``, что
означает «отношение не определено» и не приравнивается к нулю.

Контракт входного файла описан в ``docs/DATA_CONTRACT.md``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .config import ANOMALY_TECH_COLUMNS, DIM_COLUMNS, METRIC_COLUMNS
from .segment_keys import (
    SEGMENT_KEY_SEPARATOR,
    TOTAL_SEGMENT_KEY,
    parse_segment_key_parts,
)


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
    """FIXED: Создать однозначный JSON-идентификатор сегмента.

    Args:
        row: Строка таблицы.
        dim_cols: Список признаков в фиксированном порядке.

    Returns:
        Компактный JSON-массив, где пропуск представлен JSON null.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> segment_id_from_row(pd.Series({'geo': 'РФ', 'channel': None}), ['geo', 'channel'])
        '["РФ",null]'
        >>> segment_id_from_row(pd.Series({'geo': '∅'}), ['geo'])
        '["∅"]'
    """

    parts = [normalize_dim_value(row.get(col)) for col in dim_cols]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


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
        return TOTAL_SEGMENT_KEY, TOTAL_SEGMENT_KEY, 0
    # FIXED: Ключ склеивается тем же разделителем, которым его разбирает
    # segment_keys.parse_segment_key_parts; литерал больше не дублируется.
    return (
        SEGMENT_KEY_SEPARATOR.join(key_parts),
        SEGMENT_KEY_SEPARATOR.join(used),
        len(used),
    )


def _validate_segment_metadata(df: pd.DataFrame, dim_cols: Sequence[str]) -> None:
    """ADDED: Проверить согласованность иерархии сегментов до построения панели.

    Args:
        df: Подготовленная входная таблица с вычисленными сегментными полями.
        dim_cols: Разрешённый упорядоченный список dimensions.

    Returns:
        None.

    Raises:
        ValueError: Если ключ, глубина или metadata сегмента противоречат
            контракту, либо ``segment_key`` не разбирается.

    Examples:
        >>> frame = pd.DataFrame([{'segment_id': '["РФ",null]', 'segment_key': 'geo=РФ', 'segment_level': 'geo', 'slice_depth': 1, 'geo': 'РФ', 'products': None}])
        >>> _validate_segment_metadata(frame, ['geo', 'products'])
    """

    known_dims = set(dim_cols)
    issues: List[str] = []
    for row in df.itertuples(index=False):
        segment_id = str(row.segment_id)
        depth = int(row.slice_depth)
        key = str(row.segment_key)
        # FIXED: Неразбираемый ключ больше не пропускается молча: общий парсер
        # выбрасывает ValueError, иначе глубина сегмента была бы занижена.
        parts = parse_segment_key_parts(key)
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


def period_to_weeks(period: str) -> int:
    """ADDED: Извлечь число недель из обязательного периода витрины.

    Args:
        period: Период строго в формате ``<положительное целое>W``, например
            ``"1W"`` или ``"13W"``.

    Returns:
        Число недель в одном интервале.

    Raises:
        ValueError: Если период пустой или не соответствует формату.

    Examples:
        >>> period_to_weeks("13W")
        13
    """

    match = re.fullmatch(r"([1-9]\d*)W", str(period).strip().upper())
    if match is None:
        raise ValueError(
            "period обязателен и должен иметь формат '<положительное целое>W', "
            "например '1W' или '13W'"
        )
    return int(match.group(1))


def load_history_table(
    input_path: str | Path,
    sheet_name: int | str = 0,
    *,
    period: str,
    dim_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Загрузить историческую таблицу срезов GMV.

    Args:
        input_path: Путь к Excel- или CSV-файлу.
        sheet_name: Имя или номер листа Excel.
        period: Обязательное значение периода для фильтрации в формате ``<N>W``.
        dim_cols: Явно заданные признаки. Если None, признаки определяются автоматически.

    Returns:
        Кортеж: очищенная таблица, список признаков, список недель cal_date из total-слоя.

    Raises:
        ValueError: Если входной файл не поддерживается, нет обязательных колонок или total-слой некорректен.

    Examples:
        >>> # df, dims, dates = load_history_table('payoffline_pulse_hier_4_13w.xlsx')
    """

    period_weeks = period_to_weeks(period)
    normalized_period = str(period).strip().upper()
    input_path = Path(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Поддерживаются только .xlsx, .xls и .csv")

    required = {"period", "cal_date", "slice_depth", "gmv", "tx", "au", "am"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Не хватает обязательных колонок: {missing_required}")

    # FIXED: Период является обязательной частью входного контракта.
    # ADDED: Первая строка YT-выгрузки — служебный заголовок типов
    # (period='string', cal_date='uint16', gmv='double', ...), а не данные.
    # Именно из-за него pandas читает числовые колонки как object. Строку
    # отбрасываем до приведения типов. Подробности: docs/DATA_CONTRACT.md.
    df = df[df["period"].astype(str) != "string"].copy()
    df = df[df["period"].astype(str).str.strip().str.upper() == normalized_period].copy()
    if df.empty:
        raise ValueError(f"После фильтра периода {normalized_period!r} не осталось строк")
    unique_periods = sorted(df["period"].astype(str).unique().tolist())
    if len(unique_periods) != 1:
        raise ValueError(
            "После фильтра должен остаться ровно один период; "
            f"получено: {unique_periods}"
        )
    if unique_periods[0].strip().upper() != normalized_period:
        raise ValueError(
            f"Оставшийся период {unique_periods[0]!r} не совпадает с запрошенным {normalized_period!r}"
        )

    # FIXED: Повреждённая обязательная строка не должна исчезнуть и затем
    # восстановиться нулём при построении полной сетки.
    required_numeric = ["cal_date", "slice_depth", "gmv", "tx", "au", "am"]
    numeric_issues: List[str] = []
    for column in required_numeric:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = converted.isna() | ~converted.map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else False
        )
        if invalid_mask.any():
            bad_rows = df.loc[invalid_mask, ["period", column]].head(5)
            examples = ", ".join(
                f"index={index}, value={row[column]!r}"
                for index, row in bad_rows.iterrows()
            )
            numeric_issues.append(
                f"{column}: {int(invalid_mask.sum())} невалидных значений ({examples})"
            )
        df[column] = converted
    if numeric_issues:
        raise ValueError(
            "Обязательные числовые поля должны быть заполненными, числовыми и конечными: "
            + "; ".join(numeric_issues)
        )

    df["cal_date"] = df["cal_date"].astype(int)
    df["slice_depth"] = df["slice_depth"].astype(int)
    df["gmv"] = df["gmv"].astype(float)
    for metric in ("tx", "au", "am"):
        df[metric] = df[metric].astype(float)
    # FIXED: Ratio-метрики могут быть NULL при нулевом знаменателе в upstream YQL.
    for metric in ("aov", "tpm", "freq"):
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

    expected_calendar_step = 7 * period_weeks
    date_diffs = pd.Series(dates).diff().dropna().astype(int).tolist()
    if date_diffs and any(diff != expected_calendar_step for diff in date_diffs):
        raise ValueError(
            "В total-слое нарушен шаг календарного периода: "
            f"ожидалось {expected_calendar_step} дней для period={normalized_period!r}; "
            "без total GMV интервал нельзя восстановить"
        )
    if (total_by_date["gmv"].astype(float) <= 0).any():
        raise ValueError("Total GMV должен быть положительным на каждой неделе")

    # ADDED: Ни одна сегментная дата не должна молча исчезнуть при построении
    # сетки, временную ось которой задаёт total-слой.
    orphan_mask = ~df["cal_date"].isin(dates)
    if orphan_mask.any():
        orphan_rows = (
            df.loc[orphan_mask, ["period", "cal_date", "segment_key"]]
            .sort_values(["cal_date", "segment_key"])
            .head(5)
        )
        examples = "; ".join(
            f"period={row.period}, cal_date={int(row.cal_date)}, segment={row.segment_key}"
            for row in orphan_rows.itertuples(index=False)
        )
        raise ValueError(
            "Найдены даты сегментов вне total-календаря: "
            f"{int(orphan_mask.sum())} строк; примеры: {examples}"
        )

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
    # FIXED: Нулями восстанавливаются только отсутствующие строки и аддитивные
    # показатели. NULL ratio-метрики существующих и восстановленных строк
    # сохраняют семантику «отношение не определено».
    panel["gmv"] = panel["gmv"].fillna(0.0).astype(float)
    for metric in ("tx", "au", "am"):
        if metric in available_metric_cols:
            panel[metric] = panel[metric].fillna(0.0).astype(float)
    for metric in ("aov", "tpm", "freq"):
        if metric in available_metric_cols:
            panel[metric] = panel[metric].astype(float)
    panel = panel.merge(meta, how="left", on="segment_id")
    return panel.sort_values(["segment_id", "cal_date"]).reset_index(drop=True)
