"""UDF-адаптер продуктового поиска аномалий над потоком строк YT."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from .anomaly_scoring import (
    apply_hierarchy_score_adjustment,
    build_anomaly_candidates,
    build_atomic_coverage,
    build_ratio_anomaly_candidates,
)
from .config import AnomalyThresholds, DIM_COLUMNS, RATIO_METRICS
from .data_preparation import build_full_week_grid, prepare_history_dataframe
from .set_packing import search_anomal


# ADDED: Реальная схема входной YT-таблицы проверена через @schema 2026-08-05.
UDF_INPUT_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("period", "String"),
    # FIXED: YQL интерпретирует legacy uint16 как Date; перед UDF значение
    # явно приводится к Int64, чтобы Python получил дни от эпохи.
    ("cal_date", "Int64?"),
    ("slice_depth", "Uint8?"),
    ("geo", "String?"),
    ("products", "String?"),
    ("merchants_type", "String?"),
    ("is_terminal_or_cpqr", "String?"),
    ("gmv", "Double"),
    ("tx0", "Uint64"),
    ("tx", "Uint64"),
    ("au", "Uint64"),
    ("am", "Uint64"),
    ("refund_tx_numerator", "Uint64"),
    ("authzone_tx_numerator", "Uint64"),
    ("payapp_tx_numerator", "Uint64"),
    ("split_gmv_numerator", "Double"),
    ("credlim_gmv_numerator", "Double"),
    ("tips_gmv_numerator", "Double"),
    ("cashback_gmv_numerator", "Double"),
    ("aov", "Double?"),
    ("tpm", "Double?"),
    ("freq", "Double?"),
    ("success_rate", "Double?"),
    ("refund_tx_share", "Double?"),
    ("authzone_tx_share", "Double?"),
    ("payapp_tx_share", "Double?"),
    ("split_gmv_share", "Double?"),
    ("credlim_gmv_share", "Double?"),
    ("tips_gmv_share", "Double?"),
    ("cashback_gmv_share", "Double?"),
)


# ADDED: Числа остаются числами; старый UDF ошибочно сериализовал всё в String.
UDF_OUTPUT_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("Период", "String"),
    ("Название метрики", "String"),
    ("segment_id", "String"),
    ("Сегмент", "String"),
    ("Глубина", "Int64"),
    ("geo", "String?"),
    ("products", "String?"),
    ("merchants_type", "String?"),
    ("is_terminal_or_cpqr", "String?"),
    ("Текущая дата", "Int64"),
    ("Предыдущая дата", "Int64"),
    ("Текущее значение", "Double?"),
    ("Предыдущее значение", "Double?"),
    ("Единица значения", "String"),
    ("Изменение метрики", "Double?"),
    ("Единица изменения", "String"),
    ("Изменение структуры", "String?"),
    ("GMV WoW %", "Double?"),
    ("TX WoW %", "Double?"),
    ("AU WoW %", "Double?"),
    ("AM WoW %", "Double?"),
    ("AOV WoW %", "Double?"),
    ("TPM WoW %", "Double?"),
    ("Freq WoW %", "Double?"),
    ("robust_z", "Double"),
    ("materiality_share", "Double"),
    ("anomaly_score", "Double"),
    ("rank", "Int64?"),
)


_STRUCTURE_LABELS = {
    "новый сегмент": "новый",
    "возобновившийся сегмент": "возобновившийся",
    "исчезнувший сегмент": "исчезнувший",
}


def _decode_yql_value(value: object) -> object:
    """Преобразовать YQL String в обычную Python-строку.

    Args:
        value: Значение поля входной строки UDF.

    Returns:
        Декодированная строка либо исходное значение.

    Raises:
        UnicodeDecodeError: Не выбрасывается, ошибочные байты заменяются.

    Examples:
        >>> _decode_yql_value(b"1W")
        '1W'
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _optional_float(value: object, *, multiplier: float = 1.0) -> float | None:
    """Вернуть конечное число либо None для Optional-поля YQL.

    Args:
        value: Исходное числовое значение.
        multiplier: Множитель единиц результата.

    Returns:
        Конечный float или None.

    Raises:
        TypeError: Не выбрасывается для нечислового значения.

    Examples:
        >>> _optional_float(0.12, multiplier=100.0)
        12.0
    """

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return normalized * float(multiplier)


def _rows_to_dataframe(rows: Iterable[object]) -> pd.DataFrame:
    """Материализовать поток YT в DataFrame строго по входному контракту.

    Args:
        rows: Поток структур YQL.

    Returns:
        DataFrame всех строк одного периода.

    Raises:
        ValueError: Если поток пуст или строка не содержит обязательное поле.

    Examples:
        >>> _rows_to_dataframe([{"period": "1W", "cal_date": 1}]).shape[0]
        1
    """

    column_names = [name for name, _ in UDF_INPUT_SCHEMA]
    records: List[Dict[str, object]] = []
    for row in rows:
        if isinstance(row, Mapping):
            record = {
                column: _decode_yql_value(row.get(column))
                for column in column_names
            }
        else:
            record = {
                column: _decode_yql_value(getattr(row, column, None))
                for column in column_names
            }
        records.append(record)
    if not records:
        raise ValueError("Поток входных строк UDF пуст")
    return pd.DataFrame(records, columns=column_names)


def _apply_hierarchy_adjustment(
    candidates: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    thresholds: AnomalyThresholds,
    *,
    ratio_mode: bool,
    contribution_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Применить общую hierarchy-корректировку с режимом нужной метрики.

    Args:
        candidates: Кандидаты GMV или долевой метрики.
        coverage: Атомарное покрытие сегментов.
        thresholds: Пороговая конфигурация.
        ratio_mode: Использовать семантику exact ratio contribution.
        contribution_tolerance: Допуск аддитивности долевой метрики.

    Returns:
        Кандидаты с финальным anomaly_score.

    Raises:
        ValueError: Если нарушен контракт hierarchy-расчёта.

    Examples:
        >>> # adjusted = _apply_hierarchy_adjustment(df, coverage, thresholds, ratio_mode=False)
    """

    common = {
        "aggregation_bonus_lambda": thresholds.aggregation_bonus_lambda,
        "single_child_factor": thresholds.single_child_factor,
        "dominant_child_capture_threshold": (
            thresholds.dominant_child_capture_threshold
        ),
        "dominant_child_score_margin": thresholds.dominant_child_score_margin,
        "max_hierarchy_descendants": thresholds.max_hierarchy_descendants,
    }
    if ratio_mode:
        return apply_hierarchy_score_adjustment(
            candidates,
            coverage,
            movement_column="hierarchy_movement",
            allow_zero_movement=True,
            contribution_mode="exact_atomic",
            contribution_reconciliation_tolerance=contribution_tolerance,
            **common,
        )
    return apply_hierarchy_score_adjustment(candidates, coverage, **common)


def _wow_fields(gmv_row: pd.Series) -> Dict[str, float | None]:
    """Сформировать семь WoW-полей в процентных пунктах шкалы 0–100.

    Args:
        gmv_row: GMV-диагностика того же сегмента.

    Returns:
        Словарь продуктовых WoW-колонок.

    Raises:
        ValueError: Не выбрасывается; неопределённые проценты становятся None.

    Examples:
        >>> _wow_fields(pd.Series({"relative_wow": 0.1}))["GMV WoW %"]
        10.0
    """

    return {
        "GMV WoW %": _optional_float(gmv_row.get("relative_wow"), multiplier=100.0),
        "TX WoW %": _optional_float(gmv_row.get("tx_wow_pct"), multiplier=100.0),
        "AU WoW %": _optional_float(gmv_row.get("au_wow_pct"), multiplier=100.0),
        "AM WoW %": _optional_float(gmv_row.get("am_wow_pct"), multiplier=100.0),
        "AOV WoW %": _optional_float(gmv_row.get("aov_wow_pct"), multiplier=100.0),
        "TPM WoW %": _optional_float(gmv_row.get("tpm_wow_pct"), multiplier=100.0),
        "Freq WoW %": _optional_float(gmv_row.get("freq_wow_pct"), multiplier=100.0),
    }


def _product_record(
    row: pd.Series,
    gmv_row: pd.Series,
    *,
    period: str,
    metric_name: str,
    structure_change: str | None,
) -> Dict[str, object]:
    """Преобразовать выбранную строку алгоритма в продуктовую схему.

    Args:
        row: Выбранная аномалия или GMV lifecycle-строка.
        gmv_row: GMV-диагностика этого же сегмента для общих WoW-полей.
        period: Анализируемый период.
        metric_name: ``GMV`` либо имя долевой метрики.
        structure_change: Бизнес-структурное изменение GMV.

    Returns:
        Одна строка выходного Struct UDF.

    Raises:
        ValueError: Если metric_name не поддержан.

    Examples:
        >>> # record = _product_record(row, gmv_row, period="1W", metric_name="GMV", structure_change=None)
    """

    is_gmv = metric_name == "GMV"
    if is_gmv:
        current_value = _optional_float(row.get("gmv_current"))
        previous_value = _optional_float(row.get("gmv_previous"))
        metric_change = _optional_float(row.get("wow_delta_gmv"))
        value_unit = "руб."
        change_unit = "руб."
    else:
        current_value = _optional_float(
            row.get("metric_value_current"), multiplier=100.0
        )
        previous_value = _optional_float(
            row.get("metric_value_previous"), multiplier=100.0
        )
        metric_change = _optional_float(row.get("metric_delta_pp"))
        value_unit = "%"
        change_unit = "п.п."

    rank_raw = row.get("rank")
    rank = None if pd.isna(rank_raw) else int(rank_raw)
    record: Dict[str, object] = {
        "Период": str(period),
        "Название метрики": str(metric_name),
        "segment_id": str(row["segment_id"]),
        "Сегмент": str(row["segment_key"]),
        "Глубина": int(row["slice_depth"]),
        "geo": row.get("geo"),
        "products": row.get("products"),
        "merchants_type": row.get("merchants_type"),
        "is_terminal_or_cpqr": row.get("is_terminal_or_cpqr"),
        "Текущая дата": int(gmv_row["current_cal_date"]),
        "Предыдущая дата": int(gmv_row["previous_cal_date"]),
        "Текущее значение": current_value,
        "Предыдущее значение": previous_value,
        "Единица значения": value_unit,
        "Изменение метрики": metric_change,
        "Единица изменения": change_unit,
        "Изменение структуры": structure_change,
        "robust_z": float(row["robust_z"]),
        "materiality_share": float(row["materiality_share"]),
        "anomaly_score": float(row["anomaly_score"]),
        "rank": rank,
        **_wow_fields(gmv_row),
    }
    # FIXED: Optional dimension не должен передавать pandas.NaN в YQL.
    for dimension in DIM_COLUMNS:
        if pd.isna(record[dimension]):
            record[dimension] = None
        elif isinstance(record[dimension], bytes):
            record[dimension] = _decode_yql_value(record[dimension])
        elif record[dimension] is not None:
            record[dimension] = str(record[dimension])
    return {column: record.get(column) for column, _ in UDF_OUTPUT_SCHEMA}


def run_algorithm(rows: Iterable[object]) -> List[Dict[str, object]]:
    """Рассчитать один период и вернуть выбранные аномалии и GMV lifecycle.

    Args:
        rows: Полный поток строк ровно одного периода YT.

    Returns:
        Строки выбранных GMV/ratio-аномалий и GMV-структурных изменений.

    Raises:
        ValueError: Если вход, метрики, hierarchy или solver некорректны.
        RuntimeError: Если оптимизационная задача не доказала optimum.

    Examples:
        >>> # rows = run_algorithm(input_rows)
    """

    source = _rows_to_dataframe(rows)
    actual_periods = sorted(source["period"].dropna().astype(str).unique().tolist())
    if len(actual_periods) != 1 or actual_periods[0] not in {"1W", "4W", "13W"}:
        raise ValueError(
            "Один вызов UDF должен получать ровно один поддерживаемый период; "
            f"получено: {actual_periods}"
        )
    period = actual_periods[0]
    required_ratio_columns = {
        column
        for spec in RATIO_METRICS
        for column in (
            spec.value_column,
            spec.numerator_column,
            spec.denominator_column,
        )
    }
    missing_ratio_columns = sorted(required_ratio_columns - set(source.columns))
    if missing_ratio_columns:
        raise ValueError(
            "В продуктовой UDF отсутствуют обязательные колонки долевых метрик: "
            + ", ".join(missing_ratio_columns)
        )

    thresholds = AnomalyThresholds()
    history, dims, dates = prepare_history_dataframe(
        source,
        period=period,
        dim_cols=DIM_COLUMNS,
    )
    current = int(dates[-1])
    panel = build_full_week_grid(history, dims, dates)
    metadata = panel.drop_duplicates(subset=["segment_id"]).reset_index(drop=True)
    coverage = build_atomic_coverage(metadata, dims)

    gmv_candidates, _ = build_anomaly_candidates(
        panel,
        dims,
        dates,
        thresholds,
        current,
        coverage=coverage,
    )
    gmv_candidates = _apply_hierarchy_adjustment(
        gmv_candidates,
        coverage,
        thresholds,
        ratio_mode=False,
    )
    gmv_final, gmv_diagnostics, _ = search_anomal(
        gmv_candidates,
        thresholds,
        coverage=coverage,
    )
    gmv_by_segment = {
        str(row["segment_id"]): row
        for _, row in gmv_diagnostics.iterrows()
    }

    # ADDED: Итог GMV — объединение выбранных solver-ом строк и всех
    # структурных изменений. Пересечение выдаётся одной строкой.
    gmv_output_rows: Dict[str, pd.Series] = {
        str(row["segment_id"]): row for _, row in gmv_final.iterrows()
    }
    structure_by_segment: Dict[str, str] = {}
    structural = gmv_diagnostics[
        gmv_diagnostics["slice_depth"].astype(int).gt(0)
        & gmv_diagnostics["state"].isin(_STRUCTURE_LABELS)
    ]
    for _, row in structural.iterrows():
        segment_id = str(row["segment_id"])
        gmv_output_rows.setdefault(segment_id, row)
        structure_by_segment[segment_id] = _STRUCTURE_LABELS[str(row["state"])]

    result: List[Dict[str, object]] = []
    for segment_id, row in gmv_output_rows.items():
        result.append(
            _product_record(
                row,
                gmv_by_segment[segment_id],
                period=period,
                metric_name="GMV",
                structure_change=structure_by_segment.get(segment_id),
            )
        )

    # ADDED: Доли возвращают только строки, выбранные независимым Set Packing.
    for spec in RATIO_METRICS:
        ratio_candidates = build_ratio_anomaly_candidates(
            panel,
            dims,
            dates,
            thresholds,
            spec,
            current,
            coverage,
        )
        ratio_candidates = _apply_hierarchy_adjustment(
            ratio_candidates,
            coverage,
            thresholds,
            ratio_mode=True,
            contribution_tolerance=spec.validation_abs_tolerance,
        )
        ratio_final, _, _ = search_anomal(
            ratio_candidates,
            thresholds,
            coverage=coverage,
        )
        for _, row in ratio_final.iterrows():
            segment_id = str(row["segment_id"])
            result.append(
                _product_record(
                    row,
                    gmv_by_segment[segment_id],
                    period=period,
                    metric_name=spec.name,
                    structure_change=None,
                )
            )

    metric_order = {"GMV": 0, **{spec.name: i + 1 for i, spec in enumerate(RATIO_METRICS)}}
    return sorted(
        result,
        key=lambda row: (
            metric_order[str(row["Название метрики"])],
            row["rank"] is None,
            row["rank"] if row["rank"] is not None else 10**9,
            str(row["Сегмент"]),
        ),
    )


__all__ = ["UDF_INPUT_SCHEMA", "UDF_OUTPUT_SCHEMA", "run_algorithm"]
