"""UDF-адаптер продуктового поиска GMV-трендов над потоком строк YT."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .config import DIM_COLUMNS, PERIOD
from .data_preparation import build_full_week_grid, prepare_history_dataframe
from .trend_analysis import (
    TrendModelConfig,
    TrendThresholds,
    build_configured_trend_analysis,
)
from .trend_manager_output import (
    MANAGER_TREND_DATE_COLUMNS,
    build_manager_trend_output,
    manager_trend_columns,
)
from .trend_scoring import TrendScoringConfig, build_trend_selection
from .udf_runtime import UDF_INPUT_SCHEMA, _decode_yql_value, _rows_to_dataframe


# ADDED: Excel хранит дату как число дней от 1899-12-30, тогда как входная
# витрина и внутренний алгоритм используют дни от Unix-эпохи 1970-01-01.
_EXCEL_UNIX_EPOCH_OFFSET_DAYS = 25_569

# ADDED: Python3 UDF использует технические имена manager_output. Русские
# заголовки добавляет внешний SELECT генератора: YQL Struct запрещает точки в
# именах полей, а в Excel-контракте есть несколько заголовков с «п.п.».
_MANAGER_TREND_YQL_TYPES = {
    "segment_name": "String",
    "segment_level": "String",
    **{dimension: "String?" for dimension in DIM_COLUMNS},
    "current_gmv": "Double",
    "trend_direction": "String",
    "local_trend_start_date": "Int64",
    "local_trend_end_date": "Int64",
    "local_trend_length": "Int64",
    "local_trend_gmv_change_abs": "Double",
    "local_trend_gmv_change_relative": "Double",
    "local_trend_avg_gmv_change_per_period": "Double",
    "global_trend_start_date": "Int64",
    "global_trend_end_date": "Int64",
    "global_trend_length": "Int64",
    "global_trend_gmv_change_abs": "Double",
    "global_trend_gmv_change_relative": "Double",
    "global_trend_avg_gmv_change_per_period": "Double",
    "global_trend_start_total_gmv_share": "Double",
    "global_trend_end_total_gmv_share": "Double",
    "global_trend_total_gmv_share_change_pp": "Double",
    "global_trend_total_gmv_share_dynamics": "String",
    "global_trend_reversal_detected": "String",
    "pre_global_trend_direction": "String",
    "structural_change_detected": "String",
    "last_structural_change_type": "String",
    "last_structural_level_shift": "Double?",
    "global_trend_structure": "String",
    "last_local_trend_contribution_to_global_change_pct": "Double?",
    "local_trend_contributions_to_global_change": "String",
    "trend_selected": "String",
}

_YT_MANAGER_COLUMNS = tuple(
    column for column in manager_trend_columns(DIM_COLUMNS) if column != "segment_id"
)

if set(_YT_MANAGER_COLUMNS) != set(_MANAGER_TREND_YQL_TYPES):
    missing_types = sorted(set(_YT_MANAGER_COLUMNS) - set(_MANAGER_TREND_YQL_TYPES))
    extra_types = sorted(set(_MANAGER_TREND_YQL_TYPES) - set(_YT_MANAGER_COLUMNS))
    raise RuntimeError(
        "Схема trend UDF не согласована с manager_output: "
        f"нет типов={missing_types}, лишние типы={extra_types}"
    )

TREND_UDF_OUTPUT_SCHEMA: Tuple[Tuple[str, str], ...] = tuple(
    (column, _MANAGER_TREND_YQL_TYPES[column])
    for column in _YT_MANAGER_COLUMNS
)


def _is_missing(value: object) -> bool:
    """Проверить, является ли скаляр отсутствующим значением pandas.

    Args:
        value: Значение одной ячейки manager_output.

    Returns:
        True для None, NaN и pd.NA.

    Raises:
        ValueError: Не выбрасывается для скалярных значений витрины.

    Examples:
        >>> _is_missing(float("nan"))
        True
    """

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_manager_value(
    column: str,
    value: object,
    yql_type: str,
) -> object:
    """Преобразовать одну ячейку менеджерского dataframe в тип YQL Struct.

    Args:
        column: Техническое имя колонки manager_output.
        value: Исходное скалярное значение.
        yql_type: Тип поля в выходном Struct.

    Returns:
        Строку, int, float либо None для Optional-поля.

    Raises:
        ValueError: Если обязательное значение отсутствует или не является
            конечным числом.

    Examples:
        >>> _normalize_manager_value("local_trend_start_date", 0, "Int64")
        25569
    """

    optional = yql_type.endswith("?")
    base_type = yql_type[:-1] if optional else yql_type
    if _is_missing(value):
        if optional:
            return None
        raise ValueError(f"Обязательное поле manager_output пусто: {column}")

    if base_type == "String":
        return str(_decode_yql_value(value))
    if base_type == "Int64":
        normalized = int(value)
        if column in MANAGER_TREND_DATE_COLUMNS:
            normalized += _EXCEL_UNIX_EPOCH_OFFSET_DAYS
        return normalized
    if base_type == "Double":
        normalized = float(value)
        if not math.isfinite(normalized):
            if optional:
                return None
            raise ValueError(
                f"Обязательное поле manager_output не является конечным: {column}"
            )
        return normalized
    raise ValueError(f"Неподдерживаемый YQL-тип manager_output: {yql_type}")


def _manager_output_records(manager_output: pd.DataFrame) -> List[Dict[str, object]]:
    """Преобразовать manager_output в техническую UDF-схему без пояснений.

    Args:
        manager_output: Результат ``build_manager_trend_output``.

    Returns:
        Строки с техническими именами manager_output; финальный YQL SELECT
        переименовывает их в заголовки листа «Менеджерский вывод».

    Raises:
        ValueError: Если dataframe не соответствует менеджерскому контракту.

    Examples:
        >>> _manager_output_records(pd.DataFrame(columns=manager_trend_columns(DIM_COLUMNS)))
        []
    """

    missing_columns = sorted(set(manager_trend_columns(DIM_COLUMNS)) - set(manager_output.columns))
    if missing_columns:
        raise ValueError(
            "manager_output не содержит обязательные колонки: "
            f"{missing_columns}"
        )

    records: List[Dict[str, object]] = []
    for _, row in manager_output.iterrows():
        record: Dict[str, object] = {}
        for column in _YT_MANAGER_COLUMNS:
            record[column] = _normalize_manager_value(
                column,
                row[column],
                _MANAGER_TREND_YQL_TYPES[column],
            )
        records.append(record)
    return records


def run_algorithm(rows: Iterable[object]) -> List[Dict[str, object]]:
    """Рассчитать тренды одного периода и вернуть менеджерскую YT-витрину.

    Args:
        rows: Полный поток строк настроенного периода из исходной YT-таблицы.

    Returns:
        Бизнес-строки листа «Менеджерский вывод» без Excel-строки пояснений.

    Raises:
        ValueError: Если вход, трендовый расчёт, hierarchy или solver нарушают
            существующий контракт алгоритма.
        RuntimeError: Если оптимизационная задача не доказала optimum.

    Examples:
        >>> # result = run_algorithm(input_rows)
    """

    source = _rows_to_dataframe(rows)
    actual_periods = sorted(source["period"].dropna().astype(str).unique().tolist())
    if actual_periods != [PERIOD]:
        raise ValueError(
            "Один вызов trend UDF должен получать ровно период из config.PERIOD; "
            f"ожидалось {[PERIOD]}, получено {actual_periods}"
        )

    history, dims, dates = prepare_history_dataframe(
        source,
        period=PERIOD,
        dim_cols=DIM_COLUMNS,
    )
    panel = build_full_week_grid(history, dims, dates)
    thresholds = TrendThresholds()
    analysis = build_configured_trend_analysis(
        panel,
        dates,
        thresholds=thresholds,
        model_config=TrendModelConfig(trend_search_method="most_recent_cp"),
    )
    selection = build_trend_selection(
        analysis["trend_summary"],
        panel,
        dates,
        dims,
        thresholds=thresholds,
        config=TrendScoringConfig(),
    )
    manager_output = build_manager_trend_output(
        selection["trend_summary"],
        panel,
        analysis["trend_segmentation"],
        analysis["trend_changepoints"],
        dims,
    )
    return _manager_output_records(manager_output)


__all__ = ["TREND_UDF_OUTPUT_SCHEMA", "UDF_INPUT_SCHEMA", "run_algorithm"]
