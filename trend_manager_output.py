"""Собрать компактную витрину трендов для менеджерского вывода.

Модуль не участвует в поиске тренда, сегментации, scoring или Set Packing.
Он только объединяет уже рассчитанные таблицы в одну строку на eligible-сегмент,
пригодную для Excel и последующей выгрузки в YT.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd

from .trend_analysis import DECLINE, GROWTH
from .trend_most_recent_cp import (
    LEVEL_AND_SLOPE,
    LEVEL_SHIFT,
    LOCAL_FLAT,
    SLOPE_CHANGE,
    WEAK_OR_UNCLASSIFIED,
)


MANAGER_TREND_BASE_COLUMNS = (
    "segment_id",
    "segment_name",
    "segment_level",
    "current_gmv",
    "trend_direction",
    "local_trend_start_date",
    "local_trend_end_date",
    "local_trend_length",
    "local_trend_gmv_change_abs",
    "local_trend_gmv_change_relative",
    "local_trend_avg_gmv_change_per_period",
    "global_trend_start_date",
    "global_trend_end_date",
    "global_trend_length",
    "global_trend_gmv_change_abs",
    "global_trend_gmv_change_relative",
    "global_trend_avg_gmv_change_per_period",
    "global_trend_start_total_gmv_share",
    "global_trend_end_total_gmv_share",
    "global_trend_total_gmv_share_change_pp",
    "global_trend_total_gmv_share_dynamics",
    "global_trend_reversal_detected",
    "pre_global_trend_direction",
    "structural_change_detected",
    "last_structural_change_type",
    "last_structural_level_shift",
    "global_trend_structure",
    "last_local_trend_contribution_to_global_change_pct",
    "local_trend_contributions_to_global_change",
    "trend_selected",
)

MANAGER_TREND_COLUMN_LABELS = {
    # FIXED: в заголовках менеджерского листа единицы относительных показателей указаны в п.п.
    "segment_name": "Название сегмента",
    "segment_level": "Уровень сегмента",
    "current_gmv": "Текущий GMV",
    "trend_direction": "Динамика",
    "local_trend_start_date": "Начало локального тренда",
    "local_trend_end_date": "Конец локального тренда",
    "local_trend_length": "Длительность локального тренда, периодов",
    "local_trend_gmv_change_abs": "Абсолютное изменение локального тренда, GMV",
    "local_trend_gmv_change_relative": "Относительное изменение локального тренда, п.п.",
    "local_trend_avg_gmv_change_per_period": "Среднее изменение GMV за период, локальный тренд",
    "global_trend_start_date": "Начало глобального тренда",
    "global_trend_end_date": "Конец глобального тренда",
    "global_trend_length": "Длительность глобального тренда, периодов",
    "global_trend_gmv_change_abs": "Абсолютное изменение глобального тренда, GMV",
    "global_trend_gmv_change_relative": "Относительное изменение глобального тренда, п.п.",
    "global_trend_avg_gmv_change_per_period": "Среднее изменение GMV за период, глобальный тренд",
    "global_trend_start_total_gmv_share": "Доля в Total GMV в начале глобального тренда, п.п.",
    "global_trend_end_total_gmv_share": "Доля в Total GMV в конце глобального тренда, п.п.",
    "global_trend_total_gmv_share_change_pp": "Изменение доли в Total GMV, п.п.",
    "global_trend_total_gmv_share_dynamics": "Динамика доли в Total GMV",
    "global_trend_reversal_detected": "Разворот перед глобальным трендом",
    "pre_global_trend_direction": "Направление до глобального тренда",
    "structural_change_detected": "Есть структурная точка",
    "last_structural_change_type": "Тип последнего структурного изменения",
    "last_structural_level_shift": "Изменение уровня в последней структурной точке, GMV",
    "global_trend_structure": "Структура глобального тренда",
    "last_local_trend_contribution_to_global_change_pct": "Вклад последнего локального тренда в изменение глобального тренда, п.п.",
    "local_trend_contributions_to_global_change": "Вклады локальных трендов в изменение глобального тренда",
    "trend_selected": "Отобран оптимизацией",
}

MANAGER_TREND_COLUMN_DESCRIPTIONS = {
    "segment_name": "Читаемое описание сегмента.",
    "segment_level": "Уровень детализации сегмента.",
    "current_gmv": "GMV сегмента в последнем доступном периоде.",
    "trend_direction": "Направление подтверждённого текущего тренда: рост или падение.",
    "local_trend_start_date": "Дата начала последнего локального режима.",
    "local_trend_end_date": "Дата конца последнего локального режима.",
    "local_trend_length": "Число периодов в последнем локальном тренде.",
    "local_trend_gmv_change_abs": "Фактическое изменение GMV от начала до конца последнего локального режима.",
    "local_trend_gmv_change_relative": "Фактическое изменение локального GMV относительно его начального значения.",
    "local_trend_avg_gmv_change_per_period": "Фактическое среднее изменение GMV за один переход локального тренда.",
    "global_trend_start_date": "Дата начала объединённого устойчивого глобального тренда.",
    "global_trend_end_date": "Дата конца глобального тренда.",
    "global_trend_length": "Число периодов в глобальном тренде.",
    "global_trend_gmv_change_abs": "Фактическое изменение GMV от начала до конца глобального тренда.",
    "global_trend_gmv_change_relative": "Фактическое изменение глобального GMV относительно его начального значения.",
    "global_trend_avg_gmv_change_per_period": "Фактическое среднее изменение GMV за один переход глобального тренда.",
    "global_trend_start_total_gmv_share": "Доля GMV сегмента в Total GMV на старте глобального тренда.",
    "global_trend_end_total_gmv_share": "Доля GMV сегмента в Total GMV на конце глобального тренда.",
    "global_trend_total_gmv_share_change_pp": "Изменение доли сегмента в Total GMV в процентных пунктах.",
    "global_trend_total_gmv_share_dynamics": "Доля сегмента в Total GMV на старте и конце глобального тренда.",
    "global_trend_reversal_detected": "Был ли перед глобальным трендом подтверждённый тренд противоположного направления.",
    "pre_global_trend_direction": "Направление подтверждённого режима непосредственно перед глобальным трендом.",
    "structural_change_detected": "Найдена ли structural changepoint перед текущим локальным режимом.",
    "last_structural_change_type": "Классификация последнего structural changepoint.",
    "last_structural_level_shift": "Оценка скачкообразного изменения fitted уровня GMV в последнем changepoint.",
    "global_trend_structure": "Хронологическое описание глобального тренда, его режимов и структурных изменений.",
    "last_local_trend_contribution_to_global_change_pct": "Вклад последнего локального режима в изменение глобального тренда; значение может быть отрицательным или больше 100%.",
    "local_trend_contributions_to_global_change": "Хронологический список вкладов всех локальных режимов глобального тренда; сумма не обязана быть равна 100%.",
    "trend_selected": "Выбран ли сегмент exact Set Packing для непересекающегося оптимизационного фокуса.",
}

MANAGER_TREND_DATE_COLUMNS = frozenset(
    {
        "local_trend_start_date",
        "local_trend_end_date",
        "global_trend_start_date",
        "global_trend_end_date",
    }
)

MANAGER_TREND_PERCENTAGE_COLUMNS = frozenset(
    {
        "local_trend_gmv_change_relative",
        "global_trend_gmv_change_relative",
        "global_trend_start_total_gmv_share",
        "global_trend_end_total_gmv_share",
    }
)

MANAGER_TREND_COLUMN_WIDTHS = {
    "global_trend_total_gmv_share_dynamics": 30,
    "global_trend_structure": 52,
    "local_trend_contributions_to_global_change": 38,
}


def manager_trend_columns(dim_cols: Sequence[str]) -> tuple[str, ...]:
    """Вернуть стабильный порядок полей менеджерской витрины.

    Args:
        dim_cols: Упорядоченные измерения сегмента из конфигурации.

    Returns:
        Кортеж технических имён колонок, включая все ``dim_cols``.

    Raises:
        ValueError: Если измерения повторяются или пересекаются с полями витрины.

    Examples:
        >>> manager_trend_columns(["geo"])[3]
        'geo'
    """

    normalized_dims = tuple(str(column) for column in dim_cols)
    if len(normalized_dims) != len(set(normalized_dims)):
        raise ValueError("dim_cols содержит повторяющиеся имена")
    reserved = set(MANAGER_TREND_BASE_COLUMNS)
    collisions = sorted(set(normalized_dims) & reserved)
    if collisions:
        raise ValueError(
            "Измерения сегмента пересекаются с техническими полями "
            f"менеджерской витрины: {collisions}"
        )
    return (
        "segment_id",
        "segment_name",
        "segment_level",
        *normalized_dims,
        *MANAGER_TREND_BASE_COLUMNS[3:],
    )


def manager_trend_column_descriptions(
    dim_cols: Sequence[str],
) -> Mapping[str, str]:
    """Вернуть пояснения колонок менеджерской витрины.

    Args:
        dim_cols: Упорядоченные измерения сегмента из конфигурации.

    Returns:
        Словарь ``техническое имя -> описание`` для Excel-вывода.

    Raises:
        ValueError: Если ``dim_cols`` невалиден.

    Examples:
        >>> manager_trend_column_descriptions(["geo"])["geo"]
        'Значение атрибута сегмента geo.'
    """

    columns = manager_trend_columns(dim_cols)
    descriptions = dict(MANAGER_TREND_COLUMN_DESCRIPTIONS)
    descriptions["segment_id"] = "Технический уникальный ID сегмента для YT и DataLens."
    for dimension in dim_cols:
        descriptions[str(dimension)] = f"Значение атрибута сегмента {dimension}."
    return {column: descriptions[column] for column in columns}


def _require_columns(
    frame: pd.DataFrame,
    frame_name: str,
    required: set[str],
) -> None:
    """Проверить наличие обязательных полей входной таблицы.

    Args:
        frame: Проверяемая таблица.
        frame_name: Имя таблицы для текста ошибки.
        required: Множество обязательных колонок.

    Returns:
        None.

    Raises:
        ValueError: Если обязательные колонки отсутствуют.

    Examples:
        >>> _require_columns(pd.DataFrame({'x': [1]}), 'test', {'x'})
    """

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} не содержит колонки: {missing}")


def _as_finite_float(value: object) -> float:
    """Привести значение к конечному float либо вернуть ``nan``.

    Args:
        value: Произвольное значение таблицы.

    Returns:
        Конечный float либо ``math.nan``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _as_finite_float('1.5')
        1.5
    """

    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) and math.isfinite(float(converted)) else math.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Рассчитать отношение без бесконечностей при нулевом знаменателе.

    Args:
        numerator: Числитель отношения.
        denominator: Знаменатель отношения.

    Returns:
        Конечное отношение либо ``math.nan``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _safe_ratio(2.0, 4.0)
        0.5
    """

    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return float(numerator / denominator)


def _format_percent(value: float, signed: bool = False) -> str:
    """Сформатировать долю в виде компактного русскоязычного процента.

    Args:
        value: Значение в долях единицы.
        signed: Нужен ли явный знак для положительного значения.

    Returns:
        Строка процента либо ``н/д``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _format_percent(0.125)
        '12,5%'
    """

    if not math.isfinite(value):
        return "н/д"
    text = f"{value * 100:{'+' if signed else ''}.1f}".replace(".", ",")
    if text.endswith(",0"):
        text = text[:-2]
    return f"{text}%"


def _direction_label(direction: object, capitalize: bool = True) -> str:
    """Перевести каноническое направление тренда в подпись для менеджера.

    Args:
        direction: Каноническое направление тренда.
        capitalize: Нужна ли прописная первая буква.

    Returns:
        Строка ``Рост``, ``Падение`` либо ``Не подтверждено``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _direction_label(GROWTH)
        'Рост'
    """

    labels = {GROWTH: "рост", DECLINE: "падение"}
    label = labels.get(str(direction), "не подтверждено")
    return label.capitalize() if capitalize else label


def _direction_genitive_label(direction: object) -> str:
    """Вернуть направление тренда в форме после слова «начало».

    Args:
        direction: Каноническое направление тренда.

    Returns:
        Строка ``роста``, ``падения`` либо ``неподтверждённого тренда``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _direction_genitive_label(GROWTH)
        'роста'
    """

    return {
        GROWTH: "роста",
        DECLINE: "падения",
    }.get(str(direction), "неподтверждённого тренда")


def _structural_change_label(change_type: object) -> str:
    """Перевести технический тип structural CP в менеджерское описание.

    Args:
        change_type: Значение ``structural_change_type``.

    Returns:
        Короткое русскоязычное описание типа изменения.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _structural_change_label(LEVEL_SHIFT)
        'Сдвиг уровня'
    """

    labels = {
        "NONE": "Нет",
        LEVEL_SHIFT: "Сдвиг уровня",
        SLOPE_CHANGE: "Изменение скорости",
        LEVEL_AND_SLOPE: "Сдвиг уровня и изменение скорости",
        WEAK_OR_UNCLASSIFIED: "Структурное изменение без уверенной классификации",
    }
    return labels.get(str(change_type), "Структурное изменение без уверенной классификации")


def _speed_change_label(changepoint: Mapping[str, object], current_class: str) -> str:
    """Описать ускорение или замедление подтверждённого направления.

    Args:
        changepoint: Строка structural CP с OLS-наклонами соседних режимов.
        current_class: Класс правого локального режима.

    Returns:
        Менеджерская подпись изменения скорости.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _speed_change_label({'classification_previous_slope_ols': 1.0, 'classification_current_slope_ols': 2.0}, GROWTH)
        'ускорение роста'
    """

    previous_slope = _as_finite_float(changepoint.get("classification_previous_slope_ols"))
    current_slope = _as_finite_float(changepoint.get("classification_current_slope_ols"))
    direction = _direction_label(current_class, capitalize=False)
    if current_class not in {GROWTH, DECLINE} or not (
        math.isfinite(previous_slope) and math.isfinite(current_slope)
    ):
        return "изменение скорости"
    if math.isclose(abs(previous_slope), abs(current_slope), rel_tol=0.0, abs_tol=1e-12):
        return f"изменение скорости {direction}"
    noun = "роста" if current_class == GROWTH else "падения"
    return (
        f"ускорение {noun}"
        if abs(current_slope) > abs(previous_slope)
        else f"замедление {noun}"
    )


def _transition_label(
    changepoint: Mapping[str, object],
    current_class: str,
) -> str:
    """Собрать краткое описание CP перед очередным глобальным режимом.

    Args:
        changepoint: Строка CP между предыдущим и текущим режимами.
        current_class: Класс текущего правого режима.

    Returns:
        Подпись transition для поля структуры глобального тренда.

    Raises:
        ValueError: Если CP не содержит поддерживаемую классификацию.

    Examples:
        >>> _transition_label({'structural_change_type': LEVEL_SHIFT, 'level_shift': 5.0}, GROWTH)
        'сдвиг вверх'
    """

    if current_class == LOCAL_FLAT:
        return "FLAT"
    change_type = str(changepoint.get("structural_change_type"))
    level_shift = _as_finite_float(changepoint.get("level_shift"))
    level_label = (
        "сдвиг вверх"
        if math.isfinite(level_shift) and level_shift > 0.0
        else "сдвиг вниз"
        if math.isfinite(level_shift) and level_shift < 0.0
        else "сдвиг уровня"
    )
    if change_type == LEVEL_SHIFT:
        return level_label
    if change_type == SLOPE_CHANGE:
        return _speed_change_label(changepoint, current_class)
    if change_type == LEVEL_AND_SLOPE:
        return f"{level_label} и {_speed_change_label(changepoint, current_class)}"
    if change_type == WEAK_OR_UNCLASSIFIED:
        return "структурное изменение без уверенной классификации"
    raise ValueError(f"Неподдерживаемый тип structural CP: {change_type}")


def _regime_trend_value(
    regime: Mapping[str, object],
    trend_key: str,
    structural_key: str,
) -> object:
    """Прочитать поле окна тренда с fallback на прежний структурный контракт.

    Args:
        regime: Строка локального режима.
        trend_key: Имя нового поля окна тренда.
        structural_key: Имя совместимого структурного поля старого результата.

    Returns:
        Значение нового поля либо прежнего структурного поля.

    Raises:
        KeyError: Если в строке отсутствуют оба поля.

    Examples:
        >>> _regime_trend_value({'points': 4, 'trend_points': 5}, 'trend_points', 'points')
        5
    """

    if trend_key in regime:
        return regime[trend_key]
    return regime[structural_key]


def _global_structure_text(
    segment_regimes: pd.DataFrame,
    segment_changepoints: pd.DataFrame,
    global_direction: str,
) -> tuple[str, bool, str]:
    """Описать глобальный тренд и проверить разворот непосредственно перед ним.

    Args:
        segment_regimes: Все локальные режимы одного сегмента.
        segment_changepoints: Все structural CP одного сегмента.
        global_direction: Подтверждённое направление глобального тренда.

    Returns:
        Тройка из структуры, флага разворота и направления перед глобальным трендом.

    Raises:
        ValueError: Если глобальные режимы или их CP несогласованы.

    Examples:
        >>> regimes = pd.DataFrame([{'segment_index': 0, 'points': 4, 'local_regime_class': GROWTH, 'in_global_trend': True}])
        >>> _global_structure_text(regimes, pd.DataFrame(), GROWTH)[0]
        'Начало роста (4)'
    """

    ordered = segment_regimes.sort_values("segment_index", kind="stable")
    global_regimes = ordered.loc[
        ordered["in_global_trend"].fillna(False).astype(bool)
    ].copy()
    if global_regimes.empty:
        raise ValueError("Подтверждённый менеджерский тренд не содержит глобальных режимов")
    first = global_regimes.iloc[0]
    first_index = int(first["segment_index"])
    previous = ordered.loc[ordered["segment_index"].eq(first_index - 1)]
    reversal = False
    pre_global_direction = "Нет истории до глобального тренда"
    if not previous.empty:
        previous_row = previous.iloc[0]
        previous_class = str(previous_row["local_regime_class"])
        if previous_class in {GROWTH, DECLINE}:
            pre_global_direction = _direction_label(previous_class)
            reversal = previous_class != global_direction
        else:
            pre_global_direction = "Не подтверждено"

    first_class = str(first["local_regime_class"])
    # [FIXED] Длительность подписи относится к окну классификации тренда, а не
    # к непересекающемуся структурному сегменту CP/OLS.
    first_points = int(
        _regime_trend_value(first, "trend_points", "points")
    )
    if reversal:
        previous_row = previous.iloc[0]
        tokens = [
            "Разворот: "
            f"{_direction_label(previous_row['local_regime_class'], capitalize=False)} "
            f"({int(_regime_trend_value(previous_row, 'trend_points', 'points'))}) → "
            f"{_direction_label(first_class, capitalize=False)} ({first_points})"
        ]
    else:
        tokens = [
            f"Начало {_direction_genitive_label(first_class)} ({first_points})"
        ]

    changepoints_by_current_index = {
        int(row["current_regime_index"]): row.to_dict()
        for _, row in segment_changepoints.iterrows()
    }
    for _, regime in global_regimes.iloc[1:].iterrows():
        regime_index = int(regime["segment_index"])
        if regime_index not in changepoints_by_current_index:
            raise ValueError(
                "Для глобального режима отсутствует structural CP: "
                f"segment_index={regime_index}"
            )
        label = _transition_label(
            changepoints_by_current_index[regime_index],
            str(regime["local_regime_class"]),
        )
        regime_points = int(
            _regime_trend_value(regime, "trend_points", "points")
        )
        tokens.append(f"{label} ({regime_points})")
    return " → ".join(tokens), reversal, pre_global_direction


def build_manager_trend_output(
    trend_summary: pd.DataFrame,
    panel_df: pd.DataFrame,
    trend_segmentation: pd.DataFrame,
    trend_changepoints: pd.DataFrame,
    dim_cols: Sequence[str],
) -> pd.DataFrame:
    """Собрать одну менеджерскую строку на прошедший trend-фильтр сегмент.

    Args:
        trend_summary: Summary после trend scoring и Set Packing.
        panel_df: Полная иерархическая GMV-панель.
        trend_segmentation: Локальные режимы выбранной сегментации.
        trend_changepoints: Structural CP выбранной сегментации.
        dim_cols: Упорядоченные измерения сегмента из конфигурации.

    Returns:
        YT-совместимый DataFrame с техническими английскими именами колонок.

    Raises:
        ValueError: Если входные таблицы нарушают контракт полного трендового результата.

    Examples:
        >>> build_manager_trend_output(
        ...     pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
        ... )
        Traceback (most recent call last):
        ...
        ValueError: trend_summary не содержит колонки: ['current_trend_direction', 'global_trend_end_date', 'global_trend_gmv_change_abs', 'global_trend_gmv_change_relative', 'global_trend_length', 'global_trend_start_date', 'level_shift', 'segment_id', 'segment_key', 'segment_level', 'slice_depth', 'structural_change_detected', 'structural_change_type', 'trend_eligible', 'trend_selected']
    """

    output_columns = manager_trend_columns(dim_cols)
    summary_required = {
        "segment_id",
        "segment_key",
        "segment_level",
        "slice_depth",
        "trend_eligible",
        "trend_selected",
        "current_trend_direction",
        "global_trend_start_date",
        "global_trend_end_date",
        "global_trend_length",
        "global_trend_direction",
        "global_trend_gmv_change_abs",
        "global_trend_gmv_change_relative",
        "global_trend_start_gmv",
        "global_trend_end_gmv",
        "structural_change_detected",
        "structural_change_type",
        "level_shift",
    }
    segmentation_required = {
        "segment_id",
        "segment_index",
        "start_date",
        "end_date",
        "points",
        "local_start_gmv",
        "local_end_gmv",
        "local_gmv_change_abs",
        "local_gmv_change_relative",
        "local_regime_class",
        "in_global_trend",
    }
    changepoint_required = {
        "segment_id",
        "current_regime_index",
        "structural_change_type",
        "level_shift",
        "classification_previous_slope_ols",
        "classification_current_slope_ols",
    }
    panel_required = {
        "segment_id",
        "slice_depth",
        "cal_date",
        "gmv",
        *set(dim_cols),
    }
    _require_columns(trend_summary, "trend_summary", summary_required)
    _require_columns(trend_segmentation, "trend_segmentation", segmentation_required)
    _require_columns(trend_changepoints, "trend_changepoints", changepoint_required)
    _require_columns(panel_df, "panel_df", panel_required)

    summary = trend_summary.copy()
    summary["segment_id"] = summary["segment_id"].astype(str)
    if summary["segment_id"].duplicated().any():
        raise ValueError("trend_summary содержит дубли segment_id")
    eligible = summary.loc[summary["trend_eligible"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=output_columns)

    panel = panel_df.copy()
    panel["segment_id"] = panel["segment_id"].astype(str)
    panel["cal_date"] = pd.to_numeric(panel["cal_date"], errors="coerce")
    panel["gmv"] = pd.to_numeric(panel["gmv"], errors="coerce")
    if panel[["cal_date", "gmv"]].isna().any().any():
        raise ValueError("panel_df содержит нечисловые cal_date или gmv")
    panel["cal_date"] = panel["cal_date"].astype(int)
    if panel.duplicated(["segment_id", "cal_date"]).any():
        raise ValueError("panel_df содержит дубли segment_id x cal_date")

    metadata = panel[["segment_id", *dim_cols]].drop_duplicates().copy()
    if metadata["segment_id"].duplicated().any():
        raise ValueError("Атрибуты сегмента меняются внутри panel_df")
    metadata_by_id = metadata.set_index("segment_id").to_dict("index")

    latest_date = int(panel["cal_date"].max())
    current_rows = panel.loc[panel["cal_date"].eq(latest_date), ["segment_id", "gmv"]]
    current_gmv_by_id = current_rows.set_index("segment_id")["gmv"].to_dict()

    total_ids = panel.loc[
        pd.to_numeric(panel["slice_depth"], errors="coerce").eq(0),
        "segment_id",
    ].unique()
    if len(total_ids) != 1:
        raise ValueError(f"Ожидался один Total-сегмент, найдено: {len(total_ids)}")
    total_by_date = panel.loc[
        panel["segment_id"].eq(str(total_ids[0])), ["cal_date", "gmv"]
    ].set_index("cal_date")["gmv"].to_dict()
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in total_by_date.values()):
        raise ValueError("GMV Total должен быть конечным и положительным на всех датах")

    segmentation = trend_segmentation.copy()
    segmentation["segment_id"] = segmentation["segment_id"].astype(str)
    if segmentation.duplicated(["segment_id", "segment_index"]).any():
        raise ValueError("trend_segmentation содержит дубли segment_id x segment_index")
    regimes_by_id = {
        segment_id: frame.sort_values("segment_index", kind="stable").copy()
        for segment_id, frame in segmentation.groupby("segment_id", sort=False)
    }

    changepoints = trend_changepoints.copy()
    changepoints["segment_id"] = changepoints["segment_id"].astype(str)
    if changepoints.duplicated(["segment_id", "current_regime_index"]).any():
        raise ValueError("trend_changepoints содержит дубли segment_id x current_regime_index")
    changepoints_by_id = {
        segment_id: frame.copy()
        for segment_id, frame in changepoints.groupby("segment_id", sort=False)
    }

    records: list[dict[str, object]] = []
    for _, summary_row in eligible.sort_values("segment_key", kind="stable").iterrows():
        segment_id = str(summary_row["segment_id"])
        if segment_id not in metadata_by_id:
            raise ValueError(f"Для eligible-сегмента нет metadata: {segment_id}")
        if segment_id not in current_gmv_by_id:
            raise ValueError(f"Для eligible-сегмента нет GMV на последней дате: {segment_id}")
        if segment_id not in regimes_by_id:
            raise ValueError(f"Для eligible-сегмента нет trend_segmentation: {segment_id}")

        segment_regimes = regimes_by_id[segment_id]
        last_regime = segment_regimes.iloc[-1]
        global_start_date = int(summary_row["global_trend_start_date"])
        global_end_date = int(summary_row["global_trend_end_date"])
        if global_start_date not in total_by_date or global_end_date not in total_by_date:
            raise ValueError(f"Нет Total GMV на границе глобального тренда: {segment_id}")

        global_change = _as_finite_float(summary_row["global_trend_gmv_change_abs"])
        local_change = _as_finite_float(last_regime["local_gmv_change_abs"])
        global_rows = segment_regimes.loc[
            segment_regimes["in_global_trend"].fillna(False).astype(bool)
        ]
        if global_rows.empty:
            raise ValueError(f"Eligible-сегмент не содержит глобальных режимов: {segment_id}")
        global_direction = str(summary_row["global_trend_direction"])
        structure, reversal, pre_direction = _global_structure_text(
            segment_regimes,
            changepoints_by_id.get(segment_id, pd.DataFrame(columns=trend_changepoints.columns)),
            global_direction,
        )

        local_points = int(
            _regime_trend_value(last_regime, "trend_points", "points")
        )
        global_length = int(summary_row["global_trend_length"])
        start_share = _safe_ratio(
            _as_finite_float(summary_row["global_trend_start_gmv"]),
            float(total_by_date[global_start_date]),
        )
        end_share = _safe_ratio(
            _as_finite_float(summary_row["global_trend_end_gmv"]),
            float(total_by_date[global_end_date]),
        )
        contributions = (
            " → ".join(
                _format_percent(
                    _safe_ratio(
                        _as_finite_float(regime["local_gmv_change_abs"]),
                        global_change,
                    ),
                    signed=True,
                )
                for _, regime in global_rows.iterrows()
            )
            if global_change != 0.0 and math.isfinite(global_change)
            else "н/д"
        )
        attributes = metadata_by_id[segment_id]
        records.append(
            {
                "segment_id": segment_id,
                "segment_name": summary_row["segment_key"],
                "segment_level": summary_row["segment_level"],
                **{dimension: attributes[dimension] for dimension in dim_cols},
                "current_gmv": float(current_gmv_by_id[segment_id]),
                "trend_direction": _direction_label(global_direction),
                "local_trend_start_date": int(
                    _regime_trend_value(
                        last_regime,
                        "trend_start_date",
                        "start_date",
                    )
                ),
                "local_trend_end_date": int(
                    _regime_trend_value(
                        last_regime,
                        "trend_end_date",
                        "end_date",
                    )
                ),
                "local_trend_length": local_points,
                "local_trend_gmv_change_abs": local_change,
                "local_trend_gmv_change_relative": _as_finite_float(last_regime["local_gmv_change_relative"]),
                "local_trend_avg_gmv_change_per_period": _safe_ratio(local_change, float(local_points - 1)),
                "global_trend_start_date": global_start_date,
                "global_trend_end_date": global_end_date,
                "global_trend_length": global_length,
                "global_trend_gmv_change_abs": global_change,
                "global_trend_gmv_change_relative": _as_finite_float(summary_row["global_trend_gmv_change_relative"]),
                "global_trend_avg_gmv_change_per_period": _safe_ratio(global_change, float(global_length - 1)),
                "global_trend_start_total_gmv_share": start_share,
                "global_trend_end_total_gmv_share": end_share,
                "global_trend_total_gmv_share_change_pp": (
                    (end_share - start_share) * 100.0
                    if math.isfinite(start_share) and math.isfinite(end_share)
                    else math.nan
                ),
                "global_trend_total_gmv_share_dynamics": f"{_format_percent(start_share)} → {_format_percent(end_share)}",
                "global_trend_reversal_detected": "Да" if reversal else "Нет",
                "pre_global_trend_direction": pre_direction,
                "structural_change_detected": (
                    "Да" if bool(summary_row["structural_change_detected"]) else "Нет"
                ),
                "last_structural_change_type": _structural_change_label(summary_row["structural_change_type"]),
                "last_structural_level_shift": _as_finite_float(summary_row["level_shift"]),
                "global_trend_structure": structure,
                "last_local_trend_contribution_to_global_change_pct": (
                    _safe_ratio(local_change, global_change) * 100.0
                    if global_change != 0.0 and math.isfinite(global_change)
                    else math.nan
                ),
                "local_trend_contributions_to_global_change": contributions,
                "trend_selected": "Да" if bool(summary_row["trend_selected"]) else "Нет",
            }
        )
    return pd.DataFrame(records, columns=output_columns)


__all__ = [
    "MANAGER_TREND_COLUMN_DESCRIPTIONS",
    "MANAGER_TREND_COLUMN_LABELS",
    "MANAGER_TREND_COLUMN_WIDTHS",
    "MANAGER_TREND_DATE_COLUMNS",
    "MANAGER_TREND_PERCENTAGE_COLUMNS",
    "build_manager_trend_output",
    "manager_trend_column_descriptions",
    "manager_trend_columns",
]
