"""
Итоговый алгоритм v2 для bottom-up adaptive contribution analysis по GMV.

Назначение:
- вход: уже корректная предагрегированная таблица со всеми срезами от total до максимальной глубины;
- выход: Excel-файл с дельтами, диагностикой родителей, журналом решений, итоговым разбиением,
  детализацией детей, исключёнными конфликтами и контролем суммы;
- предобработки качества данных, проверки дат, дублей и полноты атомарного слоя здесь намеренно не являются
  частью основного алгоритма.

Ключевые решения v2:
- алгоритм динамически работает с произвольным набором признаков;
- ручной приоритет уровней не используется;
- защита от «слишком общего родителя» не используется;
- если общий родитель проходит правила, он может поглотить нижние блоки;
- конфликт родителей разрешается детерминированно через универсальный порядок сортировки;
- блок взаимной компенсации поглощает детей как специальный блок, а не как основной драйвер;
- при доминирующем ребёнке мелкие соседи сворачиваются в «прочее внутри родителя»;
- total-уровень не выбирается как основной драйвер.

Пример запуска:
    python gmv_final_algorithm_v2.py \
        --input payoffline_pulse_hier_full_cuts_filtered.xlsx \
        --output gmv_algorithm_v2_result.xlsx \
        --period 1W
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = Path(__file__).with_name("payoffline_pulse_hier_1.xlsx")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("Результат_1.xlsx")


TECH_COLUMNS = {
    "period",
    "cal_date",
    "calendar_date",
    "slice_depth",
    "gmv",
    "share_in_total_gmv",
    "gmv_previous",
    "gmv_current",
    "delta_gmv",
    "abs_delta_gmv",
    "share_in_total_delta",
    "share_in_current_total_gmv",
    "share_in_previous_total_gmv",
}


@dataclass(frozen=True)
class Thresholds:
    """Пороги алгоритма.

    Args:
        min_segment_delta: Минимальный модуль чистого изменения для самостоятельного сегмента.
        min_child_delta: Минимальный модуль изменения ребёнка, чтобы оставить его как существенный.
        min_gross_movement: Минимальное внутреннее движение для признания сегмента существенным.
        net_share_threshold: Порог чистой доли: чем выше, тем меньше компенсация внутри родителя.
        same_direction_threshold: Порог доли одного направления внутри родителя.
        concentration_threshold: Порог доминирования одного ребёнка.

    Returns:
        Экземпляр Thresholds.

    Raises:
        ValueError: Не выбрасывается автоматически.

    Examples:
        >>> th = Thresholds()
        >>> th.net_share_threshold
        0.75
    """

    min_segment_delta: float = 2_000_000.0
    min_child_delta: float = 1_000_000.0
    min_gross_movement: float = 1_500_000.0
    net_share_threshold: float = 0.6
    same_direction_threshold: float = 0.75
    concentration_threshold: float = 0.80


@dataclass
class ActiveBlock:
    """Активный блок итогового технического разбиения.

    Args:
        block_id: Уникальный идентификатор блока.
        block_key: Человекочитаемое описание блока.
        block_level: Уровень группировки блока.
        slice_depth: Глубина блока.
        delta_gmv: Чистое изменение GMV блока.
        atomic_ids: Атомарные сегменты, покрытые блоком.
        output_block: Тип итогового блока.
        reason: Причина попадания блока в итоговое разбиение.
        source_parent_id: Родитель, из которого создан блок.

    Returns:
        Экземпляр ActiveBlock.

    Raises:
        ValueError: Не выбрасывается автоматически.

    Examples:
        >>> b = ActiveBlock('a', 'канал=QR', 'канал', 1, 10.0, frozenset({'a1'}), 'основной драйвер', 'пример')
        >>> b.delta_gmv
        10.0
    """

    block_id: str
    block_key: str
    block_level: str
    slice_depth: int
    delta_gmv: float
    atomic_ids: frozenset[str]
    output_block: str
    reason: str
    source_parent_id: str = ""


@dataclass
class ParentCandidate:
    """Кандидат-родитель для разрешения конфликтов внутри одной глубины.

    Args:
        segment_id: Идентификатор родительского сегмента.
        segment_key: Человекочитаемое описание родителя.
        segment_level: Уровень группировки родителя.
        slice_depth: Глубина родителя.
        parent_atoms: Атомы, которые покрывает родитель.
        children: Текущие активные блоки, полностью лежащие внутри родителя.
        metrics: Метрики родителя, рассчитанные по текущим детям.
        action: Решение классификатора для родителя.
        reason: Причина решения классификатора.
        action_priority: Приоритет действия из классификатора.
        score: Численная оценка аналитической полезности кандидата.
        decision_type: Тип решения: primary_absorb, dominant_guard, residual_absorb или excluded.

    Returns:
        Экземпляр ParentCandidate.

    Raises:
        ValueError: Не выбрасывается автоматически.

    Examples:
        >>> # candidate = ParentCandidate(...)
    """

    segment_id: str
    segment_key: str
    segment_level: str
    slice_depth: int
    parent_atoms: frozenset[str]
    children: List[ActiveBlock]
    metrics: Dict[str, float | str]
    action: str
    reason: str
    action_priority: int
    score: float
    decision_type: str


EXACT_CONFLICT_COMPONENT_LIMIT = 18


def _is_missing(value: object) -> bool:
    """Проверить, является ли значение пустым для измерения.

    Args:
        value: Проверяемое значение.

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
    """Нормализовать значение признака.

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


def infer_dimension_columns(df: pd.DataFrame, explicit_dims: Optional[Sequence[str]] = None) -> List[str]:
    """Определить признаки для алгоритма.

    Args:
        df: Входная таблица.
        explicit_dims: Явно заданные признаки. Если переданы, используются они.

    Returns:
        Список признаков в стабильном порядке.

    Raises:
        ValueError: Если явно заданный признак отсутствует в таблице.

    Examples:
        >>> infer_dimension_columns(pd.DataFrame(columns=['period', 'gmv', 'geo']))
        ['geo']
    """

    if explicit_dims:
        missing = [c for c in explicit_dims if c not in df.columns]
        if missing:
            raise ValueError(f"В таблице отсутствуют явно заданные признаки: {missing}")
        return list(explicit_dims)
    return [c for c in df.columns if c not in TECH_COLUMNS]


def segment_id_from_row(row: pd.Series, dim_cols: Sequence[str]) -> str:
    """Создать технический идентификатор сегмента.

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
    for c in dim_cols:
        value = normalize_dim_value(row.get(c))
        parts.append(value if value is not None else "∅")
    return "|".join(parts)


def build_segment_key_and_level(row: pd.Series, dim_cols: Sequence[str]) -> Tuple[str, str, int]:
    """Построить человекочитаемый ключ и уровень сегмента.

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
    for c in dim_cols:
        value = normalize_dim_value(row.get(c))
        if value is not None:
            used.append(c)
            key_parts.append(f"{c}={value}")
    if not used:
        return "ИТОГО", "ИТОГО", 0
    return " × ".join(key_parts), " × ".join(used), len(used)


def load_grouped_table(
    input_path: str | Path,
    sheet_name: int | str = 0,
    period: Optional[str] = None,
    dim_cols: Optional[Sequence[str]] = None,
    previous_cal_date: Optional[int] = None,
    current_cal_date: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Загрузить уже сгруппированную таблицу и посчитать дельты по двум датам.

    Args:
        input_path: Путь к Excel- или CSV-файлу.
        sheet_name: Имя или номер листа Excel.
        period: Значение периода для фильтрации, например '1W'. Если None, фильтр не применяется.
        dim_cols: Явно заданные признаки. Если None, признаки определяются автоматически.
        previous_cal_date: Предыдущая дата. Если None, берётся минимальная cal_date.
        current_cal_date: Текущая дата. Если None, берётся максимальная cal_date.

    Returns:
        Кортеж: таблица срезов с дельтами и список признаков.

    Raises:
        ValueError: Если нет обязательных колонок или невозможно определить две даты.

    Examples:
        >>> # load_grouped_table('payoffline_pulse_hier_full_cuts_filtered.xlsx', period='1W')
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

    if period is not None and "period" in df.columns:
        df = df[df["period"].astype(str) == str(period)].copy()

    dims = infer_dimension_columns(df, dim_cols)
    for c in dims:
        df[c] = df[c].map(normalize_dim_value)

    dates = sorted(pd.Series(df["cal_date"].dropna().unique()).astype(int).tolist())
    if not dates:
        raise ValueError("Не найдены значения cal_date")
    prev_date = int(previous_cal_date) if previous_cal_date is not None else dates[0]
    curr_date = int(current_cal_date) if current_cal_date is not None else dates[-1]
    if prev_date == curr_date:
        raise ValueError("previous_cal_date и current_cal_date должны различаться")

    key_cols = ["slice_depth", *dims]
    prev = df[df["cal_date"].astype(int) == prev_date][key_cols + ["gmv"]].copy()
    curr = df[df["cal_date"].astype(int) == curr_date][key_cols + ["gmv"]].copy()
    prev = prev.rename(columns={"gmv": "gmv_previous"})
    curr = curr.rename(columns={"gmv": "gmv_current"})

    merged = curr.merge(prev, how="outer", on=key_cols)
    merged["gmv_previous"] = merged["gmv_previous"].fillna(0.0).astype(float)
    merged["gmv_current"] = merged["gmv_current"].fillna(0.0).astype(float)
    merged["delta_gmv"] = merged["gmv_current"] - merged["gmv_previous"]
    merged["abs_delta_gmv"] = merged["delta_gmv"].abs()
    merged["previous_cal_date"] = prev_date
    merged["current_cal_date"] = curr_date

    total_delta = float(merged.loc[merged["slice_depth"].astype(int) == 0, "delta_gmv"].sum())
    current_total = float(merged.loc[merged["slice_depth"].astype(int) == 0, "gmv_current"].sum())
    previous_total = float(merged.loc[merged["slice_depth"].astype(int) == 0, "gmv_previous"].sum())

    merged["share_in_total_delta"] = None if total_delta == 0 else merged["delta_gmv"] / total_delta
    merged["share_in_current_total_gmv"] = None if current_total == 0 else merged["gmv_current"] / current_total
    merged["share_in_previous_total_gmv"] = None if previous_total == 0 else merged["gmv_previous"] / previous_total

    merged["segment_id"] = merged.apply(lambda r: segment_id_from_row(r, dims), axis=1)
    key_level_depth = merged.apply(lambda r: build_segment_key_and_level(r, dims), axis=1)
    merged["segment_key"] = [x[0] for x in key_level_depth]
    merged["segment_level"] = [x[1] for x in key_level_depth]
    # Берём slice_depth из данных, но если она пустая, восстанавливаем по числу заполненных признаков.
    recovered_depth = pd.Series([x[2] for x in key_level_depth], index=merged.index)
    merged["slice_depth"] = merged["slice_depth"].where(merged["slice_depth"].notna(), recovered_depth).astype(int)

    ordered_cols = [
        "segment_id",
        "segment_key",
        "segment_level",
        "slice_depth",
        *dims,
        "gmv_previous",
        "gmv_current",
        "delta_gmv",
        "abs_delta_gmv",
        "share_in_total_delta",
        "share_in_previous_total_gmv",
        "share_in_current_total_gmv",
        "previous_cal_date",
        "current_cal_date",
    ]
    return merged[ordered_cols].sort_values(["slice_depth", "segment_key"]).reset_index(drop=True), dims


def candidate_covers_atomic(candidate: pd.Series, atomic: pd.Series, dim_cols: Sequence[str]) -> bool:
    """Проверить, покрывает ли кандидат атомарный сегмент.

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

    for c in dim_cols:
        value = normalize_dim_value(candidate.get(c))
        if value is not None and value != normalize_dim_value(atomic.get(c)):
            return False
    return True


def build_candidate_atomic_sets(slice_df: pd.DataFrame, dim_cols: Sequence[str]) -> Dict[str, frozenset[str]]:
    """Построить покрытие каждого сегмента атомами.

    Args:
        slice_df: Таблица срезов с дельтами.
        dim_cols: Список признаков.

    Returns:
        Словарь segment_id -> множество atomic_id.

    Raises:
        ValueError: Если атомарный уровень пустой.

    Examples:
        >>> # build_candidate_atomic_sets(slice_df, ['geo', 'products', 'channel'])
    """

    max_depth = len(dim_cols)
    atomic_df = slice_df[slice_df["slice_depth"].astype(int) == max_depth].copy()
    if atomic_df.empty:
        raise ValueError("Атомарный уровень пустой. Для алгоритма нужен максимальный уровень по всем признакам.")

    coverage: Dict[str, frozenset[str]] = {}
    for _, cand in slice_df.iterrows():
        atoms = []
        for _, atom in atomic_df.iterrows():
            if candidate_covers_atomic(cand, atom, dim_cols):
                atoms.append(str(atom["segment_id"]))
        coverage[str(cand["segment_id"])] = frozenset(atoms)
    return coverage


def calculate_metrics(children: Iterable[ActiveBlock]) -> Dict[str, float | str]:
    """Посчитать метрики родителя по текущим активным детям.

    Args:
        children: Активные дети, полностью покрытые родителем.

    Returns:
        Словарь метрик: чистое изменение, внутреннее движение, чистая доля и прочее.

    Raises:
        ValueError: Если список детей пуст.

    Examples:
        >>> calculate_metrics([ActiveBlock('a','a','x',1,10,frozenset({'a'}),'атом',''), ActiveBlock('b','b','x',1,-5,frozenset({'b'}),'атом','')])['covered_net_delta']
        5.0
    """

    child_list = list(children)
    if not child_list:
        raise ValueError("Нельзя считать метрики без детей")

    deltas = [float(ch.delta_gmv) for ch in child_list]
    net_delta = float(sum(deltas))
    gross_movement = float(sum(abs(x) for x in deltas))
    positive_movement = float(sum(x for x in deltas if x > 0))
    negative_movement = float(sum(x for x in deltas if x < 0))
    dominant = max(child_list, key=lambda ch: abs(ch.delta_gmv))
    dominant_abs = float(abs(dominant.delta_gmv))

    if gross_movement == 0:
        net_share = 1.0
        cancellation_ratio = 0.0
        concentration_share = 0.0
        same_direction_share = 1.0
    else:
        net_share = abs(net_delta) / gross_movement
        cancellation_ratio = 1.0 - net_share
        concentration_share = dominant_abs / gross_movement
        same_direction_share = max(positive_movement, abs(negative_movement)) / gross_movement

    return {
        "covered_net_delta": net_delta,
        "gross_movement": gross_movement,
        "positive_movement": positive_movement,
        "negative_movement": negative_movement,
        "net_share": net_share,
        "cancellation_ratio": cancellation_ratio,
        "concentration_share": concentration_share,
        "same_direction_share": same_direction_share,
        "dominant_child_id": dominant.block_id,
        "dominant_child_key": dominant.block_key,
        "dominant_child_abs_delta": dominant_abs,
        "child_count": float(len(child_list)),
    }


def classify_parent(metrics: Dict[str, float | str], thresholds: Thresholds) -> Tuple[str, str, int]:
    """Классифицировать родителя по обновлённым правилам v2.

    Args:
        metrics: Метрики родителя по текущим активным детям.
        thresholds: Пороги алгоритма.

    Returns:
        Кортеж: действие, причина, числовой приоритет действия для сортировки.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> classify_parent({'covered_net_delta': 10, 'gross_movement': 10, 'net_share': 1, 'same_direction_share': 1, 'concentration_share': 0.5, 'dominant_child_abs_delta': 5}, Thresholds())[0]
        'остаток_малое_движение'
    """

    net_delta = float(metrics["covered_net_delta"])
    gross = float(metrics["gross_movement"])
    net_share = float(metrics["net_share"])
    same_direction = float(metrics["same_direction_share"])
    concentration = float(metrics["concentration_share"])
    dominant_abs = float(metrics["dominant_child_abs_delta"])

    if abs(net_delta) < thresholds.min_segment_delta and gross < thresholds.min_gross_movement:
        return "остаток_малое_движение", "чистый вклад и внутреннее движение ниже порогов", 3

    if net_share >= thresholds.net_share_threshold and same_direction >= thresholds.same_direction_threshold and concentration < thresholds.concentration_threshold:
        return "основной_родитель", "однонаправленное движение без доминирующего ребёнка", 1

    if concentration >= thresholds.concentration_threshold and dominant_abs >= thresholds.min_child_delta:
        return "доминирующий_ребёнок", "почти весь эффект сосредоточен в одном ребёнке", 2

    if net_share < thresholds.net_share_threshold and gross >= thresholds.min_gross_movement:
        return "блок_компенсации", "сильное внутреннее движение с взаимной компенсацией", 1

    return "остаток_неясный_сигнал", "сегмент не прошёл правила основного драйвера, компенсации или доминирования", 4


def make_parent_active_block(parent: pd.Series, metrics: Dict[str, float | str], atomic_ids: frozenset[str], output_block: str, reason: str) -> ActiveBlock:
    """Создать активный блок на основе родителя.

    Args:
        parent: Строка родительского сегмента.
        metrics: Метрики родителя.
        atomic_ids: Покрываемые атомы.
        output_block: Тип итогового блока.
        reason: Причина решения.

    Returns:
        Активный блок.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # make_parent_active_block(parent, metrics, frozenset({'a'}), 'основной драйвер', 'пример')
    """

    return ActiveBlock(
        block_id=str(parent["segment_id"]),
        block_key=str(parent["segment_key"]),
        block_level=str(parent["segment_level"]),
        slice_depth=int(parent["slice_depth"]),
        delta_gmv=float(metrics["covered_net_delta"]),
        atomic_ids=atomic_ids,
        output_block=output_block,
        reason=reason,
        source_parent_id=str(parent["segment_id"]),
    )


def candidate_decision_type(action: str) -> str:
    """Определить эксплуатационный тип решения по действию классификатора.

    Args:
        action: Действие, возвращённое classify_parent.

    Returns:
        Тип решения для конфликтного выбора.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> candidate_decision_type('основной_родитель')
        'primary_absorb'
    """

    if action in {"основной_родитель", "блок_компенсации"}:
        return "primary_absorb"
    if action == "доминирующий_ребёнок":
        return "dominant_guard"
    if action == "остаток_малое_движение":
        return "residual_absorb"
    return "excluded"


def calculate_candidate_score(action: str, metrics: Dict[str, float | str]) -> float:
    """Оценить аналитическую полезность родителя внутри конфликтной группы.

    Args:
        action: Действие, возвращённое classify_parent.
        metrics: Метрики родителя по текущим активным детям.

    Returns:
        Численный score. Чем выше значение, тем полезнее кандидат.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> calculate_candidate_score('остаток_неясный_сигнал', {'covered_net_delta': 0, 'gross_movement': 0, 'net_share': 0, 'same_direction_share': 0, 'concentration_share': 0, 'cancellation_ratio': 0, 'dominant_child_abs_delta': 0})
        -1.0
    """

    net_abs = abs(float(metrics["covered_net_delta"]))
    gross = float(metrics["gross_movement"])
    net_share = float(metrics["net_share"])
    same_direction = float(metrics["same_direction_share"])
    concentration = float(metrics["concentration_share"])
    cancellation = float(metrics["cancellation_ratio"])
    dominant_abs = float(metrics["dominant_child_abs_delta"])

    if action == "основной_родитель":
        # return net_abs * net_share * same_direction * (1.0 - 0.35 * concentration)
        return net_abs * net_share * (1.0 - 0.35 * concentration)
    if action == "блок_компенсации":
        return gross * cancellation * (1.0 - 0.20 * concentration)
    if action == "доминирующий_ребёнок":
        return dominant_abs * concentration
    if action == "остаток_малое_движение":
        return min(net_abs, gross) * 0.01
    return -1.0


def candidate_rank(candidate: ParentCandidate) -> Tuple:
    """Построить детерминированный ключ сортировки кандидатов.

    Args:
        candidate: Кандидат-родитель.

    Returns:
        Кортеж сортировки. Чем меньше кортеж, тем сильнее кандидат.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # sorted(candidates, key=candidate_rank)
    """

    return (
        -float(candidate.score),
        -abs(float(candidate.metrics["covered_net_delta"])),
        -float(candidate.metrics["gross_movement"]),
        float(candidate.metrics["concentration_share"]),
        -len(candidate.parent_atoms),
        candidate.segment_key,
        candidate.segment_id,
    )


def candidate_diag_row(candidate: ParentCandidate) -> Dict[str, object]:
    """Преобразовать кандидата в строку диагностики.

    Args:
        candidate: Кандидат-родитель.

    Returns:
        Словарь с публичными диагностическими полями.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # row = candidate_diag_row(candidate)
    """

    return {
        "segment_id": candidate.segment_id,
        "segment_key": candidate.segment_key,
        "segment_level": candidate.segment_level,
        "slice_depth": candidate.slice_depth,
        "action": candidate.action,
        "reason": candidate.reason,
        "action_priority": candidate.action_priority,
        "useful_score": candidate.score,
        "candidate_score": candidate.score,
        "decision_type": candidate.decision_type,
        "covered_atomic_count": len(candidate.parent_atoms),
        "active_child_count": len(candidate.children),
        "covered_net_delta": candidate.metrics["covered_net_delta"],
        "gross_movement": candidate.metrics["gross_movement"],
        "positive_movement": candidate.metrics["positive_movement"],
        "negative_movement": candidate.metrics["negative_movement"],
        "net_share": candidate.metrics["net_share"],
        "cancellation_ratio": candidate.metrics["cancellation_ratio"],
        "concentration_share": candidate.metrics["concentration_share"],
        "same_direction_share": candidate.metrics["same_direction_share"],
        "dominant_child_id": candidate.metrics["dominant_child_id"],
        "dominant_child_key": candidate.metrics["dominant_child_key"],
        "dominant_child_abs_delta": candidate.metrics["dominant_child_abs_delta"],
    }


def build_parent_candidate(
    parent: pd.Series,
    parent_atoms: frozenset[str],
    children: List[ActiveBlock],
    thresholds: Thresholds,
) -> ParentCandidate:
    """Создать кандидата-родителя по текущему состоянию active.

    Args:
        parent: Строка родительского сегмента.
        parent_atoms: Атомы, покрываемые родителем.
        children: Активные дети, полностью лежащие внутри родителя.
        thresholds: Пороги алгоритма.

    Returns:
        Кандидат-родитель с action, metrics и score.

    Raises:
        ValueError: Если список детей пуст.

    Examples:
        >>> # candidate = build_parent_candidate(parent, atoms, children, Thresholds())
    """

    metrics = calculate_metrics(children)
    action, reason, action_priority = classify_parent(metrics, thresholds)
    score = calculate_candidate_score(action, metrics)
    return ParentCandidate(
        segment_id=str(parent["segment_id"]),
        segment_key=str(parent["segment_key"]),
        segment_level=str(parent["segment_level"]),
        slice_depth=int(parent["slice_depth"]),
        parent_atoms=parent_atoms,
        children=children,
        metrics=metrics,
        action=action,
        reason=reason,
        action_priority=action_priority,
        score=score,
        decision_type=candidate_decision_type(action),
    )


def get_active_children(active: Dict[str, ActiveBlock], parent_atoms: frozenset[str]) -> List[ActiveBlock]:
    """Найти активные блоки, полностью покрытые родителем.

    Args:
        active: Текущее активное разбиение.
        parent_atoms: Атомы родителя.

    Returns:
        Список активных блоков внутри родителя.
    """

    return [block for block in active.values() if block.atomic_ids.issubset(parent_atoms)]


def fully_covers_parent(children: Sequence[ActiveBlock], parent_atoms: frozenset[str]) -> bool:
    """Проверить, собирают ли дети родителя целиком без разрезания блоков.

    Args:
        children: Активные дети родителя.
        parent_atoms: Атомы родителя.

    Returns:
        True, если объединение детей равно покрытию родителя.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> fully_covers_parent([], frozenset())
        False
    """

    if not children:
        return False
    union_atoms = frozenset().union(*(child.atomic_ids for child in children))
    return union_atoms == parent_atoms


def build_conflict_components(candidates: Sequence[ParentCandidate]) -> List[List[ParentCandidate]]:
    """Построить компоненты конфликтов по пересечению атомов.

    Args:
        candidates: Кандидаты одной глубины.

    Returns:
        Список компонент. Внутри компоненты кандидаты связаны пересечениями атомов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> build_conflict_components([])
        []
    """

    if not candidates:
        return []

    adjacency: Dict[int, Set[int]] = {i: set() for i in range(len(candidates))}
    for i, left in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            if left.parent_atoms & right.parent_atoms:
                adjacency[i].add(j)
                adjacency[j].add(i)

    components: List[List[ParentCandidate]] = []
    seen: Set[int] = set()
    for start in range(len(candidates)):
        if start in seen:
            continue
        stack = [start]
        component_indexes: List[int] = []
        seen.add(start)
        while stack:
            idx = stack.pop()
            component_indexes.append(idx)
            for nxt in sorted(adjacency[idx]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append([candidates[i] for i in component_indexes])
    return components


def selection_objective(candidates: Sequence[ParentCandidate]) -> Tuple[float, float, float, int, int]:
    """Посчитать целевую функцию выбранного набора кандидатов.

    Args:
        candidates: Непересекающийся набор кандидатов.

    Returns:
        Кортеж для сравнения наборов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> selection_objective([])
        (0.0, 0.0, 0.0, 0, 0)
    """

    return (
        float(sum(c.score for c in candidates)),
        float(sum(abs(float(c.metrics["covered_net_delta"])) for c in candidates)),
        float(sum(float(c.metrics["gross_movement"]) for c in candidates)),
        int(sum(len(c.parent_atoms) for c in candidates)),
        -len(candidates),
    )


def better_selection(left: Sequence[ParentCandidate], right: Sequence[ParentCandidate]) -> bool:
    """Сравнить два непересекающихся набора кандидатов детерминированно.

    Args:
        left: Новый набор кандидатов.
        right: Текущий лучший набор кандидатов.

    Returns:
        True, если left лучше right.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> better_selection([], [])
        False
    """

    left_objective = selection_objective(left)
    right_objective = selection_objective(right)
    if left_objective != right_objective:
        return left_objective > right_objective
    left_signature = tuple((c.segment_key, c.segment_id) for c in sorted(left, key=candidate_rank))
    right_signature = tuple((c.segment_key, c.segment_id) for c in sorted(right, key=candidate_rank))
    return left_signature < right_signature


def select_best_component_candidates(component: Sequence[ParentCandidate]) -> List[ParentCandidate]:
    """Выбрать лучший непересекающийся набор внутри конфликтной компоненты.

    Args:
        component: Кандидаты одной конфликтной компоненты.

    Returns:
        Список выбранных кандидатов.
    """

    ordered = sorted(component, key=candidate_rank)
    if len(ordered) > EXACT_CONFLICT_COMPONENT_LIMIT:
        selected: List[ParentCandidate] = []
        used_atoms: Set[str] = set()
        for candidate in ordered:
            if not (candidate.parent_atoms & used_atoms):
                selected.append(candidate)
                used_atoms.update(candidate.parent_atoms)
        return selected

    best: List[ParentCandidate] = []

    def search(index: int, selected: List[ParentCandidate], used_atoms: Set[str]) -> None:
        nonlocal best
        if index >= len(ordered):
            if better_selection(selected, best):
                best = list(selected)
            return

        candidate = ordered[index]
        if not (candidate.parent_atoms & used_atoms):
            selected.append(candidate)
            search(index + 1, selected, used_atoms | set(candidate.parent_atoms))
            selected.pop()
        search(index + 1, selected, used_atoms)

    search(0, [], set())
    return sorted(best, key=candidate_rank)


def resolve_primary_conflicts(candidates: Sequence[ParentCandidate]) -> Tuple[List[ParentCandidate], List[Dict[str, object]]]:
    """Разрешить конфликты между поглощающими родителями одной глубины.

    Args:
        candidates: Primary-кандидаты: основной родитель или блок компенсации.

    Returns:
        Кортеж: выбранные кандидаты и строки отклонений для журнала.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> resolve_primary_conflicts([])
        ([], [])
    """

    selected: List[ParentCandidate] = []
    rejected_rows: List[Dict[str, object]] = []
    components = build_conflict_components(candidates)

    for group_idx, component in enumerate(components, start=1):
        winners = select_best_component_candidates(component)
        selected.extend(winners)
        winner_ids = {winner.segment_id for winner in winners}
        for candidate in component:
            if candidate.segment_id in winner_ids:
                continue
            overlapping_winners = [winner for winner in winners if winner.parent_atoms & candidate.parent_atoms]
            winner = sorted(overlapping_winners or winners, key=candidate_rank)[0] if winners else None
            rejected_rows.append({
                **candidate_diag_row(candidate),
                "conflict_group_id": group_idx,
                "final_decision": "исключён из-за более сильного конфликтующего родителя",
                "final_reason": "проиграл конфликт по score внутри текущей глубины",
                "winner_segment_id": "" if winner is None else winner.segment_id,
                "winner_key": "" if winner is None else winner.segment_key,
                "winner_score": None if winner is None else winner.score,
                "overlap_atomic_count": 0 if winner is None else len(candidate.parent_atoms & winner.parent_atoms),
            })

    return sorted(selected, key=candidate_rank), rejected_rows


def choose_sort_key(row: Dict[str, float | str]) -> Tuple:
    """Построить универсальный ключ сортировки родителей без ручного приоритета уровней.

    Args:
        row: Диагностическая строка родителя.

    Returns:
        Кортеж для сортировки. Чем меньше кортеж, тем раньше рассматривается родитель.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> choose_sort_key({'action_priority': 1, 'useful_score': 10, 'net_share': 1, 'concentration_share': 0.2, 'covered_atomic_count': 3, 'slice_depth': 2, 'segment_key': 'x'})[-1]
        'x'
    """

    return (
        int(row["action_priority"]),
        -float(row["useful_score"]),
        -float(row["net_share"]),
        float(row["concentration_share"]),
        -int(row["covered_atomic_count"]),
        int(row["slice_depth"]),
        str(row["segment_key"]),
    )


def run_bottom_up_algorithm(slice_df: pd.DataFrame, dim_cols: Sequence[str], thresholds: Thresholds) -> Dict[str, pd.DataFrame]:
    """Запустить итоговый bottom-up алгоритм v2.

    Args:
        slice_df: Таблица срезов с рассчитанными дельтами.
        dim_cols: Список признаков.
        thresholds: Пороги алгоритма.

    Returns:
        Словарь таблиц результата: диагностика, журнал решений, итоговое разбиение,
        детализация детей, исключённые сегменты и контроль суммы.

    Raises:
        ValueError: Если атомарный уровень пустой.

    Examples:
        >>> # result = run_bottom_up_algorithm(slice_df, ['geo', 'products', 'is_terminal_or_cpqr'], Thresholds())
    """

    max_depth = len(dim_cols)
    coverage = build_candidate_atomic_sets(slice_df, dim_cols)
    segment_lookup = {str(r["segment_id"]): r for _, r in slice_df.iterrows()}
    atomic_df = slice_df[slice_df["slice_depth"].astype(int) == max_depth].copy()

    active: Dict[str, ActiveBlock] = {}
    for _, atom in atomic_df.iterrows():
        sid = str(atom["segment_id"])
        active[sid] = ActiveBlock(
            block_id=sid,
            block_key=str(atom["segment_key"]),
            block_level=str(atom["segment_level"]),
            slice_depth=int(atom["slice_depth"]),
            delta_gmv=float(atom["delta_gmv"]),
            atomic_ids=frozenset([sid]),
            output_block="атомарный кандидат",
            reason="стартовое атомарное разбиение",
            source_parent_id=sid,
        )

    parent_diag_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []

    for depth in range(max_depth - 1, 0, -1):
        parent_rows = slice_df[slice_df["slice_depth"].astype(int) == depth].copy()
        candidates_for_depth: List[ParentCandidate] = []

        for _, parent in parent_rows.iterrows():
            parent_id = str(parent["segment_id"])
            parent_atoms = coverage.get(parent_id, frozenset())
            children = get_active_children(active, parent_atoms)
            if not children:
                continue
            if not fully_covers_parent(children, parent_atoms):
                row = {
                    "segment_id": parent_id,
                    "segment_key": parent["segment_key"],
                    "segment_level": parent["segment_level"],
                    "slice_depth": depth,
                    "action": "исключить_частичное_пересечение",
                    "reason": "родитель частично пересекается с уже выбранными активными блоками",
                    "final_decision": "исключён",
                    "final_reason": "частичное пересечение до выбора кандидатов текущей глубины",
                    "covered_atomic_count": len(parent_atoms),
                    "active_child_count": len(children),
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            candidate = build_parent_candidate(parent, parent_atoms, children, thresholds)
            candidates_for_depth.append(candidate)
            parent_diag_rows.append(candidate_diag_row(candidate))

        primary_candidates = [candidate for candidate in candidates_for_depth if candidate.decision_type == "primary_absorb"]
        selected_primary, rejected_primary_rows = resolve_primary_conflicts(primary_candidates)
        selected_primary_ids = {candidate.segment_id for candidate in selected_primary}
        blocked_atoms: Set[str] = set()

        for row in rejected_primary_rows:
            excluded_rows.append(row)
            decision_rows.append(row)

        for candidate in selected_primary:
            parent_id = candidate.segment_id
            parent = segment_lookup[parent_id]
            parent_atoms = coverage.get(parent_id, frozenset())
            children = get_active_children(active, parent_atoms)
            if not fully_covers_parent(children, parent_atoms):
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "кандидат устарел после выбора других родителей",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            metrics = calculate_metrics(children)
            action, reason, _ = classify_parent(metrics, thresholds)
            if action != candidate.action:
                candidate = build_parent_candidate(parent, parent_atoms, children, thresholds)
                metrics = candidate.metrics
                action = candidate.action
                reason = candidate.reason
            if action not in {"основной_родитель", "блок_компенсации"}:
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "после пересчёта кандидат перестал быть поглощающим родителем",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            output_block = "основной драйвер" if action == "основной_родитель" else "блок взаимной компенсации"
            detail_reason = "поглощён основным родителем" if action == "основной_родитель" else "детализация блока взаимной компенсации"
            final_decision = "поглощён как основной драйвер" if action == "основной_родитель" else "поглощён как блок взаимной компенсации"
            for child in children:
                active.pop(child.block_id, None)
                detail_rows.append(_detail_row(parent, child, detail_reason))
            active[parent_id] = make_parent_active_block(parent, metrics, parent_atoms, output_block, reason)
            blocked_atoms.update(parent_atoms)
            decision_rows.append({**candidate_diag_row(candidate), "final_decision": final_decision, "final_reason": reason})

        dominant_candidates = sorted(
            [candidate for candidate in candidates_for_depth if candidate.decision_type == "dominant_guard" and candidate.segment_id not in selected_primary_ids],
            key=candidate_rank,
        )
        protected_atoms: Set[str] = set()
        for candidate in dominant_candidates:
            if candidate.parent_atoms & blocked_atoms:
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "зона уже поглощена более сильным родителем текущей глубины",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            parent_id = candidate.segment_id
            parent = segment_lookup[parent_id]
            parent_atoms = coverage.get(parent_id, frozenset())
            children = get_active_children(active, parent_atoms)
            if not fully_covers_parent(children, parent_atoms):
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "частичное пересечение после выбора поглощающих родителей",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            metrics = calculate_metrics(children)
            action, reason, _ = classify_parent(metrics, thresholds)
            if action != "доминирующий_ребёнок":
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "не выбран",
                    "final_reason": "после пересчёта доминирующий ребёнок не подтверждён",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            small_children = [ch for ch in children if abs(ch.delta_gmv) < thresholds.min_child_delta]
            if small_children:
                small_atoms = frozenset().union(*(ch.atomic_ids for ch in small_children))
                small_delta = float(sum(ch.delta_gmv for ch in small_children))
                other_id = f"прочее|{parent_id}"
                other_key = f"прочее внутри: {parent['segment_key']}"
                for child in small_children:
                    active.pop(child.block_id, None)
                    detail_rows.append(_detail_row(parent, child, "свёрнут в прочее внутри родителя"))
                active[other_id] = ActiveBlock(
                    block_id=other_id,
                    block_key=other_key,
                    block_level=str(parent["segment_level"]),
                    slice_depth=int(parent["slice_depth"]),
                    delta_gmv=small_delta,
                    atomic_ids=small_atoms,
                    output_block="прочее внутри родителя",
                    reason="мелкие соседи доминирующего ребёнка свёрнуты",
                    source_parent_id=parent_id,
                )
            protected_atoms.update(parent_atoms)
            excluded_rows.append({
                **candidate_diag_row(candidate),
                "reason": f"родитель не выбран: доминирует ребёнок {metrics['dominant_child_key']}",
            })
            decision_rows.append({
                **candidate_diag_row(candidate),
                "final_decision": "родитель не выбран, оставлен доминирующий ребёнок",
                "final_reason": reason,
            })

        residual_candidates = sorted(
            [candidate for candidate in candidates_for_depth if candidate.decision_type == "residual_absorb" and candidate.segment_id not in selected_primary_ids],
            key=candidate_rank,
        )
        residual_selected, residual_rejected = resolve_primary_conflicts(residual_candidates)
        for row in residual_rejected:
            row["final_reason"] = "остаток проиграл конфликт другому остатку текущей глубины"
            excluded_rows.append(row)
            decision_rows.append(row)

        for candidate in residual_selected:
            if candidate.parent_atoms & (blocked_atoms | protected_atoms):
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "остаток пересекается с выбранным родителем или доминирующим ребёнком",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            parent_id = candidate.segment_id
            parent = segment_lookup[parent_id]
            parent_atoms = coverage.get(parent_id, frozenset())
            children = get_active_children(active, parent_atoms)
            if not fully_covers_parent(children, parent_atoms):
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "исключён",
                    "final_reason": "остаток стал частичным пересечением после предыдущих решений",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            metrics = calculate_metrics(children)
            action, reason, _ = classify_parent(metrics, thresholds)
            if action != "остаток_малое_движение":
                row = {
                    **candidate_diag_row(candidate),
                    "final_decision": "не выбран",
                    "final_reason": "после пересчёта остаток не подтверждён",
                }
                excluded_rows.append(row)
                decision_rows.append(row)
                continue

            for child in children:
                active.pop(child.block_id, None)
                detail_rows.append(_detail_row(parent, child, "свёрнут в остаток малого движения"))
            active[parent_id] = make_parent_active_block(parent, metrics, parent_atoms, "остаток / прочее", reason)
            blocked_atoms.update(parent_atoms)
            decision_rows.append({**candidate_diag_row(candidate), "final_decision": "свёрнут в остаток", "final_reason": reason})

        for candidate in candidates_for_depth:
            if candidate.decision_type != "excluded":
                continue
            row = {
                **candidate_diag_row(candidate),
                "final_decision": "не выбран",
                "final_reason": candidate.reason,
            }
            excluded_rows.append(row)
            decision_rows.append(row)

    final_df = pd.DataFrame([{
        "block_id": block.block_id,
        "block_key": block.block_key,
        "block_level": block.block_level,
        "slice_depth": block.slice_depth,
        "output_block": block.output_block,
        "delta_gmv": block.delta_gmv,
        "abs_delta_gmv": abs(block.delta_gmv),
        "covered_atomic_count": len(block.atomic_ids),
        "reason": block.reason,
        "source_parent_id": block.source_parent_id,
    } for block in active.values()]).sort_values(["output_block", "abs_delta_gmv"], ascending=[True, False])

    total_delta = float(atomic_df["delta_gmv"].sum())
    final_delta = float(final_df["delta_gmv"].sum()) if not final_df.empty else 0.0
    assignment_rows = []
    for block in active.values():
        for atomic_id in block.atomic_ids:
            assignment_rows.append({"block_id": block.block_id, "atomic_id": atomic_id})
    assignment_df = pd.DataFrame(assignment_rows)
    duplicated_atoms = 0 if assignment_df.empty else int((assignment_df.groupby("atomic_id")["block_id"].nunique() > 1).sum())

    control_df = pd.DataFrame([{
        "total_delta_from_atomic": total_delta,
        "final_partition_delta": final_delta,
        "difference": final_delta - total_delta,
        "atomic_count": int(len(atomic_df)),
        "assigned_atomic_count": int(assignment_df["atomic_id"].nunique()) if not assignment_df.empty else 0,
        "double_count_violation_count": duplicated_atoms,
    }])

    return {
        "parent_diagnostics": pd.DataFrame(parent_diag_rows),
        "decision_log": pd.DataFrame(decision_rows), #весь журнал решений
        "final_partition": final_df,
        "detail_children": pd.DataFrame(detail_rows),
        "excluded_segments": pd.DataFrame(excluded_rows), #только отклонённые / невыбранные кандидаты
        "atomic_assignment": assignment_df,
        "control": control_df,
    }


def _public_diag(row: Dict[str, object]) -> Dict[str, object]:
    """Оставить только публичные поля диагностики без вложенных объектов.

    Args:
        row: Диагностическая строка с техническими объектами.

    Returns:
        Словарь без множеств атомов и списков детей.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _public_diag({'a': 1, 'children': [1], 'parent_atoms': {1}})
        {'a': 1}
    """

    return {k: v for k, v in row.items() if k not in {"children", "parent_atoms"}}


def _detail_row(parent: pd.Series, child: ActiveBlock, reason: str) -> Dict[str, object]:
    """Создать строку детализации ребёнка.

    Args:
        parent: Родительский сегмент.
        child: Активный ребёнок.
        reason: Причина переноса ребёнка в детализацию.

    Returns:
        Словарь для таблицы детализации.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> # _detail_row(parent, child, 'пример')
    """

    return {
        "parent_id": str(parent["segment_id"]),
        "parent_key": str(parent["segment_key"]),
        "parent_level": str(parent["segment_level"]),
        "child_id": child.block_id,
        "child_key": child.block_key,
        "child_level": child.block_level,
        "child_output_block_before": child.output_block,
        "child_delta_gmv": child.delta_gmv,
        "covered_atomic_count": len(child.atomic_ids),
        "reason": reason,
    }


def _format_rub(value: float) -> str:
    """Отформатировать вклад GMV в рублях без дробной части.

    Args:
        value: Значение GMV в рублях.

    Returns:
        Строка с суммой в рублях и разделителем групп разрядов.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _format_rub(25894762.45)
        '+25 894 762 руб.'
    """

    sign = "+" if value >= 0 else "-"
    amount = f"{int(abs(value)):,}".replace(",", " ")
    return f"{sign}{amount} руб."


def _short_segment_name(segment_key: object) -> str:
    """Сократить технический ключ сегмента до менеджерского названия.

    Args:
        segment_key: Технический ключ сегмента.

    Returns:
        Короткое название сегмента.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _short_segment_name("products=FULLPAYMENT x is_terminal_or_cpqr=QR")
        'FULLPAYMENT x QR'
    """

    parts = []
    for part in str(segment_key).split(" x "):
        if "=" in part:
            parts.append(part.split("=", 1)[1])
        else:
            parts.append(part)
    return " x ".join(parts)


def build_manager_summary(result: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Сформировать менеджерский вывод по итоговому изменению GMV.

    Args:
        result: Словарь таблиц результата алгоритма.

    Returns:
        DataFrame для отдельного листа Excel с менеджерским выводом.

    Raises:
        KeyError: Если в result отсутствуют обязательные таблицы.

    Examples:
        >>> # summary = build_manager_summary(result)
    """

    final_df = result["final_partition"].copy()
    detail_df = result["detail_children"].copy()
    control = result["control"].iloc[0].to_dict()
    total_delta = float(control["total_delta_from_atomic"])

    rows: List[Dict[str, object]] = [
        {
            "раздел": "Заголовок",
            "блок": "",
            "сегмент": "Менеджерский вывод по изменению GMV WoW",
            "вклад": _format_rub(total_delta),
            "интерпретация": f"За неделю GMV изменился на {_format_rub(total_delta)}.",
        }
    ]

    if final_df.empty:
        return pd.DataFrame(rows)

    main_driver = final_df.sort_values("abs_delta_gmv", ascending=False).iloc[0]
    rows.append(
        {
            "раздел": "Краткий вывод",
            "блок": "Главный фактор",
            "сегмент": _short_segment_name(main_driver["block_key"]),
            "вклад": _format_rub(float(main_driver["delta_gmv"])),
            "интерпретация": (
                "Главный вклад в изменение GMV даёт сегмент "
                f"{_short_segment_name(main_driver['block_key'])}."
            ),
        }
    )

    for _, block in final_df.sort_values("abs_delta_gmv", ascending=False).iterrows():
        output_block = str(block["output_block"])
        delta = float(block["delta_gmv"])
        segment = _short_segment_name(block["block_key"])

        if output_block == "основной драйвер":
            manager_block = "Основной драйвер"
            interpretation = "Главная самостоятельная причина изменения GMV."
        elif output_block == "блок взаимной компенсации":
            manager_block = "Компенсационный блок"
            children = detail_df.loc[detail_df["parent_id"] == block["block_id"]].copy()
            if not children.empty:
                children["abs_child_delta"] = children["child_delta_gmv"].abs()
                top_children = children.sort_values("abs_child_delta", ascending=False).head(3)
                child_text = "; ".join(
                    f"{_short_segment_name(row['child_key'])}: {_format_rub(float(row['child_delta_gmv']))}"
                    for _, row in top_children.iterrows()
                )
                interpretation = (
                    "Внутри есть разнонаправленные движения. "
                    f"Крупнейшие дети: {child_text}."
                )
            else:
                interpretation = "Внутри есть разнонаправленные движения; детализация детей отсутствует."
        elif delta < 0:
            manager_block = "Сдерживающий фактор"
            interpretation = "Сегмент снижает общий результат GMV."
        else:
            manager_block = "Остаток / прочее"
            interpretation = "Не формирует отдельную крупную бизнес-причину."

        rows.append(
            {
                "раздел": "Таблица факторов",
                "блок": manager_block,
                "сегмент": segment,
                "вклад": _format_rub(delta),
                "интерпретация": interpretation,
            }
        )

    rows.extend(
        [
            {
                "раздел": "Основной вывод",
                "блок": "",
                "сегмент": "",
                "вклад": "",
                "интерпретация": (
                    "Рост или снижение GMV следует читать через крупнейшие основные драйверы; "
                    "компенсационные блоки требуют отдельного просмотра детей."
                ),
            },
            {
                "раздел": "Что проверить дальше",
                "блок": "1",
                "сегмент": "",
                "вклад": "",
                "интерпретация": "За счёт чего изменился главный драйвер: регион, мерчанты, пользователи, средний чек или количество платежей.",
            },
            {
                "раздел": "Что проверить дальше",
                "блок": "2",
                "сегмент": "",
                "вклад": "",
                "интерпретация": "В компенсационных блоках отдельно посмотреть растущие и падающие дочерние сегменты.",
            },
            {
                "раздел": "Что проверить дальше",
                "блок": "3",
                "сегмент": "",
                "вклад": "",
                "интерпретация": "По отрицательным блокам проверить, является ли снижение разовым эффектом или устойчивым трендом.",
            },
        ]
    )
    return pd.DataFrame(rows)


def write_result_excel(result: Dict[str, pd.DataFrame], slice_df: pd.DataFrame, output_path: str | Path, thresholds: Thresholds, dim_cols: Sequence[str]) -> None:
    """Записать результат алгоритма в один Excel-файл.

    Args:
        result: Словарь таблиц результата.
        slice_df: Таблица входных срезов с дельтами.
        output_path: Путь к итоговому Excel-файлу.
        thresholds: Пороги алгоритма.
        dim_cols: Список признаков.

    Returns:
        None.

    Raises:
        OSError: Если файл невозможно записать.

    Examples:
        >>> # write_result_excel(result, slice_df, 'result.xlsx', Thresholds(), ['geo'])
    """

    output_path = Path(output_path)
    params = pd.DataFrame([
        {"параметр": "признаки", "значение": " × ".join(dim_cols)},
        {"параметр": "min_segment_delta", "значение": thresholds.min_segment_delta},
        {"параметр": "min_child_delta", "значение": thresholds.min_child_delta},
        {"параметр": "min_gross_movement", "значение": thresholds.min_gross_movement},
        {"параметр": "net_share_threshold", "значение": thresholds.net_share_threshold},
        {"параметр": "same_direction_threshold", "значение": thresholds.same_direction_threshold},
        {"параметр": "concentration_threshold", "значение": thresholds.concentration_threshold},
        {"параметр": "ручной приоритет уровней", "значение": "не используется"},
        {"параметр": "защита от переукрупнения", "значение": "не используется"},
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        params.to_excel(writer, sheet_name="00_Параметры", index=False)
        slice_df.to_excel(writer, sheet_name="01_Входные_срезы", index=False)
        result["parent_diagnostics"].to_excel(writer, sheet_name="02_Диагностика родителей", index=False)
        result["decision_log"].to_excel(writer, sheet_name="03_Журнал_решений", index=False)
        result["final_partition"].to_excel(writer, sheet_name="04_Итог", index=False)
        result["detail_children"].to_excel(writer, sheet_name="05_Детализация поглощения", index=False)
        result["excluded_segments"].to_excel(writer, sheet_name="06_Исключённые", index=False)
        result["atomic_assignment"].to_excel(writer, sheet_name="07_Назначение_атомов", index=False)
        result["control"].to_excel(writer, sheet_name="08_Контроль", index=False)
        build_manager_summary(result).to_excel(writer, sheet_name="09_Менеджерский_вывод", index=False)
    # "parent_diagnostics": pd.DataFrame(parent_diag_rows),
    #     "decision_log": pd.DataFrame(decision_rows), #весь журнал решений
    #     "final_partition": final_df,
    #     "detail_children": pd.DataFrame(detail_rows),
    #     "excluded_segments": pd.DataFrame(excluded_rows), #только отклонённые / невыбранные кандидаты
    #     "atomic_assignment": assignment_df,
    #     "control": control_df,
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            for col_cells in worksheet.columns:
                max_length = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                worksheet.column_dimensions[col_letter].width = min(max(max_length + 2, 10), 42)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        Нет аргументов.

    Returns:
        Объект argparse.Namespace с параметрами запуска.

    Raises:
        SystemExit: Если аргументы командной строки некорректны.

    Examples:
        >>> # args = parse_args()
    """

    parser = argparse.ArgumentParser(description="Итоговый алгоритм v2 для анализа вклада сегментов в изменение GMV.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Путь к входному .xlsx/.xls/.csv файлу с готовыми группировками.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь к итоговому .xlsx файлу.",
    )
    parser.add_argument("--sheet-name", default=0, help="Имя или номер листа Excel. По умолчанию первый лист.")
    parser.add_argument("--period", default=None, help="Период для фильтрации, например 1W.")
    parser.add_argument("--dims", nargs="*", default=None, help="Список признаков. Если не указан, признаки определяются автоматически.")
    parser.add_argument("--previous-cal-date", type=int, default=None, help="Предыдущая дата cal_date. Если не указана, берётся минимум.")
    parser.add_argument("--current-cal-date", type=int, default=None, help="Текущая дата cal_date. Если не указана, берётся максимум.")
    parser.add_argument("--min-segment-delta", type=float, default=Thresholds.min_segment_delta)
    parser.add_argument("--min-child-delta", type=float, default=Thresholds.min_child_delta)
    parser.add_argument("--min-gross-movement", type=float, default=Thresholds.min_gross_movement)
    parser.add_argument("--net-share-threshold", type=float, default=Thresholds.net_share_threshold)
    parser.add_argument("--same-direction-threshold", type=float, default=Thresholds.same_direction_threshold)
    parser.add_argument("--concentration-threshold", type=float, default=Thresholds.concentration_threshold)
    return parser.parse_args()


def main() -> None:
    """Запустить алгоритм из командной строки.

    Args:
        Нет аргументов.

    Returns:
        None.

    Raises:
        ValueError: При некорректной входной таблице.
        OSError: При ошибке чтения или записи файла.

    Examples:
        >>> # main()
    """

    args = parse_args()
    sheet_name: int | str
    try:
        sheet_name = int(args.sheet_name)
    except (TypeError, ValueError):
        sheet_name = args.sheet_name

    thresholds = Thresholds(
        min_segment_delta=args.min_segment_delta,
        min_child_delta=args.min_child_delta,
        min_gross_movement=args.min_gross_movement,
        net_share_threshold=args.net_share_threshold,
        same_direction_threshold=args.same_direction_threshold,
        concentration_threshold=args.concentration_threshold,
    )

    slice_df, dim_cols = load_grouped_table(
        input_path=args.input,
        sheet_name=sheet_name,
        period=args.period,
        dim_cols=args.dims,
        previous_cal_date=args.previous_cal_date,
        current_cal_date=args.current_cal_date,
    )
    result = run_bottom_up_algorithm(slice_df, dim_cols, thresholds)
    write_result_excel(result, slice_df, args.output, thresholds, dim_cols)

    control = result["control"].iloc[0].to_dict()
    print("Готово.")
    print(f"Признаки: {' x '.join(dim_cols)}")
    print(f"Итоговый файл: {args.output}")
    print(f"Сумма атомарных дельт: {control['total_delta_from_atomic']:.2f}")
    print(f"Сумма итогового разбиения: {control['final_partition_delta']:.2f}")
    print(f"Разница: {control['difference']:.2f}")
    print(f"Нарушения двойного счёта: {int(control['double_count_violation_count'])}")


if __name__ == "__main__":
    main()
