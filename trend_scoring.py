"""ADDED: Скоринг, hierarchy-корректировка и exact-отбор GMV-трендов.

Модуль работает только после завершения trend analysis. Он не меняет поиск
тренда, changepoint, slopes или бизнес-контракт :func:`evaluate_trend`.
Атомарные сегменты используются для расчёта движений и покрытия, но в
hierarchy-правилах и Set Packing участвуют только подтверждённые тренды.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .anomaly_scoring import (
    _select_best_disjoint_descendant_group,
    build_atomic_coverage,
    validate_hierarchy_reconciliation,
)
from .config import AnomalyThresholds
from .set_packing import search_anomal
from .trend_analysis import DECLINE, GROWTH, TrendThresholds


# ADDED: Trend score имеет собственную конфигурацию и не меняет anomaly-пороги.
@dataclass(frozen=True)
class TrendScoringConfig:
    """Настроить прозрачные корректировки и оптимизационный отбор трендов.

    Args:
        own_score_adjustment_max: Максимальный бонус или штраф к impact за
            качество направления.
        aggregation_bonus_lambda: Наклон hierarchy-множителя. Значение 0.3
            переносит anomaly-диапазон ``[0.85, 1.15]``.
        single_child_factor: Множитель родителя при подтверждённом доминировании.
        dominant_child_capture_threshold: Минимальная доля gross-движения
            родителя, объяснённая направленным движением потомка.
        dominant_child_score_margin: Запас, удерживающий score родителя ниже
            score доминирующего потомка.
        hierarchy_reconciliation_abs_tolerance: Абсолютный допуск сверки GMV
            родителя с атомарным слоем.
        max_hierarchy_descendants: Лимит физического перебора hierarchy-групп;
            выше применяется существующий exact Set Packing.
        set_packing_gap_tolerance: Допуск доказанного оптимума Set Packing.
        max_exact_fallback_size: Максимальный размер exact fallback без MILP.

    Returns:
        Неизменяемую конфигурацию trend scoring.

    Raises:
        ValueError: Если параметр нарушает числовой контракт.

    Examples:
        >>> TrendScoringConfig().dominant_child_capture_threshold
        0.85
    """

    own_score_adjustment_max: float = 0.10
    aggregation_bonus_lambda: float = 0.30
    single_child_factor: float = 0.85
    dominant_child_capture_threshold: float = 0.85
    dominant_child_score_margin: float = 0.02
    hierarchy_reconciliation_abs_tolerance: float = 0.01
    max_hierarchy_descendants: int = 25
    set_packing_gap_tolerance: float = 1e-9
    max_exact_fallback_size: int = 25

    def __post_init__(self) -> None:
        """Проверить параметры до расчёта.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            ValueError: Если значение не поддерживается.

        Examples:
            >>> TrendScoringConfig(aggregation_bonus_lambda=0.3).aggregation_bonus_lambda
            0.3
        """

        unit_interval = {
            "own_score_adjustment_max": self.own_score_adjustment_max,
            "single_child_factor": self.single_child_factor,
            "dominant_child_capture_threshold": (
                self.dominant_child_capture_threshold
            ),
        }
        for name, value in unit_interval.items():
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} должен находиться в диапазоне (0, 1]")
        if (
            not math.isfinite(float(self.aggregation_bonus_lambda))
            or not 0.0 <= float(self.aggregation_bonus_lambda) < 2.0
        ):
            raise ValueError(
                "aggregation_bonus_lambda должен находиться в диапазоне [0, 2)"
            )
        if (
            not math.isfinite(float(self.dominant_child_score_margin))
            or not 0.0 <= float(self.dominant_child_score_margin) < 1.0
        ):
            raise ValueError(
                "dominant_child_score_margin должен находиться в диапазоне [0, 1)"
            )
        if (
            not math.isfinite(float(self.hierarchy_reconciliation_abs_tolerance))
            or float(self.hierarchy_reconciliation_abs_tolerance) < 0.0
        ):
            raise ValueError(
                "hierarchy_reconciliation_abs_tolerance должен быть конечным "
                "неотрицательным числом"
            )
        integer_positive = {
            "max_hierarchy_descendants": self.max_hierarchy_descendants,
            "max_exact_fallback_size": self.max_exact_fallback_size,
        }
        for name, value in integer_positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{name} должен быть положительным целым числом")
        if (
            not math.isfinite(float(self.set_packing_gap_tolerance))
            or float(self.set_packing_gap_tolerance) < 0.0
        ):
            raise ValueError(
                "set_packing_gap_tolerance должен быть конечным "
                "неотрицательным числом"
            )


def _minimum_direction_quality(thresholds: TrendThresholds) -> float:
    """Рассчитать минимальное геометрическое качество направления.

    Args:
        thresholds: Действующие пороги подтверждения тренда.

    Returns:
        ``sqrt(min_count_share * min_movement_share)``.

    Raises:
        ValueError: Не выбрасывается для валидного ``TrendThresholds``.

    Examples:
        >>> _minimum_direction_quality(TrendThresholds())
        0.75
    """

    return math.sqrt(
        float(thresholds.min_direction_count_share)
        * float(thresholds.min_direction_movement_share)
    )


def _direction_adjustment_signal(
    direction_quality: float,
    minimum_quality: float,
) -> float:
    """Перевести качество подтверждённого направления в диапазон ``[-1, 1]``.

    Args:
        direction_quality: Геометрическое среднее двух direction shares.
        minimum_quality: Геометрическое среднее их действующих порогов.

    Returns:
        Линейно нормированный и ограниченный сигнал.

    Raises:
        ValueError: Если входы не конечны или выходят за ``[0, 1]``.

    Examples:
        >>> _direction_adjustment_signal(0.75, 0.75)
        -1.0
        >>> _direction_adjustment_signal(1.0, 0.75)
        1.0
    """

    for name, value in {
        "direction_quality": direction_quality,
        "minimum_quality": minimum_quality,
    }.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} должен быть конечным числом в [0, 1]")
    if math.isclose(float(minimum_quality), 1.0, rel_tol=0.0, abs_tol=1e-15):
        return 1.0 if float(direction_quality) >= 1.0 else -1.0
    position = (
        (float(direction_quality) - float(minimum_quality))
        / (1.0 - float(minimum_quality))
    )
    return float(min(1.0, max(-1.0, 2.0 * position - 1.0)))


def calculate_own_trend_scores(
    trend_summary: pd.DataFrame,
    current_total_gmv: float,
    thresholds: TrendThresholds,
    config: TrendScoringConfig,
) -> pd.DataFrame:
    """Рассчитать собственный score подтверждённых трендов без hierarchy.

    Формулы:
    ``impact = sqrt(abs(slope_abs)/total_gmv * abs(slope_relative))`` и
    ``own_trend_score = impact * (1 + max_adjustment * direction_signal)``.
    Длительность намеренно не входит в score: в текущей витрине она сильно
    связана с глубиной и создаёт структурный перекос против глубоких сегментов.

    Args:
        trend_summary: Готовый результат trend analysis, одна строка на сегмент.
        current_total_gmv: GMV Total-сегмента в последнем периоде.
        thresholds: Пороги, которыми подтверждался тренд.
        config: Параметры scoring.

    Returns:
        Копию summary с impact, direction quality и собственным score.

    Raises:
        ValueError: Если отсутствуют поля или Total GMV не положителен.

    Examples:
        >>> frame = pd.DataFrame([{'segment_id': 's', 'slice_depth': 1,
        ...     'current_trend_exists': True, 'trend_slope_abs': 4.0,
        ...     'trend_slope_relative': 0.04,
        ...     'direction_count_share': 1.0,
        ...     'direction_movement_share': 1.0}])
        >>> round(float(calculate_own_trend_scores(
        ...     frame, 100.0, TrendThresholds(), TrendScoringConfig()
        ... ).at[0, 'trend_impact']), 3)
        0.04
    """

    required = {
        "segment_id",
        "slice_depth",
        "current_trend_exists",
        "trend_slope_abs",
        "trend_slope_relative",
        "direction_count_share",
        "direction_movement_share",
    }
    missing = sorted(required - set(trend_summary.columns))
    if missing:
        raise ValueError(f"Для trend score не хватает колонок: {missing}")
    if not math.isfinite(float(current_total_gmv)) or float(current_total_gmv) <= 0.0:
        raise ValueError("current_total_gmv должен быть конечным положительным числом")

    result = trend_summary.copy()
    result["current_total_gmv"] = float(current_total_gmv)
    result["absolute_rate_share"] = math.nan
    result["relative_rate"] = math.nan
    result["trend_impact"] = math.nan
    result["direction_quality"] = math.nan
    result["minimum_direction_quality"] = _minimum_direction_quality(thresholds)
    result["direction_adjustment_signal"] = math.nan
    # ADDED: Явные поля раскрывают именно величину ``0.10 × signal``;
    # пользователю не нужно восстанавливать штраф/бонус из factor.
    result["own_adjustment_signal"] = math.nan
    result["own_score_adjustment"] = math.nan
    result["own_score_adjustment_factor"] = math.nan
    result["own_trend_score"] = math.nan
    # ADDED: Ограничение видно в выходе; duration не скрыт и не равен нулевому бонусу.
    result["trend_duration_score_status"] = "NOT_INCLUDED_BY_DESIGN"
    result["trend_eligible"] = (
        result["current_trend_exists"].fillna(False).astype(bool)
        & pd.to_numeric(result["slice_depth"], errors="coerce").gt(0)
    )

    confirmed_mask = result["current_trend_exists"].fillna(False).astype(bool)
    for index, row in result.loc[confirmed_mask].iterrows():
        slope_abs = pd.to_numeric(pd.Series([row["trend_slope_abs"]]), errors="coerce").iloc[0]
        slope_relative = pd.to_numeric(
            pd.Series([row["trend_slope_relative"]]), errors="coerce"
        ).iloc[0]
        count_share = pd.to_numeric(
            pd.Series([row["direction_count_share"]]), errors="coerce"
        ).iloc[0]
        movement_share = pd.to_numeric(
            pd.Series([row["direction_movement_share"]]), errors="coerce"
        ).iloc[0]
        values = [slope_abs, slope_relative, count_share, movement_share]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                f"Подтверждённый тренд {row['segment_id']} содержит "
                "нечисловые scoring-метрики"
            )
        if not 0.0 <= float(count_share) <= 1.0 or not 0.0 <= float(movement_share) <= 1.0:
            raise ValueError(
                f"Direction shares сегмента {row['segment_id']} должны быть в [0, 1]"
            )
        absolute_rate_share = abs(float(slope_abs)) / float(current_total_gmv)
        relative_rate = abs(float(slope_relative))
        impact = math.sqrt(absolute_rate_share * relative_rate)
        direction_quality = math.sqrt(float(count_share) * float(movement_share))
        minimum_quality = float(result.at[index, "minimum_direction_quality"])
        direction_signal = _direction_adjustment_signal(
            direction_quality,
            minimum_quality,
        )
        own_score_adjustment = (
            float(config.own_score_adjustment_max) * direction_signal
        )
        adjustment_factor = 1.0 + own_score_adjustment
        own_score = impact * adjustment_factor
        result.at[index, "absolute_rate_share"] = absolute_rate_share
        result.at[index, "relative_rate"] = relative_rate
        result.at[index, "trend_impact"] = impact
        result.at[index, "direction_quality"] = direction_quality
        result.at[index, "direction_adjustment_signal"] = direction_signal
        result.at[index, "own_adjustment_signal"] = direction_signal
        result.at[index, "own_score_adjustment"] = own_score_adjustment
        result.at[index, "own_score_adjustment_factor"] = adjustment_factor
        result.at[index, "own_trend_score"] = own_score

    return result


def _active_transition_dates(
    dates: Sequence[int],
    start_date: object,
    end_date: object,
) -> Tuple[int, ...]:
    """Вернуть правые даты переходов, полностью лежащих в активном окне.

    Args:
        dates: Полная упорядоченная календарная ось.
        start_date: Первая точка текущего тренда.
        end_date: Последняя точка текущего тренда.

    Returns:
        Даты ``t`` переходов ``t-1 -> t``; сама start-date не входит.

    Raises:
        ValueError: Если границы окна отсутствуют на календарной оси.

    Examples:
        >>> _active_transition_dates([0, 7, 14, 21], 7, 21)
        (14, 21)
    """

    ordered_dates = tuple(int(date) for date in dates)
    if len(ordered_dates) != len(set(ordered_dates)) or tuple(sorted(ordered_dates)) != ordered_dates:
        raise ValueError("dates должны быть уникальными и строго возрастающими")
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("У подтверждённого тренда отсутствуют границы активного окна")
    start = int(start_date)
    end = int(end_date)
    if start not in ordered_dates or end not in ordered_dates or start > end:
        raise ValueError(
            f"Некорректное активное окно [{start}, {end}] относительно dates"
        )
    return tuple(date for date in ordered_dates if start < date <= end)


def _atomic_window_movements(
    atomic_deltas: pd.DataFrame,
    atom_ids: Sequence[str],
    transition_dates: Sequence[int],
) -> Tuple[float, float]:
    """Рассчитать net и gross наблюдаемые движения атомов на переходах.

    Args:
        atomic_deltas: Матрица недельных изменений ``segment_id x cal_date``.
        atom_ids: Атомарное покрытие сегмента.
        transition_dates: Правые даты выбранных переходов.

    Returns:
        Пару ``(net_movement, gross_movement)``.

    Raises:
        ValueError: Если покрытие/даты отсутствуют или движение нечисловое.

    Examples:
        >>> deltas = pd.DataFrame({1: [2.0, -1.0], 2: [3.0, 4.0]}, index=['a', 'b'])
        >>> _atomic_window_movements(deltas, ['a', 'b'], [1, 2])
        (8.0, 10.0)
    """

    normalized_atoms = [str(atom_id) for atom_id in atom_ids]
    missing_atoms = sorted(set(normalized_atoms) - set(atomic_deltas.index.astype(str)))
    missing_dates = sorted(set(int(date) for date in transition_dates) - set(atomic_deltas.columns))
    if missing_atoms or missing_dates:
        raise ValueError(
            "Невозможно рассчитать атомарное движение: "
            f"missing_atoms={missing_atoms[:5]}, missing_dates={missing_dates[:5]}"
        )
    if not normalized_atoms or not transition_dates:
        return 0.0, 0.0
    values = atomic_deltas.loc[normalized_atoms, list(transition_dates)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Атомарные недельные движения должны быть конечными")
    return float(values.sum()), float(np.abs(values).sum())


def _build_set_packing_adapter(
    candidates: pd.DataFrame,
    score_column: str,
    eligible_ids: Sequence[str],
) -> pd.DataFrame:
    """Подготовить внутренний адаптер к существующему exact Set Packing.

    Args:
        candidates: Trend-кандидаты и строки атомарной поддержки.
        score_column: Колонка trend score, являющаяся objective coefficient.
        eligible_ids: Сегменты, допускаемые в конкретную задачу.

    Returns:
        Таблицу технического anomaly-контракта. Поля не меняют бизнес-смысл
        trend score и не экспортируются как anomaly-результат.

    Raises:
        ValueError: Если отсутствуют metadata или score.

    Examples:
        >>> frame = pd.DataFrame([{'segment_id': 's', 'segment_key': 'x',
        ...     'slice_depth': 1, 'trend_score': 2.0}])
        >>> float(_build_set_packing_adapter(frame, 'trend_score', ['s']).at[0, 'anomaly_score'])
        2.0
    """

    required = {"segment_id", "segment_key", "slice_depth", score_column}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Для Set Packing adapter не хватает колонок: {missing}")
    adapter = candidates.copy()
    eligible_set = {str(segment_id) for segment_id in eligible_ids}
    adapter["segment_id"] = adapter["segment_id"].astype(str)
    adapter["passes_initial_anomaly_filter"] = adapter["segment_id"].isin(eligible_set)
    adapter["anomaly_score"] = pd.to_numeric(adapter[score_column], errors="coerce")
    adapter["robust_z"] = 1.0
    adapter["abs_robust_z"] = 1.0
    adapter["wow_delta_gmv"] = 1.0
    adapter["materiality_share"] = 1.0
    adapter["reliability_factor"] = 1.0
    return adapter


def apply_trend_hierarchy_adjustment(
    scored_summary: pd.DataFrame,
    panel_df: pd.DataFrame,
    dates: Sequence[int],
    coverage: Mapping[str, frozenset[str]],
    config: TrendScoringConfig,
) -> pd.DataFrame:
    """Применить отдельный hierarchy-множитель к собственному trend score.

    Движения считаются по наблюдаемым атомарным недельным дельтам. Для каждого
    потомка используются только переходы, входящие одновременно в активное окно
    родителя и собственное активное окно потомка. Более ранние периоды не
    экстраполируются OLS-линией.

    Args:
        scored_summary: Summary после ``calculate_own_trend_scores``.
        panel_df: Полная панель ``segment_id x cal_date``.
        dates: Полная календарная ось.
        coverage: Фактическое атомарное покрытие всех сегментов.
        config: Параметры scoring и hierarchy.

    Returns:
        Копию summary с hierarchy-диагностикой и ``trend_score``.

    Raises:
        ValueError: Если нарушены metadata, окна, coverage или движения.

    Examples:
        >>> # Функция вызывается build_trend_selection после сверки иерархии.
    """

    required = {
        "segment_id",
        "segment_key",
        "slice_depth",
        "trend_eligible",
        "own_trend_score",
        "current_trend_direction",
        "current_trend_start_date",
        "current_trend_end_date",
    }
    missing = sorted(required - set(scored_summary.columns))
    if missing:
        raise ValueError(f"Для trend hierarchy не хватает колонок: {missing}")
    result = scored_summary.copy()
    result["segment_id"] = result["segment_id"].astype(str)
    if result["segment_id"].duplicated().any():
        raise ValueError("trend_summary содержит дубли segment_id")

    diagnostic_defaults = {
        "trend_hierarchy_eligible_descendant_count": 0,
        "trend_hierarchy_group_count": 0,
        "trend_hierarchy_group_selection_method": "NOT_APPLICABLE",
        "trend_hierarchy_best_group_size": 0,
        "trend_hierarchy_best_group_ids_json": "[]",
        "trend_hierarchy_best_group_segment_keys": "[]",
        "trend_hierarchy_best_group_score": 0.0,
        "trend_hierarchy_parent_gross_movement": math.nan,
        "trend_hierarchy_direction_unity": math.nan,
        "trend_hierarchy_dominant_share": math.nan,
        "trend_hierarchy_balance_max": math.nan,
        "trend_hierarchy_balance_effective": math.nan,
        "trend_hierarchy_balance": math.nan,
        "trend_hierarchy_coherence": math.nan,
        "trend_hierarchy_dominant_child_id": "",
        "trend_hierarchy_dominant_child_segment_key": "",
        "trend_hierarchy_dominant_child_movement": math.nan,
        "trend_hierarchy_dominant_child_capture": math.nan,
        "trend_hierarchy_dominant_child_direction_match": pd.NA,
        "trend_hierarchy_dominance_rule_matches": False,
        "trend_hierarchy_single_child_uncapped_score": math.nan,
        "trend_hierarchy_dominance_cap_score": math.nan,
        "trend_hierarchy_dominance_cap_applied": False,
        "trend_hierarchy_dominance_cap_status": "NOT_APPLICABLE",
        "trend_hierarchy_score_factor": 1.0,
    }
    for column, default in diagnostic_defaults.items():
        result[column] = default
    result["trend_score"] = pd.to_numeric(result["own_trend_score"], errors="coerce")

    metadata = panel_df[["segment_id", "cal_date", "gmv"]].copy()
    metadata["segment_id"] = metadata["segment_id"].astype(str)
    if metadata.duplicated(["segment_id", "cal_date"]).any():
        raise ValueError("panel_df содержит дубли segment_id x cal_date")
    values = metadata.pivot(index="segment_id", columns="cal_date", values="gmv")
    values = values.reindex(columns=[int(date) for date in dates])
    values = values.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Полная панель должна содержать конечный GMV на всех датах")
    atomic_deltas = values.diff(axis=1)
    first_date = int(dates[0])
    atomic_deltas.loc[:, first_date] = 0.0

    normalized_coverage: Dict[str, frozenset[str]] = {
        str(segment_id): frozenset(str(atom_id) for atom_id in atom_ids)
        for segment_id, atom_ids in coverage.items()
    }
    index_by_id = {
        str(segment_id): index
        for index, segment_id in result["segment_id"].items()
    }
    eligible = result.loc[result["trend_eligible"].fillna(False).astype(bool)].copy()
    eligible_ids = eligible["segment_id"].astype(str).tolist()
    if not eligible_ids:
        return result
    for segment_id in eligible_ids:
        score = float(result.at[index_by_id[segment_id], "trend_score"])
        if not math.isfinite(score) or score <= 0.0:
            raise ValueError(
                f"Eligible-тренд {segment_id} должен иметь положительный конечный score"
            )
        if not normalized_coverage.get(segment_id):
            raise ValueError(f"Eligible-тренд {segment_id} имеет пустое coverage")

    depth_by_id = {
        segment_id: int(result.at[index_by_id[segment_id], "slice_depth"])
        for segment_id in eligible_ids
    }
    row_by_id = {
        segment_id: result.loc[index_by_id[segment_id]]
        for segment_id in eligible_ids
    }
    transition_dates_by_id = {
        segment_id: _active_transition_dates(
            dates,
            row_by_id[segment_id]["current_trend_start_date"],
            row_by_id[segment_id]["current_trend_end_date"],
        )
        for segment_id in eligible_ids
    }

    for parent_depth in sorted(set(depth_by_id.values()), reverse=True):
        for parent_id in sorted(
            segment_id
            for segment_id, depth in depth_by_id.items()
            if depth == parent_depth
        ):
            parent_atoms = normalized_coverage[parent_id]
            descendant_ids = sorted(
                child_id
                for child_id in eligible_ids
                if depth_by_id[child_id] > parent_depth
                and normalized_coverage[child_id].issubset(parent_atoms)
            )
            parent_index = index_by_id[parent_id]
            result.at[
                parent_index, "trend_hierarchy_eligible_descendant_count"
            ] = len(descendant_ids)
            parent_transitions = transition_dates_by_id[parent_id]
            _, parent_gross = _atomic_window_movements(
                atomic_deltas,
                sorted(parent_atoms),
                parent_transitions,
            )
            result.at[
                parent_index, "trend_hierarchy_parent_gross_movement"
            ] = parent_gross
            if not descendant_ids:
                continue

            # ADDED: Сильнейшая непересекающаяся группа выбирается существующим
            # exact-контрактом; trend score передаётся только как objective.
            adapter = _build_set_packing_adapter(
                result,
                "trend_score",
                descendant_ids,
            )
            (
                best_group_score,
                best_group_size,
                best_group,
                group_count,
                group_method,
            ) = _select_best_disjoint_descendant_group(
                adapter,
                descendant_ids,
                normalized_coverage,
                parent_atoms,
                max_enumerated_descendants=config.max_hierarchy_descendants,
            )
            result.at[parent_index, "trend_hierarchy_group_count"] = group_count
            result.at[
                parent_index, "trend_hierarchy_group_selection_method"
            ] = group_method
            result.at[parent_index, "trend_hierarchy_best_group_size"] = best_group_size
            result.at[parent_index, "trend_hierarchy_best_group_ids_json"] = json.dumps(
                list(best_group), ensure_ascii=False
            )
            result.at[
                parent_index, "trend_hierarchy_best_group_segment_keys"
            ] = json.dumps(
                [str(result.at[index_by_id[child_id], "segment_key"]) for child_id in best_group],
                ensure_ascii=False,
            )
            result.at[
                parent_index, "trend_hierarchy_best_group_score"
            ] = best_group_score

            child_movement: Dict[str, float] = {}
            child_capture: Dict[str, float] = {}
            child_direction_match: Dict[str, bool] = {}
            parent_direction = str(
                result.at[parent_index, "current_trend_direction"]
            )
            for child_id in descendant_ids:
                common_transitions = tuple(
                    sorted(
                        set(parent_transitions)
                        & set(transition_dates_by_id[child_id])
                    )
                )
                # FIXED: Direction uses the signed movement, while dominance
                # capture uses the child's gross movement on the same window.
                # Thus a child with the same atomic coverage as its parent has
                # capture 100%, even if the common series is volatile.
                net_movement, child_gross = _atomic_window_movements(
                    atomic_deltas,
                    sorted(normalized_coverage[child_id]),
                    common_transitions,
                )
                direction = str(
                    result.at[index_by_id[child_id], "current_trend_direction"]
                )
                sign_matches = (
                    parent_direction == GROWTH and net_movement > 0.0
                ) or (
                    parent_direction == DECLINE and net_movement < 0.0
                )
                direction_matches = direction == parent_direction and sign_matches
                capture = (
                    min(1.0, child_gross / parent_gross)
                    if parent_gross > 0.0
                    else math.nan
                )
                child_movement[child_id] = net_movement
                child_capture[child_id] = capture
                child_direction_match[child_id] = direction_matches

            dominant_ids = [
                child_id
                for child_id in descendant_ids
                if child_direction_match[child_id]
                and math.isfinite(child_capture[child_id])
                and child_capture[child_id]
                >= float(config.dominant_child_capture_threshold)
            ]
            if dominant_ids:
                # ADDED: Dominance не зависит от размера best group. При нескольких
                # кандидатах выбирается наиболее ценный для objective потомок.
                dominant_id = sorted(
                    dominant_ids,
                    key=lambda child_id: (
                        -float(result.at[index_by_id[child_id], "trend_score"]),
                        -float(child_capture[child_id]),
                        -int(depth_by_id[child_id]),
                        str(result.at[index_by_id[child_id], "segment_key"]),
                        child_id,
                    ),
                )[0]
                base_parent_score = float(result.at[parent_index, "own_trend_score"])
                uncapped_score = base_parent_score * float(config.single_child_factor)
                cap_score = float(
                    result.at[index_by_id[dominant_id], "trend_score"]
                ) * (1.0 - float(config.dominant_child_score_margin))
                adjusted_score = min(uncapped_score, cap_score)
                factor = adjusted_score / base_parent_score
                result.at[parent_index, "trend_hierarchy_dominant_child_id"] = dominant_id
                result.at[
                    parent_index, "trend_hierarchy_dominant_child_segment_key"
                ] = str(result.at[index_by_id[dominant_id], "segment_key"])
                result.at[
                    parent_index, "trend_hierarchy_dominant_child_movement"
                ] = child_movement[dominant_id]
                result.at[
                    parent_index, "trend_hierarchy_dominant_child_capture"
                ] = child_capture[dominant_id]
                result.at[
                    parent_index, "trend_hierarchy_dominant_child_direction_match"
                ] = True
                result.at[
                    parent_index, "trend_hierarchy_dominance_rule_matches"
                ] = True
                result.at[
                    parent_index, "trend_hierarchy_single_child_uncapped_score"
                ] = uncapped_score
                result.at[
                    parent_index, "trend_hierarchy_dominance_cap_score"
                ] = cap_score
                result.at[
                    parent_index, "trend_hierarchy_dominance_cap_applied"
                ] = adjusted_score < uncapped_score
                result.at[
                    parent_index, "trend_hierarchy_dominance_cap_status"
                ] = "APPLIED" if adjusted_score < uncapped_score else "RULE_MATCHED_NO_CAP"
            elif best_group_size >= 2:
                movements = [child_movement[child_id] for child_id in best_group]
                gross_movement = float(sum(abs(value) for value in movements))
                if gross_movement <= 0.0:
                    factor = 1.0
                else:
                    direction_unity = min(
                        1.0,
                        max(0.0, abs(sum(movements)) / gross_movement),
                    )
                    shares = [abs(value) / gross_movement for value in movements]
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
                    factor = 1.0 + float(config.aggregation_bonus_lambda) * (
                        coherence - 0.5
                    )
                    result.at[
                        parent_index, "trend_hierarchy_direction_unity"
                    ] = direction_unity
                    result.at[
                        parent_index, "trend_hierarchy_dominant_share"
                    ] = dominant_share
                    result.at[
                        parent_index, "trend_hierarchy_balance_max"
                    ] = balance_max
                    result.at[
                        parent_index, "trend_hierarchy_balance_effective"
                    ] = balance_effective
                    result.at[parent_index, "trend_hierarchy_balance"] = balance
                    result.at[
                        parent_index, "trend_hierarchy_coherence"
                    ] = coherence
            else:
                factor = 1.0

            result.at[parent_index, "trend_hierarchy_score_factor"] = factor
            result.at[parent_index, "trend_score"] = (
                float(result.at[parent_index, "own_trend_score"]) * factor
            )

    return result


def _current_total_gmv(panel_df: pd.DataFrame, dates: Sequence[int]) -> float:
    """Получить GMV единственного Total-сегмента на последней дате.

    Args:
        panel_df: Полная панель.
        dates: Полная календарная ось.

    Returns:
        Положительный текущий Total GMV.

    Raises:
        ValueError: Если Total отсутствует, неоднозначен или невалиден.

    Examples:
        >>> panel = pd.DataFrame({'segment_id': ['t'], 'slice_depth': [0],
        ...     'cal_date': [1], 'gmv': [100.0]})
        >>> _current_total_gmv(panel, [1])
        100.0
    """

    required = {"segment_id", "slice_depth", "cal_date", "gmv"}
    missing = sorted(required - set(panel_df.columns))
    if missing:
        raise ValueError(f"Для определения Total GMV не хватает колонок: {missing}")
    latest_date = int(dates[-1])
    total_ids = panel_df.loc[
        pd.to_numeric(panel_df["slice_depth"], errors="coerce").eq(0),
        "segment_id",
    ].astype(str).unique()
    if len(total_ids) != 1:
        raise ValueError(
            f"Ожидался один Total-сегмент, найдено: {len(total_ids)}"
        )
    rows = panel_df.loc[
        panel_df["segment_id"].astype(str).eq(total_ids[0])
        & pd.to_numeric(panel_df["cal_date"], errors="coerce").eq(latest_date),
        "gmv",
    ]
    if len(rows) != 1:
        raise ValueError("Не найдена единственная текущая строка Total GMV")
    value = float(pd.to_numeric(rows, errors="coerce").iloc[0])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Текущий Total GMV должен быть конечным и положительным")
    return value


def build_trend_selection(
    trend_summary: pd.DataFrame,
    panel_df: pd.DataFrame,
    dates: Sequence[int],
    dim_cols: Sequence[str],
    thresholds: TrendThresholds | None = None,
    config: TrendScoringConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """Рассчитать score и выбрать все совместимые тренды exact Set Packing.

    Кардинального ограничения ``K`` нет: оптимизация максимизирует сумму
    ``trend_score`` при ограничении, что каждый атом покрыт не более чем одним
    выбранным сегментом.

    Args:
        trend_summary: Готовая таблица trend analysis.
        panel_df: Полная иерархическая GMV-панель.
        dates: Полная календарная ось.
        dim_cols: Иерархические измерения входной витрины.
        thresholds: Фактические пороги подтверждения тренда.
        config: Параметры scoring/hierarchy/Set Packing.

    Returns:
        Словарь с обогащённым summary, всеми выбранными трендами и техническим
        decision log существующего exact Set Packing.

    Raises:
        ValueError: Если входные данные нарушают scoring или hierarchy-контракт.

    Examples:
        >>> # result = build_trend_selection(summary, panel, dates, dims)
        >>> # result['selected_trends'] содержит все выбранные сегменты без K.
    """

    thresholds = thresholds or TrendThresholds()
    config = config or TrendScoringConfig()
    if not dates:
        raise ValueError("dates не должен быть пустым")
    metadata_columns = ["segment_id", "segment_key", "slice_depth", *dim_cols]
    missing_metadata = sorted(set(metadata_columns) - set(panel_df.columns))
    if missing_metadata:
        raise ValueError(
            f"Для атомарного покрытия не хватает колонок: {missing_metadata}"
        )
    metadata = panel_df[metadata_columns].drop_duplicates("segment_id").copy()
    coverage = build_atomic_coverage(metadata, dim_cols)
    validate_hierarchy_reconciliation(
        panel_df,
        dim_cols,
        dates,
        absolute_tolerance=config.hierarchy_reconciliation_abs_tolerance,
        coverage=coverage,
    )
    current_total = _current_total_gmv(panel_df, dates)
    scored = calculate_own_trend_scores(
        trend_summary,
        current_total,
        thresholds,
        config,
    )
    hierarchy_scored = apply_trend_hierarchy_adjustment(
        scored,
        panel_df,
        dates,
        coverage,
        config,
    )
    eligible_ids = hierarchy_scored.loc[
        hierarchy_scored["trend_eligible"].fillna(False).astype(bool),
        "segment_id",
    ].astype(str).tolist()
    adapter = _build_set_packing_adapter(
        hierarchy_scored,
        "trend_score",
        eligible_ids,
    )
    solver_thresholds = AnomalyThresholds(
        min_anomaly_abs=0.0,
        min_z_score=0.0,
        min_materiality_share=0.0,
        set_packing_gap_tolerance=config.set_packing_gap_tolerance,
        max_exact_fallback_size=config.max_exact_fallback_size,
    )
    selected_adapter, diagnostics, decision_log = search_anomal(
        adapter,
        solver_thresholds,
        coverage=coverage,
    )
    selected_ids = set(selected_adapter["segment_id"].astype(str))
    rank_by_id = {
        str(row["segment_id"]): int(row["rank"])
        for _, row in selected_adapter.iterrows()
    }
    diagnostic_by_id = diagnostics.set_index("segment_id")
    result = hierarchy_scored.copy()
    result["trend_selected"] = result["segment_id"].astype(str).isin(selected_ids)
    result["trend_selection_rank"] = result["segment_id"].astype(str).map(rank_by_id)
    raw_reason_by_id = diagnostic_by_id["set_packing_reason"].to_dict()
    result["trend_selection_status"] = "TREND_FILTER_NOT_PASSED"
    result.loc[
        pd.to_numeric(result["slice_depth"], errors="coerce").eq(0),
        "trend_selection_status",
    ] = "TOTAL_EXCLUDED"
    result.loc[
        result["trend_eligible"].fillna(False).astype(bool),
        "trend_selection_status",
    ] = "NOT_SELECTED_BY_SET_PACKING"
    result.loc[result["trend_selected"], "trend_selection_status"] = "SELECTED"
    result["trend_selection_reason"] = result["segment_id"].astype(str).map(
        raw_reason_by_id
    )
    result.loc[
        result["trend_selection_status"].eq("TREND_FILTER_NOT_PASSED"),
        "trend_selection_reason",
    ] = "сегмент не прошёл действующий бизнес-контракт подтверждения тренда"
    result.loc[
        result["trend_selection_status"].eq("TOTAL_EXCLUDED"),
        "trend_selection_reason",
    ] = "Total-сегмент диагностируется, но не участвует в оптимизационном отборе"
    result["trend_selection_reason"] = (
        result["trend_selection_reason"]
        .astype(str)
        .str.replace("anomaly_score", "trend_score", regex=False)
        .str.replace("аномалий", "трендов", regex=False)
        .str.replace("аномальности", "тренда", regex=False)
    )
    result["trend_selection_component_id"] = result["segment_id"].astype(str).map(
        diagnostic_by_id["set_packing_component_id"].to_dict()
    )
    result["trend_selection_solver"] = result["segment_id"].astype(str).map(
        diagnostic_by_id["set_packing_solver"].to_dict()
    )
    result["trend_selection_solver_status"] = result["segment_id"].astype(str).map(
        diagnostic_by_id["set_packing_solver_status"].to_dict()
    )
    result["trend_selection_conflict_count"] = result["segment_id"].astype(str).map(
        diagnostic_by_id["conflict_count"].to_dict()
    )
    result["trend_selection_conflict_segment_ids"] = result["segment_id"].astype(str).map(
        diagnostic_by_id["conflict_segment_ids"].to_dict()
    )
    selected_trends = result.loc[result["trend_selected"]].copy()
    selected_trends = selected_trends.sort_values(
        ["trend_selection_rank", "segment_key"],
        kind="stable",
    ).reset_index(drop=True)
    return {
        "trend_summary": result,
        "selected_trends": selected_trends,
        "trend_selection_decision_log": decision_log,
    }


__all__ = [
    "TrendScoringConfig",
    "apply_trend_hierarchy_adjustment",
    "build_trend_selection",
    "calculate_own_trend_scores",
]
