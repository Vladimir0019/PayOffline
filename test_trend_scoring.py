"""Unit- и интеграционные тесты trend score, hierarchy и Set Packing."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from gmv_anomaly.trend_analysis import GROWTH, TrendThresholds
from gmv_anomaly.trend_scoring import (
    TrendScoringConfig,
    _active_transition_dates,
    _atomic_window_movements,
    build_trend_selection,
    calculate_own_trend_scores,
)


def _hierarchy_panel(
    atomic_values: dict[str, list[float]],
) -> tuple[pd.DataFrame, list[int]]:
    """Построить аддитивную двухуровневую панель для trend scoring.

    Args:
        atomic_values: Значения атомов, ключ является значением product.

    Returns:
        Полную панель и календарные даты.

    Raises:
        ValueError: Если длины атомарных рядов различаются.

    Examples:
        >>> len(_hierarchy_panel({'X': [1.0, 2.0]})[0])
        6
    """

    lengths = {len(values) for values in atomic_values.values()}
    if len(lengths) != 1:
        raise ValueError("Длины атомарных рядов должны совпадать")
    dates = list(range(next(iter(lengths))))
    parent_values = [
        sum(float(values[index]) for values in atomic_values.values())
        for index in dates
    ]
    rows: list[dict[str, object]] = []
    segments = [
        ("total", "Total", 0, None, None, parent_values),
        ("parent", "geo=A", 1, "A", None, parent_values),
        *[
            (
                f"atom_{product}",
                f"geo=A × product={product}",
                2,
                "A",
                product,
                values,
            )
            for product, values in atomic_values.items()
        ],
    ]
    for segment_id, segment_key, depth, geo, product, values in segments:
        for date, value in zip(dates, values):
            rows.append(
                {
                    "segment_id": segment_id,
                    "segment_key": segment_key,
                    "segment_level": "Total" if depth == 0 else "test",
                    "slice_depth": depth,
                    "geo": geo,
                    "product": product,
                    "cal_date": date,
                    "gmv": float(value),
                }
            )
    return pd.DataFrame(rows), dates


def _trend_summary(
    panel: pd.DataFrame,
    dates: list[int],
    eligible: set[str],
    starts: dict[str, int] | None = None,
    slopes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Построить минимальный summary уже выполненного trend analysis.

    Args:
        panel: Полная тестовая панель.
        dates: Календарная ось.
        eligible: ID сегментов с подтверждённым трендом.
        starts: Необязательные начала активных окон.
        slopes: Необязательные абсолютные slopes.

    Returns:
        Одну summary-строку на сегмент.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> panel, dates = _hierarchy_panel({'X': [1.0, 2.0]})
        >>> len(_trend_summary(panel, dates, {'atom_X'}))
        3
    """

    starts = starts or {}
    slopes = slopes or {}
    metadata = panel[
        ["segment_id", "segment_key", "segment_level", "slice_depth"]
    ].drop_duplicates("segment_id")
    rows = []
    for _, row in metadata.iterrows():
        segment_id = str(row["segment_id"])
        exists = segment_id in eligible
        slope = float(slopes.get(segment_id, 1.0)) if exists else math.nan
        rows.append(
            {
                **row.to_dict(),
                "current_trend_exists": exists,
                "current_trend_direction": GROWTH if exists else "NONE",
                "current_trend_length": (
                    len(dates) - int(starts.get(segment_id, dates[0])) if exists else 0
                ),
                "current_trend_start_date": (
                    int(starts.get(segment_id, dates[0])) if exists else None
                ),
                "current_trend_end_date": int(dates[-1]) if exists else None,
                "trend_slope_abs": slope,
                "trend_slope_relative": 0.10 if exists else math.nan,
                "direction_count_share": 1.0 if exists else math.nan,
                "direction_movement_share": 1.0 if exists else math.nan,
            }
        )
    return pd.DataFrame(rows)


class TrendOwnScoreTests(unittest.TestCase):
    """Проверить формулу impact и отсутствие duration adjustment."""

    def test_geometric_impact_and_direction_adjustment(self) -> None:
        """Проверить согласованные формулы собственного score."""

        frame = pd.DataFrame(
            [
                {
                    "segment_id": "s",
                    "slice_depth": 1,
                    "current_trend_exists": True,
                    "current_trend_length": 4,
                    "trend_slope_abs": 4.0,
                    "trend_slope_relative": 0.04,
                    "direction_count_share": 1.0,
                    "direction_movement_share": 1.0,
                }
            ]
        )
        result = calculate_own_trend_scores(
            frame,
            100.0,
            TrendThresholds(),
            TrendScoringConfig(),
        )
        self.assertAlmostEqual(float(result.at[0, "absolute_rate_share"]), 0.04)
        self.assertAlmostEqual(float(result.at[0, "relative_rate"]), 0.04)
        self.assertAlmostEqual(float(result.at[0, "trend_impact"]), 0.04)
        self.assertAlmostEqual(float(result.at[0, "direction_quality"]), 1.0)
        self.assertAlmostEqual(float(result.at[0, "own_adjustment_signal"]), 1.0)
        self.assertAlmostEqual(float(result.at[0, "own_score_adjustment"]), 0.10)
        self.assertAlmostEqual(float(result.at[0, "own_trend_score"]), 0.044)

    def test_duration_is_explicitly_excluded_and_does_not_change_score(self) -> None:
        """Разная длина при равных метриках не должна менять score."""

        base = {
            "slice_depth": 1,
            "current_trend_exists": True,
            "trend_slope_abs": 5.0,
            "trend_slope_relative": 0.05,
            "direction_count_share": 0.9,
            "direction_movement_share": 0.9,
        }
        frame = pd.DataFrame(
            [
                {**base, "segment_id": "short", "current_trend_length": 4},
                {**base, "segment_id": "long", "current_trend_length": 13},
            ]
        )
        result = calculate_own_trend_scores(
            frame,
            100.0,
            TrendThresholds(),
            TrendScoringConfig(),
        )
        self.assertAlmostEqual(
            float(result.at[0, "own_trend_score"]),
            float(result.at[1, "own_trend_score"]),
        )
        self.assertTrue(
            result["trend_duration_score_status"].eq(
                "NOT_INCLUDED_BY_DESIGN"
            ).all()
        )

    def test_score_is_scale_invariant(self) -> None:
        """GMV-масштаб не должен менять dimensionless trend score."""

        frame = pd.DataFrame(
            [
                {
                    "segment_id": "s",
                    "slice_depth": 1,
                    "current_trend_exists": True,
                    "trend_slope_abs": 4.0,
                    "trend_slope_relative": 0.04,
                    "direction_count_share": 0.9,
                    "direction_movement_share": 0.8,
                }
            ]
        )
        original = calculate_own_trend_scores(
            frame,
            100.0,
            TrendThresholds(),
            TrendScoringConfig(),
        )
        scaled_frame = frame.copy()
        scaled_frame["trend_slope_abs"] *= 1000.0
        scaled = calculate_own_trend_scores(
            scaled_frame,
            100_000.0,
            TrendThresholds(),
            TrendScoringConfig(),
        )
        self.assertAlmostEqual(
            float(original.at[0, "trend_impact"]),
            float(scaled.at[0, "trend_impact"]),
        )
        self.assertAlmostEqual(
            float(original.at[0, "own_trend_score"]),
            float(scaled.at[0, "own_trend_score"]),
        )


class TrendHierarchyTests(unittest.TestCase):
    """Проверить активные окна, coherence и независимый dominance cap."""

    def test_atomic_movement_uses_intersection_of_active_windows(self) -> None:
        """Падение до старта child-тренда нельзя заменять OLS-ростом."""

        panel, dates = _hierarchy_panel(
            {"X": [100.0, 80.0, 60.0, 80.0, 100.0, 120.0]}
        )
        atom_values = panel.loc[panel["segment_id"].eq("atom_X")].set_index(
            "cal_date"
        )["gmv"]
        deltas = pd.DataFrame([atom_values.diff().fillna(0.0)], index=["atom_X"])
        parent_transitions = _active_transition_dates(dates, 0, 5)
        child_transitions = _active_transition_dates(dates, 2, 5)
        parent_net, parent_gross = _atomic_window_movements(
            deltas,
            ["atom_X"],
            parent_transitions,
        )
        child_net, child_gross = _atomic_window_movements(
            deltas,
            ["atom_X"],
            sorted(set(parent_transitions) & set(child_transitions)),
        )
        self.assertAlmostEqual(parent_net, 20.0)
        self.assertAlmostEqual(parent_gross, 100.0)
        self.assertAlmostEqual(child_net, 60.0)
        self.assertAlmostEqual(child_gross, 60.0)

    def test_parent_factor_uses_full_anomaly_range_formula(self) -> None:
        """Согласованная группа должна получать интерпретируемый bonus."""

        panel, dates = _hierarchy_panel(
            {
                "X": [100.0, 110.0, 120.0, 130.0],
                "Y": [200.0, 220.0, 240.0, 260.0],
            }
        )
        summary = _trend_summary(
            panel,
            dates,
            {"parent", "atom_X", "atom_Y"},
        )
        result = build_trend_selection(
            summary,
            panel,
            dates,
            ["geo", "product"],
        )["trend_summary"].set_index("segment_id")
        parent = result.loc["parent"]
        self.assertEqual(int(parent["trend_hierarchy_best_group_size"]), 2)
        self.assertAlmostEqual(float(parent["trend_hierarchy_direction_unity"]), 1.0)
        self.assertAlmostEqual(float(parent["trend_hierarchy_balance"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(parent["trend_hierarchy_score_factor"]), 1.05)

    def test_dominant_child_is_checked_independently_of_best_group_size(self) -> None:
        """Группа из двух потомков не должна отключать dominance rule."""

        panel, dates = _hierarchy_panel(
            {
                "X": [100.0, 130.0, 160.0, 190.0],
                "Y": [100.0, 103.0, 106.0, 110.0],
            }
        )
        summary = _trend_summary(
            panel,
            dates,
            {"parent", "atom_X", "atom_Y"},
            slopes={"parent": 10.0, "atom_X": 20.0, "atom_Y": 1.0},
        )
        result = build_trend_selection(
            summary,
            panel,
            dates,
            ["geo", "product"],
        )["trend_summary"].set_index("segment_id")
        parent = result.loc["parent"]
        self.assertEqual(int(parent["trend_hierarchy_best_group_size"]), 2)
        self.assertTrue(bool(parent["trend_hierarchy_dominance_rule_matches"]))
        self.assertEqual(parent["trend_hierarchy_dominant_child_id"], "atom_X")
        self.assertGreaterEqual(
            float(parent["trend_hierarchy_dominant_child_capture"]),
            0.85,
        )
        self.assertLessEqual(
            float(parent["trend_hierarchy_score_factor"]),
            0.85,
        )

    def test_identical_volatile_child_has_full_gross_capture(self) -> None:
        """FIXED: Проверить gross-capture для идентичного волатильного ребёнка.

        Args:
            Нет аргументов, кроме ``self``.

        Returns:
            None.

        Raises:
            AssertionError: Если ребёнок с идентичным atomic coverage не
                признан доминирующим с capture 100%.

        Examples:
            Запускается через ``python -m unittest gmv_anomaly.test_trend_scoring``.
        """

        panel, dates = _hierarchy_panel(
            {"X": [100.0, 80.0, 60.0, 80.0, 100.0, 120.0]}
        )
        summary = _trend_summary(
            panel,
            dates,
            {"parent", "atom_X"},
            slopes={"parent": 1.0, "atom_X": 1.0},
        )
        result = build_trend_selection(
            summary,
            panel,
            dates,
            ["geo", "product"],
        )["trend_summary"].set_index("segment_id")
        parent = result.loc["parent"]
        self.assertTrue(bool(parent["trend_hierarchy_dominance_rule_matches"]))
        self.assertEqual(parent["trend_hierarchy_dominant_child_id"], "atom_X")
        self.assertAlmostEqual(
            float(parent["trend_hierarchy_dominant_child_capture"]),
            1.0,
        )

    def test_dominance_cap_keeps_parent_below_dominant_child(self) -> None:
        """Проверить перенесённый anomaly-cap, когда множителя 0.85 мало."""

        panel, dates = _hierarchy_panel(
            {
                "X": [100.0, 130.0, 160.0, 190.0],
                "Y": [100.0, 103.0, 106.0, 110.0],
            }
        )
        summary = _trend_summary(
            panel,
            dates,
            {"parent", "atom_X", "atom_Y"},
            slopes={"parent": 100.0, "atom_X": 1.0, "atom_Y": 1.0},
        )
        result = build_trend_selection(
            summary,
            panel,
            dates,
            ["geo", "product"],
        )["trend_summary"].set_index("segment_id")
        parent = result.loc["parent"]
        child = result.loc[str(parent["trend_hierarchy_dominant_child_id"])]
        self.assertTrue(bool(parent["trend_hierarchy_dominance_cap_applied"]))
        self.assertAlmostEqual(
            float(parent["trend_score"]),
            float(child["trend_score"]) * 0.98,
        )


class TrendSetPackingTests(unittest.TestCase):
    """Проверить eligibility атомов и отсутствие ограничения K."""

    def test_all_disjoint_eligible_atomic_segments_are_selected_without_k(self) -> None:
        """Exact Set Packing должен вернуть все совместимые positive-score строки."""

        panel, dates = _hierarchy_panel(
            {
                "X": [100.0, 110.0, 120.0, 130.0],
                "Y": [200.0, 220.0, 240.0, 260.0],
                "Z": [300.0, 330.0, 360.0, 390.0],
            }
        )
        summary = _trend_summary(
            panel,
            dates,
            {"atom_X", "atom_Y", "atom_Z"},
        )
        selection = build_trend_selection(
            summary,
            panel,
            dates,
            ["geo", "product"],
        )
        self.assertEqual(
            set(selection["selected_trends"]["segment_id"]),
            {"atom_X", "atom_Y", "atom_Z"},
        )
        diagnostics = selection["trend_summary"].set_index("segment_id")
        self.assertFalse(bool(diagnostics.at["parent", "trend_eligible"]))
        self.assertEqual(
            diagnostics.at["parent", "trend_selection_status"],
            "TREND_FILTER_NOT_PASSED",
        )


if __name__ == "__main__":
    unittest.main()
