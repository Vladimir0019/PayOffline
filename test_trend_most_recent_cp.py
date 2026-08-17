"""Тесты Most Recent CP и exact DP по рассчитанным segment costs."""

from __future__ import annotations

from itertools import combinations
import json
import math
from time import perf_counter
import unittest

import numpy as np
import pandas as pd

from gmv_anomaly.trend_analysis import (
    GROWTH,
    NO_DIRECTION,
    TrendModelConfig,
    build_configured_trend_analysis,
    build_trend_analysis,
)
from gmv_anomaly.trend_most_recent_cp import (
    DIFF_MAD,
    PERFECT_FIT,
    _fit_ols_line,
    _huber_rho,
    _select_most_recent_profile_candidate,
    analyze_most_recent_cp_series,
    build_most_recent_cp_trend_analysis,
    calculate_segment_cost,
)


def _panel(
    values: list[float],
    dates: list[int] | None = None,
) -> pd.DataFrame:
    """Построить готовую панель одного тестового сегмента.

    Args:
        values: GMV по периодам.
        dates: Необязательная календарная ось.

    Returns:
        DataFrame контракта trend analysis.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> len(_panel([1, 2, 3, 4]))
        4
    """

    normalized_dates = list(range(len(values))) if dates is None else dates
    return pd.DataFrame(
        {
            "segment_id": ["segment"] * len(values),
            "segment_key": ["geo=A"] * len(values),
            "segment_level": ["geo"] * len(values),
            "slice_depth": [1] * len(values),
            "cal_date": normalized_dates,
            "gmv": values,
        }
    )


def _reference_capped_cost(
    values: list[float],
    sigma: float,
    capped_k: float,
) -> float:
    """Получить независимый reference по OLS всех inlier subsets.

    Args:
        values: Маленький тестовый сегмент.
        sigma: Фиксированный scale.
        capped_k: Порог K.

    Returns:
        Минимальный capped objective среди subset-OLS кандидатов.

    Raises:
        ValueError: Если ни один subset не задаёт линию.

    Examples:
        >>> round(_reference_capped_cost([0, 0, 1, 5], 1.0, 2.0), 6)
        4.166667
    """

    array = np.asarray(values, dtype=float)
    time = np.arange(len(array), dtype=float)
    best = math.inf
    for size in range(2, len(array) + 1):
        for indices in combinations(range(len(array)), size):
            subset = np.asarray(indices, dtype=int)
            design = np.column_stack(
                (np.ones(len(subset), dtype=float), time[subset])
            )
            if np.linalg.matrix_rank(design) < 2:
                continue
            coefficients, _, _, _ = np.linalg.lstsq(
                design,
                array[subset],
                rcond=None,
            )
            residuals = (array - (coefficients[0] + coefficients[1] * time)) / sigma
            bounded = np.minimum(np.abs(residuals), capped_k)
            best = min(best, float(np.dot(bounded, bounded)))
    if not math.isfinite(best):
        raise ValueError("Reference search не нашёл допустимую линию")
    return best


class MostRecentCPTests(unittest.TestCase):
    """Проверить T01–T20, batch API и вычислительную стоимость C2."""

    def test_t01_perfect_line_has_no_cp_and_zero_sigma(self) -> None:
        """Проверить perfect-fit fallback без NaN и ложного CP.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T01 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        result = analyze_most_recent_cp_series(
            [100, 110, 120, 130, 140, 150, 160, 170]
        )
        summary = result.summary
        self.assertFalse(summary["structural_change_detected"])
        self.assertIsNone(summary["last_cp_index"])
        self.assertEqual(summary["sigma"], 0.0)
        self.assertEqual(summary["sigma_source"], PERFECT_FIT)
        self.assertEqual(summary["objective_selected"], 0.0)
        self.assertEqual(summary["current_trend_direction"], GROWTH)

    def test_t02_t04_slope_level_and_level_growth_changes(self) -> None:
        """Найти slope change, noisy level shift и level shift + growth.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T02–T04 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        slope_change = analyze_most_recent_cp_series(
            [100, 110, 120, 130, 140, 135, 130, 125, 120]
        )
        self.assertEqual(slope_change.summary["last_cp_index"], 5)

        noisy_level = analyze_most_recent_cp_series(
            [100, 101, 100, 102, 300, 302, 301, 303]
        )
        self.assertEqual(noisy_level.summary["last_cp_index"], 4)
        self.assertTrue(noisy_level.summary["structural_change_detected"])
        self.assertEqual(noisy_level.summary["current_trend_direction"], NO_DIRECTION)
        self.assertGreater(noisy_level.summary["level_shift"], 190.0)

        level_growth = analyze_most_recent_cp_series(
            [100, 101, 100, 102, 300, 320, 340, 360]
        )
        self.assertEqual(level_growth.summary["last_cp_index"], 4)
        self.assertEqual(level_growth.summary["current_trend_direction"], GROWTH)

    def test_t03_exact_plateau_edge_follows_sigma_and_penalty_formulas(self) -> None:
        """Зафиксировать противоречивый exact-plateau edge без скрытого порога.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если формулы sigma/beta заменены подгонкой.

        Examples:
            >>> # Запускается через unittest.
        """

        result = analyze_most_recent_cp_series(
            [100, 100, 100, 100, 300, 300, 300, 300]
        )
        self.assertIsNone(result.summary["last_cp_index"])
        self.assertAlmostEqual(result.summary["beta"], 3.0 * math.log(8.0))
        self.assertLess(
            result.summary["objective_no_change"],
            result.summary["beta"],
        )
        profile = {row["tau"]: row for row in result.cp_profile}
        self.assertIn(4, profile)
        self.assertEqual(profile[4]["current_segment_cost"], 0.0)

    def test_t05_t06_robust_costs_resist_one_outlier(self) -> None:
        """Сравнить OLS, capped и Huber на одиночном выбросе.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T05–T06 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [100, 110, 120, 130, 500, 150, 160, 170, 180, 190, 200, 210]
        results = {
            cost: analyze_most_recent_cp_series(
                values,
                model_config=TrendModelConfig(
                    trend_search_method="most_recent_cp",
                    most_recent_cp_cost=cost,
                ),
            )
            for cost in ("ols", "capped", "huber")
        }
        self.assertIsNotNone(results["ols"].summary["last_cp_index"])
        self.assertIsNone(results["capped"].summary["last_cp_index"])
        self.assertIsNone(results["huber"].summary["last_cp_index"])
        self.assertLess(
            results["huber"].summary["objective_no_change"],
            results["ols"].summary["objective_no_change"],
        )

    def test_t07_t09_minimum_length_and_zero_handling(self) -> None:
        """Проверить L_min, ведущие нули и сохранение внутреннего нуля.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T07–T09 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        three = analyze_most_recent_cp_series([100, 110, 120])
        self.assertEqual(three.summary["status"], "INSUFFICIENT_HISTORY")
        for points in (4, 7):
            result = analyze_most_recent_cp_series(list(range(1, points + 1)))
            self.assertEqual([row["tau"] for row in result.cp_profile], [0])
        eight = analyze_most_recent_cp_series(list(range(1, 9)))
        self.assertEqual([row["tau"] for row in eight.cp_profile], [0, 4])

        leading = analyze_most_recent_cp_series(
            [0, 0, 0, 100, 110, 120, 130]
        )
        self.assertEqual(leading.summary["history_points_raw"], 7)
        self.assertEqual(leading.summary["history_points_used"], 4)
        self.assertEqual(leading.summary["leading_zero_points_trimmed"], 3)

        internal = analyze_most_recent_cp_series(
            [100, 110, 0, 120, 130, 140, 150, 160]
        )
        self.assertEqual(internal.summary["history_points_used"], 8)

    def test_t10_t11_scale_invariance_and_c2_determinism(self) -> None:
        """Сохранить tau/direction при scaling и повторных C2-запусках.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T10–T11 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [100, 101, 100, 102, 300, 320, 340, 360]
        for cost in ("ols", "capped", "huber"):
            config = TrendModelConfig(
                trend_search_method="most_recent_cp",
                most_recent_cp_cost=cost,
            )
            base = analyze_most_recent_cp_series(values, model_config=config)
            scaled = analyze_most_recent_cp_series(
                [7.0 * value for value in values],
                model_config=config,
            )
            with self.subTest(cost=cost):
                self.assertEqual(
                    base.summary["last_cp_index"],
                    scaled.summary["last_cp_index"],
                )
                self.assertEqual(
                    base.summary["current_trend_direction"],
                    scaled.summary["current_trend_direction"],
                )

        capped_config = TrendModelConfig(
            trend_search_method="most_recent_cp",
            most_recent_cp_cost="capped",
        )
        repeated = [
            analyze_most_recent_cp_series(values, model_config=capped_config)
            for _ in range(3)
        ]
        self.assertEqual(
            [result.summary["last_cp_index"] for result in repeated],
            [4, 4, 4],
        )
        self.assertEqual(
            [result.summary["objective_selected"] for result in repeated],
            [repeated[0].summary["objective_selected"]] * 3,
        )
        self.assertEqual(
            [result.segmentation for result in repeated],
            [repeated[0].segmentation] * 3,
        )

    def test_t12_t15_config_validation_beta_and_legacy_dispatcher(self) -> None:
        """Проверить fail-fast, active n в beta и backward compatibility.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T12, T13 или T15 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        invalid_configs = (
            {"trend_search_method": "unknown"},
            {"most_recent_cp_cost": "unknown"},
            {"most_recent_cp_capped_k": 0.0},
            {"most_recent_cp_huber_delta": 0.0},
            {"most_recent_cp_min_segment_points": 3},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrendModelConfig(**kwargs)

        active = analyze_most_recent_cp_series(
            [0, 0, 0, 100, 101, 100, 102, 300, 302, 301, 303]
        )
        self.assertEqual(active.summary["history_points_used"], 8)
        self.assertAlmostEqual(active.summary["beta"], 3.0 * math.log(8.0))

        panel = _panel([100, 110, 120, 130, 140, 150, 160, 170])
        dates = list(range(8))
        direct = build_trend_analysis(panel, dates)
        dispatched = build_configured_trend_analysis(panel, dates)
        self.assertEqual(set(direct), set(dispatched))
        for key in direct:
            pd.testing.assert_frame_equal(direct[key], dispatched[key])

    def test_t14_constant_and_perfect_line_are_deterministic(self) -> None:
        """Проверить оба sigma=0 сценария без crash/NaN objective.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T14 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        for values in ([100.0] * 8, [100 + 10 * index for index in range(8)]):
            result = analyze_most_recent_cp_series(values)
            with self.subTest(values=values):
                self.assertEqual(result.summary["sigma"], 0.0)
                self.assertEqual(result.summary["sigma_source"], PERFECT_FIT)
                self.assertTrue(math.isfinite(result.summary["objective_selected"]))
                self.assertIsNone(result.summary["last_cp_index"])

    def test_t16_multiple_historical_breaks_restore_last_cp(self) -> None:
        """Восстановить три режима и вернуть последнюю границу.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T16 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [
            100, 110, 120, 130,
            180, 170, 160, 150,
            200, 220, 240, 260,
        ]
        result = analyze_most_recent_cp_series(values)
        self.assertEqual(result.summary["last_cp_index"], 8)
        self.assertEqual(
            json.loads(result.summary["optimal_breakpoints_json"]),
            [4, 8],
        )
        self.assertEqual(
            [(row["start"], row["end"]) for row in result.segmentation],
            [(0, 4), (4, 8), (8, 12)],
        )

    def test_t17_c2_matches_reference_and_is_not_ols_clipping(self) -> None:
        """Сверить bounded regression с независимым subset reference.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T17 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [0.0, 0.0, 1.0, 5.0]
        result = calculate_segment_cost(
            values,
            0,
            4,
            1.0,
            "capped",
            capped_k=2.0,
        )
        reference = _reference_capped_cost(values, 1.0, 2.0)
        _, _, ols_residuals, _ = _fit_ols_line(
            np.arange(4.0),
            np.asarray(values),
        )
        clipped = np.minimum(np.abs(ols_residuals), 2.0)
        ols_then_clip = float(np.dot(clipped, clipped))
        self.assertAlmostEqual(result.cost, reference, places=10)
        self.assertLess(result.cost, ols_then_clip)
        self.assertIn("ACTIVE_SET_MULTI_START", result.optimizer_status)

    def test_t18_huber_formula_uses_c1_quadratic_scale(self) -> None:
        """Проверить Huber при |u| меньше, равно и больше delta.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T18 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        actual = _huber_rho([-0.5, 1.0, -2.0], 1.0)
        np.testing.assert_allclose(actual, [0.25, 1.0, 3.0])

    def test_t19_dates_and_full_batch_diagnostics(self) -> None:
        """Проверить даты half-open CP и четыре выходных DataFrame.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T19 или batch-контракт нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [100, 101, 100, 102, 300, 302, 301, 303]
        dates = [100 + 7 * index for index in range(8)]
        config = TrendModelConfig(trend_search_method="most_recent_cp")
        result = build_most_recent_cp_trend_analysis(
            _panel(values, dates),
            dates,
            model_config=config,
        )
        self.assertEqual(
            set(result),
            {
                "trend_summary",
                "trend_cp_profile",
                "trend_segment_diagnostics",
                "trend_segmentation",
            },
        )
        summary = result["trend_summary"].iloc[0]
        self.assertEqual(summary["last_cp_left_end_date"], dates[3])
        self.assertEqual(summary["last_cp_right_start_date"], dates[4])
        self.assertEqual(
            result["trend_cp_profile"]["tau"].tolist(),
            [0, 4],
        )
        self.assertGreater(len(result["trend_segment_diagnostics"]), 0)
        self.assertEqual(len(result["trend_segmentation"]), 2)

        dispatched = build_configured_trend_analysis(
            _panel(values, dates),
            dates,
            model_config=config,
        )
        pd.testing.assert_frame_equal(
            result["trend_summary"],
            dispatched["trend_summary"],
        )

    def test_t20_machine_tie_prefers_later_but_worse_candidate_loses(self) -> None:
        """Разрешить recency только внутри machine tolerance.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T20 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        epsilon = np.finfo(float).eps
        selected, tied, _ = _select_most_recent_profile_candidate(
            [
                {"tau": 4, "G_tau": 1.0},
                {"tau": 5, "G_tau": 1.0 + epsilon},
            ]
        )
        self.assertTrue(tied)
        self.assertEqual(selected["tau"], 5)

        selected, tied, _ = _select_most_recent_profile_candidate(
            [
                {"tau": 4, "G_tau": 1.0},
                {"tau": 5, "G_tau": 1.0 + 1e-8},
            ]
        )
        self.assertFalse(tied)
        self.assertEqual(selected["tau"], 4)

    def test_c2_n20_performance_is_measured(self) -> None:
        """Ограничить регрессию времени C2 на максимальной длине ряда.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если один n=20 ряд стал практически непригодным.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [
            100 + 3 * index + (2 if index % 4 == 0 else -1 if index % 5 == 0 else 0)
            for index in range(20)
        ]
        config = TrendModelConfig(
            trend_search_method="most_recent_cp",
            most_recent_cp_cost="capped",
        )
        started = perf_counter()
        result = analyze_most_recent_cp_series(values, model_config=config)
        elapsed = perf_counter() - started
        self.assertEqual(len(result.segment_diagnostics), 153)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
