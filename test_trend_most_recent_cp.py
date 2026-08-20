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
    DECLINE,
    GROWTH,
    NO_DIRECTION,
    TrendModelConfig,
    TrendThresholds,
    build_configured_trend_analysis,
    build_trend_analysis,
    evaluate_trend,
)
from gmv_anomaly.trend_most_recent_cp import (
    DIFF_MAD,
    LEVEL_AND_SLOPE,
    LEVEL_SHIFT,
    LOCAL_FLAT,
    LOCAL_UNCONFIRMED,
    NO_STRUCTURAL_CHANGE,
    PERFECT_FIT,
    SLOPE_CHANGE,
    WEAK_OR_UNCLASSIFIED,
    _calculate_level_shift_statistics,
    _calculate_slope_change_statistics,
    _classify_structural_change,
    _classify_local_regime,
    _evaluate_selected_regimes,
    _fit_ols_regime_diagnostic,
    _fit_ols_line,
    _huber_rho,
    _select_most_recent_profile_candidate,
    _select_global_regime_indices,
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
        # FIXED: Источник истины — действующий penalty 3*ln(n)*1.5.
        self.assertAlmostEqual(
            result.summary["beta"],
            3.0 * math.log(8.0) * 1.5,
        )
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
            {"most_recent_cp_change_z_threshold": 0.0},
            {"most_recent_cp_change_z_threshold": math.inf},
            {"global_flat_max_relative_slope": -0.01},
            {"global_flat_max_total_change": -0.01},
            {"global_flat_max_noise_ratio": -0.01},
            {"global_max_flat_bridge_regimes": -1},
            {"global_max_flat_bridge_regimes": True},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrendModelConfig(**kwargs)

        active = analyze_most_recent_cp_series(
            [0, 0, 0, 100, 101, 100, 102, 300, 302, 301, 303]
        )
        self.assertEqual(active.summary["history_points_used"], 8)
        self.assertAlmostEqual(
            active.summary["beta"],
            3.0 * math.log(8.0) * 1.5,
        )

        panel = _panel([100, 110, 120, 130, 140, 150, 160, 170])
        dates = list(range(8))
        direct = build_trend_analysis(panel, dates)
        # FIXED: Действующий default selector — most_recent_cp; legacy
        # backward-compatibility проверяется только явной конфигурацией.
        dispatched = build_configured_trend_analysis(
            panel,
            dates,
            model_config=TrendModelConfig(trend_search_method="legacy"),
        )
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
        self.assertEqual(
            [row["cp_index"] for row in result.changepoints],
            [4, 8],
        )
        self.assertEqual(
            result.changepoints[-1]["structural_change_type"],
            result.summary["structural_change_type"],
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
                "trend_changepoints",
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

    def test_structural_post_classification_level_slope_and_combined(self) -> None:
        """Различить чистые level, slope и одновременные изменения.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если standardized post-classification нарушена.

        Examples:
            >>> # Запускается через unittest.
        """

        config = TrendModelConfig(
            trend_search_method="most_recent_cp",
            most_recent_cp_cost="ols",
        )
        level = analyze_most_recent_cp_series(
            [100, 101, 100, 102, 300, 302, 301, 303],
            model_config=config,
        ).summary
        self.assertEqual(level["structural_change_type"], LEVEL_SHIFT)
        self.assertGreaterEqual(level["level_shift_z"], level["change_z_threshold"])
        self.assertLess(level["slope_change_z"], level["change_z_threshold"])
        self.assertFalse(level["direction_change"])

        slope = analyze_most_recent_cp_series(
            [100, 110, 120, 130, 140, 150, 145, 140, 135],
            model_config=config,
        ).summary
        self.assertEqual(slope["last_cp_index"], 5)
        self.assertEqual(slope["structural_change_type"], SLOPE_CHANGE)
        self.assertLess(slope["level_shift_z"], slope["change_z_threshold"])
        self.assertGreaterEqual(slope["slope_change_z"], slope["change_z_threshold"])

        combined = analyze_most_recent_cp_series(
            [100, 101, 100, 102, 101, 103, 300, 320, 340, 360, 380, 400],
            model_config=config,
        ).summary
        self.assertEqual(combined["last_cp_index"], 6)
        self.assertEqual(combined["structural_change_type"], LEVEL_AND_SLOPE)
        self.assertGreaterEqual(
            combined["level_shift_z"],
            combined["change_z_threshold"],
        )
        self.assertGreaterEqual(
            combined["slope_change_z"],
            combined["change_z_threshold"],
        )

    def test_direction_change_uses_evaluate_trend_business_contract(self) -> None:
        """Отделить GROWTH/DECLINE reversal от знака OLS slope.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если direction_change не ортогонален structural type.

        Examples:
            >>> # Запускается через unittest.
        """

        config = TrendModelConfig(
            trend_search_method="most_recent_cp",
            most_recent_cp_cost="ols",
        )
        scenarios = (
            (
                [100, 110, 120, 130, 140, 150, 145, 140, 135],
                GROWTH,
                DECLINE,
                True,
            ),
            (
                [300, 290, 280, 270, 260, 250, 260, 270, 280],
                DECLINE,
                GROWTH,
                True,
            ),
            (
                [100, 101, 99, 100, 100, 120, 140, 160],
                NO_DIRECTION,
                GROWTH,
                False,
            ),
            (
                [100, 105, 110, 115, 120, 125, 145, 165, 185],
                GROWTH,
                GROWTH,
                False,
            ),
        )
        for values, previous_direction, current_direction, expected in scenarios:
            with self.subTest(values=values):
                summary = analyze_most_recent_cp_series(
                    values,
                    model_config=config,
                ).summary
                self.assertEqual(summary["structural_change_type"], SLOPE_CHANGE)
                self.assertEqual(
                    summary["previous_regime_trend_direction"],
                    previous_direction,
                )
                self.assertEqual(
                    summary["current_trend_direction"],
                    current_direction,
                )
                self.assertEqual(summary["direction_change"], expected)

        combined_reversal = analyze_most_recent_cp_series(
            [100, 110, 120, 130, 140, 150, 300, 280, 260, 240, 220, 200],
            model_config=config,
        ).summary
        self.assertEqual(
            combined_reversal["structural_change_type"],
            LEVEL_AND_SLOPE,
        )
        self.assertTrue(combined_reversal["direction_change"])

    def test_weak_and_threshold_boundary_classification_helper(self) -> None:
        """Проверить weak fallback и включающую границу ``>= threshold``.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если таблица structural classification нарушена.

        Examples:
            >>> # Запускается через unittest.
        """

        threshold = 2.0
        below = math.nextafter(threshold, 0.0)
        above = math.nextafter(threshold, math.inf)
        self.assertEqual(
            _classify_structural_change(True, below, below, threshold),
            WEAK_OR_UNCLASSIFIED,
        )
        self.assertEqual(
            _classify_structural_change(True, threshold, below, threshold),
            LEVEL_SHIFT,
        )
        self.assertEqual(
            _classify_structural_change(True, below, threshold, threshold),
            SLOPE_CHANGE,
        )
        self.assertEqual(
            _classify_structural_change(True, above, above, threshold),
            LEVEL_AND_SLOPE,
        )
        self.assertEqual(
            _classify_structural_change(False, math.nan, math.nan, threshold),
            NO_STRUCTURAL_CHANGE,
        )
        high_threshold_summary = analyze_most_recent_cp_series(
            [100, 101, 100, 102, 300, 302, 301, 303],
            model_config=TrendModelConfig(
                trend_search_method="most_recent_cp",
                most_recent_cp_cost="ols",
                most_recent_cp_change_z_threshold=200.0,
            ),
        ).summary
        self.assertEqual(
            high_threshold_summary["structural_change_type"],
            WEAK_OR_UNCLASSIFIED,
        )
        self.assertEqual(high_threshold_summary["change_z_threshold"], 200.0)

    def test_diagnostic_se_formulas_use_global_time_and_tau_boundary(self) -> None:
        """Независимо проверить Sxx=5, slope SE и fitted-mean level SE.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если Sxx, leverage или half-open индексы ошибочны.

        Examples:
            >>> # Запускается через unittest.
        """

        values = np.arange(8.0)
        previous = _fit_ols_regime_diagnostic(values, 0, 4)
        current = _fit_ols_regime_diagnostic(values, 4, 8)
        sigma = 3.0
        self.assertEqual(previous.sxx, 5.0)
        self.assertEqual(current.sxx, 5.0)
        _, slope_se, slope_z = _calculate_slope_change_statistics(
            previous,
            current,
            sigma,
            values,
        )
        _, level_se, level_z = _calculate_level_shift_statistics(
            previous,
            current,
            4,
            sigma,
            values,
        )
        self.assertAlmostEqual(slope_se, sigma * math.sqrt(0.4), places=12)
        self.assertAlmostEqual(level_se, sigma * math.sqrt(2.2), places=12)
        self.assertEqual(slope_z, 0.0)
        self.assertEqual(level_z, 0.0)
        with self.assertRaises(ValueError):
            _calculate_level_shift_statistics(previous, current, 3, sigma, values)

    def test_post_classification_scale_invariance_and_raw_scaling(self) -> None:
        """Сохранить Z/type/direction и масштабировать raw effects с GMV.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если диагностический слой зависит от единиц GMV.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [100, 101, 100, 102, 101, 103, 300, 320, 340, 360, 380, 400]
        factor = 1000.0
        config = TrendModelConfig(
            trend_search_method="most_recent_cp",
            most_recent_cp_cost="ols",
        )
        base = analyze_most_recent_cp_series(values, model_config=config).summary
        scaled = analyze_most_recent_cp_series(
            [factor * value for value in values],
            model_config=config,
        ).summary
        for field in ("structural_change_type", "direction_change", "last_cp_index"):
            self.assertEqual(base[field], scaled[field])
        for field in ("level_shift_z", "slope_change_z"):
            self.assertAlmostEqual(base[field], scaled[field], places=10)
        for field in (
            "classification_previous_slope_ols",
            "classification_current_slope_ols",
            "classification_delta_slope_ols",
            "classification_level_shift_ols",
            "slope_change_se",
            "level_shift_se",
            "sigma",
        ):
            self.assertAlmostEqual(scaled[field], factor * base[field], places=7)

    def test_no_cp_and_zero_sigma_classification_are_deterministic(self) -> None:
        """Вернуть neutral-поля без CP и 0/inf без деления на ноль.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если no-CP или PERFECT_FIT обработан нестабильно.

        Examples:
            >>> # Запускается через unittest.
        """

        no_cp = analyze_most_recent_cp_series(
            [100 + 10 * index for index in range(8)],
            model_config=TrendModelConfig(
                trend_search_method="most_recent_cp",
                most_recent_cp_cost="ols",
            ),
        ).summary
        self.assertEqual(no_cp["structural_change_type"], NO_STRUCTURAL_CHANGE)
        self.assertFalse(no_cp["direction_change"])
        self.assertFalse(no_cp["previous_regime_trend_exists"])
        self.assertEqual(no_cp["previous_regime_trend_status"], "NOT_EVALUATED")
        for field in (
            "classification_previous_slope_ols",
            "classification_current_slope_ols",
            "classification_delta_slope_ols",
            "classification_level_shift_ols",
            "slope_change_se",
            "level_shift_se",
            "slope_change_z",
            "level_shift_z",
        ):
            self.assertTrue(math.isnan(no_cp[field]))

        zero_effect_values = np.arange(8.0)
        previous = _fit_ols_regime_diagnostic(zero_effect_values, 0, 4)
        current = _fit_ols_regime_diagnostic(zero_effect_values, 4, 8)
        self.assertEqual(
            _calculate_slope_change_statistics(
                previous, current, 0.0, zero_effect_values
            )[2],
            0.0,
        )
        self.assertEqual(
            _calculate_level_shift_statistics(
                previous, current, 4, 0.0, zero_effect_values
            )[2],
            0.0,
        )

        nonzero_effect_values = np.asarray([0, 1, 2, 3, 4, 6, 8, 10], dtype=float)
        previous = _fit_ols_regime_diagnostic(nonzero_effect_values, 0, 4)
        current = _fit_ols_regime_diagnostic(nonzero_effect_values, 4, 8)
        first = _calculate_slope_change_statistics(
            previous, current, 0.0, nonzero_effect_values
        )
        second = _calculate_slope_change_statistics(
            previous, current, 0.0, nonzero_effect_values
        )
        self.assertEqual(first, second)
        self.assertEqual(first[1], 0.0)
        self.assertTrue(math.isinf(first[2]))

    def test_robust_search_coefficients_are_not_used_with_ols_se(self) -> None:
        """Сохранить C2 coefficients и отдельно вычислить OLS diagnostics.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если robust slope смешан с OLS standard error.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [100, 101, 100, 102, 130, 103, 300, 302, 301, 303, 302, 304]
        summary = analyze_most_recent_cp_series(
            values,
            model_config=TrendModelConfig(
                trend_search_method="most_recent_cp",
                most_recent_cp_cost="capped",
            ),
        ).summary
        self.assertEqual(summary["last_cp_index"], 6)
        self.assertNotAlmostEqual(
            summary["previous_regime_slope"],
            summary["classification_previous_slope_ols"],
        )
        expected = _fit_ols_regime_diagnostic(values, 0, 6)
        self.assertAlmostEqual(
            summary["classification_previous_slope_ols"],
            expected.slope,
        )

    def test_global_trend_extends_only_when_combined_window_is_confirmed(self) -> None:
        """Объединить небольшую коррекцию и остановиться на большом level drop.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если повторный evaluate_trend не ограничивает merge.

        Examples:
            >>> # Запускается через unittest.
        """

        thresholds = TrendThresholds()
        config = TrendModelConfig(trend_search_method="most_recent_cp")
        segments = [(0, 4), (4, 8)]
        dates = list(range(8))

        small_drop = np.asarray(
            [100, 110, 120, 130, 125, 135, 145, 155],
            dtype=float,
        )
        rows, evaluations = _evaluate_selected_regimes(
            small_drop,
            dates,
            segments,
            thresholds,
            config,
        )
        included, global_evaluation = _select_global_regime_indices(
            small_drop,
            rows,
            evaluations,
            thresholds,
            config,
        )
        self.assertEqual(included, (0, 1))
        self.assertIsNotNone(global_evaluation)
        self.assertEqual(global_evaluation.direction, GROWTH)

        large_drop = np.asarray(
            [100, 110, 120, 130, 80, 90, 100, 110],
            dtype=float,
        )
        rows, evaluations = _evaluate_selected_regimes(
            large_drop,
            dates,
            segments,
            thresholds,
            config,
        )
        included, global_evaluation = _select_global_regime_indices(
            large_drop,
            rows,
            evaluations,
            thresholds,
            config,
        )
        self.assertEqual(included, (1,))
        self.assertIsNotNone(global_evaluation)
        self.assertEqual(global_evaluation.direction, GROWTH)

    def test_flat_bridge_is_single_internal_and_noise_is_not_flat(self) -> None:
        """Разрешить один спокойный FLAT только между трендами одного знака.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если FLAT-policy или noise guard нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        thresholds = TrendThresholds()
        config = TrendModelConfig(trend_search_method="most_recent_cp")
        one_flat = np.asarray(
            [
                100, 110, 120, 130,
                135, 135, 135, 135,
                140, 150, 160, 170,
            ],
            dtype=float,
        )
        rows, evaluations = _evaluate_selected_regimes(
            one_flat,
            list(range(12)),
            [(0, 4), (4, 8), (8, 12)],
            thresholds,
            config,
        )
        self.assertEqual(
            [row["local_regime_class"] for row in rows],
            [GROWTH, LOCAL_FLAT, GROWTH],
        )
        included, evaluation = _select_global_regime_indices(
            one_flat,
            rows,
            evaluations,
            thresholds,
            config,
        )
        self.assertEqual(included, (0, 1, 2))
        self.assertEqual(evaluation.direction, GROWTH)

        two_flats = np.asarray(
            [
                100, 110, 120, 130,
                134, 134, 134, 134,
                136, 136, 136, 136,
                140, 150, 160, 170,
            ],
            dtype=float,
        )
        rows, evaluations = _evaluate_selected_regimes(
            two_flats,
            list(range(16)),
            [(0, 4), (4, 8), (8, 12), (12, 16)],
            thresholds,
            config,
        )
        included, _ = _select_global_regime_indices(
            two_flats,
            rows,
            evaluations,
            thresholds,
            config,
        )
        self.assertEqual(included, (3,))

        trailing_flat = np.asarray(
            [100, 110, 120, 130, 135, 135, 135, 135],
            dtype=float,
        )
        rows, evaluations = _evaluate_selected_regimes(
            trailing_flat,
            list(range(8)),
            [(0, 4), (4, 8)],
            thresholds,
            config,
        )
        included, evaluation = _select_global_regime_indices(
            trailing_flat,
            rows,
            evaluations,
            thresholds,
            config,
        )
        self.assertEqual(included, tuple())
        self.assertIsNone(evaluation)

        noisy_horizontal = evaluate_trend(
            [100, 200, 200, 100],
            thresholds,
        )
        self.assertEqual(
            _classify_local_regime(noisy_horizontal, config),
            LOCAL_UNCONFIRMED,
        )

    def test_global_summary_and_changepoint_membership_are_additive(self) -> None:
        """Сохранить current summary и добавить структуру/CP глобального роста.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если новая long-диагностика не согласована.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [
            100, 110, 120, 130,
            180, 190, 200, 210,
            260, 270, 280, 290,
        ]
        result = analyze_most_recent_cp_series(values)
        summary = result.summary
        self.assertTrue(summary["current_trend_exists"])
        self.assertTrue(summary["global_trend_exists"])
        self.assertEqual(summary["global_trend_direction"], GROWTH)
        self.assertGreaterEqual(
            summary["global_trend_length"],
            summary["current_trend_length"],
        )
        self.assertEqual(
            summary["global_trend_changepoint_count"],
            sum(row["cp_inside_global_trend"] for row in result.changepoints),
        )
        self.assertEqual(
            len(json.loads(summary["global_trend_structure_json"])),
            summary["global_trend_regime_count"],
        )

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
