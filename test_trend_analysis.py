"""Регрессионные, QA- и invariance-тесты независимого анализа GMV-тренда."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from gmv_anomaly.data_preparation import build_full_week_grid
from gmv_anomaly.trend_analysis import (
    DECLINE,
    GROWTH,
    NO_DIRECTION,
    TrendChangeEvaluation,
    TrendEvaluation,
    TrendThresholds,
    TrendWindowEvaluation,
    _select_current_suffix_trend,
    analyze_segment_trend,
    build_trend_analysis,
    evaluate_change_candidates,
    evaluate_suffix_trends,
    evaluate_trend,
    fit_continuous_piecewise_model,
    fit_single_linear_model,
    select_change_point,
    theil_sen_slope,
    trim_leading_zero_history,
)


def _panel(values: list[float], dates: list[int] | None = None) -> pd.DataFrame:
    """Построить минимальную готовую панель одного сегмента для unit-теста.

    Args:
        values: GMV по периодам.
        dates: Необязательная ось; по умолчанию ``0..N-1``.

    Returns:
        DataFrame контракта ``analyze_segment_trend``.

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


def _piecewise_values(
    points: int,
    k: int,
    *,
    intercept: float = 100.0,
    slope_before: float = 10.0,
    slope_after: float = -10.0,
) -> list[float]:
    """Сгенерировать точную непрерывную hinge-модель с известным k.

    Args:
        points: Полная длина ряда.
        k: Breakpoint в формуле ``max(0, t-k)``.
        intercept: Уровень при ``t=0``.
        slope_before: Наклон до точки.
        slope_after: Наклон после точки.

    Returns:
        Значения точной кусочно-линейной модели.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _piecewise_values(8, 4)[:5]
        [100.0, 110.0, 120.0, 130.0, 120.0]
    """

    slope_change = slope_after - slope_before
    return [
        intercept
        + slope_before * time
        + slope_change * max(0, time - (k - 1))
        for time in range(points)
    ]


def _confirmed(direction: str, slope: float) -> TrendEvaluation:
    """Создать подтверждённый результат для изолированного теста выбора.

    Args:
        direction: GROWTH или DECLINE.
        slope: Наклон с согласованным знаком.

    Returns:
        Валидный TrendEvaluation.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _confirmed(GROWTH, 1.0).trend_exists
        True
    """

    return TrendEvaluation(
        points=4,
        trend_exists=True,
        direction=direction,
        slope=slope,
        relative_slope=slope / 100.0,
        total_change=3.0 * slope / 100.0,
        direction_count_share=1.0,
        direction_movement_share=1.0,
        trend_to_noise=math.inf,
        typical_scale=100.0,
        intercept=100.0,
        residual_mad=0.0,
        noise_scale=0.0,
        noise_scale_source="PERFECT_FIT",
        status="TREND",
    )


def _candidate(k: int, f_statistic: float) -> TrendChangeEvaluation:
    """Создать валидного кандидата для точного теста near-best правила.

    Args:
        k: Точка разделения.
        f_statistic: Значение F.

    Returns:
        Валидную смену GROWTH_TO_DECLINE.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _candidate(4, 10.0).valid_change
        True
    """

    return TrendChangeEvaluation(
        k=k,
        left_points=k,
        right_points=4,
        rss_single=10.0,
        rss_piecewise=1.0,
        f_statistic=f_statistic,
        left_trend=_confirmed(GROWTH, 1.0),
        right_trend=_confirmed(DECLINE, -1.0),
        passes_f_threshold=True,
        passes_direction_change=True,
        valid_change=True,
    )


# [ADDED] Тесты сгруппированы по T01–T50 из постановки.
class TrendAnalysisTests(unittest.TestCase):
    """Проверить математический контракт и интеграцию поверх full week grid."""

    def test_t01_t04_clean_and_noisy_growth_decline(self) -> None:
        """Проверить чистые направления и небольшие встречные движения.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T01–T04 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        cases = (
            ([100, 110, 120, 130, 140], GROWTH),
            ([200, 190, 180, 170, 160], DECLINE),
            ([120, 112, 105, 108, 98, 90, 92, 82], DECLINE),
            ([80, 88, 95, 92, 102, 110, 108, 118], GROWTH),
        )
        # [FIXED] T03–T04 проверяют формулу исходного MVP-порога 60% / 70%,
        # независимо от пользовательской настройки production-порогов.
        mvp_thresholds = TrendThresholds(
            min_direction_count_share=0.60,
            min_direction_movement_share=0.70,
        )
        for values, direction in cases:
            with self.subTest(direction=direction, values=values):
                result = evaluate_trend(values, mvp_thresholds)
                self.assertTrue(result.trend_exists)
                self.assertEqual(result.direction, direction)

    def test_initial_change_direction_is_required_but_later_reversal_is_allowed(
        self,
    ) -> None:
        """Требовать знак первой пары, не запрещая дальнейшее встречное движение.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если правило первого изменения нарушено.

        Examples:
            >>> # Запускается через unittest.
        """

        first_change_against_growth = [100, 90, 130, 190, 250]
        rejected = evaluate_trend(first_change_against_growth)
        self.assertGreater(rejected.slope, 0.0)
        self.assertEqual(rejected.first_change, -10.0)
        self.assertFalse(rejected.passes_initial_direction)
        self.assertFalse(rejected.trend_exists)
        self.assertEqual(rejected.status, "INITIAL_DIRECTION_MISMATCH")

        later_reversal = [100, 140, 130, 190, 250]
        accepted = evaluate_trend(later_reversal)
        self.assertGreater(accepted.slope, 0.0)
        self.assertEqual(accepted.first_change, 40.0)
        self.assertTrue(accepted.passes_initial_direction)
        self.assertTrue(accepted.trend_exists)
        self.assertEqual(accepted.direction, GROWTH)

    def test_t05_t08_flat_noise_outliers_and_constant(self) -> None:
        """Не объявлять тренд на шуме, выбросе и постоянном уровне.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T05–T08 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        cases = (
            [100, 103, 98, 102, 99, 101],
            [100, 101, 99, 100, 150, 101, 100],
            [100, 101, 99, 100, 50, 101, 100],
            [100, 100.2, 99.8, 100.1, 99.9, 100.0],
        )
        for values in cases:
            with self.subTest(values=values):
                self.assertFalse(evaluate_trend(values).trend_exists)
        constant = evaluate_trend([100] * 8)
        self.assertEqual(constant.slope, 0.0)
        self.assertEqual(constant.direction_count_share, 0.0)
        self.assertEqual(constant.direction_movement_share, 0.0)
        self.assertEqual(constant.direction, NO_DIRECTION)

    def test_t09_t10_zero_mad_perfect_and_mae_fallback(self) -> None:
        """Различать идеальную линию и вырожденный MAD с остатком.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T09–T10 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        exact = evaluate_trend([100, 110, 120, 130])
        self.assertTrue(math.isinf(exact.trend_to_noise))
        self.assertEqual(exact.noise_scale_source, "PERFECT_FIT")
        fallback = evaluate_trend([100, 110, 120, 130, 200])
        self.assertEqual(fallback.residual_mad, 0.0)
        self.assertEqual(fallback.noise_scale_source, "MAE_FALLBACK")
        self.assertTrue(math.isfinite(fallback.trend_to_noise))

    def test_t11_t15_minimum_and_unbounded_history_length(self) -> None:
        """Поддержать 3, 4, 13, 14 и 20 точек без лимита 13.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T11–T15 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        self.assertEqual(
            evaluate_trend([100, 110, 120]).status,
            "INSUFFICIENT_HISTORY",
        )
        self.assertTrue(evaluate_trend([100, 110, 120, 130]).trend_exists)
        self.assertTrue(evaluate_trend([130, 120, 110, 100]).trend_exists)
        for points in (13, 14, 20):
            values = [100.0 + 10.0 * index for index in range(points)]
            result = analyze_segment_trend(_panel(values), list(range(points)))
            self.assertEqual(result.summary["history_points_used"], points)
            self.assertEqual(result.window_diagnostics[-1]["window_length"], points)
            self.assertEqual(
                [row["k"] for row in result.change_diagnostics],
                list(range(4, points - 3)),
            )

    def test_t16_t20_leading_internal_and_trailing_zeros(self) -> None:
        """Обрезать только ведущие нули один раз на полном ряду.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T16–T20 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        values, dates = trim_leading_zero_history(
            [0, 0, 0, 100, 110, 120, 130],
            list(range(7)),
        )
        self.assertEqual(values, [100.0, 110.0, 120.0, 130.0])
        self.assertEqual(dates, [3, 4, 5, 6])
        leading_result = analyze_segment_trend(
            _panel([0, 0, 0, 100, 110, 120, 130]),
            list(range(7)),
        )
        self.assertEqual(leading_result.summary["history_points_raw"], 7)
        self.assertEqual(leading_result.summary["history_points_used"], 4)
        self.assertEqual(
            [row["window_length"] for row in leading_result.window_diagnostics],
            [4],
        )
        for raw, expected in (
            ([100, 110, 0, 90, 80], [100, 110, 0, 90, 80]),
            ([100, 90, 80, 0], [100, 90, 80, 0]),
            ([100, 110, 0, 0, 80, 90], [100, 110, 0, 0, 80, 90]),
        ):
            self.assertEqual(trim_leading_zero_history(raw)[0], expected)
        all_zero = analyze_segment_trend(_panel([0] * 8), list(range(8)))
        self.assertEqual(all_zero.summary["status"], "NO_ACTIVE_HISTORY")
        self.assertFalse(all_zero.summary["current_trend_exists"])
        empty_tables = build_trend_analysis(_panel([0] * 8), list(range(8)))
        self.assertIn("trend_exists", empty_tables["trend_window_diagnostics"].columns)
        self.assertIn("selected", empty_tables["trend_change_diagnostics"].columns)

    def test_t21_missing_grid_row_is_real_internal_zero(self) -> None:
        """Переиспользовать build_full_week_grid для отсутствующей строки.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T21 или preprocessing-контракт нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        dates = [1, 8, 15, 22, 29]
        rows = []
        for date in dates:
            rows.append(
                {
                    "segment_id": "total",
                    "segment_key": "ИТОГО",
                    "segment_level": "ИТОГО",
                    "slice_depth": 0,
                    "geo": None,
                    "cal_date": date,
                    "gmv": 1_000.0,
                }
            )
        for date, gmv in ((1, 100.0), (15, 90.0), (22, 80.0), (29, 70.0)):
            rows.append(
                {
                    "segment_id": "segment",
                    "segment_key": "geo=A",
                    "segment_level": "geo",
                    "slice_depth": 1,
                    "geo": "A",
                    "cal_date": date,
                    "gmv": gmv,
                }
            )
        panel = build_full_week_grid(pd.DataFrame(rows), ["geo"], dates)
        segment = panel[panel["segment_id"].eq("segment")]
        missing = segment[segment["cal_date"].eq(8)].iloc[0]
        self.assertEqual(float(missing["gmv"]), 0.0)
        self.assertTrue(bool(missing["row_missing_in_source"]))
        result = analyze_segment_trend(segment, dates)
        self.assertEqual(result.summary["history_points_used"], 5)
        self.assertEqual(segment.sort_values("cal_date")["gmv"].tolist(), [100, 0, 90, 80, 70])

    def test_t22_suffix_direction_starts_at_first_confirmed_window(self) -> None:
        """Выбрать DECLINE по 5 точкам, когда последние 4 неустойчивы.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T22 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        values = [160, 150, 140, 130, 135, 125]
        result = analyze_segment_trend(_panel(values), list(range(6)))
        states = [
            (row["window_length"], row["trend_exists"], row["direction"])
            for row in result.window_diagnostics
        ]
        self.assertEqual(
            states,
            [(4, False, NO_DIRECTION), (5, True, DECLINE), (6, True, DECLINE)],
        )
        self.assertEqual(result.summary["current_trend_direction"], DECLINE)
        self.assertEqual(result.summary["current_trend_length"], 6)

    def test_t23_no_trend_does_not_stop_but_opposite_trend_does(self) -> None:
        """Проверить точное правило DECLINE/NONE/DECLINE/NONE/GROWTH.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T23 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        no_trend = TrendEvaluation(
            points=5,
            trend_exists=False,
            direction=NO_DIRECTION,
            slope=0.0,
            relative_slope=0.0,
            total_change=0.0,
            direction_count_share=0.0,
            direction_movement_share=0.0,
            trend_to_noise=0.0,
            typical_scale=100.0,
            intercept=100.0,
            residual_mad=0.0,
            noise_scale=0.0,
            noise_scale_source="PERFECT_FIT",
            status="NO_TREND",
        )
        windows = [
            TrendWindowEvaluation(4, _confirmed(DECLINE, -4.0)),
            TrendWindowEvaluation(5, no_trend),
            TrendWindowEvaluation(6, _confirmed(DECLINE, -3.0)),
            TrendWindowEvaluation(7, no_trend),
            TrendWindowEvaluation(8, _confirmed(GROWTH, 2.0)),
        ]
        length, evaluation = _select_current_suffix_trend(windows)
        self.assertEqual(length, 6)
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation.direction, DECLINE)

    def test_t24_t26_recent_decline_not_absorbed_or_counted_as_confidence(self) -> None:
        """Сохранить свежий спад независимо от старого роста и числа окон.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T24–T26 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        values = _piecewise_values(12, 6, slope_before=8.0, slope_after=-10.0)
        result = build_trend_analysis(_panel(values), list(range(12)))
        self.assertEqual(
            set(result),
            {
                "trend_summary",
                "trend_window_diagnostics",
                "trend_change_diagnostics",
            },
        )
        self.assertIn("F", result["trend_change_diagnostics"].columns)
        self.assertIn("selected", result["trend_change_diagnostics"].columns)
        summary = result["trend_summary"].iloc[0]
        self.assertEqual(summary["current_trend_direction"], DECLINE)
        self.assertNotIn("confidence", summary.index)
        self.assertNotIn("confirming_window_count", summary.index)
        noisy_tail = [200, 190, 180, 170, 160, 150, 140, 130, 132]
        tail_summary = analyze_segment_trend(
            _panel(noisy_tail), list(range(len(noisy_tail)))
        ).summary
        self.assertNotEqual(tail_summary["current_trend_direction"], GROWTH)

    def test_t27_t28_growth_decline_and_decline_growth_change(self) -> None:
        """Определить обе симметричные смены направления.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T27–T28 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        for before, after, expected in (
            (10.0, -10.0, "GROWTH_TO_DECLINE"),
            (-10.0, 10.0, "DECLINE_TO_GROWTH"),
        ):
            values = _piecewise_values(
                9,
                4,
                intercept=200.0,
                slope_before=before,
                slope_after=after,
            )
            summary = analyze_segment_trend(_panel(values), list(range(9))).summary
            self.assertTrue(summary["change_detected"])
            self.assertEqual(summary["change_type"], expected)
            self.assertLessEqual(
                int(summary["current_trend_length"]),
                9 - int(summary["change_k"]),
            )

    def test_t29_t30_earliest_and_latest_allowed_change(self) -> None:
        """Поддержать k=4 и k=N-4 без off-by-one.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T29–T30 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        earliest = analyze_segment_trend(
            _panel(_piecewise_values(8, 4)), list(range(8))
        )
        self.assertEqual(earliest.summary["change_k"], 4)
        for points in (13, 14):
            k = points - 4
            values = _piecewise_values(points, k)
            summary = analyze_segment_trend(_panel(values), list(range(points))).summary
            self.assertEqual(summary["change_k"], k)

    def test_t31_t33_minimum_right_regime_contract(self) -> None:
        """Не искать смену до 8 точек и разрешить последние четыре точки.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T31–T33 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        seven = analyze_segment_trend(
            _panel([100, 110, 120, 130, 140, 150, 160]), list(range(7))
        )
        self.assertTrue(seven.summary["current_trend_exists"])
        self.assertEqual(len(seven.change_diagnostics), 0)
        # Последние три наблюдения снижаются, но допустимое правое окно из
        # четырёх точек захватывает предшествующий рост и тренд не подтверждает.
        last_three = [100, 110, 120, 130, 140, 150, 180, 160, 140]
        self.assertFalse(
            analyze_segment_trend(_panel(last_three), list(range(9))).summary[
                "change_detected"
            ]
        )
        points = 13
        k = points - 4
        late = analyze_segment_trend(
            _panel(_piecewise_values(points, k)), list(range(points))
        ).summary
        self.assertTrue(late["change_detected"])
        self.assertEqual(late["change_k"], k)

    def test_t34_t38_reject_speed_only_noise_one_sided_and_level_shift(self) -> None:
        """Не путать смену скорости/уровня и плохую сторону со сменой направления.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T34–T38 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        same_growth = _piecewise_values(
            9,
            4,
            intercept=50.0,
            slope_before=10.0,
            slope_after=6.0,
        )
        same_decline = _piecewise_values(
            9,
            4,
            intercept=250.0,
            slope_before=-10.0,
            slope_after=-6.0,
        )
        one_line_noise = [100, 110, 119, 131, 140, 149, 161, 170, 180]
        random_noise = [100, 106, 95, 103, 98, 108, 96, 102, 99]
        one_side_flat = _piecewise_values(
            9,
            4,
            intercept=100.0,
            slope_before=0.0,
            slope_after=10.0,
        )
        level_shift = [100, 100, 100, 100, 70, 70, 70, 70]
        for values in (
            same_growth,
            same_decline,
            one_line_noise,
            random_noise,
            one_side_flat,
            level_shift,
        ):
            with self.subTest(values=values):
                summary = analyze_segment_trend(
                    _panel(values), list(range(len(values)))
                ).summary
                self.assertFalse(summary["change_detected"])

    def test_t39_t40_strong_recent_change_requires_two_stable_sides(self) -> None:
        """Подтвердить сильный свежий спад, но не слабые последние колебания.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T39–T40 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        strong_recent = _piecewise_values(
            12,
            6,
            intercept=100.0,
            slope_before=3.0,
            slope_after=-15.0,
        )
        summary = analyze_segment_trend(
            _panel(strong_recent), list(range(12))
        ).summary
        self.assertEqual(summary["current_trend_direction"], DECLINE)
        self.assertEqual(summary["change_type"], "GROWTH_TO_DECLINE")
        weak_tail = [100, 120, 140, 160, 180, 200, 198, 202, 199, 201]
        weak_summary = analyze_segment_trend(
            _panel(weak_tail), list(range(10))
        ).summary
        self.assertFalse(weak_summary["change_detected"])

    def test_t41_mvp_returns_at_most_one_change_for_13_and_20_points(self) -> None:
        """Зафиксировать ограничение одной смены при нескольких режимах.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T41 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        series = (
            [100, 120, 140, 160, 180, 150, 120, 90, 60, 90, 120, 150, 180],
            [
                200,
                220,
                240,
                260,
                280,
                250,
                220,
                190,
                160,
                130,
                100,
                130,
                160,
                190,
                220,
                250,
                280,
                310,
                340,
                370,
            ],
        )
        for values in series:
            result = analyze_segment_trend(
                _panel(values), list(range(len(values)))
            )
            self.assertLessEqual(
                sum(bool(row["selected"]) for row in result.change_diagnostics),
                1,
            )

    def test_t42_outlier_near_true_change_keeps_robust_sides(self) -> None:
        """Не уводить breakpoint далеко из-за одного умеренного выброса.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T42 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        expected_k = 6
        base = _piecewise_values(13, expected_k)
        for index, delta in ((5, 20.0), (6, -20.0), (7, 20.0)):
            values = base.copy()
            values[index] += delta
            summary = analyze_segment_trend(
                _panel(values), list(range(13))
            ).summary
            self.assertTrue(summary["change_detected"])
            self.assertLessEqual(abs(int(summary["change_k"]) - expected_k), 1)

    def test_t43_t44_single_outlier_does_not_create_direction_change(self) -> None:
        """Не создавать GROWTH_TO_DECLINE из одиночного выброса любого знака.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T43–T44 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        for outlier in (250.0, 0.0):
            values = [100.0] * 13
            values[6] = outlier
            summary = analyze_segment_trend(
                _panel(values), list(range(13))
            ).summary
            self.assertFalse(summary["change_detected"])

    def test_t45_t46_f_degenerate_cases(self) -> None:
        """Вернуть F=0 для одной линии и F=inf для точной смены наклона.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T45–T46 нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        one_line = evaluate_change_candidates([100 + 10 * i for i in range(9)])
        self.assertTrue(all(candidate.f_statistic == 0.0 for candidate in one_line))
        exact = evaluate_change_candidates(_piecewise_values(9, 4))
        selected = select_change_point(exact)
        self.assertIsNotNone(selected)
        self.assertTrue(math.isinf(selected.f_statistic))
        self.assertGreater(selected.rss_single, 0.0)
        self.assertAlmostEqual(selected.rss_piecewise, 0.0, places=20)

    def test_t47_all_dynamic_k_values(self) -> None:
        """Оценить ровно k=4,...,N-4 для пяти разных N.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T47 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        for points in (8, 9, 13, 14, 20):
            values = [100.0 + index for index in range(points)]
            self.assertEqual(
                [candidate.k for candidate in evaluate_change_candidates(values)],
                list(range(4, points - 3)),
            )

    def test_t48_breakpoint_dates_use_left_exclusive_right_split(self) -> None:
        """Вернуть даты для left=values[:k] и right=values[k:].

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T48 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        dates = [100 + 7 * index for index in range(9)]
        result = analyze_segment_trend(
            _panel(_piecewise_values(9, 4), dates), dates
        )
        self.assertEqual(result.summary["change_k"], 4)
        self.assertEqual(result.summary["change_left_end_date"], dates[3])
        self.assertEqual(result.summary["change_right_start_date"], dates[4])

    def test_t49_near_best_prefers_later_candidate(self) -> None:
        """Выбрать поздний k при F не ниже 95% лучшего.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T49 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        selected = select_change_point([_candidate(4, 100.0), _candidate(5, 95.0)])
        self.assertIsNotNone(selected)
        self.assertEqual(selected.k, 5)

    def test_t50_later_materially_worse_candidate_loses(self) -> None:
        """Не применять свежесть при F ниже 95% лучшего.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если T50 нарушен.

        Examples:
            >>> # Запускается через unittest.
        """

        selected = select_change_point([_candidate(4, 100.0), _candidate(5, 94.9)])
        self.assertIsNotNone(selected)
        self.assertEqual(selected.k, 4)

    def test_scale_invariance_and_signed_absolute_unit_slope(self) -> None:
        """Сохранить относительные решения при умножении GMV на константу.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если масштабная инвариантность нарушена.

        Examples:
            >>> # Запускается через unittest.
        """

        values = _piecewise_values(13, 6)
        scaled = [7.0 * value for value in values]
        base_result = analyze_segment_trend(_panel(values), list(range(13)))
        scaled_result = analyze_segment_trend(_panel(scaled), list(range(13)))
        base = base_result.summary
        other = scaled_result.summary
        for key in (
            "current_trend_exists",
            "current_trend_direction",
            "change_detected",
            "change_k",
        ):
            self.assertEqual(base[key], other[key])
        for key in (
            "trend_slope_relative",
            "trend_total_change",
            "direction_count_share",
            "direction_movement_share",
            "trend_to_noise",
        ):
            if math.isinf(float(base[key])):
                self.assertTrue(math.isinf(float(other[key])))
            else:
                self.assertAlmostEqual(float(base[key]), float(other[key]), places=10)
        self.assertAlmostEqual(
            float(other["trend_slope_abs"]),
            7.0 * float(base["trend_slope_abs"]),
            places=10,
        )

    def test_mirror_symmetry_and_theil_sen_pair_median(self) -> None:
        """Проверить зеркальность направлений и собственный Theil–Sen.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если симметрия или медиана наклонов нарушена.

        Examples:
            >>> # Запускается через unittest.
        """

        growth = [100, 110, 121, 130, 140, 151]
        decline = [300.0 - value for value in growth]
        growth_result = evaluate_trend(growth)
        decline_result = evaluate_trend(decline)
        self.assertEqual(growth_result.direction, GROWTH)
        self.assertEqual(decline_result.direction, DECLINE)
        self.assertAlmostEqual(growth_result.slope, -decline_result.slope)
        self.assertAlmostEqual(theil_sen_slope([0, 2, 4, 9]), 2.5)

    def test_model_contract_and_threshold_validation(self) -> None:
        """Проверить непрерывную hinge-модель и fail-fast конфигурации.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если модель или валидация порогов нарушены.

        Examples:
            >>> # Запускается через unittest.
        """

        values = _piecewise_values(9, 4)
        piecewise_fit = fit_continuous_piecewise_model(values, 4)
        self.assertAlmostEqual(piecewise_fit.rss, 0.0, places=20)
        self.assertAlmostEqual(piecewise_fit.coefficients[1], 10.0)
        self.assertAlmostEqual(
            piecewise_fit.coefficients[1] + piecewise_fit.coefficients[2],
            -10.0,
        )
        self.assertEqual(values[:4], [100.0, 110.0, 120.0, 130.0])
        self.assertEqual(values[4:8], [120.0, 110.0, 100.0, 90.0])
        self.assertGreater(fit_single_linear_model(values).rss, 0.0)
        with self.assertRaises(ValueError):
            TrendThresholds(min_trend_points=3)
        with self.assertRaises(ValueError):
            TrendThresholds(near_best_change_ratio=1.1)
        with self.assertRaises(ValueError):
            evaluate_trend([100, math.nan, 120, 130])


if __name__ == "__main__":
    unittest.main()
