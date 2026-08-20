"""Проверки компактной менеджерской витрины GMV-трендов."""

from __future__ import annotations

import unittest

import pandas as pd

from gmv_anomaly.trend_manager_output import build_manager_trend_output
from gmv_anomaly.trend_most_recent_cp import LEVEL_AND_SLOPE, LEVEL_SHIFT


class ManagerTrendOutputTests(unittest.TestCase):
    """Проверить сборку менеджерской строки из готовых трендовых таблиц."""

    def test_output_describes_reversal_flat_and_dynamic_dimensions(self) -> None:
        """Проверить полный контракт менеджерского вывода для одного сегмента.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если витрина теряет согласованные поля или меняет их смысл.

        Examples:
            >>> ManagerTrendOutputTests('test_output_describes_reversal_flat_and_dynamic_dimensions').run().wasSuccessful()
            True
        """

        dates = list(range(17))
        segment_values = [60.0, 50.0, 40.0, 30.0, 35.0, 45.0, 55.0, 65.0,
                          65.0, 65.0, 65.0, 65.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        panel_rows = []
        for date, segment_gmv in zip(dates, segment_values):
            panel_rows.extend(
                [
                    {
                        "segment_id": "total",
                        "slice_depth": 0,
                        "cal_date": date,
                        "gmv": 200.0,
                        "geo": None,
                        "new_attribute": None,
                    },
                    {
                        "segment_id": "segment",
                        "slice_depth": 2,
                        "cal_date": date,
                        "gmv": segment_gmv,
                        "geo": "РФ",
                        "new_attribute": "X",
                    },
                ]
            )
        panel = pd.DataFrame(panel_rows)
        summary = pd.DataFrame(
            [
                {
                    "segment_id": "segment",
                    "segment_key": "geo=РФ × new_attribute=X",
                    "segment_level": "test",
                    "slice_depth": 2,
                    "trend_eligible": True,
                    "trend_selected": True,
                    "current_trend_direction": "GROWTH",
                    "global_trend_direction": "GROWTH",
                    "global_trend_start_date": 4,
                    "global_trend_end_date": 16,
                    "global_trend_length": 13,
                    "global_trend_start_gmv": 35.0,
                    "global_trend_end_gmv": 120.0,
                    "global_trend_gmv_change_abs": 85.0,
                    "global_trend_gmv_change_relative": 85.0 / 35.0,
                    "structural_change_detected": True,
                    "structural_change_type": LEVEL_AND_SLOPE,
                    "level_shift": 10.0,
                }
            ]
        )
        segmentation = pd.DataFrame(
            [
                {
                    "segment_id": "segment",
                    "segment_index": 0,
                    "start_date": 0,
                    "end_date": 3,
                    "points": 4,
                    "local_start_gmv": 60.0,
                    "local_end_gmv": 30.0,
                    "local_gmv_change_abs": -30.0,
                    "local_gmv_change_relative": -0.5,
                    "local_regime_class": "DECLINE",
                    "in_global_trend": False,
                },
                {
                    "segment_id": "segment",
                    "segment_index": 1,
                    "start_date": 4,
                    "end_date": 7,
                    "points": 4,
                    "local_start_gmv": 35.0,
                    "local_end_gmv": 65.0,
                    "local_gmv_change_abs": 30.0,
                    "local_gmv_change_relative": 30.0 / 35.0,
                    "local_regime_class": "GROWTH",
                    "in_global_trend": True,
                },
                {
                    "segment_id": "segment",
                    "segment_index": 2,
                    "start_date": 8,
                    "end_date": 11,
                    "points": 4,
                    "local_start_gmv": 65.0,
                    "local_end_gmv": 65.0,
                    "local_gmv_change_abs": 0.0,
                    "local_gmv_change_relative": 0.0,
                    "local_regime_class": "FLAT",
                    "in_global_trend": True,
                },
                {
                    "segment_id": "segment",
                    "segment_index": 3,
                    "start_date": 12,
                    "end_date": 16,
                    "points": 5,
                    "local_start_gmv": 80.0,
                    "local_end_gmv": 120.0,
                    "local_gmv_change_abs": 40.0,
                    "local_gmv_change_relative": 0.5,
                    "local_regime_class": "GROWTH",
                    "in_global_trend": True,
                },
            ]
        )
        changepoints = pd.DataFrame(
            [
                {
                    "segment_id": "segment",
                    "current_regime_index": 1,
                    "structural_change_type": LEVEL_SHIFT,
                    "level_shift": 5.0,
                    "classification_previous_slope_ols": -10.0,
                    "classification_current_slope_ols": 10.0,
                },
                {
                    "segment_id": "segment",
                    "current_regime_index": 2,
                    "structural_change_type": LEVEL_SHIFT,
                    "level_shift": 0.0,
                    "classification_previous_slope_ols": 10.0,
                    "classification_current_slope_ols": 0.0,
                },
                {
                    "segment_id": "segment",
                    "current_regime_index": 3,
                    "structural_change_type": LEVEL_AND_SLOPE,
                    "level_shift": 10.0,
                    "classification_previous_slope_ols": 0.0,
                    "classification_current_slope_ols": 10.0,
                },
            ]
        )

        result = build_manager_trend_output(
            summary,
            panel,
            segmentation,
            changepoints,
            ["geo", "new_attribute"],
        )
        row = result.iloc[0]

        self.assertEqual(row["geo"], "РФ")
        self.assertEqual(row["new_attribute"], "X")
        self.assertEqual(row["trend_direction"], "Рост")
        self.assertEqual(row["global_trend_reversal_detected"], "Да")
        self.assertEqual(row["pre_global_trend_direction"], "Падение")
        self.assertEqual(
            row["global_trend_structure"],
            "Разворот: падение (4) → рост (4) → FLAT (4) → сдвиг вверх и ускорение роста (5)",
        )
        self.assertEqual(row["global_trend_total_gmv_share_dynamics"], "17,5% → 60%")
        self.assertAlmostEqual(float(row["global_trend_total_gmv_share_change_pp"]), 42.5)
        self.assertAlmostEqual(float(row["local_trend_avg_gmv_change_per_period"]), 10.0)
        self.assertAlmostEqual(float(row["global_trend_avg_gmv_change_per_period"]), 85.0 / 12.0)
        self.assertAlmostEqual(float(row["last_local_trend_contribution_to_global_change_pct"]), 40.0 / 85.0 * 100.0)
        self.assertEqual(row["local_trend_contributions_to_global_change"], "+35,3% → +0% → +47,1%")
        self.assertEqual(row["trend_selected"], "Да")


if __name__ == "__main__":
    unittest.main()
