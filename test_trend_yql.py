"""Проверки runtime-контракта и генератора product-YQL для GMV-трендов."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from gmv_anomaly.build_trend_yql import (
    _yt_output_label,
    build_yql,
    render_yql,
)
from gmv_anomaly.config import DIM_COLUMNS
from gmv_anomaly.trend_manager_output import (
    MANAGER_TREND_COLUMN_LABELS,
    manager_trend_columns,
)
from gmv_anomaly.trend_udf_runtime import (
    TREND_UDF_OUTPUT_SCHEMA,
    _MANAGER_TREND_YQL_TYPES,
    _manager_output_records,
)


class TrendYqlTests(unittest.TestCase):
    """Проверить совпадение YQL-вывода с менеджерским Excel-контрактом."""

    def test_output_schema_matches_excel_without_segment_id(self) -> None:
        """Проверить 33 поля листа и отсутствие внутреннего segment_id.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если публичная схема YT расходится с Excel-листом.

        Examples:
            >>> TrendYqlTests("test_output_schema_matches_excel_without_segment_id").run().wasSuccessful()
            True
        """

        output_names = [name for name, _ in TREND_UDF_OUTPUT_SCHEMA]
        self.assertEqual(len(output_names), 33)
        self.assertEqual(output_names[0], "segment_name")
        self.assertNotIn("segment_id", output_names)

    def test_manager_records_omit_description_row_and_convert_excel_dates(self) -> None:
        """Проверить преобразование готового dataframe без служебной строки Excel.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если строка или типы расходятся с целевой YT-схемой.

        Examples:
            >>> TrendYqlTests("test_manager_records_omit_description_row_and_convert_excel_dates").run().wasSuccessful()
            True
        """

        row: dict[str, object] = {"segment_id": "technical-id"}
        for column in manager_trend_columns(DIM_COLUMNS):
            if column == "segment_id":
                continue
            yql_type = _MANAGER_TREND_YQL_TYPES[column]
            if column in DIM_COLUMNS:
                row[column] = None
            elif yql_type.startswith("String"):
                row[column] = "значение"
            elif yql_type.startswith("Int64"):
                row[column] = 0
            else:
                row[column] = 1.0

        records = _manager_output_records(pd.DataFrame([row]))

        self.assertEqual(len(records), 1)
        self.assertNotIn("segment_id", records[0])
        self.assertEqual(records[0]["segment_name"], "значение")
        self.assertEqual(records[0]["local_trend_start_date"], 25_569)
        self.assertNotIn("Читаемое описание сегмента.", records[0].values())

    def test_render_yql_writes_only_business_rows(self) -> None:
        """Проверить фильтр периода и отсутствие anomaly-технической строки.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если render добавляет лишнюю строку или неверный путь.

        Examples:
            >>> TrendYqlTests("test_render_yql_writes_only_business_rows").run().wasSuccessful()
            True
        """

        rendered = render_yql(
            input_table="//input",
            output_table="//output",
            period="1W",
            generated_at="2026-08-19T00:00:00+03:00",
            algorithm_version="test-version",
            sources={},
        )

        self.assertIn('WHERE source.period == "1W"', rendered)
        self.assertIn("INSERT INTO `//output` WITH TRUNCATE", rendered)
        self.assertIn("result.segment_name AS `Название сегмента`", rendered)
        self.assertIn(
            "result.global_trend_total_gmv_share_change_pp "
            "AS `Изменение доли в Total GMV, пп`",
            rendered,
        )
        self.assertIn("FROM (\n    REDUCE $input", rendered)
        self.assertNotIn("$result =", rendered)
        self.assertNotIn("FROM $result", rendered)
        self.assertNotIn("Техническая информация", rendered)
        self.assertNotIn("Читаемое описание сегмента.", rendered)

    def test_yt_labels_repeat_excel_headers_without_dots(self) -> None:
        """Проверить минимальную и однозначную нормализацию Excel-заголовков.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если YT-имена отклоняются от Excel-контракта или
                содержат опасные для YQL/DataLens символы.

        Examples:
            >>> TrendYqlTests("test_yt_labels_repeat_excel_headers_without_dots").run().wasSuccessful()
            True
        """

        labels = [_yt_output_label(column) for column, _ in TREND_UDF_OUTPUT_SCHEMA]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all("." not in label and "`" not in label for label in labels))
        for column, _ in TREND_UDF_OUTPUT_SCHEMA:
            expected = MANAGER_TREND_COLUMN_LABELS.get(column, column).replace(
                "п.п.",
                "пп",
            )
            self.assertEqual(_yt_output_label(column), expected)

    def test_build_yql_validates_embedded_import_before_write(self) -> None:
        """Проверить полную локальную сборку embedded Python3 UDF.

        Args:
            Нет аргументов кроме созданного экземпляра.

        Returns:
            None.

        Raises:
            AssertionError: Если генератор не создал самодостаточный YQL.

        Examples:
            >>> TrendYqlTests("test_build_yql_validates_embedded_import_before_write").run().wasSuccessful()
            True
        """

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "trend_test.yql"
            path, _, version = build_yql(output_path)
            rendered = path.read_text(encoding="utf-8")

        self.assertEqual(len(version), 16)
        self.assertIn("gmv_anomaly.trend_udf_runtime", rendered)
        self.assertIn("from gmv_anomaly.trend_udf_runtime import run_algorithm", rendered)


if __name__ == "__main__":
    unittest.main()
