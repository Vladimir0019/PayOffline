"""Пакетные регрессионные тесты модульной реализации поиска GMV-аномалий."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from gmv_anomaly import config as runtime_config
from gmv_anomaly import main as main_module
from gmv_anomaly.anomaly_scoring import (
    apply_local_depth_penalty,
    build_anomaly_candidates,
    build_atomic_coverage,
)
from gmv_anomaly.config import AnomalyThresholds
from gmv_anomaly.data_preparation import (
    build_full_week_grid,
    infer_anomaly_dimension_columns,
    load_history_table,
)
from gmv_anomaly.pipeline import run_pipeline
from gmv_anomaly.set_packing import search_anomal


DATES = [1, 8, 15, 22]


def _history_rows() -> list[dict[str, object]]:
    """Сформировать синтетическую историю с обычным, новым и исчезнувшим сегментами.

    Args:
        Нет аргументов.

    Returns:
        Строки входной таблицы.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> len(_history_rows()) > 4
        True
    """

    rows: list[dict[str, object]] = []
    for cal_date in DATES:
        rows.append(
            {
                "period": "1W",
                "cal_date": cal_date,
                "slice_depth": 0,
                "gmv": 1_000.0,
                "geo": None,
            }
        )
    for cal_date, gmv in zip(DATES, [400.0, 400.0, 400.0, 800.0]):
        rows.append(
            {
                "period": "1W",
                "cal_date": cal_date,
                "slice_depth": 1,
                "gmv": gmv,
                "geo": "A",
            }
        )
    for cal_date in DATES[:-1]:
        rows.append(
            {
                "period": "1W",
                "cal_date": cal_date,
                "slice_depth": 1,
                "gmv": 600.0,
                "geo": "B",
            }
        )
    rows.append(
        {
            "period": "1W",
            "cal_date": DATES[-1],
            "slice_depth": 1,
            "gmv": 200.0,
            "geo": "C",
        }
    )
    # ADDED: Синтетическая витрина соблюдает фиксированный dimension-контракт.
    for row in rows:
        row.setdefault("products", None)
        row.setdefault("merchants_type", None)
        row.setdefault("is_terminal_or_cpqr", None)
    return rows


def _candidate_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Сформировать минимальный контракт кандидатов для Set Packing.

    Args:
        rows: Описания сегментов с измерениями и исходным score.

    Returns:
        Таблица кандидатов до применения depth penalty.

    Raises:
        KeyError: Если отсутствуют обязательные поля строки.

    Examples:
        >>> _candidate_frame([{
        ...     'segment_id': 'a', 'segment_key': 'geo=A', 'slice_depth': 1,
        ...     'geo': 'A', 'product': None, 'score': 2.0,
        ... }]).shape[0]
        1
    """

    normalized: list[dict[str, object]] = []
    for row in rows:
        score = float(row["score"])
        normalized.append(
            {
                "segment_id": row["segment_id"],
                "segment_key": row["segment_key"],
                "segment_level": "test",
                "slice_depth": int(row["slice_depth"]),
                "geo": row["geo"],
                "product": row["product"],
                "passes_initial_anomaly_filter": True,
                "robust_z": score,
                "robust_z_capped": score,
                "abs_z_capped": abs(score),
                "wow_delta_gmv": score * 100.0,
                "abnormal_gmv": score * 100.0,
                "abs_abnormal_gmv": abs(score * 100.0),
                "materiality_share": 1.0,
                "reliability_factor": 1.0,
                "state": "обычный",
            }
        )
    return pd.DataFrame(normalized)


class GmvAnomalyRefactorTests(unittest.TestCase):
    """Проверить ключевые неизменённые контракты GMV-pipeline."""

    def test_main_uses_only_config_values(self) -> None:
        """Проверить передачу всех параметров запуска из ``config.py``.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если main использует значение вне конфигурации.

        Examples:
            >>> # Запускается через unittest.
        """

        control = pd.DataFrame(
            {
                "показатель": [
                    "candidate_count",
                    "selected_count",
                    "double_count_violation_count",
                ],
                "значение": [0, 0, 0],
            }
        )
        with (
            patch.object(main_module, "run_pipeline", return_value={"control": control}) as mocked_pipeline,
            patch("builtins.print"),
        ):
            main_module.main()

        mocked_pipeline.assert_called_once_with(
            input_path=runtime_config.INPUT_PATH,
            output_path=runtime_config.OUTPUT_PATH,
            sheet_name=runtime_config.SHEET_NAME,
            period=runtime_config.PERIOD,
            dim_cols=runtime_config.DIM_COLUMNS,
            current_cal_date=runtime_config.CURRENT_CAL_DATE,
            thresholds=runtime_config.THRESHOLDS,
            tree_output_path=runtime_config.TREE_OUTPUT_PATH,
        )

    def test_dimensions_are_fixed_and_ignore_extra_excel_columns(self) -> None:
        """FIXED: Новая колонка Excel не должна становиться dimension.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если состав dimensions зависит от лишней колонки.

        Examples:
            >>> # Запускается через unittest.
        """

        columns = [*runtime_config.DIM_COLUMNS, "source_system", "segment_id"]
        self.assertEqual(
            infer_anomaly_dimension_columns(pd.DataFrame(columns=columns)),
            list(runtime_config.DIM_COLUMNS),
        )
        self.assertIn("source_system", runtime_config.ANOMALY_TECH_COLUMNS)
        self.assertIn("segment_id", runtime_config.ANOMALY_TECH_COLUMNS)

    def test_loading_panel_and_lifecycle_states(self) -> None:
        """Проверить загрузку, недельную сетку, новый и исчезнувший сегменты.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если контракт подготовки данных изменился.

        Examples:
            >>> # Запускается через unittest.
        """

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "history.csv"
            pd.DataFrame(_history_rows()).to_csv(input_path, index=False)
            history, dims, dates = load_history_table(input_path)
            panel = build_full_week_grid(history, dims, dates)
            candidates, _ = build_anomaly_candidates(
                panel,
                dims,
                dates,
                AnomalyThresholds(),
            )

        self.assertEqual(dims, list(runtime_config.DIM_COLUMNS))
        self.assertEqual(dates, DATES)
        by_key = candidates.set_index("segment_key")
        self.assertEqual(by_key.at["geo=B", "state"], "исчезнувший сегмент")
        self.assertEqual(by_key.at["geo=C", "state"], "новый сегмент")
        missing_b = panel[
            (panel["segment_key"] == "geo=B")
            & (panel["cal_date"] == DATES[-1])
        ].iloc[0]
        self.assertEqual(float(missing_b["gmv"]), 0.0)
        self.assertTrue(bool(missing_b["row_missing_in_source"]))

    def test_loading_rejects_duplicate_segment_date_key(self) -> None:
        """Проверить fail-fast при нарушении уникальности ``segment_id × cal_date``.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если дубликат входного ключа не отклонён.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        rows.append(dict(rows[0]))
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "history_with_duplicate.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "segment_id × cal_date"):
                load_history_table(input_path)

    def test_loading_rejects_slice_depth_inconsistent_with_dimensions(self) -> None:
        """ADDED: Отклонить вход, где depth не соответствует dimension values.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если несогласованная иерархия попадёт в panel.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        next(row for row in rows if row.get("geo") == "A")["slice_depth"] = 2
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "history_with_invalid_depth.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "slice_depth=2, вычисленная глубина=1"):
                load_history_table(input_path)

    def test_atomic_coverage_and_parent_child_set_packing(self) -> None:
        """Проверить coverage, конфликт parent-child и точный выбор решения.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если coverage или оптимальный выбор изменились.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "geo=A",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                },
                {
                    "segment_id": "c1",
                    "segment_key": "geo=A × product=X",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 6.0,
                },
                {
                    "segment_id": "c2",
                    "segment_key": "geo=A × product=Y",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 6.0,
                },
                {
                    "segment_id": "q",
                    "segment_key": "geo=B × product=Z",
                    "slice_depth": 2,
                    "geo": "B",
                    "product": "Z",
                    "score": 4.0,
                },
            ]
        )
        coverage = build_atomic_coverage(candidates, ["geo", "product"])
        self.assertEqual(coverage["p"], frozenset({"c1", "c2"}))
        self.assertTrue(coverage["c1"].isdisjoint(coverage["q"]))

        scored = apply_local_depth_penalty(candidates, coverage)
        final, diagnostics, decision_log = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertEqual(set(final["segment_id"]), {"c1", "c2", "q"})
        self.assertFalse(bool(diagnostics.loc[diagnostics["segment_id"] == "p", "selected"].iloc[0]))
        self.assertFalse(decision_log.empty)

    def test_empty_anomaly_set(self) -> None:
        """Проверить корректный результат при отсутствии eligible-кандидатов.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если пустой результат обрабатывается некорректно.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "a",
                    "segment_key": "geo=A",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 1.0,
                }
            ]
        )
        candidates["passes_initial_anomaly_filter"] = False
        coverage = {"a": frozenset({"a"})}
        scored = apply_local_depth_penalty(candidates, coverage)
        final, diagnostics, decision_log = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertTrue(final.empty)
        self.assertEqual(int(diagnostics["selected"].sum()), 0)
        self.assertFalse(decision_log.empty)

    def test_pipeline_excel_sheet_contract(self) -> None:
        """Проверить end-to-end pipeline и неизменный состав Excel-листов.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если orchestration или Excel-контракт изменились.

        Examples:
            >>> # Запускается через unittest.
        """

        expected_sheets = [
            "00_Параметры_и_контроль",
            "01_Менеджерский_вывод",
            "02_Итог_аномалий",
            "Анализ аномалий",
            "03_История_top",
            "04_Диагностика_кандидатов",
            "05_Пропуски_и_нули",
            "06_Контроль",
            "07_Журнал_set_packing",
        ]
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "history.csv"
            output_path = Path(temp_dir) / "result.xlsx"
            pd.DataFrame(_history_rows()).to_csv(input_path, index=False)
            result = run_pipeline(
                input_path,
                output_path,
                thresholds=AnomalyThresholds(
                    min_anomaly_abs=0.0,
                    min_z_score=0.0,
                    min_materiality_share=0.0,
                ),
            )
            # FIXED: Явно закрываем ExcelFile, чтобы Windows освободил файл до cleanup.
            with pd.ExcelFile(output_path) as workbook:
                sheet_names = workbook.sheet_names

        self.assertEqual(sheet_names, expected_sheets)
        self.assertEqual(
            list(result),
            [
                "history",
                "panel",
                "candidates",
                "final",
                "manager",
                "optimization_decision_log",
                "control",
            ],
        )


if __name__ == "__main__":
    unittest.main()
