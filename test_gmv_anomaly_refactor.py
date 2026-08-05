"""Пакетные регрессионные тесты модульной реализации поиска GMV-аномалий."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from gmv_anomaly import config as runtime_config
from gmv_anomaly import main as main_module
from gmv_anomaly.anomaly_scoring import (
    _enumerate_disjoint_descendant_groups,
    _history_reliability,
    apply_hierarchy_score_adjustment,
    build_anomaly_candidates,
    build_atomic_coverage,
    build_ratio_anomaly_candidates,
    calculate_segment_anomaly,
    calculate_exact_ratio_contribution,
    calculate_ratio_segment_anomaly,
    validate_hierarchy_reconciliation,
)
from gmv_anomaly.config import (
    AnomalyThresholds,
    PILOT_RATIO_METRICS,
    RATIO_METRICS,
)
from gmv_anomaly.data_preparation import (
    build_full_week_grid,
    build_segment_key_and_level,
    infer_anomaly_dimension_columns,
    load_history_table,
    segment_id_from_row,
)
from gmv_anomaly.pipeline import run_pipeline
from gmv_anomaly.reporting import (
    build_anomaly_tree_from_excel,
    build_manager_summary,
    build_ratio_analysis_sheets,
)
from gmv_anomaly.segment_keys import (
    SEGMENT_KEY_SEPARATOR,
    parse_segment_key_parts,
)
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
    # ADDED: Синтетическая витрина соблюдает фиксированный dimension-контракт
    # и обязательную числовую схему upstream YQL.
    for row in rows:
        row.setdefault("products", None)
        row.setdefault("merchants_type", None)
        row.setdefault("is_terminal_or_cpqr", None)
        row.setdefault("tx", 10.0)
        row.setdefault("au", 5.0)
        row.setdefault("am", 2.0)
        row.setdefault("aov", float(row["gmv"]) / float(row["tx"]))
        row.setdefault("tpm", float(row["tx"]) / float(row["am"]))
        row.setdefault("freq", float(row["tx"]) / float(row["au"]))
    return rows


def _ratio_history_rows() -> list[dict[str, object]]:
    """ADDED: Сформировать иерархически согласованную историю ``authzone_tx_share``.

    Args:
        Нет аргументов.

    Returns:
        Строки с готовой долей, числителем и знаменателем.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> any('authzone_tx_share' in row for row in _ratio_history_rows())
        True
    """

    rows = _history_rows()
    numerator_by_segment = {
        None: dict(zip(DATES, [5.0, 5.0, 5.0, 6.0])),
        "A": dict(zip(DATES, [2.0, 2.0, 2.0, 5.0])),
        "B": dict(zip(DATES[:-1], [3.0, 3.0, 3.0])),
        "C": {DATES[-1]: 1.0},
    }
    for row in rows:
        geo = row["geo"]
        row["tx"] = 20.0 if geo is None else 10.0
        numerator = numerator_by_segment[geo][int(row["cal_date"])]
        row["authzone_tx_numerator"] = numerator
        row["authzone_tx_share"] = numerator / float(row["tx"])
        row["aov"] = float(row["gmv"]) / float(row["tx"])
        row["tpm"] = float(row["tx"]) / float(row["am"])
        row["freq"] = float(row["tx"]) / float(row["au"])
    return rows


def _all_ratio_history_rows() -> list[dict[str, object]]:
    """ADDED: Сформировать валидную историю всех восьми относительных метрик.

    Args:
        Нет аргументов.

    Returns:
        Иерархически согласованные строки со всеми компонентами отношений.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> len({spec.value_column for spec in RATIO_METRICS})
        8
    """

    rows = _ratio_history_rows()
    for row in rows:
        tx = float(row["tx"])
        gmv = float(row["gmv"])
        row["tx0"] = tx / 0.8
        row["success_rate"] = 0.8
        ratio_components = {
            "refund_tx": (0.10, tx),
            "payapp_tx": (0.20, tx),
            "split_gmv": (0.10, gmv),
            "credlim_gmv": (0.15, gmv),
            # FIXED: Эти две метрики по бизнес-контракту не ограничены единицей.
            "tips_gmv": (1.20, gmv),
            "cashback_gmv": (1.10, gmv),
        }
        for prefix, (ratio_value, denominator) in ratio_components.items():
            row[f"{prefix}_numerator"] = ratio_value * denominator
            row[f"{prefix}_share"] = ratio_value
    return rows


def _candidate_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Сформировать минимальный контракт кандидатов для Set Packing.

    Args:
        rows: Описания сегментов с измерениями и исходным score.

    Returns:
        Таблица кандидатов до hierarchy-корректировки score.

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
        delta_gmv = float(row.get("delta_gmv", score * 100.0))
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
                "abs_robust_z": abs(score),
                "wow_delta_gmv": delta_gmv,
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
            ratio_trees_output_dir=runtime_config.RATIO_TREES_OUTPUT_DIR,
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
            history, dims, dates = load_history_table(input_path, period="1W")
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
                load_history_table(input_path, period="1W")

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
                load_history_table(input_path, period="1W")

    def test_loading_rejects_invalid_required_numeric_values(self) -> None:
        """ADDED: Отклонить все варианты повреждения обязательных числовых полей.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если невалидное значение удалено молча.

        Examples:
            >>> # Запускается через unittest.
        """

        required_numeric = ("cal_date", "slice_depth", "gmv", "tx", "au", "am")
        invalid_values = ("bad", None, math.nan, math.inf, -math.inf)
        for column in required_numeric:
            for invalid_value in invalid_values:
                with self.subTest(column=column, invalid_value=invalid_value):
                    rows = _history_rows()
                    rows[4][column] = invalid_value
                    with TemporaryDirectory() as temp_dir:
                        input_path = Path(temp_dir) / f"invalid_{column}.csv"
                        pd.DataFrame(rows).to_csv(input_path, index=False)
                        with self.assertRaisesRegex(ValueError, column):
                            load_history_table(input_path, period="1W")

    def test_loading_rejects_empty_input_and_each_missing_required_column(self) -> None:
        """ADDED: Отклонить пустой вход и отсутствие любой обязательной колонки.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если неполная схема проходит загрузку.

        Examples:
            >>> # Запускается через unittest.
        """

        required = ("period", "cal_date", "slice_depth", "gmv", "tx", "au", "am")
        valid_frame = pd.DataFrame(_history_rows())
        with TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "empty.csv"
            pd.DataFrame(columns=valid_frame.columns).to_csv(empty_path, index=False)
            with self.assertRaisesRegex(ValueError, "не осталось строк"):
                load_history_table(empty_path, period="1W")

            for column in required:
                with self.subTest(missing_column=column):
                    missing_path = Path(temp_dir) / f"missing_{column}.csv"
                    valid_frame.drop(columns=column).to_csv(missing_path, index=False)
                    with self.assertRaisesRegex(ValueError, column):
                        load_history_table(missing_path, period="1W")

    def test_loading_requires_single_requested_period(self) -> None:
        """ADDED: Применить период строго и отклонить неоднозначный вход.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если период проигнорирован.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        with TemporaryDirectory() as temp_dir:
            no_period_path = Path(temp_dir) / "no_period.csv"
            pd.DataFrame(rows).drop(columns="period").to_csv(no_period_path, index=False)
            with self.assertRaisesRegex(ValueError, "period"):
                load_history_table(no_period_path, period="1W")

            absent_path = Path(temp_dir) / "absent_period.csv"
            pd.DataFrame(rows).to_csv(absent_path, index=False)
            with self.assertRaisesRegex(ValueError, "не осталось строк"):
                load_history_table(absent_path, period="4W")

            mixed_rows = rows + [{**row, "period": "4W"} for row in rows]
            mixed_path = Path(temp_dir) / "mixed_period.csv"
            pd.DataFrame(mixed_rows).to_csv(mixed_path, index=False)
            with self.assertRaisesRegex(ValueError, "period обязателен"):
                load_history_table(mixed_path, period=None)

            typed_rows = [
                {column: "string" for column in pd.DataFrame(rows).columns},
                *rows,
                *[{**row, "period": "4W"} for row in rows],
            ]
            filtered_path = Path(temp_dir) / "typed_and_mixed.csv"
            pd.DataFrame(typed_rows).to_csv(filtered_path, index=False)
            history, _, _ = load_history_table(filtered_path, period="1W")
            self.assertEqual(history["period"].astype(str).unique().tolist(), ["1W"])
            self.assertEqual(len(history), len(rows))

    def test_total_calendar_step_depends_on_required_period(self) -> None:
        """ADDED: Проверить календарный шаг 7 × число недель в period.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если период ``13W`` валидируется как недельный.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        dates_13w = {date: 1 + index * 7 * 13 for index, date in enumerate(DATES)}
        for row in rows:
            row["period"] = "13W"
            row["cal_date"] = dates_13w[int(row["cal_date"])]

        with TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid_13w.csv"
            pd.DataFrame(rows).to_csv(valid_path, index=False)
            _, _, dates = load_history_table(valid_path, period="13W")
            self.assertEqual(dates, list(dates_13w.values()))

            invalid_rows = [dict(row) for row in rows]
            invalid_rows[0]["cal_date"] = 8
            invalid_path = Path(temp_dir) / "invalid_13w.csv"
            pd.DataFrame(invalid_rows).to_csv(invalid_path, index=False)
            with self.assertRaisesRegex(ValueError, "ожидалось 91 дней"):
                load_history_table(invalid_path, period="13W")

    def test_loading_rejects_dates_outside_total_calendar(self) -> None:
        """ADDED: Не терять строку сегмента с датой вне total-календаря.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если orphan date прошла в panel.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        orphan = dict(next(row for row in rows if row.get("geo") == "A"))
        orphan["cal_date"] = 29
        rows.append(orphan)
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "orphan_date.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "вне total-календаря"):
                load_history_table(input_path, period="1W")

    def test_segment_id_json_is_unambiguous(self) -> None:
        """ADDED: Отличать пропуск, reserved token и разделитель в значении.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если разные dimension tuple дают одинаковый ID.

        Examples:
            >>> # Запускается через unittest.
        """

        missing_id = segment_id_from_row(pd.Series({"x": None, "y": "a|b"}), ["x", "y"])
        literal_id = segment_id_from_row(pd.Series({"x": "∅", "y": "a|b"}), ["x", "y"])
        self.assertNotEqual(missing_id, literal_id)
        self.assertEqual(json.loads(missing_id), [None, "a|b"])
        self.assertEqual(json.loads(literal_id), ["∅", "a|b"])

    def test_missing_grid_row_zeros_only_additive_metrics(self) -> None:
        """ADDED: Восстановить missing row нулями только для аддитивных метрик.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если ratio-метрики искусственно стали нулями.

        Examples:
            >>> # Запускается через unittest.
        """

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "missing_row.csv"
            pd.DataFrame(_history_rows()).to_csv(input_path, index=False)
            history, dims, dates = load_history_table(input_path, period="1W")
            panel = build_full_week_grid(history, dims, dates)

        missing_b = panel[
            panel["segment_key"].eq("geo=B")
            & panel["cal_date"].eq(DATES[-1])
        ].iloc[0]
        self.assertTrue(bool(missing_b["row_missing_in_source"]))
        for metric in ("gmv", "tx", "au", "am"):
            self.assertEqual(float(missing_b[metric]), 0.0)
        for metric in ("aov", "tpm", "freq"):
            self.assertTrue(math.isnan(float(missing_b[metric])))

    def test_ratio_nan_is_preserved_and_does_not_create_false_wow(self) -> None:
        """ADDED: Не превращать неопределённую ratio-метрику в ноль или −100%.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если NULL ratio потерялся.

        Examples:
            >>> # Запускается через unittest.
        """

        for ratio_metric in ("aov", "tpm", "freq"):
            for missing_side_date in (DATES[-2], DATES[-1]):
                with self.subTest(
                    ratio_metric=ratio_metric,
                    missing_side_date=missing_side_date,
                ):
                    rows = _history_rows()
                    row_a = next(
                        row for row in rows
                        if row.get("geo") == "A"
                        and row["cal_date"] == missing_side_date
                    )
                    row_a[ratio_metric] = None
                    with TemporaryDirectory() as temp_dir:
                        input_path = Path(temp_dir) / "ratio_nan.csv"
                        pd.DataFrame(rows).to_csv(input_path, index=False)
                        history, dims, dates = load_history_table(input_path, period="1W")
                        panel = build_full_week_grid(history, dims, dates)
                    segment_panel = panel[panel["segment_key"] == "geo=A"]
                    metrics = calculate_segment_anomaly(
                        segment_panel,
                        panel[panel["slice_depth"] == 0].set_index("cal_date")["gmv"],
                        dates,
                        dates[-1],
                        AnomalyThresholds(),
                    )
                    self.assertTrue(math.isnan(float(metrics[f"{ratio_metric}_wow_pct"])))

    def test_technical_threshold_preconditions_fail_fast(self) -> None:
        """ADDED: Проверить технические предусловия расчёта и отчёта.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если некорректная настройка приводит к случайной ошибке.

        Examples:
            >>> # Запускается через unittest.
        """

        panel = pd.DataFrame({"cal_date": DATES, "gmv": [1.0, 1.0, 1.0, 2.0]})
        total = pd.Series([10.0, 10.0, 10.0, 10.0], index=DATES)
        with self.assertRaisesRegex(ValueError, "sigma_floor"):
            calculate_segment_anomaly(
                panel,
                total,
                DATES,
                DATES[-1],
                AnomalyThresholds(sigma_floor=0.0),
            )
        with self.assertRaisesRegex(ValueError, "max_manager_facts"):
            build_manager_summary(
                pd.DataFrame(),
                AnomalyThresholds(max_manager_facts=0),
                10.0,
            )

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

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        final, diagnostics, decision_log = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertEqual(set(final["segment_id"]), {"c1", "c2", "q"})
        self.assertFalse(bool(diagnostics.loc[diagnostics["segment_id"] == "p", "selected"].iloc[0]))
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]
        self.assertEqual(int(parent["hierarchy_group_count"]), 3)
        self.assertEqual(
            json.loads(parent["hierarchy_best_group_ids_json"]),
            ["c1", "c2"],
        )
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.15)
        self.assertFalse(decision_log.empty)

    def test_exhaustive_mixed_depth_groups_choose_strongest_alternative(self) -> None:
        """ADDED: Перечислить все mixed-level группы и выбрать максимум score.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если полный перебор или hierarchy-метрики изменились.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 20.0,
                    "delta_gmv": 1_000.0,
                },
                {
                    "segment_id": "product_a",
                    "segment_key": "product=A",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "A",
                    "score": 6.0,
                    "delta_gmv": 400.0,
                },
                {
                    "segment_id": "product_b",
                    "segment_key": "product=B",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "B",
                    "score": 5.0,
                    "delta_gmv": 400.0,
                },
                {
                    "segment_id": "merchant_x",
                    "segment_key": "merchant=X",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": 600.0,
                },
                {
                    "segment_id": "atom_2",
                    "segment_key": "atom=2",
                    "slice_depth": 3,
                    "geo": "A",
                    "product": "2",
                    "score": 4.0,
                    "delta_gmv": 200.0,
                },
                {
                    "segment_id": "atom_4",
                    "segment_key": "atom=4",
                    "slice_depth": 3,
                    "geo": "A",
                    "product": "4",
                    "score": 3.0,
                    "delta_gmv": 200.0,
                },
                {
                    "segment_id": "a1",
                    "segment_key": "atom=1",
                    "slice_depth": 3,
                    "geo": "A",
                    "product": "1",
                    "score": 1.0,
                    "delta_gmv": 200.0,
                },
                {
                    "segment_id": "a3",
                    "segment_key": "atom=3",
                    "slice_depth": 3,
                    "geo": "A",
                    "product": "3",
                    "score": 1.0,
                    "delta_gmv": 200.0,
                },
            ]
        )
        candidates.loc[
            candidates["segment_id"].isin({"a1", "a3"}),
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"a1", "atom_2", "a3", "atom_4"}),
            "product_a": frozenset({"a1", "atom_2"}),
            "product_b": frozenset({"a3", "atom_4"}),
            "merchant_x": frozenset({"a1", "a3"}),
            "a1": frozenset({"a1"}),
            "atom_2": frozenset({"atom_2"}),
            "a3": frozenset({"a3"}),
            "atom_4": frozenset({"atom_4"}),
        }

        groups = _enumerate_disjoint_descendant_groups(
            ["product_a", "product_b", "merchant_x", "atom_2", "atom_4"],
            coverage,
        )
        self.assertEqual(len(groups), 12)
        self.assertIn(("atom_2", "atom_4", "merchant_x"), groups)
        self.assertNotIn(("merchant_x", "product_a"), groups)

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]

        self.assertEqual(int(parent["hierarchy_group_count"]), 12)
        self.assertEqual(
            json.loads(parent["hierarchy_best_group_ids_json"]),
            ["atom_2", "atom_4", "merchant_x"],
        )
        self.assertAlmostEqual(float(parent["hierarchy_best_group_score"]), 15.0)
        self.assertAlmostEqual(float(parent["hierarchy_direction_unity"]), 1.0)
        self.assertAlmostEqual(float(parent["hierarchy_dominant_share"]), 0.6)
        self.assertAlmostEqual(float(parent["hierarchy_balance_max"]), 0.6)
        self.assertAlmostEqual(
            float(parent["hierarchy_balance_effective"]),
            7.0 / 11.0,
        )
        self.assertAlmostEqual(float(parent["hierarchy_balance"]), 0.6)
        self.assertAlmostEqual(float(parent["hierarchy_coherence"]), 0.6)
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.03)
        self.assertAlmostEqual(float(parent["anomaly_score"]), 20.6)

    def test_single_eligible_child_without_dominance_keeps_parent_score(self) -> None:
        """FIXED: Не штрафовать одного ребёнка без dominance.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если неeligible-потомок участвует в группе или
                недоминирующий ребёнок снижает score родителя.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 1_000.0,
                },
                {
                    "segment_id": "eligible_child",
                    "segment_key": "child=eligible",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": 800.0,
                },
                {
                    "segment_id": "filtered_child",
                    "segment_key": "child=filtered",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 100.0,
                    "delta_gmv": 10_000.0,
                },
            ]
        )
        candidates.loc[
            candidates["segment_id"] == "filtered_child",
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"eligible_child", "filtered_child"}),
            "eligible_child": frozenset({"eligible_child"}),
            "filtered_child": frozenset({"filtered_child"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]
        final, _, _ = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertEqual(int(parent["hierarchy_eligible_descendant_count"]), 1)
        self.assertEqual(int(parent["hierarchy_group_count"]), 1)
        self.assertEqual(
            json.loads(parent["hierarchy_best_group_ids_json"]),
            ["eligible_child"],
        )
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.0)
        self.assertAlmostEqual(float(parent["anomaly_score"]), 10.0)
        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_capture"]),
            800.0 / 10_800.0,
        )
        self.assertTrue(
            bool(parent["hierarchy_single_child_direction_match"])
        )
        self.assertFalse(bool(parent["hierarchy_dominance_cap_applied"]))
        self.assertEqual(set(final["segment_id"]), {"p"})

    def test_dominant_single_child_caps_parent_below_child(self) -> None:
        """ADDED: Ограничить score родителя при capture не ниже 80%.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если родитель, пересказывающий одного потомка,
                не ограничен на 2% ниже score потомка.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 1_000.0,
                },
                {
                    "segment_id": "eligible_child",
                    "segment_key": "child=eligible",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": 800.0,
                },
                {
                    "segment_id": "filtered_child",
                    "segment_key": "child=filtered",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 1.0,
                    "delta_gmv": 200.0,
                },
            ]
        )
        candidates.loc[
            candidates["segment_id"] == "filtered_child",
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"eligible_child", "filtered_child"}),
            "eligible_child": frozenset({"eligible_child"}),
            "filtered_child": frozenset({"filtered_child"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]
        final, _, _ = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_capture"]),
            0.80,
        )
        self.assertTrue(
            bool(parent["hierarchy_single_child_direction_match"])
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_uncapped_score"]),
            8.5,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_dominance_cap_score"]),
            7.84,
        )
        self.assertTrue(bool(parent["hierarchy_dominance_cap_applied"]))
        self.assertAlmostEqual(float(parent["anomaly_score"]), 7.84)
        self.assertAlmostEqual(
            float(parent["hierarchy_score_factor"]),
            0.784,
        )
        self.assertEqual(set(final["segment_id"]), {"eligible_child"})

    def test_dominant_child_with_opposite_direction_does_not_cap_parent(
        self,
    ) -> None:
        """ADDED: Не применять dominance cap при разных знаках движения.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если высокий capture без совпадения направления
                ограничивает score родителя.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 1_000.0,
                },
                {
                    "segment_id": "eligible_child",
                    "segment_key": "child=eligible",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": -800.0,
                },
                {
                    "segment_id": "filtered_child",
                    "segment_key": "child=filtered",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 1.0,
                    "delta_gmv": 200.0,
                },
            ]
        )
        candidates.loc[
            candidates["segment_id"] == "filtered_child",
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"eligible_child", "filtered_child"}),
            "eligible_child": frozenset({"eligible_child"}),
            "filtered_child": frozenset({"filtered_child"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]

        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_capture"]),
            0.80,
        )
        self.assertFalse(
            bool(parent["hierarchy_single_child_direction_match"])
        )
        self.assertFalse(bool(parent["hierarchy_dominance_cap_applied"]))
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.0)
        self.assertAlmostEqual(float(parent["anomaly_score"]), 10.0)

    def test_exact_parent_contribution_caps_dominant_child(self) -> None:
        """ADDED: Применить cap по parent-relative exact contributions.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если exact-вклады не дают capture 80% либо не
                ограничивают score родителя ниже score ребёнка.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 0.25,
                },
                {
                    "segment_id": "child",
                    "segment_key": "child=X",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": 0.40,
                },
                {
                    "segment_id": "other_atom",
                    "segment_key": "child=Y",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 1.0,
                    "delta_gmv": 0.10,
                },
            ]
        )
        components = {
            "p": (50.0, 100.0, 200.0, 200.0),
            "child": (40.0, 80.0, 100.0, 100.0),
            "other_atom": (10.0, 20.0, 100.0, 100.0),
        }
        for segment_id, values in components.items():
            mask = candidates["segment_id"].eq(segment_id)
            candidates.loc[mask, "numerator_previous"] = values[0]
            candidates.loc[mask, "numerator_current"] = values[1]
            candidates.loc[mask, "denominator_previous"] = values[2]
            candidates.loc[mask, "denominator_current"] = values[3]
        candidates.loc[
            candidates["segment_id"].eq("other_atom"),
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"child", "other_atom"}),
            "child": frozenset({"child"}),
            "other_atom": frozenset({"other_atom"}),
        }

        scored = apply_hierarchy_score_adjustment(
            candidates,
            coverage,
            contribution_mode="exact_atomic",
        )
        parent = scored.loc[scored["segment_id"].eq("p")].iloc[0]

        self.assertAlmostEqual(
            float(parent["hierarchy_parent_exact_metric_delta"]),
            0.25,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_parent_exact_gross_contribution"]),
            0.25,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_exact_net_contribution"]),
            0.20,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_single_child_capture"]),
            0.80,
        )
        self.assertTrue(
            bool(parent["hierarchy_single_child_direction_match"])
        )
        self.assertTrue(bool(parent["hierarchy_dominance_cap_applied"]))
        self.assertAlmostEqual(float(parent["anomaly_score"]), 7.84)

    def test_exact_parent_contributions_drive_multi_child_coherence(self) -> None:
        """ADDED: Считать coherence в едином parent-relative масштабе.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если multi-child coherence использует legacy
                movement детей вместо точных вкладов относительно родителя.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 1.0,
                },
                {
                    "segment_id": "left",
                    "segment_key": "child=left",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 4.0,
                    "delta_gmv": 100.0,
                },
                {
                    "segment_id": "right",
                    "segment_key": "child=right",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 3.0,
                    "delta_gmv": -100.0,
                },
            ]
        )
        components = {
            "p": (20.0, 40.0, 200.0, 200.0),
            "left": (10.0, 20.0, 100.0, 100.0),
            "right": (10.0, 20.0, 100.0, 100.0),
        }
        for segment_id, values in components.items():
            mask = candidates["segment_id"].eq(segment_id)
            candidates.loc[mask, "numerator_previous"] = values[0]
            candidates.loc[mask, "numerator_current"] = values[1]
            candidates.loc[mask, "denominator_previous"] = values[2]
            candidates.loc[mask, "denominator_current"] = values[3]
        coverage = {
            "p": frozenset({"left", "right"}),
            "left": frozenset({"left"}),
            "right": frozenset({"right"}),
        }

        scored = apply_hierarchy_score_adjustment(
            candidates,
            coverage,
            contribution_mode="exact_atomic",
        )
        parent = scored.loc[scored["segment_id"].eq("p")].iloc[0]

        self.assertEqual(int(parent["hierarchy_best_group_size"]), 2)
        self.assertAlmostEqual(float(parent["hierarchy_direction_unity"]), 1.0)
        self.assertAlmostEqual(float(parent["hierarchy_balance"]), 1.0)
        self.assertAlmostEqual(float(parent["hierarchy_coherence"]), 1.0)
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.15)

    def test_nonfinite_atomic_movement_skips_dominance_cap(self) -> None:
        """ADDED: Не завышать capture при неопределённом движении атома.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если cap применяется без полного атомарного
                движения либо расчёт аварийно завершается.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 1_000.0,
                },
                {
                    "segment_id": "eligible_child",
                    "segment_key": "child=eligible",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "X",
                    "score": 8.0,
                    "delta_gmv": 800.0,
                },
                {
                    "segment_id": "unknown_atom",
                    "segment_key": "child=unknown",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "Y",
                    "score": 1.0,
                    "delta_gmv": math.nan,
                },
            ]
        )
        candidates.loc[
            candidates["segment_id"] == "unknown_atom",
            "passes_initial_anomaly_filter",
        ] = False
        coverage = {
            "p": frozenset({"eligible_child", "unknown_atom"}),
            "eligible_child": frozenset({"eligible_child"}),
            "unknown_atom": frozenset({"unknown_atom"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]

        self.assertTrue(math.isnan(float(parent["hierarchy_single_child_capture"])))
        self.assertTrue(pd.isna(parent["hierarchy_single_child_direction_match"]))
        self.assertFalse(bool(parent["hierarchy_dominance_cap_applied"]))
        self.assertEqual(
            parent["hierarchy_dominance_cap_status"],
            "SKIPPED_NONFINITE_ATOMIC_MOVEMENT",
        )
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 1.0)
        self.assertAlmostEqual(float(parent["anomaly_score"]), 10.0)

    def test_effective_balance_penalizes_unequal_tail(self) -> None:
        """ADDED: Учесть всё распределение eligible-изменений через B_eff.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если итоговый баланс не равен минимуму B_max/B_eff.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 10.0,
                    "delta_gmv": 20_000_000.0,
                },
                {
                    "segment_id": "a",
                    "segment_key": "child=a",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "a",
                    "score": 3.0,
                    "delta_gmv": 12_000_000.0,
                },
                {
                    "segment_id": "b",
                    "segment_key": "child=b",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "b",
                    "score": 3.0,
                    "delta_gmv": 7_800_000.0,
                },
                {
                    "segment_id": "c",
                    "segment_key": "child=c",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "c",
                    "score": 3.0,
                    "delta_gmv": 200_000.0,
                },
            ]
        )
        coverage = {
            "p": frozenset({"a1", "a2", "a3"}),
            "a": frozenset({"a1"}),
            "b": frozenset({"a2"}),
            "c": frozenset({"a3"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]
        expected_effective_balance = ((1.0 / 0.5122) - 1.0) / 2.0

        self.assertAlmostEqual(float(parent["hierarchy_balance_max"]), 0.6)
        self.assertAlmostEqual(
            float(parent["hierarchy_balance_effective"]),
            expected_effective_balance,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_balance"]),
            expected_effective_balance,
        )
        self.assertAlmostEqual(
            float(parent["hierarchy_score_factor"]),
            1.0 + 0.3 * (expected_effective_balance - 0.5),
        )

    def test_opposite_child_directions_penalize_parent(self) -> None:
        """ADDED: Получить минимальный коэффициент при полной компенсации GMV.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если `U=0` не даёт коэффициент 0.85.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 12.0,
                    "delta_gmv": 100.0,
                },
                {
                    "segment_id": "up",
                    "segment_key": "child=up",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "up",
                    "score": 6.0,
                    "delta_gmv": 500.0,
                },
                {
                    "segment_id": "down",
                    "segment_key": "child=down",
                    "slice_depth": 2,
                    "geo": "A",
                    "product": "down",
                    "score": 6.0,
                    "delta_gmv": -500.0,
                },
            ]
        )
        coverage = {
            "p": frozenset({"a1", "a2"}),
            "up": frozenset({"a1"}),
            "down": frozenset({"a2"}),
        }

        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        parent = scored.loc[scored["segment_id"] == "p"].iloc[0]

        self.assertAlmostEqual(float(parent["hierarchy_direction_unity"]), 0.0)
        self.assertAlmostEqual(float(parent["hierarchy_balance"]), 1.0)
        self.assertAlmostEqual(float(parent["hierarchy_coherence"]), 0.0)
        self.assertAlmostEqual(float(parent["hierarchy_score_factor"]), 0.85)
        self.assertAlmostEqual(float(parent["anomaly_score"]), 10.2)

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
        scored = apply_hierarchy_score_adjustment(candidates, coverage)
        final, diagnostics, decision_log = search_anomal(
            scored,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertTrue(final.empty)
        self.assertEqual(int(diagnostics["selected"].sum()), 0)
        self.assertFalse(decision_log.empty)

    def test_uncapped_z_contract_and_scale_source(self) -> None:
        """ADDED: Проверить новые имена и диагностику неограниченного z-score.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если z ограничен или возвращаются удалённые aliases.

        Examples:
            >>> # Запускается через unittest.
        """

        dates = [1, 8, 15, 22, 29]
        segment_panel = pd.DataFrame(
            {
                "cal_date": dates,
                "gmv": [100.0, 100.0, 100.0, 100.0, 200.0],
            }
        )
        metrics = calculate_segment_anomaly(
            segment_panel,
            pd.Series(1_000.0, index=dates),
            dates,
            dates[-1],
            AnomalyThresholds(),
        )

        self.assertGreater(float(metrics["robust_z"]), 6.0)
        self.assertEqual(metrics["abs_robust_z"], abs(metrics["robust_z"]))
        self.assertEqual(metrics["z_scale_source"], "SIGMA_FLOOR")
        self.assertTrue(bool(metrics["z_uses_sigma_floor"]))
        self.assertNotIn("robust_z_capped", metrics)
        self.assertNotIn("abs_z_capped", metrics)
        self.assertNotIn("abnormal_gmv", metrics)
        self.assertNotIn("abs_abnormal_gmv", metrics)

        mad_metrics = calculate_segment_anomaly(
            pd.DataFrame(
                {
                    "cal_date": [1, 8, 15, 22, 29],
                    "gmv": [100.0, 110.0, 99.0, 118.8, 130.0],
                }
            ),
            pd.Series(1_000.0, index=[1, 8, 15, 22, 29]),
            [1, 8, 15, 22, 29],
            29,
            AnomalyThresholds(),
        )
        self.assertEqual(mad_metrics["z_scale_source"], "MAD")
        self.assertFalse(bool(mad_metrics["z_uses_sigma_floor"]))

        lifecycle_metrics = calculate_segment_anomaly(
            pd.DataFrame({"cal_date": DATES, "gmv": [0.0, 0.0, 0.0, 10.0]}),
            pd.Series(1_000.0, index=DATES),
            DATES,
            DATES[-1],
            AnomalyThresholds(lifecycle_z_score=7.0),
        )
        self.assertEqual(float(lifecycle_metrics["robust_z"]), 7.0)

    def test_hierarchy_reconciliation_fails_fast(self) -> None:
        """ADDED: Остановить расчёт при расхождении parent и depth-max атомов.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если корректная панель отклонена или расхождение принято.

        Examples:
            >>> # Запускается через unittest.
        """

        rows: list[dict[str, object]] = []
        for cal_date, atom_x, atom_y in [(1, 10.0, 20.0), (8, 15.0, 20.0)]:
            total = atom_x + atom_y
            rows.extend(
                [
                    {"segment_id": "total", "segment_key": "TOTAL", "slice_depth": 0, "cal_date": cal_date, "gmv": total, "geo": None, "product": None},
                    {"segment_id": "parent", "segment_key": "geo=A", "slice_depth": 1, "cal_date": cal_date, "gmv": total, "geo": "A", "product": None},
                    {"segment_id": "atom_x", "segment_key": "geo=A × product=X", "slice_depth": 2, "cal_date": cal_date, "gmv": atom_x, "geo": "A", "product": "X"},
                    {"segment_id": "atom_y", "segment_key": "geo=A × product=Y", "slice_depth": 2, "cal_date": cal_date, "gmv": atom_y, "geo": "A", "product": "Y"},
                ]
            )
        panel = pd.DataFrame(rows)
        validate_hierarchy_reconciliation(
            panel,
            ["geo", "product"],
            [1, 8],
            absolute_tolerance=1e-4,
        )

        broken_panel = panel.copy()
        broken_panel.loc[
            broken_panel["segment_id"].eq("parent")
            & broken_panel["cal_date"].eq(8),
            "gmv",
        ] += 0.001
        with self.assertRaisesRegex(ValueError, "Нарушена сверка GMV"):
            validate_hierarchy_reconciliation(
                broken_panel,
                ["geo", "product"],
                [1, 8],
                absolute_tolerance=1e-4,
            )

    def test_factual_coverage_is_required_and_strict(self) -> None:
        """ADDED: Проверить обязательный factual coverage и явный fallback.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если повреждённое coverage допускается в оптимизацию.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [{"segment_id": "a", "segment_key": "geo=A", "slice_depth": 1, "geo": "A", "product": None, "score": 1.0}]
        )
        candidates["anomaly_score"] = 1.0

        with self.assertRaisesRegex(ValueError, "требуется factual coverage"):
            search_anomal(candidates, AnomalyThresholds())
        with self.assertRaisesRegex(ValueError, "требуется factual coverage"):
            search_anomal(candidates.iloc[0:0], AnomalyThresholds())

        final, diagnostics, _ = search_anomal(
            candidates,
            AnomalyThresholds(),
            allow_segment_key_fallback=True,
        )
        self.assertEqual(final["segment_id"].tolist(), ["a"])
        self.assertEqual(
            diagnostics["atomic_coverage_source"].iloc[0],
            "SEGMENT_KEY_FALLBACK",
        )

        malformed = candidates.copy()
        malformed["segment_key"] = "broken-key"
        with self.assertRaisesRegex(ValueError, "Некорректная часть segment_key"):
            search_anomal(malformed, AnomalyThresholds(), allow_segment_key_fallback=True)
        with self.assertRaisesRegex(ValueError, "не str"):
            search_anomal(candidates, AnomalyThresholds(), coverage={"a": "a"})
        with self.assertRaisesRegex(ValueError, "не bytes"):
            search_anomal(candidates, AnomalyThresholds(), coverage={"a": b"a"})
        with self.assertRaisesRegex(ValueError, "неизвестные атомы"):
            search_anomal(candidates, AnomalyThresholds(), coverage={"a": ["unknown"]})
        with self.assertRaisesRegex(ValueError, "self-coverage"):
            search_anomal(candidates, AnomalyThresholds(), coverage={"a": []})

    def test_score_dynamic_range_is_reported(self) -> None:
        """ADDED: Проверить solver и диагностику при разных порядках score.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если масштаб objective теряется или выбор неверен.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {"segment_id": "parent", "segment_key": "geo=A", "slice_depth": 1, "geo": "A", "product": None, "score": 1e-6},
                {"segment_id": "atom", "segment_key": "geo=A × product=X", "slice_depth": 2, "geo": "A", "product": "X", "score": 1e6},
            ]
        )
        candidates["anomaly_score"] = candidates["robust_z"].astype(float)
        coverage = {"parent": frozenset({"atom"}), "atom": frozenset({"atom"})}

        final, diagnostics, decision_log = search_anomal(
            candidates,
            AnomalyThresholds(),
            coverage=coverage,
        )

        self.assertEqual(final["segment_id"].tolist(), ["atom"])
        self.assertEqual(
            float(diagnostics["set_packing_component_score_dynamic_range"].max()),
            1e12,
        )
        component_log = decision_log[
            decision_log["event_type"].eq("COMPONENT_SOLVE")
        ].iloc[0]
        self.assertEqual(float(component_log["score_min"]), 1e-6)
        self.assertEqual(float(component_log["score_max"]), 1e6)

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
            "Анализ долевых метрик",
            "Аномалии долевых метрик",
        ]
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "history.csv"
            output_path = Path(temp_dir) / "result.xlsx"
            pd.DataFrame(_ratio_history_rows()).to_csv(input_path, index=False)
            with patch(
                "gmv_anomaly.pipeline.search_anomal",
                wraps=search_anomal,
            ) as optimization_call:
                result = run_pipeline(
                    input_path,
                    output_path,
                    period="1W",
                    thresholds=AnomalyThresholds(
                        min_anomaly_abs=0.0,
                        min_z_score=0.0,
                        min_materiality_share=0.0,
                    ),
                )
            # FIXED: Явно закрываем ExcelFile, чтобы Windows освободил файл до cleanup.
            with pd.ExcelFile(output_path) as workbook:
                sheet_names = workbook.sheet_names
                analysis_columns = list(
                    pd.read_excel(workbook, sheet_name="Анализ аномалий").columns
                )
                ratio_analysis = pd.read_excel(
                    workbook,
                    sheet_name="Анализ долевых метрик",
                )

        self.assertEqual(sheet_names, expected_sheets)
        self.assertIn("hierarchy_group_count", analysis_columns)
        self.assertIn("hierarchy_best_group_ids_json", analysis_columns)
        self.assertIn("hierarchy_score_factor", analysis_columns)
        self.assertIn("hierarchy_single_child_capture", analysis_columns)
        self.assertIn("hierarchy_dominance_cap_applied", analysis_columns)
        self.assertIn("hierarchy_dominance_cap_status", analysis_columns)
        self.assertIn("z_scale_source", analysis_columns)
        self.assertIn("z_uses_sigma_floor", analysis_columns)
        self.assertNotIn("hierarchy_best_group_ids", analysis_columns)
        self.assertNotIn("depth_score_weight", analysis_columns)
        self.assertNotIn("robust_z_capped", result["candidates"].columns)
        self.assertNotIn("abs_z_capped", result["candidates"].columns)
        self.assertNotIn("abnormal_gmv", result["candidates"].columns)
        self.assertNotIn("abs_abnormal_gmv", result["candidates"].columns)
        self.assertEqual(set(ratio_analysis["metric_name"]), {"authzone_tx_share"})
        # ADDED: GMV и пилотная доля решают две отдельные оптимизационные задачи.
        self.assertEqual(optimization_call.call_count, 2)
        self.assertIn("numerator_current", ratio_analysis.columns)
        self.assertIn("denominator_current", ratio_analysis.columns)
        self.assertIn("numerator_delta", ratio_analysis.columns)
        self.assertIn("denominator_delta", ratio_analysis.columns)
        self.assertEqual(
            list(result),
            [
                "history",
                "panel",
                "candidates",
                "final",
                "manager",
                "optimization_decision_log",
                "ratio_candidates",
                "ratio_final",
                "ratio_optimization_decision_log",
                "ratio_status",
                "control",
            ],
        )

    def test_pipeline_calculates_all_ratio_metrics_and_builds_png_trees(self) -> None:
        """ADDED: Рассчитать восемь метрик и сохранить восемь PNG в каталог.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если метрика пропущена либо её PNG не создан.

        Examples:
            >>> # Запускается через unittest.
        """

        expected_metrics = {spec.name for spec in RATIO_METRICS}
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "all_ratios.csv"
            output_path = Path(temp_dir) / "result.xlsx"
            tree_dir = Path(temp_dir) / "ratio_trees"
            pd.DataFrame(_all_ratio_history_rows()).to_csv(input_path, index=False)

            result = run_pipeline(
                input_path,
                output_path,
                period="1W",
                thresholds=AnomalyThresholds(
                    min_anomaly_abs=0.0,
                    min_z_score=0.1,
                    min_materiality_share=0.0,
                ),
                ratio_trees_output_dir=tree_dir,
            )

            status = result["ratio_status"].set_index("metric_name")
            self.assertEqual(set(status.index), expected_metrics)
            self.assertEqual(set(status["status"]), {"CALCULATED"})
            self.assertEqual(set(status["tree_status"]), {"GENERATED"})
            expected_paths = {tree_dir / f"{metric}.png" for metric in expected_metrics}
            self.assertEqual(set(tree_dir.glob("*.png")), expected_paths)
            for path in expected_paths:
                self.assertGreater(path.stat().st_size, 0)

    def test_ratio_metric_registry_has_expected_contracts(self) -> None:
        """ADDED: Зафиксировать состав и границы относительных метрик.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если реестр содержит лишнюю метрику или неверный bound.

        Examples:
            >>> # Запускается через unittest.
        """

        specs = {spec.name: spec for spec in RATIO_METRICS}
        self.assertEqual(
            set(specs),
            {
                "success_rate",
                "refund_tx_share",
                "authzone_tx_share",
                "payapp_tx_share",
                "split_gmv_share",
                "credlim_gmv_share",
                "tips_gmv_share",
                "cashback_gmv_share",
            },
        )
        self.assertNotIn("share_in_total_gmv", specs)
        self.assertEqual(
            {
                name: (spec.numerator_column, spec.denominator_column)
                for name, spec in specs.items()
            },
            {
                "success_rate": ("tx", "tx0"),
                "refund_tx_share": ("refund_tx_numerator", "tx"),
                "authzone_tx_share": ("authzone_tx_numerator", "tx"),
                "payapp_tx_share": ("payapp_tx_numerator", "tx"),
                "split_gmv_share": ("split_gmv_numerator", "gmv"),
                "credlim_gmv_share": ("credlim_gmv_numerator", "gmv"),
                "tips_gmv_share": ("tips_gmv_numerator", "gmv"),
                "cashback_gmv_share": ("cashback_gmv_numerator", "gmv"),
            },
        )
        self.assertFalse(specs["tips_gmv_share"].bounded)
        self.assertFalse(specs["cashback_gmv_share"].bounded)
        self.assertTrue(
            all(
                spec.bounded
                for name, spec in specs.items()
                if name not in {"tips_gmv_share", "cashback_gmv_share"}
            )
        )

    def test_ratio_score_uses_exact_contribution_materiality(self) -> None:
        """FIXED: Проверить z-score и exact contribution-materiality доли.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если exact contribution или legacy-диагностика
                рассчитаны неверно.

        Examples:
            >>> # Запускается через unittest.
        """

        thresholds = AnomalyThresholds(
            min_z_score=2.0,
            min_materiality_share=0.0,
            sigma_floor=0.01,
        )
        history = pd.DataFrame(_ratio_history_rows())
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ratio.csv"
            history.to_csv(path, index=False)
            loaded, dims, dates = load_history_table(path, period="1W")
        panel = build_full_week_grid(loaded, dims, dates)
        coverage = build_atomic_coverage(
            panel.drop_duplicates("segment_id"), dims
        )
        spec = PILOT_RATIO_METRICS[0]
        candidates = build_ratio_anomaly_candidates(
            panel, dims, dates, thresholds, spec, DATES[-1], coverage
        )
        segment_a = candidates[candidates["segment_key"].eq("geo=A")].iloc[0]

        self.assertAlmostEqual(float(segment_a["metric_delta"]), 0.3)
        self.assertAlmostEqual(float(segment_a["metric_delta_pp"]), 30.0)
        self.assertAlmostEqual(float(segment_a["baseline_metric_delta"]), 0.0)
        self.assertAlmostEqual(float(segment_a["robust_z"]), 30.0)
        self.assertAlmostEqual(float(segment_a["hierarchy_movement"]), 0.15)
        self.assertAlmostEqual(float(segment_a["legacy_hierarchy_movement"]), 3.0)
        self.assertAlmostEqual(float(segment_a["numerator_delta"]), 3.0)
        self.assertAlmostEqual(float(segment_a["denominator_delta"]), 0.0)
        self.assertAlmostEqual(float(segment_a["materiality_share"]), 0.60)
        self.assertAlmostEqual(
            float(segment_a["legacy_materiality_share"]),
            5.0 / 6.0,
        )
        self.assertEqual(segment_a["exact_contribution_status"], "OK")
        self.assertAlmostEqual(float(segment_a["reliability_factor"]), 0.4)
        atomic = candidates[candidates["slice_depth"].eq(1)]
        self.assertAlmostEqual(
            float(atomic["exact_materiality_share"].sum()),
            1.0,
        )
        self.assertAlmostEqual(
            float(atomic["exact_global_net_contribution"].sum()),
            float(atomic["exact_global_metric_delta"].iloc[0]),
        )

    def test_exact_ratio_contribution_is_additive_and_handles_missing_atoms(
        self,
    ) -> None:
        """ADDED: Проверить точность формулы и отсутствующие недели атома.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если сумма вкладов не равна изменению scope либо
                пропуск одной/обеих недель создаёт нечисловой результат.

        Examples:
            >>> # Запускается через unittest.
        """

        scope_ratio_previous = 0.25
        scope_ratio_current = 0.50
        scope_denominator_previous = 200.0
        scope_denominator_current = 200.0
        first = calculate_exact_ratio_contribution(
            40.0,
            80.0,
            100.0,
            100.0,
            scope_ratio_previous,
            scope_ratio_current,
            scope_denominator_previous,
            scope_denominator_current,
        )
        second = calculate_exact_ratio_contribution(
            10.0,
            20.0,
            100.0,
            100.0,
            scope_ratio_previous,
            scope_ratio_current,
            scope_denominator_previous,
            scope_denominator_current,
        )
        self.assertAlmostEqual(first, 0.20)
        self.assertAlmostEqual(second, 0.05)
        self.assertAlmostEqual(first + second, 0.25)

        missing_current = calculate_exact_ratio_contribution(
            2.0, 0.0, 10.0, 0.0, 0.30, 0.40, 100.0, 120.0
        )
        missing_previous = calculate_exact_ratio_contribution(
            0.0, 2.0, 0.0, 10.0, 0.30, 0.40, 100.0, 120.0
        )
        missing_both = calculate_exact_ratio_contribution(
            0.0, 0.0, 0.0, 0.0, 0.30, 0.40, 100.0, 120.0
        )
        self.assertTrue(math.isfinite(missing_current))
        self.assertTrue(math.isfinite(missing_previous))
        self.assertEqual(missing_both, 0.0)

        with self.assertRaisesRegex(ValueError, "знаменатели scope"):
            calculate_exact_ratio_contribution(
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            )

    def test_ratio_zero_numerator_is_reported_as_structure_change(self) -> None:
        """ADDED: Оставить новые и исчезнувшие операции вне solver-отбора.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если lifecycle-сегмент пропал или попал в optimization.

        Examples:
            >>> # Запускается через unittest.
        """

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ratio.csv"
            pd.DataFrame(_ratio_history_rows()).to_csv(path, index=False)
            history, dims, dates = load_history_table(path, period="1W")
        panel = build_full_week_grid(history, dims, dates)
        coverage = build_atomic_coverage(panel.drop_duplicates("segment_id"), dims)
        candidates = build_ratio_anomaly_candidates(
            panel,
            dims,
            dates,
            AnomalyThresholds(min_z_score=0.0, min_materiality_share=0.0),
            PILOT_RATIO_METRICS[0],
            DATES[-1],
            coverage,
        )
        states = candidates.set_index("segment_key")["state"].to_dict()
        self.assertEqual(states["geo=B"], "исчезнувший")
        self.assertEqual(states["geo=C"], "новый")
        self.assertFalse(
            candidates.set_index("segment_key").loc["geo=B", "passes_initial_anomaly_filter"]
        )
        self.assertFalse(
            candidates.set_index("segment_key").loc["geo=C", "passes_initial_anomaly_filter"]
        )
        analysis, selected = build_ratio_analysis_sheets(candidates, pd.DataFrame())
        self.assertEqual(set(analysis["сегмент"]), {"geo=A", "geo=B", "geo=C"})
        self.assertTrue(selected.empty)

    def test_ratio_two_missing_weeks_have_zero_hierarchy_movement(self) -> None:
        """ADDED: Сохранить невалидность доли при нулевом вкладе атома.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если отсутствие строки превращает долю в ноль
                либо оставляет ненужный NaN в hierarchy movement.

        Examples:
            >>> # Запускается через unittest.
        """

        spec = PILOT_RATIO_METRICS[0]
        segment_panel = pd.DataFrame(
            {
                "cal_date": DATES,
                "gmv": [100.0, 100.0, 0.0, 0.0],
                spec.numerator_column: [1.0, 1.0, 0.0, 0.0],
                spec.denominator_column: [10.0, 10.0, 0.0, 0.0],
                spec.value_column: [0.1, 0.1, math.nan, math.nan],
                "row_missing_in_source": [False, False, True, True],
            }
        )

        result = calculate_ratio_segment_anomaly(
            segment_panel,
            DATES,
            DATES[-1],
            AnomalyThresholds(),
            spec,
        )

        self.assertFalse(bool(result["metric_valid_for_scoring"]))
        self.assertTrue(math.isnan(float(result["metric_delta"])))
        self.assertEqual(result["metric_status"], "METRIC_ROW_MISSING_IN_SOURCE")
        self.assertEqual(float(result["hierarchy_movement"]), 0.0)

    def test_ratio_reliability_uses_valid_metric_history_not_gmv(self) -> None:
        """FIXED: Не завышать reliability при неопределённой истории доли.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если healthy GMV ошибочно заменяет историю доли.

        Examples:
            >>> # Запускается через unittest.
        """

        spec = PILOT_RATIO_METRICS[0]
        dates = list(range(1, 10))
        segment_panel = pd.DataFrame(
            {
                "cal_date": dates,
                "gmv": [100.0] * len(dates),
                spec.numerator_column: [0.0] * len(dates),
                spec.denominator_column: [10.0] * len(dates),
                # В истории есть только один валидный переход: 0.2 -> 0.2.
                spec.value_column: [math.nan] * 6 + [0.2, 0.2, 0.5],
            }
        )

        result = calculate_ratio_segment_anomaly(
            segment_panel, dates, dates[-1], AnomalyThresholds(), spec
        )

        self.assertEqual(int(result["history_points"]), 1)
        self.assertEqual(int(result["history_nonzero_weeks"]), 1)
        self.assertAlmostEqual(float(result["reliability_factor"]), 0.4)
        self.assertNotEqual(float(result["reliability_factor"]), 1.0)

    def test_ratio_input_contract_rejects_inconsistent_yql_value(self) -> None:
        """ADDED: Не пересчитывать долю в Python, а отклонять её расхождение с YQL-компонентами.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если повреждённая доля принята.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _ratio_history_rows()
        rows[0]["authzone_tx_share"] = 0.99
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_ratio.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Нарушен контракт долевой метрики"):
                load_history_table(path, period="1W")

    def test_yql_outputs_ratio_components_for_python(self) -> None:
        """ADDED: Защитить передачу числителя и знаменателя из YQL.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если сырые компоненты исчезли из output.

        Examples:
            >>> # Запускается через unittest.
        """

        yql_path = (
            Path(__file__).resolve().parent.parent
            / ".proposal"
            / "fintech"
            / "fdt"
            / "payoffline"
            / "projects"
            / "qr_yandex_pay"
            / "bi"
            / "pred_insight.yql"
        )
        # FIXED: Это внешний integration-контракт, а не обязательный локальный
        # fixture. Отсутствие checkout не должно маскировать регрессии Python.
        if not yql_path.exists():
            self.skipTest(f"Внешний YQL-файл недоступен: {yql_path}")
        yql = yql_path.read_text(encoding="utf-8")
        final_select = yql.split("INSERT INTO $output_table WITH TRUNCATE", 1)[1]
        for column in (
            "tx0",
            "authzone_tx_numerator",
            "authzone_tx_share",
        ):
            with self.subTest(column=column):
                self.assertIn(column, final_select)

    def test_ratio_tree_accepts_long_sheet_with_components(self) -> None:
        """ADDED: Построить отдельный граф доли с компонентами и дельтами.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если граф не создан.

        Examples:
            >>> # Запускается через unittest.
        """

        row = {
            "metric_name": "authzone_tx_share",
            "сегмент": "geo=A",
            "глубина": 1,
            "z_scope": 3.0,
            "anomaly_score": 2.0,
            "metric_delta_pp": 5.0,
            "выбран": True,
            "numerator_current": 50.0,
            "denominator_current": 100.0,
            "numerator_delta": 15.0,
            "denominator_delta": -20.0,
            # ADDED: Вклад хранится в долях, а на графе показывается в п.п.
            "exact_global_net_contribution": 0.059,
        }
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "ratio.xlsx"
            tree_path = Path(temp_dir) / "ratio.svg"
            pd.DataFrame([row]).to_excel(
                report_path,
                sheet_name="Анализ долевых метрик",
                index=False,
            )
            created = build_anomaly_tree_from_excel(
                report_path,
                tree_path,
                sheet_name="Анализ долевых метрик",
                metric_name="authzone_tx_share",
                delta_column="metric_delta_pp",
                delta_label="Δ доли, п.п.",
                selected_column="выбран",
                numerator_column="numerator_current",
                denominator_column="denominator_current",
                numerator_delta_column="numerator_delta",
                denominator_delta_column="denominator_delta",
                contribution_column="exact_global_net_contribution",
                contribution_label="C",
            )
            self.assertEqual(created, tree_path.resolve())
            self.assertGreater(tree_path.stat().st_size, 0)
            svg = tree_path.read_text(encoding="utf-8")
            self.assertIn("C: +5.900", svg)
            self.assertLess(svg.index("C: +5.900"), svg.index("Δ доли, п.п.: +5.000"))

    def test_ratio_report_calculates_aggregation_bonus(self) -> None:
        """ADDED: Вывести для доли ту же hierarchy-поправку, что и для GMV.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если поправка не рассчитана для нескольких детей.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = pd.DataFrame(
            [{
                "metric_name": "authzone_tx_share",
                "segment_id": "parent",
                "segment_key": "geo=A",
                "slice_depth": 1,
                "robust_z": 3.0,
                "passes_initial_anomaly_filter": True,
                "state": "обычный",
                "hierarchy_best_group_size": 2,
                "hierarchy_coherence": 1.0,
            }]
        )
        analysis, _ = build_ratio_analysis_sheets(
            candidates,
            pd.DataFrame(),
            AnomalyThresholds(aggregation_bonus_lambda=0.3),
        )
        self.assertAlmostEqual(
            analysis.iloc[0]["hierarchy_coherence_adjustment"],
            0.15,
        )

    def test_ratio_report_exposes_dominant_child_for_tree(self) -> None:
        """ADDED: Передать dominance rule доли в таблицу для зачёркивания.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если long-лист не содержит признака и ключа ребёнка.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = pd.DataFrame(
            [
                {
                    "metric_name": "authzone_tx_share",
                    "segment_id": "parent",
                    "segment_key": "geo=A",
                    "slice_depth": 1,
                    "robust_z": 3.0,
                    "passes_initial_anomaly_filter": True,
                    "state": "обычный",
                    "hierarchy_dominance_rule_matches": True,
                    "hierarchy_best_group_ids_json": '["child"]',
                },
                {
                    "metric_name": "authzone_tx_share",
                    "segment_id": "child",
                    "segment_key": "geo=A × products=X",
                    "slice_depth": 2,
                    "robust_z": 3.1,
                    "passes_initial_anomaly_filter": True,
                    "state": "обычный",
                    "hierarchy_dominance_rule_matches": False,
                    "hierarchy_best_group_ids_json": "[]",
                },
            ]
        )

        analysis, _ = build_ratio_analysis_sheets(candidates, pd.DataFrame())
        parent = analysis.loc[analysis["segment_id"].eq("parent")].iloc[0]
        self.assertTrue(bool(parent["hierarchy_dominance_rule_matches"]))
        self.assertEqual(
            parent["hierarchy_dominant_child_segment"],
            "geo=A × products=X",
        )

    def test_tree_crosses_nodes_with_dominant_child_status(self) -> None:
        """ADDED: Проверить перечёркивание обоих статусов доминирования.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если крест отсутствует на карточке графа.

        Examples:
            >>> # Запускается через unittest.
        """

        tree_rows = [
            {
                "сегмент": "geo=A",
                "глубина": 1,
                "z_scope": 2.0,
                "anomaly_score": 1.0,
                "Delta GMV": 100.0,
                "hierarchy_dominance_rule_matches": True,
                "hierarchy_dominance_cap_applied": False,
                "hierarchy_dominant_child_segment": "geo=A × products=X",
            },
            {
                "сегмент": "geo=B",
                "глубина": 1,
                "z_scope": 2.1,
                "anomaly_score": 1.1,
                "Delta GMV": -110.0,
                "hierarchy_dominance_rule_matches": True,
                "hierarchy_dominance_cap_applied": True,
                "hierarchy_dominant_child_segment": "geo=B × products=Y",
            },
            {
                "сегмент": "geo=A × products=X",
                "глубина": 2,
                "z_scope": 2.2,
                "anomaly_score": 1.2,
                "Delta GMV": 120.0,
            },
            {
                "сегмент": "geo=B × products=Y",
                "глубина": 2,
                "z_scope": 2.3,
                "anomaly_score": 1.3,
                "Delta GMV": -130.0,
            },
        ]
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.xlsx"
            tree_path = Path(temp_dir) / "tree.svg"
            pd.DataFrame(tree_rows).to_excel(
                report_path,
                sheet_name="Анализ аномалий",
                index=False,
            )
            build_anomaly_tree_from_excel(report_path, tree_path)
            svg = tree_path.read_text(encoding="utf-8")

        # Две диагонали для каждого из двух узлов: обычный статус и ``cut``.
        self.assertEqual(svg.count("#5f6764"), 4)

    def test_segment_key_parser_is_shared_and_strict(self) -> None:
        """ADDED: Проверить единый строгий разбор ключа сегмента.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если разбор ключа расходится между модулями или
                молча пропускает неразбираемую часть.

        Examples:
            >>> # Запускается через unittest.
        """

        from gmv_anomaly import data_preparation, reporting, set_packing

        # Все модули обязаны использовать один и тот же парсер: раньше копии
        # в data_preparation и set_packing вели себя по-разному.
        self.assertIs(set_packing.parse_segment_key_parts, parse_segment_key_parts)
        self.assertIs(reporting.parse_segment_key_parts, parse_segment_key_parts)
        self.assertIs(data_preparation.parse_segment_key_parts, parse_segment_key_parts)

        self.assertEqual(
            parse_segment_key_parts("geo=РФ × products=QR"),
            [("geo", "РФ"), ("products", "QR")],
        )
        self.assertEqual(parse_segment_key_parts("ИТОГО"), [])
        self.assertEqual(parse_segment_key_parts(""), [])

        # Латинская "x" больше не считается разделителем, поэтому значение
        # вида "A x B" разбирается как единое значение, а не ломает ключ.
        row = pd.Series({"geo": "A x B", "products": None})
        segment_key, _, depth = build_segment_key_and_level(row, ["geo", "products"])
        self.assertEqual(depth, 1)
        self.assertEqual(parse_segment_key_parts(segment_key), [("geo", "A x B")])

        # Знак "=" внутри значения сохраняется целиком: режем по первому "=".
        self.assertEqual(
            parse_segment_key_parts("geo=A=1"),
            [("geo", "A=1")],
        )

        # Неразбираемая часть — ошибка контракта, а не повод её пропустить.
        with self.assertRaisesRegex(ValueError, "без '='"):
            parse_segment_key_parts(f"geo=РФ{SEGMENT_KEY_SEPARATOR}broken")
        with self.assertRaisesRegex(ValueError, "пустая dimension/value"):
            parse_segment_key_parts("=РФ")
        with self.assertRaisesRegex(ValueError, "пустая dimension/value"):
            parse_segment_key_parts("geo=")

    def test_reliability_penalizes_segment_without_history(self) -> None:
        """ADDED: Отличить сегмент без истории от сегмента с короткой историей.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если нулевая история получает тот же вес, что и
                история из нескольких недель.

        Examples:
            >>> # Запускается через unittest.
        """

        # Раньше обе ветки возвращали 0.4, поэтому сегмент вообще без истории
        # считался таким же надёжным, как сегмент с тремя неделями.
        self.assertEqual(_history_reliability(0), 0.1)
        self.assertEqual(_history_reliability(1), 0.4)
        self.assertEqual(_history_reliability(3), 0.4)
        self.assertEqual(_history_reliability(4), 0.7)
        self.assertEqual(_history_reliability(8), 1.0)
        self.assertLess(_history_reliability(0), _history_reliability(1))

        # Новый сегмент проходит через полный расчёт метрик и получает
        # пониженный reliability_factor.
        metrics = calculate_segment_anomaly(
            pd.DataFrame({"cal_date": DATES, "gmv": [0.0, 0.0, 0.0, 500.0]}),
            pd.Series(1_000.0, index=DATES),
            DATES,
            DATES[-1],
            AnomalyThresholds(),
        )
        self.assertEqual(metrics["state"], "новый сегмент")
        self.assertEqual(int(metrics["history_nonzero_weeks"]), 0)
        self.assertEqual(float(metrics["reliability_factor"]), 0.1)

    def test_hierarchy_enumeration_limit_fails_fast(self) -> None:
        """ADDED: Остановить экспоненциальный перебор групп по явному лимиту.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если превышение лимита не останавливает расчёт или
                допустимый размер отклоняется.

        Examples:
            >>> # Запускается через unittest.
        """

        # Непересекающиеся потомки дают 2^n − 1 групп, поэтому размер входа
        # обязан быть ограничен явно.
        coverage = {
            f"c{index}": frozenset({f"atom{index}"})
            for index in range(30)
        }

        allowed_ids = [f"c{index}" for index in range(10)]
        groups = _enumerate_disjoint_descendant_groups(
            allowed_ids,
            coverage,
            max_descendants=10,
        )
        self.assertEqual(len(groups), 2 ** 10 - 1)

        with self.assertRaisesRegex(ValueError, "max_hierarchy_descendants"):
            _enumerate_disjoint_descendant_groups(
                [f"c{index}" for index in range(11)],
                coverage,
                max_descendants=10,
            )

        self.assertEqual(AnomalyThresholds().max_hierarchy_descendants, 25)

    def test_hierarchy_uses_exact_set_packing_above_enumeration_limit(self) -> None:
        """ADDED: Масштабировать выбор hierarchy-группы без перебора ``2^n``.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если fallback не находит точную непересекающуюся группу.

        Examples:
            >>> # Запускается через unittest.
        """

        candidates = _candidate_frame(
            [
                {
                    "segment_id": "p",
                    "segment_key": "parent",
                    "slice_depth": 1,
                    "geo": "A",
                    "product": None,
                    "score": 2.0,
                    "delta_gmv": 300.0,
                },
                *[
                    {
                        "segment_id": f"a{index}",
                        "segment_key": f"child={index}",
                        "slice_depth": 2,
                        "geo": "A",
                        "product": str(index),
                        "score": 1.0,
                        "delta_gmv": 100.0,
                    }
                    for index in range(3)
                ],
            ]
        )
        coverage = {
            "p": frozenset({"a0", "a1", "a2"}),
            **{f"a{index}": frozenset({f"a{index}"}) for index in range(3)},
        }

        adjusted = apply_hierarchy_score_adjustment(
            candidates,
            coverage,
            max_hierarchy_descendants=2,
        )
        parent = adjusted.loc[adjusted["segment_id"].eq("p")].iloc[0]

        self.assertEqual(parent["hierarchy_group_selection_method"], "SET_PACKING")
        self.assertEqual(int(parent["hierarchy_group_count"]), -1)
        self.assertEqual(int(parent["hierarchy_best_group_size"]), 3)
        self.assertEqual(
            json.loads(parent["hierarchy_best_group_ids_json"]),
            ["a0", "a1", "a2"],
        )

    def test_loading_accepts_yt_type_header_and_string_numbers(self) -> None:
        """ADDED: Принять YT-выгрузку со строкой типов и числами-строками.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если служебная строка типов попадает в панель или
                числовые значения-строки не приводятся к числам.

        Examples:
            >>> # Запускается через unittest.
        """

        rows = _history_rows()
        frame = pd.DataFrame(rows)
        # Реальная выгрузка YT: первая строка содержит имена типов, а числовые
        # колонки приезжают строками, поэтому pandas читает их как object.
        type_header = {
            "period": "string",
            "cal_date": "uint16",
            "slice_depth": "uint8",
            "geo": "string",
            "products": "string",
            "merchants_type": "string",
            "is_terminal_or_cpqr": "string",
            "gmv": "double",
            "tx": "uint64",
            "au": "uint64",
            "am": "uint64",
            "aov": "double",
            "tpm": "double",
            "freq": "double",
        }
        string_rows = [
            {
                column: (
                    str(value)
                    if column in {"gmv", "cal_date", "slice_depth", "tx", "au", "am"}
                    and value is not None
                    else value
                )
                for column, value in row.items()
            }
            for row in rows
        ]
        with_header = pd.DataFrame(
            [{column: type_header.get(column, "string") for column in frame.columns}]
            + string_rows
        )

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "yt_export.csv"
            with_header.to_csv(input_path, index=False)
            history, dims, dates = load_history_table(input_path, period="1W")

        self.assertEqual(len(history), len(rows))
        self.assertNotIn("string", history["period"].astype(str).tolist())
        self.assertEqual(dates, DATES)
        self.assertEqual(dims, list(runtime_config.DIM_COLUMNS))
        self.assertTrue(pd.api.types.is_integer_dtype(history["cal_date"]))
        self.assertTrue(pd.api.types.is_integer_dtype(history["slice_depth"]))
        self.assertTrue(pd.api.types.is_float_dtype(history["gmv"]))


class GoldenAndPropertyBaselineTests(unittest.TestCase):
    """Зафиксировать текущее поведение аудита как golden/property-baseline.

    Эти тесты НЕ утверждают, что текущая математика корректна — аудит нашёл
    в ней ряд дефектов (net/gross асимметрия materiality, reliability по GMV
    для долевых метрик, двойной учёт иерархии). Тесты фиксируют фактическое
    поведение системы до правок, чтобы любое последующее изменение формул
    было осознанным и видимым в diff тестов, а не тихой регрессией.
    """

    _REAL_INPUT_PATH = Path(__file__).resolve().parent.parent / "payoffline_pulse_hier_03_08.xlsx"

    def test_golden_real_input_selection_and_score(self) -> None:
        """Зафиксировать выбор и score пайплайна на реальном входном файле.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если состав выбранных сегментов, их score или
                objective изменились относительно зафиксированного baseline.

        Examples:
            >>> # Запускается через unittest; пропускается без реального файла.
        """

        if not self._REAL_INPUT_PATH.exists():
            self.skipTest(f"Реальный входной файл отсутствует: {self._REAL_INPUT_PATH}")

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "golden_result.xlsx"
            result = run_pipeline(
                input_path=self._REAL_INPUT_PATH,
                output_path=output_path,
                sheet_name=0,
                period="1W",
                dim_cols=list(runtime_config.DIM_COLUMNS),
                current_cal_date=None,
                thresholds=AnomalyThresholds(),
                tree_output_path=None,
                ratio_tree_output_path=None,
            )

        final_gmv = result["final"]
        final_ratio = result["ratio_final"]
        authzone_final = final_ratio[
            final_ratio["metric_name"].eq("authzone_tx_share")
        ]

        expected_gmv_keys = {
            "products=FULLPAYMENT × is_terminal_or_cpqr=QR",
            "geo=РФ × products=FULLPAYMENT × merchants_type=SMB × is_terminal_or_cpqr=Терминал",
        }
        expected_ratio_keys = {
            "geo=РФ × products=FULLPAYMENT × merchants_type=Лэтуаль",
            "geo=Другое × products=FULLPAYMENT × merchants_type=Прочее × is_terminal_or_cpqr=QR",
        }

        self.assertEqual(set(final_gmv["segment_key"]), expected_gmv_keys)
        self.assertEqual(set(authzone_final["segment_key"]), expected_ratio_keys)
        self.assertAlmostEqual(
            float(final_gmv["selection_score"].sum()), 1.991405, places=3
        )
        self.assertAlmostEqual(
            float(authzone_final["selection_score"].sum()), 3.245533, places=3
        )

    def test_exact_ratio_contribution_additivity_property(self) -> None:
        """Проверить аддитивность exact contribution на наборе сценариев.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если сумма вкладов атомов отклоняется от Δscope.

        Examples:
            >>> # Запускается через unittest.
        """

        # (numerator_prev, numerator_curr, denominator_prev, denominator_curr) на атом.
        scenarios = {
            "только числитель растёт": ((5.0, 8.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0)),
            "только знаменатель растёт": ((5.0, 5.0, 10.0, 20.0), (5.0, 5.0, 10.0, 10.0)),
            "оба растут": ((5.0, 9.0, 10.0, 15.0), (5.0, 6.0, 10.0, 12.0)),
            "смена знака эффекта": ((5.0, 2.0, 10.0, 10.0), (5.0, 9.0, 10.0, 10.0)),
            "сильная компенсация": ((5.0, 9.0, 10.0, 10.0), (5.0, 1.0, 10.0, 10.0)),
        }
        for label, (atom_a, atom_b) in scenarios.items():
            with self.subTest(scenario=label):
                num_prev = atom_a[0] + atom_b[0]
                num_curr = atom_a[1] + atom_b[1]
                den_prev = atom_a[2] + atom_b[2]
                den_curr = atom_a[3] + atom_b[3]
                ratio_prev = num_prev / den_prev
                ratio_curr = num_curr / den_curr
                contributions = [
                    calculate_exact_ratio_contribution(
                        atom[0], atom[1], atom[2], atom[3],
                        ratio_prev, ratio_curr, den_prev, den_curr,
                    )
                    for atom in (atom_a, atom_b)
                ]
                self.assertAlmostEqual(
                    sum(contributions), ratio_curr - ratio_prev, places=9
                )

        forward = calculate_exact_ratio_contribution(5, 8, 10, 12, 0.5, 0.6, 20, 24)
        backward = calculate_exact_ratio_contribution(8, 5, 12, 10, 0.6, 0.5, 24, 20)
        self.assertAlmostEqual(forward, -backward, places=9)

    def test_ratio_net_materiality_bounded_by_gross_capture(self) -> None:
        """Проверить 0 <= |net|/global_gross <= gross/global_gross <= 1.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если инвариант нарушен хотя бы для одного eligible
                сегмента.

        Examples:
            >>> # Запускается через unittest.
        """

        thresholds = AnomalyThresholds(
            min_z_score=0.0, min_materiality_share=0.0, sigma_floor=0.01
        )
        history = pd.DataFrame(_ratio_history_rows())
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ratio.csv"
            history.to_csv(path, index=False)
            loaded, dims, dates = load_history_table(path, period="1W")
        panel = build_full_week_grid(loaded, dims, dates)
        coverage = build_atomic_coverage(panel.drop_duplicates("segment_id"), dims)
        spec = PILOT_RATIO_METRICS[0]
        candidates = build_ratio_anomaly_candidates(
            panel, dims, dates, thresholds, spec, DATES[-1], coverage
        )
        valid = candidates[candidates["exact_contribution_valid"].eq(True)]
        self.assertGreater(len(valid), 0)
        for row in valid.itertuples(index=False):
            gross_total = float(row.exact_global_gross_atomic_contribution)
            if gross_total <= 0.0:
                continue
            net_materiality = abs(float(row.exact_global_net_contribution)) / gross_total
            gross_capture = float(row.exact_global_gross_contribution) / gross_total
            self.assertGreaterEqual(net_materiality, -1e-9)
            self.assertLessEqual(net_materiality, gross_capture + 1e-9)
            self.assertLessEqual(gross_capture, 1.0 + 1e-9)
            # FIXED: В score идёт чистый вклад; gross-capture — отдельная
            # диагностика компенсирующихся эффектов.
            self.assertAlmostEqual(
                float(row.exact_materiality_share), net_materiality
            )
            self.assertAlmostEqual(
                float(row.exact_gross_materiality_share), gross_capture
            )

    def test_search_anomal_deterministic_under_row_permutation(self) -> None:
        """Проверить независимость выбора и objective от порядка строк.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если перестановка строк меняет выбор или objective.

        Examples:
            >>> # Запускается через unittest.
        """

        atoms = [f"at{i}" for i in range(4)]
        coverage: dict[str, frozenset[str]] = {}
        rows: list[dict[str, object]] = []
        for index, atom in enumerate(atoms):
            rows.append(
                {
                    "segment_id": atom,
                    "segment_key": f"d1=v{index} × d2=w{index}",
                    "slice_depth": 2,
                    "passes_initial_anomaly_filter": True,
                    "robust_z": 3.0,
                    "abs_robust_z": 3.0,
                    "wow_delta_gmv": 100.0,
                    "materiality_share": 0.5,
                    "reliability_factor": 1.0,
                    "anomaly_score": 1.0,
                }
            )
            coverage[atom] = frozenset({atom})
        rows.append(
            {
                "segment_id": "p1",
                "segment_key": "d1=v0",
                "slice_depth": 1,
                "passes_initial_anomaly_filter": True,
                "robust_z": 3.0,
                "abs_robust_z": 3.0,
                "wow_delta_gmv": 200.0,
                "materiality_share": 0.5,
                "reliability_factor": 1.0,
                "anomaly_score": 2.0,
            }
        )
        coverage["p1"] = frozenset({"at0", "at1"})
        frame = pd.DataFrame(rows)
        thresholds = AnomalyThresholds()

        base_final, _, _ = search_anomal(frame, thresholds, coverage=coverage)
        base_ids = tuple(sorted(base_final["segment_id"]))
        base_objective = float(base_final["selection_score"].sum())

        for seed in range(8):
            shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            final, _, _ = search_anomal(shuffled, thresholds, coverage=coverage)
            with self.subTest(seed=seed):
                self.assertEqual(tuple(sorted(final["segment_id"])), base_ids)
                self.assertAlmostEqual(
                    float(final["selection_score"].sum()), base_objective, places=9
                )

    def test_hierarchy_adjustment_current_behavior_locks_double_counting(self) -> None:
        """Зафиксировать текущее (дефектное) поведение hierarchy adjustment.

        Родитель получает бонус от потомков через ``aggregation_bonus_lambda``,
        даже если сами потомки не выбираются оптимизацией. Это зафиксированный
        аудитом дефект (двойной учёт иерархии в score и в Set Packing), а не
        желаемое поведение — тест существует, чтобы будущее исправление было
        осознанным изменением этого теста, а не тихой регрессией.

        Args:
            Нет аргументов.

        Returns:
            None.

        Raises:
            AssertionError: Если текущее поведение изменилось без обновления
                этого теста.

        Examples:
            >>> # Запускается через unittest.
        """

        coverage = {
            "p": frozenset({"a1", "a2", "a3", "a4"}),
            "c1": frozenset({"a1", "a2"}),
            "c2": frozenset({"a3", "a4"}),
            "a1": frozenset({"a1"}),
            "a2": frozenset({"a2"}),
            "a3": frozenset({"a3"}),
            "a4": frozenset({"a4"}),
        }
        specs = [
            ("p", 1, 400.0, 1.00),
            ("c1", 2, 200.0, 0.55),
            ("c2", 2, 200.0, 0.55),
            ("a1", 3, 100.0, 0.20),
            ("a2", 3, 100.0, 0.20),
            ("a3", 3, 100.0, 0.20),
            ("a4", 3, 100.0, 0.20),
        ]
        frame = pd.DataFrame(
            [
                {
                    "segment_id": segment_id,
                    "segment_key": segment_id,
                    "slice_depth": depth,
                    "passes_initial_anomaly_filter": True,
                    "abs_robust_z": 3.0,
                    "materiality_share": score / 3.0,
                    "reliability_factor": 1.0,
                    "wow_delta_gmv": movement,
                    "robust_z": 3.0,
                }
                for segment_id, depth, movement, score in specs
            ]
        )
        thresholds = AnomalyThresholds()
        adjusted = apply_hierarchy_score_adjustment(
            frame,
            coverage,
            aggregation_bonus_lambda=0.3,
            single_child_factor=0.85,
            dominant_child_capture_threshold=0.80,
            dominant_child_score_margin=0.02,
        )
        final, _, _ = search_anomal(adjusted, thresholds, coverage=coverage)

        parent_row = adjusted.loc[adjusted["segment_id"].eq("p")].iloc[0]
        self.assertGreater(float(parent_row["hierarchy_score_factor"]), 1.0)
        self.assertAlmostEqual(
            float(parent_row["anomaly_score"]), 1.15, places=6
        )
        # ADDED: Baseline фиксирует, что c1/c2 сейчас выбираются вместо p —
        # но score p всё равно вырос из-за coherence детей, которые
        # проиграли им же в Set Packing.
        self.assertEqual(set(final["segment_id"]), {"c1", "c2"})
        self.assertNotIn("p", set(final["segment_id"]))


if __name__ == "__main__":
    unittest.main()
