"""Orchestration полного анализа GMV-аномалий."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .anomaly_scoring import (
    apply_hierarchy_score_adjustment,
    build_anomaly_candidates,
    build_atomic_coverage,
    build_ratio_anomaly_candidates,
)
from .config import AnomalyThresholds, PILOT_RATIO_METRICS
from .data_preparation import build_full_week_grid, load_history_table
from .reporting import (
    build_anomaly_tree_from_excel,
    build_control_table,
    build_manager_summary,
    write_anomaly_excel,
)
from .set_packing import search_anomal


def run_anomaly_analysis(
    input_path: str | Path,
    output_path: str | Path,
    sheet_name: int | str = 0,
    *,
    period: str,
    dim_cols: Optional[Sequence[str]] = None,
    current_cal_date: Optional[int] = None,
    thresholds: Optional[AnomalyThresholds] = None,
    tree_output_path: str | Path | None = None,
    ratio_tree_output_path: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Запустить полный анализ необычных сегментов.

    Args:
        input_path: Путь к входному файлу.
        output_path: Путь к итоговому Excel-файлу.
        sheet_name: Имя или номер листа Excel.
        period: Обязательный период для фильтрации в формате ``<N>W``.
        dim_cols: Явно заданные признаки.
        current_cal_date: Анализируемая неделя. Если None, берётся последняя.
        thresholds: Пороги алгоритма. Если None, используются значения по умолчанию.
        tree_output_path: Необязательный путь к PNG, SVG или PDF с деревом
            листа «Анализ аномалий».
        ratio_tree_output_path: Необязательный явный путь к графу
            `authzone_tx_share`. Если None, долевой граф не создаётся.

    Returns:
        Словарь таблиц результата.

    Raises:
        ValueError: Если входные данные некорректны.
        ImportError: Если запрошено дерево, но matplotlib недоступен.
        OSError: Если отчёт или дерево невозможно записать.

    Examples:
        >>> # result = run_anomaly_analysis('input.xlsx', 'output.xlsx')
    """

    thresholds = thresholds or AnomalyThresholds()
    history_df, dims, dates = load_history_table(input_path, sheet_name=sheet_name, period=period, dim_cols=dim_cols)
    current = int(current_cal_date) if current_cal_date is not None else int(dates[-1])
    panel_df = build_full_week_grid(history_df, dims, dates)
    # FIXED: Покрытие считается один раз по metadata панели и переиспользуется
    # сверкой иерархии, hierarchy-корректировкой и Set Packing.
    segment_metadata = panel_df.drop_duplicates(subset=["segment_id"]).reset_index(drop=True)
    coverage = build_atomic_coverage(segment_metadata, dims)
    candidates, total_by_date = build_anomaly_candidates(
        panel_df,
        dims,
        dates,
        thresholds,
        current,
        coverage=coverage,
    )
    # REMOVED: Штраф за расстояние до eligible-глубины заменён hierarchy coherence.
    candidates = apply_hierarchy_score_adjustment(
        candidates,
        coverage,
        aggregation_bonus_lambda=thresholds.aggregation_bonus_lambda,
        single_child_factor=thresholds.single_child_factor,
        dominant_child_capture_threshold=(
            thresholds.dominant_child_capture_threshold
        ),
        dominant_child_score_margin=thresholds.dominant_child_score_margin,
        max_hierarchy_descendants=thresholds.max_hierarchy_descendants,
    )
    final_df, diagnostics, optimization_decision_log = search_anomal(candidates, thresholds, coverage=coverage)

    # ADDED: Каждая долевая метрика строит своих кандидатов и вызывает Set Packing
    # независимо от GMV и от других метрик.
    ratio_candidate_frames = []
    ratio_final_frames = []
    ratio_log_frames = []
    ratio_status_rows = []
    for spec in PILOT_RATIO_METRICS:
        required_metric_columns = {
            spec.value_column,
            spec.numerator_column,
            spec.denominator_column,
        }
        if not required_metric_columns.issubset(panel_df.columns):
            ratio_status_rows.append(
                {
                    "metric_name": spec.name,
                    "status": "SKIPPED_MISSING_COLUMNS",
                    "details": ", ".join(
                        sorted(required_metric_columns - set(panel_df.columns))
                    ),
                }
            )
            continue
        ratio_candidates = build_ratio_anomaly_candidates(
            panel_df,
            dims,
            dates,
            thresholds,
            spec,
            current,
            coverage,
        )
        ratio_candidates = apply_hierarchy_score_adjustment(
            ratio_candidates,
            coverage,
            aggregation_bonus_lambda=thresholds.aggregation_bonus_lambda,
            single_child_factor=thresholds.single_child_factor,
            dominant_child_capture_threshold=thresholds.dominant_child_capture_threshold,
            dominant_child_score_margin=thresholds.dominant_child_score_margin,
            max_hierarchy_descendants=thresholds.max_hierarchy_descendants,
            movement_column="hierarchy_movement",
            allow_zero_movement=True,
            contribution_mode=spec.contribution_mode,
            contribution_reconciliation_tolerance=(
                spec.validation_abs_tolerance
            ),
        )
        ratio_final, ratio_diagnostics, ratio_log = search_anomal(
            ratio_candidates,
            thresholds,
            coverage=coverage,
        )
        ratio_candidate_frames.append(ratio_diagnostics)
        ratio_final_frames.append(ratio_final)
        ratio_log_frames.append(ratio_log.assign(metric_name=spec.name))
        ratio_status_rows.append(
            {
                "metric_name": spec.name,
                "status": "CALCULATED",
                "details": "",
            }
        )
    ratio_diagnostics = (
        pd.concat(ratio_candidate_frames, ignore_index=True)
        if ratio_candidate_frames
        else pd.DataFrame()
    )
    ratio_final = (
        pd.concat(ratio_final_frames, ignore_index=True)
        if ratio_final_frames
        else pd.DataFrame()
    )
    ratio_optimization_decision_log = (
        pd.concat(ratio_log_frames, ignore_index=True)
        if ratio_log_frames
        else pd.DataFrame(columns=["metric_name"])
    )
    ratio_status = pd.DataFrame(ratio_status_rows)
    manager_df = build_manager_summary(final_df, thresholds, float(total_by_date.loc[current]), diagnostics)
    write_anomaly_excel(
        output_path,
        thresholds,
        dims,
        history_df,
        panel_df,
        diagnostics,
        final_df,
        manager_df,
        total_by_date,
        dates,
        current,
        coverage,
        optimization_decision_log,
        ratio_diagnostics,
        ratio_final,
    )
    if tree_output_path is not None:
        build_anomaly_tree_from_excel(output_path, tree_output_path)
    if ratio_tree_output_path is not None:
        ratio_tree_rows = (
            ratio_diagnostics[
                ratio_diagnostics["metric_name"].astype(str).eq(
                    "authzone_tx_share"
                )
                & ratio_diagnostics["passes_initial_anomaly_filter"].eq(True)
            ]
            if not ratio_diagnostics.empty
            else pd.DataFrame()
        )
        if not ratio_tree_rows.empty:
            build_anomaly_tree_from_excel(
                output_path,
                ratio_tree_output_path,
                sheet_name="Анализ долевых метрик",
                metric_name="authzone_tx_share",
                delta_column="metric_delta_pp",
                delta_label="Δ доли, п.п.",
                selected_column="выбран",
                numerator_column="numerator_current",
                denominator_column="denominator_current",
                numerator_delta_column="numerator_delta",
                denominator_delta_column="denominator_delta",
                # ADDED: Точный вклад долевого сегмента относительно Total.
                contribution_column="exact_global_net_contribution",
                contribution_label="Contribution",
            )

    return {
        "history": history_df,
        "panel": panel_df,
        "candidates": diagnostics,
        "final": final_df,
        "manager": manager_df,
        "optimization_decision_log": optimization_decision_log,
        "ratio_candidates": ratio_diagnostics,
        "ratio_final": ratio_final,
        "ratio_optimization_decision_log": ratio_optimization_decision_log,
        "ratio_status": ratio_status,
        "control": build_control_table(history_df, panel_df, diagnostics, final_df, coverage, dates, current, total_by_date),
    }


# ADDED: Каноническое имя orchestration-функции без удаления прежнего API.
run_pipeline = run_anomaly_analysis

__all__ = ["run_anomaly_analysis", "run_pipeline"]
