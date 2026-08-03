"""Orchestration полного анализа GMV-аномалий."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .anomaly_scoring import (
    apply_hierarchy_score_adjustment,
    build_anomaly_candidates,
    build_atomic_coverage,
)
from .config import AnomalyThresholds
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
    )
    if tree_output_path is not None:
        build_anomaly_tree_from_excel(output_path, tree_output_path)

    return {
        "history": history_df,
        "panel": panel_df,
        "candidates": diagnostics,
        "final": final_df,
        "manager": manager_df,
        "optimization_decision_log": optimization_decision_log,
        "control": build_control_table(history_df, panel_df, diagnostics, final_df, coverage, dates, current, total_by_date),
    }


# ADDED: Каноническое имя orchestration-функции без удаления прежнего API.
run_pipeline = run_anomaly_analysis

__all__ = ["run_anomaly_analysis", "run_pipeline"]
