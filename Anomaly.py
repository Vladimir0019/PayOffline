"""Compatibility wrapper для прежнего модуля поиска GMV-аномалий.

Расчётная реализация перенесена в пакет :mod:`gmv_anomaly`. Имена прежнего
публичного API сохранены через re-export.
"""

from __future__ import annotations

from .anomaly_scoring import (
    _history_reliability,
    _safe_float,
    apply_hierarchy_score_adjustment,
    build_anomaly_candidates,
    build_atomic_coverage,
    build_ratio_anomaly_candidates,
    calculate_exact_ratio_contribution,
    calculate_segment_anomaly,
    calculate_ratio_segment_anomaly,
    validate_hierarchy_reconciliation,
)
from .config import (
    ANOMALY_TECH_COLUMNS,
    CURRENT_CAL_DATE,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RATIO_TREE_OUTPUT_PATH,
    DEFAULT_RATIO_TREES_OUTPUT_DIR,
    DEFAULT_TREE_OUTPUT_PATH,
    DIM_COLUMNS,
    INPUT_PATH,
    MANAGER_METRIC_PCT_COLUMNS,
    METRIC_COLUMNS,
    OUTPUT_PATH,
    PERIOD,
    RATIO_METRICS,
    RATIO_TREE_OUTPUT_PATH,
    RATIO_TREES_OUTPUT_DIR,
    SHEET_NAME,
    THRESHOLDS,
    TREE_OUTPUT_PATH,
    AnomalyThresholds,
    PILOT_RATIO_METRICS,
    RATIO_ADDITIVE_COLUMNS,
    RatioMetricSpec,
)
from .data_preparation import (
    _is_missing,
    build_full_week_grid,
    build_segment_key_and_level,
    candidate_covers_atomic,
    infer_anomaly_dimension_columns,
    load_history_table,
    normalize_dim_value,
    period_to_weeks,
    segment_id_from_row,
)
from .pipeline import run_anomaly_analysis, run_pipeline
from .reporting import (
    _format_pct,
    _format_rub,
    _order_tree_layers_by_shared_children,
    _tree_edge_port_offset,
    build_anomaly_analysis_sheet,
    build_anomaly_tree_from_excel,
    build_control_table,
    build_history_for_selected,
    build_manager_summary,
    build_missing_zero_report,
    build_ratio_analysis_sheets,
    highlight_manager_rows_on_anomaly_analysis,
    write_anomaly_excel,
)
# FIXED: Парсер ключа переехал в segment_keys.py; прежние приватные имена
# сохранены как aliases, чтобы не сломать внешний код.
from .segment_keys import (
    parse_segment_key_parts as _segment_key_parts,
    segment_feature_set_from_key as _segment_feature_set_from_key,
)
from .set_packing import (
    _build_coverage_from_segment_keys,
    _build_set_packing_components,
    _build_set_packing_conflicts,
    _build_set_packing_decision_log,
    _component_atom_to_segments,
    _prepare_set_packing_coverage,
    _set_packing_canonical_key,
    _set_packing_result_is_proven_optimal,
    _set_packing_solver_result,
    _solve_component_exact_branch_and_bound,
    _solve_set_packing_component,
    _try_solve_component_with_gurobi,
    _try_solve_component_with_scipy,
    _validate_set_packing_duplicates,
    search_anomal,
    validate_set_packing_solution,
)
from .main import main

__all__ = [
    "ANOMALY_TECH_COLUMNS",
    "CURRENT_CAL_DATE",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TREE_OUTPUT_PATH",
    "DEFAULT_RATIO_TREE_OUTPUT_PATH",
    "DEFAULT_RATIO_TREES_OUTPUT_DIR",
    "DIM_COLUMNS",
    "INPUT_PATH",
    "MANAGER_METRIC_PCT_COLUMNS",
    "METRIC_COLUMNS",
    "OUTPUT_PATH",
    "PERIOD",
    "RATIO_TREE_OUTPUT_PATH",
    "RATIO_TREES_OUTPUT_DIR",
    "SHEET_NAME",
    "THRESHOLDS",
    "TREE_OUTPUT_PATH",
    "AnomalyThresholds",
    "RatioMetricSpec",
    "PILOT_RATIO_METRICS",
    "RATIO_METRICS",
    "RATIO_ADDITIVE_COLUMNS",
    "apply_hierarchy_score_adjustment",
    "build_anomaly_analysis_sheet",
    "build_anomaly_candidates",
    "build_anomaly_tree_from_excel",
    "build_atomic_coverage",
    "build_ratio_anomaly_candidates",
    "build_ratio_analysis_sheets",
    "build_control_table",
    "build_full_week_grid",
    "build_history_for_selected",
    "build_manager_summary",
    "build_missing_zero_report",
    "build_segment_key_and_level",
    "calculate_segment_anomaly",
    "calculate_exact_ratio_contribution",
    "calculate_ratio_segment_anomaly",
    "candidate_covers_atomic",
    "highlight_manager_rows_on_anomaly_analysis",
    "infer_anomaly_dimension_columns",
    "load_history_table",
    "main",
    "normalize_dim_value",
    "period_to_weeks",
    "run_anomaly_analysis",
    "run_pipeline",
    "search_anomal",
    "segment_id_from_row",
    "validate_set_packing_solution",
    "validate_hierarchy_reconciliation",
    "write_anomaly_excel",
]


if __name__ == "__main__":
    main()
