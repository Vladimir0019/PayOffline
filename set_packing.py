"""Точный отбор непересекающихся GMV-аномалий через Maximum Weighted Set Packing.

Модуль решает задачу ``max Σ anomaly_score_i × x_i`` при ограничении «каждый
атомарный сегмент покрыт не более одного раза». Граф конфликтов разбивается на
независимые компоненты, каждая решается точно: Gurobi, затем SciPy/HiGHS, затем
собственный branch and bound. Решение принимается только с доказанным статусом
OPTIMAL в пределах ``set_packing_gap_tolerance``.

Оптимум глобален в пределах множества кандидатов, прошедших первичный фильтр
аномальности, и допуска solver-а.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from itertools import combinations
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .anomaly_scoring import _safe_float
from .config import AnomalyThresholds
# FIXED: Единственный парсер ключа сегмента; локальная копия удалена.
from .segment_keys import parse_segment_key_parts, segment_feature_set_from_key


def _build_coverage_from_segment_keys(candidates: pd.DataFrame) -> Dict[str, frozenset[str]]:
    """ADDED: Построить атомарное покрытие по `segment_key` без внешнего `dim_cols`.

    Args:
        candidates: Таблица кандидатов с `segment_id`, `segment_key`, `slice_depth`.

    Returns:
        Словарь `segment_id -> frozenset(atomic_segment_id)`.

    Raises:
        ValueError: Если таблица не содержит физический атомарный слой.

    Examples:
        >>> df = pd.DataFrame([
        ...     {'segment_id': 'p', 'segment_key': 'a=1', 'slice_depth': 1},
        ...     {'segment_id': 'c', 'segment_key': 'a=1 × b=2', 'slice_depth': 2},
        ... ])
        >>> _build_coverage_from_segment_keys(df)['p']
        frozenset({'c'})
    """

    if candidates.empty:
        return {}
    max_depth = int(candidates["slice_depth"].astype(int).max())
    atomic_df = candidates[candidates["slice_depth"].astype(int).eq(max_depth)].copy()
    if atomic_df.empty:
        raise ValueError("Физический атомарный слой search_anomal пуст")

    feature_sets: Dict[str, frozenset[Tuple[str, str]]] = {}
    for _, row in candidates.iterrows():
        segment_id = str(row["segment_id"])
        depth = int(row["slice_depth"])
        if depth == 0:
            feature_sets[segment_id] = frozenset()
            continue
        parts = parse_segment_key_parts(row.get("segment_key", ""))
        dimensions = [dimension for dimension, _ in parts]
        if len(parts) != depth or len(dimensions) != len(set(dimensions)):
            raise ValueError(
                "Некорректный segment_key для fallback coverage: "
                f"segment_id={segment_id}, slice_depth={depth}, "
                f"parsed_parts={parts}"
            )
        feature_sets[segment_id] = frozenset(parts)
    atomic_features = {
        str(row["segment_id"]): feature_sets[str(row["segment_id"])]
        for _, row in atomic_df.iterrows()
    }

    coverage: Dict[str, frozenset[str]] = {}
    for _, row in candidates.iterrows():
        segment_id = str(row["segment_id"])
        features = feature_sets.get(segment_id, frozenset())
        depth = int(row["slice_depth"])
        if not features and depth == max_depth:
            coverage[segment_id] = frozenset({segment_id})
            continue
        if not features:
            coverage[segment_id] = frozenset()
            continue
        coverage[segment_id] = frozenset(
            atom_id
            for atom_id, atom_features in atomic_features.items()
            if features.issubset(atom_features)
        )
    return coverage


def _set_packing_canonical_key(segment_id: str, lookup: Dict[str, pd.Series]) -> Tuple[str, int, str]:
    """ADDED: Build a deterministic segment key for solver-independent ordering.

    Args:
        segment_id: Segment identifier.
        lookup: Mapping `segment_id -> candidate row`.

    Returns:
        Tuple with human segment key, depth and technical identifier.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> _set_packing_canonical_key('s', {'s': pd.Series({'segment_key': 'a=1', 'slice_depth': 1})})
        ('a=1', 1, 's')
    """

    row = lookup[segment_id]
    return (str(row.get("segment_key", "")), int(row.get("slice_depth", 0)), str(segment_id))


def _validate_set_packing_duplicates(candidates: pd.DataFrame) -> None:
    """ADDED: Validate that candidates do not contain duplicate segment identifiers or keys.

    Args:
        candidates: Candidate table.

    Returns:
        None.

    Raises:
        ValueError: If duplicate segment ids or duplicate segment keys are found.

    Examples:
        >>> _validate_set_packing_duplicates(pd.DataFrame({'segment_id': ['a'], 'segment_key': ['x']}))
    """

    duplicated_ids = sorted(candidates.loc[candidates["segment_id"].astype(str).duplicated(), "segment_id"].astype(str).unique())
    if duplicated_ids:
        raise ValueError(f"Duplicate segment_id values in candidates: {duplicated_ids[:10]}")
    duplicated_keys = sorted(candidates.loc[candidates["segment_key"].astype(str).duplicated(), "segment_key"].astype(str).unique())
    if duplicated_keys:
        raise ValueError(f"Duplicate segment_key values in candidates: {duplicated_keys[:10]}")


def _prepare_set_packing_coverage(
    candidates: pd.DataFrame,
    coverage: Optional[Mapping[str, Collection[str]]],
    allow_segment_key_fallback: bool = False,
) -> Tuple[Dict[str, frozenset[str]], str, Dict[str, str]]:
    """ADDED: Prepare factual atomic coverage or explicitly marked segment-key fallback.

    Args:
        candidates: Candidate table.
        coverage: Optional factual mapping `segment_id -> atomic segment ids`.
        allow_segment_key_fallback: Явное разрешение восстановить coverage из
            человекочитаемого ``segment_key``.

    Returns:
        Tuple with normalized coverage, coverage source and per-segment validation issue.

    Raises:
        ValueError: Если factual coverage не передано без явного fallback,
            имеет неправильный тип или нарушает атомарные инварианты.

    Examples:
        >>> df = pd.DataFrame([{
        ...     'segment_id': 'a', 'segment_key': 'x=1', 'slice_depth': 1,
        ...     'passes_initial_anomaly_filter': True,
        ... }])
        >>> _prepare_set_packing_coverage(df, None, allow_segment_key_fallback=True)[1]
        'SEGMENT_KEY_FALLBACK'
    """

    if coverage is None:
        if not allow_segment_key_fallback:
            raise ValueError(
                "Для production-вызова search_anomal требуется factual coverage; "
                "fallback по segment_key разрешается только через "
                "allow_segment_key_fallback=True"
            )
        coverage = _build_coverage_from_segment_keys(candidates)
        coverage_source = "SEGMENT_KEY_FALLBACK"
    else:
        coverage_source = "FACTUAL_ATOMIC_COVERAGE"
    if not isinstance(coverage, Mapping):
        raise ValueError("coverage должен быть mapping segment_id -> collection атомов")

    normalized: Dict[str, frozenset[str]] = {}
    validation_errors: List[str] = []
    candidate_ids = set(candidates["segment_id"].astype(str))
    extra_segment_ids = sorted(str(segment_id) for segment_id in coverage if str(segment_id) not in candidate_ids)
    if extra_segment_ids:
        validation_errors.append(
            f"coverage содержит неизвестные segment_id: {extra_segment_ids[:10]}"
        )

    for raw_segment_id, raw_atoms in coverage.items():
        segment_id = str(raw_segment_id)
        if segment_id not in candidate_ids:
            continue
        if isinstance(raw_atoms, (str, bytes)) or isinstance(raw_atoms, Mapping):
            validation_errors.append(
                f"coverage[{segment_id!r}] должен быть коллекцией атомов, а не "
                f"{type(raw_atoms).__name__}"
            )
            continue
        if not isinstance(raw_atoms, Collection):
            validation_errors.append(
                f"coverage[{segment_id!r}] имеет некорректный тип "
                f"{type(raw_atoms).__name__}"
            )
            continue
        atom_list = [str(atom_id).strip() for atom_id in raw_atoms]
        if any(not atom_id for atom_id in atom_list):
            validation_errors.append(
                f"coverage[{segment_id!r}] содержит пустой atomic id"
            )
        if len(atom_list) != len(set(atom_list)):
            validation_errors.append(
                f"coverage[{segment_id!r}] содержит дубли atomic ids"
            )
        normalized[segment_id] = frozenset(atom_list)

    for segment_id in sorted(candidate_ids):
        if segment_id not in normalized:
            validation_errors.append(
                f"coverage отсутствует для segment_id={segment_id}"
            )
            normalized[segment_id] = frozenset()

    max_depth = int(candidates["slice_depth"].astype(int).max())
    atomic_ids = set(
        candidates.loc[
            candidates["slice_depth"].astype(int).eq(max_depth),
            "segment_id",
        ].astype(str)
    )
    for segment_id, atoms in normalized.items():
        unknown_atomic_ids = sorted(atoms - atomic_ids)
        if unknown_atomic_ids:
            validation_errors.append(
                f"coverage[{segment_id!r}] содержит неизвестные атомы: "
                f"{unknown_atomic_ids[:10]}"
            )
    for atomic_id in sorted(atomic_ids):
        if normalized.get(atomic_id) != frozenset({atomic_id}):
            validation_errors.append(
                "Нарушено self-coverage атома: "
                f"coverage[{atomic_id!r}]={sorted(normalized.get(atomic_id, frozenset()))}"
            )

    eligible_ids = set(
        candidates.loc[
            candidates["passes_initial_anomaly_filter"].eq(True)
            & candidates["slice_depth"].astype(int).gt(0),
            "segment_id",
        ].astype(str)
    )
    empty_eligible_ids = sorted(
        segment_id
        for segment_id in eligible_ids
        if not normalized.get(segment_id, frozenset())
    )
    if empty_eligible_ids:
        validation_errors.append(
            "Eligible-сегменты имеют пустое coverage: "
            f"{empty_eligible_ids[:10]}"
        )

    if validation_errors:
        raise ValueError(
            "Некорректное атомарное coverage: "
            + "; ".join(validation_errors[:10])
        )
    return normalized, coverage_source, {}


def _build_set_packing_conflicts(
    segment_ids: Sequence[str],
    coverage: Dict[str, frozenset[str]],
    lookup: Dict[str, pd.Series],
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], frozenset[str]], Dict[str, int]]:
    """ADDED: Build conflict graph through an atom-to-segments inverted index.

    Args:
        segment_ids: Eligible segment ids.
        coverage: Mapping `segment_id -> atomic segment ids`.
        lookup: Mapping `segment_id -> candidate row`.

    Returns:
        Tuple with `atom_to_segments`, conflict pairs and conflict count by segment.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> lookup = {'a': pd.Series({'segment_key': 'a', 'slice_depth': 1})}
        >>> _build_set_packing_conflicts(['a'], {'a': frozenset({'atom'})}, lookup)[2]['a']
        0
    """

    sorted_segment_ids = sorted(segment_ids, key=lambda segment_id: _set_packing_canonical_key(segment_id, lookup))
    atom_to_segments: Dict[str, List[str]] = {}
    for segment_id in sorted_segment_ids:
        for atom_id in sorted(coverage.get(segment_id, frozenset())):
            atom_to_segments.setdefault(atom_id, []).append(segment_id)

    for atom_id, atom_segment_ids in atom_to_segments.items():
        atom_to_segments[atom_id] = sorted(atom_segment_ids, key=lambda segment_id: _set_packing_canonical_key(segment_id, lookup))

    pair_atoms: Dict[Tuple[str, str], set[str]] = {}
    conflict_neighbors: Dict[str, set[str]] = {segment_id: set() for segment_id in sorted_segment_ids}
    for atom_id, atom_segment_ids in atom_to_segments.items():
        if len(atom_segment_ids) <= 1:
            continue
        for left_id, right_id in combinations(atom_segment_ids, 2):
            pair = tuple(sorted((left_id, right_id), key=lambda segment_id: _set_packing_canonical_key(segment_id, lookup)))
            pair_atoms.setdefault(pair, set()).add(atom_id)
            conflict_neighbors[left_id].add(right_id)
            conflict_neighbors[right_id].add(left_id)

    conflict_pair_atoms = {
        pair: frozenset(atoms)
        for pair, atoms in pair_atoms.items()
    }
    conflict_count_by_segment = {
        segment_id: len(conflict_neighbors.get(segment_id, set()))
        for segment_id in sorted_segment_ids
    }
    return atom_to_segments, conflict_pair_atoms, conflict_count_by_segment


def _build_set_packing_components(
    segment_ids: Sequence[str],
    conflict_pair_atoms: Dict[Tuple[str, str], frozenset[str]],
    coverage: Dict[str, frozenset[str]],
    lookup: Dict[str, pd.Series],
    scores: Dict[str, float],
) -> List[Dict[str, object]]:
    """ADDED: Split the conflict graph into exact independent components.

    Args:
        segment_ids: Eligible segment ids.
        conflict_pair_atoms: Mapping conflict pair -> shared atomic ids.
        coverage: Mapping `segment_id -> atomic segment ids`.
        lookup: Mapping `segment_id -> candidate row`.
        scores: Objective coefficient by segment.

    Returns:
        List of component dictionaries with deterministic component ids.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> lookup = {'a': pd.Series({'segment_key': 'a', 'slice_depth': 1})}
        >>> _build_set_packing_components(['a'], {}, {'a': frozenset({'atom'})}, lookup, {'a': 1.0})[0]['component_id']
        'C001'
    """

    adjacency: Dict[str, set[str]] = {segment_id: set() for segment_id in segment_ids}
    for left_id, right_id in conflict_pair_atoms:
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)

    raw_components: List[List[str]] = []
    remaining = set(segment_ids)
    while remaining:
        seed_id = min(remaining, key=lambda segment_id: _set_packing_canonical_key(segment_id, lookup))
        stack = [seed_id]
        remaining.remove(seed_id)
        component: List[str] = []
        while stack:
            segment_id = stack.pop()
            component.append(segment_id)
            for neighbor_id in sorted(adjacency.get(segment_id, set()), key=lambda sid: _set_packing_canonical_key(sid, lookup)):
                if neighbor_id in remaining:
                    remaining.remove(neighbor_id)
                    stack.append(neighbor_id)
        raw_components.append(sorted(component, key=lambda sid: _set_packing_canonical_key(sid, lookup)))

    raw_components.sort(key=lambda component: _set_packing_canonical_key(component[0], lookup))
    components: List[Dict[str, object]] = []
    for component_index, component_segment_ids in enumerate(raw_components, start=1):
        component_id = f"C{component_index:03d}"
        component_atom_ids = sorted(
            set().union(*(coverage.get(segment_id, frozenset()) for segment_id in component_segment_ids))
        )
        component_pair_count = sum(
            1
            for left_id, right_id in conflict_pair_atoms
            if left_id in component_segment_ids and right_id in component_segment_ids
        )
        depths = [int(lookup[segment_id]["slice_depth"]) for segment_id in component_segment_ids]
        component_scores = [
            float(scores.get(segment_id, 0.0))
            for segment_id in component_segment_ids
        ]
        min_score = min(component_scores)
        max_score = max(component_scores)
        components.append(
            {
                "component_id": component_id,
                "segment_ids": component_segment_ids,
                "atom_ids": component_atom_ids,
                "conflict_pair_count": component_pair_count,
                "segment_count": len(component_segment_ids),
                "atom_count": len(component_atom_ids),
                "min_depth": min(depths),
                "max_depth": max(depths),
                "score_sum": float(sum(component_scores)),
                # ADDED: Диагностика численного масштаба objective MILP.
                "score_min": min_score,
                "score_max": max_score,
                "score_dynamic_range": (
                    max_score / min_score if min_score > 0.0 else math.inf
                ),
            }
        )
    return components


def _component_atom_to_segments(
    component_segment_ids: Sequence[str],
    coverage: Dict[str, frozenset[str]],
    lookup: Dict[str, pd.Series],
) -> Dict[str, List[str]]:
    """ADDED: Build atom constraints for one optimization component.

    Args:
        component_segment_ids: Segment ids inside one conflict component.
        coverage: Mapping `segment_id -> atomic segment ids`.
        lookup: Mapping `segment_id -> candidate row`.

    Returns:
        Mapping `atomic_segment_id -> segment ids covering this atom`.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> lookup = {'a': pd.Series({'segment_key': 'a', 'slice_depth': 1})}
        >>> _component_atom_to_segments(['a'], {'a': frozenset({'atom'})}, lookup)['atom']
        ['a']
    """

    atom_to_segments: Dict[str, List[str]] = {}
    for segment_id in component_segment_ids:
        for atom_id in coverage.get(segment_id, frozenset()):
            atom_to_segments.setdefault(atom_id, []).append(segment_id)
    return {
        atom_id: sorted(atom_segment_ids, key=lambda sid: _set_packing_canonical_key(sid, lookup))
        for atom_id, atom_segment_ids in sorted(atom_to_segments.items())
    }


def _set_packing_solver_result(
    component: Dict[str, object],
    solver_name: str,
    solver_status: str,
    selected_ids: Sequence[str],
    objective_value: float,
    best_bound: float,
    absolute_gap: float,
    relative_gap: float,
    solve_time_sec: float,
    variable_count: int,
    constraint_count: int,
    message: str = "",
) -> Dict[str, object]:
    """ADDED: Normalize solver output for diagnostics.

    Args:
        component: Component metadata.
        solver_name: Solver label.
        solver_status: Normalized status.
        selected_ids: Selected segment ids.
        objective_value: Proven or incumbent objective value.
        best_bound: Solver bound in the same maximization scale.
        absolute_gap: Absolute MIP gap.
        relative_gap: Relative MIP gap.
        solve_time_sec: Solver runtime.
        variable_count: Number of binary variables.
        constraint_count: Number of atom constraints.
        message: Optional solver message.

    Returns:
        Component result dictionary.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> _set_packing_solver_result({'component_id': 'C001'}, 'TRIVIAL', 'OPTIMAL', [], 0, 0, 0, 0, 0, 0, 0)['component_id']
        'C001'
    """

    return {
        **component,
        "solver_name": solver_name,
        "solver_status": solver_status,
        "selected_ids": list(selected_ids),
        "objective_value": float(objective_value),
        "best_bound": float(best_bound),
        "absolute_gap": float(absolute_gap),
        "relative_gap": float(relative_gap),
        "solve_time_sec": float(solve_time_sec),
        "variable_count": int(variable_count),
        "constraint_count": int(constraint_count),
        "message": message,
    }


def _set_packing_result_is_proven_optimal(result: Dict[str, object], gap_tolerance: float) -> bool:
    """ADDED: Проверить, что результат компоненты доказанно оптимален.

    Args:
        result: Нормализованный результат solver-а.
        gap_tolerance: Допустимый gap для признания оптимума доказанным.

    Returns:
        True, если статус OPTIMAL и absolute или relative gap находится в допустимой точности.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> _set_packing_result_is_proven_optimal({'solver_status': 'OPTIMAL', 'relative_gap': 0.0, 'absolute_gap': 0.0}, 1e-9)
        True
    """

    if str(result.get("solver_status", "")) != "OPTIMAL":
        return False
    relative_gap = _safe_float(result.get("relative_gap"), math.nan)
    absolute_gap = _safe_float(result.get("absolute_gap"), math.nan)
    tolerance = float(gap_tolerance) + 1e-12
    return relative_gap <= tolerance or absolute_gap <= tolerance


def _try_solve_component_with_gurobi(
    component: Dict[str, object],
    atom_to_segments: Dict[str, List[str]],
    lookup: Dict[str, pd.Series],
    scores: Dict[str, float],
    gap_tolerance: float,
) -> Optional[Dict[str, object]]:
    """ADDED: Try solving one component with Gurobi if it is installed and licensed.

    Args:
        component: Component metadata.
        atom_to_segments: Atom constraints for the component.
        lookup: Mapping `segment_id -> candidate row`.
        scores: Objective coefficient by segment.
        gap_tolerance: Allowed relative MIP gap.

    Returns:
        Solver result or None if Gurobi is unavailable.

    Raises:
        ValueError: Not raised; solver errors trigger fallback to the next solver.

    Examples:
        >>> # result = _try_solve_component_with_gurobi(component, atoms, lookup, scores, 1e-9)
    """

    try:
        import gurobipy as gp  # type: ignore[import-not-found]
        from gurobipy import GRB  # type: ignore[import-not-found]
    except Exception:
        return None

    start_time = time.perf_counter()
    component_segment_ids = list(component["segment_ids"])
    try:
        model = gp.Model("gmv_set_packing")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = gap_tolerance
        variables = {
            segment_id: model.addVar(vtype=GRB.BINARY, name=f"x_{position}")
            for position, segment_id in enumerate(component_segment_ids)
        }
        model.setObjective(
            gp.quicksum(float(scores[segment_id]) * variables[segment_id] for segment_id in component_segment_ids),
            GRB.MAXIMIZE,
        )
        for atom_segment_ids in atom_to_segments.values():
            model.addConstr(gp.quicksum(variables[segment_id] for segment_id in atom_segment_ids) <= 1)
        model.optimize()
    except Exception as exc:
        return _set_packing_solver_result(
            component,
            "GUROBI",
            "ERROR",
            [],
            0.0,
            math.nan,
            math.nan,
            math.nan,
            time.perf_counter() - start_time,
            len(component_segment_ids),
            len(atom_to_segments),
            str(exc),
        )

    status = "OPTIMAL" if model.Status == GRB.OPTIMAL else str(model.Status)
    selected_ids = [
        segment_id
        for segment_id in component_segment_ids
        if variables[segment_id].X >= 0.5
    ]
    objective_value = float(model.ObjVal) if model.SolCount else 0.0
    best_bound = float(model.ObjBound) if model.SolCount else math.nan
    relative_gap = float(model.MIPGap) if model.SolCount else math.nan
    absolute_gap = abs(best_bound - objective_value) if not math.isnan(best_bound) else math.nan
    return _set_packing_solver_result(
        component,
        "GUROBI",
        status,
        selected_ids,
        objective_value,
        best_bound,
        absolute_gap,
        relative_gap,
        time.perf_counter() - start_time,
        len(component_segment_ids),
        len(atom_to_segments),
        "",
    )


def _try_solve_component_with_scipy(
    component: Dict[str, object],
    atom_to_segments: Dict[str, List[str]],
    lookup: Dict[str, pd.Series],
    scores: Dict[str, float],
    gap_tolerance: float,
) -> Optional[Dict[str, object]]:
    """ADDED: Solve one component through scipy.optimize.milp and HiGHS.

    Args:
        component: Component metadata.
        atom_to_segments: Atom constraints for the component.
        lookup: Mapping `segment_id -> candidate row`.
        scores: Objective coefficient by segment.
        gap_tolerance: Allowed relative MIP gap.

    Returns:
        Solver result or None if SciPy MILP is unavailable.

    Raises:
        ValueError: Not raised; solver errors trigger fallback.

    Examples:
        >>> # result = _try_solve_component_with_scipy(component, atoms, lookup, scores, 1e-9)
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except Exception:
        return None

    start_time = time.perf_counter()
    component_segment_ids = sorted(
        list(component["segment_ids"]),
        key=lambda segment_id: _set_packing_canonical_key(segment_id, lookup),
    )
    variable_index = {segment_id: position for position, segment_id in enumerate(component_segment_ids)}
    row_indices: List[int] = []
    col_indices: List[int] = []
    data_values: List[float] = []
    for row_index, atom_id in enumerate(sorted(atom_to_segments)):
        for segment_id in atom_to_segments[atom_id]:
            row_indices.append(row_index)
            col_indices.append(variable_index[segment_id])
            data_values.append(1.0)

    try:
        constraint_matrix = coo_array(
            (data_values, (row_indices, col_indices)),
            shape=(len(atom_to_segments), len(component_segment_ids)),
        ).tocsr()
        constraints = LinearConstraint(
            constraint_matrix,
            lb=-np.inf * np.ones(len(atom_to_segments)),
            ub=np.ones(len(atom_to_segments)),
        )
        result = milp(
            c=-np.array([float(scores[segment_id]) for segment_id in component_segment_ids], dtype=float),
            integrality=np.ones(len(component_segment_ids), dtype=int),
            bounds=Bounds(np.zeros(len(component_segment_ids)), np.ones(len(component_segment_ids))),
            constraints=constraints,
            options={"mip_rel_gap": gap_tolerance},
        )
    except Exception as exc:
        return _set_packing_solver_result(
            component,
            "SCIPY_HIGHS",
            "ERROR",
            [],
            0.0,
            math.nan,
            math.nan,
            math.nan,
            time.perf_counter() - start_time,
            len(component_segment_ids),
            len(atom_to_segments),
            str(exc),
        )

    selected_ids = []
    if getattr(result, "x", None) is not None:
        selected_ids = [
            segment_id
            for segment_id, value in zip(component_segment_ids, result.x)
            if float(value) >= 0.5
        ]
    objective_value = -float(result.fun) if getattr(result, "fun", None) is not None else 0.0
    raw_dual_bound = getattr(result, "mip_dual_bound", math.nan)
    best_bound = -float(raw_dual_bound) if raw_dual_bound is not None and not math.isnan(float(raw_dual_bound)) else objective_value
    raw_gap = getattr(result, "mip_gap", 0.0 if int(getattr(result, "status", -1)) == 0 else math.nan)
    relative_gap = float(raw_gap) if raw_gap is not None else math.nan
    absolute_gap = abs(best_bound - objective_value) if not math.isnan(best_bound) else math.nan
    status = "OPTIMAL" if int(getattr(result, "status", -1)) == 0 and relative_gap <= gap_tolerance + 1e-12 else str(getattr(result, "status", "UNKNOWN"))
    return _set_packing_solver_result(
        component,
        "SCIPY_HIGHS",
        status,
        selected_ids,
        objective_value,
        best_bound,
        absolute_gap,
        relative_gap,
        time.perf_counter() - start_time,
        len(component_segment_ids),
        len(atom_to_segments),
        str(getattr(result, "message", "")),
    )


def _solve_component_exact_branch_and_bound(
    component: Dict[str, object],
    atom_to_segments: Dict[str, List[str]],
    lookup: Dict[str, pd.Series],
    coverage: Dict[str, frozenset[str]],
    scores: Dict[str, float],
) -> Dict[str, object]:
    """ADDED: Solve set packing exactly without external MILP dependencies.

    Args:
        component: Component metadata.
        atom_to_segments: Atom constraints for the component.
        lookup: Mapping `segment_id -> candidate row`.
        coverage: Mapping `segment_id -> atomic segment ids`.
        scores: Objective coefficient by segment.

    Returns:
        Proven optimal component result.

    Raises:
        RecursionError: If Python recursion depth is exceeded on an unusually large component.

    Examples:
        >>> lookup = {'a': pd.Series({'segment_key': 'a', 'slice_depth': 1})}
        >>> component = {'component_id': 'C001', 'segment_ids': ['a'], 'atom_ids': ['atom']}
        >>> _solve_component_exact_branch_and_bound(component, {'atom': ['a']}, lookup, {'a': frozenset({'atom'})}, {'a': 1.0})['solver_status']
        'OPTIMAL'
    """

    start_time = time.perf_counter()
    ordered_ids = sorted(
        list(component["segment_ids"]),
        key=lambda sid: (-float(scores.get(sid, 0.0)), _set_packing_canonical_key(sid, lookup)),
    )
    suffix_positive_score = [0.0] * (len(ordered_ids) + 1)
    for index in range(len(ordered_ids) - 1, -1, -1):
        suffix_positive_score[index] = suffix_positive_score[index + 1] + max(0.0, float(scores.get(ordered_ids[index], 0.0)))

    best_score = 0.0
    best_selected: List[str] = []
    best_signature: Tuple[int, Tuple[Tuple[str, int, str], ...]] = (0, tuple())

    def solution_signature(selected_ids: Sequence[str]) -> Tuple[int, Tuple[Tuple[str, int, str], ...]]:
        """ADDED: Deterministic tie-break signature among equal-score optima.

        Args:
            selected_ids: Selected segment ids.

        Returns:
            Signature preferring fewer rows, then lexical segment order.

        Raises:
            ValueError: Not raised.

        Examples:
            >>> solution_signature([])
            (0, ())
        """

        return (
            len(selected_ids),
            tuple(_set_packing_canonical_key(segment_id, lookup) for segment_id in sorted(selected_ids, key=lambda sid: _set_packing_canonical_key(sid, lookup))),
        )

    def update_best(selected_ids: Sequence[str], score_value: float) -> None:
        """ADDED: Update incumbent solution with deterministic tie handling.

        Args:
            selected_ids: Candidate selected segment ids.
            score_value: Candidate objective value.

        Returns:
            None.

        Raises:
            ValueError: Not raised.

        Examples:
            >>> # update_best(['a'], 1.0)
        """

        nonlocal best_score, best_selected, best_signature
        signature = solution_signature(selected_ids)
        if score_value > best_score + 1e-12 or (abs(score_value - best_score) <= 1e-12 and signature < best_signature):
            best_score = float(score_value)
            best_selected = sorted(selected_ids, key=lambda sid: _set_packing_canonical_key(sid, lookup))
            best_signature = signature

    def branch(index: int, used_atoms: frozenset[str], selected_ids: Tuple[str, ...], score_value: float) -> None:
        """ADDED: Recursive exact branch-and-bound search.

        Args:
            index: Current position in ordered ids.
            used_atoms: Atoms already covered by selected segments.
            selected_ids: Current selected segment ids.
            score_value: Current objective value.

        Returns:
            None.

        Raises:
            RecursionError: If recursion depth is exceeded.

        Examples:
            >>> # branch(0, frozenset(), tuple(), 0.0)
        """

        if score_value + suffix_positive_score[index] < best_score - 1e-12:
            return
        if index >= len(ordered_ids):
            update_best(selected_ids, score_value)
            return
        segment_id = ordered_ids[index]
        segment_score = float(scores.get(segment_id, 0.0))
        segment_atoms = coverage.get(segment_id, frozenset())
        if segment_score > 0.0 and not (used_atoms & segment_atoms):
            branch(index + 1, used_atoms | segment_atoms, (*selected_ids, segment_id), score_value + segment_score)
        branch(index + 1, used_atoms, selected_ids, score_value)

    branch(0, frozenset(), tuple(), 0.0)
    return _set_packing_solver_result(
        component,
        "EXACT_BRANCH_AND_BOUND",
        "OPTIMAL",
        best_selected,
        best_score,
        best_score,
        0.0,
        0.0,
        time.perf_counter() - start_time,
        len(ordered_ids),
        len(atom_to_segments),
        "",
    )


def _solve_set_packing_component(
    component: Dict[str, object],
    coverage: Dict[str, frozenset[str]],
    lookup: Dict[str, pd.Series],
    scores: Dict[str, float],
    gap_tolerance: float,
    max_exact_fallback_size: int,
) -> Dict[str, object]:
    """ADDED: Solve one independent set-packing component with the best available exact solver.

    Args:
        component: Component metadata.
        coverage: Mapping `segment_id -> atomic segment ids`.
        lookup: Mapping `segment_id -> candidate row`.
        scores: Objective coefficient by segment.
        gap_tolerance: Allowed relative MIP gap.
        max_exact_fallback_size: Largest component allowed for internal exact fallback.

    Returns:
        Component solver result.

    Raises:
        RuntimeError: If MILP solvers do not prove optimum and the component is too large for fallback.
        RecursionError: If exact fallback is allowed but exceeds recursion depth.

    Examples:
        >>> lookup = {'a': pd.Series({'segment_key': 'a', 'slice_depth': 1})}
        >>> component = {'component_id': 'C001', 'segment_ids': ['a'], 'atom_ids': ['atom']}
        >>> _solve_set_packing_component(component, {'a': frozenset({'atom'})}, lookup, {'a': 1.0}, 1e-9, 25)['solver_status']
        'OPTIMAL'
    """

    component_segment_ids = list(component["segment_ids"])
    atom_to_segments = _component_atom_to_segments(component_segment_ids, coverage, lookup)
    if all(len(segment_ids) <= 1 for segment_ids in atom_to_segments.values()):
        selected_ids = [
            segment_id
            for segment_id in component_segment_ids
            if float(scores.get(segment_id, 0.0)) > 0.0
        ]
        objective_value = float(sum(scores.get(segment_id, 0.0) for segment_id in selected_ids))
        return _set_packing_solver_result(
            component,
            "TRIVIAL",
            "OPTIMAL",
            sorted(selected_ids, key=lambda sid: _set_packing_canonical_key(sid, lookup)),
            objective_value,
            objective_value,
            0.0,
            0.0,
            0.0,
            len(component_segment_ids),
            len(atom_to_segments),
            "",
        )

    last_result: Optional[Dict[str, object]] = None
    for solver in (_try_solve_component_with_gurobi, _try_solve_component_with_scipy):
        result = solver(component, atom_to_segments, lookup, scores, gap_tolerance)
        if result is None:
            continue
        last_result = result
        if _set_packing_result_is_proven_optimal(result, gap_tolerance):
            return result

    if len(component_segment_ids) > int(max_exact_fallback_size):
        last_solver = str(last_result["solver_name"]) if last_result is not None else "NONE"
        last_status = str(last_result["solver_status"]) if last_result is not None else "UNAVAILABLE"
        raise RuntimeError(
            "MILP solver не доказал optimum, exact fallback слишком велик: "
            f"component={component.get('component_id')}, "
            f"segment_count={len(component_segment_ids)}, "
            f"limit={int(max_exact_fallback_size)}, "
            f"last_solver={last_solver}, last_status={last_status}. "
            "Для production-расчёта нужен рабочий точный MILP solver."
        )

    exact_result = _solve_component_exact_branch_and_bound(component, atom_to_segments, lookup, coverage, scores)
    if last_result is not None and exact_result["message"] == "":
        exact_result["message"] = f"Fallback after {last_result['solver_name']} status {last_result['solver_status']}"
    return exact_result


def _build_set_packing_decision_log(
    component_results: Sequence[Dict[str, object]],
    atom_to_segments: Dict[str, List[str]],
    conflict_pair_atoms: Dict[Tuple[str, str], frozenset[str]],
    lookup: Dict[str, pd.Series],
    scores: Dict[str, float],
    global_status: str,
) -> pd.DataFrame:
    """ADDED: Build a transparent journal for set-packing optimization.

    Args:
        component_results: Solver results by independent component.
        atom_to_segments: Inverted index `atomic_segment_id -> segment ids`.
        conflict_pair_atoms: Mapping conflict pair -> shared atomic ids.
        lookup: Mapping `segment_id -> candidate row`.
        scores: Objective coefficient by segment.
        global_status: Overall optimization status.

    Returns:
        Decision log DataFrame.

    Raises:
        ValueError: Not raised.

    Examples:
        >>> _build_set_packing_decision_log([], {}, {}, {}, {}, 'EMPTY').empty
        False
    """

    rows: List[Dict[str, object]] = [
        {
            "event_type": "GLOBAL_OPTIMIZATION_SUMMARY",
            "global_status": global_status,
            "component_count": len(component_results),
            "selected_count": sum(len(result.get("selected_ids", [])) for result in component_results),
            "objective_value": sum(float(result.get("objective_value", 0.0)) for result in component_results),
            "best_bound": sum(float(result.get("best_bound", 0.0)) for result in component_results),
            "absolute_gap": sum(float(result.get("absolute_gap", 0.0)) for result in component_results),
            "relative_gap": max((float(result.get("relative_gap", 0.0)) for result in component_results), default=0.0),
            "solver_status": global_status,
        }
    ]

    for atom_id, segment_ids in sorted(atom_to_segments.items()):
        rows.append(
            {
                "event_type": "ATOM_TO_SEGMENTS",
                "atomic_segment_id": atom_id,
                "segment_count": len(segment_ids),
                "segment_ids": " || ".join(segment_ids),
                "segment_keys": " || ".join(str(lookup[segment_id]["segment_key"]) for segment_id in segment_ids),
            }
        )

    for (left_id, right_id), shared_atoms in sorted(
        conflict_pair_atoms.items(),
        key=lambda item: (
            _set_packing_canonical_key(item[0][0], lookup),
            _set_packing_canonical_key(item[0][1], lookup),
        ),
    ):
        rows.append(
            {
                "event_type": "CONFLICT_PAIR",
                "left_segment_id": left_id,
                "left_segment_key": str(lookup[left_id]["segment_key"]),
                "right_segment_id": right_id,
                "right_segment_key": str(lookup[right_id]["segment_key"]),
                "shared_atomic_count": len(shared_atoms),
                "shared_atomic_ids": " || ".join(sorted(shared_atoms)),
            }
        )

    for result in component_results:
        selected_ids = set(result.get("selected_ids", []))
        rows.append(
            {
                "event_type": "COMPONENT_SOLVE",
                "component_id": result["component_id"],
                "solver_name": result["solver_name"],
                "solver_status": result["solver_status"],
                "objective_value": result["objective_value"],
                "best_bound": result["best_bound"],
                "absolute_gap": result["absolute_gap"],
                "relative_gap": result["relative_gap"],
                "solve_time_sec": result["solve_time_sec"],
                "variable_count": result["variable_count"],
                "constraint_count": result["constraint_count"],
                "segment_count": result["segment_count"],
                "atom_count": result["atom_count"],
                "conflict_pair_count": result["conflict_pair_count"],
                "min_depth": result["min_depth"],
                "max_depth": result["max_depth"],
                "score_sum": result["score_sum"],
                "score_min": result["score_min"],
                "score_max": result["score_max"],
                "score_dynamic_range": result["score_dynamic_range"],
                "selected_ids": " || ".join(sorted(selected_ids, key=lambda sid: _set_packing_canonical_key(sid, lookup))),
                "message": result.get("message", ""),
            }
        )
        for segment_id in result["segment_ids"]:
            rows.append(
                {
                    "event_type": "SEGMENT_DECISION",
                    "component_id": result["component_id"],
                    "segment_id": segment_id,
                    "segment_key": str(lookup[segment_id]["segment_key"]),
                    "slice_depth": int(lookup[segment_id]["slice_depth"]),
                    "anomaly_score": float(scores.get(segment_id, 0.0)),
                    "selected": segment_id in selected_ids,
                    "solver_name": result["solver_name"],
                    "solver_status": result["solver_status"],
                    "global_status": global_status,
                }
            )
    return pd.DataFrame(rows)


def validate_set_packing_solution(
    final_df: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: Dict[str, frozenset[str]],
    component_results: Sequence[Dict[str, object]],
    scores: Dict[str, float],
    global_status: str,
    gap_tolerance: float,
) -> None:
    """ADDED: Обязательно проверить корректность найденного Set Packing решения.

    Args:
        final_df: Итоговые выбранные сегменты.
        diagnostics: Диагностика всех кандидатов после оптимизации.
        coverage: Покрытие `segment_id -> atomic_segment_id`.
        component_results: Результаты решения независимых компонент.
        scores: Вес `anomaly_score` по каждому оптимизируемому сегменту.
        global_status: Итоговый статус оптимизации.
        gap_tolerance: Допустимый solver gap.

    Returns:
        None.

    Raises:
        RuntimeError: Если решение не доказано оптимальным, нарушает атомарные ограничения
            или objective не совпадает с суммой score выбранных сегментов.

    Examples:
        >>> validate_set_packing_solution(pd.DataFrame(), pd.DataFrame(), {}, [], {}, "OPTIMAL", 1e-9)
    """

    objective_tolerance = max(1e-7, float(gap_tolerance) + 1e-12)
    if str(global_status) != "OPTIMAL":
        raise RuntimeError(f"Set Packing solution is not globally OPTIMAL: global_status={global_status}")

    if not diagnostics.empty and {"passes_initial_anomaly_filter", "slice_depth", "set_packing_status"}.issubset(diagnostics.columns):
        passed_mask = diagnostics["passes_initial_anomaly_filter"].eq(True) & diagnostics["slice_depth"].astype(int).gt(0)
        valid_final_statuses = {"SET_PACKING_SELECTED", "SET_PACKING_NOT_SELECTED"}
        unresolved = diagnostics.loc[passed_mask & ~diagnostics["set_packing_status"].isin(valid_final_statuses)]
        if not unresolved.empty:
            examples = [
                f"{row.segment_id}: {row.set_packing_status}"
                for row in unresolved.head(10).itertuples(index=False)
            ]
            raise RuntimeError(
                "После оптимизации остались прошедшие первичный фильтр сегменты вне доказанного Set Packing: "
                + "; ".join(examples)
            )

    selected_ids_from_components: List[str] = []
    component_objective_sum = 0.0
    for result in component_results:
        if not _set_packing_result_is_proven_optimal(result, gap_tolerance):
            raise RuntimeError(
                "Компонента Set Packing не доказала OPTIMAL: "
                f"component={result.get('component_id')}, "
                f"solver={result.get('solver_name')}, "
                f"status={result.get('solver_status')}, "
                f"abs_gap={result.get('absolute_gap')}, rel_gap={result.get('relative_gap')}"
            )
        component_selected_ids = [str(segment_id) for segment_id in result.get("selected_ids", [])]
        if len(component_selected_ids) != len(set(component_selected_ids)):
            raise RuntimeError(f"Компонента {result.get('component_id')} содержит дубли selected_ids")
        missing_scores = sorted(set(component_selected_ids) - set(scores))
        if missing_scores:
            raise RuntimeError(
                f"Для выбранных сегментов компоненты {result.get('component_id')} отсутствует score: {missing_scores[:10]}"
            )
        selected_score_sum = float(sum(float(scores[segment_id]) for segment_id in component_selected_ids))
        objective_value = _safe_float(result.get("objective_value"), math.nan)
        if not math.isclose(selected_score_sum, objective_value, rel_tol=1e-9, abs_tol=objective_tolerance):
            raise RuntimeError(
                "Objective компоненты не равен сумме score выбранных сегментов: "
                f"component={result.get('component_id')}, objective={objective_value}, selected_score_sum={selected_score_sum}"
            )
        component_objective_sum += objective_value
        selected_ids_from_components.extend(component_selected_ids)

    if len(selected_ids_from_components) != len(set(selected_ids_from_components)):
        raise RuntimeError("Один и тот же сегмент выбран более чем в одной компоненте Set Packing")

    final_ids = final_df["segment_id"].astype(str).tolist() if not final_df.empty and "segment_id" in final_df.columns else []
    if set(final_ids) != set(selected_ids_from_components):
        raise RuntimeError(
            "final_df не совпадает с selected_ids solver-а: "
            f"final_only={sorted(set(final_ids) - set(selected_ids_from_components))[:10]}, "
            f"solver_only={sorted(set(selected_ids_from_components) - set(final_ids))[:10]}"
        )
    final_score_sum = (
        float(pd.to_numeric(final_df["selection_score"], errors="coerce").sum())
        if not final_df.empty and "selection_score" in final_df.columns
        else 0.0
    )
    if not math.isclose(final_score_sum, component_objective_sum, rel_tol=1e-9, abs_tol=objective_tolerance):
        raise RuntimeError(
            "Глобальный objective не равен сумме objective компонент или score итоговых сегментов: "
            f"final_score_sum={final_score_sum}, component_objective_sum={component_objective_sum}"
        )

    atom_to_selected_segments: Dict[str, List[str]] = {}
    for segment_id in selected_ids_from_components:
        segment_atoms = coverage.get(segment_id, frozenset())
        if not segment_atoms:
            raise RuntimeError(f"Выбранный сегмент {segment_id} не имеет атомарного покрытия")
        for atom_id in segment_atoms:
            atom_to_selected_segments.setdefault(str(atom_id), []).append(segment_id)
    atom_violations = {
        atom_id: segment_ids
        for atom_id, segment_ids in atom_to_selected_segments.items()
        if len(segment_ids) > 1
    }
    if atom_violations:
        examples = [
            f"{atom_id}: {' || '.join(segment_ids)}"
            for atom_id, segment_ids in list(atom_violations.items())[:10]
        ]
        raise RuntimeError(
            "Нарушено ограничение Set Packing sum(x_i covering atom) <= 1: "
            + "; ".join(examples)
        )


def search_anomal(
    candidates: pd.DataFrame,
    thresholds: AnomalyThresholds,
    coverage: Optional[Mapping[str, Collection[str]]] = None,
    allow_segment_key_fallback: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FIXED: Select anomalies via exact Maximum Weighted Set Packing.

    Args:
        candidates: Candidate table after `build_anomaly_candidates`.
        thresholds: Algorithm thresholds; only anomaly filters and set-packing gap tolerance affect selection.
        coverage: Factual mapping `segment_id -> observed atomic segment ids`.
        allow_segment_key_fallback: Явно разрешить менее надёжное восстановление
            coverage из ``segment_key``, если factual coverage не передано.

    Returns:
        Tuple with final selected anomalies, candidate diagnostics and optimization decision log.

    Raises:
        ValueError: If required columns are missing, duplicate candidates exist, or atomic coverage is invalid.
        RecursionError: If all MILP solvers are unavailable and exact fallback exceeds recursion depth.

    Examples:
        >>> # final_df, diagnostics, log = search_anomal(candidates, AnomalyThresholds(), coverage)
    """

    required_columns = {
        "segment_id",
        "segment_key",
        "slice_depth",
        "passes_initial_anomaly_filter",
        "robust_z",
        "abs_robust_z",
        "wow_delta_gmv",
        "anomaly_score",
    }
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Для search_anomal не хватает колонок: {missing_columns}")

    diagnostics = candidates.copy()
    if diagnostics.empty:
        # FIXED: Обязательность factual coverage действует и для пустого
        # production-вызова; иначе контракт зависел бы от числа строк.
        if coverage is None and not allow_segment_key_fallback:
            raise ValueError(
                "Для production-вызова search_anomal требуется factual coverage; "
                "fallback по segment_key разрешается только через "
                "allow_segment_key_fallback=True"
            )
        if coverage is not None:
            if not isinstance(coverage, Mapping):
                raise ValueError(
                    "coverage должен быть mapping segment_id -> collection атомов"
                )
            if coverage:
                raise ValueError(
                    "Для пустой таблицы candidates factual coverage должен быть пустым"
                )
        return diagnostics.copy(), diagnostics, _build_set_packing_decision_log([], {}, {}, {}, {}, "EMPTY")

    diagnostics["segment_id"] = diagnostics["segment_id"].astype(str)
    diagnostics["segment_key"] = diagnostics["segment_key"].astype(str)
    diagnostics["slice_depth"] = diagnostics["slice_depth"].astype(int)
    _validate_set_packing_duplicates(diagnostics)
    normalized_coverage, coverage_source, coverage_issues = (
        _prepare_set_packing_coverage(
            diagnostics,
            coverage,
            allow_segment_key_fallback=allow_segment_key_fallback,
        )
    )

    string_columns = [
        "action",
        "output_block",
        "reason",
        "original_atomic_descendants",
        "selected_atomic_descendants",
        "set_packing_status",
        "set_packing_global_status",
        "set_packing_component_id",
        "set_packing_solver",
        "set_packing_solver_status",
        "set_packing_reason",
        "atomic_coverage_source",
        "atomic_coverage_validation_status",
        "conflict_segment_ids",
        "conflict_segment_keys",
    ]
    for column in string_columns:
        diagnostics[column] = ""

    diagnostics["is_eligible"] = False
    # FIXED: Не подменяем нечисловой score нулём; прошедший первичный фильтр кандидат
    # с некорректным score должен остановить расчёт, а не исчезнуть из оптимизации.
    diagnostics["selection_score"] = pd.to_numeric(diagnostics["anomaly_score"], errors="coerce")
    diagnostics["covered_atomic_count"] = diagnostics["segment_id"].map(lambda sid: len(normalized_coverage.get(str(sid), frozenset())))
    diagnostics["original_atomic_count"] = 0
    diagnostics["selected_atomic_count"] = 0
    diagnostics["is_resolved"] = False
    diagnostics["selected"] = False
    diagnostics["selection_exclusion_reason"] = ""
    diagnostics["conflict_count"] = 0
    diagnostics["set_packing_objective_value"] = math.nan
    diagnostics["set_packing_best_bound"] = math.nan
    diagnostics["set_packing_abs_gap"] = math.nan
    diagnostics["set_packing_rel_gap"] = math.nan
    diagnostics["set_packing_solve_time_sec"] = math.nan
    diagnostics["set_packing_variable_count"] = 0
    diagnostics["set_packing_constraint_count"] = 0
    diagnostics["set_packing_component_segment_count"] = 0
    diagnostics["set_packing_component_atom_count"] = 0
    diagnostics["set_packing_component_conflict_pair_count"] = 0
    diagnostics["set_packing_component_score_sum"] = 0.0
    diagnostics["set_packing_component_score_min"] = math.nan
    diagnostics["set_packing_component_score_max"] = math.nan
    diagnostics["set_packing_component_score_dynamic_range"] = math.nan

    lookup = {
        str(row["segment_id"]): row.copy()
        for _, row in diagnostics.iterrows()
    }
    index_by_id = {
        str(segment_id): index
        for index, segment_id in diagnostics["segment_id"].items()
    }

    eligible_ids: List[str] = []
    fatal_input_issues: List[str] = []
    for _, row in diagnostics.iterrows():
        segment_id = str(row["segment_id"])
        index = index_by_id[segment_id]
        depth = int(row["slice_depth"])
        passed_initial_filter = bool(row.get("passes_initial_anomaly_filter", False))
        atoms = sorted(normalized_coverage.get(segment_id, frozenset()))
        coverage_issue = coverage_issues.get(segment_id, "")
        score_value = _safe_float(row.get("selection_score"), math.nan)
        diagnostics.at[index, "atomic_coverage_source"] = coverage_source
        diagnostics.at[index, "atomic_coverage_validation_status"] = coverage_issue or "OK"
        diagnostics.at[index, "original_atomic_descendants"] = " || ".join(atoms)
        diagnostics.at[index, "original_atomic_count"] = len(atoms)
        diagnostics.at[index, "covered_atomic_count"] = len(atoms)

        if depth == 0:
            reason = "total-слой исключён из оптимизационного отбора аномалий"
            status = "NOT_IN_SET_PACKING_GRAPH"
        elif not passed_initial_filter:
            reason = "сегмент не прошёл первичный фильтр аномальности"
            status = "NOT_IN_SET_PACKING_GRAPH"
        elif coverage_issue:
            reason = coverage_issue
            status = "INVALID_ATOMIC_COVERAGE"
        elif not atoms:
            reason = "у сегмента пустое фактическое атомарное покрытие"
            status = "EMPTY_ATOMIC_COVERAGE"
        elif not math.isfinite(score_value):
            reason = "anomaly_score не является конечным числом"
            status = "INVALID_SCORE"
        elif score_value <= 0.0:
            reason = "anomaly_score неположительный"
            status = "NONPOSITIVE_SCORE"
        else:
            reason = "сегмент участвует в точной задаче Maximum Weighted Set Packing"
            status = "SET_PACKING_CANDIDATE"
            eligible_ids.append(segment_id)
            diagnostics.at[index, "is_eligible"] = True

        diagnostics.at[index, "action"] = status
        diagnostics.at[index, "output_block"] = "кандидат Set Packing" if status == "SET_PACKING_CANDIDATE" else "исключён"
        diagnostics.at[index, "reason"] = reason
        diagnostics.at[index, "set_packing_status"] = status
        diagnostics.at[index, "set_packing_reason"] = reason
        diagnostics.at[index, "selection_exclusion_reason"] = "" if status == "SET_PACKING_CANDIDATE" else reason
        if passed_initial_filter and depth > 0 and status != "SET_PACKING_CANDIDATE":
            fatal_input_issues.append(
                f"{segment_id} ({row.get('segment_key', '')}): {status}: {reason}"
            )

    if fatal_input_issues:
        raise ValueError(
            "Нельзя доказать глобальный optimum: часть сегментов, прошедших первичный фильтр, "
            "не может быть корректно включена в Set Packing. "
            + "; ".join(fatal_input_issues[:10])
        )

    if not eligible_ids:
        diagnostics["set_packing_global_status"] = "EMPTY"
        decision_log = _build_set_packing_decision_log([], {}, {}, lookup, {}, "EMPTY")
        return diagnostics.iloc[0:0].copy(), diagnostics, decision_log

    scores = {
        segment_id: float(diagnostics.at[index_by_id[segment_id], "selection_score"])
        for segment_id in eligible_ids
    }
    atom_to_segments, conflict_pair_atoms, conflict_count_by_segment = _build_set_packing_conflicts(
        eligible_ids,
        normalized_coverage,
        lookup,
    )
    components = _build_set_packing_components(
        eligible_ids,
        conflict_pair_atoms,
        normalized_coverage,
        lookup,
        scores,
    )
    gap_tolerance = float(getattr(thresholds, "set_packing_gap_tolerance", 1e-9))
    max_exact_fallback_size = int(getattr(thresholds, "max_exact_fallback_size", 25))
    component_results = [
        _solve_set_packing_component(component, normalized_coverage, lookup, scores, gap_tolerance, max_exact_fallback_size)
        for component in components
    ]
    global_status = (
        "OPTIMAL"
        if all(_set_packing_result_is_proven_optimal(result, gap_tolerance) for result in component_results)
        else "NOT_OPTIMAL"
    )
    selected_ids = {
        segment_id
        for result in component_results
        for segment_id in result.get("selected_ids", [])
    }
    result_by_segment: Dict[str, Dict[str, object]] = {}
    for result in component_results:
        for segment_id in result["segment_ids"]:
            result_by_segment[segment_id] = result

    conflict_neighbors: Dict[str, set[str]] = {segment_id: set() for segment_id in eligible_ids}
    for left_id, right_id in conflict_pair_atoms:
        conflict_neighbors[left_id].add(right_id)
        conflict_neighbors[right_id].add(left_id)

    for segment_id in eligible_ids:
        index = index_by_id[segment_id]
        result = result_by_segment[segment_id]
        component_id = str(result["component_id"])
        atoms = sorted(normalized_coverage.get(segment_id, frozenset()))
        selected = segment_id in selected_ids
        segment_conflicts = sorted(conflict_neighbors.get(segment_id, set()), key=lambda sid: _set_packing_canonical_key(sid, lookup))
        conflict_keys = [str(lookup[conflict_id]["segment_key"]) for conflict_id in segment_conflicts]
        if selected:
            status = "SET_PACKING_SELECTED"
            reason = (
                "выбран оптимизацией Maximum Weighted Set Packing в пределах "
                f"gap_tolerance={gap_tolerance}; component={component_id}; "
                f"solver={result['solver_name']}; status={result['solver_status']}"
            )
        elif result["solver_status"] == "OPTIMAL":
            status = "SET_PACKING_NOT_SELECTED"
            reason = (
                f"не выбран глобальным оптимумом компоненты {component_id}; "
                "выбранный непересекающийся набор даёт большую или равную сумму anomaly_score"
            )
        else:
            status = "SET_PACKING_NOT_PROVEN"
            reason = f"компонента {component_id} не вернула доказанный статус OPTIMAL"

        diagnostics.at[index, "selected"] = selected
        diagnostics.at[index, "is_resolved"] = result["solver_status"] == "OPTIMAL"
        diagnostics.at[index, "action"] = status
        diagnostics.at[index, "output_block"] = "оптимальная аномалия Set Packing" if selected else "исключён Set Packing"
        diagnostics.at[index, "reason"] = reason
        diagnostics.at[index, "set_packing_status"] = status
        diagnostics.at[index, "conflict_count"] = int(conflict_count_by_segment.get(segment_id, 0))
        diagnostics.at[index, "conflict_segment_ids"] = " || ".join(segment_conflicts)
        diagnostics.at[index, "conflict_segment_keys"] = " || ".join(conflict_keys)
        diagnostics.at[index, "set_packing_global_status"] = global_status
        diagnostics.at[index, "set_packing_component_id"] = component_id
        diagnostics.at[index, "set_packing_solver"] = str(result["solver_name"])
        diagnostics.at[index, "set_packing_solver_status"] = str(result["solver_status"])
        diagnostics.at[index, "set_packing_reason"] = reason
        diagnostics.at[index, "set_packing_objective_value"] = float(result["objective_value"])
        diagnostics.at[index, "set_packing_best_bound"] = float(result["best_bound"])
        diagnostics.at[index, "set_packing_abs_gap"] = float(result["absolute_gap"])
        diagnostics.at[index, "set_packing_rel_gap"] = float(result["relative_gap"])
        diagnostics.at[index, "set_packing_solve_time_sec"] = float(result["solve_time_sec"])
        diagnostics.at[index, "set_packing_variable_count"] = int(result["variable_count"])
        diagnostics.at[index, "set_packing_constraint_count"] = int(result["constraint_count"])
        diagnostics.at[index, "set_packing_component_segment_count"] = int(result["segment_count"])
        diagnostics.at[index, "set_packing_component_atom_count"] = int(result["atom_count"])
        diagnostics.at[index, "set_packing_component_conflict_pair_count"] = int(result["conflict_pair_count"])
        diagnostics.at[index, "set_packing_component_score_sum"] = float(result["score_sum"])
        diagnostics.at[index, "set_packing_component_score_min"] = float(result["score_min"])
        diagnostics.at[index, "set_packing_component_score_max"] = float(result["score_max"])
        diagnostics.at[index, "set_packing_component_score_dynamic_range"] = float(result["score_dynamic_range"])
        diagnostics.at[index, "selected_atomic_descendants"] = " || ".join(atoms) if selected else ""
        diagnostics.at[index, "selected_atomic_count"] = len(atoms) if selected else 0
        diagnostics.at[index, "selection_exclusion_reason"] = "" if selected else reason

    diagnostics.loc[diagnostics["set_packing_global_status"].eq(""), "set_packing_global_status"] = global_status
    final_df = diagnostics[diagnostics["selected"].astype(bool)].copy()
    # FIXED: Сохраняем прежний tie-sort по абсолютному денежному движению без
    # удалённых aliases abnormal_gmv/abs_abnormal_gmv.
    final_df["_abs_wow_delta_gmv_sort"] = pd.to_numeric(
        final_df["wow_delta_gmv"],
        errors="coerce",
    ).abs()
    final_df = final_df.sort_values(
        by=[
            "selection_score",
            "abs_robust_z",
            "materiality_share",
            "_abs_wow_delta_gmv_sort",
            "reliability_factor",
            "segment_key",
        ],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).drop(columns=["_abs_wow_delta_gmv_sort"]).reset_index(drop=True)
    final_df.insert(0, "rank", range(1, len(final_df) + 1))
    validate_set_packing_solution(
        final_df,
        diagnostics,
        normalized_coverage,
        component_results,
        scores,
        global_status,
        gap_tolerance,
    )
    decision_log = _build_set_packing_decision_log(
        component_results,
        atom_to_segments,
        conflict_pair_atoms,
        lookup,
        scores,
        global_status,
    )
    return final_df, diagnostics, decision_log
