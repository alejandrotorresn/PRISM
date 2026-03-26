from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .plan_representation import ExecutionPlan


@dataclass
class SimulationConfig:
    mode: str = "robust"  # robust or nominal
    k_sigma: float = 1.0
    w_time: float = 1.0
    w_energy: float = 0.0
    w_transfer: float = 1.0
    gpu_mem_budget_mb: float = 1e18
    cpu_mem_budget_mb: float = 1e18
    strict_transfer_mapping: bool = False
    strict_graph_subset: bool = False
    strict_topology: bool = False


@dataclass
class SimulationResult:
    status: str
    objective_value: float
    total_time_ms: float
    total_energy_j: float
    total_transfer_ms: float
    gpu_mem_used_mb: float
    cpu_mem_used_mb: float
    layers_total: int
    layers_gpu: int
    layers_cpu: int
    cut_edges_count: int
    violations: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "total_time_ms": self.total_time_ms,
            "total_energy_j": self.total_energy_j,
            "total_transfer_ms": self.total_transfer_ms,
            "gpu_mem_used_mb": self.gpu_mem_used_mb,
            "cpu_mem_used_mb": self.cpu_mem_used_mb,
            "layers_total": self.layers_total,
            "layers_gpu": self.layers_gpu,
            "layers_cpu": self.layers_cpu,
            "cut_edges_count": self.cut_edges_count,
            "violations": self.violations,
            "warnings": self.warnings,
        }


def _validate_topology(
    assignment_layers: Set[str],
    graph_edges: List[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    violations: List[str] = []

    # Keep only edges fully contained in the assignment layer set.
    edges = [(u, v) for (u, v) in graph_edges if u in assignment_layers and v in assignment_layers]

    indegree: Dict[str, int] = {n: 0 for n in assignment_layers}
    adj: Dict[str, List[str]] = {n: [] for n in assignment_layers}
    for (u, v) in edges:
        adj[u].append(v)
        indegree[v] += 1

    # Preserve original indegrees before Kahn's BFS mutates them.  Sink nodes
    # (incoming edges but no outgoing edges) are NOT isolated; their indegrees
    # get decremented to 0 during traversal and would otherwise be misclassified.
    original_indegree = dict(indegree)

    # Kahn-style topological validation.
    queue = [n for n in assignment_layers if indegree[n] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(assignment_layers):
        violations.append(
            "Graph topology is not a DAG over assigned layers (cycle detected or inconsistent indegree)"
        )

    # A truly isolated node has no outgoing AND no incoming edges in the assignment
    # subgraph.  Use original (pre-BFS) indegrees so that legitimate sink nodes
    # are not falsely reported as isolated.
    isolated = [n for n in assignment_layers if not adj[n] and original_indegree[n] == 0]
    if isolated and len(assignment_layers) > 1:
        warnings.append(
            f"{len(isolated)} assigned layers are isolated in graph topology"
        )

    if not edges and len(assignment_layers) > 1:
        warnings.append(
            "No internal graph edges found among assigned layers; topology checks are limited"
        )

    return warnings, violations


def _robust_value(df: pd.DataFrame, base_col: str, k_sigma: float) -> pd.Series:
    mu_col = f"{base_col}_mean"
    sd_col = f"{base_col}_std"
    if mu_col not in df.columns:
        raise KeyError(f"Expected column not found: {mu_col}")

    mu = pd.to_numeric(df[mu_col], errors="coerce").fillna(0.0)
    if sd_col in df.columns:
        sd = pd.to_numeric(df[sd_col], errors="coerce").fillna(0.0)
    else:
        sd = 0.0
    return mu + (k_sigma * sd)


def _layer_profiles(metrics_stats_csv: str | Path, mode: str, k_sigma: float) -> pd.DataFrame:
    path = Path(metrics_stats_csv)
    if not path.exists():
        raise FileNotFoundError(f"metrics_stats csv not found: {path}")

    df = pd.read_csv(path).copy()
    if "layer" not in df.columns:
        raise KeyError(f"Missing required column 'layer' in {path}")

    df["layer"] = df["layer"].astype(str)

    if mode not in {"robust", "nominal"}:
        raise ValueError(f"Unsupported mode: {mode}. Expected robust|nominal")

    if mode == "nominal":
        for col in [
            "gpu_fwd_time_ms_mean",
            "gpu_bwd_time_ms_mean",
            "cpu_fwd_time_ms_mean",
            "cpu_bwd_time_ms_mean",
            "gpu_fwd_energy_j_mean",
            "gpu_bwd_energy_j_mean",
            "cpu_fwd_energy_j_mean",
            "cpu_bwd_energy_j_mean",
            "gpu_mem_peak_mb_mean",
            "cpu_mem_mb_mean",
        ]:
            if col not in df.columns:
                raise KeyError(f"Missing required nominal column '{col}' in {path}")

        df["gpu_fwd_time_ms"] = pd.to_numeric(df["gpu_fwd_time_ms_mean"], errors="coerce").fillna(0.0)
        df["gpu_bwd_time_ms"] = pd.to_numeric(df["gpu_bwd_time_ms_mean"], errors="coerce").fillna(0.0)
        df["cpu_fwd_time_ms"] = pd.to_numeric(df["cpu_fwd_time_ms_mean"], errors="coerce").fillna(0.0)
        df["cpu_bwd_time_ms"] = pd.to_numeric(df["cpu_bwd_time_ms_mean"], errors="coerce").fillna(0.0)

        df["gpu_fwd_energy_j"] = pd.to_numeric(df["gpu_fwd_energy_j_mean"], errors="coerce").fillna(0.0)
        df["gpu_bwd_energy_j"] = pd.to_numeric(df["gpu_bwd_energy_j_mean"], errors="coerce").fillna(0.0)
        df["cpu_fwd_energy_j"] = pd.to_numeric(df["cpu_fwd_energy_j_mean"], errors="coerce").fillna(0.0)
        df["cpu_bwd_energy_j"] = pd.to_numeric(df["cpu_bwd_energy_j_mean"], errors="coerce").fillna(0.0)
    else:
        df["gpu_fwd_time_ms"] = _robust_value(df, "gpu_fwd_time_ms", k_sigma)
        df["gpu_bwd_time_ms"] = _robust_value(df, "gpu_bwd_time_ms", k_sigma)
        df["cpu_fwd_time_ms"] = _robust_value(df, "cpu_fwd_time_ms", k_sigma)
        df["cpu_bwd_time_ms"] = _robust_value(df, "cpu_bwd_time_ms", k_sigma)

        df["gpu_fwd_energy_j"] = _robust_value(df, "gpu_fwd_energy_j", k_sigma)
        df["gpu_bwd_energy_j"] = _robust_value(df, "gpu_bwd_energy_j", k_sigma)
        df["cpu_fwd_energy_j"] = _robust_value(df, "cpu_fwd_energy_j", k_sigma)
        df["cpu_bwd_energy_j"] = _robust_value(df, "cpu_bwd_energy_j", k_sigma)

    df["gpu_time_ms"] = df["gpu_fwd_time_ms"] + df["gpu_bwd_time_ms"]
    df["cpu_time_ms"] = df["cpu_fwd_time_ms"] + df["cpu_bwd_time_ms"]
    df["gpu_energy_j"] = df["gpu_fwd_energy_j"] + df["gpu_bwd_energy_j"]
    df["cpu_energy_j"] = df["cpu_fwd_energy_j"] + df["cpu_bwd_energy_j"]

    if "gpu_mem_peak_mb_mean" in df.columns:
        gpu_mem = pd.to_numeric(df["gpu_mem_peak_mb_mean"], errors="coerce").fillna(0.0)
    else:
        gpu_mem = pd.Series(0.0, index=df.index)

    if "cpu_mem_mb_mean" in df.columns:
        cpu_mem = pd.to_numeric(df["cpu_mem_mb_mean"], errors="coerce").fillna(0.0)
    else:
        cpu_mem = pd.Series(0.0, index=df.index)

    df["gpu_mem_mb"] = gpu_mem
    df["cpu_mem_mb"] = cpu_mem

    keep = [
        "layer",
        "gpu_fwd_time_ms",
        "gpu_bwd_time_ms",
        "cpu_fwd_time_ms",
        "cpu_bwd_time_ms",
        "gpu_fwd_energy_j",
        "gpu_bwd_energy_j",
        "cpu_fwd_energy_j",
        "cpu_bwd_energy_j",
        "gpu_time_ms",
        "cpu_time_ms",
        "gpu_energy_j",
        "cpu_energy_j",
        "gpu_mem_mb",
        "cpu_mem_mb",
    ]
    return df[keep]


def simulate_plan(
    plan: ExecutionPlan,
    metrics_stats_csv: str | Path,
    graph_edges: List[Tuple[str, str]],
    transfer_costs: Dict[Tuple[str, str], float],
    cfg: SimulationConfig,
) -> SimulationResult:
    prof = _layer_profiles(metrics_stats_csv, mode=cfg.mode, k_sigma=cfg.k_sigma)
    prof_map = {str(r["layer"]): r for _, r in prof.iterrows()}

    warnings: List[str] = []
    violations: List[str] = []

    assigned_layers = set(plan.assignment_forward.keys()) | set(plan.assignment_backward.keys())
    profile_layers = set(prof_map.keys())

    missing_profiles = sorted(assigned_layers - profile_layers)
    if missing_profiles:
        violations.append(
            f"{len(missing_profiles)} assigned layers are missing from metrics_stats: {missing_profiles[:5]}"
        )

    extra_profiles = sorted(profile_layers - assigned_layers)
    if extra_profiles:
        warnings.append(
            f"{len(extra_profiles)} layers exist in metrics_stats but are not assigned in the plan"
        )

    graph_set = set(graph_edges)
    if cfg.strict_graph_subset:
        missing_graph = [e for e in (plan.cut_edges_forward + plan.cut_edges_backward) if e not in graph_set]
        if missing_graph:
            violations.append(
                f"{len(missing_graph)} cut edges are not present in graph_edges"
            )

    topo_warnings, topo_violations = _validate_topology(assigned_layers, graph_edges)
    warnings.extend(topo_warnings)
    if cfg.strict_topology:
        violations.extend(topo_violations)
    elif topo_violations:
        warnings.extend(topo_violations)

    # Topology and transfer validation on cut edges.
    transfer_missing = 0
    transfer_missing_edges: List[Tuple[str, str]] = []
    for phase_name, phase_edges, assignment in [
        ("forward", plan.cut_edges_forward, plan.assignment_forward),
        ("backward", plan.cut_edges_backward, plan.assignment_backward),
    ]:
        for (u, v) in phase_edges:
            if u not in assignment or v not in assignment:
                violations.append(f"{phase_name.title()} cut edge ({u}, {v}) references unknown layer")
                continue
            if assignment[u] == assignment[v]:
                violations.append(
                    f"{phase_name.title()} cut edge ({u}, {v}) is not a cut: both endpoints assigned to {assignment[u]}"
                )
            if (u, v) not in transfer_costs:
                transfer_missing += 1
                transfer_missing_edges.append((u, v))

    if transfer_missing > 0:
        msg = f"Missing transfer costs for {transfer_missing} cut edges; defaulting to 0.0"
        if cfg.strict_transfer_mapping:
            violations.append(msg)
        else:
            warnings.append(msg)

    # Cost aggregation.
    total_time_ms = 0.0
    total_energy_j = 0.0
    gpu_mem_forward_mb = 0.0
    cpu_mem_forward_mb = 0.0
    gpu_mem_backward_mb = 0.0
    cpu_mem_backward_mb = 0.0

    for layer, device in sorted(plan.assignment_forward.items()):
        if layer not in prof_map:
            continue
        row = prof_map[layer]
        if device == "GPU":
            total_time_ms += float(row["gpu_fwd_time_ms"])
            total_energy_j += float(row["gpu_fwd_energy_j"])
            gpu_mem_forward_mb += float(row["gpu_mem_mb"])
        elif device == "CPU":
            total_time_ms += float(row["cpu_fwd_time_ms"])
            total_energy_j += float(row["cpu_fwd_energy_j"])
            cpu_mem_forward_mb += float(row["cpu_mem_mb"])
        else:
            violations.append(f"Invalid forward device '{device}' for layer '{layer}'")

    for layer, device in sorted(plan.assignment_backward.items()):
        if layer not in prof_map:
            continue
        row = prof_map[layer]
        if device == "GPU":
            total_time_ms += float(row["gpu_bwd_time_ms"])
            total_energy_j += float(row["gpu_bwd_energy_j"])
            gpu_mem_backward_mb += float(row["gpu_mem_mb"])
        elif device == "CPU":
            total_time_ms += float(row["cpu_bwd_time_ms"])
            total_energy_j += float(row["cpu_bwd_energy_j"])
            cpu_mem_backward_mb += float(row["cpu_mem_mb"])
        else:
            violations.append(f"Invalid backward device '{device}' for layer '{layer}'")

    total_transfer_ms = 0.0
    for edge in plan.cut_edges_forward:
        total_transfer_ms += float(transfer_costs.get(edge, 0.0))
    for edge in plan.cut_edges_backward:
        total_transfer_ms += float(transfer_costs.get(edge, 0.0))
    for layer, _ in plan.cross_phase_edges:
        if layer in prof_map:
            total_transfer_ms += float(prof_map[layer]["gpu_fwd_time_ms"] + prof_map[layer]["gpu_bwd_time_ms"]) * 0.15

    # During backward pass, forward activations are retained for gradient computation,
    # so peak memory is the sum of both phases rather than the maximum of either alone.
    gpu_mem_used_mb = gpu_mem_forward_mb + gpu_mem_backward_mb
    cpu_mem_used_mb = cpu_mem_forward_mb + cpu_mem_backward_mb

    # Check combined peak against budget (consistent with how gpu_mem_used_mb is computed).
    if gpu_mem_used_mb > cfg.gpu_mem_budget_mb:
        violations.append(
            f"GPU memory violation: peak={gpu_mem_used_mb:.6f} "
            f"(forward={gpu_mem_forward_mb:.6f} + backward={gpu_mem_backward_mb:.6f}), "
            f"budget={cfg.gpu_mem_budget_mb:.6f}"
        )
    if cpu_mem_used_mb > cfg.cpu_mem_budget_mb:
        violations.append(
            f"CPU memory violation: peak={cpu_mem_used_mb:.6f} "
            f"(forward={cpu_mem_forward_mb:.6f} + backward={cpu_mem_backward_mb:.6f}), "
            f"budget={cfg.cpu_mem_budget_mb:.6f}"
        )

    objective = (
        (cfg.w_time * total_time_ms)
        + (cfg.w_energy * total_energy_j)
        + (cfg.w_transfer * total_transfer_ms)
    )

    status = "ok" if not violations else "invalid"

    layers_gpu = sum(1 for d in plan.assignment_forward.values() if d == "GPU")
    layers_cpu = sum(1 for d in plan.assignment_forward.values() if d == "CPU")

    return SimulationResult(
        status=status,
        objective_value=float(objective),
        total_time_ms=float(total_time_ms),
        total_energy_j=float(total_energy_j),
        total_transfer_ms=float(total_transfer_ms),
        gpu_mem_used_mb=float(gpu_mem_used_mb),
        cpu_mem_used_mb=float(cpu_mem_used_mb),
        layers_total=len(plan.assignment_forward),
        layers_gpu=layers_gpu,
        layers_cpu=layers_cpu,
        cut_edges_count=len(plan.cut_edges_forward) + len(plan.cut_edges_backward) + len(plan.cross_phase_edges),
        violations=violations,
        warnings=warnings,
    )


@dataclass
class SimulationResult4(SimulationResult):
    """Extended simulation result for Phase 4 with activation strategy costs."""

    activation_strategies: Dict[str, str] = field(default_factory=dict)
    recompute_layers: List[str] = field(default_factory=list)
    checkpoint_layers: List[str] = field(default_factory=list)
    total_recompute_cost_ms: float = 0.0
    total_checkpoint_cost_ms: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "activation_strategies": self.activation_strategies,
                "recompute_layers": self.recompute_layers,
                "checkpoint_layers": self.checkpoint_layers,
                "total_recompute_cost_ms": self.total_recompute_cost_ms,
                "total_checkpoint_cost_ms": self.total_checkpoint_cost_ms,
            }
        )
        return base_dict


def simulate_plan_phase4(
    plan: ExecutionPlan,
    metrics_stats_csv: str | Path,
    graph_edges: List[Tuple[str, str]],
    transfer_costs: Dict[Tuple[str, str], float],
    cfg: SimulationConfig,
    activation_strategies: Optional[Dict[str, str]] = None,
    recompute_cost_fraction: float = 0.50,
    checkpoint_cost_fraction: float = 0.15,
) -> SimulationResult4:
    """Simulate a plan under Phase 4 activation persistence strategies."""

    base_result = simulate_plan(plan, metrics_stats_csv, graph_edges, transfer_costs, cfg)

    if activation_strategies is None:
        activation_strategies = {layer: "retain" for layer in plan.assignment_forward}

    prof = _layer_profiles(metrics_stats_csv, mode=cfg.mode, k_sigma=cfg.k_sigma)
    prof_map = {str(r["layer"]): r for _, r in prof.iterrows()}

    recompute_layers: List[str] = []
    checkpoint_layers: List[str] = []
    total_recompute_cost_ms = 0.0
    total_checkpoint_cost_ms = 0.0
    adjusted_time_ms = base_result.total_time_ms

    for layer, strategy in activation_strategies.items():
        if layer not in plan.assignment_forward or layer not in prof_map:
            continue

        row = prof_map[layer]
        device = plan.assignment_forward[layer]
        forward_time = (
            float(row["gpu_fwd_time_ms"]) if device == "GPU" else float(row["cpu_fwd_time_ms"])
        )

        if strategy == "recompute":
            recompute_cost = forward_time * recompute_cost_fraction
            adjusted_time_ms += recompute_cost
            total_recompute_cost_ms += recompute_cost
            recompute_layers.append(layer)
        elif strategy == "checkpoint":
            checkpoint_cost = forward_time * checkpoint_cost_fraction
            adjusted_time_ms += checkpoint_cost
            total_checkpoint_cost_ms += checkpoint_cost
            checkpoint_layers.append(layer)

    adjusted_objective = (
        (cfg.w_time * adjusted_time_ms)
        + (cfg.w_energy * base_result.total_energy_j)
        + (cfg.w_transfer * base_result.total_transfer_ms)
    )

    return SimulationResult4(
        status=base_result.status,
        objective_value=float(adjusted_objective),
        total_time_ms=float(adjusted_time_ms),
        total_energy_j=base_result.total_energy_j,
        total_transfer_ms=base_result.total_transfer_ms,
        gpu_mem_used_mb=base_result.gpu_mem_used_mb,
        cpu_mem_used_mb=base_result.cpu_mem_used_mb,
        layers_total=base_result.layers_total,
        layers_gpu=base_result.layers_gpu,
        layers_cpu=base_result.layers_cpu,
        cut_edges_count=base_result.cut_edges_count,
        violations=base_result.violations,
        warnings=base_result.warnings,
        activation_strategies=activation_strategies,
        recompute_layers=recompute_layers,
        checkpoint_layers=checkpoint_layers,
        total_recompute_cost_ms=total_recompute_cost_ms,
        total_checkpoint_cost_ms=total_checkpoint_cost_ms,
    )
