from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import importlib
from typing import Any, Dict, List, Tuple

from .data_loader import ILPInputData
from .model_builder import ILPConfig, build_problem_data, build_problem_data_dual, ILPConfig4, build_problem_data_phase4
from .advanced_terms import ActivationStrategy


@dataclass
class ILPSolution:
    status: str
    backend: str
    objective_value: float
    assignment: Dict[str, str]
    gpu_mem_used_mb: float
    cpu_mem_used_mb: float
    cut_edges: List[Tuple[str, str]]
    forward_assignment: Dict[str, str] | None = None
    backward_assignment: Dict[str, str] | None = None
    backward_cut_edges: List[Tuple[str, str]] | None = None
    cross_phase_edges: List[Tuple[str, str]] | None = None
    activation_strategies: Dict[str, Any] | None = None


def _eval_assignment(
    bits: Dict[str, int],
    data: ILPInputData,
    cfg: ILPConfig,
    problem,
):
    gpu_mem = sum(problem.gpu_mem[n] for n in data.nodes if bits[n] == 1)
    cpu_mem = sum(problem.cpu_mem[n] for n in data.nodes if bits[n] == 0)
    if gpu_mem > cfg.gpu_mem_budget_mb or cpu_mem > cfg.cpu_mem_budget_mb:
        return None

    obj = 0.0
    for n in data.nodes:
        obj += problem.objective_node_gpu[n] if bits[n] == 1 else problem.objective_node_cpu[n]

    cut_edges: List[Tuple[str, str]] = []
    for e in data.edges:
        u, v = e
        if bits[u] != bits[v]:
            obj += problem.objective_edge_cut[e]
            cut_edges.append(e)

    return obj, gpu_mem, cpu_mem, cut_edges


def _solve_exhaustive(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    problem = build_problem_data(data, cfg)

    n = len(data.nodes)
    if n > 22:
        raise RuntimeError(
            f"Exhaustive backend is limited to <=22 nodes, got {n}. "
            "Install PuLP for scalable MILP solving."
        )

    best = None
    best_bits = None
    best_cut = None
    best_gpu = 0.0
    best_cpu = 0.0

    for combo in product([0, 1], repeat=n):
        bits = {data.nodes[i]: combo[i] for i in range(n)}
        out = _eval_assignment(bits, data, cfg, problem)
        if out is None:
            continue
        obj, gpu_mem, cpu_mem, cut_edges = out
        if best is None or obj < best:
            best = obj
            best_bits = bits
            best_cut = cut_edges
            best_gpu = gpu_mem
            best_cpu = cpu_mem

    if best is None or best_bits is None or best_cut is None:
        return ILPSolution(
            status="infeasible",
            backend="exhaustive",
            objective_value=float("inf"),
            assignment={},
            gpu_mem_used_mb=0.0,
            cpu_mem_used_mb=0.0,
            cut_edges=[],
        )

    assignment = {k: ("GPU" if v == 1 else "CPU") for k, v in best_bits.items()}
    return ILPSolution(
        status="optimal",
        backend="exhaustive",
        objective_value=float(best),
        assignment=assignment,
        gpu_mem_used_mb=float(best_gpu),
        cpu_mem_used_mb=float(best_cpu),
        cut_edges=best_cut,
    )


def _solve_with_pulp(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    pulp = importlib.import_module("pulp")

    problem_data = build_problem_data(data, cfg)
    prob = pulp.LpProblem("cpu_gpu_partition", pulp.LpMinimize)

    x = {n: pulp.LpVariable(f"x_{n}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    y = {
        e: pulp.LpVariable(f"y_{e[0]}__{e[1]}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for e in data.edges
    }

    prob += (
        pulp.lpSum(problem_data.objective_node_gpu[n] * x[n] + problem_data.objective_node_cpu[n] * (1 - x[n]) for n in data.nodes)
        + pulp.lpSum(problem_data.objective_edge_cut[e] * y[e] for e in data.edges)
    )

    for (u, v) in data.edges:
        prob += y[(u, v)] >= x[u] - x[v]
        prob += y[(u, v)] >= x[v] - x[u]
        prob += y[(u, v)] <= x[u] + x[v]
        prob += y[(u, v)] <= 2 - x[u] - x[v]

    prob += pulp.lpSum(problem_data.gpu_mem[n] * x[n] for n in data.nodes) <= cfg.gpu_mem_budget_mb
    prob += pulp.lpSum(problem_data.cpu_mem[n] * (1 - x[n]) for n in data.nodes) <= cfg.cpu_mem_budget_mb

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, "unknown")
    if status.lower() not in {"optimal", "feasible"}:
        return ILPSolution(
            status=status.lower(),
            backend="pulp_cbc",
            objective_value=float("inf"),
            assignment={},
            gpu_mem_used_mb=0.0,
            cpu_mem_used_mb=0.0,
            cut_edges=[],
        )

    bits = {n: int(round(pulp.value(x[n]) or 0.0)) for n in data.nodes}
    assignment = {n: ("GPU" if bits[n] == 1 else "CPU") for n in data.nodes}

    cut_edges = [e for e in data.edges if bits[e[0]] != bits[e[1]]]

    gpu_mem = sum(problem_data.gpu_mem[n] for n in data.nodes if bits[n] == 1)
    cpu_mem = sum(problem_data.cpu_mem[n] for n in data.nodes if bits[n] == 0)

    return ILPSolution(
        status=status.lower(),
        backend="pulp_cbc",
        objective_value=float(pulp.value(prob.objective) or 0.0),
        assignment=assignment,
        gpu_mem_used_mb=float(gpu_mem),
        cpu_mem_used_mb=float(cpu_mem),
        cut_edges=cut_edges,
    )


def _eval_assignment_dual(
    fwd_bits: Dict[str, int],
    bwd_bits: Dict[str, int],
    data: ILPInputData,
    cfg: ILPConfig,
    problem,
):
    gpu_mem_fwd = sum(problem.gpu_mem[n] for n in data.nodes if fwd_bits[n] == 1)
    cpu_mem_fwd = sum(problem.cpu_mem[n] for n in data.nodes if fwd_bits[n] == 0)
    gpu_mem_bwd = sum(problem.gpu_mem[n] for n in data.nodes if bwd_bits[n] == 1)
    cpu_mem_bwd = sum(problem.cpu_mem[n] for n in data.nodes if bwd_bits[n] == 0)
    if (
        gpu_mem_fwd > cfg.gpu_mem_budget_mb
        or cpu_mem_fwd > cfg.cpu_mem_budget_mb
        or gpu_mem_bwd > cfg.gpu_mem_budget_mb
        or cpu_mem_bwd > cfg.cpu_mem_budget_mb
    ):
        return None

    obj = 0.0
    for n in data.nodes:
        obj += problem.objective_fwd_gpu[n] if fwd_bits[n] == 1 else problem.objective_fwd_cpu[n]
        obj += problem.objective_bwd_gpu[n] if bwd_bits[n] == 1 else problem.objective_bwd_cpu[n]
        if fwd_bits[n] != bwd_bits[n]:
            obj += problem.objective_cross_phase[n]

    cut_edges_fwd: List[Tuple[str, str]] = []
    cut_edges_bwd: List[Tuple[str, str]] = []
    for e in data.edges:
        u, v = e
        if fwd_bits[u] != fwd_bits[v]:
            obj += problem.objective_edge_cut_forward[e]
            cut_edges_fwd.append(e)
        if bwd_bits[u] != bwd_bits[v]:
            obj += problem.objective_edge_cut_backward[e]
            cut_edges_bwd.append(e)

    cross_phase_edges = [(n, n) for n in data.nodes if fwd_bits[n] != bwd_bits[n]]
    return obj, max(gpu_mem_fwd, gpu_mem_bwd), max(cpu_mem_fwd, cpu_mem_bwd), cut_edges_fwd, cut_edges_bwd, cross_phase_edges


def _solve_exhaustive_dual(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    problem = build_problem_data_dual(data, cfg)

    n = len(data.nodes)
    if n > 14:
        raise RuntimeError(
            f"Dual exhaustive backend is limited to <=14 nodes, got {n}. Install PuLP for scalable MILP solving."
        )

    best = None
    best_fwd_bits = None
    best_bwd_bits = None
    best_cut_fwd = None
    best_cut_bwd = None
    best_cross = None
    best_gpu = 0.0
    best_cpu = 0.0

    for combo_fwd in product([0, 1], repeat=n):
        fwd_bits = {data.nodes[i]: combo_fwd[i] for i in range(n)}
        for combo_bwd in product([0, 1], repeat=n):
            bwd_bits = {data.nodes[i]: combo_bwd[i] for i in range(n)}
            out = _eval_assignment_dual(fwd_bits, bwd_bits, data, cfg, problem)
            if out is None:
                continue
            obj, gpu_mem, cpu_mem, cut_fwd, cut_bwd, cross_edges = out
            if best is None or obj < best:
                best = obj
                best_fwd_bits = fwd_bits
                best_bwd_bits = bwd_bits
                best_cut_fwd = cut_fwd
                best_cut_bwd = cut_bwd
                best_cross = cross_edges
                best_gpu = gpu_mem
                best_cpu = cpu_mem

    if best is None or best_fwd_bits is None or best_bwd_bits is None or best_cut_fwd is None or best_cut_bwd is None or best_cross is None:
        return ILPSolution(
            status="infeasible",
            backend="exhaustive_dual",
            objective_value=float("inf"),
            assignment={},
            gpu_mem_used_mb=0.0,
            cpu_mem_used_mb=0.0,
            cut_edges=[],
            forward_assignment={},
            backward_assignment={},
            backward_cut_edges=[],
            cross_phase_edges=[],
        )

    forward_assignment = {k: ("GPU" if v == 1 else "CPU") for k, v in best_fwd_bits.items()}
    backward_assignment = {k: ("GPU" if v == 1 else "CPU") for k, v in best_bwd_bits.items()}
    return ILPSolution(
        status="optimal",
        backend="exhaustive_dual",
        objective_value=float(best),
        assignment=dict(forward_assignment),
        gpu_mem_used_mb=float(best_gpu),
        cpu_mem_used_mb=float(best_cpu),
        cut_edges=best_cut_fwd,
        forward_assignment=forward_assignment,
        backward_assignment=backward_assignment,
        backward_cut_edges=best_cut_bwd,
        cross_phase_edges=best_cross,
    )


def _solve_with_pulp_dual(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    pulp = importlib.import_module("pulp")

    problem_data = build_problem_data_dual(data, cfg)
    prob = pulp.LpProblem("cpu_gpu_partition_dual", pulp.LpMinimize)

    xf = {n: pulp.LpVariable(f"xf_{n}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    xb = {n: pulp.LpVariable(f"xb_{n}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    yf = {e: pulp.LpVariable(f"yf_{e[0]}__{e[1]}", lowBound=0, upBound=1, cat=pulp.LpBinary) for e in data.edges}
    yb = {e: pulp.LpVariable(f"yb_{e[0]}__{e[1]}", lowBound=0, upBound=1, cat=pulp.LpBinary) for e in data.edges}
    z = {n: pulp.LpVariable(f"z_{n}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}

    prob += (
        pulp.lpSum(problem_data.objective_fwd_gpu[n] * xf[n] + problem_data.objective_fwd_cpu[n] * (1 - xf[n]) for n in data.nodes)
        + pulp.lpSum(problem_data.objective_bwd_gpu[n] * xb[n] + problem_data.objective_bwd_cpu[n] * (1 - xb[n]) for n in data.nodes)
        + pulp.lpSum(problem_data.objective_edge_cut_forward[e] * yf[e] for e in data.edges)
        + pulp.lpSum(problem_data.objective_edge_cut_backward[e] * yb[e] for e in data.edges)
        + pulp.lpSum(problem_data.objective_cross_phase[n] * z[n] for n in data.nodes)
    )

    for (u, v) in data.edges:
        prob += yf[(u, v)] >= xf[u] - xf[v]
        prob += yf[(u, v)] >= xf[v] - xf[u]
        prob += yf[(u, v)] <= xf[u] + xf[v]
        prob += yf[(u, v)] <= 2 - xf[u] - xf[v]
        prob += yb[(u, v)] >= xb[u] - xb[v]
        prob += yb[(u, v)] >= xb[v] - xb[u]
        prob += yb[(u, v)] <= xb[u] + xb[v]
        prob += yb[(u, v)] <= 2 - xb[u] - xb[v]

    for n in data.nodes:
        prob += z[n] >= xf[n] - xb[n]
        prob += z[n] >= xb[n] - xf[n]
        prob += z[n] <= xf[n] + xb[n]
        prob += z[n] <= 2 - xf[n] - xb[n]

    prob += pulp.lpSum(problem_data.gpu_mem[n] * xf[n] for n in data.nodes) <= cfg.gpu_mem_budget_mb
    prob += pulp.lpSum(problem_data.cpu_mem[n] * (1 - xf[n]) for n in data.nodes) <= cfg.cpu_mem_budget_mb
    prob += pulp.lpSum(problem_data.gpu_mem[n] * xb[n] for n in data.nodes) <= cfg.gpu_mem_budget_mb
    prob += pulp.lpSum(problem_data.cpu_mem[n] * (1 - xb[n]) for n in data.nodes) <= cfg.cpu_mem_budget_mb

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, "unknown")
    if status.lower() not in {"optimal", "feasible"}:
        return ILPSolution(
            status=status.lower(),
            backend="pulp_cbc_dual",
            objective_value=float("inf"),
            assignment={},
            gpu_mem_used_mb=0.0,
            cpu_mem_used_mb=0.0,
            cut_edges=[],
            forward_assignment={},
            backward_assignment={},
            backward_cut_edges=[],
            cross_phase_edges=[],
        )

    fwd_bits = {n: int(round(pulp.value(xf[n]) or 0.0)) for n in data.nodes}
    bwd_bits = {n: int(round(pulp.value(xb[n]) or 0.0)) for n in data.nodes}
    forward_assignment = {n: ("GPU" if fwd_bits[n] == 1 else "CPU") for n in data.nodes}
    backward_assignment = {n: ("GPU" if bwd_bits[n] == 1 else "CPU") for n in data.nodes}
    cut_edges_fwd = [e for e in data.edges if fwd_bits[e[0]] != fwd_bits[e[1]]]
    cut_edges_bwd = [e for e in data.edges if bwd_bits[e[0]] != bwd_bits[e[1]]]
    cross_edges = [(n, n) for n in data.nodes if fwd_bits[n] != bwd_bits[n]]
    gpu_mem = max(
        sum(problem_data.gpu_mem[n] for n in data.nodes if fwd_bits[n] == 1),
        sum(problem_data.gpu_mem[n] for n in data.nodes if bwd_bits[n] == 1),
    )
    cpu_mem = max(
        sum(problem_data.cpu_mem[n] for n in data.nodes if fwd_bits[n] == 0),
        sum(problem_data.cpu_mem[n] for n in data.nodes if bwd_bits[n] == 0),
    )

    return ILPSolution(
        status=status.lower(),
        backend="pulp_cbc_dual",
        objective_value=float(pulp.value(prob.objective) or 0.0),
        assignment=dict(forward_assignment),
        gpu_mem_used_mb=float(gpu_mem),
        cpu_mem_used_mb=float(cpu_mem),
        cut_edges=cut_edges_fwd,
        forward_assignment=forward_assignment,
        backward_assignment=backward_assignment,
        backward_cut_edges=cut_edges_bwd,
        cross_phase_edges=cross_edges,
    )


def solve_partition_ilp(data: ILPInputData, cfg: ILPConfig, backend: str = "auto") -> ILPSolution:
    if backend == "auto":
        try:
            importlib.import_module("pulp")

            return _solve_with_pulp_dual(data, cfg)
        except Exception:
            return _solve_exhaustive_dual(data, cfg)

    if backend == "pulp":
        return _solve_with_pulp_dual(data, cfg)
    if backend == "exhaustive":
        return _solve_exhaustive_dual(data, cfg)

    raise ValueError(f"Unsupported backend: {backend}")


# ==================== Phase 4 Extensions ====================

@dataclass
class ILPSolution4(ILPSolution):
    """Extended solution for Phase 4 with activation strategy decisions."""
    activation_strategies: Dict[str, ActivationStrategy] = None
    mode: str = "phase3"  # "phase3" or "phase4"


def _solve_phase4_greedy(data: ILPInputData, cfg: ILPConfig4) -> ILPSolution4:
    """
    Phase 4 MVP solver using greedy heuristic for activation strategies.
    
    Strategy:
    1. Solve Phase 3 baseline (all nodes retain activations)
    2. For GPU nodes with high memory pressure, consider recompute/checkpoint
    3. Apply heuristic: if peak GPU mem > 80% of budget, try recompute on forward-heavy layers
    """
    # First, solve the Phase 3 baseline
    baseline_solution = solve_partition_ilp(data, cfg, backend="auto")
    
    if baseline_solution.status != "optimal":
        # Fallback: return Phase 3 solution with empty strategies
        return ILPSolution4(
            status=baseline_solution.status,
            backend="greedy_phase4",
            objective_value=baseline_solution.objective_value,
            assignment=baseline_solution.assignment,
            gpu_mem_used_mb=baseline_solution.gpu_mem_used_mb,
            cpu_mem_used_mb=baseline_solution.cpu_mem_used_mb,
            cut_edges=baseline_solution.cut_edges,
            activation_strategies={n: ActivationStrategy(n, retain=True) for n in data.nodes},
            mode="phase3",
        )
    
    # Build Phase 4 problem data
    problem4 = build_problem_data_phase4(data, cfg)
    
    # Initialize all nodes with retain strategy
    strategies = {n: ActivationStrategy(n, retain=True) for n in data.nodes}
    
    # Check if recompute/checkpoint can improve memory efficiency and objective.
    forward_assignment = baseline_solution.forward_assignment or baseline_solution.assignment
    backward_assignment = baseline_solution.backward_assignment or baseline_solution.assignment
    gpu_nodes = [n for n, device in forward_assignment.items() if device == "GPU"]
    gpu_mem_used = baseline_solution.gpu_mem_used_mb
    cpu_mem_used = baseline_solution.cpu_mem_used_mb
    gpu_budget = cfg.gpu_mem_budget_mb

    adjusted_objective = float(baseline_solution.objective_value)

    def _node_delta_time_ms(node: str, strategy_name: str) -> float:
        on_gpu = forward_assignment.get(node, "CPU") == "GPU"
        if strategy_name == "recompute":
            table = problem4.recompute_cost_gpu if on_gpu else problem4.recompute_cost_cpu
            return float(table.get(node, 0.0))
        if strategy_name == "checkpoint":
            table = problem4.checkpoint_cost_gpu if on_gpu else problem4.checkpoint_cost_cpu
            return float(table.get(node, 0.0))
        return 0.0

    def _node_delta_energy_j(node: str, strategy_name: str) -> float:
        if strategy_name == "checkpoint":
            return float(problem4.activation_meta.node_energy_io_j.get(node, 0.0))
        return 0.0

    def _apply_strategy(node: str, strategy_name: str) -> None:
        nonlocal gpu_mem_used, cpu_mem_used, adjusted_objective
        if strategy_name == "recompute":
            strategies[node] = ActivationStrategy(node, recompute=True)
            if forward_assignment.get(node, "CPU") == "GPU":
                gpu_mem_used = max(0.0, gpu_mem_used - problem4.activation_meta.node_mem_activation_mb.get(node, 0.0))
        elif strategy_name == "checkpoint":
            strategies[node] = ActivationStrategy(node, checkpoint=True)
            act_mem = problem4.activation_meta.node_mem_activation_mb.get(node, 0.0)
            if forward_assignment.get(node, "CPU") == "GPU":
                gpu_mem_used = max(0.0, gpu_mem_used - act_mem)
                cpu_mem_used += act_mem

        delta_time = _node_delta_time_ms(node, strategy_name)
        delta_energy = _node_delta_energy_j(node, strategy_name)
        adjusted_objective += (cfg.w_time * delta_time) + (cfg.w_energy * delta_energy) + (cfg.w_io * delta_time)
    
    # If GPU memory utilization > 60%, consider recompute/checkpoint.
    # Prefer larger memory savings for smaller time penalties.
    if gpu_mem_used > 0.6 * gpu_budget:
        candidates: List[Tuple[float, str, str]] = []
        for n in gpu_nodes:
            act_mem = float(problem4.activation_meta.node_mem_activation_mb.get(n, 0.0))
            if act_mem <= 0.0:
                continue

            if cfg.enable_recompute:
                recomp_cost = float(problem4.recompute_cost_gpu.get(n, 0.0))
                score = act_mem / (recomp_cost + 1e-6)
                candidates.append((score, n, "recompute"))

            if cfg.enable_checkpoint:
                checkpoint_cost = float(problem4.checkpoint_cost_gpu.get(n, 0.0))
                score = act_mem / (checkpoint_cost + 1e-6)
                candidates.append((score, n, "checkpoint"))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: set[str] = set()
        target_gpu_mem = min(gpu_budget, gpu_mem_used) * 0.90

        for _, node, strategy_name in candidates:
            if node in selected:
                continue
            if gpu_mem_used <= target_gpu_mem:
                break
            _apply_strategy(node, strategy_name)
            selected.add(node)
    
    # Return Phase 4 solution
    return ILPSolution4(
        status=baseline_solution.status,
        backend="greedy_phase4",
        objective_value=float(adjusted_objective),
        assignment=dict(forward_assignment),
        gpu_mem_used_mb=float(gpu_mem_used),
        cpu_mem_used_mb=float(cpu_mem_used),
        cut_edges=list(baseline_solution.cut_edges),
        forward_assignment=dict(forward_assignment),
        backward_assignment=dict(backward_assignment),
        backward_cut_edges=list(baseline_solution.backward_cut_edges or []),
        cross_phase_edges=list(baseline_solution.cross_phase_edges or []),
        activation_strategies=strategies,
        mode="phase4",
    )


def solve_partition_ilp_phase4(
    data: ILPInputData,
    cfg: ILPConfig4,
    backend: str = "greedy"
) -> ILPSolution4:
    """
    Solve Phase 4 problem with activation persistence strategies.
    
    Args:
      data: ILP input data (extended with activation metadata)
      cfg: ILPConfig4 with Phase 4 parameters
      backend: "greedy" (available now) or "pulp" (future)
    
    Returns:
      ILPSolution4 with activation strategies per node
    """
    if backend == "greedy":
        return _solve_phase4_greedy(data, cfg)
    
    raise ValueError(f"Unsupported Phase 4 backend: {backend}")

