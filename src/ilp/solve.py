from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import importlib
from typing import Dict, List, Tuple

from .data_loader import ILPInputData
from .model_builder import ILPConfig, build_problem_data


@dataclass
class ILPSolution:
    status: str
    backend: str
    objective_value: float
    assignment: Dict[str, str]
    gpu_mem_used_mb: float
    cpu_mem_used_mb: float
    cut_edges: List[Tuple[str, str]]


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


def solve_partition_ilp(data: ILPInputData, cfg: ILPConfig, backend: str = "auto") -> ILPSolution:
    if backend == "auto":
        try:
            importlib.import_module("pulp")

            return _solve_with_pulp(data, cfg)
        except Exception:
            return _solve_exhaustive(data, cfg)

    if backend == "pulp":
        return _solve_with_pulp(data, cfg)
    if backend == "exhaustive":
        return _solve_exhaustive(data, cfg)

    raise ValueError(f"Unsupported backend: {backend}")
