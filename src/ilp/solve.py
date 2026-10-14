from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import importlib
from typing import Any, Dict, List, Tuple

from .data_loader import ILPInputData
from .model_builder import ILPConfig, build_problem_data, build_problem_data_dual, ILPConfig4, build_problem_data_phase4
from .advanced_terms import ActivationStrategy


def _interpolate_recompute_curve(points: List[Tuple[int, float]]) -> Dict[int, float]:
    """Linearly interpolates measured (n_blocks, peak_gpu_mem_mb) points
    (see src/runner/recompute_curve_calibration.py) to every integer
    n_blocks between the smallest and largest measured, returning each
    one's delta relative to the n_blocks=0 point (negative = real
    savings, positive = real increase -- both occur, see
    fix/07_gpu_recompute_joint_decision/README.md). Returns {} if points
    is empty or has no n_blocks=0 entry to anchor the baseline against.
    """
    if not points:
        return {}
    pts = sorted(points, key=lambda p: p[0])
    baseline = next((mb for n, mb in pts if n == 0), None)
    if baseline is None:
        return {}
    delta: Dict[int, float] = {pts[0][0]: pts[0][1] - baseline}
    for (n0, mb0), (n1, mb1) in zip(pts, pts[1:]):
        if n1 <= n0:
            continue
        for k in range(n0, n1 + 1):
            t = (k - n0) / (n1 - n0)
            delta[k] = (mb0 + t * (mb1 - mb0)) - baseline
    return delta


def _block_id_for_layer_name(layer_name: str) -> str:
    # Keep in sync with src/runtime/hybrid_executor.py's own
    # _block_id_for_layer_name (duplicated rather than imported: `ilp` is
    # the lower layer, `runtime` already imports FROM `ilp`, so importing
    # the other way would invert that dependency). Identifies which
    # repeated block (nn.ModuleList-style, e.g. "transformer.h.{i}" for
    # GPT-2, "model.layers.{i}" for LLaMA-style models) a leaf layer
    # belongs to, by truncating its name at the last purely-numeric path
    # segment. A layer with no numeric segment (e.g. "lm_head") is its own
    # block of size 1.
    parts = layer_name.split(".")
    last_numeric_idx = None
    for i, part in enumerate(parts):
        if part.isdigit():
            last_numeric_idx = i
    if last_numeric_idx is None:
        return layer_name
    return ".".join(parts[: last_numeric_idx + 1])


def _group_nodes_into_activation_candidates(
    nodes: List[str], forward_assignment: Dict[str, str]
) -> List[List[str]]:
    # For a decoder-only (torch.export, operator-granularity) graph,
    # recompute/checkpoint only reduces real memory when applied to an
    # entire repeated block's worth of leaves together -- confirmed
    # empirically, see fix/06 README: a single leaf's own ATen ops save
    # ~0% in isolation (checkpointing protects a node's small INPUTS, not
    # its much larger OUTPUT, which is what's actually expensive), while a
    # whole transformer block saved ~70-74%. The runtime's own grouping
    # (_detect_checkpoint_groups) silently gets ZERO benefit from a
    # strategy applied to only SOME of a block's leaves, so the greedy
    # heuristic must offer (and later apply) a whole block atomically, or
    # its predicted memory savings won't match what the runtime actually
    # delivers -- exactly the mismatch that motivated this fix (measured:
    # the ILP assumed ~68ms/some-MB for a partial selection, the runtime
    # delivered 0 either way). A block is only offered as a candidate when
    # ALL its leaves share the same device assignment -- the runtime's own
    # group-level device-consistency check (_prepare_segment_fn) would
    # otherwise refuse to group it too, for the same reason.
    by_block: Dict[str, List[str]] = {}
    for n in nodes:
        by_block.setdefault(_block_id_for_layer_name(n), []).append(n)
    groups: List[List[str]] = []
    for block_nodes in by_block.values():
        devices = {forward_assignment.get(n, "CPU") for n in block_nodes}
        if len(devices) == 1 and next(iter(devices)) == "GPU":
            groups.append(block_nodes)
    return groups


def _sanitize_lp_name(name: str) -> str:
    """Sanitize a layer name for use as a PuLP/CBC variable identifier.
    CBC rejects variable names that contain dots, brackets, or hyphens."""
    return (
        name.replace(".", "_")
            .replace("[", "_lb_")
            .replace("]", "_rb_")
            .replace("-", "_")
    )


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


def _build_cbc_solver(pulp):
    """Create a CBC solver bound to the conda-installed system 'cbc' binary.

    Uses COIN_CMD rather than PULP_CBC_CMD: on a conda-forge PuLP install
    (no bundled cbc binary shipped -- this project's environment.yml pairs
    `pulp` with the separate `coin-or-cbc` conda package), PULP_CBC_CMD's
    class-level availability check does `os.access("cbc", os.X_OK)` on the
    bare command name as a relative path (not a PATH-aware lookup), which is
    always False, and permanently disables the solver for the process on
    the resulting exception. COIN_CMD's availability check correctly uses
    shutil.which, resolving the same system 'cbc' either way -- both classes
    end up invoking the identical conda coin-or-cbc binary in this
    environment, so this only fixes detection, not which binary runs.
    """
    return pulp.COIN_CMD(msg=False, timeLimit=300, gapRel=0.01)


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
    """
    DEPRECATED: Single-phase (inference-style) ILP solver.

    This solver treats the model as a single pass and does not distinguish between
    forward and backward phases. It is NOT correct for training workloads.
    No active call path invokes this function; all training paths use
    _solve_with_pulp_dual() which models both phases (xf, xb) correctly.
    """
    pulp = importlib.import_module("pulp")


    problem_data = build_problem_data(data, cfg)
    prob = pulp.LpProblem("cpu_gpu_partition", pulp.LpMinimize)

    x = {n: pulp.LpVariable(f"x_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    y = {
        e: pulp.LpVariable(f"y_{_sanitize_lp_name(e[0])}__{_sanitize_lp_name(e[1])}", lowBound=0, upBound=1, cat=pulp.LpBinary)
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

    solver = _build_cbc_solver(pulp)
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


def _derive_congestion_knee_ms(data: ILPInputData, cfg: ILPConfig) -> float:
    if cfg.congestion_knee_ms > 0.0:
        return float(cfg.congestion_knee_ms)

    positive_edges = sorted(max(float(data.edge_transfer_ms.get(e, 0.0)), 0.0) for e in data.edges)
    positive_edges = [v for v in positive_edges if v > 0.0]
    if not positive_edges:
        return float("inf")

    # Auto-knee: allow roughly one-third of the lightest cuts before extra congestion penalty.
    quota = max(1, len(positive_edges) // 3)
    return float(sum(positive_edges[:quota]))


def _eval_assignment_dual(
    fwd_bits: Dict[str, int],
    bwd_bits: Dict[str, int],
    data: ILPInputData,
    cfg: ILPConfig,
    problem,
    congestion_knee_ms: float | None = None,
):
    # Base memory (weights, gradients, optimizer states) must reside in GPU whenever
    # a layer runs on GPU in EITHER phase — use OR.
    #
    # Activation memory must be present on whichever device runs BACKWARD,
    # full stop — that is what actually consumes it (gradient computation).
    # The original code used AND(fwd==GPU, bwd==GPU) instead, which happens
    # to agree with "charge the backward device" whenever fwd==bwd (the
    # common case), but for a cross-phase node (fwd=CPU/bwd=GPU or
    # fwd=GPU/bwd=CPU) it charged NEITHER device anything: that silently
    # told the ILP cross-phase placement was memory-free, which is exactly
    # what let it produce the pathologically fragmented, still-OOMing plans
    # seen for resnet152 at large batch sizes (hybrid_executor.py's
    # dual-placement recompute does a full forward replay on the backward
    # device first, which very much needs that device's memory). See
    # src/ilp/topology_memory.py's module docstring for the full context.
    combined_gpu_base = sum(problem.gpu_base_mem[n] for n in data.nodes if fwd_bits[n] == 1 or bwd_bits[n] == 1)
    combined_gpu_act = sum(problem.gpu_act_mem[n] for n in data.nodes if bwd_bits[n] == 1)
    combined_cpu_base = sum(problem.cpu_base_mem[n] for n in data.nodes if fwd_bits[n] == 0 or bwd_bits[n] == 0)
    combined_cpu_act = sum(problem.cpu_act_mem[n] for n in data.nodes if bwd_bits[n] == 0)
    combined_gpu_mem = combined_gpu_base + combined_gpu_act
    combined_cpu_mem = combined_cpu_base + combined_cpu_act
    if (
        combined_gpu_mem > cfg.gpu_mem_budget_mb
        or combined_cpu_mem > cfg.cpu_mem_budget_mb
    ):
        return None

    obj = 0.0
    for n in data.nodes:
        obj += problem.objective_fwd_gpu[n] if fwd_bits[n] == 1 else problem.objective_fwd_cpu[n]
        obj += problem.objective_bwd_gpu[n] if bwd_bits[n] == 1 else problem.objective_bwd_cpu[n]
        if fwd_bits[n] != bwd_bits[n]:
            # fwd=CPU,bwd=GPU (fwd_bits=0,bwd_bits=1) -> recompute on GPU.
            # fwd=GPU,bwd=CPU (fwd_bits=1,bwd_bits=0) -> recompute on CPU.
            obj += (
                problem.objective_cross_phase_cpu_to_gpu[n]
                if bwd_bits[n] == 1
                else problem.objective_cross_phase_gpu_to_cpu[n]
            )

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

    if cfg.w_congestion > 0.0 and data.edges:
        knee = congestion_knee_ms if congestion_knee_ms is not None else _derive_congestion_knee_ms(data, cfg)
        frontier_fwd_ms = sum(max(float(data.edge_transfer_ms.get(e, 0.0)), 0.0) for e in cut_edges_fwd)
        frontier_bwd_ms = sum(max(float(data.edge_transfer_ms.get(e, 0.0)), 0.0) for e in cut_edges_bwd)
        overflow_fwd = max(0.0, frontier_fwd_ms - knee)
        overflow_bwd = max(0.0, frontier_bwd_ms - knee)
        obj += cfg.w_congestion * (overflow_fwd + overflow_bwd)

    cross_phase_edges = [(n, n) for n in data.nodes if fwd_bits[n] != bwd_bits[n]]
    return obj, combined_gpu_mem, combined_cpu_mem, cut_edges_fwd, cut_edges_bwd, cross_phase_edges


def _solve_exhaustive_dual(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    problem = build_problem_data_dual(data, cfg)
    congestion_knee = _derive_congestion_knee_ms(data, cfg)

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
            out = _eval_assignment_dual(
                fwd_bits,
                bwd_bits,
                data,
                cfg,
                problem,
                congestion_knee_ms=congestion_knee,
            )
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


def _build_tight_topological_gpu_constraints(
    prob: Any,
    data: ILPInputData,
    xb: Dict[str, Any],
    gpu_budget_mb: float,
) -> None:
    """Replaces the single aggregate `Σ gpu_act_mem[n]*xb[n] <= budget`
    constraint (memory_model="topological") with a family of linear
    "critical point" constraints -- one per node's forward position and one
    per node's backward position in the REAL topological walk order -- that
    together are a faithful, validated-to-floating-point-precision replica
    of src/ilp/topology_memory.py's PreparedTopologyMemory.evaluate() (see
    fix/09_.../README.md: validated against 6 real plans across
    resnet50/resnet152/distilgpt2/gpt2_small, all within +/-0.02%).

    Only valid when every node's forward and backward device assignment are
    forced equal (see the `xf[n] == xb[n]` constraint _solve_with_pulp_dual
    adds for memory_model="topological_tight" before calling this) -- a
    node is then simply retained continuously from its forward position to
    its backward position, with no cross-phase/checkpoint/recompute
    lifetime complexity to model. Callers must not call this otherwise.

    The topological order and per-node retained/transient/weight-group data
    come from `data.prepared_topology` (a `PreparedTopologyMemory`, the
    SAME already-validated object topology_memory.py's post-solve repair
    already uses) -- never recomputed here.
    """
    pulp = importlib.import_module("pulp")

    prepared = data.prepared_topology
    if prepared is None:
        raise ValueError(
            "memory_model='topological_tight' requires ILPInputData.prepared_topology to be set "
            "before calling solve_partition_ilp (see validation/run_ilp_partition.py)."
        )

    node_layer = prepared.node_layer
    retained = prepared.retained
    transient_fwd = prepared.transient_fwd
    transient_bwd = prepared.transient_bwd
    params = prepared.params
    weight_group = prepared.weight_group
    group_static_mb = prepared.group_static_mb

    # Only node_ids whose layer is an actual ILP decision node participate --
    # topology_memory.py's own graph can include layers this ILP run never
    # loaded stats for (already filtered out by PreparedTopologyMemory.prepare,
    # but double-checked here defensively).
    relevant = [nid for nid in prepared.topo if node_layer[nid] in xb]

    # Static (weights) memory, deduplicated by weight-sharing group -- same
    # dedup PreparedTopologyMemory.evaluate() itself uses for tied weights
    # (e.g. GPT-2's lm_head/wte). A group with a single member uses xb[m]
    # directly; a group with >1 member (the real case: tied pairs) gets an
    # auxiliary OR-indicator variable, standard linear MILP technique.
    groups_members: Dict[str, List[str]] = {}
    for nid in relevant:
        name = node_layer[nid]
        group = weight_group.get(name, name)
        members = groups_members.setdefault(group, [])
        if name not in members:
            members.append(name)

    group_gpu_indicator: Dict[str, Any] = {}
    for group, members in groups_members.items():
        if len(members) == 1:
            group_gpu_indicator[group] = xb[members[0]]
        else:
            indicator = pulp.LpVariable(
                f"topo_tight_group_gpu_{_sanitize_lp_name(group)}", lowBound=0, upBound=1, cat=pulp.LpBinary
            )
            for m in members:
                prob += indicator >= xb[m]
            prob += indicator <= pulp.lpSum(xb[m] for m in members)
            group_gpu_indicator[group] = indicator

    static_gpu_expr = pulp.lpSum(
        group_static_mb.get(group, 0.0) * indicator for group, indicator in group_gpu_indicator.items()
    )

    # Forward walk: one constraint per node, using the running sum of every
    # node produced strictly before it (and not yet freed -- irrelevant
    # here since nothing is freed during forward) -- mirrors
    # PreparedTopologyMemory.evaluate()'s own forward loop exactly.
    running_retained: Any = 0
    constraint_idx = 0
    for nid in relevant:
        name = node_layer[nid]
        prob += (
            static_gpu_expr + running_retained + transient_fwd[nid] * xb[name] <= gpu_budget_mb,
            f"topo_tight_fwd_{constraint_idx}",
        )
        constraint_idx += 1
        running_retained = running_retained + retained[nid] * xb[name]

    # Backward walk (reverse order): grad memory accrues progressively
    # (once per weight group, never subtracted -- mirrors .evaluate()'s own
    # grad_charged_groups bookkeeping), retained sum shrinks as each node's
    # own activation is freed right after its own constraint is built.
    grad_accum: Any = 0
    charged_grad_groups: set = set()
    for nid in reversed(relevant):
        name = node_layer[nid]
        group = weight_group.get(name, name)
        prob += (
            static_gpu_expr + grad_accum + running_retained + transient_bwd[nid] * xb[name] <= gpu_budget_mb,
            f"topo_tight_bwd_{constraint_idx}",
        )
        constraint_idx += 1
        if group not in charged_grad_groups:
            grad_accum = grad_accum + params[nid] * group_gpu_indicator[group]
            charged_grad_groups.add(group)
        running_retained = running_retained - retained[nid] * xb[name]


def _solve_with_pulp_dual(data: ILPInputData, cfg: ILPConfig) -> ILPSolution:
    pulp = importlib.import_module("pulp")

    problem_data = build_problem_data_dual(data, cfg)
    congestion_knee = _derive_congestion_knee_ms(data, cfg)
    prob = pulp.LpProblem("cpu_gpu_partition_dual", pulp.LpMinimize)

    # Joint device+activation-strategy decision (see
    # fix/07_gpu_recompute_joint_decision/README.md): only ILPConfig4
    # callers that explicitly opt in set this; a plain ILPConfig (every
    # existing call site) has no such attribute, so getattr's default keeps
    # this branch fully inert and the MILP mathematically identical to
    # before this change.
    enable_joint = bool(getattr(cfg, "enable_joint_activation_strategy", False))
    enable_joint_recompute = enable_joint and bool(getattr(cfg, "enable_recompute", True))
    enable_joint_checkpoint = enable_joint and bool(getattr(cfg, "enable_checkpoint", False))
    if cfg.memory_model == "topological_tight" and (enable_joint_recompute or enable_joint_checkpoint):
        # See fix/09_.../README.md: the tight per-critical-point GPU memory
        # constraint (built below) assumes every node is retained
        # continuously from its forward position to its backward position
        # -- modeling recompute/checkpoint correctly would need linearizing
        # products of binary variables (whether a node is still "resident"
        # at a given point would itself depend on a decision variable),
        # deliberately out of scope for this version. use_calibrated_curve
        # (below) can only be true when enable_joint_recompute already is,
        # so checking these two covers it too.
        raise ValueError(
            "memory_model='topological_tight' does not support "
            "enable_joint_recompute/enable_joint_checkpoint (Phase4 activation "
            "strategies) yet -- use memory_model='topological' for those."
        )
    w_recompute_penalty = float(getattr(cfg, "w_recompute_penalty", 0.5))
    w_io = float(getattr(cfg, "w_io", 0.0))
    recompute_cost_gpu = {
        n: w_recompute_penalty * float(data.node_cost_gpu_fwd_ms.get(n, 0.0)) for n in data.nodes
    }
    checkpoint_cost_gpu = {
        n: w_io * float(data.node_time_io_ms.get(n, 0.0)) for n in data.nodes
    }

    xf = {n: pulp.LpVariable(f"xf_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    xb = {n: pulp.LpVariable(f"xb_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    if cfg.memory_model == "topological_tight":
        # Forces same-device placement, eliminating cross-phase assignment
        # by construction -- see fix/09_.../README.md for why: the tight
        # constraint below assumes a node is retained continuously from its
        # forward position to its backward position, which is only valid
        # when fwd==bwd for every node (no cheaper to model correctly for
        # cross-phase here -- would need linearizing products of binaries).
        for n in data.nodes:
            prob += xf[n] == xb[n], f"topological_tight_same_device_{_sanitize_lp_name(n)}"

    # Calibrated, piecewise-linear recompute memory curve (see
    # fix/07_gpu_recompute_joint_decision/README.md and
    # src/runner/recompute_curve_calibration.py): recompute's real memory
    # benefit is non-additive -- it rises, peaks, then reverses to WORSE
    # than nothing as more blocks are marked simultaneously, in a way two
    # simpler hypotheses (a fixed per-frame cost; an inherently "bad"
    # block) both failed to explain. Rather than guess a formula, the real
    # curve is measured once per model/batch and handed to the MILP as
    # data. Only active when a real curve was actually calibrated and
    # handed in via cfg -- otherwise this whole mechanism is inert and the
    # plain per-node term below (unchanged from before) is used instead.
    recompute_block_order: List[List[str]] = list(getattr(cfg, "recompute_block_order", None) or [])
    recompute_curve_points: List[Tuple[int, float]] = list(getattr(cfg, "recompute_curve_points", None) or [])
    recompute_curve_delta = _interpolate_recompute_curve(recompute_curve_points)
    use_calibrated_curve = bool(enable_joint_recompute and recompute_block_order and recompute_curve_delta)

    node_to_block_idx: Dict[str, int] = {}
    block_recompute: List[Any] = []
    predicted_recompute_delta_mb: Any = 0
    if use_calibrated_curve:
        for b_idx, group in enumerate(recompute_block_order):
            for n in group:
                if n not in data.shared_call_site_layers:
                    node_to_block_idx[n] = b_idx
        n_blocks_total = len(recompute_block_order)
        block_recompute = [
            pulp.LpVariable(f"block_recompute_{b_idx}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            for b_idx in range(n_blocks_total)
        ]
        # The calibration measured peak memory for PREFIXES of
        # recompute_block_order (highest theoretical savings first) --
        # never for an arbitrary subset. Force the MILP to only ever
        # choose a prefix, or the calibrated curve would be applied to a
        # combination it was never measured for.
        for b_idx in range(n_blocks_total - 1):
            prob += block_recompute[b_idx] >= block_recompute[b_idx + 1]

        max_measured_k = max(recompute_curve_delta.keys())
        total_recompute_blocks = pulp.lpSum(block_recompute)
        if max_measured_k < n_blocks_total:
            # Never let the solver pick more blocks than were actually
            # calibrated -- extrapolating this curve past what was
            # measured has no empirical support either way.
            prob += total_recompute_blocks <= max_measured_k
        is_k = {
            k: pulp.LpVariable(f"recompute_is_k_{k}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            for k in range(max_measured_k + 1)
        }
        prob += pulp.lpSum(is_k.values()) == 1
        prob += total_recompute_blocks == pulp.lpSum(k * is_k[k] for k in is_k)
        predicted_recompute_delta_mb = pulp.lpSum(
            is_k[k] * recompute_curve_delta.get(k, 0.0) for k in is_k
        )

    # A layer name reused at multiple call sites (data.shared_call_site_layers
    # -- see its docstring) never gets a recompute[n]/checkpoint[n] variable
    # at all: it is a plain int 0, exactly like the "joint flag off" case
    # below, for that specific node. This is a complete exclusion, not a
    # best-effort skip -- a single strategy value cannot safely describe
    # multiple distinct call sites of the same module (confirmed to crash
    # the runtime otherwise, see fix/07_gpu_recompute_joint_decision/
    # README.md), so the MILP must never even consider the option.
    #
    # When use_calibrated_curve is active, a node's recompute[n] is the
    # SAME shared variable as every other node in its block
    # (block_recompute[b]) -- one decision per block, matching the
    # granularity the runtime actually delivers savings at (see
    # fix/07's grouping work) and the granularity the curve above was
    # calibrated at. A node outside every calibrated block gets 0: this
    # mode covers exactly what was measured, nothing more.
    if use_calibrated_curve:
        recompute = {n: block_recompute[node_to_block_idx[n]] if n in node_to_block_idx else 0 for n in data.nodes}
    else:
        recompute = (
            {
                n: (0 if n in data.shared_call_site_layers else
                    pulp.LpVariable(f"recompute_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary))
                for n in data.nodes
            }
            if enable_joint_recompute else {n: 0 for n in data.nodes}
        )
    checkpoint = (
        {
            n: (0 if n in data.shared_call_site_layers else
                pulp.LpVariable(f"checkpoint_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary))
            for n in data.nodes
        }
        if enable_joint_checkpoint else {n: 0 for n in data.nodes}
    )
    if enable_joint_recompute or enable_joint_checkpoint:
        for n in data.nodes:
            if n in data.shared_call_site_layers:
                continue  # no variable exists for this node; nothing to constrain
            # Only meaningful when this node's backward already runs on
            # GPU (xb[n]=1) -- that is the memory this strategy would
            # relieve. Mutually exclusive with each other; both at 0 means
            # "retain", identical to today's only behavior. Skipped for
            # calibrated-block nodes when use_calibrated_curve: that
            # constraint is enforced once per BLOCK instead, right after
            # block_recompute is built above (same inequality, same
            # effect, just not duplicated once per member node).
            if n in node_to_block_idx:
                continue
            if enable_joint_recompute:
                prob += recompute[n] <= xb[n]
            if enable_joint_checkpoint:
                prob += checkpoint[n] <= xb[n]
            if enable_joint_recompute and enable_joint_checkpoint:
                prob += recompute[n] + checkpoint[n] <= xb[n]
    if use_calibrated_curve:
        for n, b_idx in node_to_block_idx.items():
            prob += block_recompute[b_idx] <= xb[n]
    yf = {e: pulp.LpVariable(f"yf_{_sanitize_lp_name(e[0])}__{_sanitize_lp_name(e[1])}", lowBound=0, upBound=1, cat=pulp.LpBinary) for e in data.edges}
    yb = {e: pulp.LpVariable(f"yb_{_sanitize_lp_name(e[0])}__{_sanitize_lp_name(e[1])}", lowBound=0, upBound=1, cat=pulp.LpBinary) for e in data.edges}
    # Direction matters: fwd=CPU/bwd=GPU needs a GPU recompute (cheap if GPU
    # is fast); fwd=GPU/bwd=CPU needs a CPU recompute (expensive, given this
    # project's measured ~15-17x CPU/GPU compute gap). A single undirected
    # indicator would price both the same, which is exactly the bug being
    # fixed here -- see objective_cross_phase_cpu_to_gpu/_gpu_to_cpu.
    c_cpu_gpu = {n: pulp.LpVariable(f"c_cpu_gpu_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    c_gpu_cpu = {n: pulp.LpVariable(f"c_gpu_cpu_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    overflow_fwd = pulp.LpVariable("congestion_overflow_fwd", lowBound=0.0, cat=pulp.LpContinuous)
    overflow_bwd = pulp.LpVariable("congestion_overflow_bwd", lowBound=0.0, cat=pulp.LpContinuous)
    edge_congestion_ms = {e: max(float(data.edge_transfer_ms.get(e, 0.0)), 0.0) for e in data.edges}

    prob += (
        pulp.lpSum(problem_data.objective_fwd_gpu[n] * xf[n] + problem_data.objective_fwd_cpu[n] * (1 - xf[n]) for n in data.nodes)
        + pulp.lpSum(problem_data.objective_bwd_gpu[n] * xb[n] + problem_data.objective_bwd_cpu[n] * (1 - xb[n]) for n in data.nodes)
        + pulp.lpSum(problem_data.objective_edge_cut_forward[e] * yf[e] for e in data.edges)
        + pulp.lpSum(problem_data.objective_edge_cut_backward[e] * yb[e] for e in data.edges)
        + pulp.lpSum(
            problem_data.objective_cross_phase_cpu_to_gpu[n] * c_cpu_gpu[n]
            + problem_data.objective_cross_phase_gpu_to_cpu[n] * c_gpu_cpu[n]
            for n in data.nodes
        )
        + cfg.w_congestion * (overflow_fwd + overflow_bwd)
        + (
            pulp.lpSum(recompute_cost_gpu[n] * recompute[n] for n in data.nodes)
            if enable_joint_recompute else 0
        )
        + (
            pulp.lpSum(checkpoint_cost_gpu[n] * checkpoint[n] for n in data.nodes)
            if enable_joint_checkpoint else 0
        )
    )

    total_frontier_fwd = pulp.lpSum(edge_congestion_ms[e] * yf[e] for e in data.edges)
    total_frontier_bwd = pulp.lpSum(edge_congestion_ms[e] * yb[e] for e in data.edges)
    prob += overflow_fwd >= total_frontier_fwd - congestion_knee
    prob += overflow_bwd >= total_frontier_bwd - congestion_knee

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
        # c_cpu_gpu[n] = 1 iff xf[n]=0 (fwd=CPU) and xb[n]=1 (bwd=GPU).
        prob += c_cpu_gpu[n] >= xb[n] - xf[n]
        prob += c_cpu_gpu[n] <= xb[n]
        prob += c_cpu_gpu[n] <= 1 - xf[n]
        # c_gpu_cpu[n] = 1 iff xf[n]=1 (fwd=GPU) and xb[n]=0 (bwd=CPU).
        prob += c_gpu_cpu[n] >= xf[n] - xb[n]
        prob += c_gpu_cpu[n] <= xf[n]
        prob += c_gpu_cpu[n] <= 1 - xb[n]

    z_v = {n: pulp.LpVariable(f"z_v_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}
    z_cpu = {n: pulp.LpVariable(f"z_cpu_{_sanitize_lp_name(n)}", lowBound=0, upBound=1, cat=pulp.LpBinary) for n in data.nodes}

    for n in data.nodes:
        prob += z_v[n] >= xf[n]
        prob += z_v[n] >= xb[n]
        prob += z_v[n] <= xf[n] + xb[n]
        prob += z_cpu[n] >= (1 - xf[n])
        prob += z_cpu[n] >= (1 - xb[n])
        prob += z_cpu[n] <= (1 - xf[n]) + (1 - xb[n])

    # Activation memory belongs to whichever device runs BACKWARD (xb), full
    # stop -- that is what actually consumes it (gradient computation needs
    # the saved activation). Previously this used an AND(xf,xb) indicator
    # (w_v/w_v_else), which happens to agree with xb alone when fwd==bwd
    # (the common case) but charged NEITHER device for a cross-phase node
    # (fwd=CPU/bwd=GPU or fwd=GPU/bwd=CPU) -- silently telling the solver
    # cross-phase placement was memory-free. That is exactly what let it
    # produce the pathologically fragmented, still-OOMing plans seen for
    # resnet152 at large batch sizes (hybrid_executor.py's dual-placement
    # recompute does a full forward replay on the backward device first,
    # which very much needs that device's memory). See
    # src/ilp/topology_memory.py's module docstring for the full context.
    # xb itself is already a linear 0/1 decision variable, so no extra AND
    # indicator variable is needed at all.
    # A node whose joint strategy is recompute/checkpoint no longer keeps
    # its activation resident on GPU for the duration implied by xb[n]
    # alone -- recompute never materializes it outside the backward
    # recompute-then-consume window, and checkpoint moves it to CPU
    # instead. Both are plain ints (0) when the joint flag is off, so this
    # is algebraically identical to `xb[n]` alone in that case -- see
    # fix/07_gpu_recompute_joint_decision/README.md.
    # Calibrated-block nodes contribute their FULL retained cost here --
    # recompute's real effect for them is captured once, in aggregate, by
    # predicted_recompute_delta_mb below (added directly to total_gpu_act),
    # never by subtracting gpu_act_mem[n] per node too: that would double
    # count the same savings once per-node and once via the curve.
    gpu_act_coeff = {
        n: (xb[n] if n in node_to_block_idx else xb[n] - recompute[n] - checkpoint[n])
        for n in data.nodes
    }
    # checkpoint additionally spills that activation to CPU memory for the
    # gap between forward and backward (recompute does not: it never
    # stores the activation anywhere, it just redoes the forward compute).
    checkpoint_cpu_spill = pulp.lpSum(problem_data.cpu_act_mem[n] * checkpoint[n] for n in data.nodes) \
        if enable_joint_checkpoint else 0

    if cfg.memory_model == "topological":
        total_gpu_base = pulp.lpSum(problem_data.gpu_base_mem[n] * z_v[n] for n in data.nodes)
        total_cpu_base = pulp.lpSum(problem_data.cpu_base_mem[n] * z_cpu[n] for n in data.nodes)
        total_gpu_act = (
            pulp.lpSum(problem_data.gpu_act_mem[n] * gpu_act_coeff[n] for n in data.nodes)
            + predicted_recompute_delta_mb
        )
        total_cpu_act = pulp.lpSum(problem_data.cpu_act_mem[n] * (1 - xb[n]) for n in data.nodes)

        prob += total_gpu_base + total_gpu_act <= cfg.gpu_mem_budget_mb, "topo_gpu_mem_peak"
        prob += total_cpu_base + total_cpu_act + checkpoint_cpu_spill <= cfg.cpu_mem_budget_mb, "topo_cpu_mem_peak"
    elif cfg.memory_model == "topological_tight":
        # GPU: the real, per-critical-point constraint family (see
        # _build_tight_topological_gpu_constraints's own docstring).
        # CPU: unchanged from "topological" above -- CPU was never the
        # problem this memory_model exists to fix.
        _build_tight_topological_gpu_constraints(prob, data, xb, cfg.gpu_mem_budget_mb)
        total_cpu_base = pulp.lpSum(problem_data.cpu_base_mem[n] * z_cpu[n] for n in data.nodes)
        total_cpu_act = pulp.lpSum(problem_data.cpu_act_mem[n] * (1 - xb[n]) for n in data.nodes)
        prob += total_cpu_base + total_cpu_act + checkpoint_cpu_spill <= cfg.cpu_mem_budget_mb, "topo_tight_cpu_mem_peak"

    solver = _build_cbc_solver(pulp)
    solver.msg = True
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
    recompute_bits = (
        {n: int(round(pulp.value(recompute[n]) or 0.0)) for n in data.nodes}
        if enable_joint_recompute else {n: 0 for n in data.nodes}
    )
    checkpoint_bits = (
        {n: int(round(pulp.value(checkpoint[n]) or 0.0)) for n in data.nodes}
        if enable_joint_checkpoint else {n: 0 for n in data.nodes}
    )
    forward_assignment = {n: ("GPU" if fwd_bits[n] == 1 else "CPU") for n in data.nodes}
    backward_assignment = {n: ("GPU" if bwd_bits[n] == 1 else "CPU") for n in data.nodes}
    cut_edges_fwd = [e for e in data.edges if fwd_bits[e[0]] != fwd_bits[e[1]]]
    cut_edges_bwd = [e for e in data.edges if bwd_bits[e[0]] != bwd_bits[e[1]]]
    cross_edges = [(n, n) for n in data.nodes if fwd_bits[n] != bwd_bits[n]]

    def _gpu_act_contributes(n: str) -> bool:
        # Calibrated-block nodes (n in node_to_block_idx) always count
        # their full gpu_act_mem here when their backward is on GPU --
        # recompute's real effect for them is the REALIZED curve delta
        # added once below, not a per-node subtraction (see gpu_act_coeff
        # above for why, during solving, this is the same split).
        if n in node_to_block_idx:
            return bwd_bits[n] == 1
        return bwd_bits[n] == 1 and recompute_bits[n] == 0 and checkpoint_bits[n] == 0

    realized_recompute_delta_mb = (
        float(pulp.value(predicted_recompute_delta_mb) or 0.0) if use_calibrated_curve else 0.0
    )

    if cfg.memory_model == "topological":
        total_base_gpu = sum(problem_data.gpu_base_mem[n] for n in data.nodes if fwd_bits[n] == 1 or bwd_bits[n] == 1)
        total_base_cpu = sum(problem_data.cpu_base_mem[n] for n in data.nodes if fwd_bits[n] == 0 or bwd_bits[n] == 0)

        # Activation memory belongs to whichever device runs backward (see
        # the memory-constraint comment above for why this replaced the
        # AND(xf,xb) logic) -- except a node using recompute/checkpoint,
        # which no longer keeps that activation resident on GPU the way
        # "retain" does (see fix/07_gpu_recompute_joint_decision/README.md).
        gpu_act_mem = sum(problem_data.gpu_act_mem[n] for n in data.nodes if _gpu_act_contributes(n)) \
            + realized_recompute_delta_mb
        cpu_act_mem = sum(problem_data.cpu_act_mem[n] for n in data.nodes if bwd_bits[n] == 0)
        checkpoint_spill_mem = sum(problem_data.cpu_act_mem[n] for n in data.nodes if checkpoint_bits[n] == 1)

        gpu_mem = total_base_gpu + gpu_act_mem
        cpu_mem = total_base_cpu + cpu_act_mem + checkpoint_spill_mem
    elif cfg.memory_model == "topological_tight":
        # Report the REAL peak directly from the same trusted evaluator the
        # tight constraints were built to replicate, instead of re-deriving
        # it from the MILP's own variable values -- self-consistent with
        # what was actually optimized, and the most direct way to confirm
        # (see fix/09_.../README.md's verification step 1) that what the
        # solver believed matches what topology_memory.py measures for the
        # SAME assignment.
        gpu_mem, cpu_mem = data.prepared_topology.evaluate(forward_assignment, backward_assignment, None)
    else:
        gpu_mem = sum(problem_data.gpu_mem[n] for n in data.nodes if fwd_bits[n] == 1 or bwd_bits[n] == 1)
        cpu_mem = sum(problem_data.cpu_mem[n] for n in data.nodes if fwd_bits[n] == 0 or bwd_bits[n] == 0)

    activation_strategies = None
    if enable_joint_recompute or enable_joint_checkpoint:
        activation_strategies = {}
        for n in data.nodes:
            if recompute_bits[n] == 1:
                activation_strategies[n] = ActivationStrategy(n, recompute=True)
            elif checkpoint_bits[n] == 1:
                activation_strategies[n] = ActivationStrategy(n, checkpoint=True)
            else:
                activation_strategies[n] = ActivationStrategy(n, retain=True)

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
        activation_strategies=activation_strategies,
    )


def solve_partition_ilp(data: ILPInputData, cfg: ILPConfig, backend: str = "auto") -> ILPSolution:
    if backend == "auto":
        try:
            importlib.import_module("pulp")
            return _solve_with_pulp_dual(data, cfg)
        except Exception as e:
            print(f"PuLP failed with exception: {repr(e)}", flush=True)
            return _solve_exhaustive_dual(data, cfg)

    if backend in ("pulp", "pulp_cbc_dual"):
        return _solve_with_pulp_dual(data, cfg)
    if backend == "exhaustive":
        return _solve_exhaustive_dual(data, cfg)

    raise ValueError(f"Unsupported backend: {backend}")


def _assignment_to_bits(assignment: Dict[str, str], nodes: List[str]) -> Dict[str, int]:
    return {n: 1 if str(assignment.get(n, "CPU")).upper() == "GPU" else 0 for n in nodes}


def _bits_to_assignment(bits: Dict[str, int], nodes: List[str]) -> Dict[str, str]:
    return {n: ("GPU" if bits.get(n, 0) == 1 else "CPU") for n in nodes}


def refine_solution_hierarchical_local(
    data: ILPInputData,
    cfg: ILPConfig,
    base_solution: ILPSolution,
    max_assignment_changes: int,
) -> ILPSolution:
    """Hierarchical local search with explicit assignment-change budget.

    Stage order is backward then forward. Each accepted move flips one layer in
    one phase and consumes one unit of the change budget.
    """
    if max_assignment_changes <= 0:
        return base_solution

    if not base_solution.forward_assignment or not base_solution.backward_assignment:
        return base_solution

    problem = build_problem_data_dual(data, cfg)
    congestion_knee = _derive_congestion_knee_ms(data, cfg)
    fwd_bits = _assignment_to_bits(base_solution.forward_assignment, data.nodes)
    bwd_bits = _assignment_to_bits(base_solution.backward_assignment, data.nodes)

    current_eval = _eval_assignment_dual(
        fwd_bits,
        bwd_bits,
        data,
        cfg,
        problem,
        congestion_knee_ms=congestion_knee,
    )
    if current_eval is None:
        return base_solution

    current_obj, current_gpu, current_cpu, current_cut_fwd, current_cut_bwd, current_cross = current_eval
    changes_used = 0

    for stage in ("backward", "forward"):
        while changes_used < max_assignment_changes:
            best_candidate = None
            best_node = None

            for node in data.nodes:
                cand_fwd = dict(fwd_bits)
                cand_bwd = dict(bwd_bits)
                if stage == "backward":
                    cand_bwd[node] = 1 - cand_bwd[node]
                else:
                    cand_fwd[node] = 1 - cand_fwd[node]

                cand_eval = _eval_assignment_dual(
                    cand_fwd,
                    cand_bwd,
                    data,
                    cfg,
                    problem,
                    congestion_knee_ms=congestion_knee,
                )
                if cand_eval is None:
                    continue
                cand_obj = float(cand_eval[0])
                if cand_obj + 1e-9 < current_obj:
                    if best_candidate is None or cand_obj < float(best_candidate[0]):
                        best_candidate = cand_eval
                        best_node = node

            if best_candidate is None or best_node is None:
                break

            if stage == "backward":
                bwd_bits[best_node] = 1 - bwd_bits[best_node]
            else:
                fwd_bits[best_node] = 1 - fwd_bits[best_node]

            current_obj = float(best_candidate[0])
            current_gpu = float(best_candidate[1])
            current_cpu = float(best_candidate[2])
            current_cut_fwd = list(best_candidate[3])
            current_cut_bwd = list(best_candidate[4])
            current_cross = list(best_candidate[5])
            changes_used += 1

    forward_assignment = _bits_to_assignment(fwd_bits, data.nodes)
    backward_assignment = _bits_to_assignment(bwd_bits, data.nodes)

    return ILPSolution(
        status=base_solution.status,
        backend=f"{base_solution.backend}_hier_local",
        objective_value=float(current_obj),
        assignment=dict(forward_assignment),
        gpu_mem_used_mb=float(current_gpu),
        cpu_mem_used_mb=float(current_cpu),
        cut_edges=list(current_cut_fwd),
        forward_assignment=forward_assignment,
        backward_assignment=backward_assignment,
        backward_cut_edges=list(current_cut_bwd),
        cross_phase_edges=list(current_cross),
        activation_strategies=getattr(base_solution, "activation_strategies", None),
    )


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
    
    # If GPU memory utilization exceeds cfg.target_gpu_utilization (default
    # 0.60, matching this heuristic's original hardcoded threshold),
    # consider recompute/checkpoint. Prefer larger memory savings for
    # smaller time penalties.
    if gpu_mem_used > cfg.target_gpu_utilization * gpu_budget:
        # For a decoder-only (torch.export) graph, a strategy only ever
        # yields real runtime savings when applied to a WHOLE repeated
        # block's leaves together (or to the model's single final layer,
        # lm_head -- a genuine size-1 block that gets its own real saving via
        # hybrid_executor.py's dedicated final-node + chunked-loss fusion,
        # which has no vision-graph equivalent) -- never to an arbitrary
        # subset. See _group_nodes_into_activation_candidates.
        if data.graph_trace_source == "torch_export_decoder_only":
            candidate_groups = _group_nodes_into_activation_candidates(data.nodes, forward_assignment)
        else:
            # Was: candidate_groups = [[n] for n in gpu_nodes] -- every GPU
            # node its own isolated candidate. Confirmed (fix/08 README):
            # checkpointing an isolated nn.Module in a vision graph does not
            # save real memory -- it makes it WORSE (+36% measured in
            # hybrid_executor.py::_detect_checkpoint_groups_call_module's own
            # controlled test), because its only expensive tensor is its own
            # output, which checkpoint does not free; that function only
            # delivers a real saving when it groups >=2 leaves of the SAME
            # structural block (_block_id_for_layer_name). Reuse the same
            # block-grouping the decoder-only branch already uses (a vision
            # layer name like "layer2.0.conv1" groups into block "layer2.0",
            # i.e. exactly one resnet Bottleneck), but -- unlike the
            # decoder-only branch -- drop any group that is still size 1:
            # a vision graph has no final-node/loss-fusion mechanism to save
            # a standalone leaf (not even "fc"; see
            # _detect_checkpoint_groups_call_module's docstring), so an
            # isolated stem/head layer must stay "retain".
            candidate_groups = [
                group
                for group in _group_nodes_into_activation_candidates(data.nodes, forward_assignment)
                if len(group) > 1
            ]

        candidates: List[Tuple[float, Tuple[str, ...], str]] = []
        for group in candidate_groups:
            act_mem = sum(float(problem4.activation_meta.node_mem_activation_mb.get(n, 0.0)) for n in group)
            if act_mem <= 0.0:
                continue

            if cfg.enable_recompute:
                recomp_cost = sum(float(problem4.recompute_cost_gpu.get(n, 0.0)) for n in group)
                score = act_mem / (recomp_cost + 1e-6)
                candidates.append((score, tuple(group), "recompute"))

            if cfg.enable_checkpoint:
                checkpoint_cost = sum(float(problem4.checkpoint_cost_gpu.get(n, 0.0)) for n in group)
                score = act_mem / (checkpoint_cost + 1e-6)
                candidates.append((score, tuple(group), "checkpoint"))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: set[str] = set()
        target_gpu_mem = cfg.target_gpu_utilization * gpu_budget  # reduce to just below the trigger threshold

        for _, group, strategy_name in candidates:
            if any(n in selected for n in group):
                continue
            if gpu_mem_used <= target_gpu_mem:
                break
            # Before applying a checkpoint strategy, verify it won't violate the
            # CPU memory budget.  Checkpointing moves activation memory from GPU
            # to CPU; if the CPU budget is already tight, skip this candidate.
            if strategy_name == "checkpoint":
                group_act_mem_on_gpu = sum(
                    float(problem4.activation_meta.node_mem_activation_mb.get(n, 0.0))
                    for n in group
                    if forward_assignment.get(n, "CPU") == "GPU"
                )
                if cpu_mem_used + group_act_mem_on_gpu > cfg.cpu_mem_budget_mb:
                    continue
            for node in group:
                _apply_strategy(node, strategy_name)
            selected.update(group)
    
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


def _solve_phase4_exhaustive(data: ILPInputData, cfg: ILPConfig4) -> ILPSolution4:
    """Phase 4 exhaustive strategy search on top of dual placement baseline."""
    baseline_solution = solve_partition_ilp(data, cfg, backend="auto")

    if baseline_solution.status != "optimal":
        return ILPSolution4(
            status=baseline_solution.status,
            backend="exhaustive_phase4",
            objective_value=baseline_solution.objective_value,
            assignment=baseline_solution.assignment,
            gpu_mem_used_mb=baseline_solution.gpu_mem_used_mb,
            cpu_mem_used_mb=baseline_solution.cpu_mem_used_mb,
            cut_edges=baseline_solution.cut_edges,
            forward_assignment=baseline_solution.forward_assignment,
            backward_assignment=baseline_solution.backward_assignment,
            backward_cut_edges=baseline_solution.backward_cut_edges,
            cross_phase_edges=baseline_solution.cross_phase_edges,
            activation_strategies={n: ActivationStrategy(n, retain=True) for n in data.nodes},
            mode="phase3",
        )

    if len(data.nodes) > 18:
        raise RuntimeError(
            f"Exhaustive Phase 4 backend is limited to <=18 nodes, got {len(data.nodes)}. "
            "Use backend='greedy' for larger graphs."
        )

    problem4 = build_problem_data_phase4(data, cfg)
    forward_assignment = baseline_solution.forward_assignment or baseline_solution.assignment
    backward_assignment = baseline_solution.backward_assignment or baseline_solution.assignment

    options_by_node: Dict[str, List[str]] = {n: ["retain"] for n in data.nodes}
    for n in data.nodes:
        if cfg.enable_recompute:
            options_by_node[n].append("recompute")
        if cfg.enable_checkpoint:
            options_by_node[n].append("checkpoint")

    search_space = 1
    for node in data.nodes:
        search_space *= len(options_by_node[node])
    if search_space > 2_000_000:
        raise RuntimeError(
            "Exhaustive Phase 4 search space is too large "
            f"({search_space} combinations). Use backend='greedy' or backend='auto'."
        )

    best_obj = None
    best_gpu_mem = 0.0
    best_cpu_mem = 0.0
    best_strategies: Dict[str, ActivationStrategy] = {n: ActivationStrategy(n, retain=True) for n in data.nodes}

    baseline_gpu_mem = float(baseline_solution.gpu_mem_used_mb)
    baseline_cpu_mem = float(baseline_solution.cpu_mem_used_mb)
    baseline_obj = float(baseline_solution.objective_value)

    node_order = list(data.nodes)
    for combo in product(*[options_by_node[n] for n in node_order]):
        gpu_mem = baseline_gpu_mem
        cpu_mem = baseline_cpu_mem
        obj = baseline_obj

        strategies: Dict[str, ActivationStrategy] = {}
        feasible = True

        for idx, node in enumerate(node_order):
            strategy_name = combo[idx]
            on_gpu = forward_assignment.get(node, "CPU") == "GPU"
            act_mem = float(problem4.activation_meta.node_mem_activation_mb.get(node, 0.0))

            if strategy_name == "retain":
                strategies[node] = ActivationStrategy(node, retain=True)
                continue

            if strategy_name == "recompute":
                strategies[node] = ActivationStrategy(node, recompute=True)
                delta_time = float((problem4.recompute_cost_gpu if on_gpu else problem4.recompute_cost_cpu).get(node, 0.0))
                obj += (cfg.w_time * delta_time) + (cfg.w_io * delta_time)
                if on_gpu:
                    gpu_mem = max(0.0, gpu_mem - act_mem)

            elif strategy_name == "checkpoint":
                strategies[node] = ActivationStrategy(node, checkpoint=True)
                delta_time = float((problem4.checkpoint_cost_gpu if on_gpu else problem4.checkpoint_cost_cpu).get(node, 0.0))
                delta_energy = float(problem4.activation_meta.node_energy_io_j.get(node, 0.0))
                obj += (cfg.w_time * delta_time) + (cfg.w_energy * delta_energy) + (cfg.w_io * delta_time)
                if on_gpu:
                    gpu_mem = max(0.0, gpu_mem - act_mem)
                    cpu_mem += act_mem

            if gpu_mem > cfg.gpu_mem_budget_mb or cpu_mem > cfg.cpu_mem_budget_mb:
                feasible = False
                break

        if not feasible:
            continue

        if best_obj is None or obj < best_obj:
            best_obj = float(obj)
            best_gpu_mem = float(gpu_mem)
            best_cpu_mem = float(cpu_mem)
            best_strategies = strategies

    if best_obj is None:
        return ILPSolution4(
            status="infeasible",
            backend="exhaustive_phase4",
            objective_value=float("inf"),
            assignment=dict(forward_assignment),
            gpu_mem_used_mb=float(baseline_gpu_mem),
            cpu_mem_used_mb=float(baseline_cpu_mem),
            cut_edges=list(baseline_solution.cut_edges),
            forward_assignment=dict(forward_assignment),
            backward_assignment=dict(backward_assignment),
            backward_cut_edges=list(baseline_solution.backward_cut_edges or []),
            cross_phase_edges=list(baseline_solution.cross_phase_edges or []),
            activation_strategies={n: ActivationStrategy(n, retain=True) for n in data.nodes},
            mode="phase4",
        )

    return ILPSolution4(
        status="optimal",
        backend="exhaustive_phase4",
        objective_value=float(best_obj),
        assignment=dict(forward_assignment),
        gpu_mem_used_mb=float(best_gpu_mem),
        cpu_mem_used_mb=float(best_cpu_mem),
        cut_edges=list(baseline_solution.cut_edges),
        forward_assignment=dict(forward_assignment),
        backward_assignment=dict(backward_assignment),
        backward_cut_edges=list(baseline_solution.backward_cut_edges or []),
        cross_phase_edges=list(baseline_solution.cross_phase_edges or []),
        activation_strategies=best_strategies,
        mode="phase4",
    )


def _solve_phase4_joint(data: ILPInputData, cfg: ILPConfig4) -> ILPSolution4:
    """Phase 4 via a single joint MILP solve: device placement (CPU/GPU,
    forward/backward) and activation strategy (retain/recompute/checkpoint)
    are decided TOGETHER by the real solver (_solve_with_pulp_dual), instead
    of _solve_phase4_greedy/_solve_phase4_exhaustive's approach of fixing
    the device assignment first and only adjusting activation strategy on
    nodes the baseline already placed on GPU. This lets a node that would
    otherwise be forced entirely onto CPU by GPU memory pressure stay on
    GPU via recompute/checkpoint instead -- see
    fix/07_gpu_recompute_joint_decision/README.md for the measured
    motivation (CPU can be ~100x slower than GPU per layer at large batch
    sizes) and the full design. Does not modify _solve_phase4_greedy,
    _solve_phase4_exhaustive, or _solve_with_pulp_dual's behavior for any
    caller that does not set enable_joint_activation_strategy=True."""
    joint_cfg = replace(cfg, enable_joint_activation_strategy=True)
    sol = solve_partition_ilp(data, joint_cfg, backend="auto")

    fallback_strategies = {n: ActivationStrategy(n, retain=True) for n in data.nodes}
    return ILPSolution4(
        status=sol.status,
        backend="joint_phase4",
        objective_value=sol.objective_value,
        assignment=sol.assignment,
        gpu_mem_used_mb=sol.gpu_mem_used_mb,
        cpu_mem_used_mb=sol.cpu_mem_used_mb,
        cut_edges=sol.cut_edges,
        forward_assignment=sol.forward_assignment,
        backward_assignment=sol.backward_assignment,
        backward_cut_edges=sol.backward_cut_edges,
        cross_phase_edges=sol.cross_phase_edges,
        activation_strategies=sol.activation_strategies or fallback_strategies,
        mode="phase4" if sol.status in ("optimal", "feasible") else "phase3",
    )


def solve_partition_ilp_phase4(
    data: ILPInputData,
    cfg: ILPConfig4,
    backend: str = "auto"
) -> ILPSolution4:
    """
    Solve Phase 4 problem with activation persistence strategies.

    Args:
      data: ILP input data (extended with activation metadata)
      cfg: ILPConfig4 with Phase 4 parameters
    backend: "auto", "greedy", "exhaustive", or "joint"

    Returns:
      ILPSolution4 with activation strategies per node
    """
    if backend == "auto":
        options_per_node = 1 + int(bool(cfg.enable_recompute)) + int(bool(cfg.enable_checkpoint))
        search_space = options_per_node ** len(data.nodes)
        # Keep auto backend conservative for campaign stability.
        if len(data.nodes) <= 14 and search_space <= 250_000:
            return _solve_phase4_exhaustive(data, cfg)
        return _solve_phase4_greedy(data, cfg)

    if backend == "greedy":
        return _solve_phase4_greedy(data, cfg)

    if backend == "exhaustive":
        return _solve_phase4_exhaustive(data, cfg)

    if backend == "joint":
        return _solve_phase4_joint(data, cfg)

    raise ValueError(f"Unsupported Phase 4 backend: {backend}")

