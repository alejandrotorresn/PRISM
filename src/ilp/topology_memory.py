"""Real, call-site-aware topological peak-memory simulator for a candidate ILP plan.

Root-cause context (see fix/06 README and PLAN_ACCION_JURADO.md item 42 for the
full investigation): the ILP's own memory formulas (nodal_sum / peak_approx /
the misleadingly-named "topological" MILP constraint in model_builder.py) are
all aggregate *sums* over per-layer coefficients. None of them walk the real
execution order, so none of them can express:
  - that a layer's *transient* (workspace) memory does not accumulate across
    layers the way its *retained* activation does (only the running max
    matters, since GPU execution is sequential on one stream);
  - that forward and backward have genuinely different transient costs
    (backward's own transient was never even measured before this session's
    fix to training_profiler.py -- gpu_bwd_mem_peak_mb);
  - that a "checkpoint"-strategy activation is NOT resident across the
    forward->backward gap (it is offloaded/discarded, not retained);
  - that a cross-phase node (forward on one device, backward on the other)
    pays a real, transient recompute cost on whichever device runs its
    backward.

This module walks the REAL graph, in real execution order, exactly once per
assignment candidate, to compute the figure that actually matters: the peak
concurrent memory on each device. It is intentionally architecture-agnostic
(no CNN- or LLM-specific logic) -- it only consumes measured per-layer stats,
real graph topology, and the candidate assignment, which is what lets it
generalize from the vision models it was validated against (resnet50: -0.4%
error; resnet152/batch_64: +5.5% error, vs the ILP's own -19.7% / -5.3%) to
attention/LLM layers without special-casing either family.

Known remaining limitation (documented honestly, not hidden): for very
heavily fragmented plans (many forward/backward cut edges relative to total
layer count -- empirically, resnet152/batch_96 with ~400 cut/cross-phase
events over 364 layers), real GPU usage can still exceed this simulator's
prediction, because PyTorch's async prefetch (hybrid_executor.py,
_enable_prefetch) can keep more than one cross-device duplicate buffer alive
at once -- a pipelining effect this sequential, one-thing-at-a-time walk does
not model. Wiring this simulator into the ILP's own feasibility check (see
solve.py's post-solve repair loop) is expected to make the solver stop
*producing* such pathologically fragmented plans in the first place, since it
will no longer see cross-phase placement as memory-free.

Design: loading+indexing the graph and stats (I/O-bound) is separated from
evaluating a candidate assignment (pure Python, assignment-independent data
reused) -- PreparedTopologyMemory.prepare() does the former once;
.evaluate() is cheap enough to call repeatedly inside a repair/local-search
loop that tries many candidate assignments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _nkey(node_id: str) -> Any:
    suffix = node_id[1:] if node_id and node_id[0] == "n" else node_id
    return int(suffix) if suffix.isdigit() else suffix


def _topological_order(relevant_ids: set, edges: List[Tuple[str, str]]) -> Tuple[List[str], set]:
    """Kahn's algorithm, breaking ties by node_id (real trace/execution order).

    node_id values are "n<k>" assigned in forward-trace order by
    graph_extractor.py -- using them as the tie-break (instead of arbitrary
    Python set-iteration order) was confirmed, empirically, to matter: it
    took resnet50's error from +7.0% to -0.4% on its own, with no other
    change. producer_name/consumer_name (layer names) are NOT used for
    topology here because they collapse reused modules (e.g. a Bottleneck's
    shared self.relu, called 3x) to one label, breaking the DAG.

    Decoder-only (torch.export) graphs can contain a genuine cycle, not just
    collapsed-name ambiguity: a weight-tied pair (e.g. GPT-2's lm_head and
    transformer.wte sharing one tensor) gets a spurious producer->consumer
    edge from graph_extractor.py treating the shared parameter as a data
    dependency, closing a cycle around the entire network (confirmed:
    distilgpt2's graph_edges.csv has a literal lm_head->transformer.wte
    edge). Plain Kahn's algorithm would then stall with most of the graph
    still unprocessed, which silently produced a near-zero peak (58/59
    nodes dropped) rather than failing loudly -- far worse than a slightly
    wrong order. When no zero-indegree node remains but unprocessed nodes
    do, force progress by taking the unprocessed node with the fewest
    remaining unsatisfied predecessors (then smallest node_id) -- this
    breaks exactly the spurious back-edge's cycle without needing to know
    which specific edge is spurious.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {nid: 0 for nid in relevant_ids}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    ready = sorted([n for n in relevant_ids if indeg[n] == 0], key=_nkey)
    topo: List[str] = []
    indeg_work = dict(indeg)
    remaining = set(relevant_ids)
    while remaining:
        if not ready:
            # Cycle: no node has indegree 0. Force the closest-to-ready
            # remaining node (fewest unsatisfied predecessors) to break it.
            forced = min(remaining, key=lambda n: (indeg_work[n], _nkey(n)))
            ready.append(forced)
        ready.sort(key=_nkey)
        n = ready.pop(0)
        if n not in remaining:
            continue
        remaining.discard(n)
        topo.append(n)
        for v in adj[n]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0 and v in remaining:
                ready.append(v)

    missing = relevant_ids - set(topo)
    return topo, missing


def _safe_float(row: Any, col: str, default: float = 0.0) -> float:
    try:
        val = float(row.get(col, default))
        if val != val:  # NaN
            return default
        return val
    except (TypeError, ValueError):
        return default


@dataclass
class PreparedTopologyMemory:
    """Graph + stats indexed once; .evaluate() tries a candidate assignment cheaply."""

    topo: List[str]
    node_layer: Dict[str, str]
    retained: Dict[str, float]
    transient_fwd: Dict[str, float]
    transient_bwd: Dict[str, float]
    params: Dict[str, float]
    optimizer_states: Dict[str, float]
    # layer name -> weight-sharing group id (see training_profiler.py's
    # _detect_shared_weight_groups). A tied pair (e.g. GPT-2's lm_head/wte
    # sharing one tensor) maps to the SAME group; an untied layer maps to
    # its own name as a singleton group.
    weight_group: Dict[str, str]
    # group id -> that group's real, deduplicated static memory
    # (params_mb + optimizer_states_mb), counted ONCE regardless of how many
    # layer names/call-sites reference the shared tensor.
    group_static_mb: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def prepare(graph_edges_csv: str | Path, metrics_stats_csv: str | Path) -> "PreparedTopologyMemory":
        graph_path = Path(graph_edges_csv)
        stats_path = Path(metrics_stats_csv)
        if not graph_path.exists():
            raise FileNotFoundError(f"graph_edges csv not found: {graph_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"metrics_stats csv not found: {stats_path}")

        stats = pd.read_csv(stats_path)
        stats_map: Dict[str, Any] = {row["layer"]: row for _, row in stats.iterrows()}

        edges_df = pd.read_csv(graph_path)
        node_layer: Dict[str, str] = {}
        for _, r in edges_df.iterrows():
            node_layer[str(r["src_id"])] = str(r["producer_name"])
            node_layer[str(r["dst_id"])] = str(r["consumer_name"])

        measured_layers = set(stats_map.keys())
        relevant_ids = {nid for nid, name in node_layer.items() if name in measured_layers}

        edges: List[Tuple[str, str]] = [
            (str(r["src_id"]), str(r["dst_id"]))
            for _, r in edges_df.iterrows()
            if str(r["src_id"]) in relevant_ids and str(r["dst_id"]) in relevant_ids
        ]

        topo, missing = _topological_order(relevant_ids, edges)

        retained: Dict[str, float] = {}
        transient_fwd: Dict[str, float] = {}
        transient_bwd: Dict[str, float] = {}
        params: Dict[str, float] = {}
        optimizer_states: Dict[str, float] = {}
        for nid in relevant_ids:
            name = node_layer[nid]
            row = stats_map.get(name)
            if row is None:
                retained[nid] = transient_fwd[nid] = transient_bwd[nid] = params[nid] = optimizer_states[nid] = 0.0
                continue
            act = _safe_float(row, "activations_mb_mean")
            fwd_peak = _safe_float(row, "gpu_mem_peak_mb_mean")
            bwd_peak = _safe_float(row, "gpu_bwd_mem_peak_mb_mean")
            retained[nid] = act
            transient_fwd[nid] = max(0.0, fwd_peak - act)
            transient_bwd[nid] = max(0.0, bwd_peak - act)
            params[nid] = _safe_float(row, "params_mb_mean")
            # Optimizer state (e.g. AdamW's two fp32 moment buffers, ~2x
            # params_mb; 0 for plain SGD -- see OPTIMIZER_OVERHEAD_MAP in
            # core/constants.py) is allocated once, statically, same as
            # params -- never during backward like grads_mb. Only shows up
            # with a non-SGD optimizer: the vision campaigns validated so
            # far all used plain SGD (factor 0), which is why this gap went
            # unnoticed until a model profiled with AdamW (distilgpt2,
            # gpt2_small) was checked.
            optimizer_states[nid] = _safe_float(row, "optimizer_states_mb_mean")

        # Weight-sharing groups, keyed by LAYER NAME (a module-level
        # property, not per call-site): fall back to the layer's own name
        # as a singleton group when the column is absent (older
        # metrics_stats.csv predating this field) or blank for this row.
        weight_group: Dict[str, str] = {}
        group_static_mb: Dict[str, float] = {}
        for name, row in stats_map.items():
            group = str(row.get("shared_weight_group", "") or "").strip() or name
            weight_group[name] = group
            if group not in group_static_mb:
                group_static_mb[group] = _safe_float(row, "params_mb_mean") + _safe_float(
                    row, "optimizer_states_mb_mean"
                )

        return PreparedTopologyMemory(
            topo=topo,
            node_layer=node_layer,
            retained=retained,
            transient_fwd=transient_fwd,
            transient_bwd=transient_bwd,
            params=params,
            optimizer_states=optimizer_states,
            weight_group=weight_group,
            group_static_mb=group_static_mb,
            diagnostics={
                "nodes_total": len(relevant_ids),
                "nodes_missing_from_topo_order": len(missing),
                "layers_total": len(measured_layers),
            },
        )

    def evaluate(
        self,
        forward_assignment: Dict[str, str],
        backward_assignment: Dict[str, str],
        activation_strategy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Return (peak_gpu_mb, peak_cpu_mb) for this candidate assignment.

        forward_assignment / backward_assignment: {layer_name: "GPU"|"CPU"},
        keyed exactly like ILPSolution.forward_assignment/backward_assignment
        -- i.e. at the ILP's own per-layer granularity, not per call-site.
        Multiple real call-sites of a reused layer (e.g. a shared ReLU)
        therefore all receive the SAME device decision here, which matches
        what the runtime actually does (one nn.Module placement decision
        governs every call to it) -- only the topological *order* needed
        call-site granularity, already resolved once in .prepare().
        """
        node_layer = self.node_layer
        retained = self.retained
        transient_fwd = self.transient_fwd
        transient_bwd = self.transient_bwd
        params = self.params
        optimizer_states = self.optimizer_states

        def strategy_of(nid: str) -> str:
            if activation_strategy is None:
                return "retain"
            entry = activation_strategy.get(node_layer[nid])
            if entry is None:
                return "retain"
            if getattr(entry, "checkpoint", False):
                return "checkpoint"
            if getattr(entry, "recompute", False):
                return "recompute"
            return "retain"

        gpu_resident = 0.0
        cpu_resident = 0.0
        peak_gpu = 0.0
        peak_cpu = 0.0

        for nid in self.topo:
            name = node_layer[nid]
            fwd = forward_assignment.get(name)
            if fwd is None:
                continue
            strategy = strategy_of(nid)
            if fwd == "GPU":
                peak_gpu = max(peak_gpu, gpu_resident + retained[nid] + transient_fwd[nid])
                if strategy == "retain":
                    gpu_resident += retained[nid]
            elif fwd == "CPU":
                peak_cpu = max(peak_cpu, cpu_resident + retained[nid] + transient_fwd[nid])
                if strategy == "retain":
                    cpu_resident += retained[nid]

        # Deduplicate by weight-sharing group before summing: a tied pair
        # (e.g. GPT-2's lm_head/wte) must only pay for its real, single
        # underlying tensor once per device, no matter how many layer
        # names/call-sites reference it. Confirmed bug otherwise (see
        # weight_group's docstring): ~442MB double-counted for distilgpt2.
        gpu_groups = {
            self.weight_group.get(node_layer[nid], node_layer[nid])
            for nid in self.topo
            if forward_assignment.get(node_layer[nid]) == "GPU"
            or backward_assignment.get(node_layer[nid]) == "GPU"
        }
        cpu_groups = {
            self.weight_group.get(node_layer[nid], node_layer[nid])
            for nid in self.topo
            if forward_assignment.get(node_layer[nid]) == "CPU"
            or backward_assignment.get(node_layer[nid]) == "CPU"
        }
        static_gpu = sum(self.group_static_mb.get(g, 0.0) for g in gpu_groups)
        static_cpu = sum(self.group_static_mb.get(g, 0.0) for g in cpu_groups)
        peak_gpu += static_gpu
        peak_cpu += static_cpu
        gpu_resident += static_gpu
        cpu_resident += static_cpu

        # Same weight-sharing dedup as static_gpu/static_cpu above, but for the
        # gradient buffer: autograd accumulates a tied weight's gradient into
        # ONE physical buffer regardless of how many layer names/node_ids
        # reference it (e.g. GPT-2's lm_head/wte), so its params_mb-sized
        # contribution to resident memory must only be charged once across
        # the whole backward walk, not once per node_id that shares it.
        grad_charged_groups: set = set()

        for nid in reversed(self.topo):
            name = node_layer[nid]
            fwd = forward_assignment.get(name)
            bwd = backward_assignment.get(name)
            if fwd is None or bwd is None:
                continue
            strategy = strategy_of(nid)
            needs_recompute = strategy in ("checkpoint", "recompute") or fwd != bwd
            was_retained_here = strategy == "retain" and fwd == bwd
            group = self.weight_group.get(name, name)
            if bwd == "GPU":
                if needs_recompute:
                    peak_gpu = max(peak_gpu, gpu_resident + retained[nid] + transient_fwd[nid])
                    peak_gpu = max(peak_gpu, gpu_resident + retained[nid] + transient_bwd[nid])
                else:
                    peak_gpu = max(peak_gpu, gpu_resident + transient_bwd[nid])
                if group not in grad_charged_groups:
                    gpu_resident += params[nid]  # grads_mb == params_mb
                    grad_charged_groups.add(group)
                if was_retained_here:
                    gpu_resident -= retained[nid]
            elif bwd == "CPU":
                if needs_recompute:
                    peak_cpu = max(
                        peak_cpu, cpu_resident + retained[nid] + max(transient_fwd[nid], transient_bwd[nid])
                    )
                else:
                    peak_cpu = max(peak_cpu, cpu_resident + transient_bwd[nid])
                if group not in grad_charged_groups:
                    cpu_resident += params[nid]
                    grad_charged_groups.add(group)
                if was_retained_here:
                    cpu_resident -= retained[nid]

        return float(peak_gpu), float(peak_cpu)

    def layer_gpu_weight(self, layer_name: str) -> float:
        """Rough per-layer GPU footprint (retained + forward transient, summed
        across every real call-site of this layer) -- used only to RANK which
        layer to move to CPU first during repair, not as a memory formula."""
        total = 0.0
        for nid in self.topo:
            if self.node_layer[nid] == layer_name:
                total += self.retained[nid] + self.transient_fwd[nid]
        return total


def correct_solution_memory(
    prepared: PreparedTopologyMemory,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Optional[Dict[str, Any]],
    gpu_budget_mb: float,
    cpu_budget_mb: float,
    repair_budget: int = 200,
) -> Dict[str, Any]:
    """Replace a solver's memory estimate with the real one, and greedily
    repair (move layers GPU->CPU, both phases) if the solver's plan does not
    actually fit under gpu_budget_mb per the real walk.

    This is intentionally a *repair*, not a re-optimization: the solver
    already chose assignments to minimize its (possibly wrong) objective;
    this only intervenes when its own memory estimate was too optimistic,
    moving the largest GPU-resident layers off GPU first (cheapest way to
    free the most memory per flip) until the real simulator says it fits,
    or the repair budget / available GPU layers run out.
    """
    fwd = dict(forward_assignment)
    bwd = dict(backward_assignment)
    peak_gpu, peak_cpu = prepared.evaluate(fwd, bwd, activation_strategy)

    flips_applied = 0
    if peak_gpu > gpu_budget_mb:
        gpu_layers = sorted(
            {name for name, dev in fwd.items() if dev == "GPU"} | {name for name, dev in bwd.items() if dev == "GPU"},
            key=prepared.layer_gpu_weight,
            reverse=True,
        )
        for layer_name in gpu_layers:
            if flips_applied >= repair_budget:
                break
            if peak_gpu <= gpu_budget_mb:
                break
            fwd[layer_name] = "CPU"
            bwd[layer_name] = "CPU"
            peak_gpu, peak_cpu = prepared.evaluate(fwd, bwd, activation_strategy)
            flips_applied += 1

    return {
        "forward_assignment": fwd,
        "backward_assignment": bwd,
        "peak_gpu_mb": peak_gpu,
        "peak_cpu_mb": peak_cpu,
        "flips_applied": flips_applied,
        "feasible": peak_gpu <= gpu_budget_mb and peak_cpu <= cpu_budget_mb,
    }


def expand_solution_memory(
    prepared: PreparedTopologyMemory,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Optional[Dict[str, Any]],
    node_cost_cpu_fwd_ms: Dict[str, float],
    node_cost_cpu_bwd_ms: Dict[str, float],
    node_cost_gpu_fwd_ms: Dict[str, float],
    node_cost_gpu_bwd_ms: Dict[str, float],
    gpu_budget_mb: float,
    cpu_budget_mb: float,
    expand_budget: int = 200,
) -> Dict[str, Any]:
    """Symmetric to correct_solution_memory, but greedy in the CPU->GPU
    direction instead of GPU->CPU: the MILP's own memory constraint
    (src/ilp/solve.py, Σ gpu_act_mem[n]*xb[n] <= budget) treats every
    GPU-resident node's activation as simultaneously live, which is a far
    more conservative over-approximation than the real sequential peak
    this module already computes correctly -- confirmed empirically on
    resnet50: the solver believed it was at 100% of budget while the real
    peak was only 41-54%, leaving thousands of real MB of GPU headroom
    unused and most of the network stuck on CPU for no real memory reason.
    This greedily reclaims that margin post-solve, using the SAME trusted
    evaluator (never a new estimate), rather than changing the MILP's own
    constraint (which would stop being a simple linear sum and risk
    solve times blowing up -- the same tractability trade-off already
    made for the recompute-block prefix restriction).

    Candidates: layers with BOTH phases on CPU today (no new cross-phase
    placement is introduced here -- keeps this pass simple and orthogonal
    to the existing recompute/checkpoint machinery, which still operates
    on top of whatever this pass decides). Ordered by real time saved by
    moving to GPU, descending (same ranking principle as
    layer_gpu_weight, for time instead of memory, in the opposite
    direction). Each candidate is tentatively moved, re-validated with
    prepared.evaluate() (the same real walk `correct_solution_memory`
    already trusts), and kept only if both budgets still hold -- otherwise
    reverted and the next (cheaper-ranked) candidate is still tried.
    """
    fwd = dict(forward_assignment)
    bwd = dict(backward_assignment)

    cpu_resident_layers = [
        name for name in fwd
        if fwd.get(name) == "CPU" and bwd.get(name) == "CPU"
    ]

    def _time_saved(name: str) -> float:
        cpu_time = node_cost_cpu_fwd_ms.get(name, 0.0) + node_cost_cpu_bwd_ms.get(name, 0.0)
        gpu_time = node_cost_gpu_fwd_ms.get(name, 0.0) + node_cost_gpu_bwd_ms.get(name, 0.0)
        return cpu_time - gpu_time

    cpu_resident_layers.sort(key=_time_saved, reverse=True)

    peak_gpu, peak_cpu = prepared.evaluate(fwd, bwd, activation_strategy)
    flips_applied = 0
    flip_order: List[str] = []
    for layer_name in cpu_resident_layers:
        if flips_applied >= expand_budget:
            break
        fwd[layer_name] = "GPU"
        bwd[layer_name] = "GPU"
        trial_peak_gpu, trial_peak_cpu = prepared.evaluate(fwd, bwd, activation_strategy)
        if trial_peak_gpu <= gpu_budget_mb and trial_peak_cpu <= cpu_budget_mb:
            peak_gpu, peak_cpu = trial_peak_gpu, trial_peak_cpu
            flips_applied += 1
            flip_order.append(layer_name)
        else:
            fwd[layer_name] = "CPU"
            bwd[layer_name] = "CPU"

    return {
        "forward_assignment": fwd,
        "backward_assignment": bwd,
        "peak_gpu_mb": peak_gpu,
        "peak_cpu_mb": peak_cpu,
        "flips_applied": flips_applied,
        # Layers in the exact order they were successfully moved CPU->GPU --
        # lets a caller re-validate against something topology_memory never
        # models (e.g. the hybrid engine's real transient overhead) and trim
        # the expansion to a smaller prefix if needed, without rerunning the
        # greedy ranking (see run_ilp_partition.py's
        # _validate_gpu_expansion_against_real_overhead).
        "flip_order": flip_order,
        "feasible": peak_gpu <= gpu_budget_mb and peak_cpu <= cpu_budget_mb,
    }


def rebalance_solution_memory(
    prepared: PreparedTopologyMemory,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Optional[Dict[str, Any]],
    node_cost_cpu_fwd_ms: Dict[str, float],
    node_cost_cpu_bwd_ms: Dict[str, float],
    node_cost_gpu_fwd_ms: Dict[str, float],
    node_cost_gpu_bwd_ms: Dict[str, float],
    gpu_budget_mb: float,
    cpu_budget_mb: float,
    expand_budget: int = 200,
    repair_budget: int = 200,
) -> Dict[str, Any]:
    """expand_solution_memory, then correct_solution_memory as a final
    safety net (unmodified, reused as-is) -- always in that order, so this
    is safe regardless of the solution's initial state: if there is no
    real margin, expansion commits nothing (every tentative move fails
    validation and reverts) and the repair step then also has nothing to
    do (already within budget); if expansion overshoots for any reason,
    the repair step still catches it exactly like it does for any other
    caller of correct_solution_memory today.
    """
    expanded = expand_solution_memory(
        prepared=prepared,
        forward_assignment=forward_assignment,
        backward_assignment=backward_assignment,
        activation_strategy=activation_strategy,
        node_cost_cpu_fwd_ms=node_cost_cpu_fwd_ms,
        node_cost_cpu_bwd_ms=node_cost_cpu_bwd_ms,
        node_cost_gpu_fwd_ms=node_cost_gpu_fwd_ms,
        node_cost_gpu_bwd_ms=node_cost_gpu_bwd_ms,
        gpu_budget_mb=gpu_budget_mb,
        cpu_budget_mb=cpu_budget_mb,
        expand_budget=expand_budget,
    )
    repaired = correct_solution_memory(
        prepared=prepared,
        forward_assignment=expanded["forward_assignment"],
        backward_assignment=expanded["backward_assignment"],
        activation_strategy=activation_strategy,
        gpu_budget_mb=gpu_budget_mb,
        cpu_budget_mb=cpu_budget_mb,
        repair_budget=repair_budget,
    )
    repaired["flips_applied"] = expanded["flips_applied"] + repaired["flips_applied"]
    repaired["expand_flips_applied"] = expanded["flips_applied"]
    repaired["expand_flip_order"] = expanded["flip_order"]
    return repaired


def simulate_assignment_peak_memory(
    graph_edges_csv: str | Path,
    metrics_stats_csv: str | Path,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Convenience one-shot wrapper around PreparedTopologyMemory for a single evaluation."""
    prepared = PreparedTopologyMemory.prepare(graph_edges_csv, metrics_stats_csv)
    peak_gpu, peak_cpu = prepared.evaluate(forward_assignment, backward_assignment, activation_strategy)
    return peak_gpu, peak_cpu, dict(prepared.diagnostics)
