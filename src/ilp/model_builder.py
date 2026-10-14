from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

from .data_loader import ILPInputData
from .advanced_terms import ActivationMetadata

logger = logging.getLogger(__name__)


@dataclass
class ILPConfig:
    w_time: float = 1.0
    w_energy: float = 0.0
    enforce_convex_weights: bool = False
    convex_weight_tolerance: float = 1e-9
    w_transfer: float = 1.0
    w_fragmentation: float = 0.0
    w_congestion: float = 0.0
    congestion_knee_ms: float = 0.0
    gpu_mem_budget_mb: float = 1e18
    cpu_mem_budget_mb: float = 1e18
    memory_model: str = "topological"
    peak_activation_overlap: float = 0.35


@dataclass
class ILPConfig4(ILPConfig):
    """Extended configuration for Phase 4 activation persistence strategies."""
    w_io: float = 0.0  # Weight for I/O costs in checkpoint/recompute decisions
    w_recompute_penalty: float = 0.5  # Penalty multiplier for recompute strategy
    w_checkpoint_penalty: float = 0.3  # Penalty multiplier for checkpoint strategy
    enable_recompute: bool = True  # Allow recompute strategy
    enable_checkpoint: bool = False  # Allow checkpoint strategy (I/O-based)
    # Fraction of gpu_mem_budget_mb that _solve_phase4_greedy (src/ilp/solve.py)
    # treats as both the trigger ("start applying recompute/checkpoint once
    # GPU memory usage exceeds this fraction of budget") and the target
    # ("stop as soon as usage drops back to this fraction").
    #
    # Raised from the original hardcoded 0.60 to 0.95 once the underlying bug
    # that motivated such a steep margin was fixed at the source: the
    # baseline (Phase 3) solve this heuristic starts from already enforces
    # gpu_mem_budget_mb as a hard ILP constraint, so gpu_mem_used can never
    # exceed that budget by construction. 0.60 was compensating for that
    # budget's own per-layer memory estimate (training_profiler.py's
    # output_bytes/mem_mb) silently underestimating real usage for layers
    # past an OOM-retry point during profiling -- now that measurement is
    # rescaled to the real batch size, a second independent ~40% haircut on
    # top of an already-correct, already-budget-respecting solution has no
    # evidence behind it: it just forces recompute/checkpoint (trading
    # compute time for no real memory-safety benefit) far more often than
    # the real constraint requires. 0.95 means Phase 4 now only intervenes
    # when the baseline solution is genuinely within 5% of the real budget.
    target_gpu_utilization: float = 0.95
    # When True, _solve_with_pulp_dual (src/ilp/solve.py) adds recompute[n]/
    # checkpoint[n] decision variables to the REAL MILP itself, so a node
    # whose GPU memory would otherwise force it entirely onto CPU can
    # instead stay on GPU by recomputing/checkpointing its activation --
    # a joint device+activation-strategy decision, instead of today's
    # greedy/exhaustive Phase 4 heuristics, which only ever adjust the
    # activation strategy of nodes the baseline MILP already placed on GPU
    # and can never move a CPU-assigned node back to GPU this way. Default
    # False: every existing call site (the whole validated 4-model
    # campaign) is unaffected unless this is explicitly turned on. See
    # fix/07_gpu_recompute_joint_decision/README.md.
    enable_joint_activation_strategy: bool = False
    # Real, measured (not theoretical) recompute memory-savings curve for
    # THIS model/batch -- see src/runner/recompute_curve_calibration.py and
    # fix/07_gpu_recompute_joint_decision/README.md. Confirmed empirically
    # that recompute's real memory benefit is NOT additive per node: it
    # rises as more candidate blocks are marked recompute, peaks, then
    # reverses to WORSE than doing nothing at all once too many are marked
    # simultaneously -- in two different architectures (resnet50, peak at
    # ~25% of blocks; distilgpt2, peak at ~83%), and two simpler
    # hypotheses (a fixed per-"frame" memory cost; an inherently "bad"
    # block regardless of context) were each tested and refuted. Rather
    # than guess a formula, the real curve is measured for the real case
    # and handed to the MILP as piecewise-linear data.
    #
    # recompute_block_order: candidate blocks (each a list of layer names
    # sharing one activation-strategy decision -- see
    # _block_id_for_layer_name), ordered from highest to lowest theoretical
    # savings (typically by summed gpu_act_mem). The MILP is constrained to
    # only ever choose a PREFIX of this list (see _solve_with_pulp_dual):
    # the calibration below was measured for exactly that ordering, never
    # for an arbitrary subset, so the MILP must not be free to pick one.
    #
    # recompute_curve_points: measured (n_blocks, peak_gpu_mem_mb) pairs
    # from measure_recompute_curve, MUST include a point at n_blocks=0
    # (the no-recompute baseline this curve's deltas are measured against).
    #
    # Both default to empty: enable_joint_activation_strategy=True with no
    # curve calibrated falls back to the plain per-node additive term
    # (today's behavior, known to be an overestimate of real savings but a
    # safe, conservative one -- it only ever OVER-predicts savings, never
    # under-predicts memory use, so it cannot by itself cause an OOM it
    # wouldn't already cause without this feature).
    recompute_block_order: list = None
    recompute_curve_points: list = None

    def __post_init__(self):
        if self.recompute_block_order is None:
            self.recompute_block_order = []
        if self.recompute_curve_points is None:
            self.recompute_curve_points = []


@dataclass
class ILPProblemData:
    objective_node_gpu: Dict[str, float]
    objective_node_cpu: Dict[str, float]
    objective_edge_cut: Dict[Tuple[str, str], float]
    gpu_mem: Dict[str, float]
    cpu_mem: Dict[str, float]
    gpu_base_mem: Dict[str, float] = None
    gpu_act_mem: Dict[str, float] = None
    cpu_base_mem: Dict[str, float] = None
    cpu_act_mem: Dict[str, float] = None


@dataclass
class ILPProblemDataDual:
    objective_fwd_gpu: Dict[str, float]
    objective_fwd_cpu: Dict[str, float]
    objective_bwd_gpu: Dict[str, float]
    objective_bwd_cpu: Dict[str, float]
    objective_edge_cut_forward: Dict[Tuple[str, str], float]
    objective_edge_cut_backward: Dict[Tuple[str, str], float]
    # Cost of a node running forward on one device and backward on the
    # other. Direction-specific because the real runtime
    # (src/runtime/hybrid_executor.py::_run_layer_with_dual_placement) does
    # a full extra forward recompute on the BACKWARD device before it can
    # compute gradients there -- that recompute's cost depends entirely on
    # which device it runs on (CPU recompute vs GPU recompute differ by the
    # same large factor as this project's own measured CPU/GPU TFLOPS gap).
    # A single undirected cost here would price a cheap CPU-side recompute
    # the same as an expensive CPU-side recompute, which is exactly the
    # modeling gap that let the ILP under-price this: it was using a small
    # transfer-time proxy (node_time_io_ms) for what is actually a second
    # full forward pass.
    objective_cross_phase_cpu_to_gpu: Dict[str, float]  # fwd=CPU, bwd=GPU -> recompute on GPU
    objective_cross_phase_gpu_to_cpu: Dict[str, float]  # fwd=GPU, bwd=CPU -> recompute on CPU
    gpu_mem: Dict[str, float]
    cpu_mem: Dict[str, float]
    gpu_base_mem: Dict[str, float] = None
    gpu_act_mem: Dict[str, float] = None
    cpu_base_mem: Dict[str, float] = None
    cpu_act_mem: Dict[str, float] = None


def validate_ilp_config(cfg: ILPConfig) -> None:
    if cfg.w_time < 0:
        raise ValueError(f"w_time must be >= 0, got {cfg.w_time}")
    if cfg.w_energy < 0:
        raise ValueError(f"w_energy must be >= 0, got {cfg.w_energy}")
    if cfg.enforce_convex_weights:
        weight_sum = float(cfg.w_time + cfg.w_energy)
        if abs(weight_sum - 1.0) > float(cfg.convex_weight_tolerance):
            raise ValueError(
                "w_time + w_energy must be 1.0 when enforce_convex_weights=true; "
                f"got {weight_sum} (w_time={cfg.w_time}, w_energy={cfg.w_energy})"
            )
    if cfg.w_transfer < 0:
        raise ValueError(f"w_transfer must be >= 0, got {cfg.w_transfer}")
    if cfg.w_fragmentation < 0:
        raise ValueError(f"w_fragmentation must be >= 0, got {cfg.w_fragmentation}")
    if cfg.w_congestion < 0:
        raise ValueError(f"w_congestion must be >= 0, got {cfg.w_congestion}")
    if cfg.congestion_knee_ms < 0:
        raise ValueError(f"congestion_knee_ms must be >= 0, got {cfg.congestion_knee_ms}")
    if cfg.gpu_mem_budget_mb < 0:
        raise ValueError(f"gpu_mem_budget_mb must be >= 0, got {cfg.gpu_mem_budget_mb}")
    if cfg.cpu_mem_budget_mb < 0:
        raise ValueError(f"cpu_mem_budget_mb must be >= 0, got {cfg.cpu_mem_budget_mb}")
    if cfg.memory_model not in {"topological", "topological_tight"}:
        raise ValueError(
            "memory_model must be 'topological' or 'topological_tight', "
            f"got {cfg.memory_model}"
        )
    if not (0.0 <= cfg.peak_activation_overlap <= 1.0):
        raise ValueError(
            f"peak_activation_overlap must be in [0,1], got {cfg.peak_activation_overlap}"
        )


def _build_effective_memory(data: ILPInputData, cfg: ILPConfig) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    gpu_mem: Dict[str, float] = {}
    cpu_mem: Dict[str, float] = {}
    gpu_base_mem: Dict[str, float] = {}
    gpu_act_mem: Dict[str, float] = {}
    cpu_base_mem: Dict[str, float] = {}
    cpu_act_mem: Dict[str, float] = {}

    for n in data.nodes:
        gpu_base = float(data.node_mem_gpu_mb.get(n, 0.0))
        cpu_base = float(data.node_mem_cpu_mb.get(n, 0.0))

        # Real measured per-phase GPU peak (training_profiler.py's hook-based
        # delta), NOT the theoretical node_mem_activation_mb sum: that value
        # only ever reflected the activation TENSOR's own size, never the
        # real transient/workspace cost (cuDNN workspace, intermediate
        # buffers) layered on top of it, and never had any backward-specific
        # figure at all before this session -- gpu_bwd_mem_peak_mb is new.
        # Measured, empirically, to matter a lot: backward's own peak is
        # frequently 4-5x the forward activation for conv-heavy layers (see
        # fix/06 README / PLAN_ACCION_JURADO.md item 42). Using max(fwd,bwd)
        # rather than their sum reflects that forward's and backward's own
        # transient peaks for the SAME node happen sequentially in real
        # execution, never simultaneously -- confirmed by the validated
        # topological simulator (src/ilp/topology_memory.py), which performs
        # the same max() rather than a sum for exactly this reason.
        # gpu_base bundles params+activations_mb(theoretical)+grads+optimizer
        # in one sum (node_mem_gpu_mb). To swap in the real peak as the
        # activation term without losing the static params/grads/optimizer
        # portion, subtract the OLD theoretical activation estimate (never
        # the new real one) to recover that static portion first.
        theoretical_act = max(float(data.node_mem_activation_mb.get(n, gpu_base * 0.70)), 0.0)
        static_non_act_gpu = max(0.0, gpu_base - theoretical_act)

        real_fwd_peak = max(float(data.node_mem_gpu_fwd_peak_mb.get(n, 0.0)), 0.0)
        real_bwd_peak = max(float(data.node_mem_gpu_bwd_peak_mb.get(n, 0.0)), 0.0)
        act_gpu = max(real_fwd_peak, real_bwd_peak)
        if act_gpu <= 0.0:
            # No real peak measurement available (e.g. an older
            # metrics_stats.csv without gpu_mem_peak_mb/gpu_bwd_mem_peak_mb) --
            # fall back to the theoretical activation estimate rather than 0,
            # which would silently tell the ILP this node is free on GPU.
            act_gpu = theoretical_act

        non_act_gpu = static_non_act_gpu
        gpu_base = non_act_gpu + act_gpu  # keep gpu_base consistent with the real-peak-based act_gpu below
        act_ratio = (act_gpu / gpu_base) if gpu_base > 1e-12 else 0.0
        act_cpu = min(max(cpu_base * act_ratio, 0.0), max(cpu_base, 0.0))
        non_act_cpu = max(0.0, cpu_base - act_cpu)

        gpu_base_mem[n] = non_act_gpu
        gpu_act_mem[n] = act_gpu
        cpu_base_mem[n] = non_act_cpu
        cpu_act_mem[n] = act_cpu

        # memory_model is always "topological" or "topological_tight" here
        # (validated above). gpu_mem represents the raw peak without
        # heuristic overlap. topological_tight uses the SAME per-node
        # gpu_base_mem/gpu_act_mem/cpu_base_mem/cpu_act_mem values (needed by
        # Phase4's recompute-cost bookkeeping and by reporting) -- only the
        # GPU memory CONSTRAINT itself differs (see solve.py's
        # _build_tight_topological_gpu_constraints), not this per-node data.
        gpu_mem[n] = gpu_base
        cpu_mem[n] = cpu_base

    return gpu_mem, cpu_mem, gpu_base_mem, gpu_act_mem, cpu_base_mem, cpu_act_mem


def build_problem_data(data: ILPInputData, cfg: ILPConfig) -> ILPProblemData:
    validate_ilp_config(cfg)

    node_gpu = {}
    node_cpu = {}
    for n in data.nodes:
        node_gpu[n] = (cfg.w_time * data.node_cost_gpu_ms[n]) + (cfg.w_energy * data.node_energy_gpu_j[n])
        node_cpu[n] = (cfg.w_time * data.node_cost_cpu_ms[n]) + (cfg.w_energy * data.node_energy_cpu_j[n])

    edge_cut = {
        e: (cfg.w_transfer * data.edge_transfer_ms[e]) + cfg.w_fragmentation
        for e in data.edges
    }
    gpu_mem, cpu_mem, gpu_base, gpu_act, cpu_base, cpu_act = _build_effective_memory(data, cfg)

    return ILPProblemData(
        objective_node_gpu=node_gpu,
        objective_node_cpu=node_cpu,
        objective_edge_cut=edge_cut,
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        gpu_base_mem=gpu_base,
        gpu_act_mem=gpu_act,
        cpu_base_mem=cpu_base,
        cpu_act_mem=cpu_act,
    )


def build_problem_data_dual(data: ILPInputData, cfg: ILPConfig) -> ILPProblemDataDual:
    validate_ilp_config(cfg)

    objective_fwd_gpu = {
        n: (cfg.w_time * data.node_cost_gpu_fwd_ms[n]) + (cfg.w_energy * data.node_energy_gpu_fwd_j[n])
        for n in data.nodes
    }
    objective_fwd_cpu = {
        n: (cfg.w_time * data.node_cost_cpu_fwd_ms[n]) + (cfg.w_energy * data.node_energy_cpu_fwd_j[n])
        for n in data.nodes
    }
    objective_bwd_gpu = {
        n: (cfg.w_time * data.node_cost_gpu_bwd_ms[n]) + (cfg.w_energy * data.node_energy_gpu_bwd_j[n])
        for n in data.nodes
    }
    objective_bwd_cpu = {
        n: (cfg.w_time * data.node_cost_cpu_bwd_ms[n]) + (cfg.w_energy * data.node_energy_cpu_bwd_j[n])
        for n in data.nodes
    }
    edge_cut_forward = {
        e: (cfg.w_transfer * data.edge_transfer_ms[e]) + cfg.w_fragmentation
        for e in data.edges
    }
    edge_cut_backward = {
        e: (cfg.w_transfer * data.edge_transfer_ms[e]) + cfg.w_fragmentation
        for e in data.edges
    }
    # Real cost of a dual-placed node: the backward device must recompute
    # the ENTIRE forward pass before it can compute gradients
    # (_run_layer_with_dual_placement does exactly this), so the dominant
    # term is a full extra forward pass priced on the backward device --
    # not the small activation-transfer proxy (node_time_io_ms) this used
    # to be. node_time_io_ms is kept as an additive term since there is
    # still a real transfer (the saved CPU snapshot + moving the layer's
    # parameters between devices), just no longer the whole story.
    cross_phase_cpu_to_gpu = {  # fwd=CPU, bwd=GPU -> recompute happens on GPU
        n: objective_fwd_gpu[n] + (cfg.w_transfer * data.node_time_io_ms[n]) + cfg.w_fragmentation
        for n in data.nodes
    }
    cross_phase_gpu_to_cpu = {  # fwd=GPU, bwd=CPU -> recompute happens on CPU
        n: objective_fwd_cpu[n] + (cfg.w_transfer * data.node_time_io_ms[n]) + cfg.w_fragmentation
        for n in data.nodes
    }
    gpu_mem, cpu_mem, gpu_base, gpu_act, cpu_base, cpu_act = _build_effective_memory(data, cfg)

    return ILPProblemDataDual(
        objective_fwd_gpu=objective_fwd_gpu,
        objective_fwd_cpu=objective_fwd_cpu,
        objective_bwd_gpu=objective_bwd_gpu,
        objective_bwd_cpu=objective_bwd_cpu,
        objective_edge_cut_forward=edge_cut_forward,
        objective_edge_cut_backward=edge_cut_backward,
        objective_cross_phase_cpu_to_gpu=cross_phase_cpu_to_gpu,
        objective_cross_phase_gpu_to_cpu=cross_phase_gpu_to_cpu,
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        gpu_base_mem=gpu_base,
        gpu_act_mem=gpu_act,
        cpu_base_mem=cpu_base,
        cpu_act_mem=cpu_act,
    )


@dataclass
class ILPProblemData4(ILPProblemData):
    """Extended problem data for Phase 4 with activation strategies."""
    activation_meta: ActivationMetadata = None
    # Cost multipliers for different activation strategies per node
    recompute_cost_gpu: Dict[str, float] = None  # Additional time cost
    recompute_cost_cpu: Dict[str, float] = None
    checkpoint_cost_gpu: Dict[str, float] = None  # I/O time cost
    checkpoint_cost_cpu: Dict[str, float] = None


def build_problem_data_phase4(data: ILPInputData, cfg: ILPConfig4) -> ILPProblemData4:
    """Build extended problem data for Phase 4 activation persistence optimization."""
    validate_ilp_config(cfg)

    # Get base problem data
    base_data = build_problem_data(data, cfg)

    if data.activation_metadata_source != "provided" or data.io_metadata_source != "provided":
        raise ValueError(
            "Phase 4 requires explicit activation and I/O metadata derived from measured artifacts. "
            "Heuristic defaults are disabled for thesis-grade execution."
        )

    activation_meta = ActivationMetadata(
        node_mem_activation_mb=dict(data.node_mem_activation_mb),
        node_time_recompute_ms={
            # Recompute = forward pass only. Use fwd_ms if available; else estimate
            # as 1/(1+BACKWARD_FACTOR) fraction of total (e.g., 1/3 when factor=2.0).
            n: cfg.w_recompute_penalty * (
                data.node_cost_gpu_fwd_ms.get(n, 0.0)
                if data.node_cost_gpu_fwd_ms.get(n, 0.0) > 0
                else data.node_cost_gpu_ms.get(n, 0.0) / 3.0
            )
            for n in data.nodes
        },
        node_time_checkpoint_ms=dict(data.node_time_io_ms),
        node_energy_io_j=dict(data.node_energy_io_j),
    )

    # Compute recompute cost = cost of re-running the forward pass of this layer.
    # Use measured forward time directly; do NOT use total * 0.5 (incorrect when bwd != fwd).
    recompute_cost_gpu = {
        n: cfg.w_recompute_penalty * data.node_cost_gpu_fwd_ms.get(n, 0.0)
        for n in data.nodes
    }
    recompute_cost_cpu = {
        n: cfg.w_recompute_penalty * data.node_cost_cpu_fwd_ms.get(n, 0.0)
        for n in data.nodes
    }
    
    # Compute checkpoint cost (I/O time)
    checkpoint_cost_gpu = {
        n: cfg.w_io * activation_meta.node_time_checkpoint_ms.get(n, 0.0)
        for n in data.nodes
    }
    checkpoint_cost_cpu = {
        n: cfg.w_io * activation_meta.node_time_checkpoint_ms.get(n, 0.0)
        for n in data.nodes
    }

    return ILPProblemData4(
        objective_node_gpu=base_data.objective_node_gpu,
        objective_node_cpu=base_data.objective_node_cpu,
        objective_edge_cut=base_data.objective_edge_cut,
        gpu_mem=base_data.gpu_mem,
        cpu_mem=base_data.cpu_mem,
        activation_meta=activation_meta,
        recompute_cost_gpu=recompute_cost_gpu,
        recompute_cost_cpu=recompute_cost_cpu,
        checkpoint_cost_gpu=checkpoint_cost_gpu,
        checkpoint_cost_cpu=checkpoint_cost_cpu,
    )
