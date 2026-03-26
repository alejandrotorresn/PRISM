from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .data_loader import ILPInputData
from .advanced_terms import ActivationMetadata


@dataclass
class ILPConfig:
    w_time: float = 1.0
    w_energy: float = 0.0
    w_transfer: float = 1.0
    gpu_mem_budget_mb: float = 1e18
    cpu_mem_budget_mb: float = 1e18


@dataclass
class ILPConfig4(ILPConfig):
    """Extended configuration for Phase 4 activation persistence strategies."""
    w_io: float = 0.0  # Weight for I/O costs in checkpoint/recompute decisions
    w_recompute_penalty: float = 0.5  # Penalty multiplier for recompute strategy
    w_checkpoint_penalty: float = 0.3  # Penalty multiplier for checkpoint strategy
    enable_recompute: bool = True  # Allow recompute strategy
    enable_checkpoint: bool = False  # Allow checkpoint strategy (I/O-based)


@dataclass
class ILPProblemData:
    objective_node_gpu: Dict[str, float]
    objective_node_cpu: Dict[str, float]
    objective_edge_cut: Dict[Tuple[str, str], float]
    gpu_mem: Dict[str, float]
    cpu_mem: Dict[str, float]


@dataclass
class ILPProblemDataDual:
    objective_fwd_gpu: Dict[str, float]
    objective_fwd_cpu: Dict[str, float]
    objective_bwd_gpu: Dict[str, float]
    objective_bwd_cpu: Dict[str, float]
    objective_edge_cut_forward: Dict[Tuple[str, str], float]
    objective_edge_cut_backward: Dict[Tuple[str, str], float]
    objective_cross_phase: Dict[str, float]
    gpu_mem: Dict[str, float]
    cpu_mem: Dict[str, float]


def validate_ilp_config(cfg: ILPConfig) -> None:
    if cfg.w_time < 0:
        raise ValueError(f"w_time must be >= 0, got {cfg.w_time}")
    if cfg.w_energy < 0:
        raise ValueError(f"w_energy must be >= 0, got {cfg.w_energy}")
    if cfg.w_transfer < 0:
        raise ValueError(f"w_transfer must be >= 0, got {cfg.w_transfer}")
    if cfg.gpu_mem_budget_mb < 0:
        raise ValueError(f"gpu_mem_budget_mb must be >= 0, got {cfg.gpu_mem_budget_mb}")
    if cfg.cpu_mem_budget_mb < 0:
        raise ValueError(f"cpu_mem_budget_mb must be >= 0, got {cfg.cpu_mem_budget_mb}")


def build_problem_data(data: ILPInputData, cfg: ILPConfig) -> ILPProblemData:
    validate_ilp_config(cfg)

    node_gpu = {}
    node_cpu = {}
    for n in data.nodes:
        node_gpu[n] = (cfg.w_time * data.node_cost_gpu_ms[n]) + (cfg.w_energy * data.node_energy_gpu_j[n])
        node_cpu[n] = (cfg.w_time * data.node_cost_cpu_ms[n]) + (cfg.w_energy * data.node_energy_cpu_j[n])

    edge_cut = {e: cfg.w_transfer * data.edge_transfer_ms[e] for e in data.edges}

    return ILPProblemData(
        objective_node_gpu=node_gpu,
        objective_node_cpu=node_cpu,
        objective_edge_cut=edge_cut,
        gpu_mem=data.node_mem_gpu_mb,
        cpu_mem=data.node_mem_cpu_mb,
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
    edge_cut_forward = {e: cfg.w_transfer * data.edge_transfer_ms[e] for e in data.edges}
    edge_cut_backward = {e: cfg.w_transfer * data.edge_transfer_ms[e] for e in data.edges}
    cross_phase = {n: cfg.w_transfer * data.node_time_io_ms[n] for n in data.nodes}

    return ILPProblemDataDual(
        objective_fwd_gpu=objective_fwd_gpu,
        objective_fwd_cpu=objective_fwd_cpu,
        objective_bwd_gpu=objective_bwd_gpu,
        objective_bwd_cpu=objective_bwd_cpu,
        objective_edge_cut_forward=edge_cut_forward,
        objective_edge_cut_backward=edge_cut_backward,
        objective_cross_phase=cross_phase,
        gpu_mem=data.node_mem_gpu_mb,
        cpu_mem=data.node_mem_cpu_mb,
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
            n: cfg.w_recompute_penalty * data.node_cost_gpu_ms.get(n, 0.0) * 0.5
            for n in data.nodes
        },
        node_time_checkpoint_ms=dict(data.node_time_io_ms),
        node_energy_io_j=dict(data.node_energy_io_j),
    )

    # Compute recompute cost (additional forward pass time)
    recompute_cost_gpu = {
        n: cfg.w_recompute_penalty * data.node_cost_gpu_ms.get(n, 0.0) * 0.5
        for n in data.nodes
    }
    recompute_cost_cpu = {
        n: cfg.w_recompute_penalty * data.node_cost_cpu_ms.get(n, 0.0) * 0.5
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
