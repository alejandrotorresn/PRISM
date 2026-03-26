"""
Advanced terms for Fase 4: activation persistence, recomputation, and checkpointing.

This module extends the Phase 3 baseline model with binary decisions per node:
  - retain(n) = 1: keep activation in memory on assigned device
  - recompute(n) = 1: do not retain; recompute in backward
  - checkpoint(n) = 1: save activation to intermediate storage (CPU/disk)

These three decisions are mutually exclusive: at most one can be 1 per node.

Memory impact:
  mem_effective_n = mem_forward_n + (retain(n) * mem_activation_n) 
                    + (checkpoint(n) * mem_activation_n)

Time impact:
  time_effective_n = time_forward_n 
                     + (recompute(n) * time_forward_n)
                     + (checkpoint(n) * time_io_n)

Energy impact:
  energy_effective_n = energy_forward_n + energy_backward_n
                       + (checkpoint(n) * energy_io_n)

Forward/backward independence (Phase 0 requirement):
  - x_f_n: assignment of forward pass
  - x_b_n: assignment of backward pass
  - Not modeled in baseline; Phase 4 MVP focuses on single-assignment variant.
    Full forward/backward independence deferred to Phase 5 if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ActivationStrategy:
    """
    Describes activation handling for a node in Phase 4.
    
    Exactly one of {retain, recompute, checkpoint} must be True.
    """

    node: str
    retain: bool = False
    recompute: bool = False
    checkpoint: bool = False

    def __post_init__(self):
        count = sum([self.retain, self.recompute, self.checkpoint])
        if count != 1:
            raise ValueError(
                f"Node {self.node}: exactly one of {{retain, recompute, checkpoint}} must be True, "
                f"got {count} active (retain={self.retain}, recompute={self.recompute}, checkpoint={self.checkpoint})"
            )

    @property
    def is_valid(self) -> bool:
        return sum([self.retain, self.recompute, self.checkpoint]) == 1


@dataclass
class ActivationMetadata:
    """
    Profiling data for activation persistence and recomputation.
    """

    node_mem_activation_mb: Dict[str, float]
    # Time to recompute forward pass (typically ~equal to forward time)
    node_time_recompute_ms: Dict[str, float]
    # Time to checkpoint/restore activation (I/O cost)
    node_time_checkpoint_ms: Dict[str, float]
    # Energy cost of I/O operations
    node_energy_io_j: Dict[str, float]


def estimate_activation_metadata(
    nodes: List[str],
    node_cost_gpu_ms: Dict[str, float],
    node_cost_cpu_ms: Dict[str, float],
    node_mem_gpu_mb: Dict[str, float],
    activation_mem_fraction: float = 0.70,
    io_time_fraction: float = 0.15,
    io_energy_fraction: float = 0.05,
) -> ActivationMetadata:
    """
    Estimate activation metadata from baseline profiling data.
    
    Args:
      nodes: list of layer names
      node_cost_gpu_ms: forward+backward time on GPU per node (ms)
      node_cost_cpu_ms: forward+backward time on CPU per node (ms)
      node_mem_gpu_mb: forward memory on GPU per node (MB)
      activation_mem_fraction: fraction of forward memory used by activation (default 0.70)
      io_time_fraction: checkpoint I/O time as fraction of forward time (default 0.15)
      io_energy_fraction: checkpoint I/O energy as fraction of forward energy (default 0.05)
      
    Returns:
      ActivationMetadata with reasonable defaults for Phase 4 optimization.
    """
    node_mem_activation_mb = {
        n: node_mem_gpu_mb.get(n, 0.0) * activation_mem_fraction for n in nodes
    }
    
    # Recompute cost = forward time (backward time is independent)
    node_time_recompute_ms = {
        n: node_cost_gpu_ms.get(n, 0.0) * 0.5 for n in nodes  # approximation: ~50% is forward
    }
    
    # Checkpoint I/O cost
    node_time_checkpoint_ms = {
        n: node_cost_gpu_ms.get(n, 0.0) * io_time_fraction for n in nodes
    }
    
    # Energy for I/O (conservative estimate)
    node_energy_io_j = {n: io_energy_fraction for n in nodes}
    
    return ActivationMetadata(
        node_mem_activation_mb=node_mem_activation_mb,
        node_time_recompute_ms=node_time_recompute_ms,
        node_time_checkpoint_ms=node_time_checkpoint_ms,
        node_energy_io_j=node_energy_io_j,
    )


def compute_effective_costs_phase4(
    nodes: List[str],
    assignment: Dict[str, str],  # node -> "GPU" or "CPU"
    strategies: Dict[str, ActivationStrategy],  # node -> strategy
    node_cost_gpu_ms: Dict[str, float],
    node_cost_cpu_ms: Dict[str, float],
    node_energy_gpu_j: Dict[str, float],
    node_energy_cpu_j: Dict[str, float],
    node_mem_gpu_mb: Dict[str, float],
    node_mem_cpu_mb: Dict[str, float],
    activation_meta: ActivationMetadata,
    w_time: float = 1.0,
    w_energy: float = 0.0,
    w_io: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute effective time, energy, and memory costs under Phase 4 strategies.
    
    Returns:
      (total_time_ms, total_energy_j, total_mem_mb)
    """
    total_time = 0.0
    total_energy = 0.0
    total_mem_gpu = 0.0
    total_mem_cpu = 0.0

    for n in nodes:
        if n not in assignment:
            continue
        
        On_gpu = assignment[n] == "GPU"
        base_cost = node_cost_gpu_ms[n] if On_gpu else node_cost_cpu_ms[n]
        base_energy = node_energy_gpu_j[n] if On_gpu else node_energy_cpu_j[n]
        base_mem = node_mem_gpu_mb[n] if On_gpu else node_mem_cpu_mb[n]
        
        strategy = strategies.get(n, ActivationStrategy(n, retain=True))
        
        # Time impact
        time_cost = base_cost
        if strategy.recompute:
            time_cost += activation_meta.node_time_recompute_ms.get(n, 0.0)
        if strategy.checkpoint:
            time_cost += activation_meta.node_time_checkpoint_ms.get(n, 0.0)
        
        # Energy impact
        energy_cost = base_energy
        if strategy.checkpoint:
            energy_cost += activation_meta.node_energy_io_j.get(n, 0.0)
        
        # Memory impact
        mem_cost = base_mem
        if strategy.retain or strategy.checkpoint:
            mem_cost += activation_meta.node_mem_activation_mb.get(n, 0.0)
        
        total_time += time_cost
        total_energy += energy_cost
        
        if On_gpu:
            total_mem_gpu += mem_cost
        else:
            total_mem_cpu += mem_cost
    
    # Aggregate objective
    total_obj = (w_time * total_time) + (w_energy * total_energy) + (w_io * (total_time + total_energy))
    
    return total_obj, total_mem_gpu, total_mem_cpu


def validate_strategies_phase4(
    strategies: Dict[str, ActivationStrategy],
) -> None:
    """Validate that all strategies are mutually exclusive. Raises ValueError on violation."""
    for node, strat in strategies.items():
        if not strat.is_valid:
            raise ValueError(
                f"Strategy for node '{node}' is invalid "
                f"(exactly one of retain/recompute/checkpoint must be active): {strat}"
            )


def estimate_memory_savings_phase4(
    assignment: Dict[str, str],
    strategies_base: Dict[str, ActivationStrategy],  # all retain (Phase 3 baseline)
    strategies_phase4: Dict[str, ActivationStrategy],  # Phase 4 with recompute/checkpoint
    activation_meta: ActivationMetadata,
) -> float:
    """
    Estimate peak GPU memory savings from Phase 4 strategies vs. baseline retention.
    
    Returns:
      savings_mb: reduction in peak GPU memory usage.
    """
    savings = 0.0
    for n in assignment:
        if assignment[n] != "GPU":
            continue
        
        if n not in activation_meta.node_mem_activation_mb:
            continue
        
        strat4 = strategies_phase4.get(n, ActivationStrategy(n, retain=True))
        strat_base = strategies_base.get(n, ActivationStrategy(n, retain=True))
        
        # Both base and phase4 retain: no savings
        if strat_base.retain and strat4.retain:
            savings += 0.0
        # Base retains but phase4 recomputes: save activation memory
        elif strat_base.retain and strat4.recompute:
            savings += activation_meta.node_mem_activation_mb[n]
        # Base retains but phase4 checkpoints: no GPU memory savings (move to CPU)
        elif strat_base.retain and strat4.checkpoint:
            savings += 0.0
    
    return savings
