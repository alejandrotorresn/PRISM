from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .data_loader import ILPInputData


@dataclass
class ILPConfig:
    w_time: float = 1.0
    w_energy: float = 0.0
    w_transfer: float = 1.0
    gpu_mem_budget_mb: float = 1e18
    cpu_mem_budget_mb: float = 1e18


@dataclass
class ILPProblemData:
    objective_node_gpu: Dict[str, float]
    objective_node_cpu: Dict[str, float]
    objective_edge_cut: Dict[Tuple[str, str], float]
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
