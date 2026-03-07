from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class ILPInputData:
    nodes: List[str]
    node_cost_gpu_ms: Dict[str, float]
    node_cost_cpu_ms: Dict[str, float]
    node_energy_gpu_j: Dict[str, float]
    node_energy_cpu_j: Dict[str, float]
    node_mem_gpu_mb: Dict[str, float]
    node_mem_cpu_mb: Dict[str, float]
    edges: List[Tuple[str, str]]
    edge_transfer_ms: Dict[Tuple[str, str], float]


def _require_columns(df: pd.DataFrame, columns: List[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")


def _safe_num(val, default: float = 0.0) -> float:
    try:
        out = float(val)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


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


def load_ilp_inputs(
    metrics_stats_csv: str,
    graph_edges_csv: str,
    transfer_edges_csv: str,
    k_sigma: float = 1.0,
    strict_graph_mapping: bool = False,
    strict_transfer_mapping: bool = False,
) -> ILPInputData:
    if k_sigma < 0:
        raise ValueError(f"k_sigma must be >= 0, got {k_sigma}")

    stats_path = Path(metrics_stats_csv)
    graph_path = Path(graph_edges_csv)
    transfer_path = Path(transfer_edges_csv)

    if not stats_path.exists():
        raise FileNotFoundError(f"metrics_stats csv not found: {stats_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"graph_edges csv not found: {graph_path}")
    if not transfer_path.exists():
        raise FileNotFoundError(f"transfer_edges csv not found: {transfer_path}")

    stats = pd.read_csv(stats_path)
    graph = pd.read_csv(graph_path)
    transfer = pd.read_csv(transfer_path)

    _require_columns(
        stats,
        [
            "layer",
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
        ],
        stats_path,
    )
    _require_columns(graph, ["producer_name", "consumer_name"], graph_path)
    _require_columns(transfer, ["producer_name", "consumer_name", "transfer_sym_ms"], transfer_path)

    stats = stats.copy()
    stats["layer"] = stats["layer"].astype(str)

    gpu_time = _robust_value(stats, "gpu_fwd_time_ms", k_sigma) + _robust_value(stats, "gpu_bwd_time_ms", k_sigma)
    cpu_time = _robust_value(stats, "cpu_fwd_time_ms", k_sigma) + _robust_value(stats, "cpu_bwd_time_ms", k_sigma)
    gpu_energy = _robust_value(stats, "gpu_fwd_energy_j", k_sigma) + _robust_value(stats, "gpu_bwd_energy_j", k_sigma)
    cpu_energy = _robust_value(stats, "cpu_fwd_energy_j", k_sigma) + _robust_value(stats, "cpu_bwd_energy_j", k_sigma)

    stats["gpu_time_robust_ms"] = gpu_time
    stats["cpu_time_robust_ms"] = cpu_time
    stats["gpu_energy_robust_j"] = gpu_energy
    stats["cpu_energy_robust_j"] = cpu_energy

    nodes = sorted(stats["layer"].unique().tolist())

    node_cost_gpu_ms = {row["layer"]: _safe_num(row["gpu_time_robust_ms"]) for _, row in stats.iterrows()}
    node_cost_cpu_ms = {row["layer"]: _safe_num(row["cpu_time_robust_ms"]) for _, row in stats.iterrows()}
    node_energy_gpu_j = {row["layer"]: _safe_num(row["gpu_energy_robust_j"]) for _, row in stats.iterrows()}
    node_energy_cpu_j = {row["layer"]: _safe_num(row["cpu_energy_robust_j"]) for _, row in stats.iterrows()}
    node_mem_gpu_mb = {row["layer"]: _safe_num(row.get("gpu_mem_peak_mb_mean", 0.0)) for _, row in stats.iterrows()}
    node_mem_cpu_mb = {row["layer"]: _safe_num(row.get("cpu_mem_mb_mean", 0.0)) for _, row in stats.iterrows()}

    edges_raw = []
    dropped_graph_edges = 0
    for _, row in graph.iterrows():
        u = str(row["producer_name"])
        v = str(row["consumer_name"])
        if u in node_cost_gpu_ms and v in node_cost_gpu_ms:
            edges_raw.append((u, v))
        else:
            dropped_graph_edges += 1

    if dropped_graph_edges > 0:
        msg = (
            f"Dropped {dropped_graph_edges} graph edges due to node-name mismatch between "
            f"graph_edges and metrics_stats layer names"
        )
        if strict_graph_mapping:
            raise ValueError(msg)
        logger.warning(msg)

    transfer_map: Dict[Tuple[str, str], float] = {}
    for _, row in transfer.iterrows():
        u = str(row["producer_name"])
        v = str(row["consumer_name"])
        transfer_map[(u, v)] = _safe_num(row.get("transfer_sym_ms", 0.0))

    edges = sorted(set(edges_raw))
    edge_transfer_ms: Dict[Tuple[str, str], float] = {}
    missing_transfer_edges = 0
    for e in edges:
        if e in transfer_map:
            edge_transfer_ms[e] = transfer_map[e]
        else:
            edge_transfer_ms[e] = 0.0
            missing_transfer_edges += 1

    if missing_transfer_edges > 0:
        msg = (
            f"Missing transfer costs for {missing_transfer_edges} matched graph edges; "
            f"defaulted to 0.0"
        )
        if strict_transfer_mapping:
            raise ValueError(msg)
        logger.warning(msg)

    return ILPInputData(
        nodes=nodes,
        node_cost_gpu_ms=node_cost_gpu_ms,
        node_cost_cpu_ms=node_cost_cpu_ms,
        node_energy_gpu_j=node_energy_gpu_j,
        node_energy_cpu_j=node_energy_cpu_j,
        node_mem_gpu_mb=node_mem_gpu_mb,
        node_mem_cpu_mb=node_mem_cpu_mb,
        edges=edges,
        edge_transfer_ms=edge_transfer_ms,
    )
