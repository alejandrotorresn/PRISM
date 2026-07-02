from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ILPInputData:
    nodes: List[str]
    node_cost_gpu_ms: Dict[str, float]
    node_cost_cpu_ms: Dict[str, float]
    node_cost_gpu_fwd_ms: Dict[str, float] = None
    node_cost_gpu_bwd_ms: Dict[str, float] = None
    node_cost_cpu_fwd_ms: Dict[str, float] = None
    node_cost_cpu_bwd_ms: Dict[str, float] = None
    node_energy_gpu_j: Dict[str, float] = None
    node_energy_cpu_j: Dict[str, float] = None
    node_energy_gpu_fwd_j: Dict[str, float] = None
    node_energy_gpu_bwd_j: Dict[str, float] = None
    node_energy_cpu_fwd_j: Dict[str, float] = None
    node_energy_cpu_bwd_j: Dict[str, float] = None
    node_mem_gpu_mb: Dict[str, float] = None
    node_mem_cpu_mb: Dict[str, float] = None
    edges: List[Tuple[str, str]] = None
    edge_transfer_ms: Dict[Tuple[str, str], float] = None
    # Fase 4: activation persistence metadata
    node_mem_activation_mb: Dict[str, float] = None
    node_time_io_ms: Dict[str, float] = None
    node_energy_io_j: Dict[str, float] = None
    activation_metadata_source: str = "unknown"
    io_metadata_source: str = "unknown"
    graph_trace_source: str = "unknown"

    def __post_init__(self):
        if self.node_mem_gpu_mb is None:
            self.node_mem_gpu_mb = {n: 0.0 for n in self.nodes}
        if self.node_mem_cpu_mb is None:
            self.node_mem_cpu_mb = {n: 0.0 for n in self.nodes}
        if self.edges is None:
            self.edges = []
        if self.edge_transfer_ms is None:
            self.edge_transfer_ms = {}

        if self.node_cost_gpu_fwd_ms is None:
            self.node_cost_gpu_fwd_ms = {n: self.node_cost_gpu_ms.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_cost_gpu_bwd_ms is None:
            self.node_cost_gpu_bwd_ms = {n: self.node_cost_gpu_ms.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_cost_cpu_fwd_ms is None:
            self.node_cost_cpu_fwd_ms = {n: self.node_cost_cpu_ms.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_cost_cpu_bwd_ms is None:
            self.node_cost_cpu_bwd_ms = {n: self.node_cost_cpu_ms.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_energy_gpu_j is None:
            # Cannot derive joules from ms without power data; use 0.0 as a neutral default.
            self.node_energy_gpu_j = {n: 0.0 for n in self.nodes}
        if self.node_energy_cpu_j is None:
            # Cannot derive joules from ms without power data; use 0.0 as a neutral default.
            self.node_energy_cpu_j = {n: 0.0 for n in self.nodes}
        if self.node_energy_gpu_fwd_j is None:
            self.node_energy_gpu_fwd_j = {n: self.node_energy_gpu_j.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_energy_gpu_bwd_j is None:
            self.node_energy_gpu_bwd_j = {n: self.node_energy_gpu_j.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_energy_cpu_fwd_j is None:
            self.node_energy_cpu_fwd_j = {n: self.node_energy_cpu_j.get(n, 0.0) * 0.5 for n in self.nodes}
        if self.node_energy_cpu_bwd_j is None:
            self.node_energy_cpu_bwd_j = {n: self.node_energy_cpu_j.get(n, 0.0) * 0.5 for n in self.nodes}
        # Initialize Fase 4 fields with defaults if not provided
        if self.node_mem_activation_mb is None:
            self.node_mem_activation_mb = {
                n: self.node_mem_gpu_mb.get(n, 0.0) * 0.70 for n in self.nodes
            }
            self.activation_metadata_source = "heuristic_default"
        elif self.activation_metadata_source == "unknown":
            self.activation_metadata_source = "provided"
        if self.node_time_io_ms is None:
            self.node_time_io_ms = {
                n: self.node_cost_gpu_fwd_ms.get(n, 0.0) * 0.15 for n in self.nodes
            }
            self.io_metadata_source = "heuristic_default"
        elif self.io_metadata_source == "unknown":
            self.io_metadata_source = "provided"
        if self.node_energy_io_j is None:
            self.node_energy_io_j = {n: 0.05 for n in self.nodes}
            if self.io_metadata_source == "unknown":
                self.io_metadata_source = "heuristic_default"
        elif self.io_metadata_source == "unknown":
            self.io_metadata_source = "provided"


def _weighted_mean(values: List[float], weights: Optional[List[float]] = None) -> float:
    if not values:
        return 0.0
    if not weights:
        return float(sum(values) / len(values))
    wsum = float(sum(weights))
    if wsum <= 0:
        return float(sum(values) / len(values))
    return float(sum(v * w for v, w in zip(values, weights)) / wsum)


def _weighted_std(values: List[float], weights: Optional[List[float]] = None) -> float:
    if not values or len(values) == 1:
        return 0.0
    mu = _weighted_mean(values, weights)
    if not weights:
        var = sum((v - mu) ** 2 for v in values) / len(values)
        return float(math.sqrt(max(var, 0.0)))
    wsum = float(sum(weights))
    if wsum <= 0:
        var = sum((v - mu) ** 2 for v in values) / len(values)
        return float(math.sqrt(max(var, 0.0)))
    var = sum(w * ((v - mu) ** 2) for v, w in zip(values, weights)) / wsum
    return float(math.sqrt(max(var, 0.0)))


def _aggregate_values(
    values: List[float],
    strategy: str,
    dispersion_k: float,
    weights: Optional[List[float]] = None,
) -> float:
    if not values:
        return 0.0
    if strategy == "max":
        return float(max(values))
    if strategy == "mean":
        mu = _weighted_mean(values, weights)
        if dispersion_k <= 0:
            return mu
        sigma = _weighted_std(values, weights)
        return float(mu + (dispersion_k * sigma))
    raise ValueError(f"Unsupported multi-hardware aggregation strategy: {strategy}")


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


def _collapse_graph_to_measured_edges(
    graph: pd.DataFrame,
    transfer: pd.DataFrame,
    measured_nodes: set[str],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float], int]:
    required_graph_ids = {"src_id", "dst_id"}
    required_transfer_ids = {"src_id", "dst_id"}
    if not required_graph_ids.issubset(graph.columns) or not required_transfer_ids.issubset(transfer.columns):
        return [], {}, 0

    graph_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    measured_ids: List[Tuple[str, str]] = []
    transfer_by_id: Dict[Tuple[str, str], float] = {}

    for _, row in transfer.iterrows():
        src_id = str(row["src_id"])
        dst_id = str(row["dst_id"])
        transfer_by_id[(src_id, dst_id)] = _safe_num(row.get("transfer_sym_ms", 0.0))

    for _, row in graph.iterrows():
        src_id = str(row["src_id"])
        dst_id = str(row["dst_id"])
        producer = str(row["producer_name"])
        consumer = str(row["consumer_name"])
        graph_adj[src_id].append((dst_id, consumer))
        if producer in measured_nodes:
            measured_ids.append((src_id, producer))

    collapsed_edges: set[Tuple[str, str]] = set()
    collapsed_transfer: Dict[Tuple[str, str], float] = {}
    traversed_intermediate_edges = 0

    for start_id, start_name in measured_ids:
        queue = deque([(start_id, 0.0, 0)])
        visited = {start_id}
        best_depth: int | None = None
        found_edges: Dict[Tuple[str, str], float] = {}
        while queue:
            current_id, path_transfer, depth = queue.popleft()
            if best_depth is not None and depth >= best_depth:
                continue
            for next_id, next_name in graph_adj.get(current_id, []):
                if next_id in visited:
                    continue
                visited.add(next_id)

                edge_transfer = transfer_by_id.get((current_id, next_id), 0.0)
                next_transfer = max(path_transfer, edge_transfer)
                next_depth = depth + 1

                if next_name in measured_nodes:
                    if next_name != start_name:
                        edge = (start_name, next_name)
                        if best_depth is None or next_depth < best_depth:
                            best_depth = next_depth
                        if next_depth == best_depth:
                            found_edges[edge] = max(found_edges.get(edge, 0.0), next_transfer)
                    continue

                traversed_intermediate_edges += 1
                queue.append((next_id, next_transfer, next_depth))

        for edge, transfer_ms in found_edges.items():
            collapsed_edges.add(edge)
            collapsed_transfer[edge] = max(collapsed_transfer.get(edge, 0.0), transfer_ms)

    return sorted(collapsed_edges), collapsed_transfer, traversed_intermediate_edges


def load_measured_graph_artifacts(
    graph_edges_csv: str | Path,
    transfer_edges_csv: str | Path,
    measured_nodes: set[str],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
    graph_path = Path(graph_edges_csv)
    transfer_path = Path(transfer_edges_csv)

    if not graph_path.exists():
        raise FileNotFoundError(f"graph_edges csv not found: {graph_path}")
    if not transfer_path.exists():
        raise FileNotFoundError(f"transfer_edges csv not found: {transfer_path}")

    graph = pd.read_csv(graph_path)
    transfer = pd.read_csv(transfer_path)

    _require_columns(graph, ["producer_name", "consumer_name"], graph_path)
    _require_columns(transfer, ["producer_name", "consumer_name", "transfer_sym_ms"], transfer_path)

    collapsed_edges, collapsed_transfer, _ = _collapse_graph_to_measured_edges(
        graph=graph,
        transfer=transfer,
        measured_nodes=measured_nodes,
    )
    return collapsed_edges, collapsed_transfer


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


def _resolve_reference_meta_path(
    metrics_stats_csv: str | Path,
    graph_edges_csv: str | Path,
    transfer_edges_csv: str | Path,
) -> Path | None:
    stats_path = Path(metrics_stats_csv)
    graph_path = Path(graph_edges_csv)
    transfer_path = Path(transfer_edges_csv)

    candidates: List[Path] = []
    for artifact_path, suffix in (
        (graph_path, "_graph_edges.csv"),
        (transfer_path, "_transfer_edges.csv"),
    ):
        if artifact_path.name.endswith(suffix):
            candidates.append(artifact_path.with_name(artifact_path.name.replace(suffix, "_meta.json")))
        candidates.append(artifact_path.with_name(f"{artifact_path.stem}_meta.json"))
    if stats_path.name.endswith("_metrics_stats.csv"):
        candidates.append(stats_path.with_name(stats_path.name.replace("_metrics_stats.csv", "_meta.json")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_reference_meta(
    metrics_stats_csv: str | Path,
    graph_edges_csv: str | Path,
    transfer_edges_csv: str | Path,
) -> Dict[str, object] | None:
    meta_path = _resolve_reference_meta_path(metrics_stats_csv, graph_edges_csv, transfer_edges_csv)
    if meta_path is None:
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ilp_inputs(
    metrics_stats_csv: str,
    graph_edges_csv: str,
    transfer_edges_csv: str,
    k_sigma: float = 1.0,
    k_sigma_time: float | None = None,
    k_sigma_energy: float | None = None,
    strict_graph_mapping: bool = False,
    strict_transfer_mapping: bool = False,
    strict_metric_validity: bool = True,
    strict_sample_quality: bool = True,
    strict_transfer_calibration: bool = True,
    strict_graph_trace_source: bool = True,
) -> ILPInputData:
    if k_sigma < 0:
        raise ValueError(f"k_sigma must be >= 0, got {k_sigma}")
    if k_sigma_time is not None and k_sigma_time < 0:
        raise ValueError(f"k_sigma_time must be >= 0, got {k_sigma_time}")
    if k_sigma_energy is not None and k_sigma_energy < 0:
        raise ValueError(f"k_sigma_energy must be >= 0, got {k_sigma_energy}")

    # Backward-compatible behaviour: when not explicitly configured, use k_sigma
    # for both timing and energy uncertainty margins.
    k_sigma_time_eff = k_sigma if k_sigma_time is None else k_sigma_time
    k_sigma_energy_eff = k_sigma if k_sigma_energy is None else k_sigma_energy

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

    def _stats_series(col_name: str, default: float = 0.0) -> pd.Series:
        if col_name in stats.columns:
            return pd.to_numeric(stats[col_name], errors="coerce").fillna(default)
        return pd.Series([default] * len(stats), index=stats.index, dtype=float)

    if strict_sample_quality:
        required_quality_cols = ["quality_flag", "n_runs", "n_samples"]
        missing_quality_cols = [col for col in required_quality_cols if col not in stats.columns]
        if missing_quality_cols:
            raise ValueError(
                "metrics_stats.csv is missing required sample-quality audit columns: "
                f"{missing_quality_cols}. Re-aggregate profiling artifacts before solving ILP."
            )
        bad_quality = stats[stats["quality_flag"].astype(str).str.lower() != "ok"]
        if not bad_quality.empty:
            flagged_layers = bad_quality["layer"].astype(str).tolist()
            logger.warning(
                f"metrics_stats.csv contains {len(flagged_layers)} low-quality profiling rows (e.g. {flagged_layers[:5]}). "
                "Empirical GPU measurements are preserved and used as-is; the ILP will use "
                "real hardware telemetry even for flagged layers. Re-profile with more "
                "replicas to improve statistical reliability."
            )
            # NOTE: Empirical GPU data (timing and energy) is intentionally NOT zeroed
            # out here. Destroying measured silicon telemetry in favour of a theoretical
            # FLOPS-based lower bound produces ILP objectives that are orders of magnitude
            # below reality (e.g. 0.91 J vs 57.5 J NVML-measured for ResNet-152).
            # The analytical fallback regime is reserved for datasets where GPU timing is
            # genuinely zero (true OOM / hardware absence), which is caught by the
            # all_gpu_zero guard below (line ~448).

    meta: Dict[str, object] | None = None
    if strict_transfer_calibration or strict_graph_trace_source:
        meta = _load_reference_meta(stats_path, graph_path, transfer_path)
        if meta is None:
            raise ValueError(
                "Could not locate reference profiling metadata for strict artifact validation. "
                "Re-profile with the current pipeline or disable strict validation only for diagnostics."
            )

    if strict_transfer_calibration:
        calibration_source = str(meta.get("transfer_calibration_source", "unknown"))
        if calibration_source != "measured":
            raise ValueError(
                "Refusing ILP solve because transfer calibration is not empirically measured "
                f"(transfer_calibration_source={calibration_source}). "
                "Re-profile until calibration succeeds or disable strict transfer calibration enforcement only for diagnostics."
            )

    if strict_graph_trace_source:
        graph_trace_source = str(meta.get("graph_trace_source", "unknown"))
        allowed_graph_trace_sources = {"torch_fx", "torch_export_decoder_only"}
        if graph_trace_source not in allowed_graph_trace_sources:
            raise ValueError(
                "Refusing ILP solve because graph topology is not derived from an accepted structured trace "
                f"(graph_trace_source={graph_trace_source}). "
                "Re-profile with torch.fx/torch.export or disable strict graph trace validation only for diagnostics."
            )

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

    # Guardrail against degenerate profiling datasets that produce invalid ILP
    # conclusions (e.g., all CPU times equal to zero across layers).
    cpu_time_mean = (
        pd.to_numeric(stats["cpu_fwd_time_ms_mean"], errors="coerce").fillna(0.0)
        + pd.to_numeric(stats["cpu_bwd_time_ms_mean"], errors="coerce").fillna(0.0)
    )
    gpu_time_mean = (
        pd.to_numeric(stats["gpu_fwd_time_ms_mean"], errors="coerce").fillna(0.0)
        + pd.to_numeric(stats["gpu_bwd_time_ms_mean"], errors="coerce").fillna(0.0)
    )

    if (cpu_time_mean < 0).any() or (gpu_time_mean < 0).any():
        raise ValueError("Invalid profiling data: negative timing values detected")

    all_cpu_zero = bool((cpu_time_mean <= 0).all())
    all_gpu_zero = bool((gpu_time_mean <= 0).all())
    
    analytical_fallback = False
    if meta is not None:
        analytical_fallback = bool(meta.get("analytical_fallback_gpu", False))

    # For AdamW degeneration or complete GPU absence, force analytical fallback
    if all_gpu_zero and not analytical_fallback:
        logger.warning(
            f"Degenerate GPU telemetry detected in {stats_path} (all times are zero). "
            "Forcing analytical fallback regime to rescue partitioning."
        )
        analytical_fallback = True

    if all_cpu_zero:
        raise ValueError(
            "Invalid profiling data for ILP: "
            f"all CPU mean times are zero across layers in {stats_path}. "
            "This dataset is degenerate and cannot be partitioned even analytically."
        )

    zero_gpu_mask = gpu_time_mean <= 0

    if zero_gpu_mask.any():
        if all_gpu_zero and analytical_fallback:
            logger.warning(
                "Model structural requirements exceed physical VRAM (OOM). "
                "Triggering analytical estimation regime for GPU costs using CPU profiling and empirical TFLOPS."
            )
        else:
            logger.warning(
                "Some layers have missing or rejected (low-quality) empirical GPU metrics. "
                "Triggering analytical estimation regime for these specific layers."
            )
            
        tflops = 1.0
        tdp_w = 250.0
        mem_bw_gbps = 900.0  # Heuristic fallback bandwidth
        if meta is not None:
            tflops = float(meta.get("measured_gpu_peak_tflops", 0.0))
            tdp_w = float(meta.get("gpu_tdp_w", 250.0))
            mem_bw_gbps = float(meta.get("measured_gpu_mem_bw_gbps", 900.0))
            
        if tflops <= 0.0:
            logger.warning("Empirical GPU TFLOPS measurement missing. Falling back to safe 1.0 TFLOPS for projection.")
            tflops = 1.0
        if tdp_w <= 0.0:
            tdp_w = 250.0
        if mem_bw_gbps <= 0.0:
            mem_bw_gbps = 900.0

        flops_col = _stats_series("flops", default=0.0)
        cpu_mem = _stats_series("cpu_mem_mb_mean", default=0.0)
        
        # Roofline Projection: max(Compute Bound, Memory Bound)
        compute_time_ms = flops_col / (tflops * 1e9)
        memory_time_ms = cpu_mem / mem_bw_gbps
        
        # T_gpu_ms = max(compute_bound, memory_bound). Half for fwd, half for bwd.
        projected_time_ms = np.maximum(compute_time_ms, memory_time_ms) / 2.0
        
        stats.loc[zero_gpu_mask, "gpu_fwd_time_ms_mean"] = projected_time_ms[zero_gpu_mask]
        stats.loc[zero_gpu_mask, "gpu_bwd_time_ms_mean"] = projected_time_ms[zero_gpu_mask]
        stats.loc[zero_gpu_mask, "gpu_fwd_time_ms_std"] = 0.0
        stats.loc[zero_gpu_mask, "gpu_bwd_time_ms_std"] = 0.0
        
        # GPU Energy Projection: E_gpu_j = Power_w * (T_gpu_ms / 1000)
        projected_energy_j = tdp_w * (projected_time_ms / 1000.0)
        
        stats.loc[zero_gpu_mask, "gpu_fwd_energy_j_mean"] = projected_energy_j[zero_gpu_mask]
        stats.loc[zero_gpu_mask, "gpu_bwd_energy_j_mean"] = projected_energy_j[zero_gpu_mask]
        stats.loc[zero_gpu_mask, "gpu_fwd_energy_j_std"] = 0.0
        stats.loc[zero_gpu_mask, "gpu_bwd_energy_j_std"] = 0.0
        
        # GPU Memory Projection: Same as CPU memory (tensors are isomorphic)
        stats.loc[zero_gpu_mask, "gpu_mem_peak_mb_mean"] = cpu_mem[zero_gpu_mask]
        
        # Re-compute robust variables now that stats has been populated analytically
        gpu_time_mean = (
            pd.to_numeric(stats["gpu_fwd_time_ms_mean"], errors="coerce").fillna(0.0)
            + pd.to_numeric(stats["gpu_bwd_time_ms_mean"], errors="coerce").fillna(0.0)
        )

    if strict_metric_validity:
        zero_layer_count = int(((cpu_time_mean <= 0) | (gpu_time_mean <= 0)).sum())
        if zero_layer_count > 0:
            raise ValueError(
                "Invalid profiling data for ILP: "
                f"{zero_layer_count} layer(s) have non-positive CPU or GPU mean time. "
                "Disable strict_metric_validity only for explicit diagnostic runs."
            )

    gpu_time = _robust_value(stats, "gpu_fwd_time_ms", k_sigma_time_eff) + _robust_value(stats, "gpu_bwd_time_ms", k_sigma_time_eff)
    cpu_time = _robust_value(stats, "cpu_fwd_time_ms", k_sigma_time_eff) + _robust_value(stats, "cpu_bwd_time_ms", k_sigma_time_eff)
    gpu_energy = _robust_value(stats, "gpu_fwd_energy_j", k_sigma_energy_eff) + _robust_value(stats, "gpu_bwd_energy_j", k_sigma_energy_eff)
    cpu_energy = _robust_value(stats, "cpu_fwd_energy_j", k_sigma_energy_eff) + _robust_value(stats, "cpu_bwd_energy_j", k_sigma_energy_eff)

    activations_mb_mean = _stats_series("activations_mb_mean")
    transfer_d2h_ms_mean = _stats_series("transfer_d2h_ms_mean")
    transfer_edge_aware_total_ms_mean = _stats_series("transfer_edge_aware_total_ms_mean")

    stats["gpu_time_robust_ms"] = gpu_time
    stats["cpu_time_robust_ms"] = cpu_time
    stats["gpu_energy_robust_j"] = gpu_energy
    stats["cpu_energy_robust_j"] = cpu_energy
    stats["activations_mb_effective"] = activations_mb_mean
    stats["transfer_d2h_ms_effective"] = transfer_d2h_ms_mean
    stats["transfer_edge_aware_total_ms_effective"] = transfer_edge_aware_total_ms_mean

    nodes = sorted(stats["layer"].unique().tolist())

    node_cost_gpu_ms = {row["layer"]: _safe_num(row["gpu_time_robust_ms"]) for _, row in stats.iterrows()}
    node_cost_cpu_ms = {row["layer"]: _safe_num(row["cpu_time_robust_ms"]) for _, row in stats.iterrows()}
    node_energy_gpu_j = {row["layer"]: _safe_num(row["gpu_energy_robust_j"]) for _, row in stats.iterrows()}
    node_energy_cpu_j = {row["layer"]: _safe_num(row["cpu_energy_robust_j"]) for _, row in stats.iterrows()}
    node_cost_gpu_fwd_ms = {
        row["layer"]: _safe_num(row.get("gpu_fwd_time_ms_mean", 0.0)) + (k_sigma_time_eff * _safe_num(row.get("gpu_fwd_time_ms_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_cost_gpu_bwd_ms = {
        row["layer"]: _safe_num(row.get("gpu_bwd_time_ms_mean", 0.0)) + (k_sigma_time_eff * _safe_num(row.get("gpu_bwd_time_ms_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_cost_cpu_fwd_ms = {
        row["layer"]: _safe_num(row.get("cpu_fwd_time_ms_mean", 0.0)) + (k_sigma_time_eff * _safe_num(row.get("cpu_fwd_time_ms_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_cost_cpu_bwd_ms = {
        row["layer"]: _safe_num(row.get("cpu_bwd_time_ms_mean", 0.0)) + (k_sigma_time_eff * _safe_num(row.get("cpu_bwd_time_ms_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_energy_gpu_fwd_j = {
        row["layer"]: _safe_num(row.get("gpu_fwd_energy_j_mean", 0.0)) + (k_sigma_energy_eff * _safe_num(row.get("gpu_fwd_energy_j_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_energy_gpu_bwd_j = {
        row["layer"]: _safe_num(row.get("gpu_bwd_energy_j_mean", 0.0)) + (k_sigma_energy_eff * _safe_num(row.get("gpu_bwd_energy_j_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_energy_cpu_fwd_j = {
        row["layer"]: _safe_num(row.get("cpu_fwd_energy_j_mean", 0.0)) + (k_sigma_energy_eff * _safe_num(row.get("cpu_fwd_energy_j_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_energy_cpu_bwd_j = {
        row["layer"]: _safe_num(row.get("cpu_bwd_energy_j_mean", 0.0)) + (k_sigma_energy_eff * _safe_num(row.get("cpu_bwd_energy_j_std", 0.0)))
        for _, row in stats.iterrows()
    }
    node_mem_gpu_mb = {}
    node_mem_cpu_mb = {}
    missing_power_rate_layers = []

    for _, row in stats.iterrows():
        layer = row["layer"]
        isolated_mem = (
            _safe_num(row.get("params_mb_mean", 0.0)) +
            _safe_num(row.get("activations_mb_mean", 0.0)) +
            _safe_num(row.get("grads_mb_mean", 0.0)) +
            _safe_num(row.get("optimizer_states_mb_mean", 0.0))
        )
        if isolated_mem <= 0.0:
            isolated_mem = _safe_num(row.get("cpu_mem_mb_mean", 0.0))
        node_mem_gpu_mb[layer] = isolated_mem
        node_mem_cpu_mb[layer] = _safe_num(row.get("cpu_mem_mb_mean", 0.0))

    node_mem_activation_mb = {}
    node_time_io_ms = {}
    node_energy_io_j = {}

    for _, row in stats.iterrows():
        layer = row["layer"]
        gpu_base = max(node_mem_gpu_mb.get(layer, 0.0), 0.0)
        activation_mb = _safe_num(row.get("activations_mb_effective", row.get("activations_mb_mean", 0.0)))
        if activation_mb <= 0.0:
            activation_mb = gpu_base * 0.70  # Heuristic fallback: ~70% of total is activation
        gpu_base_for_clip = gpu_base if gpu_base > 1e-12 else activation_mb
        if activation_mb > gpu_base_for_clip + 1e-6 and gpu_base_for_clip > 1e-12:
            logger.warning(
                "Layer '%s': measured activations_mb=%.2f MB > gpu_total_mb=%.2f MB. "
                "This indicates inconsistent profiling data (likely global GPU snapshot vs. per-layer delta mismatch). "
                "Clipping to gpu_total. Re-run profiler with per-layer delta measurement to fix.",
                layer, activation_mb, gpu_base_for_clip,
            )
        node_mem_activation_mb[layer] = min(max(activation_mb, 0.0), gpu_base_for_clip)


        io_ms = _safe_num(row.get("transfer_d2h_ms_effective", row.get("transfer_d2h_ms_mean", 0.0)))
        if io_ms <= 0.0:
            io_ms = _safe_num(row.get("transfer_edge_aware_total_ms_effective", row.get("transfer_edge_aware_total_ms_mean", 0.0)))
        if io_ms <= 0.0:
            io_ms = _safe_num(row.get("gpu_fwd_time_ms_mean", 0.0)) * 0.15
        node_time_io_ms[layer] = max(io_ms, 0.0)

        gpu_time_mean = max(_safe_num(row.get("gpu_fwd_time_ms_mean", 0.0)) + _safe_num(row.get("gpu_bwd_time_ms_mean", 0.0)), 0.0)
        cpu_time_mean = max(_safe_num(row.get("cpu_fwd_time_ms_mean", 0.0)) + _safe_num(row.get("cpu_bwd_time_ms_mean", 0.0)), 0.0)
        gpu_energy_mean = max(_safe_num(row.get("gpu_fwd_energy_j_mean", 0.0)) + _safe_num(row.get("gpu_bwd_energy_j_mean", 0.0)), 0.0)
        cpu_energy_mean = max(_safe_num(row.get("cpu_fwd_energy_j_mean", 0.0)) + _safe_num(row.get("cpu_bwd_energy_j_mean", 0.0)), 0.0)

        power_rates: List[float] = []
        if gpu_time_mean > 0 and gpu_energy_mean > 0:
            power_rates.append(gpu_energy_mean / gpu_time_mean)
        if cpu_time_mean > 0 and cpu_energy_mean > 0:
            power_rates.append(cpu_energy_mean / cpu_time_mean)
        if power_rates:
            node_energy_io_j[layer] = max(node_time_io_ms[layer] * (sum(power_rates) / len(power_rates)), 0.0)
        else:
            node_energy_io_j[layer] = 0.0
            missing_power_rate_layers.append(layer)

    if missing_power_rate_layers:
        logger.warning(
            "I/O energy fallback used zero power-rate on %s layer(s) (example: %s); "
            "node_energy_io_j was set to 0.0 for those layers due to missing positive CPU/GPU energy-time telemetry.",
            len(missing_power_rate_layers),
            missing_power_rate_layers[:5],
        )

    measured_node_names = set(node_cost_gpu_ms)
    edges_raw = []
    dropped_graph_edges = 0
    dropped_boundary_edges = 0
    dropped_mismatch_edges = 0
    for _, row in graph.iterrows():
        u = str(row["producer_name"])
        v = str(row["consumer_name"])
        u_in = u in measured_node_names
        v_in = v in measured_node_names
        if u_in and v_in:
            edges_raw.append((u, v))
        else:
            dropped_graph_edges += 1

            # Boundary edges are expected in FX graphs (e.g., input placeholder -> first module,
            # or last module -> output node) and should not be treated as mapping errors.
            if u_in ^ v_in:
                dropped_boundary_edges += 1
            else:
                dropped_mismatch_edges += 1

    collapsed_edges, collapsed_transfer_ms, traversed_intermediate_edges = _collapse_graph_to_measured_edges(
        graph=graph,
        transfer=transfer,
        measured_nodes=measured_node_names,
    )
    if collapsed_edges:
        edges_raw = sorted(set(edges_raw).union(collapsed_edges))
        dropped_mismatch_edges = 0

    if dropped_mismatch_edges > 0:
        msg = (
            f"Dropped {dropped_mismatch_edges} graph edges due to node-name mismatch between "
            f"graph_edges and metrics_stats layer names"
        )
        if strict_graph_mapping:
            raise ValueError(msg)
        logger.warning(msg)

    if dropped_boundary_edges > 0:
        logger.info(
            "Dropped %s boundary graph edges (expected placeholder/output nodes not present in layer metrics)",
            dropped_boundary_edges,
        )
    if traversed_intermediate_edges > 0:
        logger.info(
            "Contracted %s FX graph edges through non-profiled intermediate nodes",
            traversed_intermediate_edges,
        )

    transfer_map: Dict[Tuple[str, str], float] = {}
    for _, row in transfer.iterrows():
        u = str(row["producer_name"])
        v = str(row["consumer_name"])
        transfer_map[(u, v)] = _safe_num(row.get("transfer_sym_ms", 0.0))

    edges = sorted(set(edges_raw))
    edge_transfer_ms: Dict[Tuple[str, str], float] = {}
    missing_transfer_edges = 0
    for e in edges:
        if e in collapsed_transfer_ms:
            edge_transfer_ms[e] = collapsed_transfer_ms[e]
        elif e in transfer_map:
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
        node_cost_gpu_fwd_ms=node_cost_gpu_fwd_ms,
        node_cost_gpu_bwd_ms=node_cost_gpu_bwd_ms,
        node_cost_cpu_fwd_ms=node_cost_cpu_fwd_ms,
        node_cost_cpu_bwd_ms=node_cost_cpu_bwd_ms,
        node_energy_gpu_j=node_energy_gpu_j,
        node_energy_cpu_j=node_energy_cpu_j,
        node_energy_gpu_fwd_j=node_energy_gpu_fwd_j,
        node_energy_gpu_bwd_j=node_energy_gpu_bwd_j,
        node_energy_cpu_fwd_j=node_energy_cpu_fwd_j,
        node_energy_cpu_bwd_j=node_energy_cpu_bwd_j,
        node_mem_gpu_mb=node_mem_gpu_mb,
        node_mem_cpu_mb=node_mem_cpu_mb,
        node_mem_activation_mb=node_mem_activation_mb,
        node_time_io_ms=node_time_io_ms,
        node_energy_io_j=node_energy_io_j,
        edges=edges,
        edge_transfer_ms=edge_transfer_ms,
        activation_metadata_source=("provided" if "activations_mb_mean" in stats.columns else "heuristic_default"),
        io_metadata_source=("provided" if ("transfer_d2h_ms_mean" in stats.columns or "transfer_edge_aware_total_ms_mean" in stats.columns) else "heuristic_default"),
        graph_trace_source=str(meta.get("graph_trace_source", "unknown")) if meta is not None else "unknown",
    )


def merge_ilp_inputs_multi_hardware(
    profiles: List[ILPInputData],
    strategy: str = "max",
    dispersion_k: float = 0.0,
    weights: Optional[List[float]] = None,
    strict_schema: bool = True,
) -> ILPInputData:
    if not profiles:
        raise ValueError("profiles must contain at least one ILPInputData")
    if dispersion_k < 0:
        raise ValueError(f"dispersion_k must be >= 0, got {dispersion_k}")
    if strategy not in {"max", "mean"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if weights is not None and len(weights) != len(profiles):
        raise ValueError("weights length must match number of profiles")

    base_nodes = profiles[0].nodes
    base_edges = profiles[0].edges
    if strict_schema:
        for idx, p in enumerate(profiles[1:], start=1):
            if p.nodes != base_nodes:
                raise ValueError(f"Node schema mismatch for hardware profile #{idx}")
            if p.edges != base_edges:
                raise ValueError(f"Edge schema mismatch for hardware profile #{idx}")

    nodes = list(base_nodes)
    edges = list(base_edges)

    def agg_node(metric_getter):
        out: Dict[str, float] = {}
        for n in nodes:
            vals = [float(metric_getter(p, n)) for p in profiles]
            out[n] = _aggregate_values(vals, strategy=strategy, dispersion_k=dispersion_k, weights=weights)
        return out

    def agg_edge(metric_getter):
        out: Dict[Tuple[str, str], float] = {}
        for e in edges:
            vals = [float(metric_getter(p, e)) for p in profiles]
            out[e] = _aggregate_values(vals, strategy=strategy, dispersion_k=dispersion_k, weights=weights)
        return out

    merged = ILPInputData(
        nodes=nodes,
        node_cost_gpu_ms=agg_node(lambda p, n: p.node_cost_gpu_ms[n]),
        node_cost_cpu_ms=agg_node(lambda p, n: p.node_cost_cpu_ms[n]),
        node_cost_gpu_fwd_ms=agg_node(lambda p, n: p.node_cost_gpu_fwd_ms[n]),
        node_cost_gpu_bwd_ms=agg_node(lambda p, n: p.node_cost_gpu_bwd_ms[n]),
        node_cost_cpu_fwd_ms=agg_node(lambda p, n: p.node_cost_cpu_fwd_ms[n]),
        node_cost_cpu_bwd_ms=agg_node(lambda p, n: p.node_cost_cpu_bwd_ms[n]),
        node_energy_gpu_j=agg_node(lambda p, n: p.node_energy_gpu_j[n]),
        node_energy_cpu_j=agg_node(lambda p, n: p.node_energy_cpu_j[n]),
        node_energy_gpu_fwd_j=agg_node(lambda p, n: p.node_energy_gpu_fwd_j[n]),
        node_energy_gpu_bwd_j=agg_node(lambda p, n: p.node_energy_gpu_bwd_j[n]),
        node_energy_cpu_fwd_j=agg_node(lambda p, n: p.node_energy_cpu_fwd_j[n]),
        node_energy_cpu_bwd_j=agg_node(lambda p, n: p.node_energy_cpu_bwd_j[n]),
        node_mem_gpu_mb=agg_node(lambda p, n: p.node_mem_gpu_mb[n]),
        node_mem_cpu_mb=agg_node(lambda p, n: p.node_mem_cpu_mb[n]),
        edges=edges,
        edge_transfer_ms=agg_edge(lambda p, e: p.edge_transfer_ms[e]),
        graph_trace_source=(
            profiles[0].graph_trace_source
            if all(p.graph_trace_source == profiles[0].graph_trace_source for p in profiles)
            else "mixed"
        ),
    )

    logger.info(
        "Merged %s hardware profiles into one ILPInputData (strategy=%s, dispersion_k=%.3f)",
        len(profiles),
        strategy,
        dispersion_k,
    )
    return merged
