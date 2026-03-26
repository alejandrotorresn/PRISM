from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.errors import EmptyDataError

VALID_DEVICES = {"CPU", "GPU"}


@dataclass
class ExecutionPlan:
    assignment_forward: Dict[str, str]
    assignment_backward: Dict[str, str]
    cut_edges_forward: List[Tuple[str, str]]
    cut_edges_backward: List[Tuple[str, str]]
    cross_phase_edges: List[Tuple[str, str]]
    activation_strategies: Dict[str, str] = field(default_factory=dict)

    @property
    def assignment(self) -> Dict[str, str]:
        return self.assignment_forward

    @property
    def cut_edges(self) -> List[Tuple[str, str]]:
        return self.cut_edges_forward


@dataclass
class ILPInputPaths:
    metrics_stats_csv: Path
    graph_edges_csv: Path
    transfer_edges_csv: Path


def infer_ilp_input_paths(config_dir: Path, model_name: str) -> ILPInputPaths:
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([p for p in config_dir.glob("run_*") if p.is_dir()])
    if run_dirs:
        ref_run = run_dirs[0]
        graph_edges = ref_run / f"{model_name}_graph_edges.csv"
        transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    else:
        graph_edges = config_dir / f"{model_name}_graph_edges.csv"
        transfer_edges = config_dir / f"{model_name}_transfer_edges.csv"

    missing = [
        str(p)
        for p in (stats, graph_edges, transfer_edges)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Could not resolve required ILP input files: " + ", ".join(missing)
        )

    return ILPInputPaths(
        metrics_stats_csv=stats,
        graph_edges_csv=graph_edges,
        transfer_edges_csv=transfer_edges,
    )


def load_execution_plan(assignment_csv: str | Path, cut_edges_csv: str | Path) -> ExecutionPlan:
    assign_path = Path(assignment_csv)
    cut_path = Path(cut_edges_csv)

    if not assign_path.exists():
        raise FileNotFoundError(f"assignment_csv not found: {assign_path}")
    if not cut_path.exists():
        raise FileNotFoundError(f"cut_edges_csv not found: {cut_path}")

    assign_df = pd.read_csv(assign_path)
    try:
        cut_df = pd.read_csv(cut_path)
    except EmptyDataError:
        cut_df = pd.DataFrame(columns=["src_layer", "dst_layer"])

    required_assign = {"layer"}
    required_cut = {"src_layer", "dst_layer"}

    if not required_assign.issubset(assign_df.columns):
        raise KeyError(
            f"Missing columns in assignment CSV {assign_path}: "
            f"{sorted(required_assign - set(assign_df.columns))}"
        )
    if not required_cut.issubset(cut_df.columns):
        raise KeyError(
            f"Missing columns in cut CSV {cut_path}: "
            f"{sorted(required_cut - set(cut_df.columns))}"
        )

    forward_col = "device_forward" if "device_forward" in assign_df.columns else "device"
    backward_col = "device_backward" if "device_backward" in assign_df.columns else forward_col

    assignment_forward: Dict[str, str] = {}
    assignment_backward: Dict[str, str] = {}
    activation_strategies: Dict[str, str] = {}
    for _, row in assign_df.iterrows():
        layer = str(row["layer"])
        device_forward = str(row[forward_col]).upper()
        device_backward = str(row[backward_col]).upper()
        for device in [device_forward, device_backward]:
            if device not in VALID_DEVICES:
                raise ValueError(
                    f"Invalid device '{device}' for layer '{layer}'. Expected one of {sorted(VALID_DEVICES)}"
                )
        if layer in assignment_forward:
            raise ValueError(f"Duplicated layer in assignment CSV: {layer}")
        assignment_forward[layer] = device_forward
        assignment_backward[layer] = device_backward
        strategy = str(row.get("activation_strategy", "retain")).lower()
        if strategy in {"", "nan", "none"}:
            strategy = "retain"
        activation_strategies[layer] = strategy

    if "phase" in cut_df.columns:
        cut_edges_forward = [(str(r["src_layer"]), str(r["dst_layer"])) for _, r in cut_df[cut_df["phase"] == "forward"].iterrows()]
        cut_edges_backward = [(str(r["src_layer"]), str(r["dst_layer"])) for _, r in cut_df[cut_df["phase"] == "backward"].iterrows()]
        cross_phase_edges = [(str(r["src_layer"]), str(r["dst_layer"])) for _, r in cut_df[cut_df["phase"] == "cross_phase"].iterrows()]
    else:
        cut_edges_forward = [(str(r["src_layer"]), str(r["dst_layer"])) for _, r in cut_df.iterrows()]
        cut_edges_backward = list(cut_edges_forward)
        cross_phase_edges = []

    return ExecutionPlan(
        assignment_forward=assignment_forward,
        assignment_backward=assignment_backward,
        cut_edges_forward=cut_edges_forward,
        cut_edges_backward=cut_edges_backward,
        cross_phase_edges=cross_phase_edges,
        activation_strategies=activation_strategies,
    )


def load_graph_edges(
    graph_edges_csv: str | Path,
    transfer_edges_csv: str | Path | None = None,
    measured_layers: set[str] | None = None,
) -> List[Tuple[str, str]]:
    path = Path(graph_edges_csv)
    if not path.exists():
        raise FileNotFoundError(f"graph_edges_csv not found: {path}")

    if measured_layers and transfer_edges_csv is not None:
        from ilp.data_loader import load_measured_graph_artifacts

        edges, _ = load_measured_graph_artifacts(
            graph_edges_csv=path,
            transfer_edges_csv=transfer_edges_csv,
            measured_nodes=measured_layers,
        )
        if edges:
            return edges

    df = pd.read_csv(path)
    required = {"producer_name", "consumer_name"}
    if not required.issubset(df.columns):
        raise KeyError(
            f"Missing columns in graph edges CSV {path}: "
            f"{sorted(required - set(df.columns))}"
        )

    has_node_ids = {"src_id", "dst_id"}.issubset(df.columns)
    if not has_node_ids:
        edges = [
            (str(r["producer_name"]), str(r["consumer_name"]))
            for _, r in df.iterrows()
        ]
        return sorted(set(edges))

    def _parse_node_index(node_id: object) -> int | None:
        text = str(node_id)
        if not text.startswith("n"):
            return None
        suffix = text[1:]
        if not suffix.isdigit():
            return None
        return int(suffix)

    records: List[Tuple[str, str, int | None, int | None]] = []
    first_seen_idx: Dict[str, int] = {}

    for _, row in df.iterrows():
        producer = str(row["producer_name"])
        consumer = str(row["consumer_name"])
        src_idx = _parse_node_index(row["src_id"])
        dst_idx = _parse_node_index(row["dst_id"])

        records.append((producer, consumer, src_idx, dst_idx))

        if src_idx is not None:
            first_seen_idx[producer] = min(first_seen_idx.get(producer, src_idx), src_idx)
        if dst_idx is not None:
            first_seen_idx[consumer] = min(first_seen_idx.get(consumer, dst_idx), dst_idx)

    if not first_seen_idx:
        edges = [(producer, consumer) for producer, consumer, _, _ in records]
        return sorted(set(edges))

    # Collapse to layer-name edges while preserving chronological orientation derived
    # from node IDs. This avoids artificial cycles when a reused module name appears
    # multiple times in a traced graph (e.g., shared ReLU modules in ResNet blocks).
    oriented_edges: set[Tuple[str, str]] = set()
    for producer, consumer, src_idx, dst_idx in records:
        producer_first = first_seen_idx.get(producer)
        consumer_first = first_seen_idx.get(consumer)

        if producer_first is None or consumer_first is None:
            oriented_edges.add((producer, consumer))
            continue

        if producer_first < consumer_first:
            oriented_edges.add((producer, consumer))
            continue

        if producer_first == consumer_first:
            if src_idx is not None and dst_idx is not None and src_idx <= dst_idx:
                oriented_edges.add((producer, consumer))
            continue

        # producer_first > consumer_first: drop backward-in-time collapsed edge.

    return sorted(oriented_edges)


def load_transfer_costs(
    transfer_edges_csv: str | Path,
    graph_edges_csv: str | Path | None = None,
    measured_layers: set[str] | None = None,
) -> Dict[Tuple[str, str], float]:
    path = Path(transfer_edges_csv)
    if not path.exists():
        raise FileNotFoundError(f"transfer_edges_csv not found: {path}")

    if measured_layers and graph_edges_csv is not None:
        from ilp.data_loader import load_measured_graph_artifacts

        _, transfer_costs = load_measured_graph_artifacts(
            graph_edges_csv=graph_edges_csv,
            transfer_edges_csv=path,
            measured_nodes=measured_layers,
        )
        if transfer_costs:
            return transfer_costs

    df = pd.read_csv(path)
    required = {"producer_name", "consumer_name", "transfer_sym_ms"}
    if not required.issubset(df.columns):
        raise KeyError(
            f"Missing columns in transfer edges CSV {path}: "
            f"{sorted(required - set(df.columns))}"
        )

    out: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        edge = (str(row["producer_name"]), str(row["consumer_name"]))
        out[edge] = float(row["transfer_sym_ms"])
    return out
