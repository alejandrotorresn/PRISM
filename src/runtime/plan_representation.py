from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.errors import EmptyDataError

VALID_DEVICES = {"CPU", "GPU"}


@dataclass
class ExecutionPlan:
    assignment: Dict[str, str]
    cut_edges: List[Tuple[str, str]]


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

    required_assign = {"layer", "device"}
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

    assignment: Dict[str, str] = {}
    for _, row in assign_df.iterrows():
        layer = str(row["layer"])
        device = str(row["device"]).upper()
        if device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}' for layer '{layer}'. Expected one of {sorted(VALID_DEVICES)}"
            )
        if layer in assignment:
            raise ValueError(f"Duplicated layer in assignment CSV: {layer}")
        assignment[layer] = device

    cut_edges = [(str(r["src_layer"]), str(r["dst_layer"])) for _, r in cut_df.iterrows()]

    return ExecutionPlan(assignment=assignment, cut_edges=cut_edges)


def load_graph_edges(graph_edges_csv: str | Path) -> List[Tuple[str, str]]:
    path = Path(graph_edges_csv)
    if not path.exists():
        raise FileNotFoundError(f"graph_edges_csv not found: {path}")

    df = pd.read_csv(path)
    required = {"producer_name", "consumer_name"}
    if not required.issubset(df.columns):
        raise KeyError(
            f"Missing columns in graph edges CSV {path}: "
            f"{sorted(required - set(df.columns))}"
        )

    edges = [
        (str(r["producer_name"]), str(r["consumer_name"]))
        for _, r in df.iterrows()
    ]
    return sorted(set(edges))


def load_transfer_costs(transfer_edges_csv: str | Path) -> Dict[Tuple[str, str], float]:
    path = Path(transfer_edges_csv)
    if not path.exists():
        raise FileNotFoundError(f"transfer_edges_csv not found: {path}")

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
