from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ilp.data_loader import load_ilp_inputs


def _write_graph(path: Path) -> None:
    pd.DataFrame(
        [
            {"producer_name": "a", "consumer_name": "b"},
        ]
    ).to_csv(path, index=False)


def _write_transfer(path: Path) -> None:
    pd.DataFrame(
        [
            {"producer_name": "a", "consumer_name": "b", "transfer_sym_ms": 0.1},
        ]
    ).to_csv(path, index=False)


def _write_stats(path: Path, cpu_zero: bool) -> None:
    rows = []
    for layer in ["a", "b"]:
        rows.append(
            {
                "layer": layer,
                "gpu_fwd_time_ms_mean": 1.0,
                "gpu_bwd_time_ms_mean": 1.0,
                "gpu_fwd_time_ms_std": 0.1,
                "gpu_bwd_time_ms_std": 0.1,
                "cpu_fwd_time_ms_mean": 0.0 if cpu_zero else 2.0,
                "cpu_bwd_time_ms_mean": 0.0 if cpu_zero else 2.0,
                "cpu_fwd_time_ms_std": 0.1,
                "cpu_bwd_time_ms_std": 0.1,
                "gpu_fwd_energy_j_mean": 0.5,
                "gpu_bwd_energy_j_mean": 0.5,
                "gpu_fwd_energy_j_std": 0.05,
                "gpu_bwd_energy_j_std": 0.05,
                "cpu_fwd_energy_j_mean": 1.0,
                "cpu_bwd_energy_j_mean": 1.0,
                "cpu_fwd_energy_j_std": 0.05,
                "cpu_bwd_energy_j_std": 0.05,
                "gpu_mem_peak_mb_mean": 10.0,
                "cpu_mem_mb_mean": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_ilp_inputs_rejects_all_cpu_zero(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats, cpu_zero=True)
    _write_graph(graph)
    _write_transfer(transfer)

    with pytest.raises(ValueError, match="all CPU mean times are zero"):
        _ = load_ilp_inputs(
            metrics_stats_csv=str(stats),
            graph_edges_csv=str(graph),
            transfer_edges_csv=str(transfer),
        )


def test_load_ilp_inputs_accepts_valid_stats(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats, cpu_zero=False)
    _write_graph(graph)
    _write_transfer(transfer)

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
    )

    assert data.nodes == ["a", "b"]
    assert len(data.edges) == 1
    assert data.edge_transfer_ms[("a", "b")] == 0.1
