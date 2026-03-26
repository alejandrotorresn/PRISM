from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ilp.data_loader import ILPInputData, load_ilp_inputs


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


def _write_meta(path: Path, *, transfer_calibration_source: str = "measured", graph_trace_source: str = "torch_fx") -> None:
    path.write_text(
        json.dumps(
            {
                "transfer_calibration_source": transfer_calibration_source,
                "graph_trace_source": graph_trace_source,
            }
        ),
        encoding="utf-8",
    )


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
                "quality_flag": "ok",
                "n_runs": 5,
                "n_samples": 5,
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
    _write_meta(tmp_path / "graph_edges_meta.json")

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
    _write_meta(tmp_path / "graph_edges_meta.json")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
    )

    assert data.nodes == ["a", "b"]
    assert len(data.edges) == 1
    assert data.edge_transfer_ms[("a", "b")] == 0.1
    assert data.graph_trace_source == "torch_fx"


def test_load_ilp_inputs_rejects_fallback_graph_trace_by_default(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats, cpu_zero=False)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json", graph_trace_source="fallback_leaf_modules")

    with pytest.raises(ValueError, match="graph topology is not derived from an accepted structured trace"):
        _ = load_ilp_inputs(
            metrics_stats_csv=str(stats),
            graph_edges_csv=str(graph),
            transfer_edges_csv=str(transfer),
        )


def test_load_ilp_inputs_can_allow_fallback_graph_trace_for_diagnostics(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats, cpu_zero=False)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json", graph_trace_source="fallback_leaf_modules")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
        strict_graph_trace_source=False,
    )

    assert data.graph_trace_source == "fallback_leaf_modules"


def test_ilp_input_data_minimal_init_is_safe() -> None:
    data = ILPInputData(
        nodes=["a"],
        node_cost_gpu_ms={"a": 1.0},
        node_cost_cpu_ms={"a": 2.0},
    )

    assert data.node_mem_gpu_mb["a"] == 0.0
    assert data.node_mem_cpu_mb["a"] == 0.0
    assert data.node_mem_activation_mb["a"] == 0.0
    assert data.edges == []
    assert data.edge_transfer_ms == {}


def test_load_ilp_inputs_contracts_unmeasured_fx_nodes(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats, cpu_zero=False)
    pd.DataFrame(
        [
            {"src_id": "n0", "dst_id": "n1", "producer_name": "x", "consumer_name": "a"},
            {"src_id": "n1", "dst_id": "n2", "producer_name": "a", "consumer_name": "hidden_1"},
            {"src_id": "n2", "dst_id": "n3", "producer_name": "hidden_1", "consumer_name": "hidden_2"},
            {"src_id": "n3", "dst_id": "n4", "producer_name": "hidden_2", "consumer_name": "b"},
            {"src_id": "n4", "dst_id": "n5", "producer_name": "b", "consumer_name": "y"},
        ]
    ).to_csv(graph, index=False)
    pd.DataFrame(
        [
            {"src_id": "n0", "dst_id": "n1", "producer_name": "x", "consumer_name": "a", "transfer_sym_ms": 0.1},
            {"src_id": "n1", "dst_id": "n2", "producer_name": "a", "consumer_name": "hidden_1", "transfer_sym_ms": 0.7},
            {"src_id": "n2", "dst_id": "n3", "producer_name": "hidden_1", "consumer_name": "hidden_2", "transfer_sym_ms": 0.2},
            {"src_id": "n3", "dst_id": "n4", "producer_name": "hidden_2", "consumer_name": "b", "transfer_sym_ms": 0.5},
            {"src_id": "n4", "dst_id": "n5", "producer_name": "b", "consumer_name": "y", "transfer_sym_ms": 0.1},
        ]
    ).to_csv(transfer, index=False)
    _write_meta(tmp_path / "graph_edges_meta.json")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
        strict_graph_mapping=True,
    )

    assert data.edges == [("a", "b")]
    assert data.edge_transfer_ms[("a", "b")] == 0.7
