"""
Regression tests ensuring empirical GPU energy/timing measurements are preserved
when quality_flag is 'low_sample' (e.g. n_runs=2 in HPC campaigns).

Prior to the fix, the data_loader.py zeroed out gpu_*_time_ms_mean and
gpu_*_energy_j_mean for any layer with quality_flag != 'ok', causing the ILP
solver to fall back to a theoretical FLOPS-based energy estimate that could be
~685x lower than the real NVML-measured value.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ilp.data_loader import load_ilp_inputs


def _write_graph(path: Path) -> None:
    pd.DataFrame([{"producer_name": "a", "consumer_name": "b"}]).to_csv(path, index=False)


def _write_transfer(path: Path) -> None:
    pd.DataFrame(
        [{"producer_name": "a", "consumer_name": "b", "transfer_sym_ms": 0.1}]
    ).to_csv(path, index=False)


def _write_meta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "transfer_calibration_source": "measured",
                "graph_trace_source": "torch_fx",
                "measured_gpu_peak_tflops": 19.17,
                "gpu_tdp_w": 250.0,
            }
        ),
        encoding="utf-8",
    )


def _write_stats_low_sample(path: Path, gpu_energy_fwd: float = 28.75) -> None:
    """Simulate a 2-replica HPC profile: low_sample quality_flag but real GPU energy."""
    rows = []
    for layer in ["a", "b"]:
        rows.append(
            {
                "layer": layer,
                # Real empirical GPU timing (not zero)
                "gpu_fwd_time_ms_mean": 93.5,
                "gpu_bwd_time_ms_mean": 93.5,
                "gpu_fwd_time_ms_std": 2.1,
                "gpu_bwd_time_ms_std": 2.1,
                # Real empirical CPU timing
                "cpu_fwd_time_ms_mean": 1110.0,
                "cpu_bwd_time_ms_mean": 1110.0,
                "cpu_fwd_time_ms_std": 15.0,
                "cpu_bwd_time_ms_std": 15.0,
                # Real empirical GPU energy (e.g. NVML-measured on A100)
                "gpu_fwd_energy_j_mean": gpu_energy_fwd,
                "gpu_bwd_energy_j_mean": gpu_energy_fwd,
                "gpu_fwd_energy_j_std": 0.5,
                "gpu_bwd_energy_j_std": 0.5,
                # Real CPU energy
                "cpu_fwd_energy_j_mean": 204.0,
                "cpu_bwd_energy_j_mean": 204.0,
                "cpu_fwd_energy_j_std": 3.0,
                "cpu_bwd_energy_j_std": 3.0,
                "gpu_mem_peak_mb_mean": 3506.0,
                "cpu_mem_mb_mean": 3276.0,
                # quality_flag = low_sample simulates n_runs=2 < MIN_RECOMMENDED_RUNS
                "quality_flag": "low_sample",
                "n_runs": 2,
                "n_samples": 2,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_empirical_gpu_energy_preserved_when_low_sample(tmp_path: Path) -> None:
    """
    With quality_flag='low_sample' but non-zero GPU energy, the loader must
    preserve the empirical NVML measurements and NOT replace them with zero.

    This is a regression test for the destructive overwrite bug where:
        stats.loc[bad_quality.index, "gpu_fwd_energy_j_mean"] = 0.0
    would destroy real measured data in favour of a theoretical FLOPS estimate.
    """
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"
    _write_stats_low_sample(stats, gpu_energy_fwd=28.75)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
    )

    total_gpu_fwd = sum(data.node_energy_gpu_fwd_j.values())
    total_gpu_bwd = sum(data.node_energy_gpu_bwd_j.values())

    # With k_sigma=1.0 (default), robust value = mean + 1.0 * std.
    # For each layer: fwd = 28.75 + 1.0*0.5 = 29.25. Two layers -> 58.5 J.
    # The result must be >> 0 (empirical), not ~0.91 J (analytical FLOPS estimate).
    assert total_gpu_fwd > 1.0, (
        f"GPU forward energy {total_gpu_fwd:.4f} J is suspiciously low — "
        "the empirical NVML data may have been destroyed by the analytical fallback."
    )
    assert total_gpu_bwd > 1.0, (
        f"GPU backward energy {total_gpu_bwd:.4f} J is suspiciously low — "
        "the empirical NVML data may have been destroyed by the analytical fallback."
    )


def test_empirical_gpu_timing_preserved_when_low_sample(tmp_path: Path) -> None:
    """
    Same scenario: GPU timing should also not be zeroed for low_sample layers.
    Zeroing timing would incorrectly trigger the 'all_gpu_zero' analytical fallback.
    """
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"
    _write_stats_low_sample(stats)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
    )

    total_gpu_fwd_ms = sum(data.node_cost_gpu_fwd_ms.values())
    # Each layer has 93.5 ms fwd mean + 1.0*2.1 std = 95.6 ms; two layers -> 191.2 ms.
    # Must be >> 0 (empirical), not the Roofline TFLOPS projection.
    assert total_gpu_fwd_ms > 10.0, (
        f"GPU forward timing {total_gpu_fwd_ms:.4f} ms is suspiciously low — "
        "empirical timing may have been zeroed out, triggering the wrong fallback."
    )


def test_analytical_fallback_still_triggers_on_genuine_oom(tmp_path: Path) -> None:
    """
    The analytical fallback must still activate when GPU timing is genuinely
    zero (true OOM, not a quality-flag issue).
    """
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    # Simulate a true OOM case: gpu times are zero but cpu times are valid.
    rows = []
    for layer in ["a", "b"]:
        rows.append(
            {
                "layer": layer,
                "gpu_fwd_time_ms_mean": 0.0,   # genuinely zero (OOM)
                "gpu_bwd_time_ms_mean": 0.0,
                "gpu_fwd_time_ms_std": 0.0,
                "gpu_bwd_time_ms_std": 0.0,
                "cpu_fwd_time_ms_mean": 1110.0,
                "cpu_bwd_time_ms_mean": 1110.0,
                "cpu_fwd_time_ms_std": 15.0,
                "cpu_bwd_time_ms_std": 15.0,
                "gpu_fwd_energy_j_mean": 0.0,
                "gpu_bwd_energy_j_mean": 0.0,
                "gpu_fwd_energy_j_std": 0.0,
                "gpu_bwd_energy_j_std": 0.0,
                "cpu_fwd_energy_j_mean": 204.0,
                "cpu_bwd_energy_j_mean": 204.0,
                "cpu_fwd_energy_j_std": 3.0,
                "cpu_bwd_energy_j_std": 3.0,
                "gpu_mem_peak_mb_mean": 0.0,
                "cpu_mem_mb_mean": 3276.0,
                "quality_flag": "ok",
                "n_runs": 3,
                "n_samples": 3,
                "flops": 1e9,
            }
        )
    pd.DataFrame(rows).to_csv(stats, index=False)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json")

    # Must NOT raise — the analytical fallback should rescue the dataset.
    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
        strict_metric_validity=False,  # allow zero metrics for this diagnostic test
    )
    # Nodes should still be populated (fallback succeeded)
    assert data.nodes == ["a", "b"]
