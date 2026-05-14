from types import SimpleNamespace

import torch.nn as nn

from src.runner.training_profiler import TrainingProfiler, _piecewise_transfer_ms


def test_piecewise_transfer_ms_penalizes_large_messages() -> None:
    small = _piecewise_transfer_ms(
        tensor_mb=16.0,
        alpha_ms=0.05,
        beta_nominal_mb_per_ms=12.0,
        beta_congested_mb_per_ms=6.0,
        congestion_knee_mb=32.0,
    )
    large = _piecewise_transfer_ms(
        tensor_mb=256.0,
        alpha_ms=0.05,
        beta_nominal_mb_per_ms=12.0,
        beta_congested_mb_per_ms=6.0,
        congestion_knee_mb=32.0,
    )

    assert large > small


def test_edge_transfer_costs_increase_under_branch_pressure() -> None:
    profiler = TrainingProfiler(
        model=nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
        model_name="toy",
        args=SimpleNamespace(no_gpu=True, gpu_id=0, rapl=False),
    )

    graph_edges = [
        {"src_id": "n0", "dst_id": "n1", "producer_name": "a", "consumer_name": "b", "tensor_mb": 64.0},
        {"src_id": "n0", "dst_id": "n2", "producer_name": "a", "consumer_name": "c", "tensor_mb": 64.0},
        {"src_id": "n3", "dst_id": "n4", "producer_name": "d", "consumer_name": "e", "tensor_mb": 64.0},
    ]
    pci_stats = {
        "alpha_h2d": 0.05,
        "beta_h2d": 12.0,
        "beta_h2d_congested": 6.0,
        "congestion_knee_h2d_mb": 32.0,
        "alpha_d2h": 0.05,
        "beta_d2h": 12.0,
        "beta_d2h_congested": 6.0,
        "congestion_knee_d2h_mb": 32.0,
        "overlap_ratio_sigma": 0.0,
    }

    edge_rows, _, _ = profiler._build_edge_transfer_costs(graph_edges, pci_stats)
    branched = next(row for row in edge_rows if row["producer_name"] == "a" and row["consumer_name"] == "b")
    isolated = next(row for row in edge_rows if row["producer_name"] == "d" and row["consumer_name"] == "e")

    assert branched["branch_pressure"] > 0.0
    assert isolated["branch_pressure"] == 0.0
    assert branched["transfer_sym_ms"] > isolated["transfer_sym_ms"]
