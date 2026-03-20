from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.runtime.plan_representation import ExecutionPlan, load_execution_plan
from src.runtime.simulator import SimulationConfig, simulate_plan


def _write_metrics(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "layer": "a",
                "gpu_fwd_time_ms_mean": 1.0,
                "gpu_bwd_time_ms_mean": 1.0,
                "cpu_fwd_time_ms_mean": 2.0,
                "cpu_bwd_time_ms_mean": 2.0,
                "gpu_fwd_energy_j_mean": 0.5,
                "gpu_bwd_energy_j_mean": 0.5,
                "cpu_fwd_energy_j_mean": 1.0,
                "cpu_bwd_energy_j_mean": 1.0,
                "gpu_mem_peak_mb_mean": 10.0,
                "cpu_mem_mb_mean": 1.0,
            },
            {
                "layer": "b",
                "gpu_fwd_time_ms_mean": 1.5,
                "gpu_bwd_time_ms_mean": 1.5,
                "cpu_fwd_time_ms_mean": 2.5,
                "cpu_bwd_time_ms_mean": 2.5,
                "gpu_fwd_energy_j_mean": 0.7,
                "gpu_bwd_energy_j_mean": 0.7,
                "cpu_fwd_energy_j_mean": 1.2,
                "cpu_bwd_energy_j_mean": 1.2,
                "gpu_mem_peak_mb_mean": 12.0,
                "cpu_mem_mb_mean": 1.2,
            },
        ]
    )
    df.to_csv(path, index=False)


def test_load_execution_plan_rejects_invalid_device(tmp_path: Path) -> None:
    assignment = tmp_path / "ilp_assignment.csv"
    cut_edges = tmp_path / "ilp_cut_edges.csv"

    pd.DataFrame([{"layer": "a", "device": "TPU"}]).to_csv(assignment, index=False)
    pd.DataFrame([{"src_layer": "a", "dst_layer": "b"}]).to_csv(cut_edges, index=False)

    with pytest.raises(ValueError):
        _ = load_execution_plan(assignment_csv=assignment, cut_edges_csv=cut_edges)


def test_simulate_plan_flags_non_cut_edge(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics_stats.csv"
    _write_metrics(metrics_csv)

    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "GPU"},
        assignment_backward={"a": "GPU", "b": "GPU"},
        cut_edges_forward=[("a", "b")],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    cfg = SimulationConfig(mode="nominal", w_time=1.0, w_energy=0.0, w_transfer=1.0)
    result = simulate_plan(
        plan=plan,
        metrics_stats_csv=metrics_csv,
        graph_edges=[("a", "b")],
        transfer_costs={("a", "b"): 0.3},
        cfg=cfg,
    )

    assert result.status == "invalid"
    assert any("is not a cut" in msg for msg in result.violations)


def test_simulate_plan_nominal_computes_objective(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics_stats.csv"
    _write_metrics(metrics_csv)

    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "CPU"},
        assignment_backward={"a": "GPU", "b": "CPU"},
        cut_edges_forward=[("a", "b")],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    cfg = SimulationConfig(
        mode="nominal",
        w_time=1.0,
        w_energy=0.0,
        w_transfer=1.0,
        gpu_mem_budget_mb=100.0,
        cpu_mem_budget_mb=100.0,
    )

    result = simulate_plan(
        plan=plan,
        metrics_stats_csv=metrics_csv,
        graph_edges=[("a", "b")],
        transfer_costs={("a", "b"): 0.4},
        cfg=cfg,
    )

    # Forward+backward time = a on GPU (1.0+1.0) + b on CPU (2.5+2.5) = 7.0
    # Transfer only in forward = 0.4 ; objective = 7.4
    assert result.status == "ok"
    assert abs(result.total_time_ms - 7.0) < 1e-9
    assert abs(result.total_transfer_ms - 0.4) < 1e-9
    assert abs(result.objective_value - 7.4) < 1e-9


def test_simulate_plan_strict_topology_flags_cycle(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics_stats.csv"
    _write_metrics(metrics_csv)

    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "CPU"},
        assignment_backward={"a": "CPU", "b": "CPU"},
        cut_edges_forward=[("a", "b")],
        cut_edges_backward=[],
        cross_phase_edges=[("a", "a")],
    )

    cfg = SimulationConfig(
        mode="nominal",
        strict_topology=True,
    )

    result = simulate_plan(
        plan=plan,
        metrics_stats_csv=metrics_csv,
        graph_edges=[("a", "b"), ("b", "a")],
        transfer_costs={("a", "b"): 0.1},
        cfg=cfg,
    )

    assert result.status == "invalid"
    assert any("not a DAG" in msg for msg in result.violations)


def test_load_execution_plan_supports_dual_assignment_columns(tmp_path: Path) -> None:
    assignment = tmp_path / "ilp_assignment.csv"
    cut_edges = tmp_path / "ilp_cut_edges.csv"

    pd.DataFrame(
        [
            {"layer": "a", "device_forward": "GPU", "device_backward": "CPU"},
            {"layer": "b", "device_forward": "CPU", "device_backward": "CPU"},
        ]
    ).to_csv(assignment, index=False)
    pd.DataFrame(
        [
            {"src_layer": "a", "dst_layer": "b", "phase": "forward"},
            {"src_layer": "a", "dst_layer": "a", "phase": "cross_phase"},
        ]
    ).to_csv(cut_edges, index=False)

    plan = load_execution_plan(assignment_csv=assignment, cut_edges_csv=cut_edges)

    assert plan.assignment_forward["a"] == "GPU"
    assert plan.assignment_backward["a"] == "CPU"
    assert plan.cut_edges_forward == [("a", "b")]
    assert plan.cross_phase_edges == [("a", "a")]
