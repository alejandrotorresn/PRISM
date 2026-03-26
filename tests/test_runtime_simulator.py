from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.runtime.plan_representation import ExecutionPlan, load_execution_plan, load_graph_edges, load_transfer_costs
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


def test_simulate_plan_uses_phase_specific_assignment_costs(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics_stats.csv"
    _write_metrics(metrics_csv)

    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "CPU"},
        assignment_backward={"a": "CPU", "b": "GPU"},
        cut_edges_forward=[("a", "b")],
        cut_edges_backward=[("a", "b")],
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

    # Forward: a GPU fwd=1.0, b CPU fwd=2.5 -> 3.5
    # Backward: a CPU bwd=2.0, b GPU bwd=1.5 -> 3.5
    # Transfer: forward cut + backward cut = 0.8
    # Objective: 7.0 + 0.8 = 7.8
    assert result.status == "ok"
    assert abs(result.total_time_ms - 7.0) < 1e-9
    assert abs(result.total_transfer_ms - 0.8) < 1e-9
    assert abs(result.objective_value - 7.8) < 1e-9


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


def test_load_graph_edges_uses_node_order_to_avoid_name_collapsing_cycles(tmp_path: Path) -> None:
    graph_edges = tmp_path / "graph_edges.csv"
    pd.DataFrame(
        [
            {"src_id": "n1", "dst_id": "n2", "producer_name": "x", "consumer_name": "relu"},
            {"src_id": "n2", "dst_id": "n3", "producer_name": "relu", "consumer_name": "y"},
            {"src_id": "n4", "dst_id": "n5", "producer_name": "z", "consumer_name": "relu"},
            # Same module name reused later; naive name-collapse would introduce relu->x and x->relu.
            {"src_id": "n6", "dst_id": "n7", "producer_name": "relu", "consumer_name": "x"},
        ]
    ).to_csv(graph_edges, index=False)

    edges = load_graph_edges(graph_edges)

    assert ("x", "relu") in edges
    assert ("relu", "x") not in edges


def test_load_graph_edges_contracts_to_nearest_measured_successors(tmp_path: Path) -> None:
    graph_edges = tmp_path / "graph_edges.csv"
    transfer_edges = tmp_path / "transfer_edges.csv"

    pd.DataFrame(
        [
            {"src_id": "n0", "dst_id": "n1", "producer_name": "x", "consumer_name": "a"},
            {"src_id": "n1", "dst_id": "n2", "producer_name": "a", "consumer_name": "hidden_1"},
            {"src_id": "n2", "dst_id": "n3", "producer_name": "hidden_1", "consumer_name": "b"},
            {"src_id": "n3", "dst_id": "n4", "producer_name": "b", "consumer_name": "hidden_2"},
            {"src_id": "n4", "dst_id": "n5", "producer_name": "hidden_2", "consumer_name": "c"},
        ]
    ).to_csv(graph_edges, index=False)
    pd.DataFrame(
        [
            {"src_id": "n0", "dst_id": "n1", "producer_name": "x", "consumer_name": "a", "transfer_sym_ms": 0.1},
            {"src_id": "n1", "dst_id": "n2", "producer_name": "a", "consumer_name": "hidden_1", "transfer_sym_ms": 0.7},
            {"src_id": "n2", "dst_id": "n3", "producer_name": "hidden_1", "consumer_name": "b", "transfer_sym_ms": 0.5},
            {"src_id": "n3", "dst_id": "n4", "producer_name": "b", "consumer_name": "hidden_2", "transfer_sym_ms": 0.4},
            {"src_id": "n4", "dst_id": "n5", "producer_name": "hidden_2", "consumer_name": "c", "transfer_sym_ms": 0.3},
        ]
    ).to_csv(transfer_edges, index=False)

    edges = load_graph_edges(
        graph_edges,
        transfer_edges_csv=transfer_edges,
        measured_layers={"a", "b", "c"},
    )
    transfer_costs = load_transfer_costs(
        transfer_edges,
        graph_edges_csv=graph_edges,
        measured_layers={"a", "b", "c"},
    )

    assert edges == [("a", "b"), ("b", "c")]
    assert transfer_costs[("a", "b")] == 0.7
    assert transfer_costs[("b", "c")] == 0.4


def test_simulate_plan_sink_nodes_do_not_produce_isolated_warning(tmp_path: Path) -> None:
    """Sink nodes (real incoming edges, no outgoing edges in the assignment subgraph)
    must not be reported as isolated.  This guards against the false-positive that
    arose when Kahn's BFS decremented all non-cycle indegrees to 0, making sink nodes
    look indistinguishable from truly unconnected nodes."""
    metrics_csv = tmp_path / "metrics_stats.csv"
    pd.DataFrame(
        [
            {
                "layer": name,
                "gpu_fwd_time_ms_mean": 1.0,
                "gpu_bwd_time_ms_mean": 0.5,
                "cpu_fwd_time_ms_mean": 2.0,
                "cpu_bwd_time_ms_mean": 1.0,
                "gpu_fwd_energy_j_mean": 0.1,
                "gpu_bwd_energy_j_mean": 0.05,
                "cpu_fwd_energy_j_mean": 0.2,
                "cpu_bwd_energy_j_mean": 0.1,
                "gpu_mem_peak_mb_mean": 10.0,
                "cpu_mem_mb_mean": 5.0,
            }
            for name in ("a", "b", "c")
        ]
    ).to_csv(metrics_csv, index=False)

    # Linear chain a -> b -> c.  Node 'c' is a sink: it has an incoming edge from 'b'
    # but no outgoing edges inside the assignment set.
    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "GPU", "c": "CPU"},
        assignment_backward={"a": "GPU", "b": "GPU", "c": "CPU"},
        cut_edges_forward=[("b", "c")],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    cfg = SimulationConfig(
        mode="nominal",
        w_time=1.0,
        w_energy=0.0,
        w_transfer=1.0,
        strict_topology=True,
    )
    result = simulate_plan(
        plan=plan,
        metrics_stats_csv=metrics_csv,
        graph_edges=[("a", "b"), ("b", "c")],
        transfer_costs={("b", "c"): 0.5},
        cfg=cfg,
    )

    isolated_warnings = [w for w in result.warnings if "isolated" in w.lower()]
    assert isolated_warnings == [], (
        f"Sink node 'c' was falsely reported as isolated: {isolated_warnings}"
    )


def test_simulate_plan_combined_gpu_peak_triggers_violation(tmp_path: Path) -> None:
    """Each phase's GPU memory individually fits within budget, but the combined
    peak (forward activations retained during backward) must trigger a violation."""
    metrics_csv = tmp_path / "metrics_stats.csv"
    _write_metrics(metrics_csv)  # a: 10 MB GPU peak, b: 12 MB GPU peak

    # Forward: a on GPU (10 MB), backward: b on GPU (12 MB).
    # Per-phase: max(10, 12) = 12 MB — both individually <= 15 MB budget.
    # Combined peak: 10 + 12 = 22 MB > 15 MB budget -> must be a violation.
    plan = ExecutionPlan(
        assignment_forward={"a": "GPU", "b": "CPU"},
        assignment_backward={"a": "CPU", "b": "GPU"},
        cut_edges_forward=[("a", "b")],
        cut_edges_backward=[("a", "b")],
        cross_phase_edges=[],
    )

    cfg = SimulationConfig(
        mode="nominal",
        w_time=1.0,
        w_energy=0.0,
        w_transfer=1.0,
        gpu_mem_budget_mb=15.0,  # > each individual phase (10, 12) but < combined (22)
        cpu_mem_budget_mb=100.0,
    )

    result = simulate_plan(
        plan=plan,
        metrics_stats_csv=metrics_csv,
        graph_edges=[("a", "b")],
        transfer_costs={("a", "b"): 0.4},
        cfg=cfg,
    )

    assert result.status == "invalid", "Combined peak exceeds budget; plan should be invalid"
    assert any("GPU memory violation" in msg for msg in result.violations)
    assert result.gpu_mem_used_mb > cfg.gpu_mem_budget_mb
