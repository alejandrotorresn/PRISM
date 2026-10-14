from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ilp.backward_meta_model import apply_backward_meta_model, train_backward_meta_model
from src.ilp.data_loader import ILPInputData, load_ilp_inputs
from src.ilp.model_builder import ILPConfig, build_problem_data_dual
from src.ilp.solve import ILPSolution, refine_solution_hierarchical_local, solve_partition_ilp


def _write_graph(path: Path) -> None:
    pd.DataFrame(
        [
            {"producer_name": "a", "consumer_name": "b"},
        ]
    ).to_csv(path, index=False)


def _write_transfer(path: Path) -> None:
    pd.DataFrame(
        [
            {"producer_name": "a", "consumer_name": "b", "transfer_sym_ms": 0.2},
        ]
    ).to_csv(path, index=False)


def _write_meta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "transfer_calibration_source": "measured",
                "graph_trace_source": "torch_fx",
            }
        ),
        encoding="utf-8",
    )


def _write_stats(path: Path) -> None:
    rows = []
    for i, layer in enumerate(["a", "b", "c", "d"]):
        rows.append(
            {
                "layer": layer,
                "gpu_fwd_time_ms_mean": 1.0 + i,
                "gpu_bwd_time_ms_mean": 2.0 + (0.5 * i),
                "gpu_fwd_time_ms_std": 0.1,
                "gpu_bwd_time_ms_std": 0.2,
                "cpu_fwd_time_ms_mean": 3.0 + i,
                "cpu_bwd_time_ms_mean": 4.0 + i,
                "cpu_fwd_time_ms_std": 0.3,
                "cpu_bwd_time_ms_std": 0.4,
                "gpu_fwd_energy_j_mean": 0.5 + (0.1 * i),
                "gpu_bwd_energy_j_mean": 0.7 + (0.1 * i),
                "gpu_fwd_energy_j_std": 0.01,
                "gpu_bwd_energy_j_std": 0.02,
                "cpu_fwd_energy_j_mean": 0.9 + (0.1 * i),
                "cpu_bwd_energy_j_mean": 1.2 + (0.1 * i),
                "cpu_fwd_energy_j_std": 0.02,
                "cpu_bwd_energy_j_std": 0.03,
                "gpu_mem_peak_mb_mean": 10.0 + i,
                "cpu_mem_mb_mean": 2.0 + i,
                "quality_flag": "ok",
                "n_runs": 5,
                "n_samples": 10,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_ilp_inputs_supports_independent_time_energy_uncertainty(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
        k_sigma=0.0,
        k_sigma_time=2.0,
        k_sigma_energy=0.0,
    )

    # Time should include robust margin while energy should stay nominal.
    assert data.node_cost_gpu_fwd_ms["a"] == 1.0 + (2.0 * 0.1)
    assert data.node_energy_gpu_fwd_j["a"] == 0.5


def test_backward_meta_model_train_and_apply_updates_time_and_energy(tmp_path: Path) -> None:
    stats = tmp_path / "metrics_stats.csv"
    graph = tmp_path / "graph_edges.csv"
    transfer = tmp_path / "transfer_edges.csv"

    _write_stats(stats)
    _write_graph(graph)
    _write_transfer(transfer)
    _write_meta(tmp_path / "graph_edges_meta.json")

    model_path = tmp_path / "backward_meta_model.json"
    payload = train_backward_meta_model(
        metrics_stats_csv=stats,
        output_json=model_path,
        validation_ratio=0.25,
        ridge_lambda=1e-6,
        seed=7,
    )

    assert payload["schema"] == "prism.backward_meta_model.v1"
    assert set(payload["targets"].keys()) == {"gpu_time", "cpu_time", "gpu_energy", "cpu_energy"}

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats),
        graph_edges_csv=str(graph),
        transfer_edges_csv=str(transfer),
        k_sigma=0.0,
    )
    before_bwd_gpu_time = data.node_cost_gpu_bwd_ms["a"]
    before_bwd_gpu_energy = data.node_energy_gpu_bwd_j["a"]

    updated = apply_backward_meta_model(data, payload, blend=1.0)

    assert updated.node_cost_gpu_bwd_ms["a"] >= 0.0
    assert updated.node_energy_gpu_bwd_j["a"] >= 0.0
    assert (
        updated.node_cost_gpu_bwd_ms["a"] != before_bwd_gpu_time
        or updated.node_energy_gpu_bwd_j["a"] != before_bwd_gpu_energy
    )


def test_hierarchical_local_refinement_respects_budget_and_improves_objective() -> None:
    data = ILPInputData(
        nodes=["a", "b"],
        node_cost_gpu_ms={"a": 11.0, "b": 11.0},
        node_cost_cpu_ms={"a": 3.0, "b": 3.0},
        node_cost_gpu_fwd_ms={"a": 1.0, "b": 1.0},
        node_cost_gpu_bwd_ms={"a": 10.0, "b": 10.0},
        node_cost_cpu_fwd_ms={"a": 2.0, "b": 2.0},
        node_cost_cpu_bwd_ms={"a": 1.0, "b": 1.0},
        node_energy_gpu_j={"a": 1.0, "b": 1.0},
        node_energy_cpu_j={"a": 1.0, "b": 1.0},
        node_energy_gpu_fwd_j={"a": 0.5, "b": 0.5},
        node_energy_gpu_bwd_j={"a": 0.5, "b": 0.5},
        node_energy_cpu_fwd_j={"a": 0.5, "b": 0.5},
        node_energy_cpu_bwd_j={"a": 0.5, "b": 0.5},
        node_mem_gpu_mb={"a": 1.0, "b": 1.0},
        node_mem_cpu_mb={"a": 1.0, "b": 1.0},
        edges=[],
        edge_transfer_ms={},
    )
    cfg = ILPConfig(
        w_time=1.0,
        w_energy=0.0,
        w_transfer=1.0,
        w_fragmentation=0.0,
        gpu_mem_budget_mb=10.0,
        cpu_mem_budget_mb=10.0,
    )

    base = ILPSolution(
        status="optimal",
        backend="manual",
        objective_value=1_000.0,
        assignment={"a": "GPU", "b": "GPU"},
        gpu_mem_used_mb=4.0,
        cpu_mem_used_mb=0.0,
        cut_edges=[],
        forward_assignment={"a": "GPU", "b": "GPU"},
        backward_assignment={"a": "GPU", "b": "GPU"},
        backward_cut_edges=[],
        cross_phase_edges=[],
    )

    refined = refine_solution_hierarchical_local(
        data=data,
        cfg=cfg,
        base_solution=base,
        max_assignment_changes=1,
    )

    assert refined.objective_value < base.objective_value
    # With one local change, at most one backward layer should move to CPU in this setup.
    moved = sum(1 for n in data.nodes if refined.backward_assignment[n] != base.backward_assignment[n])
    assert moved <= 1


def test_explicit_congestion_term_discourages_branched_frontier_cuts() -> None:
    data = ILPInputData(
        nodes=["a", "b", "c"],
        node_cost_gpu_ms={"a": 40.0, "b": 2.0, "c": 2.0},
        node_cost_cpu_ms={"a": 0.0, "b": 12.0, "c": 12.0},
        node_cost_gpu_fwd_ms={"a": 20.0, "b": 1.0, "c": 1.0},
        node_cost_gpu_bwd_ms={"a": 20.0, "b": 1.0, "c": 1.0},
        node_cost_cpu_fwd_ms={"a": 0.0, "b": 6.0, "c": 6.0},
        node_cost_cpu_bwd_ms={"a": 0.0, "b": 6.0, "c": 6.0},
        node_energy_gpu_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_energy_cpu_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_energy_gpu_fwd_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_energy_gpu_bwd_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_energy_cpu_fwd_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_energy_cpu_bwd_j={"a": 0.0, "b": 0.0, "c": 0.0},
        node_mem_gpu_mb={"a": 1.0, "b": 1.0, "c": 1.0},
        node_mem_cpu_mb={"a": 1.0, "b": 1.0, "c": 1.0},
        edges=[("a", "b"), ("a", "c")],
        edge_transfer_ms={("a", "b"): 5.0, ("a", "c"): 5.0},
    )

    cfg_no_congestion = ILPConfig(
        w_time=1.0,
        w_energy=0.0,
        w_transfer=0.2,
        w_fragmentation=0.0,
        w_congestion=0.0,
        congestion_knee_ms=5.0,
        gpu_mem_budget_mb=100.0,
        cpu_mem_budget_mb=100.0,
    )
    sol_no_congestion = solve_partition_ilp(data, cfg_no_congestion, backend="exhaustive")

    cfg_with_congestion = ILPConfig(
        w_time=1.0,
        w_energy=0.0,
        w_transfer=0.2,
        w_fragmentation=0.0,
        w_congestion=3.0,
        congestion_knee_ms=5.0,
        gpu_mem_budget_mb=100.0,
        cpu_mem_budget_mb=100.0,
    )
    sol_with_congestion = solve_partition_ilp(data, cfg_with_congestion, backend="exhaustive")

    assert sol_no_congestion.forward_assignment["b"] == "GPU"
    assert sol_no_congestion.forward_assignment["c"] == "GPU"
    assert len(sol_no_congestion.cut_edges) == 2

    moved_to_cpu = sum(
        1
        for node in ("b", "c")
        if sol_with_congestion.forward_assignment[node] == "CPU"
    )
    assert moved_to_cpu >= 1
    assert len(sol_with_congestion.cut_edges) < len(sol_no_congestion.cut_edges)


def test_cross_phase_cost_is_direction_aware() -> None:
    """Regression test for the dual-placement cost fix.

    src/runtime/hybrid_executor.py's _run_layer_with_dual_placement recomputes
    the ENTIRE forward pass on the backward device before computing
    gradients there. That recompute's real cost depends on which device it
    runs on: cheap if the backward device is fast (GPU), expensive if it is
    slow (CPU) -- this project's own profiling shows GPU running ~15-17x
    faster than CPU. Before this fix, both directions were priced with the
    same small, undirected node_time_io_ms proxy, so the ILP treated
    "forward=GPU, backward=CPU" (expensive: the recompute lands on the slow
    device) as being just as cheap as "forward=CPU, backward=GPU" (genuinely
    cheap: the recompute lands on the fast device).

    Checks the two directional cross-phase cost coefficients
    (build_problem_data_dual's objective_cross_phase_cpu_to_gpu /
    _gpu_to_cpu) directly, rather than relying on the solver to pick a
    cross-phase placement via a memory-constrained scenario: fixing the
    real cross-phase memory-accounting bug found in the same investigation
    (see src/ilp/topology_memory.py's module docstring -- activation memory
    must belong to whichever device runs backward, not only when forward
    and backward devices match) means a cross-phase placement never needs
    LESS GPU memory than fully retaining on GPU, so a single-node,
    memory-constrained scenario can no longer force the solver into a
    cross-phase choice to exercise this cost. Asserting on the coefficients
    themselves tests the exact mechanism in isolation instead.

      - GPU fwd/bwd: 1ms each -- fast.
      - CPU fwd/bwd: 100ms each -- ~100x slower (deliberately exaggerated
        here for a clean, unambiguous assertion; this project's own real
        CPU/GPU gap is ~15-17x, same qualitative effect).
      - fwd=CPU, bwd=GPU recompute cost: 1 (GPU fwd replay) + 0.1 (io) = 1.1ms  <- cheap, recompute lands on GPU
      - fwd=GPU, bwd=CPU recompute cost: 100 (CPU fwd replay) + 0.1 (io) = 100.1ms <- expensive, recompute lands on CPU

    Under the old undirected model both directions were priced with the
    same small, undirected node_time_io_ms proxy (~0.1ms each), making them
    look equally (and wrongly) cheap regardless of which device the
    recompute actually lands on.
    """
    data = ILPInputData(
        nodes=["a"],
        node_cost_gpu_ms={"a": 2.0},
        node_cost_cpu_ms={"a": 200.0},
        node_cost_gpu_fwd_ms={"a": 1.0},
        node_cost_gpu_bwd_ms={"a": 1.0},
        node_cost_cpu_fwd_ms={"a": 100.0},
        node_cost_cpu_bwd_ms={"a": 100.0},
        node_energy_gpu_j={"a": 0.0},
        node_energy_cpu_j={"a": 0.0},
        node_energy_gpu_fwd_j={"a": 0.0},
        node_energy_gpu_bwd_j={"a": 0.0},
        node_energy_cpu_fwd_j={"a": 0.0},
        node_energy_cpu_bwd_j={"a": 0.0},
        node_mem_gpu_mb={"a": 800.0},
        node_mem_cpu_mb={"a": 10.0},
        node_mem_activation_mb={"a": 400.0},
        node_time_io_ms={"a": 0.1},
        node_energy_io_j={"a": 0.0},
        edges=[],
        edge_transfer_ms={},
    )
    cfg = ILPConfig(
        w_time=1.0,
        w_energy=0.0,
        w_transfer=1.0,
        w_fragmentation=0.0,
        gpu_mem_budget_mb=1_000_000.0,
        cpu_mem_budget_mb=1_000_000.0,
        memory_model="topological",
    )

    problem = build_problem_data_dual(data, cfg)
    assert problem.objective_cross_phase_cpu_to_gpu["a"] == pytest.approx(1.1, abs=0.05)
    assert problem.objective_cross_phase_gpu_to_cpu["a"] == pytest.approx(100.1, abs=0.05)
    assert problem.objective_cross_phase_cpu_to_gpu["a"] < problem.objective_cross_phase_gpu_to_cpu["a"]
