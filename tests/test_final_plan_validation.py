"""Task 1.1 (H1): _validate_final_plan_against_real_overhead must never turn a
plan into all-CPU because a measurement was impossible."""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
_spec = importlib.util.spec_from_file_location("rip_under_test", _ROOT / "validation" / "run_ilp_partition.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class _Topo:
    def layer_gpu_weight(self, name):
        return float(name[1:])  # l4 is the heaviest


def _plan(n=5):
    fwd = {f"l{i}": "GPU" for i in range(n)}
    return fwd, dict(fwd)


def _run(monkeypatch, measure, budget=100.0):
    monkeypatch.setattr(mod, "measure_real_peak_for_plan", measure)
    fwd, bwd = _plan()
    return mod._validate_final_plan_against_real_overhead(
        fwd, bwd, None, None, Namespace(gpu_mem_budget_mb=budget), _Topo(), object(), None
    )


def _ok(reserved):
    return {"status": "ok", "real_reserved_mb": reserved, "real_allocated_mb": reserved - 1}


def test_unmeasurable_keeps_plan_and_reports_status(monkeypatch):
    res = _run(monkeypatch, lambda **kw: {"status": "trace_error", "real_reserved_mb": None, "real_allocated_mb": None})
    assert all(d == "GPU" for d in res["forward_assignment"].values())
    assert res["flips_removed"] == 0
    assert res["validation_status"] == "unmeasurable"
    assert res["feasible"] is None


def test_measured_fit_keeps_plan(monkeypatch):
    res = _run(monkeypatch, lambda **kw: _ok(50.0))
    assert res["flips_removed"] == 0 and res["validation_status"] == "ok" and res["feasible"] is True


def test_over_budget_trims_heaviest_first_and_not_more_than_needed(monkeypatch):
    def measure(**kw):
        gpu_layers = sum(1 for d in kw["forward_assignment"].values() if d == "GPU")
        return _ok(30.0 * gpu_layers)  # budget 100 -> at most 3 GPU layers

    res = _run(monkeypatch, measure)
    gpu = {n for n, d in res["forward_assignment"].items() if d == "GPU"}
    assert res["validation_status"] == "trimmed" and res["flips_removed"] == 2
    assert gpu == {"l0", "l1", "l2"}  # l4, l3 (heaviest) were moved to CPU
    assert res["feasible"] is True


def test_nothing_fits_raises_instead_of_all_cpu(monkeypatch):
    with pytest.raises(RuntimeError, match="Refusing to emit an all-CPU plan"):
        _run(monkeypatch, lambda **kw: _ok(500.0))


def _tiny_problem():
    from src.ilp.data_loader import ILPInputData
    from src.ilp.model_builder import ILPConfig

    z = {"a": 0.0, "b": 0.0, "c": 0.0}
    data = ILPInputData(
        nodes=["a", "b", "c"],
        node_cost_gpu_ms={"a": 40.0, "b": 2.0, "c": 2.0},
        node_cost_cpu_ms={"a": 0.0, "b": 12.0, "c": 12.0},
        node_cost_gpu_fwd_ms={"a": 20.0, "b": 1.0, "c": 1.0},
        node_cost_gpu_bwd_ms={"a": 20.0, "b": 1.0, "c": 1.0},
        node_cost_cpu_fwd_ms={"a": 0.0, "b": 6.0, "c": 6.0},
        node_cost_cpu_bwd_ms={"a": 0.0, "b": 6.0, "c": 6.0},
        node_energy_gpu_j=dict(z), node_energy_cpu_j=dict(z),
        node_energy_gpu_fwd_j=dict(z), node_energy_gpu_bwd_j=dict(z),
        node_energy_cpu_fwd_j=dict(z), node_energy_cpu_bwd_j=dict(z),
        node_mem_gpu_mb={"a": 1.0, "b": 1.0, "c": 1.0},
        node_mem_cpu_mb={"a": 1.0, "b": 1.0, "c": 1.0},
        edges=[("a", "b"), ("a", "c")],
        edge_transfer_ms={("a", "b"): 5.0, ("a", "c"): 5.0},
    )
    cfg = ILPConfig(
        w_time=1.0, w_energy=0.0, w_transfer=0.2, w_fragmentation=0.0, w_congestion=3.0,
        congestion_knee_ms=5.0, gpu_mem_budget_mb=100.0, cpu_mem_budget_mb=100.0,
    )
    return data, cfg


def test_recompute_objective_matches_solver_objective():
    from src.ilp.solve import solve_partition_ilp

    data, cfg = _tiny_problem()
    sol = solve_partition_ilp(data, cfg, backend="exhaustive")
    args = Namespace(
        w_time=cfg.w_time, w_energy=cfg.w_energy, w_transfer=cfg.w_transfer,
        w_fragmentation=cfg.w_fragmentation, w_congestion=cfg.w_congestion,
        congestion_knee_ms=cfg.congestion_knee_ms, memory_model=cfg.memory_model,
        peak_activation_overlap=cfg.peak_activation_overlap,
    )
    got = mod._recompute_plan_objective(
        data, args, sol.forward_assignment or sol.assignment, sol.backward_assignment or sol.assignment,
        cfg.enforce_convex_weights,
    )
    assert got == pytest.approx(sol.objective_value, rel=1e-6)

    all_cpu = {n: "CPU" for n in data.nodes}
    worse = mod._recompute_plan_objective(data, args, all_cpu, all_cpu, cfg.enforce_convex_weights)
    assert worse is not None and worse > got
