"""Tests that run_all_gpu_training/run_prism_training actually populate the
new resource-usage fields (peak GPU/CPU memory, avg power, total energy) on
ConvergenceRunResult and its per-step TrainStepRecord entries, and that
run_convergence_study.py writes them to disk (train_loss.csv/summary.json).

These are GPU-gated: the whole point of peak_gpu_mem_mb/avg_power_w is to
measure something real, so there is no meaningful CPU-only version of this
test -- it skips cleanly when CUDA is unavailable instead of asserting
zeros, which would pass trivially without checking anything real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_CONVERGENCE_DIR = Path(__file__).resolve().parents[1]
if str(_CONVERGENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVERGENCE_DIR))

_REPO_ROOT = _CONVERGENCE_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _monotonic_nondecreasing(values: list[float]) -> bool:
    return all(b >= a - 1e-6 for a, b in zip(values, values[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_run_all_gpu_training_reports_resource_metrics(tmp_path):
    from all_gpu_loop import run_all_gpu_training

    result = run_all_gpu_training(
        model_name="resnet50",
        precision="fp32",
        batch_size=8,
        datasets_root=_REPO_ROOT / "datasets",
        epochs=1,
        lr=0.01,
        seed=42,
        optimizer_name="sgd",
        max_steps=2,
    )
    assert result.status == "ok"
    assert len(result.train_steps) == 2

    # Peaks must be real (non-zero) and monotonically non-decreasing across
    # steps, since torch.cuda.max_memory_allocated()/ru_maxrss are both
    # running peaks, never reset mid-run.
    gpu_peaks = [r.peak_gpu_mem_mb for r in result.train_steps]
    cpu_peaks = [r.peak_cpu_mem_mb for r in result.train_steps]
    assert all(v > 0 for v in gpu_peaks)
    assert all(v > 0 for v in cpu_peaks)
    assert _monotonic_nondecreasing(gpu_peaks)
    assert _monotonic_nondecreasing(cpu_peaks)

    assert result.peak_gpu_mem_mb == gpu_peaks[-1]
    assert result.peak_cpu_mem_mb == cpu_peaks[-1]
    assert result.avg_power_w >= 0.0
    assert result.total_energy_j >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_run_prism_training_reports_resource_metrics():
    from prism_loop import run_prism_training

    config_dir = _REPO_ROOT / "data" / "zephyr" / "results_thesis_mode" / "custom" / "resnet50" / "SGD" / "fp32" / "batch_64"
    solution_dir = config_dir / "ilp_solution"
    assignment_csv = solution_dir / "ilp_assignment.csv"
    cut_edges_csv = solution_dir / "ilp_cut_edges.csv"
    if not assignment_csv.exists():
        pytest.skip(f"no ILP solution found at {solution_dir} on this machine")

    result = run_prism_training(
        model_name="resnet50",
        precision="fp32",
        batch_size=64,
        datasets_root=_REPO_ROOT / "datasets",
        assignment_csv=assignment_csv,
        cut_edges_csv=cut_edges_csv,
        epochs=1,
        lr=0.01,
        seed=42,
        optimizer_name="sgd",
        max_steps=2,
    )
    assert result.status == "ok"
    assert len(result.train_steps) == 2

    gpu_peaks = [r.peak_gpu_mem_mb for r in result.train_steps]
    cpu_peaks = [r.peak_cpu_mem_mb for r in result.train_steps]
    assert all(v > 0 for v in gpu_peaks)
    assert all(v > 0 for v in cpu_peaks)
    assert _monotonic_nondecreasing(gpu_peaks)
    assert _monotonic_nondecreasing(cpu_peaks)
    assert result.avg_power_w >= 0.0
    assert result.total_energy_j >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_write_result_persists_resource_metrics_to_disk(tmp_path):
    import argparse

    from all_gpu_loop import run_all_gpu_training
    from run_convergence_study import _write_result

    result = run_all_gpu_training(
        model_name="resnet50",
        precision="fp32",
        batch_size=8,
        datasets_root=_REPO_ROOT / "datasets",
        epochs=1,
        lr=0.01,
        seed=42,
        optimizer_name="sgd",
        max_steps=2,
    )
    assert result.status == "ok"

    args = argparse.Namespace(mode="all_gpu", regime="A", epochs=1, lr=0.01, optimizer="sgd", seed=42)
    _write_result(result, tmp_path, args)

    train_csv_lines = (tmp_path / "train_loss.csv").read_text().strip().splitlines()
    header = train_csv_lines[0].split(",")
    assert "peak_gpu_mem_mb" in header
    assert "peak_cpu_mem_mb" in header
    data_row = train_csv_lines[1].split(",")
    assert float(data_row[header.index("peak_gpu_mem_mb")]) > 0.0
    assert float(data_row[header.index("peak_cpu_mem_mb")]) > 0.0

    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["peak_gpu_mem_mb"] > 0.0
    assert summary["peak_cpu_mem_mb"] > 0.0
    assert summary["avg_power_w"] >= 0.0
    assert summary["total_energy_j"] >= 0.0
    assert summary["total_wall_time_s"] == pytest.approx(
        summary["setup_wall_time_s"] + summary["total_train_wall_time_s"]
    )
