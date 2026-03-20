from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
SMOKE_CFG = PROJECT_ROOT / "data" / "zephyr" / "results_smoke" / "simple_mlp" / "SGD" / "fp32" / "batch_8"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def _python_bin() -> str:
    if PYTHON_BIN.exists():
        return str(PYTHON_BIN)
    return sys.executable


@pytest.mark.integration
def test_phase1_sweep_cli_has_greedy_columns(tmp_path: Path) -> None:
    if not SMOKE_CFG.exists():
        pytest.skip(f"Smoke config directory not found: {SMOKE_CFG}")

    out_csv = tmp_path / "sweep.csv"
    cmd = [
        _python_bin(),
        "validation/sweep_ilp_pareto.py",
        "--config_dirs",
        str(SMOKE_CFG),
        "--model",
        "simple_mlp",
        "--gpu_budgets_mb",
        "4,8,16,32,64",
        "--output_csv",
        str(out_csv),
    ]
    res = _run(cmd)
    assert res.returncode == 0, res.stderr or res.stdout
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert len(df) == 5
    for col in [
        "greedy_status",
        "greedy_objective",
        "greedy_gpu_mem_mb",
        "greedy_layers_gpu",
        "greedy_cut_edges",
    ]:
        assert col in df.columns


@pytest.mark.integration
def test_phase1_ablation_cli_has_four_variants(tmp_path: Path) -> None:
    if not SMOKE_CFG.exists():
        pytest.skip(f"Smoke config directory not found: {SMOKE_CFG}")

    out_csv = tmp_path / "ablation.csv"
    cmd = [
        _python_bin(),
        "validation/run_ilp_ablation_suite.py",
        "--config_dirs",
        str(SMOKE_CFG),
        "--model",
        "simple_mlp",
        "--gpu_budgets_mb",
        "4,8,16,32,64",
        "--output_csv",
        str(out_csv),
    ]
    res = _run(cmd)
    assert res.returncode == 0, res.stderr or res.stdout
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert len(df) == 20
    assert sorted(df["variant"].unique().tolist()) == [
        "full_model",
        "no_robustification",
        "no_topology",
        "no_transfer_edges",
    ]


@pytest.mark.integration
def test_phase1_sensitivity_cli_has_delta_columns(tmp_path: Path) -> None:
    if not SMOKE_CFG.exists():
        pytest.skip(f"Smoke config directory not found: {SMOKE_CFG}")

    out_csv = tmp_path / "sensitivity.csv"
    cmd = [
        _python_bin(),
        "validation/run_ilp_sensitivity.py",
        "--config_dirs",
        str(SMOKE_CFG),
        "--model",
        "simple_mlp",
        "--gpu_budgets_mb",
        "4,8,16,32,64",
        "--output_csv",
        str(out_csv),
    ]
    res = _run(cmd)
    assert res.returncode == 0, res.stderr or res.stdout
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert len(df) == 55
    assert sorted(df["param_name"].unique().tolist()) == ["baseline", "k_sigma", "w_transfer"]
    for col in ["baseline_objective", "delta_abs", "delta_pct"]:
        assert col in df.columns


@pytest.mark.integration
def test_phase1_metrics_quality_flags_present(tmp_path: Path) -> None:
    if not SMOKE_CFG.exists():
        pytest.skip(f"Smoke config directory not found: {SMOKE_CFG}")

    out_csv = tmp_path / "metrics_stats.csv"
    cmd = [
        _python_bin(),
        "validation/aggregate_metrics_stats.py",
        "--input_dir",
        str(SMOKE_CFG),
        "--output_csv",
        str(out_csv),
    ]
    res = _run(cmd)
    assert res.returncode == 0, res.stderr or res.stdout
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    for col in [
        "n_runs",
        "n_samples",
        "quality_flag",
        "max_cv_key_metrics",
        "gpu_fwd_time_ms_cv",
        "gpu_bwd_time_ms_cv",
        "cpu_fwd_time_ms_cv",
        "cpu_bwd_time_ms_cv",
    ]:
        assert col in df.columns
