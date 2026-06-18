from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_relpath: str) -> subprocess.CompletedProcess[str]:
    script = ROOT / script_relpath
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_advanced_thesis_plots_help_has_expected_args() -> None:
    proc = _run_help("validation/generate_advanced_thesis_plots.py")
    assert proc.returncode == 0
    assert "--input_root" in proc.stdout
    assert "--output_dir" in proc.stdout
    assert "--hybrid_csv" in proc.stdout
    assert "--ilp_pareto_csv" not in proc.stdout


def test_plot_custom_user_figures_help_available() -> None:
    proc = _run_help("scripts/plot_custom_user_figures.py")
    assert proc.returncode == 0
    assert "--models" in proc.stdout
    assert "--batches" in proc.stdout
    assert "--reports-root" in proc.stdout


def test_organize_plots_help_available() -> None:
    proc = _run_help("scripts/organize_plots.py")
    assert proc.returncode == 0
    assert "--host" in proc.stdout
    assert "--reports-root" in proc.stdout
