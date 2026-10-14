"""Tests for _resolve_hybrid_overhead_ratio in validation/run_ilp_partition.py.

This is the function that turns the per-run hybrid_overhead_ratio
measurements (src/runner/hybrid_overhead_calibration.py) into the single
correction factor applied to gpu_mem_budget_mb. It must: average only
"ok" measurements, ignore missing/invalid files, and default to 1.0 (no
correction) when nothing is available -- so older profiled data (with no
hybrid_overhead.json at all) behaves EXACTLY as before this feature
existed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "validation" / "run_ilp_partition.py"
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_spec = importlib.util.spec_from_file_location("run_ilp_partition_under_test", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_resolve_hybrid_overhead_ratio = _mod._resolve_hybrid_overhead_ratio


def _write_run(cdir: Path, run_name: str, model: str, payload: dict | None) -> None:
    run_dir = cdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (run_dir / f"{model}_hybrid_overhead.json").write_text(json.dumps(payload))


def test_defaults_to_one_when_no_files_exist(tmp_path):
    cdir = tmp_path / "batch_64"
    cdir.mkdir()
    (cdir / "run_001").mkdir()
    (cdir / "run_002").mkdir()
    ratio = _resolve_hybrid_overhead_ratio([cdir], "resnet50")
    assert ratio == 1.0


def test_averages_ok_measurements_across_runs(tmp_path):
    cdir = tmp_path / "batch_96"
    cdir.mkdir()
    _write_run(cdir, "run_001", "resnet152", {"hybrid_overhead_ratio": 1.4, "status": "ok"})
    _write_run(cdir, "run_002", "resnet152", {"hybrid_overhead_ratio": 1.6, "status": "ok"})
    ratio = _resolve_hybrid_overhead_ratio([cdir], "resnet152")
    assert abs(ratio - 1.5) < 1e-9


def test_ignores_non_ok_status(tmp_path):
    cdir = tmp_path / "batch_96"
    cdir.mkdir()
    _write_run(cdir, "run_001", "resnet152", {"hybrid_overhead_ratio": 1.4, "status": "ok"})
    _write_run(cdir, "run_002", "resnet152", {"hybrid_overhead_ratio": 1.0, "status": "skipped", "reason": "no cuda"})
    ratio = _resolve_hybrid_overhead_ratio([cdir], "resnet152")
    assert abs(ratio - 1.4) < 1e-9


def test_ignores_malformed_json(tmp_path):
    cdir = tmp_path / "batch_96"
    cdir.mkdir()
    run_dir = cdir / "run_001"
    run_dir.mkdir()
    (run_dir / "resnet152_hybrid_overhead.json").write_text("{not valid json")
    ratio = _resolve_hybrid_overhead_ratio([cdir], "resnet152")
    assert ratio == 1.0


def test_ignores_ratio_below_one(tmp_path):
    # A ratio < 1.0 would be physically impossible (reserved can never be
    # less than allocated) -- defend against a corrupt/adversarial file
    # silently SHRINKING the budget instead of growing it.
    cdir = tmp_path / "batch_96"
    cdir.mkdir()
    _write_run(cdir, "run_001", "resnet152", {"hybrid_overhead_ratio": 0.5, "status": "ok"})
    ratio = _resolve_hybrid_overhead_ratio([cdir], "resnet152")
    assert ratio == 1.0


def test_multi_hardware_config_dirs_combined(tmp_path):
    cdir_a = tmp_path / "hostA" / "batch_96"
    cdir_b = tmp_path / "hostB" / "batch_96"
    cdir_a.mkdir(parents=True)
    cdir_b.mkdir(parents=True)
    _write_run(cdir_a, "run_001", "resnet152", {"hybrid_overhead_ratio": 1.2, "status": "ok"})
    _write_run(cdir_b, "run_001", "resnet152", {"hybrid_overhead_ratio": 1.8, "status": "ok"})
    ratio = _resolve_hybrid_overhead_ratio([cdir_a, cdir_b], "resnet152")
    assert abs(ratio - 1.5) < 1e-9
