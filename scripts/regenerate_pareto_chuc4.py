#!/usr/bin/env python3
"""
Master Pareto sweep runner for chuc-4 doctoral_full campaign.
Iterates over all 378 configurations and regenerates pareto_sweep.csv
using the corrected ILP data loader (empirical GPU energy preserved).
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE = Path("data/chuc-4/thesis_runs/job_adhoc_20260624_044532/doctoral_full")
PYTHON = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
SWEEP_SCRIPT = "validation/sweep_ilp_pareto.py"

# GPU budgets used in the original chuc-4 campaign (20480 and 32768 MB)
GPU_BUDGETS = "20480,32768"
CPU_MEM_BUDGET = "1e18"

# Flags matching the original thesis campaign
EXTRA_FLAGS = [
    "--regime", "diagnostic",
    "--gpu_budgets_mb", GPU_BUDGETS,
    "--cpu_mem_budget_mb", CPU_MEM_BUDGET,
    "--allow_low_quality_stats",          # now a no-op thanks to the fix, but keep for compat
    "--allow_transfer_calibration_fallback",
    "--allow_fallback_graph_trace",
    "--w_energy", "1.0",
    "--w_time", "0.0",
    "--w_transfer", "0.0",
    "--backend", "auto",
]

def _discover_configs():
    configs = []
    for stats_csv in sorted(BASE.rglob("*_metrics_stats.csv")):
        parts = stats_csv.relative_to(BASE).parts
        if len(parts) < 5:
            continue
        seed, model, optimizer, precision, batch = parts[:5]
        config_dir = stats_csv.parent
        configs.append({
            "seed": seed, "model": model, "optimizer": optimizer,
            "precision": precision, "batch": batch,
            "config_dir": config_dir,
        })
    return configs


def _run_one(cfg: dict) -> tuple[str, bool, str]:
    label = f"{cfg['seed']}/{cfg['model']}/{cfg['optimizer']}/{cfg['precision']}/{cfg['batch']}"
    cmd = [
        PYTHON, SWEEP_SCRIPT,
        "--config_dir", str(cfg["config_dir"]),
        "--model", cfg["model"],
    ] + EXTRA_FLAGS

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return label, False, result.stderr[-800:]
    return label, True, ""


def main():
    configs = _discover_configs()
    print(f"[INFO] Found {len(configs)} configurations to sweep.")

    ok = 0
    failed = []
    t0 = time.perf_counter()

    # Use 4 parallel workers — ILP is CPU-bound but each solve is fast (<10s)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            label, success, err = fut.result()
            if success:
                ok += 1
                print(f"[{i:3d}/{len(configs)}] OK  {label}")
            else:
                failed.append((label, err))
                print(f"[{i:3d}/{len(configs)}] ERR {label}")
                print(f"         {err[:200]}")

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] {ok}/{len(configs)} succeeded in {elapsed:.1f}s")
    if failed:
        print(f"[WARN] {len(failed)} failed:")
        for lbl, _ in failed:
            print(f"  - {lbl}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
