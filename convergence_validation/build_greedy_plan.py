#!/usr/bin/env python3
"""Builds a greedy-heuristic CPU/GPU partition plan for one (model, batch,
budget) configuration, in the exact CSV schema an ILP solution already uses.

Why this exists: the thesis already implements a greedy device-partitioning
heuristic (`validation/sweep_ilp_pareto.py::_greedy_bits`) and reports its
aggregate cost (latency/energy) next to the ILP's in the Pareto sweep CSVs
-- but it never persists greedy's per-layer assignment to disk, because the
sweep only needed summary numbers for the Chapter 6 plots. Regime-B
convergence validation needs an actual per-layer plan to execute real
training under, so this script computes the SAME greedy bits the sweep
already reports and writes them out as a plan directory
(`greedy_solution_budget_<N>/{ilp_assignment.csv,ilp_cut_edges.csv}`) that
`run_convergence_study.py --mode greedy_plan` can read with the exact same
code path already used for `--mode ilp_plan` (device_forward ==
device_backward for every layer, since the greedy heuristic -- unlike the
ILP -- picks one device per node for both passes).

This is read-only reuse: it imports `_greedy_bits` and
`_build_effective_from_dual` from validation/sweep_ilp_pareto.py (loaded by
file path, so it works regardless of the caller's working directory) and
`load_ilp_inputs`/`ILPConfig` from src/ilp -- none of those files are
modified.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

load_ilp_inputs = importlib.import_module("ilp.data_loader").load_ilp_inputs
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig


def _load_sweep_module():
    """Loads validation/sweep_ilp_pareto.py by file path rather than by
    package name, so this works no matter the caller's cwd and never
    collides with an unrelated `validation` import elsewhere on sys.path
    (same reasoning as convergence_validation/datasets.py's fix for the
    repo-root data/ vs src/data/ collision)."""
    path = _REPO_ROOT / "validation" / "sweep_ilp_pareto.py"
    spec = importlib.util.spec_from_file_location("_sweep_ilp_pareto_for_greedy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _budget_slug(budget_mb: float) -> str:
    """Matches scripts/run_thesis_mode.sh's own convention for naming
    ilp_solution_pareto_best_budget_<slug>/ directories, e.g. 7369.0 ->
    "7369p0" -- so greedy's output directory name lines up with the ILP's
    for the same budget."""
    return f"{budget_mb:.1f}".replace(".", "p")


def _find_existing_pareto_budget(config_dir: Path) -> float:
    candidates = sorted(config_dir.glob("ilp_solution_pareto_best_budget_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No ilp_solution_pareto_best_budget_* directory under {config_dir} to "
            "read the regime-B GPU budget from. Pass --gpu_mem_budget_mb explicitly, "
            "or run the pareto sweep for this config first."
        )
    suffix = candidates[0].name.removeprefix("ilp_solution_pareto_best_budget_")
    return float(suffix.replace("p", "."))


def build_greedy_plan(
    config_dir: Path,
    model_name: str,
    gpu_mem_budget_mb: float,
    cpu_mem_budget_mb: float = 1e18,
) -> Path:
    sweep = _load_sweep_module()
    stats_csv, graph_csv, transfer_csv = sweep._default_paths(config_dir, model_name)
    data = load_ilp_inputs(
        metrics_stats_csv=str(stats_csv),
        graph_edges_csv=str(graph_csv),
        transfer_edges_csv=str(transfer_csv),
        k_sigma=1.0,
    )
    # Matches validation/sweep_ilp_pareto.py's own argparse defaults
    # (w_time=1.0, w_energy=0.0, w_transfer=1.0, w_fragmentation=0.0,
    # memory_model="topological", peak_activation_overlap=0.35) so this
    # reads the SAME greedy bits the Pareto sweep CSV already reports for
    # this budget, not a differently-weighted variant.
    #
    # Deliberately NOT "topological_tight" even though that's the ILP's own
    # default now (see fix/09): this greedy baseline is meant as an
    # INDEPENDENT comparison point against whatever the ILP chose, not a
    # second copy of the ILP's own memory accounting -- confirmed
    # intentional, not a stale reference (see
    # fix/10_backward_segment_misattribution_residual_blocks/README.md).
    cfg = ILPConfig(
        w_time=1.0,
        w_energy=0.0,
        enforce_convex_weights=False,
        w_transfer=1.0,
        w_fragmentation=0.0,
        gpu_mem_budget_mb=gpu_mem_budget_mb,
        cpu_mem_budget_mb=cpu_mem_budget_mb,
        memory_model="topological",
        peak_activation_overlap=0.35,
    )
    bits = sweep._greedy_bits(data, cfg)
    if bits is None:
        raise RuntimeError(
            f"Greedy heuristic found no feasible partition for {model_name} at "
            f"gpu_mem_budget_mb={gpu_mem_budget_mb}. If an ilp_solution_pareto_best_budget_* "
            "directory exists for this same budget, that means the ILP solved a budget "
            "where greedy could not -- that is itself evidence worth reporting (greedy "
            "failing where PRISM succeeds), not a bug in this script."
        )

    out_dir = config_dir / f"greedy_solution_budget_{_budget_slug(gpu_mem_budget_mb)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same schema as src/ilp/export_solution.py's ilp_assignment.csv: the
    # greedy heuristic assigns one device per node for BOTH passes (no
    # independent forward/backward choice like the ILP), so
    # device_forward == device_backward always, and activation_strategy is
    # always "retain" (the greedy heuristic here does not reason about
    # recompute/checkpoint strategies).
    assign_rows = [
        {
            "layer": layer,
            "device": "GPU" if bit == 1 else "CPU",
            "device_forward": "GPU" if bit == 1 else "CPU",
            "device_backward": "GPU" if bit == 1 else "CPU",
            "activation_strategy": "retain",
        }
        for layer, bit in sorted(bits.items())
    ]
    pd.DataFrame(
        assign_rows,
        columns=["layer", "device", "device_forward", "device_backward", "activation_strategy"],
    ).to_csv(out_dir / "ilp_assignment.csv", index=False)

    cut_rows = []
    for (u, v) in data.edges:
        if bits[u] != bits[v]:
            cut_rows.append({"src_layer": u, "dst_layer": v, "phase": "forward"})
            cut_rows.append({"src_layer": u, "dst_layer": v, "phase": "backward"})
    pd.DataFrame(cut_rows, columns=["src_layer", "dst_layer", "phase"]).to_csv(
        out_dir / "ilp_cut_edges.csv", index=False
    )

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu_mem_budget_mb",
        type=float,
        default=None,
        help="Defaults to the budget already encoded in this config_dir's "
        "ilp_solution_pareto_best_budget_<N>/ directory name, so greedy and "
        "the ILP are compared at the identical budget.",
    )
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    budget = args.gpu_mem_budget_mb
    if budget is None:
        budget = _find_existing_pareto_budget(config_dir)

    out_dir = build_greedy_plan(config_dir, args.model, budget, args.cpu_mem_budget_mb)
    print(f"[ok] greedy plan written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
