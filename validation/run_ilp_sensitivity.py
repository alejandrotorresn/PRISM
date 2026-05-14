#!/usr/bin/env python3
"""One-at-a-time (OAT) sensitivity analysis for ILP hyperparameters.

Sweeps k_sigma and w_transfer independently (keeping all other parameters at
their baseline values) and records how the optimised objective changes.  The
output is a structured CSV and JSON summary that can be cited in the thesis as
evidence of model robustness with respect to statistical dispersion tolerance
(k_sigma) and transfer-cost weighting (w_transfer).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_ilp_inputs = importlib.import_module("ilp.data_loader").load_ilp_inputs
merge_ilp_inputs_multi_hardware = importlib.import_module("ilp.data_loader").merge_ilp_inputs_multi_hardware
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp

# ---------------------------------------------------------------------------
# Default sweep grids
# ---------------------------------------------------------------------------
DEFAULT_K_SIGMA_VALUES: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_W_TRANSFER_VALUES: List[float] = [0.0, 0.5, 1.0, 2.0, 5.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_paths(config_dir: Path, model_name: str):
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([p for p in config_dir.glob("run_*") if p.is_dir()])
    if run_dirs:
        ref_run = run_dirs[0]
        graph_edges = ref_run / f"{model_name}_graph_edges.csv"
        transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    else:
        graph_edges = config_dir / f"{model_name}_graph_edges.csv"
        transfer_edges = config_dir / f"{model_name}_transfer_edges.csv"

    if not graph_edges.exists() or not transfer_edges.exists():
        raise FileNotFoundError(
            "Could not resolve graph/transfer artifacts in config_dir. "
            f"Expected either run_*/ files or direct files in: {config_dir}"
        )
    return stats, graph_edges, transfer_edges


def _parse_float_list(text: str) -> List[float]:
    vals = []
    for chunk in text.split(","):
        c = chunk.strip()
        if c:
            vals.append(float(c))
    if not vals:
        raise ValueError("At least one value must be provided")
    return vals


def _parse_budget_list(text: str) -> List[float]:
    return _parse_float_list(text)


def _solve_one(
    data,
    gpu_budget_mb: float,
    cpu_budget_mb: float,
    k_sigma: float,
    w_time: float,
    w_energy: float,
    w_transfer: float,
    w_fragmentation: float,
    backend: str,
) -> Dict[str, Any]:
    """Reload data with the given k_sigma and solve ILP."""
    # k_sigma is baked into the loaded data (it rescales the robust cost).
    # We need to pass it through ILPConfig-equivalent state.  Because the
    # data object already contains the robust costs with the k_sigma applied at
    # load time, and we want to sweep k_sigma, we accept *data* pre-loaded
    # with that k_sigma here — callers are responsible for loading per sweep.
    cfg = ILPConfig(
        w_time=w_time,
        w_energy=w_energy,
        w_transfer=w_transfer,
        w_fragmentation=w_fragmentation,
        gpu_mem_budget_mb=gpu_budget_mb,
        cpu_mem_budget_mb=cpu_budget_mb,
    )
    result = solve_partition_ilp(data, cfg, backend=backend)
    return {
        "ilp_status": result.status,
        "ilp_objective": result.objective_value,
        "ilp_gpu_mem_mb": result.gpu_mem_used_mb,
        "ilp_cpu_mem_mb": result.cpu_mem_used_mb,
        "ilp_layers_gpu": sum(1 for d in result.assignment.values() if d == "GPU"),
        "ilp_layers_cpu": sum(1 for d in result.assignment.values() if d == "CPU"),
        "ilp_cut_edges": len(result.cut_edges),
    }


def _safe_delta_pct(baseline: float, candidate: float) -> Optional[float]:
    if baseline in (float("inf"), float("-inf"), 0.0) or baseline != baseline:
        return None
    return round(((candidate - baseline) / abs(baseline)) * 100.0, 4)


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sensitivity(
    config_dirs: List[Path],
    model: str,
    gpu_budgets_mb: List[float],
    cpu_mem_budget_mb: float,
    baseline_k_sigma: float,
    baseline_k_sigma_time: float | None,
    baseline_k_sigma_energy: float | None,
    baseline_w_time: float,
    baseline_w_energy: float,
    baseline_w_transfer: float,
    baseline_w_fragmentation: float,
    k_sigma_values: List[float],
    w_transfer_values: List[float],
    backend: str,
    hw_aggregate: str,
    hw_dispersion_k: float,
    strict_graph_mapping: bool,
    strict_transfer_mapping: bool,
    strict_sample_quality: bool,
    strict_transfer_calibration: bool,
    strict_graph_trace_source: bool,
) -> pd.DataFrame:
    """Run OAT sensitivity for k_sigma and w_transfer, return tidy DataFrame."""

    def _load_data(k_sigma: float, strict_graph: bool, strict_transfer: bool):
        profiles = []
        for cdir in config_dirs:
            stats_csv, graph_csv, transfer_csv = _default_paths(cdir, model)
            profile = load_ilp_inputs(
                metrics_stats_csv=str(stats_csv),
                graph_edges_csv=str(graph_csv),
                transfer_edges_csv=str(transfer_csv),
                k_sigma=k_sigma,
                k_sigma_time=baseline_k_sigma_time,
                k_sigma_energy=baseline_k_sigma_energy,
                strict_graph_mapping=strict_graph,
                strict_transfer_mapping=strict_transfer,
                strict_sample_quality=strict_sample_quality,
                strict_transfer_calibration=strict_transfer_calibration,
                strict_graph_trace_source=strict_graph_trace_source,
            )
            profiles.append(profile)

        if len(profiles) == 1:
            return profiles[0]
        return merge_ilp_inputs_multi_hardware(
            profiles=profiles,
            strategy=hw_aggregate,
            dispersion_k=hw_dispersion_k,
            strict_schema=True,
        )

    rows: List[Dict[str, Any]] = []

    # --- Establish baseline rows (k_sigma=baseline, w_transfer=baseline) ------
    baseline_data = _load_data(baseline_k_sigma, strict_graph_mapping, strict_transfer_mapping)
    for b in gpu_budgets_mb:
        res = _solve_one(
            baseline_data, b, cpu_mem_budget_mb,
            baseline_k_sigma, baseline_w_time, baseline_w_energy, baseline_w_transfer, baseline_w_fragmentation, backend,
        )
        rows.append({
            "model": model,
            "param_name": "baseline",
            "param_value": float("nan"),
            "gpu_budget_mb": b,
            **res,
            "baseline_objective": res["ilp_objective"],
            "delta_abs": 0.0,
            "delta_pct": 0.0,
        })

    # Build baseline objective look-up keyed by budget.
    baseline_obj: Dict[float, float] = {
        r["gpu_budget_mb"]: r["ilp_objective"]
        for r in rows
        if r["param_name"] == "baseline"
    }

    # --- k_sigma sweep --------------------------------------------------------
    for ks in k_sigma_values:
        if ks == baseline_k_sigma:
            # Re-use baseline rows (they are already in the output; just skip
            # duplicate solving and emit a reference row).
            for b in gpu_budgets_mb:
                base_val = baseline_obj.get(b, float("nan"))
                rows.append({
                    "model": model,
                    "param_name": "k_sigma",
                    "param_value": ks,
                    "gpu_budget_mb": b,
                    **{k: v for k, v in next(
                        r for r in rows
                        if r["param_name"] == "baseline" and r["gpu_budget_mb"] == b
                    ).items() if k not in ("model", "param_name", "param_value",
                                           "gpu_budget_mb", "baseline_objective",
                                           "delta_abs", "delta_pct")},
                    "baseline_objective": base_val,
                    "delta_abs": 0.0,
                    "delta_pct": 0.0,
                })
            continue

        data_ks = _load_data(ks, strict_graph_mapping, strict_transfer_mapping)
        for b in gpu_budgets_mb:
            res = _solve_one(
                data_ks, b, cpu_mem_budget_mb,
                ks, baseline_w_time, baseline_w_energy, baseline_w_transfer, baseline_w_fragmentation, backend,
            )
            base_val = baseline_obj.get(b, float("nan"))
            delta_abs = res["ilp_objective"] - base_val if base_val == base_val else float("nan")
            delta_pct = _safe_delta_pct(base_val, res["ilp_objective"])
            rows.append({
                "model": model,
                "param_name": "k_sigma",
                "param_value": ks,
                "gpu_budget_mb": b,
                **res,
                "baseline_objective": base_val,
                "delta_abs": round(float(delta_abs), 6) if delta_abs == delta_abs else float("nan"),
                "delta_pct": delta_pct if delta_pct is not None else float("nan"),
            })

    # --- w_transfer sweep -----------------------------------------------------
    for wt in w_transfer_values:
        if wt == baseline_w_transfer:
            for b in gpu_budgets_mb:
                base_val = baseline_obj.get(b, float("nan"))
                rows.append({
                    "model": model,
                    "param_name": "w_transfer",
                    "param_value": wt,
                    "gpu_budget_mb": b,
                    **{k: v for k, v in next(
                        r for r in rows
                        if r["param_name"] == "baseline" and r["gpu_budget_mb"] == b
                    ).items() if k not in ("model", "param_name", "param_value",
                                           "gpu_budget_mb", "baseline_objective",
                                           "delta_abs", "delta_pct")},
                    "baseline_objective": base_val,
                    "delta_abs": 0.0,
                    "delta_pct": 0.0,
                })
            continue

        for b in gpu_budgets_mb:
            res = _solve_one(
                baseline_data, b, cpu_mem_budget_mb,
                baseline_k_sigma, baseline_w_time, baseline_w_energy, wt, baseline_w_fragmentation, backend,
            )
            base_val = baseline_obj.get(b, float("nan"))
            delta_abs = res["ilp_objective"] - base_val if base_val == base_val else float("nan")
            delta_pct = _safe_delta_pct(base_val, res["ilp_objective"])
            rows.append({
                "model": model,
                "param_name": "w_transfer",
                "param_value": wt,
                "gpu_budget_mb": b,
                **res,
                "baseline_objective": base_val,
                "delta_abs": round(float(delta_abs), 6) if delta_abs == delta_abs else float("nan"),
                "delta_pct": delta_pct if delta_pct is not None else float("nan"),
            })

    df = pd.DataFrame(rows).sort_values(
        by=["param_name", "param_value", "gpu_budget_mb"], kind="stable"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="OAT sensitivity analysis for ILP hyperparameters (k_sigma, w_transfer)"
    )
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--config_dirs", default=None, help="Comma-separated batch directories")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu_budgets_mb", required=True, help="Comma-separated budgets, e.g. 4,8,16,32,64")
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--k_sigma", type=float, default=1.0, help="Baseline k_sigma value")
    parser.add_argument("--k_sigma_time", type=float, default=None, help="Optional baseline sigma for time")
    parser.add_argument("--k_sigma_energy", type=float, default=None, help="Optional baseline sigma for energy")
    parser.add_argument("--w_time", type=float, default=1.0, help="Baseline w_time")
    parser.add_argument("--w_energy", type=float, default=0.0, help="Baseline w_energy")
    parser.add_argument("--w_transfer", type=float, default=1.0, help="Baseline w_transfer")
    parser.add_argument("--w_fragmentation", type=float, default=0.0, help="Baseline fragmentation regularizer")
    parser.add_argument("--k_sigma_values", default=",".join(str(v) for v in DEFAULT_K_SIGMA_VALUES),
                        help="Comma-separated k_sigma sweep values")
    parser.add_argument("--w_transfer_values", default=",".join(str(v) for v in DEFAULT_W_TRANSFER_VALUES),
                        help="Comma-separated w_transfer sweep values")
    parser.add_argument("--strict_graph_mapping", action="store_true")
    parser.add_argument("--strict_transfer_mapping", action="store_true")
    parser.add_argument("--allow_low_quality_stats", action="store_true", help="Allow sensitivity runs on metrics_stats.csv rows flagged as low quality (diagnostic only)")
    parser.add_argument("--allow_transfer_calibration_fallback", action="store_true", help="Allow sensitivity runs when transfer calibration fell back to neutral defaults (diagnostic only)")
    parser.add_argument("--allow_fallback_graph_trace", action="store_true", help="Allow sensitivity runs from fallback_leaf_modules graph traces (diagnostic only)")
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--hw_aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--hw_dispersion_k", type=float, default=0.0)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if args.config_dirs:
        config_dirs = [Path(p.strip()) for p in args.config_dirs.split(",") if p.strip()]
    else:
        if not args.config_dir:
            raise ValueError("Provide --config_dir or --config_dirs")
        config_dirs = [Path(args.config_dir)]

    for cdir in config_dirs:
        if not cdir.exists():
            raise FileNotFoundError(f"config_dir does not exist: {cdir}")

    gpu_budgets = _parse_budget_list(args.gpu_budgets_mb)
    k_sigma_values = _parse_float_list(args.k_sigma_values)
    w_transfer_values = _parse_float_list(args.w_transfer_values)

    df = run_sensitivity(
        config_dirs=config_dirs,
        model=args.model,
        gpu_budgets_mb=gpu_budgets,
        cpu_mem_budget_mb=args.cpu_mem_budget_mb,
        baseline_k_sigma=args.k_sigma,
        baseline_k_sigma_time=args.k_sigma_time,
        baseline_k_sigma_energy=args.k_sigma_energy,
        baseline_w_time=args.w_time,
        baseline_w_energy=args.w_energy,
        baseline_w_transfer=args.w_transfer,
        baseline_w_fragmentation=args.w_fragmentation,
        k_sigma_values=k_sigma_values,
        w_transfer_values=w_transfer_values,
        backend=args.backend,
        hw_aggregate=args.hw_aggregate,
        hw_dispersion_k=args.hw_dispersion_k,
        strict_graph_mapping=args.strict_graph_mapping,
        strict_transfer_mapping=args.strict_transfer_mapping,
        strict_sample_quality=not args.allow_low_quality_stats,
        strict_transfer_calibration=not args.allow_transfer_calibration_fallback,
        strict_graph_trace_source=not args.allow_fallback_graph_trace,
    )

    base_config_dir = config_dirs[0]

    out_csv = args.output_csv or str(base_config_dir / f"{args.model}_sensitivity.csv")
    df.to_csv(out_csv, index=False)

    # --- JSON summary ---------------------------------------------------------
    summary: Dict[str, Any] = {"model": args.model, "parameters": {}}
    for param in df["param_name"].unique():
        if param == "baseline":
            continue
        sub = df[(df["param_name"] == param) & (df["ilp_status"].isin(["optimal", "feasible"]))].copy()
        if sub.empty:
            summary["parameters"][param] = {}
            continue
        # Best budget per param_value
        idx = sub.groupby("param_value", sort=False)["ilp_objective"].idxmin()
        best = sub.loc[idx].sort_values("param_value")
        summary["parameters"][param] = [
            {
                "param_value": row["param_value"],
                "best_budget_mb": row["gpu_budget_mb"],
                "ilp_objective": row["ilp_objective"],
                "delta_pct": row["delta_pct"],
            }
            for _, row in best.iterrows()
        ]

    out_json = args.output_json or str(base_config_dir / f"{args.model}_sensitivity_summary.json")
    Path(out_json).write_text(json.dumps(summary, indent=2))

    n_k = len([v for v in k_sigma_values if v not in [args.k_sigma]])
    n_wt = len([v for v in w_transfer_values if v not in [args.w_transfer]])

    print("=" * 80)
    print("ILP SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Rows:           {len(df)}")
    print(f"k_sigma sweep:  {k_sigma_values}  (baseline={args.k_sigma})")
    print(f"w_transfer sweep: {w_transfer_values}  (baseline={args.w_transfer})")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
