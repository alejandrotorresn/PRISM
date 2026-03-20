#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _find_pareto_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_pareto_sweep.csv"))


def _find_ablation_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_ablation_suite.csv"))


def _find_sensitivity_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_sensitivity.csv"))


def _safe_pct_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - candidate) / baseline) * 100.0


def _best_feasible_rows(df: pd.DataFrame) -> pd.DataFrame:
    feasible = df[df["ilp_status"].isin(["optimal", "feasible"])].copy()
    if feasible.empty:
        return feasible
    idx = feasible.groupby("model", sort=False)["ilp_objective"].idxmin()
    return feasible.loc[idx].sort_values(by=["model"], kind="stable")


def _plot_model_objective_curves(model_df: pd.DataFrame, model: str, out_dir: Path) -> None:
    model_df = model_df.sort_values(by=["gpu_budget_mb"], kind="stable")

    x = model_df["gpu_budget_mb"].astype(float)
    y_ilp = model_df["ilp_objective"].astype(float)
    y_cpu = model_df["all_cpu_objective"].astype(float)
    y_greedy = model_df["greedy_objective"].astype(float) if "greedy_objective" in model_df.columns else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_ilp, marker="o", linewidth=2.0, label="ILP")
    if y_greedy is not None:
        ax.plot(x, y_greedy, linestyle="-.", linewidth=1.8, label="Greedy")
    ax.plot(x, y_cpu, linestyle="--", linewidth=1.8, label="All CPU")

    # Plot All GPU only where finite.
    finite_gpu = model_df[pd.to_numeric(model_df["all_gpu_objective"], errors="coerce").replace([float("inf")], pd.NA).notna()]
    if len(finite_gpu) > 0:
        ax.plot(
            finite_gpu["gpu_budget_mb"].astype(float),
            finite_gpu["all_gpu_objective"].astype(float),
            linestyle=":",
            linewidth=1.8,
            label="All GPU",
        )

    ax.set_title(f"{model}: Objective vs GPU Memory Budget")
    ax.set_xlabel("GPU Memory Budget (MB)")
    ax.set_ylabel("Objective (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = out_dir / f"{model}_objective_vs_budget.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_best_improvements(best_df: pd.DataFrame, out_dir: Path) -> None:
    if best_df.empty:
        return

    plot_df = best_df.copy()
    plot_df["improvement_vs_all_cpu_pct"] = plot_df.apply(
        lambda r: _safe_pct_improvement(float(r["all_cpu_objective"]), float(r["ilp_objective"])),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["model"], plot_df["improvement_vs_all_cpu_pct"], color="#1f77b4")
    ax.set_title("Best ILP Improvement vs All-CPU Baseline")
    ax.set_ylabel("Improvement (%)")
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.25)

    for i, v in enumerate(plot_df["improvement_vs_all_cpu_pct"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    out = out_dir / "best_ilp_vs_all_cpu_improvement.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _write_markdown_summary(full_df: pd.DataFrame, best_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# ILP Results Summary\n")
    lines.append("## Inputs")
    lines.append(f"- Pareto rows: {len(full_df)}")
    lines.append(f"- Models: {', '.join(sorted(full_df['model'].astype(str).unique().tolist()))}")
    lines.append("")

    if best_df.empty:
        lines.append("No feasible ILP rows were found.")
    else:
        lines.append("## Best Feasible Row Per Model")
        display_cols = [
            "model",
            "gpu_budget_mb",
            "ilp_objective",
            "greedy_objective",
            "ilp_gpu_mem_mb",
            "ilp_cpu_mem_mb",
            "ilp_layers_gpu",
            "ilp_layers_cpu",
            "ilp_cut_edges",
            "all_cpu_objective",
            "all_gpu_status",
        ]
        table = best_df[display_cols].copy()
        table["improvement_vs_all_cpu_pct"] = table.apply(
            lambda r: _safe_pct_improvement(float(r["all_cpu_objective"]), float(r["ilp_objective"])),
            axis=1,
        )
        if "greedy_objective" in table.columns:
            table["improvement_vs_greedy_pct"] = table.apply(
                lambda r: _safe_pct_improvement(float(r["greedy_objective"]), float(r["ilp_objective"])),
                axis=1,
            )
        try:
            lines.append(table.to_markdown(index=False))
        except Exception:
            # Fallback path when optional dependency `tabulate` is unavailable.
            lines.append("```text")
            lines.append(table.to_string(index=False))
            lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines))


def _write_ablation_markdown_summary(ablation_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# ILP Ablation Summary\n")
    lines.append("## Inputs")
    lines.append(f"- Rows: {len(ablation_df)}")
    lines.append(f"- Models: {', '.join(sorted(ablation_df['model'].astype(str).unique().tolist()))}")
    lines.append(f"- Variants: {', '.join(sorted(ablation_df['variant'].astype(str).unique().tolist()))}")
    lines.append("")

    feasible = ablation_df[ablation_df["ilp_status"].isin(["optimal", "feasible"])].copy()
    if feasible.empty:
        lines.append("No feasible ablation rows were found.")
    else:
        idx = feasible.groupby(["model", "variant"], sort=False)["ilp_objective"].idxmin()
        best = feasible.loc[idx].sort_values(by=["model", "variant"], kind="stable")
        cols = [
            "model",
            "variant",
            "gpu_budget_mb",
            "ilp_objective",
            "delta_vs_full_obj",
            "ilp_cut_edges",
            "ilp_layers_gpu",
            "ilp_layers_cpu",
        ]
        table = best[cols].copy()
        try:
            lines.append(table.to_markdown(index=False))
        except Exception:
            lines.append("```text")
            lines.append(table.to_string(index=False))
            lines.append("```")

    out_path.write_text("\n".join(lines))


def _write_sensitivity_markdown_summary(sensitivity_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# ILP Sensitivity Analysis Report\n")
    lines.append("## Overview")
    lines.append(f"- Rows: {len(sensitivity_df)}")
    lines.append(f"- Models: {', '.join(sorted(sensitivity_df['model'].astype(str).unique().tolist()))}")
    params = [p for p in sensitivity_df['param_name'].unique() if p != 'baseline']
    lines.append(f"- Parameters swept: {', '.join(sorted(params))}")
    lines.append("")

    feasible = sensitivity_df[
        sensitivity_df["ilp_status"].isin(["optimal", "feasible"]) &
        (sensitivity_df["param_name"] != "baseline")
    ].copy()

    if feasible.empty:
        lines.append("No feasible sensitivity rows were found.")
    else:
        for param in sorted(params):
            sub = feasible[feasible["param_name"] == param].copy()
            if sub.empty:
                continue
            lines.append(f"## Parameter: `{param}`")
            idx = sub.groupby(["model", "param_value"], sort=False)["ilp_objective"].idxmin()
            best = sub.loc[idx].sort_values(["model", "param_value"], kind="stable")
            cols = ["model", "param_value", "gpu_budget_mb", "ilp_objective",
                    "baseline_objective", "delta_abs", "delta_pct",
                    "ilp_cut_edges", "ilp_layers_gpu", "ilp_layers_cpu"]
            available = [c for c in cols if c in best.columns]
            table = best[available].copy()
            try:
                lines.append(table.to_markdown(index=False))
            except Exception:
                lines.append("```text")
                lines.append(table.to_string(index=False))
                lines.append("```")
            lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate consolidated ILP report assets from Pareto sweep CSV files")
    parser.add_argument("--input_root", default="data/test-m4", help="Root folder to scan for *_pareto_sweep.csv")
    parser.add_argument("--output_dir", default="reports/ilp_results", help="Output folder for tables/plots")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    pareto_files = _find_pareto_files(input_root)
    if not pareto_files:
        raise FileNotFoundError(f"No *_pareto_sweep.csv files found under: {input_root}")

    frames = []
    for p in pareto_files:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    consolidated_csv = out_dir / "ilp_pareto_consolidated.csv"
    full_df.to_csv(consolidated_csv, index=False)

    best_df = _best_feasible_rows(full_df)
    best_csv = out_dir / "ilp_best_per_model.csv"
    best_df.to_csv(best_csv, index=False)

    for model in sorted(full_df["model"].astype(str).unique().tolist()):
        _plot_model_objective_curves(full_df[full_df["model"] == model], model, out_dir)

    _plot_best_improvements(best_df, out_dir)

    md_summary = out_dir / "ILP_RESULTS_SUMMARY.md"
    _write_markdown_summary(full_df, best_df, md_summary)

    ablation_files = _find_ablation_files(input_root)
    if ablation_files:
        ablation_frames = []
        for p in ablation_files:
            adf = pd.read_csv(p)
            adf["source_csv"] = str(p)
            ablation_frames.append(adf)

        ablation_df = pd.concat(ablation_frames, ignore_index=True)
        ablation_csv = out_dir / "ilp_ablation_consolidated.csv"
        ablation_df.to_csv(ablation_csv, index=False)

        ablation_md = out_dir / "ILP_ABLATION_SUMMARY.md"
        _write_ablation_markdown_summary(ablation_df, ablation_md)

    sensitivity_files = _find_sensitivity_files(input_root)
    if sensitivity_files:
        sens_frames = []
        for p in sensitivity_files:
            sdf = pd.read_csv(p)
            sdf["source_csv"] = str(p)
            sens_frames.append(sdf)

        sensitivity_df = pd.concat(sens_frames, ignore_index=True)
        sensitivity_csv = out_dir / "ilp_sensitivity_consolidated.csv"
        sensitivity_df.to_csv(sensitivity_csv, index=False)

        sensitivity_md = out_dir / "ILP_SENSITIVITY_SUMMARY.md"
        _write_sensitivity_markdown_summary(sensitivity_df, sensitivity_md)

    print("=" * 80)
    print("ILP REPORT ASSETS GENERATED")
    print("=" * 80)
    print(f"Input Pareto files: {len(pareto_files)}")
    print(f"Consolidated CSV: {consolidated_csv}")
    print(f"Best-per-model CSV: {best_csv}")
    print(f"Markdown summary: {md_summary}")
    if ablation_files:
        print(f"Ablation files: {len(ablation_files)}")
        print(f"Ablation consolidated CSV: {out_dir / 'ilp_ablation_consolidated.csv'}")
        print(f"Ablation markdown summary: {out_dir / 'ILP_ABLATION_SUMMARY.md'}")
    if sensitivity_files:
        print(f"Sensitivity files: {len(sensitivity_files)}")
        print(f"Sensitivity consolidated CSV: {out_dir / 'ilp_sensitivity_consolidated.csv'}")
        print(f"Sensitivity markdown summary: {out_dir / 'ILP_SENSITIVITY_SUMMARY.md'}")
    print(f"Plots directory: {out_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
