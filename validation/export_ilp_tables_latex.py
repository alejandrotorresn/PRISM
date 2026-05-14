#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - candidate) / baseline) * 100.0


def _fmt_float(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def _prepare_best_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["impr_vs_cpu_pct"] = out.apply(
        lambda r: _safe_improvement(float(r["all_cpu_objective"]), float(r["ilp_objective"])),
        axis=1,
    )
    if "greedy_objective" in out.columns:
        out["impr_vs_greedy_pct"] = out.apply(
            lambda r: _safe_improvement(float(r["greedy_objective"]), float(r["ilp_objective"])),
            axis=1,
        )
    else:
        out["impr_vs_greedy_pct"] = float("nan")

    if "all_gpu_objective" in out.columns:
        all_gpu_obj = out["all_gpu_objective"].map(lambda v: _fmt_float(v, 3))
    else:
        all_gpu_obj = "-"

    if "all_gpu_status" in out.columns:
        all_gpu_status = out["all_gpu_status"]
    else:
        all_gpu_status = "-"

    if "ilp_layers_gpu_forward" in out.columns:
        fwd_gpu = out["ilp_layers_gpu_forward"].astype(int)
        fwd_cpu = out["ilp_layers_cpu_forward"].astype(int)
        bwd_gpu = out["ilp_layers_gpu_backward"].astype(int)
        bwd_cpu = out["ilp_layers_cpu_backward"].astype(int)
        cut_fwd = out["ilp_cut_edges_forward"].astype(int)
        cut_bwd = out["ilp_cut_edges_backward"].astype(int)
        cross_phase = out["ilp_cross_phase_edges"].astype(int)
    else:
        fwd_gpu = out["ilp_layers_gpu"].astype(int)
        fwd_cpu = out["ilp_layers_cpu"].astype(int)
        bwd_gpu = out["ilp_layers_gpu"].astype(int)
        bwd_cpu = out["ilp_layers_cpu"].astype(int)
        cut_fwd = out["ilp_cut_edges"].astype(int)
        cut_bwd = out["ilp_cut_edges"].astype(int)
        cross_phase = pd.Series([0] * len(out), index=out.index)

    keep = pd.DataFrame(
        {
            "Model": out["model"],
            "GPU Budget (MB)": out["gpu_budget_mb"].map(lambda v: _fmt_float(v, 0)),
            "ILP Obj": out["ilp_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-CPU Obj": out["all_cpu_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-GPU Status": all_gpu_status,
            "All-GPU Obj": all_gpu_obj,
            "Greedy Obj": out["greedy_objective"].map(lambda v: _fmt_float(v, 3)) if "greedy_objective" in out.columns else "-",
            "Improve vs CPU (\\%)": out["impr_vs_cpu_pct"].map(lambda v: _fmt_float(v, 2)),
            "Improve vs Greedy (\\%)": out["impr_vs_greedy_pct"].map(lambda v: _fmt_float(v, 2)),
            "ILP GPU Mem (MB)": out["ilp_gpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "ILP CPU Mem (MB)": out["ilp_cpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "Fwd GPU": fwd_gpu,
            "Fwd CPU": fwd_cpu,
            "Bwd GPU": bwd_gpu,
            "Bwd CPU": bwd_cpu,
            "Cut Fwd": cut_fwd,
            "Cut Bwd": cut_bwd,
            "Cross-Phase": cross_phase,
        }
    )
    return keep


def _prepare_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(by=["model", "gpu_budget_mb"], kind="stable")
    out["impr_vs_cpu_pct"] = out.apply(
        lambda r: _safe_improvement(float(r["all_cpu_objective"]), float(r["ilp_objective"])),
        axis=1,
    )
    if "greedy_objective" in out.columns:
        out["impr_vs_greedy_pct"] = out.apply(
            lambda r: _safe_improvement(float(r["greedy_objective"]), float(r["ilp_objective"])),
            axis=1,
        )
    else:
        out["impr_vs_greedy_pct"] = float("nan")

    if "all_gpu_objective" in out.columns:
        all_gpu_obj = out["all_gpu_objective"].map(lambda v: _fmt_float(v, 3))
    else:
        all_gpu_obj = "-"

    if "ilp_layers_gpu_forward" in out.columns:
        fwd_gpu = out["ilp_layers_gpu_forward"].astype(int)
        bwd_gpu = out["ilp_layers_gpu_backward"].astype(int)
        cut_bwd = out["ilp_cut_edges_backward"].astype(int)
        cross_phase = out["ilp_cross_phase_edges"].astype(int)
    else:
        fwd_gpu = out["ilp_layers_gpu"].astype(int)
        bwd_gpu = out["ilp_layers_gpu"].astype(int)
        cut_bwd = out["ilp_cut_edges"].astype(int)
        cross_phase = pd.Series([0] * len(out), index=out.index)

    keep = pd.DataFrame(
        {
            "Model": out["model"],
            "GPU Budget (MB)": out["gpu_budget_mb"].map(lambda v: _fmt_float(v, 0)),
            "ILP Status": out["ilp_status"],
            "ILP Obj": out["ilp_objective"].map(lambda v: _fmt_float(v, 3)),
            "Greedy Obj": out["greedy_objective"].map(lambda v: _fmt_float(v, 3)) if "greedy_objective" in out.columns else "-",
            "All-CPU Obj": out["all_cpu_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-GPU Status": out["all_gpu_status"],
            "All-GPU Obj": all_gpu_obj,
            "Improve vs CPU (\\%)": out["impr_vs_cpu_pct"].map(lambda v: _fmt_float(v, 2)),
            "Improve vs Greedy (\\%)": out["impr_vs_greedy_pct"].map(lambda v: _fmt_float(v, 2)),
            "ILP GPU Mem (MB)": out["ilp_gpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "ILP CPU Mem (MB)": out["ilp_cpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "Fwd GPU": fwd_gpu,
            "Bwd GPU": bwd_gpu,
            "Cut Fwd": out["ilp_cut_edges"].astype(int),
            "Cut Bwd": cut_bwd,
            "Cross-Phase": cross_phase,
        }
    )
    return keep


def _prepare_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    feasible = df[df["ilp_status"].isin(["optimal", "feasible"])].copy()
    if feasible.empty:
        return feasible
    idx = feasible.groupby(["model", "variant"], sort=False)["ilp_objective"].idxmin()
    best = feasible.loc[idx].sort_values(by=["model", "variant"], kind="stable")

    keep = pd.DataFrame(
        {
            "Model": best["model"],
            "Variant": best["variant"],
            "GPU Budget (MB)": best["gpu_budget_mb"].map(lambda v: _fmt_float(v, 0)),
            "ILP Obj": best["ilp_objective"].map(lambda v: _fmt_float(v, 3)),
            "Delta vs Full": best["delta_vs_full_obj"].map(lambda v: _fmt_float(v, 3)),
            "Cut Edges": best["ilp_cut_edges"].astype(int),
            "GPU Layers": best["ilp_layers_gpu"].astype(int),
            "CPU Layers": best["ilp_layers_cpu"].astype(int),
        }
    )
    return keep


def _prepare_hybrid_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(by=["model"], kind="stable")
    keep = pd.DataFrame(
        {
            "Model": out["model"],
            "Optimizer": out["config_optimizer"].fillna("-"),
            "Precision": out["config_precision"].fillna("-"),
            "Batch": out["config_batch_size"].map(lambda v: _fmt_float(v, 0)),
            "Avg Step (ms)": out["avg_step_ms"].map(lambda v: _fmt_float(v, 3)),
            "Final Loss": out["final_loss"].map(lambda v: _fmt_float(v, 3)),
            "Metric": out["quality_metric_name"].fillna("-"),
            "Metric Value": out["final_quality_metric"].map(lambda v: _fmt_float(v, 3)),
            "Dataset": out["dataset_name"].fillna("-"),
            "Input Source": out["input_source"].fillna("-"),
        }
    )
    return keep


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=False)
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            body,
            "\\end{table}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ILP result tables to LaTeX")
    parser.add_argument("--best_csv", default="reports/ilp_results/ilp_best_per_model.csv")
    parser.add_argument("--consolidated_csv", default="reports/ilp_results/ilp_pareto_consolidated.csv")
    parser.add_argument("--ablation_csv", default="reports/ilp_results/ilp_ablation_consolidated.csv")
    parser.add_argument("--hybrid_csv", default="reports/ilp_results/hybrid_execution_best_per_model.csv")
    parser.add_argument("--output_dir", default="reports/ilp_results/latex")
    args = parser.parse_args()

    best_path = Path(args.best_csv)
    cons_path = Path(args.consolidated_csv)
    abl_path = Path(args.ablation_csv)
    hybrid_path = Path(args.hybrid_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not best_path.exists():
        raise FileNotFoundError(f"best csv not found: {best_path}")
    if not cons_path.exists():
        raise FileNotFoundError(f"consolidated csv not found: {cons_path}")

    best_df = pd.read_csv(best_path)
    cons_df = pd.read_csv(cons_path)

    best_tbl = _prepare_best_table(best_df)
    budget_tbl = _prepare_budget_table(cons_df)

    best_tex = _latex_table(
        best_tbl,
        caption="Best feasible ILP result per model under evaluated GPU memory budgets.",
        label="tab:ilp-best-per-model",
    )
    budget_tex = _latex_table(
        budget_tbl,
        caption="ILP Pareto sweep with baseline comparison across GPU memory budgets.",
        label="tab:ilp-budget-sweep",
    )

    best_path_tex = out_dir / "ilp_best_per_model.tex"
    budget_path_tex = out_dir / "ilp_budget_sweep.tex"
    ablation_path_tex = out_dir / "ilp_ablation_best_per_variant.tex"
    hybrid_path_tex = out_dir / "ilp_hybrid_best_per_model.tex"
    all_path_tex = out_dir / "ilp_tables.tex"

    best_path_tex.write_text(best_tex)
    budget_path_tex.write_text(budget_tex)
    combined_tex = best_tex + "\n" + budget_tex

    if abl_path.exists():
        abl_df = pd.read_csv(abl_path)
        abl_tbl = _prepare_ablation_table(abl_df)
        if not abl_tbl.empty:
            abl_tex = _latex_table(
                abl_tbl,
                caption="Best feasible ablation result per model and variant.",
                label="tab:ilp-ablation-best-per-variant",
            )
            ablation_path_tex.write_text(abl_tex)
            combined_tex += "\n" + abl_tex

    if hybrid_path.exists():
        hybrid_df = pd.read_csv(hybrid_path)
        if not hybrid_df.empty:
            hybrid_tbl = _prepare_hybrid_table(hybrid_df)
            hybrid_tex = _latex_table(
                hybrid_tbl,
                caption="Best observed hybrid runtime row per model with task-quality metric and dataset provenance.",
                label="tab:ilp-hybrid-best-per-model",
            )
            hybrid_path_tex.write_text(hybrid_tex)
            combined_tex += "\n" + hybrid_tex

    all_path_tex.write_text(combined_tex)

    print("=" * 80)
    print("ILP LATEX TABLES EXPORTED")
    print("=" * 80)
    print(f"Best table: {best_path_tex}")
    print(f"Budget sweep table: {budget_path_tex}")
    if ablation_path_tex.exists():
        print(f"Ablation table: {ablation_path_tex}")
    if hybrid_path_tex.exists():
        print(f"Hybrid table: {hybrid_path_tex}")
    print(f"Combined tables: {all_path_tex}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
