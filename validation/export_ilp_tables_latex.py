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

    keep = pd.DataFrame(
        {
            "Model": out["model"],
            "GPU Budget (MB)": out["gpu_budget_mb"].map(lambda v: _fmt_float(v, 0)),
            "ILP Obj": out["ilp_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-CPU Obj": out["all_cpu_objective"].map(lambda v: _fmt_float(v, 3)),
            "Improve vs CPU (\\%)": out["impr_vs_cpu_pct"].map(lambda v: _fmt_float(v, 2)),
            "ILP GPU Mem (MB)": out["ilp_gpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "ILP CPU Mem (MB)": out["ilp_cpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "GPU Layers": out["ilp_layers_gpu"].astype(int),
            "CPU Layers": out["ilp_layers_cpu"].astype(int),
            "Cut Edges": out["ilp_cut_edges"].astype(int),
        }
    )
    return keep


def _prepare_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(by=["model", "gpu_budget_mb"], kind="stable")
    out["impr_vs_cpu_pct"] = out.apply(
        lambda r: _safe_improvement(float(r["all_cpu_objective"]), float(r["ilp_objective"])),
        axis=1,
    )

    keep = pd.DataFrame(
        {
            "Model": out["model"],
            "GPU Budget (MB)": out["gpu_budget_mb"].map(lambda v: _fmt_float(v, 0)),
            "ILP Status": out["ilp_status"],
            "ILP Obj": out["ilp_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-CPU Obj": out["all_cpu_objective"].map(lambda v: _fmt_float(v, 3)),
            "All-GPU Status": out["all_gpu_status"],
            "Improve vs CPU (\\%)": out["impr_vs_cpu_pct"].map(lambda v: _fmt_float(v, 2)),
            "ILP GPU Mem (MB)": out["ilp_gpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "ILP CPU Mem (MB)": out["ilp_cpu_mem_mb"].map(lambda v: _fmt_float(v, 2)),
            "Cut Edges": out["ilp_cut_edges"].astype(int),
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
    parser.add_argument("--output_dir", default="reports/ilp_results/latex")
    args = parser.parse_args()

    best_path = Path(args.best_csv)
    cons_path = Path(args.consolidated_csv)
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
    all_path_tex = out_dir / "ilp_tables.tex"

    best_path_tex.write_text(best_tex)
    budget_path_tex.write_text(budget_tex)
    all_path_tex.write_text(best_tex + "\n" + budget_tex)

    print("=" * 80)
    print("ILP LATEX TABLES EXPORTED")
    print("=" * 80)
    print(f"Best table: {best_path_tex}")
    print(f"Budget sweep table: {budget_path_tex}")
    print(f"Combined tables: {all_path_tex}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
