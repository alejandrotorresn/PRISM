#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _get_column_safe(df: pd.DataFrame, new_name: str, legacy_name: str = None, default = None):
    """
    Get column by new name with fallback to legacy name (backward compatibility).
    """
    if new_name in df.columns:
        return df[new_name]
    elif legacy_name and legacy_name in df.columns:
        return df[legacy_name]
    else:
        return pd.Series([default] * len(df), index=df.index)
# Set academic theme for ILP plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
sns.set_palette("colorblind")


def _find_pareto_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_pareto_sweep.csv"))


def _find_ablation_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_ablation_suite.csv"))


def _find_sensitivity_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*_sensitivity.csv"))


def _find_hybrid_protocol_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("hybrid_execution_protocol.csv"))


def _detect_gpu_vram_total_mb(input_root: Path) -> float | None:
    """
    Detect total GPU VRAM (MiB) from profiling metadata when available.
    Returns None if metadata is unavailable or malformed.
    """
    for meta_path in sorted(input_root.rglob("*_meta.json")):
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        vram_val = payload.get("gpu_vram_total_mb")
        try:
            vram_mb = float(vram_val)
        except (TypeError, ValueError):
            continue
        if vram_mb > 0:
            return vram_mb
    return None


def _safe_pct_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - candidate) / baseline) * 100.0


def _normalize_rescue_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rescue_influence_* columns are present in consolidated data.
    Falls back to False if not present (backward compatibility).
    """
    for col_type in ["profiling", "ilp", "hybrid"]:
        col_name = f"rescue_influence_{col_type}"
        if col_name not in df.columns:
            df[col_name] = False
    return df

def _resolve_batch_col(df: pd.DataFrame) -> str:
    if "config_batch_size" in df.columns:
        return "config_batch_size"
    if "batch_size" in df.columns:
        return "batch_size"
    if "source_csv" in df.columns:
        # Fallback for metrics_stats which might not have batch_size explicitly
        return "batch_size_inferred"
    return "batch_size"


def _best_feasible_rows(df: pd.DataFrame) -> pd.DataFrame:
    feasible = df[df["ilp_status"].isin(["optimal", "feasible"])].copy()
    if feasible.empty:
        return feasible
    batch_col = _resolve_batch_col(feasible)
    group_cols = ["model"]
    if "optimizer" in feasible.columns:
        group_cols.append("optimizer")
    if "precision" in feasible.columns:
        group_cols.append("precision")
    group_cols.append(batch_col)
    
    idx = feasible.groupby(group_cols, sort=False)["ilp_objective"].idxmin()
    return feasible.loc[idx].sort_values(by=group_cols, kind="stable")


def _best_hybrid_rows(df: pd.DataFrame) -> pd.DataFrame:
    runtime_df = df[(df["run_label"] == "ilp_plan") & (df["status"] == "ok")].copy()
    if runtime_df.empty:
        return runtime_df

    batch_col = _resolve_batch_col(runtime_df)
    optimizer_col = "config_optimizer" if "config_optimizer" in runtime_df.columns else ("optimizer" if "optimizer" in runtime_df.columns else None)
    precision_col = "config_precision" if "config_precision" in runtime_df.columns else ("precision" if "precision" in runtime_df.columns else None)

    group_cols = ["model"]
    if optimizer_col is not None:
        group_cols.append(optimizer_col)
    if precision_col is not None:
        group_cols.append(precision_col)
    group_cols.append(batch_col)

    selected_frames = []
    for _, model_df in runtime_df.groupby(group_cols, sort=False):
        if "plan_selection_mode" in model_df.columns:
            preferred_df = model_df[model_df["plan_selection_mode"] == "pareto_best"].copy()
            if not preferred_df.empty:
                model_df = preferred_df
        best_idx = model_df["avg_step_ms"].astype(float).idxmin()
        row = runtime_df.loc[[best_idx]].copy()
        # Ensure rescue_influence_hybrid flag is present (defensive for backward compat)
        if "rescue_influence_hybrid" not in row.columns:
            row["rescue_influence_hybrid"] = False
        selected_frames.append(row)

    return pd.concat(selected_frames, ignore_index=True).sort_values(by=group_cols, kind="stable")


def _plot_model_objective_curves(model_df: pd.DataFrame, model: str, optimizer: str, precision: str, batch_size: str, out_dir: Path, global_min_vram: float, global_max_vram: float) -> None:
    model_df = model_df.sort_values(by=["gpu_budget_mb"], kind="stable")

    x = model_df["gpu_budget_mb"].astype(float)
    y_ilp = model_df["ilp_objective"].astype(float)
    y_cpu = model_df["all_cpu_objective"].astype(float)
    y_greedy = model_df["greedy_objective"].astype(float) if "greedy_objective" in model_df.columns else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Use seaborn lineplot for better default aesthetics without error bands
    sns.lineplot(x=x, y=y_ilp, marker="o", markersize=8, linewidth=2.5, label="ILP Optimal", ax=ax, color=sns.color_palette()[0], errorbar=None)
    
    if y_greedy is not None:
        sns.lineplot(x=x, y=y_greedy, linestyle="-.", linewidth=2.0, label="Greedy Heuristic", ax=ax, color=sns.color_palette()[1], errorbar=None, marker='^', markersize=6)

    # Detect RAPL counter wrap-around: CPU ran (feasible) but energy counter overflowed
    # (32-bit RAPL register on AMD EPYC wraps after ~21 s at full load, returning 0.0).
    _cpu_energy_zero = (model_df["all_cpu_energy_j"].astype(float) == 0.0).all() if "all_cpu_energy_j" in model_df.columns else False
    _cpu_feasible = (model_df["all_cpu_status"].astype(str) == "feasible").any() if "all_cpu_status" in model_df.columns else False
    cpu_latency_label = "All CPU (RAPL Counter Overflow)" if (_cpu_energy_zero and _cpu_feasible) else "All CPU"
    sns.lineplot(x=x, y=y_cpu, linestyle="--", linewidth=2.0, label=cpu_latency_label, ax=ax, color=sns.color_palette()[2], errorbar=None, marker='o', markersize=6)

    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.yaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # All GPU is a single physical configuration, so it acts as a baseline starting from its memory requirement
    finite_gpu = model_df[model_df["all_gpu_status"] != "infeasible"]

    if len(finite_gpu) > 0:
        all_gpu_val = float(finite_gpu["all_gpu_objective"].iloc[0])
        all_gpu_mem = float(finite_gpu["all_gpu_gpu_mem_mb"].iloc[0])
        line_start = min(all_gpu_mem, global_max_vram)
        
        # Plot horizontal line from required memory to max VRAM
        ax.hlines(
            y=all_gpu_val,
            xmin=line_start,
            xmax=global_max_vram,
            linestyle="-",
            linewidth=2.0,
            label=f"All GPU (\u2265 {all_gpu_mem:.0f} MB)",
            color=sns.color_palette()[3]
        )
    else:
        ax.plot([], [], linestyle="None", marker="s", color=sns.color_palette()[3], label="All GPU (Out of Memory)")

    ax.set_title(f"{model} (Batch {batch_size}): Execution Latency vs GPU Memory Budget", pad=15, fontweight="bold")
    ax.set_xlabel("GPU Memory Budget (MB)", fontweight="bold")
    ax.set_ylabel("Total Latency (ms)", fontweight="bold")
    ax.set_xlim(left=global_min_vram, right=global_max_vram)
    ax.legend(title="Execution Strategy", frameon=True, loc="center right")
    sns.despine(left=True, bottom=True)

    # Note if it's a flat line
    if len(y_ilp) > 1 and y_ilp.max() - y_ilp.min() < 1e-6:
        ax.text(0.5, -0.15, "Note: Flat curve indicates optimal partition fits within minimum tested budget", 
                transform=ax.transAxes, ha='center', va='top', fontsize=10, color='gray')

    plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "cost_vs_budget"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"execution_cost_vs_budget_{optimizer}_{precision}_batch_{batch_size}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_model_comparisons(best_df: pd.DataFrame, out_dir: Path) -> None:
    if best_df.empty:
        return

    for _, row in best_df.iterrows():
        model = row["model"]
        batch_size = str(row.get("batch_size", "unknown"))
        optimizer = str(row.get("optimizer", "unknown"))
        precision = str(row.get("precision", "unknown"))
        
        data = []
        cpu_val = float(row["all_cpu_objective"])
        data.append({"Strategy": "All-CPU", "Cost": cpu_val, "Is_OOM": False, "Is_Overflow": (cpu_val == 0.0)})
        
        # Check if All-GPU is feasible
        all_gpu_val = pd.to_numeric(row["all_gpu_objective"], errors="coerce")
        is_all_gpu_oom = row.get("all_gpu_status") == "infeasible" or pd.isna(all_gpu_val) or all_gpu_val == float("inf")
        if is_all_gpu_oom:
            data.append({"Strategy": "All-GPU", "Cost": 0, "Is_OOM": True, "Is_Overflow": False})
        else:
            gpu_c = float(all_gpu_val)
            data.append({"Strategy": "All-GPU", "Cost": gpu_c, "Is_OOM": False, "Is_Overflow": (gpu_c == 0.0)})
            
        if "greedy_objective" in row:
            greedy_val = pd.to_numeric(row["greedy_objective"], errors="coerce")
            is_greedy_oom = row.get("greedy_status") == "infeasible" or pd.isna(greedy_val) or greedy_val == float("inf")
            if is_greedy_oom:
                data.append({"Strategy": "Greedy", "Cost": 0, "Is_OOM": True, "Is_Overflow": False})
            else:
                gc = float(greedy_val)
                data.append({"Strategy": "Greedy", "Cost": gc, "Is_OOM": False, "Is_Overflow": (gc == 0.0)})
            
        ilp_c = float(row["ilp_objective"])
        data.append({"Strategy": "ILP Optimal", "Cost": ilp_c, "Is_OOM": False, "Is_Overflow": (ilp_c == 0.0)})
        
        plot_df = pd.DataFrame(data)
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.yaxis.set_major_locator(plt.MaxNLocator(15))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot ALL strategies to ensure x-axis ticks exist
        sns.barplot(data=plot_df, x="Strategy", y="Cost", ax=ax, palette="viridis")
        
        # Determine appropriate positive y-limits
        valid_costs = plot_df[~plot_df["Is_OOM"] & ~plot_df["Is_Overflow"]]["Cost"]
        max_valid = valid_costs.max() if not valid_costs.empty and valid_costs.max() > 0 else 1000.0
        placeholder_height = max_valid * 0.9 if max_valid > 1 else 1000.0
        ax.set_ylim(0, max_valid * 1.15 if max_valid > 1 else 1150.0)
        
        # Now find the OOM/Overflow strategies and draw over their bars
        strategies = plot_df["Strategy"].tolist()
        for i, patch in enumerate(ax.patches):
            if i < len(strategies):
                strategy = strategies[i]
                row_data = plot_df[plot_df["Strategy"] == strategy].iloc[0]
                is_oom = bool(row_data.get("Is_OOM", False))
                is_overflow = bool(row_data.get("Is_Overflow", False))
                
                if is_oom:
                    patch.set_height(placeholder_height)
                    patch.set_facecolor('dimgrey')
                    patch.set_hatch('///')
                    patch.set_edgecolor('black')
                    ax.text(patch.get_x() + patch.get_width() / 2., placeholder_height * 0.5, 
                            "Out of Memory (OOM)", color="black", ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)
                elif is_overflow:
                    patch.set_height(placeholder_height)
                    patch.set_facecolor('coral')
                    patch.set_hatch('xx')
                    patch.set_edgecolor('black')
                    ax.text(patch.get_x() + patch.get_width() / 2., placeholder_height * 0.5, 
                            "RAPL Counter Overflow (>21s)", color="black", ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)
        
        # Add a horizontal line for the Pareto ceiling (ILP Optimal cost)
        ilp_cost = float(row["ilp_objective"])
        if ilp_cost > 0:
            ax.axhline(y=ilp_cost, color='r', linestyle='--', linewidth=2, label='Pareto Boundary (ILP Ceiling)')
        
        ax.set_title(f"Execution Latency Comparison: {model} (Batch {batch_size})", pad=15, fontweight="bold")
        ax.set_ylabel("Total Latency (ms)", fontweight="bold")
        ax.set_xlabel("Execution Strategy", fontweight="bold")
        
        # Add data labels (skip OOM/Overflow bars)
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            if i < len(strategies):
                row_data = plot_df[plot_df["Strategy"] == strategies[i]].iloc[0]
                is_special = bool(row_data.get("Is_OOM", False)) or bool(row_data.get("Is_Overflow", False))
            else:
                is_special = False
            if height > 0 and not is_special:
                ax.annotate(f"{height:.2f}", 
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="center", xytext=(0, 8), textcoords="offset points",
                            fontsize=10, fontweight="bold")
        
        import matplotlib.patches as mpatches
        handles, labels = ax.get_legend_handles_labels()
        if plot_df["Is_OOM"].any():
            oom_patch = mpatches.Patch(facecolor='dimgrey', hatch='///', edgecolor='black', label='Out of Memory')
            if 'Out of Memory' not in labels:
                handles.append(oom_patch)
                labels.append('Out of Memory')
        if plot_df["Is_Overflow"].any():
            of_patch = mpatches.Patch(facecolor='coral', hatch='xx', edgecolor='black', label='RAPL Overflow (>21s)')
            if 'RAPL Overflow (>21s)' not in labels:
                handles.append(of_patch)
                labels.append('RAPL Overflow (>21s)')
        
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        sns.despine()

        plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "comparisons"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / f"strategy_comparison_{optimizer}_{precision}_batch_{batch_size}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _write_markdown_summary(
    full_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_path: Path,
    hybrid_best_df: pd.DataFrame | None = None,
) -> None:
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
            "batch_size",
            "gpu_budget_mb",
            "ilp_objective",
            "greedy_objective",
            "ilp_gpu_mem_mb",
            "ilp_cpu_mem_mb",
            "ilp_layers_gpu",
            "ilp_layers_cpu",
            "all_cpu_objective",
            "all_gpu_status",
        ]
        display_cols = [c for c in display_cols if c in best_df.columns]
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
            with pd.option_context("display.max_columns", None, "display.width", 2000):
                lines.append(table.to_string(index=False))
            lines.append("```")
        lines.append("")

    if hybrid_best_df is not None and not hybrid_best_df.empty:
        lines.append("## Best Observed Hybrid Runtime Per Model")
        hybrid_cols = [
            "model",
            "config_optimizer",
            "config_precision",
            "config_batch_size",
            "plan_selection_mode",
            "plan_gpu_budget_mb",
            "avg_step_ms",
            "final_loss",
            "quality_metric_name",
            "final_quality_metric",
            "dataset_name",
            "input_source",
            "target_source",
        ]
        available = [col for col in hybrid_cols if col in hybrid_best_df.columns]
        hybrid_table = hybrid_best_df[available].copy()
        try:
            lines.append(hybrid_table.to_markdown(index=False))
        except Exception:
            lines.append("```text")
            with pd.option_context("display.max_columns", None, "display.width", 2000):
                lines.append(hybrid_table.to_string(index=False))
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


def _plot_model_energy_comparisons(best_df: pd.DataFrame, out_dir: Path) -> None:
    if best_df.empty or "ilp_energy_j" not in best_df.columns:
        return

    for _, row in best_df.iterrows():
        model = row["model"]
        batch_size = str(row.get("batch_size", "unknown"))
        optimizer = str(row.get("optimizer", "unknown"))
        precision = str(row.get("precision", "unknown"))
        
        data = []
        if "all_cpu_energy_j" in row and not pd.isna(row["all_cpu_energy_j"]):
            cpu_e = float(row["all_cpu_energy_j"])
            data.append({"Strategy": "All-CPU", "Energy": cpu_e, "Is_OOM": False, "Is_Overflow": (cpu_e == 0.0)})
        # Check if All-GPU is feasible
        all_gpu_val = pd.to_numeric(row.get("all_gpu_energy_j", float("inf")), errors="coerce")
        is_all_gpu_oom = row.get("all_gpu_status") == "infeasible" or pd.isna(all_gpu_val) or all_gpu_val == float("inf")
        if is_all_gpu_oom:
            data.append({"Strategy": "All-GPU", "Energy": 0, "Is_OOM": True, "Is_Overflow": False})
        else:
            gpu_e = float(all_gpu_val)
            data.append({"Strategy": "All-GPU", "Energy": gpu_e, "Is_OOM": False, "Is_Overflow": (gpu_e == 0.0)})
            
        if "greedy_energy_j" in row:
            greedy_val = pd.to_numeric(row.get("greedy_energy_j", float("inf")), errors="coerce")
            is_greedy_oom = row.get("greedy_status") == "infeasible" or pd.isna(greedy_val) or greedy_val == float("inf")
            if is_greedy_oom:
                data.append({"Strategy": "Greedy", "Energy": 0, "Is_OOM": True, "Is_Overflow": False})
            else:
                ge = float(greedy_val)
                data.append({"Strategy": "Greedy", "Energy": ge, "Is_OOM": False, "Is_Overflow": (ge == 0.0)})
            
        if "ilp_energy_j" in row and not pd.isna(row["ilp_energy_j"]):
            ilp_e = float(row["ilp_energy_j"])
            data.append({"Strategy": "ILP Optimal", "Energy": ilp_e, "Is_OOM": False, "Is_Overflow": (ilp_e == 0.0)})
        
        plot_df = pd.DataFrame(data)
        if plot_df.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.yaxis.set_major_locator(plt.MaxNLocator(15))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot ALL strategies to ensure x-axis ticks exist
        sns.barplot(data=plot_df, x="Strategy", y="Energy", ax=ax, palette="plasma")
        
        # Determine appropriate positive y-limits
        valid_energy = plot_df[~plot_df["Is_OOM"] & ~plot_df["Is_Overflow"]]["Energy"]
        max_valid = valid_energy.max() if not valid_energy.empty and valid_energy.max() > 0 else 1000.0
        placeholder_height = max_valid * 0.9 if max_valid > 1 else 1000.0
        ax.set_ylim(0, max_valid * 1.15 if max_valid > 1 else 1150.0)
        
        # Now find the OOM/Overflow strategies and draw over their bars
        strategies = plot_df["Strategy"].tolist()
        for i, patch in enumerate(ax.patches):
            if i < len(strategies):
                strategy = strategies[i]
                row_data = plot_df[plot_df["Strategy"] == strategy].iloc[0]
                is_oom = bool(row_data.get("Is_OOM", False))
                is_overflow = bool(row_data.get("Is_Overflow", False))
                
                if is_oom:
                    patch.set_height(placeholder_height)
                    patch.set_facecolor('dimgrey')
                    patch.set_hatch('///')
                    patch.set_edgecolor('black')
                    ax.text(patch.get_x() + patch.get_width() / 2., placeholder_height * 0.5, 
                            "Out of Memory (OOM)", color="black", ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)
                elif is_overflow:
                    patch.set_height(placeholder_height)
                    patch.set_facecolor('coral')
                    patch.set_hatch('xx')
                    patch.set_edgecolor('black')
                    ax.text(patch.get_x() + patch.get_width() / 2., placeholder_height * 0.5, 
                            "RAPL Counter Overflow (>21s)", color="black", ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)
                                    
        # Add a horizontal line for the Pareto ceiling (ILP Optimal cost)
        ilp_energy = float(row["ilp_energy_j"]) if "ilp_energy_j" in row and not pd.isna(row["ilp_energy_j"]) else 0
        if ilp_energy > 0:
            ax.axhline(y=ilp_energy, color='r', linestyle='--', linewidth=2, label='Pareto Boundary (ILP Ceiling)')
        
        ax.set_title(f"Energy Consumption Comparison: {model} (Batch {batch_size})", pad=15, fontweight="bold")
        ax.set_ylabel("Total Energy (Joules)", fontweight="bold")
        ax.set_xlabel("Execution Strategy", fontweight="bold")
        
        import matplotlib.patches as mpatches
        handles, labels = ax.get_legend_handles_labels()
        if plot_df["Is_OOM"].any():
            oom_patch = mpatches.Patch(facecolor='dimgrey', hatch='///', edgecolor='black', label='Out of Memory')
            if 'Out of Memory' not in labels:
                handles.append(oom_patch)
                labels.append('Out of Memory')
        if plot_df["Is_Overflow"].any():
            of_patch = mpatches.Patch(facecolor='coral', hatch='xx', edgecolor='black', label='RAPL Overflow (>21s)')
            if 'RAPL Overflow (>21s)' not in labels:
                handles.append(of_patch)
                labels.append('RAPL Overflow (>21s)')
        
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        sns.despine()

        plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "comparisons"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / f"energy_comparison_{optimizer}_{precision}_batch_{batch_size}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_memory_utilization_bar(metrics_csv: Path, model: str, optimizer: str, precision: str, batch_size: str, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    if "layer" not in df.columns or "gpu_mem_peak_mb_mean" not in df.columns:
        return
        
    gpu_total = df["gpu_mem_peak_mb_mean"].sum() if "gpu_mem_peak_mb_mean" in df.columns else 0.0
    cpu_total = df["cpu_mem_mb_mean"].sum() if "cpu_mem_mb_mean" in df.columns else 0.0
    
    if gpu_total == 0 and cpu_total == 0:
        return
        
    data = [
        {"Memory Domain": "All-GPU VRAM Peak Demand", "Size (MB)": gpu_total},
        {"Memory Domain": "All-CPU DRAM Peak Demand", "Size (MB)": cpu_total}
    ]
    plot_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_df, x="Memory Domain", y="Size (MB)", ax=ax, palette="magma")
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    ax.set_title(f"Model Memory Footprint Profiling\n(Peak Demand without ILP Partitioning)\n{model} (Batch {batch_size})", pad=15, fontweight="bold")
    ax.set_ylabel("Memory Size (MB)", fontweight="bold")
    
    # Add an explicit note at the bottom
    ax.text(0.5, -0.2, "Note: This represents the peak demand if the model executed 100% on the respective device.",
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color='gray')
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f} MB", 
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold")
    if gpu_total == 0 and cpu_total > 0:
        ax.text(0.5, 0.95, "[!] Measured on CPU due to GPU OOM", 
                transform=ax.transAxes, ha='center', va='top', fontsize=10, 
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'), fontweight="bold")
                    
    sns.despine()
    
    plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "memory"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"memory_utilization_{optimizer}_{precision}_batch_{batch_size}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_top15_memory_layers(metrics_csv: Path, model: str, optimizer: str, precision: str, batch_size: str, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    if "layer" not in df.columns or "gpu_mem_peak_mb_mean" not in df.columns:
        return
        
    gpu_sum = df["gpu_mem_peak_mb_mean"].sum()
    cpu_sum = df["cpu_mem_mb_mean"].sum() if "cpu_mem_mb_mean" in df.columns else 0.0
    
    use_cpu = False
    metric_col = "gpu_mem_peak_mb_mean"
    x_label = "GPU Memory Peak (MB)"
    if gpu_sum == 0 and cpu_sum > 0:
        use_cpu = True
        metric_col = "cpu_mem_mb_mean"
        x_label = "CPU Memory Peak (MB)"
        
    top_layers = df.sort_values(by=metric_col, ascending=False).head(15).copy()
    if top_layers.empty or top_layers[metric_col].sum() == 0:
        return
        
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_layers, x=metric_col, y="layer", ax=ax, palette="crest")
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)

    ax.set_title(f"Top 15 Memory-Intensive Layers: {model} (Batch {batch_size})", pad=15, fontweight="bold")
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    
    if use_cpu:
        ax.text(0.5, 0.05, "[!] Measured on CPU due to GPU OOM", 
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'), fontweight="bold")
    
    sns.despine()
    
    plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "memory"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"top15_memory_layers_{optimizer}_{precision}_batch_{batch_size}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    # Normalize rescue_influence columns (backward compatibility)
    full_df = _normalize_rescue_columns(full_df)

    # Extract info from source_csv: e.g. .../resnet50/AdamW/fp32/batch_32/...
    full_df["batch_size"] = full_df["source_csv"].str.extract(r'(batch_\d+)')
    # If regex captures 'batch_32', map to just '32'
    full_df["batch_size"] = full_df["batch_size"].str.replace("batch_", "")
    
    full_df["precision"] = full_df["source_csv"].str.extract(r'/(fp16|fp32|bf16|int8)/')
    full_df["optimizer"] = full_df["source_csv"].str.extract(r'/([^/]+)/(?:fp16|fp32|bf16|int8)/')

    # Fill NaNs with 'unknown' to avoid groupby dropping rows
    full_df["precision"] = full_df["precision"].fillna("unknown")
    full_df["optimizer"] = full_df["optimizer"].fillna("unknown")
    full_df["batch_size"] = full_df["batch_size"].fillna("unknown")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    consolidated_csv = csv_dir / "ilp_pareto_consolidated.csv"
    full_df.to_csv(consolidated_csv, index=False)

    best_df = _best_feasible_rows(full_df)
    best_csv = csv_dir / "ilp_best_per_model.csv"
    best_df.to_csv(best_csv, index=False)

    hybrid_files = _find_hybrid_protocol_files(input_root)
    hybrid_best_df = pd.DataFrame()
    if hybrid_files:
        hybrid_frames = []
        for p in hybrid_files:
            hdf = pd.read_csv(p)
            hdf["source_csv"] = str(p)
            hybrid_frames.append(hdf)
        hybrid_df = pd.concat(hybrid_frames, ignore_index=True)
        # Normalize rescue_influence columns in hybrid data
        hybrid_df = _normalize_rescue_columns(hybrid_df)
        hybrid_csv = csv_dir / "hybrid_execution_consolidated.csv"
        hybrid_df.to_csv(hybrid_csv, index=False)
        hybrid_best_df = _best_hybrid_rows(hybrid_df)
        hybrid_best_csv = csv_dir / "hybrid_execution_best_per_model.csv"
        hybrid_best_df.to_csv(hybrid_best_csv, index=False)

    base_min_vram = float(full_df["gpu_budget_mb"].min())
    base_max_vram = float(full_df["gpu_budget_mb"].max())

    # Memory-budget plots must be scaled by tested budgets, not by theoretical
    # all-GPU demand, which can exceed physical VRAM and distort the x-axis.
    detected_vram_total_mb = _detect_gpu_vram_total_mb(input_root)
    global_min_vram = max(0.0, base_min_vram * 0.95)
    global_max_vram = base_max_vram * 1.05
    if detected_vram_total_mb is not None:
        global_max_vram = min(global_max_vram, detected_vram_total_mb * 1.02)
    
    unique_seeds = full_df["source_csv"].str.extract(r'/(seed_\d+)/')[0].dropna().unique()
    is_multiseed_root = len(unique_seeds) > 1

    if not is_multiseed_root:
        for model in sorted(full_df["model"].astype(str).unique().tolist()):
            model_data = full_df[full_df["model"] == model]
            for optimizer in sorted(model_data["optimizer"].astype(str).unique().tolist()):
                opt_data = model_data[model_data["optimizer"] == optimizer]
                for precision in sorted(opt_data["precision"].astype(str).unique().tolist()):
                    prec_data = opt_data[opt_data["precision"] == precision]
                    for batch_size in sorted(prec_data["batch_size"].astype(str).unique().tolist()):
                        batch_data = prec_data[prec_data["batch_size"] == batch_size]
                        if not batch_data.empty:
                            _plot_model_objective_curves(batch_data, model, optimizer, precision, batch_size, out_dir, global_min_vram, global_max_vram)
                            _plot_model_energy_curves(batch_data, model, optimizer, precision, batch_size, out_dir, global_min_vram, global_max_vram)
                            
                            source_csv = Path(batch_data.iloc[0]["source_csv"])
                            config_dir = source_csv.parent
                            metrics_stats_csv = config_dir / f"{model}_metrics_stats.csv"
                            if metrics_stats_csv.exists():
                                _plot_memory_utilization_bar(metrics_stats_csv, model, optimizer, precision, batch_size, out_dir)
                                _plot_top15_memory_layers(metrics_stats_csv, model, optimizer, precision, batch_size, out_dir)

        _plot_model_comparisons(best_df, out_dir)
        _plot_model_energy_comparisons(best_df, out_dir)

    md_dir = out_dir / "summary_docs"
    md_dir.mkdir(parents=True, exist_ok=True)
    
    md_summary = md_dir / "ILP_RESULTS_SUMMARY.md"
    _write_markdown_summary(full_df, best_df, md_summary, hybrid_best_df=hybrid_best_df)

    ablation_files = _find_ablation_files(input_root)
    if ablation_files:
        ablation_frames = []
        for p in ablation_files:
            adf = pd.read_csv(p)
            adf["source_csv"] = str(p)
            ablation_frames.append(adf)

        ablation_df = pd.concat(ablation_frames, ignore_index=True)
        ablation_csv = csv_dir / "ilp_ablation_consolidated.csv"
        ablation_df.to_csv(ablation_csv, index=False)

        ablation_md = md_dir / "ILP_ABLATION_SUMMARY.md"
        _write_ablation_markdown_summary(ablation_df, ablation_md)

    sensitivity_files = _find_sensitivity_files(input_root)
    if sensitivity_files:
        sens_frames = []
        for p in sensitivity_files:
            sdf = pd.read_csv(p)
            sdf["source_csv"] = str(p)
            sens_frames.append(sdf)

        sensitivity_df = pd.concat(sens_frames, ignore_index=True)
        sensitivity_csv = csv_dir / "ilp_sensitivity_consolidated.csv"
        sensitivity_df.to_csv(sensitivity_csv, index=False)

        sensitivity_md = md_dir / "ILP_SENSITIVITY_SUMMARY.md"
        _write_sensitivity_markdown_summary(sensitivity_df, sensitivity_md)

    print("================================================================================")
    print("ILP REPORT ASSETS GENERATED")
    print("================================================================================")
    print(f"Input Pareto files: {len(pareto_files)}")
    if detected_vram_total_mb is not None:
        print(f"Detected GPU VRAM total (MB): {detected_vram_total_mb:.1f}")
    print(f"Budget axis limits (MB): [{global_min_vram:.1f}, {global_max_vram:.1f}]")
    print(f"Consolidated CSV: {consolidated_csv}")
    print(f"Best-per-model CSV: {best_csv}")
    print(f"Markdown summary: {md_summary}")
    if hybrid_files:
        print(f"Hybrid protocol files: {len(hybrid_files)}")
        print(f"Hybrid consolidated CSV: {csv_dir / 'hybrid_execution_consolidated.csv'}")
        print(f"Hybrid best-per-model CSV: {csv_dir / 'hybrid_execution_best_per_model.csv'}")
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



def _plot_model_energy_curves(model_df: pd.DataFrame, model: str, optimizer: str, precision: str, batch_size: str, out_dir: Path, global_min_vram: float, global_max_vram: float) -> None:
    model_df = model_df.sort_values(by=["gpu_budget_mb"], kind="stable")

    if "ilp_energy_j" not in model_df.columns:
        return

    x = model_df["gpu_budget_mb"].astype(float)
    y_ilp = model_df["ilp_energy_j"].astype(float)
    y_cpu = model_df["all_cpu_energy_j"].astype(float)
    y_greedy = model_df["greedy_energy_j"].astype(float) if "greedy_energy_j" in model_df.columns else None

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.lineplot(x=x, y=y_ilp, marker="o", markersize=8, linewidth=2.5, label="ILP Optimal", ax=ax, color=sns.color_palette()[0], errorbar=None)

    if y_greedy is not None:
        sns.lineplot(x=x, y=y_greedy, linestyle="-.", linewidth=2.0, label="Greedy Heuristic", ax=ax, color=sns.color_palette()[1], errorbar=None, marker='^', markersize=6)

    # Detect RAPL counter wrap-around: CPU ran (feasible) but the 32-bit hardware
    # energy register overflowed (AMD EPYC 7513 wraps after ~21 s at full load,
    # causing pyRAPL to return 0.0 instead of the real measured Joules).
    _cpu_energy_zero = (y_cpu == 0.0).all()
    _cpu_feasible = (model_df["all_cpu_status"].astype(str) == "feasible").any() if "all_cpu_status" in model_df.columns else False
    cpu_energy_label = "All CPU (RAPL Counter Overflow)" if (_cpu_energy_zero and _cpu_feasible) else "All CPU"
    sns.lineplot(x=x, y=y_cpu, linestyle="--", linewidth=2.0, label=cpu_energy_label, ax=ax, color=sns.color_palette()[2], errorbar=None, marker='o', markersize=6)

    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.yaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot All GPU only where objective was feasible    # All GPU is a single physical configuration
    finite_gpu = model_df[model_df["all_gpu_status"] != "infeasible"]

    if len(finite_gpu) > 0:
        all_gpu_energy_val = float(finite_gpu["all_gpu_energy_j"].iloc[0])
        all_gpu_mem = float(finite_gpu["all_gpu_gpu_mem_mb"].iloc[0])
        line_start = min(all_gpu_mem, global_max_vram)
        
        ax.hlines(
            y=all_gpu_energy_val,
            xmin=line_start,
            xmax=global_max_vram,
            linestyle="-",
            linewidth=2.0,
            label=f"All GPU (\u2265 {all_gpu_mem:.0f} MB)",
            color=sns.color_palette()[3]
        )
    else:
        ax.plot([], [], linestyle="None", marker="s", color=sns.color_palette()[3], label="All GPU (Out of Memory)")

    ax.set_title(f"{model} (Batch {batch_size}): Energy Consumption vs GPU Memory Budget", pad=15, fontweight="bold")
    ax.set_xlabel("GPU Memory Budget (MB)", fontweight="bold")
    ax.set_ylabel("Total Energy (Joules)", fontweight="bold")
    ax.set_xlim(left=global_min_vram, right=global_max_vram)
    ax.legend(title="Execution Strategy", frameon=True, loc="center right")
    sns.despine(left=True, bottom=True)

    plot_dir = out_dir / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "energy_vs_budget"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"execution_energy_vs_budget_{optimizer}_{precision}_batch_{batch_size}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
if __name__ == "__main__":
    raise SystemExit(main())
