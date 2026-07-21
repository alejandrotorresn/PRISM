#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional Plotting Aesthetics
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

VALID_MODELS = ["distilgpt2", "gpt2_small", "resnet50", "resnet152", "vit_b16"]

def plot_statistical_validity(sig_csv_path: Path, output_dir: Path):
    if not sig_csv_path.exists():
        print(f"File not found: {sig_csv_path}")
        return

    df = pd.read_csv(sig_csv_path)
    # Filter out simple_mlp or any model not in VALID_MODELS
    df = df[df["model"].isin(VALID_MODELS)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cohen's d
    df_plot = df.dropna(subset=["cohens_d_vs_cpu"]).copy()
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_plot, x="model", y="cohens_d_vs_cpu", hue="model", palette="viridis", ax=ax, legend=False, order=VALID_MODELS)
        ax.set_title("Effect Size (Cohen's d): PRISM vs. CPU Baseline", pad=15, fontweight="bold")
        ax.set_xlabel("Deep Neural Architecture", fontweight="bold")
        ax.set_ylabel("Cohen's d (Standardized Improvement)", fontweight="bold")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(VALID_MODELS)))
        ax.set_xticklabels(VALID_MODELS, rotation=20, ha='right')
        sns.despine()
        fig.tight_layout()
        fig.savefig(output_dir / "cohens_d_effect_size_vs_cpu.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Generated cohens_d_effect_size_vs_cpu.png (English, 5 models)")

    # 2. p-value
    df_p = df.dropna(subset=["p_value_vs_cpu"]).copy()
    if not df_p.empty:
        df_p["neg_log_p"] = -np.log10(df_p["p_value_vs_cpu"] + 1e-15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_p, x="model", y="neg_log_p", hue="model", palette="magma", ax=ax, legend=False, order=VALID_MODELS)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=2, label="Significance Threshold (p=0.05)")
        ax.axhline(-np.log10(0.001), color="orange", linestyle=":", linewidth=2, label="Strict Doctoral Significance (p=0.001)")
        ax.set_title("Statistical Significance (-log10 p-value): PRISM vs. CPU Baseline", pad=15, fontweight="bold")
        ax.set_xlabel("Deep Neural Architecture", fontweight="bold")
        ax.set_ylabel("-log10(p-value)", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(VALID_MODELS)))
        ax.set_xticklabels(VALID_MODELS, rotation=20, ha='right')
        sns.despine()
        fig.tight_layout()
        fig.savefig(output_dir / "p_value_significance_vs_cpu.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Generated p_value_significance_vs_cpu.png (English, 5 models)")

def plot_global_predicted_vs_actual(hybrid_csv_paths: list, output_dir: Path):
    dfs = []
    for p in hybrid_csv_paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print(f"Warning: path {p} not found during loading")
            
    if not dfs:
        print("No hybrid CSV files found!")
        return

    df = pd.concat(dfs, ignore_index=True)
    predicted_col = "plan_objective" if "plan_objective" in df.columns else None
    measured_col = "avg_step_ms" if "avg_step_ms" in df.columns else None
    model_col = "config_model" if "config_model" in df.columns else ("model" if "model" in df.columns else None)

    if predicted_col is None or measured_col is None or model_col is None:
        print("Missing columns in hybrid csv")
        return

    df_valid = df[
        (pd.to_numeric(df[predicted_col], errors="coerce") > 0)
        & (df["run_label"] != "all_cpu")
        & (df["run_label"] != "all_gpu")
    ].copy()

    # STRICTLY FILTER OUT simple_mlp AND KEEP ALL 5 DEEP MODELS
    df_valid = df_valid[df_valid[model_col].isin(VALID_MODELS)].copy()

    df_valid[predicted_col] = pd.to_numeric(df_valid[predicted_col], errors="coerce")
    df_valid[measured_col] = pd.to_numeric(df_valid[measured_col], errors="coerce")
    df_valid = df_valid.dropna(subset=[predicted_col, measured_col])

    if df_valid.empty:
        print("No valid rows remaining")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter Plot (Global Density)
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(df_valid) > 1:
        from scipy.stats import pearsonr
        r_val, _ = pearsonr(df_valid[predicted_col], df_valid[measured_col])
        r2 = r_val ** 2
        title_extra = f" (Global $R^2$ = {r2:.3f})"
    else:
        title_extra = ""

    models = [m for m in VALID_MODELS if m in df_valid[model_col].unique()]
    palette = sns.color_palette("colorblind", n_colors=len(models))
    color_map = {m: palette[i] for i, m in enumerate(models)}

    for m in models:
        subset = df_valid[df_valid[model_col] == m]
        ax.scatter(subset[predicted_col], subset[measured_col],
                   color=color_map[m], label=m, s=60, alpha=0.8, edgecolors='k', linewidths=0.5, zorder=2)

    min_val = min(df_valid[predicted_col].min(), df_valid[measured_col].min())
    max_val = max(df_valid[predicted_col].max(), df_valid[measured_col].max())
    padding = (max_val - min_val) * 0.05 if max_val > min_val else min_val * 0.05
    ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', linewidth=2, label='Ideal Perfect Prediction (y=x)')

    ax.set_title(f"Global Correlation: Predicted vs. Hardware Measured Latency{title_extra}", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted ILP Latency (ms)", fontweight="bold")
    ax.set_ylabel("Hardware Measured Latency (ms)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "global_predicted_vs_actual_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated global_predicted_vs_actual_density.png (English, all 5 models)")

    # 2. FacetGrid by Model (strictly 5 models)
    g = sns.FacetGrid(df_valid, col=model_col, col_order=models, col_wrap=3, height=4.2, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x=predicted_col, y=measured_col, alpha=0.8, s=50, color="#1f77b4", edgecolor="k", linewidth=0.3)
    
    def plot_ideal_line(**kwargs):
        ax = plt.gca()
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.8, zorder=0, label="Ideal Prediction (y=x)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
    g.map(plot_ideal_line)
    g.set_axis_labels("Predicted ILP Latency (ms)", "Hardware Measured Latency (ms)")
    g.set_titles(col_template="{col_name}", size=13, weight="bold")
    leg = g.fig.legend(
        *g.axes[0].get_legend_handles_labels(),
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=12,
    )
    g.fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.38, wspace=0.25)
    suptitle = g.fig.suptitle("Predictive Fidelity Faceted by Neural Architecture", fontweight="bold", fontsize=16, y=1.03)
    g.savefig(output_dir / "global_predicted_vs_actual_facets.png", dpi=300, bbox_inches="tight", bbox_extra_artists=(leg, suptitle))
    plt.close(g.fig)
    print("Generated global_predicted_vs_actual_facets.png (English, strictly 5 models)")

def main():
    base_dir = Path("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code")
    sig_csv = base_dir / "reports/chuc-4/doctoral_full/csv/ilp_statistical_significance.csv"
    
    # Load all full campaign CSVs across nodes to get the complete dataset for all 5 models
    hybrid_csvs = [
        base_dir / "reports/chuc-4/doctoral_full/csv/hybrid_execution_consolidated.csv",
        base_dir / "reports/kinovis-2/doctoral_full/csv/hybrid_execution_consolidated.csv",
        base_dir / "reports/paccaA100.unicartagena.edu.co/doctoral_full/csv/hybrid_execution_consolidated.csv"
    ]
    out_dir = base_dir / "final_thesis/figures/chapter6"

    print("Regenerating global and faceted plots (strictly English, 5 models) into:", out_dir)
    plot_statistical_validity(sig_csv, out_dir)
    plot_global_predicted_vs_actual(hybrid_csvs, out_dir)
    print("All global and faceted plots regenerated successfully!")

if __name__ == "__main__":
    main()
