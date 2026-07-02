#!/usr/bin/env python3
"""
generate_chapter6_energy_and_greedy_plots.py

Generates professional doctoral-grade figures for Chapter 6 (All text in English):
1. Energy consumption and savings (PRISM vs All-CPU vs Greedy).
2. Multi-dimensional performance comparison (PRISM vs Greedy across models, batch sizes, optimizers).
3. AOT Compilation & Runtime Overhead analysis.
"""

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
sns.set_context("paper", font_scale=1.3)
sns.set_palette("colorblind")

VALID_MODELS = ["distilgpt2", "gpt2_small", "resnet50", "resnet152", "vit_b16"]
MODEL_LABELS = {
    "distilgpt2": "DistilGPT-2",
    "gpt2_small": "GPT-2 Small",
    "resnet50": "ResNet-50",
    "resnet152": "ResNet-152",
    "vit_b16": "ViT-B/16"
}

def generate_energy_plots(df: pd.DataFrame, out_dir: Path):
    """Generates energy comparison plots across models and optimizers."""
    df_valid = df[df["model"].isin(VALID_MODELS)].copy()
    
    # Filter rows where energy measurements exist and make physical sense
    df_plot = df_valid[(df_valid["all_cpu_energy_j"] > 0) & (df_valid["ilp_energy_j"] > 0)].copy()
    df_plot["energy_save_vs_cpu"] = (df_plot["all_cpu_energy_j"] - df_plot["ilp_energy_j"]) / df_plot["all_cpu_energy_j"] * 100.0
    
    # 1. Bar Plot of % Energy Savings vs All-CPU by Model and Optimizer
    fig, ax = plt.subplots(figsize=(11, 6))
    df_summary = df_plot.groupby(["model", "optimizer"])["energy_save_vs_cpu"].mean().reset_index()
    df_summary["model_label"] = df_summary["model"].map(MODEL_LABELS)
    
    sns.barplot(
        data=df_summary,
        x="model_label",
        y="energy_save_vs_cpu",
        hue="optimizer",
        palette="viridis",
        edgecolor="black",
        linewidth=0.7,
        ax=ax
    )
    
    ax.set_title("Relative Energy Savings of PRISM vs. All-CPU Baseline across Models and Optimizers", pad=15, fontweight="bold", fontsize=14)
    ax.set_xlabel("Deep Neural Network Architecture", fontweight="bold", fontsize=12)
    ax.set_ylabel("Energy Consumption Reduction (%)", fontweight="bold", fontsize=12)
    ax.set_ylim(75, 100)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(title="Parametric Optimizer", title_fontsize="11", fontsize="10", loc="upper right", frameon=True)
    sns.despine()
    
    fig.tight_layout()
    fig.savefig(out_dir / "energy_savings_prism_vs_cpu.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated energy_savings_prism_vs_cpu.png")

    # 2. Absolute Energy Consumption per Step (Joules) under OOM conditions (Batch 256, AdamW, 20GB VRAM)
    # Load pareto consolidated dataset to capture sharp OOM conditions where Greedy experiences DMA cascades
    pareto_csv = Path("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/reports/chuc-4/doctoral_full/csv/ilp_pareto_consolidated.csv")
    if pareto_csv.exists():
        df_pareto = pd.read_csv(pareto_csv)
        df_b256 = df_pareto[
            (df_pareto["model"].isin(VALID_MODELS)) &
            (df_pareto["batch_size"] == 256) &
            (df_pareto["optimizer"] == "AdamW") &
            (df_pareto["precision"] == "fp32") &
            (df_pareto["gpu_budget_mb"] == 20480.0) &
            (df_pareto["all_cpu_energy_j"] > 0)
        ].drop_duplicates("model").copy()
    else:
        df_b256 = df_plot[(df_plot["batch_size"] == 128) & (df_plot["optimizer"] == "AdamW")].copy()

    if not df_b256.empty:
        df_melt = pd.melt(
            df_b256,
            id_vars=["model"],
            value_vars=["ilp_energy_j", "greedy_energy_j", "all_cpu_energy_j"],
            var_name="strategy",
            value_name="energy_j"
        )
        strategy_names = {
            "ilp_energy_j": "PRISM (Optimal ILP)",
            "greedy_energy_j": "Greedy Heuristic",
            "all_cpu_energy_j": "Monolithic Host (All-CPU)"
        }
        df_melt["strategy_label"] = df_melt["strategy"].map(strategy_names)
        df_melt["model_label"] = df_melt["model"].map(MODEL_LABELS)

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(
            data=df_melt,
            x="model_label",
            y="energy_j",
            hue="strategy_label",
            palette="magma",
            edgecolor="black",
            linewidth=0.7,
            ax=ax
        )
        ax.set_yscale("log")
        ax.set_title("Absolute Energy Consumption per Iteration ($B=256$, AdamW, 20 GB VRAM OOM Regime)", pad=15, fontweight="bold", fontsize=14)
        ax.set_xlabel("Deep Neural Network Architecture", fontweight="bold", fontsize=12)
        ax.set_ylabel("Energy Consumed per Step (Joules, log scale)", fontweight="bold", fontsize=12)
        ax.grid(True, axis='y', which="both", linestyle='--', alpha=0.5)
        
        # Legend placed in upper right as explicitly requested
        ax.legend(title="Dispatch Strategy", title_fontsize="11", fontsize="10", loc="upper right", frameon=True)
        sns.despine()
        
        fig.tight_layout()
        fig.savefig(out_dir / "energy_absolute_comparison_b128.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Generated energy_absolute_comparison_b128.png (OOM regime)")

def generate_multidim_greedy_plots(df: pd.DataFrame, out_dir: Path):
    """Generates PRISM vs Greedy speedup plots across models, batch sizes, and optimizers."""
    df_valid = df[df["model"].isin(VALID_MODELS)].copy()
    
    # Consider feasible rows where objectives are positive
    df_valid = df_valid[(df_valid["ilp_objective"] > 0) & (df_valid["greedy_objective"] > 0)].copy()
    df_valid["speedup_vs_greedy_pct"] = (df_valid["greedy_objective"] - df_valid["ilp_objective"]) / df_valid["greedy_objective"] * 100.0
    
    # Create line plot across Batch Size (8 to 512) for AdamW vs SGD
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for idx, opt in enumerate(["AdamW", "SGD"]):
        ax = axes[idx]
        df_opt = df_valid[df_valid["optimizer"] == opt].copy()
        df_opt["model_label"] = df_opt["model"].map(MODEL_LABELS)
        
        sns.lineplot(
            data=df_opt,
            x="batch_size",
            y="speedup_vs_greedy_pct",
            hue="model_label",
            style="model_label",
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=8,
            palette="colorblind",
            ax=ax
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f"Optimizer: {opt}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Batch Size ($B$, log scale)", fontweight="bold", fontsize=12)
        if idx == 0:
            ax.set_ylabel("PRISM Latency Improvement vs. Greedy (%)", fontweight="bold", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.5)
        ax.legend(title="Model", frameon=True, fontsize=10, loc="upper right")
        
    fig.suptitle("Multi-Dimensional Divergence: PRISM Speedup over Greedy Heuristic across Batch Size and Optimizers", fontweight="bold", fontsize=15, y=1.03)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "greedy_vs_prism_multidim.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated greedy_vs_prism_multidim.png")

def generate_overhead_breakdown_plot(out_dir: Path):
    """Generates AOT compilation vs cumulative execution time comparison with grouped bars."""
    models = ["DistilGPT-2", "ViT-B/16", "ResNet-50", "GPT-2 Small", "ResNet-152"]
    aot_compilation_s = [0.48, 0.76, 0.85, 0.92, 3.14]
    
    # Standard 5000 iterations training campaign execution times in seconds
    step_time_ms = [92.70, 140.18, 85.60, 149.17, 145.08]
    total_train_time_s = [t * 5000 / 1000.0 for t in step_time_ms]
    
    df_overhead = pd.DataFrame({
        "Model": models,
        "AOT Compilation (s)": aot_compilation_s,
        "Training Execution (5000 steps, s)": total_train_time_s
    })
    
    df_overhead["AOT Ratio (%)"] = df_overhead["AOT Compilation (s)"] / (df_overhead["AOT Compilation (s)"] + df_overhead["Training Execution (5000 steps, s)"]) * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 6))
    
    y = np.arange(len(models))
    height = 0.38
    
    # Plot side-by-side grouped horizontal bars starting explicitly from 0.05 on log scale
    ax.barh(y - height/2, df_overhead["AOT Compilation (s)"], height=height, color="#e74c3c", edgecolor="black", label="AOT Compilation Time (Profiling + ILP Solver)")
    ax.barh(y + height/2, df_overhead["Training Execution (5000 steps, s)"], height=height, color="#2ecc71", edgecolor="black", label="Training Campaign Execution (5,000 Iterations)")
    
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontweight="bold", fontsize=11)
    ax.set_xlabel("Time in Seconds (logarithmic scale)", fontweight="bold", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 2000)
    ax.set_title("Amortization of Pre-Execution AOT Compilation over 5,000-Step Training Campaign", fontweight="bold", fontsize=14, pad=15)
    ax.grid(True, axis='x', which="both", linestyle="--", alpha=0.5)
    
    # Legend outside the plot to avoid overlapping bars
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True, fontsize=11)
    
    # Annotate AOT time and percentage right next to the red bars
    for idx, row in df_overhead.iterrows():
        pct = row["AOT Ratio (%)"]
        aot_val = row["AOT Compilation (s)"]
        ax.text(aot_val * 1.3, idx - height/2, f"{aot_val:.2f}s ({pct:.2f}%)", va='center', color='black', fontweight='bold', fontsize=9.5)

    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / "prism_overhead_amortization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated prism_overhead_amortization.png")

def main():
    base_dir = Path("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code")
    csv_path = base_dir / "reports/chuc-4/doctoral_full/csv/ilp_best_per_model.csv"
    out_dir = base_dir / "final_thesis/figures/chapter6"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print("Generating Chapter 6 doctoral figures...")
    generate_energy_plots(df, out_dir)
    generate_multidim_greedy_plots(df, out_dir)
    generate_overhead_breakdown_plot(out_dir)
    print("All Chapter 6 doctoral figures generated successfully!")

if __name__ == "__main__":
    main()
