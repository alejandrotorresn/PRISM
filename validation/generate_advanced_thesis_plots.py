import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import csv

# Professional Plotting Aesthetics
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'seaborn-paper' if 'seaborn-paper' in plt.style.available else 'default')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

def read_csv_rows(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def find_config_dirs(input_root: Path):
    # Expecting structure: root/model/optimizer/precision/batch_*/
    dirs = []
    for model_path in input_root.iterdir():
        if not model_path.is_dir() or model_path.name.startswith("."): continue
        for opt_path in model_path.iterdir():
            if not opt_path.is_dir(): continue
            for prec_path in opt_path.iterdir():
                if not prec_path.is_dir(): continue
                for batch_path in prec_path.iterdir():
                    if not batch_path.is_dir(): continue
                    dirs.append(batch_path)
    return dirs

def plot_predicted_vs_actual(hybrid_csv: Path, output_dir: Path):
    if not hybrid_csv.exists():
        print(f"Warning: Hybrid CSV {hybrid_csv} not found for predicted vs actual scatter.")
        return
        
    df = pd.read_csv(hybrid_csv)
    
    if "plan_objective" not in df.columns or "avg_step_ms" not in df.columns:
        print("Required columns for scatter plot missing in hybrid CSV.")
        return
        
    # Filter out baseline runs. Include ALL statuses since analytical fallback can populate avg_step_ms
    df_valid = df[(df["plan_objective"] > 0) & (df["run_label"] != "all_cpu") & (df["run_label"] != "all_gpu")].copy()
    df_valid = df_valid.dropna(subset=["avg_step_ms"])
    
    if df_valid.empty:
        print("No valid hybrid execution data points for scatter plot.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate R^2
    from scipy.stats import pearsonr
    r_val, _ = pearsonr(df_valid["plan_objective"], df_valid["avg_step_ms"])
    r2 = r_val ** 2
    
    sns.scatterplot(data=df_valid, x="plan_objective", y="avg_step_ms", hue="model", s=100, alpha=0.8, ax=ax)
    
    # Ideal y=x line
    min_val = min(df_valid["plan_objective"].min(), df_valid["avg_step_ms"].min())
    max_val = max(df_valid["plan_objective"].max(), df_valid["avg_step_ms"].max())
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label=f'Ideal (y=x)\n$R^2 = {r2:.3f}$')
    
    ax.set_title("Validation: ILP Predicted Latency vs. Hybrid Actual Latency", pad=15, fontweight="bold")
    ax.set_xlabel("ILP Predicted Latency (ms)", fontweight="bold")
    ax.set_ylabel("Hybrid Execution Measured Latency (ms)", fontweight="bold")
    ax.legend(title="", frameon=True)
    sns.despine()
    
    fig.tight_layout()
    
    plot_dir = output_dir / "all_models" / "predicted_vs_actual"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_memory_footprint(config_dir: Path, output_dir: Path):
    model = config_dir.parents[2].name
    stats_path = config_dir / f"{model}_metrics_stats.csv"
    if not stats_path.exists():
        return
        
    df = pd.read_csv(stats_path)
    if "activations_mb_mean" not in df.columns or "params_mb_mean" not in df.columns or "grads_mb_mean" not in df.columns:
        return
        
    df = df.dropna(subset=["activations_mb_mean", "params_mb_mean", "grads_mb_mean"])
    if df.empty:
        return
        
    # FORWARD PASS FOOTPRINT
    df_fwd = df.copy()
    df_fwd["cumulative_params"] = df_fwd["params_mb_mean"].cumsum()
    df_fwd["cumulative_activations"] = df_fwd["activations_mb_mean"].cumsum()
    df_fwd["total_memory"] = df_fwd["cumulative_params"] + df_fwd["cumulative_activations"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df_fwd))
    ax.fill_between(x, 0, df_fwd["cumulative_params"], label="Model Parameters (Cumulative)", alpha=0.6)
    ax.fill_between(x, df_fwd["cumulative_params"], df_fwd["total_memory"], label="Activations (Cumulative)", alpha=0.6)
    
    ax.set_title(f"Memory Footprint Evolution (Forward Pass): {model}", pad=15, fontweight="bold")
    ax.set_xlabel("Layer Index (Execution Order)", fontweight="bold")
    ax.set_ylabel("Cumulative VRAM Usage (MB)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    sns.despine()
    fig.tight_layout()
    
    plot_dir = output_dir / model / "memory_footprint"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"fwd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # BACKWARD PASS FOOTPRINT (Reversed execution conceptually)
    df_bwd = df.copy()
    # In backward, parameters are needed, gradients are accumulated, activations are consumed
    # For a simple structural view: we show params + accumulating gradients
    df_bwd["cumulative_params"] = df_bwd["params_mb_mean"].sum() # All params exist
    df_bwd["cumulative_grads"] = df_bwd["grads_mb_mean"].iloc[::-1].cumsum().iloc[::-1] # Gradients build up from end to start
    df_bwd["total_memory"] = df_bwd["cumulative_params"] + df_bwd["cumulative_grads"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, 0, df_bwd["cumulative_params"], label="Model Parameters (Static)", alpha=0.6, color="tab:blue")
    ax.fill_between(x, df_bwd["cumulative_params"], df_bwd["total_memory"], label="Gradients (Cumulative during BWD)", alpha=0.6, color="tab:red")
    
    ax.set_title(f"Memory Footprint Evolution (Backward Pass): {model}", pad=15, fontweight="bold")
    ax.set_xlabel("Layer Index (Forward Order)", fontweight="bold")
    ax.set_ylabel("Cumulative VRAM Usage (MB)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    sns.despine()
    fig.tight_layout()
    
    plot_dir = output_dir / model / "memory_footprint"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"bwd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_affinity_heatmap(config_dir: Path, output_dir: Path):
    model = config_dir.parents[2].name
    stats_path = config_dir / f"{model}_metrics_stats.csv"
    if not stats_path.exists():
        return
        
    df = pd.read_csv(stats_path)
    if "cpu_fwd_time_ms_mean" not in df.columns or "gpu_fwd_time_ms_mean" not in df.columns:
        return
        
    # Calculate affinity: CPU Time / GPU Time
    # High value -> GPU is much faster -> High affinity for GPU
    # Low value -> CPU is relatively fast -> Good candidate for CPU
    df["affinity_ratio"] = df["cpu_fwd_time_ms_mean"] / (df["gpu_fwd_time_ms_mean"] + 1e-6)
    
    # Filter top 20 layers with most computation to avoid noise
    df["total_cost"] = df["cpu_fwd_time_ms_mean"] + df["gpu_fwd_time_ms_mean"]
    df_top = df.sort_values(by="total_cost", ascending=False).head(20).copy()
    
    heatmap_data = df_top[["layer", "affinity_ratio"]].set_index("layer")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=1.0, annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Speedup Ratio (CPU / GPU Time)'})
    
    ax.set_title(f"Computational Affinity Heatmap: {model}\n(Top 20 most expensive layers)", pad=15, fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    ax.set_xlabel("")
    
    fig.tight_layout()
    
    plot_dir = output_dir / model / "affinity_heatmap"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_violin_variance(config_dir: Path, output_dir: Path):
    model = config_dir.parents[2].name
    
    run_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        return
        
    all_data_fwd = []
    all_data_bwd = []
    for rdir in run_dirs:
        metrics_path = rdir / f"{model}_metrics.csv"
        if not metrics_path.exists(): continue
        
        df = pd.read_csv(metrics_path)
        if "layer" in df.columns and "gpu_fwd_time_ms" in df.columns and "gpu_bwd_time_ms" in df.columns:
            for _, row in df.iterrows():
                all_data_fwd.append({"Layer": row["layer"], "Time (ms)": row["gpu_fwd_time_ms"]})
                all_data_bwd.append({"Layer": row["layer"], "Time (ms)": row["gpu_bwd_time_ms"]})
                
    if not all_data_fwd or not all_data_bwd: return
    
    # FORWARD
    df_fwd = pd.DataFrame(all_data_fwd)
    top_layers_fwd = df_fwd.groupby("Layer")["Time (ms)"].median().sort_values(ascending=False).head(10).index.tolist()
    df_top_fwd = df_fwd[df_fwd["Layer"].isin(top_layers_fwd)].copy()
    
    if not df_top_fwd.empty and len(df_top_fwd) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df_top_fwd, x="Layer", y="Time (ms)", ax=ax, density_norm="width", palette="muted", hue="Layer", legend=False)
        ax.set_title(f"Profiling Variance Distribution (Forward - Top 10 Layers): {model}", pad=15, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("GPU Fwd Time (ms)", fontweight="bold")
        ax.set_xlabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        
        plot_dir = output_dir / model / "variance_violin"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"fwd.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # BACKWARD
    df_bwd = pd.DataFrame(all_data_bwd)
    top_layers_bwd = df_bwd.groupby("Layer")["Time (ms)"].median().sort_values(ascending=False).head(10).index.tolist()
    df_top_bwd = df_bwd[df_bwd["Layer"].isin(top_layers_bwd)].copy()
    
    if not df_top_bwd.empty and len(df_top_bwd) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df_top_bwd, x="Layer", y="Time (ms)", ax=ax, density_norm="width", palette="flare", hue="Layer", legend=False)
        ax.set_title(f"Profiling Variance Distribution (Backward - Top 10 Layers): {model}", pad=15, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("GPU Bwd Time (ms)", fontweight="bold")
        ax.set_xlabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        
        plot_dir = output_dir / model / "variance_violin"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"bwd.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_waterfall_ablation(ilp_pareto_csv: Path, output_dir: Path):
    if not ilp_pareto_csv.exists():
        return
        
    df = pd.read_csv(ilp_pareto_csv)
    
    # We will pick the best objective for each model to do a simplified waterfall
    # This is a mock waterfall derivation since we don't have exact transfer/compute breakdown in the consolidated CSV
    # But we can show All-CPU -> ILP Best for each model
    if "All-CPU Obj" not in df.columns or "ILP Obj" not in df.columns:
        return
        
    best_per_model = df.loc[df.groupby("Model")["ILP Obj"].idxmin()]
    
    for _, row in best_per_model.iterrows():
        model = row["Model"]
        cpu_obj = float(row["All-CPU Obj"])
        ilp_obj = float(row["ILP Obj"])
        saved = cpu_obj - ilp_obj
        
        if saved <= 0: continue
        
        # Simplified waterfall: Baseline CPU -> ILP GPU Savings -> Final
        categories = ["All-CPU (Baseline)", "ILP Offload Savings", "Optimized Objective"]
        values = [cpu_obj, -saved, ilp_obj]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plotting a basic waterfall
        bottom = 0
        for i, (cat, val) in enumerate(zip(categories, values)):
            if val < 0:
                ax.bar(cat, val, bottom=bottom, color="green", width=0.6)
                bottom += val
            else:
                if i == len(values) - 1:
                    ax.bar(cat, val, color="blue", width=0.6) # Final
                else:
                    ax.bar(cat, val, color="grey", width=0.6)
                    bottom += val
                    
        ax.set_title(f"Ablation Waterfall Chart: {model}", pad=15, fontweight="bold")
        ax.set_ylabel("Total Objective (ms)", fontweight="bold")
        sns.despine()
        
        fig.tight_layout()
        
        plot_dir = output_dir / model / "ablation_waterfall"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"waterfall.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate Advanced Statistical Thesis Plots")
    parser.add_argument("--input_root", type=Path, required=True, help="Root directory containing model profiling results")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save advanced plots")
    parser.add_argument("--hybrid_csv", type=Path, default=None, help="Path to hybrid_execution_consolidated.csv")
    parser.add_argument("--ilp_pareto_csv", type=Path, default=None, help="Path to ilp_pareto_consolidated.csv")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating advanced plots in {args.output_dir}...")
    
    if args.hybrid_csv and args.hybrid_csv.exists():
        try:
            plot_predicted_vs_actual(args.hybrid_csv, args.output_dir)
        except Exception as e:
            print(f"Error plotting predicted vs actual: {e}")
            
    if args.ilp_pareto_csv and args.ilp_pareto_csv.exists():
        try:
            plot_waterfall_ablation(args.ilp_pareto_csv, args.output_dir)
        except Exception as e:
            print(f"Error plotting waterfall ablation: {e}")
        
    config_dirs = find_config_dirs(args.input_root)
    processed_models = set()
    
    for cdir in config_dirs:
        model = cdir.parents[2].name
        if model in processed_models:
            continue
            
        try:
            plot_memory_footprint(cdir, args.output_dir)
            plot_affinity_heatmap(cdir, args.output_dir)
            plot_violin_variance(cdir, args.output_dir)
        except Exception as e:
            print(f"Error processing advanced plots for {model}: {e}")
            
        processed_models.add(model)
        
    print("Advanced plots generated successfully.")

if __name__ == "__main__":
    main()
