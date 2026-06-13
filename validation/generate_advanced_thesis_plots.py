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

def plot_predicted_vs_actual(config_dir: Path, output_dir: Path, model: str, hybrid_csv: Path):
    # Removed as per user request to only have the global scatter plot.
    pass

def plot_memory_footprint(config_dir: Path, output_dir: Path, model: str):
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    
    stats_path = config_dir / f"{model}_metrics_stats.csv"
    if not stats_path.exists():
        return
        
    df = pd.read_csv(stats_path)
    if "activations_mb_mean" not in df.columns or "params_mb_mean" not in df.columns or "grads_mb_mean" not in df.columns:
        return
        
    df = df.dropna(subset=["activations_mb_mean", "params_mb_mean", "grads_mb_mean"])
    if df.empty:
        return

    # Prepare data for plotting
    df = df.copy()
    df["cumulative_params"] = df["params_mb_mean"].cumsum()
    df["cumulative_acts"] = df["activations_mb_mean"].cumsum()
        
    # BWD pass: layer N down to 0
    df_bwd = df.copy()
    
    # In time, backward goes from N to 0. We'll use the backward execution step (0 to len-1)
    df_bwd["bwd_step"] = np.arange(len(df_bwd))
    
    # Gradients accumulate over time.
    # df_bwd starts with layer 0 at index 0. We reverse it so layer N is at index 0.
    df_bwd = df_bwd.iloc[::-1].copy()
    df_bwd["bwd_step"] = np.arange(len(df_bwd))
    df_bwd["cumulative_grads"] = df_bwd["grads_mb_mean"].cumsum()
    
    # Activations start at max and are consumed
    total_acts = df["activations_mb_mean"].sum()
    # Subtract activations as they are consumed layer by layer
    df_bwd["remaining_activations"] = total_acts - df_bwd["activations_mb_mean"].cumsum() + df_bwd["activations_mb_mean"]
    
    total_params = df["params_mb_mean"].sum()
    df_bwd["static_params"] = total_params
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Forward Pass
    x_fwd = np.arange(len(df))
    ax1.fill_between(x_fwd, 0, df["cumulative_params"], label="Parameters", color="#3498db", alpha=0.8)
    ax1.fill_between(x_fwd, df["cumulative_params"], df["cumulative_params"] + df["cumulative_acts"], 
                     label="Activations", color="#e74c3c", alpha=0.8)
    ax1.set_title("Forward Pass (Layer 0 $\\rightarrow$ N)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Layer Execution Order", fontsize=12)
    ax1.set_ylabel("Memory (MB)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Backward Pass (Single Axis Stacking with cleaner presentation)
    x_bwd = df_bwd["bwd_step"]
    
    # We stack: Params + Gradients + Remaining Activations using stackplot for a cleaner visual
    y_params = df_bwd["static_params"].values
    y_grads = df_bwd["cumulative_grads"].values
    y_acts = df_bwd["remaining_activations"].values
    
    # Use stackplot to automatically handle the areas without overlapping jagged lines
    ax2.stackplot(x_bwd, y_params, y_grads, y_acts, 
                  labels=["Parameters", "Gradients", "Remaining Activations"], 
                  colors=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
    
    ax2.set_xlabel("Backward Execution Step (Layer N $\\rightarrow$ 0)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title("Backward Pass", fontsize=14, fontweight="bold")
    
    fig.suptitle(f"Memory Footprint Evolution: {model} | {optimizer} | {precision} | {batch_name}", fontsize=16, fontweight="bold")
    
    # Place a single shared legend outside the plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Avoid duplicate labels (Parameters)
    by_label = dict(zip(labels2 + labels1, handles2 + handles1))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., fontsize=12)

    
    plot_dir = output_dir / model / optimizer / precision / batch_name / "memory_footprint"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_dir / f"footprint_{optimizer}_{precision}_{batch_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_affinity_heatmap(config_dir: Path, output_dir: Path, model: str):
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
    
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    
    plot_dir = output_dir / model / optimizer / precision / batch_name / "affinity_heatmap"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_violin_variance(config_dir: Path, output_dir: Path, model: str):
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    
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
        ax.set_title(f"Profiling Variance Distribution (Forward - Top 10 Layers): {model} | {optimizer} | {precision} | {batch_name}", pad=15, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("GPU Fwd Time (ms)", fontweight="bold")
        ax.set_xlabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        
        plot_dir = output_dir / model / optimizer / precision / batch_name / "variance_violin"
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
        ax.set_title(f"Profiling Variance Distribution (Backward - Top 10 Layers): {model} | {optimizer} | {precision} | {batch_name}", pad=15, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("GPU Bwd Time (ms)", fontweight="bold")
        ax.set_xlabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        
        plot_dir = output_dir / model / optimizer / precision / batch_name / "variance_violin"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"bwd.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_waterfall_ablation(config_dir: Path, output_dir: Path, model: str):
    stats_path = config_dir / f"{model}_pareto_sweep.csv"
    if not stats_path.exists():
        return
        
    df = pd.read_csv(stats_path)
    if "all_cpu_objective" not in df.columns or "ilp_objective" not in df.columns:
        return
        
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    
    feasible = df[df.get("ilp_status", "").isin(["optimal", "feasible"])].copy() if "ilp_status" in df.columns else df.copy()
    feasible["ilp_objective"] = pd.to_numeric(feasible["ilp_objective"], errors="coerce")
    feasible["all_cpu_objective"] = pd.to_numeric(feasible["all_cpu_objective"], errors="coerce")
    feasible = feasible.dropna(subset=["ilp_objective", "all_cpu_objective"])
    feasible = feasible[np.isfinite(feasible["ilp_objective"]) & np.isfinite(feasible["all_cpu_objective"])].copy()
    if feasible.empty:
        return

    best_row = feasible.loc[feasible["ilp_objective"].idxmin()]
    cpu_obj = float(best_row["all_cpu_objective"])
    ilp_obj = float(best_row["ilp_objective"])
    saved = cpu_obj - ilp_obj
    
    if saved <= 0: return
    
    categories = ["All-CPU (Baseline)", "ILP Offload Savings", "Optimized Objective"]
    values = [cpu_obj, -saved, ilp_obj]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottom = 0
    for i, (cat, val) in enumerate(zip(categories, values)):
        if val < 0:
            ax.bar(cat, val, bottom=bottom, color="green", width=0.6)
            bottom += val
        else:
            if i == len(values) - 1:
                ax.bar(cat, val, color="blue", width=0.6)
            else:
                ax.bar(cat, val, color="grey", width=0.6)
                bottom += val
                
    ax.set_title(f"Ablation Waterfall Chart: {model} | {optimizer} | {precision} | {batch_name}", pad=15, fontweight="bold")
    ax.set_ylabel("Total Objective (ms)", fontweight="bold")
    sns.despine()
    
    fig.tight_layout()
    
    plot_dir = output_dir / model / optimizer / precision / batch_name / "ablation_waterfall"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"waterfall_{optimizer}_{precision}_{batch_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_roofline_model(config_dir: Path, output_dir: Path, model: str):
    import json
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    
    stats_path = config_dir / f"{model}_metrics_stats.csv"
    if not stats_path.exists(): 
        print(f"Roofline: {stats_path} does not exist")
        return
    df = pd.read_csv(stats_path)
    
    # Use rescued_tflops if available, else fallback to tflops_mean
    tflops_col = "rescued_tflops" if "rescued_tflops" in df.columns else "tflops_mean"
    
    if tflops_col not in df.columns or "gpu_fwd_time_ms_mean" not in df.columns or "transfer_h2d_ms_mean" not in df.columns: 
        print(f"Roofline: missing columns in {stats_path}")
        return
        
    df = df[df[tflops_col] > 0].copy()
    if df.empty: 
        print(f"Roofline: {stats_path} has no rows with {tflops_col} > 0")
        return
        
    # Read meta json to extract actual hardware measurements
    peak_tflops_gpu = 20.0
    peak_tflops_cpu = 1.0
    meta_json = config_dir / "run_001" / f"{model}_meta.json"
    if meta_json.exists():
        with open(meta_json, "r") as f:
            meta = json.load(f)
            peak_tflops_gpu = meta.get("measured_peak_tflops_gpu", 20.0)
            peak_tflops_cpu = meta.get("measured_peak_tflops_cpu", 1.0)
            
    # Assuming typical Datacenter GPU (VRAM) and high-end Server CPU (DRAM) bandwidths
    vram_bandwidth_gb_s = 760.0
    dram_bandwidth_gb_s = 50.0
    
    ridge_gpu = peak_tflops_gpu / (vram_bandwidth_gb_s / 1000.0)
    ridge_cpu = peak_tflops_cpu / (dram_bandwidth_gb_s / 1000.0)
    
    # Approx Bytes transferred: (params_mb + activations_mb) * 1e6
    df["bytes"] = (df["params_mb_mean"] + df["activations_mb_mean"]) * 1e6
    
    # If gpu_fwd_time_ms_mean <= 0 (OOM), fallback to cpu time
    df["used_time_ms"] = df["gpu_fwd_time_ms_mean"]
    df.loc[df["used_time_ms"] <= 0, "used_time_ms"] = df["cpu_fwd_time_ms_mean"]
    df["flops"] = df[tflops_col] * 1e12 * (df["used_time_ms"] / 1000.0)
    
    df["arithmetic_intensity"] = df["flops"] / df["bytes"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot ceilings
    min_ai = min(0.01, df["arithmetic_intensity"].min() * 0.5)
    max_ai = max(10000, df["arithmetic_intensity"].max() * 2)
    ai_vals = np.logspace(np.log10(min_ai), np.log10(max_ai), 100)
    
    gpu_bw_roof = ai_vals * (vram_bandwidth_gb_s / 1000.0)
    gpu_roof = np.minimum(gpu_bw_roof, peak_tflops_gpu)
    
    cpu_bw_roof = ai_vals * (dram_bandwidth_gb_s / 1000.0)
    cpu_roof = np.minimum(cpu_bw_roof, peak_tflops_cpu)
    
    ax.plot(ai_vals, gpu_roof, color="red", linestyle="-", linewidth=2.5, label=f"GPU Compute ({peak_tflops_gpu:.1f} TFLOPS) / VRAM BW ({vram_bandwidth_gb_s} GB/s)")
    ax.plot(ai_vals, cpu_roof, color="blue", linestyle="--", linewidth=2.5, label=f"CPU Compute ({peak_tflops_cpu:.1f} TFLOPS) / DRAM BW ({dram_bandwidth_gb_s} GB/s)")
    
    # Regions
    ax.fill_between(ai_vals[ai_vals <= ridge_gpu], 0, gpu_roof[ai_vals <= ridge_gpu], color='orange', alpha=0.1, label='Memory Bound')
    ax.fill_between(ai_vals[ai_vals > ridge_gpu], 0, gpu_roof[ai_vals > ridge_gpu], color='purple', alpha=0.1, label='Compute Bound')
    
    # Plot layers
    scatter = ax.scatter(df["arithmetic_intensity"], df[tflops_col], 
                         c=df["used_time_ms"], cmap="viridis", s=60, alpha=0.8, edgecolors="k", label="Model Layers")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOP/s)", fontsize=12)
    ax.set_title(f"Hardware Roofline Model: {model} | {optimizer} | {precision} | {batch_name}", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Execution Time (ms)")
    
    ax.legend(loc="lower right")
    
    out_dir = output_dir / model / optimizer / precision / batch_name / "roofline_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "roofline_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_power_throughput_vs_budget(config_dir: Path, output_dir: Path, model: str):
    stats_path = config_dir / f"{model}_pareto_sweep.csv"
    if not stats_path.exists():
        return
        
    df = pd.read_csv(stats_path)
    if "gpu_budget_mb" not in df.columns or "ilp_energy_j" not in df.columns or "ilp_objective" not in df.columns: 
        return
        
    optimizer = config_dir.parents[1].name
    precision = config_dir.parents[0].name
    batch_name = config_dir.name
    try:
        batch_size = int(batch_name.split("_")[-1])
    except ValueError:
        batch_size = 32
    
    # Drop rows where latency or energy is missing or inf
    df_model = df[(df["ilp_energy_j"] > 0) & (df["ilp_objective"] > 0)].copy()
    df_model = df_model[(df_model["ilp_energy_j"] != np.inf) & (df_model["ilp_objective"] != np.inf)]
    
    if df_model.empty:
        return
        
    df_model["watts"] = df_model["ilp_energy_j"] / (df_model["ilp_objective"] / 1000.0)
    df_model["throughput"] = batch_size / (df_model["ilp_objective"] / 1000.0)
    
    # Sort by budget
    df_model = df_model.sort_values("gpu_budget_mb")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:red'
    ax1.set_xlabel('GPU Memory Budget (MB)', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Power (Watts)', color=color1, fontsize=12, fontweight="bold")
    ax1.plot(df_model["gpu_budget_mb"], df_model["watts"], color=color1, marker='o', linewidth=2, label="Power (W)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Training Throughput (Samples/s)', color=color2, fontsize=12, fontweight="bold")
    ax2.plot(df_model["gpu_budget_mb"], df_model["throughput"], color=color2, marker='s', linewidth=2, label="Throughput")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.suptitle(f"Power & Throughput vs Budget: {model} | {optimizer} | {precision} | {batch_name}", fontsize=14, fontweight="bold")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right")
    
    fig.tight_layout()
    
    plot_dir = output_dir / model / optimizer / precision / batch_name / "efficiency"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"power_throughput_{optimizer}_{precision}_{batch_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_global_predicted_vs_actual(output_dir: Path, hybrid_csv: Path):
    if not hybrid_csv or not hybrid_csv.exists():
        return
        
    df = pd.read_csv(hybrid_csv)
    predicted_col = "plan_objective" if "plan_objective" in df.columns else None
    measured_col = "avg_step_ms" if "avg_step_ms" in df.columns else None
    if predicted_col is None or measured_col is None:
        return

    model_col = "config_model" if "config_model" in df.columns else ("model" if "model" in df.columns else None)
    if model_col is None:
        return

    df_valid = df[
        (pd.to_numeric(df[predicted_col], errors="coerce") > 0)
        & (df["run_label"] != "all_cpu")
        & (df["run_label"] != "all_gpu")
    ].copy()

    df_valid[predicted_col] = pd.to_numeric(df_valid[predicted_col], errors="coerce")
    df_valid[measured_col] = pd.to_numeric(df_valid[measured_col], errors="coerce")
    df_valid = df_valid.dropna(subset=[predicted_col, measured_col])

    if df_valid.empty:
        return

    plot_dir = output_dir / "global_summary"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Density Plot (to solve saturation)
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(df_valid) > 1:
        from scipy.stats import pearsonr
        r_val, _ = pearsonr(df_valid[predicted_col], df_valid[measured_col])
        r2 = r_val ** 2
        title_extra = f" (Global $R^2$ = {r2:.3f})"
    else:
        title_extra = ""

    sns.histplot(data=df_valid, x=predicted_col, y=measured_col, bins=50, pmax=0.9, cmap="mako", cbar=True, ax=ax, cbar_kws={'label': 'Density of Configurations'})
    
    min_val = min(df_valid[predicted_col].min(), df_valid[measured_col].min())
    max_val = max(df_valid[predicted_col].max(), df_valid[measured_col].max())
    padding = (max_val - min_val) * 0.1 if max_val > min_val else min_val * 0.1
    ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', label='Ideal Perfect Prediction (y=x)')
    
    ax.set_title(f"Global Predicted vs Actual Latency (Density){title_extra}", pad=15, fontweight="bold")
    ax.set_xlabel("ILP Predicted Latency (ms)", fontweight="bold")
    ax.set_ylabel("Measured Latency (ms)", fontweight="bold")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(plot_dir / "global_predicted_vs_actual_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. FacetGrid by Model
    models = df_valid[model_col].unique()
    if len(models) <= 16:
        g = sns.FacetGrid(df_valid, col=model_col, col_wrap=4, height=4, sharex=False, sharey=False)
        g.map_dataframe(sns.scatterplot, x=predicted_col, y=measured_col, alpha=0.7, color="#1f77b4")
        
        def plot_ideal_line(**kwargs):
            ax = plt.gca()
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0, label="Ideal Perfect Prediction (y=x)")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
        g.map(plot_ideal_line)
        g.set_axis_labels("Predicted (ms)", "Actual (ms)")
        g.set_titles(col_template="{col_name}")
        g.add_legend()
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Predicted vs Actual Latency by Model", fontweight="bold", fontsize=16)
        g.savefig(plot_dir / "global_predicted_vs_actual_facets.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)

def main():
    parser = argparse.ArgumentParser(description="Generate Advanced Statistical Thesis Plots")
    parser.add_argument("--input_root", type=Path, required=True, help="Root directory containing model profiling results")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save advanced plots")
    parser.add_argument("--hybrid_csv", type=Path, default=None, help="Path to hybrid_execution_consolidated.csv")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    # Keep output path idempotent: if caller already points to a "plots" dir,
    # do not append another nested "plots" segment.
    if output_dir.name != "plots":
        output_dir = output_dir / "plots"

    print(f"Generating advanced plots in {output_dir}...")
    
    if args.hybrid_csv and args.hybrid_csv.exists():
        hybrid_csv_path = args.hybrid_csv
    else:
        hybrid_csv_path = None
    total = 0
    errors = 0

    stats_files = sorted(p for p in input_root.rglob("*_metrics_stats.csv") if "run_" not in p.parent.name)

    for stats_file in stats_files:
        cdir = stats_file.parent
        model = stats_file.name.replace("_metrics_stats.csv", "")
        total += 1
        
        # We process ALL config_dirs, not skipping if model is seen
        try:
            plot_memory_footprint(cdir, output_dir, model)
            plot_affinity_heatmap(cdir, output_dir, model)
            plot_roofline_model(cdir, output_dir, model)
            plot_power_throughput_vs_budget(cdir, output_dir, model)
            plot_violin_variance(cdir, output_dir, model)
            plot_waterfall_ablation(cdir, output_dir, model)
            if hybrid_csv_path:
                plot_predicted_vs_actual(cdir, output_dir, model, hybrid_csv_path)
        except Exception as e:
            errors += 1
            print(f"Error processing advanced plots for {cdir}: {e}")

    success = total - errors
    print(f"Advanced plots finished: processed={total}, success={success}, errors={errors}")
    
    if hybrid_csv_path:
        print("Generating global scatter plots...")
        try:
            plot_global_predicted_vs_actual(output_dir, hybrid_csv_path)
            print("Global scatter plots generated successfully.")
        except Exception as e:
            print(f"Error generating global scatter plots: {e}")
            errors += 1

    if errors > 0:
        raise SystemExit(1)
    print("Advanced plots generated successfully.")

if __name__ == "__main__":
    main()
