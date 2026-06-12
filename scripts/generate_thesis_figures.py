from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Set Seaborn academic theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
sns.set_palette("colorblind")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_profiling_layer_costs(host_tag: str, config_dir: Path, model: str, optimizer: str, precision: str, batch: str, output_dir: Path) -> None:
    stats_path = config_dir / f"{model}_metrics_stats.csv"
    if not stats_path.exists():
        stats_path = config_dir / "metrics_stats.csv"
        
    if not stats_path.exists():
        return
        
    try:
        rows = read_csv_rows(stats_path)
        fwd_data = []
        bwd_data = []
        for row in rows:
            layer = row.get("layer", "")
            transfer_time = float(row.get("transfer_edge_aware_total_ms_mean", 0))
            
            fwd_data.append({"Layer": layer, "Time (ms)": float(row.get("gpu_fwd_time_ms_mean", 0)), "Component": "GPU Forward"})
            fwd_data.append({"Layer": layer, "Time (ms)": float(row.get("cpu_fwd_time_ms_mean", 0)), "Component": "CPU Forward"})
            fwd_data.append({"Layer": layer, "Time (ms)": transfer_time, "Component": "PCIe Transfer"})
            
            bwd_data.append({"Layer": layer, "Time (ms)": float(row.get("gpu_bwd_time_ms_mean", 0)), "Component": "GPU Backward"})
            bwd_data.append({"Layer": layer, "Time (ms)": float(row.get("cpu_bwd_time_ms_mean", 0)), "Component": "CPU Backward"})
            bwd_data.append({"Layer": layer, "Time (ms)": transfer_time, "Component": "PCIe Transfer"})
            
        if not fwd_data or not bwd_data:
            return
            
        def generate_plot(data_list, title_prefix, file_suffix):
            df = pd.DataFrame(data_list)
            
            # Calculate total cost per layer to find the Top 15
            total_costs = df.groupby("Layer")["Time (ms)"].sum().reset_index()
            top_15_layers = total_costs.sort_values(by="Time (ms)", ascending=False).head(15)["Layer"].tolist()
            
            # Filter dataframe for top 15 layers
            df_top15 = df[df["Layer"].isin(top_15_layers)].copy()
            
            # We want to maintain the sort order in the plot
            df_top15["Layer"] = pd.Categorical(df_top15["Layer"], categories=top_15_layers, ordered=True)
            df_top15 = df_top15.sort_values("Layer")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=df_top15, y="Layer", x="Time (ms)", hue="Component", ax=ax, palette="deep", orient="h")
            
            ax.set_title(f"Top 15 Per-layer {title_prefix} Latencies: {model} (Opt: {optimizer}, Prec: {precision}, Batch: {batch})", pad=15, fontweight="bold")
            ax.set_ylabel("Layer Name", fontweight="bold")
            ax.set_xlabel("Time (ms)", fontweight="bold")
            ax.legend(title="", frameon=True, loc="lower right")
            sns.despine()
            
            fig.tight_layout()
            plot_dir = output_dir / model / "layer_profiling"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"profiling_{model}_{optimizer}_{precision}_{batch}_{file_suffix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        generate_plot(fwd_data, "Forward Pass", "fwd_layer_costs")
        generate_plot(bwd_data, "Backward Pass", "bwd_layer_costs")
    except Exception as e:
        print(f"Error plotting layer costs for {config_dir}: {e}")
        raise


def plot_prediction_vs_observation(host_tag: str, config_dir: Path, model: str, optimizer: str, precision: str, batch: str, output_dir: Path) -> None:
    simulation_path = config_dir / "ilp_solution" / f"{model}_pareto_summary.json"
    runtime_path = config_dir / "ilp_solution" / "hybrid_execution" / "hybrid_execution_summary.json"

    if not simulation_path.exists() or not runtime_path.exists():
        return

    try:
        pareto_data = json.loads(simulation_path.read_text(encoding="utf-8"))
        if "best_feasible_row" in pareto_data:
            simulation = pareto_data["best_feasible_row"]
        else:
            return
            
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))

        metrics = ["Total Latency (ms)", "Total Energy (J)", "Peak GPU Memory (MB)"]
        predicted = [
            simulation.get("ilp_objective", 0.0),
            simulation.get("total_energy_j", 0.0),
            simulation.get("gpu_mem_used_mb", 0.0),
        ]
        observed = [
            runtime.get("avg_step_ms", 0.0),
            runtime.get("total_energy_j", 0.0),
            runtime.get("peak_gpu_mem_mb", 0.0),
        ]

        data = []
        for m, p, o in zip(metrics, predicted, observed):
            data.append({"Metric": m, "Value": p, "Type": "Predicted (ILP)"})
            data.append({"Metric": m, "Value": o, "Type": "Observed (Runtime)"})

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for i, metric in enumerate(metrics):
            subset = df[df["Metric"] == metric]
            sns.barplot(data=subset, x="Type", y="Value", ax=axes[i], palette="muted")
            axes[i].set_title(metric, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            
            # Add value labels
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.2f}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha="center", va="center", xytext=(0, 8), textcoords="offset points",
                                 fontsize=10, fontweight="bold")
        
        sns.despine()
        fig.suptitle(f"ILP Simulation vs Hybrid Execution: {model} (Opt: {optimizer}, Prec: {precision}, Batch: {batch})", fontsize=14, fontweight="bold", y=1.05)
        fig.tight_layout()
        
        plot_dir = output_dir / model / "hybrid_vs_ilp"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"prediction_vs_observation_{model}_{optimizer}_{precision}_{batch}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting prediction vs observation for {config_dir}: {e}")
        raise

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis figures from profiling data.")
    parser.add_argument(
        "--host_tag", 
        type=str, 
        default="all", 
        help="Specify the host tag to process, or 'all' to process all directories in data/."
    )
    args = parser.parse_args()
    
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return 1

    if args.host_tag.lower() == "all":
        hosts = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    else:
        hosts = [args.host_tag]
        
    for host in hosts:
        print(f"Generating figures for host: {host}")
        host_results_dir = data_dir / host / "results_thesis_mode"
        if not host_results_dir.exists():
            continue
            
        output_dir = ROOT / "reports" / host / "doctoral_minimal" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        errors = 0

        # Discover all configurations
        for batch_dir in sorted(host_results_dir.rglob("batch_*")):
            if not batch_dir.is_dir() or len(batch_dir.parts) < 4:
                continue
            
            try:
                batch = batch_dir.name.replace("batch_", "")
                precision = batch_dir.parent.name
                optimizer = batch_dir.parent.parent.name
                model = batch_dir.parent.parent.parent.name
                processed += 1
                
                plot_profiling_layer_costs(host, batch_dir, model, optimizer, precision, batch, output_dir)
                plot_prediction_vs_observation(host, batch_dir, model, optimizer, precision, batch, output_dir)
            except Exception as e:
                errors += 1
                print(f"Error processing {batch_dir}: {e}")
        
        print(
            f"Finished host {host}: processed={processed}, success={processed - errors}, "
            f"errors={errors}. Output saved to {output_dir}\n"
        )
        if errors > 0:
            return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())