from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_custom_figures(
    host: str,
    model: str,
    batch_size: int,
    optimizer: str,
    precision: str,
    results_root: Path,
    reports_root: Path,
) -> bool:
    base_dir = results_root / host / "results_thesis_mode" / "doctoral_full" / model / optimizer / precision / f"batch_{batch_size}"
    stats_csv = base_dir / f"{model}_metrics_stats.csv"

    run_001 = base_dir / "run_001"
    transfer_csv = None
    if (run_001 / f"{model}_transfer_edges.csv").exists():
        transfer_csv = run_001 / f"{model}_transfer_edges.csv"
    elif (base_dir / f"{model}_transfer_edges.csv").exists():
        transfer_csv = base_dir / f"{model}_transfer_edges.csv"

    out_dir = reports_root / host / "doctoral_minimal" / "plots" / model / optimizer / precision / f"batch_{batch_size}" / "custom_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not stats_csv.exists():
        print(f"Skipping {model} batch {batch_size}: {stats_csv} not found")
        return False

    df_stats = pd.read_csv(stats_csv)

    gpu_mem = df_stats["gpu_mem_peak_mb_mean"].sum() if "gpu_mem_peak_mb_mean" in df_stats.columns else 0
    cpu_mem = df_stats["cpu_mem_mb_mean"].sum() if "cpu_mem_mb_mean" in df_stats.columns else 0

    total_bandwidth_mb = 0
    if transfer_csv and transfer_csv.exists():
        try:
            df_trans = pd.read_csv(transfer_csv)
            if "transfer_bytes" in df_trans.columns:
                total_bandwidth_mb = df_trans["transfer_bytes"].sum() / (1024 * 1024)
        except Exception as exc:
            print(f"Warning: failed reading transfer edges for {model} batch {batch_size}: {exc}")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = pd.DataFrame(
        [
            {"Metric": "GPU Memory (MB)", "Value": gpu_mem},
            {"Metric": "CPU Memory (MB)", "Value": cpu_mem},
            {"Metric": "Total Bandwidth (MB)", "Value": total_bandwidth_mb},
        ]
    )
    sns.barplot(data=bars, x="Metric", y="Value", ax=ax, palette="mako")
    ax.set_title(f"Memory and Bandwidth Consumption: {model} (Batch {batch_size})", pad=15, fontweight="bold")
    ax.set_ylabel("Megabytes (MB)", fontweight="bold")
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f} MB",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_dir / f"total_consumption_{model}_batch_{batch_size}.png", dpi=300)
    plt.close(fig)

    if "gpu_mem_peak_mb_mean" in df_stats.columns and "layer" in df_stats.columns:
        top_mem = df_stats.sort_values(by="gpu_mem_peak_mb_mean", ascending=False).head(15).copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top_mem, x="gpu_mem_peak_mb_mean", y="layer", ax=ax, palette="crest")
        ax.set_title(f"Top 15 Memory-Intensive Layers: {model} (Batch {batch_size})", pad=15, fontweight="bold")
        ax.set_xlabel("GPU Memory Peak (MB)", fontweight="bold")
        ax.set_ylabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        fig.savefig(out_dir / f"top15_memory_{model}_batch_{batch_size}.png", dpi=300)
        plt.close(fig)

    if "layer" in df_stats.columns:
        df_stats = df_stats.copy()
        df_stats["total_energy_j"] = (
            pd.to_numeric(df_stats.get("gpu_fwd_energy_j_mean", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df_stats.get("gpu_bwd_energy_j_mean", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df_stats.get("cpu_fwd_energy_j_mean", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df_stats.get("cpu_bwd_energy_j_mean", 0), errors="coerce").fillna(0)
        )
        top_en = df_stats.sort_values(by="total_energy_j", ascending=False).head(15).copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top_en, x="total_energy_j", y="layer", ax=ax, palette="flare")
        ax.set_title(f"Top 15 Energy-Intensive Layers: {model} (Batch {batch_size})", pad=15, fontweight="bold")
        ax.set_xlabel("Total Energy (Joules)", fontweight="bold")
        ax.set_ylabel("Layer", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        fig.savefig(out_dir / f"top15_energy_{model}_batch_{batch_size}.png", dpi=300)
        plt.close(fig)

    print(f"Generated custom figures for {model} batch {batch_size} in {out_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate custom user figures for thesis profiling artifacts.")
    parser.add_argument("--host", default="zephyr", help="Host namespace under data/<host>/results_thesis_mode")
    parser.add_argument("--models", nargs="+", default=["vit_b16", "resnet152"], help="Model list")
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64], help="Batch sizes")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer subfolder name")
    parser.add_argument("--precision", default="fp32", help="Precision subfolder name")
    parser.add_argument("--results-root", type=Path, default=Path("data"), help="Results root directory")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"), help="Reports root directory")
    args = parser.parse_args()

    generated = 0
    skipped = 0
    for model in args.models:
        for batch_size in args.batches:
            ok = plot_custom_figures(
                host=args.host,
                model=model,
                batch_size=batch_size,
                optimizer=args.optimizer,
                precision=args.precision,
                results_root=args.results_root,
                reports_root=args.reports_root,
            )
            if ok:
                generated += 1
            else:
                skipped += 1

    print(f"Custom figures completed: generated={generated}, skipped={skipped}")
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
