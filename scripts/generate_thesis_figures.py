from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "thesis_figures"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_profiling_layer_costs() -> None:
    stats_path = (
        ROOT
        / "data"
        / "zephyr"
        / "results_smoke"
        / "simple_mlp"
        / "SGD"
        / "fp32"
        / "batch_8"
        / "simple_mlp_metrics_stats.csv"
    )
    rows = read_csv_rows(stats_path)
    layers = [row["layer"] for row in rows]
    gpu_fwd = [float(row["gpu_fwd_time_ms_mean"]) for row in rows]
    cpu_fwd = [float(row["cpu_fwd_time_ms_mean"]) for row in rows]
    transfer = [float(row["transfer_edge_aware_total_ms_mean"]) for row in rows]

    positions = list(range(len(layers)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([p - width for p in positions], gpu_fwd, width=width, label="Forward GPU (ms)", color="#2B6CB0")
    ax.bar(positions, cpu_fwd, width=width, label="Forward CPU (ms)", color="#D97706")
    ax.bar([p + width for p in positions], transfer, width=width, label="Transferencia arista-aware (ms)", color="#2F855A")

    ax.set_xticks(positions)
    ax.set_xticklabels(layers)
    ax.set_ylabel("Milisegundos")
    ax.set_title("Ejemplo real de profiling robusto por capa: simple_mlp, batch 8, fp32")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "profiling_simple_mlp_layer_costs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_vs_observation() -> None:
    simulation_path = (
        ROOT
        / "reports"
        / "ilp_results_phase4_controlled"
        / "simple_mlp_dual_runtime_evidence"
        / "simulation"
        / "simulation_summary.json"
    )
    runtime_path = (
        ROOT
        / "reports"
        / "ilp_results_phase4_controlled"
        / "simple_mlp_dual_runtime_evidence"
        / "runtime"
        / "hybrid_execution_summary.json"
    )

    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    comparisons = [
        ("Latencia total (ms)", simulation["total_time_ms"], runtime["avg_step_ms"]),
        ("Energia total (J)", simulation["total_energy_j"], runtime["total_energy_j"]),
        ("Memoria GPU pico (MB)", simulation["gpu_mem_used_mb"], runtime["peak_gpu_mem_mb"]),
    ]

    for ax, (title, predicted, observed) in zip(axes, comparisons):
        ax.bar([0, 1], [predicted, observed], color=["#4A5568", "#C05621"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicho", "Observado"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ymax = max(predicted, observed)
        ax.set_ylim(0, ymax * 1.22 if ymax else 1)
        ax.text(0, predicted + ymax * 0.05, f"{predicted:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(1, observed + ymax * 0.05, f"{observed:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Caso controlado simple_mlp: simulacion frente a ejecucion hibrida observada", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_vs_observation_simple_mlp_dual.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_profiling_layer_costs()
    plot_prediction_vs_observation()


if __name__ == "__main__":
    main()