import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Professional Plotting Aesthetics
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'seaborn-paper' if 'seaborn-paper' in plt.style.available else 'default')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")


def _find_metrics_csv(input_root: Path, preferred_model: str) -> Path | None:
    # Aggregated stats live at the batch directory level (e.g.
    # .../resnet50/SGD/fp32/batch_64/resnet50_metrics_stats.csv), never inside
    # a run_NNN/ replicate subdirectory -- same convention already used by
    # validation/generate_advanced_thesis_plots.py.
    candidates = sorted(p for p in input_root.rglob("*_metrics_stats.csv") if "run_" not in p.parent.name)
    if not candidates:
        return None
    preferred = [c for c in candidates if preferred_model in c.name]
    return preferred[0] if preferred else candidates[0]


def _find_meta_json(input_root: Path, preferred_model: str) -> Path | None:
    # Unlike *_metrics_stats.csv, *_meta.json (which carries the real PCIe
    # transfer calibration: transfer_alpha_h2d/transfer_beta_h2d/etc.) is only
    # ever written inside a run_NNN/ replicate subdirectory -- confirmed by
    # direct inspection, there is no aggregated top-level copy.
    candidates = sorted(input_root.rglob("*_meta.json"))
    if not candidates:
        return None
    preferred = [c for c in candidates if preferred_model in c.name]
    return preferred[0] if preferred else candidates[0]


def plot_temporal_decomposition(metrics_csv: Path, output_dir: Path):
    """Generates Figure 4.2: T_wall vs T_kernel vs T_dispatch, from real per-layer
    profiling data (training_profiler.py already measures and stores kernel_ms
    and dispatch_ms separately per layer -- metrics_stats.csv only persists
    their ratio, gpu_fwd_time_ms_mean * dispatch_overhead_ratio_mean reconstructs
    the absolute dispatch time exactly; no new instrumentation needed)."""
    df = pd.read_csv(metrics_csv)
    # Per-layer reconstruction first (t_wall * ratio), THEN average across all
    # real layers of a type -- averaging a single cherry-picked "heaviest"
    # layer per category would bias the figure; this reports the honest
    # per-category mean over every real layer of that type.
    df = df.copy()
    df["t_dispatch_abs"] = df["gpu_fwd_time_ms_mean"] * df["dispatch_overhead_ratio_mean"]
    df["t_kernel_abs"] = df["gpu_fwd_time_ms_mean"] - df["t_dispatch_abs"]

    type_order = ["Conv2d", "BatchNorm2d", "ReLU", "Linear"]
    display_labels = {
        "Conv2d": "Conv2d (Heavy)",
        "BatchNorm2d": "BatchNorm",
        "ReLU": "ReLU (Light)",
        "Linear": "Linear",
    }

    chosen = []
    for t in type_order:
        sub = df[df["type"] == t]
        if sub.empty:
            continue
        chosen.append((t, sub["t_kernel_abs"].mean(), sub["t_dispatch_abs"].mean(), len(sub)))

    if not chosen:
        raise ValueError(f"No layers matching {type_order} found in {metrics_csv}")

    labels = [display_labels[t] for t, _, _, _ in chosen]
    t_kernel = np.array([k for _, k, _, _ in chosen])
    t_dispatch = np.array([d for _, _, d, _ in chosen])

    print("  Real per-category averages used for temporal decomposition (mean over all real layers of that type):")
    for t, k, d, n in chosen:
        print(f"    {display_labels[t]:20s} <- {n} real layer(s), mean t_kernel={k:.3f}ms  mean t_dispatch={d:.3f}ms")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.5

    ax.bar(x, t_kernel, width, label=r'$T^{\mathrm{kernel}}$ (Effective Compute)', color='#2ca02c')
    ax.bar(x, t_dispatch, width, bottom=t_kernel, label=r'$T^{\mathrm{dispatch}}$ (OS/Driver Overhead)', color='#d62728')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Time (ms)", fontweight="bold")
    ax.set_title(r"Temporal Latency Decomposition per Layer ($T^{\mathrm{wall}}$)", pad=15, fontweight="bold")
    ax.legend(frameon=True)
    sns.despine()

    fig.tight_layout()
    fig.savefig(output_dir / "temporal_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_congestion_knee(meta_json_path: Path | None, output_dir: Path):
    """Generates Figure 4.3: Congestion Knee using the REAL measured PCIe
    calibration (transfer_alpha_h2d/transfer_beta_h2d/transfer_beta_h2d_congested/
    transfer_congestion_knee_h2d_mb), written by
    training_profiler.py::_measure_pci_bandwidth_detailed into every
    <model>_meta.json -- never fabricated/forced."""
    if meta_json_path is None or not meta_json_path.exists():
        print(f"Warning: no real *_meta.json found under the given input root. Using fallback parameters (NOT measured).")
        alpha = 0.05
        beta_nom = 12.0
        beta_cong = 6.0
        knee = 128.0
        source = "fallback (no real meta.json found)"
    else:
        with open(meta_json_path, "r") as f:
            meta = json.load(f)
        alpha = meta.get("transfer_alpha_h2d", 0.05)
        beta_nom = meta.get("transfer_beta_h2d", 12.0)
        beta_cong = meta.get("transfer_beta_h2d_congested", beta_nom * 0.8)
        knee = meta.get("transfer_congestion_knee_h2d_mb", 128.0)
        source = meta.get("transfer_calibration_source", "unknown")
        print(f"  Real PCIe calibration used (source={source}, file={meta_json_path}):")
        print(f"    alpha={alpha:.4f}ms  beta_nom={beta_nom:.2f}GB/s  beta_cong={beta_cong:.2f}GB/s  knee={knee:.1f}MB")

    sizes_mb = np.linspace(1, 300, 300)

    def calc_latency(s):
        if s <= knee:
            return alpha + (s / beta_nom)
        else:
            return alpha + (knee / beta_nom) + ((s - knee) / beta_cong)

    latencies = np.array([calc_latency(s) for s in sizes_mb])
    ideal_latencies = alpha + (sizes_mb / beta_nom)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes_mb, latencies, color='#1f77b4', linewidth=3, label=r'Piecewise Calibrated Model ($t_{\mathrm{dir}}$)')
    ax.plot(sizes_mb, ideal_latencies, color='gray', linestyle='--', label=r'Nominal Extrapolation (Ideal)')

    ax.axvline(x=knee, color='#d62728', linestyle=':', linewidth=2, label=rf'Congestion Knee ($S_{{\mathrm{{knee}}}}={knee:.0f}$ MB)')

    ax.set_xlabel("Tensor Transfer Size $S$ (MB)", fontweight="bold")
    ax.set_ylabel("PCIe Latency (ms)", fontweight="bold")
    ax.set_title("Directional Calibration and Congestion Knee", pad=15, fontweight="bold")
    ax.legend(frameon=True, loc='lower right', fontsize=10)
    sns.despine()

    fig.tight_layout()
    fig.savefig(output_dir / "congestion_knee.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_tension(output_dir: Path):
    """Generates Figure 4.4: Overlap factor vs Branching pressure.

    Deliberately NOT backed by a per-layer measurement: this figure visualizes
    the SHAPE of the parametric cost model (tau as a function of the overlap
    degree sigma and branching pressure p_u), not a specific measured
    quantity -- t_base/kappa are illustrative constants for that purpose, same
    as the thesis caption frames it ("superficie parametrica").
    """
    sigma_vals = np.linspace(0, 1, 50)
    p_u_vals = [0, 1, 2, 4]  # Diferentes niveles de ramificación
    t_base = 10.0  # ms arbitrario (ilustrativo, no medido -- ver docstring)
    kappa = 0.5  # factor de penalización (ilustrativo, no medido -- ver docstring)

    fig, ax = plt.subplots(figsize=(8, 6))  # Increased height to accommodate bottom legend

    colors = sns.color_palette("rocket", len(p_u_vals))

    for i, p_u in enumerate(p_u_vals):
        t_eff = t_base * (1 - 0.5 * sigma_vals) * (1 + kappa * p_u)
        ax.plot(sigma_vals, t_eff, linewidth=2.5, color=colors[i], label=rf'Branching Pressure $p_u = {p_u}$')

    ax.set_xlabel(r"Asynchronous Overlap Degree ($\sigma$)", fontweight="bold")
    ax.set_ylabel(r"Effective Latency $t_{\mathrm{edge}}^{\mathrm{eff}}$ (ms)", fontweight="bold")
    ax.set_title(r"Overlap Attenuation vs. Branching Penalty", pad=15, fontweight="bold")

    ax.set_ylim(bottom=0)

    ax.annotate("More efficient\n(Independent streams)", xy=(0.95, 5.2), xytext=(0.5, 2.5),
                arrowprops=dict(arrowstyle="->", linestyle="--", color='black', shrinkA=5, shrinkB=5),
                fontsize=9, fontweight="bold", ha='center')

    ax.annotate("Massive congestion\n(Fan-out blocking)", xy=(0.05, 29), xytext=(0.4, 20),
                arrowprops=dict(arrowstyle="->", linestyle="--", color='black', shrinkA=5, shrinkB=5),
                fontsize=9, fontweight="bold", ha='center')

    ax.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    sns.despine()

    fig.tight_layout()
    fig.savefig(output_dir / "overlap_tension.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate methodological figures for Chapter 4")
    parser.add_argument("--input_root", type=Path, required=True,
                         help="Root directory containing model profiling results (e.g. data/<host>/results_thesis_mode/<profile>) -- searched recursively for a real *_metrics_stats.csv and *_meta.json")
    parser.add_argument("--preferred_model", default="resnet50",
                         help="Model name to prefer when more than one is present under input_root (default: resnet50, a CNN with Conv2d/BatchNorm2d/ReLU/Linear layer diversity)")
    parser.add_argument("--output_dir", type=Path, default=Path("reports/zephyr/doctoral_minimal/methodology_plots"), help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = _find_metrics_csv(args.input_root, args.preferred_model)
    meta_path = _find_meta_json(args.input_root, args.preferred_model)

    print(f"Generando figuras metodológicas en {args.output_dir}...")

    if metrics_csv is None:
        print(f"Warning: no *_metrics_stats.csv found under {args.input_root} -- skipping temporal_decomposition.png")
    else:
        print(f"  Using real metrics: {metrics_csv}")
        plot_temporal_decomposition(metrics_csv, args.output_dir)
        print("✓ Generada Descomposición Temporal (T_wall vs T_kernel) -- datos reales")

    plot_congestion_knee(meta_path, args.output_dir)
    print("✓ Generada Rodilla de Congestión (PCIe Knee)" + (" -- datos reales" if meta_path is not None else " -- SIN datos reales, usando respaldo"))

    plot_overlap_tension(args.output_dir)
    print("✓ Generada Curva de Tensión Solapamiento/Presión (visualización teórica del modelo, ver docstring)")

    print("Proceso completado con éxito.")


if __name__ == "__main__":
    main()
