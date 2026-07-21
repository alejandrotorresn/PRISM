#!/usr/bin/env python3
"""
generate_energy_gpu_prism_comparison.py

Generates a publication-quality doctoral figure for Chapter 6 illustrating:
Absolute electrical energy consumption per iteration ($E_{iter}$ in Joules/step) across
representative batch regimes where PRISM distributes the graph across memory hierarchy
or where All-GPU execution is feasible/infeasible, contrasting:
1. All-GPU Monolithic Baseline (or OOM crash indication)
2. Greedy Heuristic Dispatch
3. PRISM Combinatorial ILP (Optimal Partition)

All text inside the figure is strictly in English.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set publication-grade aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def create_energy_gpu_prism_plot(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Empirical energy measurements (Joules per step) from CHUC cluster evaluation
    # Regimes:
    # 1. ViT-B/16 (B=64, 32GB VRAM): Feasible on All-GPU
    # 2. ViT-B/16 (B=512, 32GB VRAM): All-GPU OOM (requires 42.7 GB), PRISM vs Greedy partition
    # 3. ResNet-50 (B=256, 32GB VRAM): All-GPU feasible/borderline, PRISM vs Greedy
    # 4. ResNet-152 (B=256, 32GB VRAM): All-GPU OOM (requires 41.5 GB), PRISM vs Greedy partition
    
    regimes = [
        "ViT-B/16\n($B=64$)",
        "ViT-B/16\n($B=512$, OOM)",
        "ResNet-50\n($B=256$)",
        "ResNet-152\n($B=256$, OOM)"
    ]
    
    # Energy per step in Joules (J)
    all_gpu_j = np.array([142.50, np.nan, 238.59, np.nan]) # NaN where All-GPU crashes with OOM
    greedy_j  = np.array([140.34, 403.14, 525.52, 2386.89])
    prism_j   = np.array([140.21, 357.08, 326.78, 892.94])
    
    x = np.arange(len(regimes))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    
    # Plot grouped bars
    b_gpu = ax.bar(x - width, np.nan_to_num(all_gpu_j, nan=0.0), width, label="Monolithic GPU (All-GPU)", color=sns.color_palette("mako", 4)[1], edgecolor="black", alpha=0.9)
    b_grd = ax.bar(x, greedy_j, width, label="Greedy Heuristic Partition", color=sns.color_palette("rocket", 3)[1], edgecolor="black", alpha=0.88)
    b_prm = ax.bar(x + width, prism_j, width, label="PRISM Combinatorial Optimal (ILP)", color=sns.color_palette("viridis", 4)[2], edgecolor="black", alpha=0.92)
    
    # Add explicit annotations and OOM indicators with pristine spacing on log scale
    for i in range(len(x)):
        # 1. Annotate All-GPU bar
        if np.isnan(all_gpu_j[i]):
            # Place a clean rotated badge directly in the empty slot at x[i] - width, near the bottom so it never touches other bars or labels
            ax.text(x[i] - width, 52.0, "All-GPU: OOM Crash", ha='center', va='bottom', rotation=90,
                    color='crimson', fontweight='bold', fontsize=8.5, 
                    bbox=dict(boxstyle="round,pad=0.2", fc="lavenderblush", ec="crimson", lw=1.0, alpha=0.95))
        else:
            # Multiplicative offset on log scale for clean placement slightly above bar
            ax.text(x[i] - width, all_gpu_j[i] * 1.06, f"{all_gpu_j[i]:.1f} J", ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='darkslategray')
            
        # 2. Annotate Greedy bar
        ax.text(x[i], greedy_j[i] * 1.06, f"{greedy_j[i]:.1f} J", ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='darkred')
        
        # 3. Annotate PRISM bar
        ax.text(x[i] + width, prism_j[i] * 1.06, f"{prism_j[i]:.1f} J", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='darkgreen')
        
        # 4. Clean percentage savings or efficiency badges placed well above the bar labels (zero overlap/collision)
        save_pct = ((greedy_j[i] - prism_j[i]) / greedy_j[i]) * 100.0
        if save_pct > 5.0:
            # Place badge high above the tallest bar in the group
            badge_y = max(greedy_j[i], prism_j[i]) * 1.45
            if i == 3: # For ResNet-152, ensure enough headroom below top limit
                badge_y = greedy_j[i] * 1.35
            ax.annotate(f"-{save_pct:.1f}% Energy\nvs. Greedy", 
                        xy=(x[i] + width/2, max(greedy_j[i], prism_j[i]) * 1.15), 
                        xytext=(x[i] + width/2, badge_y),
                        ha='center', va='bottom', fontweight='bold', color='darkgreen', fontsize=9.0,
                        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.6),
                        bbox=dict(boxstyle="round,pad=0.25", fc="honeydew", ec="darkgreen", lw=1.0, alpha=0.98))
        elif not np.isnan(all_gpu_j[i]):
            # Place "Matches GPU" cleanly above the pair for ViT-B/16 B=64
            ax.annotate("Matches GPU\nEfficiency", 
                        xy=(x[i] - width/2, all_gpu_j[i] * 1.15), 
                        xytext=(x[i] - width/2, all_gpu_j[i] * 1.55),
                        ha='center', va='bottom', fontweight='bold', color='darkslategray', fontsize=8.5,
                        arrowprops=dict(arrowstyle='->', color='darkslategray', lw=1.4),
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightcyan", ec="darkslategray", lw=0.9, alpha=0.95))

    ax.set_yscale("log")
    ax.set_ylim(35, 22000)
    ax.set_ylabel("Electrical Energy Consumption per Step ($E_{iter}$ in Joules, log scale)", fontweight="bold", labelpad=14)
    ax.set_xlabel("Neural Architecture and Batch Size Regime ($B$)", fontweight="bold", labelpad=12)
    ax.set_title("Energy Expenditure Comparison: All-GPU vs. Greedy Heuristic vs. PRISM Optimal Partitioning", fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontweight="bold")
    ax.grid(True, axis='y', which="both", linestyle='--', alpha=0.5)
    
    # Add summary text box inside the plot at top left with clean padding
    ax.text(0.02, 0.965, "Under OOM conditions where All-GPU crashes, PRISM eliminates PCIe congestion thrashing,\nreducing electrical expenditure per step by up to 62.6% compared to Greedy dispatch.",
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="darkgoldenrod", lw=1.2, alpha=0.95))
            
    # Legend placed upper right, now floating well above ResNet-152 thanks to ylim=22000
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, borderpad=0.6)
    sns.despine()
    
    fig.tight_layout()
    
    png_path = output_dir / "energy_comparison_gpu_prism_greedy.png"
    pdf_path = output_dir / "energy_comparison_gpu_prism_greedy.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Generated energy plot successfully:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    out_dir = Path("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/final_thesis/figures/chapter6")
    create_energy_gpu_prism_plot(out_dir)
    
    rep_dir = Path("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/reports/chuc-4/doctoral_full/plots/energy")
    create_energy_gpu_prism_plot(rep_dir)
