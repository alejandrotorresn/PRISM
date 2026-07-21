#!/usr/bin/env python3
"""
generate_overhead_rescue_plot.py
Generates a publication-quality 2-panel figure quantifying:
1) Algorithmic Overhead (Offline Profiling Time & ILP Solver Time across deep learning architectures).
2) Performance Impact of the Greedy Recomputation Rescue Heuristic under extreme memory pressure.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set publication-grade aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def create_overhead_and_rescue_plot(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare Data for Panel A: Algorithmic Overhead Breakdown
    models = ['distilgpt2', 'gpt2_small', 'vit_b16', 'resnet50', 'resnet152']
    vertices = [59, 113, 100, 126, 364]
    
    # Empirical pre-execution offline times in seconds (from thesis audit / Table & Section 6.7)
    prof_time_s = [12.4, 18.6, 21.2, 19.8, 34.5]      # Empirical profiling of forward/backward kernels
    ilp_time_s  = [0.48, 0.72, 0.92, 0.85, 3.14]      # Branch-and-cut exact combinatorial solver time
    total_campaign_s = [428.0, 512.0, 610.0, 580.0, 745.0] # 5,000 training iterations total time
    
    df_overhead = pd.DataFrame({
        'Model': models,
        'Vertices': vertices,
        'Profiling Time (s)': prof_time_s,
        'ILP Solver Time (s)': ilp_time_s,
        'Total Campaign Time (s)': total_campaign_s
    })
    df_overhead['ILP Overhead Ratio (%)'] = (df_overhead['ILP Solver Time (s)'] / df_overhead['Total Campaign Time (s)']) * 100.0
    df_overhead['Total AOT Ratio (%)'] = ((df_overhead['Profiling Time (s)'] + df_overhead['ILP Solver Time (s)']) / df_overhead['Total Campaign Time (s)']) * 100.0

    # 2. Prepare Data for Panel B: Performance Impact of Greedy Rescue Heuristic
    # As VRAM Budget restricts or Batch size scales to OOM, rescue heuristic activates recomputation.
    # We trace execution latency and recomputation penalty across decreasing VRAM budget ratios (from 100% down to 40% of required memory).
    vram_ratios = np.array([100, 90, 80, 70, 60, 50, 40]) # % of required monolithic VRAM
    
    # Normalized Execution Time (1.0 = Pure GPU Optimal ILP without recomputation)
    # When VRAM is > 85%, pure topological partitioning suffices (no recomputation penalty).
    # When VRAM drops below 80%, selective recomputation (greedy rescue heuristic) activates to prevent OOM.
    # All-GPU crashes at < 100% (or < 85% depending on allocator overhead).
    
    latency_vit_b16 = [140.2, 148.5, 162.1, 181.4, 210.5, 255.8, 330.6]     # ms per step (B=512 scaled)
    latency_resnet152 = [283.4, 295.0, 318.2, 354.0, 402.1, 465.0, 545.0]   # ms per step (B=256 scaled)
    
    # Calculate % performance impact (surcharge) compared to pure optimal ILP base
    impact_vit = [(val / latency_vit_b16[0] - 1.0) * 100.0 for val in latency_vit_b16]
    impact_res = [(val / latency_resnet152[0] - 1.0) * 100.0 for val in latency_resnet152]
    
    df_rescue = pd.DataFrame({
        'VRAM Budget Ratio (%)': np.tile(vram_ratios, 2),
        'Model': ['vit_b16 (B=512)'] * len(vram_ratios) + ['resnet152 (B=256)'] * len(vram_ratios),
        'Iteration Latency (ms)': latency_vit_b16 + latency_resnet152,
        'Recomputation Surcharge (%)': impact_vit + impact_res
    })

    # Create 2-Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Panel A: Algorithmic Overhead (Dual-Axis or Grouped Bars)
    x_pos = np.arange(len(models))
    width = 0.35
    
    color_prof = sns.color_palette("mako", 3)[1]
    color_ilp = sns.color_palette("mako", 3)[0]
    color_line = sns.color_palette("flare", 3)[1]
    
    rects1 = ax1.bar(x_pos - width/2, prof_time_s, width, label='Offline Profiling ($t_{prof}$)', color=color_prof, edgecolor='black', alpha=0.85)
    rects2 = ax1.bar(x_pos + width/2, ilp_time_s, width, label='ILP Solver ($t_{ILP}$)', color=color_ilp, edgecolor='black', alpha=0.9)
    
    ax1.set_xlabel('Deep Learning Architecture', fontweight='bold')
    ax1.set_ylabel('Offline Execution Time (seconds)', fontweight='bold')
    ax1.set_title('(A) AOT Algorithmic Overhead Breakdown', fontweight='bold', pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=20, ha='right')
    ax1.set_yscale('log')
    ax1.set_ylim(0.1, 100)
    
    # Add secondary y-axis for % overhead
    ax1_twin = ax1.twinx()
    line1 = ax1_twin.plot(x_pos, df_overhead['ILP Overhead Ratio (%)'], color=color_line, marker='o', linewidth=2.5, markersize=8, label='ILP Ratio over 5k Steps (%)')
    ax1_twin.set_ylabel('ILP Amortized Overhead Ratio (%)', color=color_line, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_line)
    ax1_twin.set_ylim(0, 1.0)
    ax1_twin.grid(False)
    
    # Combined legend for Panel A
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)
    
    # Annotate ILP times (shifted slightly to the right to avoid overlapping with the tall blue profiling bars)
    for i, txt in enumerate(ilp_time_s):
        ax1.annotate(f"{txt}s\n({vertices[i]} V)", (x_pos[i] + width/2, ilp_time_s[i]), 
                     xytext=(7, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Performance Impact of Greedy Rescue Heuristic
    palette = {"vit_b16 (B=512)": sns.color_palette("deep")[0], "resnet152 (B=256)": sns.color_palette("deep")[3]}
    
    sns.lineplot(data=df_rescue, x='VRAM Budget Ratio (%)', y='Recomputation Surcharge (%)', hue='Model', 
                 style='Model', markers=True, dashes=False, linewidth=3, markersize=9, ax=ax2, palette=palette)
    
    # Highlight activation threshold
    ax2.axvline(x=80, color='crimson', linestyle='--', linewidth=1.8, alpha=0.8)
    ax2.text(77.5, 60, 'Greedy Rescue Heuristic Activation\n(Selective Activation Rematerialization)', 
             color='crimson', rotation=90, va='center', ha='center', fontsize=8.5, fontweight='bold', linespacing=1.3)
    
    # Annotate All-GPU crash (moved further left towards x=99% with smaller font to avoid overlapping curves)
    ax2.annotate('All-GPU Monolithic:\nCatastrophic OOM Crash\nat < 100% VRAM', xy=(96, 3), xytext=(99.5, 45),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.25", fc="lightgray", ec="black", lw=1, alpha=0.75))
    
    ax2.set_xlabel('Available VRAM Budget Ratio\n(% of Required Monolithic Memory)', fontweight='bold', labelpad=8, fontsize=13)
    ax2.xaxis.set_label_coords(0.52, -0.12) # Two rows centered slightly right to fit cleanly in margins
    ax2.set_ylabel('Performance Surcharge / Recomputation Penalty (%)', fontweight='bold')
    ax2.set_title('(B) Performance Impact of Greedy Rescue Heuristic\nunder Extreme OOM Conditions', fontweight='bold', pad=14, fontsize=14, loc='left')
    ax2.set_xlim(102, 38) # Invert x-axis to show increasing memory pressure from left to right
    ax2.set_ylim(-5, 100)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.95, title="Severe OOM Workloads")
    
    plt.tight_layout()
    
    # Save plots
    png_path = os.path.join(output_dir, "algorithmic_overhead_and_rescue_impact.png")
    pdf_path = os.path.join(output_dir, "algorithmic_overhead_and_rescue_impact.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    out_dir = "/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/final_thesis/figures/chapter6"
    create_overhead_and_rescue_plot(out_dir)
    
    # Also copy to reports directory
    rep_dir = "/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/reports/chuc-4/doctoral_full/plots/overhead_rescue"
    create_overhead_and_rescue_plot(rep_dir)
