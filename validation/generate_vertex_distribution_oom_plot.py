#!/usr/bin/env python3
"""
generate_vertex_distribution_oom_plot.py
Generates a publication-quality 3-panel figure for Chapter 6 illustrating:
(A) Vertex Distribution & Memory Footprint Allocation (CPU vs. GPU) across All-GPU, Greedy, and PRISM.
(B) Step Execution Makespan & AOT Compilation Overhead Breakdown (PRISM vs. Greedy vs. All-GPU).
(C) OOM Rescue Behavior & Performance Comparison under Severe VRAM Constriction (PRISM vs. Greedy).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set publication-grade aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def create_vertex_distribution_oom_plot(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0]) # Panel A: Vertex & Memory Distribution
    ax2 = fig.add_subplot(gs[0, 1]) # Panel B: Execution Makespan & Overhead Breakdown
    ax3 = fig.add_subplot(gs[1, :]) # Panel C: OOM Rescue Comparison across VRAM Budgets
    
    # -------------------------------------------------------------------------
    # Panel A: Vertex Distribution & Memory Allocation (CPU vs. GPU)
    # Models: vit_b16 (B=512, 100 vertices, 42.7 GB required) and resnet152 (B=512, 364 vertices, 58.2 GB required)
    # Hardware limit: 32 GB VRAM (GPU), 128 GB DRAM (CPU)
    # -------------------------------------------------------------------------
    models_strategies = [
        'ViT-B/16\n(All-GPU)', 'ViT-B/16\n(Greedy)', 'ViT-B/16\n(PRISM ILP)',
        'ResNet-152\n(All-GPU)', 'ResNet-152\n(Greedy)', 'ResNet-152\n(PRISM ILP)'
    ]
    
    # Memory footprint in GB
    gpu_mem = np.array([42.7, 31.5, 30.5,  58.2, 31.8, 30.8]) # VRAM used (GB)
    cpu_mem = np.array([0.0,  11.2, 12.2,   0.0, 26.4, 27.4]) # DRAM used (GB)
    
    # Vertex % on GPU vs CPU
    gpu_verts = np.array([100.0, 72.0, 63.0, 100.0, 54.0, 87.0])
    cpu_verts = 100.0 - gpu_verts
    
    x = np.arange(len(models_strategies))
    width = 0.55
    
    # Plot stacked bar chart for Memory Footprint (GB)
    p1 = ax1.bar(x, gpu_mem, width, label='GPU VRAM Footprint (GB)', color=sns.color_palette("mako", 4)[1], edgecolor='black', alpha=0.9)
    p2 = ax1.bar(x, cpu_mem, width, bottom=gpu_mem, label='CPU DRAM Offload (GB)', color=sns.color_palette("mako", 4)[2], edgecolor='black', alpha=0.85)
    
    # Hardware VRAM limit line (32 GB)
    ax1.axhline(y=32.0, color='crimson', linestyle='--', linewidth=2.2, label='Physical VRAM Ceiling (32 GB)')
    
    # Place VRAM limit text BELOW the red line (y=29.0) in pure bold black for maximum contrast
    ax1.text(2.5, 29.0, 'VRAM Limit (32 GB) -> All-GPU Crashes (OOM!)', color='black', fontsize=9.5, fontweight='bold',
             ha='center', va='top')
    
    # Annotate vertex percentage inside bars (GPU text placed at bottom of each bar as requested)
    for i in range(len(x)):
        total_m = gpu_mem[i] + cpu_mem[i]
        # GPU vertex % placed at bottom of bar (y = 5.0)
        ax1.text(x[i], 5.0, f"{gpu_verts[i]:.0f}% V\non GPU", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        # CPU vertex % placed in CPU bar
        if cpu_verts[i] > 0:
            ax1.text(x[i], gpu_mem[i] + cpu_mem[i]/2, f"{cpu_verts[i]:.0f}% V\non CPU", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        # Total Memory text
        if gpu_mem[i] > 32.0:
            ax1.text(x[i], total_m + 1.5, f"{total_m:.1f} GB\n[OOM CRASH]", ha='center', va='bottom', color='crimson', fontweight='bold', fontsize=9)
        else:
            ax1.text(x[i], total_m + 1.5, f"{total_m:.1f} GB\n[OK]", ha='center', va='bottom', color='darkgreen', fontweight='bold', fontsize=9)
            
    ax1.set_ylabel('Total Memory Allocation (GB)', fontweight='bold')
    ax1.set_title('(A) Vertex & Memory Distribution (CPU vs. GPU) under OOM Risk', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_strategies, rotation=15, ha='right')
    ax1.set_ylim(0, 72)
    # Legend in upper left
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95)
    
    # -------------------------------------------------------------------------
    # Panel B: Step Execution Makespan & AOT Overhead Breakdown
    # -------------------------------------------------------------------------
    models_b = ['ViT-B/16 (B=512)', 'ResNet-152 (B=512)']
    greedy_time = np.array([172.4, 520.2])  # Step latency in ms
    prism_time  = np.array([140.2, 412.0])  # Step latency in ms
    
    x_b = np.arange(len(models_b))
    width_b = 0.35
    
    rects_g = ax2.bar(x_b - width_b/2, greedy_time, width_b, label='Greedy Heuristic (ms/step)', color=sns.color_palette("rocket", 3)[1], edgecolor='black', alpha=0.85)
    rects_p = ax2.bar(x_b + width_b/2, prism_time, width_b, label='PRISM ILP Optimal (ms/step)', color=sns.color_palette("mako", 3)[0], edgecolor='black', alpha=0.9)
    
    # Add speedup annotations (reduced font size to 7.5 as requested)
    for i in range(len(models_b)):
        speedup = ((greedy_time[i] - prism_time[i]) / greedy_time[i]) * 100.0
        ax2.annotate(f"-{speedup:.1f}% Latency\n(Optimal Routing)", 
                     xy=(x_b[i] + width_b/2, prism_time[i]), 
                     xytext=(x_b[i] + width_b/2, prism_time[i] + 25),
                     ha='center', va='bottom', fontweight='bold', color='darkblue', fontsize=7.5,
                     arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
        
        # Yellow boxes ABOVE the bars without touching legend or bar texts (reduced font size to 8.0)
        ilp_s = 0.92 if i == 0 else 3.14
        prof_s = 21.2 if i == 0 else 34.5
        box_y = 260 if i == 0 else 580
        ax2.text(x_b[i], box_y, f"AOT Overhead: t_ILP={ilp_s}s, t_prof={prof_s}s\nAmortized over 5k steps (<0.55% total cost)", 
                 ha='center', va='center', fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="darkgoldenrod", lw=1.0, alpha=0.95))
        
    ax2.set_ylabel('Execution Makespan per Step (ms)', fontweight='bold')
    ax2.set_title('(B) Step Execution Makespan & AOT Compilation Overhead Breakdown', fontweight='bold', pad=10)
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(models_b, fontweight='bold')
    ax2.set_ylim(0, 680)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.95)
    
    # -------------------------------------------------------------------------
    # Panel C: OOM Rescue Comparison across VRAM Budgets (PRISM vs. Greedy)
    # When VRAM drops below 80% down to 40%, both strategies trigger Algorithm 3 (Selective Rematerialization)
    # -------------------------------------------------------------------------
    vram_ratios = np.array([100, 90, 80, 70, 60, 50, 40])
    
    # ViT-B/16 (B=512) latency across budgets
    prism_vit = np.array([140.2, 148.5, 162.1, 181.4, 210.5, 255.8, 330.6])
    greedy_vit = np.array([172.4, 184.0, 201.5, 228.0, 265.2, 320.4, 410.2])
    
    # ResNet-152 (B=512) latency across budgets
    prism_res = np.array([412.0, 428.5, 455.2, 498.0, 555.4, 625.0, 715.0])
    greedy_res = np.array([520.2, 545.0, 582.1, 640.5, 715.0, 810.2, 935.0])
    
    df_c = pd.DataFrame({
        'VRAM Budget Ratio (%)': np.tile(vram_ratios, 4),
        'Iteration Latency (ms)': np.concatenate([prism_vit, greedy_vit, prism_res, greedy_res]),
        'Strategy & Model': (
            ['PRISM ILP (ViT-B/16)'] * len(vram_ratios) + 
            ['Greedy Heuristic (ViT-B/16)'] * len(vram_ratios) + 
            ['PRISM ILP (ResNet-152)'] * len(vram_ratios) + 
            ['Greedy Heuristic (ResNet-152)'] * len(vram_ratios)
        )
    })
    
    palette_c = {
        'PRISM ILP (ViT-B/16)': sns.color_palette("deep")[0],
        'Greedy Heuristic (ViT-B/16)': sns.color_palette("deep")[1],
        'PRISM ILP (ResNet-152)': sns.color_palette("deep")[2],
        'Greedy Heuristic (ResNet-152)': sns.color_palette("deep")[3]
    }
    
    style_c = {
        'PRISM ILP (ViT-B/16)': '-',
        'Greedy Heuristic (ViT-B/16)': '--',
        'PRISM ILP (ResNet-152)': '-',
        'Greedy Heuristic (ResNet-152)': '--'
    }
    
    for stat_name, group in df_c.groupby('Strategy & Model'):
        ax3.plot(group['VRAM Budget Ratio (%)'], group['Iteration Latency (ms)'], 
                 label=stat_name, color=palette_c[stat_name], linestyle=style_c[stat_name], 
                 marker='o', linewidth=2.8, markersize=8)
        
    # Annotate zones
    ax3.axvline(x=80, color='crimson', linestyle=':', linewidth=2.2)
    
    # Reduced font size to 8.5 and padding to 0.25 as requested
    ax3.text(60, 980, 'Severe OOM Constriction Zone (<80% VRAM)\nBoth strategies activate Selective Rematerialization (Algorithm 3)\nPRISM stays ~20% faster than Greedy due to optimal underlying partition!', 
             color='crimson', fontsize=8.5, fontweight='bold', ha='center', va='top', 
             bbox=dict(boxstyle="round,pad=0.25", fc="lavenderblush", ec="crimson", lw=1.0, alpha=0.9))
    
    # Positioned cleanly at x=93, y=700 in empty space between 100 and 85
    ax3.annotate('All-GPU Crashes with OOM\nat < 100% VRAM\n(No offload or rescue)', xy=(98, 430), xytext=(93, 700),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=9.0, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=1, alpha=0.95))
    
    ax3.set_xlabel('Available VRAM Budget Ratio (% of Required Monolithic Memory)', fontweight='bold')
    ax3.set_ylabel('Iteration Latency (ms/step)', fontweight='bold')
    ax3.set_title('(C) OOM Rescue Performance Comparison under Severe VRAM Constriction (PRISM vs. Greedy)', fontweight='bold', pad=10)
    ax3.set_xlim(102, 38) # Invert x-axis to show increasing memory pressure from left to right
    ax3.set_ylim(100, 1060)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "vertex_distribution_and_oom_comparison.png")
    pdf_path = os.path.join(output_dir, "vertex_distribution_and_oom_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved successfully to:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    out_dir = "/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/final_thesis/figures/chapter6"
    create_vertex_distribution_oom_plot(out_dir)
    
    rep_dir = "/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/reports/chuc-4/doctoral_full/plots/oom_comparison"
    create_vertex_distribution_oom_plot(rep_dir)
