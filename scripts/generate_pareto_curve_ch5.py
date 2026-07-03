#!/usr/bin/env python3
"""
Generates an ultra-clean, publication-grade English Pareto Frontier plot for ResNet-152 (Chapter 5).
Adheres strictly to IEEE/ACM minimalist standards:
- 100% English titles, axis labels, and legend entries.
- Zero clutter: NO callout text boxes inside the canvas, NO crossing arrows.
- Smooth, continuous Pareto trade-off curves across memory constraints for Batch Size 128 and 256.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Professional academic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11.5
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 13.5
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11

def main():
    # Cache b256 grid if available in /tmp
    tmp_b256 = Path("/tmp/test_b256_grid.csv")
    cache_b256 = Path("data/resnet152_pareto_sweep_b256.csv")
    if tmp_b256.exists():
        shutil.copy(tmp_b256, cache_b256)
        
    cache_b128 = Path("data/resnet152_pareto_sweep_b128.csv")
    
    # Load consolidated ILP execution data for background candidate scatter
    csv_cons = Path("reports/chuc-4/doctoral_full/csv/ilp_pareto_consolidated.csv")
    df_cons = pd.read_csv(csv_cons) if csv_cons.exists() else pd.DataFrame()
    
    fig, ax = plt.subplots(figsize=(11.0, 7.0), dpi=300)
    
    # 1. Background Scatter: Candidate ILP Evaluations
    if not df_cons.empty:
        sub_128 = df_cons[(df_cons['model'] == 'resnet152') & (df_cons['batch_size'] == 128) & (df_cons['sim_time_ms'] > 0)]
        sub_256 = df_cons[(df_cons['model'] == 'resnet152') & (df_cons['batch_size'] == 256) & (df_cons['sim_time_ms'] > 0)]
        
        if not sub_128.empty:
            ax.scatter(sub_128['sim_time_ms']/1000.0, sub_128['sim_energy_j'], color='#1f77b4', alpha=0.25, s=50,
                       edgecolors='none', label='ILP Evaluated Candidates (Batch Size 128)', zorder=2)
        if not sub_256.empty:
            ax.scatter(sub_256['sim_time_ms']/1000.0, sub_256['sim_energy_j'], color='#d62728', alpha=0.25, s=50,
                       edgecolors='none', label='ILP Evaluated Candidates (Batch Size 256)', zorder=2)

    # 2. Smooth Pareto Frontier Curves Across Memory Budgets
    if cache_b128.exists():
        df_128 = pd.read_csv(cache_b128)
        df_128 = df_128[df_128['sim_time_ms'] > 0].drop_duplicates(subset=['sim_time_ms', 'sim_energy_j']).sort_values('sim_time_ms')
        ax.plot(df_128['sim_time_ms']/1000.0, df_128['sim_energy_j'], color='#0d47a1', linewidth=3.0,
                marker='o', markersize=8.5, markerfacecolor='#1f77b4', markeredgecolor='white', markeredgewidth=1.5,
                label='Pareto Optimal Frontier (Batch Size 128)', zorder=4)
                
    if cache_b256.exists():
        df_256 = pd.read_csv(cache_b256)
        df_256 = df_256[df_256['sim_time_ms'] > 0].drop_duplicates(subset=['sim_time_ms', 'sim_energy_j']).sort_values('sim_time_ms')
        ax.plot(df_256['sim_time_ms']/1000.0, df_256['sim_energy_j'], color='#b71c1c', linewidth=3.0,
                marker='s', markersize=8.5, markerfacecolor='#d62728', markeredgecolor='white', markeredgewidth=1.5,
                label='Pareto Optimal Frontier (Batch Size 256)', zorder=4)

    ax.set_title("Empirical Pareto Frontiers: Execution Time vs. Dynamic Energy Dissipation\n(ResNet-152 ILP-PRISM Multi-Criteria Optimization Across Memory Constraints)",
                 fontsize=13.5, fontweight='bold', pad=18)
    ax.set_xlabel("Execution Time per Iteration / Makespan ($C_T$, Seconds)", fontsize=12.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Total Dynamic Energy Dissipation ($C_E$, Joules)", fontsize=12.5, fontweight='bold', labelpad=10)
    
    # Set clean axes limits
    ax.set_xlim(0.2, 7.5)
    ax.set_ylim(150, 4200)
    
    # Grid and Legend placed cleanly
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.legend(loc='upper left', frameon=True, framealpha=0.98, facecolor='white', 
              edgecolor='#adb5bd', fontsize=11, borderpad=1.0)
    
    out_path = Path("final_thesis/figures/chapter5/pareto_frontier_resnet152.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved clean Pareto frontier plot to {out_path}")
    return 0

if __name__ == "__main__":
    main()
