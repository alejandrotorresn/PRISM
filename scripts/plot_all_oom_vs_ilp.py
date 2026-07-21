import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_oom_comparisons(base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_files = glob.glob(f"{base_dir}/*/*/*/*/*_metrics_stats.csv")
    
    generated_count = 0
    for metrics_file in metrics_files:
        config_dir = os.path.dirname(metrics_file)
        parts = config_dir.split('/')
        model, opt, prec, batch = parts[-4], parts[-3], parts[-2], parts[-1]
        
        ilp_dirs = glob.glob(f"{config_dir}/ilp_solution_pareto_best_budget_*")
        if ilp_dirs:
            ilp_dir = sorted(ilp_dirs)[0]
        else:
            ilp_dir = f"{config_dir}/ilp_solution"
            
        assignment_file = f"{ilp_dir}/ilp_assignment.csv"
        if not os.path.exists(assignment_file):
            continue
            
        try:
            metrics_df = pd.read_csv(metrics_file)
            assign_df = pd.read_csv(assignment_file)
        except Exception as e:
            print(f"Error reading {config_dir}: {e}")
            continue
            
        if len(metrics_df) != len(assign_df):
            # Mismatch in layers, probably edge case
            continue
            
        metrics_df = metrics_df.sort_index()
        
        # Merge dataframes just to be safe they align by index
        df = pd.concat([metrics_df.reset_index(drop=True), assign_df.reset_index(drop=True)], axis=1)
        
        # Determine if this was an intervened model (OOM saved)
        has_intervention = (df['device'] != 'GPU').any() or (df['activation_strategy'] != 'retain').any()
        
        total_static_mem = df['params_mb_mean'].sum() + df['grads_mb_mean'].sum() + df['optimizer_states_mb_mean'].sum()
        total_naive_peak = total_static_mem + df['activations_mb_mean'].sum()
        
        if not has_intervention and total_naive_peak < 7500:
            continue
            
        # Calculate Forward Pass Memory Accumulation
        layers = np.arange(len(df))
        
        # 1. Antes (GPU Only)
        before_vram = []
        acc_act = 0.0
        for i in range(len(df)):
            acc_act += df['activations_mb_mean'].iloc[i]
            before_vram.append(total_static_mem + acc_act)
            
        # 2. Despues (ILP)
        after_vram = []
        after_dram = []
        
        df['static_mem'] = df['params_mb_mean'] + df['grads_mb_mean'] + df['optimizer_states_mb_mean']
        static_gpu = df[df['device'] == 'GPU']['static_mem'].sum()
        static_cpu = df[df['device'] == 'CPU']['static_mem'].sum()
        
        acc_act_gpu = 0.0
        acc_act_cpu = 0.0
        
        for i in range(len(df)):
            row = df.iloc[i]
            act = row['activations_mb_mean']
            strat = row['activation_strategy']
            dev = row['device']
            
            if dev == 'GPU':
                if strat == 'retain':
                    acc_act_gpu += act
                elif strat == 'checkpoint':
                    acc_act_cpu += act
            elif dev == 'CPU':
                acc_act_cpu += act
                
            after_vram.append(static_gpu + acc_act_gpu)
            after_dram.append(static_cpu + acc_act_cpu)
            
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Memory Distribution: {model.upper()} | {opt} | {prec} | {batch.upper()}", fontsize=16, fontweight='bold', y=0.95)
        
        # Plot Antes
        ax1.fill_between(layers, before_vram, color='#d62728', alpha=0.6, label='GPU VRAM (Naïve)')
        ax1.axhline(8192, color='red', linestyle='--', linewidth=2, label='VRAM Limit (8GB)')
        ax1.set_title("Before: Native Execution (GPU-Only) - OOM Exceeded", fontsize=13)
        ax1.set_ylabel("Memory (MB)", fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot Despues
        ax2.fill_between(layers, after_vram, color='#2ca02c', alpha=0.7, label='GPU VRAM (ILP Assigned)')
        ax2.fill_between(layers, after_dram, color='#1f77b4', alpha=0.7, label='CPU DRAM (ILP Assigned / Evacuated)')
        ax2.axhline(8192, color='red', linestyle='--', linewidth=2, label='VRAM Limit (8GB)')
        ax2.axhline(32768, color='blue', linestyle='--', linewidth=2, label='DRAM Limit (32GB)')
        ax2.set_title("After: ILP Distribution (Hybrid + Recompute)", fontsize=13)
        ax2.set_xlabel("Topological Model Layers (Index)", fontsize=11)
        ax2.set_ylabel("Memory (MB)", fontsize=11)
        ax2.legend(loc='center left')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        safe_name = f"{model}_{opt}_{prec}_{batch}.png"
        out_path = os.path.join(output_dir, safe_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated_count += 1
        print(f"Generada gráfica para: {safe_name} (Max VRAM Nativa: {max(before_vram):.1f} MB -> Max VRAM ILP: {max(after_vram):.1f} MB)")

    print(f"\nProceso finalizado. Se generaron {generated_count} gráficas en: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OOM vs ILP comparison plots for a thesis-mode result tree.")
    parser.add_argument(
        "--input_root",
        default="data/zephyr/results_thesis_mode/doctoral_full",
        help="Root directory containing model/optimizer/precision/batch_* folders",
    )
    parser.add_argument(
        "--output_dir",
        default="reports/zephyr/doctoral_full/plots/oom_vs_ilp",
        help="Directory where plots will be written",
    )
    args = parser.parse_args()
    generate_oom_comparisons(args.input_root, args.output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
