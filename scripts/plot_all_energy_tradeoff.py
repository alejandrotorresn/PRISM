import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_energy_tradeoff_plots():
    base_dir = "data/zephyr/results_thesis_mode/doctoral_full"
    output_dir = "reports/zephyr/doctoral_full/plots/energy_tradeoff"
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
            continue
            
        if len(metrics_df) != len(assign_df):
            continue
            
        metrics_df = metrics_df.sort_index()
        df = pd.concat([metrics_df.reset_index(drop=True), assign_df.reset_index(drop=True)], axis=1)
        
        # Determine if this was an intervened model (OOM saved)
        has_intervention = (df['device'] != 'GPU').any() or (df['activation_strategy'] != 'retain').any()
        
        total_static_mem = df['params_mb_mean'].sum() + df['grads_mb_mean'].sum() + df['optimizer_states_mb_mean'].sum()
        total_naive_peak = total_static_mem + df['activations_mb_mean'].sum()
        
        if not has_intervention and total_naive_peak < 7500:
            continue
            
        layers = np.arange(len(df))
        
        # 1. Antes (Ejecucion Nativa)
        before_energy_acc = []
        before_power_watts = []
        acc_energy_native = 0.0
        
        # 2. Despues (ILP Hibrido)
        after_energy_acc = []
        after_power_watts = []
        acc_energy_ilp = 0.0
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # -- NATIVE (GPU Only) --
            gpu_fwd_e = row.get('gpu_fwd_energy_j_mean', 0.0)
            gpu_bwd_e = row.get('gpu_bwd_energy_j_mean', 0.0)
            if pd.isna(gpu_fwd_e): gpu_fwd_e = 0.0
            if pd.isna(gpu_bwd_e): gpu_bwd_e = 0.0
            
            native_energy = gpu_fwd_e + gpu_bwd_e
            acc_energy_native += native_energy
            before_energy_acc.append(acc_energy_native)
            
            gpu_fwd_t = row.get('gpu_fwd_time_ms_mean', 1.0)
            gpu_bwd_t = row.get('gpu_bwd_time_ms_mean', 1.0)
            if pd.isna(gpu_fwd_t): gpu_fwd_t = 1.0
            if pd.isna(gpu_bwd_t): gpu_bwd_t = 1.0
            native_time_s = (gpu_fwd_t + gpu_bwd_t) / 1000.0
            
            # Avoid division by zero
            if native_time_s > 0.0001:
                native_power = native_energy / native_time_s
            else:
                native_power = 0.0
            before_power_watts.append(native_power)
            
            # -- ILP (Hibrido) --
            dev = row['device']
            cpu_fwd_e = row.get('cpu_fwd_energy_j_mean', 0.0)
            cpu_bwd_e = row.get('cpu_bwd_energy_j_mean', 0.0)
            if pd.isna(cpu_fwd_e): cpu_fwd_e = 0.0
            if pd.isna(cpu_bwd_e): cpu_bwd_e = 0.0
            
            cpu_fwd_t = row.get('cpu_fwd_time_ms_mean', 1.0)
            cpu_bwd_t = row.get('cpu_bwd_time_ms_mean', 1.0)
            if pd.isna(cpu_fwd_t): cpu_fwd_t = 1.0
            if pd.isna(cpu_bwd_t): cpu_bwd_t = 1.0
            
            if dev == 'GPU':
                ilp_energy = gpu_fwd_e + gpu_bwd_e
                ilp_time_s = (gpu_fwd_t + gpu_bwd_t) / 1000.0
            else:
                ilp_energy = cpu_fwd_e + cpu_bwd_e
                ilp_time_s = (cpu_fwd_t + cpu_bwd_t) / 1000.0
                
            acc_energy_ilp += ilp_energy
            after_energy_acc.append(acc_energy_ilp)
            
            if ilp_time_s > 0.0001:
                ilp_power = ilp_energy / ilp_time_s
            else:
                ilp_power = 0.0
            after_power_watts.append(ilp_power)
            
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Energy and Power Tradeoff: {model.upper()} | {opt} | {prec} | {batch.upper()}", fontsize=16, fontweight='bold', y=0.95)
        
        # Plot Energy (Cumulative)
        ax1.plot(layers, before_energy_acc, color='#d62728', linewidth=2.5, label='Native GPU (OOM Trajectory)')
        ax1.plot(layers, after_energy_acc, color='#2ca02c', linewidth=2.5, label='ILP Hybrid Plan')
        ax1.fill_between(layers, before_energy_acc, after_energy_acc, where=(np.array(after_energy_acc) > np.array(before_energy_acc)), interpolate=True, color='gray', alpha=0.2, label='Energy Penalty (Tradeoff)')
        ax1.set_title("Cumulative Energy Consumption (Joules)", fontsize=13)
        ax1.set_ylabel("Total Energy (J)", fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot Power (Watts per layer)
        ax2.plot(layers, before_power_watts, color='#d62728', alpha=0.7, label='Native GPU Power')
        ax2.plot(layers, after_power_watts, color='#2ca02c', alpha=0.7, label='ILP Hybrid Power')
        ax2.set_title("Average Power Draw per Layer (Watts)", fontsize=13)
        ax2.set_xlabel("Topological Model Layers (Index)", fontsize=11)
        ax2.set_ylabel("Power (W)", fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        safe_name = f"energy_{model}_{opt}_{prec}_{batch}.png"
        out_path = os.path.join(output_dir, safe_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated_count += 1
        print(f"Generated energy plot for: {safe_name} (Native J: {max(before_energy_acc):.2f} -> ILP J: {max(after_energy_acc):.2f})")

    print(f"\\nFinished. Generated {generated_count} plots in: {output_dir}")

if __name__ == "__main__":
    generate_energy_tradeoff_plots()
