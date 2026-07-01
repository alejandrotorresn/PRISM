#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import t as t_dist

def cohens_d_one_sample(mean_baseline, std_baseline, target_value):
    if std_baseline == 0 or pd.isna(std_baseline):
        return np.nan
    return (mean_baseline - target_value) / std_baseline

def compute_p_value_t_test(mean_baseline, std_baseline, n_samples, target_value):
    """One-sample two-tailed Student t-test.

    With the small sample sizes typical in doctoral profiling campaigns (n ≈ 5),
    the standard normal (Z) approximation inflates false positives.  The t-test
    with df = n - 1 tail-corrects for this.
    """
    if std_baseline == 0 or pd.isna(std_baseline) or n_samples < 2:
        return np.nan
    # Standard error of the mean
    sem = std_baseline / np.sqrt(n_samples)
    t_score = (mean_baseline - target_value) / sem
    df = n_samples - 1
    # Two-tailed p-value using the t distribution
    p_value = 2 * (1 - t_dist.cdf(abs(t_score), df=df))
    return p_value

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def _plot_statistical_validity(out_df: pd.DataFrame, out_path: Path):
    plot_dir = out_path.parent.parent / "plots" / "statistical_validity"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    sns.set_context("paper", font_scale=1.2)

    df_plot = out_df.dropna(subset=["cohens_d_vs_cpu"]).copy()
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_plot, x="model", y="cohens_d_vs_cpu", palette="viridis", ax=ax, errorbar=None)
        ax.set_title("Effect Size (Cohen's d) of ILP Offloading vs CPU Baseline", pad=15, fontweight="bold")
        ax.set_xlabel("Architecture Model", fontweight="bold")
        ax.set_ylabel("Cohen's d (Standardized Savings)", fontweight="bold")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        sns.despine()
        fig.tight_layout()
        fig.savefig(plot_dir / "cohens_d_effect_size_vs_cpu.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    df_p = out_df.dropna(subset=["p_value_vs_cpu"]).copy()
    if not df_p.empty:
        df_p["neg_log_p"] = -np.log10(df_p["p_value_vs_cpu"] + 1e-15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_p, x="model", y="neg_log_p", palette="magma", ax=ax, errorbar=None)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=2, label="Significance Threshold (p=0.05)")
        ax.axhline(-np.log10(0.001), color="orange", linestyle=":", linewidth=2, label="High Significance (p=0.001)")
        ax.set_title("Statistical Significance (-log10 p-value) vs CPU Baseline", pad=15, fontweight="bold")
        ax.set_xlabel("Architecture Model", fontweight="bold")
        ax.set_ylabel("-log10(p-value)", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        sns.despine()
        fig.tight_layout()
        fig.savefig(plot_dir / "p_value_significance_vs_cpu.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consolidated_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    cons_path = Path(args.consolidated_csv)
    if not cons_path.exists():
        print(f"File not found: {cons_path}")
        return 1

    df = pd.read_csv(cons_path)

    # We want to compare ILP objective vs All-GPU and All-CPU
    # The variance of the baseline is estimated via max_cv_key_metrics
    # If a config was OOM on GPU, we can't test vs GPU.

    results = []
    for _, row in df.iterrows():
        model = row.get("model", "unknown")
        budget = row.get("gpu_budget_mb", 0)
        ilp_obj = float(row.get("ilp_objective", np.nan))
        n_samples = float(row.get("n_samples", 5)) # usually 5
        cv = float(row.get("max_cv_key_metrics", 0.05)) # default to 5% if missing
        if pd.isna(cv) or cv == 0:
            cv = 0.05

        cpu_obj = float(row.get("all_cpu_objective", np.nan))
        gpu_obj = float(row.get("all_gpu_objective", np.nan))

        cpu_std = cpu_obj * cv if not pd.isna(cpu_obj) else np.nan
        gpu_std = gpu_obj * cv if not pd.isna(gpu_obj) else np.nan

        res = {
            "model": model,
            "gpu_budget_mb": budget,
            "ilp_objective": ilp_obj,
            "n_samples": n_samples,
        }

        # Test vs CPU
        if not pd.isna(cpu_obj) and not pd.isna(ilp_obj):
            res["cohens_d_vs_cpu"] = cohens_d_one_sample(cpu_obj, cpu_std, ilp_obj)
            res["p_value_vs_cpu"] = compute_p_value_t_test(cpu_obj, cpu_std, n_samples, ilp_obj)
            res["significant_vs_cpu"] = res["p_value_vs_cpu"] < 0.05
        else:
            res["cohens_d_vs_cpu"] = np.nan
            res["p_value_vs_cpu"] = np.nan
            res["significant_vs_cpu"] = False

        # Test vs GPU
        if not pd.isna(gpu_obj) and not pd.isna(ilp_obj):
            res["cohens_d_vs_gpu"] = cohens_d_one_sample(gpu_obj, gpu_std, ilp_obj)
            res["p_value_vs_gpu"] = compute_p_value_t_test(gpu_obj, gpu_std, n_samples, ilp_obj)
            res["significant_vs_gpu"] = res["p_value_vs_gpu"] < 0.05
        results.append(res)

    out_df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Statistical significance tests written to {out_path}")
    try:
        _plot_statistical_validity(out_df, out_path)
        print("Statistical validity plots generated successfully.")
    except Exception as e:
        print(f"Error generating statistical validity plots: {e}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
