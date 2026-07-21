#!/usr/bin/env python3
"""Generate Chapter 6 comparison figures for PRISM vs All-GPU.

This script builds infrastructure-separated summaries and figures from:
- reports/<infra>/doctoral_full/csv/hybrid_execution_consolidated.csv
- reports/<infra>/doctoral_full/csv/ilp_best_per_model.csv

Outputs:
- final_thesis/figures/chapter6/prism_vs_allgpu_feasible_by_model.png
- final_thesis/figures/chapter6/prism_vs_allgpu_feasible_distribution.png
- reports/chapter6/ch6_prism_vs_allgpu_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "final_thesis" / "figures" / "chapter6"
OUT_DIR = ROOT / "reports" / "chapter6"

INFRA_LABELS: Dict[str, str] = {
    "chuc-4": "CHUC-4 (A100-SXM4)",
    "kinovis-2": "KINOVIS-2 (L40S)",
}

MODELS_MAIN: List[str] = ["distilgpt2", "gpt2_small", "resnet50", "resnet152", "vit_b16"]
MODEL_ORDER: List[str] = ["resnet50", "resnet152", "distilgpt2", "gpt2_small", "vit_b16"]
MODEL_TITLES: Dict[str, str] = {
    "distilgpt2": "DistilGPT2",
    "gpt2_small": "GPT2-Small",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "vit_b16": "ViT-B16",
}

COLORS: Dict[str, str] = {
    "chuc-4": "#1f77b4",
    "kinovis-2": "#d62728",
}


def _load_joined_dataset(infra: str) -> pd.DataFrame:
    """Join runtime rows with feasibility labels from ILP consolidated results."""
    hybrid_path = ROOT / "reports" / infra / "doctoral_full" / "csv" / "hybrid_execution_consolidated.csv"
    ilp_path = ROOT / "reports" / infra / "doctoral_full" / "csv" / "ilp_best_per_model.csv"

    hy = pd.read_csv(hybrid_path)
    hy = hy[hy["config_model"].isin(MODELS_MAIN)].copy()

    keys = ["config_model", "config_optimizer", "config_precision", "config_batch_size"]
    piv = (
        hy.pivot_table(index=keys, columns="run_label", values="avg_step_ms", aggfunc="mean")
        .reset_index()
        .rename(
            columns={
                "config_model": "model",
                "config_optimizer": "optimizer",
                "config_precision": "precision",
                "config_batch_size": "batch_size",
            }
        )
    )

    ilp = pd.read_csv(ilp_path)
    ilp = ilp[ilp["model"].isin(MODELS_MAIN)][
        [
            "model",
            "optimizer",
            "precision",
            "batch_size",
            "gpu_budget_mb",
            "all_gpu_status",
            "ilp_objective",
            "all_gpu_objective",
            "greedy_objective",
            "all_cpu_objective",
        ]
    ]

    merged = piv.merge(ilp, on=["model", "optimizer", "precision", "batch_size"], how="inner")
    merged["infra"] = infra
    merged["prism_vs_allgpu_rt_pct"] = (merged["all_gpu"] - merged["ilp_plan"]) / merged["all_gpu"] * 100.0
    merged["prism_vs_cpu_rt_pct"] = (merged["all_cpu"] - merged["ilp_plan"]) / merged["all_cpu"] * 100.0
    merged["prism_vs_greedy_obj_pct"] = (
        (merged["greedy_objective"] - merged["ilp_objective"]) / merged["greedy_objective"] * 100.0
    )
    return merged


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (infra, precision), d in df.groupby(["infra", "precision"], sort=True):
        feasible = d[d["all_gpu_status"] == "feasible"].copy()
        infeasible = d[d["all_gpu_status"] == "infeasible"].copy()

        if feasible.empty:
            continue

        best = feasible.loc[feasible["prism_vs_allgpu_rt_pct"].idxmax()]
        worst = feasible.loc[feasible["prism_vs_allgpu_rt_pct"].idxmin()]

        rows.append(
            {
                "scope": "global",
                "infra": infra,
                "precision": precision,
                "model": "ALL",
                "n_feasible": len(feasible),
                "n_infeasible": len(infeasible),
                "median_prism_vs_allgpu_rt_pct": feasible["prism_vs_allgpu_rt_pct"].median(),
                "p10_prism_vs_allgpu_rt_pct": feasible["prism_vs_allgpu_rt_pct"].quantile(0.1),
                "p90_prism_vs_allgpu_rt_pct": feasible["prism_vs_allgpu_rt_pct"].quantile(0.9),
                "median_prism_vs_greedy_obj_pct_in_oom": infeasible["prism_vs_greedy_obj_pct"].median()
                if not infeasible.empty
                else np.nan,
                "best_model": best["model"],
                "best_batch_size": int(best["batch_size"]),
                "best_optimizer": best["optimizer"],
                "best_prism_vs_allgpu_rt_pct": best["prism_vs_allgpu_rt_pct"],
                "best_prism_step_ms": best["ilp_plan"],
                "best_allgpu_step_ms": best["all_gpu"],
                "worst_model": worst["model"],
                "worst_batch_size": int(worst["batch_size"]),
                "worst_optimizer": worst["optimizer"],
                "worst_prism_vs_allgpu_rt_pct": worst["prism_vs_allgpu_rt_pct"],
                "worst_prism_step_ms": worst["ilp_plan"],
                "worst_allgpu_step_ms": worst["all_gpu"],
            }
        )

        for model, dm in feasible.groupby("model", sort=False):
            rows.append(
                {
                    "scope": "model",
                    "infra": infra,
                    "precision": precision,
                    "model": model,
                    "n_feasible": len(dm),
                    "n_infeasible": int((d["model"] == model).sum() - len(dm)),
                    "median_prism_vs_allgpu_rt_pct": dm["prism_vs_allgpu_rt_pct"].median(),
                    "p10_prism_vs_allgpu_rt_pct": dm["prism_vs_allgpu_rt_pct"].quantile(0.1),
                    "p90_prism_vs_allgpu_rt_pct": dm["prism_vs_allgpu_rt_pct"].quantile(0.9),
                    "median_prism_vs_greedy_obj_pct_in_oom": np.nan,
                    "best_model": "",
                    "best_batch_size": np.nan,
                    "best_optimizer": "",
                    "best_prism_vs_allgpu_rt_pct": np.nan,
                    "best_prism_step_ms": np.nan,
                    "best_allgpu_step_ms": np.nan,
                    "worst_model": "",
                    "worst_batch_size": np.nan,
                    "worst_optimizer": "",
                    "worst_prism_vs_allgpu_rt_pct": np.nan,
                    "worst_prism_step_ms": np.nan,
                    "worst_allgpu_step_ms": np.nan,
                }
            )

    return pd.DataFrame(rows)


def _plot_median_by_model(df: pd.DataFrame) -> None:
    plot_df = df[(df["all_gpu_status"] == "feasible") & (df["precision"] == "fp32")].copy()
    kinovis_bf16 = df[(df["infra"] == "kinovis-2") & (df["precision"] == "bf16") & (df["all_gpu_status"] == "feasible")]

    med = (
        plot_df.groupby(["infra", "model"]) ["prism_vs_allgpu_rt_pct"]
        .median()
        .unstack(0)
        .reindex(MODEL_ORDER)
    )

    if not kinovis_bf16.empty:
        med["kinovis-2-bf16"] = (
            kinovis_bf16.groupby("model")["prism_vs_allgpu_rt_pct"].median().reindex(MODEL_ORDER)
        )

    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.24 if "kinovis-2-bf16" in med.columns else 0.32

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200)

    offsets = []
    cols = []
    labels = []
    if "chuc-4" in med.columns:
        offsets.append(-width)
        cols.append(COLORS["chuc-4"])
        labels.append("CHUC-4 FP32")
    if "kinovis-2" in med.columns:
        offsets.append(0.0)
        cols.append(COLORS["kinovis-2"])
        labels.append("KINOVIS-2 FP32")
    if "kinovis-2-bf16" in med.columns:
        offsets.append(width)
        cols.append("#2ca02c")
        labels.append("KINOVIS-2 BF16")

    for off, col, label, colname in zip(offsets, cols, labels, med.columns):
        vals = med[colname].values
        bars = ax.bar(x + off, vals, width=width, label=label, color=col, alpha=0.92)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.6,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(0.0, color="#444444", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_TITLES[m] for m in MODEL_ORDER], rotation=0)
    ax.set_ylabel("PRISM improvement over All-GPU (%)")
    ax.set_title("Feasible regime: median by model and infrastructure")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "prism_vs_allgpu_feasible_by_model.png", bbox_inches="tight")
    plt.close(fig)


def _plot_distribution_with_extremes(df: pd.DataFrame) -> None:
    d = df[(df["all_gpu_status"] == "feasible") & (df["precision"] == "fp32")].copy()
    d["model_label"] = d["model"].map(MODEL_TITLES)

    # Rank within each infrastructure for stable x-positioning
    d = d.sort_values(["infra", "model", "batch_size", "optimizer"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=200, sharey=True)

    for ax, infra in zip(axes, ["chuc-4", "kinovis-2"]):
        di = d[d["infra"] == infra].copy()
        if di.empty:
            continue

        # jittered categorical x positions
        model_to_x = {m: i for i, m in enumerate(MODEL_ORDER)}
        rng = np.random.default_rng(7)
        x = np.array([model_to_x[m] for m in di["model"]], dtype=float)
        jitter = rng.normal(0, 0.06, size=len(di))

        ax.scatter(
            x + jitter,
            di["prism_vs_allgpu_rt_pct"],
            s=24,
            alpha=0.65,
            color=COLORS[infra],
            edgecolor="none",
        )

        best_idx = di["prism_vs_allgpu_rt_pct"].idxmax()
        worst_idx = di["prism_vs_allgpu_rt_pct"].idxmin()
        best = di.loc[best_idx]
        worst = di.loc[worst_idx]

        for row, marker, color, txt_prefix in [
            (best, "*", "#0a7f2e", "Best"),
            (worst, "X", "#8b0000", "Worst"),
        ]:
            xi = model_to_x[row["model"]]
            yi = row["prism_vs_allgpu_rt_pct"]
            ax.scatter([xi], [yi], s=170, marker=marker, color=color, zorder=5)
            ax.annotate(
                f"{txt_prefix}: {MODEL_TITLES[row['model']]} B={int(row['batch_size'])}\n{yi:.1f}%",
                xy=(xi, yi),
                xytext=(xi + 0.22, yi + (6 if yi < 25 else -12)),
                fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": color},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
            )

        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_TITLES[m] for m in MODEL_ORDER], rotation=25, ha="right")
        ax.set_title(INFRA_LABELS[infra] + " (FP32)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("PRISM improvement over All-GPU (%)")
    fig.suptitle("Per-configuration distribution in the feasible regime", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prism_vs_allgpu_feasible_distribution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = [_load_joined_dataset(infra) for infra in INFRA_LABELS]
    data = pd.concat(frames, ignore_index=True)

    _plot_median_by_model(data)
    _plot_distribution_with_extremes(data)

    summary = _build_summary(data)
    summary.to_csv(OUT_DIR / "ch6_prism_vs_allgpu_summary.csv", index=False)

    print("Generated:")
    print("-", FIG_DIR / "prism_vs_allgpu_feasible_by_model.png")
    print("-", FIG_DIR / "prism_vs_allgpu_feasible_distribution.png")
    print("-", OUT_DIR / "ch6_prism_vs_allgpu_summary.csv")


if __name__ == "__main__":
    main()
