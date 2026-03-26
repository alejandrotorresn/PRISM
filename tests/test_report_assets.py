from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation import generate_ilp_report_assets


def test_best_hybrid_rows_prefers_fastest_ilp_plan() -> None:
    df = pd.DataFrame(
        [
            {"model": "simple_mlp", "run_label": "all_cpu", "status": "ok", "avg_step_ms": 2.0},
            {"model": "simple_mlp", "run_label": "ilp_plan", "status": "ok", "avg_step_ms": 1.5},
            {"model": "simple_mlp", "run_label": "ilp_plan", "status": "ok", "avg_step_ms": 1.2},
            {"model": "resnet50", "run_label": "ilp_plan", "status": "ok", "avg_step_ms": 3.1},
        ]
    )

    best = generate_ilp_report_assets._best_hybrid_rows(df)
    assert list(best["model"]) == ["resnet50", "simple_mlp"]
    assert float(best.loc[best["model"] == "simple_mlp", "avg_step_ms"].iloc[0]) == 1.2


def test_markdown_summary_includes_hybrid_section(tmp_path: Path) -> None:
    full_df = pd.DataFrame(
        [
            {
                "model": "simple_mlp",
                "gpu_budget_mb": 200.0,
                "ilp_status": "optimal",
                "ilp_objective": 1.0,
                "greedy_objective": 1.2,
                "ilp_gpu_mem_mb": 0.0,
                "ilp_cpu_mem_mb": 0.1,
                "ilp_layers_gpu": 0,
                "ilp_layers_cpu": 5,
                "ilp_cut_edges": 0,
                "all_cpu_objective": 1.4,
                "all_gpu_status": "feasible",
            }
        ]
    )
    best_df = full_df.copy()
    hybrid_best_df = pd.DataFrame(
        [
            {
                "model": "simple_mlp",
                "config_optimizer": "SGD",
                "config_precision": "fp32",
                "config_batch_size": 8,
                "avg_step_ms": 0.9,
                "final_loss": 1.9,
                "quality_metric_name": "accuracy",
                "final_quality_metric": 1.0,
                "dataset_name": "mnist",
                "input_source": "dataset",
            }
        ]
    )

    out_path = tmp_path / "summary.md"
    generate_ilp_report_assets._write_markdown_summary(full_df, best_df, out_path, hybrid_best_df=hybrid_best_df)

    content = out_path.read_text()
    assert "Best Observed Hybrid Runtime Per Model" in content
    assert "accuracy" in content
    assert "mnist" in content