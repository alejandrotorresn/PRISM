from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "gpu_fwd_time_ms",
    "gpu_bwd_time_ms",
    "cpu_fwd_time_ms",
    "cpu_bwd_time_ms",
    "gpu_fwd_energy_j",
    "gpu_bwd_energy_j",
    "cpu_fwd_energy_j",
    "cpu_bwd_energy_j",
    "params_mb",
    "grads_mb",
    "optimizer_states_mb",
    "activations_mb",
    "gpu_mem_peak_mb",
    "cpu_mem_mb",
    "transfer_h2d_ms",
    "transfer_d2h_ms",
    "transfer_edge_aware_total_ms",
    "dispatch_overhead_ratio",
    "tflops",
    "efficiency_ratio",
    "opt_step_time_ms",
]


GROUP_COLUMNS = [
    "model",
    "batch_size",
    "precision_requested",
    "optimizer",
    "layer",
    "type",
    "cpu_precision_executed",
    "gpu_precision_executed",
]


def _discover_metrics_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in input_dir.rglob("*_metrics.csv"):
        name = path.name
        if name.endswith("_metrics_gpu_partial.csv"):
            continue
        if name.endswith("_metrics_stats.csv"):
            continue
        files.append(path)
    return sorted(files)


def _load_frames(paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _stat_series(group: pd.DataFrame, col: str) -> Dict[str, float]:
    s = pd.to_numeric(group[col], errors="coerce").dropna()
    if s.empty:
        return {
            f"{col}_mean": float("nan"),
            f"{col}_std": float("nan"),
            f"{col}_p50": float("nan"),
            f"{col}_p90": float("nan"),
            f"{col}_p95": float("nan"),
        }
    return {
        f"{col}_mean": float(s.mean()),
        f"{col}_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        f"{col}_p50": float(s.quantile(0.50)),
        f"{col}_p90": float(s.quantile(0.90)),
        f"{col}_p95": float(s.quantile(0.95)),
    }


def aggregate_metrics_stats(
    input_dir: str,
    output_csv: Optional[str] = None,
    include_skipped: bool = False,
) -> Dict[str, object]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    metric_files = _discover_metrics_files(base)
    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.csv files found under: {input_dir}")

    df = _load_frames(metric_files)
    if df.empty:
        raise ValueError(f"No data rows found in metrics files under: {input_dir}")

    if not include_skipped:
        if "run_executed" in df.columns:
            df = df[df["run_executed"] == True]  # noqa: E712
        if "layer" in df.columns:
            df = df[df["layer"] != "__profiling_skipped__"]

    if df.empty:
        raise ValueError("All rows were filtered out; no executable samples remain for aggregation")

    missing_group_cols = [c for c in GROUP_COLUMNS if c not in df.columns]
    if missing_group_cols:
        raise KeyError(f"Missing required grouping columns in metrics data: {missing_group_cols}")

    metric_cols = [c for c in DEFAULT_METRIC_COLUMNS if c in df.columns]
    if not metric_cols:
        raise KeyError("No expected metric columns were found for aggregation")

    rows: List[Dict[str, object]] = []
    grouped = df.groupby(GROUP_COLUMNS, dropna=False)

    for keys, g in grouped:
        row: Dict[str, object] = {GROUP_COLUMNS[i]: keys[i] for i in range(len(GROUP_COLUMNS))}
        row["n_samples"] = int(len(g))
        row["n_runs"] = int(g["run_id"].nunique()) if "run_id" in g.columns else int(len(g))

        for col in metric_cols:
            row.update(_stat_series(g, col))

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(
        by=["model", "batch_size", "precision_requested", "optimizer", "layer"],
        kind="stable",
    )

    if output_csv is None:
        output_csv = str(base / "metrics_stats.csv")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return {
        "output_csv": str(out_path),
        "input_files": len(metric_files),
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "metric_columns": metric_cols,
    }
