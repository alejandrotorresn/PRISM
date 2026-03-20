from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of independent runs recommended for reliable coefficient estimation.
MIN_RECOMMENDED_RUNS: int = 3
# Coefficient-of-variation threshold above which timing estimates are flagged as
# high-dispersion (std/|mean| > 30 % is considered unreliable for ILP regression).
HIGH_CV_THRESHOLD: float = 0.30

# Key timing columns used to derive the quality flag; energy columns are excluded
# because they may legitimately be absent in CPU-only or GPU-only profiles.
KEY_TIME_COLUMNS: List[str] = [
    "gpu_fwd_time_ms",
    "gpu_bwd_time_ms",
    "cpu_fwd_time_ms",
    "cpu_bwd_time_ms",
]


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


def _coefficient_of_variation(group: pd.DataFrame, col: str) -> Optional[float]:
    """Return std/|mean| for *col* in *group*, or None when not computable."""
    if col not in group.columns:
        return None
    s = pd.to_numeric(group[col], errors="coerce").dropna()
    if len(s) < 2:
        return None
    mean = s.mean()
    if mean == 0.0:
        return None
    return float(s.std(ddof=1) / abs(mean))


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
        n_runs = int(g["run_id"].nunique()) if "run_id" in g.columns else int(len(g))
        row["n_runs"] = n_runs

        # --- Coefficient of variation for key time metrics --------------------
        cv_values: List[float] = []
        for kcol in KEY_TIME_COLUMNS:
            cv = _coefficient_of_variation(g, kcol)
            row[f"{kcol}_cv"] = round(cv, 4) if cv is not None else float("nan")
            if cv is not None:
                cv_values.append(cv)

        max_cv = max(cv_values) if cv_values else float("nan")
        row["max_cv_key_metrics"] = round(max_cv, 4) if cv_values else float("nan")

        # --- Quality flag -----------------------------------------------------
        if n_runs < MIN_RECOMMENDED_RUNS:
            row["quality_flag"] = "low_sample"
        elif cv_values and max_cv > HIGH_CV_THRESHOLD:
            row["quality_flag"] = "high_dispersion"
        else:
            row["quality_flag"] = "ok"

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

    flagged_rows = [r for r in rows if r.get("quality_flag") != "ok"]
    if flagged_rows:
        layer_names = [str(r.get("layer", r)) for r in flagged_rows]
        msg = (
            f"Sample quality issues detected in {len(flagged_rows)} layer(s): "
            f"{', '.join(layer_names)}. "
            f"Check 'quality_flag' column in output CSV for details."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    low_sample = [r.get("layer") for r in rows if r.get("quality_flag") == "low_sample"]
    high_disp  = [r.get("layer") for r in rows if r.get("quality_flag") == "high_dispersion"]

    return {
        "output_csv": str(out_path),
        "input_files": len(metric_files),
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "metric_columns": metric_cols,
        "flagged_layers_count": len(flagged_rows),
        "low_sample_layers": low_sample,
        "high_dispersion_layers": high_disp,
    }
