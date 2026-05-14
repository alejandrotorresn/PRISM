from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .data_loader import ILPInputData


@dataclass
class LinearModel:
    intercept: float
    coefficients: Dict[str, float]


META_TARGETS = {
    "gpu_time": ("gpu_bwd_time_ms_mean", "gpu_fwd_time_ms_mean"),
    "cpu_time": ("cpu_bwd_time_ms_mean", "cpu_fwd_time_ms_mean"),
    "gpu_energy": ("gpu_bwd_energy_j_mean", "gpu_fwd_energy_j_mean"),
    "cpu_energy": ("cpu_bwd_energy_j_mean", "cpu_fwd_energy_j_mean"),
}


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _split_train_val(n_rows: int, validation_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 4:
        idx = np.arange(n_rows)
        return idx, idx
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = max(1, int(round(n_rows * validation_ratio)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _fit_ridge(
    design: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    xtx = design.T @ design
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=float)
    reg[0, 0] = 0.0
    return np.linalg.pinv(xtx + reg) @ (design.T @ target)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err ** 2)) if len(err) > 0 else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err))) if len(err) > 0 else 0.0
    var = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) > 0 else 0.0
    r2 = float(1.0 - (np.sum(err ** 2) / var)) if var > 0 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_backward_meta_model(
    metrics_stats_csv: str | Path,
    output_json: str | Path,
    validation_ratio: float = 0.25,
    ridge_lambda: float = 1e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    path = Path(metrics_stats_csv)
    if not path.exists():
        raise FileNotFoundError(f"metrics_stats csv not found: {path}")
    if not 0.0 < validation_ratio < 0.9:
        raise ValueError(f"validation_ratio must be in (0, 0.9), got {validation_ratio}")
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")

    df = pd.read_csv(path).copy()
    if "layer" not in df.columns:
        raise KeyError(f"Missing required column 'layer' in {path}")

    mem_gpu = _safe_col(df, "gpu_mem_peak_mb_mean", 0.0)
    mem_cpu = _safe_col(df, "cpu_mem_mb_mean", 0.0)

    models: Dict[str, Dict[str, float]] = {}
    report: Dict[str, Dict[str, float]] = {}

    for target_name, (target_col, forward_col) in META_TARGETS.items():
        if target_col not in df.columns or forward_col not in df.columns:
            raise KeyError(
                f"Missing columns for meta-model target '{target_name}': {target_col}, {forward_col}"
            )

        x_fwd = _safe_col(df, forward_col, 0.0)
        x_fwd_sq = x_fwd ** 2
        x_mem = mem_gpu if "gpu_" in target_name else mem_cpu
        x_bias = pd.Series(1.0, index=df.index, dtype=float)

        design_df = pd.DataFrame(
            {
                "bias": x_bias,
                "fwd": x_fwd,
                "fwd_sq": x_fwd_sq,
                "mem": x_mem,
            }
        )
        y = _safe_col(df, target_col, 0.0)

        design = design_df.to_numpy(dtype=float)
        target = y.to_numpy(dtype=float)

        tr_idx, va_idx = _split_train_val(len(df), validation_ratio=validation_ratio, seed=seed)

        beta = _fit_ridge(design[tr_idx], target[tr_idx], ridge_lambda=ridge_lambda)
        pred_train = design[tr_idx] @ beta
        pred_val = design[va_idx] @ beta

        models[target_name] = {
            "intercept": float(beta[0]),
            "coef_fwd": float(beta[1]),
            "coef_fwd_sq": float(beta[2]),
            "coef_mem": float(beta[3]),
        }

        report[target_name] = {
            "train_rmse": _metrics(target[tr_idx], pred_train)["rmse"],
            "train_mae": _metrics(target[tr_idx], pred_train)["mae"],
            "train_r2": _metrics(target[tr_idx], pred_train)["r2"],
            "val_rmse": _metrics(target[va_idx], pred_val)["rmse"],
            "val_mae": _metrics(target[va_idx], pred_val)["mae"],
            "val_r2": _metrics(target[va_idx], pred_val)["r2"],
        }

    payload: Dict[str, Any] = {
        "schema": "prism.backward_meta_model.v1",
        "metrics_stats_csv": str(path),
        "num_layers": int(len(df)),
        "validation_ratio": float(validation_ratio),
        "ridge_lambda": float(ridge_lambda),
        "seed": int(seed),
        "features": ["fwd", "fwd_sq", "mem"],
        "targets": models,
        "validation": report,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return payload


def _predict_one(
    coeffs: Dict[str, float],
    fwd_value: float,
    mem_value: float,
) -> float:
    pred = (
        float(coeffs.get("intercept", 0.0))
        + float(coeffs.get("coef_fwd", 0.0)) * fwd_value
        + float(coeffs.get("coef_fwd_sq", 0.0)) * (fwd_value ** 2)
        + float(coeffs.get("coef_mem", 0.0)) * mem_value
    )
    return max(0.0, float(pred))


def _safe_blend(measured: float, predicted: float, blend: float) -> float:
    return float((1.0 - blend) * measured + (blend * predicted))


def apply_backward_meta_model(
    data: ILPInputData,
    meta_model_payload: Dict[str, Any],
    blend: float = 1.0,
) -> ILPInputData:
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")

    if str(meta_model_payload.get("schema", "")) != "prism.backward_meta_model.v1":
        raise ValueError("Unsupported meta-model schema")

    targets = meta_model_payload.get("targets", {})
    required = {"gpu_time", "cpu_time", "gpu_energy", "cpu_energy"}
    missing = sorted(required - set(targets.keys()))
    if missing:
        raise KeyError(f"Meta-model payload is missing targets: {missing}")

    out = ILPInputData(
        nodes=list(data.nodes),
        node_cost_gpu_ms=dict(data.node_cost_gpu_ms),
        node_cost_cpu_ms=dict(data.node_cost_cpu_ms),
        node_cost_gpu_fwd_ms=dict(data.node_cost_gpu_fwd_ms),
        node_cost_gpu_bwd_ms=dict(data.node_cost_gpu_bwd_ms),
        node_cost_cpu_fwd_ms=dict(data.node_cost_cpu_fwd_ms),
        node_cost_cpu_bwd_ms=dict(data.node_cost_cpu_bwd_ms),
        node_energy_gpu_j=dict(data.node_energy_gpu_j),
        node_energy_cpu_j=dict(data.node_energy_cpu_j),
        node_energy_gpu_fwd_j=dict(data.node_energy_gpu_fwd_j),
        node_energy_gpu_bwd_j=dict(data.node_energy_gpu_bwd_j),
        node_energy_cpu_fwd_j=dict(data.node_energy_cpu_fwd_j),
        node_energy_cpu_bwd_j=dict(data.node_energy_cpu_bwd_j),
        node_mem_gpu_mb=dict(data.node_mem_gpu_mb),
        node_mem_cpu_mb=dict(data.node_mem_cpu_mb),
        edges=list(data.edges),
        edge_transfer_ms=dict(data.edge_transfer_ms),
        node_mem_activation_mb=dict(data.node_mem_activation_mb),
        node_time_io_ms=dict(data.node_time_io_ms),
        node_energy_io_j=dict(data.node_energy_io_j),
        activation_metadata_source=data.activation_metadata_source,
        io_metadata_source=data.io_metadata_source,
        graph_trace_source=data.graph_trace_source,
    )

    for node in out.nodes:
        mem_gpu = float(out.node_mem_gpu_mb.get(node, 0.0))
        mem_cpu = float(out.node_mem_cpu_mb.get(node, 0.0))

        gpu_bwd_time_pred = _predict_one(targets["gpu_time"], float(out.node_cost_gpu_fwd_ms.get(node, 0.0)), mem_gpu)
        cpu_bwd_time_pred = _predict_one(targets["cpu_time"], float(out.node_cost_cpu_fwd_ms.get(node, 0.0)), mem_cpu)
        gpu_bwd_energy_pred = _predict_one(targets["gpu_energy"], float(out.node_energy_gpu_fwd_j.get(node, 0.0)), mem_gpu)
        cpu_bwd_energy_pred = _predict_one(targets["cpu_energy"], float(out.node_energy_cpu_fwd_j.get(node, 0.0)), mem_cpu)

        out.node_cost_gpu_bwd_ms[node] = _safe_blend(float(out.node_cost_gpu_bwd_ms.get(node, 0.0)), gpu_bwd_time_pred, blend)
        out.node_cost_cpu_bwd_ms[node] = _safe_blend(float(out.node_cost_cpu_bwd_ms.get(node, 0.0)), cpu_bwd_time_pred, blend)
        out.node_energy_gpu_bwd_j[node] = _safe_blend(float(out.node_energy_gpu_bwd_j.get(node, 0.0)), gpu_bwd_energy_pred, blend)
        out.node_energy_cpu_bwd_j[node] = _safe_blend(float(out.node_energy_cpu_bwd_j.get(node, 0.0)), cpu_bwd_energy_pred, blend)

        out.node_cost_gpu_ms[node] = float(out.node_cost_gpu_fwd_ms.get(node, 0.0) + out.node_cost_gpu_bwd_ms.get(node, 0.0))
        out.node_cost_cpu_ms[node] = float(out.node_cost_cpu_fwd_ms.get(node, 0.0) + out.node_cost_cpu_bwd_ms.get(node, 0.0))
        out.node_energy_gpu_j[node] = float(out.node_energy_gpu_fwd_j.get(node, 0.0) + out.node_energy_gpu_bwd_j.get(node, 0.0))
        out.node_energy_cpu_j[node] = float(out.node_energy_cpu_fwd_j.get(node, 0.0) + out.node_energy_cpu_bwd_j.get(node, 0.0))

    return out


def load_backward_meta_model(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Backward meta-model json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
