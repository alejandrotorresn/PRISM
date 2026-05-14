#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

train_backward_meta_model = importlib.import_module("ilp.backward_meta_model").train_backward_meta_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and validate backward meta-model (time+energy) from metrics_stats.csv"
    )
    parser.add_argument("--metrics_stats_csv", required=True, help="Path to metrics_stats CSV")
    parser.add_argument("--output_json", required=True, help="Where to write trained meta-model JSON")
    parser.add_argument("--validation_ratio", type=float, default=0.25)
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = train_backward_meta_model(
        metrics_stats_csv=args.metrics_stats_csv,
        output_json=args.output_json,
        validation_ratio=args.validation_ratio,
        ridge_lambda=args.ridge_lambda,
        seed=args.seed,
    )

    print("=" * 80)
    print("BACKWARD META-MODEL TRAINING")
    print("=" * 80)
    print(f"metrics_stats_csv: {args.metrics_stats_csv}")
    print(f"output_json: {args.output_json}")
    print(f"num_layers: {payload.get('num_layers')}")
    print("Validation summary:")
    validation = payload.get("validation", {})
    for target in ["gpu_time", "cpu_time", "gpu_energy", "cpu_energy"]:
        row = validation.get(target, {})
        print(
            f"  - {target}: "
            f"val_rmse={row.get('val_rmse', float('nan')):.6f}, "
            f"val_mae={row.get('val_mae', float('nan')):.6f}, "
            f"val_r2={row.get('val_r2', float('nan')):.6f}"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
