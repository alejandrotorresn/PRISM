#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

runtime_plan = importlib.import_module("runtime.plan_representation")
runtime_device = importlib.import_module("runtime.device_plan")
runtime_exec = importlib.import_module("runtime.hybrid_executor")

load_execution_plan = runtime_plan.load_execution_plan
DevicePlan = runtime_device.DevicePlan
plan_requests_gpu = runtime_device.plan_requests_gpu
run_hybrid_training = runtime_exec.run_hybrid_training


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: tuple[int, int] = (512, 256), output_dim: int = 10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP hybrid execution guided by ILP plan")
    parser.add_argument("--config_dir", required=True, help="Directory containing ilp_solution folder")
    parser.add_argument("--assignment_csv", default=None)
    parser.add_argument("--cut_edges_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--strict_plan", action="store_true")
    parser.add_argument("--rapl", action="store_true", help="Enable RAPL CPU energy monitor when applicable")
    parser.add_argument("--energy_sample_interval", type=float, default=0.05)
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="Allow executing GPU-assigned layers on CPU when CUDA is unavailable",
    )
    parser.add_argument("--output_dir", default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir does not exist: {config_dir}")

    solution_dir = config_dir / "ilp_solution"
    assignment_csv = Path(args.assignment_csv) if args.assignment_csv else (solution_dir / "ilp_assignment.csv")
    cut_edges_csv = Path(args.cut_edges_csv) if args.cut_edges_csv else (solution_dir / "ilp_cut_edges.csv")

    plan = load_execution_plan(assignment_csv=assignment_csv, cut_edges_csv=cut_edges_csv)
    device_plan = DevicePlan.from_execution_plan(plan)

    if plan_requests_gpu(device_plan) and not torch.cuda.is_available() and not args.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA is not available but the ILP plan assigns one or more layers to GPU. "
            "Aborting to avoid invalid phase-3 runtime evidence. "
            "If your environment uses modules, ensure GPU/driver visibility (e.g., module load devtools/cuda-13.0) "
            "and retry. If you intentionally want CPU fallback for diagnostics, pass --allow_cpu_fallback."
        )

    model = SimpleMLP().to("cpu")
    inp = torch.randn((args.batch_size, 784), dtype=torch.float32)

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=device_plan,
        steps=args.steps,
        lr=args.lr,
        gpu_id=args.gpu_id,
        strict_plan=args.strict_plan,
        enable_rapl=args.rapl,
        energy_sample_interval=args.energy_sample_interval,
    )

    out_dir = Path(args.output_dir) if args.output_dir else (solution_dir / "hybrid_execution")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "hybrid_execution_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                **result.to_dict(),
                "inputs": {
                    "config_dir": str(config_dir),
                    "assignment_csv": str(assignment_csv),
                    "cut_edges_csv": str(cut_edges_csv),
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    "lr": args.lr,
                    "seed": args.seed,
                    "gpu_id": args.gpu_id,
                    "strict_plan": args.strict_plan,
                    "rapl": args.rapl,
                    "energy_sample_interval": args.energy_sample_interval,
                },
            },
            f,
            indent=4,
        )

    df = pd.DataFrame([s.__dict__ for s in result.per_step])
    traces_csv = out_dir / "hybrid_execution_steps.csv"
    df.to_csv(traces_csv, index=False)

    print("=" * 80)
    print("HYBRID EXECUTION MVP")
    print("=" * 80)
    print(f"Status: {result.status}")
    print(f"Steps: {result.steps}")
    print(f"Avg step (ms): {result.avg_step_ms:.6f}")
    print(f"Avg power (W): {result.avg_power_w:.6f}")
    print(f"Total energy (J): {result.total_energy_j:.6f}")
    print(f"Energy source: {result.energy_source}")
    print(f"Transfer events: {result.total_transfer_events}")
    print(f"Transfer total (MB): {result.total_transfer_mb:.6f}")
    print(f"Peak GPU mem (MB): {result.peak_gpu_mem_mb:.6f}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print(f"Summary JSON: {summary_path}")
    print(f"Step traces CSV: {traces_csv}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
