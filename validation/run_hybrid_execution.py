#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

runtime_plan = importlib.import_module("runtime.plan_representation")
runtime_device = importlib.import_module("runtime.device_plan")
runtime_exec = importlib.import_module("runtime.hybrid_executor")
model_factory = importlib.import_module("models.factory")

load_execution_plan = runtime_plan.load_execution_plan
DevicePlan = runtime_device.DevicePlan
collect_leaf_module_names = runtime_device.collect_leaf_module_names
plan_requests_gpu = runtime_device.plan_requests_gpu
run_hybrid_training = runtime_exec.run_hybrid_training
HybridExecutionResult = runtime_exec.HybridExecutionResult
HybridExecutionUnsupportedError = runtime_exec.HybridExecutionUnsupportedError
build_model_and_input = model_factory.build_model_and_input
build_model_input_target = model_factory.build_model_input_target

from core.loss_utils import build_deterministic_classification_targets, extract_classification_logits
from core.decoder_export_backend import try_export_decoder_only_trace

SUPPORTED_MODELS = ["resnet50", "resnet152", "vit_b16", "bert_base", "gpt2_small", "distilgpt2", "simple_mlp"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid execution guided by ILP plan")
    parser.add_argument("--config_dir", required=True, help="Directory containing ilp_solution folder")
    parser.add_argument("--assignment_csv", default=None)
    parser.add_argument("--cut_edges_csv", default=None)

    parser.add_argument("--model", default="simple_mlp", choices=SUPPORTED_MODELS)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--datasets_root", default="datasets")
    parser.add_argument("--require_datasets", action="store_true")
    parser.add_argument("--allow_deterministic_target_fallback", action="store_true", help="Allow derived surrogate targets for diagnostic-only quality metrics")

    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--strict_plan", action="store_true")
    parser.add_argument("--enable_async_transfer", action="store_true")
    parser.add_argument("--enable_prefetch", action="store_true")
    parser.add_argument("--execution_mode", default="auto", choices=["auto", "linear", "dag"])
    parser.add_argument("--rapl", action="store_true", help="Enable RAPL CPU energy monitor when applicable")
    parser.add_argument("--energy_sample_interval", type=float, default=0.05)
    parser.add_argument("--compare_baselines", action="store_true", help="Run all_cpu/all_gpu/ilp comparative protocol")
    parser.add_argument("--plan_selection_mode", default="default_ilp")
    parser.add_argument("--plan_source", default=None)
    parser.add_argument("--plan_source_csv", default=None)
    parser.add_argument("--plan_gpu_budget_mb", type=float, default=None)
    parser.add_argument("--plan_objective", type=float, default=None)
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="Allow executing GPU-assigned layers on CPU when CUDA is unavailable",
    )
    parser.add_argument("--output_dir", default=None)
    return parser


def _torch_dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _config_identity(config_dir: Path) -> dict[str, Any]:
    parts = config_dir.parts
    identity = {
        "config_dir": str(config_dir),
        "config_model": None,
        "config_optimizer": None,
        "config_precision": None,
        "config_batch_label": None,
        "config_batch_size": None,
    }
    if len(parts) >= 4:
        identity.update(
            {
                "config_model": parts[-4],
                "config_optimizer": parts[-3],
                "config_precision": parts[-2],
                "config_batch_label": parts[-1],
            }
        )
        if parts[-1].startswith("batch_"):
            try:
                identity["config_batch_size"] = int(parts[-1].split("_", 1)[1])
            except ValueError:
                identity["config_batch_size"] = None
    return identity


def _build_model_input(args: argparse.Namespace) -> tuple[nn.Module, Any, torch.Tensor | None, str | None, dict[str, Any]]:
    mf_args = SimpleNamespace(
        model=args.model,
        precision=args.precision,
        batch_size=args.batch_size,
        input_size=args.input_size,
        seq_length=args.seq_length,
        datasets_root=args.datasets_root,
        require_datasets=args.require_datasets,
    )
    model, inp, target, data_info = build_model_input_target(mf_args, _torch_dtype_from_precision(args.precision))

    metric_name: str | None = None
    model.eval()
    with torch.no_grad():
        if isinstance(inp, dict):
            sample_out = model(**inp)
        else:
            sample_out = model(inp)
    logits = extract_classification_logits(sample_out)
    if logits is not None and target is None:
        if args.allow_deterministic_target_fallback:
            target = build_deterministic_classification_targets(inp, num_classes=int(logits.shape[1]))
            if target is not None:
                data_info = dict(data_info)
                data_info["target_source"] = "deterministic_fallback"
        else:
            data_info = dict(data_info)
            data_info["target_source"] = "missing"
    if logits is not None and target is not None:
        metric_name = "accuracy"
    model.train()
    return model, inp, target, metric_name, data_info


def _can_execute_with_layer_chain(model: nn.Module, input_data: Any) -> tuple[bool, str]:
    # Runtime currently executes a linear leaf-layer chain. Keep a fast explicit
    # compatibility check before attempting full hybrid execution.
    x = input_data
    leaves = list(model.named_modules())
    leaf_modules = [(n, m) for n, m in leaves if n and len(list(m.children())) == 0]
    if not leaf_modules:
        return False, "model has no leaf modules"

    try:
        for _, layer in leaf_modules:
            x = layer(x)
        return True, "ok"
    except Exception as exc:
        return False, f"leaf-chain execution failed: {type(exc).__name__}: {exc}"


def _make_uniform_plan(reference_plan: DevicePlan, device_label: str) -> DevicePlan:
    label = device_label.upper()
    all_layers = set(reference_plan.assignment_forward.keys()) | set(reference_plan.assignment_backward.keys())
    assignment = {layer: label for layer in sorted(all_layers)}
    cut_edges = []
    cross_phase = []
    return DevicePlan(
        assignment_forward=dict(assignment),
        assignment_backward=dict(assignment),
        cut_edges_forward=cut_edges,
        cut_edges_backward=cut_edges,
        cross_phase_edges=cross_phase,
        activation_strategies={layer: "retain" for layer in assignment},
    )


def _run_single(
    args: argparse.Namespace,
    plan: DevicePlan,
    label: str,
) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, inp, target, task_metric_name, data_info = _build_model_input(args)
    model.to("cpu")

    if args.execution_mode == "linear":
        compatible, reason = _can_execute_with_layer_chain(model, inp)
        if not compatible:
            raise RuntimeError(
                f"Model '{args.model}' is not compatible with linear leaf-chain runtime ({reason}). "
                "Use execution_mode=dag or execution_mode=auto."
            )

    warnings = []
    warnings = []
    export_trace_ctx = try_export_decoder_only_trace(model, inp)
    leaf_names = set(export_trace_ctx.node_layer_names.values()) if export_trace_ctx is not None else set(collect_leaf_module_names(model))
    plan_layers = set(plan.assignment_forward.keys()) | set(plan.assignment_backward.keys())
    missing_layers = sorted(leaf_names - plan_layers)
    if missing_layers:
        warnings.append(
            f"{len(missing_layers)} runtime leaf layer(s) are not in plan and will default to CPU: {missing_layers[:5]}"
        )

    if plan_requests_gpu(plan) and not torch.cuda.is_available() and not args.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA is not available but selected plan requests GPU layers. "
            "Use --allow_cpu_fallback for diagnostics."
        )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        target_data=target,
        plan=plan,
        activation_strategies=plan.activation_strategies,
        steps=args.steps,
        lr=args.lr,
        gpu_id=args.gpu_id,
        strict_plan=args.strict_plan,
        enable_rapl=args.rapl,
        energy_sample_interval=args.energy_sample_interval,
        enable_async_transfer=args.enable_async_transfer,
        enable_prefetch=args.enable_prefetch,
        execution_mode=args.execution_mode,
    )

    return {
        "label": label,
        "result": result,
        "task_metric_name": task_metric_name,
        "extra_warnings": warnings,
        "data_info": data_info,
    }


def _unsupported_result(message: str) -> HybridExecutionResult:
    return HybridExecutionResult(
        status="unsupported",
        steps=0,
        avg_step_ms=0.0,
        avg_power_w=0.0,
        total_energy_j=0.0,
        energy_source="unavailable",
        total_transfer_mb=0.0,
        total_transfer_events=0,
        total_prefetch_mb=0.0,
        total_prefetch_events=0,
        peak_gpu_mem_mb=0.0,
        initial_loss=0.0,
        final_loss=0.0,
        min_loss=0.0,
        loss_delta=0.0,
        prefetch_layers=[],
        recompute_layers=[],
        checkpoint_layers=[],
        backward_relocation_layers=[],
        unsupported_checkpoint_layers=[],
        warnings=[message],
        per_step=[],
    )


def _run_single_safe(
    args: argparse.Namespace,
    plan: DevicePlan,
    label: str,
) -> Dict[str, Any]:
    try:
        return _run_single(args, plan, label)
    except HybridExecutionUnsupportedError as exc:
        return {
            "label": label,
            "result": _unsupported_result(str(exc)),
            "task_metric_name": None,
            "extra_warnings": [f"Hybrid execution unsupported for '{label}': {exc}"],
            "data_info": {},
        }


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

    out_dir = Path(args.output_dir) if args.output_dir else (solution_dir / "hybrid_execution")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, Dict[str, Any]] = {}

    if args.compare_baselines:
        runs["all_cpu"] = _run_single_safe(args, _make_uniform_plan(device_plan, "CPU"), "all_cpu")
        runs["all_gpu"] = _run_single_safe(args, _make_uniform_plan(device_plan, "GPU"), "all_gpu")
    runs["ilp_plan"] = _run_single_safe(args, device_plan, "ilp_plan")

    summary_payload: Dict[str, Any] = {
        "inputs": {
            "config_dir": str(config_dir),
            "assignment_csv": str(assignment_csv),
            "cut_edges_csv": str(cut_edges_csv),
            "model": args.model,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "input_size": args.input_size,
            "seq_length": args.seq_length,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "gpu_id": args.gpu_id,
            "strict_plan": args.strict_plan,
            "rapl": args.rapl,
            "energy_sample_interval": args.energy_sample_interval,
            "enable_async_transfer": args.enable_async_transfer,
            "enable_prefetch": args.enable_prefetch,
            "execution_mode": args.execution_mode,
            "compare_baselines": args.compare_baselines,
            "datasets_root": args.datasets_root,
            "plan_selection_mode": args.plan_selection_mode,
            "plan_source": args.plan_source,
            "plan_source_csv": args.plan_source_csv,
            "plan_gpu_budget_mb": args.plan_gpu_budget_mb,
            "plan_objective": args.plan_objective,
        },
        "runs": {},
    }
    summary_payload["inputs"].update(_config_identity(config_dir))

    rows = []
    for key, payload in runs.items():
        result = payload["result"]
        task_metric_name = payload.get("task_metric_name")
        extra_warnings = payload["extra_warnings"]
        data_info = payload.get("data_info", {})
        run_dict = result.to_dict()
        if extra_warnings:
            run_dict["warnings"] = list(run_dict.get("warnings", [])) + extra_warnings
        run_dict["data_info"] = data_info
        summary_payload["runs"][key] = run_dict

        rows.append(
            {
                **_config_identity(config_dir),
                "run_label": key,
                "model": args.model,
                "precision": args.precision,
                "runtime_batch_size": args.batch_size,
                "status": result.status,
                "steps": result.steps,
                "avg_step_ms": result.avg_step_ms,
                "total_energy_j": result.total_energy_j,
                "peak_gpu_mem_mb": result.peak_gpu_mem_mb,
                "plan_selection_mode": args.plan_selection_mode,
                "plan_source": args.plan_source,
                "plan_source_csv": args.plan_source_csv,
                "plan_gpu_budget_mb": args.plan_gpu_budget_mb,
                "plan_objective": args.plan_objective,
                "transfer_events": result.total_transfer_events,
                "final_loss": result.final_loss,
                "loss_delta": result.loss_delta,
                "quality_metric_name": result.quality_metric_name or task_metric_name,
                "final_quality_metric": result.final_quality_metric,
                "quality_metric_delta": result.quality_metric_delta,
                "input_source": data_info.get("input_source"),
                "target_source": data_info.get("target_source"),
                "dataset_name": data_info.get("dataset_name"),
                "dataset_split": data_info.get("dataset_split"),
                "dataset_path": data_info.get("dataset_path"),
            }
        )

    baseline_label = "all_gpu" if "all_gpu" in summary_payload["runs"] else ("all_cpu" if "all_cpu" in summary_payload["runs"] else None)
    if baseline_label is not None:
        baseline = summary_payload["runs"][baseline_label]
        if baseline.get("status") == "ok":
            baseline_loss = float(baseline.get("final_loss", 0.0))
            baseline_time = float(baseline.get("avg_step_ms", 0.0))
            baseline_metric = baseline.get("final_quality_metric")
            for row in rows:
                row["delta_final_loss_vs_baseline"] = float(row["final_loss"] - baseline_loss)
                row["delta_avg_step_ms_vs_baseline"] = float(row["avg_step_ms"] - baseline_time)
                if baseline_metric is not None and row["final_quality_metric"] is not None:
                    row["delta_final_quality_metric_vs_baseline"] = float(row["final_quality_metric"] - baseline_metric)
            summary_payload["protocol"] = {
                "baseline": baseline_label,
                "note": "Negative delta_final_loss_vs_baseline indicates lower final training loss than baseline.",
            }
        else:
            summary_payload["protocol"] = {
                "baseline": baseline_label,
                "note": "Baseline deltas omitted because the selected baseline did not complete successfully.",
            }

    summary_path = out_dir / "hybrid_execution_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    traces_path = out_dir / "hybrid_execution_steps.csv"
    ilp_steps = pd.DataFrame([s.__dict__ for s in runs["ilp_plan"]["result"].per_step])
    ilp_steps.to_csv(traces_path, index=False)

    protocol_path = out_dir / "hybrid_execution_protocol.csv"
    pd.DataFrame(rows).to_csv(protocol_path, index=False)

    ilp_result = runs["ilp_plan"]["result"]
    print("=" * 80)
    print("HYBRID EXECUTION")
    print("=" * 80)
    print(f"Model: {args.model} | Precision: {args.precision}")
    print(f"ILP status: {ilp_result.status}")
    print(f"ILP avg step (ms): {ilp_result.avg_step_ms:.6f}")
    print(f"ILP final loss: {ilp_result.final_loss:.6f} (delta={ilp_result.loss_delta:.6f})")
    if ilp_result.quality_metric_name and ilp_result.final_quality_metric is not None:
        print(
            f"ILP final {ilp_result.quality_metric_name}: {ilp_result.final_quality_metric:.6f} "
            f"(delta={float(ilp_result.quality_metric_delta or 0.0):.6f})"
        )
    if args.compare_baselines:
        print("Comparative protocol rows:")
        for row in rows:
            metric_suffix = ""
            if row["quality_metric_name"] and row["final_quality_metric"] is not None:
                metric_suffix = f", {row['quality_metric_name']}={row['final_quality_metric']:.6f}"
            print(
                f"  - {row['run_label']}: final_loss={row['final_loss']:.6f}, "
                f"avg_step_ms={row['avg_step_ms']:.6f}{metric_suffix}"
            )
    print(f"Summary JSON: {summary_path}")
    print(f"ILP step traces CSV: {traces_path}")
    print(f"Protocol CSV: {protocol_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
