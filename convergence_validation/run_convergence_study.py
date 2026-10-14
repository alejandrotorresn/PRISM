#!/usr/bin/env python3
"""CLI entry point for one convergence-study run.

One call of this script trains ONE configuration (model x precision x mode x
regime x seed) for real, and writes its loss/accuracy curve to disk. The
matrix across all configurations is driven by `run_convergence_study.sh`.

`--mode ilp_plan` reads an ALREADY SOLVED ILP plan from `--config_dir`
(an existing `data/<host>/.../<model>/.../batch_<N>/` directory from a prior
profiling+ILP campaign run) — it does not run profiling or the ILP solver.
See `convergence_validation/README.md` for why that is not needed here.

`--mode greedy_plan` is the same idea but reads a `greedy_solution_budget_*/`
directory (built ahead of time by `build_greedy_plan.py` from the same
greedy heuristic already used for Chapter 6's PRISM-vs-Greedy comparisons)
instead of an ILP plan -- it exercises the exact same PRISM runtime
(hybrid_executor), just handed a worse (non-ILP) device assignment, so the
loss/accuracy curve this produces is the fair "naive partitioning" baseline
next to PRISM's ILP plan and the real All-GPU run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "convergence_validation") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "convergence_validation"))

from all_gpu_loop import ConvergenceRunResult, run_all_gpu_training  # noqa: E402
from prism_loop import run_prism_training  # noqa: E402

MODES = ["ilp_plan", "greedy_plan", "all_gpu", "all_gpu_gradaccum"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["resnet50", "resnet152", "distilgpt2", "gpt2_small"])
    parser.add_argument("--precision", required=True, choices=["fp32", "bf16"])
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--regime", required=True, choices=["A", "B"], help="Only used to label/organize outputs")

    parser.add_argument(
        "--config_dir",
        default=None,
        help="Directory containing ilp_solution/ (regime A), "
        "ilp_solution_pareto_best_budget_*/ (regime B), or "
        "greedy_solution_budget_*/ (--mode greedy_plan). Required for "
        "--mode ilp_plan and --mode greedy_plan.",
    )

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Only meaningful for --mode all_gpu_gradaccum")

    parser.add_argument("--datasets_root", default=str(_REPO_ROOT / "datasets"))
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=["sgd", "adamw"],
        help="SGD at lr~0.01 suits the lightly fine-tuned vision models; the pretrained "
        "language models need AdamW at a much smaller lr (~5e-5) or their loss diverges "
        "within a few steps — see mixed_precision.py::optimizer_cls_from_name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Cap the number of optimizer steps regardless of --epochs (smoke-testing a configuration)",
    )
    return parser


def _resolve_ilp_csvs(config_dir: Path, regime: str) -> tuple[Path, Path]:
    if regime == "A":
        solution_dir = config_dir / "ilp_solution"
    else:
        candidates = sorted(config_dir.glob("ilp_solution_pareto_best_budget_*"))
        if not candidates:
            raise FileNotFoundError(
                f"No 'ilp_solution_pareto_best_budget_*' directory found under {config_dir} "
                "for regime B. This model/batch combination may need a one-off reduced-budget "
                "ILP solve (see convergence_validation/README.md) before it can be used here."
            )
        solution_dir = candidates[0]

    assignment_csv = solution_dir / "ilp_assignment.csv"
    cut_edges_csv = solution_dir / "ilp_cut_edges.csv"
    if not assignment_csv.exists() or not cut_edges_csv.exists():
        raise FileNotFoundError(f"Expected ILP solution files under {solution_dir}, found: {sorted(solution_dir.glob('*'))}")
    return assignment_csv, cut_edges_csv


def _resolve_greedy_csvs(config_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(config_dir.glob("greedy_solution_budget_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No 'greedy_solution_budget_*' directory found under {config_dir} for "
            "--mode greedy_plan. Run convergence_validation/build_greedy_plan.py for "
            "this config_dir first."
        )
    solution_dir = candidates[0]
    assignment_csv = solution_dir / "ilp_assignment.csv"
    cut_edges_csv = solution_dir / "ilp_cut_edges.csv"
    if not assignment_csv.exists() or not cut_edges_csv.exists():
        raise FileNotFoundError(f"Expected greedy plan files under {solution_dir}, found: {sorted(solution_dir.glob('*'))}")
    return assignment_csv, cut_edges_csv


def _write_result(result: ConvergenceRunResult, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_loss.csv"
    with train_csv.open("w", encoding="utf-8") as f:
        f.write("step,epoch,loss,metric_name,metric_value,step_wall_time_s,peak_gpu_mem_mb,peak_cpu_mem_mb\n")
        for record in result.train_steps:
            metric_value = "" if record.metric_value is None else record.metric_value
            metric_name = record.metric_name or ""
            f.write(
                f"{record.step},{record.epoch},{record.loss},{metric_name},{metric_value},"
                f"{record.step_wall_time_s},{record.peak_gpu_mem_mb},{record.peak_cpu_mem_mb}\n"
            )

    eval_csv = output_dir / "eval_metrics.csv"
    with eval_csv.open("w", encoding="utf-8") as f:
        f.write("after_step,epoch,loss,metric_name,metric_value\n")
        for record in result.eval_points:
            metric_value = "" if record.metric_value is None else record.metric_value
            metric_name = record.metric_name or ""
            f.write(f"{record.after_step},{record.epoch},{record.loss},{metric_name},{metric_value}\n")

    summary = {
        "label": result.label,
        "model": result.model_name,
        "precision": result.precision,
        "batch_size": result.batch_size,
        "accumulation_steps": result.accumulation_steps,
        "effective_batch_size": result.batch_size * result.accumulation_steps,
        "status": result.status,
        "error": result.error,
        "mode": args.mode,
        "regime": args.regime,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "num_train_steps": len(result.train_steps),
        "num_eval_points": len(result.eval_points),
        "final_train_loss": result.train_steps[-1].loss if result.train_steps else None,
        "final_eval_loss": result.eval_points[-1].loss if result.eval_points else None,
        "final_eval_metric_name": result.eval_points[-1].metric_name if result.eval_points else None,
        "final_eval_metric_value": result.eval_points[-1].metric_value if result.eval_points else None,
        # Wall-clock timing (torch.cuda.synchronize()'d around each step, so
        # this is real measured time, not a simulated/ILP-predicted
        # estimate) -- see all_gpu_loop.py's ConvergenceRunResult docstring
        # for why setup time (model build, tracing, provider construction)
        # is reported separately from the training-loop time.
        "setup_wall_time_s": result.setup_wall_time_s,
        "total_train_wall_time_s": result.total_wall_time_s,
        "total_wall_time_s": result.setup_wall_time_s + result.total_wall_time_s,
        "avg_step_wall_time_ms": (
            1000.0 * result.total_wall_time_s / len(result.train_steps) if result.train_steps else None
        ),
        # Run-level peaks/aggregates -- see ConvergenceRunResult's own
        # docstring in all_gpu_loop.py for the measurement methodology
        # (same EnergyMonitor/torch.cuda.max_memory_allocated/ru_maxrss
        # already used by src/runtime/hybrid_executor.py's
        # run_hybrid_training, reused here so PRISM and the baselines are
        # comparable). peak_gpu_mem_mb/peak_cpu_mem_mb are also available
        # per-step in train_loss.csv for a memory-over-training plot.
        "peak_gpu_mem_mb": result.peak_gpu_mem_mb,
        "peak_cpu_mem_mb": result.peak_cpu_mem_mb,
        "avg_power_w": result.avg_power_w,
        "total_energy_j": result.total_energy_j,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{result.status}] {result.label} model={result.model_name} precision={result.precision} "
          f"batch={result.batch_size} accum={result.accumulation_steps}")
    if result.error:
        print(f"  error: {result.error}")
    if result.eval_points:
        last = result.eval_points[-1]
        metric_text = "" if last.metric_value is None else f", {last.metric_name}={last.metric_value:.4f}"
        print(f"  final eval: loss={last.loss:.4f}{metric_text}")
    if result.train_steps:
        avg_ms = 1000.0 * result.total_wall_time_s / len(result.train_steps)
        print(f"  timing: setup={result.setup_wall_time_s:.2f}s, train_loop={result.total_wall_time_s:.2f}s, avg_step={avg_ms:.1f}ms")
        print(
            f"  memory/energy: peak_gpu={result.peak_gpu_mem_mb:.1f}MB, peak_cpu={result.peak_cpu_mem_mb:.1f}MB, "
            f"avg_power={result.avg_power_w:.1f}W, total_energy={result.total_energy_j:.1f}J"
        )
    print(f"  wrote: {train_csv}, {eval_csv}, {output_dir / 'summary.json'}")


def main() -> int:
    args = _build_parser().parse_args()
    datasets_root = Path(args.datasets_root)
    output_dir = Path(args.output_dir)

    if args.mode in ("ilp_plan", "greedy_plan"):
        if not args.config_dir:
            raise SystemExit(f"--config_dir is required for --mode {args.mode}")
        if args.mode == "ilp_plan":
            assignment_csv, cut_edges_csv = _resolve_ilp_csvs(Path(args.config_dir), args.regime)
        else:
            assignment_csv, cut_edges_csv = _resolve_greedy_csvs(Path(args.config_dir))
        result = run_prism_training(
            model_name=args.model,
            precision=args.precision,
            batch_size=args.batch_size,
            datasets_root=datasets_root,
            assignment_csv=assignment_csv,
            cut_edges_csv=cut_edges_csv,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            optimizer_name=args.optimizer,
            accumulation_steps=args.accumulation_steps,
            eval_every_steps=args.eval_every_steps,
            eval_batches=args.eval_batches,
            seq_length=args.seq_length,
            gpu_id=args.gpu_id,
            label=args.mode,
            max_steps=args.max_steps,
        )
    else:
        label = "all_gpu_gradaccum" if args.mode == "all_gpu_gradaccum" else "all_gpu"
        accumulation_steps = args.accumulation_steps if args.mode == "all_gpu_gradaccum" else 1
        result = run_all_gpu_training(
            model_name=args.model,
            precision=args.precision,
            batch_size=args.batch_size,
            datasets_root=datasets_root,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            optimizer_name=args.optimizer,
            accumulation_steps=accumulation_steps,
            eval_every_steps=args.eval_every_steps,
            eval_batches=args.eval_batches,
            seq_length=args.seq_length,
            gpu_id=args.gpu_id,
            label=label,
            max_steps=args.max_steps,
        )

    _write_result(result, output_dir, args)
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
