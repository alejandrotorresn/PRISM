"""Standard PyTorch training loop for the All-GPU baseline.

Why this is a separate, from-scratch loop
------------------------------------------
"All-GPU" means every operator runs on the accelerator — there is no CPU/GPU
placement decision to make, so there is nothing to reuse from
`src/runtime/hybrid_executor.py` (that module exists specifically to dispatch
layers across devices according to an ILP plan). Writing this as a plain
`model.to("cuda")` + standard forward/backward/step loop is both simpler and
more representative of what "All-GPU" actually means to a reader comparing it
against PRISM.

Two regimes are implemented here:

- `run_all_gpu_training(..., accumulation_steps=1)`: the direct Régimen A
  comparison (PRISM vs. All-GPU, same batch size, same everything else).
- `run_all_gpu_training(..., accumulation_steps=K > 1)`: the Régimen B
  secondary baseline — a smaller micro-batch that *does* fit under a reduced
  VRAM budget, accumulated over K micro-steps to reach the same *effective*
  batch size PRISM trains at when All-GPU cannot run that batch natively at
  all. See `convergence_validation/README.md` for why this is reported as a
  secondary comparison, not the only Régimen B evidence (BatchNorm normalizes
  per micro-batch, not per effective batch, so it is not numerically
  identical to a real large-batch run).
"""
from __future__ import annotations

import gc
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.energy import EnergyMonitor  # noqa: E402
from core.loss_utils import compute_training_objective  # noqa: E402
from models.factory import _build_model  # noqa: E402


def _peak_cpu_mem_mb() -> float:
    # ru_maxrss is already a lifetime-peak-since-process-start value on
    # Linux (kilobytes), not a point-in-time reading -- no need to sample
    # repeatedly or reset it ourselves.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _peak_gpu_mem_mb(device: torch.device) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0**2)

from datasets import EpochBatchProvider, make_provider  # noqa: E402
from mixed_precision import TrainableModel, build_trainable_model, optimizer_cls_from_name  # noqa: E402


def _torch_dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(v, device) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_to_device(v, device) for v in value]
        return type(value)(moved)
    return value


@dataclass
class TrainStepRecord:
    step: int
    epoch: int
    loss: float
    metric_name: str | None
    metric_value: float | None
    step_wall_time_s: float = 0.0
    # Peak-so-far (not per-step delta) -- torch.cuda.max_memory_allocated()
    # and ru_maxrss are both running peaks since the process/CUDA-context
    # started, so these columns trace out a monotonically non-decreasing
    # curve across steps, useful for a memory-over-training plot.
    peak_gpu_mem_mb: float = 0.0
    peak_cpu_mem_mb: float = 0.0


@dataclass
class EvalRecord:
    after_step: int
    epoch: int
    loss: float
    metric_name: str | None
    metric_value: float | None


@dataclass
class ConvergenceRunResult:
    label: str
    model_name: str
    precision: str
    batch_size: int
    accumulation_steps: int
    train_steps: list[TrainStepRecord] = field(default_factory=list)
    eval_points: list[EvalRecord] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    # Wall-clock seconds spent in the training loop (sum of per-step times,
    # measured with torch.cuda.synchronize() around each step so this
    # reflects actual GPU completion time, not just CPU-side kernel-launch
    # time) -- excludes one-time setup (model build, dataset provider
    # construction). This is what PRISM's whole value proposition is
    # actually about (speed under a memory budget, not just not crashing),
    # so convergence evidence without it is incomplete.
    total_wall_time_s: float = 0.0
    setup_wall_time_s: float = 0.0
    # Run-level peaks/aggregates, same methodology already used by
    # src/runtime/hybrid_executor.py::run_hybrid_training for the
    # run_thesis_mode.sh campaigns -- reused here (not reinvented) so PRISM
    # and the All-GPU/greedy baselines report comparable numbers.
    # avg_power_w/total_energy_j cover the training loop only (the
    # EnergyMonitor starts after setup and stops before this function
    # returns); peak_gpu_mem_mb/peak_cpu_mem_mb are the final (highest)
    # values across all recorded train_steps.
    peak_gpu_mem_mb: float = 0.0
    peak_cpu_mem_mb: float = 0.0
    avg_power_w: float = 0.0
    total_energy_j: float = 0.0


def _run_one_optimizer_step(
    trainable: TrainableModel,
    train_provider: EpochBatchProvider,
    accumulation_steps: int,
    device: torch.device,
) -> tuple[float, str | None, float | None]:
    """Run `accumulation_steps` micro-batches (>=1) under a single
    `zero_grad()`/`optimizer.step()` pair, scaling each micro-batch's loss by
    `1 / accumulation_steps` before `.backward()` so accumulated gradients
    have the same expected magnitude as a single large-batch gradient.
    Returns the (unscaled) mean loss and the last micro-batch's quality
    metric, for logging.
    """
    trainable.zero_grad()
    total_loss = 0.0
    metric_name: str | None = None
    metric_value: float | None = None

    for _ in range(accumulation_steps):
        input_data, target = train_provider()
        input_data = _move_to_device(input_data, device)
        target = _move_to_device(target, device)

        out = trainable.model(**input_data) if isinstance(input_data, dict) else trainable.model(input_data)
        loss, step_metric_name, step_metric_value = compute_training_objective(out, target=target)

        total_loss += float(loss.detach().item())
        metric_name, metric_value = step_metric_name, step_metric_value

        (loss / accumulation_steps).backward()

    trainable.step()
    return total_loss / accumulation_steps, metric_name, metric_value


def _evaluate(
    trainable: TrainableModel,
    val_provider: EpochBatchProvider,
    eval_batches: int,
    device: torch.device,
) -> tuple[float, str | None, float | None]:
    model = trainable.model
    was_training = model.training
    model.eval()
    losses: list[float] = []
    metric_values: list[float] = []
    metric_name: str | None = None
    try:
        with torch.no_grad():
            for _ in range(eval_batches):
                input_data, target = val_provider()
                input_data = _move_to_device(input_data, device)
                target = _move_to_device(target, device)
                out = model(**input_data) if isinstance(input_data, dict) else model(input_data)
                loss, step_metric_name, step_metric_value = compute_training_objective(out, target=target)
                losses.append(float(loss.item()))
                if step_metric_value is not None:
                    metric_values.append(step_metric_value)
                    metric_name = step_metric_name
        gc.collect()  # see run_all_gpu_training's own gc.collect() for why
    finally:
        model.train(was_training)

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    mean_metric = sum(metric_values) / len(metric_values) if metric_values else None
    return mean_loss, metric_name, mean_metric


def run_all_gpu_training(
    model_name: str,
    precision: str,
    batch_size: int,
    datasets_root: Path,
    epochs: int,
    lr: float,
    seed: int,
    optimizer_name: str = "sgd",
    accumulation_steps: int = 1,
    eval_every_steps: int = 50,
    eval_batches: int = 10,
    seq_length: int = 128,
    input_size: int = 224,
    gpu_id: int = 0,
    label: str = "all_gpu",
    max_steps: int | None = None,
) -> ConvergenceRunResult:
    """Train `model_name` entirely on GPU for `epochs` real epochs, logging a
    training-loss point every optimizer step and a held-out validation
    point every `eval_every_steps` optimizer steps.

    `accumulation_steps=1` is the direct Régimen A baseline. For Régimen B's
    secondary baseline, call with `batch_size` set to a micro-batch that fits
    under the reduced VRAM budget and `accumulation_steps` set so that
    `batch_size * accumulation_steps` equals the effective batch size PRISM
    is being evaluated at.

    `max_steps`, if given, caps the number of optimizer steps regardless of
    `epochs` — useful for a quick smoke check of a configuration before
    committing to a full run.
    """
    if not torch.cuda.is_available():
        return ConvergenceRunResult(
            label=label,
            model_name=model_name,
            precision=precision,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            status="unsupported",
            error="CUDA is not available; All-GPU training requires a GPU.",
        )

    setup_start = time.perf_counter()
    device = torch.device(f"cuda:{gpu_id}")
    torch_dtype = _torch_dtype_from_precision(precision)

    torch.manual_seed(seed)
    model_args = SimpleNamespace(model=model_name, batch_size=batch_size)
    try:
        model = _build_model(model_args, torch_dtype).to(device)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return ConvergenceRunResult(
            label=label,
            model_name=model_name,
            precision=precision,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            status="oom",
            error=f"All-GPU model placement ran out of VRAM at batch_size={batch_size}: {exc}",
        )
    model.train()

    trainable = build_trainable_model(model, precision=precision, optimizer_cls=optimizer_cls_from_name(optimizer_name), lr=lr)

    train_provider = make_provider(model_name, datasets_root, batch_size, "train", seed, torch_dtype, seq_length)
    val_provider = make_provider(model_name, datasets_root, batch_size, "val", seed + 1, torch_dtype, seq_length)

    steps_per_epoch = max(1, len(train_provider) // accumulation_steps)
    total_optimizer_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, max_steps)

    result = ConvergenceRunResult(
        label=label,
        model_name=model_name,
        precision=precision,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
    )
    torch.cuda.synchronize(device)
    result.setup_wall_time_s = time.perf_counter() - setup_start

    torch.cuda.reset_peak_memory_stats(device)
    energy_monitor = EnergyMonitor(device_type="cuda", gpu_id=gpu_id)
    energy_monitor.start()
    try:
        for step in range(1, total_optimizer_steps + 1):
            epoch = (step - 1) // steps_per_epoch
            step_start = time.perf_counter()
            try:
                loss, metric_name, metric_value = _run_one_optimizer_step(trainable, train_provider, accumulation_steps, device)
                torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError as exc:
                # This is the genuine Régimen-B failure mode: a batch size that
                # does not fit in physical VRAM when run monolithically on GPU.
                # Reporting it as a clean "oom" status (instead of letting the
                # exception propagate) is what lets run_convergence_study.sh
                # keep going and still write out whatever PRISM/ilp_plan produced
                # for the same configuration, which is the actual comparison of
                # interest.
                torch.cuda.empty_cache()
                result.status = "oom"
                result.error = f"All-GPU training ran out of VRAM at step {step} (batch_size={batch_size}): {exc}"
                return result

            step_wall_time_s = time.perf_counter() - step_start
            result.total_wall_time_s += step_wall_time_s
            # Defensive, not confirmed necessary on this specific path (see
            # prism_loop.py's own gc.collect() for the confirmed reference-
            # cycle leak and its measured evidence) -- this all-GPU loop
            # never moves tensors across devices, which is the mechanism
            # suspected to form the cycle, but the cost of calling this here
            # too is negligible and keeps the PRISM-vs-baseline comparison on
            # the same memory-hygiene footing either way.
            gc.collect()
            result.train_steps.append(
                TrainStepRecord(
                    step=step, epoch=epoch, loss=loss, metric_name=metric_name, metric_value=metric_value,
                    step_wall_time_s=step_wall_time_s,
                    peak_gpu_mem_mb=_peak_gpu_mem_mb(device),
                    peak_cpu_mem_mb=_peak_cpu_mem_mb(),
                )
            )

            if step % eval_every_steps == 0 or step == total_optimizer_steps:
                eval_loss, eval_metric_name, eval_metric_value = _evaluate(trainable, val_provider, eval_batches, device)
                result.eval_points.append(
                    EvalRecord(after_step=step, epoch=epoch, loss=eval_loss, metric_name=eval_metric_name, metric_value=eval_metric_value)
                )
    finally:
        energy_monitor.stop()

    result.avg_power_w = float(energy_monitor.get_avg_power())
    result.total_energy_j = result.avg_power_w * result.total_wall_time_s
    if result.train_steps:
        result.peak_gpu_mem_mb = result.train_steps[-1].peak_gpu_mem_mb
        result.peak_cpu_mem_mb = result.train_steps[-1].peak_cpu_mem_mb
    return result
