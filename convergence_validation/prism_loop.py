"""Real multi-epoch training loop driven by an already-solved PRISM/ILP plan.

Why this is not just calling `run_hybrid_training` in a loop
---------------------------------------------------------------
`src/runtime/hybrid_executor.py::run_hybrid_training` traces the model's
computation graph (via `fx.symbolic_trace`, or `torch.export` for
decoder-only language models) once per CALL, before its internal step loop.
If this module called that function once per training step (the only way to
feed it a different real batch every step, since its own loop reuses one
fixed batch), the decoder-only models would re-run `torch.export.export`
thousands of times — correct, but needlessly slow for a one-off data
collection run.

Instead, this module imports (read-only — `src/runtime/hybrid_executor.py`
is never modified) the same two building blocks `run_hybrid_training` itself
uses to decide *where each operator runs*:

- `_try_symbolic_trace` / `try_export_decoder_only_trace`: produce the traced
  graph, once, at the start of this run.
- `_DeviceAwareFXInterpreter`: executes that graph, dispatching every
  operator to the CPU or GPU exactly as the loaded ILP plan says.

and drives them with a real, epoch-cycling data provider (see
`datasets.py`) instead of one fixed batch. This is the same PRISM execution
engine used throughout the rest of the thesis, not a re-implementation of
it — the only new part is the outer loop (real data, many epochs, periodic
validation), which is exactly what was missing.

Known limitation, documented rather than hidden
--------------------------------------------------
For the decoder-only language models (`distilgpt2`, `gpt2_small`), the graph
is produced by `torch.export`, which bakes in the model's `training=True`
state at trace time (this matches how `run_hybrid_training` itself traces —
see its `model.train()` call before tracing). Calling `model.eval()` later
does not change dropout behavior inside that already-exported graph, so the
periodic validation pass for these two models runs with dropout still
active, adding mild noise (not bias in a particular direction) to the
reported validation loss/token-accuracy. This does not apply to the vision
models (`resnet50`, `resnet152`): `fx.symbolic_trace` keeps the original
calls to the real BatchNorm/Dropout submodules, which do respond to
`model.eval()` normally.
"""
from __future__ import annotations

import gc
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

from core.decoder_export_backend import try_export_decoder_only_trace  # noqa: E402
from core.energy import EnergyMonitor  # noqa: E402
from models.factory import _build_model  # noqa: E402
from runtime.device_plan import DevicePlan, validate_plan_coverage  # noqa: E402
from runtime.hybrid_executor import (  # noqa: E402
    HybridExecutionUnsupportedError,
    _DeviceAwareFXInterpreter,
    _compute_objective_possibly_deferred,
    _prepare_fx_run_args,
    _try_symbolic_trace,
)
from runtime.plan_representation import load_execution_plan  # noqa: E402

from all_gpu_loop import (  # noqa: E402
    ConvergenceRunResult,
    EvalRecord,
    TrainStepRecord,
    _peak_cpu_mem_mb,
    _peak_gpu_mem_mb,
    _torch_dtype_from_precision,
)
from datasets import EpochBatchProvider, make_provider  # noqa: E402
from mixed_precision import TrainableModel, build_trainable_model, optimizer_cls_from_name  # noqa: E402


def _empty_runtime_features() -> dict[str, Any]:
    """Same shape `_run_hybrid_training_dag` initializes per step in
    `src/runtime/hybrid_executor.py` — `_DeviceAwareFXInterpreter` writes
    into these keys as it runs, so they must exist even though this module
    does not report most of them (latency/energy instrumentation is already
    covered by the existing Chapter 6 campaign; this experiment only needs
    loss/accuracy)."""
    return {
        "recompute_layers": set(),
        "checkpoint_layers": set(),
        "backward_relocation_layers": set(),
        "prefetch_layers": set(),
        "unsupported_activation_layers": set(),
        "recompute_count": 0,
        "checkpoint_count": 0,
        "backward_relocation_count": 0,
        "prefetch_count": 0,
        "prefetch_bytes": 0.0,
    }


@dataclass
class _TracedGraph:
    graph_module: Any
    node_layer_names: dict[str, str] | None
    supports_activation_strategies: bool


def _trace_once(model: torch.nn.Module, example_input: Any) -> _TracedGraph:
    """Trace `model` exactly once, the same way
    `run_hybrid_training` does: try `fx.symbolic_trace` first (vision
    models), fall back to `torch.export`-based decoder tracing (language
    models) otherwise."""
    traced_module = _try_symbolic_trace(model)
    if traced_module is not None:
        return _TracedGraph(graph_module=traced_module, node_layer_names=None, supports_activation_strategies=True)

    export_ctx = try_export_decoder_only_trace(model, example_input)
    if export_ctx is None:
        raise HybridExecutionUnsupportedError(
            f"Could not trace model '{type(model).__name__}' via symbolic_trace or torch.export; "
            "PRISM execution requires one of these to dispatch operators to CPU/GPU."
        )
    return _TracedGraph(
        graph_module=export_ctx.graph_module,
        node_layer_names=export_ctx.node_layer_names,
        supports_activation_strategies=False,
    )


def _forward_through_plan(
    traced: _TracedGraph,
    plan: DevicePlan,
    activation_strategies: dict[str, Any] | None,
    gpu_id: int,
    warnings: list[str],
    input_data: Any,
    pinned_output_buffer_cache: dict[str, Any] | None = None,
    runtime_features: dict[str, Any] | None = None,
) -> tuple[Any, _DeviceAwareFXInterpreter]:
    """One forward pass, dispatching each operator to CPU/GPU exactly as
    `plan` specifies. A fresh `_DeviceAwareFXInterpreter` is built per call
    (it is lightweight — the expensive part, graph tracing, already happened
    once in `_trace_once`). Returns the interpreter too (not just its
    output): the caller MUST call `interpreter.env.clear()` once it is done
    with the returned value (after .backward(), or right after reading the
    loss in eval mode) -- see _run_one_optimizer_step/_evaluate. Mirrors the
    identical fix in src/runtime/hybrid_executor.py::_run_hybrid_training_dag:
    when the final layer is deferred (see _DeferredCheckpointSegment),
    self.env holds a closure that captures `self`, forming a reference
    cycle (interp -> env -> deferred -> closure -> interp) that only
    Python's periodic cyclic GC -- not plain refcounting -- can free.
    Confirmed the hard way: this exact gap (this function never clearing
    env, and never routing through _compute_objective_possibly_deferred at
    all -- see the caller) OOM'd a real distilgpt2/batch_64 run by step 3
    even though the identical plan ran 50 stable steps through the fixed
    hybrid_executor.py path.

    `pinned_output_buffer_cache` MUST be created once by the REAL caller
    (`run_prism_training`) and reused across every call for the whole run
    (training steps AND eval batches), never created fresh here — same
    requirement, same reason, as `_run_hybrid_training_dag`'s own cache in
    `hybrid_executor.py`: PyTorch's pinned-memory allocator pools
    page-locked host allocations internally and does not release them back
    to the OS promptly even after the old dict is garbage-collected, so a
    fresh cache built on every forward pass accumulates real, unbounded
    host memory across steps. Confirmed the hard way: this exact bug (this
    function had no such parameter at all, so `_DeviceAwareFXInterpreter`
    always built its own fresh `{}`) OOM-killed a real resnet50/batch_192
    run partway through its first epoch, RSS climbing ~800MB/step until the
    run's own cgroup memory ceiling SIGKILLed it at ~23GB. Default `None`
    (fresh, single-use dict, same convention as
    `_DeviceAwareFXInterpreter.__init__` itself) exists only so tests can
    call this function directly without wiring a cache through."""
    if pinned_output_buffer_cache is None:
        pinned_output_buffer_cache = {}
    if runtime_features is None:
        runtime_features = _empty_runtime_features()
    interpreter = _DeviceAwareFXInterpreter(
        traced.graph_module,
        plan,
        gpu_id,
        {"bytes": 0.0, "events": 0.0},
        activation_strategies,
        warnings,
        runtime_features,
        False,  # enable_async_transfer: not relevant to a loss/accuracy study
        False,  # enable_prefetch: ditto
        None,  # transfer_stream
        node_layer_names=traced.node_layer_names,
        supports_activation_strategies=traced.supports_activation_strategies,
        pinned_output_buffer_cache=pinned_output_buffer_cache,
    )
    out = interpreter.run(*_prepare_fx_run_args(traced.graph_module, input_data))
    return out, interpreter


def _run_one_optimizer_step(
    trainable: TrainableModel,
    traced: _TracedGraph,
    plan: DevicePlan,
    train_provider: EpochBatchProvider,
    accumulation_steps: int,
    gpu_id: int,
    warnings: list[str],
    pinned_output_buffer_cache: dict[str, Any] | None = None,
) -> tuple[float, str | None, float | None]:
    trainable.zero_grad()
    total_loss = 0.0
    metric_name: str | None = None
    metric_value: float | None = None

    for _ in range(accumulation_steps):
        input_data, target = train_provider()
        runtime_features = _empty_runtime_features()
        out, interpreter = _forward_through_plan(
            traced, plan, plan.activation_strategies, gpu_id, warnings, input_data,
            pinned_output_buffer_cache, runtime_features,
        )
        loss, step_metric_name, step_metric_value = _compute_objective_possibly_deferred(
            out, target_data=target, runtime_features=runtime_features
        )

        total_loss += float(loss.detach().item())
        metric_name, metric_value = step_metric_name, step_metric_value

        (loss / accumulation_steps).backward()
        interpreter.env.clear()

    trainable.step()
    return total_loss / accumulation_steps, metric_name, metric_value


def _evaluate(
    trainable: TrainableModel,
    traced: _TracedGraph,
    plan: DevicePlan,
    val_provider: EpochBatchProvider,
    eval_batches: int,
    gpu_id: int,
    warnings: list[str],
    pinned_output_buffer_cache: dict[str, Any] | None = None,
) -> tuple[float, str | None, float | None]:
    model = trainable.model
    was_training = model.training
    model.eval()
    try:
        traced.graph_module.eval()
    except NotImplementedError:
        pass  # see module docstring: decoder-only export graphs keep train-time behavior

    losses: list[float] = []
    metric_values: list[float] = []
    metric_name: str | None = None
    try:
        with torch.no_grad():
            for _ in range(eval_batches):
                input_data, target = val_provider()
                runtime_features = _empty_runtime_features()
                out, interpreter = _forward_through_plan(
                    traced, plan, plan.activation_strategies, gpu_id, warnings, input_data,
                    pinned_output_buffer_cache, runtime_features,
                )
                loss, step_metric_name, step_metric_value = _compute_objective_possibly_deferred(
                    out, target_data=target, runtime_features=runtime_features
                )
                losses.append(float(loss.item()))
                if step_metric_value is not None:
                    metric_values.append(step_metric_value)
                    metric_name = step_metric_name
                interpreter.env.clear()
        # Reference cycles through cross-device autograd bookkeeping (see
        # run_prism_training's own gc.collect() for the full measured
        # evidence: +73.6MB/step growth, reclaimed in full by gc.collect(),
        # confirmed independent of CUDA allocator fragmentation via
        # gc.disable()) accumulate here exactly as in training -- this
        # function shares the same _forward_through_plan/_DeviceAwareFXInterpreter
        # path, just under no_grad.
        gc.collect()
    finally:
        model.train(was_training)
        try:
            traced.graph_module.train(was_training)
        except NotImplementedError:
            pass

    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    mean_metric = sum(metric_values) / len(metric_values) if metric_values else None
    return mean_loss, metric_name, mean_metric


def run_prism_training(
    model_name: str,
    precision: str,
    batch_size: int,
    datasets_root: Path,
    assignment_csv: Path,
    cut_edges_csv: Path,
    epochs: int,
    lr: float,
    seed: int,
    optimizer_name: str = "sgd",
    accumulation_steps: int = 1,
    eval_every_steps: int = 50,
    eval_batches: int = 10,
    seq_length: int = 128,
    gpu_id: int = 0,
    strict_plan: bool = False,
    label: str = "ilp_plan",
    max_steps: int | None = None,
) -> ConvergenceRunResult:
    """Train `model_name` for `epochs` real epochs using the device
    placement already solved in `assignment_csv`/`cut_edges_csv` (an
    existing `data/<host>/.../ilp_solution*/` directory from a prior
    profiling+ILP campaign run — see `convergence_validation/README.md` for
    why no new profiling/ILP solve is needed here).

    `max_steps`, if given, caps the number of optimizer steps regardless of
    `epochs` — useful for a quick smoke check of a configuration before
    committing to a full run.
    """
    setup_start = time.perf_counter()
    torch.manual_seed(seed)
    torch_dtype = _torch_dtype_from_precision(precision)

    execution_plan = load_execution_plan(assignment_csv=assignment_csv, cut_edges_csv=cut_edges_csv)
    plan = DevicePlan.from_execution_plan(execution_plan)

    model_args = SimpleNamespace(model=model_name, batch_size=batch_size)
    model = _build_model(model_args, torch_dtype)
    model.train()

    trainable = build_trainable_model(model, precision=precision, optimizer_cls=optimizer_cls_from_name(optimizer_name), lr=lr)

    train_provider = make_provider(model_name, datasets_root, batch_size, "train", seed, torch_dtype, seq_length)
    val_provider = make_provider(model_name, datasets_root, batch_size, "val", seed + 1, torch_dtype, seq_length)

    first_input, _ = train_provider()
    try:
        traced = _trace_once(model, first_input)
    except HybridExecutionUnsupportedError as exc:
        return ConvergenceRunResult(
            label=label,
            model_name=model_name,
            precision=precision,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            status="unsupported",
            error=str(exc),
        )

    coverage_layer_names = None if traced.node_layer_names is None else set(traced.node_layer_names.values())
    warnings = validate_plan_coverage(model=model, plan=plan, strict=strict_plan, model_layer_names=coverage_layer_names)

    if not torch.cuda.is_available() and (
        any(v == "GPU" for v in plan.assignment_forward.values())
        or any(v == "GPU" for v in plan.assignment_backward.values())
    ):
        warnings.append("GPU layers requested but CUDA is unavailable; those layers will run on CPU")

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
    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu_id)
    result.setup_wall_time_s = time.perf_counter() - setup_start

    # Created ONCE for the whole run (training steps AND eval batches) and
    # passed into every _forward_through_plan call below -- see that
    # function's docstring for why a fresh cache per call is a real,
    # confirmed OOM bug, not just a missed optimization.
    pinned_output_buffer_cache: dict[str, Any] = {}

    gpu_device = torch.device(f"cuda:{gpu_id}")
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.cuda.reset_peak_memory_stats(gpu_device)
    # Same EnergyMonitor already used by
    # src/runtime/hybrid_executor.py::run_hybrid_training for the
    # run_thesis_mode.sh campaigns, reused here (not reinvented) so PRISM
    # and the All-GPU/greedy baselines (all_gpu_loop.py) report comparable
    # power/energy numbers. A no-op if CUDA is unavailable (NVML init
    # inside EnergyMonitor fails gracefully; get_avg_power() then returns 0).
    energy_monitor = EnergyMonitor(device_type="cuda", gpu_id=gpu_id)
    energy_monitor.start()

    try:
        for step in range(1, total_optimizer_steps + 1):
            epoch = (step - 1) // steps_per_epoch
            step_start = time.perf_counter()
            try:
                loss, metric_name, metric_value = _run_one_optimizer_step(
                    trainable, traced, plan, train_provider, accumulation_steps, gpu_id, warnings,
                    pinned_output_buffer_cache,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(gpu_id)
            except torch.cuda.OutOfMemoryError as exc:
                # Should not normally happen — the whole point of the ILP plan is
                # to fit the declared budget — but a genuinely tight budget plus
                # real-world fragmentation could still OOM. Reported the same way
                # as all_gpu_loop.py's handling so the orchestrator can keep going
                # and the failure is visible in the written results instead of a
                # crashed process with no CSV output.
                torch.cuda.empty_cache()
                result.status = "oom"
                result.error = f"PRISM training ran out of VRAM at step {step} (batch_size={batch_size}): {exc}"
                return result
            except RuntimeError as exc:
                if "Dual-device recompute mismatch" not in str(exc):
                    raise
                # src/runtime/hybrid_executor.py::_run_layer_with_dual_placement
                # recomputes a layer on its backward device and compares against
                # the forward-device output (rtol=1e-4, atol=1e-5) as a safety
                # check for its forward-on-one-device/backward-on-another
                # strategy. Observed in practice (see
                # fix/01_accuracy_convergence/README.md): plain FP32 conv/batchnorm
                # kernels do not guarantee bit-identical results between CPU
                # (oneDNN) and GPU (cuDNN) backends, so this check can fire on
                # entirely correct computations whenever a real (non-repeated)
                # batch happens to land close to the tolerance boundary. This is
                # reported as a distinct status rather than silently retried or
                # swallowed, because it is evidence about the existing runtime's
                # behavior under real data, not a bug in this module.
                result.status = "dual_placement_mismatch"
                result.error = f"Dual-device recompute mismatch at step {step}: {exc}"
                return result

            step_wall_time_s = time.perf_counter() - step_start
            result.total_wall_time_s += step_wall_time_s
            # Reference cycles through cross-device autograd bookkeeping
            # accumulate real, unreclaimed CUDA memory every step (confirmed
            # empirically: +73.6MB/step linear growth, independent of CUDA
            # allocator fragmentation -- reproduced with gc.disable() showing
            # the identical growth, and a manual gc.collect() reclaiming it
            # in full -- see fix/08 README). Too slow to matter for a short
            # 10-step validation probe, but a real multi-hundred-step
            # convergence run lets it ratchet peak_gpu_mem_mb up far past
            # what the plan actually needs. Placed AFTER step_wall_time_s is
            # captured so this does not inflate the timing metric also
            # reported/compared here.
            gc.collect()
            result.train_steps.append(
                TrainStepRecord(
                    step=step, epoch=epoch, loss=loss, metric_name=metric_name, metric_value=metric_value,
                    step_wall_time_s=step_wall_time_s,
                    peak_gpu_mem_mb=_peak_gpu_mem_mb(gpu_device) if has_gpu else 0.0,
                    peak_cpu_mem_mb=_peak_cpu_mem_mb(),
                )
            )

            if step % eval_every_steps == 0 or step == total_optimizer_steps:
                eval_loss, eval_metric_name, eval_metric_value = _evaluate(
                    trainable, traced, plan, val_provider, eval_batches, gpu_id, warnings,
                    pinned_output_buffer_cache,
                )
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
