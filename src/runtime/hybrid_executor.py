from __future__ import annotations

import inspect
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from core.loss_utils import compute_training_objective
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.loss_utils import compute_training_objective

try:
    from core.decoder_export_backend import try_export_decoder_only_trace
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.decoder_export_backend import try_export_decoder_only_trace

from .device_plan import DevicePlan, validate_plan_coverage
from .leaf_modules import iter_leaf_modules

try:
    from core.energy import EnergyMonitor
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.energy import EnergyMonitor


@dataclass
class TransferEvent:
    layer: str
    src_device: str
    dst_device: str
    size_mb: float
    time_ms: float


@dataclass
class StepTrace:
    step: int
    total_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    transfer_count: int
    transfer_total_mb: float
    prefetch_count: int
    prefetch_total_mb: float
    peak_gpu_mem_mb: float
    recompute_count: int
    checkpoint_count: int
    backward_relocation_count: int
    loss_value: float
    task_metric_value: float | None = None


@dataclass
class HybridExecutionResult:
    status: str
    steps: int
    avg_step_ms: float
    avg_power_w: float
    total_energy_j: float
    energy_source: str
    total_transfer_mb: float
    total_transfer_events: int
    total_prefetch_mb: float
    total_prefetch_events: int
    peak_gpu_mem_mb: float
    initial_loss: float
    final_loss: float
    min_loss: float
    loss_delta: float
    prefetch_layers: List[str]
    recompute_layers: List[str]
    checkpoint_layers: List[str]
    backward_relocation_layers: List[str]
    unsupported_checkpoint_layers: List[str]
    warnings: List[str]
    per_step: List[StepTrace]
    quality_metric_name: str | None = None
    initial_quality_metric: float | None = None
    final_quality_metric: float | None = None
    best_quality_metric: float | None = None
    quality_metric_delta: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "steps": self.steps,
            "avg_step_ms": self.avg_step_ms,
            "avg_power_w": self.avg_power_w,
            "total_energy_j": self.total_energy_j,
            "energy_source": self.energy_source,
            "total_transfer_mb": self.total_transfer_mb,
            "total_transfer_events": self.total_transfer_events,
            "total_prefetch_mb": self.total_prefetch_mb,
            "total_prefetch_events": self.total_prefetch_events,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "loss_delta": self.loss_delta,
            "prefetch_layers": self.prefetch_layers,
            "recompute_layers": self.recompute_layers,
            "checkpoint_layers": self.checkpoint_layers,
            "backward_relocation_layers": self.backward_relocation_layers,
            "unsupported_checkpoint_layers": self.unsupported_checkpoint_layers,
            "warnings": self.warnings,
            "quality_metric_name": self.quality_metric_name,
            "initial_quality_metric": self.initial_quality_metric,
            "final_quality_metric": self.final_quality_metric,
            "best_quality_metric": self.best_quality_metric,
            "quality_metric_delta": self.quality_metric_delta,
            "per_step": [asdict(s) for s in self.per_step],
        }


class HybridExecutionUnsupportedError(RuntimeError):
    pass


def _tensor_bytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.numel() * x.element_size())
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_tensor_bytes(v) for v in x.values())
    return 0


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _to_device_non_blocking(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [_to_device_non_blocking(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_non_blocking(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device_non_blocking(v, device) for k, v in x.items()}
    return x


def _is_inplace_module(module: Any) -> bool:
    # Same detection training_profiler.py's
    # _disable_inplace_modules_for_backward_hooks already uses: a module
    # that can mutate its own input's storage in place (e.g.
    # ReLU(inplace=True), Dropout(inplace=True)) declares it via an
    # `inplace` attribute. Modules with no such attribute (Conv2d,
    # BatchNorm2d, Linear, pooling, ...) always allocate a fresh output
    # tensor and cannot mutate an upstream tensor's storage.
    return bool(getattr(module, "inplace", False))


# Hard ceiling on how many bytes of page-locked (pinned) CPU memory the
# dual-placement buffer cache may hold at once, across ALL call sites
# combined. Pinned memory is never swappable, unlike the plain CPU clone it
# replaces -- the OS can reclaim a regular tensor's pages under pressure but
# never a pinned one's. Confirmed empirically: resnet50 at batch_192's real
# ILP plan dual-places 37 layers, and pinning one buffer per layer (the
# original, unbounded design) locks ~7.9GB of non-reclaimable memory for the
# life of the process -- this, stacked on top of everything else already
# running on the machine, pushed system-wide memory pressure high enough
# that systemd-oomd killed an unrelated terminal and the kernel OOM killer
# killed VS Code itself, even though PRISM's own process never leaked or
# exceeded its own budget. 512MB keeps the worst case bounded to a small,
# fixed amount regardless of how many dual-placement layers a plan has; a
# model/config with only a few such layers still gets the full speed benefit
# below this ceiling, exactly as before.
_PINNED_BUFFER_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _get_pinned_buffer(cache: Dict[Any, torch.Tensor], key: Any, like: torch.Tensor) -> torch.Tensor | None:
    # Reuses a page-locked CPU buffer across calls (keyed by the CALLER'S
    # chosen key -- see _run_layer_with_dual_placement's pinned_buffer_key
    # docs for why that key must be call-site-unique, not layer_name).
    # Re-allocates if the cached buffer's shape/dtype no longer matches
    # (e.g. a different batch size): correctness over reuse in that case,
    # since silently returning a wrong-shaped buffer would break .copy_().
    buf = cache.get(key)
    if buf is not None and buf.shape == like.shape and buf.dtype == like.dtype:
        return buf

    needed_bytes = like.numel() * like.element_size()
    # Exclude this key's own (stale/wrong-shaped) entry from the current
    # total: it is about to be replaced or dropped, not kept alongside the
    # new buffer.
    current_bytes = sum(
        v.numel() * v.element_size() for k, v in cache.items() if k != key
    )
    if current_bytes + needed_bytes > _PINNED_BUFFER_CACHE_MAX_BYTES:
        # Budget would be exceeded: do not grow the pinned cache for this
        # call site. Drop any stale entry under this key so its bytes are
        # not double-counted on a later call, and tell the caller to fall
        # back to the always-safe, unpinned (transient, swappable) clone
        # path for this node instead -- same behavior as before this cache
        # existed, just for this one call site.
        cache.pop(key, None)
        return None

    buf = torch.empty(like.shape, dtype=like.dtype).pin_memory()
    cache[key] = buf
    return buf


def _current_device(x: Any) -> torch.device | None:
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, (list, tuple)):
        for v in x:
            d = _current_device(v)
            if d is not None:
                return d
    if isinstance(x, dict):
        for v in x.values():
            d = _current_device(v)
            if d is not None:
                return d
    return None


def _all_tensor_devices(x: Any) -> List[torch.device]:
    if isinstance(x, torch.Tensor):
        return [x.device]
    if isinstance(x, (list, tuple)):
        devices: List[torch.device] = []
        for value in x:
            devices.extend(_all_tensor_devices(value))
        return devices
    if isinstance(x, dict):
        devices = []
        for value in x.values():
            devices.extend(_all_tensor_devices(value))
        return devices
    return []


def _compute_loss(out: Any) -> torch.Tensor:
    loss, _, _ = compute_training_objective(out)
    return loss


def _compute_objective(
    out: Any,
    target_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str | None, float | None]:
    return compute_training_objective(out, target=target_data)


def _substitute_deferred(value: Any, marker: Any, replacement: Any) -> Any:
    if value is marker:
        return replacement
    if isinstance(value, tuple):
        return tuple(_substitute_deferred(v, marker, replacement) for v in value)
    if isinstance(value, list):
        return [_substitute_deferred(v, marker, replacement) for v in value]
    if isinstance(value, dict):
        updated = {k: _substitute_deferred(v, marker, replacement) for k, v in value.items()}
        if type(value) is not dict:
            # HuggingFace's ModelOutput (what a decoder-only model's forward
            # actually returns, e.g. CausalLMOutputWithCrossAttentions) IS a
            # dict subclass, but compute_training_objective reads it via
            # ATTRIBUTE access (out.logits, out.loss) -- rebuilding a plain
            # dict here would silently lose that, making every one of those
            # getattr() checks return None and fall through to the WRONG
            # (surrogate) loss. Reconstruct the SAME class instead; fall
            # back to the plain dict only if that class cannot be
            # rebuilt this way.
            try:
                return type(value)(**updated)
            except TypeError:
                pass
        return updated
    return value


# Target peak size of ONE chunk's fp32 logits tensor in _chunked_causal_lm_loss.
# Chunk length is derived from vocab size (not hardcoded), so this stays the
# memory cost regardless of a model's vocabulary size.
_CHUNKED_LOSS_TARGET_CHUNK_BYTES = 128 * 1024 * 1024


def _chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    target: torch.Tensor,
    strategy: str,
) -> Tuple[torch.Tensor, float | None]:
    # Fuses the vocabulary projection (hidden_states @ weight.T + bias) with
    # the causal-LM cross-entropy loss, computed in CHUNKS along the
    # flattened (batch*seq) token axis, so the full [batch, seq, vocab]
    # logits tensor -- often the single largest tensor in a decoder-only
    # LLM's whole training step -- is never materialized at once.
    # checkpoint/recompute on this projection ALONE only avoids RETAINING
    # that tensor for backward, not the cost of producing it even once --
    # confirmed empirically to still OOM here (see fix/06 README). Verified
    # empirically that this exact composition -- plain torch.utils.checkpoint
    # per chunk, no custom backward -- reproduces the unchunked computation's
    # loss and gradients bit-for-bit while cutting peak memory ~5x for
    # distilgpt2 at batch=64/seq=128: torch.utils.checkpoint composes
    # cleanly across independent chunks sharing the same weight leaf (PyTorch
    # autograd accumulates each chunk's contribution to weight.grad on its
    # own, same mechanism as any shared/tied weight) -- each chunk's
    # recomputed activations are discarded before the next chunk's begin,
    # both during the initial forward and during backward.
    vocab_size = weight.shape[0]
    chunk_size = max(1, _CHUNKED_LOSS_TARGET_CHUNK_BYTES // (vocab_size * 4))

    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = target[:, 1:].to(hidden_states.device, dtype=torch.long).contiguous()
    flat_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    num_tokens = flat_hidden.shape[0]

    def chunk_fn(
        h_chunk: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, labels_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_chunk = F.linear(h_chunk, w, b)
        loss_sum = F.cross_entropy(logits_chunk.float(), labels_chunk, ignore_index=-100, reduction="sum")
        valid = labels_chunk.ne(-100)
        correct = (logits_chunk.argmax(dim=-1) == labels_chunk).logical_and(valid).sum()
        return loss_sum, correct.to(loss_sum.dtype)

    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    total_correct = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        h_chunk = flat_hidden[start:end]
        labels_chunk = flat_labels[start:end]
        if strategy == "checkpoint":
            with _saved_tensor_cpu_offload_context():
                loss_sum, correct = chunk_fn(h_chunk, weight, bias, labels_chunk)
        else:
            loss_sum, correct = activation_checkpoint(
                chunk_fn, h_chunk, weight, bias, labels_chunk, use_reentrant=False
            )
        total_loss = total_loss + loss_sum
        total_correct = total_correct + correct

    total_valid = shift_labels.ne(-100).sum().clamp(min=1)
    loss = total_loss / total_valid
    token_accuracy = float((total_correct / total_valid).item())
    return loss, token_accuracy


def _compute_objective_possibly_deferred(
    out: Any,
    target_data: torch.Tensor | None = None,
    runtime_features: Dict[str, Any] | None = None,
) -> tuple[torch.Tensor, str | None, float | None]:
    # See _DeferredCheckpointSegment's docstring: when the graph's final
    # layer was deferred (its own output never materialized), `out` holds
    # that marker instead of a real tensor. Fusing the deferred segment's
    # own computation TOGETHER with the loss inside one outer
    # checkpoint/recompute boundary is what actually avoids retaining that
    # layer's (often huge, e.g. a vocabulary projection's) output for the
    # whole forward-to-backward gap -- checkpointing the layer ALONE does
    # not (confirmed empirically, see fix/06 README). metric_name/
    # metric_value are plain Python values (never need gradients), so they
    # are captured via an ordinary closure side effect during the ONE
    # forward-time call torch.utils.checkpoint/the offload context always
    # performs, rather than by recomputing them a second time afterward.
    deferred = _find_deferred_segment(out)
    if deferred is None:
        return _compute_objective(out, target_data=target_data)

    if deferred.linear_args is not None and target_data is not None:
        hidden_states, weight, bias = deferred.linear_args
        # Same shape match compute_training_objective's own causal-LM
        # branch requires (3D logits, 2D integer target, matching
        # batch/seq) -- structural, not name-based, so this activates for
        # any decoder-only model's vocabulary-projection head. Anything
        # that doesn't match (e.g. a pooled classification head, whose
        # output is already small) falls through to the generic path
        # below unchanged.
        if (
            target_data.ndim == 2
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == target_data.shape[0]
            and hidden_states.shape[1] == target_data.shape[1]
        ):
            loss, token_accuracy = _chunked_causal_lm_loss(
                hidden_states, weight, bias, target_data, deferred.strategy
            )
            if runtime_features is not None:
                key = "checkpoint_layers" if deferred.strategy == "checkpoint" else "recompute_layers"
                count_key = "checkpoint_count" if deferred.strategy == "checkpoint" else "recompute_count"
                runtime_features[key].add(deferred.layer_name)
                runtime_features[count_key] += 1
            return loss, "token_accuracy", token_accuracy

    metrics_box: Dict[str, Any] = {}

    def combined(*values: torch.Tensor) -> torch.Tensor:
        segment_out = deferred.segment_fn(*values)
        reconstructed = _substitute_deferred(out, deferred, segment_out)
        loss, metric_name, metric_value = _compute_objective(reconstructed, target_data=target_data)
        metrics_box["metric_name"] = metric_name
        metrics_box["metric_value"] = metric_value
        return loss

    if deferred.strategy == "checkpoint":
        with _saved_tensor_cpu_offload_context():
            loss = combined(*deferred.external_values)
    else:
        loss = activation_checkpoint(combined, *deferred.external_values, use_reentrant=False)

    if runtime_features is not None:
        key = "checkpoint_layers" if deferred.strategy == "checkpoint" else "recompute_layers"
        count_key = "checkpoint_count" if deferred.strategy == "checkpoint" else "recompute_count"
        runtime_features[key].add(deferred.layer_name)
        runtime_features[count_key] += 1

    return loss, metrics_box.get("metric_name"), metrics_box.get("metric_value")


def _activation_strategy_name(strategy: Any) -> str:
    if strategy is None:
        return "retain"
    if isinstance(strategy, str):
        return strategy.lower()
    if getattr(strategy, "recompute", False):
        return "recompute"
    if getattr(strategy, "checkpoint", False):
        return "checkpoint"
    if getattr(strategy, "retain", False):
        return "retain"
    return "retain"


def _block_id_for_layer_name(layer_name: str) -> str:
    # Identifies which repeated block (nn.ModuleList-style, e.g.
    # "transformer.h.{i}" for GPT-2, "model.layers.{i}" for LLaMA-style
    # models) a leaf layer belongs to, by truncating its name at the last
    # purely-numeric path segment. Generic across decoder-only
    # architectures: it relies only on the universal nn.ModuleList naming
    # convention (an integer index segment), never a hardcoded module name.
    # A layer with no numeric segment (e.g. "lm_head") is its own block.
    #
    # This exists so _detect_checkpoint_groups' grouping -- which merges
    # consecutive same-strategy leaves across their own leaf-name boundary
    # (see that method's docstring) -- stops at block boundaries instead of
    # also merging DIFFERENT blocks that happen to share the same strategy.
    # Confirmed necessary by a real OOM: merging all 6 of distilgpt2's
    # transformer blocks into one recompute group made backward recompute
    # all 6 blocks' activations at once, as expensive as no recompute at
    # all -- checkpointing each block as its OWN unit is what actually
    # bounds peak memory to one block's worth.
    parts = layer_name.split(".")
    last_numeric_idx = None
    for i, part in enumerate(parts):
        if part.isdigit():
            last_numeric_idx = i
    if last_numeric_idx is None:
        return layer_name
    return ".".join(parts[: last_numeric_idx + 1])


@dataclass
class _DeferredCheckpointSegment:
    """Returned (instead of a real tensor) by _DeviceAwareFXInterpreter.run()
    when the graph's final output comes directly from a single
    checkpoint/recompute-marked node whose only consumer is the graph's own
    `output` node (see _detect_checkpoint_groups' final-node detection,
    below). Checkpointing/recomputing that node ALONE does not reduce real
    memory when its own output is immediately needed by the loss computation
    that runs right after the graph returns (confirmed empirically: a large
    vocabulary projection like `lm_head` showed ~0% saving checkpointed in
    isolation, vs ~70%+ when checkpointed/recomputed TOGETHER WITH the loss
    that consumes it -- see fix/06 README). `segment_fn`/`external_values`
    let the caller (_run_hybrid_training_dag / prism_loop.py) replay exactly
    this node's own computation itself, fused with the loss, inside ONE
    outer checkpoint/recompute boundary, instead of materializing this
    node's output for the whole forward-to-backward gap.
    """

    strategy: str
    segment_fn: Callable[..., torch.Tensor]
    external_values: Tuple[torch.Tensor, ...]
    layer_name: str
    # Populated only when this node is a plain aten.linear.default (the
    # universal shape of a decoder-only LLM's vocabulary-projection head,
    # e.g. "lm_head" -- generic: detected structurally, never by name).
    # When set, _compute_objective_possibly_deferred can bypass segment_fn
    # entirely and fuse this linear projection WITH the causal-LM loss in
    # CHUNKS along the sequence, so the full [batch, seq, vocab] logits
    # tensor -- often the single largest tensor in the whole training step
    # -- is never materialized at once. checkpoint/recompute on segment_fn
    # alone only avoids RETAINING that tensor, not the cost of producing it
    # once (confirmed empirically: still OOMs here even when checkpointed --
    # see fix/06 README); chunking is what actually bounds that peak.
    linear_args: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] | None = None


def _find_deferred_segment(value: Any) -> "_DeferredCheckpointSegment | None":
    if isinstance(value, _DeferredCheckpointSegment):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_deferred_segment(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _find_deferred_segment(item)
            if found is not None:
                return found
    return None


def _saved_tensor_cpu_offload_context() -> Any:
    def pack_hook(tensor: torch.Tensor) -> Tuple[str, torch.Tensor]:
        return (str(tensor.device), tensor.detach().to("cpu"))

    def unpack_hook(packed: Tuple[str, torch.Tensor]) -> torch.Tensor:
        device_label, tensor = packed
        return tensor.to(torch.device(device_label))

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)


def _run_layer_with_dual_placement(
    layer: nn.Module,
    layer_name: str,
    input_tensor: torch.Tensor,
    forward_device: torch.device,
    backward_device: torch.device,
    output_clone_safe: bool = False,
    pinned_buffer_cache: Dict[Any, torch.Tensor] | None = None,
    pinned_buffer_key: Any = None,
) -> Any:
    # pinned_buffer_cache/_key: when output_clone_safe is False (the output
    # snapshot must still be cloned), route that clone into a cached,
    # page-locked CPU buffer instead of a fresh regular one, reused across
    # training steps -- measured ~30% faster end-to-end (copy once pinning
    # is amortized, plus the later CPU->GPU compare-move benefits from
    # pinned-memory DMA). pinned_buffer_key MUST uniquely identify this
    # exact call site within a single forward pass, NOT just `layer_name`:
    # a module reused at multiple graph nodes (e.g. torchvision Bottleneck's
    # single self.relu, called 3x with 3 distinct FX nodes that all report
    # the SAME layer_name here, since node_layer_names is only ever
    # populated for torch.export-traced decoder models, never for
    # fx.symbolic_trace-traced vision models) would otherwise have its
    # SECOND call overwrite the cached buffer before the FIRST call's
    # backward() reads it -- silent data corruption, not a crash. Callers
    # pass the FX node's own `.name` (unique per node by construction) or a
    # sequential-loop index, never `layer_name` itself, as this key.
    # output_clone_safe: True only when the CALLER has verified, from the
    # execution graph, that NONE of this node's direct downstream consumers
    # can mutate this layer's output in place (see
    # _DeviceAwareFXInterpreter._node_output_clone_safe for the FX-graph
    # case, and the layer_order[idx+1] check in run_hybrid_training's linear
    # fallback loop). Defaults to False (always clone) so any call site that
    # does not explicitly check this keeps the original, always-safe
    # behavior.
    def _detach_to_cpu(value: Any, clone: bool = True) -> Any:
        if torch.is_tensor(value):
            if not clone:
                # Caller has verified no downstream consumer can mutate this
                # tensor's storage in place -- .detach() alone is safe and,
                # measured on this project's own resnet50 dual-placed
                # layers, ~9x+ cheaper than .clone() for a 300MB activation
                # (the clone is a real CPU memcpy; detach is a metadata-only
                # view).
                return value.detach().to("cpu")
            # .clone() is required here: when `value` is already on CPU
            # (e.g. forward_device="CPU"), .to("cpu") is a no-op that
            # returns the SAME storage, not a copy. Without cloning, this
            # "saved" snapshot silently aliases the live tensor, and a
            # downstream in-place op (e.g. the ReLU(inplace=True) that
            # follows this layer in a torchvision Bottleneck block) mutates
            # it before the forward-vs-recompute comparison runs in
            # backward() — producing a spurious mismatch against a value
            # that was never actually this layer's own output.
            return value.detach().clone().to("cpu")
        if isinstance(value, tuple):
            return tuple(_detach_to_cpu(v, clone) for v in value)
        if isinstance(value, list):
            return [_detach_to_cpu(v, clone) for v in value]
        if isinstance(value, dict):
            return {k: _detach_to_cpu(v, clone) for k, v in value.items()}
        return value

    def _compare_outputs(expected: Any, observed: Any, device: torch.device) -> Tuple[bool, float, str]:
        if torch.is_tensor(expected) and torch.is_tensor(observed):
            if expected.shape != observed.shape:
                return False, float("inf"), f"shape mismatch: expected {tuple(expected.shape)} vs observed {tuple(observed.shape)}"
            # Move `expected` (the forward snapshot, always CPU-resident --
            # see ctx.forward_output_cpu below) to `device` (the backward
            # device, where `observed`/recompute_output already lives)
            # instead of moving `observed` to CPU: `device` is normally the
            # GPU, where an elementwise diff + reduction over a full
            # activation tensor is far cheaper than on CPU. Also computes
            # the tolerance check directly instead of a second full pass via
            # torch.allclose -- same formula (allclose's own semantics:
            # |input-other| <= atol + rtol*|other|, other=observed_f here,
            # matching the original torch.allclose(expected_f, observed_f,
            # ...) call this replaces), same atol/rtol, same verdict, just
            # computed once. Measured: 526ms -> 57ms (~9x) for a 300MB
            # resnet50 activation tensor on this project's own hardware.
            expected_f = expected.to(device=device, dtype=torch.float32, non_blocking=True)
            observed_f = observed.to(torch.float32)
            diff = (expected_f - observed_f).abs()
            if diff.numel() == 0:
                return True, 0.0, ""
            # CPU (oneDNN) and GPU (cuDNN) do not produce numerically identical
            # conv/batchnorm results in FP32 even for identical weights and
            # inputs (different reduction algorithms), and the divergence
            # compounds across a real network: an isolated single-layer
            # CPU-vs-GPU conv forward already differs by ~4e-3 (for unit-scale
            # output), and measured end-to-end on resnet50 with real
            # pretrained weights the divergence at a mid-network dual-placed
            # layer reaches ~0.09-0.13. atol=0.15/rtol=0.05 (~0.2 threshold at
            # unit scale) keeps ~50% margin above the observed range instead
            # of assuming near-exact cross-backend equality.
            atol, rtol = 0.15, 0.05
            within_tol = bool((diff <= (atol + rtol * observed_f.abs())).all().item())
            max_diff = float(diff.max().item())
            return within_tol, max_diff, "tensor mismatch" if not within_tol else ""

        if type(expected) is not type(observed):
            return False, float("inf"), f"type mismatch: expected {type(expected).__name__} vs observed {type(observed).__name__}"

        if isinstance(expected, tuple):
            if len(expected) != len(observed):
                return False, float("inf"), f"tuple length mismatch: expected {len(expected)} vs observed {len(observed)}"
            max_diff = 0.0
            for exp_item, obs_item in zip(expected, observed):
                ok, item_diff, reason = _compare_outputs(exp_item, obs_item, device)
                max_diff = max(max_diff, item_diff)
                if not ok:
                    return False, max_diff, reason
            return True, max_diff, ""

        if isinstance(expected, list):
            if len(expected) != len(observed):
                return False, float("inf"), f"list length mismatch: expected {len(expected)} vs observed {len(observed)}"
            max_diff = 0.0
            for exp_item, obs_item in zip(expected, observed):
                ok, item_diff, reason = _compare_outputs(exp_item, obs_item, device)
                max_diff = max(max_diff, item_diff)
                if not ok:
                    return False, max_diff, reason
            return True, max_diff, ""

        if isinstance(expected, dict):
            if expected.keys() != observed.keys():
                return False, float("inf"), "dict key mismatch"
            max_diff = 0.0
            for key in expected:
                ok, item_diff, reason = _compare_outputs(expected[key], observed[key], device)
                max_diff = max(max_diff, item_diff)
                if not ok:
                    return False, max_diff, reason
            return True, max_diff, ""

        return True, 0.0, ""

    # True only if THIS layer itself can mutate its own input in place (e.g.
    # ReLU(inplace=True)) -- same detection training_profiler.py's
    # _disable_inplace_modules_for_backward_hooks already uses elsewhere in
    # this project. This only answers "can calling layer(x) mutate x" -- a
    # DIFFERENT question from output_clone_safe (the caller-supplied
    # parameter above), which is about whether a DOWNSTREAM layer can
    # mutate THIS layer's output after it's returned. The two are
    # independent checks for independent risks.
    _layer_is_inplace = bool(getattr(layer, "inplace", False))

    class _DualPlacementFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            # Snapshot the input BEFORE calling the layer. If forward_device
            # == x.device, `x.to(forward_device)` below is a no-op that
            # aliases x's storage, and if `layer` is in-place, calling it
            # would mutate that shared storage -- corrupting a snapshot
            # taken afterwards. .clone() is required ONLY in that case;
            # when `layer` cannot mutate its own input (the common case --
            # Conv2d/BatchNorm2d/Linear/pooling all create new output
            # tensors), .detach() alone is safe and ~34ms cheaper per call
            # on a 300MB activation (measured on this project's own
            # resnet50 dual-placed layers).
            saved_input_cpu = (
                x.detach().clone().to("cpu") if _layer_is_inplace else x.detach().to("cpu")
            )

            layer.to(forward_device)
            x_forward = x.to(forward_device)
            output = layer(x_forward)

            ctx.layer = layer
            ctx.layer_name = layer_name
            ctx.forward_device = forward_device
            ctx.backward_device = backward_device
            ctx.input_device = x.device
            ctx.save_for_backward(saved_input_cpu)
            if output_clone_safe:
                ctx.forward_output_cpu = _detach_to_cpu(output, clone=False)
            elif pinned_buffer_cache is not None and torch.is_tensor(output):
                # A clone is unavoidable here (output_clone_safe is False),
                # so route it into a reused, page-locked buffer instead of a
                # fresh regular one: the copy itself is no more expensive,
                # but the later CPU->GPU move in _compare_outputs becomes a
                # real DMA transfer instead of a slower pageable-memory one.
                # Measured end-to-end (copy + later transfer, steady state
                # once the one-time pin cost is amortized across training
                # steps): ~30% faster than the always-fresh-clone path.
                # _get_pinned_buffer returns None once the global pinned-
                # memory budget (_PINNED_BUFFER_CACHE_MAX_BYTES) is full --
                # fall back to the always-safe fresh clone for this node
                # instead of growing non-reclaimable memory without bound.
                _buf = _get_pinned_buffer(pinned_buffer_cache, pinned_buffer_key, output)
                if _buf is not None:
                    _buf.copy_(output.detach())
                    ctx.forward_output_cpu = _buf
                else:
                    ctx.forward_output_cpu = _detach_to_cpu(output, clone=True)
            else:
                ctx.forward_output_cpu = _detach_to_cpu(output, clone=True)
            # Safety tripwire for output_clone_safe: if the caller's
            # graph-level analysis was wrong about no downstream consumer
            # being able to mutate this tensor in place, that mutation will
            # bump the shared storage's autograd version counter. Comparing
            # it again in backward() turns a silent, potentially-corrupted
            # comparison into a loud, specific error instead -- this is a
            # near-zero-cost integer check (no tensor op), not a resync.
            ctx.output_clone_safe = output_clone_safe
            ctx.saved_output_version = (
                ctx.forward_output_cpu._version
                if output_clone_safe and torch.is_tensor(ctx.forward_output_cpu)
                else None
            )
            return output

        @staticmethod
        def backward(ctx, *grad_outputs: Any) -> Any:
            grad_output = grad_outputs[0]
            (saved_input_cpu,) = ctx.saved_tensors
            layer_local = ctx.layer
            layer_local.to(ctx.backward_device)

            recompute_input = saved_input_cpu.to(ctx.backward_device).detach().requires_grad_(True)
            params = tuple(param for param in layer_local.parameters() if param.requires_grad)

            with torch.enable_grad():
                # .clone() gives in-place layers (e.g. ReLU(inplace=True), used
                # throughout torchvision's ResNet) a non-leaf buffer to modify.
                # Autograd forbids an in-place op on a leaf Variable that
                # requires grad, and recompute_input above is exactly that;
                # this is the same trick torch.utils.checkpoint uses.
                recompute_output = layer_local(recompute_input.clone())

            if ctx.saved_output_version is not None and torch.is_tensor(ctx.forward_output_cpu):
                if ctx.forward_output_cpu._version != ctx.saved_output_version:
                    raise RuntimeError(
                        f"Internal error: output_clone_safe was True for layer={ctx.layer_name} but its "
                        f"saved output snapshot was mutated in place before backward() read it. This means "
                        f"the caller's graph-level in-place analysis missed a real downstream consumer -- "
                        f"fix the analysis (see _node_output_clone_safe), not this check."
                    )

            outputs_match, max_diff, reason = _compare_outputs(ctx.forward_output_cpu, recompute_output, ctx.backward_device)
            if not outputs_match:
                raise RuntimeError(
                    f"Dual-device recompute mismatch for layer={ctx.layer_name} (max_abs_diff={max_diff:.6e}): {reason}"
                )

            grad_output_local = grad_output.to(ctx.backward_device)
            grads = torch.autograd.grad(
                recompute_output,
                (recompute_input,) + params,
                grad_outputs=grad_output_local,
                allow_unused=True,
            )

            grad_input = grads[0]
            for param, grad in zip(params, grads[1:]):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad.detach().clone()
                else:
                    # In-place accumulation avoids creating an ephemeral full-size
                    # tensor copy that would spike peak memory during backward.
                    param.grad.add_(grad.detach())

            if grad_input is None:
                grad_input = torch.zeros_like(saved_input_cpu, device=ctx.input_device)
            else:
                grad_input = grad_input.to(ctx.input_device)
            return (grad_input,)

    return _DualPlacementFn.apply(input_tensor)


_STOCHASTIC_OPERATOR_NAMES = frozenset(
    {
        "aten.dropout.default",
        "aten.native_dropout.default",
        "aten.dropout_.default",
        "aten.feature_dropout.default",
    }
)


def _is_stochastic_operator(target: Any) -> bool:
    """True for operators whose output depends on fresh randomness each call
    (e.g. Dropout's mask). Forward-vs-recompute dual placement fundamentally
    cannot apply to these: recomputing during backward draws a NEW random
    mask, so the two outputs legitimately differ -- not backend noise, two
    different random events. Confirmed in practice: a real dual-placed
    dropout node produced a ~213 max-abs-diff mismatch (vs ~0.1-0.15 for
    genuine cross-device numerical noise on deterministic ops), i.e. two to
    three orders of magnitude larger, consistent with "wrong mask" rather
    than "backend rounding." These operators are cheap (no parameters, tiny
    compute) and were never the source of the latency gap this mechanism
    exists to close, so they simply keep today's single-device fallback
    rather than attempting (and necessarily failing) dual placement.
    """
    return str(target) in _STOCHASTIC_OPERATOR_NAMES


def _run_operator_with_dual_placement(
    target: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    forward_device: torch.device,
    backward_device: torch.device,
    layer_name: str,
) -> Any:
    """Op-level analog of `_run_layer_with_dual_placement` for `call_function`
    nodes in torch.export-traced (decoder-only) graphs. There is no `nn.Module`
    to call twice here -- `torch.export` decomposes each original layer into raw
    ATen op calls (`aten.linear.default`, `aten.layer_norm.default`, ...) with
    weights arriving as plain tensors via `get_attr` nodes, not `module.parameters()`.
    Only used for nodes whose output is a single Tensor -- every `call_function`
    node observed in practice (linear, layer_norm, dropout, embedding) satisfies
    this; an unexpected multi-output op raises rather than silently falling back.

    Unlike `_run_layer_with_dual_placement` (which accumulates gradients manually
    onto `param.grad`), every differentiable tensor argument is threaded through
    this `autograd.Function`'s own forward/backward signature, and gradients are
    RETURNED rather than written to `.grad` directly. This matters because GPT-2's
    `lm_head.weight` IS `transformer.wte.weight` (tied embeddings): returning the
    gradient lets PyTorch's native leaf-accumulation combine both usages'
    contributions correctly. Manually touching `.grad` here (the module-level
    approach) would race with or double-count the embedding node's own, separate,
    unwrapped backward contribution to that same shared leaf.
    """
    slots: List[Tuple[str, Any, Any]] = [("pos", i, v) for i, v in enumerate(args)]
    slots += [("kw", k, v) for k, v in kwargs.items()]

    diff_slot_indices = [i for i, (_, _, v) in enumerate(slots) if torch.is_tensor(v) and v.requires_grad]
    diff_slot_index_set = set(diff_slot_indices)
    diff_tensors = tuple(slots[i][2] for i in diff_slot_indices)

    # `_rebuild` below is referenced by both `forward` and `backward` of a
    # FRESH `torch.autograd.Function` subclass created on every single call
    # to this function (needed since `target`/`forward_device`/
    # `backward_device` differ per node) -- its closure therefore lives as
    # long as that class's grad_fn does. Closing over `slots` directly (as
    # an earlier version of this function did) keeps every differentiable
    # argument's ORIGINAL tensor (the layer's real input activation,
    # frequently the single largest tensor in the step) referenced for
    # that same lifetime. Confirmed empirically this is NOT just "delayed
    # until the next GC pass": with gc.disable() AND gc.collect() both
    # unable to free these tensors across repeated steps, measured +72MB
    # per step (one dual-placed leaf's own activation, never released) on
    # real distilgpt2 training -- a reference cycle involving the C++
    # autograd engine's own ctx/grad_fn bookkeeping, which Python's cyclic
    # GC cannot see into and therefore cannot break (confirmed by
    # isolating the exact closure-over-slots pattern in a minimal
    # reproduction -- see fix/06 README). Keeping only the STRUCTURAL
    # skeleton (which slot is which kind/key, never a value) plus
    # non-differentiable VALUES (typically small/constant, e.g. a fixed
    # attention mask) in the closure -- and always receiving the
    # differentiable tensors fresh, as `_rebuild`'s own argument, never
    # from the closure -- avoids ever capturing the one tensor that
    # actually grows unbounded.
    skeleton: List[Tuple[str, Any]] = [(kind, key) for kind, key, _ in slots]
    non_diff_values: Dict[int, Any] = {i: v for i, (_, _, v) in enumerate(slots) if i not in diff_slot_index_set}

    def _rebuild(tensor_values: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        diff_map = dict(zip(diff_slot_indices, tensor_values))
        merged: List[Tuple[str, Any, Any]] = []
        for idx, (kind, key) in enumerate(skeleton):
            if idx in diff_map:
                merged.append((kind, key, diff_map[idx]))
            else:
                # Non-differentiable tensor args (e.g. a constant attention
                # mask) still need to follow the active device per call;
                # non-tensor args (shape lists, floats, bools) pass through
                # unchanged.
                value = non_diff_values[idx]
                merged.append((kind, key, value.to(device) if torch.is_tensor(value) else value))
        new_args = tuple(v for kind, _, v in merged if kind == "pos")
        new_kwargs = {key: v for kind, key, v in merged if kind == "kw"}
        return new_args, new_kwargs

    def _compare_tensors(expected: torch.Tensor, observed: torch.Tensor) -> Tuple[bool, float, str]:
        if expected.shape != observed.shape:
            return False, float("inf"), f"shape mismatch: expected {tuple(expected.shape)} vs observed {tuple(observed.shape)}"
        expected_f = expected.to(torch.float32)
        observed_f = observed.to(torch.float32)
        diff = (expected_f - observed_f).abs()
        max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        # Same tolerance and rationale as _run_layer_with_dual_placement's
        # _compare_outputs: CPU/GPU backends do not produce numerically
        # identical FP32 results even for identical weights/inputs.
        allclose = bool(torch.allclose(expected_f, observed_f, rtol=0.05, atol=0.15))
        return allclose, max_diff, "tensor mismatch" if not allclose else ""

    if not diff_tensors:
        # No differentiable tensor argument at all -- nothing for a backward
        # device placement to apply to; just run forward on forward_device.
        moved_args, moved_kwargs = _rebuild((), forward_device)
        return target(*moved_args, **moved_kwargs)

    class _DualPlacementOpFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *diff_tensor_args: torch.Tensor) -> torch.Tensor:
            ctx.original_devices = [t.device for t in diff_tensor_args]
            forward_tensors = tuple(t.to(forward_device) for t in diff_tensor_args)
            # Clone before use: same aliasing discipline as
            # _run_layer_with_dual_placement (bug 3), even though torch.export's
            # functionalized graph means none of these ops are in-place in
            # practice (confirmed: 0 nodes with an in-place op suffix in the
            # traced distilgpt2 graph) -- cloned defensively, not reactively.
            saved_inputs_cpu = tuple(t.detach().clone().to("cpu") for t in forward_tensors)

            full_args, full_kwargs = _rebuild(forward_tensors, forward_device)
            output = target(*full_args, **full_kwargs)
            if not torch.is_tensor(output):
                raise HybridExecutionUnsupportedError(
                    f"Operator-level dual placement only supports single-Tensor outputs; "
                    f"layer={layer_name} ({target}) returned {type(output).__name__}"
                )

            # PyTorch's own docs for custom autograd.Function warn against
            # `ctx.x = tensor` for any tensor: it creates a reference cycle
            # between the tensor and ctx that "is more likely to leak
            # memory" -- confirmed exactly this, the hard way: a fresh
            # _DualPlacementOpFn class is built per call (needed since
            # `target`/`forward_device`/`backward_device` differ per node),
            # and `ctx.forward_output_cpu = ...` kept every such node's
            # saved output tensor alive indefinitely across training steps
            # (measured: +72MB/step, 3 dual-placed leaves x one
            # [64,128,768] tensor each, never freed even by gc.collect() --
            # the cycle runs through the C++ autograd engine's own ctx/
            # grad_fn bookkeeping, which Python's cyclic GC cannot see into
            # and therefore cannot break). save_for_backward is the
            # mechanism PyTorch provides specifically to avoid this: it
            # accepts any number of tensors, so the forward output joins
            # the already-correctly-saved inputs instead of a plain
            # attribute. See fix/06 README for the full investigation.
            ctx.save_for_backward(*saved_inputs_cpu, output.detach().clone().to("cpu"))
            return output

        @staticmethod
        def backward(ctx, *grad_outputs: Any) -> Any:
            grad_output = grad_outputs[0]
            *saved_inputs_cpu, forward_output_cpu = ctx.saved_tensors
            recompute_tensors = tuple(
                t.to(backward_device).detach().requires_grad_(True) for t in saved_inputs_cpu
            )

            with torch.enable_grad():
                full_args, full_kwargs = _rebuild(recompute_tensors, backward_device)
                recompute_output = target(*full_args, **full_kwargs)

            recompute_output_cpu = recompute_output.detach().clone().to("cpu")
            outputs_match, max_diff, reason = _compare_tensors(forward_output_cpu, recompute_output_cpu)
            if not outputs_match:
                raise RuntimeError(
                    f"Dual-device recompute mismatch for operator layer={layer_name} ({target}) "
                    f"(max_abs_diff={max_diff:.6e}): {reason}"
                )

            grad_output_local = grad_output.to(backward_device)
            grads = torch.autograd.grad(
                recompute_output,
                recompute_tensors,
                grad_outputs=grad_output_local,
                allow_unused=True,
            )

            # Critical: restore each gradient to the ORIGINAL device of its
            # input tensor before returning. PyTorch does not do this
            # automatically -- a gradient returned on the wrong device raises
            # immediately (confirmed via a minimal reproduction during design
            # review, before this safeguard was added).
            restored_grads = [
                None if grad is None else grad.to(original_device)
                for grad, original_device in zip(grads, ctx.original_devices)
            ]
            return tuple(restored_grads)

    return _DualPlacementOpFn.apply(*diff_tensors)


def _ordered_sequential_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        return [(f"net.{idx}", mod) for idx, mod in enumerate(model.net)]

    leaves: List[Tuple[str, nn.Module]] = []
    for name, module in iter_leaf_modules(model):
        if name:
            leaves.append((name, module))
    if not leaves:
        raise ValueError("Model has no leaf modules to execute")
    return leaves


def _try_symbolic_trace(model: nn.Module) -> Optional[fx.GraphModule]:
    was_training = model.training
    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None) if config is not None else None
    try:
        return fx.symbolic_trace(model)
    except Exception:
        return None
    finally:
        model.train(was_training)
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache


def _prepare_fx_run_args(graph_module: fx.GraphModule, input_data: Any) -> Tuple[Any, ...]:
    if isinstance(input_data, tuple):
        return input_data
    if not isinstance(input_data, dict):
        return (input_data,)

    placeholder_nodes = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    args: List[Any] = []
    missing: List[str] = []
    for node in placeholder_nodes:
        if node.name in input_data:
            args.append(input_data[node.name])
            continue

        default_value = node.args[0] if node.args else inspect._empty
        if default_value is inspect._empty:
            missing.append(node.name)
            continue
        args.append(default_value)

    if missing:
        raise HybridExecutionUnsupportedError(
            f"Structured FX input is missing placeholder values required by the traced graph: {missing}"
        )
    return tuple(args)


class _DeviceAwareFXInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        plan: DevicePlan,
        gpu_id: int,
        transfer_metrics: Dict[str, float],
        activation_strategies: Dict[str, Any] | None,
        warnings: List[str],
        runtime_features: Dict[str, Any],
        enable_async_transfer: bool,
        enable_prefetch: bool,
        transfer_stream: torch.cuda.Stream | None,
        node_layer_names: Dict[str, str] | None = None,
        supports_activation_strategies: bool = True,
        pinned_output_buffer_cache: Dict[str, torch.Tensor] | None = None,
    ):
        super().__init__(module)
        self._plan = plan
        self._gpu_id = gpu_id
        self._transfer_metrics = transfer_metrics
        self._activation_strategies = activation_strategies or {}
        self._warnings = warnings
        self._runtime_features = runtime_features
        self._enable_async_transfer = enable_async_transfer
        self._enable_prefetch = enable_prefetch
        self._transfer_stream = transfer_stream
        self._node_layer_names = node_layer_names or {}
        self._supports_activation_strategies = supports_activation_strategies
        self._prefetched_inputs: Dict[str, torch.Tensor] = {}
        self._active_node: fx.Node | None = None
        self._attr_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._warned_messages: set[str] = set()
        self._output_clone_safe_cache: Dict[str, bool] = {}
        # Keyed by FX node NAME (unique per node by construction), never by
        # layer_name -- see _run_layer_with_dual_placement's
        # pinned_buffer_key docs for why layer_name can collide across
        # distinct call sites of the same reused module.
        #
        # MUST be created ONCE by the caller and passed in here, NOT
        # created fresh in this constructor: _run_hybrid_training_dag
        # instantiates a new _DeviceAwareFXInterpreter every training step,
        # so a cache owned by this instance would silently never be reused
        # across steps. That alone wouldn't leak (the old instance should
        # be garbage-collected) -- but PyTorch's pinned-memory allocator
        # pools page-locked host allocations internally and does not
        # release them back to the OS promptly even after the Python
        # tensor is freed, so a fresh set of pinned buffers allocated every
        # step accumulates for real. Confirmed the hard way: this exact
        # per-step-fresh-cache bug OOM-killed a real run (`journalctl -k`
        # showed the kernel's own oom-killer taking the process out, not a
        # crash) after RSS grew past 10GB over a handful of steps. Default
        # None (fresh, single-use dict) exists only so direct unit-test
        # construction of this class doesn't need to care about this --
        # any REAL training loop must pass a cache created before its step
        # loop starts.
        self._pinned_output_buffer_cache: Dict[str, torch.Tensor] = (
            pinned_output_buffer_cache if pinned_output_buffer_cache is not None else {}
        )
        # Populated by _detect_checkpoint_groups (call_function nodes,
        # decoder-only graphs) and _detect_checkpoint_groups_call_module
        # (call_module nodes, vision graphs) -- see
        # fix/07_gpu_recompute_joint_decision/README.md for why single-node
        # checkpoint/recompute measurably does NOT save memory for either
        # graph kind (confirmed empirically to make it WORSE for an
        # isolated call_module, +36% in a controlled test) and must be
        # applied to a whole block's worth of nodes together instead.
        self._checkpoint_group_start: Dict[str, Tuple[str, List[fx.Node]]] = {}
        self._checkpoint_group_member: set[str] = set()
        self._checkpoint_group_results: Dict[str, Any] = {}
        # See _detect_checkpoint_groups' final-node detection: the one node
        # (if any) whose only consumer is the graph's own `output` node AND
        # that is marked checkpoint/recompute. Deferred rather than executed
        # eagerly so the DAG runner can fuse it with the loss computation --
        # see _DeferredCheckpointSegment's own docstring for why that fusion
        # (not checkpointing this node alone) is what actually saves memory.
        self._deferred_final_node: fx.Node | None = None
        self._deferred_final_strategy: str | None = None
        self._shared_call_module_layer_names: set = self._compute_shared_call_module_layer_names()
        self._detect_checkpoint_groups()
        self._detect_checkpoint_groups_call_module()

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        self._warnings.append(message)

    def _detect_checkpoint_groups(self) -> None:
        # Phase 4's checkpoint/recompute strategies only ever reduce real
        # memory when applied to a whole named layer's worth of computation
        # (e.g. one full transformer block, or a final projection fused with
        # the loss that consumes it) -- never to one individual decomposed
        # ATen op in isolation. call_module already gets this right for free
        # (one nn.Module call IS one whole layer's computation). For a
        # decoder-only torch.export graph, the SAME named layer (e.g.
        # "transformer.h.5") is decomposed into many individual
        # call_function nodes; checkpointing just one of them barely touches
        # its own tiny saved-for-backward tensors and leaves everything else
        # retained -- confirmed empirically (measured ~0% saving checkpointing
        # a single linear op in isolation, vs ~70-74% checkpointing/
        # recomputing a whole block's worth of ops together). This scan
        # groups consecutive call_function nodes that share the same
        # layer_name AND a checkpoint/recompute strategy into one segment,
        # so _run_checkpoint_group (below) can execute and
        # checkpoint/recompute the WHOLE group as a single unit.
        #
        # Deliberately only ever considers call_function nodes: call_method
        # dispatch (structurally unsupported, see its own docstring) is
        # untouched by this method. call_module dispatch (vision models)
        # has its OWN, separate grouping pass --
        # _detect_checkpoint_groups_call_module, called right after this
        # method -- rather than being handled here, because call_module
        # graphs never populate self._node_layer_names the way decoder-only
        # graphs do (see _layer_name_for_node's call_module fallback), so
        # the node-scanning details differ enough to warrant a separate,
        # additive method instead of branching this one. NOTE: a single
        # call_module's checkpoint/recompute is NOT already "correct,
        # whole-module-granularity" savings on its own, despite what an
        # earlier version of this comment claimed -- confirmed empirically
        # (fix/07_gpu_recompute_joint_decision/README.md) that checkpointing
        # one isolated nn.Module measurably makes memory usage WORSE, not
        # better, for exactly the same reason a single decoder-only
        # call_function op does: the module's only expensive tensor is its
        # OUTPUT, which checkpointing does not free (the next layer still
        # needs it) -- only a whole block's worth of modules together
        # frees anything real.
        if self._supports_activation_strategies:
            return

        # Grouping key is the STRATEGY, not the individual layer_name: a
        # decoder-only graph's node_layer_names only ever names LEAF
        # submodules (confirmed empirically -- a GPT2Block container like
        # "transformer.h.5" itself is never a value in node_layer_names,
        # only its individual children are, e.g. "transformer.h.5.ln_1",
        # "transformer.h.5.attn.c_attn", "transformer.h.5.mlp.c_proj", ...).
        # Phase 4 marking a whole block for relief therefore marks SEVERAL
        # consecutive, DIFFERENTLY-named leaf layers with the same strategy
        # -- grouping must merge across that leaf-name boundary, not require
        # an exact layer_name match, or it would never group more than one
        # leaf's own (tiny) handful of ops at a time, which measured ~0%
        # saving (see this method's own module-level docstring note above).
        current_group: List[fx.Node] = []
        current_strategy: str | None = None
        current_block_id: str | None = None

        def flush() -> None:
            nonlocal current_group, current_strategy, current_block_id
            if len(current_group) > 1 and current_strategy in ("checkpoint", "recompute"):
                group_set = set(current_group)
                # Safety requirement: no node INSIDE the group may be
                # consumed by anything OUTSIDE it, except the group's own
                # last node. This lets _run_checkpoint_group only ever need
                # the group's final value (nothing else is ever looked up
                # from outside), which avoids re-running the segment's
                # internals a second time outside the checkpoint/recompute
                # boundary -- a second run would produce tensors disconnected
                # from the real autograd graph used for backward. Typical
                # transformer-block decompositions satisfy this trivially
                # (strictly sequential; residual adds take the pre-block
                # value as an EXTERNAL input, never as an internal node
                # escaping the group). When it doesn't hold, skip grouping
                # entirely for this run rather than risk silently wrong
                # gradients -- those nodes keep running through the
                # existing, unmodified per-node path (today's
                # fall-back-to-retain behavior).
                internally_contained = all(
                    all(user in group_set for user in node.users) for node in current_group[:-1]
                )
                if internally_contained:
                    self._checkpoint_group_start[current_group[0].name] = (current_strategy, list(current_group))
                    for member_node in current_group[1:]:
                        self._checkpoint_group_member.add(member_node.name)
            current_group = []
            current_strategy = None
            current_block_id = None

        for node in self.module.graph.nodes:
            if node.op != "call_function":
                flush()
                continue
            layer_name = self._node_layer_names.get(node.name)
            if layer_name is None:
                # Structural ops with no attributable layer (reshape/view/
                # split/alias/... -- confirmed empirically: GPT-2's own
                # attention block emits several of these between its named
                # leaf sub-modules) carry no real cost of their own. If a
                # checkpoint/recompute run is already in progress, keep
                # this node INSIDE it rather than ending the group here --
                # otherwise a single block fragments into several small
                # groups instead of one, each leaving its own gap of
                # unprotected activations. _run_checkpoint_group's
                # internally_contained safety check still applies to this
                # node exactly like any other member, so this is not a
                # relaxation of correctness, only of what counts as a
                # group boundary.
                if current_strategy is not None:
                    current_group.append(node)
                continue
            strategy = _activation_strategy_name(self._activation_strategies.get(layer_name))
            block_id = _block_id_for_layer_name(layer_name)
            if strategy != "retain" and strategy == current_strategy and block_id == current_block_id:
                current_group.append(node)
            else:
                flush()
                if strategy != "retain":
                    current_strategy = strategy
                    current_block_id = block_id
                    current_group = [node]
        flush()

        # Final-node detection (see _DeferredCheckpointSegment's docstring
        # for why this is handled separately from the grouping above): a
        # node whose ONLY user is the graph's own `output` node is, by
        # construction, never read by anything else -- the single-node
        # equivalent of the internally_contained safety check above, except
        # here a group of size ONE is deliberately allowed (the main loop
        # requires >1 node to form a group; a model's final projection layer
        # -- e.g. a decoder's vocabulary head -- is typically just one
        # call_function node, like "lm_head" decomposing to a single
        # aten.linear.default). Purely structural: no layer name is ever
        # hardcoded here, so this applies equally to any decoder-only
        # torch.export graph, not just GPT-2-style models.
        output_node = next((n for n in self.module.graph.nodes if n.op == "output"), None)
        if output_node is not None:
            output_refs: List[fx.Node] = []

            def _collect(value: Any) -> None:
                if isinstance(value, fx.Node):
                    output_refs.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        _collect(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        _collect(item)

            _collect(output_node.args)
            if len(output_refs) == 1:
                final_node = output_refs[0]
                if (
                    final_node.op == "call_function"
                    and final_node.name not in self._checkpoint_group_member
                    and list(final_node.users) == [output_node]
                ):
                    layer_name = self._node_layer_names.get(final_node.name)
                    strategy = (
                        _activation_strategy_name(self._activation_strategies.get(layer_name))
                        if layer_name is not None
                        else "retain"
                    )
                    if strategy in ("checkpoint", "recompute"):
                        # If this node was ALSO the start of a multi-node
                        # group (e.g. the group's last node happens to be
                        # this one), prefer the deferred single-node path
                        # for it specifically -- deregister it from the
                        # group so run_node only ever takes one of the two
                        # paths for this node, never both.
                        self._checkpoint_group_start.pop(final_node.name, None)
                        self._deferred_final_node = final_node
                        self._deferred_final_strategy = strategy

    def _compute_shared_call_module_layer_names(self) -> set:
        """Layer names whose call_module node appears more than once in
        this graph (e.g. torchvision resnet's shared self.relu, invoked
        once after bn1, once after bn2, once after the residual add --
        same module, same layer name, three distinct nodes). A Phase 4
        activation strategy is a single value keyed by layer name, so it
        cannot distinguish between these call sites; offering recompute/
        checkpoint for one silently offers it for all of them, which
        crashes torch.utils.checkpoint's bookkeeping as soon as one call
        site turns out structurally incompatible with the others
        (confirmed empirically -- see
        fix/07_gpu_recompute_joint_decision/README.md). Computed directly
        from THIS graph rather than trusting data.shared_call_site_layers
        from the ILP's own data loader, so the exclusion holds even for a
        DevicePlan built some other way -- defense in depth, not a
        replacement for the ILP-side exclusion (which exists so the MILP's
        own cost model never assumes savings this runtime cannot deliver).
        """
        layer_name_counts: Dict[str, int] = {}
        for node in self.module.graph.nodes:
            if node.op != "call_module":
                continue
            layer_name = self._layer_name_for_node(node)
            if layer_name is not None:
                layer_name_counts[layer_name] = layer_name_counts.get(layer_name, 0) + 1
        return {name for name, count in layer_name_counts.items() if count > 1}

    def _detect_checkpoint_groups_call_module(self) -> None:
        """Vision-graph (fx.symbolic_trace) analogue of
        _detect_checkpoint_groups: groups consecutive call_module nodes
        that share both a checkpoint/recompute strategy AND a block id
        (_block_id_for_layer_name -- e.g. all of a resnet Bottleneck's
        "layer1.0.*" leaves) into one segment, so run_node's existing
        group-dispatch logic (shared with the decoder-only path below)
        checkpoints/recomputes the WHOLE block as a single unit instead of
        one module at a time. See fix/07_gpu_recompute_joint_decision/
        README.md: checkpointing a single isolated nn.Module measurably
        makes memory usage WORSE (confirmed empirically, +36% in a
        controlled test), not better, because its only expensive tensor is
        its own output, which checkpointing does not free. Writes into the
        SAME self._checkpoint_group_start/_checkpoint_group_member dicts as
        _detect_checkpoint_groups -- safe because that method only ever
        populates them from call_function nodes, this one only from
        call_module nodes, so their keys (node names) never collide within
        one graph (a node has exactly one .op). Does not touch call_method
        or the deferred-final-node mechanism (call_module graphs do not
        need it: a vision model's final layer, e.g. "fc", is already a
        whole nn.Module call like any other -- the gap this method exists
        to close is specifically single-module checkpoint/recompute, not a
        decomposed-into-many-ops final node).
        """
        shared_layer_names = self._shared_call_module_layer_names
        current_group: List[fx.Node] = []
        current_strategy: str | None = None
        current_block_id: str | None = None

        def flush() -> None:
            nonlocal current_group, current_strategy, current_block_id
            if len(current_group) > 1 and current_strategy in ("checkpoint", "recompute"):
                group_set = set(current_group)
                # Same safety requirement as _detect_checkpoint_groups: no
                # node inside the group may be consumed by anything outside
                # it except the group's own last node.
                internally_contained = all(
                    all(user in group_set for user in node.users) for node in current_group[:-1]
                )
                if internally_contained:
                    self._checkpoint_group_start[current_group[0].name] = (current_strategy, list(current_group))
                    for member_node in current_group[1:]:
                        self._checkpoint_group_member.add(member_node.name)
            current_group = []
            current_strategy = None
            current_block_id = None

        for node in self.module.graph.nodes:
            if node.op != "call_module":
                flush()
                continue
            layer_name = self._layer_name_for_node(node)
            if layer_name is None:
                flush()
                continue
            if layer_name in shared_layer_names:
                # Never independently strategizable (see above). If a
                # group is already forming, absorb this node as a
                # transparent pass-through member instead of ending the
                # group here -- otherwise the group fragments into tiny,
                # low-savings pieces at every such boundary (confirmed
                # empirically: this is exactly why grouping a resnet
                # Bottleneck's conv/bn chain around its shared ReLU calls
                # measured only ~5% savings instead of the ~20%+ a whole,
                # unbroken block achieves). If no group is forming, this
                # node is simply retained on its own, same as today.
                if current_strategy is not None:
                    current_group.append(node)
                continue
            strategy = _activation_strategy_name(self._activation_strategies.get(layer_name))
            block_id = _block_id_for_layer_name(layer_name)
            if strategy != "retain" and strategy == current_strategy and block_id == current_block_id:
                current_group.append(node)
            else:
                flush()
                if strategy != "retain":
                    current_strategy = strategy
                    current_block_id = block_id
                    current_group = [node]
        flush()

    def _build_deferred_final_segment(self, node: fx.Node, strategy: str) -> _DeferredCheckpointSegment | None:
        prepared = self._prepare_segment_fn([node])
        if prepared is None:
            return None
        segment_fn, external_values = prepared
        layer_name = self._layer_name_for_node(node) or node.name
        linear_args = (
            self._extract_linear_args(node) if node.target is torch.ops.aten.linear.default else None
        )
        return _DeferredCheckpointSegment(
            strategy=strategy,
            segment_fn=segment_fn,
            external_values=external_values,
            layer_name=layer_name,
            linear_args=linear_args,
        )

    def _extract_linear_args(
        self, node: fx.Node
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] | None:
        # aten.linear.default(input, weight, bias=None) -- this node's own
        # dependencies were already computed earlier in topological order,
        # so their values are already in self.env. Structural detection
        # only (shape/arity, never a layer name), so it applies equally to
        # any decoder-only model's vocabulary-projection head, not just
        # GPT-2's "lm_head". Returns None (never raises) for anything that
        # doesn't fit this exact shape -- the caller then falls back to the
        # existing generic (opaque) deferred-segment path.
        args = node.args
        if len(args) < 2 or len(args) > 3:
            return None
        input_node, weight_node = args[0], args[1]
        bias_node = args[2] if len(args) == 3 else None
        if not isinstance(input_node, fx.Node) or not isinstance(weight_node, fx.Node):
            return None
        input_value = self.env.get(input_node)
        weight_value = self.env.get(weight_node)
        if not isinstance(input_value, torch.Tensor) or not isinstance(weight_value, torch.Tensor):
            return None
        bias_value: torch.Tensor | None = None
        if bias_node is not None:
            if not isinstance(bias_node, fx.Node):
                return None
            resolved_bias = self.env.get(bias_node)
            if resolved_bias is not None and not isinstance(resolved_bias, torch.Tensor):
                return None
            bias_value = resolved_bias

        forward_device = self._plan.resolve_torch_device(
            layer_name=self._layer_name_for_node(node) or str(node.target), gpu_id=self._gpu_id, phase="forward"
        )
        to_move = [input_value, weight_value] + ([bias_value] if bias_value is not None else [])
        moved_values, _ = self._move_inputs_to_device(tuple(to_move), {}, forward_device)
        moved_input, moved_weight = moved_values[0], moved_values[1]
        moved_bias = moved_values[2] if bias_value is not None else None
        return moved_input, moved_weight, moved_bias

    def _prepare_segment_fn(
        self, group_nodes: List[fx.Node]
    ) -> Tuple[Callable[..., torch.Tensor], Tuple[torch.Tensor, ...]] | None:
        # Shared by _run_checkpoint_group and _build_deferred_final_segment:
        # collects this group's EXTERNAL tensor dependencies and builds a
        # closure that replays exactly the group's own node.target(...)
        # calls using a local (group-scoped) env, falling back to self.env
        # for the external references. Returns None (never raises) when the
        # group has a non-Tensor external dependency or spans more than one
        # forward device -- both callers treat that as "do not group this",
        # not as an error.
        group_node_set = set(group_nodes)
        external_nodes: List[fx.Node] = []
        external_seen: set[str] = set()

        def _collect_node_refs(value: Any) -> None:
            if isinstance(value, fx.Node):
                if value not in group_node_set and value.name not in external_seen:
                    external_seen.add(value.name)
                    external_nodes.append(value)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    _collect_node_refs(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _collect_node_refs(item)

        for gn in group_nodes:
            _collect_node_refs(gn.args)
            _collect_node_refs(gn.kwargs)

        external_tensor_nodes = [n for n in external_nodes if isinstance(self.env.get(n), torch.Tensor)]
        if len(external_tensor_nodes) != len(external_nodes):
            self._warn_once(
                f"Checkpoint group starting at '{group_nodes[0].name}' has a non-Tensor external "
                "dependency; falling back to per-node execution for this group."
            )
            return None

        # Transparent boundary nodes (layer_name is None -- structural ops
        # like view/split/alias swallowed into the group by
        # _detect_checkpoint_groups) have no device assignment of their own
        # in the plan; str(gn.target) is not a real layer_name, so querying
        # the plan with it would resolve to an arbitrary fallback device and
        # falsely look like a device mismatch. Only nodes with a real
        # layer_name mapping determine the group's device -- the trigger
        # node always has one (that's what started the group), so this set
        # is never empty.
        forward_devices = {
            self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
            for gn in group_nodes
            for layer_name in (self._layer_name_for_node(gn),)
            if layer_name is not None
        }
        if len(forward_devices) != 1:
            self._warn_once(
                f"Checkpoint group starting at '{group_nodes[0].name}' spans more than one forward "
                "device; falling back to per-node execution for this group."
            )
            return None
        forward_device = next(iter(forward_devices))

        external_values = [self.env[n] for n in external_tensor_nodes]
        external_values, _ = self._move_inputs_to_device(tuple(external_values), {}, forward_device)

        # How many times each producer (external input or internal group
        # node) is still read by a NOT-YET-EXECUTED group node, at each
        # point during segment_fn's loop. Without pruning local_env as these
        # hit zero, EVERY intermediate tensor the group ever produces stays
        # alive simultaneously for the group's whole execution -- for a
        # group spanning several transformer blocks fused together, that is
        # as expensive as no checkpointing at all (confirmed by a real OOM:
        # a 6-block merged group's own forward pass alone exhausted the
        # GPU before even reaching the next layer). Only group_nodes[-1]'s
        # value survives to be returned; everything else must be freed the
        # moment its last consumer inside the group has run.
        initial_uses: Dict[fx.Node, int] = {}

        def _count_uses(value: Any) -> None:
            if isinstance(value, fx.Node):
                initial_uses[value] = initial_uses.get(value, 0) + 1
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    _count_uses(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _count_uses(item)

        for gn in group_nodes:
            _count_uses(gn.args)
            _count_uses(gn.kwargs)

        def segment_fn(*values: torch.Tensor) -> torch.Tensor:
            local_env: Dict[fx.Node, Any] = dict(zip(external_tensor_nodes, values))
            remaining_uses = dict(initial_uses)

            def resolve(value: Any) -> Any:
                # Every fx.Node this can ever see is already in local_env:
                # group-internal nodes are seeded into it right after they
                # run (below), and every node EXTERNAL to the group was
                # validated as Tensor-valued and pre-seeded from `values`
                # above -- _prepare_segment_fn returns None otherwise (see
                # the external_tensor_nodes check). Deliberately NOT falling
                # back to self.env[value]: that fallback is unreachable by
                # construction, and capturing `self` here created a real
                # reference cycle (resolve's closure -> self -> self.env ->
                # a later step's own group result -> that step's own
                # segment_fn/resolve closure -> ITS self -> ...), chaining
                # across every training step's own interpreter instance.
                # Only Python's periodic cyclic GC (not plain refcounting)
                # could ever break that chain, so real GPU memory piled up
                # between GC passes -- confirmed empirically: removing this
                # one fallback alone fixed a steady per-step memory growth
                # that otherwise OOM'd a real training run by ~step 10.
                if isinstance(value, fx.Node):
                    return local_env[value]
                if isinstance(value, tuple):
                    return tuple(resolve(v) for v in value)
                if isinstance(value, list):
                    return [resolve(v) for v in value]
                if isinstance(value, dict):
                    return {k: resolve(v) for k, v in value.items()}
                return value

            def _release(value: Any) -> None:
                if isinstance(value, fx.Node):
                    if value not in remaining_uses:
                        return
                    remaining_uses[value] -= 1
                    if remaining_uses[value] <= 0 and value is not group_nodes[-1]:
                        local_env.pop(value, None)
                    return
                if isinstance(value, (list, tuple)):
                    for item in value:
                        _release(item)
                    return
                if isinstance(value, dict):
                    for item in value.values():
                        _release(item)

            last_value: torch.Tensor = cast(torch.Tensor, None)
            for gn in group_nodes:
                resolved_args = resolve(gn.args)
                resolved_kwargs = resolve(gn.kwargs)
                if gn.op == "call_module":
                    # call_module's own .target is the module's qualified
                    # NAME (a str), never a callable -- unlike
                    # call_function, where .target already IS the callable.
                    # Only reached by groups _detect_checkpoint_groups_call_module
                    # builds (vision graphs); decoder-only groups are always
                    # call_function, so this branch is a no-op for that path.
                    submod = self.fetch_attr(gn.target)
                    submod.to(forward_device)
                    if (
                        _is_inplace_module(submod)
                        and resolved_args
                        and isinstance(resolved_args[0], torch.Tensor)
                    ):
                        # Same fix as _run_layer_with_dual_placement's own
                        # in-place handling (fix/02, bug 2): an in-place
                        # module (e.g. ReLU(inplace=True), standard
                        # throughout torchvision's resnet) would otherwise
                        # mutate a tensor this segment's own recompute/
                        # checkpoint bookkeeping still needs intact --
                        # confirmed empirically (RuntimeError: "one of the
                        # variables needed for gradient computation has
                        # been modified by an inplace operation") the first
                        # time this grouping actually ran on a real
                        # resnet50 block.
                        resolved_args = (resolved_args[0].clone(),) + resolved_args[1:]
                    last_value = submod(*resolved_args, **resolved_kwargs)
                else:
                    last_value = gn.target(*resolved_args, **resolved_kwargs)
                local_env[gn] = last_value
                _release(gn.args)
                _release(gn.kwargs)
            # Clone the group's OWN returned value before handing it back:
            # torch.utils.checkpoint's use_reentrant=False bookkeeping
            # tracks this exact tensor object to recompute it consistently
            # during backward. The node right after this group (outside
            # it, run by the normal interpreter loop) is very often an
            # in-place op in torchvision's resnet (`out += identity` is a
            # literal in-place add in every Bottleneck/BasicBlock) -- that
            # would mutate this tensor's storage in place, incrementing
            # its autograd version past what checkpoint's saved-tensor
            # hooks recorded. Confirmed empirically (RuntimeError: "one of
            # the variables needed for gradient computation has been
            # modified by an inplace operation ... is at version 1;
            # expected version 0") the first time this grouping ran on a
            # real resnet50 block ending right before its residual add.
            if isinstance(last_value, torch.Tensor):
                return last_value.clone()
            return last_value

        return segment_fn, tuple(external_values)

    def _run_checkpoint_group(self, group_nodes: List[fx.Node], strategy: str) -> Any:
        # Executes an entire group of same-layer_name call_function nodes
        # (see _detect_checkpoint_groups) as one checkpointed/recomputed
        # unit, then caches every member node's result so the interpreter's
        # normal per-node loop (which will still visit each of them one by
        # one) just returns the cached value instead of recomputing anything
        # itself -- see run_node's override below.
        prepared = self._prepare_segment_fn(group_nodes)
        if prepared is None:
            # Group-level checkpoint/recompute isn't viable for this group
            # (non-Tensor external dependency, or a device mismatch) --
            # run every member individually instead, exactly as if no
            # grouping had been attempted. run_node (below) unconditionally
            # pops every member's name from self._checkpoint_group_results
            # once this returns, so every member must get an entry here,
            # not just the trigger node. Writing into self.env as we go
            # (not only into the results cache) is required too: a later
            # member node referencing an earlier one by its fx.Node
            # resolves that reference through self.env via
            # fetch_args_kwargs_from_env, same as normal per-node execution.
            for member_node in group_nodes:
                value = self._run_single_node_default(member_node)
                self.env[member_node] = value
                self._checkpoint_group_results[member_node.name] = value
            return
        segment_fn, external_values = prepared

        if strategy == "checkpoint":
            checkpoint_ctx = _saved_tensor_cpu_offload_context()
            with checkpoint_ctx:
                final_value = segment_fn(*external_values)
        else:  # "recompute"
            final_value = activation_checkpoint(segment_fn, *external_values, use_reentrant=False)

        # Every node's value is looked up the SAME way by run_node (via
        # self._checkpoint_group_results.pop(n.name)), including the
        # group's own first/trigger node -- only group_nodes[-1]'s entry is
        # the real final_value. Every other member (first node included)
        # gets a placeholder: internally_contained (see
        # _detect_checkpoint_groups) guarantees nothing outside the group
        # ever reads any of their individual values, only the last node's.
        for member_node in group_nodes[:-1]:
            self._checkpoint_group_results[member_node.name] = None
        self._checkpoint_group_results[group_nodes[-1].name] = final_value
        group_layer_names = {
            self._layer_name_for_node(gn) or gn.name for gn in group_nodes
        }
        self._runtime_features["checkpoint_layers" if strategy == "checkpoint" else "recompute_layers"].update(
            group_layer_names
        )
        self._runtime_features["checkpoint_count" if strategy == "checkpoint" else "recompute_count"] += 1

    def _run_single_node_default(self, n: fx.Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        return getattr(self, n.op)(n.target, args, kwargs)

    def _node_output_clone_safe(self, node: fx.Node | None) -> bool:
        # True only if EVERY direct consumer of this node's output is
        # provably unable to mutate it in place -- see _is_inplace_module
        # and _run_layer_with_dual_placement's output_clone_safe parameter
        # for what this buys (skipping an expensive .clone() of the full
        # activation tensor). Conservative (returns False, i.e. "clone it")
        # whenever this can't be established with confidence: unknown node,
        # a consumer this analysis doesn't recognize, or a submodule lookup
        # that fails. Cached per node name since the graph structure (who
        # consumes whom) never changes across training steps, even though
        # this method runs once per node per step.
        if node is None:
            return False
        cached = self._output_clone_safe_cache.get(node.name)
        if cached is not None:
            return cached

        safe = True
        for user in node.users:
            if user.op == "call_module":
                try:
                    submod = self.fetch_attr(user.target)
                except Exception:
                    safe = False
                    break
                if _is_inplace_module(submod):
                    safe = False
                    break
            elif user.op in ("call_function", "call_method"):
                # PyTorch convention: a trailing underscore names the
                # in-place variant of an op (add_, relu_, mul_, ...).
                target_name = getattr(user.target, "__name__", str(user.target))
                if target_name.endswith("_"):
                    safe = False
                    break
            else:
                # output/placeholder/get_attr consuming this node's value is
                # unusual for a plain activation tensor; be conservative
                # rather than assume it's safe.
                safe = False
                break

        self._output_clone_safe_cache[node.name] = safe
        return safe

    def _layer_name_for_node(self, node: fx.Node | None) -> str | None:
        if node is None:
            return None
        mapped = self._node_layer_names.get(node.name)
        if mapped:
            return mapped
        if node.op == "call_module":
            return str(node.target)
        return None

    def _move_inputs_to_device(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        target_device: torch.device,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        devices = _all_tensor_devices(args) + _all_tensor_devices(kwargs)
        if not devices:
            return args, kwargs
        if all(device == target_device for device in devices):
            return args, kwargs

        if self._transfer_stream is not None and self._enable_async_transfer and target_device.type == "cuda":
            with torch.cuda.stream(self._transfer_stream):
                moved_args = _to_device_non_blocking(args, target_device)
                moved_kwargs = _to_device_non_blocking(kwargs, target_device)
            torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            moved_args = _to_device(args, target_device)
            moved_kwargs = _to_device(kwargs, target_device)

        self._transfer_metrics["bytes"] += float(_tensor_bytes(args) + _tensor_bytes(kwargs))
        self._transfer_metrics["events"] += 1.0
        return moved_args, moved_kwargs

    def _record_unsupported_activation(self, layer_name: str) -> None:
        strategy_name = _activation_strategy_name(self._activation_strategies.get(layer_name))
        if strategy_name == "retain":
            return
        self._runtime_features["unsupported_activation_layers"].add(layer_name)
        self._warn_once(
            f"Layer '{layer_name}' requested activation strategy '{strategy_name}' but decoder-only export DAG executes at operator granularity; falling back to retain"
        )

    def _align_operator_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        devices = _all_tensor_devices(args) + _all_tensor_devices(kwargs)
        if len(devices) <= 1:
            return args, kwargs

        target_device = next((device for device in devices if device.type == "cuda"), devices[0])
        if all(device == target_device for device in devices):
            return args, kwargs

        if self._transfer_stream is not None and self._enable_async_transfer and target_device.type == "cuda":
            with torch.cuda.stream(self._transfer_stream):
                moved_args = _to_device_non_blocking(args, target_device)
                moved_kwargs = _to_device_non_blocking(kwargs, target_device)
            torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            moved_args = _to_device(args, target_device)
            moved_kwargs = _to_device(kwargs, target_device)

        self._transfer_metrics["bytes"] += float(_tensor_bytes(args) + _tensor_bytes(kwargs))
        self._transfer_metrics["events"] += 1.0
        return moved_args, moved_kwargs

    def get_attr(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        value = super().get_attr(target, args, kwargs)
        if not isinstance(value, torch.Tensor):
            return value

        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            return value

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        if value.device == forward_device:
            return value

        cache_key = (str(target), str(forward_device))
        cached = self._attr_cache.get(cache_key)
        if cached is not None:
            return cached

        moved = value.to(forward_device)
        self._transfer_metrics["bytes"] += float(_tensor_bytes(value))
        self._transfer_metrics["events"] += 1.0
        self._attr_cache[cache_key] = moved
        return moved

    def run_node(self, n: fx.Node) -> Any:
        self._active_node = n
        try:
            # Both checks below match whatever _detect_checkpoint_groups
            # (call_function, decoder-only graphs) or
            # _detect_checkpoint_groups_call_module (call_module, vision
            # graphs) identified during __init__ -- dispatch here is by
            # node NAME only, so it is agnostic to which op produced the
            # group. When a graph has no checkpoint/recompute-marked
            # groups at all, both dicts stay empty and every node falls
            # straight through to the original super().run_node() path,
            # unchanged.
            if n.name in self._checkpoint_group_results:
                return self._checkpoint_group_results.pop(n.name)
            if self._deferred_final_node is not None and n is self._deferred_final_node:
                deferred = self._build_deferred_final_segment(n, cast(str, self._deferred_final_strategy))
                if deferred is not None:
                    return deferred
                # _prepare_segment_fn already warned why; run this one node
                # exactly as if it had never been marked checkpoint/recompute.
                return self._run_single_node_default(n)
            if n.name in self._checkpoint_group_start:
                strategy, group_nodes = self._checkpoint_group_start[n.name]
                # Populates self._checkpoint_group_results for EVERY node in
                # the group, including this one (the group's first/trigger
                # node) -- the group's real final_value belongs to
                # group_nodes[-1], never to this first node, so it must be
                # looked up the same way every other member node is, not
                # returned directly.
                self._run_checkpoint_group(group_nodes, strategy)
                return self._checkpoint_group_results.pop(n.name)
            return super().run_node(n)
        finally:
            self._active_node = None

    def call_module(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if str(target) == "_guards_fn":
            # torch.export shape-guard helper: no parameters, no compute, never planned.
            return super().call_module(target, args, kwargs)
        layer_name = self._layer_name_for_node(self._active_node) or str(target)
        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")

        src_dev = _current_device(args)
        payload_bytes = _tensor_bytes(args) + _tensor_bytes(kwargs)

        current_node = self._active_node
        node_name = current_node.name if current_node is not None else layer_name

        used_prefetched_input = (
            node_name in self._prefetched_inputs
            and len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and len(kwargs) == 0
        )
        if used_prefetched_input:
            moved_args = (self._prefetched_inputs.pop(node_name),)
            moved_kwargs = kwargs
            if self._transfer_stream is not None and moved_args[0].device.type == "cuda":
                torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            if self._transfer_stream is not None and self._enable_async_transfer and forward_device.type == "cuda":
                with torch.cuda.stream(self._transfer_stream):
                    moved_args = _to_device_non_blocking(args, forward_device)
                    moved_kwargs = _to_device_non_blocking(kwargs, forward_device)
                torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
            else:
                moved_args = _to_device(args, forward_device)
                moved_kwargs = _to_device(kwargs, forward_device)

        if src_dev is not None and src_dev != forward_device:
            self._transfer_metrics["bytes"] += float(payload_bytes)
            self._transfer_metrics["events"] += 1.0

        submod = self.fetch_attr(target)
        submod.to(forward_device)

        # A layer name reused at multiple call sites can never be safely
        # strategized on its own (see _compute_shared_call_module_layer_names)
        # -- force "retain" here regardless of what self._activation_strategies
        # says, as a defense-in-depth backstop. The ILP already never
        # requests recompute/checkpoint for such a layer
        # (data.shared_call_site_layers), so this should be a no-op in
        # practice; it only matters for a DevicePlan built some other way.
        strategy_name = (
            "retain" if layer_name in self._shared_call_module_layer_names
            else _activation_strategy_name(self._activation_strategies.get(layer_name))
        )
        if strategy_name == "recompute":
            self._runtime_features["recompute_layers"].add(layer_name)
        elif strategy_name == "checkpoint":
            self._runtime_features["checkpoint_layers"].add(layer_name)

        wants_dual_runtime = (
            self._plan.assignment_forward.get(layer_name) != self._plan.assignment_backward.get(layer_name)
        )
        can_materialize_dual = (
            wants_dual_runtime
            and len(moved_args) == 1
            and isinstance(moved_args[0], torch.Tensor)
            and len(moved_kwargs) == 0
            and forward_device != backward_device
        )

        if wants_dual_runtime and not can_materialize_dual:
            self._warnings.append(
                f"Layer '{layer_name}' requests different backward placement but DAG runtime cannot materialize it for this call signature; using forward placement"
            )

        if can_materialize_dual:
            self._runtime_features["backward_relocation_layers"].add(layer_name)
            self._runtime_features["backward_relocation_count"] += 1
            if strategy_name in {"checkpoint", "recompute"}:
                # Dual Placement (fwd/bwd on different devices) and activation strategies
                # (checkpoint/recompute) are mutually exclusive at this time: the dual
                # placement function performs its own recompute of the forward pass during
                # backward, making an outer checkpoint redundant.  Log explicitly so the
                # combination is never silently dropped.
                self._warnings.append(
                    f"Layer '{layer_name}': Dual Placement overrides '{strategy_name}' strategy "
                    f"because dual-device recompute already handles activation retention."
                )
            return _run_layer_with_dual_placement(
                layer=submod,
                layer_name=layer_name,
                input_tensor=cast(torch.Tensor, moved_args[0]),
                forward_device=forward_device,
                backward_device=backward_device,
                output_clone_safe=self._node_output_clone_safe(current_node),
                pinned_buffer_cache=self._pinned_output_buffer_cache,
                pinned_buffer_key=node_name,
            )

        if strategy_name == "recompute":
            if len(moved_args) == 1 and isinstance(moved_args[0], torch.Tensor) and len(moved_kwargs) == 0:
                self._runtime_features["recompute_count"] += 1
                recompute_input = moved_args[0]
                if _is_inplace_module(submod):
                    # An in-place module (ReLU(inplace=True), standard in
                    # torchvision's resnet) would otherwise mutate this
                    # exact tensor's storage, incrementing its autograd
                    # version past what torch.utils.checkpoint's
                    # use_reentrant=False bookkeeping recorded when it
                    # first saved it -- confirmed empirically (fix/07_
                    # gpu_recompute_joint_decision/README.md): this exact
                    # pattern crashes with "one of the variables needed for
                    # gradient computation has been modified by an inplace
                    # operation" as soon as a shared module instance
                    # (common in torchvision -- one self.relu called at
                    # several points in the same block) gets this strategy.
                    recompute_input = recompute_input.clone()
                return activation_checkpoint(submod, recompute_input, use_reentrant=False)
            self._warnings.append(
                f"Layer '{layer_name}' requested recompute but DAG runtime supports only single-Tensor positional input; falling back to retain"
            )
            self._runtime_features["unsupported_activation_layers"].add(layer_name)

        if strategy_name == "checkpoint":
            if len(moved_args) == 1 and isinstance(moved_args[0], torch.Tensor) and len(moved_kwargs) == 0:
                self._runtime_features["checkpoint_count"] += 1
                checkpoint_input = moved_args[0].clone() if _is_inplace_module(submod) else moved_args[0]
                checkpoint_ctx = _saved_tensor_cpu_offload_context()
                with checkpoint_ctx:
                    return submod(checkpoint_input)
            self._warnings.append(
                f"Layer '{layer_name}' requested checkpoint but DAG runtime supports only single-Tensor positional input; falling back to retain"
            )
            self._runtime_features["unsupported_activation_layers"].add(layer_name)

        out = submod(*moved_args, **moved_kwargs)

        if self._enable_prefetch and current_node is not None and isinstance(out, torch.Tensor):
            for user_node in current_node.users:
                if user_node.op != "call_module":
                    continue
                user_name = user_node.name
                user_target = str(user_node.target)
                user_device = self._plan.resolve_torch_device(
                    layer_name=user_target,
                    gpu_id=self._gpu_id,
                    phase="forward",
                )
                out_device = out.device
                if out_device == user_device:
                    continue
                if self._transfer_stream is not None and self._enable_async_transfer and user_device.type == "cuda":
                    with torch.cuda.stream(self._transfer_stream):
                        prefetched = out.to(user_device, non_blocking=True)
                else:
                    prefetched = out.to(user_device)
                self._prefetched_inputs[user_name] = prefetched
                self._runtime_features["prefetch_layers"].add(user_target)
                self._runtime_features["prefetch_count"] += 1
                self._runtime_features["prefetch_bytes"] += float(_tensor_bytes(out))

        return out

    def call_function(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if not self._supports_activation_strategies and str(target) == "aten._assert_tensor_metadata.default":
            return None

        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            moved_args, moved_kwargs = self._align_operator_inputs(args, kwargs)
            return target(*moved_args, **moved_kwargs)

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")
        if not self._supports_activation_strategies:
            self._record_unsupported_activation(layer_name)

        if forward_device != backward_device and not _is_stochastic_operator(target):
            self._runtime_features["backward_relocation_layers"].add(layer_name)
            self._runtime_features["backward_relocation_count"] += 1
            return _run_operator_with_dual_placement(
                target=target,
                args=args,
                kwargs=kwargs,
                forward_device=forward_device,
                backward_device=backward_device,
                layer_name=layer_name,
            )

        if forward_device != backward_device:
            self._warn_once(
                f"Layer '{layer_name}' ({target}) requests different backward placement but is a "
                "stochastic operator (e.g. Dropout); dual placement cannot replay its random mask "
                "on recompute, so forward placement is used for both phases"
            )

        moved_args, moved_kwargs = self._move_inputs_to_device(args, kwargs, forward_device)
        return target(*moved_args, **moved_kwargs)

    def call_method(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            moved_args, moved_kwargs = self._align_operator_inputs(args, kwargs)
            self_obj, *method_args = moved_args
            return getattr(self_obj, target)(*method_args, **moved_kwargs)

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")
        if not self._supports_activation_strategies:
            self._record_unsupported_activation(layer_name)
        if forward_device != backward_device:
            self._warn_once(
                f"Layer '{layer_name}' requests different backward placement but decoder-only export DAG cannot materialize operator-level dual placement; using forward placement"
            )
        moved_args, moved_kwargs = self._move_inputs_to_device(args, kwargs, forward_device)
        self_obj, *method_args = moved_args
        return getattr(self_obj, target)(*method_args, **moved_kwargs)


def _run_hybrid_training_dag(
    model: nn.Module,
    input_data: Any,
    target_data: torch.Tensor | None,
    plan: DevicePlan,
    activation_strategies: Dict[str, Any] | None,
    steps: int,
    lr: float,
    gpu_id: int,
    enable_rapl: bool,
    energy_sample_interval: float,
    enable_async_transfer: bool,
    enable_prefetch: bool,
    warnings: List[str],
    traced_module: Optional[fx.GraphModule] = None,
    node_layer_names: Dict[str, str] | None = None,
    supports_activation_strategies: bool = True,
) -> HybridExecutionResult:
    graph_module = traced_module or _try_symbolic_trace(model)
    if graph_module is None:
        raise RuntimeError("DAG execution requested but symbolic tracing failed for this model")

    model.train()
    try:
        graph_module.train()
    except NotImplementedError:
        warnings.append("Trace backend does not support GraphModule.train(); executing with the exported module as-is")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Created ONCE here, before the step loop below constructs a fresh
    # _DeviceAwareFXInterpreter every step -- see that class's
    # pinned_output_buffer_cache parameter docstring for why this must
    # outlive any single step's interpreter instance.
    pinned_output_buffer_cache: Dict[str, torch.Tensor] = {}

    step_traces: List[StepTrace] = []
    total_transfer_bytes = 0.0
    total_transfer_events = 0.0
    total_prefetch_bytes = 0.0
    total_prefetch_events = 0
    global_peak_gpu_mem_mb = 0.0
    recompute_layers_used: set[str] = set()
    checkpoint_layers_used: set[str] = set()
    backward_relocation_layers_used: set[str] = set()
    prefetch_layers_used: set[str] = set()
    unsupported_activation_layers: set[str] = set()
    quality_metric_name_used: str | None = None

    uses_gpu = torch.cuda.is_available() and (
        any(str(v).upper() == "GPU" for v in plan.assignment_forward.values())
        or any(str(v).upper() == "GPU" for v in plan.assignment_backward.values())
    )
    energy_device_type = "cuda" if uses_gpu else "cpu"
    energy_source = "nvml" if energy_device_type == "cuda" else "rapl"
    energy_monitor = EnergyMonitor(
        device_type=energy_device_type,
        gpu_id=gpu_id,
        sample_interval=energy_sample_interval,
        enable_rapl=enable_rapl,
    )

    run_start = time.perf_counter()
    energy_monitor.start()
    transfer_stream = torch.cuda.Stream(device=gpu_id) if (enable_async_transfer and torch.cuda.is_available()) else None
    try:
        for step_idx in range(steps):
            step_start = time.perf_counter()
            optimizer.zero_grad()

            transfer_metrics = {"bytes": 0.0, "events": 0.0}
            runtime_features = {
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
            interp = _DeviceAwareFXInterpreter(
                graph_module,
                plan,
                gpu_id,
                transfer_metrics,
                activation_strategies,
                warnings,
                runtime_features,
                enable_async_transfer,
                enable_prefetch,
                transfer_stream,
                node_layer_names=node_layer_names,
                supports_activation_strategies=supports_activation_strategies,
                pinned_output_buffer_cache=pinned_output_buffer_cache,
            )

            forward_start = time.perf_counter()
            out = interp.run(*_prepare_fx_run_args(graph_module, input_data))

            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            backward_start = time.perf_counter()
            loss, quality_metric_name, quality_metric_value = _compute_objective_possibly_deferred(
                out, target_data=target_data, runtime_features=runtime_features
            )
            if quality_metric_name_used is None and quality_metric_name is not None:
                quality_metric_name_used = quality_metric_name
            loss_value = float(loss.detach().cpu().item())
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            # Break a reference cycle: when the final layer was deferred
            # (see _DeferredCheckpointSegment), self.env[final_node] holds
            # that segment's own segment_fn closure, which captures `self`
            # (this interpreter) to resolve its external node references --
            # interp -> env -> deferred -> segment_fn -> interp. Python's
            # refcounting GC cannot free a cycle; only the periodic cyclic
            # collector can, which runs far less often than once per step,
            # so every deferred step's whole self.env (every node's tensor
            # for that step) piles up as uncollected garbage until it
            # happens to run. Confirmed empirically: real distilgpt2
            # training's CURRENT (non-peak) GPU memory grew steadily
            # step-over-step and OOM'd by ~step 10 without this; clearing
            # env right after backward (nothing needs it afterward --
            # autograd holds its own separate graph, and any
            # checkpoint/recompute recompute that needed env already ran
            # DURING backward, above) breaks the cycle so everything is
            # freed immediately via ordinary refcounting instead.
            interp.env.clear()

            opt_start = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            optimizer_ms = (time.perf_counter() - opt_start) * 1000.0

            total_ms = (time.perf_counter() - step_start) * 1000.0

            peak_gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated(gpu_id) / (1024**2)
                global_peak_gpu_mem_mb = max(global_peak_gpu_mem_mb, peak_gpu_mem_mb)

            step_transfer_bytes = float(transfer_metrics["bytes"])
            step_transfer_events = int(transfer_metrics["events"])
            step_prefetch_bytes = float(runtime_features["prefetch_bytes"])
            step_prefetch_events = int(runtime_features["prefetch_count"])

            step_traces.append(
                StepTrace(
                    step=step_idx,
                    total_ms=total_ms,
                    forward_ms=forward_ms,
                    backward_ms=backward_ms,
                    optimizer_ms=optimizer_ms,
                    transfer_count=step_transfer_events,
                    transfer_total_mb=step_transfer_bytes / (1024**2),
                    prefetch_count=step_prefetch_events,
                    prefetch_total_mb=step_prefetch_bytes / (1024**2),
                    peak_gpu_mem_mb=peak_gpu_mem_mb,
                    recompute_count=int(runtime_features["recompute_count"]),
                    checkpoint_count=int(runtime_features["checkpoint_count"]),
                    backward_relocation_count=int(runtime_features["backward_relocation_count"]),
                    loss_value=loss_value,
                    task_metric_value=quality_metric_value,
                )
            )

            recompute_layers_used.update(cast(set[str], runtime_features["recompute_layers"]))
            checkpoint_layers_used.update(cast(set[str], runtime_features["checkpoint_layers"]))
            backward_relocation_layers_used.update(cast(set[str], runtime_features["backward_relocation_layers"]))
            prefetch_layers_used.update(cast(set[str], runtime_features["prefetch_layers"]))
            unsupported_activation_layers.update(cast(set[str], runtime_features["unsupported_activation_layers"]))

            total_prefetch_bytes += step_prefetch_bytes
            total_prefetch_events += step_prefetch_events

            total_transfer_bytes += step_transfer_bytes
            total_transfer_events += float(step_transfer_events)
    finally:
        run_duration_sec = max(0.0, time.perf_counter() - run_start)
        energy_monitor.stop()

    avg_power_w = float(energy_monitor.get_avg_power())
    total_energy_j = float(avg_power_w * run_duration_sec) if avg_power_w > 0 else 0.0
    if total_energy_j <= 0.0:
        warnings.append("Observed energy could not be measured (power monitor unavailable or zero readings)")

    avg_step_ms = sum(s.total_ms for s in step_traces) / len(step_traces)
    loss_values = [s.loss_value for s in step_traces]
    initial_loss = float(loss_values[0]) if loss_values else 0.0
    final_loss = float(loss_values[-1]) if loss_values else 0.0
    min_loss = float(min(loss_values)) if loss_values else 0.0
    loss_delta = float(final_loss - initial_loss) if loss_values else 0.0
    quality_values = [float(s.task_metric_value) for s in step_traces if s.task_metric_value is not None and math.isfinite(float(s.task_metric_value))]
    initial_quality_metric = float(quality_values[0]) if quality_values else None
    final_quality_metric = float(quality_values[-1]) if quality_values else None
    best_quality_metric = float(max(quality_values)) if quality_values else None
    quality_metric_delta = (
        float(final_quality_metric - initial_quality_metric)
        if initial_quality_metric is not None and final_quality_metric is not None
        else None
    )

    if unsupported_activation_layers:
        warnings.append(
            "DAG runtime could not apply some activation strategies due to unsupported call signatures; falling back to retain for: "
            f"{sorted(unsupported_activation_layers)}"
        )

    return HybridExecutionResult(
        status="ok",
        steps=steps,
        avg_step_ms=avg_step_ms,
        avg_power_w=avg_power_w,
        total_energy_j=total_energy_j,
        energy_source=energy_source,
        total_transfer_mb=total_transfer_bytes / (1024**2),
        total_transfer_events=int(total_transfer_events),
        total_prefetch_mb=total_prefetch_bytes / (1024**2),
        total_prefetch_events=total_prefetch_events,
        peak_gpu_mem_mb=global_peak_gpu_mem_mb,
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        loss_delta=loss_delta,
        prefetch_layers=sorted(prefetch_layers_used),
        recompute_layers=sorted(recompute_layers_used),
        checkpoint_layers=sorted(checkpoint_layers_used),
        backward_relocation_layers=sorted(backward_relocation_layers_used),
        unsupported_checkpoint_layers=sorted(unsupported_activation_layers),
        warnings=warnings,
        per_step=step_traces,
        quality_metric_name=quality_metric_name_used,
        initial_quality_metric=initial_quality_metric,
        final_quality_metric=final_quality_metric,
        best_quality_metric=best_quality_metric,
        quality_metric_delta=quality_metric_delta,
    )


def run_hybrid_training(
    model: nn.Module,
    input_data: Any,
    plan: DevicePlan,
    target_data: torch.Tensor | None = None,
    activation_strategies: Dict[str, Any] | None = None,
    steps: int = 5,
    lr: float = 0.01,
    gpu_id: int = 0,
    strict_plan: bool = True,
    enable_rapl: bool = False,
    energy_sample_interval: float = 0.05,
    enable_async_transfer: bool = False,
    enable_prefetch: bool = False,
    execution_mode: str = "linear",
) -> HybridExecutionResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if execution_mode not in {"auto", "linear", "dag"}:
        raise ValueError(f"execution_mode must be one of auto|linear|dag, got: {execution_mode}")

    traced_module = _try_symbolic_trace(model)
    export_trace_ctx = None if traced_module is not None else try_export_decoder_only_trace(model, input_data)
    coverage_layer_names = None if export_trace_ctx is None else set(export_trace_ctx.node_layer_names.values())
    warnings = validate_plan_coverage(
        model=model,
        plan=plan,
        strict=strict_plan,
        model_layer_names=coverage_layer_names,
    )
    if not torch.cuda.is_available() and (
        any(v == "GPU" for v in plan.assignment_forward.values())
        or any(v == "GPU" for v in plan.assignment_backward.values())
    ):
        warnings.append("GPU layers requested but CUDA is unavailable; those layers run on CPU")

    if execution_mode == "dag" and traced_module is None and export_trace_ctx is None:
        raise HybridExecutionUnsupportedError(
            "execution_mode=dag requires FX symbolic tracing support for this model"
        )

    active_trace = traced_module if traced_module is not None else (None if export_trace_ctx is None else export_trace_ctx.graph_module)
    active_node_layer_names = None if export_trace_ctx is None else export_trace_ctx.node_layer_names
    active_supports_activation_strategies = export_trace_ctx is None

    if execution_mode == "dag":
        return _run_hybrid_training_dag(
            model=model,
            input_data=input_data,
            target_data=target_data,
            plan=plan,
            activation_strategies=activation_strategies,
            steps=steps,
            lr=lr,
            gpu_id=gpu_id,
            enable_rapl=enable_rapl,
            energy_sample_interval=energy_sample_interval,
            enable_async_transfer=enable_async_transfer,
            enable_prefetch=enable_prefetch,
            warnings=warnings,
            traced_module=active_trace,
            node_layer_names=active_node_layer_names,
            supports_activation_strategies=active_supports_activation_strategies,
        )

    if execution_mode == "auto":
        can_use_dag = active_trace is not None
        if can_use_dag:
            return _run_hybrid_training_dag(
                model=model,
                input_data=input_data,
                target_data=target_data,
                plan=plan,
                activation_strategies=activation_strategies,
                steps=steps,
                lr=lr,
                gpu_id=gpu_id,
                enable_rapl=enable_rapl,
                energy_sample_interval=energy_sample_interval,
                enable_async_transfer=enable_async_transfer,
                enable_prefetch=enable_prefetch,
                warnings=warnings,
                traced_module=active_trace,
                node_layer_names=active_node_layer_names,
                supports_activation_strategies=active_supports_activation_strategies,
            )
        if active_trace is None:
            if not isinstance(input_data, torch.Tensor):
                raise HybridExecutionUnsupportedError(
                    "Auto execution mode requires FX graph tracing for structured inputs, but tracing failed for this model; "
                    "linear fallback only supports single-tensor leaf-chain execution"
                )
            warnings.append("Auto execution mode fell back to linear runtime (FX tracing unavailable)")

    layer_order = _ordered_sequential_layers(model)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    transfer_stream = torch.cuda.Stream(device=gpu_id) if (enable_async_transfer and torch.cuda.is_available()) else None
    # Keyed by layer_idx (unique per position in this strictly sequential
    # chain), never layer_name -- see _run_layer_with_dual_placement's
    # pinned_buffer_key docs.
    pinned_output_buffer_cache: Dict[int, torch.Tensor] = {}

    step_traces: List[StepTrace] = []
    total_transfer_bytes = 0
    total_transfer_events = 0
    total_prefetch_bytes = 0
    total_prefetch_events = 0
    global_peak_gpu_mem_mb = 0.0
    prefetch_layers_used: set[str] = set()
    recompute_layers_used: set[str] = set()
    checkpoint_layers_used: set[str] = set()
    backward_relocation_layers_used: set[str] = set()
    unsupported_checkpoint_layers: set[str] = set()
    quality_metric_name_used: str | None = None

    uses_gpu = torch.cuda.is_available() and (
        any(str(v).upper() == "GPU" for v in plan.assignment_forward.values())
        or any(str(v).upper() == "GPU" for v in plan.assignment_backward.values())
    )
    energy_device_type = "cuda" if uses_gpu else "cpu"
    energy_source = "nvml" if energy_device_type == "cuda" else "rapl"
    energy_monitor = EnergyMonitor(
        device_type=energy_device_type,
        gpu_id=gpu_id,
        sample_interval=energy_sample_interval,
        enable_rapl=enable_rapl,
    )
    run_start = time.perf_counter()
    energy_monitor.start()

    try:
        for step_idx in range(steps):
            step_start = time.perf_counter()
            optimizer.zero_grad()

            x = input_data
            forward_start = time.perf_counter()
            step_transfer_bytes = 0
            step_transfer_events = 0
            step_prefetch_bytes = 0
            step_prefetch_events = 0
            step_recompute_count = 0
            step_checkpoint_count = 0
            step_backward_relocation_count = 0
            prefetched_inputs: Dict[str, Any] = {}

            for layer_idx, (layer_name, layer) in enumerate(layer_order):
                target_device = plan.resolve_torch_device(layer_name=layer_name, gpu_id=gpu_id, phase="forward")
                backward_device = plan.resolve_torch_device(layer_name=layer_name, gpu_id=gpu_id, phase="backward")
                layer.to(target_device)
                strategy_name = _activation_strategy_name(
                    None if activation_strategies is None else activation_strategies.get(layer_name)
                )

                used_prefetched_input = layer_name in prefetched_inputs
                if used_prefetched_input:
                    x = prefetched_inputs.pop(layer_name)
                    if transfer_stream is not None:
                        dst_dev = _current_device(x)
                        if dst_dev is not None and dst_dev.type == "cuda":
                            torch.cuda.current_stream(gpu_id).wait_stream(transfer_stream)

                src_device = _current_device(x)
                if (not used_prefetched_input) and src_device is not None and src_device != target_device:
                    transfer_start = time.perf_counter()
                    payload_bytes = _tensor_bytes(x)
                    if transfer_stream is not None and target_device.type == "cuda":
                        with torch.cuda.stream(transfer_stream):
                            x = _to_device(x, target_device)
                        torch.cuda.current_stream(gpu_id).wait_stream(transfer_stream)
                        torch.cuda.synchronize(gpu_id)
                    else:
                        x = _to_device(x, target_device)
                        if target_device.type == "cuda":
                            torch.cuda.synchronize(gpu_id)
                    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
                    _ = TransferEvent(
                        layer=layer_name,
                        src_device=str(src_device),
                        dst_device=str(target_device),
                        size_mb=payload_bytes / (1024**2),
                        time_ms=transfer_ms,
                    )
                    step_transfer_bytes += payload_bytes
                    step_transfer_events += 1

                wants_dual_runtime = (
                    plan.assignment_forward.get(layer_name) != plan.assignment_backward.get(layer_name)
                )
                can_materialize_dual = (
                    wants_dual_runtime
                    and isinstance(x, torch.Tensor)
                    and target_device != backward_device
                )

                if wants_dual_runtime and not can_materialize_dual:
                    warnings.append(
                        f"Layer '{layer_name}' requests different backward placement but it cannot be materialized in the current runtime/device context; using forward placement for backward"
                    )

                if can_materialize_dual:
                    # Strictly sequential chain here (no FX graph), so the
                    # only downstream consumer of this layer's output is the
                    # next layer in the list -- same safety criterion as
                    # _DeviceAwareFXInterpreter._node_output_clone_safe, just
                    # expressed as "is there a next layer, and is it
                    # in-place" instead of walking node.users.
                    next_layer_inplace = (
                        layer_idx + 1 < len(layer_order)
                        and _is_inplace_module(layer_order[layer_idx + 1][1])
                    )
                    x = _run_layer_with_dual_placement(
                        layer=layer,
                        layer_name=layer_name,
                        input_tensor=cast(torch.Tensor, x),
                        forward_device=target_device,
                        backward_device=backward_device,
                        output_clone_safe=not next_layer_inplace,
                        pinned_buffer_cache=pinned_output_buffer_cache,
                        pinned_buffer_key=layer_idx,
                    )
                    backward_relocation_layers_used.add(layer_name)
                    step_backward_relocation_count += 1
                elif strategy_name == "recompute":
                    if isinstance(x, torch.Tensor):
                        x = activation_checkpoint(layer, x, use_reentrant=False)
                        recompute_layers_used.add(layer_name)
                        step_recompute_count += 1
                    else:
                        warnings.append(
                            f"Layer '{layer_name}' requested recompute but runtime input is not a Tensor; falling back to retain"
                        )
                        x = layer(x)
                elif strategy_name == "checkpoint":
                    checkpoint_ctx = _saved_tensor_cpu_offload_context()
                    with checkpoint_ctx:
                        x = layer(x)
                    checkpoint_layers_used.add(layer_name)
                    step_checkpoint_count += 1
                else:
                    x = layer(x)

                # Prefetch policy (look-ahead): proactively transfer current output to
                # the next layer forward device when devices differ.
                if enable_prefetch and (layer_idx + 1) < len(layer_order):
                    next_layer_name, _next_layer = layer_order[layer_idx + 1]
                    next_target_device = plan.resolve_torch_device(
                        layer_name=next_layer_name,
                        gpu_id=gpu_id,
                        phase="forward",
                    )
                    current_output_device = _current_device(x)
                    if current_output_device is not None and current_output_device != next_target_device:
                        prefetch_start = time.perf_counter()
                        payload_bytes = _tensor_bytes(x)

                        if transfer_stream is not None and next_target_device.type == "cuda":
                            with torch.cuda.stream(transfer_stream):
                                prefetched_inputs[next_layer_name] = _to_device(x, next_target_device)
                        else:
                            prefetched_inputs[next_layer_name] = _to_device(x, next_target_device)

                        prefetch_ms = (time.perf_counter() - prefetch_start) * 1000.0
                        _ = TransferEvent(
                            layer=f"prefetch:{next_layer_name}",
                            src_device=str(current_output_device),
                            dst_device=str(next_target_device),
                            size_mb=payload_bytes / (1024**2),
                            time_ms=prefetch_ms,
                        )

                        prefetch_layers_used.add(next_layer_name)
                        step_prefetch_bytes += payload_bytes
                        step_prefetch_events += 1

            if isinstance(x, torch.Tensor) and x.device.type == "cuda":
                torch.cuda.synchronize(gpu_id)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            backward_start = time.perf_counter()
            loss, quality_metric_name, quality_metric_value = _compute_objective(x, target_data=target_data)
            if quality_metric_name_used is None and quality_metric_name is not None:
                quality_metric_name_used = quality_metric_name
            loss_value = float(loss.detach().cpu().item())
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            opt_start = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            optimizer_ms = (time.perf_counter() - opt_start) * 1000.0

            total_ms = (time.perf_counter() - step_start) * 1000.0

            peak_gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated(gpu_id) / (1024**2)
                global_peak_gpu_mem_mb = max(global_peak_gpu_mem_mb, peak_gpu_mem_mb)

            step_trace = StepTrace(
                step=step_idx,
                total_ms=total_ms,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                optimizer_ms=optimizer_ms,
                transfer_count=step_transfer_events,
                transfer_total_mb=step_transfer_bytes / (1024**2),
                prefetch_count=step_prefetch_events,
                prefetch_total_mb=step_prefetch_bytes / (1024**2),
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                recompute_count=step_recompute_count,
                checkpoint_count=step_checkpoint_count,
                backward_relocation_count=step_backward_relocation_count,
                loss_value=loss_value,
                task_metric_value=quality_metric_value,
            )
            step_traces.append(step_trace)

            total_transfer_bytes += step_transfer_bytes
            total_transfer_events += step_transfer_events
            total_prefetch_bytes += step_prefetch_bytes
            total_prefetch_events += step_prefetch_events
    finally:
        run_duration_sec = max(0.0, time.perf_counter() - run_start)
        energy_monitor.stop()

    avg_power_w = float(energy_monitor.get_avg_power())
    total_energy_j = float(avg_power_w * run_duration_sec) if avg_power_w > 0 else 0.0
    if total_energy_j <= 0.0:
        warnings.append(
            "Observed energy could not be measured (power monitor unavailable or zero readings)"
        )

    if unsupported_checkpoint_layers:
        warnings.append(
            "Checkpoint strategy requested but runtime currently supports only recompute; falling back to retain for: "
            f"{sorted(unsupported_checkpoint_layers)}"
        )

    avg_step_ms = sum(s.total_ms for s in step_traces) / len(step_traces)
    loss_values = [s.loss_value for s in step_traces]
    initial_loss = float(loss_values[0]) if loss_values else 0.0
    final_loss = float(loss_values[-1]) if loss_values else 0.0
    min_loss = float(min(loss_values)) if loss_values else 0.0
    loss_delta = float(final_loss - initial_loss) if loss_values else 0.0
    quality_values = [float(s.task_metric_value) for s in step_traces if s.task_metric_value is not None and math.isfinite(float(s.task_metric_value))]
    initial_quality_metric = float(quality_values[0]) if quality_values else None
    final_quality_metric = float(quality_values[-1]) if quality_values else None
    best_quality_metric = float(max(quality_values)) if quality_values else None
    quality_metric_delta = (
        float(final_quality_metric - initial_quality_metric)
        if initial_quality_metric is not None and final_quality_metric is not None
        else None
    )

    return HybridExecutionResult(
        status="ok",
        steps=steps,
        avg_step_ms=avg_step_ms,
        avg_power_w=avg_power_w,
        total_energy_j=total_energy_j,
        energy_source=energy_source,
        total_transfer_mb=total_transfer_bytes / (1024**2),
        total_transfer_events=total_transfer_events,
        total_prefetch_mb=total_prefetch_bytes / (1024**2),
        total_prefetch_events=total_prefetch_events,
        peak_gpu_mem_mb=global_peak_gpu_mem_mb,
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        loss_delta=loss_delta,
        prefetch_layers=sorted(prefetch_layers_used),
        recompute_layers=sorted(recompute_layers_used),
        checkpoint_layers=sorted(checkpoint_layers_used),
        backward_relocation_layers=sorted(backward_relocation_layers_used),
        unsupported_checkpoint_layers=sorted(unsupported_checkpoint_layers),
        warnings=warnings,
        per_step=step_traces,
        quality_metric_name=quality_metric_name_used,
        initial_quality_metric=initial_quality_metric,
        final_quality_metric=final_quality_metric,
        best_quality_metric=best_quality_metric,
        quality_metric_delta=quality_metric_delta,
    )
