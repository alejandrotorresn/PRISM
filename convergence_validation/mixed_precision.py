"""FP32 master-weight mixed precision for BF16 training.

Why this exists
----------------
`src/models/factory.py::_build_model` casts the entire model (parameters and
BatchNorm buffers) to BF16 with no FP32 copy anywhere, and there is no
GradScaler/autocast anywhere in the repo. For the short profiling runs that
exist today (a handful of iterations reusing one batch) this is harmless.
For REAL multi-epoch training, it is a real risk: a plain SGD update
`param -= lr * grad` can silently round to a no-op once `lr * grad` is small
relative to BF16's 7-bit mantissa ULP at the parameter's magnitude
("weight-update underflow"). If that happens, a BF16 convergence curve could
look artificially worse than FP32 for a reason that has nothing to do with
PRISM's CPU/GPU placement — exactly the kind of confound the committee is
asking this experiment to rule out.

Note this is *not* the same problem GradScaler solves for FP16: BF16 keeps
FP32's 8-bit exponent range, so BF16 gradients do not underflow to zero the
way FP16 gradients can. The risk here is specifically in the *weight update*
magnitude, not gradient magnitude, which is why the standard fix is a FP32
"master copy" of the weights (classic mixed-precision training recipe, e.g.
as used by NVIDIA's AMP / fp16 training literature), not loss scaling.

How it works
------------
1. The model's own parameters stay in BF16 and are used for forward/backward
   (this is still what is being validated: does PRISM's *placement* hurt
   convergence, not "can we avoid mixed precision altogether").
2. A separate FP32 "master" copy of every parameter is the actual target of
   `optimizer.step()`.
3. Each step: forward+backward in BF16 -> copy the resulting BF16 gradients
   into the FP32 master copy -> `optimizer.step()` updates the FP32 master
   (full precision update, no underflow) -> the updated master is cast back
   to BF16 and copied into the model for the next forward pass.

This file does not modify `src/models/factory.py`; it wraps whatever model
that module already builds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


class Bf16MixedPrecisionWrapper:
    """Owns the FP32 master parameters for a BF16 model and keeps the two in
    sync across the zero_grad -> backward -> step cycle."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.master_params: list[torch.Tensor] = [
            p.detach().clone().float().requires_grad_(True) for p in model.parameters()
        ]

    def build_optimizer(self, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any) -> torch.optim.Optimizer:
        """The optimizer must be built over `master_params`, not
        `model.parameters()` — the whole point is that the update happens in
        FP32."""
        return optimizer_cls(self.master_params, **kwargs)

    def zero_grad(self) -> None:
        for p in self.model.parameters():
            p.grad = None
        for p in self.master_params:
            p.grad = None

    def sync_grads_to_master(self) -> None:
        """Call after `loss.backward()`, before `optimizer.step()`."""
        for master, model_param in zip(self.master_params, self.model.parameters()):
            if model_param.grad is None:
                continue
            grad_fp32 = model_param.grad.detach().float()
            master.grad = grad_fp32 if master.grad is None else master.grad.copy_(grad_fp32)

    def sync_master_to_model(self) -> None:
        """Call after `optimizer.step()`, before the next forward pass."""
        with torch.no_grad():
            for master, model_param in zip(self.master_params, self.model.parameters()):
                model_param.copy_(master.to(model_param.dtype))


@dataclass
class TrainableModel:
    """Normalizes FP32 (plain) and BF16 (mixed-precision) training behind one
    interface, so `all_gpu_loop.py` / `prism_loop.py` do not need an
    if/else branch per precision in their step logic.

    Usage (identical for fp32 and bf16):

        trainable = build_trainable_model(model, precision="bf16", optimizer_cls=torch.optim.SGD, lr=0.01)
        trainable.zero_grad()
        out = trainable.model(...)
        loss = ...
        loss.backward()
        trainable.step()
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    _wrapper: Bf16MixedPrecisionWrapper | None = field(default=None)

    def zero_grad(self) -> None:
        if self._wrapper is not None:
            self._wrapper.zero_grad()
        else:
            self.optimizer.zero_grad()

    def step(self) -> None:
        if self._wrapper is not None:
            self._wrapper.sync_grads_to_master()
            self.optimizer.step()
            self._wrapper.sync_master_to_model()
        else:
            self.optimizer.step()


def optimizer_cls_from_name(name: str) -> type[torch.optim.Optimizer]:
    """Resolve a CLI-friendly optimizer name to a torch.optim class.

    Plain SGD at a CNN-scale learning rate (0.01) is appropriate for the
    vision models, which are only lightly fine-tuned from their torchvision
    pretrained weights. It is NOT appropriate for the pretrained language
    models: testing showed distilgpt2's loss exploding from 6.2 to 10.9
    (near-random, ln(vocab_size) ~= 10.8) within 2 SGD steps at lr=0.01.
    AdamW at a much smaller learning rate (~5e-5, standard for transformer
    fine-tuning) keeps the loss stable and slowly improving instead. This is
    why `run_convergence_study.sh` picks a different (optimizer, lr) default
    per model family rather than using one global value.
    """
    if name == "adamw":
        return torch.optim.AdamW
    if name == "sgd":
        return torch.optim.SGD
    raise ValueError(f"Unsupported optimizer name: {name!r} (expected 'sgd' or 'adamw')")


def build_trainable_model(
    model: nn.Module,
    precision: str,
    optimizer_cls: type[torch.optim.Optimizer],
    **optimizer_kwargs: Any,
) -> TrainableModel:
    """Build a `TrainableModel` for `precision` in {"fp32", "bf16"}.

    For "fp32", this is exactly `optimizer_cls(model.parameters(), **kwargs)`
    with no extra machinery — identical to what the rest of the repo already
    does. For "bf16", it wraps `model` with `Bf16MixedPrecisionWrapper` so the
    optimizer actually updates FP32 master weights (see module docstring).
    """
    if precision == "bf16":
        wrapper = Bf16MixedPrecisionWrapper(model)
        optimizer = wrapper.build_optimizer(optimizer_cls, **optimizer_kwargs)
        return TrainableModel(model=model, optimizer=optimizer, _wrapper=wrapper)

    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    return TrainableModel(model=model, optimizer=optimizer, _wrapper=None)
