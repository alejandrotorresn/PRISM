from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if hasattr(value, "to_tuple"):
        try:
            return _first_tensor(value.to_tuple())
        except Exception:
            return None
    return None


def extract_primary_output_tensor(out: Any) -> torch.Tensor | None:
    if hasattr(out, "loss") and out.loss is not None and isinstance(out.loss, torch.Tensor):
        return out.loss

    for attr in ["logits", "last_hidden_state", "pooler_output"]:
        value = getattr(out, attr, None)
        if isinstance(value, torch.Tensor):
            return value

    return _first_tensor(out)


def extract_classification_logits(out: Any) -> torch.Tensor | None:
    tensor = extract_primary_output_tensor(out)
    if tensor is None:
        return None
    if tensor.ndim == 2 and tensor.shape[1] >= 2:
        return tensor
    return None


def _flatten_batch_features(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.reshape(tensor.shape[0], 1)
        else:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.is_complex():
            tensor = torch.view_as_real(tensor).reshape(tensor.shape[0], -1)
        return tensor.float().cpu()
    if isinstance(value, (tuple, list)):
        parts = [_flatten_batch_features(item) for item in value]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        batch_size = min(part.shape[0] for part in parts)
        cropped = [part[:batch_size] for part in parts]
        return torch.cat(cropped, dim=1)
    if isinstance(value, dict):
        parts = [_flatten_batch_features(item) for item in value.values()]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        batch_size = min(part.shape[0] for part in parts)
        cropped = [part[:batch_size] for part in parts]
        return torch.cat(cropped, dim=1)
    return None


def build_deterministic_classification_targets(input_data: Any, num_classes: int) -> torch.Tensor | None:
    if num_classes < 2:
        return None

    features = _flatten_batch_features(input_data)
    if features is None or features.numel() == 0:
        return None

    stride = max(1, features.shape[1] // 64)
    checksum = features[:, ::stride].sum(dim=1)
    mean = features.mean(dim=1)
    std = features.std(dim=1, correction=0)
    raw = (mean * 1_000_003.0 + std * 100_019.0 + checksum * 10_007.0).abs()
    return torch.remainder(torch.round(raw).long(), num_classes)


def compute_stable_surrogate_loss(out: Any) -> torch.Tensor:
    if hasattr(out, "loss") and out.loss is not None and isinstance(out.loss, torch.Tensor):
        return out.loss

    tensor = extract_primary_output_tensor(out)
    if tensor is None:
        return torch.tensor(0.0, requires_grad=True)
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    return tensor.float().square().mean()


def compute_training_objective(
    out: Any,
    target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str | None, float | None]:
    model_loss = getattr(out, "loss", None)
    logits = extract_classification_logits(out)
    if target is not None and logits is not None and target.ndim == 1 and target.shape[0] == logits.shape[0]:
        target_device = target.to(logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), target_device)
        accuracy = float((logits.argmax(dim=1) == target_device).float().mean().item())
        return loss, "accuracy", accuracy

    raw_logits = getattr(out, "logits", None)
    if (
        target is not None
        and isinstance(raw_logits, torch.Tensor)
        and raw_logits.ndim == 3
        and target.ndim == 2
        and raw_logits.shape[0] == target.shape[0]
        and raw_logits.shape[1] == target.shape[1]
    ):
        shift_logits = raw_logits[:, :-1, :].contiguous().float()
        shift_labels = target[:, 1:].to(raw_logits.device, dtype=torch.long).contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        valid_mask = shift_labels.ne(-100)
        token_accuracy: float | None = None
        if bool(valid_mask.any()):
            predictions = shift_logits.argmax(dim=-1)
            token_accuracy = float(predictions[valid_mask].eq(shift_labels[valid_mask]).float().mean().item())

        if isinstance(model_loss, torch.Tensor):
            return model_loss, "token_accuracy", token_accuracy
        return loss, "token_accuracy", token_accuracy

    return compute_stable_surrogate_loss(out), None, None