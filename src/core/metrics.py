from typing import Any

import torch
import torch.nn as nn


def get_tensor_size_recursive(data: Any) -> int:
    size = 0
    try:
        if data is None:
            return 0
        if isinstance(data, torch.Tensor):
            size += data.numel() * data.element_size()
        elif isinstance(data, (tuple, list)):
            for item in data:
                size += get_tensor_size_recursive(item)
        elif isinstance(data, dict):
            for v in data.values():
                size += get_tensor_size_recursive(v)
        elif hasattr(data, "to_tuple"):
            size += get_tensor_size_recursive(data.to_tuple())
    except Exception:
        pass
    return int(size)


def _numel(t: Any) -> int:
    return t.numel() if hasattr(t, "numel") else 0


def estimate_flops(module: nn.Module, inputs: Any, output: Any) -> float:
    try:
        in_t = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else inputs
        if not isinstance(in_t, torch.Tensor):
            return 0.0

        if isinstance(module, nn.Conv2d) and isinstance(output, torch.Tensor):
            try:
                Cin = module.in_channels
                Cout = module.out_channels
                Kx, Ky = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                Hout, Wout = output.shape[2], output.shape[3]
                return 2.0 * Cout * Hout * Wout * (Cin // module.groups * Kx * Ky)
            except (IndexError, AttributeError, RuntimeError):
                return 0.0

        if isinstance(module, nn.Linear):
            try:
                in_f = module.in_features
                out_f = module.out_features
                positions = int(torch.tensor(in_t.shape[:-1]).prod().item()) if in_t is not None else 1
                return 2.0 * positions * in_f * out_f
            except (IndexError, AttributeError, RuntimeError, ValueError):
                return 0.0

        if isinstance(module, (nn.ReLU, nn.GELU)):
            try:
                return float(_numel(in_t))
            except (AttributeError, RuntimeError):
                return 0.0

        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            try:
                return 5.0 * float(_numel(in_t))
            except (AttributeError, RuntimeError):
                return 0.0

        module_name = module.__class__.__name__.lower()
        if "attention" in module_name and "multi" in module_name:
            try:
                B = in_t.shape[0]
                S = in_t.shape[1] if in_t.ndim >= 3 else 1
                d = in_t.shape[-1]
                return 4.0 * B * S * (d * d) + 2.0 * B * (S * S) * d
            except (IndexError, AttributeError, RuntimeError):
                return 0.0

        return 0.0

    except Exception:
        return 0.0
