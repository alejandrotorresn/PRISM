"""Single definition of "plan layer" shared by profiler, graph, plan and runtime.

torch.fx treats every torch.nn module (e.g. nn.MultiheadAttention) as a leaf,
even when it has children. A plain ``children() == 0`` test splits MHA into
``self_attention`` (never hooked, never planned) and ``out_proj`` (never called
as a module), so attention silently disappears from profiling and the plan.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import torch.nn as nn

# Modules executed as one unit although they own child modules.
OPAQUE_MODULE_TYPES: Tuple[type, ...] = (nn.MultiheadAttention,)


def iter_leaf_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    opaque_prefixes: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(p) for p in opaque_prefixes):
            continue
        if isinstance(module, OPAQUE_MODULE_TYPES):
            if name:
                opaque_prefixes.append(name + ".")
            yield name, module
        elif len(list(module.children())) == 0:
            yield name, module
