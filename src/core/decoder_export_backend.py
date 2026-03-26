from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn


EXPORT_TRACE_SOURCE = "torch_export_decoder_only"


@dataclass
class ExportTraceContext:
    graph_module: fx.GraphModule
    node_layer_names: Dict[str, str]
    trace_source: str = EXPORT_TRACE_SOURCE


def _collect_leaf_module_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, module in model.named_modules()
        if name and len(list(module.children())) == 0
    }


def _extract_leaf_from_attr_target(target: str, leaf_names: set[str]) -> str | None:
    parts = target.split(".")
    for idx in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:idx])
        if candidate in leaf_names:
            return candidate
    return None


def _extract_leaf_from_module_stack(node: fx.Node, leaf_names: set[str]) -> str | None:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return None

    best_match: str | None = None
    best_depth = -1
    for value in stack.values():
        if not isinstance(value, tuple) or not value:
            continue
        module_path = str(value[0])
        if not module_path or module_path not in leaf_names:
            continue
        depth = module_path.count(".")
        if depth > best_depth:
            best_match = module_path
            best_depth = depth
    return best_match


def is_decoder_only_export_candidate(model: nn.Module, input_data: Any) -> bool:
    try:
        from transformers import PreTrainedModel
    except Exception:
        return False

    if not isinstance(model, PreTrainedModel):
        return False

    config = getattr(model, "config", None)
    if config is None or getattr(config, "is_encoder_decoder", False):
        return False

    if not isinstance(input_data, dict) or "input_ids" not in input_data:
        return False

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if get_output_embeddings is None:
        return False

    try:
        return get_output_embeddings() is not None
    except Exception:
        return False


def _prepare_export_inputs(input_data: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(input_data, tuple):
        return input_data, {}
    if isinstance(input_data, dict):
        return (), dict(input_data)
    return (input_data,), {}


def build_export_node_layer_names(
    graph_module: fx.GraphModule,
    model: nn.Module,
) -> Dict[str, str]:
    leaf_names = _collect_leaf_module_names(model)
    node_layer_names: Dict[str, str] = {}

    for node in graph_module.graph.nodes:
        layer_name: str | None = None
        if node.op == "call_module":
            target = str(node.target)
            if target in leaf_names:
                layer_name = target
        elif node.op == "get_attr":
            layer_name = _extract_leaf_from_attr_target(str(node.target), leaf_names)
        else:
            layer_name = _extract_leaf_from_module_stack(node, leaf_names)

        if layer_name:
            node_layer_names[node.name] = layer_name

    return node_layer_names


def try_export_decoder_only_trace(
    model: nn.Module,
    input_data: Any,
) -> Optional[ExportTraceContext]:
    if not is_decoder_only_export_candidate(model, input_data):
        return None

    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None) if config is not None else None
    try:
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        export_args, export_kwargs = _prepare_export_inputs(input_data)
        graph_module = torch.export.export(model, export_args, export_kwargs).module()
    except Exception:
        return None
    finally:
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache

    node_layer_names = build_export_node_layer_names(graph_module, model)
    if not node_layer_names:
        return None

    return ExportTraceContext(graph_module=graph_module, node_layer_names=node_layer_names)