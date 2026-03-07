import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from core.io_artifacts import write_csv_rows

logger = logging.getLogger(__name__)


def _params_mb_for_module(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    try:
        params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
        return float(params_bytes) / (1024**2)
    except Exception:
        return 0.0


def _activation_mb_from_layer_stats(layer_stats: Dict[str, Dict[str, Any]], name: str) -> float:
    try:
        out_bytes = layer_stats.get(name, {}).get("output_bytes", 0)
        return float(out_bytes) / (1024**2)
    except Exception:
        return 0.0


def _node_label(node) -> str:
    if node.op == "call_module":
        return str(node.target)
    if node.op in {"call_function", "call_method"}:
        return str(node.target)
    if node.op == "placeholder":
        return str(node.target)
    return node.name


def _shape_from_meta(node) -> str:
    # Best-effort shape extraction if metadata exists.
    try:
        tmeta = node.meta.get("tensor_meta")
        if tmeta is None:
            return ""
        shape = getattr(tmeta, "shape", None)
        if shape is None:
            return ""
        return "x".join(str(int(d)) for d in shape)
    except Exception:
        return ""


def _build_fallback_graph(model: nn.Module, layer_stats: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    leaf_modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            leaf_modules.append((name, module))

    for idx, (name, module) in enumerate(leaf_modules):
        node_id = f"n{idx}"
        nodes.append({
            "node_id": node_id,
            "node_name": name,
            "op_type": module.__class__.__name__,
            "topo_index": idx,
            "params_mb": _params_mb_for_module(module),
            "activ_out_mb": _activation_mb_from_layer_stats(layer_stats, name),
            "trace_source": "fallback_leaf_modules",
        })

    for idx in range(len(nodes) - 1):
        src = nodes[idx]
        dst = nodes[idx + 1]
        edges.append({
            "src_id": src["node_id"],
            "dst_id": dst["node_id"],
            "tensor_mb": src["activ_out_mb"],
            "tensor_shape": "",
            "producer_name": src["node_name"],
            "consumer_name": dst["node_name"],
            "trace_source": "fallback_leaf_modules",
        })

    return nodes, edges, "fallback_leaf_modules"


def _maybe_enrich_node_meta(gm, input_data: Any) -> None:
    # Optional metadata pass for tensor shapes; failures are non-fatal.
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        param_device = "cpu"
        first_param = next(gm.parameters(), None)
        if first_param is not None:
            param_device = str(first_param.device)

        input_device = "cpu"
        if isinstance(input_data, torch.Tensor):
            input_device = str(input_data.device)
        elif isinstance(input_data, dict):
            for v in input_data.values():
                if isinstance(v, torch.Tensor):
                    input_device = str(v.device)
                    break

        if param_device != input_device:
            # Avoid ShapeProp device mismatch noise in GPU-only profiling mode.
            return

        if isinstance(input_data, dict):
            ShapeProp(gm).propagate(**input_data)
        else:
            ShapeProp(gm).propagate(input_data)
    except Exception:
        pass


def _build_fx_graph(model: nn.Module, layer_stats: Dict[str, Dict[str, Any]], input_data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    gm = symbolic_trace(model)
    _maybe_enrich_node_meta(gm, input_data)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_id_map: Dict[str, str] = {}
    activation_map: Dict[str, float] = {}
    name_map: Dict[str, str] = {}

    topo_index = 0
    for node in gm.graph.nodes:
        if node.op == "output":
            continue

        node_id = f"n{topo_index}"
        node_id_map[node.name] = node_id

        label = _node_label(node)
        name_map[node.name] = label

        module = None
        op_type = node.op
        if node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            op_type = module.__class__.__name__ if module is not None else "Module"

        activ_out_mb = _activation_mb_from_layer_stats(layer_stats, label)
        activation_map[node.name] = activ_out_mb

        nodes.append({
            "node_id": node_id,
            "node_name": label,
            "op_type": op_type,
            "topo_index": topo_index,
            "params_mb": _params_mb_for_module(module),
            "activ_out_mb": activ_out_mb,
            "trace_source": "torch_fx",
        })
        topo_index += 1

    for dst in gm.graph.nodes:
        if dst.op == "output":
            continue
        dst_id = node_id_map.get(dst.name)
        if dst_id is None:
            continue

        for src in dst.all_input_nodes:
            src_id = node_id_map.get(src.name)
            if src_id is None:
                continue
            edges.append({
                "src_id": src_id,
                "dst_id": dst_id,
                "tensor_mb": activation_map.get(src.name, 0.0),
                "tensor_shape": _shape_from_meta(src),
                "producer_name": name_map.get(src.name, src.name),
                "consumer_name": name_map.get(dst.name, dst.name),
                "trace_source": "torch_fx",
            })

    return nodes, edges, "torch_fx"


def export_graph_artifacts(
    model: nn.Module,
    model_name: str,
    output_dir: str,
    input_data: Any,
    layer_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    include_records: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    layer_stats = layer_stats or {}

    nodes_path = os.path.join(output_dir, f"{model_name}_graph_nodes.csv")
    edges_path = os.path.join(output_dir, f"{model_name}_graph_edges.csv")

    try:
        nodes, edges, source = _build_fx_graph(model, layer_stats, input_data)
    except Exception as e:
        logger.warning(f"torch.fx graph extraction failed, using fallback graph. Reason: {e}")
        nodes, edges, source = _build_fallback_graph(model, layer_stats)

    write_csv_rows(nodes_path, nodes)
    write_csv_rows(edges_path, edges)

    logger.info(
        "Saved graph artifacts: %s nodes, %s edges (%s)",
        len(nodes),
        len(edges),
        source,
    )

    return {
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "trace_source": source,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "nodes": nodes if include_records else None,
        "edges": edges if include_records else None,
    }
