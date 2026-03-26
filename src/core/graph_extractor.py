import logging
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

try:
    from .decoder_export_backend import EXPORT_TRACE_SOURCE, ExportTraceContext, try_export_decoder_only_trace
    from .io_artifacts import write_csv_rows
except ImportError:
    from core.decoder_export_backend import EXPORT_TRACE_SOURCE, ExportTraceContext, try_export_decoder_only_trace
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


def _build_export_grouped_graph(
    model: nn.Module,
    layer_stats: Dict[str, Dict[str, Any]],
    trace_ctx: ExportTraceContext,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    graph_module = trace_ctx.graph_module
    node_layer_names = trace_ctx.node_layer_names

    adjacency: Dict[str, List[str]] = defaultdict(list)
    first_seen_idx: Dict[str, int] = {}
    source_nodes_by_layer: Dict[str, List[str]] = defaultdict(list)

    topo_index = 0
    for node in graph_module.graph.nodes:
        if node.op == "output":
            continue
        layer_name = node_layer_names.get(node.name)
        if layer_name and layer_name not in first_seen_idx:
            first_seen_idx[layer_name] = topo_index
        if layer_name:
            source_nodes_by_layer[layer_name].append(node.name)
        topo_index += 1

    for dst in graph_module.graph.nodes:
        if dst.op == "output":
            continue
        for src in dst.all_input_nodes:
            adjacency[src.name].append(dst.name)

    ordered_layers = sorted(first_seen_idx, key=first_seen_idx.get)
    nodes: List[Dict[str, Any]] = []
    layer_node_ids: Dict[str, str] = {}
    for idx, layer_name in enumerate(ordered_layers):
        node_id = f"n{idx}"
        layer_node_ids[layer_name] = node_id
        module = model.get_submodule(layer_name)
        nodes.append({
            "node_id": node_id,
            "node_name": layer_name,
            "op_type": module.__class__.__name__,
            "topo_index": first_seen_idx[layer_name],
            "params_mb": _params_mb_for_module(module),
            "activ_out_mb": _activation_mb_from_layer_stats(layer_stats, layer_name),
            "trace_source": EXPORT_TRACE_SOURCE,
        })

    edges: List[Dict[str, Any]] = []
    seen_edges: set[Tuple[str, str]] = set()
    for start_layer, start_nodes in source_nodes_by_layer.items():
        queue = deque((node_name, 0) for node_name in start_nodes)
        visited = set(start_nodes)
        nearest_depth: int | None = None
        reached_layers: set[str] = set()

        while queue:
            current_name, depth = queue.popleft()
            if nearest_depth is not None and depth >= nearest_depth:
                continue
            for next_name in adjacency.get(current_name, []):
                if next_name in visited:
                    continue
                visited.add(next_name)
                next_layer = node_layer_names.get(next_name)
                next_depth = depth + 1
                if next_layer and next_layer != start_layer:
                    if nearest_depth is None or next_depth < nearest_depth:
                        nearest_depth = next_depth
                        reached_layers = {next_layer}
                    elif next_depth == nearest_depth:
                        reached_layers.add(next_layer)
                    continue
                queue.append((next_name, next_depth))

        for dst_layer in sorted(reached_layers, key=lambda name: first_seen_idx.get(name, 10**9)):
            edge_key = (start_layer, dst_layer)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "src_id": layer_node_ids[start_layer],
                "dst_id": layer_node_ids[dst_layer],
                "tensor_mb": _activation_mb_from_layer_stats(layer_stats, start_layer),
                "tensor_shape": "",
                "producer_name": start_layer,
                "consumer_name": dst_layer,
                "trace_source": EXPORT_TRACE_SOURCE,
            })

    return nodes, edges, EXPORT_TRACE_SOURCE


def export_graph_artifacts(
    model: nn.Module,
    model_name: str,
    output_dir: str,
    input_data: Any,
    layer_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    include_records: bool = False,
    allow_fallback_graph: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    layer_stats = layer_stats or {}

    nodes_path = os.path.join(output_dir, f"{model_name}_graph_nodes.csv")
    edges_path = os.path.join(output_dir, f"{model_name}_graph_edges.csv")

    try:
        nodes, edges, source = _build_fx_graph(model, layer_stats, input_data)
    except Exception as e:
        trace_ctx = try_export_decoder_only_trace(model, input_data)
        if trace_ctx is None:
            if not allow_fallback_graph:
                raise RuntimeError(
                    "Structured graph extraction failed and fallback_leaf_modules is disabled. "
                    "Use allow_fallback_graph=True only for explicit diagnostic runs."
                ) from e
            logger.warning(f"torch.fx graph extraction failed, using fallback graph. Reason: {e}")
            nodes, edges, source = _build_fallback_graph(model, layer_stats)
        else:
            logger.info("torch.fx graph extraction failed; using decoder-only export backend")
            nodes, edges, source = _build_export_grouped_graph(model, layer_stats, trace_ctx)

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
