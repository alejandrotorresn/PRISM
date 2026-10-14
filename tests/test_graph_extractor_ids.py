import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.graph_extractor import _build_fx_graph, export_graph_artifacts


class _TwoRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x + x


def test_h12_fx_node_names_are_unique():
    nodes, _, _ = _build_fx_graph(_TwoRelu(), {}, torch.randn(2, 8))
    names = [n["node_name"] for n in nodes]
    assert len(names) == len(set(names))


def test_h13_edges_get_real_sizes_from_shape_prop():
    _, edges, _ = _build_fx_graph(_TwoRelu(), {}, torch.randn(2, 8))
    expected_mb = 2 * 8 * 4 / (1024**2)
    for e in edges:
        assert e["tensor_mb"] == expected_mb, e
        assert e["tensor_mb_source"] == "shape_prop"


def test_measured_layer_stats_take_precedence():
    stats = {"fc1": {"output_bytes": 1024 * 1024}}
    _, edges, _ = _build_fx_graph(_TwoRelu(), stats, torch.randn(2, 8))
    fc1_edges = [e for e in edges if e["producer_name"] == "fc1"]
    assert fc1_edges and all(e["tensor_mb"] == 1.0 and e["tensor_mb_source"] == "measured" for e in fc1_edges)


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)

    def forward(self, x):
        h = self.a(x)
        return self.c(self.b(h) + h)


def test_unknown_sizes_raise_in_strict_mode(tmp_path):
    # Input on a different device than params is not reproducible on CPU-only; simulate by
    # an untraceable size: no ShapeProp metadata when input_data is None.
    model = _Residual()
    with pytest.raises(ValueError):
        export_graph_artifacts(model, "m", str(tmp_path), None, strict_tensor_sizes=True)




def test_only_shared_modules_repeat_names_and_edge_count_matches_fx():
    from torch.fx import symbolic_trace

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return self.relu(self.fc(self.relu(x))) + x

    model = Shared()
    nodes, edges, _ = _build_fx_graph(model, {}, torch.randn(2, 4))
    dup = {n["node_name"] for n in nodes if [m["node_name"] for m in nodes].count(n["node_name"]) > 1}
    assert dup == {"relu"}
    gm = symbolic_trace(model)
    assert len(edges) == sum(len(n.all_input_nodes) for n in gm.graph.nodes if n.op != "output")
    assert all(e["tensor_mb"] > 0 for e in edges)
