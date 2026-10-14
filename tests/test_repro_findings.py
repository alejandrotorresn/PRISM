"""Reproduction tests for verified findings (fix/PLAN_MEJORA_PRISM.md, task 0.3).

Each test asserts the CORRECT behavior and is marked xfail(strict=True): it
fails today for the documented reason, and turns into an XPASS error the
moment the fix lands, forcing the marker to be removed.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.graph_extractor import _build_fx_graph
from src.runtime.device_plan import DevicePlan
from src.runtime.hybrid_executor import _run_layer_with_dual_placement


class _TwoRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x + x


@pytest.mark.xfail(strict=True, reason="H2: capa ausente del plan cae a CPU sin aviso")
def test_h2_unplanned_layer_is_not_silently_cpu():
    plan = DevicePlan({"a": "GPU"}, {"a": "GPU"}, [], [], [])
    with pytest.raises((KeyError, ValueError)):
        plan.resolve_torch_device("missing_layer")


@pytest.mark.xfail(strict=True, reason="H12: node_name usa str(node.target) y colisiona")
def test_h12_fx_node_names_are_unique():
    nodes, _, _ = _build_fx_graph(_TwoRelu(), {}, torch.randn(2, 8))
    names = [n["node_name"] for n in nodes]
    assert len(names) == len(set(names))


@pytest.mark.xfail(strict=True, reason="H5b: Dropout en dual placement lanza RuntimeError por mismatch")
def test_h5b_dropout_dual_placement_does_not_raise():
    layer = nn.Dropout(0.5).train()
    x = torch.randn(64, 64, requires_grad=True)
    cpu = torch.device("cpu")
    out = _run_layer_with_dual_placement(layer, "drop", x, cpu, cpu)
    out.sum().backward()


@pytest.mark.xfail(strict=True, reason="H5c: BatchNorm actualiza estadisticas dos veces en dual placement")
def test_h5c_batchnorm_stats_updated_once():
    layer = nn.BatchNorm1d(8).train()
    x = torch.randn(16, 8, requires_grad=True)
    cpu = torch.device("cpu")
    out = _run_layer_with_dual_placement(layer, "bn", x, cpu, cpu)
    out.sum().backward()
    assert int(layer.num_batches_tracked) == 1
