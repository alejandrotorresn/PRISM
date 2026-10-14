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


import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_validation_module(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_repro", _REPO_ROOT / "validation" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.xfail(strict=True, reason="H1: medicion no-ok mueve todas las capas a CPU (hi = n)")
def test_h1_unmeasurable_plan_is_not_emptied(monkeypatch):
    mod = _load_validation_module("run_ilp_partition")
    monkeypatch.setattr(
        mod,
        "measure_real_peak_for_plan",
        lambda **kw: {"status": "trace_error", "real_reserved_mb": None, "real_allocated_mb": None},
    )

    class _Topo:
        def layer_gpu_weight(self, name):
            return 1.0

    fwd = {f"l{i}": "GPU" for i in range(5)}
    bwd = dict(fwd)
    args = Namespace(gpu_mem_budget_mb=1000.0)
    res = mod._validate_final_plan_against_real_overhead(
        fwd, bwd, None, None, args, _Topo(), object(), None
    )
    assert any(d == "GPU" for d in res["forward_assignment"].values())


@pytest.mark.xfail(strict=True, reason="H7: ilp_objective se usa como sim_time_ms")
def test_h7_objective_is_not_reported_as_simulated_ms(tmp_path):
    mod = _load_validation_module("run_hybrid_execution")
    csv = tmp_path / "plans.csv"
    csv.write_text("model,ilp_status,ilp_objective\nresnet50,optimal,123.0\n")
    args = Namespace(
        plan_sim_time_ms=None, plan_sim_energy_j=None, plan_source_csv=str(csv),
        plan_gpu_budget_mb=None, model="resnet50", precision="fp32", batch_size=8, plan_objective=None,
    )
    # config_dir con < 4 componentes: sin identidad model/optimizer/batch inferida
    ref = mod._resolve_sim_reference_values(args, Path("/cfg"))
    assert ref["sim_time_ms"] is None


@pytest.mark.xfail(strict=True, reason="H9: el clasificador OOM lee el log global de campaña")
def test_h9_oom_classifier_not_fed_global_campaign_log():
    text = (_REPO_ROOT / "scripts" / "run_experiments.sh").read_text()
    assert 'is_oom_failure_from_log "$LOG_FILE"' not in text


@pytest.mark.xfail(strict=True, reason="H11: symbolic_trace no puede trazar modelos HF (BERT)")
def test_h11_hf_bert_graph_is_traceable():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, vocab_size=100
    )
    model = transformers.BertModel(cfg).eval()
    _build_fx_graph(model, {}, torch.randint(0, 100, (2, 8)))
