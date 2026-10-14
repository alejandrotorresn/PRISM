import pytest
import torch
import torch.nn as nn

from src.runtime.device_plan import DevicePlan, collect_leaf_module_names, validate_plan_coverage
from src.runtime.leaf_modules import iter_leaf_modules


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.ln = nn.LayerNorm(8)


def test_mha_is_one_leaf_without_its_children():
    names = [n for n, _ in iter_leaf_modules(_Block())]
    assert names == ["attn", "ln"]
    assert collect_leaf_module_names(_Block()) == ["attn", "ln"]


def test_unplanned_layer_raises_by_default_and_debug_flag_restores_cpu():
    plan = DevicePlan({"a": "GPU"}, {"a": "GPU"}, [], [], [])
    with pytest.raises(KeyError):
        plan.resolve_torch_device("missing")
    plan.allow_unplanned = True
    assert plan.resolve_torch_device("missing").type == "cpu"


def test_coverage_validation_raises_on_missing_layers_unless_allowed():
    model = _Block()
    plan = DevicePlan({"ln": "GPU"}, {"ln": "GPU"}, [], [], [])
    with pytest.raises(ValueError):
        validate_plan_coverage(model, plan)
    assert validate_plan_coverage(model, plan, allow_unplanned=True)


def test_vit_attention_is_covered_by_leaf_names():
    tv = pytest.importorskip("torchvision")
    model = tv.models.vit_b_16()
    names = set(collect_leaf_module_names(model))
    assert "encoder.layers.encoder_layer_0.self_attention" in names
    assert "encoder.layers.encoder_layer_0.self_attention.out_proj" not in names
