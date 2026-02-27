import io
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from core import precision_policy as pp


def test_evaluate_precision_policy_fp16_unsupported():
    result = pp.evaluate_precision_execution_policy(
        "fp16", {"fp16_accelerated": False, "bf16_accelerated": True}
    )
    assert result["allowed"] is False
    assert result["cpu_precision_executed"] == "fp16_requested_isa_unsupported"
    assert result["status"] == "skipped_unsupported_precision"


def test_evaluate_precision_policy_bf16_unsupported():
    result = pp.evaluate_precision_execution_policy(
        "bf16", {"fp16_accelerated": True, "bf16_accelerated": False}
    )
    assert result["allowed"] is False
    assert result["cpu_precision_executed"] == "bf16_requested_isa_unsupported"
    assert result["status"] == "skipped_unsupported_precision"


def test_evaluate_precision_policy_supported():
    result = pp.evaluate_precision_execution_policy(
        "fp32", {"fp16_accelerated": False, "bf16_accelerated": False}
    )
    assert result["allowed"] is True
    assert result["cpu_precision_executed"] == "fp32"
    assert result["status"] == "ready"


def test_get_cpu_instruction_flags_non_linux(monkeypatch):
    monkeypatch.setattr(pp.platform, "system", lambda: "Darwin")
    assert pp.get_cpu_instruction_flags() == []


def test_get_cpu_instruction_flags_linux_parse(monkeypatch):
    cpuinfo = "flags\t: sse2 avx512_fp16 amx_tile\n"
    monkeypatch.setattr(pp.platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(cpuinfo))
    flags = pp.get_cpu_instruction_flags()
    assert "sse2" in flags
    assert "avx512_fp16" in flags
    assert "amx_tile" in flags


def test_probe_cpu_precision_support(monkeypatch):
    monkeypatch.setattr(
        pp,
        "get_cpu_instruction_flags",
        lambda: ["avx512_bf16", "amx_bf16", "amx_tile", "xsave"],
    )
    result = pp.probe_cpu_precision_support()
    assert result["bf16_accelerated"] is True
    assert result["fp16_accelerated"] is False


def test_get_cpu_fp16_support_info_smoke_ok_without_isa(monkeypatch):
    monkeypatch.setattr(pp.platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("flags\t: sse2\n"))
    monkeypatch.setattr(pp.torch, "mm", lambda a, b: torch.zeros((64, 64), dtype=torch.float16))

    result = pp.get_cpu_fp16_support_info()
    assert result["supported"] is True
    assert result["smoke_test_ok"] is True
    assert result["isa_avx512_fp16"] is False
    assert "not detected" in result["reason"]


def test_get_cpu_fp16_support_info_smoke_fail(monkeypatch):
    monkeypatch.setattr(pp.platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("flags\t: avx512_fp16\n"))

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pp.torch, "mm", _raise)

    result = pp.get_cpu_fp16_support_info()
    assert result["supported"] is False
    assert result["smoke_test_ok"] is False
    assert "smoke test failed" in result["reason"]


def test_extract_loss_for_preflight_paths():
    out_loss = SimpleNamespace(loss=torch.tensor(3.0))
    assert pp._extract_loss_for_preflight(out_loss).item() == 3.0

    out_logits = SimpleNamespace(logits=torch.ones((2, 2)))
    assert pp._extract_loss_for_preflight(out_logits).item() == 4.0

    out_tuple = (torch.ones((3,)),)
    assert pp._extract_loss_for_preflight(out_tuple).item() == 3.0


def test_build_mini_input_for_cpu_fp16_tensor_and_dict():
    tensor_input = torch.randn((4, 5), dtype=torch.float32)
    mini_tensor = pp._build_mini_input_for_cpu_fp16(tensor_input)
    assert mini_tensor.shape[0] == 1
    assert mini_tensor.dtype == torch.float16

    dict_input = {
        "float_data": torch.randn((3, 2), dtype=torch.float32),
        "token_ids": torch.randint(0, 10, (3, 2), dtype=torch.long),
        "meta": "x",
    }
    mini_dict = pp._build_mini_input_for_cpu_fp16(dict_input)
    assert mini_dict["float_data"].shape[0] == 1
    assert mini_dict["float_data"].dtype == torch.float16
    assert mini_dict["token_ids"].shape[0] == 1
    assert mini_dict["token_ids"].dtype == torch.long
    assert mini_dict["meta"] == "x"


def test_run_cpu_fp16_model_preflight_success():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    input_data = torch.randn((2, 4), dtype=torch.float32)

    result = pp.run_cpu_fp16_model_preflight(model, input_data, timeout_safety_factor=1.0)
    assert result["ok"] is True
    assert "succeeded" in result["reason"]


def test_run_cpu_fp16_model_preflight_exception_path():
    class BoomModel(nn.Module):
        def forward(self, x):
            raise RuntimeError("intentional")

    model = BoomModel()
    input_data = torch.randn((1, 4), dtype=torch.float32)

    result = pp.run_cpu_fp16_model_preflight(model, input_data, timeout_safety_factor=1.0)
    assert result["ok"] is False
    assert "failed with exception" in result["reason"]
