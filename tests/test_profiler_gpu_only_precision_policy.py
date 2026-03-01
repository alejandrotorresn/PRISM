import os
import sys
import importlib
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

profiler = importlib.import_module("profiler")


def _make_args(precision: str, skip_cpu: bool, no_gpu: bool):
    return SimpleNamespace(
        precision=precision,
        skip_cpu=skip_cpu,
        no_gpu=no_gpu,
        cpu_instruction_flags=[],
        cpu_isa_probe={},
        cpu_precision_executed="",
        execution_status="ready",
        abort_profiling_due_to_isa=False,
        abort_profiling_reason="",
        cpu_fp16_supported=None,
        cpu_fp16_isa_avx512=None,
        cpu_fp16_smoke_test_ok=None,
        cpu_fp16_support_reason=None,
    )


def test_unsupported_fp16_cpu_abort_when_cpu_enabled(monkeypatch):
    args = _make_args(precision="fp16", skip_cpu=False, no_gpu=False)

    monkeypatch.setattr(profiler, "probe_cpu_precision_support", lambda: {})
    monkeypatch.setattr(
        profiler,
        "evaluate_precision_execution_policy",
        lambda precision, isa: {
            "allowed": False,
            "cpu_precision_executed": "fp16_requested_isa_unsupported",
            "reason": "missing avx512_fp16",
            "status": "skipped_unsupported_precision",
        },
    )
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        profiler,
        "get_cpu_fp16_support_info",
        lambda: {
            "supported": False,
            "isa_avx512_fp16": False,
            "smoke_test_ok": False,
            "reason": "no support",
        },
    )

    dtype = profiler._configure_precision(args)

    assert dtype == torch.float16
    assert args.abort_profiling_due_to_isa is True
    assert "avx512_fp16" in args.abort_profiling_reason


def test_unsupported_fp16_cpu_no_abort_when_gpu_only(monkeypatch):
    args = _make_args(precision="fp16", skip_cpu=True, no_gpu=False)

    monkeypatch.setattr(profiler, "probe_cpu_precision_support", lambda: {})
    monkeypatch.setattr(
        profiler,
        "evaluate_precision_execution_policy",
        lambda precision, isa: {
            "allowed": False,
            "cpu_precision_executed": "fp16_requested_isa_unsupported",
            "reason": "missing avx512_fp16",
            "status": "skipped_unsupported_precision",
        },
    )
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        profiler,
        "get_cpu_fp16_support_info",
        lambda: {
            "supported": False,
            "isa_avx512_fp16": False,
            "smoke_test_ok": False,
            "reason": "no support",
        },
    )

    dtype = profiler._configure_precision(args)

    assert dtype == torch.float16
    assert args.abort_profiling_due_to_isa is False
    assert args.execution_status == "ready"


def test_unsupported_bf16_gpu_only_uses_bf16_dtype(monkeypatch):
    args = _make_args(precision="bf16", skip_cpu=True, no_gpu=False)

    monkeypatch.setattr(profiler, "probe_cpu_precision_support", lambda: {})
    monkeypatch.setattr(
        profiler,
        "evaluate_precision_execution_policy",
        lambda precision, isa: {
            "allowed": False,
            "cpu_precision_executed": "bf16_requested_isa_unsupported",
            "reason": "missing bf16 isa",
            "status": "skipped_unsupported_precision",
        },
    )
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler, "cpu_supports_bf16", lambda: False)

    dtype = profiler._configure_precision(args)

    assert dtype == torch.bfloat16
    assert args.abort_profiling_due_to_isa is False
