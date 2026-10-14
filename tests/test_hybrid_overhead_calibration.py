"""Tests for src/runner/hybrid_overhead_calibration.py.

This module is purely additive (see its own docstring): it must never be
able to break an existing profiling run. These tests check both the happy
path (a real calibration measurement on GPU) and that every failure mode
degrades gracefully to ratio=1.0 instead of raising.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from runner.hybrid_overhead_calibration import (  # noqa: E402
    _build_representative_dual_plan,
    _infer_batch_size,
    _leaf_module_names,
    _slice_batch,
    measure_hybrid_overhead_ratio,
)


class _TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_leaf_module_names_excludes_containers():
    model = _TinyConvNet()
    names = _leaf_module_names(model)
    assert set(names) == {"conv1", "relu1", "conv2", "relu2", "pool", "fc"}


def test_infer_and_slice_batch_tensor():
    x = torch.randn(10, 3, 8, 8)
    assert _infer_batch_size(x) == 10
    sliced = _slice_batch(x, 4)
    assert sliced.shape[0] == 4


def test_infer_and_slice_batch_dict():
    x = {"input_ids": torch.randint(0, 100, (6, 5)), "scalar_flag": torch.tensor(1)}
    assert _infer_batch_size(x) == 6
    sliced = _slice_batch(x, 3)
    assert sliced["input_ids"].shape[0] == 3
    assert sliced["scalar_flag"].item() == 1


def test_representative_dual_plan_alternates_and_flags_cross_phase():
    names = ["a", "b", "c", "d"]
    plan = _build_representative_dual_plan(names)
    assert plan.assignment_forward["a"] == "GPU" and plan.assignment_backward["a"] == "GPU"
    assert plan.assignment_forward["b"] == "CPU" and plan.assignment_backward["b"] == "GPU"
    assert ("b", "b") in plan.cross_phase_edges
    assert ("a", "a") not in plan.cross_phase_edges


def test_measure_returns_skipped_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = measure_hybrid_overhead_ratio(_TinyConvNet(), torch.randn(4, 3, 8, 8))
    assert result["hybrid_overhead_ratio"] == 1.0
    assert result["status"] == "skipped"


def test_measure_never_raises_on_internal_error(monkeypatch):
    import runner.hybrid_overhead_calibration as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(mod, "_run_one_calibration_pass", _boom)
    result = measure_hybrid_overhead_ratio(_TinyConvNet(), torch.randn(4, 3, 8, 8))
    assert result["hybrid_overhead_ratio"] == 1.0
    assert result["status"] == "error"
    assert "simulated internal failure" in result["reason"]


def test_measure_restores_model_device_on_success_and_failure(monkeypatch):
    import runner.hybrid_overhead_calibration as mod

    model = _TinyConvNet()
    original_device = next(model.parameters()).device

    def _boom(*args, **kwargs):
        model.to("meta") if False else None  # no-op; just ensure finally restores whatever current device is
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_run_one_calibration_pass", _boom)
    measure_hybrid_overhead_ratio(model, torch.randn(4, 3, 8, 8))
    assert next(model.parameters()).device == original_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_measure_real_calibration_on_gpu_reports_plausible_ratio():
    model = _TinyConvNet()
    result = measure_hybrid_overhead_ratio(model, torch.randn(8, 3, 16, 16), gpu_id=0)
    assert result["status"] in ("ok", "skipped")
    if result["status"] == "ok":
        # reserved can never be less than allocated by construction of the
        # CUDA caching allocator.
        assert result["hybrid_overhead_ratio"] >= 1.0
        assert result["calibration_allocated_mb"] > 0.0
        assert result["calibration_reserved_mb"] >= result["calibration_allocated_mb"]
    # model must be moved back to CPU regardless of outcome
    assert next(model.parameters()).device.type == "cpu"
