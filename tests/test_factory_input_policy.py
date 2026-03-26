from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models import factory


def test_build_model_input_target_requires_datasets_by_default() -> None:
    args = SimpleNamespace(
        model="simple_mlp",
        precision="fp32",
        batch_size=2,
        input_size=224,
        seq_length=16,
        datasets_root="",
        require_datasets=True,
    )

    with pytest.raises(ValueError, match="Dataset-backed execution is required"):
        factory.build_model_input_target(args, torch.float32)


def test_build_model_input_target_allows_explicit_synthetic_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(
        model="simple_mlp",
        precision="fp32",
        batch_size=2,
        input_size=224,
        seq_length=16,
        datasets_root="missing-datasets",
        require_datasets=False,
    )

    monkeypatch.setattr(factory, "load_model_batch", lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))

    model, inp, target, info = factory.build_model_input_target(args, torch.float32)

    assert model is not None
    assert isinstance(inp, torch.Tensor)
    assert target is None
    assert info["input_source"] == "synthetic"