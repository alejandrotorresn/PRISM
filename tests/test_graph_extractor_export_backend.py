from __future__ import annotations

from pathlib import Path

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from src.core.graph_extractor import export_graph_artifacts
from src.runtime.device_plan import collect_leaf_module_names


def _tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = 0
    model.config.use_cache = False
    model.train()
    return model


def test_export_graph_artifacts_uses_decoder_only_export_backend(tmp_path: Path) -> None:
    model = _tiny_gpt2()
    input_data = {
        "input_ids": torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
    }
    layer_stats = {
        name: {"output_bytes": 0}
        for name in collect_leaf_module_names(model)
    }

    result = export_graph_artifacts(
        model=model,
        model_name="tiny_gpt2",
        output_dir=str(tmp_path),
        input_data=input_data,
        layer_stats=layer_stats,
        include_records=True,
    )

    producer_consumer_names = {
        edge["producer_name"]
        for edge in result["edges"]
    } | {
        edge["consumer_name"]
        for edge in result["edges"]
    }
    leaf_names = set(collect_leaf_module_names(model))

    assert result["trace_source"] == "torch_export_decoder_only"
    assert result["edges_count"] > 0
    assert result["nodes_count"] > 0
    assert producer_consumer_names
    assert producer_consumer_names.issubset(leaf_names)


def test_export_graph_artifacts_rejects_fallback_graph_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    input_data = torch.randn(2, 4)

    monkeypatch.setattr("src.core.graph_extractor._build_fx_graph", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fx failed")))
    monkeypatch.setattr("src.core.graph_extractor.try_export_decoder_only_trace", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="fallback_leaf_modules is disabled"):
        export_graph_artifacts(
            model=model,
            model_name="tiny_mlp",
            output_dir=str(tmp_path),
            input_data=input_data,
            include_records=True,
        )


def test_export_graph_artifacts_allows_fallback_graph_for_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    input_data = torch.randn(2, 4)

    monkeypatch.setattr("src.core.graph_extractor._build_fx_graph", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fx failed")))
    monkeypatch.setattr("src.core.graph_extractor.try_export_decoder_only_trace", lambda *args, **kwargs: None)

    result = export_graph_artifacts(
        model=model,
        model_name="tiny_mlp",
        output_dir=str(tmp_path),
        input_data=input_data,
        include_records=True,
        allow_fallback_graph=True,
    )

    assert result["trace_source"] == "fallback_leaf_modules"
    assert result["nodes_count"] > 0
