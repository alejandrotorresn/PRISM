from __future__ import annotations

import json
from pathlib import Path

from src.data import dataset_registry


def test_dataset_key_for_model_resolves_expected_dataset() -> None:
    assert dataset_registry.dataset_key_for_model("simple_mlp") == "mnist"
    assert dataset_registry.dataset_key_for_model("resnet50") == "imagenette2-160"
    assert dataset_registry.dataset_key_for_model("bert_base") == "ag_news"
    assert dataset_registry.dataset_key_for_model("gpt2_small") == "tiny_shakespeare"
    assert dataset_registry.dataset_key_for_model("distilgpt2") == "tiny_shakespeare"


def test_download_required_datasets_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    def fake_prepare(name: str):
        def _impl(root: Path, force: bool = False) -> Path:
            out = root / name
            out.mkdir(parents=True, exist_ok=True)
            return out

        return _impl

    monkeypatch.setattr(dataset_registry, "_ensure_mnist", fake_prepare("mnist"))
    monkeypatch.setattr(dataset_registry, "_ensure_imagenette", fake_prepare("imagenette2-160"))
    monkeypatch.setattr(dataset_registry, "_ensure_ag_news", fake_prepare("ag_news"))
    monkeypatch.setattr(dataset_registry, "_ensure_tiny_shakespeare", fake_prepare("tiny_shakespeare"))

    results = dataset_registry.download_required_datasets(
        models=["simple_mlp", "resnet50", "bert_base", "gpt2_small", "distilgpt2"],
        datasets_root=tmp_path,
    )

    assert [item["dataset_key"] for item in results] == [
        "mnist",
        "imagenette2-160",
        "ag_news",
        "tiny_shakespeare",
    ]

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["models"] == ["simple_mlp", "resnet50", "bert_base", "gpt2_small", "distilgpt2"]
    assert [item["dataset_key"] for item in manifest["datasets"]] == [
        "mnist",
        "imagenette2-160",
        "ag_news",
        "tiny_shakespeare",
    ]