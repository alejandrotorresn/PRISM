from __future__ import annotations

from pathlib import Path

import torch

from src.data import dataset_registry


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        input_ids = torch.tensor([[11, 12, 13, 0], [21, 22, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_tiny_shakespeare_loader_masks_padding_labels(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "tiny_shakespeare"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.txt").write_text("To be\nOr not\n", encoding="utf-8")

    monkeypatch.setattr(dataset_registry.AutoTokenizer, "from_pretrained", lambda _: _FakeTokenizer())

    inputs, labels, info = dataset_registry._load_tiny_shakespeare_batch(tmp_path, batch_size=2, seq_length=4)

    assert isinstance(inputs, dict)
    assert labels.tolist() == [[11, 12, 13, -100], [21, 22, -100, -100]]
    assert info["dataset_name"] == "tiny_shakespeare"
    assert info["target_source"] == "next_token_labels"