"""Real, cycling train/val dataloaders for the convergence study.

Why this file exists
---------------------
`src/data/dataset_registry.py` was built for the thesis's profiling campaign:
every loader there returns exactly ONE batch (`DataLoader(..., shuffle=False);
next(iter(loader))`), always the first N samples of the training split, with
no notion of "next batch" or "validation split". That is correct for
profiling (you only need one representative batch to measure a layer's
latency), but it cannot produce a loss/accuracy curve, because the model
would see the exact same data on every step.

This module provides real epoch-cycling dataloaders instead: shuffled
training batches that advance every call (and reshuffle once the dataset is
exhausted = a new epoch), plus a genuinely held-out validation pass. It is a
NEW, separate implementation — `src/data/dataset_registry.py` is not
imported for its one-shot loaders, and is not modified. Where this module
needs logic that is already correct and tested there (mapping an Imagenette
folder class to the true ImageNet-1000 index the pretrained models expect),
it imports those small helper functions read-only, so the accuracy numbers
produced here are defined identically to the rest of the thesis.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

# Load src/data/dataset_registry.py directly by file path rather than via
# `from data.dataset_registry import ...`. The package-style import is what
# validation/run_hybrid_execution.py uses, but it is fragile here: the repo
# also has a top-level `data/` DIRECTORY (experiment outputs, not Python
# code) with no `dataset_registry.py` in it. When this module is imported in
# the same pytest session as other test files that also touch sys.path
# (e.g. running `pytest tests/ convergence_validation/tests/` together),
# Python can resolve the bare name `data` to that unrelated directory
# instead of `src/data/`, and once `sys.modules["data"]` is cached wrong it
# stays wrong for the rest of the process -- confirmed in practice: isolated
# `pytest convergence_validation/tests/` passes, but running it alongside
# `tests/` raised `ModuleNotFoundError: No module named 'data.dataset_registry'`.
# Loading by explicit file path sidesteps sys.path/sys.modules entirely.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_REGISTRY_PATH = _REPO_ROOT / "src" / "data" / "dataset_registry.py"
_spec = importlib.util.spec_from_file_location("_prism_dataset_registry", _DATASET_REGISTRY_PATH)
assert _spec is not None and _spec.loader is not None
_dataset_registry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _dataset_registry
_spec.loader.exec_module(_dataset_registry)

_load_synset_to_imagenet_index = _dataset_registry._load_synset_to_imagenet_index
_vision_weights_for_model = _dataset_registry._vision_weights_for_model


@dataclass
class EpochBatchProvider:
    """A callable that returns a fresh `(input, target)` batch each time it
    is invoked, cycling through a real `DataLoader` indefinitely.

    This is the object `prism_loop.py` and `all_gpu_loop.py` call once per
    training step. Internally it holds a live Python iterator over a
    `torch.utils.data.DataLoader`; when that iterator is exhausted (one full
    epoch has been consumed), a fresh iterator is created automatically,
    which for a `shuffle=True` loader means a new random ordering for the
    next epoch — real epoch-cycling, not batch repetition.
    """

    dataset: Dataset
    batch_size: int
    shuffle: bool
    collate_fn: Callable[[list[Any]], Any] | None = None
    generator_seed: int | None = None

    def __post_init__(self) -> None:
        generator = None
        if self.generator_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.generator_seed)
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            # drop_last=True everywhere (train AND val): the PRISM path traces
            # the model's computation graph exactly once (see prism_loop.py)
            # using the first batch's shape. For the decoder-only language
            # models (distilgpt2, gpt2_small) that trace is produced by
            # torch.export, which specializes to the exact batch dimension it
            # was given — a later batch with a different size would not run
            # safely through that graph. Keeping the batch size constant for
            # every batch, train and val alike, removes that failure mode
            # entirely instead of having to special-case it per model family.
            drop_last=True,
            generator=generator,
            collate_fn=self.collate_fn,
        )
        self._iterator: Iterator[Any] = iter(self._loader)

    def __call__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            # Epoch boundary: rebuild the iterator. For shuffle=True this
            # reshuffles (the DataLoader's generator state advances), so the
            # next epoch sees a different sample order, not a repeat.
            self._iterator = iter(self._loader)
            return next(self._iterator)

    def __len__(self) -> int:
        """Number of batches in one pass over the underlying dataset."""
        return len(self._loader)


# ---------------------------------------------------------------------------
# Vision: imagenette2-160 (resnet50, resnet152)
# ---------------------------------------------------------------------------


class _RemappedImageFolder(Dataset):
    """Wraps `torchvision.datasets.ImageFolder` so that labels are the true
    ImageNet-1000 class indices the pretrained torchvision heads expect,
    instead of Imagenette's own 0-9 folder-order indices.

    This remapping logic (synset -> ImageNet index) is identical to what
    `src/data/dataset_registry.py::_load_imagenette_batch` already does; the
    two small lookup helpers are imported from there read-only so both code
    paths agree on what "correct" means for this dataset.
    """

    def __init__(self, image_folder: Any, synset_to_imagenet_idx: dict[str, int]) -> None:
        self._image_folder = image_folder
        idx_to_synset = {idx: synset for synset, idx in image_folder.class_to_idx.items()}
        self._remap = {
            local_idx: synset_to_imagenet_idx[synset]
            for local_idx, synset in idx_to_synset.items()
        }

    def __len__(self) -> int:
        return len(self._image_folder)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, local_label = self._image_folder[index]
        return image, self._remap[local_label]


def make_vision_provider(
    model_name: str,
    datasets_root: Path,
    batch_size: int,
    split: str,
    seed: int,
    torch_dtype: torch.dtype,
) -> EpochBatchProvider:
    """Real train/val provider over `datasets/imagenette2-160/{train,val}`.

    `split="train"` shuffles every epoch; `split="val"` does not (evaluation
    order should not matter, and keeping it fixed makes debugging easier).
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got: {split}")

    from torchvision.datasets import ImageFolder

    dataset_dir = Path(datasets_root) / "imagenette2-160"
    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Imagenette '{split}' split not found: {split_dir}")

    weights = _vision_weights_for_model(model_name)
    image_folder = ImageFolder(root=str(split_dir), transform=weights.transforms())
    synset_to_imagenet_idx = _load_synset_to_imagenet_index(
        Path(datasets_root) / "metadata" / "imagenet_class_index.json"
    )
    dataset = _RemappedImageFolder(image_folder, synset_to_imagenet_idx)

    def _collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch]).to(dtype=torch_dtype)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    return EpochBatchProvider(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=_collate,
        generator_seed=seed,
    )


# ---------------------------------------------------------------------------
# Language: tiny_shakespeare (distilgpt2, gpt2_small)
# ---------------------------------------------------------------------------


class _TokenizedLineDataset(Dataset):
    """Pre-tokenizes every line once at construction time (the corpus is
    small — tens of thousands of short lines — so this is cheap and avoids
    re-running the tokenizer on every `__getitem__` call)."""

    def __init__(self, lines: list[str], tokenizer: Any, seq_length: int) -> None:
        self._encodings = tokenizer(
            lines,
            padding="max_length",
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
        )
        labels = self._encodings["input_ids"].clone().long()
        # Mask padded positions the same way
        # src/data/dataset_registry.py::_load_tiny_shakespeare_batch does,
        # so the next-token cross-entropy ignores padding identically.
        labels[self._encodings["attention_mask"] == 0] = -100
        self._labels = labels

    def __len__(self) -> int:
        return self._encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        item = {key: value[index] for key, value in self._encodings.items()}
        return item, self._labels[index]


def make_lm_provider(
    datasets_root: Path,
    batch_size: int,
    split: str,
    seed: int,
    seq_length: int,
) -> EpochBatchProvider:
    """Real train/val provider over the held-out tiny_shakespeare split
    produced by `prepare_tiny_shakespeare_split.py`.

    Raises `FileNotFoundError` with an actionable message if that one-time
    script has not been run yet.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got: {split}")

    split_file = Path(__file__).resolve().parent / "data" / "tiny_shakespeare" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} not found. Run "
            "`python convergence_validation/prepare_tiny_shakespeare_split.py` once "
            "before the first language-model convergence run."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lines = [line for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = _TokenizedLineDataset(lines, tokenizer, seq_length)

    return EpochBatchProvider(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=None,
        generator_seed=seed,
    )


def make_provider(
    model_name: str,
    datasets_root: Path,
    batch_size: int,
    split: str,
    seed: int,
    torch_dtype: torch.dtype,
    seq_length: int = 128,
) -> EpochBatchProvider:
    """Dispatch to the right provider for a given model, matching the
    model->dataset wiring already used across the thesis
    (resnet50/resnet152 -> imagenette2-160, distilgpt2/gpt2_small ->
    tiny_shakespeare)."""
    if model_name in _dataset_registry.VISION_MODELS:
        return make_vision_provider(model_name, datasets_root, batch_size, split, seed, torch_dtype)
    if model_name in {"distilgpt2", "gpt2_small"}:
        return make_lm_provider(datasets_root, batch_size, split, seed, seq_length)
    raise ValueError(
        f"No convergence-study dataloader wired for model '{model_name}'. "
        "Supported: resnet50, resnet152, vit_b16, distilgpt2, gpt2_small."
    )
