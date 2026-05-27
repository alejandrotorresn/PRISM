from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, MNIST
from torchvision.models import ResNet50_Weights, ResNet152_Weights, ViT_B_16_Weights
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"
IMAGENET_CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
AG_NEWS_URLS = {
    "train.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VISION_MODELS = {"resnet50", "resnet152", "vit_b16"}

MODEL_DATASET_MAP = {
    "simple_mlp": "mnist",
    "resnet50": "imagenette2-160",
    "resnet152": "imagenette2-160",
    "vit_b16": "imagenette2-160",
    "bert_base": "ag_news",
    "gpt2_small": "tiny_shakespeare",
    "distilgpt2": "tiny_shakespeare",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    local_path: str


DATASET_SPECS = {
    "mnist": DatasetSpec("mnist", "MNIST handwritten digits for simple_mlp.", "mnist"),
    "imagenette2-160": DatasetSpec(
        "imagenette2-160",
        "Imagenette 160px subset mapped into ImageNet-1K label ids for torchvision classifiers.",
        "imagenette2-160",
    ),
    "ag_news": DatasetSpec("ag_news", "AG News text classification corpus for bert_base token inputs.", "ag_news"),
    "tiny_shakespeare": DatasetSpec(
        "tiny_shakespeare",
        "Tiny Shakespeare language modeling corpus for gpt2_small and distilgpt2 token inputs.",
        "tiny_shakespeare",
    ),
}


def resolve_datasets_root(root: str | Path | None = None) -> Path:
    if root is None or str(root).strip() == "":
        return DEFAULT_DATASETS_ROOT
    return Path(root)


def dataset_key_for_model(model_name: str) -> str:
    if model_name not in MODEL_DATASET_MAP:
        raise ValueError(f"Unsupported model for dataset resolution: {model_name}")
    return MODEL_DATASET_MAP[model_name]


def _download_file(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return
    logger.info("Downloading %s -> %s", url, dest)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, dest.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass

    if shutil.which("wget"):
        subprocess.run(["wget", "-O", str(dest), url], check=True)
        return
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
        return
    raise RuntimeError(f"Failed to download {url} and neither wget nor curl is available")


def _extract_tar(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def _extract_zip(archive_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest_dir)


def _ensure_mnist(root: Path, force: bool = False) -> Path:
    dataset_dir = root / "mnist"
    marker = dataset_dir / "MNIST" / "raw"
    if marker.exists() and not force:
        return dataset_dir
    MNIST(root=str(dataset_dir), train=True, download=True)
    MNIST(root=str(dataset_dir), train=False, download=True)
    return dataset_dir


def _ensure_imagenette(root: Path, force: bool = False) -> Path:
    dataset_dir = root / "imagenette2-160"
    train_dir = dataset_dir / "train"
    archive_path = root / "archives" / "imagenette2-160.tgz"
    if not train_dir.exists() or force:
        _download_file(IMAGENETTE_URL, archive_path, force=force)
        if force and dataset_dir.exists():
            for path in sorted(dataset_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        root.mkdir(parents=True, exist_ok=True)
        _extract_tar(archive_path, root)

    class_index_path = root / "metadata" / "imagenet_class_index.json"
    _download_file(IMAGENET_CLASS_INDEX_URL, class_index_path, force=force)
    return dataset_dir


def _ensure_ag_news(root: Path, force: bool = False) -> Path:
    dataset_dir = root / "ag_news"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in AG_NEWS_URLS.items():
        _download_file(url, dataset_dir / filename, force=force)
    return dataset_dir


def _ensure_tiny_shakespeare(root: Path, force: bool = False) -> Path:
    dataset_dir = root / "tiny_shakespeare"
    train_file = dataset_dir / "train.txt"
    if not train_file.exists() or force:
        _download_file(TINY_SHAKESPEARE_URL, train_file, force=force)
    return dataset_dir


def download_required_datasets(
    models: Iterable[str],
    datasets_root: str | Path | None = None,
    force: bool = False,
) -> list[dict[str, str]]:
    root = resolve_datasets_root(datasets_root)
    root.mkdir(parents=True, exist_ok=True)

    requested_keys = []
    for model in models:
        key = dataset_key_for_model(model)
        if key not in requested_keys:
            requested_keys.append(key)

    results: list[dict[str, str]] = []
    for key in requested_keys:
        if key == "mnist":
            dataset_path = _ensure_mnist(root, force=force)
        elif key == "imagenette2-160":
            dataset_path = _ensure_imagenette(root, force=force)
        elif key == "ag_news":
            dataset_path = _ensure_ag_news(root, force=force)
        elif key == "tiny_shakespeare":
            dataset_path = _ensure_tiny_shakespeare(root, force=force)
        else:
            raise ValueError(f"Unsupported dataset key: {key}")
        spec = DATASET_SPECS[key]
        results.append(
            {
                "dataset_key": spec.key,
                "description": spec.description,
                "path": str(dataset_path),
            }
        )

    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({"models": list(models), "datasets": results}, indent=2))
    return results


def _repeat_to_batch(values: Sequence[Any], batch_size: int) -> list[Any]:
    if not values:
        raise ValueError("Cannot build batch from empty dataset sample")
    out = list(values[:batch_size])
    idx = 0
    while len(out) < batch_size:
        out.append(values[idx % len(values)])
        idx += 1
    return out


def _next_loader_batch(dataset: Any, batch_size: int) -> tuple[Any, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _vision_weights_for_model(model_name: str):
    if model_name == "resnet50":
        return ResNet50_Weights.DEFAULT
    if model_name == "resnet152":
        return ResNet152_Weights.DEFAULT
    if model_name == "vit_b16":
        return ViT_B_16_Weights.DEFAULT
    raise ValueError(f"No vision weights for model: {model_name}")


def _load_synset_to_imagenet_index(metadata_path: Path) -> dict[str, int]:
    data = json.loads(metadata_path.read_text())
    return {value[0]: int(idx) for idx, value in data.items()}


def _load_mnist_batch(root: Path, batch_size: int, torch_dtype: torch.dtype) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    dataset_dir = root / "mnist"
    dataset = MNIST(root=str(dataset_dir), train=True, download=False, transform=ToTensor())
    images, labels = _next_loader_batch(dataset, batch_size)
    return (
        images.view(images.shape[0], -1).to(dtype=torch_dtype),
        labels.long(),
        {
            "dataset_name": "mnist",
            "dataset_split": "train",
            "dataset_path": str(dataset_dir),
            "input_source": "dataset",
            "target_source": "dataset_label",
        },
    )


def _load_imagenette_batch(
    model_name: str,
    root: Path,
    batch_size: int,
    torch_dtype: torch.dtype,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    dataset_dir = root / "imagenette2-160"
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Imagenette train split not found: {train_dir}")

    weights = _vision_weights_for_model(model_name)
    dataset = ImageFolder(root=str(train_dir), transform=weights.transforms())
    images, labels = _next_loader_batch(dataset, batch_size)
    synset_to_imagenet_idx = _load_synset_to_imagenet_index(root / "metadata" / "imagenet_class_index.json")
    idx_to_synset = {idx: synset for synset, idx in dataset.class_to_idx.items()}
    mapped_targets = torch.tensor(
        [synset_to_imagenet_idx[idx_to_synset[int(label)]] for label in labels.tolist()],
        dtype=torch.long,
    )
    return (
        images.to(dtype=torch_dtype),
        mapped_targets,
        {
            "dataset_name": "imagenette2-160",
            "dataset_split": "train",
            "dataset_path": str(dataset_dir),
            "input_source": "dataset",
            "target_source": "dataset_label_imagenet_index",
        },
    )


def _load_ag_news_batch(root: Path, batch_size: int, seq_length: int) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    dataset_dir = root / "ag_news"
    train_csv = dataset_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"AG News train split not found: {train_csv}")

    rows: list[tuple[str, int]] = []
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            label = max(0, int(raw[0]) - 1)
            title = raw[1].strip() if len(raw) > 1 else ""
            description = raw[2].strip() if len(raw) > 2 else ""
            text = f"{title}. {description}".strip()
            if text:
                rows.append((text, label))
            if len(rows) >= batch_size:
                break

    batch_rows = _repeat_to_batch(rows, batch_size)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(
        [text for text, _ in batch_rows],
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt",
    )
    targets = torch.tensor([label for _, label in batch_rows], dtype=torch.long)
    return (
        dict(tokens),
        targets,
        {
            "dataset_name": "ag_news",
            "dataset_split": "train",
            "dataset_path": str(dataset_dir),
            "input_source": "dataset",
            "target_source": "dataset_label",
        },
    )


def _load_tiny_shakespeare_batch(root: Path, batch_size: int, seq_length: int) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    dataset_dir = root / "tiny_shakespeare"
    train_file = dataset_dir / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"Tiny Shakespeare corpus not found: {train_file}")

    texts: list[str] = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) >= batch_size:
                break

    batch_texts = _repeat_to_batch(texts, batch_size)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt",
    )
    labels = tokens["input_ids"].clone().long()
    labels[tokens["attention_mask"] == 0] = -100
    return (
        dict(tokens),
        labels,
        {
            "dataset_name": "tiny_shakespeare",
            "dataset_split": "train",
            "dataset_path": str(dataset_dir),
            "input_source": "dataset",
            "target_source": "next_token_labels",
        },
    )


def load_model_batch(
    model_name: str,
    batch_size: int,
    input_size: int,
    seq_length: int,
    datasets_root: str | Path | None,
    torch_dtype: torch.dtype,
) -> tuple[Any, torch.Tensor | None, dict[str, Any]]:
    root = resolve_datasets_root(datasets_root)
    if model_name == "simple_mlp":
        return _load_mnist_batch(root, batch_size, torch_dtype)
    if model_name in VISION_MODELS:
        return _load_imagenette_batch(model_name, root, batch_size, torch_dtype)
    if model_name == "bert_base":
        return _load_ag_news_batch(root, batch_size, seq_length)
    if model_name in {"gpt2_small", "distilgpt2"}:
        return _load_tiny_shakespeare_batch(root, batch_size, seq_length)
    raise ValueError(f"Unsupported model for dataset loading: {model_name}")