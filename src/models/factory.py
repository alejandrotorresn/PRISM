import logging
from typing import Any, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    ResNet152_Weights,
    ViT_B_16_Weights,
    resnet50,
    resnet152,
    vit_b_16,
)
from transformers import BertForSequenceClassification, GPT2LMHeadModel

try:
    from data.dataset_registry import load_model_batch
except ModuleNotFoundError:  # pragma: no cover - exercised by package-style imports in tests
    from src.data.dataset_registry import load_model_batch

logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(512, 256), output_dim=10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model_and_input(args, torch_dtype: torch.dtype) -> Tuple[nn.Module, Any]:
    model, inp, _, _ = build_model_input_target(args, torch_dtype)
    return model, inp


def _build_model(args, torch_dtype: torch.dtype) -> nn.Module:
    logger.info(f"Initializing {args.model} with batch size {args.batch_size}...")

    if args.model == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(dtype=torch_dtype)
    elif args.model == "resnet152":
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights).to(dtype=torch_dtype)
    elif args.model == "vit_b16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights).to(dtype=torch_dtype)
    elif args.model == "bert_base":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    elif args.model == "gpt2_small":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.config.pad_token_id = model.config.eos_token_id
    elif args.model == "distilgpt2":
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        model.config.pad_token_id = model.config.eos_token_id
    elif args.model == "simple_mlp":
        model = SimpleMLP().to(dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def _build_synthetic_input(args, torch_dtype: torch.dtype) -> Tuple[Any, torch.Tensor | None, dict[str, Any]]:
    if args.model in {"resnet50", "resnet152", "vit_b16"}:
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model in {"bert_base", "gpt2_small", "distilgpt2"}:
        inp = torch.randint(0, 1000, (args.batch_size, args.seq_length), dtype=torch.long)
    elif args.model == "simple_mlp":
        inp = torch.randn((args.batch_size, 784), dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return inp, None, {
        "dataset_name": None,
        "dataset_split": None,
        "dataset_path": None,
        "input_source": "synthetic",
        "target_source": None,
    }


def build_model_input_target(args, torch_dtype: torch.dtype) -> tuple[nn.Module, Any, torch.Tensor | None, dict[str, Any]]:
    model = _build_model(args, torch_dtype)
    datasets_root = getattr(args, "datasets_root", None)
    require_datasets = bool(getattr(args, "require_datasets", False))

    if require_datasets and not datasets_root:
        raise ValueError(
            "Dataset-backed execution is required, but datasets_root is empty. "
            "Provide --datasets_root or disable dataset enforcement only for diagnostics."
        )

    if datasets_root:
        try:
            inp, target, data_info = load_model_batch(
                model_name=args.model,
                batch_size=args.batch_size,
                input_size=args.input_size,
                seq_length=args.seq_length,
                datasets_root=datasets_root,
                torch_dtype=torch_dtype,
            )
        except FileNotFoundError:
            if require_datasets:
                raise
            logger.warning("Dataset-backed input unavailable for %s. Falling back to synthetic input.", args.model)
            inp, target, data_info = _build_synthetic_input(args, torch_dtype)
    else:
        inp, target, data_info = _build_synthetic_input(args, torch_dtype)

    if args.precision in ["fp16", "bf16"] and args.model not in ["bert_base", "gpt2_small", "distilgpt2"]:
        if isinstance(inp, torch.Tensor):
            inp = inp.to(dtype=torch_dtype)

    return model, inp, target, data_info
