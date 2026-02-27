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
from transformers import BertModel, GPT2Model

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
    logger.info(f"Initializing {args.model} with batch size {args.batch_size}...")

    if args.model == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "resnet152":
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "vit_b16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "bert_base":
        model = BertModel.from_pretrained("bert-base-uncased")
        inp = torch.randint(0, 1000, (args.batch_size, args.seq_length), dtype=torch.long)
    elif args.model == "gpt2_small":
        model = GPT2Model.from_pretrained("gpt2")
        inp = torch.randint(0, 1000, (args.batch_size, args.seq_length), dtype=torch.long)
    elif args.model == "simple_mlp":
        model = SimpleMLP().to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 784), dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.precision in ["fp16", "bf16"] and args.model not in ["bert_base", "gpt2_small"]:
        if isinstance(inp, torch.Tensor):
            inp = inp.to(dtype=torch_dtype)

    return model, inp
