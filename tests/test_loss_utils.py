from __future__ import annotations

from types import SimpleNamespace

import torch

from src.core.loss_utils import compute_training_objective


def test_compute_training_objective_reports_classification_accuracy() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.5]], dtype=torch.float32)
    out = SimpleNamespace(logits=logits)
    target = torch.tensor([0, 1], dtype=torch.long)

    loss, metric_name, metric_value = compute_training_objective(out, target=target)

    assert loss.item() > 0.0
    assert metric_name == "accuracy"
    assert metric_value == 1.0


def test_compute_training_objective_reports_causal_token_accuracy() -> None:
    logits = torch.tensor(
        [
            [
                [0.1, 5.0, 0.1],
                [0.1, 0.1, 4.0],
                [4.0, 0.1, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = SimpleNamespace(logits=logits)

    loss, metric_name, metric_value = compute_training_objective(out, target=target)

    assert loss.item() > 0.0
    assert metric_name == "token_accuracy"
    assert metric_value == 1.0


def test_compute_training_objective_uses_model_loss_for_causal_lm() -> None:
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    target = torch.tensor([[0, 1, -100]], dtype=torch.long)
    out = SimpleNamespace(logits=logits, loss=torch.tensor(2.5, requires_grad=True))

    loss, metric_name, metric_value = compute_training_objective(out, target=target)

    assert float(loss.item()) == 2.5
    assert metric_name == "token_accuracy"
    assert metric_value is not None