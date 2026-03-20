from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .plan_representation import ExecutionPlan


@dataclass
class DevicePlan:
    assignment: Dict[str, str]
    cut_edges: List[Tuple[str, str]]

    @classmethod
    def from_execution_plan(cls, plan: ExecutionPlan) -> "DevicePlan":
        return cls(assignment=dict(plan.assignment), cut_edges=list(plan.cut_edges))

    def resolve_torch_device(self, layer_name: str, gpu_id: int = 0) -> torch.device:
        label = self.assignment.get(layer_name, "CPU").upper()
        if label == "GPU" and torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")


def collect_leaf_module_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            names.append(name)
    return names


def plan_requests_gpu(plan: DevicePlan) -> bool:
    return any(str(dev).upper() == "GPU" for dev in plan.assignment.values())


def validate_plan_coverage(
    model: nn.Module,
    plan: DevicePlan,
    strict: bool = False,
) -> List[str]:
    warnings: List[str] = []
    leaf_names = set(collect_leaf_module_names(model))
    plan_layers = set(plan.assignment.keys())

    missing_in_model = sorted(plan_layers - leaf_names)
    missing_in_plan = sorted(leaf_names - plan_layers)

    if missing_in_model:
        msg = f"{len(missing_in_model)} layer(s) in plan are absent from model: {missing_in_model[:5]}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    if missing_in_plan:
        msg = f"{len(missing_in_plan)} model layer(s) are not explicitly assigned and default to CPU: {missing_in_plan[:5]}"
        warnings.append(msg)

    return warnings
