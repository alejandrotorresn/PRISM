from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .plan_representation import ExecutionPlan


@dataclass
class DevicePlan:
    assignment_forward: Dict[str, str]
    assignment_backward: Dict[str, str]
    cut_edges_forward: List[Tuple[str, str]]
    cut_edges_backward: List[Tuple[str, str]]
    cross_phase_edges: List[Tuple[str, str]]
    activation_strategies: Dict[str, str] = field(default_factory=dict)

    @property
    def assignment(self) -> Dict[str, str]:
        return self.assignment_forward

    @property
    def cut_edges(self) -> List[Tuple[str, str]]:
        return self.cut_edges_forward

    @classmethod
    def from_execution_plan(cls, plan: ExecutionPlan) -> "DevicePlan":
        return cls(
            assignment_forward=dict(plan.assignment_forward),
            assignment_backward=dict(plan.assignment_backward),
            cut_edges_forward=list(plan.cut_edges_forward),
            cut_edges_backward=list(plan.cut_edges_backward),
            cross_phase_edges=list(plan.cross_phase_edges),
            activation_strategies=dict(plan.activation_strategies),
        )

    def resolve_torch_device(self, layer_name: str, gpu_id: int = 0, phase: str = "forward") -> torch.device:
        assignment = self.assignment_forward if phase == "forward" else self.assignment_backward
        label = assignment.get(layer_name, "CPU").upper()
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
    return any(str(dev).upper() == "GPU" for dev in plan.assignment_forward.values()) or any(
        str(dev).upper() == "GPU" for dev in plan.assignment_backward.values()
    )


def validate_plan_coverage(
    model: nn.Module,
    plan: DevicePlan,
    strict: bool = False,
    model_layer_names: Iterable[str] | None = None,
) -> List[str]:
    warnings: List[str] = []
    leaf_names = set(model_layer_names) if model_layer_names is not None else set(collect_leaf_module_names(model))
    plan_layers = set(plan.assignment_forward.keys()) | set(plan.assignment_backward.keys())

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
