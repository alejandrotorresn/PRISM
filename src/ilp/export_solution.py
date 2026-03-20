from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .solve import ILPSolution


def save_ilp_solution(solution: ILPSolution, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assign_path = out / "ilp_assignment.csv"
    cut_path = out / "ilp_cut_edges.csv"
    summary_path = out / "ilp_solution_summary.json"

    def _strategy_name(raw) -> str:
        if raw is None:
            return "retain"
        if isinstance(raw, str):
            return raw.lower()
        if getattr(raw, "recompute", False):
            return "recompute"
        if getattr(raw, "checkpoint", False):
            return "checkpoint"
        return "retain"

    activation_strategies = getattr(solution, "activation_strategies", None) or {}

    forward_assignment = solution.forward_assignment or solution.assignment
    backward_assignment = solution.backward_assignment or solution.assignment
    assign_rows = [
        {
            "layer": layer,
            "device": forward_assignment[layer],
            "device_forward": forward_assignment[layer],
            "device_backward": backward_assignment[layer],
            "activation_strategy": _strategy_name(activation_strategies.get(layer)),
        }
        for layer in sorted(forward_assignment.keys())
    ]
    pd.DataFrame(assign_rows).to_csv(assign_path, index=False)

    cut_rows = ([{"src_layer": u, "dst_layer": v, "phase": "forward"} for (u, v) in solution.cut_edges] +
                [{"src_layer": u, "dst_layer": v, "phase": "backward"} for (u, v) in (solution.backward_cut_edges or [])] +
                [{"src_layer": u, "dst_layer": v, "phase": "cross_phase"} for (u, v) in (solution.cross_phase_edges or [])])
    pd.DataFrame(cut_rows, columns=["src_layer", "dst_layer", "phase"]).to_csv(cut_path, index=False)

    payload = {
        "status": solution.status,
        "backend": solution.backend,
        "objective_value": solution.objective_value,
        "gpu_mem_used_mb": solution.gpu_mem_used_mb,
        "cpu_mem_used_mb": solution.cpu_mem_used_mb,
        "layers_total": len(forward_assignment),
        "layers_gpu_forward": sum(1 for _, dev in forward_assignment.items() if dev == "GPU"),
        "layers_cpu_forward": sum(1 for _, dev in forward_assignment.items() if dev == "CPU"),
        "layers_gpu_backward": sum(1 for _, dev in backward_assignment.items() if dev == "GPU"),
        "layers_cpu_backward": sum(1 for _, dev in backward_assignment.items() if dev == "CPU"),
        "cut_edges_forward": len(solution.cut_edges),
        "cut_edges_backward": len(solution.backward_cut_edges or []),
        "cross_phase_edges": len(solution.cross_phase_edges or []),
        "activation_strategy_counts": {
            "retain": sum(1 for layer in forward_assignment if _strategy_name(activation_strategies.get(layer)) == "retain"),
            "recompute": sum(1 for layer in forward_assignment if _strategy_name(activation_strategies.get(layer)) == "recompute"),
            "checkpoint": sum(1 for layer in forward_assignment if _strategy_name(activation_strategies.get(layer)) == "checkpoint"),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=4)

    return {
        "assignment_csv": str(assign_path),
        "cut_edges_csv": str(cut_path),
        "summary_json": str(summary_path),
    }
