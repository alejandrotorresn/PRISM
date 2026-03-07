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

    assign_rows = [{"layer": k, "device": v} for k, v in sorted(solution.assignment.items())]
    pd.DataFrame(assign_rows).to_csv(assign_path, index=False)

    cut_rows = [{"src_layer": u, "dst_layer": v} for (u, v) in solution.cut_edges]
    pd.DataFrame(cut_rows).to_csv(cut_path, index=False)

    payload = {
        "status": solution.status,
        "backend": solution.backend,
        "objective_value": solution.objective_value,
        "gpu_mem_used_mb": solution.gpu_mem_used_mb,
        "cpu_mem_used_mb": solution.cpu_mem_used_mb,
        "layers_total": len(solution.assignment),
        "layers_gpu": sum(1 for _, dev in solution.assignment.items() if dev == "GPU"),
        "layers_cpu": sum(1 for _, dev in solution.assignment.items() if dev == "CPU"),
        "cut_edges": len(solution.cut_edges),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=4)

    return {
        "assignment_csv": str(assign_path),
        "cut_edges_csv": str(cut_path),
        "summary_json": str(summary_path),
    }
