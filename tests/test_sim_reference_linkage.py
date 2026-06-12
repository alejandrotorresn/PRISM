from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_resolve_sim_reference_values_from_plan_source_csv(tmp_path: Path) -> None:
    # Import lazily so environments without torch can skip at collection-time upstream if needed.
    import pytest

    pytest.importorskip("torch")
    from validation.run_hybrid_execution import _resolve_sim_reference_values

    src = tmp_path / "pareto.csv"
    pd.DataFrame(
        [
            {
                "model": "resnet50",
                "optimizer": "SGD",
                "precision": "fp32",
                "batch_size": 32,
                "gpu_budget_mb": 2048,
                "ilp_status": "optimal",
                "ilp_objective": 123.0,
                "sim_time_ms": 130.0,
                "sim_energy_j": 55.0,
            }
        ]
    ).to_csv(src, index=False)

    args = SimpleNamespace(
        model="resnet50",
        precision="fp32",
        batch_size=32,
        plan_sim_time_ms=None,
        plan_sim_energy_j=None,
        plan_source_csv=str(src),
        plan_gpu_budget_mb=2048.0,
        plan_objective=999.0,
    )

    cfg_dir = tmp_path / "resnet50" / "SGD" / "fp32" / "batch_32"
    values = _resolve_sim_reference_values(args, cfg_dir)

    assert values["sim_time_ms"] == 130.0
    assert values["sim_energy_j"] == 55.0
    assert str(values["source"]).startswith("plan_source_csv:")
