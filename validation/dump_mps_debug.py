import sys
import pulp
from pathlib import Path
sys.path.append("/home/zephyr/Documents/University/PhD/Code/Final Thesis Code")
from src.ilp.data_loader import load_ilp_inputs
from src.ilp.model_builder import ILPConfig
from src.ilp.solve import _build_cbc_solver, ILPInputData

cfg = ILPConfig(
    gpu_mem_budget_mb=800.0,
    cpu_mem_budget_mb=1e18,
    w_time=0.5,
    w_energy=0.5,
    w_transfer=1.0
)
from src.runtime.plan_representation import infer_ilp_input_paths

config_dir = Path("data/zephyr/results_thesis_mode/resnet50/AdamW/fp32/batch_64")
inferred = infer_ilp_input_paths(config_dir=config_dir, model_name="resnet50")

data = load_ilp_inputs(
    metrics_stats_csv=str(inferred.metrics_stats_csv),
    graph_edges_csv=str(inferred.graph_edges_csv),
    transfer_edges_csv=str(inferred.transfer_edges_csv),
    strict_metric_validity=False,
    strict_sample_quality=False,
    strict_transfer_calibration=False,
    strict_graph_trace_source=False,
)

import src.ilp.solve as solve_mod

prob, problem_data, congestion_knee = solve_mod._build_pulp_problem(data, cfg)

solver = _build_cbc_solver(pulp)
solver.msg = True
try:
    prob.solve(solver)
except Exception as e:
    print("FAILED:", repr(e))

