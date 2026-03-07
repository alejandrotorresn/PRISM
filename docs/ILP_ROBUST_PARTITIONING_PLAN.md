# ILP Robust Layer Partitioning Plan (CPU/GPU)

This document defines a thesis-grade plan to upgrade the current profiler into a robust ILP-based partitioning pipeline.

## 1. Objective

Build a reproducible optimization framework that:
- models layer dependencies explicitly as a DAG,
- uses edge-level CPU<->GPU transfer costs,
- incorporates stochastic variability (not only mean values),
- validates solutions across batches/precisions/hardware conditions.

## 2. Mathematical Formulation (Thesis-Ready)

### 2.1 Sets and Graph

- `V`: set of computation nodes (layer/op-level units).
- `E`: set of directed edges (activation dependencies), `e = (u, v)`.
- `D = {CPU, GPU}`: execution devices.

Graph: `G = (V, E)` is a DAG from model trace.

### 2.2 Decision Variables

- `x_v in {0,1}` for `v in V`:
  - `x_v = 1` => node `v` assigned to GPU,
  - `x_v = 0` => node `v` assigned to CPU.
- `y_e in {0,1}` for `e = (u,v) in E`:
  - `y_e = 1` if edge crosses devices (transfer required),
  - `y_e = 0` otherwise.

Linearization constraints for each `e=(u,v)`:
- `y_e >= x_u - x_v`
- `y_e >= x_v - x_u`

### 2.3 Per-Node Costs

For each node `v`:
- `T_gpu(v), T_cpu(v)`: forward+backward time (ms),
- `E_gpu(v), E_cpu(v)`: energy (J),
- `M_gpu(v), M_cpu(v)`: memory contribution (MB).

Robust time/energy option:
- `T_hat_d(v) = mu_T_d(v) + k * sigma_T_d(v)`
- `E_hat_d(v) = mu_E_d(v) + k * sigma_E_d(v)`
with `d in {CPU, GPU}` and `k` chosen by confidence policy.

### 2.4 Edge Transfer Cost

For each edge `e=(u,v)` with activation size `S_e` (MB):
- H2D: `C_h2d(e) = alpha_h2d + S_e / beta_h2d + sync_h2d(e)`
- D2H: `C_d2h(e) = alpha_d2h + S_e / beta_d2h + sync_d2h(e)`

Use direction implied by assignment pair:
- CPU->GPU if `x_u=0, x_v=1`,
- GPU->CPU if `x_u=1, x_v=0`.

A practical symmetric approximation for first robust implementation:
- `C_e = alpha_e + S_e / beta_e + sync_e`
- transfer objective uses `y_e * C_e`.

### 2.5 Objective Function

Weighted single-objective form:

`min Z = w_t * Z_time + w_e * Z_energy + w_m * Z_memory`

where
- `Z_time = sum_v [x_v*T_hat_gpu(v) + (1-x_v)*T_hat_cpu(v)] + sum_e y_e*C_e`
- `Z_energy = sum_v [x_v*E_hat_gpu(v) + (1-x_v)*E_hat_cpu(v)] + sum_e y_e*E_transfer(e)`
- `Z_memory` can be modeled as penalty or hard constraints.

Alternative: epsilon-constraint (recommended for thesis analysis)
- minimize time,
- subject to energy <= budget and memory <= budget.

### 2.6 Constraints

- Assignment: `x_v in {0,1}`.
- Boundary transfer activation: `y_e` constraints above.
- GPU memory budget: `M_gpu_peak(x) <= B_gpu`.
- CPU memory budget: `M_cpu_peak(x) <= B_cpu`.
- Optional latency SLA: `Z_time <= L_max`.

Note: exact peak memory is sequence-dependent. First implementation can use conservative linear upper bound, then tighten with schedule-aware approximation.

## 3. Data Model Additions Required

Current profiler already exports many useful per-layer fields. Add the following for robust ILP:

### 3.1 Node/Edge Topology Artifacts

New artifacts per run:
- `{model}_graph_nodes.csv`
- `{model}_graph_edges.csv`

Minimum columns:
- nodes: `node_id, node_name, op_type, topo_index, params_mb, activ_out_mb`
- edges: `src_id, dst_id, tensor_mb, tensor_shape, producer_name, consumer_name`

### 3.2 Variability Statistics

New aggregated artifact:
- `{model}_metrics_stats.csv`

Columns per node and device:
- `time_mean_ms, time_std_ms, time_p50_ms, time_p90_ms, time_p95_ms`
- `energy_mean_j, energy_std_j, energy_p50_j, energy_p90_j, energy_p95_j`
- `n_samples`

### 3.3 Transfer Calibration by Size Class

New artifact:
- `transfer_calibration.csv`

Columns:
- `direction, size_mb, alpha_ms, beta_mb_s, sigma_overlap, contention_class, n_trials`

## 4. Implementation Plan in This Repository

## 4.1 Phase A: Graph Extraction

Add module:
- `src/core/graph_extractor.py`

Responsibilities:
- trace model graph using `torch.fx` (fallback strategy if tracing fails),
- build DAG node/edge tables,
- compute activation sizes from representative input.

Integrate from:
- `src/runner/training_profiler.py` (after model/input initialization and before final artifact save).

## 4.2 Phase B: Edge-Aware Transfer Costs

Update:
- `src/runner/training_profiler.py`

Changes:
- replace global transfer approximation columns with edge-aware transfer aggregation,
- store both per-edge and per-node boundary costs,
- keep legacy columns for backward compatibility during migration.

## 4.3 Phase C: Variability Collection

Update:
- `scripts/run_experiments.sh` (replicate runs with seed schedule),
- `src/profiler.py` (add seed/run_id args),
- `src/core/io_artifacts.py` (append/aggregate helpers).

Add module:
- `src/core/stats_aggregator.py`

Responsibilities:
- aggregate replicate artifacts,
- compute means/std/percentiles,
- export `metrics_stats.csv`.

## 4.4 Phase D: ILP Solver Interface

Add package:
- `src/ilp/`

Suggested files:
- `src/ilp/model_builder.py` (variables, objective, constraints)
- `src/ilp/robust_terms.py` (mu+k*sigma or scenario sampling)
- `src/ilp/solve.py` (solver interface; CBC/Gurobi/OR-Tools)
- `src/ilp/export_solution.py` (partition map and diagnostics)

CLI integration:
- new command script `scripts/run_ilp_partition.sh`.

## 4.5 Phase E: Validation Harness

Add:
- `validation/validate_ilp_pipeline.py`

Checks:
- graph consistency (`|V|`, `|E|`, acyclicity),
- edge transfer non-negativity and unit sanity,
- robust stats completeness (`n_samples` threshold),
- ILP feasibility under defined budgets,
- prediction vs observed execution gap.

## 5. Experimental Protocol (Thesis)

## 5.1 Factorial Matrix

At minimum:
- Models: `resnet50, resnet152, vit_b16, bert_base, gpt2_small, simple_mlp`
- Batch sizes: small/medium/large per model memory envelope
- Precision: `fp32, fp16, bf16` (when supported)
- Optimizer: at least `SGD, AdamW`
- Replicates: `N >= 20` per configuration (or justify lower with CI width)

## 5.2 Baselines

Compare ILP against:
- all-GPU,
- all-CPU,
- greedy heuristic (e.g., offload largest activations first),
- optional pipeline-parallel heuristic.

## 5.3 Evaluation Metrics

- End-to-end step latency (ms)
- Energy per step (J)
- GPU memory peak (MB)
- CPU memory footprint (MB)
- Prediction error:
  - `|pred_time - obs_time| / obs_time`
  - `|pred_energy - obs_energy| / obs_energy`

## 5.4 Ablation Study

Mandatory ablations:
- no graph topology (flat layer list),
- no edge-aware transfer costs,
- no variability robustness,
- full model (all components).

This demonstrates contribution of each thesis component.

## 6. Risks and Mitigations

- Tracing failures for dynamic models:
  - fallback to module-level DAG approximation and document limitations.
- Non-stationary performance under thermal throttling:
  - randomize run order, include cooldown periods, log temperature/power states if available.
- Memory peak non-linearity:
  - begin with conservative bounds, validate with replay runs and tighten constraints iteratively.

## 7. Incremental Delivery Milestones

1. `M1`: Graph extraction artifacts generated for all models.
2. `M2`: Edge-aware transfer terms integrated into profiling outputs.
3. `M3`: Replicate aggregation and robust stats exported.
4. `M4`: ILP solver produces feasible partitions under memory budget.
5. `M5`: Full thesis evaluation and ablations complete.

## 8. Immediate Next Actions (Practical)

1. Implement `src/core/graph_extractor.py` and export `nodes/edges` CSV files.
2. Add replicate runner mode in `scripts/run_experiments.sh` (`REPEATS`, `SEED_BASE`).
3. Add `stats_aggregator.py` to compute robust per-layer stats.
4. Build initial ILP with deterministic means, then upgrade to robust (`mu + k*sigma`).
5. Run pilot study on `simple_mlp` and `resnet50` before full matrix.
