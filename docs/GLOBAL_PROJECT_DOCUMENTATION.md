# Global Project Documentation

## 1. Document scope and objective

This document serves as the comprehensive technical reference for the project.

It explains, with direct code traceability:

- how profiling data is captured (time, energy, memory, FLOPs, transfer)
- how model graphs are extracted and persisted
- how outputs are stored and aggregated
- how the ILP model is built and solved
- how multi-hardware data is merged
- how scripts are executed end-to-end
- how report plots and LaTeX tables are generated
- how validation is implemented

Primary implementation references:

- `src/profiler.py`
- `src/runner/training_profiler.py`
- `src/core/graph_extractor.py`
- `src/core/metrics.py`
- `src/core/precision_policy.py`
- `src/core/energy.py`
- `src/core/stats_aggregator.py`
- `src/ilp/data_loader.py`
- `src/ilp/model_builder.py`
- `src/ilp/solve.py`
- `src/ilp/export_solution.py`

Supporting runtime and orchestration references:

- `scripts/run_experiments.sh`
- `scripts/run_thesis_smoke_workflow.sh`
- `scripts/run_ilp_partition.sh`
- `scripts/run_ilp_pareto_sweep.sh`
- `scripts/discover_ilp_config_dirs.sh`
- `scripts/generate_ilp_report_assets.sh`
- `scripts/export_ilp_tables_latex.sh`
- `validation/*.py`

---

## 2. What this project builds

The project implements a complete empirical-to-optimization pipeline:

1. Profile training layer-by-layer on CPU and GPU.
2. Export metrics and metadata artifacts.
3. Export a DAG graph and edge transfer costs.
4. Aggregate repeated runs into robust statistics.
5. Build and solve a robust ILP partition model (CPU/GPU assignment per layer).
6. Run Pareto sweeps under GPU memory budgets.
7. Generate consolidated plots and LaTeX tables for reporting.

At a conceptual level:

- profiling side = measurement + feature engineering
- ILP side = decision optimization using measured costs and constraints

---

## 3. End-to-end architecture

### 3.1 Runtime flow

Main profiling entrypoint:

- `main()` in `src/profiler.py`

Primary orchestration steps:

1. Parse CLI args (`_build_parser`).
2. Normalize output directory under host namespace (`normalize_output_dir_for_host` in `src/core/system.py`).
3. Configure CPU runtime and determinism (`configure_cpu_runtime`, `set_determinism`).
4. Evaluate precision execution policy (`_configure_precision` in `src/profiler.py` + `src/core/precision_policy.py`).
5. Build model and input (`build_model_and_input` in `src/models/factory.py`).
6. Execute profiling (`TrainingProfiler.run_profiling` in `src/runner/training_profiler.py`).

### 3.2 Artifact pipeline

Per-run profiling outputs:

- `<model>_metrics.csv`
- `<model>_meta.json`
- `<model>_graph_nodes.csv`
- `<model>_graph_edges.csv`
- `<model>_transfer_edges.csv`
- optional early safety checkpoints:
  - `<model>_metrics_gpu_partial.csv`
  - `<model>_meta_gpu_partial.json`

Replicate aggregation output:

- `<model>_metrics_stats.csv` (or `metrics_stats.csv`)

ILP outputs:

- `ilp_solution/ilp_assignment.csv`
- `ilp_solution/ilp_cut_edges.csv`
- `ilp_solution/ilp_solution_summary.json`

Pareto outputs:

- `<model>_pareto_sweep.csv`
- `<model>_pareto_summary.json`

Report outputs:

- `ilp_pareto_consolidated.csv`
- `ilp_best_per_model.csv`
- `ILP_RESULTS_SUMMARY.md`
- `*_objective_vs_budget.png`
- `best_ilp_vs_all_cpu_improvement.png`
- LaTeX files in `reports/.../latex/*.tex`

---

## 4. Profiling internals: exactly how data is captured

### 4.1 Layer-level instrumentation

Implemented in `TrainingProfiler._register_hooks` (`src/runner/training_profiler.py`).

The profiler attaches hooks only to leaf modules (`_get_leaf_modules`) in order to avoid double counting nested modules.

For each leaf layer:

- the pre-hook stores wall-clock start time (`time.perf_counter()`)
- for CUDA, the pre-hook records a CUDA event
- the post-hook computes:
  - kernel time (`kernel_ms`) from CUDA events (GPU) or wall-clock delta (CPU)
  - dispatch overhead:

$$
\text{dispatch\_ms} = \max(0, \text{wall\_ms} - \text{kernel\_ms})
$$

  - output size in bytes (`get_tensor_size_recursive`)
  - parameter size in MB
  - FLOPs estimate (`estimate_flops`)

### 4.2 Training-step execution

Implemented in `TrainingProfiler._run_epoch`:

- the input is moved to the selected device
- the optimizer is constructed from the user-selected optimizer
- the loop executes `steps` iterations:
  - `forward`
  - `loss.backward()`
  - `optimizer.step()` timing
- the energy monitor runs in parallel (`EnergyMonitor`)

Energy per phase is computed as:

$$
E_{\text{total}} = P_{\text{avg}} \cdot T
$$

where:

- $P_{\text{avg}}$ = average measured power (W)
- $T$ = measured wall-clock duration (s)

### 4.3 FLOPs formulas used

Implemented in `src/core/metrics.py` (`estimate_flops`).

The purpose of this subsection is to make explicit how computational work is approximated when direct hardware counters are not available in a portable way. FLOPs are not used as a direct optimization objective in this project, but they are essential for normalization and interpretation: they let us compare layers with very different shapes and operator types under a common unit of computational demand.

Convolution 2D:

$$
\text{FLOPs}_{\text{conv}} = 2 \cdot C_{out} \cdot H_{out} \cdot W_{out} \cdot \left(\frac{C_{in}}{\text{groups}} \cdot K_x \cdot K_y\right)
$$

Linear:

$$
\text{FLOPs}_{\text{linear}} = 2 \cdot P \cdot \text{in\_features} \cdot \text{out\_features}
$$

where $P$ is the product of all position dimensions before the feature axis.

Attention-like module approximation:

$$
\text{FLOPs}_{\text{attn}} \approx 4BSD^2 + 2BS^2D
$$

with:

- $B$: batch
- $S$: sequence length
- $D$: hidden dimension

Why these formulas are needed and how to read them:

- The convolution expression captures multiply-accumulate effort over output spatial positions and channels. The factor `2` represents multiply plus add.
- The linear expression scales with input and output features and any leading positional dimensions (`P`), which is why transformer MLP blocks and classifier heads can be compared directly.
- The attention formula is intentionally approximate. It captures the dominant complexity terms ($D^2$ projection work and $S^2D$ attention score/value mixing) to preserve model-scale sensitivity without requiring architecture-specific internals for each implementation.

Practical interpretation:

- Large FLOPs with low measured TFLOPS often indicates memory or dispatch bottlenecks.
- Similar FLOPs but very different runtime can indicate transfer overhead, kernel efficiency differences, or precision-path effects.

### 4.4 Empirical peak TFLOPS benchmark

Implemented in `TrainingProfiler._measure_peak_flops`.

This benchmark exists to provide a hardware-relative normalization baseline. Raw TFLOPS per layer can be hard to interpret in isolation because they depend on device generation and runtime settings. By measuring an empirical peak in the same environment, the project derives an efficiency ratio that is comparable across runs and hardware profiles.

For matrix size $N$ and average duration $\Delta t$ over iterations:

$$
\text{TFLOPS}_{\text{peak}} = \frac{2N^3}{10^{12} \cdot \Delta t}
$$

Layer efficiency ratio:

$$
\text{efficiency\_ratio}_{\ell} = \frac{\text{TFLOPS}_{\ell}}{\text{TFLOPS}_{\text{peak}}}
$$

Why this benchmark is important:

- It separates layer-level algorithmic work from platform ceiling effects.
- It improves diagnostic power: a layer with low efficiency ratio can be investigated for memory pressure, transfer stalls, or suboptimal kernel paths.
- It enables fairer multi-server analysis, because each server is normalized against its own empirical peak before robust merging.

### 4.5 Forward/backward heuristic

`BACKWARD_FACTOR = 2.0` in `src/core/constants.py`.

When backward measurement is not directly separated at layer level, backward time and energy per layer are approximated as:

$$
T_{\ell}^{bwd} = 2 \cdot T_{\ell}^{fwd}, \quad
E_{\ell}^{bwd} = 2 \cdot E_{\ell}^{fwd}
$$

This is applied in row creation inside `TrainingProfiler.run_profiling`.

Why `BACKWARD_FACTOR = 2.0` is used and why it is not arbitrary:

- In gradient-based training, backward computation is commonly of the same order as forward, and often larger due to gradient propagation and parameter-gradient accumulation.
- A factor near `2` is a pragmatic and conservative engineering prior used when precise per-layer backward separation is unavailable.
- The value is fixed to keep runs comparable and deterministic; changing it ad hoc would introduce confounding effects in downstream robust statistics and ILP coefficients.

Methodological role:

- It is a fallback approximation, not a claim of exact physics.
- It preserves pipeline continuity and avoids missing cost channels in ILP input construction.
- Its impact is later moderated by replicate statistics (`mean`, `std`, quantiles) and robustification (`mu + k*sigma`).

---

## 5. Precision policy and runtime gating

Implemented in:

- `src/core/precision_policy.py`
- `_configure_precision` in `src/profiler.py`

### 5.1 ISA probing

`probe_cpu_precision_support()` checks CPU flags from `/proc/cpuinfo`.

Acceleration policy:

- `fp16` requires `avx512_fp16`
- `bf16` requires `avx512_bf16` OR (`amx_bf16` and `amx_tile`)

### 5.2 Execution policy

`evaluate_precision_execution_policy(precision, isa_info)` returns:

- `allowed`
- `cpu_precision_executed`
- `reason`
- `status`

Meaning of each returned field:

- `allowed`: Boolean gate indicating whether the requested precision can proceed under detected ISA/runtime constraints.
- `cpu_precision_executed`: Effective CPU precision mode that will actually run (`fp32`, `fp16`, `bf16`, or empty when skipped), used for artifact traceability.
- `reason`: Human-readable justification for the policy decision (for example unsupported ISA path), designed for auditability and debugging.
- `status`: Compact machine-readable state label (for example `ready` or `skipped_unsupported_precision`) used by wrappers and validation checks.

If unsupported precision is requested:

- with `--skip_cpu` and GPU available: continue GPU-only
- otherwise: run is skipped with explicit reason artifacts

### 5.3 CPU FP16 model preflight and timeout mathematics

Implemented in `run_cpu_fp16_model_preflight`.

Three-stage behavior:

1. forward-phase join timeout fixed at 60s
2. adaptive backward timeout computed as:

$$
\tau_{bwd} = \max\left(10,\; T_{fwd} \cdot \text{BACKWARD\_FACTOR} \cdot s\right)
$$

where:

- $T_{fwd}$ = measured forward time in seconds
- `BACKWARD_FACTOR = 2.0`
- $s$ = timeout safety factor (default 2.5)

3. final join uses $\tau_{bwd}$

This mechanism avoids hangs while still preserving valid execution when feasible.

Why this staged behavior is used:

- CPU FP16 support can be syntactically available but operationally unstable on some platforms.
- A single monolithic timeout is either too short for large models or too long for failure detection.
- Splitting timeout into a fixed forward gate and adaptive backward gate improves both safety and fairness.

Design rationale of the stages:

- Stage 1 (fixed forward timeout) quickly rejects clearly non-viable runs.
- Stage 2 (adaptive backward timeout) scales expected budget to observed forward behavior, reducing false negatives on larger models.
- Stage 3 (final guarded join) ensures the process cannot stall indefinitely.

In short, this is a reliability mechanism that protects campaign integrity while preserving potentially valid executions.

---

## 6. Graph extraction: how nodes and edges are built

This section explains how structural model information is transformed into an analyzable graph representation. The graph is not merely descriptive; it is the structural substrate used for transfer-cost construction and downstream ILP edge-cut penalties.

Implemented in `src/core/graph_extractor.py`.

### 6.1 Primary method: torch.fx

`_build_fx_graph(model, layer_stats, input_data)`:

- traces the model via `symbolic_trace`
- optionally propagates shape metadata (`ShapeProp`) when devices are compatible
- creates node records for non-output FX nodes
- creates edge records from input dependencies (`dst.all_input_nodes`)

Node naming:

- `call_module`: module target name
- `call_function` / `call_method`: target representation
- `placeholder`: input name

### 6.2 Fallback method

If FX fails, `_build_fallback_graph` creates a linear chain over leaf modules.

This guarantees that graph artifacts exist even for non-traceable models, albeit with reduced structural fidelity.

Why this fallback is necessary:

- Some models include dynamic control flow or tracing-incompatible operations.
- Without fallback, those runs would produce no graph artifacts, blocking transfer-edge modeling and ILP construction.
- The linear-chain fallback is intentionally conservative: it preserves ordering and artifact availability, even if it cannot represent full branching structure.

### 6.3 Node and edge schemas

Graph node columns (written by `write_csv_rows`):

- `node_id`
- `node_name`
- `op_type`
- `topo_index`
- `params_mb`
- `activ_out_mb`
- `trace_source`

Node column interpretation:

- `node_id`: Stable numeric key used for joins across graph, transfer, and ILP artifacts.
- `node_name`: Human-readable identifier of the operation/module represented by the node.
- `op_type`: Structural category (`call_module`, `call_function`, `call_method`, `placeholder`) used for semantic filtering.
- `topo_index`: Topological order index used to preserve dependency-consistent execution ordering.
- `params_mb`: Parameter memory footprint attributable to the node.
- `activ_out_mb`: Output activation size estimate; key driver of transfer cost when cuts occur.
- `trace_source`: Provenance marker (`fx` or `fallback`) indicating structural fidelity level.

Graph edge columns:

- `src_id`
- `dst_id`
- `tensor_mb`
- `tensor_shape`
- `producer_name`
- `consumer_name`
- `trace_source`

Edge column interpretation:

- `src_id`, `dst_id`: Directed dependency endpoints in node-id space.
- `tensor_mb`: Estimated tensor payload size moving across the dependency.
- `tensor_shape`: Shape metadata used for sanity checks and reproducibility.
- `producer_name`, `consumer_name`: Human-readable endpoint names for diagnostics and report readability.
- `trace_source`: Provenance marker aligning with node trace source.

### 6.4 How nodes are represented and viewed

Nodes are represented as tabular graph entities in `*_graph_nodes.csv`.

You can inspect the artifact quickly with:

```bash
python - <<'PY'
import pandas as pd
p='data/<host>/results/<model>/<optimizer>/<precision>/batch_<N>/run_001/<model>_graph_nodes.csv'
df=pd.read_csv(p)
print(df[['node_id','node_name','op_type','topo_index']].head(20).to_string(index=False))
PY
```

No dedicated graph-visualization renderer is currently shipped; graph data is instead exported for downstream ILP processing and external visualization tools.

---

## 7. Transfer calibration and edge-aware transfer cost model

This section formalizes communication cost between CPU and GPU. In practice, partition quality is often determined as much by transfer behavior as by compute speed, so this model is a first-class component of optimization validity.

Implemented in `TrainingProfiler`:

- `_measure_pci_and_overlap`
- `_measure_pci_bandwidth_detailed`
- `_build_edge_transfer_costs`

### 7.1 Direction-specific alpha-beta

For each direction (`h2d`, `d2h`), alpha and beta are estimated from measured transfers at two tensor sizes (10MB and 100MB in current detailed calibration):

$$
t_{dir}(S) = \alpha_{dir} + \frac{S}{\beta_{dir}}
$$

where:

- $S$: tensor size in MB
- $\alpha_{dir}$: latency term in ms
- $\beta_{dir}$: throughput term in MB/ms

### 7.2 Overlap attenuation

An overlap ratio $\sigma \in [0,1]$ is estimated, and transfer penalties are then attenuated:

$$
\text{overlap\_factor} = 1 - 0.5\sigma
$$

$$
t^{eff}_{h2d} = t^{raw}_{h2d} \cdot \text{overlap\_factor},
\quad
t^{eff}_{d2h} = t^{raw}_{d2h} \cdot \text{overlap\_factor}
$$

Why overlap attenuation is modeled:

- Real systems may overlap communication and computation; assuming fully serial transfer overestimates cut penalties.
- Assuming full overlap would be overly optimistic and can under-penalize fragmented partitions.
- The attenuation model provides a middle ground: it discounts transfer cost in proportion to observed overlap while preserving conservative behavior.

Interpretation of the factor:

- When `sigma_overlap = 0`, no overlap is observed and effective cost equals raw cost.
- As overlap increases, effective transfer cost decreases.
- The `0.5` coefficient limits maximum discount to 50%, deliberately avoiding unrealistically aggressive optimism.

### 7.3 Transfer edge artifact schema

`<model>_transfer_edges.csv` contains:

- `edge_id`
- `src_id`, `dst_id`
- `producer_name`, `consumer_name`
- `tensor_mb`
- `transfer_h2d_ms_raw`, `transfer_d2h_ms_raw`
- `transfer_h2d_ms`, `transfer_d2h_ms`
- `transfer_sym_ms`
- `alpha_h2d_ms`, `beta_h2d_mb_s`
- `alpha_d2h_ms`, `beta_d2h_mb_s`
- `sigma_overlap`

`transfer_sym_ms` is the edge transfer scalar consumed by ILP.

---

## 8. Metrics and metadata storage model

This section defines the artifact contract that makes the full pipeline reproducible. Each field is designed to be interpretable by both humans (for audit and reporting) and automated stages (for aggregation and ILP loading).

### 8.1 Per-layer CSV full column set

Final metrics rows are built in `TrainingProfiler.run_profiling`.

Columns:

- `layer`
- `model`
- `batch_size`
- `run_id`
- `seed`
- `type`
- `params_mb`
- `grads_mb`
- `optimizer_states_mb`
- `activations_mb`
- `theoretical_flops`
- `tflops`
- `efficiency_ratio`
- `gpu_fwd_time_ms`
- `gpu_bwd_time_ms`
- `gpu_fwd_energy_j`
- `gpu_bwd_energy_j`
- `gpu_mem_peak_mb`
- `layer_j_per_tflop_gpu`
- `dispatch_overhead_ratio`
- `cpu_fwd_time_ms`
- `cpu_bwd_time_ms`
- `cpu_fwd_energy_j`
- `cpu_bwd_energy_j`
- `cpu_mem_mb`
- `layer_j_per_tflop_cpu`
- `transfer_h2d_ms`
- `transfer_d2h_ms`
- `transfer_h2d_ms_legacy`
- `transfer_d2h_ms_legacy`
- `transfer_edge_aware_total_ms`
- `remat_penalty_ms`
- `precision_requested`
- `cpu_precision_executed`
- `gpu_precision_executed`
- `run_executed`
- `skip_unsupported_precision`
- `skip_reason`
- `optimizer`
- `opt_step_time_ms`

Column-by-column interpretation and utility:

- `layer`: Layer/module identifier; primary unit of ILP assignment.
- `model`: Model family/name; used for campaign partitioning and reporting.
- `batch_size`: Workload scale parameter; conditions all timing/energy observations.
- `run_id`: Replica identifier for robust aggregation.
- `seed`: Reproducibility anchor for initialization and data path randomness.
- `type`: Operator/module class; useful for stratified diagnostics.
- `params_mb`: Parameter memory footprint of the layer.
- `grads_mb`: Approximate gradient footprint; used in training-memory context.
- `optimizer_states_mb`: Optimizer-state memory component (for example moments).
- `activations_mb`: Activation memory footprint estimate.
- `theoretical_flops`: Workload estimate independent of runtime implementation.
- `tflops`: Observed compute throughput for the layer.
- `efficiency_ratio`: Layer throughput normalized by empirical peak.
- `gpu_fwd_time_ms`: GPU forward latency term.
- `gpu_bwd_time_ms`: GPU backward latency term (measured or heuristic path).
- `gpu_fwd_energy_j`: GPU forward energy term.
- `gpu_bwd_energy_j`: GPU backward energy term.
- `gpu_mem_peak_mb`: Peak GPU memory observed in the layer context.
- `layer_j_per_tflop_gpu`: Energy-intensity indicator on GPU.
- `dispatch_overhead_ratio`: Fraction of non-kernel framework overhead.
- `cpu_fwd_time_ms`: CPU forward latency term.
- `cpu_bwd_time_ms`: CPU backward latency term.
- `cpu_fwd_energy_j`: CPU forward energy term.
- `cpu_bwd_energy_j`: CPU backward energy term.
- `cpu_mem_mb`: CPU memory contribution estimate.
- `layer_j_per_tflop_cpu`: Energy-intensity indicator on CPU.
- `transfer_h2d_ms`: Host-to-device transfer estimate.
- `transfer_d2h_ms`: Device-to-host transfer estimate.
- `transfer_h2d_ms_legacy`: Legacy transfer estimator output (kept for comparability).
- `transfer_d2h_ms_legacy`: Legacy reverse transfer estimator output.
- `transfer_edge_aware_total_ms`: Edge-aware aggregate transfer penalty.
- `remat_penalty_ms`: Optional recomputation-related penalty term.
- `precision_requested`: Requested precision mode from CLI.
- `cpu_precision_executed`: Effective precision actually executed on CPU.
- `gpu_precision_executed`: Effective precision actually executed on GPU.
- `run_executed`: Boolean execution flag for run validity filtering.
- `skip_unsupported_precision`: Explicit unsupported-precision skip flag.
- `skip_reason`: Machine/human-readable reason for skip.
- `optimizer`: Optimizer name used in run.
- `opt_step_time_ms`: Optimizer-step latency contribution.

### 8.2 Metadata JSON key groups

Generated in `TrainingProfiler.run_profiling` and in skip paths.

Main groups:

1. Hardware identity
- host, torch version, OS, CPU model, GPU identity

2. Graph and transfer artifacts
- counts and file paths for nodes/edges/transfers

3. Timing and overhead
- total layer times and step times for CPU/GPU
- framework overhead and per-layer dispatch vector

4. Energy
- per-step and total energy CPU/GPU
- per-layer energy distribution vector

5. Memory and model-size totals
- global GPU peak
- totals for params, grads, activations

6. Transfer calibration
- alpha/beta values and raw PCIe calibration dict

7. Precision and policy diagnostics
- requested/executed precision
- ISA probe flags and FP16 preflight diagnostics
- execution status and skip reason if any

8. Optimizer timing
- total and average optimizer step timing

How to read these groups and why grouping exists:

- The JSON file is a run-level context envelope, while CSV is layer-level measurement detail.
- Grouping separates concerns (hardware identity, performance, energy, memory, policy diagnostics) so post-processing tools can parse only what they need.
- It also supports audit trails: decisions seen in ILP output can be traced back to device, precision-policy, and calibration context captured in metadata.

### 8.3 Host-scoped storage normalization

`normalize_output_dir_for_host` in `src/core/system.py` enforces one host namespace insertion under `data/<host>/...` to avoid collisions across machines.

Why this matters operationally:

- Multi-server campaigns frequently write similarly named artifacts.
- Without host scoping, runs from different machines can overwrite each other or become impossible to disambiguate.
- Host-scoped normalization guarantees deterministic, conflict-free storage and enables robust multi-hardware merges.

Typical pattern:

- Input root (user-provided): `data/results/...`
- Normalized output: `data/<host>/results/...`

This convention is fundamental for reproducibility and traceability in distributed experimental workflows.

---

## 9. Robust replicate statistics

Aggregation is not cosmetic post-processing: it is the mechanism that transforms noisy replica observations into robust optimization coefficients. Without this stage, ILP decisions would be highly sensitive to outliers and transient runtime conditions.

Implemented in `src/core/stats_aggregator.py` and CLI wrapper `validation/aggregate_metrics_stats.py`.

### 9.1 Aggregation behavior

Input discovery:

- recursively reads `*_metrics.csv`
- excludes `*_metrics_gpu_partial.csv`
- excludes existing `*_metrics_stats.csv`

Grouping keys (`GROUP_COLUMNS`):

- `model`
- `batch_size`
- `precision_requested`
- `optimizer`
- `layer`
- `type`
- `cpu_precision_executed`
- `gpu_precision_executed`

Per metric, it computes:

- mean
- std (sample std)
- p50
- p90
- p95

for each metric in `DEFAULT_METRIC_COLUMNS`.

What these metrics represent and why they are aggregated:

- Timing metrics (`*_time_ms`) capture latency channels used by ILP objective construction.
- Energy metrics (`*_energy_j`) capture energy channels used when `w_energy > 0`.
- Memory metrics (`*_mem_*`, `*_mb`) support feasibility constraints and diagnostics.
- Transfer-related metrics capture communication penalties that affect partition quality.

Why aggregation is useful:

- Single-run values are noisy due to runtime variability.
- Aggregation yields stable central tendencies and risk-aware tails.
- ILP coefficients built from aggregated statistics are more robust than coefficients built from single-run measurements.

### 9.2 Statistics equations

For values $x_1,\dots,x_n$:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\mu)^2}
$$

Quantiles $Q_{0.50}, Q_{0.90}, Q_{0.95}$ are empirical quantiles over run samples.

Why these equations are explicitly included:

- They define the exact statistical semantics of fields consumed by ILP loaders.
- They make robustification auditable and reproducible in thesis reporting.
- They clarify how uncertainty is carried from raw profiling data into optimization coefficients.

### 9.3 Output schema pattern

Each aggregated metric `m` yields:

- `m_mean`
- `m_std`
- `m_p50`
- `m_p90`
- `m_p95`

Also included:

- `n_samples`
- `n_runs`

How to use this schema in practice:

- `_mean` fields provide nominal coefficient values.
- `_std` fields quantify uncertainty and feed robustification (`mu + k*sigma`).
- `_p90`/`_p95` fields support tail-risk analysis and reporting.
- `n_samples`/`n_runs` provide confidence context; low sample counts should be interpreted conservatively.

---

## 10. ILP model: concept, mathematics, implementation

### 10.1 What an ILP is

An ILP (Integer Linear Program) optimizes a linear objective under linear constraints with integer, often binary, decision variables.

In this project, each layer is assigned to CPU or GPU with binary decisions.

Expanded intuition:

- ILP is used because the decision space is combinatorial: each layer can be assigned to one of two devices, and edge cuts create interaction costs.
- The linear form allows exact or near-exact solving with mature MILP tooling.
- Binary decisions map naturally to executable deployment actions (place layer on CPU or GPU), which is critical for practical adoption.

### 10.2 Sets and parameters

Let:

- $V$: layer nodes
- $E \subseteq V \times V$: directed graph edges
- for each node $v$:
  - $T_{gpu}(v), T_{cpu}(v)$: robust time cost
  - $E_{gpu}(v), E_{cpu}(v)$: robust energy cost
  - $M_{gpu}(v), M_{cpu}(v)$: memory contributions
- for each edge $e$:
  - $C_{tr}(e)$: transfer cut cost (`transfer_sym_ms`-based)

Why these parameters are included:

- Time and energy parameters encode execution-cost tradeoffs.
- Memory parameters encode physical feasibility.
- Transfer parameters encode communication penalties induced by partition cuts.

Together, they represent the minimum sufficient state to express a realistic heterogeneous partition objective under constraints.

### 10.3 Decision variables

Binary assignment:

$$
x_v \in \{0,1\}, \quad
x_v=1 \Rightarrow \text{GPU},\; x_v=0 \Rightarrow \text{CPU}
$$

Binary cut variable per edge:

$$
y_{uv} \in \{0,1\}, \quad (u,v) \in E
$$

Linearization constraints used in `src/ilp/solve.py`:

$$
y_{uv} \ge x_u - x_v
$$

$$
y_{uv} \ge x_v - x_u
$$

$$
y_{uv} \le x_u + x_v
$$

$$
y_{uv} \le 2 - x_u - x_v
$$

These enforce $y_{uv}=1$ iff assignments differ.

Interpretive note:

- `x_v` captures placement decisions.
- `y_uv` captures placement interaction effects along graph dependencies.
- This separation is what allows the model to remain linear while still representing cut costs.

### 10.4 Objective function in implementation

`build_problem_data` in `src/ilp/model_builder.py` constructs:

$$
\text{node\_gpu}(v) = w_t T_{gpu}(v) + w_e E_{gpu}(v)
$$

$$
\text{node\_cpu}(v) = w_t T_{cpu}(v) + w_e E_{cpu}(v)
$$

$$
\text{edge\_cut}(u,v) = w_{tr} C_{tr}(u,v)
$$

Optimized objective:

$$
\min Z = \sum_{v \in V}\left[x_v\,\text{node\_gpu}(v) + (1-x_v)\,\text{node\_cpu}(v)\right]
+ \sum_{(u,v)\in E} y_{uv}\,\text{edge\_cut}(u,v)
$$

Why this construction is used:

- The node term selects CPU or GPU cost for each layer based on `x_v`.
- The edge term activates only when assignments differ (via `y_uv`), capturing communication overhead.
- Weighting (`w_t`, `w_e`, `w_tr`) makes objective preferences explicit and experimentally tunable.

### 10.5 Memory constraints

Implemented in `src/ilp/solve.py`:

$$
\sum_{v\in V} M_{gpu}(v)\,x_v \le B_{gpu}
$$

$$
\sum_{v\in V} M_{cpu}(v)\,(1-x_v) \le B_{cpu}
$$

where:

- $B_{gpu}$ = `gpu_mem_budget_mb`
- $B_{cpu}$ = `cpu_mem_budget_mb`

Why these constraints are essential:

- They prevent formally optimal but physically impossible assignments.
- They provide direct control for Pareto analysis under varying GPU budgets.
- They encode deployment realism, which is required for thesis-grade practical validity.

### 10.6 Robust parametrization from statistics

`load_ilp_inputs` in `src/ilp/data_loader.py` computes robust values using:

$$
\hat{m} = \mu_m + k_\sigma \sigma_m
$$

for time and energy channels, where `k_sigma` is configurable.

Specifically:

$$
T_{gpu}(v)=\widehat{gpu\_fwd\_time}(v)+\widehat{gpu\_bwd\_time}(v)
$$

$$
T_{cpu}(v)=\widehat{cpu\_fwd\_time}(v)+\widehat{cpu\_bwd\_time}(v)
$$

$$
E_{gpu}(v)=\widehat{gpu\_fwd\_energy}(v)+\widehat{gpu\_bwd\_energy}(v)
$$

$$
E_{cpu}(v)=\widehat{cpu\_fwd\_energy}(v)+\widehat{cpu\_bwd\_energy}(v)
$$

Why this robustification is used:

- Runtime measurements contain stochastic variability.
- Using `mu + k*sigma` creates uncertainty-aware coefficients.
- The parameter `k_sigma` controls conservatism and can be tuned per experimental objective.

### 10.7 Multi-hardware aggregation

`merge_ilp_inputs_multi_hardware` supports two strategies:

1. Conservative max:

$$
\bar{c} = \max_i c_i
$$

2. Mean with dispersion margin:

$$
\bar{c} = \mu(c) + k_d\sigma(c)
$$

where $k_d$ is `hw_dispersion_k`.

This is applied to node costs, energies, memories, and edge transfer costs across hardware profiles.

Why multi-hardware aggregation is needed:

- Single-machine optimization can overfit to one platform.
- Aggregation builds policies that remain valid across heterogeneous server pools.
- `max` favors worst-case robustness; `mean + k_d*sigma` favors balanced robustness-performance tradeoffs.

### 10.8 Why this ILP formulation was chosen

Practical reasons aligned with thesis constraints:

- Binary layer assignment is interpretable and auditable.
- Linear objective and constraints support mature MILP solvers.
- Transfer costs naturally map to edge cut variables.
- Memory constraints are directly representable.
- Robust terms from measured variability are easy to inject as linear coefficients.

### 10.9 Solver backends

`solve_partition_ilp` in `src/ilp/solve.py`:

- `auto`: PuLP CBC if available, else exhaustive search
- `pulp`: MILP solver path
- `exhaustive`: brute force (guarded: max 22 nodes)

Backend selection rationale:

- `pulp` is preferred for realistic graph sizes due to solver scalability.
- `exhaustive` is retained as a correctness oracle for small instances.
- `auto` maximizes portability by degrading gracefully when solver dependencies are unavailable.

---

## 11. How measured data is used by ILP, end-to-end

The goal of this section is methodological closure: it shows the complete causal path from empirical measurements to optimization decisions, so final assignments can be traced back to observable runtime behavior.

1. Profiling writes per-run metrics and graph artifacts.
2. Replicate stats computes robust moments by `(model,batch,precision,optimizer,layer,...)`.
3. ILP loader maps:
   - robust node time and energy from `*_metrics_stats.csv`
   - graph edges from `*_graph_edges.csv`
   - transfer costs from `*_transfer_edges.csv`
4. ILP builder creates weighted objective terms and memory vectors.
5. Solver returns assignment and cut edges.
6. Pareto sweep repeats optimization under multiple GPU budgets.

For multiple hardware configurations:

- each hardware profile contributes one ILP input
- profiles are merged with `max` or `mean + k*std`
- a single robust ILP instance is solved

---

## 12. Script execution catalog (how to run everything)

This section is operational by design. It documents not only command syntax, but also the experimental dimensions controlled by each parameter, which is essential for reproducibility across hardware classes.

### 12.1 Main profiling campaign

Script: `scripts/run_experiments.sh`

Canonical command:

```bash
bash scripts/run_experiments.sh
```

Common environment controls:

- grid override:
  - `MODELS_CSV`
  - `BATCH_SIZES_CSV`
  - `PRECISIONS_CSV`
  - `OPTIMIZERS_CSV`
- runtime:
  - `USE_SKIP_CPU=true|false`
  - `ENABLE_RAPL=true|false`
  - `FORCE_THREADS=N`
  - `REPEATS=N`
  - `WARMUP=N`
  - `MEASURE=N`
  - `FAIL_FAST=true|false`
  - `DRY_RUN=true|false`

Example:

```bash
MODELS_CSV=simple_mlp,resnet50 \
BATCH_SIZES_CSV=8,16 \
PRECISIONS_CSV=fp32 \
OPTIMIZERS_CSV=SGD,AdamW \
REPEATS=3 \
USE_SKIP_CPU=true \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

### 12.2 End-to-end thesis smoke workflow

Script: `scripts/run_thesis_smoke_workflow.sh`

Runs the complete reduced pipeline:

1. profile reduced grid
2. aggregate stats
3. ILP partition
4. Pareto sweep
5. generate report assets
6. export LaTeX tables

Command:

```bash
bash scripts/run_thesis_smoke_workflow.sh
```

### 12.3 Single ILP partition

Wrapper: `scripts/run_ilp_partition.sh`

```bash
MODEL=simple_mlp \
CONFIG_DIR=data/<host>/results/simple_mlp/SGD/fp32/batch_8 \
K_SIGMA=1.0 W_TIME=1.0 W_ENERGY=0.0 W_TRANSFER=1.0 \
GPU_MEM_BUDGET_MB=1e18 CPU_MEM_BUDGET_MB=1e18 \
BACKEND=auto \
bash scripts/run_ilp_partition.sh
```

### 12.4 ILP Pareto sweep

Wrapper: `scripts/run_ilp_pareto_sweep.sh`

```bash
MODEL=resnet50 \
CONFIG_DIR=data/<host>/results/resnet50/SGD/fp32/batch_8 \
GPU_BUDGETS_MB=400,600,800,1000 \
CPU_MEM_BUDGET_MB=1e18 \
BACKEND=auto \
bash scripts/run_ilp_pareto_sweep.sh
```

### 12.5 Multi-node config discovery

Script: `scripts/discover_ilp_config_dirs.sh`

```bash
MODEL=simple_mlp OPTIMIZER=SGD PRECISION=fp32 BATCH=8 \
MODE=print \
bash scripts/discover_ilp_config_dirs.sh
```

`MODE=partition` and `MODE=pareto` can launch wrappers directly.

### 12.6 Consolidated report assets and plots

Wrapper: `scripts/generate_ilp_report_assets.sh`

```bash
INPUT_ROOT=data/<host>/results_smoke \
OUTPUT_DIR=reports/ilp_results/<host>_smoke \
bash scripts/generate_ilp_report_assets.sh
```

Plot generation implementation: `validation/generate_ilp_report_assets.py`

Generated plots:

- `<model>_objective_vs_budget.png`
- `best_ilp_vs_all_cpu_improvement.png`

### 12.7 LaTeX table export

Wrapper: `scripts/export_ilp_tables_latex.sh`

```bash
BEST_CSV=reports/ilp_results/ilp_best_per_model.csv \
CONSOLIDATED_CSV=reports/ilp_results/ilp_pareto_consolidated.csv \
OUT_DIR=reports/ilp_results/latex \
bash scripts/export_ilp_tables_latex.sh
```

Implementation: `validation/export_ilp_tables_latex.py`

### 12.8 Legacy HPC launcher

Script: `scripts/launch_grid5k.sh`

Important note:

- this script currently uses CLI argument names that do not match current `src/profiler.py` parser (for example `--batch-size`, `--gpu-id` with hyphen style and model names like `bert`, `vit`), so it should be treated as legacy and updated before production use.

---

## 13. Validation and test framework

This section addresses scientific rigor at the workflow level: generating outputs is not sufficient unless execution policies, structural assumptions, and failure guards are systematically verified.

### 13.1 Unit tests

Run:

```bash
bash validation/run_unit_tests.sh
```

Tests in `tests/` include:

- `test_precision_policy_unit.py`
- `test_profiler_gpu_only_precision_policy.py`
- `test_timeout_validation.py`

### 13.2 Structural and behavior guards

- `validation/validate_code.py`: timeout and integration integrity checks
- `validation/validate_zombie_fix.py`: checks `--skip_cpu` and `--num_threads` integration
- `validation/validate_all_models.py`: broad model and preflight validation
- `validation/comprehensive_check.sh`: grep-based architecture checks

---

## 14. Column-level utility for ILP (why each data block matters)

The focus here is the interpretability of coefficients. It explains why each column family exists and how each one influences objective terms, constraints, or data-quality filters in the optimization stage.

### 14.1 Compute cost block

Used for node objective terms:

- `gpu_fwd_time_ms`, `gpu_bwd_time_ms`
- `cpu_fwd_time_ms`, `cpu_bwd_time_ms`

These become robust node time costs in ILP.

### 14.2 Energy block

Used for optional energy-weighted objective:

- `gpu_fwd_energy_j`, `gpu_bwd_energy_j`
- `cpu_fwd_energy_j`, `cpu_bwd_energy_j`

Weighted by `w_energy` in ILP objective.

### 14.3 Memory block

Used in hard memory constraints:

- `gpu_mem_peak_mb` -> node GPU memory contribution
- `cpu_mem_mb` -> node CPU memory contribution

### 14.4 Transfer block

Used in edge cut penalty:

- from transfer artifact: `transfer_sym_ms`

Mapped to edge objective coefficient and multiplied by cut variable $y_{uv}$.

### 14.5 Precision diagnostics block

Useful for filtering and run-quality control:

- `precision_requested`
- `cpu_precision_executed`
- `gpu_precision_executed`
- `run_executed`
- `skip_unsupported_precision`
- `skip_reason`

These fields prevent invalid rows from contaminating aggregation/ILP input.

---

## 15. Known limitations and operational cautions

Any applied optimization pipeline has validity limits. Making those limits explicit helps prevent over-interpretation and creates a concrete roadmap for future methodological improvements.

1. `launch_grid5k.sh` is legacy and not aligned with current CLI names.
2. Graph fallback is linearized and less expressive than FX graph.
3. Energy quality depends on sensor availability (NVML/pyRAPL).
4. Exhaustive ILP backend is only for small node counts.
5. Transfer model is first-order alpha-beta with overlap approximation.

---

## 16. Recommended reproducible workflow for thesis-grade runs

The recommended sequence prioritizes experimental risk control: validate the environment and execution policy first, scale to full campaigns second, and only then consolidate optimization and reporting artifacts.

1. Environment setup and validation.
2. Run canonical preflight (`SMOKE_MODE=true`, `DRY_RUN=true`).
3. Run real smoke (small true execution) on each hardware class.
4. Run full or profile-specific campaign with `run_experiments.sh`.
5. Confirm artifact completeness and aggregate stats generation.
6. Execute ILP partition and Pareto sweeps.
7. Generate report plots and LaTeX tables.
8. For multi-hardware analysis, merge profiles with `hw_aggregate=max` or `mean` + `hw_dispersion_k`.

---

## 17. Quick command index

This index is intended as a fast operational reference. It accelerates repeated execution, while the deeper methodological rationale remains documented in the earlier sections.

Run full campaign:

```bash
bash scripts/run_experiments.sh
```

Run smoke end-to-end thesis workflow:

```bash
bash scripts/run_thesis_smoke_workflow.sh
```

Aggregate a configuration folder manually:

```bash
python validation/aggregate_metrics_stats.py --input_dir <config_dir> --output_csv <config_dir>/<model>_metrics_stats.csv
```

Run ILP partition:

```bash
python validation/run_ilp_partition.py --config_dir <config_dir> --model <model>
```

Run ILP Pareto sweep:

```bash
python validation/sweep_ilp_pareto.py --config_dir <config_dir> --model <model> --gpu_budgets_mb 400,600,800
```

Generate report assets:

```bash
python validation/generate_ilp_report_assets.py --input_root <root> --output_dir <reports_dir>
```

Export LaTeX tables:

```bash
python validation/export_ilp_tables_latex.py --best_csv <best.csv> --consolidated_csv <consolidated.csv> --output_dir <latex_dir>
```

---

## 18. Additional project references

- `README.md`
- `docs/README.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/MULTI_NODE_ILP_RUNBOOK.md`
- `docs/ILP_ROBUST_PARTITIONING_PLAN.md`
- `docs/SERVER_LAUNCH_PROFILES.md`
- `docs/documentation.md`

---

## 19. Extended Academic Narrative

### 19.1 Pedagogical introduction: from practical problem to scientific model

In a heterogeneous environment (CPU + GPU), training a deep learning model requires deciding where each computational block should run. This decision is not trivial: moving more layers to GPU often reduces raw compute time, but it can also increase memory pressure and inter-device transfer overhead. Running layers on CPU may reduce GPU memory pressure, but usually at the cost of higher latency.

This project transforms that practical dilemma into a reproducible scientific optimization workflow. First, it measures real per-layer behavior (time, energy, memory, FLOPs, and transfer costs). Then, it converts these measurements into coefficients of an Integer Linear Programming (ILP) problem that decides an optimal CPU/GPU assignment under physical constraints (memory budgets) and performance criteria (time, energy, transfer).

This methodology is suitable for doctoral-level reporting for two reasons. First, every optimization decision is grounded in observed data rather than abstract assumptions. Second, the full pipeline is auditable end-to-end, so every final numerical result can be traced back to the measurements and transformations that produced it.

### 19.2 Guided reading for mixed audiences (non-specialist and specialist)

For non-specialists, each layer can be viewed as a task in a production chain. Some tasks run faster on a specialized machine (GPU), while others can remain on the general-purpose machine (CPU). Moving outputs between machines incurs an additional cost (transfer). The problem is therefore to distribute tasks so that total cost is minimized while temporary storage limits (memory) are respected.

For specialists, the formulation is a binary partitioning of a DAG with edge-cut penalties and resource constraints. Node costs are affine combinations of robustified time and energy terms, and edge costs are derived from an alpha-beta transfer model attenuated by overlap. The resulting MILP instance is solved with an exact backend (CBC via PuLP) or an exhaustive fallback for small instances.

### 19.3 Why the pipeline starts with profiling instead of direct ILP

An ILP requires numerical coefficients. If those coefficients do not represent real hardware and model behavior, the solution can be formally optimal yet practically invalid. Therefore, profiling is not an accessory step; it is the empirical foundation of model validity.

Profiling captures run-to-run variability (system noise, scheduler jitter, thermal effects, cache states, background load). That variability is integrated through robust statistics, so ILP optimization targets a conservative, defensible operating regime rather than an idealized one.

---

## 20. ILP Deep Dive: detailed conceptual and mathematical explanation

### 20.1 Rigorous definition and mapping to this use case

An ILP (Integer Linear Programming) problem minimizes (or maximizes) a linear objective under linear constraints, with part of the variables constrained to integer values. In this project, decision variables are binary, yielding a 0-1 ILP formulation.

The mapping is direct:

- each graph node (layer) gets a binary variable `x_v`
- `x_v = 1` means execute on GPU
- `x_v = 0` means execute on CPU

Because layers are connected by dependencies (edges), assigning adjacent nodes to different devices introduces transfer overhead. That overhead is modeled with a second binary variable `y_uv` per edge `(u,v)`, activated when a cut occurs (different assignments at edge endpoints).

### 20.2 Physical meaning of the objective function

The objective combines three components:

1. node-level time cost
2. node-level energy cost
3. transfer cost for cut edges

Interpretation:

- the time term approximates total compute latency
- the energy term captures efficiency and operational sustainability
- the transfer term penalizes overly fragmented partitions

Compactly:

$$
\min Z = \underbrace{\sum_{v\in V}\left[x_v C_{gpu}(v) + (1-x_v)C_{cpu}(v)\right]}_{\text{node cost}}
+ \underbrace{\sum_{(u,v)\in E} y_{uv} C_{cut}(u,v)}_{\text{edge-cut cost}}
$$

con:

$$
C_{gpu}(v)=w_t T_{gpu}(v)+w_e E_{gpu}(v),
\quad
C_{cpu}(v)=w_t T_{cpu}(v)+w_e E_{cpu}(v)
$$

$$
C_{cut}(u,v)=w_{tr}C_{tr}(u,v)
$$

The weights $w_t$, $w_e$, and $w_{tr}$ are not universal constants; they encode an experimental preference profile. If the primary objective is latency, increase $w_t$. If energy efficiency is critical, increase $w_e$. If interconnect bandwidth is the bottleneck, increase $w_{tr}$.

### 20.3 Why edge-cut linearization is correct

Conceptually, the desired cut term is $|x_u - x_v|$. Because absolute value is not used directly in this linear binary formulation, `y_uv` is introduced with inequalities that enforce:

- `y_uv = 0` when `x_u = x_v`
- `y_uv = 1` when `x_u != x_v`

The four constraints:

$$
y_{uv}\ge x_u-x_v,
\quad
y_{uv}\ge x_v-x_u,
\quad
y_{uv}\le x_u+x_v,
\quad
y_{uv}\le 2-x_u-x_v
$$

exactly encode this logic for binary variables. This is a standard combinatorial optimization technique for modeling XOR/cut behavior while preserving linearity.

### 20.4 Memory constraints as physical feasibility guards

Without memory constraints, the optimizer could place too many layers on GPU to reduce latency, generating an infeasible out-of-memory solution. Therefore, hard limits are imposed:

$$
\sum_{v\in V} M_{gpu}(v)x_v \le B_{gpu},
\quad
\sum_{v\in V} M_{cpu}(v)(1-x_v) \le B_{cpu}
$$

These constraints are the mathematical translation of physical device capacity. In thesis terms, they bridge formal elegance and executable reality.

### 20.5 Statistical robustness: why not use mean only

In real systems, two identical runs can produce different timings. A mean-only model may underestimate operational risk. Therefore, the project uses:

$$
\hat{m}=\mu_m + k_\sigma \sigma_m
$$

The constant $k_\sigma$ controls conservatism:

- $k_\sigma=0$: nominal (aggressive) optimization
- $k_\sigma>0$: robust (more conservative) optimization

Conceptually, this accepts a small expected over-cost in order to reduce degradation risk in production.

### 20.6 Multi-hardware integration: scientific interpretation

When a single policy is needed across multiple servers, two aggregation options are available:

1. `max`: adopts the worst case per coefficient
2. `mean + k*std`: adopts central tendency with a dispersion margin

The `max` option prioritizes operational safety (worst-node robustness), while `mean + k*std` provides a tradeoff between average performance and tolerance to cross-machine variability.

---

## 21. Narrative data dictionary (with meaning, role, and interpretation)

### 21.1 Graph node fields (`*_graph_nodes.csv`)

This table describes the computational units that ILP assigns to CPU or GPU. Each row represents one execution-graph node.

`node_id`:
Unique integer identifier within the graph. It is the technical key used to join nodes with edges and ILP assignments. It has referential, not semantic, meaning.

`node_name`:
Human-readable node name (for example, module or function name). It is central for interpretability and traceability in result analysis.

`op_type`:
Operation type (for example, `call_module`, `call_function`, `call_method`, `placeholder`). It distinguishes operational semantics among parametrized layers, functional operators, and input placeholders.

`topo_index`:
Topological index (dependency-consistent order). It helps reconstruct dataflow and verify acyclic graph consistency.

`params_mb`:
Parameter size in MB. It approximates persistent model state associated with the node.

`activ_out_mb`:
Output activation size in MB. It is key to estimating transfer cost when CPU/GPU cuts occur.

`trace_source`:
Trace provenance (`fx` or `fallback`). This field communicates structural fidelity: `fx` is usually closer to true execution dependencies, while `fallback` guarantees artifact availability with reduced topological detail.

### 21.2 Graph edge fields (`*_graph_edges.csv`)

Edges represent data dependencies between nodes; in ILP they become transfer-penalty candidates when cut across devices.

`src_id` and `dst_id`:
Source and destination node IDs, defining dependency direction (producer to consumer).

`tensor_mb`:
Approximate transferred tensor size. This is the primary physical variable for transfer-latency estimation.

`tensor_shape`:
Tensor shape (when available), useful for technical auditing and dimensional consistency checks.

`producer_name` and `consumer_name`:
Human-readable endpoint names, improving interpretation of cut edges in reports.

`trace_source`:
Same semantics as nodes: indicates whether the edge came from full FX tracing or fallback reconstruction.

### 21.3 Transfer edge fields (`*_transfer_edges.csv`)

This table quantifies expected CPU/GPU data-movement cost for each candidate edge.

`edge_id`:
Unique transfer-edge identifier.

`transfer_h2d_ms_raw` and `transfer_d2h_ms_raw`:
Baseline directional times before overlap attenuation, derived from calibrated alpha-beta transfer models.

`transfer_h2d_ms` and `transfer_d2h_ms`:
Effective directional times after overlap adjustment, offering a more realistic approximation under potential compute-communication overlap.

`transfer_sym_ms`:
Symmetric scalar transfer cost consumed by ILP for edge-cut penalties. This is the key transfer column for objective construction.

`alpha_*`, `beta_*`, `sigma_overlap`:
Calibration and overlap parameters documenting the physical origin of transfer costs, supporting reproducibility and experimental auditability.

### 21.4 Layer metrics fields (`*_metrics.csv`)

This table contains empirical per-layer observations per run. It is not only a log; it is the raw material for robust optimization inputs.

Identity fields (`layer`, `model`, `batch_size`, `run_id`, `seed`, `type`):
These fields ensure that only comparable replicas are grouped together. Without them, aggregation can mix non-equivalent scenarios.

Memory fields (`params_mb`, `grads_mb`, `optimizer_states_mb`, `activations_mb`, `gpu_mem_peak_mb`, `cpu_mem_mb`):
These quantify memory footprint from multiple perspectives (persistent model state, training state, and observed peaks), and are essential for feasibility constraints.

Compute fields (`theoretical_flops`, `tflops`, `efficiency_ratio`):
These connect theoretical workload to measured throughput. `efficiency_ratio` indicates proximity to empirically measured device peak.

Time and energy fields (`*_time_ms`, `*_energy_j`):
These are the primary node-cost channels. Combined with objective weights, they drive device preference at node level.

Transfer and overhead fields (`dispatch_overhead_ratio`, `transfer_*`, `remat_penalty_ms`):
These capture non-FLOP effects that materially impact runtime behavior. Ignoring them often yields over-optimistic models.

Policy/state fields (`precision_requested`, `cpu_precision_executed`, `gpu_precision_executed`, `run_executed`, `skip_unsupported_precision`, `skip_reason`):
These enforce experimental hygiene by allowing exclusion of skipped or invalid executions.

`opt_step_time_ms`:
Optimizer-step timing, relevant for end-to-end training analysis beyond pure forward/backward cost.

### 21.5 Aggregated metrics fields (`*_metrics_stats.csv`)

Each base metric yields derived columns (`_mean`, `_std`, `_p50`, `_p90`, `_p95`).

Methodological interpretation:

- `_mean`: central expected behavior
- `_std`: run-to-run volatility
- `_p50`: outlier-robust median behavior
- `_p90`/`_p95`: upper-tail operational risk behavior

`n_samples` and `n_runs`:
These provide statistical confidence context; means computed from few samples are less robust inferentially.

### 21.6 ILP solution fields (outputs)

`ilp_assignment.csv`:
Node-level CPU/GPU decisions and associated costs.

`ilp_cut_edges.csv`:
Edges effectively cut by the final assignment, exposing where transfer penalties are paid.

`ilp_solution_summary.json`:
Total objective, objective decomposition, and solver execution metadata.

Taken together, these outputs enable causal analysis: not only what objective value was achieved, but also why that assignment was selected.

---

## 22. Justification of design choices for doctoral-level discussion

### 22.1 Decision granularity at layer level

Layer-level granularity was selected because it balances expressiveness and tractability. A finer granularity (kernel- or primitive-operation level) increases problem size and measurement noise, whereas a coarser granularity (large blocks) may hide meaningful partition opportunities.

### 22.2 Binary assignment versus fractional allocation

A binary formulation better matches runtime execution reality: each layer runs on CPU or GPU in a given execution instance. Fractional models (continuous relaxations) can be informative for theoretical bounds, but they require post-processing to recover executable decisions.

### 22.3 Linear objective and constraints

Maintaining linearity enables robust, mature solvers with stable behavior and reproducible explanations. In an applied thesis context, this improves experiment comparability and methodological transparency.

### 22.4 Robust statistics and scientific validity

Including observed variance via $\mu + k\sigma$ reduces the risk of conclusions tied to idealized, non-repeatable conditions. This strengthens external validity by making the resulting policy less sensitive to platform noise.

### 22.5 Multi-hardware aggregation as transferability mechanism

Merging multiple hardware profiles supports partition policies that transfer across servers. Instead of optimizing for a single machine (infrastructure overfitting risk), the method builds a more generalizable policy.

---

## 23. Reading map for thesis chapter integration

To integrate this material into a thesis chapter, the following narrative sequence is recommended:

1. practical motivation: heterogeneous assignment problem
2. empirical measurement methodology (profiling)
3. structural representation (graph)
4. statistical robustification (replicas and dispersion)
5. ILP formulation (variables, objective, constraints)
6. results (partitioning, Pareto, reports)
7. limitations and future work

This sequence supports progressive reading: non-specialists can follow the physical intuition, while specialists can drill down into the equations and modeling decisions.

---

## 24. Suggested future enrichments (optional)

For an even more complete thesis version, the following extensions can be added:

1. complete numeric example on a small DAG (3-5 nodes) with manual ILP walkthrough
2. systematic sensitivity analysis over `w_time`, `w_energy`, `w_transfer`, and `k_sigma`
3. formal comparison against baselines (`all_cpu`, `all_gpu`, heuristic policies)
4. temporal stability analysis (day-level drift, thermal variation, shared-load effects)
5. computational-complexity appendix by solver backend

These extensions do not change the current pipeline, but they strengthen the scientific argument and the transferability of conclusions.

Internal references:

- `README.md`
- `docs/README.md`
- `docs/GLOBAL_PROJECT_DOCUMENTATION.md`
- `docs/GLOBAL_PROJECT_DOCUMENTATION_ES.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/MULTI_NODE_ILP_RUNBOOK.md`
- `docs/ILP_ROBUST_PARTITIONING_PLAN.md`
- `docs/SERVER_LAUNCH_PROFILES.md`
- `docs/documentation.md`

---

Last updated: March 14, 2026.
