# Architecture Audit

This audit consolidates the current file-level inventory for `src/` and `validation/`, plus technical debt and operational risks observed in the current implementation.

## Scope and Method

- Scope: current repository state as of March 12, 2026.
- Focus areas: source architecture, validation toolchain, profiler-to-ILP pipeline robustness.
- Evidence basis: repository files under `src/`, `validation/`, and active documentation.

## File Inventory: `src/`

### Root

- `src/__init__.py`: Root package marker.
- `src/profiler.py`: Main CLI, argument parsing, precision policy, runtime initialization, orchestration entrypoint.
- `src/profiler_en.md`: English profiler usage/technical notes.
- `src/profiler_es.md`: Spanish profiler usage/technical notes.

### Core (`src/core/`)

- `src/core/__init__.py`: Core package marker.
- `src/core/constants.py`: Shared constants (`BACKWARD_FACTOR`, defaults, optimizer overhead map).
- `src/core/energy.py`: GPU/CPU energy monitor abstraction (NVML, optional pyRAPL).
- `src/core/graph_extractor.py`: Graph export using `torch.fx` with fallback leaf-module DAG.
- `src/core/io_artifacts.py`: CSV/JSON write helpers and partial artifact cleanup.
- `src/core/metrics.py`: Tensor size recursion and per-layer FLOPs estimation.
- `src/core/precision_policy.py`: ISA probing, precision policy evaluation, CPU FP16 preflight.
- `src/core/stats_aggregator.py`: Replicate aggregation into robust stats table (`mean/std/p50/p90/p95`).
- `src/core/system.py`: Determinism setup, CPU thread config, hardware metadata collection.

### Models (`src/models/`)

- `src/models/__init__.py`: Models package marker.
- `src/models/factory.py`: Supported model/input factory (`resnet50`, `resnet152`, `vit_b16`, `bert_base`, `gpt2_small`, `simple_mlp`).

### Runner (`src/runner/`)

- `src/runner/__init__.py`: Runner package marker.
- `src/runner/training_profiler.py`: End-to-end profiling pipeline (warmup/measure loops, hooks, energy, graph/transfer artifacts, final CSV/JSON).

### ILP (`src/ilp/`)

- `src/ilp/__init__.py`: ILP package marker.
- `src/ilp/data_loader.py`: Loads robust metrics, graph edges, and transfer edges into ILP input structure.
- `src/ilp/model_builder.py`: ILP config validation and objective component assembly.
- `src/ilp/solve.py`: Solver backend selection (`pulp` or exhaustive fallback).
- `src/ilp/export_solution.py`: Writes ILP solution outputs (`assignment`, `cut_edges`, summary).

## File Inventory: `validation/`

- `validation/VALIDATION_SUMMARY.sh`: Wrapper summary script for validation runs.
- `validation/aggregate_metrics_stats.py`: CLI wrapper for robust stats aggregation.
- `validation/comprehensive_check.sh`: Grep-based structural integrity audit.
- `validation/export_ilp_tables_latex.py`: Export ILP tables to LaTeX.
- `validation/generate_ilp_report_assets.py`: Generate ILP report assets.
- `validation/run_ilp_partition.py`: Solve one ILP partition configuration from artifacts.
- `validation/run_unit_tests.sh`: Unified unit test launcher.
- `validation/sweep_ilp_pareto.py`: GPU memory budget sweep for ILP Pareto analysis.
- `validation/validate_all_models.py`: Multi-model validation and configurable preflight scope.
- `validation/validate_code.py`: Structural checks for timeout/preflight implementation.
- `validation/validate_zombie_fix.py`: Structural checks for `--skip_cpu`/`--num_threads` flow.

## Technical Debt and Real Risks

### 1) Documentation drift

- `docs/PROJECT_STRUCTURE.md` was previously behind actual code evolution (ILP package and robust aggregation pipeline additions).
- Risk: onboarding errors, incorrect runbook assumptions, and thesis reproducibility friction.

### 2) Monolithic runtime orchestration

- `src/runner/training_profiler.py` centralizes many concerns (measurement, preflight gating, graph export, transfer modeling, CSV/JSON assembly).
- Risk: high regression surface for small changes and difficult unit isolation.

### 3) Structural tests coupled to source text

- Validation scripts rely on source-introspection patterns in `run_profiling` rather than behavior-only tests.
- Risk: harmless refactors can fail checks even if runtime behavior is unchanged.

### 4) Solver scalability fallback

- `src/ilp/solve.py` exhaustive backend is capped at small node counts (<=22) when `pulp` is unavailable.
- Risk: practical ILP execution can silently become unavailable for realistic graphs in constrained environments.

### 5) Environment-dependent energy fidelity

- GPU energy requires NVML; CPU energy requires optional pyRAPL.
- Risk: partial or incomparable energy data across machines if sensors/packages are missing.

### 6) Graph extraction fallback quality

- When `torch.fx` tracing fails, fallback graph is sequential leaf-module approximation.
- Risk: topology simplification can bias transfer edge modeling and ILP cut-cost realism.

### 7) Robust stats quality guardrails

- Aggregation computes robust moments/percentiles but does not enforce strict minimum sample thresholds by default.
- Risk: ILP decisions based on under-sampled variability estimates.

### 8) Artifact naming/mapping fragility

- `src/ilp/data_loader.py` depends on naming consistency between metrics, graph, and transfer artifacts.
- Risk: mismatches lead to dropped edges or zero-filled transfers (warnings unless strict flags are enabled).

### 9) Campaign combinatorics and runtime cost

- `scripts/run_experiments.sh` supports large cartesian sweeps (models x precisions x optimizers x batches x repeats).
- Risk: long runtimes, OOM incidents, and operational overhead without careful campaign constraints.

### 10) Validation emphasis balance

- Strong structural checks exist, but full behavioral/performance regression suites are comparatively thinner.
- Risk: semantic drift that remains undetected by text/structure-focused validators.

## Recommended Mitigations

1. Keep architecture docs synced as part of release checklist.
2. Continue incremental decomposition of `training_profiler.py` into testable helper methods.
3. Preserve structural validator expectations while adding behavior-first tests.
4. Treat `pulp` availability as deployment prerequisite for medium/large ILP runs.
5. Add environment capability report to experiment metadata (sensor availability + quality flags).
6. Add confidence checks in ILP preprocessing (minimum `n_samples` thresholds).
7. Run strict mapping mode in CI for ILP pipelines (`--strict_graph_mapping`, `--strict_transfer_mapping`).
8. Define bounded default campaign profiles (`smoke`, `standard`, `thesis_full`).

## End-to-End Flow (Summary)

1. `src/profiler.py` configures run, precision policy, and runtime.
2. `src/runner/training_profiler.py` profiles model and emits per-run artifacts.
3. `src/core/stats_aggregator.py` aggregates replicate runs into robust metrics stats.
4. `validation/run_ilp_partition.py` or `validation/sweep_ilp_pareto.py` loads robust artifacts.
5. `src/ilp/solve.py` computes assignment; `src/ilp/export_solution.py` exports outputs.

---

Last updated: March 12, 2026.
