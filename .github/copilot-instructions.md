# Copilot / Agent Instructions — Hybrid CPU↔GPU Profiler

Short, actionable guidance to help coding agents be immediately productive in this repo.

## Quick context (big picture)
- Purpose: produce per-layer profiling metrics (time, FLOPs, energy, memory, transfers) used to parameterize an ILP optimizer (see `documentation.md`).
- Entry point: `src/profiler.py` (main profiling workflow). A partial variant exists as `src/profiler_update.py`.
- Outputs (default): `data/{model_name}_metrics.csv` (per-layer rows) and `data/{model_name}_meta.json` (global metadata).

## Key components & responsibilities
- `src/profiler.py`:
  - Model factory and precision casting (HuggingFace models use `torch_dtype` in `from_pretrained`).
  - `TrainingProfiler` class: registers pre/post hooks on **leaf** modules and accumulates metrics.
  - `EnergyMonitor` thread: samples NVML (GPU) or pyRAPL (CPU) and returns average power.
  - GEMM micro‑benchmark (`_measure_peak_flops`) to compute empirical TFLOPS.
  - PCIe alpha–beta calibration (`_measure_pci_bandwidth`) to estimate transfer times.
  - Outputs: CSV + JSON written to `--output-dir` (default `data`).
- `src/profiler_update.py`: supplementary/experimental changes — read for ideas but trust `src/profiler.py` as canonical.
- `documentation.md` and `src/profiler_en.md`: authoritative descriptions and the CSV/JSON schema.

## Project-specific conventions & patterns (use these when editing code)
- Profiling granularity is "leaf module" only — add hooks only to modules with no children.
- Timing strategy: GPU kernel time via CUDA Events; CPU via `time.perf_counter()`; dispatch overhead = max(0, wall_ms - kernel_ms).
- Energy attribution: global device energy is split among layers proportionally to layer execution time.
- BF16 on CPU: the code checks for AVX512_BF16; if missing, it sets `args.cpu_precision_executed = "fp32_fallback"`.
- Optimizer memory heuristic: use `OPTIMIZER_OVERHEAD_MAP` constant when computing `optimizer_states_mb`.
- NVML is wrapped with safe helpers (`safe_nvml_init`, `safe_nvml_shutdown`); threads must be daemonized and shut down cleanly.
- FLOPs estimation: implemented in `estimate_flops` with geometric formulas (Conv2d, Linear, Attention heuristics).

## Developer workflows & quick commands
- Run minimal CPU smoke test (no GPU required):
  - `python src/profiler.py --model mlp --no-gpu --precision fp32 --warmup 1 --measure 2 --output-dir data/test`
- GPU run with CPU energy measurement (if available):
  - `python src/profiler.py --model resnet50 --precision fp16 --rapl`
- Lower GEMM sizes to avoid OOM/dev boxes: `--gpu-gemm-n` / `--cpu-gemm-n`.
- Use `--nvml-sample-interval` to adjust NVML sampling frequency (default `0.05s`).

## Integration & testing tips for agents
- Add automated checks: a small CI job that runs `--model mlp --no-gpu` with very small warmup/measure values to validate runtime and output CSV/JSON generation.
- When adding model support in `main()`: ensure
  - Input shapes are explicit and deterministic.
  - Precision casting follows existing branches (HF models use `torch_dtype` argument; other models use `.half()` / `.to(torch.bfloat16)`).
  - If model imports are added, update `requirements` in the doc (e.g., `transformers`).
- When changing any metric name or schema, update both `documentation.md` and the `*_meta.json` writing site in `TrainingProfiler.run()`.

## What agents must not assume
- Do not assume NVML or pyRAPL are always available; code uses fallbacks and may set metric fields to `None`.
- Do not assume CSV rows are ordered or complete; rely on column names described in `documentation.md`.
- Do not change heuristics (e.g., `BACKWARD_FACTOR`, `OPTIMIZER_OVERHEAD_MAP`) without adding a short note in `documentation.md` and tests verifying downstream ILP compatibility.

## Files to reference for examples and patterns
- `src/profiler.py` — canonical implementation
- `src/profiler_update.py` — experimental ideas / partial duplication
- `documentation.md`, `src/profiler_en.md` — conceptual descriptions and data dictionary
- `data/` — target for outputs (CSV/JSON); tests should clean or write to `data/test`

## Example quick tasks for a coding agent
1. Add a new model (e.g., `efficientnet`) — implement branch in `main()` with matching casting and input shape, then run small CPU test.
2. Improve NVML safety — ensure `safe_nvml_init()` is always used before NVML reads and add a unit test simulating NVML failure.
3. Add a CI smoke test that runs `python src/profiler.py --model mlp --no-gpu --warmup 1 --measure 1` and asserts output files exist.

---
Please review this guidance and tell me if any sections are unclear or if you want me to include copy for a CI workflow or a sample GitHub Actions job. Thanks — I can iterate quickly. 
