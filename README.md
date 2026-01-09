# Hybrid CPU↔GPU Profiler for ILP Optimization

## Overview
- Purpose: profile deep learning models per layer to produce time, FLOPs, energy, memory, and transfer metrics that parameterize an ILP optimizer. Outputs CSV (per layer) and JSON (global metadata).
- Entry point: src/profiler.py. Spanish and English manuals in src/profiler_es.md and src/profiler_en.md. Data schema in documentation.md.

## Environment Setup
- Conda (GPU by default): see environment.yml.
- CPU-only: remove `pytorch-cuda` in environment.yml and add `cpuonly` (pytorch channel).
- Pip fallback: use requirements.txt, then install torch/torchvision per your CUDA/CPU target.

Commands (Linux):
```bash
# Conda (GPU)
conda env create -f environment.yml
conda activate hybrid-profiler

# Pip (CPU example)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Quick Start
```bash
# Minimal CPU smoke test (no GPU required)
python src/profiler.py --model mlp --no-gpu --precision fp32 --warmup 1 --measure 2 --output-dir data/test

# GPU run with CPU energy (requires RAPL permissions)
python src/profiler.py --model resnet50 --precision fp16 --rapl

# Adjust GEMM sizes to avoid OOM on dev boxes
python src/profiler.py --model resnet50 --gpu-gemm-n 4096 --cpu-gemm-n 1024
```

## Outputs
- CSV: data/{model_name}_metrics.csv — per-layer metrics (time, energy, FLOPs, memory, transfers, optimizer step time, precision).
- JSON: data/{model_name}_meta.json — global metadata (hardware, precision executed, energy totals/averages, PCIe α–β, TFLOPS peaks, optimizer timings).

## Notes & Troubleshooting
- RAPL (CPU energy): requires read access to /sys/class/powercap/intel-rapl; use --rapl only if permitted. When unavailable, CPU energy is 0.0 in CSV and None in JSON.
- NVML (GPU energy): ensure nvidia drivers and pynvml are available; see nvml_status in JSON if readings fail.
- Transfer model: H2D uses params_mb; D2H uses activations_mb. α–β are calibrated automatically.
- Precision: CPU bf16 falls back to fp32 if AVX512-BF16 is not supported; recorded in metadata.

## Experiment Script
- See run_experiments.sh for grid runs over models, optimizers, precisions, and batch sizes. Logs written to logs/.

## Docs
- Spanish: src/profiler_es.md
- English: src/profiler_en.md
- Data schema & ILP mapping: documentation.md

## License
- Academic research use within the PhD project. Contact the author for broader usage.

## CI Smoke Test
- GitHub Actions runs a fast CPU-only check to ensure CSV/JSON are generated.
- It installs CPU wheels for torch/torchvision and executes a minimal run.

Workflow snippet:
```yaml
name: CI Smoke Test
on:
	push:
		branches: [ main, master ]
	pull_request:
jobs:
	smoke:
		runs-on: ubuntu-latest
		steps:
			- uses: actions/checkout@v4
			- uses: actions/setup-python@v5
				with:
					python-version: '3.10'
			- name: Install dependencies
				run: |
					python -m pip install --upgrade pip
					pip install -r requirements.txt
					pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
			- name: Run profiler smoke test (CPU-only)
				run: |
					mkdir -p data/test-ci
					python src/profiler.py --model mlp --no-gpu --precision fp32 --warmup 1 --measure 1 --output-dir data/test-ci
			- name: Verify outputs
				run: |
					test -f data/test-ci/mlp_metrics.csv
					test -f data/test-ci/mlp_meta.json
```
