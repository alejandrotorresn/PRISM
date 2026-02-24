# Quick Start Guide

**For general project overview and setup, see [../README.md](../README.md) in the repository root.**

This document provides detailed instructions and examples for using the Advanced Hybrid Profiler.

## Installation

### Option 1: Conda (Recommended)
```bash
conda env create -f config/environment.yml
conda activate thesis_env
```

### Option 2: Pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
# Install PyTorch separately for your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Validation

Before profiling, verify installation:

```bash
# Quick syntax check
python validation/validate_code.py

# Test all model loading
python validation/validate_all_models.py

# Verify zombie thread fixes
python validation/validate_zombie_fix.py

# Full validation suite
bash validation/comprehensive_check.sh
```

## Basic Usage

### Profile Single Model

```bash
# Minimal test (CPU, fastest)
python src/profiler.py \
  --model simple_mlp \
  --precision fp32 \
  --batch_size 8 \
  --warmup 1 \
  --measure 2 \
  --skip_cpu

# Vision model with GPU
python src/profiler.py \
  --model resnet50 \
  --precision fp32 \
  --batch_size 32

# NLP model with energy measurement
python src/profiler.py \
  --model bert_base \
  --precision fp16 \
  --batch_size 16 \
  --rapl
```

### Important: Zombie Thread Avoidance

When profiling ViT-B/16 or other models with slow CPU FP16:

```bash
# CPU FP16 emulation can be very slow on non-AVX512_FP16 CPUs:
# Solution: Use --skip_cpu flag

python src/profiler.py \
  --model vit_b16 \
  --precision fp16 \
  --skip_cpu \
  --num_threads 16 \
  --batch_size 32
```

This extracts GPU metrics in ~3 minutes instead of ~15 minutes.

### Override SLURM Single-Core Limitation

On HPC systems, SLURM may allocate only 1 core. Force more threads:

```bash
# SLURM limitation: CPU affinity set to [0] = 1 core
# torch.set_num_threads(1) disables OpenMP
# Solution: Use --num_threads to override

python src/profiler.py \
  --model resnet50 \
  --num_threads 16 \
  --precision fp32
```

## Output Interpretation

### Metrics CSV
- **Location**: `data/{model_name}_metrics.csv`
- **Rows**: One per layer
- **Columns**: Time, energy, FLOPs, memory, efficiency, overhead
- **Purpose**: Input data for ILP model

```bash
# View first few rows
head -20 data/resnet50_metrics.csv
```

### Metadata JSON
- **Location**: `data/{model_name}_meta.json`
- **Content**: Hardware info, total energy, calibration data, precision executed
- **Purpose**: Global context for metrics

```bash
# Pretty-print metadata
python -m json.tool data/resnet50_meta.json
```

## Run Full Experiment Grid

Edit `scripts/run_experiments.sh` to configure grid search:

```bash
# Models to profile
MODELS=("resnet50" "vit_b16" "bert_base")

# Batch sizes
BATCH_SIZES=(32 64 128)

# Precisions
PRECISIONS=("fp32" "fp16")

# Optimizers
OPTIMIZERS=("SGD" "Adam")

# NEW: Zombie thread fix options
USE_SKIP_CPU=false      # Set to true for GPU-only
FORCE_THREADS=0         # Set to >0 to override SLURM
```

Then run:

```bash
bash scripts/run_experiments.sh
# Results saved to: data/results/{model}/{optimizer}/{precision}/
```

## Supported Models

```python
# Vision Models
- resnet50      # ResNet 50 layers
- resnet152     # ResNet 152 layers  
- vit_b16       # Vision Transformer B/16

# NLP Models
- bert_base     # BERT-base-uncased
- gpt2_small    # GPT2 small

# Baseline
- simple_mlp    # Simple 3-layer MLP (fast test)
```

## Troubleshooting

### Process Hangs During Profiling

**Symptom**: Stuck at "Profiling ViT-B/16 with FP16"

**Cause**: CPU FP16 emulation on non-AVX512_FP16 machine

**Solution**: Use `--skip_cpu` or `--precision fp32`

```bash
python src/profiler.py --model vit_b16 --skip_cpu
```

### RAPL Permission Denied

**Symptom**: `PermissionError: /sys/class/powercap/intel-rapl`

**Cause**: RAPL requires read permissions (Linux only)

**Solution**: 
- Option A: Don't use `--rapl` flag
- Option B: Run with `sudo` (not recommended)
- Option C: Grant permissions: `sudo chmod -r o+r /sys/class/powercap/`

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Cause**: Batch size too large

**Solution**: Reduce `--batch_size`

```bash
python src/profiler.py --model resnet50 --batch_size 8
```

### GPU Not Detected

**Symptom**: `nvidia-smi: command not found` or runs on CPU

**Cause**: NVIDIA drivers or pytorch CUDA build not installed

**Solution**: Install PyTorch with CUDA support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Advanced Usage

### Custom Output Location

```bash
python src/profiler.py \
  --model resnet50 \
  --output_dir /tmp/my_results
```

### Different GEMM Benchmark Sizes

(Note: Current version uses fixed GEMM size. See profiler.py for details.)

### No GPU / CPU-Only

```bash
python src/profiler.py \
  --model simple_mlp \
  --no_gpu
```

## Benchmarking

Time a single model profiling run:

```bash
time python src/profiler.py \
  --model simple_mlp \
  --batch_size 32 \
  --warmup 3 \
  --measure 10 \
  --skip_cpu
```

Typical times:
- simple_mlp + GPU: ~5 minutes
- resnet50 + GPU + CPU: ~20 minutes
- vit_b16 + GPU-only: ~10 minutes

## Batch Processing

Profile multiple models programmatically:

```bash
#!/bin/bash
for model in simple_mlp resnet50 vit_b16; do
  echo "Profiling: $model"
  python src/profiler.py \
    --model "$model" \
    --precision fp32 \
    --batch_size 32 \
    --skip_cpu \
    --warmup 2 \
    --measure 5
done
```

## Integration with ILP Model

The CSV and JSON outputs are formatted for direct use in the ILP optimizer:

1. Read CSV into optimization framework
2. Use column names as variable names
3. Reference JSON for hardware constraints
4. See thesis Chapter 3 for ILP formulation

## Documentation Index

| Document | Purpose |
|----------|---------|
| [../README.md](../README.md) | Project overview (START HERE) |
| [documentation.md](documentation.md) | Data schema & technical details |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Project organization |
| [FOLDER_GUIDE.md](FOLDER_GUIDE.md) | Folder purposes & rationale |
| [ZOMBIE_THREAD_FIX_SUMMARY.md](ZOMBIE_THREAD_FIX_SUMMARY.md) | Blocking issue & solutions |

## Support

For additional help:
1. Check [../README.md](../README.md) for overview
2. Review [documentation.md](documentation.md) for data schema
3. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for project details
4. Run validation scripts: `python validation/validate_*.py`

---

*Last Updated*: February 24, 2026
