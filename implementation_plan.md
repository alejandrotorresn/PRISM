# Implementation Plan: Fix Simulation Budget Violations

## Goal
Address the user's observation that the ILP simulator incorrectly aborts valid hybrid plans by treating the Pareto budget (e.g., 1000 MB) as a hard physical VRAM limit. Also address the root cause of the "540 GB" memory accumulation bug and the excessive assignment of layers to the CPU.

## Root Cause Analysis
1. **Simulation Aborts on Budget:** The `simulator.py` currently appends a `violation` if `gpu_mem_used_mb > cfg.gpu_mem_budget_mb`. A `violation` changes the status to `invalid` and aborts the physical execution. However, as the user correctly stated, a plan that exceeds the 1000 MB target might still perfectly fit in the 12GB physical VRAM and execute successfully.
2. **The 540 GB Bug & Excessive CPU Layers:** The profiling data (`gpu_mem_peak_mb_mean`) captured by `torch.cuda.max_memory_allocated()` is *cumulative* across the PyTorch forward pass, not isolated per-layer. Thus, the last layer (`fc`) reports the peak of the *entire* model (e.g., 5.5 GB). 
   - **Simulator Impact:** When the simulator sums these values, it sums cumulative totals ($N^2$ explosion), resulting in 540 GB.
   - **ILP Impact:** The ILP sees `fc` as requiring 5.5 GB and is forced to send it (and other late layers) to the CPU to respect the 1000 MB budget, which slows down the model unnecessarily.

## Proposed Changes

### 1. Demote Simulation Violations to Warnings
#### [MODIFY] `src/runtime/simulator.py`
- Change the `gpu_mem_budget_mb` and `cpu_mem_budget_mb` checks to append to `warnings` instead of `violations`.
- This ensures that if a plan mathematically exceeds the Pareto budget but fits in physical VRAM, PyTorch will still attempt to execute it.

### 2. Patch ILP Memory Loader for Cumulative Profiling
#### [MODIFY] `src/ilp/data_loader.py`
- To prevent the ILP from being terrified of the artificially massive cumulative memory of late layers (and sending them to CPU), we will add a normalization step in `_load_node_weights`.
- We will cap the maximum `gpu_mem` of any individual layer to its estimated incremental size (e.g., fallback to `cpu_mem` which represents parameter size, or apply a scaling factor). 

## User Review Required
> [!IMPORTANT]
> The cumulative profiling data is the reason the ILP sent so many layers to the CPU. Are you okay with patching `data_loader.py` to use `cpu_mem_mb_mean` (which correctly measures the isolated tensor sizes) as the definitive size for both CPU and GPU in the optimization? This will allow the ILP to send many more layers back to the GPU!
