WARMUP_STEPS = 5
MEASURE_STEPS = 15
OUTPUT_DIR = "data"
# Fallback backward/forward time ratio used ONLY when backward hooks fail to fire
# (e.g., in-place ops, fused kernels, or autograd incompatibilities).
# Mathematical basis: backward requires 2 matrix multiplications per 1 forward matmul
# — one to propagate gradients and one to compute weight gradients (Rajbhandari et al., 2020).
# Empirically: 1.8–2.2× for CNN/ResNet; 1.5–2.5× for transformers (Epoch.ai, 2023).
# Conservative value 2.0 is intentional: overestimating backward cost is safer than
# underestimating it, as it biases the ILP toward GPU placement (avoids false CPU offloads).
# NOTE: when empirical backward measurement is available (bwd_count > 0), this constant
# is NOT used — see training_profiler.py where the measured value takes precedence.
BACKWARD_FACTOR = 2.0
OPTIMIZER_OVERHEAD_FACTOR = 2.0

OPTIMIZER_OVERHEAD_MAP = {
    "SGD": 0.0,
    "SGD_momentum": 1.0,
    "Adam": 2.0,
    "AdamW": 2.0,
    "RMSprop": 1.0,
    "Adagrad": 1.0,
    "Adadelta": 2.0
}
