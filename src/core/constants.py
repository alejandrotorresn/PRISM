WARMUP_STEPS = 5
MEASURE_STEPS = 15
OUTPUT_DIR = "data"
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
