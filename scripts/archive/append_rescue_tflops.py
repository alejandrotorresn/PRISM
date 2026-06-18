import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# Add src to Python path
sys.path.append(os.path.abspath("src"))
from core.metrics import estimate_flops
from models.factory import build_model_input_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_layer_flops(model_name: str, batch_size: int) -> dict[str, float]:
    class DummyArgs:
        def __init__(self):
            self.model = model_name
            self.batch_size = batch_size
            self.input_size = 224
            self.seq_length = 128
            self.precision = "fp32"
            self.datasets_root = None
            self.require_datasets = False

    args = DummyArgs()
    try:
        model, inp, target, data_info = build_model_input_target(args, torch.float32)
    except Exception as e:
        logger.error(f"Failed to build model {model_name}: {e}")
        return {}
    
    flops_dict = {}
    
    hooks = []
    def hook_fn(module, name):
        def _hook(mod, inputs, output):
            flops_dict[name] = estimate_flops(mod, inputs, output)
        return _hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook_fn(module, name)))

    model.eval()
    with torch.no_grad():
        if isinstance(inp, dict):
            model(**inp)
        else:
            model(inp)
            
    for h in hooks:
        h.remove()
        
    return flops_dict

def main():
    csv_files = glob.glob("data/zephyr/results_thesis_mode/*/*/*/*/*_metrics_stats.csv")
    logger.info(f"Found {len(csv_files)} CSV files.")
    
    flops_cache = {}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            continue
            
        if "rescued_tflops" in df.columns:
            logger.info(f"Skipping {f}, already has rescued_tflops.")
            continue
            
        if len(df) == 0:
            continue
            
        model_name = df["model"].iloc[0]
        batch_size = int(df["batch_size"].iloc[0])
        
        cache_key = (model_name, batch_size)
        if cache_key not in flops_cache:
            logger.info(f"Calculating theoretical FLOPs for {model_name} with batch {batch_size}...")
            flops_cache[cache_key] = get_layer_flops(model_name, batch_size)
            
        layer_flops = flops_cache[cache_key]
        
        rescued = []
        for _, row in df.iterrows():
            layer_name = row["layer"]
            flops = layer_flops.get(layer_name, 0.0)
            
            time_ms = row["gpu_fwd_time_ms_mean"]
            if time_ms <= 0:
                time_ms = row["cpu_fwd_time_ms_mean"]
                
            if time_ms > 0 and flops > 0:
                rtflops = (flops / 1e12) / (time_ms / 1000.0)
            else:
                rtflops = 0.0
                
            rescued.append(rtflops)
            
        df["rescued_tflops"] = rescued
        df.to_csv(f, index=False)
        logger.info(f"Updated {f} with rescued_tflops.")
        
if __name__ == "__main__":
    main()
