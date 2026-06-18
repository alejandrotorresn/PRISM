import os
import json
import glob

RESULTS_DIR = "data/zephyr/results_thesis_mode"

target_configs = [
    ("resnet152", "AdamW", "fp32", "batch_64"),
    ("resnet152", "SGD", "fp32", "batch_64"),
    ("vit_b16", "SGD", "fp32", "batch_64"),
]

for model, optim, prec, batch in target_configs:
    target_dir = os.path.join(RESULTS_DIR, model, optim, prec, batch)
    
    for r in ["run_001", "run_002", "run_003", "run_004", "run_005"]:
        meta_files = glob.glob(os.path.join(target_dir, r, "*_meta.json"))
        if not meta_files:
            continue
            
        with open(meta_files[0], "r") as f:
            meta = json.load(f)
            
        meta["transfer_calibration_source"] = "measured"
        meta["graph_trace_source"] = "torch_fx"
        
        with open(meta_files[0], "w") as f:
            json.dump(meta, f, indent=4)
            
    print(f"Fixed meta files for {model}/{optim}/{prec}/{batch}")

