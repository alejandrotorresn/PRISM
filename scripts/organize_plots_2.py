import os
import shutil
from pathlib import Path

reports_dir = Path("reports/zephyr/doctoral_minimal")
plots_dir = reports_dir / "plots"

# Move advanced_analytics
advanced_dir = Path("reports/zephyr/advanced_analytics")
if advanced_dir.exists():
    for model_dir in advanced_dir.iterdir():
        if model_dir.is_dir():
            target_model_dir = plots_dir / model_dir.name
            target_model_dir.mkdir(parents=True, exist_ok=True)
            for plot_type_dir in model_dir.iterdir():
                if plot_type_dir.is_dir():
                    target_plot_dir = target_model_dir / plot_type_dir.name
                    if target_plot_dir.exists():
                        shutil.rmtree(target_plot_dir)
                    shutil.move(str(plot_type_dir), str(target_model_dir))
    shutil.rmtree(advanced_dir)

# Move thesis_figures
thesis_dir = Path("reports/zephyr/thesis_figures")
if thesis_dir.exists():
    for file in thesis_dir.glob("*.png"):
        model = None
        for m in ["resnet152", "resnet50", "vit_b16", "simple_mlp"]:
            if m in file.name:
                model = m
                break
        if model:
            dest_dir = plots_dir / model / "hybrid_vs_ilp"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(dest_dir / file.name))
    shutil.rmtree(thesis_dir)

print("Second pass organized successfully")
