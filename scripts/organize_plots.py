import os
import shutil
from pathlib import Path

reports_dir = Path("reports/zephyr/doctoral_minimal")
plots_dir = reports_dir / "plots"
advanced_dir = reports_dir / "advanced_analytics"

# Move advanced_analytics plots to plots/
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

# Move pareto and bottlenecks to plots/<model>/...
for ptype in ["pareto", "bottlenecks"]:
    src = plots_dir / ptype
    if src.exists():
        for file in src.glob("*.png"):
            # Extract model name
            # Format is usually <model>_pareto.png or <model>_top15_fwd.png
            model = None
            for m in ["resnet152", "resnet50", "vit_b16", "simple_mlp"]:
                if m in file.name:
                    model = m
                    break
            if model:
                dest_dir = plots_dir / model / ptype
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(dest_dir / file.name))
        shutil.rmtree(src)

# Move CSVs and Markdown
csv_dir = reports_dir / "csv"
csv_dir.mkdir(exist_ok=True)
for file in reports_dir.glob("*.csv"):
    shutil.move(str(file), str(csv_dir / file.name))

md_dir = reports_dir / "summary_docs"
md_dir.mkdir(exist_ok=True)
for file in reports_dir.glob("*.md"):
    if file.name != "THESIS_MODE_PROTOCOL_CHECKLIST.md":
        shutil.move(str(file), str(md_dir / file.name))

print("Organized successfully")
