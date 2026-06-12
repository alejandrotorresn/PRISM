from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KEYWORDS = {
    "pareto",
    "top15",
    "strategy",
    "cost",
    "memory",
    "energy",
    "latency",
    "comparison",
    "execution",
    "hybrid",
    "batch",
}


def _infer_model_from_name(name: str) -> str | None:
    stem = Path(name).stem
    tokens = stem.split("_")
    model_tokens = []
    for token in tokens:
        if token.lower() in KEYWORDS:
            break
        model_tokens.append(token)
    if not model_tokens:
        return None
    return "_".join(model_tokens)


def _move_advanced_dirs(plots_dir: Path, reports_dir: Path) -> None:
    candidate_dirs = [reports_dir / "advanced_analytics", reports_dir.parent / "advanced_analytics"]
    for advanced_dir in candidate_dirs:
        if not advanced_dir.exists():
            continue
        for model_dir in advanced_dir.iterdir():
            if not model_dir.is_dir():
                continue
            target_model_dir = plots_dir / model_dir.name
            target_model_dir.mkdir(parents=True, exist_ok=True)
            for plot_type_dir in model_dir.iterdir():
                if not plot_type_dir.is_dir():
                    continue
                target_plot_dir = target_model_dir / plot_type_dir.name
                if target_plot_dir.exists():
                    shutil.rmtree(target_plot_dir)
                shutil.move(str(plot_type_dir), str(target_model_dir))
        shutil.rmtree(advanced_dir)


def _move_legacy_flat_plot_dirs(plots_dir: Path) -> None:
    for ptype in ["pareto", "bottlenecks"]:
        src = plots_dir / ptype
        if not src.exists():
            continue
        for file in src.glob("*.png"):
            model = _infer_model_from_name(file.name)
            if not model:
                continue
            dest_dir = plots_dir / model / ptype
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(dest_dir / file.name))
        shutil.rmtree(src)


def _move_legacy_thesis_figures(plots_dir: Path, reports_dir: Path) -> None:
    thesis_dir = reports_dir.parent / "thesis_figures"
    if not thesis_dir.exists():
        return

    for file in thesis_dir.glob("*.png"):
        model = _infer_model_from_name(file.name)
        if not model:
            continue
        dest_dir = plots_dir / model / "hybrid_vs_ilp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(dest_dir / file.name))
    shutil.rmtree(thesis_dir)


def _move_top_level_docs(reports_dir: Path) -> None:
    csv_dir = reports_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for file in reports_dir.glob("*.csv"):
        shutil.move(str(file), str(csv_dir / file.name))

    md_dir = reports_dir / "summary_docs"
    md_dir.mkdir(exist_ok=True)
    for file in reports_dir.glob("*.md"):
        if file.name != "THESIS_MODE_PROTOCOL_CHECKLIST.md":
            shutil.move(str(file), str(md_dir / file.name))


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize thesis plot artifacts into canonical layout.")
    parser.add_argument("--host", default="zephyr", help="Host namespace under reports/<host>/")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"), help="Reports root directory")
    args = parser.parse_args()

    reports_dir = args.reports_root / args.host / "doctoral_minimal"
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _move_advanced_dirs(plots_dir, reports_dir)
    _move_legacy_flat_plot_dirs(plots_dir)
    _move_legacy_thesis_figures(plots_dir, reports_dir)
    _move_top_level_docs(reports_dir)

    print(f"Organized successfully under {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
