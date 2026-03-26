#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset_registry import MODEL_DATASET_MAP, download_required_datasets, resolve_datasets_root


def _parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(MODEL_DATASET_MAP.keys())
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare datasets used by the profiling and hybrid runtime campaigns")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'")
    parser.add_argument("--datasets_root", default="datasets", help="Target root folder for dataset storage")
    parser.add_argument("--force", action="store_true", help="Re-download archives and refresh extracted datasets")
    args = parser.parse_args()

    models = _parse_models(args.models)
    results = download_required_datasets(models=models, datasets_root=args.datasets_root, force=args.force)
    payload = {
        "datasets_root": str(resolve_datasets_root(args.datasets_root)),
        "models": models,
        "datasets": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())