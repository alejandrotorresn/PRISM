#!/usr/bin/env python3
"""One-time preparation of a held-out validation split for tiny_shakespeare.

Why this exists
----------------
`datasets/tiny_shakespeare/` only ships `train.txt`. Every other dataset used by
this thesis (imagenette2-160, mnist) already has a real train/val split on disk;
tiny_shakespeare does not, because `src/data/dataset_registry.py` was built for
short profiling batches (always the first N lines of `train.txt`), not for real
training with held-out evaluation.

This script does NOT touch `datasets/tiny_shakespeare/train.txt` or any file
under `src/`. It reads the original corpus, shuffles the non-empty lines with a
fixed seed, and writes two NEW files under
`convergence_validation/data/tiny_shakespeare/`:

- `train.txt`: ~90% of the lines, for training.
- `val.txt`: ~10% of the lines, held out, never seen during training.

Why shuffle before splitting (not just take the last 10% of the file)
-----------------------------------------------------------------------
tiny_shakespeare is organized by play/act/scene. A contiguous tail slice would
likely make the validation split one specific scene or character's dialogue,
which is a stylistic/content distribution shift, not a random held-out sample.
Shuffling once with a fixed seed avoids that bias while staying perfectly
reproducible (same seed -> same split, every time this script is re-run).

Run once, before the first convergence_validation run for a language model:

    python convergence_validation/prepare_tiny_shakespeare_split.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_SEED = 1234
DEFAULT_VAL_FRACTION = 0.10


def load_nonempty_lines(train_txt: Path) -> list[str]:
    """Read train.txt and keep only non-blank lines, matching the filtering
    already done by `src/data/dataset_registry.py::_load_tiny_shakespeare_batch`
    (blank lines are formatting separators between speakers/scenes, not corpus
    text)."""
    with train_txt.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def split_lines(
    lines: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_lines = shuffled[:n_val]
    train_lines = shuffled[n_val:]
    return train_lines, val_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="Path to the original tiny_shakespeare train.txt "
        "(default: <repo_root>/datasets/tiny_shakespeare/train.txt)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write the new train.txt/val.txt "
        "(default: <repo_root>/convergence_validation/data/tiny_shakespeare)",
    )
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = Path(args.source) if args.source else repo_root / "datasets" / "tiny_shakespeare" / "train.txt"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "data" / "tiny_shakespeare"

    if not source.exists():
        raise FileNotFoundError(f"Original tiny_shakespeare corpus not found: {source}")

    lines = load_nonempty_lines(source)
    train_lines, val_lines = split_lines(lines, args.val_fraction, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"Source corpus: {source} ({len(lines)} non-empty lines)")
    print(f"Wrote {len(train_lines)} training lines -> {output_dir / 'train.txt'}")
    print(f"Wrote {len(val_lines)} validation lines -> {output_dir / 'val.txt'}")
    print(f"(seed={args.seed}, val_fraction={args.val_fraction})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
