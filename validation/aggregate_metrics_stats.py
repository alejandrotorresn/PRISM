#!/usr/bin/env python3
import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.stats_aggregator import aggregate_metrics_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate replicate metrics into robust stats table")
    parser.add_argument("--input_dir", required=True, help="Directory containing run_* subfolders with *_metrics.csv")
    parser.add_argument("--output_csv", default=None, help="Output CSV path (default: <input_dir>/metrics_stats.csv)")
    parser.add_argument("--include_skipped", action="store_true", help="Include skipped profiling rows in aggregation")
    args = parser.parse_args()

    result = aggregate_metrics_stats(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        include_skipped=args.include_skipped,
    )

    print("=" * 80)
    print("METRICS STATS AGGREGATION")
    print("=" * 80)
    print(f"Input files: {result['input_files']}")
    print(f"Rows in: {result['rows_in']}")
    print(f"Rows out: {result['rows_out']}")
    print(f"Output: {result['output_csv']}")
    flagged = result.get("flagged_layers_count", 0)
    if flagged == 0:
        print("Quality flags: all layers OK")
    else:
        print(f"Quality flags: {flagged} layer(s) flagged")
        low = result.get("low_sample_layers") or []
        hd  = result.get("high_dispersion_layers") or []
        if low:
            print(f"  low_sample      : {', '.join(str(l) for l in low)}")
        if hd:
            print(f"  high_dispersion : {', '.join(str(l) for l in hd)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
