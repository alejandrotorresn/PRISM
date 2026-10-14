"""Classify why one profiler run failed, using only evidence from THAT run.

Evidence: exit code, whether the wall time reached the timeout, and the part of
the (shared, append-only) campaign log written after ``log_offset_bytes``.
Categories: GPU_OOM, CPU_OOM, TRACE_ERROR, TIMEOUT, CODE_ERROR.
Stdlib only so it can run as a script from the campaign shell.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Optional

GPU_OOM = "GPU_OOM"
CPU_OOM = "CPU_OOM"
TRACE_ERROR = "TRACE_ERROR"
TIMEOUT = "TIMEOUT"
CODE_ERROR = "CODE_ERROR"

SIGKILL_EXIT_CODES = (137, 9)
TIMEOUT_EXIT_CODE = 124

_GPU_OOM_RE = re.compile(
    r"cuda out of memory|cuda error: out of memory|hip out of memory|cublas.*alloc|"
    r"oom retries were exhausted|torch\.(cuda\.)?outofmemoryerror"
)
_CPU_OOM_RE = re.compile(r"defaultcpuallocator: can't allocate memory|std::bad_alloc|memoryerror")
_TRACE_RE = re.compile(
    r"symbolic_trace|symbolically traced|torch\.fx\.proxy|'proxy' object|"
    r"structured graph extraction failed|torch\.export|export_trace|graph extraction failed"
)


def read_run_log(log_path: str, log_offset_bytes: int = 0, tail_bytes: int = 2_000_000) -> str:
    if not log_path or not os.path.isfile(log_path):
        return ""
    size = os.path.getsize(log_path)
    start = max(int(log_offset_bytes), size - tail_bytes, 0)
    with open(log_path, "rb") as fh:
        fh.seek(start)
        return fh.read().decode("utf-8", errors="replace")


def classify_failure(
    exit_code: int,
    log_text: str = "",
    elapsed_s: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> str:
    text = log_text.lower()
    if exit_code == TIMEOUT_EXIT_CODE:
        return TIMEOUT
    if exit_code in SIGKILL_EXIT_CODES:
        # `timeout --signal=KILL` also exits 137: tell it apart by elapsed time.
        if timeout_s and elapsed_s is not None and elapsed_s >= timeout_s - 1:
            return TIMEOUT
        return CPU_OOM
    if _GPU_OOM_RE.search(text):
        return GPU_OOM
    if _CPU_OOM_RE.search(text):
        return CPU_OOM
    if _TRACE_RE.search(text):
        return TRACE_ERROR
    return CODE_ERROR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit_code", type=int, required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--log_offset_bytes", type=int, default=0)
    parser.add_argument("--elapsed_s", type=float, default=None)
    parser.add_argument("--timeout_s", type=float, default=None)
    args = parser.parse_args()
    text = read_run_log(args.log, args.log_offset_bytes)
    print(classify_failure(args.exit_code, text, args.elapsed_s, args.timeout_s))


if __name__ == "__main__":
    main()
