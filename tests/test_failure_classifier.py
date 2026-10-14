import subprocess
import sys
from pathlib import Path

from src.core.failure_classifier import (
    CODE_ERROR, CPU_OOM, GPU_OOM, TIMEOUT, TRACE_ERROR, classify_failure, read_run_log,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_exit_codes_and_timeout_disambiguation():
    assert classify_failure(137, "") == CPU_OOM
    assert classify_failure(137, "", elapsed_s=600, timeout_s=600) == TIMEOUT
    assert classify_failure(124, "") == TIMEOUT


def test_text_categories():
    assert classify_failure(1, "RuntimeError: CUDA out of memory. Tried to allocate") == GPU_OOM
    assert classify_failure(1, "DefaultCPUAllocator: can't allocate memory") == CPU_OOM
    assert classify_failure(1, "TypeError: 'Proxy' object does not support item assignment") == TRACE_ERROR
    assert classify_failure(1, "Structured graph extraction failed and fallback is disabled") == TRACE_ERROR
    assert classify_failure(1, "ValueError: bad argument") == CODE_ERROR


def test_old_oom_in_shared_campaign_log_is_ignored(tmp_path):
    log = tmp_path / "campaign.log"
    log.write_text("earlier run: CUDA out of memory\n")
    offset = log.stat().st_size
    with log.open("a") as fh:
        fh.write("this run: TypeError: 'Proxy' object does not support item assignment\n")
    text = read_run_log(str(log), offset)
    assert "cuda" not in text.lower()
    assert classify_failure(1, text) == TRACE_ERROR
    # the old behavior (reading the whole log) would have misreported GPU_OOM
    assert classify_failure(1, read_run_log(str(log), 0)) == GPU_OOM


def test_cli_prints_category(tmp_path):
    log = tmp_path / "l.log"
    log.write_text("x\nValueError: nope\n")
    out = subprocess.run(
        [sys.executable, str(_ROOT / "src/core/failure_classifier.py"), "--exit_code", "1", "--log", str(log)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == CODE_ERROR
