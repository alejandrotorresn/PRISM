import logging
import platform
import re
import threading
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from core.constants import BACKWARD_FACTOR

logger = logging.getLogger(__name__)


def cpu_supports_bf16() -> bool:
    try:
        if platform.system() != "Linux":
            return False
        with open("/proc/cpuinfo", "r") as f:
            return "avx512_bf16" in f.read()
    except Exception:
        return False


def get_cpu_instruction_flags() -> List[str]:
    if platform.system() != "Linux":
        return []

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
    except Exception:
        return []

    matches = re.findall(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if not matches:
        return []

    flags = set()
    for line in matches:
        for flag in line.split():
            flags.add(flag.strip())
    return sorted(flags)


def probe_cpu_precision_support() -> Dict[str, Any]:
    flags = set(get_cpu_instruction_flags())
    has_avx512_fp16 = "avx512_fp16" in flags
    has_avx512_bf16 = "avx512_bf16" in flags
    has_amx_bf16 = "amx_bf16" in flags
    has_amx_tile = "amx_tile" in flags
    has_amx_bf16_path = has_amx_bf16 and has_amx_tile

    return {
        "flags": sorted(flags),
        "avx512_fp16": has_avx512_fp16,
        "avx512_bf16": has_avx512_bf16,
        "amx_bf16": has_amx_bf16,
        "amx_tile": has_amx_tile,
        "fp16_accelerated": has_avx512_fp16,
        "bf16_accelerated": has_avx512_bf16 or has_amx_bf16_path,
    }


def evaluate_precision_execution_policy(precision: str, isa_info: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "allowed": True,
        "cpu_precision_executed": precision,
        "reason": "",
        "status": "ready",
    }

    if precision == "fp16":
        if not isa_info.get("fp16_accelerated", False):
            result.update({
                "allowed": False,
                "cpu_precision_executed": "fp16_requested_isa_unsupported",
                "reason": "missing avx512_fp16; fp16 would require non-accelerated/emulated path",
                "status": "skipped_unsupported_precision",
            })
    elif precision == "bf16":
        if not isa_info.get("bf16_accelerated", False):
            result.update({
                "allowed": False,
                "cpu_precision_executed": "bf16_requested_isa_unsupported",
                "reason": "missing avx512_bf16 and amx_bf16/amx_tile; bf16 would require non-accelerated/emulated path",
                "status": "skipped_unsupported_precision",
            })

    return result


def get_cpu_fp16_support_info() -> Dict[str, Any]:
    info = {
        "supported": False,
        "isa_avx512_fp16": False,
        "smoke_test_ok": False,
        "reason": "unknown",
    }

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                info["isa_avx512_fp16"] = "avx512_fp16" in cpuinfo
    except Exception:
        info["isa_avx512_fp16"] = False

    try:
        a = torch.randn((64, 64), dtype=torch.float16, device="cpu")
        b = torch.randn((64, 64), dtype=torch.float16, device="cpu")
        _ = torch.mm(a, b)
        info["smoke_test_ok"] = True
    except Exception as e:
        info["smoke_test_ok"] = False
        info["reason"] = f"fp16 cpu smoke test failed: {e}"

    if info["smoke_test_ok"]:
        info["supported"] = True
        if info["isa_avx512_fp16"]:
            info["reason"] = "functional support validated; avx512_fp16 detected"
        else:
            info["reason"] = "functional support validated; avx512_fp16 not detected (possible emulation/slower path)"
    elif info["reason"] == "unknown":
        info["reason"] = "functional support not available"

    return info


def _extract_loss_for_preflight(out: Any) -> torch.Tensor:
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    if hasattr(out, "logits"):
        return out.logits.sum()
    if isinstance(out, torch.Tensor):
        return out.sum()
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0].sum()
    return torch.tensor(0.0, requires_grad=True)


def _build_mini_input_for_cpu_fp16(input_data: Any) -> Any:
    def _slice_first(x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 0 and x.shape[0] > 1:
            return x[:1].clone()
        return x.clone()

    def _to_cpu_fp16_if_float(x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.to("cpu")
        if torch.is_floating_point(x_cpu):
            return x_cpu.to(dtype=torch.float16)
        return x_cpu

    if isinstance(input_data, dict):
        mini = {}
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                mini[key] = _to_cpu_fp16_if_float(_slice_first(value))
            else:
                mini[key] = value
        return mini

    if isinstance(input_data, torch.Tensor):
        return _to_cpu_fp16_if_float(_slice_first(input_data))

    return input_data


def run_cpu_fp16_model_preflight(model: nn.Module, input_data: Any, timeout_safety_factor: float = 2.5) -> Dict[str, Any]:
    result = {"ok": False, "reason": "unknown"}
    execution_result = {
        "forward_completed": False,
        "forward_time_ms": 0.0,
        "backward_completed": False,
        "exception": None,
    }

    def _run_training_step():
        try:
            model.train()
            model.to("cpu")
            model.to(dtype=torch.float16)

            mini_inp = _build_mini_input_for_cpu_fp16(input_data)
            opt = torch.optim.SGD(model.parameters(), lr=1e-6)
            opt.zero_grad(set_to_none=True)

            t_forward_start = time.perf_counter()
            if isinstance(mini_inp, dict):
                out = model(**mini_inp)
            else:
                out = model(mini_inp)
            t_forward_end = time.perf_counter()
            execution_result["forward_time_ms"] = (t_forward_end - t_forward_start) * 1000.0
            execution_result["forward_completed"] = True

            loss = _extract_loss_for_preflight(out)
            loss.backward()
            opt.step()

            execution_result["backward_completed"] = True
        except Exception as e:
            execution_result["exception"] = e

    preflight_thread = threading.Thread(target=_run_training_step, daemon=True)
    preflight_thread.start()
    preflight_thread.join(timeout=60.0)

    if execution_result["forward_completed"]:
        forward_time_sec = execution_result["forward_time_ms"] / 1000.0
        backward_timeout = max(10.0, forward_time_sec * BACKWARD_FACTOR * timeout_safety_factor)
        logger.debug(
            f"FP16 preflight forward time: {execution_result['forward_time_ms']:.2f}ms, "
            f"calculated backward timeout: {backward_timeout:.2f}s "
            f"(formula: {forward_time_sec:.4f}s × {BACKWARD_FACTOR} × {timeout_safety_factor})"
        )
    else:
        backward_timeout = 10.0
        logger.warning(
            f"FP16 preflight forward pass did not complete within 60s timeout; "
            f"using minimum backward timeout of {backward_timeout:.2f}s"
        )

    preflight_thread.join(timeout=backward_timeout)

    if execution_result["exception"] is not None:
        result["ok"] = False
        result["reason"] = f"cpu fp16 training-step preflight failed with exception: {execution_result['exception']}"
    elif execution_result["backward_completed"]:
        result["ok"] = True
        result["reason"] = (
            f"cpu fp16 training-step preflight succeeded "
            f"(forward={execution_result['forward_time_ms']:.2f}ms, "
            f"backward allowed {backward_timeout:.2f}s timeout)"
        )
    elif execution_result["forward_completed"]:
        result["ok"] = False
        result["reason"] = (
            f"cpu fp16 backward pass blocked after {backward_timeout:.2f}s timeout "
            f"(forward took {execution_result['forward_time_ms']:.2f}ms, "
            f"calculated timeout=forward×{BACKWARD_FACTOR}×{timeout_safety_factor}={backward_timeout:.2f}s); "
            f"likely missing AVX512_FP16 ISA flag, insufficient CPU resources, or model too complex for CPU FP16"
        )
    else:
        result["ok"] = False
        result["reason"] = (
            f"cpu fp16 forward pass timeout after 60s; "
            f"model layers too large for CPU FP16 on this hardware or severe resource limitation"
        )

    return result
