import importlib.util
import logging
import os
import platform
import random
import socket
import time
from typing import Any, Dict

import numpy as np
import psutil
import pynvml
import torch

PYRAPL_AVAILABLE = importlib.util.find_spec("pyRAPL") is not None

logger = logging.getLogger(__name__)


def get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown_host"


def normalize_output_dir_for_host(output_dir: str, host_name: str) -> str:
    """
    Ensure output paths under data/ are namespaced as data/<host>/... once.
    This keeps historical paths stable while isolating per-node profiling artifacts.
    """
    if not output_dir:
        return output_dir

    normalized = output_dir.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return output_dir

    try:
        data_idx = parts.index("data")
    except ValueError:
        # Not a data-rooted path; keep user-provided output unchanged.
        return output_dir

    if data_idx + 1 < len(parts) and parts[data_idx + 1] == host_name:
        return output_dir

    new_parts = parts[: data_idx + 1] + [host_name] + parts[data_idx + 1 :]
    if normalized.startswith("/"):
        return "/" + "/".join(new_parts)
    return "/".join(new_parts)


def set_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_cpu_runtime(force_threads: int = 0) -> None:
    env_threads = os.getenv("OMP_NUM_THREADS")
    target_threads = None
    source = None

    if force_threads > 0:
        target_threads = force_threads
        source = "user_forced"
    elif env_threads and env_threads.isdigit() and int(env_threads) > 0:
        target_threads = int(env_threads)
        source = "OMP_NUM_THREADS"
    else:
        try:
            affinity = psutil.Process().cpu_affinity()
            affinity_count = len(affinity) if affinity is not None else 0
        except Exception:
            affinity_count = 0

        physical_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        target_threads = affinity_count if affinity_count > 0 else physical_count
        source = "cpu_affinity" if affinity_count > 0 else "physical_cores"

    target_threads = max(1, int(target_threads))

    try:
        torch.set_num_threads(target_threads)
    except Exception as e:
        logger.warning(f"Failed to set torch intra-op threads: {e}")

    try:
        torch.set_num_interop_threads(max(1, min(4, target_threads)))
    except Exception as e:
        logger.warning(f"Failed to set torch inter-op threads: {e}")

    try:
        torch.set_flush_denormal(True)
    except Exception as e:
        logger.warning(f"Failed to enable flush denormals: {e}")

    logger.info(
        f"CPU runtime configured: threads={target_threads} "
        f"(source={source}), interop={max(1, min(4, target_threads))}, flush_denormal=True"
    )


def get_cpu_model() -> str:
    cpu_model = platform.processor()
    if not cpu_model or cpu_model == "x86_64":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.strip().split(":")[1].strip()
                        break
        except Exception:
            cpu_model = "Unknown"
    return cpu_model


def get_hardware_metadata() -> Dict[str, Any]:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host_name": get_hostname(),
        "torch_version": torch.__version__,
        "os": platform.platform(),
        "cpu_model": get_cpu_model(),
        "gpu_name": "None",
        "gpu_driver": "None",
        "rapl_available": PYRAPL_AVAILABLE,
    }
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meta["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            meta["gpu_driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception as e:
            logger.warning(f"NVML Init failed in metadata check: {e}")
    return meta
