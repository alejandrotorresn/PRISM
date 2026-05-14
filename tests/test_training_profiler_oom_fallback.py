from types import SimpleNamespace

import torch
import torch.nn as nn

from src.runner.training_profiler import TrainingProfiler, _is_oom_runtime_error


def _build_profiler() -> TrainingProfiler:
    return TrainingProfiler(
        model=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
        model_name="toy",
        args=SimpleNamespace(no_gpu=True, gpu_id=0, rapl=False),
    )


def test_is_oom_runtime_error_detects_cuda_oom_message() -> None:
    ex = RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB")
    assert _is_oom_runtime_error(ex) is True


def test_is_oom_runtime_error_ignores_non_oom_runtime_error() -> None:
    ex = RuntimeError("some unrelated autograd runtime issue")
    assert _is_oom_runtime_error(ex) is False


def test_slice_batch_and_infer_batch_size_for_dict_input() -> None:
    profiler = _build_profiler()
    batch = {
        "x": torch.randn(8, 4),
        "mask": torch.ones(8, 4),
    }

    assert profiler._infer_batch_size(batch) == 8
    sliced = profiler._slice_batch(batch, 0, 3)

    assert sliced["x"].shape[0] == 3
    assert sliced["mask"].shape[0] == 3
