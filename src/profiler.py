import argparse
import logging

import torch

from core.constants import (
    BACKWARD_FACTOR,
    MEASURE_STEPS,
    OPTIMIZER_OVERHEAD_MAP,
    OUTPUT_DIR,
    WARMUP_STEPS,
)
from core.precision_policy import (
    _build_mini_input_for_cpu_fp16,
    _extract_loss_for_preflight,
    cpu_supports_bf16,
    evaluate_precision_execution_policy,
    get_cpu_fp16_support_info,
    probe_cpu_precision_support,
    run_cpu_fp16_model_preflight,
)
from core.system import configure_cpu_runtime, set_determinism
from models.factory import SimpleMLP, build_model_and_input
from runner.training_profiler import TrainingProfiler


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced Profiler for Deep Learning Training")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "resnet152", "vit_b16", "bert_base", "gpt2_small", "simple_mlp"], help="Model architecture to profile")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision mode for profiling")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for input data")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS, help="Number of warmup steps")
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS, help="Number of measurement steps")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save profiling data")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU profiling even if available")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for profiling")
    parser.add_argument("--rapl", action="store_true", help="Enable RAPL energy measurement on CPU (Linux only)")
    parser.add_argument("--input_size", type=int, default=224, help="Input size (for vision models)")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length (for NLP models)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=list(OPTIMIZER_OVERHEAD_MAP.keys()), help="Optimizer type for overhead estimation")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (for metadata)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for metadata)")
    parser.add_argument("--skip_cpu", action="store_true", help="Skip CPU profiling entirely")
    parser.add_argument("--num_threads", type=int, default=0, help="Force CPU thread count (0 = auto-detect)")
    parser.add_argument("--keep_partial_artifacts", action="store_true", help="Keep intermediate *_gpu_partial.csv/json artifacts after successful final save")
    return parser


def _initialize_precision_state(args) -> None:
    args.cpu_fp16_supported = None
    args.cpu_fp16_isa_avx512 = None
    args.cpu_fp16_smoke_test_ok = None
    args.cpu_fp16_model_smoke_ok = None
    args.cpu_fp16_model_smoke_reason = None
    args.cpu_fp16_support_reason = None
    args.cpu_instruction_flags = []
    args.cpu_isa_probe = {}
    args.abort_profiling_due_to_isa = False
    args.abort_profiling_reason = ""
    args.execution_status = "ready"


def _configure_precision(args) -> torch.dtype:
    torch_dtype = torch.float32

    isa_info = probe_cpu_precision_support()
    args.cpu_instruction_flags = isa_info.get("flags", [])
    args.cpu_isa_probe = isa_info

    precision_policy = evaluate_precision_execution_policy(args.precision, isa_info)
    args.cpu_precision_executed = precision_policy["cpu_precision_executed"]
    args.execution_status = precision_policy["status"]
    if not precision_policy["allowed"]:
        args.abort_profiling_due_to_isa = True
        args.abort_profiling_reason = precision_policy["reason"]
        logger.warning(
            "Requested precision is not ISA-accelerated on this CPU. Profiling run will be skipped. "
            f"Reason: {args.abort_profiling_reason}"
        )

    if args.precision == "fp16":
        fp16_info = get_cpu_fp16_support_info()
        args.cpu_fp16_supported = fp16_info["supported"]
        args.cpu_fp16_isa_avx512 = fp16_info["isa_avx512_fp16"]
        args.cpu_fp16_smoke_test_ok = fp16_info["smoke_test_ok"]
        args.cpu_fp16_support_reason = fp16_info["reason"]

        if not fp16_info["supported"]:
            logger.warning(
                "CPU FP16 requested but functional support is not available. "
                "Continuing without FP32 fallback. "
                f"Reason: {fp16_info['reason']}"
            )
        elif not fp16_info["isa_avx512_fp16"]:
            logger.warning(
                "CPU FP16 is functionally available, but AVX512_FP16 was not detected. "
                "Execution may use a slower path."
            )

        torch_dtype = torch.float16
    elif args.precision == "bf16":
        if cpu_supports_bf16():
            torch_dtype = torch.bfloat16
        else:
            logger.warning("CPU does not support BF16 accelerated ISA path.")
            torch_dtype = torch.float32

    return torch_dtype


def main() -> None:
    args = _build_parser().parse_args()

    configure_cpu_runtime(force_threads=args.num_threads)
    set_determinism()
    _initialize_precision_state(args)

    torch_dtype = _configure_precision(args)
    model, inp = build_model_and_input(args, torch_dtype)

    if args.precision == "fp16" and args.cpu_fp16_supported is False and args.cpu_precision_executed == "fp16":
        args.cpu_precision_executed = "fp16_requested_no_cpu_support"
    args.gpu_precision_executed = args.precision

    TrainingProfiler(model, args.model, args).run_profiling(inp)


if __name__ == "__main__":
    main()
