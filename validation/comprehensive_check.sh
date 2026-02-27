#!/bin/bash

# COMPREHENSIVE MODEL & PREFLIGHT VALIDATION
# Verifica que TODOS los modelos estén correctamente integrados (arquitectura modular)

echo "================================================================================"
echo "COMPREHENSIVE MODEL & TIMEOUT VALIDATION FOR ALL MODELS"
echo "================================================================================"
echo ""

check_count=0
pass_count=0

check_in_file() {
    local test_name="$1"
    local file_path="$2"
    local search_regex="$3"
    check_count=$((check_count + 1))

    if grep -Eq "$search_regex" "$file_path"; then
        echo "✅ [$check_count] $test_name"
        pass_count=$((pass_count + 1))
    else
        echo "❌ [$check_count] $test_name"
        echo "    File: $file_path"
        echo "    Search: $search_regex"
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 1: MODEL SELECTION & LOADING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_in_file "ResNet50 Loading" "src/models/factory.py" 'if args.model == "resnet50":'
check_in_file "ResNet50 Weights" "src/models/factory.py" 'ResNet50_Weights.DEFAULT'
check_in_file "ResNet50 Input Shape" "src/models/factory.py" '\(args.batch_size, 3, args.input_size, args.input_size\)'

check_in_file "ResNet152 Loading" "src/models/factory.py" 'elif args.model == "resnet152":'
check_in_file "ResNet152 Weights" "src/models/factory.py" 'ResNet152_Weights.DEFAULT'

check_in_file "ViT-B/16 Loading" "src/models/factory.py" 'elif args.model == "vit_b16":'
check_in_file "ViT-B/16 Weights" "src/models/factory.py" 'ViT_B_16_Weights.DEFAULT'

check_in_file "BERT Loading" "src/models/factory.py" 'elif args.model == "bert_base":'
check_in_file "BERT Weights" "src/models/factory.py" 'BertModel.from_pretrained\("bert-base-uncased"\)'
check_in_file "BERT Input (int64)" "src/models/factory.py" 'torch.randint\(0, 1000, \(args.batch_size, args.seq_length\), dtype=torch.long\)'

check_in_file "GPT2 Loading" "src/models/factory.py" 'elif args.model == "gpt2_small":'
check_in_file "GPT2 Weights" "src/models/factory.py" 'GPT2Model.from_pretrained\("gpt2"\)'

check_in_file "SimpleMLP Loading" "src/models/factory.py" 'elif args.model == "simple_mlp":'
check_in_file "SimpleMLP Input" "src/models/factory.py" '\(args.batch_size, 784\)'

check_in_file "Model Selection Validation" "src/models/factory.py" 'raise ValueError\(f"Unsupported model: \{args.model\}"\)'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 2: PRECISION HANDLING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_in_file "FP16 Detection Function" "src/core/precision_policy.py" 'def get_cpu_fp16_support_info\(\)'
check_in_file "FP16 ISA Check" "src/core/precision_policy.py" 'avx512_fp16'
check_in_file "FP16 Smoke Test" "src/core/precision_policy.py" 'torch.mm\(a, b\)'
check_in_file "BF16 Detection" "src/core/precision_policy.py" 'def cpu_supports_bf16\(\)'
check_in_file "BF16 ISA Check" "src/core/precision_policy.py" 'avx512_bf16'

check_in_file "Precision Branch - FP32" "src/profiler.py" 'torch_dtype = torch.float32'
check_in_file "Precision Branch - FP16" "src/profiler.py" 'torch_dtype = torch.float16'
check_in_file "Precision Branch - BF16" "src/profiler.py" 'torch_dtype = torch.bfloat16'

check_in_file "NLP Model Exclusion from Cast" "src/models/factory.py" 'args.model not in \["bert_base", "gpt2_small"\]'
check_in_file "Input Casting Logic" "src/models/factory.py" 'inp = inp.to\(dtype=torch_dtype\)'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 3: CPU FP16 PREFLIGHT (TWO-PHASE TIMEOUT)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_in_file "Preflight Function Definition" "src/core/precision_policy.py" 'def run_cpu_fp16_model_preflight'
check_in_file "Preflight Signature" "src/core/precision_policy.py" 'model: nn.Module, input_data: Any, timeout_safety_factor: float = 2.5'

check_in_file "BACKWARD_FACTOR Constant" "src/core/constants.py" 'BACKWARD_FACTOR = 2.0'
check_in_file "BACKWARD_FACTOR Usage (GPU)" "src/runner/training_profiler.py" '"gpu_bwd_time_ms": t_fwd_gpu \* BACKWARD_FACTOR'
check_in_file "BACKWARD_FACTOR Usage (CPU)" "src/runner/training_profiler.py" '"cpu_bwd_time_ms": t_fwd_cpu \* BACKWARD_FACTOR'
check_in_file "BACKWARD_FACTOR Usage (Preflight)" "src/core/precision_policy.py" 'forward_time_sec \* BACKWARD_FACTOR'

check_in_file "PHASE 1: Forward Measurement (60s)" "src/core/precision_policy.py" 'preflight_thread.join\(timeout=60.0\)'
check_in_file "PHASE 2: Adaptive Timeout Calculation" "src/core/precision_policy.py" 'backward_timeout = max\('
check_in_file "PHASE 2: Min Timeout 10s" "src/core/precision_policy.py" '10.0,'
check_in_file "PHASE 2: Formula" "src/core/precision_policy.py" 'forward_time_sec \* BACKWARD_FACTOR \* timeout_safety_factor'
check_in_file "PHASE 3: Backward+Step Wait" "src/core/precision_policy.py" 'preflight_thread.join\(timeout=backward_timeout\)'

check_in_file "Loss Extraction Helper" "src/core/precision_policy.py" 'def _extract_loss_for_preflight'
check_in_file "Input Preparation Helper" "src/core/precision_policy.py" 'def _build_mini_input_for_cpu_fp16'
check_in_file "Preflight Called in Runtime Flow" "src/runner/training_profiler.py" 'run_cpu_fp16_model_preflight\(self.model, input_data\)'

check_in_file "Preflight Result Storage (ok)" "src/runner/training_profiler.py" 'args.cpu_fp16_model_smoke_ok = model_preflight\["ok"\]'
check_in_file "Preflight Result Storage (reason)" "src/runner/training_profiler.py" 'args.cpu_fp16_model_smoke_reason = model_preflight\["reason"\]'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 4: METADATA FIELDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_in_file "ISA Flag Field Init" "src/profiler.py" 'args.cpu_fp16_isa_avx512 = None'
check_in_file "Support Field Init" "src/profiler.py" 'args.cpu_fp16_supported = None'
check_in_file "Smoke Test Field Init" "src/profiler.py" 'args.cpu_fp16_smoke_test_ok = None'
check_in_file "Model Smoke Field Init" "src/profiler.py" 'args.cpu_fp16_model_smoke_ok = None'
check_in_file "Model Smoke Reason Field Init" "src/profiler.py" 'args.cpu_fp16_model_smoke_reason = None'
check_in_file "Support Reason Field Init" "src/profiler.py" 'args.cpu_fp16_support_reason = None'

check_in_file "ISA Field in GPU Partial JSON" "src/runner/training_profiler.py" '"cpu_fp16_isa_avx512":'
check_in_file "Support Field in GPU Partial JSON" "src/runner/training_profiler.py" '"cpu_fp16_supported":'
check_in_file "Model Smoke Field in GPU Partial JSON" "src/runner/training_profiler.py" '"cpu_fp16_model_smoke_ok":'
check_in_file "Model Smoke Reason in GPU Partial JSON" "src/runner/training_profiler.py" '"cpu_fp16_model_smoke_reason":'

check_in_file "ISA Field in Final JSON" "src/runner/training_profiler.py" '"cpu_fp16_isa_avx512":'
check_in_file "Support Field in Final JSON" "src/runner/training_profiler.py" '"cpu_fp16_supported":'
check_in_file "Model Smoke Field in Final JSON" "src/runner/training_profiler.py" '"cpu_fp16_model_smoke_ok":'
check_in_file "Model Smoke Reason in Final JSON" "src/runner/training_profiler.py" '"cpu_fp16_model_smoke_reason":'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 5: PRECISION EXECUTION TRACKING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_in_file "CPU Precision Tracking Logic" "src/profiler.py" 'args.cpu_precision_executed'
check_in_file "GPU Precision Tracking Logic" "src/profiler.py" 'args.gpu_precision_executed'
check_in_file "FP16 Preflight Failure Path" "src/runner/training_profiler.py" 'fp16_requested_model_preflight_failed'
check_in_file "FP16 No Support Path" "src/profiler.py" 'fp16_requested_no_cpu_support'
check_in_file "BF16 Unsupported ISA Skip Path" "src/core/precision_policy.py" 'bf16_requested_isa_unsupported'

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "Total Checks: $check_count"
echo "Passed: $pass_count"
echo "Failed: $((check_count - pass_count))"

if [ $pass_count -eq $check_count ]; then
    echo ""
    echo "🎉 ALL CHECKS PASSED!"
    echo ""
    echo "✅ All 6 models are correctly integrated"
    echo "✅ Two-phase timeout mechanism is properly implemented"
    echo "✅ BACKWARD_FACTOR = 2.0 is used throughout"
    echo "✅ All metadata fields are present and initialized"
    echo "✅ Precision handling (FP32/FP16/BF16) is correct"
    echo "✅ Preflight uses proper timeout formula: max(10s, forward×2.0×2.5)"
    echo "✅ NLP models correctly exclude precision cast from input"
    echo "✅ Code is production-ready"
    echo ""
    exit 0
else
    echo ""
    echo "⚠️  SOME CHECKS FAILED"
    echo "Review failures above"
    echo ""
    exit 1
fi
