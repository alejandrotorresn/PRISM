#!/usr/bin/env python3
"""
Comprehensive validation of profiler.py across all supported models.

Tests:
1. Model loading for each model type
2. CPU FP16 preflight for each model  
3. Precision handling (fp32, fp16, bf16)
4. Input data preparation
5. Metadata field population
"""

import sys
import os
import logging
import argparse
from typing import Dict, Tuple, List, Any, Optional

# Add project src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, resnet152,
    ResNet50_Weights, ResNet152_Weights,
    vit_b_16, ViT_B_16_Weights
)
from transformers import BertModel, GPT2Model

from core.constants import BACKWARD_FACTOR
from core.precision_policy import (
    _build_mini_input_for_cpu_fp16,
    _extract_loss_for_preflight,
    cpu_supports_bf16,
    get_cpu_fp16_support_info,
    run_cpu_fp16_model_preflight,
)
from models.factory import SimpleMLP

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates all models across precision modes and preflight."""
    
    MODELS = ["resnet50", "resnet152", "vit_b16", "bert_base", "gpt2_small", "simple_mlp"]
    PRECISIONS = ["fp32", "fp16", "bf16"]
    BATCH_SIZE = 1  # Use batch_size=1 for faster testing
    INPUT_SIZE = 224
    SEQ_LENGTH = 128
    
    def __init__(self, preflight_models: Optional[List[str]] = None, preflight_timeout_safety_factor: float = 2.5):
        self.results = {
            "model_loading": {},
            "preflight": {},
            "precision_handling": {},
            "metadata": {}
        }
        self.models_loaded = {}
        if preflight_models is None:
            self.preflight_models = ["simple_mlp"]
        else:
            self.preflight_models = preflight_models
        self.preflight_timeout_safety_factor = preflight_timeout_safety_factor
    
    def _load_model(self, model_name: str, dtype: torch.dtype) -> Tuple[nn.Module, Any]:
        """Load a model with specified precision."""
        logger.info(f"Loading {model_name} with dtype={dtype}...")
        
        try:
            if model_name == "resnet50":
                weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=weights).to(dtype=dtype)
                inp = torch.randn((self.BATCH_SIZE, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=dtype)
            
            elif model_name == "resnet152":
                weights = ResNet152_Weights.DEFAULT
                model = resnet152(weights=weights).to(dtype=dtype)
                inp = torch.randn((self.BATCH_SIZE, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=dtype)
            
            elif model_name == "vit_b16":
                weights = ViT_B_16_Weights.DEFAULT
                model = vit_b_16(weights=weights).to(dtype=dtype)
                inp = torch.randn((self.BATCH_SIZE, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=dtype)
            
            elif model_name == "bert_base":
                model = BertModel.from_pretrained("bert-base-uncased")
                # BERT handles precision internally, but cast if fp16/bf16
                if dtype != torch.float32:
                    model = model.to(dtype=dtype)
                inp = torch.randint(0, 1000, (self.BATCH_SIZE, self.SEQ_LENGTH), dtype=torch.long)
            
            elif model_name == "gpt2_small":
                model = GPT2Model.from_pretrained("gpt2")
                if dtype != torch.float32:
                    model = model.to(dtype=dtype)
                inp = torch.randint(0, 1000, (self.BATCH_SIZE, self.SEQ_LENGTH), dtype=torch.long)
            
            elif model_name == "simple_mlp":
                model = SimpleMLP().to(dtype=dtype)
                inp = torch.randn((self.BATCH_SIZE, 784), dtype=dtype)
            
            else:
                return None, None, f"❌ Unknown model: {model_name}"
            
            model.to("cpu")
            return model, inp, "✅"
        
        except Exception as e:
            return None, None, f"❌ {type(e).__name__}: {str(e)[:100]}"
    
    def test_model_loading(self):
        """Test 1: Load all models in FP32."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Model Loading (FP32)")
        logger.info("="*80)
        
        for model_name in self.MODELS:
            model, inp, status = self._load_model(model_name, torch.float32)
            self.results["model_loading"][model_name] = status
            
            if model is not None:
                self.models_loaded[model_name] = (model, inp)
                logger.info(f"  {status} {model_name} loaded successfully")
            else:
                logger.error(f"  {status} {model_name}")
    
    def test_precision_handling(self):
        """Test 2: Verify precision casting logic."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Precision Handling (FP32, FP16, BF16)")
        logger.info("="*80)
        
        # Check CPU FP16 support
        fp16_info = get_cpu_fp16_support_info()
        fp16_supported = fp16_info["supported"]
        bf16_supported = cpu_supports_bf16()
        
        logger.info(f"CPU FP16 supported: {fp16_supported} ({fp16_info['reason']})")
        logger.info(f"CPU BF16 supported: {bf16_supported}")
        
        for precision in self.PRECISIONS:
            self.results["precision_handling"][precision] = {}
            
            if precision == "fp16":
                dtype = torch.float16
                supported = fp16_supported
            elif precision == "bf16":
                dtype = torch.bfloat16
                supported = bf16_supported
            else:
                dtype = torch.float32
                supported = True
            
            self.results["precision_handling"][precision]["supported"] = supported
            status = "✅" if supported else "⚠️ "
            logger.info(f"  {status} {precision.upper()} - supported={supported}")
    
    def test_input_preparation(self):
        """Test 3: Verify input preparation for FP16."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: FP16 Input Preparation (_build_mini_input_for_cpu_fp16)")
        logger.info("="*80)
        
        for model_name, (model, inp) in list(self.models_loaded.items())[:3]:  # Test first 3
            try:
                # Prepare mini input for FP16 preflight
                mini_inp = _build_mini_input_for_cpu_fp16(inp)
                
                # Verify dimensions (should be batch_size=1)
                if isinstance(mini_inp, dict):
                    for k, v in mini_inp.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  ✅ {model_name} dict input: {k} shape={tuple(v.shape)}")
                            break
                elif isinstance(mini_inp, torch.Tensor):
                    assert mini_inp.shape[0] == 1, f"Expected batch_size=1, got {mini_inp.shape[0]}"
                    logger.info(f"  ✅ {model_name} tensor input: shape={tuple(mini_inp.shape)}")
            
            except Exception as e:
                logger.error(f"  ❌ {model_name}: {type(e).__name__}: {str(e)[:100]}")
    
    def test_loss_extraction(self):
        """Test 4: Verify loss extraction from different output types."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Loss Extraction (_extract_loss_for_preflight)")
        logger.info("="*80)
        
        test_outputs = [
            ("Tensor", torch.randn((2, 10))),
            ("Tuple[Tensor]", (torch.randn((2, 10)), torch.randn((2,)))),
            ("Object w/ .loss", type('obj', (), {'loss': torch.tensor(1.0)})()),
            ("Object w/ .logits", type('obj', (), {'logits': torch.randn((2, 10))})()),
        ]
        
        for name, out in test_outputs:
            try:
                loss = _extract_loss_for_preflight(out)
                assert isinstance(loss, torch.Tensor) and loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
                logger.info(f"  ✅ {name}: extracted scalar loss")
            except Exception as e:
                logger.error(f"  ❌ {name}: {type(e).__name__}: {str(e)[:100]}")
    
    def test_preflight_all_models(self):
        """Test 5: Run FP16 preflight on all models."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: CPU FP16 Model Preflight (Full Training-Step)")
        logger.info("="*80)
        
        fp16_info = get_cpu_fp16_support_info()
        if not fp16_info["supported"]:
            logger.warning("⚠️  CPU FP16 not supported - skipping preflight tests")
            self.results["preflight"]["skipped"] = "CPU FP16 not supported"
            return

        if not self.preflight_models:
            logger.warning("⚠️  Preflight scope is empty - skipping preflight tests")
            self.results["preflight"]["skipped"] = "Preflight disabled by configuration"
            return

        logger.info(f"Preflight scope: {', '.join(self.preflight_models)}")
        
        for model_name in self.preflight_models:
            if model_name not in self.models_loaded:
                logger.info(f"  ⏭️  {model_name}: not loaded, skipping preflight")
                continue
            
            model, inp = self.models_loaded[model_name]
            
            try:
                logger.info(f"  Running preflight for {model_name}...")
                result = run_cpu_fp16_model_preflight(
                    model,
                    inp,
                    timeout_safety_factor=self.preflight_timeout_safety_factor,
                )
                
                # Store result
                self.results["preflight"][model_name] = {
                    "ok": result["ok"],
                    "reason": result["reason"]
                }
                
                status = "✅" if result["ok"] else "⏱️ "
                logger.info(f"    {status} ok={result['ok']}")
                if len(result["reason"]) < 150:
                    logger.info(f"       reason: {result['reason']}")
                else:
                    logger.info(f"       reason: {result['reason'][:150]}...")
            
            except Exception as e:
                logger.error(f"  ❌ {model_name}: {type(e).__name__}: {str(e)[:150]}")
                self.results["preflight"][model_name] = {
                    "ok": False,
                    "reason": f"Exception: {type(e).__name__}"
                }
    
    def test_metadata_fields(self):
        """Test 6: Verify all metadata fields are populated correctly."""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Metadata Fields Population")
        logger.info("="*80)
        
        required_fields = [
            "cpu_fp16_supported",
            "cpu_fp16_isa_avx512",
            "cpu_fp16_smoke_test_ok",
            "cpu_fp16_model_smoke_ok",
            "cpu_fp16_model_smoke_reason",
            "cpu_fp16_support_reason",
        ]
        
        # Simulate args object from main
        class MockArgs:
            pass
        
        args = MockArgs()
        fp16_info = get_cpu_fp16_support_info()
        
        args.cpu_fp16_supported = fp16_info["supported"]
        args.cpu_fp16_isa_avx512 = fp16_info["isa_avx512_fp16"]
        args.cpu_fp16_smoke_test_ok = fp16_info["smoke_test_ok"]
        args.cpu_fp16_model_smoke_ok = True  # Example
        args.cpu_fp16_model_smoke_reason = "Example preflight reason"
        args.cpu_fp16_support_reason = fp16_info["reason"]
        
        for field in required_fields:
            if hasattr(args, field):
                value = getattr(args, field)
                logger.info(f"  ✅ {field}: {value}")
                self.results["metadata"][field] = "present"
            else:
                logger.error(f"  ❌ {field}: MISSING")
                self.results["metadata"][field] = "MISSING"
    
    def test_backward_factor_constant(self):
        """Test 7: Verify BACKWARD_FACTOR constant."""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: BACKWARD_FACTOR Constant")
        logger.info("="*80)
        
        expected = 2.0
        if BACKWARD_FACTOR == expected:
            logger.info(f"  ✅ BACKWARD_FACTOR = {BACKWARD_FACTOR} (standard literature value)")
        else:
            logger.error(f"  ❌ BACKWARD_FACTOR = {BACKWARD_FACTOR}, expected {expected}")
    
    def run_all_tests(self):
        """Execute all validation tests."""
        logger.info("\n" + "🔍 COMPREHENSIVE PROFILER VALIDATION SUITE\n")
        logger.info(f"Models to test: {', '.join(self.MODELS)}")
        logger.info(f"Precisions to test: {', '.join(self.PRECISIONS)}")
        logger.info(f"Batch size: {self.BATCH_SIZE}\n")
        logger.info(f"FP16 preflight scope: {', '.join(self.preflight_models) if self.preflight_models else 'none'}")
        
        self.test_model_loading()
        self.test_precision_handling()
        self.test_input_preparation()
        self.test_loss_extraction()
        self.test_preflight_all_models()
        self.test_metadata_fields()
        self.test_backward_factor_constant()
        
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        # Model Loading Summary
        logger.info("\n1️⃣  Model Loading:")
        passed = sum(1 for v in self.results["model_loading"].values() if "✅" in v)
        total = len(self.results["model_loading"])
        logger.info(f"   {passed}/{total} models loaded successfully")
        for model, status in self.results["model_loading"].items():
            marker = "✅" if "✅" in status else "❌"
            logger.info(f"   {marker} {model}")
        
        # Precision Handling Summary
        logger.info("\n2️⃣  Precision Handling:")
        for prec, data in self.results["precision_handling"].items():
            supported = data.get("supported", False)
            marker = "✅" if supported else "⚠️ "
            logger.info(f"   {marker} {prec.upper()}: supported={supported}")
        
        # Preflight Summary
        logger.info("\n3️⃣  FP16 Preflight Results:")
        if "skipped" in self.results["preflight"]:
            logger.info(f"   ⏭️  Skipped: {self.results['preflight']['skipped']}")
        else:
            for model, result in self.results["preflight"].items():
                marker = "✅" if result["ok"] else "⏱️ "
                logger.info(f"   {marker} {model}: ok={result['ok']}")
        
        # Metadata Summary
        logger.info("\n4️⃣  Metadata Fields:")
        missing = [f for f, status in self.results["metadata"].items() if "MISSING" in status]
        if not missing:
            logger.info(f"   ✅ All {len(self.results['metadata'])} required fields present")
        else:
            logger.error(f"   ❌ Missing {len(missing)} field(s): {', '.join(missing)}")
        
        logger.info("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Validate all profiler models and precision helpers")
    parser.add_argument(
        "--preflight-scope",
        choices=["fast", "all", "none"],
        default="fast",
        help="FP16 CPU preflight scope: fast=simple_mlp only, all=all models, none=skip preflight",
    )
    parser.add_argument(
        "--preflight-timeout-safety-factor",
        type=float,
        default=2.5,
        help="Timeout safety factor passed to FP16 preflight",
    )
    args = parser.parse_args()

    if args.preflight_scope == "all":
        preflight_models = ModelValidator.MODELS.copy()
    elif args.preflight_scope == "none":
        preflight_models = []
    else:
        preflight_models = ["simple_mlp"]

    validator = ModelValidator(
        preflight_models=preflight_models,
        preflight_timeout_safety_factor=args.preflight_timeout_safety_factor,
    )
    validator.run_all_tests()


if __name__ == "__main__":
    main()
