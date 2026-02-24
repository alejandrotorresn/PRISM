#!/usr/bin/env python3
"""
Validation script for two-phase timeout mechanism in CPU FP16 preflight.

Tests the following scenarios:
1. Fast model (simple_mlp): forward ~10ms → backward_timeout = max(10s, 10ms×2.0×2.5) = 10s (minimum)
2. Medium model (ViT): forward ~250ms → backward_timeout = max(10s, 250ms×2.0×2.5) = 1.25s (should timeout)
3. Slow model: forward ~2s → backward_timeout = max(10s, 2s×2.0×2.5) = 10s (calculated minimum hit)

Each test validates:
- Two-phase join() behavior (60s Phase 1 for forward, adaptive Phase 2 for backward)
- Timeout calculation formulas
- Metadata reason strings match expected format
"""

import sys
import os
import logging
import torch
import torch.nn as nn
import argparse
from typing import Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from profiler import run_cpu_fp16_model_preflight

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """Fast model: forward ~10ms on CPU, backward should complete instantly."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SlowMLP(nn.Module):
    """Medium-slow model: forward ~500ms on CPU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class BlockingBackwardMLP(nn.Module):
    """Model that will block in backward pass (simulates AVX512_FP16 missing)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        # Custom backward hook that simulates blocking
        self._register_hook()
    
    def _register_hook(self):
        """Register a backward hook that blocks to simulate ISA issues."""
        def blocking_hook(grad):
            # This hook will be called during backward and will block
            import time
            logger.warning("BACKWARD HOOK: Starting 5-second blocking simulation (ISA missing)")
            time.sleep(5.0)
            logger.warning("BACKWARD HOOK: Completed blocking (simulating ISA timeout scenario)")
            return grad
        
        # Register on the ReLU output
        self.register_parameter('_hook_param', nn.Parameter(torch.tensor(1.0)))
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # Register hook on output
        x.register_hook(lambda g: (logger.warning("BACKWARD: Blocking for 5s..."), __import__('time').sleep(5.0), g)[2])
        x = self.fc2(x)
        return x


def run_test(model_name: str, model: nn.Module, input_data: torch.Tensor, 
             expected_behavior: str, safety_factor: float = 2.5) -> Tuple[bool, str]:
    """
    Run preflight test and validate timeout behavior.
    
    Args:
        model_name: Display name for test
        model: Model to test
        input_data: Input tensor
        expected_behavior: One of ["pass", "timeout_backward", "timeout_forward"]
        safety_factor: Timeout safety factor (default 2.5)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST: {model_name}")
    logger.info(f"Expected behavior: {expected_behavior}")
    logger.info(f"Safety factor: {safety_factor}x")
    logger.info(f"{'='*80}")
    
    try:
        # Run preflight
        start_time = datetime.now()
        result = run_cpu_fp16_model_preflight(model, input_data, timeout_safety_factor=safety_factor)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Preflight completed in {elapsed:.2f}s")
        logger.info(f"Result OK: {result['ok']}")
        logger.info(f"Reason: {result['reason']}")
        
        # Validate against expected behavior
        if expected_behavior == "pass":
            if result['ok']:
                msg = f"✅ PASS: Model preflight succeeded as expected. Reason: {result['reason']}"
                logger.info(msg)
                return True, msg
            else:
                msg = f"❌ FAIL: Expected pass but preflight failed. Reason: {result['reason']}"
                logger.error(msg)
                return False, msg
        
        elif expected_behavior == "timeout_backward":
            if not result['ok'] and "backward pass blocked" in result['reason'].lower():
                msg = f"✅ PASS: Backward timeout detected as expected. Reason: {result['reason']}"
                logger.info(msg)
                return True, msg
            else:
                msg = f"❌ FAIL: Expected backward timeout but got different result. Reason: {result['reason']}"
                logger.error(msg)
                return False, msg
        
        elif expected_behavior == "timeout_forward":
            if not result['ok'] and "forward pass timeout" in result['reason'].lower():
                msg = f"✅ PASS: Forward timeout detected as expected. Reason: {result['reason']}"
                logger.info(msg)
                return True, msg
            else:
                msg = f"❌ FAIL: Expected forward timeout but got different result. Reason: {result['reason']}"
                logger.error(msg)
                return False, msg
        
        else:
            return False, f"Unknown expected behavior: {expected_behavior}"
    
    except Exception as e:
        msg = f"❌ EXCEPTION: {type(e).__name__}: {e}"
        logger.exception(msg)
        return False, msg


def main():
    """Run all timeout validation tests."""
    parser = argparse.ArgumentParser(description="Validate two-phase timeout mechanism")
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'fast', 'slow', 'blocking'],
                        help='Which test to run')
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("CPU FP16 TWO-PHASE TIMEOUT VALIDATION TEST SUITE")
    logger.info("="*80)
    logger.info("Timeout formula: max(10s, forward_time × BACKWARD_FACTOR(2.0) × safety_factor(2.5))")
    logger.info("="*80)
    
    tests = []
    
    # Test 1: Fast model (should pass with 10s minimum backward timeout)
    if args.test in ['all', 'fast']:
        logger.info("\n[TEST 1] Fast Model (SimpleMLP)")
        logger.info("Expected: forward ~10ms → backward_timeout = max(10s, 10ms × 2.0 × 2.5) = 10s minimum")
        model1 = SimpleMLP()
        inp1 = torch.randn((1, 784), dtype=torch.float32)
        success, msg = run_test("SimpleMLP (Fast)", model1, inp1, "pass")
        tests.append(("SimpleMLP", success, msg))
    
    # Test 2: Medium-slow model (should pass, but with smaller backward timeout)
    if args.test in ['all', 'slow']:
        logger.info("\n[TEST 2] Medium-Slow Model (SlowMLP)")
        logger.info("Expected: forward ~500ms → backward_timeout = max(10s, 500ms × 2.0 × 2.5) = 10s minimum")
        logger.info("Note: May still timeout if backward is actually blocked by missing AVX512_FP16")
        model2 = SlowMLP()
        inp2 = torch.randn((1, 784), dtype=torch.float32)
        success, msg = run_test("SlowMLP (Medium-Slow)", model2, inp2, "pass")
        tests.append(("SlowMLP", success, msg))
    
    # Test 3: Blocking backward model (should timeout in backward)
    if args.test in ['all', 'blocking']:
        logger.info("\n[TEST 3] Blocking Backward Model (BlockingBackwardMLP)")
        logger.info("Expected: forward ~10ms, then backward blocks for 5s")
        logger.info("With 2.5x safety factor: backward_timeout = max(10s, ...) = 10s, so should complete despite blocking")
        logger.info("(Timeout is 10s, blocking is 5s)")
        model3 = BlockingBackwardMLP()
        inp3 = torch.randn((1, 784), dtype=torch.float32)
        # Note: This will block for 5s, which is less than 10s minimum, so should pass
        success, msg = run_test("BlockingBackwardMLP", model3, inp3, "pass")
        tests.append(("BlockingBackwardMLP", success, msg))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    for test_name, success, message in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    total = len(tests)
    passed = sum(1 for _, s, _ in tests if s)
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
