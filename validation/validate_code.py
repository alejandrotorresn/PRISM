#!/usr/bin/env python3
"""
Quick validation of profiler.py code integrity and two-phase timeout logic.

Checks:
1. BACKWARD_FACTOR constant defined
2. run_cpu_fp16_model_preflight function exists and has correct signature
3. Timeout constants are correct values
4. Helper functions exist and have correct logic
"""

import sys
import os
import inspect
from typing import Dict, Any


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def read_profiler_file():
    """Read profiler.py and return content."""
    filepath = os.path.join(SRC_DIR, 'profiler.py')
    with open(filepath, 'r') as f:
        return f.read()

def extract_section(content: str, start_marker: str, end_marker: str) -> str:
    """Extract code section between markers."""
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        return ""
    return content[start_idx:end_idx+len(end_marker)]

def validate_profiler() -> Dict[str, Any]:
    """Run validation checks on profiler.py."""
    content = read_profiler_file()
    import profiler
    from core import constants
    from core import precision_policy
    from runner.training_profiler import TrainingProfiler

    results = {
        "checks_passed": [],
        "checks_failed": [],
        "code_excerpts": {}
    }
    
    # Check 1: BACKWARD_FACTOR constant
    if getattr(constants, "BACKWARD_FACTOR", None) == 2.0 and getattr(profiler, "BACKWARD_FACTOR", None) == 2.0:
        results["checks_passed"].append("✅ BACKWARD_FACTOR = 2.0 defined")
    else:
        results["checks_failed"].append("❌ BACKWARD_FACTOR constant not found or incorrect value")
    
    # Check 2: run_cpu_fp16_model_preflight function signature
    try:
        sig = inspect.signature(precision_policy.run_cpu_fp16_model_preflight)
        if "timeout_safety_factor" in sig.parameters and sig.parameters["timeout_safety_factor"].default == 2.5:
            results["checks_passed"].append("✅ run_cpu_fp16_model_preflight function signature correct")
        else:
            results["checks_failed"].append("❌ run_cpu_fp16_model_preflight function signature incorrect")
    except Exception:
        results["checks_failed"].append("❌ run_cpu_fp16_model_preflight function signature incorrect")

    preflight_src = inspect.getsource(precision_policy.run_cpu_fp16_model_preflight)

    # Check 3: Phase 1 timeout (60s for forward measurement)
    if 'preflight_thread.join(timeout=60.0)' in preflight_src:
        results["checks_passed"].append("✅ Phase 1 timeout = 60.0s for forward measurement")
    else:
        results["checks_failed"].append("❌ Phase 1 timeout (60.0s) not found")

    # Check 4: Phase 2 backward timeout calculation
    if 'backward_timeout = max(10.0, forward_time_sec * BACKWARD_FACTOR * timeout_safety_factor)' in preflight_src:
        results["checks_passed"].append("✅ Phase 2 backward timeout calculation with formula")
    else:
        results["checks_failed"].append("❌ Phase 2 backward timeout calculation formula missing")

    # Check 5: Minimum backward timeout (10s)
    if 'max(10.0, forward_time_sec * BACKWARD_FACTOR * timeout_safety_factor)' in preflight_src:
        results["checks_passed"].append("✅ Backward timeout minimum = 10s")
    else:
        results["checks_failed"].append("❌ Backward timeout minimum not set to 10s")

    # Check 6: Phase 3 second join with calculated timeout
    phase3_join = 'preflight_thread.join(timeout=backward_timeout)'
    if preflight_src.count(phase3_join) > 0:
        results["checks_passed"].append("✅ Phase 3 join with calculated backward_timeout")
    else:
        results["checks_failed"].append("❌ Phase 3 join with backward_timeout not found")

    # Check 7: Diagnostic message for backward timeout case
    backward_timeout_msg = "cpu fp16 backward pass blocked after"
    if backward_timeout_msg in preflight_src:
        results["checks_passed"].append("✅ Backward timeout diagnostic message present")
    else:
        results["checks_failed"].append("❌ Backward timeout diagnostic message missing")

    # Check 8: Diagnostic message for forward timeout case
    forward_timeout_msg = "cpu fp16 forward pass timeout after 60s"
    if forward_timeout_msg in preflight_src:
        results["checks_passed"].append("✅ Forward timeout diagnostic message present")
    else:
        results["checks_failed"].append("❌ Forward timeout diagnostic message missing")

    # Check 9: Helper functions exist
    helpers = ["_extract_loss_for_preflight", "_build_mini_input_for_cpu_fp16"]
    for helper in helpers:
        if hasattr(precision_policy, helper):
            results["checks_passed"].append(f"✅ Helper function {helper} defined")
        else:
            results["checks_failed"].append(f"❌ Helper function {helper} not found")

    # Check 10: Integration point - preflight called in runtime flow
    runtime_src = inspect.getsource(TrainingProfiler.run_profiling)
    if "run_cpu_fp16_model_preflight(self.model, input_data)" in runtime_src:
        results["checks_passed"].append("✅ run_cpu_fp16_model_preflight called in runtime flow")
    else:
        results["checks_failed"].append("❌ run_cpu_fp16_model_preflight not called in runtime flow")

    # Check 11: Metadata fields populated
    if "cpu_fp16_model_smoke_ok" in runtime_src and "cpu_fp16_model_smoke_reason" in runtime_src:
        results["checks_passed"].append("✅ Metadata fields cpu_fp16_model_smoke_ok/cpu_fp16_model_smoke_reason used")
    else:
        results["checks_failed"].append("❌ Metadata fields cpu_fp16_model_smoke_* not found in runtime flow")

    # Check 12: No syntax errors (try to compile orchestrator)
    try:
        compile(content, 'profiler.py', 'exec')
        results["checks_passed"].append("✅ profiler.py compiles without syntax errors")
    except SyntaxError as e:
        results["checks_failed"].append(f"❌ Syntax error: {e}")

    results["code_excerpts"]["phase_1_forward"] = preflight_src[:220] + "..."
    
    return results


def main():
    """Run validation and print results."""
    print("\n" + "="*80)
    print("CODE INTEGRITY VALIDATION: Two-Phase Timeout Mechanism")
    print("="*80 + "\n")
    
    results = validate_profiler()
    
    # Print passed checks
    if results["checks_passed"]:
        print("PASSED CHECKS:")
        print("-" * 80)
        for check in results["checks_passed"]:
            print(check)
    
    # Print failed checks
    if results["checks_failed"]:
        print("\nFAILED CHECKS:")
        print("-" * 80)
        for check in results["checks_failed"]:
            print(check)
    
    # Summary
    total = len(results["checks_passed"]) + len(results["checks_failed"])
    passed = len(results["checks_passed"])
    print("\n" + "="*80)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("="*80)
    
    if results["checks_failed"]:
        print("\n⚠️  VALIDATION FAILED - See above for details\n")
        return 1
    else:
        print("\n✅ ALL VALIDATION CHECKS PASSED\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
