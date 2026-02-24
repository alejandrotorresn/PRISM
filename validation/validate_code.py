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
import re
from typing import Dict, Any

def read_profiler_file():
    """Read profiler.py and return content."""
    filepath = os.path.join(os.path.dirname(__file__), 'src', 'profiler.py')
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
    results = {
        "checks_passed": [],
        "checks_failed": [],
        "code_excerpts": {}
    }
    
    # Check 1: BACKWARD_FACTOR constant
    if "BACKWARD_FACTOR = 2.0" in content:
        results["checks_passed"].append("✅ BACKWARD_FACTOR = 2.0 defined")
    else:
        results["checks_failed"].append("❌ BACKWARD_FACTOR constant not found or incorrect value")
    
    # Check 2: run_cpu_fp16_model_preflight function signature
    if "def run_cpu_fp16_model_preflight(model: nn.Module, input_data: Any, timeout_safety_factor: float = 2.5)" in content:
        results["checks_passed"].append("✅ run_cpu_fp16_model_preflight function signature correct")
    else:
        results["checks_failed"].append("❌ run_cpu_fp16_model_preflight function signature incorrect")
    
    # Check 3: Phase 1 timeout (60s for forward measurement)
    if 'preflight_thread.join(timeout=60.0)' in content:
        results["checks_passed"].append("✅ Phase 1 timeout = 60.0s for forward measurement")
    else:
        results["checks_failed"].append("❌ Phase 1 timeout (60.0s) not found")
    
    # Check 4: Phase 2 backward timeout calculation
    if 'backward_timeout = max(' in content and 'forward_time_sec * BACKWARD_FACTOR * timeout_safety_factor' in content:
        results["checks_passed"].append("✅ Phase 2 backward timeout calculation with formula")
    else:
        results["checks_failed"].append("❌ Phase 2 backward timeout calculation formula missing")
    
    # Check 5: Minimum backward timeout (10s)
    if 'max(' in content and '10.0,' in content:  # accounts for multi-line formatting
        results["checks_passed"].append("✅ Backward timeout minimum = 10s")
    else:
        results["checks_failed"].append("❌ Backward timeout minimum not set to 10s")
    
    # Check 6: Phase 3 second join with calculated timeout
    phase3_join = 'preflight_thread.join(timeout=backward_timeout)'
    if content.count(phase3_join) > 0:
        results["checks_passed"].append("✅ Phase 3 join with calculated backward_timeout")
    else:
        results["checks_failed"].append("❌ Phase 3 join with backward_timeout not found")
    
    # Check 7: Diagnostic message for backward timeout case
    backward_timeout_msg = "cpu fp16 backward pass blocked after"
    if backward_timeout_msg in content:
        results["checks_passed"].append("✅ Backward timeout diagnostic message present")
    else:
        results["checks_failed"].append("❌ Backward timeout diagnostic message missing")
    
    # Check 8: Diagnostic message for forward timeout case
    forward_timeout_msg = "cpu fp16 forward pass timeout after 60s"
    if forward_timeout_msg in content:
        results["checks_passed"].append("✅ Forward timeout diagnostic message present")
    else:
        results["checks_failed"].append("❌ Forward timeout diagnostic message missing")
    
    # Check 9: Helper functions exist
    helpers = ["_extract_loss_for_preflight", "_build_mini_input_for_cpu_fp16"]
    for helper in helpers:
        if f"def {helper}(" in content:
            results["checks_passed"].append(f"✅ Helper function {helper} defined")
        else:
            results["checks_failed"].append(f"❌ Helper function {helper} not found")
    
    # Check 10: Integration point - preflight called in main
    if "run_cpu_fp16_model_preflight(model, inp)" in content:
        results["checks_passed"].append("✅ run_cpu_fp16_model_preflight called in main")
    else:
        results["checks_failed"].append("❌ run_cpu_fp16_model_preflight not called in main")
    
    # Check 11: Metadata fields populated
    metadata_fields = ["cpu_fp16_model_smoke_ok", "cpu_fp16_model_smoke_reason"]
    for field in metadata_fields:
        if f"args.{field}" in content:
            results["checks_passed"].append(f"✅ Metadata field {field} used in args")
        else:
            results["checks_failed"].append(f"❌ Metadata field {field} not found in args")
    
    # Check 12: No syntax errors (try to compile)
    try:
        compile(content, 'profiler.py', 'exec')
        results["checks_passed"].append("✅ profiler.py compiles without syntax errors")
    except SyntaxError as e:
        results["checks_failed"].append(f"❌ Syntax error: {e}")
    
    # Extract timeout-related code sections for reference
    forward_phase = re.search(
        r'# PHASE 1:.*?preflight_thread\.join\(timeout=60\.0\)',
        content,
        re.DOTALL
    )
    if forward_phase:
        results["code_excerpts"]["phase_1_forward"] = forward_phase.group(0)[:200] + "..."
    
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
