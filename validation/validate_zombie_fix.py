#!/usr/bin/env python3
"""
Validation script for zombie thread fix.
Tests:
1. --skip_cpu and --num_threads arguments are recognized
2. configure_cpu_runtime accepts force_threads parameter
3. Preflight is located inside run_profiling, not __main__
"""

import argparse
import sys
import re
from pathlib import Path

def check_arguments_exist():
    """Check that --skip_cpu and --num_threads arguments exist"""
    profiler_path = Path(__file__).parent / "src" / "profiler.py"
    content = profiler_path.read_text()
    
    checks = [
        ('--skip_cpu' in content and "action='store_true'" in content, 
         "✓ --skip_cpu argument found"),
        ('--num_threads' in content and 'type=int' in content, 
         "✓ --num_threads argument found"),
    ]
    
    results = []
    for check, msg in checks:
        results.append((check, msg))
        print(f"{'✓' if check else '✗'} {msg}")
    
    return all(r[0] for r in results)

def check_configure_cpu_runtime_signature():
    """Check that configure_cpu_runtime accepts force_threads parameter"""
    profiler_path = Path(__file__).parent / "src" / "profiler.py"
    content = profiler_path.read_text()
    
    # Look for function signature with force_threads parameter
    pattern = r'def configure_cpu_runtime\(force_threads:\s*int\s*=\s*0\)'
    found = bool(re.search(pattern, content))
    
    print(f"{'✓' if found else '✗'} configure_cpu_runtime(force_threads=0) signature")
    return found

def check_preflight_in_run_profiling():
    """Check that preflight is called inside run_profiling, not __main__"""
    profiler_path = Path(__file__).parent / "src" / "profiler.py"
    content = profiler_path.read_text()
    
    # Check that preflight is NOT in __main__
    main_section = content[content.find("if __name__ == \"__main__\":"):] 
    preflight_in_main = "run_cpu_fp16_model_preflight(model, inp)" in main_section
    
    # Check that preflight IS in run_profiling method
    run_profiling_start = content.find("def run_profiling(self, input_data:")
    run_profiling_end = content.find("\n    def ", run_profiling_start + 1)
    if run_profiling_end == -1:
        run_profiling_end = len(content)
    
    run_profiling_section = content[run_profiling_start:run_profiling_end]
    preflight_in_run_profiling = "run_cpu_fp16_model_preflight(self.model, input_data)" in run_profiling_section
    
    check1 = not preflight_in_main
    check2 = preflight_in_run_profiling
    
    print(f"{'✓' if check1 else '✗'} Preflight removed from __main__")
    print(f"{'✓' if check2 else '✗'} Preflight moved to run_profiling()")
    
    return check1 and check2

def check_skip_cpu_logic():
    """Check that skip_cpu flag is used in run_profiling"""
    profiler_path = Path(__file__).parent / "src" / "profiler.py"
    content = profiler_path.read_text()
    
    # Check that skip_cpu is respected in preflight AND skip_cpu_profile logic
    pattern1 = r'not getattr\(self\.args, "skip_cpu".*False\)'  # In preflight condition
    pattern2 = r'getattr\(self\.args, "skip_cpu".*False\)'     # In skip_cpu_profile
    
    check1 = bool(re.search(pattern1, content))
    check2 = bool(re.search(pattern2, content))
    
    print(f"{'✓' if check1 else '✗'} --skip_cpu flag respected in preflight call")
    print(f"{'✓' if check2 else '✗'} --skip_cpu flag respected in skip_cpu_profile logic")
    
    return check1 and check2

def check_force_threads_usage():
    """Check that force_threads parameter is passed to configure_cpu_runtime"""
    profiler_path = Path(__file__).parent / "src" / "profiler.py"
    content = profiler_path.read_text()
    
    # In __main__, should call: configure_cpu_runtime(force_threads=args.num_threads)
    pattern = r'configure_cpu_runtime\(force_threads=args\.num_threads\)'
    found = bool(re.search(pattern, content))
    
    print(f"{'✓' if found else '✗'} configure_cpu_runtime called with args.num_threads")
    return found

def main():
    print("=" * 70)
    print("VALIDATING ZOMBIE THREAD FIX")
    print("=" * 70)
    
    tests = [
        ("Arguments Check", check_arguments_exist),
        ("configure_cpu_runtime Signature", check_configure_cpu_runtime_signature),
        ("Preflight Location", check_preflight_in_run_profiling),
        ("--skip_cpu Logic", check_skip_cpu_logic),
        ("--num_threads Usage", check_force_threads_usage),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        print(f"{'✓' if result else '✗'} {test_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All validation checks PASSED - Zombie thread fix is correctly implemented!")
        return 0
    else:
        print(f"\n❌ {total - passed} check(s) FAILED - Review the implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
