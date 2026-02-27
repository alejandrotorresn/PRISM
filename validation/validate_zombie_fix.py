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
import inspect
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def check_arguments_exist():
    """Check that --skip_cpu and --num_threads arguments exist"""
    import profiler

    parser_src = inspect.getsource(profiler._build_parser)
    
    checks = [
        ('--skip_cpu' in parser_src and 'store_true' in parser_src,
         "✓ --skip_cpu argument found"),
        ('--num_threads' in parser_src and 'type=int' in parser_src,
         "✓ --num_threads argument found"),
    ]
    
    results = []
    for check, msg in checks:
        results.append((check, msg))
        print(f"{'✓' if check else '✗'} {msg}")
    
    return all(r[0] for r in results)

def check_configure_cpu_runtime_signature():
    """Check that configure_cpu_runtime accepts force_threads parameter"""
    from core.system import configure_cpu_runtime

    sig = inspect.signature(configure_cpu_runtime)
    found = "force_threads" in sig.parameters
    
    print(f"{'✓' if found else '✗'} configure_cpu_runtime(force_threads=0) signature")
    return found

def check_preflight_in_run_profiling():
    """Check that preflight is called inside run_profiling, not __main__"""
    import profiler
    from runner.training_profiler import TrainingProfiler

    main_src = inspect.getsource(profiler.main)
    preflight_in_main = "run_cpu_fp16_model_preflight" in main_src

    run_profiling_src = inspect.getsource(TrainingProfiler.run_profiling)
    preflight_in_run_profiling = "run_cpu_fp16_model_preflight(self.model, input_data)" in run_profiling_src
    
    check1 = not preflight_in_main
    check2 = preflight_in_run_profiling
    
    print(f"{'✓' if check1 else '✗'} Preflight removed from __main__")
    print(f"{'✓' if check2 else '✗'} Preflight moved to run_profiling()")
    
    return check1 and check2

def check_skip_cpu_logic():
    """Check that skip_cpu flag is used in run_profiling"""
    from runner.training_profiler import TrainingProfiler

    runtime_src = inspect.getsource(TrainingProfiler.run_profiling)

    pattern1 = r'not getattr\(self\.args, "skip_cpu".*False\)'
    pattern2 = r'getattr\(self\.args, "skip_cpu".*False\)'

    check1 = bool(re.search(pattern1, runtime_src))
    check2 = bool(re.search(pattern2, runtime_src))
    
    print(f"{'✓' if check1 else '✗'} --skip_cpu flag respected in preflight call")
    print(f"{'✓' if check2 else '✗'} --skip_cpu flag respected in skip_cpu_profile logic")
    
    return check1 and check2

def check_force_threads_usage():
    """Check that force_threads parameter is passed to configure_cpu_runtime"""
    import profiler

    content = inspect.getsource(profiler.main)

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
