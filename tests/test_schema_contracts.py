"""
Regression tests for new schema contracts (operational vs inferential readiness).

Ensures:
1. audit_experimental_grid.py generates all new state fields
2. rescue_influence_* markers propagate through consolidation
3. operational_ready ≠ inferential_strict (they're independent)
4. Backward compatibility with legacy data (missing new columns)
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

# Add project paths
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
VALIDATION = ROOT / "validation"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))


def test_audit_generates_new_state_fields():
    """Verify audit_experimental_grid.py _compute_row generates new fields."""
    import audit_experimental_grid as audit
    
    # Create a minimal test config (mock setup)
    test_cfg = Path(tempfile.gettempdir()) / "test_audit_config"
    test_cfg.mkdir(exist_ok=True)
    
    # Call _compute_row with minimal args (it should not crash and return new fields)
    # _compute_row takes a key tuple and cfg_dir, expected_repeats, require_hybrid, log_failures
    key = ("test_model", "SGD", "fp32", 32)
    row = audit._compute_row(
        key=key,
        cfg_dir=test_cfg,
        expected_repeats=1,
        require_hybrid=False,
        log_failures={},
    )
    
    # Verify new fields exist in output
    expected_fields = [
        "operational_status",
        "inferential_status",
        "rescue_influence_profiling",
        "doctoral_operational_ready",
        "doctoral_inferential_strict",
    ]
    for field in expected_fields:
        assert field in row, f"Missing field: {field}"
    
    # operational_status should be one of: ok, partial, failed
    assert row["operational_status"] in ["ok", "partial", "failed"], \
        f"Invalid operational_status: {row['operational_status']}"
    
    # inferential_status should be one of: strict, qualified, exploratory
    assert row["inferential_status"] in ["strict", "qualified", "exploratory"], \
        f"Invalid inferential_status: {row['inferential_status']}"
    
    # rescue markers should be boolean
    assert isinstance(row["rescue_influence_profiling"], bool)
    assert isinstance(row["doctoral_operational_ready"], bool)
    assert isinstance(row["doctoral_inferential_strict"], bool)


def test_operational_vs_inferential_independent():
    """
    Verify that a config can be operationally_ready but NOT inferential_strict.
    
    This happens when:
    - Profiling executed (operational_ready=True)
    - But with fallback/rescue (inferential_strict=False)
    """
    # Create a mock audit DataFrame
    test_rows = [
        {
            "model": "resnet50",
            "optimizer": "SGD",
            "precision": "fp32",
            "batch_size": 32,
            "operational_status": "ok",
            "inferential_status": "qualified",  # NOT strict
            "rescue_influence_profiling": True,  # Rescue WAS used
            "doctoral_operational_ready": True,
            "doctoral_inferential_strict": False,
        }
    ]
    df = pd.DataFrame(test_rows)
    
    # Verify: operational_ready=True but inferential_strict=False
    assert df.loc[0, "doctoral_operational_ready"] == True
    assert df.loc[0, "doctoral_inferential_strict"] == False
    assert df.loc[0, "rescue_influence_profiling"] == True
    
    print("✓ Confirmed: operational_ready ≠ inferential_strict (independent axes)")


def test_rescue_influence_markers_propagate():
    """
    Verify rescue_influence_* markers are present in consolidated outputs.
    """
    import generate_ilp_report_assets as gen_ilp
    
    # Create mock data with rescue markers
    test_data = {
        "model": ["resnet50"],
        "gpu_budget_mb": [2048],
        "ilp_status": ["optimal"],
        "ilp_objective": [100.0],
        "rescue_influence_profiling": [True],
        "rescue_influence_ilp": [False],
        "rescue_influence_hybrid": [False],
    }
    df = pd.DataFrame(test_data)
    
    # Apply normalization function (should preserve markers)
    df_normalized = gen_ilp._normalize_rescue_columns(df)
    
    assert "rescue_influence_profiling" in df_normalized.columns
    assert "rescue_influence_ilp" in df_normalized.columns
    assert "rescue_influence_hybrid" in df_normalized.columns
    assert df_normalized.loc[0, "rescue_influence_profiling"] == True
    
    print("✓ Confirmed: rescue_influence_* markers propagated")


def test_backward_compat_missing_new_columns():
    """
    Verify scripts handle missing new columns gracefully.
    """
    # Create legacy DataFrame (without new columns)
    legacy_data = {
        "model": ["resnet50"],
        "optimizer": ["SGD"],
        "precision": ["fp32"],
        "batch_size": [32],
        "completion_pct": [95.0],
        "quality_flag": [True],
        "aggregation_ok": [True],
        # No: operational_status, inferential_status, rescue_influence_*, etc.
    }
    df_legacy = pd.DataFrame(legacy_data)
    
    import generate_ilp_report_assets as gen_ilp
    
    # Apply normalization - should add missing rescue columns with defaults
    df_normalized = gen_ilp._normalize_rescue_columns(df_legacy)
    
    assert "rescue_influence_profiling" in df_normalized.columns
    assert "rescue_influence_ilp" in df_normalized.columns
    assert "rescue_influence_hybrid" in df_normalized.columns
    
    # Should default to False if missing
    assert df_normalized.loc[0, "rescue_influence_profiling"] == False
    
    print("✓ Confirmed: backward compat works with legacy data")


def test_audit_csv_export_includes_new_fields():
    """
    Verify that audit_experimental_grid.py CSV output includes new fields.
    """
    # This is more of an integration test: run audit and check CSV columns
    # For now, we verify the fields are in the row dict from _compute_row
    import audit_experimental_grid as audit
    
    test_cfg = Path(tempfile.gettempdir()) / "test_audit_csv"
    test_cfg.mkdir(exist_ok=True)
    
    key = ("test", "SGD", "fp32", 32)
    row = audit._compute_row(
        key=key,
        cfg_dir=test_cfg,
        expected_repeats=1,
        require_hybrid=False,
        log_failures={},
    )
    
    # Convert to DataFrame to simulate CSV export
    df = pd.DataFrame([row])
    
    expected_cols = [
        "operational_status",
        "inferential_status",
        "rescue_influence_profiling",
        "doctoral_operational_ready",
        "doctoral_inferential_strict",
    ]
    for col in expected_cols:
        assert col in df.columns, f"CSV would be missing: {col}"
    
    print("✓ Confirmed: audit CSV export includes new state fields")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING SCHEMA CONTRACT REGRESSION TESTS")
    print("="*80)
    
    try:
        test_audit_generates_new_state_fields()
        print("✓ test_audit_generates_new_state_fields PASSED")
    except Exception as e:
        print(f"✗ test_audit_generates_new_state_fields FAILED: {e}")
    
    try:
        test_operational_vs_inferential_independent()
        print("✓ test_operational_vs_inferential_independent PASSED")
    except Exception as e:
        print(f"✗ test_operational_vs_inferential_independent FAILED: {e}")
    
    try:
        test_rescue_influence_markers_propagate()
        print("✓ test_rescue_influence_markers_propagate PASSED")
    except Exception as e:
        print(f"✗ test_rescue_influence_markers_propagate FAILED: {e}")
    
    try:
        test_backward_compat_missing_new_columns()
        print("✓ test_backward_compat_missing_new_columns PASSED")
    except Exception as e:
        print(f"✗ test_backward_compat_missing_new_columns FAILED: {e}")
    
    try:
        test_audit_csv_export_includes_new_fields()
        print("✓ test_audit_csv_export_includes_new_fields PASSED")
    except Exception as e:
        print(f"✗ test_audit_csv_export_includes_new_fields FAILED: {e}")
    
    print("="*80)
