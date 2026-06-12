"""
Backward-compatible column access utilities for plot/report scripts.

Handles migration from legacy schema to new schema (with operational_status, inferential_status, etc.)
by gracefully falling back to legacy names if new names not present.
"""

import pandas as pd
from typing import Any, Optional


def get_column_safe(
    df: pd.DataFrame,
    new_name: str,
    legacy_name: Optional[str] = None,
    default: Any = None,
) -> pd.Series:
    """
    Get a column by new name with fallback to legacy name.

    Args:
        df: DataFrame to get column from
        new_name: New column name (preferred)
        legacy_name: Old column name (fallback)
        default: Default value if neither name exists

    Returns:
        pd.Series with values from new_name, legacy_name, or default
    """
    if new_name in df.columns:
        return df[new_name]
    elif legacy_name and legacy_name in df.columns:
        return df[legacy_name]
    else:
        return pd.Series([default] * len(df), index=df.index)


def ensure_columns_exist(df: pd.DataFrame, required_columns: dict) -> pd.DataFrame:
    """
    Ensure DataFrame has all required columns, adding with defaults if missing.

    Args:
        df: DataFrame to ensure columns for
        required_columns: Dict mapping {col_name: default_value}

    Returns:
        DataFrame with all required columns (potentially with defaults added)
    """
    df_out = df.copy()
    for col, default in required_columns.items():
        if col not in df_out.columns:
            df_out[col] = default
    return df_out


def infer_operational_status_from_legacy(df: pd.DataFrame) -> pd.Series:
    """
    Infer operational_status from legacy columns if not present.

    Falls back to inferring from 'completion_pct', 'successful_runs', etc.
    """
    if "operational_status" in df.columns:
        return df["operational_status"]

    # Fallback logic: operational_ok if completion_pct >= 90
    if "completion_pct" in df.columns:
        result = df["completion_pct"].apply(lambda x: "ok" if x >= 90 else ("partial" if x > 0 else "failed"))
        return result
    else:
        return pd.Series(["exploratory"] * len(df), index=df.index)


def infer_inferential_status_from_legacy(df: pd.DataFrame) -> pd.Series:
    """
    Infer inferential_status from legacy columns if not present.

    Falls back to inferring from 'quality_flag', 'aggregation_ok', etc.
    """
    if "inferential_status" in df.columns:
        return df["inferential_status"]

    # Fallback logic: strict if quality_flag and aggregation_ok
    if "quality_flag" in df.columns and "aggregation_ok" in df.columns:
        result = df.apply(
            lambda row: "strict"
            if (row.get("quality_flag", False) and row.get("aggregation_ok", False))
            else ("qualified" if row.get("aggregation_ok", False) else "exploratory"),
            axis=1,
        )
        return result
    else:
        return pd.Series(["exploratory"] * len(df), index=df.index)


def infer_rescue_influence_from_legacy(df: pd.DataFrame, col_type: str = "profiling") -> pd.Series:
    """
    Infer rescue_influence_* flags from legacy columns if not present.

    Args:
        df: DataFrame to infer from
        col_type: 'profiling', 'ilp', or 'hybrid'

    Returns:
        pd.Series of boolean rescue_influence flags
    """
    col_name = f"rescue_influence_{col_type}"
    if col_name in df.columns:
        return df[col_name]

    # Fallback logic by type
    if col_type == "profiling":
        # Mark rescue if transfer_fallback_runs > 0 or non_structured_graph_runs > 0
        if "transfer_fallback_runs" in df.columns or "non_structured_graph_runs" in df.columns:
            result = (
                (df.get("transfer_fallback_runs", 0) > 0)
                | (df.get("non_structured_graph_runs", 0) > 0)
            )
            return pd.Series(result, index=df.index)
    elif col_type == "ilp":
        # Mark rescue if ILP used analytical fallback
        if "ilp_analytical_fallback_used" in df.columns:
            return df["ilp_analytical_fallback_used"]
    elif col_type == "hybrid":
        # Mark rescue if hybrid detected fallback in baseline
        if "baseline_rescue_triggered" in df.columns:
            return df["baseline_rescue_triggered"]

    return pd.Series([False] * len(df), index=df.index)
