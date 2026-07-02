#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONFIG_KEY = Tuple[str, str, str, int]


def _parse_csv_values(text: Optional[str], cast_int: bool = False) -> List[str] | List[int]:
    if text is None:
        return []
    values: List[str] = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if cast_int:
        out: List[int] = []
        for value in values:
            out.append(int(value))
        return out
    return values


def _discover_batch_dirs(input_root: Path) -> Dict[CONFIG_KEY, Path]:
    out: Dict[CONFIG_KEY, Path] = {}
    for batch_dir in sorted(input_root.rglob("batch_*")):
        if not batch_dir.is_dir():
            continue
        if len(batch_dir.parts) < 4:
            continue
        batch_token = batch_dir.name
        if not batch_token.startswith("batch_"):
            continue
        try:
            batch_size = int(batch_token.split("_", 1)[1])
        except Exception:
            continue

        precision = batch_dir.parent.name
        optimizer = batch_dir.parent.parent.name
        model = batch_dir.parent.parent.parent.name
        out[(model, optimizer, precision, batch_size)] = batch_dir
    return out


def _find_stats_csv(cfg_dir: Path, model: str) -> Optional[Path]:
    p1 = cfg_dir / f"{model}_metrics_stats.csv"
    if p1.exists():
        return p1
    p2 = cfg_dir / "metrics_stats.csv"
    if p2.exists():
        return p2
    return None


def _load_meta_files(cfg_dir: Path, model: str) -> List[Dict[str, object]]:
    metas: List[Dict[str, object]] = []
    for run_dir in sorted(cfg_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        model_meta = run_dir / f"{model}_meta.json"
        generic_meta = run_dir / "meta.json"
        meta_path = model_meta if model_meta.exists() else generic_meta
        if not meta_path.exists():
            continue
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return metas


def _load_log_failures(logs_root: Optional[Path]) -> Dict[CONFIG_KEY, List[str]]:
    if logs_root is None or not logs_root.exists():
        return {}

    files = sorted(logs_root.rglob("experiments_*.txt"))
    if not files:
        return {}

    block_re = re.compile(
        r"Profiling Block:\s*Model=(?P<model>[^|]+)\|\s*Optimizer=(?P<opt>[^|]+)\|\s*Precision=(?P<prec>.+)$"
    )
    fail_re = re.compile(r"FAILURE:\s*Batch\s*(?P<batch>\d+)\s*replicate\s*(?P<run>run_\d+)")

    failures: Dict[CONFIG_KEY, List[str]] = {}
    cur_model = ""
    cur_opt = ""
    cur_prec = ""

    for fp in files:
        try:
            lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            m_block = block_re.search(line)
            if m_block:
                cur_model = m_block.group("model").strip()
                cur_opt = m_block.group("opt").strip()
                cur_prec = m_block.group("prec").strip()
                continue

            m_fail = fail_re.search(line)
            if not m_fail or not cur_model or not cur_opt or not cur_prec:
                continue

            batch_size = int(m_fail.group("batch"))
            msg_parts = [line.strip()]
            if i + 1 < len(lines) and "Probable causes" in lines[i + 1]:
                msg_parts.append(lines[i + 1].strip())
            key = (cur_model, cur_opt, cur_prec, batch_size)
            failures.setdefault(key, []).append(" | ".join(msg_parts))

    return failures


def _bool_to_stage(flag: bool, ok: str, bad: str) -> str:
    return ok if flag else bad


def _compute_row(
    key: CONFIG_KEY,
    cfg_dir: Optional[Path],
    expected_repeats: int,
    require_hybrid: bool,
    log_failures: Dict[CONFIG_KEY, List[str]],
) -> Dict[str, object]:
    model, optimizer, precision, batch_size = key

    cfg_exists = bool(cfg_dir is not None and cfg_dir.exists())
    run_dirs = sorted(cfg_dir.glob("run_*")) if cfg_exists else []
    run_dirs_count = len([p for p in run_dirs if p.is_dir()])

    metrics_files = []
    if cfg_exists:
        for path in cfg_dir.rglob("*_metrics.csv"):
            name = path.name
            if name.endswith("_metrics_stats.csv") or name.endswith("_metrics_gpu_partial.csv"):
                continue
            metrics_files.append(path)
    metrics_files_count = len(metrics_files)

    meta_records = _load_meta_files(cfg_dir, model) if cfg_exists else []
    successful_runs = 0
    skipped_runs = 0
    skip_reasons: List[str] = []
    transfer_fallback_runs = 0
    non_structured_graph_runs = 0

    for meta in meta_records:
        executed = bool(meta.get("run_executed", False))
        status = str(meta.get("execution_status", "")).strip()
        if executed and status in {"", "completed", "ready"}:
            successful_runs += 1
        elif not executed or status.startswith("skipped"):
            skipped_runs += 1
            reason = str(meta.get("execution_skip_reason", "")).strip()
            if reason:
                skip_reasons.append(reason)

        if str(meta.get("transfer_calibration_source", "")) != "measured":
            transfer_fallback_runs += 1
        if str(meta.get("graph_trace_source", "")) not in {"torch_fx", "torch_export_decoder_only"}:
            non_structured_graph_runs += 1

    if successful_runs == 0 and metrics_files_count > 0:
        successful_runs = min(metrics_files_count, max(run_dirs_count, 1))

    expected_repeats_eff = max(1, int(expected_repeats))
    profiling_completion_pct = min(100.0, (100.0 * successful_runs / expected_repeats_eff))

    stats_csv = _find_stats_csv(cfg_dir, model) if cfg_exists else None
    aggregation_ok = bool(stats_csv and stats_csv.exists())
    quality_flag_ok = False
    quality_flag_value = "missing"
    flagged_layers = 0
    if aggregation_ok:
        try:
            stats_df = pd.read_csv(stats_csv)
            if "quality_flag" in stats_df.columns:
                bad = stats_df[stats_df["quality_flag"].astype(str).str.lower() != "ok"]
                flagged_layers = int(len(bad))
                quality_flag_ok = flagged_layers == 0
                quality_flag_value = "ok" if quality_flag_ok else "flagged"
            else:
                quality_flag_value = "unknown"
                quality_flag_ok = False
        except Exception:
            quality_flag_value = "unreadable"

    ilp_summary_path = (cfg_dir / "ilp_solution" / "ilp_solution_summary.json") if cfg_exists else None
    ilp_summary_exists = bool(ilp_summary_path and ilp_summary_path.exists())
    ilp_status = "missing"
    ilp_ok = False
    if ilp_summary_exists:
        try:
            payload = json.loads(ilp_summary_path.read_text(encoding="utf-8"))
            ilp_status = str(payload.get("status", "unknown"))
            ilp_ok = ilp_status.lower() in {"optimal", "feasible"}
        except Exception:
            ilp_status = "unreadable"

    pareto_path = (cfg_dir / f"{model}_pareto_sweep.csv") if cfg_exists else None
    pareto_exists = bool(pareto_path and pareto_path.exists())
    pareto_feasible_rows = 0
    pareto_ok = False
    if pareto_exists:
        try:
            pareto_df = pd.read_csv(pareto_path)
            if "ilp_status" in pareto_df.columns:
                feasible = pareto_df[pareto_df["ilp_status"].astype(str).str.lower().isin(["optimal", "feasible"])]
                pareto_feasible_rows = int(len(feasible))
                pareto_ok = pareto_feasible_rows > 0
            else:
                pareto_ok = False
        except Exception:
            pareto_ok = False

    hybrid_protocol_files = list(cfg_dir.rglob("hybrid_execution_protocol.csv")) if cfg_exists else []
    hybrid_ok = len(hybrid_protocol_files) > 0

    # ==================== NEW: Separate operational viability from scientific validity ====================
    # Operational viability: did the pipeline execute successfully (with rescue allowed)?
    operational_status = "failed"
    if cfg_exists and successful_runs > 0:
        if successful_runs >= expected_repeats_eff:
            operational_status = "ok"
        else:
            operational_status = "partial"
    elif cfg_exists and successful_runs == 0:
        operational_status = "partial"  # Profiling partial or failed but may have rescue data

    # Mark if rescue/fallback were invoked
    rescue_influence_profiling = transfer_fallback_runs > 0 or non_structured_graph_runs > 0
    
    # Inferential status depends on quality of measured data (not on rescue itself)
    inferential_status = "exploratory"
    if aggregation_ok and quality_flag_ok:
        inferential_status = "strict"
    elif aggregation_ok:
        inferential_status = "qualified"

    # Doctoral operational readiness: executed (even with rescue) + ILP found solution + hybrid ran
    doctoral_operational_ready = all(
        [
            cfg_exists,
            successful_runs > 0,  # At least one replicate
            ilp_ok,
            pareto_ok,
            hybrid_ok if require_hybrid else True,
        ]
    )

    # Doctoral inferential strictness: operational ready + high-quality data + measured transfers + structured graph
    strict_transfer_measured = (len(meta_records) > 0 and transfer_fallback_runs == 0)
    strict_graph_trace_structured = (len(meta_records) > 0 and non_structured_graph_runs == 0)
    doctoral_inferential_strict = all(
        [
            doctoral_operational_ready,
            aggregation_ok,
            quality_flag_ok,
            strict_transfer_measured,
            strict_graph_trace_structured,
        ]
    )

    stage_scores = [
        profiling_completion_pct,
        100.0 if aggregation_ok else 0.0,
        100.0 if ilp_ok else 0.0,
        100.0 if pareto_ok else 0.0,
    ]
    if require_hybrid:
        stage_scores.append(100.0 if hybrid_ok else 0.0)
    completion_pct = round(sum(stage_scores) / len(stage_scores), 2)

    probable_failure_stage = "none"
    probable_cause = "none"

    if not cfg_exists:
        probable_failure_stage = "profiling"
        probable_cause = "Batch directory not created: combination not executed or failed early"
    elif successful_runs == 0:
        probable_failure_stage = "profiling"
        if skip_reasons:
            probable_cause = skip_reasons[0]
        elif key in log_failures and log_failures[key]:
            probable_cause = log_failures[key][0]
        else:
            probable_cause = "Profiling failure (OOM, unsupported precision, or hardware constraint)"
    elif successful_runs < expected_repeats_eff:
        probable_failure_stage = "profiling"
        probable_cause = "Partial replicate coverage: not all expected repetitions completed"
    elif not aggregation_ok:
        probable_failure_stage = "aggregation"
        probable_cause = "metrics_stats.csv does not exist: robust aggregation failed"
    elif not quality_flag_ok:
        probable_failure_stage = "quality"
        probable_cause = "quality_flag different from ok in metrics_stats.csv"
    elif not ilp_summary_exists:
        probable_failure_stage = "ilp"
        probable_cause = "ilp_solution_summary.json does not exist: ILP partitioning failed"
    elif not ilp_ok:
        probable_failure_stage = "ilp"
        probable_cause = f"ILP optimal/feasible status missing (status={ilp_status})"
    elif not pareto_exists:
        probable_failure_stage = "pareto"
        probable_cause = "Pareto sweep file does not exist"
    elif not pareto_ok:
        probable_failure_stage = "pareto"
        probable_cause = "Pareto generated but contains no feasible rows"
    elif require_hybrid and not hybrid_ok:
        probable_failure_stage = "hybrid"
        probable_cause = "hybrid_execution_protocol.csv does not exist"

    if probable_cause == "none":
        if transfer_fallback_runs > 0:
            probable_cause = "Transfer calibration fallback in at least one replicate"
        elif non_structured_graph_runs > 0:
            probable_cause = "Unstructured graph trace in at least one replicate"

    return {
        "model": model,
        "optimizer": optimizer,
        "precision": precision,
        "batch_size": batch_size,
        "config_dir": str(cfg_dir) if cfg_exists else "",
        "expected_repeats": expected_repeats_eff,
        "run_dirs_count": run_dirs_count,
        "metrics_files_count": metrics_files_count,
        "successful_runs": successful_runs,
        "skipped_runs": skipped_runs,
        "profiling_completion_pct": round(profiling_completion_pct, 2),
        "aggregation_ok": aggregation_ok,
        "stats_quality": quality_flag_value,
        "flagged_layers": flagged_layers,
        "ilp_status": ilp_status,
        "ilp_ok": ilp_ok,
        "pareto_exists": pareto_exists,
        "pareto_feasible_rows": pareto_feasible_rows,
        "pareto_ok": pareto_ok,
        "hybrid_ok": hybrid_ok,
        "completion_pct": completion_pct,
        "strict_transfer_measured": strict_transfer_measured,
        "strict_graph_trace_structured": strict_graph_trace_structured,
        "operational_status": operational_status,
        "inferential_status": inferential_status,
        "rescue_influence_profiling": rescue_influence_profiling,
        "doctoral_operational_ready": bool(doctoral_operational_ready),
        "doctoral_inferential_strict": bool(doctoral_inferential_strict),
        "probable_failure_stage": probable_failure_stage,
        "probable_cause": probable_cause,
        "profiling_stage_status": _bool_to_stage(successful_runs >= expected_repeats_eff, "ok", "incomplete"),
        "aggregation_stage_status": _bool_to_stage(aggregation_ok, "ok", "missing"),
        "ilp_stage_status": _bool_to_stage(ilp_ok, "ok", "failed_or_missing"),
        "pareto_stage_status": _bool_to_stage(pareto_ok, "ok", "failed_or_missing"),
        "hybrid_stage_status": _bool_to_stage(hybrid_ok, "ok", "missing"),
    }


def _group_completeness(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(col, dropna=False)
        .agg(
            n_configs=("completion_pct", "count"),
            avg_completion_pct=("completion_pct", "mean"),
            operational_ready_configs=("doctoral_operational_ready", "sum"),
            inferential_strict_configs=("doctoral_inferential_strict", "sum"),
        )
        .reset_index()
        .sort_values(by=[col], kind="stable")
    )
    grouped["avg_completion_pct"] = grouped["avg_completion_pct"].round(2)
    grouped["operational_ready_ratio_pct"] = (
        100.0 * grouped["operational_ready_configs"] / grouped["n_configs"]
    ).round(2)
    grouped["inferential_strict_ratio_pct"] = (
        100.0 * grouped["inferential_strict_configs"] / grouped["n_configs"]
    ).round(2)
    return grouped


def _table_to_markdown_or_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def _write_markdown_summary(
    out_path: Path,
    full_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    by_model_df: pd.DataFrame,
    by_optimizer_df: pd.DataFrame,
    by_precision_df: pd.DataFrame,
    by_batch_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Experimental Grid Coverage Audit")
    lines.append("")
    lines.append("## General Summary")
    lines.append(f"- Audited configurations: {len(full_df)}")
    lines.append(f"- Operationally ready configurations: {int(full_df['doctoral_operational_ready'].sum())}")
    lines.append(f"- Scientifically strict configurations: {int(full_df['doctoral_inferential_strict'].sum())}")
    lines.append(f"- Configurations with incidents: {len(failed_df)}")
    lines.append("")

    lines.append("## Completeness by Configuration (model/optimizer/precision/batch)")
    cols = [
        "model",
        "optimizer",
        "precision",
        "batch_size",
        "completion_pct",
        "profiling_completion_pct",
        "doctoral_operational_ready",
        "doctoral_inferential_strict",
        "operational_status",
        "inferential_status",
    ]
    disp_df = full_df[cols].copy().sort_values(by=["model", "optimizer", "precision", "batch_size"])
    disp_df["doctoral_operational_ready"] = disp_df["doctoral_operational_ready"].apply(
        lambda x: "✓ Ready" if x else "✗ Not Ready"
    )
    disp_df["doctoral_inferential_strict"] = disp_df["doctoral_inferential_strict"].apply(
        lambda x: "🟢 Ready" if x else "🔴 Incomplete"
    )
    disp_df["completion_pct"] = disp_df["completion_pct"].apply(
        lambda x: f"🟢 {x}%" if x == 100 else (f"🟡 {x}%" if x >= 50 else f"🔴 {x}%")
    )
    disp_df["profiling_completion_pct"] = disp_df["profiling_completion_pct"].apply(
        lambda x: f"🟢 {x}%" if x == 100 else (f"🟡 {x}%" if x >= 50 else f"🔴 {x}%")
    )
    lines.append(_table_to_markdown_or_text(disp_df))
    lines.append("")

    lines.append("## Failed Configurations and Probable Causes")
    if failed_df.empty:
        lines.append("🟢 No configurations with incidents detected.")
    else:
        cols = [
            "model",
            "optimizer",
            "precision",
            "batch_size",
            "probable_failure_stage",
            "probable_cause",
        ]
        f_df = failed_df[cols].copy()
        f_df["probable_failure_stage"] = f_df["probable_failure_stage"].apply(lambda x: f"⚠️ {x}")
        lines.append(_table_to_markdown_or_text(f_df))
    lines.append("")

    for title, df in [
        ("Completeness by Model", by_model_df),
        ("Completeness by Optimizer", by_optimizer_df),
        ("Completeness by Precision", by_precision_df),
        ("Completeness by Batch Size", by_batch_df),
    ]:
        lines.append(f"## {title}")
        g_df = df.copy()
        if "avg_completion_pct" in g_df.columns:
            g_df["avg_completion_pct"] = g_df["avg_completion_pct"].apply(
                lambda x: f"🟢 {x}%" if x == 100 else (f"🟡 {x}%" if x >= 50 else f"🔴 {x}%")
            )
        if "operational_ready_ratio_pct" in g_df.columns:
            g_df["operational_ready_ratio_pct"] = g_df["operational_ready_ratio_pct"].apply(
                lambda x: f"🟢 {x}%" if x == 100 else (f"🟡 {x}%" if x >= 50 else f"🔴 {x}%")
            )
        if "inferential_strict_ratio_pct" in g_df.columns:
            g_df["inferential_strict_ratio_pct"] = g_df["inferential_strict_ratio_pct"].apply(
                lambda x: f"🟢 {x}%" if x == 100 else (f"🟡 {x}%" if x >= 50 else f"🔴 {x}%")
            )
        lines.append(_table_to_markdown_or_text(g_df))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_expected_keys(
    observed_keys: Iterable[CONFIG_KEY],
    models: List[str],
    optimizers: List[str],
    precisions: List[str],
    batches: List[int],
) -> List[CONFIG_KEY]:
    if models and optimizers and precisions and batches:
        return list(itertools.product(models, optimizers, precisions, batches))
    return sorted(observed_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit grid completeness, failures and doctoral readiness")
    parser.add_argument("--input_root", required=True, help="Root directory containing model/optimizer/precision/batch_* tree")
    parser.add_argument("--output_dir", required=True, help="Directory where audit CSV/MD outputs will be saved")
    parser.add_argument("--expected_models_csv", default=None, help="Optional expected models CSV list")
    parser.add_argument("--expected_optimizers_csv", default=None, help="Optional expected optimizers CSV list")
    parser.add_argument("--expected_precisions_csv", default=None, help="Optional expected precisions CSV list")
    parser.add_argument("--expected_batches_csv", default=None, help="Optional expected batch sizes CSV list")
    parser.add_argument("--expected_repeats", type=int, default=1, help="Expected replicates per configuration")
    parser.add_argument("--require_hybrid", action="store_true", help="Require hybrid_execution_protocol.csv for doctoral readiness")
    parser.add_argument("--logs_root", default=None, help="Optional logs root to infer probable failure cause from experiments logs")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = _discover_batch_dirs(input_root)
    expected_models = _parse_csv_values(args.expected_models_csv)
    expected_optimizers = _parse_csv_values(args.expected_optimizers_csv)
    expected_precisions = _parse_csv_values(args.expected_precisions_csv)
    expected_batches = _parse_csv_values(args.expected_batches_csv, cast_int=True)

    keys = _build_expected_keys(
        observed_keys=observed.keys(),
        models=list(expected_models),
        optimizers=list(expected_optimizers),
        precisions=list(expected_precisions),
        batches=list(expected_batches),
    )
    if not keys:
        raise ValueError("No configurations were discovered and no expected grid was provided")

    log_failures = _load_log_failures(Path(args.logs_root) if args.logs_root else None)

    rows = [
        _compute_row(
            key=key,
            cfg_dir=observed.get(key),
            expected_repeats=args.expected_repeats,
            require_hybrid=args.require_hybrid,
            log_failures=log_failures,
        )
        for key in keys
    ]

    full_df = pd.DataFrame(rows).sort_values(by=["model", "optimizer", "precision", "batch_size"], kind="stable")
    full_df = full_df[~full_df["model"].astype(str).str.contains("mlp|simple", case=False)].copy()
    failed_df = full_df[~full_df["doctoral_operational_ready"]].copy()

    by_model_df = _group_completeness(full_df, "model")
    by_optimizer_df = _group_completeness(full_df, "optimizer")
    by_precision_df = _group_completeness(full_df, "precision")
    by_batch_df = _group_completeness(full_df, "batch_size")

    by_config_csv = output_dir / "grid_completeness_by_config.csv"
    failed_csv = output_dir / "grid_failed_configurations.csv"
    by_model_csv = output_dir / "grid_completeness_by_model.csv"
    by_optimizer_csv = output_dir / "grid_completeness_by_optimizer.csv"
    by_precision_csv = output_dir / "grid_completeness_by_precision.csv"
    by_batch_csv = output_dir / "grid_completeness_by_batch.csv"
    summary_md = output_dir / "GRID_AUDIT_SUMMARY.md"

    full_df.to_csv(by_config_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)
    by_model_df.to_csv(by_model_csv, index=False)
    by_optimizer_df.to_csv(by_optimizer_csv, index=False)
    by_precision_df.to_csv(by_precision_csv, index=False)
    by_batch_df.to_csv(by_batch_csv, index=False)
    _write_markdown_summary(
        out_path=summary_md,
        full_df=full_df,
        failed_df=failed_df,
        by_model_df=by_model_df,
        by_optimizer_df=by_optimizer_df,
        by_precision_df=by_precision_df,
        by_batch_df=by_batch_df,
    )

    print("=" * 80)
    print("GRID AUDIT GENERATED")
    print("=" * 80)
    print(f"Configurations audited: {len(full_df)}")
    print(f"Operationally ready: {int(full_df['doctoral_operational_ready'].sum())}")
    print(f"Scientifically strict: {int(full_df['doctoral_inferential_strict'].sum())}")
    print(f"Configurations with incidents: {len(failed_df)}")
    print(f"By-config CSV: {by_config_csv}")
    print(f"Failed-config CSV: {failed_csv}")
    print(f"Summary MD: {summary_md}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())