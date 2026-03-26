# Thesis Mode Protocol Checklist

This run executed the project-wide thesis mode orchestration.

## Generated roots
- Base output: data/zephyr/test-audit-thesis-mode
- Reports: reports/ilp_results/test-audit-thesis-mode
- LaTeX: reports/ilp_results/test-audit-thesis-mode/latex
- Log: logs/thesis_mode_20260326_112907.txt

## Automatically covered by script
- Profiling campaign grid
- Replicate aggregation
- ILP partition and Pareto sweep per config
- Consolidated report assets and LaTeX tables
- Optional hybrid runtime traces (FX and export-backed DAG paths when supported)

## Must still be validated explicitly (doctoral criteria)
- Final model quality metrics (accuracy/loss/AUC) vs baselines in target tasks
- Statistical significance and effect size for claimed gains
- Multi-hardware robustness matrix and threats-to-validity analysis

