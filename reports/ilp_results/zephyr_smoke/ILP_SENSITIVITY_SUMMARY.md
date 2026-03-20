# ILP Sensitivity Analysis Report

## Overview
- Rows: 55
- Models: simple_mlp
- Parameters swept: k_sigma, w_transfer

## Parameter: `k_sigma`
```text
     model  param_value  gpu_budget_mb  ilp_objective  baseline_objective  delta_abs  delta_pct  ilp_cut_edges  ilp_layers_gpu  ilp_layers_cpu
simple_mlp          0.0           64.0       1.310637            1.810442  -0.499805   -27.6068              1               3               2
simple_mlp          0.5           64.0       1.560539            1.810442  -0.249902   -13.8034              1               3               2
simple_mlp          1.0           64.0       1.810442            1.810442   0.000000     0.0000              1               3               2
simple_mlp          1.5           64.0       2.060344            1.810442   0.249902    13.8034              1               3               2
simple_mlp          2.0           64.0       2.310246            1.810442   0.499805    27.6068              1               3               2
```

## Parameter: `w_transfer`
```text
     model  param_value  gpu_budget_mb  ilp_objective  baseline_objective  delta_abs  delta_pct  ilp_cut_edges  ilp_layers_gpu  ilp_layers_cpu
simple_mlp          0.0           64.0       1.372013            1.810442  -0.438429   -24.2167              3               2               3
simple_mlp          0.5           64.0       1.700111            1.810442  -0.110331    -6.0941              1               3               2
simple_mlp          1.0           64.0       1.810442            1.810442   0.000000     0.0000              1               3               2
simple_mlp          2.0           64.0       2.031104            1.810442   0.220662    12.1883              1               3               2
simple_mlp          5.0           64.0       2.693090            1.810442   0.882648    48.7532              1               3               2
```
