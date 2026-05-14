# ILP Results Summary

## Inputs
- Pareto rows: 24
- Models: resnet50, simple_mlp

## Best Feasible Row Per Model
```text
     model  gpu_budget_mb  ilp_objective  greedy_objective  ilp_gpu_mem_mb  ilp_cpu_mem_mb  ilp_layers_gpu  ilp_layers_cpu  ilp_cut_edges  all_cpu_objective all_gpu_status  improvement_vs_all_cpu_pct  improvement_vs_greedy_pct
  resnet50         2000.0    2230.172923       2176.662097     1968.032227     1473.248535               0             126              0        2344.673882     infeasible                    4.883449                  -2.458389
simple_mlp          500.0       1.261982          1.303656      141.448242        0.064941               3               2              1          27.388898       feasible                   95.392361                   3.196744
```
