# ILP Results Summary

## Inputs
- Pareto rows: 24
- Models: resnet50, simple_mlp

## Best Feasible Row Per Model
```text
     model  gpu_budget_mb  ilp_objective  greedy_objective  ilp_gpu_mem_mb  ilp_cpu_mem_mb  ilp_layers_gpu  ilp_layers_cpu  ilp_cut_edges  all_cpu_objective all_gpu_status  improvement_vs_all_cpu_pct  improvement_vs_greedy_pct
  resnet50         2000.0     720.402938        704.035102     1986.173828     1344.623535               2             124              4         821.457066     infeasible                   12.301815                  -2.324861
simple_mlp          500.0       0.748324          1.218835      234.333008        0.000000               5               0              0           0.910162       feasible                   17.781165                  38.603324
```
