# ILP Results Summary

## Inputs
- Pareto rows: 8
- Models: resnet50, vit_b16

## Best Feasible Row Per Model
```text
   model  gpu_budget_mb  ilp_objective  ilp_gpu_mem_mb  ilp_cpu_mem_mb  ilp_layers_gpu  ilp_layers_cpu  ilp_cut_edges  all_cpu_objective all_gpu_status  improvement_vs_all_cpu_pct
resnet50         1100.0    2101.867638     1063.369629      681.499268               5             121              9        2491.547138     infeasible                   15.640061
 vit_b16         1500.0     418.485696     1483.362305      454.808228               4              96              3         507.397185     infeasible                   17.523055
```
