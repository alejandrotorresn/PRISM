# ILP Ablation Summary

## Inputs
- Rows: 20
- Models: simple_mlp
- Variants: full_model, no_robustification, no_topology, no_transfer_edges

```text
     model            variant  gpu_budget_mb  ilp_objective  delta_vs_full_obj  ilp_cut_edges  ilp_layers_gpu  ilp_layers_cpu
simple_mlp         full_model           64.0       1.810442           0.000000              1               3               2
simple_mlp no_robustification           64.0       1.310637          -0.499805              1               3               2
simple_mlp        no_topology           64.0       1.372013          -0.438429              0               2               3
simple_mlp  no_transfer_edges           64.0       1.372013          -0.438429              3               2               3
```