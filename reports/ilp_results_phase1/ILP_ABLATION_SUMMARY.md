# ILP Ablation Summary

## Inputs
- Rows: 20
- Models: simple_mlp
- Variants: full_model, no_robustification, no_topology, no_transfer_edges

```text
     model            variant  gpu_budget_mb  ilp_objective  delta_vs_full_obj  ilp_cut_edges  ilp_layers_gpu  ilp_layers_cpu
simple_mlp         full_model           16.0            0.0                0.0              0               0               5
simple_mlp no_robustification           16.0            0.0                0.0              0               0               5
simple_mlp        no_topology           16.0            0.0                0.0              0               0               5
simple_mlp  no_transfer_edges           16.0            0.0                0.0              0               0               5
```