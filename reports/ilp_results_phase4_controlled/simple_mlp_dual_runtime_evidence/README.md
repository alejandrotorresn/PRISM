# Evidencia controlada de Fase 4

Este directorio contiene un caso dual controlado para `simple_mlp` sobre el dataset real de smoke `data/zephyr/results_smoke/simple_mlp/SGD/fp32/batch_8`.

El objetivo metodologico del artefacto es forzar una separacion materializable entre forward y backward para verificar que el runtime no solo preserva el plan dual, sino que ejecuta efectivamente el backward en un dispositivo distinto cuando el plan asi lo exige.

Plan aplicado:
- `net.0`: `device_forward = GPU`, `device_backward = CPU`
- `net.1` a `net.4`: `CPU` en ambas fases
- corte `forward`: `net.0 -> net.1`
- corte `cross_phase`: `net.0 -> net.0`

Artefactos relevantes:
- `ilp_assignment.csv`: asignacion dual controlada
- `ilp_cut_edges.csv`: cortes forward y entre fases
- `simulation/simulation_summary.json`: validacion topologica y simulacion del plan
- `runtime/hybrid_execution_summary.json`: ejecucion fisica observada
- `runtime/hybrid_execution_steps.csv`: trazas por paso

Resultado clave esperado y verificado:
- `backward_relocation_layers = ["net.0"]`
- `backward_relocation_count = 1`
