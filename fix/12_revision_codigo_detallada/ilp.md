# Revisión `src/ilp/*` + `src/runtime/{device_plan,plan_representation}.py` (+ llamador `validation/run_ilp_partition.py`)

Solo lectura. Números de línea de `solve.py` verificados con grep; el resto son aproximados (±5).

---
## 0. BUG (a): `ilp_assignment.csv` 100 % CPU con `gpu_mem≈5585` y `objective≈100 ms` — CAUSA RAÍZ CONFIRMADA

**No está en `src/ilp/` ni en el exportador ni en el sentido de x.** Es post-proceso en `validation/run_ilp_partition.py`.

Evidencia (log `logs/zephyr/thesis_mode/thesis_mode_20261014_023618.txt`, gpt2_small batch_32, ratio 0.90, trial budget 6632.1):
```
[topology_memory] real peak: gpu=5585.2MB ... flips_applied=0 feasible=True
[real_overhead] WARNING: the final plan ... exceeds the real measured peak (NoneMB vs budget 6632.1MB, status=skipped) ... Trimming further...
[real_overhead] Final plan validated: real_allocated_mb=None real_reserved_mb=None, removed 113 additional layer(s)
[real_overhead] WARNING: final plan still exceeds ... even after removing every GPU-resident layer
Objective: 100.120635 / GPU mem used (MB): 5585.165
```
y `ilp_solution_pareto_best_budget_6632p1/ilp_solution_summary.json`: `layers_gpu_forward=0`, `gpu_mem_used_mb=5585.17`, `cpu_mem_used_mb=0.0`, `objective=100.12`. El directorio base `ilp_solution/` (sin calibración) sí sale 113/113 GPU. Mismo patrón en distilgpt2 batch_16/64 (`pareto_best_budget_*`: layers_gpu_forward=0, objective idéntico al del plan all-GPU 100.41).

Cadena causal:
1. `run_ilp_partition.py` ~L905-935 `_validate_final_plan_against_real_overhead` se ejecuta SIEMPRE que haya `calib_model` (hay `--calibration_batch_size` en el path pareto/safety-ratio).
2. `measure_real_peak_for_plan` (`src/runner/hybrid_overhead_calibration.py:304`) → `_run_real_plan_pass` (~L271) devuelve `{"status":"skipped","reason":"symbolic_trace unsupported for this model"}` para modelos HF decoder-only (GPT-2: `torch.fx.symbolic_trace` falla; no trae `real_reserved_mb`).
3. En `_validate_final_plan_against_real_overhead` (~L300-330): `feasible = status=="ok" and reserved is not None and reserved<=budget` → **False** para "skipped" (se confunde "no medible" con "no cabe").
4. Entra al bucle de recorte: `removal_order` = todas las capas GPU; `_measure(_plan_removing(n))` también "skipped" → rama `else: hi = n` ("degenerate case, use it anyway") → **mueve las 113 capas a CPU** (`flips_removed=113`) y reescribe `sol.forward/backward_assignment/assignment`, cut_edges, cross_phase.
5. `real_allocated_mb is None` → `sol.gpu_mem_used_mb` NO se actualiza (queda 5585 del paso topology_memory) y `sol.objective_value`/`activation_strategies` nunca se recalculan (siguen siendo los del all-GPU; las 113 capas CPU conservan 'checkpoint'/'recompute'). Por eso el resumen es incoherente: objective≈100 ms (all-GPU), gpu_mem 5585, pero CSV 0 GPU. `cpu_mem_used_mb=0` viene de topology_memory previo.
6. Consecuencia: el "Pareto-best" ejecutado en hybrid runtime es en realidad all-CPU; la sonda no OOMa y la búsqueda de safety-ratio "tiene éxito" con un plan inválido (resultados de tiempo/energía de la tesis para decoder-only contaminados).

Arreglo (en `run_ilp_partition.py`):
- Tratar `status != "ok"` como "no validado": `return None` (o `{"validated": False}`) y NO tocar el plan; solo recortar cuando `status=="ok"` y `reserved>budget`. Si `status=="error"` por OOM → sí recortar.
- Eliminar la rama `hi = n` (degenerate) o hacerla fallar duro (`raise`) en lugar de vaciar la GPU.
- Tras cualquier modificación del plan, recalcular `objective_value` (re-evaluar con `_eval_assignment_dual`/phase4), `gpu_mem_used_mb`, `cpu_mem_used_mb`, `activation_strategies` (poner 'retain' en capas movidas a CPU) y avisar si divergen del solver. Añadir aserción: `layers_gpu_forward>0` si el solver reportó gpu_mem>0.
- Medir con `torch.export`/hooks en vez de `symbolic_trace` para HF (o usar el `hybrid_executor` real con el modelo traced por `torch_export_decoder_only`).
- El mismo patrón "skipped ⇒ infeasible" existe en `_validate_gpu_expansion_against_real_overhead` (~L190-230): ahí solo avisa (inocuo), pero imprime mensaje engañoso.

---
## 1. MAPA POR ARCHIVO

### 1.1 `src/ilp/solve.py` (1461 l.)
| Símbolo | Línea | Qué hace / E→S | Llamado por |
|---|---|---|---|
| `_interpolate_recompute_curve` | 13 | Interpola puntos (k bloques→ms) a curva entera | `_solve_with_pulp_dual` |
| `_block_id_for_layer_name` | 38 | Deriva id de bloque desde nombre de capa (regex) | `_group_nodes_into_activation_candidates`, MILP |
| `_group_nodes_into_activation_candidates` | 58 | Agrupa nodos en bloques candidatos a recompute (solo bloques 100 % GPU) | MILP dual |
| `_sanitize_lp_name` | 89 | Nombre válido LP | MILP |
| `ILPSolution` | 101 | Dataclass resultado (assignment, forward/backward_assignment, cut_edges, backward_cut_edges, cross_phase_edges, objective, gpu/cpu_mem_used, status, backend) | todo |
| `_build_cbc_solver` | 116 | COIN_CMD CBC, `timeLimit=300, gapRel=0.01` | `_solve_with_pulp*` |
| `_eval_assignment` / `_solve_exhaustive` | 133 / 158 | Fase única (1 x por nodo), ≤22 nodos | `solve_partition_ilp(backend=exhaustive)` |
| `_solve_with_pulp` | 210 | MILP single-phase **DEPRECADO** (muerto) | nadie salvo rutas antiguas |
| `_derive_congestion_knee_ms` | 279 | Auto-knee = suma 1/3 aristas más baratas | MILP dual, exhaustive_dual |
| `_eval_assignment_dual` / `_solve_exhaustive_dual` | 293 / 364 | Enumera 4^n (≤14 nodos) | fallback de `solve_partition_ilp` |
| `_build_tight_topological_gpu_constraints` | 440 | Restricciones GPU por punto crítico (orden topológico real, dedup pesos compartidos) | MILP dual si `memory_model=="topological_tight"` |
| `_solve_with_pulp_dual` | 555-~950 | MILP principal fwd/bwd (formulación §2) | `solve_partition_ilp`, `_solve_phase4_joint` |
| `solve_partition_ilp` | 952 | Dispatcher: `auto`→pulp, `except Exception` → exhaustive_dual | run_ilp_partition, sweeps |
| `refine_solution_hierarchical_local` | 977 | Búsqueda local (≤budget flips) sobre una solución | run_ilp_partition (`--local_refine_budget`) |
| `ILPSolution4` | 1080 | Extiende con `activation_strategies`, `mode` | fase 4 |
| `_solve_phase4_greedy` | 1086 | Heurística: parte de all-GPU, expulsa/estrategia por nodo hasta caber | `solve_partition_ilp_phase4` |
| `_solve_phase4_exhaustive` | 1255 | Enumeración (≤18 nodos, espacio ≤2 000 000) | idem |
| `_solve_phase4_joint` | 1392 | Llama `_solve_with_pulp_dual` con recompute/checkpoint activados y empaqueta `ILPSolution4` (acepta "optimal"/"feasible") | idem |
| `solve_partition_ilp_phase4` | 1427 | Dispatcher: auto = exhaustive si ≤14 nodos y espacio ≤250 000, si no greedy; `backend` explícito `joint_phase4` | run_ilp_partition, tests |

**Flujo `solve_partition_ilp`**: `validate_ilp_config` → `build_problem_data_dual` → (auto) intenta `_solve_with_pulp_dual`; cualquier excepción se imprime y cae a `_solve_exhaustive_dual` (>14 nodos ⇒ RuntimeError que enmascara la causa original de PuLP/CBC).

**Flujo `solve_partition_ilp_phase4`**: valida strategies → `build_problem_data_phase4` (exige activation/io "provided") → según backend: *greedy* (inicia all-GPU; mientras gpu_mem > presupuesto elige nodo/estrategia con menor delta de coste ponderado por MB liberado; descarta solución si `status != "optimal"`; devuelve `forward_assignment=None` en baseline fallback), *exhaustive* (enumera {GPU,CPU}×{retain,recompute,checkpoint} por nodo, escoge min objetivo factible), *joint* (MILP dual con variables de recompute/checkpoint, curva calibrada opcional `_solve_with_calibrated_recompute_curve` en el llamador).

### 1.2 `model_builder.py`
- `ILPConfig` L14 (w_time=1, w_energy, w_transfer=1, w_fragmentation, w_congestion, `peak_activation_overlap=0.35` [validado, **no usado**], budgets 1e18, `memory_model="topological"`); `ILPConfig4` L30 (w_recompute_penalty 0.5, w_checkpoint_penalty 0.3 [no usado por solve], target_gpu_utilization 0.95, enable_joint…, recompute_block_order, curve_points).
- `validate_ilp_config` L153; `_build_effective_memory` L188 (act_gpu = max(fwd_peak,bwd_peak); fallback `gpu_base*0.70`; cpu_act = cpu_base·ratio); `build_problem_data` L256; `build_problem_data_dual` L284 (cross-phase = obj_fwd_dev + w_transfer·time_io + w_fragmentation; edge_cut_fwd==edge_cut_bwd); `ILPProblemData4` L348; `build_problem_data_phase4` L358 (recompute_cost = w_recompute_penalty·fwd_ms; `/3.0` fallback; checkpoint_cost_cpu==gpu).

### 1.3 `data_loader.py`
`ILPInputData` L22 (+`__post_init__` heurísticos: split 0.5/0.5, energía 0, activation 0.70·gpu_mem, io 0.15·fwd, energy_io 0.05 J); `_weighted_mean/_std/_aggregate_values` L148-190; `_collapse_graph_to_measured_edges` L209 (BFS, max(transfer) en el camino, no suma); `load_measured_graph_artifacts` L278; `_robust_value` L305 (mean+k·σ, k=1); `load_ilp_inputs` L356 (fallback analítico tflops=1, tdp=250 W, bw=900 GB/s L528-542); `merge_ilp_inputs_multi_hardware` L860 (no propaga activation/io/prepared_topology). Llamadores: run_ilp_partition, sweeps, hierarchical pipeline.

### 1.4 `topology_memory.py`
`_topological_order` L63 (Kahn + ruptura de ciclos), `PreparedTopologyMemory` L131 (`prepare`, `evaluate`, `layer_gpu_weight`), `correct_solution_memory` L375 (repair GPU→CPU por peso desc., flips fwd+bwd, `repair_budget=200`), `expand_solution_memory` L426, `rebalance_solution_memory` L514, `simulate_assignment_peak_memory` L565.

### 1.5 `advanced_terms.py`, `backward_meta_model.py`, `export_solution.py`, `__init__.py`
- `ActivationStrategy`, `estimate_activation_metadata` (0.70/0.15/0.05), `compute_effective_costs_phase4`, `validate_strategies_phase4`, `estimate_memory_savings_phase4` (inconsistente con compute_effective_costs: checkpoint=0 ahorro GPU) → mayormente muertos fuera de tests.
- `backward_meta_model.py`: ridge (λ=1e-6, val 0.25, seed 42), `apply_backward_meta_model` pierde peaks/shared_call_site/prepared_topology; run_ilp_partition lo ignora ("deprecado") → código muerto.
- `save_ilp_solution`: exporta fielmente fwd/bwd del objeto recibido (CSV assignment: layer, device, device_forward, device_backward, activation_strategy; cut edges incl. cross-phase como (n,n); summary JSON). **No invierte x**; usa `assignment` si `forward_assignment` es None.

### 1.6 `src/runtime/device_plan.py`, `plan_representation.py`
- `DevicePlan.resolve_torch_device`: cae silenciosamente a CPU si no hay CUDA o si la capa no está en el plan (default "CPU"). `validate_plan_coverage`: solo warning para capas sin asignar. Import duplicado `typing`.
- `infer_ilp_input_paths`: prioriza `run_*` y busca en hermanos `batch_*` (fallback silencioso a artefactos de otro batch). `load_execution_plan`: `backward_col` default = forward_col. `load_graph_edges`: orienta por primer id y descarta aristas "hacia atrás" (grafo distinto del ILP). `load_transfer_costs`.

---
## 2. FORMULACIÓN MATEMÁTICA (`_solve_with_pulp_dual`)

Conjuntos: nodos N, aristas E, bloques B. Variables binarias: x^f_n, x^b_n (1=GPU) ; y^f_e, y^b_e (corte por fase; linealizados con 4 desigualdades: y ≥ x_i−x_j, y ≥ x_j−x_i, y ≤ x_i+x_j, y ≤ 2−x_i−x_j); c^{cg}_n, c^{gc}_n (cruce de fase CPU→GPU / GPU→CPU, direccional); z^v_n = x^f_n ∨ x^b_n; z^c_n = (1−x^f_n) ∨ (1−x^b_n); continuas o^f, o^b ≥ 0 (congestión sobre knee). Opcionales: r_n (recompute), k_n (checkpoint), r_n+k_n ≤ x^b_n; s_b (bloque recompute, prefijo monótono), is_k (curva calibrada, Σis_k=1, Σ s_b = Σ k·is_k).

Objetivo (suma serial):
min Σ_n [t^f_G x^f + t^f_C(1−x^f) + t^b_G x^b + t^b_C(1−x^b)] + Σ_e (cut^f_e y^f_e + cut^b_e y^b_e) + Σ_n (cross_{c→g} c^{cg} + cross_{g→c} c^{gc}) + w_cong (o^f+o^b) [+ w_rec·t^f_G·r_n] [+ w_io·t_io·k_n], donde t = w_time·ms + w_energy·J (sin normalizar).

Memoria GPU ("topological"): Σ gpu_base_n·z^v_n + Σ gpu_act_n·(x^b_n − r_n − k_n) [+Δcurva] ≤ M_gpu. CPU: Σ cpu_base·z^c + Σ cpu_act·(1−x^b) + spill ≤ M_cpu. "topological_tight": una restricción GPU por punto crítico (fwd y bwd) con dedup de pesos compartidos; CPU igual que topological (sin pico real). Exige x^f=x^b, sin recompute/checkpoint.

---
## 3. HALLAZGOS

### CRÍTICO
**C1** `run_ilp_partition.py` ~L300-345 (`_validate_final_plan_against_real_overhead`) — ver §0: "skipped" tratado como infeasible, vacía la GPU, no recalcula objetivo/mem. Arreglo en §0.
**C2** `run_ilp_partition.py` ~L905-960 — tras `correct_solution_memory`/validaciones, `objective_value`, `activation_strategies`, y `gpu_mem_used_mb` (si real is None) no se recalculan ⇒ summary incoherente con el CSV. Recalcular y registrar `post_process_flips`.
**C3** `solve.py:952-967` (`solve_partition_ilp`) — `except Exception` en backend auto traga errores de PuLP/CBC (CBC ausente, infeasible, memory_model no soportado) y cae a exhaustive_dual, que con >14 nodos lanza RuntimeError sin la causa raíz. Propagar con `raise ... from exc`, o reintentar solo ante `PulpSolverError`.
**C4** `solve.py:130` — `timeLimit=300, gapRel=0.01` hardcodeados; `_solve_with_pulp_dual` solo mapea status de PuLP: una solución por timeout se reporta "optimal"/"feasible" sin gap; y greedy/exhaustive rechazan "feasible" mientras joint lo acepta. Exponer `--solver_time_limit_s/--mip_gap`, registrar gap real y `status="feasible_timeout"`.

### ALTO
**A1** `model_builder.py:188-255` — `act_gpu = max(fwd_peak, bwd_peak)` es el pico transitorio medido, y la restricción "topological" lo SUMA sobre todos los nodos retenidos: el modelo sobreestima mucho (gpu_mem solver 6578 vs 5585 real; 15344 vs 9739; 38098 vs 21350 en logs) ⇒ empuja capas a CPU innecesariamente (contradice "priorizar GPU"). Usar activación retenida (no pico) + pico transitorio máx (max, no suma): `Σ retenida + max_n transiente`, o usar `topological_tight` por defecto.
**A2** `topology_memory.py` `evaluate` — supone grads==params, un solo stream secuencial, no modela prefetch/doble buffer de transferencias en cortes, optimizer states (AdamW: 2× params) ni overhead de contexto CUDA/allocator ⇒ NO conservador para GPU (el propio README fix/08 lo reconoce) mientras el MILP es muy conservador: los dos modelos de memoria divergen ~1.2–1.8×. Unificar y añadir término de estados del optimizador + factor medido de fragmentación (hybrid_overhead_ratio de la calibración, no hardcodeado).
**A3** `model_builder.py`/`solve.py` — `memory_model="topological"` en CPU no incluye pico ni overhead del framework; `topological_tight` no restringe CPU por pico.
**A4** `solve.py` objetivo — suma serial de fwd+bwd+cortes: no es makespan; no modela solapamiento CPU/GPU ni pipelining de transferencias (async). Si el runtime solapa (hybrid_executor prefetch), el ILP sobreestima cortes; documentar o añadir variable T ≥ camino crítico.
**A5** Unidades — objetivo mezcla `w_time·ms + w_energy·J + w_transfer·ms + w_fragmentation (adimensional) + w_congestion·ms` sin normalizar; ms y J no son comparables (ponderación depende de escala del modelo). Normalizar por baseline (all-GPU time/energy) y reportar objetivo en ms puros en el summary (`time_ms`, `energy_j` por separado). Además `objective≈100` en summary es "ponderado", no tiempo.
**A6** `data_loader.py` — grafo por nombre de capa genera ciclos espurios en decoder-only (lm_head↔wte por pesos atados); `_topological_order` los rompe arbitrariamente ⇒ orden de picos incorrecto. Marcar aristas de pesos compartidos como no-dataflow.
**A7** `data_loader.merge_ilp_inputs_multi_hardware` (~L860) — no propaga `node_mem_activation_mb/time_io/energy_io/prepared_topology` ⇒ Phase4 con datos "heuristic_default" o falla. Propagar con `_aggregate_values`.
**A8** `_solve_phase4_greedy` (solve.py 1086-1255) — decide coste GPU/CPU del nodo con forward_assignment ignorando backward; suma `w_io` dos veces (dentro de checkpoint_cost y como `w_io*delta`); recompute suma `w_io·delta_time` sin sentido; `gpu_mem_used - act` ignora cross-phase. Resultados greedy no comparables con joint.
**A9** `build_problem_data_phase4` — `recompute_cost = w_recompute_penalty·fwd_ms` (0.5 por defecto) REDUCE el coste real de recompute (debería ser ≥1·fwd_ms); sesga hacia recompute. `w_checkpoint_penalty` no se usa. `checkpoint_cost_cpu==gpu`.

### MEDIO
**M1** `solve.py` `_derive_congestion_knee_ms` L279 — knee automático = suma del tercio más barato de aristas (1/3 mágico, sin base física); `w_congestion` default sin calibrar.
**M2** `solve.py:847` — `solver.msg = True` hardcodeado: inunda logs (decenas de miles de líneas Cbc en logs). Parametrizar `--solver_verbose`.
**M3** `_solve_with_pulp_dual` — `_build_tight_topological_gpu_constraints` L440 requiere `prepared_topology`; si falta lanza ValueError → capturado por C3.
**M4** `load_ilp_inputs` L528-560 — fallback analítico silencioso (tflops=1.0, tdp=250, bw=900); no falla ni marca `ilp_input_quality="analytical"` salvo strict. Todos tiempos GPU=0 → fuerza fallback.
**M5** `data_loader` — transferencias faltantes → 0.0 con warning (solo strict falla): cortes "gratis" ⇒ el ILP corta en cualquier lado.
**M6** `data_loader._collapse_graph_to_measured_edges` — coste de arista colapsada = max(transfer) en el camino, no el tamaño del tensor que realmente cruza; subestima/ sobreestima.
**M7** `ILPSolution4` redefine `activation_strategies`/`mode` — drift de dataclass; `export` usa `getattr`.
**M8** `plan_representation.infer_ilp_input_paths` — fallback a artefactos de otro `batch_*` silencioso (mezcla datos de distinto batch).
**M9** `plan_representation.load_graph_edges` descarta aristas "hacia atrás" ⇒ grafo de ejecución ≠ grafo de ILP (cuts distintos).
**M10** `DevicePlan.resolve_torch_device` — default "CPU" silencioso para capas no listadas y caída a CPU sin CUDA (ya hay flag `--allow_cpu_fallback`, pero este default lo ignora). Falla ruidosa en modo estricto.
**M11** `solve_partition_ilp_phase4` auto — exhaustive solo ≤14 nodos y espacio ≤250 000; para modelos reales (59–113 nodos) siempre greedy salvo `backend=joint_phase4` explícito; greedy ≠ óptimo. Documentar/forzar joint por defecto.
**M12** `topology_memory.correct_solution_memory` — repair voltea fwd Y bwd, descarta cualquier recompute/checkpoint de esas capas sin actualizar el objetivo; `repair_budget=200` mágico (con >200 capas no converge).

### BAJO
**B1** Código muerto: `_solve_with_pulp` (solve.py:210), `peak_activation_overlap` (validado, no usado), `w_checkpoint_penalty`, `estimate_memory_savings_phase4`, `backward_meta_model.py` completo (+ `apply_backward_meta_model`), `compute_effective_costs_phase4` fuera de tests.
**B2** `device_plan.py` import `typing` duplicado.
**B3** Strings "GPU"/"CPU" literales en todo el código (sin enum).
**B4** `graph_trace_source == "torch_export_decoder_only"` en solve.py (nombre hardcodeado de modelo/trazador) y `allowed_graph_trace_sources` en data_loader.
**B5** `topology_memory._safe_float` default 0.0 silencioso (columnas ausentes ⇒ memoria 0 ⇒ plan "factible" falso); nodos sin stats se omiten sin aviso.
**B6** Excepciones tragadas: `solve_partition_ilp` (C3), `_resolve_hybrid_overhead_ratio` (`except (OSError, ValueError): continue`, silencioso ⇒ ratio=1.0), `measure_real_peak_for_plan` (`except Exception` → "error"), `_load_calibration_model` (except Exception → None), `hybrid_overhead_calibration finally: except Exception: pass`.

---
## 4. HARDCODEADOS (valor → file:línea → fuente propuesta)
| Valor | Ubicación | Debería venir de |
|---|---|---|
| timeLimit=300 s, gapRel=0.01 | solve.py:130 | CLI `--solver_time_limit_s`, `--mip_gap` |
| solver.msg=True | solve.py:847 | CLI `--solver_verbose` |
| límites exhaustive 22 / 14 / 18, espacio 2 000 000, auto 14 / 250 000 | solve.py ~158, 364, 1255, 1427 | config `ILPConfig`/CLI |
| tol 1e-9, 1e-6 (score), `//3`, knee=1/3 | solve.py 279-293, greedy | constantes con nombre + knee medido de la curva de congestión |
| `w_recompute_penalty` getattr 0.5, `w_checkpoint_penalty` 0.3 | model_builder.py:30-45 | CLI + calibración (≥1.0 en coste real) |
| `peak_activation_overlap` 0.35 | model_builder.py:~20 | medición (fwd/bwd peaks) o eliminar |
| activation = 0.70·gpu_mem, io=0.15·fwd, energy_io 0.05 J, split 0.5/0.5, `/3.0` recompute | model_builder.py ~200, 370; data_loader `__post_init__`; advanced_terms | medición por capa (profiling); marcar `source="heuristic"` y abortar en modo thesis |
| budgets 1e18, target_gpu_utilization 0.95 | model_builder.py:23-24,35 | `detect_gpu_vram_mb` × safety ratio (ya en scripts) |
| tflops=1.0, tdp_w=250, mem_bw=900 GB/s | data_loader.py:528-542 | `reference_meta` medido (`measured_gpu_peak_tflops`, `nvidia-smi`) |
| k_sigma=1.0, clip 1e-6 | data_loader.py ~305 | CLI `--k_sigma` (existe) / constante |
| **Ancho de banda PCIe**: no hardcodeado en src/ilp; viene de `*_transfer_edges.csv` (medido). Riesgo: fallback 0.0 ms cuando falta (M5) | data_loader | medición por transferencia; error si falta |
| repair_budget=200, expand_budget=200 | topology_memory.py:382,437,525-526 | `len(layers)` o CLI (ya CLI en runner: `--topology_memory_repair_budget`) |
| grads==params, factor optimizer states ausente | topology_memory evaluate | optimizador real (AdamW 2×, SGD 0/1×) |
| ridge λ=1e-6, val 0.25, seed 42 | backward_meta_model.py | CLI (módulo muerto) |
| `input_size=224` en calibración | run_ilp_partition.py ~L100 | `--calibration_input_size` (rompe modelos no 224) |
| bf16/float32 mapping | run_ilp_partition.py ~L103 | config de precisión (soportar fp16) |
| Safety ratios 0.90,0.85; CPU 0.9; VRAM 1.0 | scripts/run_thesis_mode.sh:205,310,332 | Medición de calibración (`hybrid_overhead_ratio`), no escalones fijos |

---
## 5. RESUMEN (c)-(f)
- (c) Unidades: objetivo ponderado mezcla ms/J/adimensional; summary `objective_value` no es tiempo. Report time_ms y energy_j por separado.
- (d) Memoria: MILP sobre-conservador (suma de picos), topology_memory no conservador (sin optimizer states/prefetch); desacuerdo explícito en logs (solver 5190 vs real 3128 MB; 15344 vs 9739). Presupuesto CPU sin overhead.
- (e) Makespan: no; suma serial.
- (f) Ver B1, B6, C3, M10.
