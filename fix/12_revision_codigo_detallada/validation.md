# Revisión de `validation/` y `convergence_validation/` (PRISM)

Alcance: lectura completa de los 22 `.py` de `validation/` y de `convergence_validation/` (`prism_loop.py`, `all_gpu_loop.py`, `run_convergence_study.py/.sh`, `datasets.py`, `build_greedy_plan.py`, `mixed_precision.py`, `prepare_tiny_shakespeare_split.py`, README). `tests/` de convergence: solo inventario de nombres de test, no lectura línea a línea. Las líneas son aproximadas (`~`) donde se indica. No se modificó nada del proyecto.

---------------------------------------------------------------------
# PARTE 1 — MAPA

## 1.1 Flujo extremo a extremo: elección del plan y export

```
run_thesis_mode.sh paso 3
  ├─ scripts/run_ilp_partition.sh → validation/run_ilp_partition.py  → <cfg>/ilp_solution/
  └─ scripts/run_ilp_pareto_sweep.sh → validation/sweep_ilp_pareto.py
        → <model>_pareto_sweep.csv + <model>_pareto_summary.json (best_feasible_row)
run_thesis_mode.sh paso 4 (HYBRID_PLAN_SELECTION=pareto_best, L~850-1160)
  ├─ lee best_feasible_row.gpu_mem_budget_mb  (base)
  ├─ bucle GPU_VRAM_SAFETY_RATIO_CSV × PHASE4_TARGET_GPU_UTILIZATION_CSV
  │     trial_budget = base*ratio → OUT_DIR=<cfg>/ilp_solution_pareto_best_budget_<slug> (slug 5157.9→5157p9)
  │     run_ilp_partition.sh (CALIBRATION_BATCH_SIZE=probe_batch) → re-materializa el plan
  │     run_hybrid_execution.py SIN --compare_baselines y SIN --plan_source_csv (probe)
  │     éxito = exit code 0 → primer ratio sin OOM gana → safety_ratio_search_<model>.json
  └─ hybrid_already_executed=true
```
Selección "best": `sweep_ilp_pareto.py` ~L555-560: `feasible = ilp_status∈{optimal,feasible}`; `best_idx = ilp_objective.idxmin()` (empates → primera fila = menor presupuesto; igualdad exacta de floats).

## 1.2 `validation/run_ilp_partition.py` (~1000 líneas)
Pipeline de post-proceso (orden): resolver (ILP / phase4 greedy|joint, opcional `--phase4_calibrate_recompute_curve`) → refine local opcional → `PreparedTopologyMemory` + `correct_solution_memory` → opcional `expand_solution_memory` + `_validate_gpu_expansion_against_real_overhead` → **`_validate_final_plan_against_real_overhead` (L~262-355)** → `save_ilp_solution` (`src/ilp/export_solution.py`).
- `_resolve_hybrid_overhead_ratio` (silencia OSError/ValueError → 1.0); el presupuesto se divide por el ratio (L~710-720).
- `_load_calibration_model` (except genérico → `(None,None)`), `_solve_with_calibrated_recompute_curve`, `refine_cfg`.
- CLI: `--config_dir --model --gpu_mem_budget_mb --cpu_mem_budget_mb(1e18) --k_sigma(1.0) --w_time(1) --w_energy(0) --w_transfer(1) --peak_activation_overlap(0.35) --topology_memory_repair_budget(200) --gpu_expansion_budget(200) --phase4_* (recompute_penalty 0.5, checkpoint_penalty 0.3, target_gpu_utilization 0.95) --calibration_seq_length(128) --calibration_datasets_root(datasets)`, `--output_dir`. Llamador: `scripts/run_ilp_partition.sh`.
- Salidas: `ilp_assignment.csv` (layer, device, device_forward, device_backward, activation_strategy), `ilp_cut_edges.csv`, `ilp_solution_summary.json`.
- `export_solution.py` es limpio: `device`=forward, usa `forward_assignment or assignment`, sin inversión. Riesgo: `backward_assignment[layer]` → KeyError.

## 1.3 `validation/sweep_ilp_pareto.py`
Barrido de presupuestos GPU; baselines `_eval_fixed_policy("all_cpu"/"all_gpu")`, `_greedy_bits`, `_build_effective_from_dual`, `_default_paths`. Salidas: `<model>_pareto_sweep.csv` (ilp_status, ilp_objective, ilp_energy_j, all_*_objective, greedy_*, gpu_mem...) y `_pareto_summary.json`. **No escribe** `sim_*`. Backends incl. `pulp_cbc_dual`. Llamador: `run_ilp_pareto_sweep.sh`; `build_greedy_plan.py` importa `_greedy_bits` y `_default_paths`.

## 1.4 `validation/run_hybrid_execution.py`
Ejecuta el motor híbrido real y compara `all_cpu → all_gpu → ilp_plan` (mismo proceso, ese orden) con `--compare_baselines`. `_resolve_sim_reference_values` (L~145-215), `_run_single_safe` (solo captura `HybridExecutionUnsupportedError`), `_config_identity` (parsea `parts[-4..-1]`). Salidas: `hybrid_execution_protocol.csv` (`avg_step_ms`, `status`, `run_label`, `plan_selection_mode`, `delta_time_sim_vs_real_pct` L~494-497...), `ilp_steps.csv` (solo traza de ilp_plan).

## 1.5 Resto de `validation/`
| Archivo | Función | Notas |
|---|---|---|
| `run_statistical_significance.py` | t-test vs objetivo ILP | L~101-107 std inventada |
| `aggregate_metrics_stats.py`, `train_backward_meta_model.py` | wrappers finos (el 2.º deprecado, sin uso) | |
| `run_ilp_hierarchical_pipeline.py` | pipeline jerárquico | `w_energy=1.0` por defecto (otros 0.0) |
| `run_ilp_sensitivity.py` | rejilla k_sigma × w_transfer (L~34-35) | |
| `run_ilp_ablation_suite.py` | variantes `_clone_without_*` | |
| `validate_all_models.py` (413) | smoke del profiler (`ModelValidator`) | test tautológico |
| `validate_code.py`, `validate_zombie_fix.py` | "tests" por regex sobre `inspect.getsource` | frágiles |
| `validate_ilp_pipeline.py` | evalúa plan con `PreparedTopologyMemory`; exit 2 si excede | presupuesto 1e18 por defecto |
| `audit_experimental_grid.py` (596) | completitud de la rejilla | solo mira `ilp_solution/` |
| `export_ilp_tables_latex.py` (355) | tablas LaTeX | |
| `generate_ilp_report_assets.py` (967) | consolida CSV + gráficos | `_best_feasible_rows` L~96, `_best_hybrid_rows` L~108-135 |
| `generate_advanced_thesis_plots.py` (599) | roofline, waterfall, pred-vs-real | |
| `generate_chapter6_energy_and_greedy_plots.py`, `generate_energy_gpu_prism_comparison.py`, `generate_overhead_rescue_plot.py`, `generate_vertex_distribution_oom_plot.py`, `regenerate_chapter6_global_plots_no_mlp.py` | figuras del cap. 6 | datos hardcodeados / rutas absolutas |

## 1.6 `convergence_validation/`
| Archivo:línea | Contenido |
|---|---|
| `run_convergence_study.py` L15 `MODES`; `_build_parser` L~30-80; `_resolve_ilp_csvs` L~83; `_resolve_greedy_csvs` L~103; `_write_result` L~120-195; `main` L~200 | CLI de una celda. Salida: `train_loss.csv`, `eval_metrics.csv`, `summary.json`; exit 0 solo si status=ok, si no 2 |
| `prism_loop.py`: `_trace_once` L~99, `_forward_through_plan` L~118, `_run_one_optimizer_step` L~175, `_evaluate` L~220, `run_prism_training` L~300 | Traza una vez; `_DeviceAwareFXInterpreter` por llamada; grad-accum con `loss/accumulation_steps` |
| `all_gpu_loop.py`: dataclasses L~75-135, `_run_one_optimizer_step` L~140, `_evaluate` L~170, `run_all_gpu_training` L~205 | Bucle PyTorch estándar |
| `datasets.py`: `EpochBatchProvider` L~70, `_RemappedImageFolder`, `make_vision_provider`, `_TokenizedLineDataset`, `make_lm_provider`, `make_provider` | `drop_last=True`, generador con seed |
| `mixed_precision.py`: `Bf16MixedPrecisionWrapper`, `TrainableModel`, `optimizer_cls_from_name`, `build_trainable_model` | bf16 con master FP32 |
| `build_greedy_plan.py`: `_load_sweep_module`, `_budget_slug`, `_find_existing_pareto_budget` L~58, `build_greedy_plan` L~70 | escribe `greedy_solution_budget_<slug>/` |
| `prepare_tiny_shakespeare_split.py` | split 90/10 con seed 1234 |
| `run_convergence_study.sh` | matriz modelo×precisión×régimen×semilla |
| `tests/` | smoke (cycling, traza única), grupos de checkpoint, métricas de recursos, pesos atados (no leídos línea a línea) |

### Regímenes y grad-accum (cómo funciona)
- **Régimen A** (suficiencia): plan `<cfg>/ilp_solution/`; corre `ilp_plan` (PRISM) y `all_gpu`, mismo batch (por defecto 256/128/128/128 para resnet50/resnet152/distilgpt2/gpt2_small, `DEFAULT_BATCH_A`).
- **Régimen B** (rescate OOM): plan `ilp_solution_pareto_best_budget_*` (`glob(...)` ordenado, **`candidates[0]`**); corre `ilp_plan`, `all_gpu_gradaccum` (micro-batch `GRADACCUM_MICRO_BATCH=8`, `accum = batch // 8`) y `greedy_plan` (construido al mismo presupuesto con `build_greedy_plan.py`). Batches B: 512/256/128/128.
- Grad-accum: en ambos bucles, `zero_grad` → K micro-batches con `(loss/K).backward()` → `step`; un "paso" = 1 optimizer step; `steps_per_epoch = len(provider)//K`.
- Optimizador/LR: visión SGD 0.01; LM AdamW 5e-5. Semillas 42,43,44. Épocas 7. Eval cada 50 pasos con 10 batches. bf16 se salta si el host no es kinovis (`BF16_HOST_HINT`).

---------------------------------------------------------------------
# PARTE 2 — HALLAZGOS

## CRÍTICO

**C1. Los planes Pareto de LLM terminan 100 % CPU (causa raíz confirmada).**
- Dónde: `run_ilp_partition.py::_validate_final_plan_against_real_overhead` (L~262-355; uso L~870-905). `feasible = status=="ok" and reserved<=budget`. Para distilgpt2/gpt2_small, `measure_real_peak_for_plan` (`src/runner/hybrid_overhead_calibration.py` L~277) devuelve `status="skipped"` ("symbolic_trace unsupported"; `_try_symbolic_trace` no traza LLM de HF, mientras que el motor híbrido sí lo hace con `try_export_decoder_only_trace`).
- Efecto: skipped ⇒ "no factible" ⇒ `removal_order` recorre TODAS las capas GPU; `_measure(n)` también se salta y cae en `else: hi = n` ("degenerate case, use it anyway") ⇒ todo a CPU.
- Evidencia: logs `thesis_mode_2026101[34]*.txt`: "exceeds the real measured peak (NoneMB vs budget 5157.9MB, status=skipped)" y "removed 58 additional layer(s)" (57 / 113 en otros). Datos medidos: `ilp_solution_pareto_best_budget_{5157p9,6263p1}` de distilgpt2 → CPU=59; gpt2_small `{6632p1,5157p9}` → CPU=113. En cambio resnet50/152/vit sí tienen GPU>0. El sweep da 58 GPU/1 CPU, objetivo 100.41 frente a `all_cpu_objective` 1775.7, y all-GPU factible desde 5731 MB (5202.7 MB).
- Estado inconsistente: `ilp_solution_summary.json` queda con `objective_value=100.41`, `gpu_mem_used_mb=3104.9`, status "optimal", `layers_cpu_forward=59` (valores obsoletos, no se recalculan tras recortar).
- El probe "pasa": `safety_ratio_search` reporta success en el intento 1 porque un plan todo-CPU nunca hace OOM. Es decir, el criterio "exit code 0" premia justo el plan degenerado.
- Consecuencias en cascada: (i) el estudio de convergencia régimen B de LLM entrena "PRISM" como todo-CPU (`run_convergence_study.sh`/`prism_loop.py` no lo detectan) y lo compara con all-GPU grad-accum; (ii) `generate_*` y `export_ilp_tables_latex` usan `ilp_objective` del sweep (100.41), que no es el del plan ejecutado; (iii) `audit_experimental_grid` mira solo `ilp_solution/` y da OK.
- Arreglo: (1) distinguir `skipped`/error no-OOM ("sin validar": conservar el plan o validar con `topology_memory`) de overflow real o OOM; recortar solo si `status=="ok"` y excede, o si hay OOM; (2) hacer que `_try_symbolic_trace` en la calibración use el export decoder (reutilizar `try_export_decoder_only_trace`); (3) tras recortar, recalcular objective, cut_edges, gpu_mem_used y marcar `status="repaired"`/`"degraded_all_cpu"`; (4) el probe de safety-ratio debe fallar si el plan es todo-CPU mientras all-GPU era factible, o si `#GPU < ε·#GPU_del_best_sweep`; (5) alinear `_validate_gpu_expansion_against_real_overhead` (L~370+), que con `feasible0=False` conserva el plan preexpansión (inconsistente con la validación final); (6) unit test: LLM mock con status skipped ⇒ plan sin cambios.
- Se descartan las hipótesis de export: columnas/orden/inversión (el export es correcto) y la métrica de selección del sweep (elige un plan 58 GPU; el recorte posterior es lo que rompe).

**C2. `sim_time_ms / sim_energy_j / sim_status` no se escriben y hay fallback silencioso a `ilp_objective`.**
- `sweep_ilp_pareto.py` (dict de fila ~L470-520) no emite `sim_status, sim_objective, sim_time_ms, sim_energy_j, sim_transfer_ms, sim_gpu_mem_mb, sim_cpu_mem_mb, sim_warnings, sim_violations, sim_error` (los CSV de `data/legacy/chuc-4` sí los tienen ⇒ un escritor antiguo fue eliminado).
- `run_hybrid_execution.py::_resolve_sim_reference_values` (L~145-215): `sim_time_ms` ← fila → **`ilp_objective`** (objetivo ponderado, no ms) → `--plan_objective`; `sim_energy_j` ← fila → `ilp_energy_j`. `delta_time_sim_vs_real_pct`/`delta_energy_sim_vs_real_pct` (L~494-497) comparan un objetivo adimensional con ms reales (y posiblemente por-muestra vs N pasos): carecen de sentido y alimentan `plot_global_predicted_vs_actual` (R²).
- Además: el orden por `_budget_abs_delta` se anula con el orden posterior por `_ilp_obj_num` (se ignora el presupuesto pedido); los filtros optimizer/precision/batch nunca aplican (el CSV del sweep no tiene esas columnas); el probe no pasa `--plan_source_csv`; `try/except Exception` silencia errores.
- Arreglo: escribir `sim_*` en el sweep (o eliminar el campo) y fallar/NaN si falta, nunca sustituir por `ilp_objective`; calibrar objetivo→ms con una constante documentada; propagar `--plan_source_csv` y corregir el orden.

**C3. Test estadístico inválido (`run_statistical_significance.py` L~101-107).**
- La desviación estándar se fabrica como `objective*cv` (`cv=max_cv_key_metrics` o 0.05 por defecto), `n_samples=5` por defecto: t-test de una muestra contra el objetivo ILP determinista (salida de modelo, no medición); bilateral ⇒ "significativo" incluso cuando el ILP es peor; sin corrección por comparaciones múltiples; el objetivo no está en ms. `significant_vs_gpu` falta en la rama `else` (KeyError en `export_ilp_tables_latex`); `inf` no lo detecta `isna`; regex `mlp|simple` hardcodeado; el plot silencia errores.
- Arreglo: medir n≥5 repeticiones reales (`run_hybrid_execution --repeats`, semillas distintas) de ilp_plan vs all_gpu/all_cpu en ms; test de Welch (o Mann-Whitney) unilateral/pareado + IC bootstrap + Holm; CV medido, no 0.05.

**C4. Figuras del capítulo 6 con datos hardcodeados/inventados.**
- `generate_energy_gpu_prism_comparison.py`: todas las energías en el código (all_gpu_j [142.50,nan,238.59,nan], greedy_j, prism_j), "up to 62.6%" fijo, memorias OOM 42.7/41.5 GB, "32GB VRAM".
- `generate_overhead_rescue_plot.py`: vértices, tiempos de profiling/ILP, `total_campaign_s` [428,512,610,580,745] (inconsistente con `step_time_ms×5000` de otro script: 92.7 ms×5000=463 s≠428; 140.18×5000=701≠512), curvas de rescate (vram_ratios 100..40) inventadas, "se activa por debajo del 80 %".
- `generate_vertex_distribution_oom_plot.py`: todo hardcodeado ("~20 % más rápido", "<0.55 % del coste").
- `generate_chapter6_energy_and_greedy_plots.py::generate_overhead_breakdown_plot`: `ilp_solver_s [0.48,0.72,0.92,0.85,3.14]`, `step_time_ms [92.70,140.18,85.60,149.17,145.08]`, 5000 pasos fijos; `ax.set_ylim(75,100)` (oculta ahorros <75 %); el nombre `_b128.png` no coincide con los datos B=256.
- Arreglo: leer de `reports/**/hybrid_execution_consolidated.csv` y logs reales; si son ilustrativas, marcarlas en el título/pie como "ilustrativa", nunca citar sus números en el texto.

**C5. Régimen B de LLM no es "PRISM híbrido" (consecuencia de C1) y se compara con una evaluación no equivalente.** Ver F1/F2 abajo; `run_convergence_study.sh` no valida que el plan tenga GPU>0.

## ALTO

**A1. Métrica de selección del mejor plan = `ilp_objective` ponderado únicamente** (sweep ~L555-560; `generate_ilp_report_assets._best_feasible_rows` L~96): ignora memoria y energía (w_energy=0), empates → menor presupuesto por igualdad exacta de floats; el presupuesto del sweep no se corrige por overhead híbrido ni por reparación topológica (el `run_ilp_partition.py` divide otra vez por `hybrid_overhead_ratio`, por lo que el presupuesto se reduce dos veces: ratio×overhead). Arreglo: Pareto (tiempo, memoria pico, energía), desempate por menor VRAM con tolerancia relativa; usar el mismo modelo de memoria que el plan materializado; el objetivo final debe medirse con ejecución real.

**A2. Cherry-picking en `_best_hybrid_rows` (`generate_ilp_report_assets.py` L~108-135):** elige el mínimo `avg_step_ms` entre todas las filas `ilp_plan`/ok candidatas (best-of-N) ⇒ sesgo optimista; ignora la comparación contra all_gpu/all_cpu. Arreglo: fijar el plan por criterio previo (no por el resultado medido), o reportar mediana ± IC de todos.

**A3. Comparación híbrido vs baselines en `run_hybrid_execution.py`:** un solo proceso con orden fijo all_cpu → all_gpu → ilp_plan (calentamiento/cachés/fragmentación sesgan al último), sin repeticiones, `--steps` default 5, semilla 42 fija. `_run_single_safe` no captura OOM ⇒ un OOM de all_gpu aborta toda la corrida. Línea base = all_gpu si no, all_cpu. `divergence_explained_by_rescue` es una heurística sobre texto de warnings. Arreglo: rotar orden/aleatorizar, calentamiento explícito, ≥5 repeticiones, capturar `torch.cuda.OutOfMemoryError` y registrar `status="oom"`.

**A4. `audit_experimental_grid.py`:** lee el estado ILP solo de `<cfg>/ilp_solution/ilp_solution_summary.json` (L~190), nunca de `ilp_solution_pareto_best_budget_*`; `ilp_ok` solo mira optimal/feasible ⇒ no detecta plan todo-CPU ni objetivo obsoleto; `hybrid_ok` = existe `hybrid_execution_protocol.csv` (no que corrió OK); `expected_repeats` default 1 (n=1 cuenta como 100 %); `successful_runs=min(metrics_files,max(run_dirs,1))` puede sobrecontar; `except Exception: continue`. Arreglo: auditar `layers_gpu>0`, coherencia summary↔CSV, status "ok" del ilp_plan.

**A5. `sweep_ilp_pareto.py::_eval_fixed_policy("all_gpu")`:** factibilidad con memoria agregada (suma), no topológica ⇒ all_gpu se marca factible con más facilidad que el ILP (comparación injusta); el greedy empata hacia GPU (`<=`); `ilp_energy_j` calculado manualmente; filas `profile_invalid` con objetivo inf; `except Exception` al cargar perfiles se silencia salvo `--fail_fast_profile_errors`.

**A6. Ablación (`run_ilp_ablation_suite.py`):** `full_result` se resuelve dos veces; los `_clone_without_*` construyen `ILPInputData` con un subconjunto de campos (los fwd/bwd, energía, pico, activación, io y `shared_call_site_layers` existen en `ILPInputData`); si se descartan, las variantes usan otro modelo de costes y Δ vs full queda confundido; `no_robust` pone k_sigma=0 y los objetivos no son comparables entre variantes. Arreglo: re-evaluar todas las variantes con el modelo de costes completo.

**A7. Sensibilidad (`run_ilp_sensitivity.py`):** `delta_pct` del objetivo entre parámetros no es comparable (la escala cambia con w_transfer/k_sigma); omite `strict_metric_validity/memory_model`; parámetros sin uso (`k_sigma` en `_solve_one`, `n_k`, `n_wt`).

**A8. Gráficos de predicho vs real:** `generate_advanced_thesis_plots.plot_global_predicted_vs_actual` y `regenerate_chapter6_global_plots_no_mlp.py` correlacionan `plan_objective` (ponderado, sin ms) con `avg_step_ms`, con Pearson R² agrupado entre modelos/batches (inflado por escala) y mezclan tres nodos con hardware distinto (chuc-4, kinovis-2, paccaA100). Dependen de `ilp_statistical_significance.csv` (de C3). Roofline con `vram_bandwidth_gb_s=760`, `dram_bandwidth_gb_s=50`, `peak_tflops 20/1` supuestos, no medidos; bytes=(params+act)·1e6.

**A9. `generate_ilp_report_assets.py`:** `Is_Overflow = (cost==0.0)` etiquetado como "RAPL Counter Overflow (>21s)" también en gráficas de latencia (hardware EPYC 7513 asumido); `_plot_memory_utilization_bar` suma picos por capa ("Peak Demand") ≠ pico real; batch/optimizer/precision extraídos de `source_csv` por regex ('unknown' como fallback); `source_csv` con más filas se elige silenciosamente.

**A10. Convergencia — `candidates[0]` y orden lexicográfico** (`run_convergence_study.py` L~83-100, `build_greedy_plan.py` L~58-68): `sorted(glob("ilp_solution_pareto_best_budget_*"))[0]` coge el primer directorio alfabético, no el mejor ni el último (p. ej. "10000p0" < "9000p0"); con varios intentos de safety-ratio conviven varios directorios y el elegido es arbitrario. Arreglo: leer `safety_ratio_search_<model>.json` (intento exitoso) o ordenar numéricamente + exigir uno.

**A11. Convergencia — evaluación no comparable entre PRISM y baselines.**
- (F1) `_evaluate` en LLM de PRISM: el grafo exportado con `torch.export` mantiene dropout activo (documentado en el docstring de `prism_loop.py`); all_gpu evalúa con `model.eval()` real ⇒ `eval_loss` de LM de PRISM sesgado al alza frente al baseline; la curva no es comparable sin corregirlo. Arreglo: exportar el grafo en modo eval para validación (segundo trace) o fijar `p=0` de dropout en todos los modos.
- (F2) En `all_gpu_gradaccum` el provider de validación usa `batch_size=8` y `eval_batches=10` ⇒ 80 muestras frente a 5120 (batch 512) en PRISM; subconjuntos distintos y ruido distinto. Arreglo: `eval_batches = ceil(N_eval/batch)` o evaluar todo el split de validación en todos los modos.
- (F3) El orden de datos depende del `batch_size` del provider (`generator_seed` igual pero tamaño de batch distinto ⇒ permutación distinta en gradaccum respecto a PRISM con batch efectivo equivalente): no se ven las mismas muestras por paso. Arreglo: muestrear índices por paso con un generador común y trocear en micro-batches.
- (F4) En régimen B con BN (ResNet) el grad-accum no es equivalente (documentado en el README, bien); LM usa LayerNorm, correcto.
- (F5) División entera `accumulation_steps = batch // GRADACCUM_MICRO_BATCH` (sh L~375): si no es divisible, el batch efectivo difiere (no ocurre con 512/256/128 y micro 8, sí con otros).

## MEDIO

- M1. Hardcodes (b): ver tabla del apartado "Hardcodes" más abajo.
- M2. `run_hybrid_execution.py` `_config_identity` parsea `parts[-4..-1]` del path (frágil si cambia el layout); `ilp_steps.csv` solo contiene ilp_plan.
- M3. `run_ilp_hierarchical_pipeline.py`: `w_energy=1.0` por defecto frente a 0.0 en el resto; sin `strict_*`/`memory_model`; `_default_paths` toma `run_dirs[0]` sin comprobar existencia; `ILPConfig` sin `enforce_convex_weights`; `topology_feasible` contra presupuesto crudo; tag de presupuesto `str(int(budget))` (colisiona 5157.9/5157.2).
- M4. `export_ilp_tables_latex.py`: "Mejora vs CPU/Greedy" se calcula sobre el objetivo (no ms); `hybrid_tex` se escribe dos veces con captions/labels distintos (tab:hybrid-execution-best vs tab:ilp-hybrid-best-per-model) y se añade de nuevo a `combined_tex` (tabla duplicada); `sig_path = out_dir.parent/"csv"/...` frágil; `_prepare_best_table` KeyError si falta `all_cpu_objective`; la ablación elige min objetivo entre presupuestos; ruta de CSV vacío escribe placeholders y devuelve 0.
- M5. Energía de convergencia: `total_energy_j = avg_power_w × wall_time` (potencia GPU solo vía NVML; ignora CPU/RAPL) ⇒ energía de PRISM (que trabaja en CPU) infravalorada frente a all_gpu; no comparable con las cifras de energía de la campaña. `peak_cpu_mem_mb` usa `ru_maxrss` de todo el proceso (incluye carga/tokenizado/trace), `peak_gpu_mem` se reinicia tras montar el modelo en all_gpu pero tras trace en PRISM (ventanas distintas).
- M6. Convergencia: sin warmup ni scheduler ni clipping de gradiente (aplica igual a todos, bien), `weight_decay` por defecto AdamW (0.01) en `optimizer_cls(**{lr})`; `bf16`: modelo en bf16 con master FP32; fp16 no soportado en el CLI aunque `_torch_dtype_from_precision` lo mapea. `torch.manual_seed(seed)` no cubre RNG de CUDA por dispositivo ni `numpy`/`random`; dropout RNG difiere CPU/GPU ⇒ sin igualdad bit a bit incluso con misma semilla.
- M7. `prism_loop.run_prism_training`: `except RuntimeError` convierte "Dual-device recompute mismatch" en `status="dual_placement_mismatch"` y termina la celda (se pierde la curva); un OOM de CPU no se contempla (lo mata el cgroup). `strict_plan=False` ⇒ capas no cubiertas por el plan se ejecutan sin aviso salvo `warnings` (la lista se descarta: nunca se escribe en `summary.json`).
- M8. `build_greedy_plan.py`: usa `k_sigma=1.0`, `topological` (no `topological_tight`, deliberado según comentario), sin corrección de overhead real ni de topología ⇒ el greedy se evalúa con otro presupuesto efectivo que el ILP (que se redujo ×ratio×overhead): comparación de presupuestos no idéntica; `activation_strategy` siempre "retain" (el ILP usa recompute/checkpoint ⇒ ventaja del ILP confundida con la de la colocación); `_budget_slug` formatea con 1 decimal (7369.04 vs 7369.0). Si el único plan ILP es todo-CPU (C1), el greedy se construye al mismo presupuesto pero sí tiene capas GPU: el "greedy" ganaría a PRISM.
- M9. `run_convergence_study.sh`: `find_config_dir | head -n 1` (primer match ordenado, p. ej. varios `custom/` vs `calibration/`), `HOST_TAG=$(hostname)`, `BF16_HOST_HINT=kinovis` y `CPU_MEM_LIMIT_PCT=95`; `run_one` captura todo fallo con `|| echo ... continuing` (falla silenciosa; no hay resumen final de celdas fallidas); `is_vision_model` abre un intérprete Python por llamada; `GRADACCUM_MICRO_BATCH=8` fijo (valdría calcularlo con memoria disponible).
- M10. `validate_all_models.py`: `test_metadata_fields` tautológico (comprueba atributos de su propio MockArgs); test 3 solo cubre los 3 primeros modelos; los resultados nunca cambian el exit code; `except Exception` truncado a 100 caracteres. `validate_code.py`/`validate_zombie_fix.py`: pruebas por texto de código fuente (frágiles, one-off). `validate_ilp_pipeline.py`: presupuesto 1e18 por defecto (siempre pasa) y nunca verifica que el plan no sea todo-CPU.
- M11. `run_ilp_partition.py`: `model.cuda()` carga el modelo completo a GPU durante la calibración (puede OOM en modelos grandes); `refine_cfg` omite `w_congestion/congestion_knee_ms/phase4`; `_solve_with_calibrated_recompute_curve` resuelve su base con greedy y `enable_recompute=False`.

## BAJO

- B1. Código muerto: `run_ilp_partition.py` (params `data`/`activation_strategy` sin uso; args `backward_meta_*` ignorados; falta `import Optional`, funciona por `__future__`); `train_backward_meta_model.py` (deprecado); `generate_advanced_thesis_plots.plot_predicted_vs_actual` (`pass`); `generate_ilp_report_assets.py` (`if True:` L~740, `unique_seeds`/`is_multiseed_root`, `_get_column_safe` sin uso; `_plot_model_energy_curves` definida tras `main`; mensajes de ruta de ablación/sensibilidad erróneos); `run_ilp_sensitivity.py` (`n_k`,`n_wt`).
- B2. `palette` sin `hue` (deprecación seaborn), violin sin hue.
- B3. `generate_advanced_thesis_plots`: `plot_waterfall_ablation` ("Ablation" = all-CPU vs ILP); `batch=32` si no se puede parsear; watts = `ilp_energy_j/(ilp_objective/1000)` mezcla objetivo ponderado con ms.
- B4. `datasets.py`: `AutoTokenizer.from_pretrained("gpt2")` en cada `make_lm_provider` (descarga/lectura repetida, ×2 por run); `drop_last=True` descarta muestras (documentado); val de visión sin shuffle (bien) pero con transforms de `weights.transforms()` en train (sin augmentación: más fácil sobreajustar; igual para todos).
- B5. `prepare_tiny_shakespeare_split.py`: el split es por líneas barajadas (líneas consecutivas de la misma escena caen en train y val ⇒ fuga/optimismo leve); seed 1234 y 10 % hardcodeados (CLI-parametrizables).
- B6. `_write_result` escribe CSV a mano (sin escapes); `summary.json` no guarda `warnings`, versión de torch/commit, ni hardware.

---------------------------------------------------------------------
# HARDCODES (apartado b) — dónde están y de dónde deberían salir

| Valor | Ubicación | Debería venir de |
|---|---|---|
| Rutas `/home/zephyr/Documents/University/PhD/Code/Final Thesis Code/...` | `generate_chapter6_energy_and_greedy_plots.py` L~75, ~230; `generate_energy_gpu_prism_comparison.py` L~139-142; `generate_overhead_rescue_plot.py` L~153-157; `generate_vertex_distribution_oom_plot.py` L~207-210; `regenerate_chapter6_global_plots_no_mlp.py` L~173 | `Path(__file__).resolve().parents[1]` o `--output_dir` |
| Nombres de nodo chuc-4 / kinovis-2 / paccaA100.unicartagena.edu.co | `regenerate_chapter6_global_plots_no_mlp.py`, `run_convergence_study.sh` (BF16_HOST_HINT=kinovis) | `--hosts_csv` / config |
| Semillas 42 (hybrid, convergence CLI), 42,43,44 (sh), 1234 (split) | `run_hybrid_execution.py`, `run_convergence_study.py` L~55, `.sh` `SEEDS_CSV`, `prepare_tiny_shakespeare_split.py` | CLI `--seeds`; n≥5 para estadística |
| n=5, CV=0.05 | `run_statistical_significance.py` L~101-107 | medición real (C3) |
| Batch 8 / input 224 / seq 128 / lr 0.01 / steps 5 / intervalo energía 0.05 | `run_hybrid_execution.py`; `input_size=224` en `run_ilp_partition.py` L~124, L~437; `validate_all_models.py` (BATCH_SIZE=1) | argumentos de CLI + metadatos del perfilado |
| Presupuestos GPU/CPU 1e18, `peak_activation_overlap` 0.35, `topology_memory_repair_budget` 200, `gpu_expansion_budget` 200, `phase4_*` 0.5/0.3, `target_gpu_utilization` 0.95, k_sigma 1.0, w_* | `run_ilp_partition.py`, `sweep_ilp_pareto.py`, `build_greedy_plan.py`, `validate_ilp_pipeline.py` | una sola `ILPConfig` por defecto compartida |
| Rejillas k_sigma [0,0.5,1,1.5,2], w_transfer [0,0.5,1,2,5] | `run_ilp_sensitivity.py` L~34-35 | CLI |
| Modelos SUPPORTED_MODELS, VALID_MODELS, regex `mlp|simple`, MODELS list | `run_hybrid_execution.py` L~43, `validate_all_models.py`, `audit_experimental_grid.py` L~575, `generate_*` | `MODELS_CSV` / registro |
| Batches 256/128/128/128 y 512/256/128/128; micro-batch 8; épocas 7; lr 0.01/5e-5; eval 50/10 | `run_convergence_study.sh` | derivados de VRAM medida |
| Presupuesto 20480 MB, B=256, AdamW, fp32; `ylim(75,100)`; 5000 pasos | `generate_chapter6_energy_and_greedy_plots.py` | argumentos |
| peak_tflops 20/1; VRAM bw 760 GB/s; DRAM bw 50 GB/s | `generate_advanced_thesis_plots.py` L~320 | metadatos del perfil (PCIe/BW medidos, `*_meta.json`) |
| PCIe | no se ha encontrado un valor de PCIe hardcodeado en estos scripts; el coste de transferencia procede de `transfer_edges_csv` (perfilado) | — |
| Umbrales 0.05/0.001 de p-valor, epsilon 1e-15 | `regenerate_chapter6_global_plots_no_mlp.py` | CLI |
| Datos de energía/tiempos/vértices de las figuras | scripts de C4 | CSV reales |

---------------------------------------------------------------------
# VALIDEZ ESTADÍSTICA (e) — resumen
Una sola ejecución de 5 pasos, semilla 42 y sin repeticiones en el híbrido; std fabricada; objetivo ILP tratado como muestra; mejor-de-N en `_best_hybrid_rows`; R² agrupado entre hosts; convergencia con 3 semillas (aceptable para curvas, pero no hay IC ni test sobre `final_eval_loss`; `summary.json` ofrece solo valores finales por semilla, sin agregación entre semillas).

# COMPARABILIDAD DE CURVAS (f) — resumen
Iguales: dataset/split, seed del generador de entrenamiento, optimizer/lr, precisión, sin scheduler, cadencia de eval por pasos de optimizador. Distintas: dropout en la evaluación LM de PRISM (F1), tamaño del subconjunto de validación en gradaccum (F2), orden de datos con micro-batch distinto (F3), plan PRISM LLM todo-CPU en régimen B (C1), greedy con otro presupuesto efectivo y sin recompute/checkpoint (M8), energía solo GPU (M5), ventanas de pico de memoria diferentes (M5).

# EXCEPCIONES TRAGADAS / FALLBACKS / CÓDIGO MUERTO (g) — índice
`_resolve_hybrid_overhead_ratio` →1.0 silencioso; `_load_calibration_model` except genérico; `run_hybrid_execution._resolve_sim_reference_values` try/except + fallback a `ilp_objective`; sweep: perfil inválido salvo `--fail_fast_profile_errors`; `audit_experimental_grid` `except: continue`; `generate_ilp_report_assets._detect_gpu_vram_total_mb` traga excepciones; gráficos con try/except que imprimen; `run_one` del .sh `|| echo`; `measure_real_peak_for_plan` "skipped" tratado como infactible (C1). Código muerto: ver B1.

# PRIORIDAD DE ARREGLOS
1. C1 (validación final `skipped` ⇒ no recortar; recalcular summary; safety probe que rechace todo-CPU). 2. Re-generar los planes Pareto de LLM y repetir el estudio de convergencia régimen B de LLM. 3. C2 + C3 (escribir `sim_*`; estadística con mediciones reales). 4. F1/F2 (evaluación equivalente) y A10 (elección de directorio). 5. Eliminar datos hardcodeados de figuras (C4) y rutas absolutas. 6. A1–A4 (métrica de selección y auditoría).
