# Revisión scripts/, tests/, config/, docs/ — PRISM

Alcance: solo lectura. `scripts/` (29 archivos), `tests/` (25), `convergence_validation/tests` (4), `pytest.ini`, `config/`, `README.md`, `docs/`.
No existen `Makefile` ni `requirements.txt` en la raíz (solo `config/requirements.txt` y `config/environment.yml`).
Las líneas son las del archivo actual; las marcadas "~" son aproximadas.
Suite ejecutada con `prism_env` (torch 2.14+cu130, GPU disponible): **125 passed, 7 skipped, 0 failed**. Con el Python base (sin torch) la colección falla.

---
## 1. MAPA

### 1.1 Orquestador `scripts/run_thesis_mode.sh` (1262 líneas)

**Perfiles** (l.~50-130)

| Perfil | Modelos | Optimizadores | Precisiones | Batches | REPEATS / WARMUP / MEASURE |
|---|---|---|---|---|---|
| `doctoral_minimal` y `doctoral_diagnostic` | simple_mlp, resnet50, resnet152, vit_b16 | SGD, AdamW | fp32 | 8, 32, 64 | 5 / 3 / 10 |
| `doctoral_full` | + bert_base, gpt2_small, distilgpt2 | SGD, AdamW, RMSprop | fp32, bf16 | 8, 16, 32, 64, 128, 256, 512 | 7 / 5 / 15 |
| `quick_smoke` | simple_mlp | SGD | fp32 | 8 | 1 / 1 / 1 |
| `custom` | MODELS_CSV, etc. | por defecto SGD, AdamW | — | por defecto "8,32" | REPEATS=5 |

- Todos usan `GPU_BUDGETS_MB=auto` y `K_SIGMA=1.0`.
- Los perfiles *strict* (minimal y full) fijan `STRICT_GRAPH/TRANSFER=true`, `REGIME=deterministic` y `ENFORCE_CONVEX_WEIGHTS=true` (l.241-248).
- `REPEATS<2` en un perfil doctoral aborta (l.230-235).

**Variables clave y defaults**
- Pesos: `W_ENERGY=0.5`, `W_TIME=0.5`, `W_TRANSFER=1.0`.
- Presupuestos: `GPU_MEM_BUDGET_MB=1e18` (ILP principal); `CPU_MEM_BUDGET_MB=auto` (MemAvailable×0.9, leído en el momento).
- Memoria y backend: `PEAK_ACTIVATION_OVERLAP=0.35`, `MEMORY_MODEL=topological`, `BACKEND=auto`, `HW_AGGREGATE=max`.
- Cgroup: `CPU_MEM_LIMIT_PCT=95` (systemd-run MemoryMax, swap=0).
- Learning rates y semilla: `LR_VISION=0.01`, `LR_LM=5e-5`, `SEED_BASE=42`.
- Phase 4: penalización recompute 0.5, checkpoint 0.3, `W_IO=0`.
- Calibración: `seq_len=128`, fracciones 0, .25, .5, .75, 1.
- Híbrido: `GPU_EXPANSION_BUDGET=200`, `HYBRID_STEPS=10`, `HYBRID_BATCH_SIZE=8`, `HYBRID_PLAN_SELECTION=pareto_best`.
- Control de fallos: `FAIL_FAST=false`, `ALLOW_PARTIAL_PROFILING_FAILURES=true`, `STRICT_METRIC_VALIDITY=false`, `FAIL_FAST_PROFILE_ERRORS=false`. Valen igual en los perfiles doctorales.
- Banderas de fase: `RUN_PROFILING`, `RUN_ILP`, `RUN_HYBRID`, `RUN_REPORTS`, `DRY_RUN`, `DOWNLOAD_DATASETS`.
- Rutas: `BASE_OUTPUT_DIR`, `REPORTS_DIR`, `LOG_DIR`, `DATASETS_DIR`, `HOST_TAG`, `GPU_VRAM_FALLBACK_MB`, `USE_CGROUP_MEM_LIMIT`.

**Orden de fases**
1. *Preflight*: detecta VRAM (nvidia-smi `-i 0`) y RAM; activa conda `~/anaconda3`/`prism_env`.
2. STEP1 `download_datasets.py` (corre incluso con DRY_RUN).
3. STEP2 `run_experiments.sh`: perfilado por (modelo, optimizador, precisión, batch, repetición).
4. STEP3 por cada `batch_*`:
   - `run_ilp_partition.sh`, sin calibración Phase4 ni expansión;
   - `run_ilp_pareto_sweep.sh` con la escalera de presupuestos.
   - Un fallo hace `continue`.
5. STEP4 híbrido: toma la mejor fila factible del Pareto (`best_feasible_row`). Para cada (safety_ratio, target_util) materializa el plan con Phase4 y lo ejecuta con `run_sandboxed validation/run_hybrid_execution.py`.
6. STEP5 reportes, en este orden:
   - `generate_ilp_report_assets.sh`
   - `run_statistical_significance.sh`
   - `export_ilp_tables_latex.sh`
   - `validation/generate_advanced_thesis_plots.py`
   - `plot_all_oom_vs_ilp.py`
   - `plot_all_energy_tradeoff.py`
   - `generate_methodological_plots.py`
   - `audit_experimental_grid.sh` (LOGS_ROOT="logs")
   - `THESIS_MODE_PROTOCOL_CHECKLIST.md`

**Presupuestos GPU**
- Escalera de porcentajes fija 90, 85, 80, 75, 70, 50 sobre la VRAM total (`compute_dynamic_budgets`, l.608-625 y 674). Está duplicada en `run_ilp_pareto_sweep.sh:48`. Nunca explora por debajo del 50 %.
- Búsqueda adaptativa (l.~920-1095): `GPU_VRAM_SAFETY_RATIO_CSV=0.90,0.85` × `PHASE4_TARGET_GPU_UTILIZATION_CSV=0.95,0.90,0.85`.
  - El batch de la sonda sale del nombre del directorio `batch_N`.
  - Escribe `safety_ratio_search_<model>.json`.

**Cómo se decide OOM-SKIP y se clasifica un fallo**

*En el perfilado (`run_experiments.sh` l.676-732)*, en orden de prioridad:
1. exit 137 o 9 → "CPU OOM KILL" (`oom_cpu_skipped_config.json`) y `break`.
2. exit 124 → timeout (rama muerta, ver H-C2).
3. `is_oom_failure_from_log` verdadero → "OOM SKIP" (`oom_skipped_config.json`), seguido de un rescate con `--no_gpu`.
4. Otro caso → `FAILED_RUNS`, `handle_failure` y continúa.

*Reintentos OOM*: el batch baja con `OOM_RETRY_BACKOFF=2` hasta `OOM_RETRY_MIN_BATCH=1`.

*En la sonda híbrida (l.1064-1070)*: **cualquier** fallo se registra como "Probe OOM'd (or failed)" y se prueba el siguiente ratio. No hay clasificación. Si se agotan los ratios: "Skipping".

*Código de salida*: ILP, Pareto e híbrido hacen `continue`. El script termina en 0 aunque haya fallos parciales.

### 1.2 Resto de scripts

| Script | Propósito | Entradas / salidas / vars |
|---|---|---|
| `run_experiments.sh` (795) | Perfilado en grid, con agregación | Grids por defecto (ver §2.3). Vars: `WARMUP`, `MEASURE`, `REPEATS`, `SEED_BASE`, `OOM_SKIP_CONFIGS`, `OOM_RETRY_*`, `PROFILER_TIMEOUT_SECS`, `ALLOW_PARTIAL_FAILURES`, `FORCE_NO_GPU`. Salida: `run_NNN/`, `*_metrics_stats.csv`, marcadores `oom_*_skipped_config.json` |
| `run_ilp_partition.sh` | Resuelve un ILP (wrapper de `validation/*`) | `CONFIG_DIR`, `OUT_DIR`, pesos, presupuestos, flags Phase4 y calibración. Exige el binario CBC |
| `run_ilp_pareto_sweep.sh` | Barrido por presupuestos de GPU | `GPU_BUDGETS_MB`, `CPU_MEM_BUDGET_MB`. Salida: `<model>_pareto_summary.json` y CSVs |
| `generate_ilp_report_assets.sh` | CSV consolidados y mejores filas | `INPUT_ROOT`, `OUTPUT_DIR` |
| `export_ilp_tables_latex.sh` | CSV → tablas LaTeX | `BEST_CSV`, `CONSOLIDATED_CSV`, `OUT_DIR` |
| `run_statistical_significance.sh` | Wrapper de tests estadísticos | `PYTHON_CMD` |
| `audit_experimental_grid.sh` | Cobertura del grid esperado | `EXPECTED_*_CSV`, `INPUT_ROOT`, `LOGS_ROOT` |
| `discover_ilp_config_dirs.sh` | Localiza config_dirs | `MODEL`, `BATCH`, `GPU_BUDGETS_MB` |
| `sanitize_cuda_env.sh` | Limpia `LD_LIBRARY_PATH`/`CUDA_HOME` | Se hace `source` |
| `download_datasets.py` | Descarga y manifiesto | `--models`, `--datasets_root` |
| `find_free_gpus.py` | Busca GPUs libres en Grid5000 por SSH | Sin parámetros |
| `generate_pareto_curve_ch5.py` | Figura Pareto del cap. 5 | CSV fijo |
| `generate_chapter6_prism_vs_allgpu_figures.py` | PRISM vs All-GPU del cap. 6 | CSVs `hybrid_*` |
| `generate_methodological_plots.py` | Gráficas metodológicas | `meta.json` |
| `generate_thesis_figures.py` | Predicción vs observación | `ilp_solution/*` |
| `generate_multiserver_summary.py` | Resumen multi-servidor | `reports/*/` |
| `plot_all_oom_vs_ilp.py` | Memoria GPU/CPU antes y después del ILP | `--input_root`, `--output_dir` |
| `plot_all_energy_tradeoff.py` | Energía y potencia por capa | idem |
| `plot_custom_user_figures.py` | Memoria, ancho de banda y top-15 | `--host`, `--models`, `--batches` |
| `organize_plots.py` | Reorganiza carpetas de figuras | `--host`, `--reports-root` |
| `launch_full_validation_campaign.sh` | Campaña por modelo: thesis_mode custom + convergencia | Modelos, batches y semillas fijos |
| `launch_grid5k.sh` / `_v2` | Lanzador en nodo Grid5000 (v2 añade convergencia) | `CAMPAIGN_PROFILE`, `FULL_SEEDS_CSV`, `DATA_MOUNT_SRC`… |
| `launch_slurm.sh` / `_v2` | Idem para Slurm | `MODULE_NAME`, `CONDA_EXE`, `GPU_VRAM_FALLBACK_MB=40960` |
| `run_thesis.sh` | Lado OAR: kadeploy + rsync + lanza `launch_grid5k.sh` | `KADEPLOY_FILE`, `SYNC_*` |
| `run_thesis_slurm.sh` | Cabecera `#SBATCH` y llamada a `launch_slurm.sh` | `--profile`, `--memory-model`, etc. |
| `run_thesis_smoke_workflow.sh` | Flujo reducido de 7 pasos | `MODELS`, `OPTIMIZERS`, `BATCH_SIZES`, `GPU_BUDGETS_MB` |

### 1.3 Cobertura de tests

Resultado: 125 passed, 7 skipped. **Ningún test cubre los scripts `.sh`** (clasificación OOM, perfiles, presupuestos, `FORCE_NO_GPU`). Tampoco cubre ninguno de los generadores de figuras salvo `--help`.

| Test | Cubre | No cubre / notas |
|---|---|---|
| test_dataset_registry, test_factory_input_policy, test_nlp_dataset_targets | Mapeo dataset↔modelo, política de datos reales vs sintéticos, máscara de padding | Descarga real |
| test_graph_extractor_export_backend | Exportación decoder-only; rechaza el grafo fallback por defecto | bert_base (el caso del "trace error") |
| test_hybrid_executor (678 líneas) | Plan CPU, recompute/checkpoint, modo DAG, auto-fallback HF, entradas estructuradas | GPU real: 2 tests con `skipif`/`if cuda` bifurcan comportamiento |
| test_hybrid_overhead_calibration, test_run_ilp_partition_hybrid_overhead | Calibración del overhead y su promedio entre runs | La rama GPU real depende de CUDA |
| test_ilp_data_loader_validity, test_ilp_energy_empirical_preservation, test_ilp_nextgen_features | Validez de entradas, preservación empírica, refinamiento jerárquico, congestión, coste direccional | Solver CBC end-to-end |
| test_phase4_activation / comparison / decoder_block_grouping / synthetic | Estrategias de activación y solver Phase4 | `setUpClass` busca `data/zephyr/results_smoke`, `data/test-m3*`: si no están, se omiten (3 skipped) |
| test_phase1_integrity | CLIs sweep/ablation/sensitivity | **4 skipped**: requiere `data/zephyr/results_smoke/...` y `.venv/bin/python` |
| test_plot_scripts_cli | `--help` de 3 scripts | Solo ayuda; no ejecuta ninguna figura |
| test_precision_policy_unit, test_profiler_gpu_only_precision_policy | Política fp16/bf16 y preflight CPU | — |
| test_report_assets | `best_hybrid_rows` | Bien acotado |
| test_schema_contracts | Esquemas de auditoría | — |
| test_sim_reference_linkage | `sim_*` ↔ plan | Usa `importorskip("torch")` |
| test_timeout_validation | **0 tests** (script manual con `main()`, sin funciones `test_*`) | Ver H-T1 |
| test_training_profiler_oom_fallback | `is_oom_runtime_error` en Python | **No** el equivalente shell `is_oom_failure_from_log` |
| test_training_profiler_transfer_model | Costes por mensaje y ramificación | — |
| test_loss_utils | Pérdida/accuracy | — |
| convergence tests (4) | Pinned cache, trazado único, grupos de checkpoint decoder, métricas de recursos, pesos atados | 3 de `resource_metrics` requieren CUDA; `smoke` requiere imagenette; la ILP-solution en la máquina |

Notas de ejecución:
- `pytest.ini` fija `testpaths = tests`, así que `convergence_validation/tests` **no** corre por defecto.
- Hay un marker `integration` sin uso real de filtrado.

---
## 2. HALLAZGOS

### 2.1 CRÍTICO

**C1. `FORCE_NO_GPU=true` persistente** — `run_experiments.sh:~711`
- Tras un rescate `--no_gpu` se fija `FORCE_NO_GPU=true` y no se resetea.
- Todas las configuraciones y modelos posteriores se perfilan sin GPU. Los datos de la campaña quedan contaminados sin aviso.
- Arreglo: usar una variable local por configuración y restaurarla al terminar. Etiquetar las filas con `profiling_mode=cpu_only_rescue`.

**C2. Timeout clasificado como "CPU OOM KILL"** — `run_experiments.sh:~676-690`
- `timeout --signal=KILL` devuelve 137 (verificado). La rama `exit 124` (l.~688) nunca se alcanza.
- Un timeout marca la configuración como `oom_cpu_skipped` y hace `break`.
- Arreglo: usar `timeout -k` con SIGTERM primero (124), o un wrapper que escriba el motivo. Para el OOM real, comprobar `dmesg`/cgroup `memory.events` (`oom_kill`) en lugar del código 137.

**C3. `is_oom_failure_from_log` mira el log global** — `run_experiments.sh:329-337`
- Hace `tail -n 200` del log de campaña, no de la ejecución actual. El regex es amplio ("out of memory", "cublas.*alloc", …).
- Cualquier OOM reciente, incluso uno recuperado por reintento, etiqueta como "OOM SKIP" un fallo de otra causa. Es el caso del bert_base: error de trace etiquetado "OOM SKIP". Además dispara el rescate `--no_gpu` (C1).
- Los marcadores `oom_skipped_config.json` persisten y la reanudación (l.591-606) salta esa configuración para siempre.
- Arreglo:
  1. Guardar el log por ejecución (`$RUN_DIR/stderr.log`) y clasificar solo sobre él.
  2. Exigir `CUDA out of memory` o `torch.OutOfMemoryError` como último error del traceback.
  3. Un fallo de trace/export debe ser `FAIL_TRACE` (reintentable con otro backend), no `OOM`.
  4. Escribir `failure_class` en el marcador.

**C4. Perfiles "strict" que no paran** — `run_thesis_mode.sh:241-248`
- `STRICT_METRIC_VALIDITY=false`, `FAIL_FAST_PROFILE_ERRORS=false`, `FAIL_FAST=false` y `ALLOW_PARTIAL_PROFILING_FAILURES=true` en minimal y full.
- `FAIL_FAST_PROFILE_ERRORS` ni siquiera se pasa a `run_ilp_partition.sh` (solo al Pareto).
- El script termina en 0 aunque falte la mitad del grid.
- Arreglo:
  - Los perfiles doctorales deben fijar `STRICT_METRIC_VALIDITY=true` y `FAIL_FAST_PROFILE_ERRORS=true`.
  - Al final, comparar contra `audit_experimental_grid` y salir con código ≠0 si la cobertura < 100 %, salvo OOM legítimos explícitamente registrados.

**C5. Sonda híbrida sin baselines ni clasificación; fallback con batch 8** — `run_thesis_mode.sh:1025-1038, 1064-1070, 1097, 1099`
- La sonda no usa `--compare_baselines` y marca `hybrid_already_executed=true`.
- `HYBRID_COMPARE_BASELINES` se ignora en el modo por defecto, así que no hay filas all_gpu/all_cpu.
- `generate_chapter6_prism_vs_allgpu_figures.py` pivota por run_label `all_gpu`/`ilp_plan`/`all_cpu` y falla con KeyError.
- Cualquier fallo se trata como OOM.
- El fallback `default_ilp` (l.1099) ejecuta con `HYBRID_BATCH_SIZE=8` aunque la config sea otro batch.
- Arreglo: pasar `--compare_baselines` también a la sonda. Derivar el batch del config_dir. Clasificar OOM vs otro con el log de la sonda y registrar la clase en el JSON.

**C6. `discover_ilp_config_dirs.sh:41` no encuentra la salida de thesis mode**
- Usa `-path "*/results/$MODEL/..."`, que no coincide con `results_thesis_mode`.
- Arreglo: parametrizar el patrón o usar `-path "*/$MODEL/*/*/batch_*"`.

**C7. `generate_thesis_figures.py` rutas incorrectas**
- Busca `config_dir/ilp_solution/<model>_pareto_summary.json`, pero el sweep escribe `config_dir/<model>_pareto_summary.json`. La figura "prediction_vs_observation" nunca se genera.
- En modo pareto_best el runtime está en `ilp_solution_pareto_best_budget_X/hybrid_execution`.
- Grafica `ilp_objective` (costo ponderado) como "Total Latency (ms)" frente a `avg_step_ms`.
- Claves `total_energy_j` y `gpu_mem_used_mb`/`peak_gpu_mem_mb` con default 0.0: barras en cero, silenciosas.

**C8. Datos inventados o de respaldo en figuras de tesis**
- `generate_methodological_plots.py`:
  - sin `meta.json` usa constantes de respaldo (alpha 0.05, beta 12, congested 6, knee 128 MB), solo con un warning;
  - `plot_overlap_tension` usa `t_base=10`, `kappa=0.5`, factor 0.5, `p_u=[0,1,2,4]` y anotaciones con coordenadas fijas.
  - Arreglo: marcar claramente como "ilustrativo" en la figura, o fallar sin meta.
- `generate_pareto_curve_ch5.py`: lee `/tmp/test_b256_grid.csv` (ruta temporal que desaparece), host `chuc-4` fijo, `xlim (0.2,7.5)` y `ylim (150,4200)` fijos. El título dice "Empirical"/"Pareto Optimal Frontier" pero usa `sim_*` (simulados) y solo ordena por tiempo, sin calcular no-dominancia.

### 2.2 ALTO

- **A1.** `run_thesis_mode.sh`: el Pareto corre sin Phase4 pero la materialización sí lo usa. Las filas `sim_*` del sweep no corresponden al plan ejecutado.
- **A2.** `GPU_MEM_BUDGET_MB=1e18` en el ILP principal: no se restringe la GPU. Solo el sweep usa presupuestos reales. Si el ILP principal es el "plan" que se compara, es en realidad all-GPU salvo por la jerarquía de tiempo. Pasar el presupuesto de safety ratio.
- **A3.** `CPU_MEM_BUDGET_MB=auto` lee `MemAvailable` en el momento: no reproducible. Registrarlo en `meta.json` y permitir fijarlo.
- **A4.** `run_ilp_partition.sh` / `run_ilp_pareto_sweep.sh` defaults W_TIME=1.0, W_ENERGY=0.0 frente a 0.5/0.5 de thesis_mode. El mismo script da planes distintos según quien lo invoque. `run_thesis_smoke_workflow.sh` usa 1.0/0.0. Centralizar en un archivo de config.
- **A5.** `generate_chapter6_prism_vs_allgpu_figures.py`:
  - infra hardcodeada {chuc-4: A100-SXM4, kinovis-2: L40S}, perfil `doctoral_full`, modelos sin bert_base;
  - solo régimen "feasible": excluye los casos donde all-GPU hace OOM, que es donde PRISM aporta;
  - `all_gpu_status` viene del ILP (predicho), no de un OOM real;
  - offsets de anotación y seed 7 fijos; posible desajuste en `zip(offsets, cols, labels, med.columns)`.
- **A6.** `plot_all_oom_vs_ilp.py` y `plot_all_energy_tradeoff.py`:
  - `if not has_intervention and total_naive_peak < 7500: continue` (7500 MB fijo; debería ser VRAM medida de `meta.json`);
  - `ilp_dirs = sorted(...)[0]`: elige el primer `pareto_best_budget_*` por orden alfabético, no el realmente ejecutado;
  - el título de energía dice "Native GPU (OOM Trajectory)" aunque no haya OOM;
  - el modelo de memoria es simplificado (retain/checkpoint suma activaciones en CPU); no equivale al modelo del ILP;
  - `except Exception: continue` silencioso y `len(metrics)!=len(assign)` se omite sin aviso.
- **A7.** `run_thesis.sh` / `run_thesis_slurm.sh`:
  - `rsync --delete` sobre `$PROJECT_ROOT` remoto, con excludes fijos;
  - `StrictHostKeyChecking=no`, `root@`;
  - `run_thesis_slurm.sh` fija `--nodelist=paccaA100` y `--time=48:00:00`.
- **A8.** `launch_grid5k.sh` y `_v2` usan `"$MODELS_CSV"` sin definir (`set -u`): si se lanzan directamente (sin `run_thesis.sh`) abortan con "unbound variable". En v1, `LOG_FILE` tiene una `}` sobrante (`...log}"`) que acaba en el nombre del archivo.
- **A9.** `launch_slurm.sh` no pasa `MODELS_CSV`, `MEMORY_MODEL`, `GPU_EXPANSION` ni `CALIBRATION_BATCH_SIZE` a `run_thesis_mode.sh`, aunque `run_thesis_slurm.sh` los acepta como flags. Esas flags quedan ignoradas (verificar en `_v2`).

### 2.3 Inventario de valores hardcodeados

**Modelos / grids**
| Valor | Dónde | Debería salir de |
|---|---|---|
| minimal: simple_mlp, resnet50, resnet152, vit_b16; full: + bert_base, gpt2_small, distilgpt2 | `run_thesis_mode.sh` l.~50-130 | `config/campaign_profiles.yaml` |
| 7 modelos × batches 8..256 × fp32/fp16/bf16 × 7 optimizadores | `run_experiments.sh` | idem; el run_thesis ya los sobreescribe, solo es default |
| batches 8,32,64 (minimal) / 8..512 (full) | `run_thesis_mode.sh` | Config; para full, descubrir el máximo por calibración (ya hay OOM backoff) |
| `HYBRID_BATCH_SIZE=8` | `run_thesis_mode.sh` | Debe ser el batch de cada config (ver C5) |
| smoke: simple_mlp, resnet50, SGD/AdamW, batch 8/32, GPU budgets 500,1000,2000, CPU 1e18 | `run_thesis_smoke_workflow.sh:49-77` | Medición (VRAM) o config |
| launch_full_validation: distilgpt2 16/64, gpt2_small 8/32, vit_b16 16/64, bert_base 16/64, AdamW, fp32, REPEATS=2, MEASURE=3, MAX_STEPS=200 | `launch_full_validation_campaign.sh:68-181` | Perfil; los batches provienen de la medición de OOM |
| Convergencia: resnet50, resnet152, distilgpt2, gpt2_small; regímenes A,B; semillas 42,43,44; EPOCHS=7 | `launch_grid5k_v2.sh:42-46`, `launch_slurm_v2.sh:40` | Config |
| `convergence_validation/run_convergence_study.sh`: BATCH_SIZE=64, GRADACCUM_MICRO_BATCH=8, EVAL_EVERY_STEPS=50, EVAL_BATCHES=10, SEQ_LENGTH=128, BF16_HOST_HINT=kinovis, PRECISIONS fp32,bf16 | l.47-150 | Config; `BF16_HOST_HINT` debería ser detección de capacidad |

**Presupuestos y límites (%)**
| Valor | Dónde | Origen ideal |
|---|---|---|
| Escalera 90,85,80,75,70,50 % de VRAM | `run_thesis_mode.sh:608-625, 674`; `run_ilp_pareto_sweep.sh:48` | Parámetro `GPU_BUDGET_PCTS`; ampliar por debajo del 50 % si hay OOM |
| `GPU_VRAM_SAFETY_RATIO_CSV=0.90,0.85`, `PHASE4_TARGET_GPU_UTILIZATION_CSV=0.95,0.90,0.85` | `run_thesis_mode.sh` l.~920 | Parámetro; derivarlo de la sonda |
| `CPU_MEM_LIMIT_PCT=95`; CPU budget = MemAvailable×0.9 | `run_thesis_mode.sh`, `run_convergence_study.sh:119` | Parámetro y registro en meta |
| `GPU_BUDGETS_MB` default 400,600,800,1000,1200 | `discover_ilp_config_dirs.sh` | Debe derivarse de la VRAM |
| `GPU_VRAM_FALLBACK_MB=40960` | `launch_slurm*.sh:16`, `run_thesis_slurm.sh:28` | Nodo (nvidia-smi); si falla, abortar |
| `GPU_MEM_BUDGET_MB=1e18`, `CPU_MEM_BUDGET_MB=1e18` | `run_ilp_partition.sh`, smoke | Parámetro obligatorio |
| `PEAK_ACTIVATION_OVERLAP=0.35` | `run_thesis_mode.sh` | Calibración (debe medirse) |
| Penalizaciones recompute 0.5 / checkpoint 0.3, `W_IO=0` | `run_thesis_mode.sh` | Calibración (existe `--calibrate_recompute_curve`) |
| `GPU_EXPANSION_BUDGET=200`, fracciones 0, .25, .5, .75, 1, `seq_len=128` | `run_thesis_mode.sh` | Config |
| `plot_all_*`: umbral 7500 MB | `plot_all_oom_vs_ilp.py`, `plot_all_energy_tradeoff.py` | `meta.json` (VRAM medida) |
| `generate_methodological_plots.py`: alpha 0.05, beta 12, congested 6, knee 128 MB, t_base 10, kappa 0.5 | l.~ varias | `meta.json` / fallar |
| `generate_pareto_curve_ch5.py`: xlim (0.2,7.5), ylim (150,4200) | script | Autoescala |

**Pesos / hiperparámetros / semillas**
| Valor | Dónde |
|---|---|
| `W_ENERGY=0.5`, `W_TIME=0.5`, `W_TRANSFER=1.0`, `K_SIGMA=1.0`, `HW_AGGREGATE=max` | `run_thesis_mode.sh` (y 1.0/0.0 en `run_ilp_*.sh`) |
| `LR_VISION=0.01`, `LR_LM=5e-5` | `run_thesis_mode.sh`, `run_convergence_study.sh:88-91` |
| `SEED_BASE=42`; semillas 42,43,44; `SINGLE_SEED=42`; seed 7 en el cap. 6 | varios |
| `REPEATS 5/7`, `WARMUP 3/5`, `MEASURE 10/15`; `FULL_REPEATS_PER_SEED=2`, `NON_FULL_REPEATS=1` | thesis_mode, launchers |
| `HYBRID_STEPS=10` | `run_thesis_mode.sh` |
| `SMOKE REPEATS=3, WARMUP=2, MEASURE=5` | smoke workflow |

**Tiempos y timeouts**
| Valor | Dónde |
|---|---|
| `PROFILER_TIMEOUT_SECS=0` (sin timeout) | `run_experiments.sh` |
| `find_free_gpus.py`: timeout SSH 45 s | script |
| `SSH_CONNECT_TIMEOUT=8`, `SSH_MAX_ATTEMPTS=40`, `SSH_RETRY_SECONDS=5` | `run_thesis.sh:104-106` |
| `#SBATCH --time=48:00:00`, `--cpus-per-task=16` | `run_thesis_slurm.sh:13-15` |
| Preflight CPU fp16: 60 s fase 1 y mínimo 10 s | `src/profiler.py` (ver test_timeout_validation) |

**Rutas absolutas y específicas del usuario/host**
| Valor | Dónde |
|---|---|
| `/home/ltorresnino/data`, `/home/ltorresnino/kadeploy_images/PRISM/rocky9_profiling.yaml`, `/home/ltorresnino` | `launch_grid5k*.sh:20/29`, `run_thesis.sh:22-23,44` |
| `/root/PRISM` | `run_thesis.sh:14` |
| `/home/latorresn/PRISM`, partición `GPU`, nodo `paccaA100` | `run_thesis_slurm.sh:9-10,22` |
| `/opt/ohpc/pub/Analytics/anaconda3/bin/conda` | `launch_slurm*.sh:23`, `run_thesis_slurm.sh:26` |
| `~/anaconda3`, `prism_env` | `run_thesis_mode.sh`, `run_thesis_smoke_workflow.sh:22-29`, `launch_full_validation_campaign.sh:9-10`, `validation/run_unit_tests.sh` |
| `ltorresnino@access.grid5000.fr`, `StrictHostKeyChecking=no`, lista de sitios | `find_free_gpus.py` |
| `/tmp/test_b256_grid.csv`, host `chuc-4`, copia a `data/resnet152_pareto_sweep_b256.csv` | `generate_pareto_curve_ch5.py` |
| `{chuc-4: A100-SXM4, kinovis-2: L40S}`, perfil `doctoral_full` | `generate_chapter6_prism_vs_allgpu_figures.py` |
| `data/test-m3-r2/simple_mlp/SGD/fp32/batch_8` | `run_ilp_partition.sh:12` |
| `data/test-m4/resnet50/...` | `run_ilp_pareto_sweep.sh:13` |
| `data/test-m4` | `generate_ilp_report_assets.sh`, `audit_experimental_grid.sh` |
| `reports/ilp_results/...` | `export_ilp_tables_latex.sh` |
| `--host zephyr`, `reports/zephyr/doctoral_minimal/...`, `data/zephyr/results_thesis_mode/doctoral_full` | `organize_plots.py`, `plot_custom_user_figures.py`, `plot_all_*.py`, `generate_methodological_plots.py`, `generate_thesis_figures.py` |
| Salida fija `doctoral_minimal` sin importar el perfil | `plot_custom_user_figures.py`, `organize_plots.py`, `generate_thesis_figures.py` |
| Escritura bajo `doctoral_minimal` pero lectura de `doctoral_full` | `plot_custom_user_figures.py` (base_dir full, out_dir minimal) |

### 2.4 MEDIO

- **M1.** Sin exit code global de fallos parciales en thesis_mode (ver C4).
- **M2.** `run_thesis_mode.sh` muestra el error con `tail -n 15` sobre el log compartido: puede mostrar el error de otra configuración.
- **M3.** La reanudación por `metrics_stats.csv` ignora cambios de REPEATS/WARMUP/MEASURE (`run_experiments.sh:591-606`).
- **M4.** `run_experiments.sh`: un fallo no-OOM no hace `break`; el agregador corre sobre réplicas parciales.
- **M5.** `run_ilp_partition.sh` añade flags de calibración aunque `PHASE4_ACTIVATION=false`; exige CBC sin comprobación previa clara.
- **M6.** `discover_ilp_config_dirs.sh` no propaga presupuestos CPU ni `memory_model`.
- **M7.** `generate_multiserver_summary.py`: baseline greedy con fallback a all_cpu, mezcla "OOM/Infeasible" con `ilp_status`, carga de CSV con `except Exception: continue` silencioso.
- **M8.** `generate_methodological_plots.py`: `_find_meta_json` toma el primer meta.json ordenado de cualquier modelo (ignora `preferred_model` si no coincide). Lista fija de tipos (Conv2d, BatchNorm2d, ReLU, Linear): NLP/ViT no se grafican.
- **M9.** `generate_thesis_figures.py`: el tiempo de transferencia se suma igual a fwd y bwd; top-15 fijo.
- **M10.** `plot_custom_user_figures.py`: `gpu_mem` es la **suma** de picos por capa (`gpu_mem_peak_mb_mean`.sum()), no el pico real; mezcla con CPU y ancho de banda en un único eje en MB, sin significado físico común. Usa `sns.barplot(palette=...)` sin `hue` (deprecado en seaborn ≥0.13).
- **M11.** `run_thesis_smoke_workflow.sh`: aborta en el primer fallo del profiler (sin manejo de OOM); numeración de pasos inconsistente ("STEP 0/6", "1/7").
- **M12.** `organize_plots.py`: `shutil.rmtree` de `advanced_analytics`/`thesis_figures` tras mover; `_infer_model_from_name` falla con nombres de modelo que contengan una KEYWORD (p. ej. "batch").
- **M13.** `find_free_gpus.py`: cuenta "recursos" aunque son nodos; las líneas de guiones (115 vs 95) son inconsistentes.

### 2.5 BAJO

- Funciones duplicadas (`normalize_output_dir_for_host_sh`, `is_vision_model`) entre `run_thesis_mode.sh` y `run_experiments.sh`.
- `mem_limit_mb` es `local` en `run_profiler_sandboxed`: siempre sale "unknown" en el JSON.
- `sanitize_cuda_env.sh`: solo quita rutas `stubs` y `CUDA_HOME`/`CUDA_PATH` iguales a `/usr/local/cuda`.
- `run_thesis.sh`: la ayuda no lista `--models` (que sí existe); `log_msg` se usa en `parse_args` antes de su definición textual (funciona porque se invoca después).
- `launch_grid5k.sh` y `_v2` casi idénticos (duplicación; igual para slurm).
- `generate_chapter6...py`: offsets de anotación manuales.
- `STRICT_…=true` en el smoke workflow, pero `ALLOW_LOW_QUALITY_STATS=true`.

### 2.6 Tests: rotos, desactualizados, dependientes de CUDA o rutas fijas

- **T1 (Medio)** `tests/test_timeout_validation.py` (247 líneas): no tiene ninguna función `test_*` (pytest recoge 0). Es un script manual con `main()` y `logging.DEBUG`. Además su docstring es incoherente: "ViT 250 ms → backward_timeout = max(10 s, 250 ms×2.0×2.5)=1.25 s (should timeout)". El máximo con 10 s da 10 s, no 1.25 s. Convertir en tests reales con tiempos bajos, o moverlo a `validation/`.
- **T2 (Medio)** `tests/test_phase1_integrity.py:11-14`: `PYTHON_BIN=.venv/bin/python` y `SMOKE_CFG=data/zephyr/results_smoke/...` fijos. Los 4 tests se omiten siempre en CI o en otra máquina. Generar un fixture sintético en `tmp_path` (como en `test_ilp_data_loader_validity`) en lugar de depender de `data/`.
- **T3 (Medio)** `tests/test_phase4_activation.py:78-83`: busca `data/zephyr/results_smoke`, `data/test-m3`, `data/test-m3-r2` con rutas relativas al cwd. 3 skipped ("Compatible Phase 4 fixtures not available"). Usar fixtures sintéticos versionados en `tests/fixtures/`.
- **T4 (Bajo)** CUDA: `test_hybrid_executor.py:617` (skipif), `test_hybrid_overhead_calibration.py:112` (skipif), 3 de `convergence_validation/tests/test_resource_metrics.py` (skipif), `test_tied_weight_dual_placement.py:101` (skip). En esta máquina corrieron. Sin GPU quedan sin cubrir las ramas GPU y no hay marker `gpu` para seleccionarlos.
- **T5 (Medio)** Cobertura ausente:
  - ningún test de `run_experiments.sh` (clasificación OOM / 137 / timeout / `FORCE_NO_GPU`);
  - ninguno de `run_thesis_mode.sh` (`compute_dynamic_budgets`, búsqueda de safety ratio, `best_feasible_row`);
  - ninguno de las figuras (solo `--help`, que no habría detectado C7);
  - no hay test de regresión del caso bert_base (trace error ≠ OOM).
  - Sugerencia: tests con `bats` o `pytest` + `subprocess` sobre funciones `source`adas, con logs sintéticos.
- **T6 (Bajo)** `pytest.ini`: `testpaths = tests` no incluye `convergence_validation/tests`; el marker `integration` no filtra. `tests/test_plot_scripts_cli.py` usa `sys.executable`, mientras `test_phase1_integrity` usa `.venv`: criterio inconsistente.
- **T7 (Bajo)** `tests/test_sim_reference_linkage.py` hace `importorskip("torch")` en el cuerpo, pero el resto de los tests importan torch al nivel del módulo (el Python base falla en la colección, no se omite).
- **T8 (Bajo)** `tests/test_training_profiler_oom_fallback.py` prueba el detector en Python, mientras el detector que decide el skip es shell (C3).

### 2.7 Documentación desactualizada

- `README.md:~224-230` (OOM): describe `OOM_SKIP_CONFIGS=true` como "cuando se agotan los reintentos OOM". En realidad marca OOM cualquier fallo si el log reciente contiene un patrón OOM (C3), y 137 siempre se marca como CPU OOM (C2).
- `README.md:~98-100`: llama a minimal y full "official strict doctoral profile", pero no paran ante fallos (C4).
- `README.md:~120, 151`: usa `data/<hostname>/results/...`, mientras thesis_mode escribe en `results_thesis_mode`.
- `README.md:55` y `docs/README.md:29`: `bash validation/run_unit_tests.sh` exige `~/anaconda3/envs/prism_env/bin/python`; los tests de convergencia no están en `testpaths`.
- `docs/SCRIPTS_DOCUMENTATION.md` (123 líneas): documenta 14 scripts. **No documenta** (en README ni docs): `launch_grid5k_v2.sh`, `launch_slurm.sh`, `launch_slurm_v2.sh`, `run_thesis_slurm.sh`, `launch_full_validation_campaign.sh`, `generate_pareto_curve_ch5.py`, y los generadores `plot_*`, `generate_*` (solo 1 mención en docs de `generate_chapter6…`). Dice `CONFIG_DIR` default `data/.../batch_8` sin indicar que está fijo en test-m3-r2.
- `docs/SCRIPTS_DOCUMENTATION.md:20`: "no requiere argumentos CLI estrictos…" (vago, no refleja las flags reales).
- `config/requirements.txt` fija `transformers>=5.0.0` y `huggingface_hub>=1.0.0`, y comenta `torch` sin versión. `prism_env` real: Python 3.10, torch 2.14+cu130. Ni `environment.yml` ni el README fijan esta combinación; verificar que coincidan.
- No hay `requirements.txt` ni `Makefile` en la raíz, pese a que `README.md` apunta a `config/`.

---
## 3. Prioridad de arreglos (orden sugerido)
1. C3 + C1 + C2: clasificación por log de ejecución, `FORCE_NO_GPU` local, no usar 137 para timeout.
2. C4: perfiles strict con parada real y exit code global.
3. C5 + A1/A2: sonda con baselines, batch de la config, y presupuestos reales en el ILP principal.
4. C6 + C7: rutas de `discover` y de `generate_thesis_figures`.
5. C8 + inventario (§2.3): mover valores a `config/campaign_profiles.yaml`; marcar figuras ilustrativas.
6. T1-T5: tests sintéticos, marker `gpu`, tests de shell para clasificación de fallos.
7. Documentación (§2.7).
