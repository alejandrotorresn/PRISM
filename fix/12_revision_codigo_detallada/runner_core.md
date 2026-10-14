# Revisión PRISM: src/runner, src/core, src/data, src/models (solo lectura)

Nota: los números de línea de `training_profiler.py` son aproximados (±10) salvo los indicados explícitamente.

## PARTE 1 — MAPA

### src/runner/training_profiler.py (1686 líneas)
| Símbolo | Línea ~ | Qué hace | Llamadores |
|---|---|---|---|
| `_is_oom_runtime_error` | 27 | Detecta OOM por subcadena en el mensaje | `_run_step_with_oom_fallback`, `_run_epoch`, `run_profiling` (duplicado en 3 sitios) |
| `_new_layer_stat` | 38 | Dict acumulador por capa (tiempos, mem, bytes, flops, counts) | `_register_hooks` |
| `_piecewise_transfer_ms` | 58 | t = alpha + size/beta; beta congestionada sobre `knee` | `_build_edge_transfer_costs` |
| `_build_branch_pressure_maps` | 76 | Presión de ramas (fan-in/out) por nombre de nodo | `_build_edge_transfer_costs` |
| `TrainingProfiler` | 93 | Clase principal | `src/profiler.py` |
| `_detect_shared_weight_groups` | ~110 | Pesos atados por `id(Parameter)` → `shared_weight_group` | `__init__` |
| `_disable_inplace_modules_for_backward_hooks` | ~140 | Desactiva `inplace` en ReLU etc. | `_run_epoch_attempt` |
| `_compute_loss` | ~190 | Loss sustituta (`loss_utils`) | `_run_step` |
| `_register_hooks` | 203–345 | Hooks fwd pre/post (CUDA Events + sync + `memory_allocated`/`reset_peak_memory_stats`) y bwd por tensor-hook en inputs del módulo (segmento encadenado) | `_run_epoch_attempt` |
| `_arm_backward_marker`/`_finalize_pending_backward_marker` | ~350 | Cierra/arma segmentos backward | hooks |
| `_build_optimizer` | ~420 | Crea optimizador (desconocido → SGD) | `_run_epoch_attempt` |
| `_infer_batch_size`/`_slice_batch` | ~440 | Batch y micro-batch | OOM fallback (duplicado en hybrid_overhead_calibration) |
| `_run_step` | ~470 | fwd+loss+bwd+opt.step | `_run_step_with_oom_fallback` |
| `_run_step_with_oom_fallback` | 495–570 | Divide a la mitad el micro-batch en OOM | `_run_epoch_attempt` |
| `_run_epoch`/`_run_epoch_attempt` | 600–720 | Optimizador+hooks+`EnergyMonitor`, sleep 0.05 s | `_profile_device_phase` |
| `_profile_device_phase`, `_merge_layer_stats`, `_compute_phase_overheads` | ~720–795 | Fase GPU/CPU, fusión, overhead de framework | `run_profiling` |
| `_export_graph_artifacts_safe` | ~790 | Envuelve `graph_extractor.export_graph_artifacts` | `run_profiling` |
| `_measure_pci_and_overlap` | ~795 | H2D/D2H + solape cómputo/comunicación | `run_profiling` |
| `_measure_pci_bandwidth_detailed` | ~850 | alpha/beta/knee H2D y D2H | `run_profiling` |
| `_measure_peak_flops` | ~920 | GEMM sintético (GPU N=8192, CPU N=2048) | `run_profiling` |
| `_save_gpu_partial_results`, `_cleanup_partial_artifacts`, `_save_skip_artifacts` | ~960–1190 | CSV/JSON parcial y de skip | `run_profiling` |
| `_build_edge_transfer_costs` | ~1190 | Coste de transferencia por arista | `run_profiling` |
| `run_profiling` | 1270–1686 | Orquestación completa y escritura de salidas | `src/profiler.py` |

**Qué se mide**
- Tiempo fwd/bwd por capa hoja: GPU con `cuda.Event` entre syncs; CPU con wall (`perf_counter`). 15 pasos de medición (`MEASURE_STEPS`), 5 de warmup (`WARMUP_STEPS`).
- Memoria GPU por capa: pico (`max_memory_allocated` − baseline) fwd y bwd. CPU: no se mide (`cpu_mem_mb = params + act`).
- Params/grads: `numel × dtype` de parámetros directos del módulo hoja. Optimizador: `params × OPTIMIZER_OVERHEAD_MAP[opt]`, estático. Activaciones: `output_bytes` de la primera llamada × batch_scale.
- Tamaño de tensor en aristas: `graph_edges.csv` `tensor_mb` = salida del productor (de `layer_stats`).
- Ancho de banda PCIe: `_measure_pci_bandwidth_detailed` (8/32/128/256 MB), `_measure_pci_and_overlap`.
- Energía: `EnergyMonitor` (NVML GPU + pyRAPL CPU) por fase, repartida por capa proporcionalmente al tiempo.

**Salidas**
- `{model}_metrics.csv`: layer, model, batch_size, run_id, seed, type, shared_weight_group, params_mb, grads_mb, optimizer_states_mb, activations_mb, theoretical_flops, tflops, efficiency_ratio, gpu_fwd_time_ms, gpu_bwd_time_ms, gpu_mem_peak_mb, gpu_bwd_mem_peak_mb, gpu_fwd/bwd_energy_j, cpu_fwd/bwd_energy_j, layer_j_per_tflop_gpu/cpu, dispatch_overhead_ratio, cpu_fwd_time_ms, cpu_bwd_time_ms, cpu_mem_mb, transfer_h2d_ms, transfer_d2h_ms, *_legacy, transfer_edge_aware_total_ms, remat_penalty_ms, precision_*, run_executed, skip_*, optimizer, opt_step_time_ms.
- `{model}_meta.json`: hardware, tiempos de step, framework overhead, energía total, `pcie_stats_raw`, info de OOM-retry, `analytical_fallback_gpu`, `gpu_tdp_w`, `measured_peak_tflops_*`.
- `{model}_graph_nodes.csv` (node_id, node_name, op_type, topo_index, params_mb, activ_out_mb, trace_source); `{model}_graph_edges.csv` (src_id, dst_id, tensor_mb, tensor_shape, producer_name, consumer_name, trace_source); `{model}_transfer_edges.csv` (H2D/D2H por arista, branch_pressure, sigma, alpha/beta/knee).
- `*_metrics_gpu_partial.csv` y `_meta_gpu_partial.json`: intermedios (se eliminan).
- `*_metrics_stats.csv` (stats_aggregator).
- Consumidores: `ilp/data_loader.py` (lee `gpu_tdp_w` en ~L533), `ilp/model_builder.py`, `validation/run_ilp_partition.py`.

### src/runner/hybrid_overhead_calibration.py (351)
- `measure_hybrid_overhead_ratio`: ratio reserved/allocated con plan alterno representativo. Usada por `profiler.py`.
- `measure_real_peak_for_plan`: pico real de un plan. Usada por `validation/run_ilp_partition.py`.
- Depende de `runtime.hybrid_executor` (`_try_symbolic_trace`, `_DeviceAwareFXInterpreter`).

### src/runner/recompute_curve_calibration.py (309)
- `measure_recompute_curve`, `build_ordered_candidate_blocks`, `select_gpu_resident_candidate_blocks`. Usadas por `ilp/model_builder.py` y `validation/run_ilp_partition.py`.

### src/core
- `graph_extractor.py`: `_build_fx_graph` (~128, `symbolic_trace` en 131), `_maybe_enrich_node_meta` (ShapeProp), `_build_fallback_graph`, `_build_export_grouped_graph`, `export_graph_artifacts` (~290).
- `decoder_export_backend.py`: `try_export_decoder_only_trace`, `is_decoder_only_export_candidate`, `build_export_node_layer_names`.
- `energy.py`: `EnergyMonitor` (hilo; NVML cada 50 ms; pyRAPL dominio package).
- `metrics.py`: `get_tensor_size_recursive`, `estimate_flops`.
- `loss_utils.py`: `compute_stable_surrogate_loss`, `compute_training_objective`, `build_deterministic_classification_targets`.
- `precision_policy.py`: sondeo ISA, `run_cpu_fp16_model_preflight`, `evaluate_precision_execution_policy`, `cpu_supports_bf16`, `get_cpu_fp16_support_info`, `probe_cpu_precision_support`.
- `system.py`: `set_determinism`, `configure_cpu_runtime`, `get_hardware_metadata`, `normalize_output_dir_for_host`.
- `stats_aggregator.py`: `aggregate_metrics_stats` (media/std/p50/p90/p95/CV/quality_flag por `GROUP_COLUMNS`).
- `constants.py`: WARMUP_STEPS=5, MEASURE_STEPS=15, BACKWARD_FACTOR=2.0, OPTIMIZER_OVERHEAD_FACTOR=2.0, OPTIMIZER_OVERHEAD_MAP, OUTPUT_DIR.
- `io_artifacts.py`: `write_csv_rows`, escritura JSON.

### src/models/factory.py (125) y src/data/dataset_registry.py (386)
- `build_model_input_target` / `build_model_and_input`: modelos cerrados por nombre, entrada sintética o real.
- `MODEL_DATASET_MAP`, `load_model_batch`, `download_required_datasets`.

---

## PARTE 2 — HALLAZGOS

### CRÍTICO

**C1. Fallo `symbolic_trace` en bert_base** (`graph_extractor.py:131`, `_build_fx_graph`)
- Reproducido (prism_env: torch 2.14.1, transformers 5.17.0, py3.10): `ValueError: code: varnames is too small` en `torch.fx._symbolic_trace._patch_function`.
- Causa raíz: el `forward` de `BertForSequenceClassification` está decorado por HF (`can_return_tuple`/`check_model_inputs`) con un `wrapper(self, *args, **kwargs)`. FX recrea el `CodeType` para funciones con varargs y falla porque `co_varnames` es demasiado corto.
- Un wrapper `nn.Module` propio con `input_ids` tampoco sirve: `TypeError: slice indices must be integers` (código HF dependiente de datos).
- `transformers.utils.fx` NO existe en transformers 5.x (`ModuleNotFoundError`), así que el fallback `hybrid_executor._try_huggingface_symbolic_trace` (~L1050-1077) es código muerto y `_try_symbolic_trace` devuelve None. Efecto: hybrid/calibraciones se degradan (ratio=1.0 o skip).
- `torch.export.export(m, (), {"input_ids": ids})` SÍ funciona para Bert (128 nodos) y `build_export_node_layer_names` mapea 31 de 33 capas hoja.
- `is_decoder_only_export_candidate` devuelve False para Bert por: (a) `get_output_embeddings()` es None en `BertForSequenceClassification`; (b) exige entrada dict pero `factory.py` entrega un Tensor.
- Con `allow_fallback_graph=False` (estricto) `export_graph_artifacts` lanza excepción.
- Arreglo:
  1. Para `PreTrainedModel` usar `torch.export` como camino primario; relajar `is_decoder_only_export_candidate` (quitar el requisito de `get_output_embeddings`, no exigir dict).
  2. Envolver el Tensor como `{"input_ids": x}` (+`attention_mask`) en `factory.py`.
  3. Reservar `symbolic_trace` para modelos no-HF.
  4. Sustituir el import muerto `transformers.utils.fx` (usar export; o `torch.fx` con `concrete_args`/tracer HF solo si `transformers<5`).
  5. `try_export_decoder_only_trace` traga excepciones: loguear el motivo y exponerlo en meta (`trace_source`, `trace_error`).

**C2. Nombres de nodo no únicos / aristas sin tamaño** (`graph_extractor.py` `_node_label`)
- Para call_function/call_method usa `str(node.target)`: todos los `add` comparten `<built-in function add>`. Los mapas de presión e `incoming_h2d_by_layer` se indexan por nombre y colisionan.
- Esos nodos (add residual, getitem, placeholder) no tienen `layer_stats` ⇒ `tensor_mb=0` en sus aristas (residuales sin coste de comunicación para el ILP). Tuplas: se suma el tamaño total, no el del elemento `getitem`.
- Fix: usar `node.name`; calcular `tensor_mb` desde `tensor_meta` de ShapeProp (shape × dtype), por elemento.

**C3. Aristas residuales perdidas** (`_build_export_grouped_graph`)
- El BFS con poda `nearest_depth` elimina skip-edges (A→LN vía add queda más profundo que A→siguiente lineal). El ILP no ve su coste de transferencia.
- Fix: conservar todas las aristas productor→consumidor a nivel de grupo (sin poda por profundidad) y deduplicar.

**C4. Fallback GPU por OOM rellena ceros** (`run_profiling`, `_save_skip_artifacts`)
- Con `gpu_oom_fallback_triggered` se omite la fase GPU y el CSV lleva `gpu_fwd/bwd_time_ms=0`, energía 0 y flops 0 (flops solo se leen de `g_s`). El ILP puede interpretar "GPU gratis". Fix: escribir NaN/vacío y `run_executed=False`, y que `data_loader` rechace/impute usando `analytical_fallback_gpu`. El stats_aggregator también promedia esos ceros.
- `tdp` cae a una constante de 250 W.

**C5. Atribución errónea del backward** (`_register_hooks` 203–345)
- Los segmentos se encadenan con tensor-hooks sobre los inputs del módulo:
  - Capas cuyos inputs no requieren grad (embeddings con ids, posicionales, primera conv) nunca cierran segmento; solo se vacía la primera hoja en orden fwd.
  - Un tensor que alimenta varios módulos (residual/ramas) dispara varios hooks: el primero recibe todo el segmento, los demás ≈0.
  - Ops funcionales (add, softmax, matmul de atención) no tienen hook: su tiempo se atribuye al vecino o a `framework_overhead`.
  - La loss usa solo la salida primaria (`last_hidden_state` en Bert): pooler/cabezas auxiliares tienen `bwd_count=0` y caen silenciosamente a `BACKWARD_FACTOR=2.0`. El modo `_use_backward_hooks=False` también es silencioso; el chequeo del string "BackwardHookFunction" es casi código muerto.
  - Un sync por hook serializa el pipeline.
- Fix: perfilar bwd con `torch.autograd.profiler`/`torch.profiler` (kernels CUDA por `nn.Module` vía record_function) o hooks `register_full_backward_pre_hook`+`register_full_backward_hook` por módulo; usar una loss que toque todas las salidas; marcar en el CSV `bwd_estimated=True` cuando se use el factor.

### ALTO

**A1. `gpu_mem_peak_mb_global` sin sentido** (`run_profiling`): `reset_peak_memory_stats` en cada hook ⇒ el valor final es el pico del último segmento. Guardar el pico global antes de los resets (o con un contador propio). `gpu_mem_reserved_mb_global`, `cpu_uss/pss_mb_global`, `efficiency_ratio_avg`, `avg_tflops…`, `energy_efficiency…` son 0 hardcodeados (placeholders engañosos).

**A2. Clasificación de OOM** (`_is_oom_runtime_error` L27 y duplicados en `_run_epoch`/`run_profiling`)
- Detección por subcadena del mensaje, triplicada. No cubre fallos de CPU ("DefaultCPUAllocator: can't allocate memory") ni `torch.OutOfMemoryError` por tipo en todos los casos; por otro lado un RuntimeError "out of memory" de otra causa (p.ej. cuDNN workspace, CUBLAS_STATUS_ALLOC_FAILED) y errores de bugs se mezclan.
- Los frames de la excepción mantienen memoria viva durante el fallback: `del exc; gc.collect(); empty_cache()` antes de reintentar.
- En el reintento cada paso arranca con el batch completo y vuelve a fallar; los `layer_stats` acumulan intentos fallidos (count/tiempo) y la duración incluye reintentos.
- La memoria se escala por `target/chunk` pero el TIEMPO no (se subestima con micro-batching).
- `_oom_retry_events` se acumula entre warmup, GPU y CPU.
- `_measure_peak_flops` traga errores devolviendo 0.0.
- En `hybrid_overhead_calibration` un OOM real devuelve `status="error"`, indistinguible de bugs.
- Fix: función única `classify_failure(exc)` → {oom_gpu, oom_cpu, other} por tipo (`torch.cuda.OutOfMemoryError`) + mensaje; resetear stats al reintentar; registrar `failure_kind` en meta.

**A3. Promedios con módulos reutilizados**: el CSV usa `/count` (media por llamada) mientras `_compute_phase_overheads` usa `/measure`. Un módulo llamado varias veces por step (ReLU compartida) queda infra-contado en tiempo, activaciones (solo 1ª llamada) y memoria. Fix: reportar por-step (suma/measure) y por-llamada (columna `calls_per_step`).

**A4. Calentamiento y varianza**: warmup solo en `cuda` (o `cpu` si no hay GPU); la fase CPU incluye init de oneDNN/allocator. `cudnn.deterministic=True`, `benchmark=False` sesgan tiempos GPU. 15 pasos sin std/mediana por capa (el CV solo existe entre ejecuciones en el agregador). Tiempos de step incluyen overhead de hooks/sync ⇒ `framework_overhead` distorsionado. Fix: warmup por dispositivo, guardar mediana/std/p95 por capa, y medir una pasada sin hooks para el tiempo de step real.

**A5. Calibración PCIe** (`_measure_pci_*` ~795–905)
- Se ejecuta DESPUÉS del perfilado, una vez, sin controlar estado PCIe.
- Unidades inconsistentes: fallback `pci_bw_mb_s=12.0` (MB/s, absurdo) frente a medición en MB/s; `beta_*` en MB/ms pero columnas `beta_*_mb_s`.
- Overlap: `t_comp`/`t_comm` sin warmup y `alpha_est=0.05` fijo.
- Detallada: tamaños 8/32/128/256 MB, una muestra, beta con solo los dos primeros puntos, alpha extrapolada (no se miden KB–MB, que son los tamaños típicos de aristas), knee fijo 128 MB; fallback beta=12.0/alpha=0.05 en el cuerpo y como defaults en `_build_edge_transfer_costs`.
- D2H va a memoria pageable mientras H2D sale de pinned.
- `overlap_factor = 1-0.5σ`, fórmula de branch_pressure (pesos 0.5, knees reutilizados entre H2D y D2H) y H2D legacy (params/beta) son heurísticas sin validación.
- El valor por capa mezcla edge-aware (activaciones) con legacy (params) según aparezca la capa en las aristas.
- Fix: regresión lineal (alpha, beta) con ≥8 tamaños (4 KB…512 MB) y ≥10 repeticiones, mediana; pinned en ambos sentidos; unidades unificadas (MB/ms); knee medido.

**A6. Energía** (`energy.py`)
- NVML: solo potencia de placa de la GPU, muestreo 50 ms con `nvmlDeviceGetPowerUsage` (promedia ~1 s en muchas GPUs). Las excepciones añaden lecturas de 0.0 W ⇒ media sesgada a la baja. Energía = `avg_power × duración`, no integral (usar `nvmlDeviceGetTotalEnergyConsumption` en Volta+). `nvmlShutdown` solo si hubo lecturas.
- pyRAPL: solo `pkg[0]` (socket 0, sin DRAM); requiere `intel-rapl` y permisos de lectura (`energy_uj` restringido desde kernel 5.10); no funciona en AMD. `rapl_available` en meta solo comprueba `find_spec("pyRAPL")`.
- **Por qué 0 W en LLM**: si RAPL falla ⇒ `avg_power=0` ⇒ `total_energy=None` y la energía CPU por capa se escribe como 0.0 (no NaN), "gratis" para el ILP. Se suma el fallback GPU por OOM (energía 0.0, C4), `--skip_cpu`/fase CPU omitida, y que la energía CPU no se mide durante la fase GPU.
- La energía por capa es energía de step repartida proporcional al tiempo, no medida.
- `gpu_bwd_energy_j` con factor fallback usa un denominador que no incluye ese tiempo.
- `cpu_fwd_power_w`/`cpu_bwd_power_w` solo se leen en `_save_skip_artifacts` (L~1178).
- `gpu_id=0` hardcodeado en `get_hardware_metadata`.
- Fix: NaN cuando no hay medida; `nvmlDeviceGetTotalEnergyConsumption`; leer `/sys/class/powercap/intel-rapl*` (todos los sockets + dram) o `perf`; `rapl_available` real (probar lectura).

**A7. `opt_step_time_ms` mal etiquetado**: en el CSV es la SUMA de `measure` pasos (no la media) y la última época (CPU) lo sobrescribe, no hay uno por dispositivo.

**A8. Memoria infra-contada**: parámetros directos de módulos no hoja (cls_token/pos_embed de ViT) y buffers (BN stats) no se cuentan. Estado del optimizador = factor estático (`OPTIMIZER_OVERHEAD_MAP`; fuera del mapa → 2.0; `_build_optimizer` con nombre desconocido → SGD sin avisar). `activations_mb` = solo la salida del módulo en la 1ª llamada (excluye intermedios guardados para backward). Pesos atados duplicados salvo que el ILP use `shared_weight_group`. `cpu_mem_mb = params + act` (faltan grads y optimizador; RSS no se mide). Fix: medir con `memory_allocated` antes/después de fwd (saved tensors) y `torch.cuda.memory_stats`; contar buffers y `recurse=False` completo.

**A9. FLOPs** (`metrics.py`): solo Conv2d, Linear, ReLU/GELU, BN/LN y MHA. Embedding, Conv1D de HF (GPT-2), pooling, softmax, RMSNorm, SiLU ⇒ 0. Módulos llamados con kwargs no tienen `inp` en el hook ⇒ flops 0 y sin hook backward. Fix: `torch.utils.flop_counter.FlopCounterMode` por módulo.

**A10. Sync por capa / dispatch**: tiempo GPU = `Event.elapsed_time` entre eventos sincronizados (correcto pero serializa y elimina el solape real); tiempo "kernel" CPU = wall ⇒ `dispatch_overhead_ratio=0` en CPU.

### MEDIO

- **`precision_policy.run_cpu_fp16_model_preflight`**: convierte el modelo COMPARTIDO a fp16 y ejecuta `opt.step` en un hilo daemon; tras timeout el hilo sigue mutando el modelo y consumiendo CPU durante el perfilado CPU. Loss distinta (`_extract_loss_for_preflight`); reutiliza `BACKWARD_FACTOR` como factor de timeout; timeouts mágicos (`FORWARD_PREFLIGHT_TIMEOUT_S=60`, `BACKWARD_PREFLIGHT_MIN_TIMEOUT_S=10`, factor 2.5). `cpu_supports_bf16`/`get_cpu_fp16_support_info` duplican `probe_cpu_precision_support`. Fix: usar una copia (`deepcopy`) del modelo y un subproceso con kill.
- **`loss_utils.compute_stable_surrogate_loss`**: `logits.float().square().mean()` (no es CE; targets sin uso; copia fp32 de B×S×V en LLMs, riesgo de OOM artificial). Salida sin tensores → `torch.tensor(0.0, requires_grad=True)` en silencio (backward nulo). `build_deterministic_classification_targets` usa multiplicadores hash mágicos.
- **`stats_aggregator`**: `MIN_RECOMMENDED_RUNS=2` justificado solo por Grid5000; `HIGH_CV_THRESHOLD=0.30` fijo; p90/p95 con n=2 sin sentido; ejecuciones con fallback GPU (ceros) contaminan medias; edges/transfer_edges no se agregan.
- **`graph_extractor`**: ShapeProp ejecuta un forward completo en CPU en modo train (muta BN y RNG) y los fallos se silencian (`except Exception: pass`). `_build_fallback_graph` fabrica cadena secuencial (errónea con residuales). `edge.tensor_mb` siempre = salida del productor.
- **`hybrid_overhead_calibration`**: una pasada sin warmup (workspace cuDNN/cuBLAS infla `reserved`); loss `out.sum()`; plan alterno hardcodeado y flags async/prefetch=True fijos; llama al privado `_DeviceAwareFXInterpreter` por posición (frágil); ratio medido con batch reducido se aplica al batch objetivo; tras OOM los frames retienen memoria (`empty_cache` sin `gc.collect`); duplica `_infer_batch_size`/`_slice_batch`.
- **`recompute_curve_calibration`**: un solo paso sin warmup ni repetición; fracciones por defecto (0,.25,.5,.75,1); `exclude_substrings=("downsample",)` específico de ResNet; agrupación de bloques por último segmento numérico duplicada en 3 archivos; `lr=1e-6` por defecto; `model.to("cuda:{gpu_id}" if device_count>1 else "cuda")`.
- **`factory.py`**: mapa cerrado de nombres de modelo; `num_labels=4`; `bert-base-uncased`/`gpt2`/`distilgpt2` con `from_pretrained` (requiere red); `torch.randint(0,1000)` como ids; entrada NLP = Tensor sin `attention_mask` (y no dict, lo que rompe el candidato export, ver C1); fallback silencioso a datos sintéticos en `FileNotFoundError` (queda en `input_source`); modelo con `.to(dtype)` (fp16/bf16 puro, no AMP); dimensiones de SimpleMLP fijas.
- **`dataset_registry.py`**: URLs y `MODEL_DATASET_MAP` hardcodeados; un solo batch con las primeras `batch_size` muestras, `shuffle=False`, sin seed; AG News: primeras filas del CSV, labels `int(raw[0])-1`; `tarfile.extractall` sin validar rutas (path traversal); fallback con subprocess `wget`/`curl`; `AutoTokenizer.from_pretrained` online; `_repeat_to_batch` repite muestras (batch con duplicados).
- **`system.py`/`energy.py`**: `import pynvml` obligatorio a nivel de módulo (rompe en máquinas sin NVML); `set_determinism` seed 42 + cudnn determinista; `set_flush_denormal(True)` altera numéricos; threads interop `min(4,n)` fijo; nº de hilos CPU no se guarda en el CSV (necesario para reproducir tiempos CPU).

### BAJO / código muerto / fallbacks silenciosos

- `'measured_gpu_peak_tflops' in locals()` (frágil); `_measure_peak_flops("cuda")` se ejecuta dos veces y descarta el 2º; pico CPU con N=2048 fp32 sea cual sea la precisión; GPU N=8192, ITER=5 sin control de TF32/precisión.
- `self.args.cpu_precision_executed` accedido directo (AttributeError) mientras el resto usa `getattr`.
- `remat_penalty_ms = t_fwd_gpu` (sin medir recompute real; ya existe `recompute_curve_calibration`).
- Defaults `run_id="run_001"` y seed 42 en filas.
- `_run_epoch`: fallback con while-loop poco claro.
- `io_artifacts.write_csv_rows`: `os.makedirs(os.path.dirname(path))` falla si la ruta no tiene directorio.
- `hybrid_executor._try_huggingface_symbolic_trace`: muerto con transformers 5.x.
- Excepciones tragadas: `try_export_decoder_only_trace`, ShapeProp, `_measure_peak_flops`, lecturas NVML (añaden 0.0).

## Tabla de constantes mágicas y su origen recomendado
| Valor | Lugar | Debe venir de |
|---|---|---|
| beta=12.0, alpha=0.05, knee=128 MB | training_profiler (PCIe y `_build_edge_transfer_costs`) | medición (regresión multi-tamaño) |
| `alpha_est=0.05` overlap | `_measure_pci_and_overlap` | medición |
| tamaños 8/32/128/256 MB | `_measure_pci_bandwidth_detailed` | parámetro + sweep KB→GB |
| WARMUP=5, MEASURE=15 | constants.py | CLI + criterio de convergencia (CV) |
| BACKWARD_FACTOR=2.0, OPTIMIZER_OVERHEAD_MAP | constants.py | medición (bwd real; `torch.cuda.memory_stats` tras `opt.step`) |
| TDP 250 W | fallback `gpu_tdp_w` | `nvmlDeviceGetPowerManagementLimit` |
| N=8192/2048, ITER=5 | `_measure_peak_flops` | parámetro por precisión |
| sleep 0.05 s | `_run_epoch_attempt` | sincronización explícita del monitor |
| seed 42, run_id "run_001" | system.py / rows | CLI |
| timeouts 60/10/2.5 | precision_policy | parámetro |
| `num_labels=4`, `randint(0,1000)`, nombres de modelo | factory.py | config por modelo/dataset y `config.vocab_size` |
| `gpu_id=0` | system.py | argumento |
| `MIN_RECOMMENDED_RUNS=2`, `HIGH_CV=0.30` | stats_aggregator | parámetro |
| `exclude_substrings=("downsample",)`, fracciones, lr=1e-6 | recompute_curve_calibration | parámetro |

## Prioridad de arreglos
1. C1 (export para HF + entrada dict) y C2/C3 (grafo correcto con tamaños de arista).
2. C4/A2 (no escribir ceros; clasificación OOM correcta).
3. C5/A4 (medición bwd fiable y varianza).
4. A5/A6 (PCIe y energía medidos de verdad, NaN si no hay medida).
5. Resto (memoria/flops completos, limpieza de constantes y código muerto).
