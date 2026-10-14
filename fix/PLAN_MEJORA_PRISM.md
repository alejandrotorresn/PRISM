# Plan de mejora de PRISM (paso a paso)

Objetivo de PRISM que este plan debe restaurar:

1. Distribuir las capas entre GPU y CPU **teniendo en cuenta el grafo** (aristas, tamaños de tensor, cortes).
2. **Minimizar el tiempo de ejecución y el uso de memoria.**
3. **Dar prioridad a la GPU**, sin producir OOM.
4. **Nada hardcodeado**: todo valor es un parámetro (CLI/config) o sale de resultados escritos de mediciones reales.

Base: revisión del código (anexo en [`12_revision_codigo_detallada/`](12_revision_codigo_detallada/)) y análisis de `data/zephyr`, `reports/zephyr` y `logs/zephyr`.
Después de este plan se cruzará cada fase con los 42 ítems de [`PLAN_ACCION_JURADO.md`](PLAN_ACCION_JURADO.md).

---

## 0. Aclaración: qué es la "verificación de igualdad" del dual placement

Una capa con `device_forward != device_backward` (por ejemplo forward en GPU y backward en CPU) es el caso "mixto". Ocurre **en cada paso**, no en pasos distintos. La implementación está en `_run_layer_with_dual_placement` (`src/runtime/hybrid_executor.py:559`) y funciona así:

| Etapa | Qué hace |
|---|---|
| Forward (dispositivo A) | Ejecuta la capa en A. Guarda en CPU una copia de la entrada y otra de la salida. |
| Backward (dispositivo B) | Mueve la capa a B y **recomputa el forward en B** a partir de la entrada guardada. Calcula los gradientes con esa recomputación. |
| Comparación (líneas ~639–660, 943–946) | Compara la salida guardada del forward con la recomputada en B (`atol=0.15`, `rtol=0.05`). Si no coinciden lanza `RuntimeError`. |

- La **recomputación** es necesaria para la correctitud: los tensores que autograd guardó viven en A y no se pueden mover a B. Es el coste real del modo mixto y debe entrar en el ILP.
- La **comparación** es solo una verificación (test de cordura). No cambia los gradientes. Hoy se ejecuta en **cada paso y cada capa dual**, con copia D2H de la salida, transferencia de vuelta y `.item()` (sincronización). Ese coste extra no lo modela el ILP.
- `--verify_dual_every N` significa: hacer la comparación solo cada N pasos (N=0 desactiva, N=1 la hace siempre, como hoy). En los pasos sin verificación se omite la copia de la salida y los `.item()`. **La recomputación y los gradientes no cambian.** Los experimentos de tesis correrían con la verificación espaciada y los tests con N=1.
- Con N>1, el snapshot de entrada se sigue guardando (hace falta para recomputar), y el de salida se guarda solo en los pasos verificados.

---

## 1. Doble verificación de los hallazgos

Cada hallazgo se contrastó contra el código y los datos **por mí** en esta sesión, salvo los marcados "Pendiente", que vienen de los agentes de revisión y deben reproducirse con un test antes de implementar el arreglo (paso 0.3).

| # | Hallazgo | Estado | Evidencia |
|---|---|---|---|
| H1 | Los planes Pareto de LLM salen todo-CPU por post-proceso, no por el solver | **Verificado** | `validation/run_ilp_partition.py:265-372`: si la medición no es `ok`, entra a recorte y en la rama `else` (L~366) hace `hi = n` (todas las capas a CPU). Log: `removed 113 additional layer(s)`. `ilp_assignment.csv` de gpt2_small b32 en el plan Pareto: todo CPU. El resumen conserva objetivo 100 ms y 5585 MB del plan GPU. |
| H2 | vit_b16 no tiene atención en el plan | **Verificado** | `ilp_assignment.csv` de batch_64: 101 líneas, 0 coincidencias con `attn`/`attention`/`out_proj`. `DevicePlan.resolve_torch_device` (`device_plan.py:36`) usa `assignment.get(layer_name, "CPU")`: toda capa ausente va a CPU sin aviso. |
| H3 | El optimizador del runtime es SGD fijo | **Verificado** | `hybrid_executor.py:2202` y `:2518`: `torch.optim.SGD(model.parameters(), lr=lr)`. Los experimentos usan AdamW. |
| H4 | La tolerancia del dual placement está fija | **Verificado** | `atol, rtol = 0.15, 0.05` (L653) y `rtol=0.05, atol=0.15` (L946). |
| H5 | El camino `call_module` del dual placement no excluye Dropout | **Verificado** | La guarda `_is_stochastic_operator` solo se aplica al camino por operadores (L2133). En L2031–2053 (`call_module`) no hay guarda. |
| H6 | Los grupos de checkpoint ignoran el dispositivo | **Verificado en el código leído** | La clave de agrupación es estrategia + bloque (L1237–1275). El aviso "spans >1 device" aparece 90–150 veces por corrida. El detalle de `_prepare_segment_fn` (~1596) queda para el paso 3.2. |
| H7 | El sweep no escribe `sim_time_ms`/`sim_energy_j`/`sim_status`; el runner usa `ilp_objective` como si fueran ms | **Verificado** | `grep sim_` en `sweep_ilp_pareto.py` sin resultados. `run_hybrid_execution.py:193-196` toma `row.get("ilp_objective")` como `sim_time_ms`. |
| H8 | El solver tiene límites fijos y traga excepciones | **Verificado** | `solve.py:130` `timeLimit=300, gapRel=0.01`; `solve.py:957` `except Exception` en `backend="auto"` con fallback a `_solve_exhaustive_dual`. |
| H9 | La clasificación OOM mira el log global de campaña y `FORCE_NO_GPU` persiste | **Verificado** | `run_experiments.sh:46` define un único `LOG_FILE` por campaña. L329-337 hace `tail -n 200` de ese archivo. L696 lo usa para marcar OOM. L711 pone `FORCE_NO_GPU=true` y no lo restablece. |
| H10 | `HYBRID_BATCH_SIZE` fijo en 8 | **Verificado** | `scripts/run_thesis_mode.sh:347`. |
| H11 | `symbolic_trace` falla en bert_base | **Verificado en logs** | `ValueError: code: varnames is too small` en `graph_extractor._build_fx_graph`; el agente lo reprodujo con torch 2.14.1 y transformers 5.17.0. Pendiente reproducir el arreglo con `torch.export`. |
| H12 | Etiquetas de nodo colisionan (`str(node.target)`) | **Verificado** (datos) | `resnet50_graph_nodes.csv` (b64): 47 de 176 `node_name` duplicados (16 `<built-in function add>`, 3 por cada `layerX.Y.relu`). Causa: `graph_extractor.py:38-44`. |
| H13 | `tensor_mb=0` en add/placeholder | **Verificado** (datos) | `resnet50_graph_edges.csv` (b64): 18 de 191 aristas con `tensor_mb=0`, p. ej. `add -> layer1.0.relu` con forma 64x256x56x56 (≈205 MB) y `x -> conv1`. Esas aristas tienen coste de corte 0 en el ILP. La poda BFS de aristas residuales sigue **pendiente**. |
| H14 | El dual placement añade un sobrecoste grande | **Verificado en parte** (micro-benchmark) | Capa Conv+BN, entrada 16x64x56x56, media de 4 pasos: normal 2,4 ms; dual GPU→GPU 17,2 ms; dual GPU→CPU 69,8 ms. Queda **pendiente** separar cuánto es recomputación y cuánto verificación, y que sea la causa principal del backward 1,4–3,3× del campaña (paso 3.1.1). |
| H5b | Dropout en dual placement falla | **Verificado** (ejecución) | `_run_layer_with_dual_placement` con `nn.Dropout(0.5)` en train: `RuntimeError: Dual-device recompute mismatch ... max_abs_diff=8.2`. |
| H5c | BatchNorm en dual placement actualiza las estadísticas dos veces | **Verificado** (ejecución) | `BatchNorm1d` en train: `num_batches_tracked` pasa de 1 (forward) a 2 tras el backward, y `running_mean` se actualiza otra vez con la recomputación. |
| H15 | Memoria topológica sobreestima (15344 vs 9739 MB) y subestima (sin optimizador/prefetch) | **Pendiente** | Cifras de logs vistas por el agente. Reproducir contra `peak_gpu_mb` medido. |
| H16 | Backward mal atribuido en capas con input sin grad, y `BACKWARD_FACTOR=2.0` silencioso | **Pendiente** | Coincide con el documento `fix/10`. |
| H17 | Energía: RAPL solo `pkg[0]`, ceros de excepciones, TDP 250 W de respaldo | **Pendiente** | Reportado por el agente; verificar en la máquina. |
| H18 | Planes con `device_forward != device_backward` están casi ausentes en la campaña nueva (8 capas en 1 de 24 planes) | **Verificado** (datos) | Ver informe anterior. |
| H19 | `scripts/` y varios tests solo se revisaron con `grep`/`awk` | **Limitación** | La revisión línea a línea completa de `scripts/`, `docs/`, `config/` queda como tarea 7.0. |

Cualquier hallazgo "Pendiente" que el test de reproducción no confirme se descarta del plan.

---

## 2. Plan por fases

Convenciones: `[ID]` identifica la tarea. "Aceptación" es el criterio verificable. Orden: cada fase solo se inicia cuando la anterior cumple su aceptación, salvo donde se indica paralelismo.

### Fase 0. Preparación

- **[0.1]** Crear la rama de trabajo. Mover los resultados actuales a `data|reports/zephyr_v2_pre_fix` (solo etiquetar, sin borrar).
- **[0.2]** Crear `config/prism.yaml` (y su loader) con todos los parámetros que hoy están en el código. Cada parámetro tiene el valor por defecto en un solo sitio, con nombre, unidad y origen (CLI, medición o derivado). Los resultados deben guardar la configuración efectiva usada (`effective_config.json`).
- **[0.3]** Tests de reproducción (deben **fallar** antes del arreglo) para H1, H2, H5, H6, H7, H9, H11, H12 y H13. Cada test usa un modelo pequeño y no requiere GPU salvo marca explícita.
- **Aceptación:** los tests existen, fallan por la razón correcta y están en `pytest.ini`.

### Fase 1. Bugs que invalidan los resultados

- **[1.1] Post-proceso del plan (H1).** En `_validate_final_plan_against_real_overhead`:
  1. Si `measured["status"] != "ok"`, no modificar el plan. Registrar `validation_status="unmeasurable"` y validar con el modelo de memoria calibrado (Fase 2.3).
  2. Eliminar la rama `hi = n`. Si ni vaciar la GPU cabe, abortar con error explícito.
  3. Tras cualquier cambio del plan, recalcular `objective_value`, `gpu_mem_used_mb`, `cpu_mem_used_mb`, cortes y `activation_strategy`.
  4. El probe de safety-ratio debe rechazar un plan con 0 capas en GPU si existe uno factible con capas en GPU.
  - Aceptación: distilgpt2 y gpt2_small producen planes con capas en GPU, y el resumen coincide con el CSV (test que recalcula el objetivo desde el CSV).
- **[1.2] Cobertura del plan (H2).**
  1. El ILP debe asignar **todas** las hojas ejecutables del grafo FX, incluyendo `nn.MultiheadAttention` y cualquier módulo que FX trate como hoja.
  2. `DevicePlan` lanza error si un nodo `call_module` ejecutado no está en el plan (flag `--allow_unplanned_layers` solo para depuración, con aviso). Los nodos `call_function`/`placeholder` no son capas del plan y siguen el dispositivo de su productor. `resolve_torch_device` tiene 10 llamadores en `hybrid_executor.py`; el cambio se hace en un único punto y se prueba con los modelos existentes antes de generalizarlo.
  3. Validar plan contra el grafo ejecutado y no contra `named_modules()`.
  - Aceptación: vit_b16 con 100 % de nodos `call_module` cubiertos, sin avisos de capas ausentes.
- **[1.3] Grafo del ILP (H12, H13).** Nombres de nodo únicos (`node.name`), `tensor_mb` real de todos los nodos desde ShapeProp, aristas residuales conservadas, y transferencia desconocida = error (no 0).
  - Aceptación: en resnet50, el número de aristas del CSV iguala el del FX; ninguna arista con tamaño 0 salvo las de escalares.
- **[1.4] Clasificación de fallos (H9).**
  1. Un único clasificador (módulo Python o función de shell) que lea el log **de esa ejecución**, el código de salida y la señal.
  2. Categorías: `GPU_OOM`, `CPU_OOM`, `TRACE_ERROR`, `TIMEOUT`, `CODE_ERROR`.
  3. Solo `GPU_OOM` activa el rescate `--no_gpu` y solo para esa configuración. `FORCE_NO_GPU` no persiste entre configuraciones.
  4. Los marcadores de salto se escriben con la categoría y se ignoran en una reanudación si el motivo no era OOM.
  - Aceptación: bert_base se marca `TRACE_ERROR` y la campaña continúa con el resto en GPU.
- **[1.5] Modelos HF (H11).** Usar `torch.export` como camino principal para bert_base y demás modelos HF. Aceptar entrada dict (`{"input_ids": x}`), relajar el candidato que exige `get_output_embeddings()` y eliminar el fallback muerto de `transformers.utils.fx`.
  - Aceptación: bert_base se perfila y se ejecuta end-to-end en b16 y b64.

### Fase 2. Fidelidad del modelo ILP

- **[2.1] Objetivo en ms.** Un solo tiempo estimado (ms): `makespan = max(ruta CPU, ruta GPU) + transferencias no solapadas`, o una suma serial documentada con su límite de validez. Energía y memoria como restricciones o segunda fase lexicográfica, nunca sumadas sin normalizar.
- **[2.2] Prioridad GPU declarada.** Minimizar tiempo con `memoria_pico ≤ presupuesto`. Desempate: menos capas en CPU, luego menos cortes. Test: si todo cabe en GPU, el plan es 100 % GPU.
- **[2.3] Modelo de memoria único y calibrado.**
  1. Incluir parámetros, gradientes, estados del optimizador según el optimizador real, activaciones, buffers de transferencia y prefetch.
  2. Calibrar con las mediciones (`hybrid_overhead.json`) y escribir el error entre pico predicho y medido en cada corrida.
  3. Reportar el error en el resumen del plan.
- **[2.4] Solver honesto (H8).** Límite de tiempo y gap como parámetros. Reportar `status` real (`optimal`, `time_limit`, `infeasible`) y el gap final. Eliminar el `except Exception` general: si PuLP/CBC falla se informa el error y no se cambia de algoritmo sin avisar. Reemplazar el límite de 14 nodos del exhaustivo por una política explícita.
- **[2.5] Fase 4 (recompute/checkpoint).** Penalización de recompute a partir de la curva medida (`recompute_curve_calibration`), sin factor fijo de 0,5. Quitar el doble conteo de `w_io` en greedy. El coste del dual placement (recomputación más transferencias) entra al modelo de tiempo.
- **[2.6] Datos.** Resolver los ciclos espurios por pesos atados (lm_head ↔ wte), propagar los campos de Fase 4 en `merge_ilp_inputs_multi_hardware` y exigir `--allow_analytic_fallback` para usar tflops/TDP/ancho de banda analíticos, anotándolo en el resumen.
- **[2.7] Código muerto (con cautela).** La comprobación con `grep` mostró que **no** son código muerto: `peak_activation_overlap` (usado en `model_builder.py`, `run_ilp_partition.py`, `run_ilp_partition.sh`, `run_ilp_pareto_sweep.sh`, `convergence_validation/build_greedy_plan.py`), `w_checkpoint_penalty` (`model_builder.py`, `run_ilp_partition.py`) y `backward_meta_model` (`run_ilp_hierarchical_pipeline.py`, `train_backward_meta_model.py`, `run_ilp_partition.py`, `tests/test_ilp_nextgen_features.py`). No se borran. Solo `estimate_memory_savings_phase4` (1 aparición, su propia definición) y `_solve_with_pulp` (`solve.py:210`, sin llamadores confirmados) son candidatos. Antes de borrar cualquiera: `usages` del símbolo, tests en verde y un commit separado y revertible.
- **Aceptación de la fase:** en cada configuración, error entre tiempo simulado y medido dentro del umbral configurado, y error de memoria predicha vs medida reportado.

### Fase 3. Runtime híbrido

- **[3.1] Dual placement (H4, H5, H14).**
  1. Medir primero cuánto cuesta (snapshots, recompute, comparación) para confirmar H14.
  2. Añadir `--verify_dual_every N` (sección 0). Con N=0 no se copian salidas ni se llama a `.item()`.
  3. Tolerancia por dtype y escala (parámetro), no `0.15/0.05` fijos.
  4. En `call_module`, tratar Dropout igual que ya se hace en el camino por operadores (`_is_stochastic_operator`, L2133): sin dual placement, con aviso y registro. BatchNorm en train actualiza `running_mean/var` dos veces (H5c, verificado): la recomputación del backward debe correr con las estadísticas congeladas (`momentum=0` temporal o copia y restauración) o excluir BatchNorm del dual placement.
  5. Mientras el ILP no modele ese coste, exigirlo (Fase 2.5).
- **[3.2] Grupos de checkpoint (H6).** Dividir los grupos al cambiar de dispositivo, decidir una sola vez por plan y deduplicar los avisos globalmente. Aceptación: 0 avisos "spans >1 device" en resnet50 b64 y resnet152 b32, y checkpoints ejecutados = checkpoints planificados.
- **[3.3] Optimizador (H3).** El tipo (SGD/AdamW), sus hiperparámetros y la semilla vienen de los argumentos. Sus estados entran al modelo de memoria.
- **[3.4] Mediciones.** `reset_peak_memory_stats` antes de cada paso, warm-up descartado, `status` real (capturar OOM), sin doble conteo de prefetch.
- **[3.5] Caché pinned.** Tamaño como parámetro o derivado de la RAM libre. Política por reutilización. `record_stream` en los streams.
- **[3.6] Modo train/eval.** `model.train()/eval()` explícito antes y después del export. Evaluación sin dropout en PRISM y baselines.
- **Aceptación:** en vit_b16 b64, resnet50 b192 y gpt2 b32, el tiempo medido por paso mejora respecto a la línea base, el pico de GPU no supera el presupuesto y no hay `RuntimeError` por mismatch.

### Fase 4. Perfilado y mediciones

- **[4.1]** Backward por segmento con eventos CUDA o `register_full_backward_hook`. Prohibir el fallback silencioso `BACKWARD_FACTOR=2.0`: si falta el dato, la capa se marca como no medida y el ILP falla.
- **[4.2]** Los fallos de OOM guardan NaN, no 0. Eliminar el TDP de 250 W de respaldo.
- **[4.3] Energía:** GPU (NVML) más CPU (RAPL, todos los sockets y DRAM) en una cifra total comparable, o NaN si no hay permiso. Registrar la fuente. Corregir el 0 W en corridas LLM.
- **[4.4] PCIe:** muestreo de tamaños pequeños a grandes, pinned y pageable, unidades consistentes (MB/ms), y alpha/knee ajustados por regresión sobre las muestras.
- **[4.5]** Varianza por capa, warm-up también en CPU, FLOPs reales de Embedding/Conv1D/softmax.
- **Aceptación:** los CSV de perfil no contienen constantes de respaldo y cada columna indica si fue medida.

### Fase 5. Validación y estadística

- **[5.1]** El sweep escribe `sim_time_ms`, `sim_energy_j` y `sim_status`. `run_hybrid_execution.py` deja de usar `ilp_objective` como tiempo (H7).
- **[5.2]** El mejor plan se elige por tiempo simulado con restricción de memoria. Eliminar la doble reducción de presupuesto y el `min(avg_step_ms)` entre candidatos.
- **[5.3]** Estadística con muestras reales: ≥ 5 semillas, intervalos de confianza, corrección por comparaciones múltiples. Sin desviación estándar inventada (`run_statistical_significance.py`).
- **[5.4]** Convergencia comparable: mismas muestras de evaluación, mismo orden de datos, mismo presupuesto en greedy e ILP, sin dropout al evaluar, detección de planes todo-CPU. Añadir vit_b16 y bert_base y validar la divisibilidad de la acumulación.
- **[5.5]** Baselines externos (ítem 4 del jurado): activation checkpointing de PyTorch, `save_on_cpu` y offload tipo ZeRO. El régimen B debe incluir un caso donde all-GPU hace OOM de verdad.
- **Aceptación:** los reportes citan intervalos de confianza y los baselines se ejecutaron con las mismas condiciones.

### Fase 6. Cero hardcodeado (orden expresa)

- **[6.1]** Para cada literal del inventario (ver anexo, tablas de `ilp.md`, `runtime.md`, `runner_core.md`, `validation.md`, `scripts_tests.md`): moverlo a `config/prism.yaml`/CLI, o a una medición escrita en JSON/CSV. Ejemplos: escalera 90/85/80/75/70/50 %, safety 0,90/0,85, pesos 0,5/0,5 vs 1/0, semillas 42/43/44, `HYBRID_BATCH_SIZE=8`, `GPU_VRAM_FALLBACK_MB`, caché de 512 MB, `timeLimit`/`gapRel`, tolerancias, `num_labels`, `gpu_id`.
- **[6.2]** Eliminar rutas y hosts fijos (`/home/zephyr`, `/home/ltorresnino`, `/home/latorresn/PRISM`, `/tmp/test_b256_grid.csv`, `chuc-4`, `paccaA100`, etc.).
- **[6.3]** Las figuras del capítulo 6 con datos fijos o inventados (`generate_vertex_distribution_oom_plot.py`, `generate_energy_gpu_prism_comparison.py`, `generate_overhead_rescue_plot.py`, `generate_chapter6_energy_and_greedy_plots.py`) se regeneran solo desde resultados medidos. Si falta el dato, la figura falla con error.
- **[6.4] Test de CI** que busca literales numéricos o rutas absolutas fuera de `config/`, `tests/` y una lista de excepciones justificadas por escrito.
- **Aceptación:** el test 6.4 pasa y las figuras se regeneran en otra máquina.

### Fase 7. Scripts, campañas, tests y documentación

- **[7.0]** Completar la revisión línea a línea de `scripts/` (`launch_*`, `run_thesis*`), `docs/`, `config/` y `convergence_validation/tests` (H19).
- **[7.1]** Corregir: `FORCE_NO_GPU` persistente, `discover_ilp_config_dirs.sh` (busca `*/results/`, no encuentra `results_thesis_mode`), `MODELS_CSV` indefinido con `set -u`, variables no propagadas en `launch_slurm.sh`, y la sonda híbrida (usar `--compare_baselines` y el batch del plan, sin fallback a batch 8).
- **[7.2]** Perfiles "strict": `FAIL_FAST=true`, `STRICT_METRIC_VALIDITY=true`, código de salida ≠ 0 ante fallos parciales.
- **[7.3]** Tests: incluir `convergence_validation/tests` en `pytest.ini`; convertir o eliminar `tests/test_timeout_validation.py` (no recoge ningún test); tests de los `.sh` (clasificación de fallos, perfiles) y del export del plan; `skipif` explícito para CUDA o rutas fijas.
- **[7.4]** Documentación: `README.md` (referencia a `generate_grid_audit.py` inexistente, descripción de perfiles strict y `OOM_SKIP_CONFIGS`), `docs/SCRIPTS_DOCUMENTATION.md`, CUDA 12.1 frente a cu128, y `requirements.txt` o `pyproject.toml`.

### Fase 8. Re-ejecución y criterios de aceptación globales

- **[8.1]** Repetir la campaña completa (6 modelos, 2 batches, régimen A y B, ≥ 5 semillas).
- **[8.2]** Criterios de aceptación globales:
  1. Ningún plan LLM todo-CPU cuando existe uno con capas en GPU.
  2. Cero OOM en los planes PRISM.
  3. Error de tiempo simulado vs medido dentro del umbral configurado (hoy 1,5×–26×).
  4. Pico de GPU medido ≤ presupuesto en todas las corridas.
  5. bert_base operativo.
  6. En régimen B, PRISM se compara con grad-accum, activation checkpointing y `save_on_cpu`, y se documenta cuándo gana y cuándo no.
  7. Test 6.4 (sin hardcodeado) en verde.
- **[8.3]** Regenerar el informe por paso (tiempo, memoria, energía total, distribución de capas por dispositivo).

---

## 3. Reglas de seguridad para no dañar el proyecto

Estado inicial verificado: hay 56 archivos modificados o eliminados sin commit (último commit `97f3a15`). Reglas obligatorias:

1. **Estrategia de ramas (ver sección 6).** El estado publicado en GitHub no se modifica. Ya existe localmente la rama `v1` y la etiqueta `v1-legacy-snapshot`, ambas en `97f3a15`.
2. **Línea base de tests (medida):** `pytest` en `prism_env` con el árbol de trabajo actual: **109 pasan, 7 se omiten**, 17 s. Ningún cambio puede bajar esa cifra sin explicación escrita.
3. **Un cambio por commit**, con su test de reproducción, para poder revertirlo solo.
4. **Resultados dorados:** guardar la salida numérica (pérdida por paso, asignación de capas) de resnet50 b8, vit_b16 b16 y distilgpt2 b16 antes de los cambios. Tras cada fase, la pérdida debe coincidir dentro de tolerancia y el plan solo puede cambiar donde la fase lo declara.
5. **Primero aditivo:** los parámetros nuevos conservan por defecto el comportamiento actual hasta que el test del hallazgo pase. El cambio del valor por defecto va en un commit aparte.
6. **No se borran ni se sobrescriben** datos de `data/` y `reports/`; las corridas nuevas van a un directorio nuevo.
7. **Nada se borra por inferencia:** el código "muerto" solo se elimina tras `usages`, tests en verde y commit separado (el paso 2.7 ya corrigió un error de este tipo).
8. **Validar por fase** en pequeño (modelos chicos en la GPU de 8 GB) antes de gastar la campaña completa.

## 4. Orden y dependencias

`0 → 1 → (2 ∥ 3 ∥ 4) → 5 → 6 → 7 → 8`. Las fases 2, 3 y 4 son independientes entre sí tras la fase 1. La Fase 6 se aplica de forma incremental (cada fase nueva no introduce literales); su prueba de CI se activa al final.

## 5. Riesgos abiertos

- El dual placement puede resultar caro incluso sin verificación, por la recomputación. Si el coste medido supera el beneficio, el ILP debe prohibir planes mixtos o penalizarlos con el coste medido.
- Si la memoria topológica se vuelve demasiado conservadora, el plan dejará capas en CPU sin necesidad. Medir el error con signo, no solo el valor absoluto.
- Algunas mejoras (baselines externos, convergencia con varias semillas) requieren el clúster (PACCA/Grid5000) y no pueden validarse localmente en 8 GB de GPU.
