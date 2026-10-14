# Hallazgo nuevo — OOM reales en campañas largas y a escala (`resnet152`): caché de buffers pinneados sin tope, falta de sandboxing, y ~35-52% de subestimación de memoria GPU del ILP

**No fue pedido por ningún jurado.** Continuación directa del ítem 39 (`fix/03_phase4_oom_avoidance_and_adaptive_search/README.md`): aquel arreglo resolvió el OOM para `resnet50`/`resnet152` a batch moderado en corridas cortas. Este hallazgo aparece al escalar a campañas reales, largas y repetidas — `resnet50` batch 192, `resnet152` batch 96, convergencia de época completa — donde aparecieron **cuatro bugs reales distintos**, cada uno descubierto con medición directa, no con sospecha.

Estado: ✅ **los cuatro corregidos y verificados con ejecución real repetida; el caso límite (`resnet152` batch 96) completa limpio por primera vez en toda la investigación.**

## El problema original

Al relanzar campañas reales de `run_thesis_mode.sh` y `convergence_validation/run_convergence_study.sh` a escala (`resnet50` batch 192, `resnet152` batch 96/32, convergencia de época completa en vez de solo unos pasos de humo), el proceso de entrenamiento **mataba el sistema completo dos veces** (la terminal del usuario y VS Code se reiniciaron), y después, incluso con sandboxing puesto, `resnet152` batch 96 seguía sin caber en la GPU real a pesar de que el ILP reportaba su propio plan como factible.

## Hallazgo 1 — Caché de buffers pinneados sin tope de bytes

El fix anterior (sesión previa, no documentado aquí) había introducido una caché de buffers de memoria *pinned* (`torch.Tensor.pin_memory()`) en `src/runtime/hybrid_executor.py::_get_pinned_buffer`, reutilizada entre pasos de entrenamiento para evitar re-pinnear memoria en cada forward/backward de una capa con *dual placement*. El bug: **no tenía ningún límite de tamaño**. Para `resnet152` batch 96, el plan real tiene 37 nodos con `forward≠backward` — pinnear un buffer por cada uno acumula memoria **no intercambiable por el sistema operativo** (a diferencia de un clon normal, que sí puede liberarse bajo presión).

Medido directamente: **7,88 GB** de memoria pinneada permanente para ese plan — suficiente para que, sumado a lo demás abierto en la máquina, `systemd-oomd` y el OOM killer del kernel mataran procesos ajenos (otra terminal, VS Code) en vez de solo el proceso de PRISM.

**Corrección:** tope global configurable (`_PINNED_BUFFER_CACHE_MAX_BYTES`, 512 MB por defecto) en `_get_pinned_buffer`. Al excederlo, la capa cae al camino seguro anterior (clon transitorio, intercambiable) para ESA capa específica, en vez de seguir pinneando sin límite. Verificado con el caso real: de 8.066,6 MB a 496,1 MB pinneados para el mismo plan de `resnet152`/batch 96 — **94% de reducción**, sin tocar ninguna lógica de corrección (el camino de verificación forward-vs-recompute no cambia).

## Hallazgo 2 — El mismo bug, de forma independiente, en `convergence_validation/prism_loop.py`

El arreglo original (de la sesión previa a este hallazgo) había corregido el problema de fondo en `hybrid_executor.py::_run_hybrid_training_dag` (la caché debe crearse **una sola vez** para toda la corrida, no en cada step — `_DeviceAwareFXInterpreter` se recreaba por step, y con él su caché). Pero `convergence_validation/prism_loop.py` reimplementa su propio bucle de entrenamiento (documentado en su propio docstring: no reusa `run_hybrid_training` para no re-trazar el grafo en cada step) y **nunca recibió ese mismo arreglo** — su función `_forward_through_plan` construía un `_DeviceAwareFXInterpreter` nuevo en cada forward pass (cada step de entrenamiento Y cada batch de evaluación), sin pasarle ninguna caché persistente.

Confirmado en vivo: el proceso creció ~800 MB/step durante una época completa de `resnet152` batch 96 hasta que el kernel lo mató (`constraint=CONSTRAINT_MEMCG`, contenido al cgroup gracias al Hallazgo 3, no al sistema completo). Verificado numéricamente: con el bug, se esperarían ~5 asignaciones nuevas de buffer por step (60 en 12 steps); con el fix, exactamente 5 en total en 12 steps reales.

**Corrección:** `_forward_through_plan`, `_run_one_optimizer_step`, `_evaluate` y `run_prism_training` ahora reciben/crean una única caché persistente para toda la corrida (mismo patrón que `hybrid_executor.py`, con default `None` solo para no romper llamadas directas en tests).

## Hallazgo 3 — Sin sandboxing real en `run_convergence_study.sh`

A diferencia de `scripts/run_thesis_mode.sh` (que ya envuelve sus subprocesos pesados con `run_sandboxed`/`systemd-run --scope -p MemoryMax=...`), `convergence_validation/run_convergence_study.sh` corría cada celda **directo en la terminal del usuario**, sin ningún techo de memoria. Cuando algo se salía de control (Hallazgo 1/2, antes de corregirlos), la presión de memoria subía lo suficiente para que `systemd-oomd` —un demonio de espacio de usuario que actúa sobre presión sostenida medida vía PSI, **distinto del OOM killer del kernel**— matara el cgroup completo de la terminal (7 procesos de un golpe, incluyendo el shell), en vez de solo el proceso ofensor.

**Corrección:** se portó el mismo mecanismo `detect_available_ram_mb`/`run_sandboxed` a `run_convergence_study.sh`. Verificado: tras el fix, un OOM real durante la búsqueda de seguridad de `resnet152` quedó con `constraint=CONSTRAINT_MEMCG` (contenido al scope `run-pXXXXX-...scope` del proceso individual) — ni la terminal ni el resto del sistema se vieron afectados, y la campaña siguió avanzando a la siguiente celda en vez de abortar completa.

## Hallazgo 4 — El ILP subestima la memoria GPU real de su propia ejecución híbrida (no es fragmentación, no es recompute cross-phase, no es el reescalado del profiler)

Con los Hallazgos 1-3 corregidos, `resnet152` batch 96 seguía sin caber en la GPU real (7,70 GiB) a pesar de que el ILP reportaba su plan como factible (`gpu_mem_used_mb=5.586,68 MB`, por debajo del presupuesto). La investigación pasó por **tres hipótesis descartadas con medición directa** antes de encontrar la causa real — documentado en detalle porque el propio usuario pidió explícitamente verificación rigurosa en cada paso, tras dos conclusiones previas erróneas en la misma sesión:

1. **Descartada — fragmentación del *allocator* escala con la profundidad del modelo.** Medido `memory_reserved()` vs `memory_allocated()` en ejecución nativa (un solo dispositivo): el gap es *menor* en `resnet152` (14,5%) que en `resnet50` (21,2%) — lo opuesto a lo esperado si la causa fuera puramente el número de capas.
2. **Descartada — el recompute cross-phase (`forward=CPU, backward=GPU`) no se cuenta contra el presupuesto de GPU.** Es un hueco estructural real en `_eval_assignment_dual` (`src/ilp/solve.py`: la memoria de activación de un nodo solo cuenta contra GPU si *ambas* fases corren en GPU — ver comentario "Base memory... use OR... Activation memory... use AND"), pero medido aislando forward-only vs forward+backward en el proceso real: el backward (que es donde ocurre el recompute cross-phase) solo agrega +79,7 MB / +81,7 MB en dos mediciones independientes — no es la causa dominante. **Queda documentado como hueco teórico real, pendiente de corregir en el modelo del ILP, pero no es lo que causó este OOM.**
3. **Descartada — el reescalado lineal del profiler tras un OOM-retry.** El perfilado de `resnet152` batch 96 sí tuvo que retroceder a micro-batch 24 (`oom_retry_triggered=true`, 4 reintentos) y reescalar ×4 linealmente. Hipótesis plausible, pero medido directamente: la memoria real de `resnet152` nativo escala de forma *sublineal* entre batch 12/24/36 (MB/muestra decrece de 195 a 179), así que una extrapolación lineal desde un batch menor en realidad *sobreestimaría* ligeramente, no subestimaría.

4. **Causa real, confirmada instrumentando el proceso real (no una reconstrucción propia) con un `monkeypatch` global de `torch.Tensor.backward`:** el ejecutor híbrido de PRISM (intérprete FX + `_run_layer_with_dual_placement`: mueve capas entre dispositivos, guarda snapshots, usa *streams* de transferencia asíncrona) produce un patrón de asignación de memoria **mucho más fragmentado** que una ejecución nativa de un solo dispositivo — el profiler nunca mide esto porque perfila cada capa corriendo nativamente, nunca a través del motor híbrido real. Medido en el proceso real: `memory_allocated=2.247,2 MB` pero `memory_reserved=3.434,0 MB` — **52,8% más alto**, y ese número de `reserved` coincide con lo que `nvidia-smi` mide externamente (3.672 MB), y con el propio mensaje de error de CUDA ("*reserved but unallocated*") del OOM original.

## La solución — medir el overhead real, no adivinarlo ni hardcodearlo

Nuevo módulo aislado `src/runner/hybrid_overhead_calibration.py`: tras el perfilado normal (sin tocarlo), una pasada adicional de calibración construye un plan sintético representativo (mitad de las capas "retain", mitad `forward=CPU/backward=GPU`) y lo corre **una vez, a través del motor real** (`_DeviceAwareFXInterpreter`/`_run_layer_with_dual_placement`, no una reimplementación), midiendo `memory_reserved()/memory_allocated()`. El resultado (`hybrid_overhead_ratio`, un número *medido*, nunca una constante) se guarda en `<model>_hybrid_overhead.json`. `validation/run_ilp_partition.py` lo promedia entre `run_*` disponibles y divide `gpu_mem_budget_mb` por ese factor antes de pasarlo al ILP — **la fórmula de optimización del ILP no se toca**, solo el presupuesto de entrada que recibe (ver bandera de escape `--disable_hybrid_overhead_correction`).

**Limitación conocida, documentada:** la calibración es representativa, no exacta — si el batch objetivo tampoco cabe nativamente durante la propia calibración, ésta retrocede a un micro-batch menor y el ratio medido sale más bajo de lo real (confirmado: para `resnet152` batch 96 dio solo `1,020`, muy por debajo del `1,528` real medido directamente sobre el plan verdadero). En ese caso específico, lo que realmente cerró la brecha fue la combinación de esta corrección parcial **con** la búsqueda adaptativa de dos niveles ya existente (ítem 39/`fix/03_...`), que en su cuarto intento encontró un presupuesto suficientemente conservador.

## Bonus — métricas de recursos (RAM/VRAM/energía) ahora persistidas en `convergence_validation/`

Pedido explícito del usuario: las curvas de convergencia (pérdida/precisión) no sirven solas para las gráficas de la tesis sin tiempo por paso, tiempo total, memoria y energía. Se agregó, reusando el **mismo mecanismo** que ya usa `src/runtime/hybrid_executor.py::run_hybrid_training` (no uno nuevo): `EnergyMonitor` (NVML) para potencia/energía, `torch.cuda.max_memory_allocated`/`resource.getrusage().ru_maxrss` para picos de VRAM/RAM. `TrainStepRecord` ahora incluye `peak_gpu_mem_mb`/`peak_cpu_mem_mb` por step (trayectoria completa, no solo un número final); `ConvergenceRunResult`/`summary.json` incluyen además `avg_power_w`, `total_energy_j`, y `total_wall_time_s` explícito (antes había que sumar dos campos a mano). Mismo mecanismo en `all_gpu_loop.py` y `prism_loop.py`, para que PRISM y los baselines sean comparables.

## Verificación con datos reales (no solo revisión de código)

- Suite completa: `pytest tests/ convergence_validation/tests/` → **128 passed, 7 skipped**, sin regresiones en ningún punto de esta investigación (verificado repetidamente, tras cada uno de los cuatro hallazgos).
- `resnet152` batch 96, el caso que falló repetidamente durante toda la sesión: **`status: ok`**, `peak_gpu_mem_mb=7.377,4 MB` (cabe en los 7,70 GiB físicos con margen), tanto en la ejecución híbrida de `run_thesis_mode.sh` como en la convergencia completa (`ilp_plan`, régimen B): pérdida 6,34→0,56 / 0,769, exactamente el mismo pico de VRAM en ambos caminos de ejecución — confirma consistencia.
- `resnet50` (control, nunca tuvo el problema severo): corrección de overhead modesta (ratio≈1,035), sin degradación — `status: ok` en todas las celdas, mismos números que antes del cambio.
- Cero eventos de `systemd-oomd` o del OOM killer del kernel en ninguna de las corridas finales de verificación (confirmado con `journalctl -k`/`journalctl` completo, no solo con la ausencia de crash visible).
- Trade-off de PRISM cuantificado con datos reales por primera vez: régimen B (`resnet152`) gasta ~9× más RAM y ~10× más tiempo por paso que el baseline de gradient-accumulation, a cambio de caber en un presupuesto de VRAM donde el baseline `greedy_plan` directamente no puede ejecutar (`status: oom`, esperado y documentado, no un fallo).

## Archivos modificados

- `src/runtime/hybrid_executor.py`: `_get_pinned_buffer` con tope de bytes (`_PINNED_BUFFER_CACHE_MAX_BYTES`), fallback seguro al camino sin pinnear.
- `convergence_validation/prism_loop.py`: caché de buffers pinneados persistente para toda la corrida (`_forward_through_plan`, `_run_one_optimizer_step`, `_evaluate`, `run_prism_training`); instrumentación de `EnergyMonitor`/picos de memoria por step.
- `convergence_validation/all_gpu_loop.py`: `TrainStepRecord`/`ConvergenceRunResult` con los campos nuevos de memoria/energía; helpers `_peak_cpu_mem_mb`/`_peak_gpu_mem_mb`; instrumentación del bucle de entrenamiento.
- `convergence_validation/run_convergence_study.sh`: sandboxing real (`detect_available_ram_mb`/`run_sandboxed`, portado de `run_thesis_mode.sh`).
- `convergence_validation/run_convergence_study.py::_write_result`: columnas/campos nuevos en `train_loss.csv`/`summary.json` (`peak_gpu_mem_mb`, `peak_cpu_mem_mb`, `avg_power_w`, `total_energy_j`, `total_wall_time_s`).
- `src/runner/hybrid_overhead_calibration.py` (nuevo, aislado): mide `hybrid_overhead_ratio` real tras el perfilado normal.
- `src/profiler.py`: llamada adicional, envuelta en `try/except` (nunca puede romper el perfilado existente), que invoca la calibración y escribe `<model>_hybrid_overhead.json`.
- `validation/run_ilp_partition.py`: `_resolve_hybrid_overhead_ratio` + corrección de `gpu_mem_budget_mb` antes de construir `ILPConfig`/`ILPConfig4`; bandera `--disable_hybrid_overhead_correction`.
- Tests nuevos: `tests/test_hybrid_overhead_calibration.py` (8), `tests/test_run_ilp_partition_hybrid_overhead.py` (6), `convergence_validation/tests/test_resource_metrics.py` (3, con GPU real), más el test de regresión del Hallazgo 2 en `convergence_validation/tests/test_convergence_smoke.py`.

`final_thesis/` no se tocó. `src/ilp/solve.py`/`model_builder.py` (la fórmula de optimización del ILP en sí) no se tocaron — el hueco teórico del recompute cross-phase (hipótesis 2, descartada como causa dominante pero real) queda documentado como pendiente, no corregido.

## Pendiente

- El hueco de modelado de `_eval_assignment_dual` para nodos cross-phase (memoria de activación nunca contada contra el presupuesto de GPU cuando `backward=GPU`) sigue sin corregirse — medido como pequeño (+80 MB) en los casos probados, pero podría no serlo en un plan con muchos más nodos cross-phase simultáneamente solapados (prefetch/async_transfer agresivo).
- La calibración de `hybrid_overhead_ratio` es representativa, no exacta, y puede subestimar si el propio batch objetivo tampoco cabe durante la calibración (ver Hallazgo 4). Posible mejora futura: calibrar directamente sobre el plan real ya resuelto, en vez de un split sintético genérico — requeriría reordenar el pipeline (hoy la calibración ocurre antes de que exista ningún plan ILP).
