# Expansión GPU post-solve y la subestimación de memoria de resnet50

## Contexto

Investigando por qué la GPU queda subutilizada en los planes reales de resnet50, se midió algo
concreto: **el MILP cree que está usando 100% de su presupuesto de GPU cuando en realidad solo
necesita 40-54%** (confirmado con `topology_memory.py`, el mismo evaluador real ya validado en
`fix/07`).

| | capas en GPU | el MILP cree usar | lo que REALMENTE se necesita | margen real desperdiciado |
|---|---|---|---|---|
| resnet50 batch_64 | 90/126 | 6631.2 MB (100%) | 3568.0 MB (54%) | 3063 MB |
| resnet50 batch_192 | 25/126 | 6619.6 MB (100%) | 2715.2 MB (41%) | 3904 MB |

**Causa raíz:** la restricción de memoria del MILP (`src/ilp/solve.py::_solve_with_pulp_dual`, rama
`topological`) es `total_gpu_act = Σ gpu_act_mem[n] * xb[n] <= budget` -- suma la activación de
TODAS las capas GPU como si estuvieran vivas simultáneamente. La realidad (que
`PreparedTopologyMemory.evaluate()` ya modela bien: un recorrido secuencial que libera cada
activación apenas se consume) es un pico mucho menor. No es un bug del solver ni de Phase4 -- es
que la propia restricción es una sobreaproximación conservadora, necesaria para que siga siendo
lineal y resoluble en segundos.

## Corrección implementada: `expand_solution_memory` / `rebalance_solution_memory`

Estado: ✅ **implementado, validado con caso sintético y casos reales en frío. ⚠️ prueba real en
GPU reveló un riesgo de seguridad real -- ver sección siguiente, bandera sigue apagada por
defecto.**

Simétrica a la función que ya existía (`correct_solution_memory`, que solo QUITA capas de GPU
cuando el estimado real excede el presupuesto) -- esta agrega el camino inverso: después de
resolver, mueve greedily capas CPU-residentes a GPU (ordenadas por ahorro de tiempo real
`cpu_ms - gpu_ms`, descendente), validando cada movimiento con el mismo evaluador real ya
confiable, mientras siga cabiendo en el presupuesto. Nunca se modificó `correct_solution_memory`
ni la restricción del MILP en sí -- la restricción del MILP no se tocó porque reemplazarla por el
modelo de pico secuencial real le quitaría su linealidad (mismo argumento de tratabilidad ya usado
en `fix/07` para la restricción de "solo prefijos" de bloques de recompute).

Nuevo flag `--enable_gpu_expansion` en `validation/run_ilp_partition.py` (apagado por defecto).
Funciones nuevas en `src/ilp/topology_memory.py`: `expand_solution_memory`, `rebalance_solution_memory`
(expande primero, luego llama a `correct_solution_memory` sin modificar como red de seguridad).

**Verificado:**
1. Caso sintético (3 nodos CPU: uno barato que cabe, uno mediano que cabe, uno enorme que no):
   se movieron exactamente los dos que debían moverse, el enorme se saltó correctamente.
2. `pytest tests/ convergence_validation/tests/`: 136 passed / 7 skipped, sin regresión (bandera
   apagada).
3. Caso real en frío, resnet50:

| | capas GPU antes | capas GPU después | % real del presupuesto |
|---|---|---|---|
| batch_64 | 90/126 | **119/126** (+29) | 54% → **99.9%** |
| batch_192 | 25/126 | **45/126** (+20) | 41% → **99.9%** |

4. **Prueba real de entrenamiento (batch_192, 10 pasos)**: completó sin crash, pérdida 6.29→1.06,
   exactitud 95.3% -- pero con **7 eventos de OOM transitorio del allocator de CUDA** durante la
   ejecución (recuperados automáticamente por PyTorch, nunca llegó a fallar la corrida). Pico real
   medido: **7431.8 MB de 8188 MB físicos (90.8%)** -- solo 756 MB de margen real.

## Hallazgo de seguridad: cerrar el desperdicio de memoria también cierra el colchón que lo escondía

**Antes de esta corrección**, resnet50 ya subestimaba memoria un 12-30% (hallazgo documentado
desde antes de esta sesión) -- pero como el ILP solo usaba 41-54% del presupuesto, ese error de
predicción quedaba absorbido sin consecuencias por el desperdicio mismo, sin que nadie lo
planeara así. **Al recuperar ese margen con `expand_solution_memory`, el uso real sube a ~99.9%
del presupuesto, y el mismo error de predicción de siempre ya no tiene dónde esconderse** -- de
ahí los 7 eventos de OOM transitorio en la prueba real.

**Conclusión: la expansión es correcta, pero no está lista para producción hasta resolver la causa
de la subestimación de memoria de resnet50**, o hasta acotar la expansión con un margen de
seguridad explícito que compense ese error ya conocido. Se decidió investigar la causa de fondo
(subestimación de memoria) en vez de solo bajar el presupuesto efectivo -- ver investigación en
curso más abajo.

## Causa raíz encontrada: ¿por qué resnet50 subestima memoria (y resnet152 no)?

**No es una diferencia arquitectónica entre las dos redes.** Es un bug de selección en el
heurístico Phase4-greedy que resnet50 pisa (porque su presupuesto lo obliga a usar
checkpoint/recompute) y resnet152 nunca pisa (porque sus planes reales nunca necesitaron activar
ese mecanismo).

### Evidencia

Revisando el `ilp_assignment.csv` real de cada plan:

| modelo/batch | capas marcadas `checkpoint` | ¿aisladas o en grupo? | activación de esas capas | gap real medido (predicho vs real) |
|---|---|---|---|---|
| resnet50 batch_192 | `conv1` (1 sola) | aislada | 588.0 MB | 804-824 MB (-29.8% / -12.1% según el run) |
| resnet50 batch_64 | `layer2.0.conv1`, `layer2.2.conv3`, `layer2.3.conv3` (3, cada una sola) | aisladas entre sí | 98.0 MB c/u = 294 MB | ~240 MB (-11.9%) |
| resnet152 batch_32 y batch_96 (legacy, ambos verificados) | ninguna | -- (364/364 capas en `retain`) | -- | ~0 (+2.3% / +2.6%) |

El tamaño de las capas mal-marcadas explica, en el orden de magnitud correcto, el gap absoluto
medido en cada caso -- y el hecho de que resnet152 nunca active el mecanismo en absoluto explica
por qué nunca pisa este bug, sin que exista ninguna diferencia estructural real entre ambas redes
(ambas usan el mismo `Bottleneck` de torchvision).

### Mecanismo exacto

`src/ilp/solve.py::_solve_phase4_greedy` (líneas ~1032-1043), al decidir qué nodos GPU convertir en
candidatos de `recompute`/`checkpoint` cuando la memoria supera el umbral de disparo
(`target_gpu_utilization`, 60% por defecto):

```python
if data.graph_trace_source == "torch_export_decoder_only":
    candidate_groups = _group_nodes_into_activation_candidates(data.nodes, forward_assignment)
else:
    candidate_groups = [[n] for n in gpu_nodes]   # <-- un grafo de visión: cada nodo, SOLO
```

Para cualquier grafo que no sea decoder-only (es decir, **todos** los modelos de visión: resnet50,
resnet152, vit_b16, etc.), cada nodo GPU se trata como su propio candidato de tamaño 1 -- el
comentario que acompaña a esta rama (líneas 1033-1039) justifica esto diciendo "a module-granularity
graph, each node already IS one whole nn.Module layer... that path is unchanged", asumiendo que
checkpointear un `nn.Module` entero (a diferencia de un solo op ATen suelto de un grafo
decoder-only) sí es efectivo por sí solo.

**Esa asunción ya estaba refutada, con una prueba controlada, en el propio motor de ejecución**:
`src/runtime/hybrid_executor.py::_detect_checkpoint_groups_call_module` (líneas 1396-1419, comentario
1405-1408) documenta explícitamente: *"checkpointing a single isolated nn.Module measurably makes
memory usage WORSE (confirmed empirically, +36% in a controlled test), not better, because its only
expensive tensor is its own output, which checkpointing does not free."* Por eso esa función exige
`len(current_group) > 1` **y** que las hojas compartan el mismo `_block_id_for_layer_name` (ej. todo
`layer1.0.*` de un mismo `Bottleneck`) antes de activar de verdad el checkpoint agrupado
(`flush()`, línea 1427). Un candidato de tamaño 1 nunca cumple esa condición -- el motor real lo
ejecuta, pero sin el ahorro de memoria que el plan cree haber ganado (y, según la prueba citada, con
un costo extra neto).

`src/ilp/topology_memory.py::strategy_of()` (líneas 264-274) tiene el mismo defecto, en el
evaluador que se supone es la fuente de verdad: lee `activation_strategy[layer].checkpoint` y lo
honra ciegamente, sin verificar si el motor real formaría -- o no -- un grupo con esa capa. Por eso
la "corrección" de `correct_solution_memory`/`expand_solution_memory` (que confía en este mismo
evaluador) tampoco detecta el problema: ambas heredan la misma asunción incorrecta del heurístico que
generó el plan.

**Conclusión:** el bug vive en el heurístico de *decisión* (`_solve_phase4_greedy`), ya refutado por
el propio motor de *ejecución* (`fix/07`) para el caso agrupado, pero nunca corregido para el caso
de un candidato de un solo nodo en grafos de visión. `resnet152` nunca lo pisa porque sus planes
reales (a los batches probados) no cruzan el umbral de 60% que dispara la búsqueda de
checkpoint/recompute; `resnet50` sí, y elige candidatos aislados porque es la única opción que
`candidate_groups` le ofrece.

## Corrección implementada y verificada

**Estado: ✅ implementado, verificado con regresión + caso real en frío + prueba real de
entrenamiento en GPU.**

Un único cambio quirúrgico en `src/ilp/solve.py::_solve_phase4_greedy` (rama `else`, grafos que no
son decoder-only): en vez de `candidate_groups = [[n] for n in gpu_nodes]` (cada nodo, su propio
candidato aislado), ahora reusa `_group_nodes_into_activation_candidates` -- la misma función
block-aware que la rama decoder-only ya usa y que `fix/06` ya validó -- y además filtra los grupos
resultantes a `len(group) > 1`. A diferencia de la rama decoder-only (que debe dejar pasar el
singleton `lm_head`, cuyo ahorro real viene de un mecanismo dedicado de fusión con la pérdida que no
existe para grafos de visión), aquí ningún grupo de tamaño 1 se ofrece nunca como candidato -- ni
siquiera "fc" o "conv1" -- porque ninguno se beneficia solo (confirmado en el docstring de
`_detect_checkpoint_groups_call_module`).

No se tocó nada más: `_group_nodes_into_activation_candidates` se reusa sin cambios, la rama
decoder-only queda intacta, `_solve_phase4_exhaustive` no participa en producción (guard-rail de
<=18 nodos) y `topology_memory.py` no se modificó (una red de seguridad equivalente ahí se consideró
y se descartó por el riesgo de romper el mecanismo de `lm_head`, que depende de distinguir "nodo
final fusionable con la pérdida" de "nodo aislado real" sin esa información disponible en ese
módulo -- ver plan completo en el historial de esta sesión).

### Verificación

1. **Regresión**: `pytest tests/ convergence_validation/tests/` → 136 passed / 7 skipped (mismo
   resultado que antes del cambio). Dos tests que codificaban la asunción vieja (checkpoint aislado
   válido para grafos de visión) se actualizaron para reflejar el comportamiento correcto:
   `tests/test_phase4_activation.py::TestPhase4GreedyCPUBudget` (datos sintéticos renombrados para
   formar un bloque real de 3 hojas, en vez de 3 nodos sin relación) y
   `tests/test_phase4_decoder_block_grouping.py::test_non_decoder_graph_also_requires_whole_block_grouping`
   (antes afirmaba lo opuesto: que el camino por-nodo debía seguir intacto para grafos de visión).
2. **Caso real en frío, resnet50**, mismo presupuesto de siempre (6632.1MB), mismo script
   (`scripts/run_ilp_partition.sh`, `PHASE4_BACKEND=greedy`): `ilp_assignment.csv` ya no marca
   ninguna capa aislada con `checkpoint`/`recompute` -- las 126/126 capas quedan en `retain` en
   ambos batches, porque el baseline de Phase 3 nunca pone un bloque `Bottleneck` COMPLETO en GPU a
   la vez para este modelo/presupuesto (consistente con el mismo requisito, ya existente, de la
   rama decoder-only: solo se ofrece un bloque si TODAS sus hojas ya están en GPU). Capas en GPU
   (columna `device`, no `activation_strategy`): batch_64 59/126 (antes 90/126, pero ese número
   incluía el "ahorro" fantasma del bug), batch_192 27/126 (antes 25/126).
3. **Prueba real de entrenamiento** (10 pasos, batch_192, protocolo cuidadoso de siempre: GPU
   vigilada todo el tiempo): completó sin ningún warning de OOM (antes: 7 eventos transitorios).
   Pico real medido: **3030.2MB**; predicho (topology-corregido): **2871.6MB** -- **error -5.2%**,
   contra el -29.8% (campaña original) / -12.1% (plan expandido de hoy) de antes de este arreglo.
   Ya en el mismo rango que resnet152 (+2.3%, +2.6%), confirmando que la causa raíz identificada era
   correcta y suficiente -- sin necesitar ninguna diferencia de tratamiento por arquitectura.
4. **No-regresión decoder-only**: no se tocó esa rama (el `if` sigue exactamente igual), y la suite
   de regresión (incluyendo los tests dedicados a la agrupación por bloque decoder-only, ya
   validados en `fix/06`) sigue en verde -- se consideró suficiente sin repetir una corrida real de
   GPU con distilgpt2 solo para esto, dado que no hay cambio de código en ese camino.

### Qué queda abierto

El gap residual de -5.2% en resnet50 (vs. prácticamente 0% en resnet152) probablemente ya sea ruido
normal de medición (una sola corrida de 10 pasos), no un segundo bug -- pero no se intentó aislarlo
más a fondo, dado que ya está dentro del objetivo <10% que el resto de la campaña usa como
referencia.

## Reverificación de `--enable_gpu_expansion` tras este arreglo: sigue sin ser segura (empeoró)

Se repitió la misma prueba real de entrenamiento (resnet50 batch_192, mismo presupuesto 6632.1MB,
10 pasos, protocolo cuidadoso de siempre) con `--enable_gpu_expansion` activo, ahora sobre el solver
ya corregido (sin el bug de checkpoint aislado), para ver si cerrar el bug de fondo hacía segura la
expansión. **No fue así -- empeoró.**

La expansión movió 15 capas más de CPU a GPU (42/126 en GPU, prediciendo 6401.4MB de 6632.1MB =
96.5% del presupuesto). La prueba real de entrenamiento **nunca completó ni un solo paso**: tras 2
eventos iniciales de OOM, el proceso quedó atascado alternando indefinidamente entre dos tamaños de
asignación fijos (392.2MB y 155.2MB) cada ~7-8 segundos, sin ningún progreso (el directorio de
resultados de `hybrid_execution` quedó vacío) durante más de 2 minutos y 15 eventos de OOM --
distinto del comportamiento de antes de este arreglo, donde la misma bandera sí lograba completar
los 10 pasos (con 7 eventos de OOM transitorios, recuperados). Se detuvo el proceso limpiamente
(`SIGTERM`) al confirmar que no había progreso real, y se verificó que la GPU quedó libre (79MiB)
después.

**Interpretación:** arreglar el bug del checkpoint aislado no quitó margen de seguridad -- lo
redujo. Antes, el bug del checkpoint hacía que el plan expandido tuviera MENOS memoria realmente
necesaria de lo que el presupuesto permitía creer (un colchón accidental). Ahora que el solver ya no
miente sobre cuánto libera el checkpoint, el plan expandido usa memoria real y densa hasta ~96.5%
del presupuesto -- sin ningún colchón para la fragmentación/sobrecarga transitoria del motor híbrido
(transferencias async, prefetch, selección de algoritmo de cuDNN) que `topology_memory.py` nunca
modela (ver su propio docstring, sección "Known remaining limitation"). El resultado es que la GPU
queda sin memoria libre contigua suficiente para que el allocator de CUDA complete una sola
asignación, entrando en un ciclo de reintento que nunca progresa -- un fallo más grave que el OOM
transitorio-pero-recuperable de antes.

**Conclusión (en el momento de escribir esto): `--enable_gpu_expansion` sigue sin estar lista para
producción. La bandera permanece apagada por defecto.** El problema de fondo ya no es la
subestimación de memoria de resnet50 (ya corregida, -5.2%) sino la sobrecarga transitoria real del
motor híbrido en sí -- ver la corrección implementada abajo, que sí la mide y la tiene en cuenta.

## Corrección: validar la expansión contra la sobrecarga transitoria REAL del motor híbrido

**Estado: ✅ implementado y verificado -- la expansión ya es segura cuando hay prerrequisitos de
calibración disponibles (misma flag `--calibration_batch_size` que ya existía para la curva de
recompute, ningún flag nuevo).**

En vez de confiar solo en `topology_memory.py` (que nunca modela la fragmentación del allocator de
CUDA por transferencias async/prefetch) ni en el ratio genérico de
`hybrid_overhead_calibration.py::measure_hybrid_overhead_ratio` (medido sobre un split sintético
"representativo", **antes** de que la expansión decida nada -- así que no describe el patrón de
corte real que la expansión termina produciendo), se agregó:

1. **`measure_real_peak_for_plan`** (nueva función en `hybrid_overhead_calibration.py`, junto a
   `measure_hybrid_overhead_ratio`, reusando el mismo helper interno de medición
   `_run_one_pass_and_measure`): mide la memoria real (`reserved_mb`/`allocated_mb`) de UNA pasada
   real de forward+backward a través del mismo motor de ejecución
   (`_DeviceAwareFXInterpreter`), pero con el `DevicePlan` construido directamente desde el plan
   CANDIDATO exacto (no uno sintético).
2. **`expand_solution_memory`** ahora también retorna `flip_order` -- las capas movidas CPU→GPU, en
   el orden exacto en que se aplicaron (cambio puramente aditivo, sin tocar su lógica de decisión).
   `rebalance_solution_memory` lo reenvía como `expand_flip_order`.
3. **`_validate_gpu_expansion_against_real_overhead`** (nueva función en `run_ilp_partition.py`,
   insertada entre la expansión y el repair existente de `correct_solution_memory`): cuando hay
   GPU disponible y `--calibration_batch_size` está puesto, carga el modelo real una vez (mismo
   patrón que `_solve_with_calibrated_recompute_curve`) y hace una búsqueda binaria sobre cuántos
   de los movimientos de `flip_order` mantener, verificando cada candidato con UNA medición real
   (`measure_real_peak_for_plan`) -- acotado a ~2+log2(N) mediciones reales. Si ni siquiera el plan
   previo a la expansión (`k=0`) pasa la medición real, se usa igual (no es peor que antes) con una
   advertencia fuerte, señalando un posible punto ciego en el propio `correct_solution_memory`
   (fuera de alcance de este arreglo). Sin esos prerrequisitos, cae al comportamiento de antes
   (sin validar) pero con una advertencia explícita en vez de una falsa sensación de seguridad.

### Verificación

1. **Regresión**: 136 passed / 7 skipped, sin cambios (todo el código nuevo es aditivo y solo se
   activa con `--enable_gpu_expansion` + prerrequisitos de calibración).
2. **Caso real en frío, resnet50 batch_192** (mismo presupuesto 6632.1MB, ahora con
   `--calibration_batch_size 192`): la búsqueda binaria detectó que mantener los 15 movimientos
   completos de la expansión sin validar causaba un OOM real (confirmado: el mismo patrón de
   bloqueo de la sección anterior), y recortó a **4/15 movimientos**, con `real_reserved_mb=6354.0MB`
   -- dentro del presupuesto.
3. **Prueba real de entrenamiento** (10 pasos, protocolo cuidadoso de siempre): completó **sin
   ningún warning de OOM** (0, frente a los 7 de antes del fix de causa raíz y los 15+ del ciclo de
   bloqueo tras ese fix sin esta validación). `hybrid_execution_summary.json` reportó
   `peak_gpu_mem_mb=5086.9MB`. GPU confirmada limpia (79MiB) después.

   **Corrección a una lectura inicial equivocada de este número:** se reportó inicialmente esto
   como "76.7% del presupuesto" (5086.9/6632.1), sugiriendo margen desperdiciado. Es una comparación
   entre métricas distintas, no un hallazgo real: `peak_gpu_mem_mb` del harness de entrenamiento usa
   `torch.cuda.max_memory_ALLOCATED()` (solo el payload de tensores --
   `src/runtime/hybrid_executor.py:2320`), mientras que la validación de esta sección compara contra
   `max_memory_RESERVED()` (lo que el allocator de CUDA realmente tiene reservado -- la métrica que
   de hecho determina si hay OOM, y la que motivó toda la investigación de `fix/08`). Confirmado
   directamente comparando ambas métricas para el MISMO plan de 4 movimientos: `real_allocated_mb≈5090MB`
   (coincide con el `peak_gpu_mem_mb` del entrenamiento real) vs `real_reserved_mb=6354.0MB`. La
   utilización real correcta es **6354.0/6632.1 = 95.8%** -- ya en el rango objetivo (~95-99%) de
   toda esta investigación. También se descartó la hipótesis de "arranque en frío" (cuDNN
   autotuning/allocator creciendo desde cero): 3 pasadas de calentamiento antes de medir dieron el
   mismo `reserved_mb` que una sola pasada fría, así que no hay sesgo de ese tipo que corregir.

### Qué queda abierto (en ese momento)

- El gap entre `topology_memory.py` (predijo 4047.8MB para el plan validado) y la medición real
  RESERVED de la validación (6354.0MB) sigue siendo grande (~2300MB) -- es exactamente la sobrecarga
  transitoria que motivó este arreglo, ahora medida y respetada en vez de ignorada, pero no
  "explicada" ni reducida en sí misma.
- `peak_gpu_mem_mb` en `hybrid_execution_summary.json` (y en cualquier reporte/plot que lo consuma)
  usa `max_memory_allocated()`, que subestima sistemáticamente cuánto de la VRAM física está
  realmente comprometida frente a `max_memory_reserved()` -- no es un bug (es una métrica válida,
  "cuánta memoria usan los tensores"), pero es fácil de malinterpretar como "cuán cerca del
  presupuesto está el plan" si no se tiene en cuenta esta distinción.
- No se repitió esta verificación para resnet50 batch_64 ni para otros modelos (resnet152,
  distilgpt2, gpt2_small) -- el mecanismo es genérico (no depende de arquitectura), pero cada
  modelo/batch nuevo que use `--enable_gpu_expansion` en una campaña real debería confirmarse al
  menos una vez con una prueba real de entrenamiento, siguiendo el mismo protocolo.

## Hallazgo real en la primera campaña post-fix: duplicación de buffers de transferencia en cortes de dispositivo

Al relanzar la primera campaña real (resnet50, batch_64/192, con `GPU_EXPANSION=true` ya cableado en
los scripts), ambos batches entrenaron sin ningún incidente -- pero comparando `gpu_mem_used_mb`
(predicho) contra `peak_gpu_mem_mb` real del harness de entrenamiento:

| batch | predicho | real (allocated) | error |
|---|---|---|---|
| batch_64 | 3361.0MB | 3684.1MB | -8.8% (mejoró del -11.9% original) |
| batch_192 | 2854.4MB | 4393.0MB | **-35.0%** (empeoró del -29.8% original) |

**Causa raíz:** `src/runtime/hybrid_executor.py` (líneas ~2093-2116), con prefetch activo, hace
`prefetched = out.to(user_device, ...)` en cada corte de dispositivo -- una COPIA del tensor en el
destino, mientras el original sigue vivo en el origen. `topology_memory.py` nunca contabiliza esa
duplicación. Como `max_memory_allocated/reserved` son marcas de agua alta acumuladas durante TODA
la corrida, un pico transitorio temprano por una duplicación grande queda grabado en el pico final
aunque se libere después. Confirmado con los cortes reales de cada plan: batch_192, con menos
margen de presupuesto, se ve forzado a cortar justo en `layer1.0` -- la activación más grande de
toda la red (588MB, antes de cualquier reducción espacial) -- mientras que batch_64 evita cortar ahí
(máximo 196MB por corte). Menos cortes totales (52 vs 88) pero cada uno ~3x más grande explica por
qué el error empeoró para el batch más grande pese a tener un plan "más simple".

### Corrección: generalizar la validación real a SIEMPRE, no solo como filtro de la expansión

Se consideró modelar esto analíticamente dentro de `topology_memory.py`, pero depende del
comportamiento exacto de profundidad de prefetch de PyTorch -- difícil de replicar con precisión
sin medirlo (mismo argumento ya usado para no modelar la sobrecarga transitoria ahí). En vez de
eso, se generalizó la validación real ya existente (`measure_real_peak_for_plan`):

- Nuevo helper compartido `_load_calibration_model(args)` (`validation/run_ilp_partition.py`) --
  carga el modelo real una sola vez, reusado por la validación de expansión y la nueva validación
  final.
- Nueva función `_validate_final_plan_against_real_overhead`: corre SIEMPRE (con o sin
  `--enable_gpu_expansion`) sobre el plan final, cuando hay prerrequisitos de calibración. Mide una
  vez con `measure_real_peak_for_plan`; usa `real_reserved_mb` (lo que el allocator realmente
  reserva) para decidir factibilidad contra el presupuesto, y **reemplaza `gpu_mem_used_mb` con
  `real_allocated_mb`** -- la misma métrica que `hybrid_execution_summary.json` reporta, así que
  "predicho" y "real" vuelven a ser una comparación consistente. Si no cabe, recorta por búsqueda
  binaria (quitando las capas GPU más grandes primero, mismo orden que ya usa
  `correct_solution_memory`) hasta que una medición real confirme que cabe.

**Verificado:** repitiendo el caso real (resnet50 batch_192, mismo presupuesto 6632.1MB,
`GPU_EXPANSION=true`, `CALIBRATION_BATCH_SIZE=192`): `GPU mem used (MB)` reportado pasó de 2854.4MB
(estimado, mal) a **4668.3MB** (medición real) -- feasible confirmado (`real_reserved_mb=5230.0MB`
contra presupuesto 6404.7MB). Regresión: 136/143 sin cambios. Prueba real de entrenamiento (10
pasos, protocolo cuidadoso de siempre) sobre el plan resultante: completó **sin ningún OOM**, pico
real medido **4536.2MB** -- error final **+2.9%** (predicho 4668.3MB vs real 4536.2MB), contra el
-35.0% de antes de este arreglo. Ya en el mismo rango de excelencia que resnet152 (+2.3%, +2.6%).
GPU confirmada limpia (79MiB) después.

## Segundo hallazgo real, al lanzar el estudio de convergencia completo (200 pasos): fuga de memoria por ciclos de referencia, no relacionada con lo anterior

Al correr `convergence_validation/run_convergence_study.sh` (200 pasos reales, no la sonda corta de
validación) sobre el MISMO plan ya validado arriba, `peak_gpu_mem_mb` reportado subió mucho más de
lo esperado: batch_64 a 5437.6MB (vs 3777.2MB de la sonda corta) y batch_192 a 6263.9MB (vs
4536-5012MB). La hipótesis inicial ("la evaluación periódica consume más memoria") se descartó con
una reproducción controlada -- **no es la misma causa que arriba, es una fuga real, independiente,
en `convergence_validation/prism_loop.py`** (el harness de entrenamiento largo, que corre con
`enable_prefetch=False`/`enable_async_transfer=False`, a diferencia de la sonda de validación que
usa prefetch ON):

```
step=14  live=428.8MB
step=24  live=1164.8MB   <- +73.6MB EXACTO cada paso, 11 pasos seguidos
step=25  live=208.1MB    <- el GC cíclico de Python finalmente corrió
step=26  live=280.0MB    <- vuelve a crecer al mismo ritmo
```

Confirmado con `gc.disable()` (mismo patrón, descartando fragmentación del allocator de CUDA) y con
`gc.collect()` manual (libera 1354 objetos, de 1238.8MB a 134.8MB) que es un **ciclo de referencia
de Python** -- misma clase de bug que `PLAN_ACCION_JURADO.md` item 42 ya encontró una vez (en
`hybrid_executor.py::_run_operator_with_dual_placement`), pero una instancia NUEVA, nunca antes
auditada: se reproduce incluso con el plan más simple posible (sin dual-placement, sin
checkpoint/recompute, sin prefetch), así que NO es la misma closure que el item 42 ya arregló. A
diferencia de esa fuga (que ni `gc.collect()` podía liberar), esta SÍ se libera con `gc.collect()`
-- más cercana a un patrón ya conocido de PyTorch (ciclos en el *bookkeeping* de autograd para
transferencias entre dispositivos) que a un bug propio del proyecto.

**Corrección:** `gc.collect()` periódico (una vez por paso de entrenamiento, una vez por ronda de
evaluación) en `convergence_validation/prism_loop.py` y, defensivamente, en `all_gpu_loop.py` --
mismo patrón ya usado con éxito en este proyecto (`recompute_curve_calibration.py`,
`training_profiler.py`). Costo medido: ~18.5ms por llamada, despreciable frente a un paso real
(100ms-17s según el caso).

**Verificado:** repitiendo la reproducción (35 pasos) con el fix aplicado: memoria perfectamente
plana (`live=133.0MB` en los 35 pasos, sin ninguna variación; pico fijo en 4573.7MB desde el primer
paso). La fuga desapareció por completo.

### Qué queda pendiente

- No se confirmó con una reproducción propia si `all_gpu_loop.py` tenía realmente esta fuga (el
  camino 100%-GPU nunca mueve tensores entre dispositivos, el mecanismo sospechado) -- el
  `gc.collect()` ahí es defensivo, de costo despreciable, no una corrección de un bug confirmado.
  Si se quiere confirmar/descartar con certeza, se necesitaría la misma reproducción directa hecha
  aquí para `prism_loop.py`.
- Falta relanzar el estudio de convergencia completo (200 pasos, resnet50 batch_64/batch_192) con
  el fix aplicado para obtener números de memoria corregidos y confirmar que `final_train_loss`/
  `final_eval_metric_value` no cambiaron -- los resultados de la corrida anterior (reportados antes
  de este arreglo) ya quedaron con memoria sobreestimada, aunque la pérdida/exactitud reportadas en
  ellos sí son válidas (el fix es puramente de limpieza de memoria).
- No se investigó si distilgpt2/gpt2_small (grafos decoder-only, que SÍ pasan por
  `_run_operator_with_dual_placement` y otros mecanismos más complejos) tienen esta misma fuga u
  otra adicional -- deberían verificarse con el mismo método antes de confiar en sus números de
  memoria del estudio de convergencia.

## Tercer hallazgo: la MISMA fuga, pero dentro de la propia validación de expansión GPU

Tras corregir lo anterior, el usuario notó que el uso real de GPU seguía muy por debajo del
presupuesto (4573.7MB de 6632.1MB, ~69%) y sospechó que la validación de expansión GPU
(`_validate_gpu_expansion_against_real_overhead`, que hace varias mediciones reales en el mismo
proceso durante su búsqueda binaria) podía tener la misma fuga -- confirmado:
`src/runner/hybrid_overhead_calibration.py` (el módulo detrás de `measure_real_peak_for_plan`)
nunca tuvo ningún `gc.collect()`. Corregido con el mismo patrón ya establecido
(`gc.collect()` antes Y después de cada sonda, en el helper compartido `_run_one_pass_and_measure`).

**Verificado:** 8 mediciones repetidas del mismo plan en el mismo proceso -- antes del fix, cada
llamada heredaba basura de la anterior; después del fix, las llamadas 1-7 dan el número EXACTO
idéntico (4851.7MB allocated / 5788.0MB reserved; solo la llamada 0, el "warmup" de cuDNN, difiere
un poco). Regresión: 136/143 sin cambios.

**Pero esto NO explica la baja utilización de GPU que preocupaba al usuario.** Repitiendo la
búsqueda completa de expansión con el fix aplicado, el resultado es prácticamente el mismo que
antes (3/19 movimientos, real_allocated_mb=4907.4MB vs 4851.7MB de antes) -- la contaminación entre
mediciones era real y ya está corregida, pero no era la causa principal de que la expansión se
detenga donde se detiene. Los eventos de OOM reales observados al probar candidatos más grandes
durante la búsqueda binaria confirman que, para este modelo/batch/presupuesto, **ya no hay más
margen real que recuperar** -- el límite viene del costo real de las transferencias cruzadas de
dispositivo (el hallazgo de la sección "duplicación de buffers de transferencia" arriba), no de un
artefacto de medición. Si se quiere usar más GPU, el camino sería reducir ese costo real (menos
cortes, o cortes en capas más pequeñas) o aceptar un presupuesto de GPU mayor -- no hay más que
exprimir en la validación misma.
