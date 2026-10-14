# Investigación de rendimiento (tiempo real vs predicho) y diseño de recompute conjunto en el ILP

**Disparador:** la campaña completa de 4 modelos (`logs/zephyr/claude_monitor_20261011.log`) mostró
planes híbridos reales (`ilp_plan`/`greedy_plan`) **2.3x–18x más lentos** que un baseline 100% GPU
(con gradient accumulation donde hacía falta), escalando con cuánto fragmenta el plan. El usuario
pidió investigar la causa raíz y, en paralelo, se corrigió un hallazgo de memoria independiente
(doble conteo de peso compartido `lm_head`/`wte`, ver sección final).

Estado: 🔶 **causa raíz confirmada con datos reales; la corrección queda diseñada en detalle pero
NO implementada todavía** — pendiente de aprobación explícita antes de tocar el solver (`solve.py`),
dado que modifica el núcleo del ILP.

## Hallazgo A (descartado) — el backend `greedy_phase4` no decide la partición CPU/GPU

Primera hipótesis: el backend `greedy_phase4` visible en los `ilp_solution_summary.json` de la
campaña (99/126 capas en CPU para resnet50/batch_192) era un heurístico eligiendo mal. **Descartado**
leyendo `src/ilp/solve.py:724-869` (`_solve_phase4_greedy`): la partición real viene de
`solve_partition_ilp(data, cfg, backend="auto")` -- el MILP real (`pulp_cbc_dual`). La capa "greedy"
solo decide, **encima** de esa partición ya fija, si aplicar `checkpoint`/`recompute` a capas que
**ya están en GPU** -- nunca puede mover una capa de CPU a GPU. Esto en sí mismo ya es parte de la
causa raíz real (ver Hallazgo C).

## Hallazgo B (descartado por experimento real) — la penalización de fragmentación no ayuda

Los parámetros `w_fragmentation`/`w_congestion` existen en `ILPConfig` (`src/ilp/model_builder.py`)
pero nunca se activan en ningún script de lanzamiento (confirmado por `grep`, cero ocurrencias en
`run_thesis_mode.sh`/`run_ilp_partition.sh`). Hipótesis: activar `w_fragmentation` reduciría los
saltos CPU↔GPU y el tiempo real.

**Experimento controlado** (resnet50 batch_192, mismo presupuesto de GPU, solo variando
`w_fragmentation`, resuelto en frío con el código actual):

| w_fragmentation | capas en GPU (fwd) | aristas cortadas (fwd) | memoria GPU real (topology_memory) |
|---|---|---|---|
| 0 (como corrió la campaña) | 25/126 | 48 | 2256.9 MB |
| 50 | 40/126 | 4 | 2763.2 MB |

Con `w_fragmentation=50` se logró 60% más capas en GPU y 12x menos cortes, sin violar el
presupuesto de VRAM (confirmado con `correct_solution_memory`, `feasible=True`, 0 reparaciones).

**Prueba real ejecutada** (10 pasos reales, misma metodología de `run_hybrid_execution.py` que usó
la campaña, GPU monitoreada en todo momento, pico real 2568 MB de 8188 MB disponibles -- sin
riesgo):

| | transferencias/paso | datos movidos/paso | tiempo/paso |
|---|---|---|---|
| Baseline (`w_fragmentation=0`, el real de la campaña) | 40 | 4.4 GB | 16.6 s |
| `w_fragmentation=50` | 6 | 1.9 GB | **17.4 s (igual o peor)** |

**Conclusión: reducir la fragmentación NO reduce el tiempo real.** Mover más capas a GPU con la
estrategia `retain` (la única que existe hoy) no ayuda porque el verdadero costo no es la
transferencia/sincronización -- es el cómputo en sí.

## Hallazgo C (confirmado) — la causa real es que CPU es ~100x más lento que GPU para estas capas, y no hay forma de evitarlo con la estrategia de activación actual

Del perfilado real (`resnet50_metrics_stats.csv`, batch_192):

| capa | GPU fwd+bwd | CPU fwd+bwd | factor |
|---|---|---|---|
| `conv1` | 6.5 ms | 850.8 ms | **130x** |
| `layer2.0.bn3` | 11.1 ms | 805.4 ms | **72x** |
| `layer2.0.conv1` | 5.2 ms | 761.0 ms | **146x** |

La suma de costo CPU de **las 126 capas completas** (si todo corriera en CPU) es **24.2 segundos** --
coincide casi exactamente con los ~17 s/paso medidos, dado que solo una fracción de esas 126 capas
corre realmente en CPU en cada plan. Esto no es un bug: es la realidad física de hacer convoluciones
en CPU a este batch size. La razón por la que tantas capas terminan en CPU es que correr **todo**
resnet50 en GPU a batch 192 necesita **33.5 GB** de VRAM (confirmado:
`resnet50_pareto_sweep.csv`, columna `all_gpu_gpu_mem_mb`), y la tarjeta real tiene **8 GB**.

Se descartó una hipótesis barata antes de llegar a esta conclusión: hilos de CPU mal configurados
(`torch.get_num_threads()` = 16 por defecto en esta máquina de 24 núcleos, nada anómalo).

## La palanca real, pedida explícitamente por el usuario: recompute en GPU en vez de CPU completo

La tabla de arriba lo deja claro: si una capa no cabe en el presupuesto de VRAM porque su
**activación retenida** (`gpu_act_mem`) es grande, hoy la única alternativa que el ILP considera es
mandar **todo el cómputo** de esa capa a CPU (130x más lento). Pero casi siempre sería mucho más
barato dejarla en GPU y **recomputar** su forward durante el backward (pagar el forward dos veces en
GPU) en vez de pagar el forward+backward completos en CPU una vez. Ejemplo con `conv1`: GPU con
recompute ≈ 2.7ms (fwd) + 2.7ms (recompute) + 3.8ms (bwd) ≈ **9.2ms**, contra **850.8ms** en CPU --
**~92x más rápido**, incluso pagando el forward dos veces.

**Por qué esto no pasa hoy:** la estrategia de activación (`retain`/`recompute`/`checkpoint`) de
Fase 4 (`_solve_phase4_greedy`/`_solve_phase4_exhaustive`, `src/ilp/solve.py`) se decide **después**
de que el MILP real (`_solve_with_pulp_dual`) ya fijó qué capas van a CPU/GPU -- y solo puede
aplicarse a capas que **ya están en GPU**. Nunca puede "rescatar" una capa que el MILP mandó a CPU
por presión de memoria, aunque recomputarla en GPU sería muchísimo más barato que dejarla en CPU.
Confirmado leyendo las tres funciones: ninguna de las dos variantes de Fase 4 existentes reconsidera
el dispositivo.

## Diseño propuesto (no implementado todavía) -- decisión conjunta dispositivo + estrategia de activación

Extender el MILP real (`_solve_with_pulp_dual`, `src/ilp/solve.py`), no los heurísticos de Fase 4,
para que la elección de estrategia de activación sea una variable de decisión más del mismo
problema de optimización, en vez de un parche posterior. Cambio **aditivo y apagado por defecto**
(flag nueva en `ILPConfig4`, default `False`) para no arriesgar ningún comportamiento ya validado en
la campaña de 4 modelos.

### Variables nuevas (solo si la nueva flag está activa)

- `recompute[n]` (binaria): capa `n` usa recompute en vez de retener su activación en GPU durante el
  backward.
- `checkpoint[n]` (binaria): capa `n` vuelca su activación a CPU durante el forward y la recupera
  para el backward (ya existe como concepto en Fase 4, nunca como variable del MILP real).
- Restricciones: `recompute[n] <= xb[n]`, `checkpoint[n] <= xb[n]`,
  `recompute[n] + checkpoint[n] <= xb[n]` -- solo tienen sentido cuando el backward de esa capa ya
  corre en GPU (`xb[n]=1`); si ambas son 0, el comportamiento es "retener", exactamente igual que
  hoy.

### Restricción de memoria GPU modificada

Hoy (`src/ilp/solve.py:503`, caso `memory_model == "topological"`):
```python
total_gpu_act = pulp.lpSum(problem_data.gpu_act_mem[n] * xb[n] for n in data.nodes)
```
Pasaría a:
```python
total_gpu_act = pulp.lpSum(
    problem_data.gpu_act_mem[n] * (xb[n] - recompute[n] - checkpoint[n]) for n in data.nodes
)
```
Sigue siendo **perfectamente lineal** (no hay producto de dos binarias: `xb`, `recompute` y
`checkpoint` son variables independientes relacionadas solo por las restricciones de arriba) -- no
hace falta ningún truco de linealización. Con la flag apagada, `recompute[n]=checkpoint[n]=0`
siempre (ni siquiera se crean las variables), así que la fórmula es **matemáticamente idéntica** a
la actual.

`checkpoint[n]` además debe sumar `cpu_act_mem[n]` al lado CPU del presupuesto (el volcado ocupa
memoria de host temporalmente):
```python
total_cpu_act += pulp.lpSum(problem4.cpu_act_mem[n] * checkpoint[n] for n in data.nodes)
```

### Objetivo modificado

Se suma el costo de recomputar/volcar, reusando exactamente las mismas fórmulas que ya existen en
`build_problem_data_phase4` (`recompute_cost_gpu`, `checkpoint_cost_gpu` -- basadas en
`node_cost_gpu_fwd_ms` y `node_time_io_ms` respectivamente, ya medidas y disponibles):
```python
+ pulp.lpSum(recompute_cost_gpu[n] * recompute[n] + checkpoint_cost_gpu[n] * checkpoint[n]
             for n in data.nodes)
```

### Integración, sin tocar nada existente

- `ILPConfig4` gana un campo nuevo: `enable_joint_activation_strategy: bool = False`.
- `_solve_with_pulp_dual(data, cfg)` revisa `getattr(cfg, "enable_joint_activation_strategy", False)`
  -- si es `False` (el caso de **cualquier** llamada existente, incluida toda la campaña ya
  validada), el código nuevo ni siquiera se ejecuta. Se accede a los demás atributos de Fase 4
  (`w_recompute_penalty`, etc.) también vía `getattr` con el mismo default que usa `ILPConfig4` hoy,
  así que una llamada con un `ILPConfig` simple (no `ILPConfig4`) sigue funcionando exactamente igual
  que ahora.
- Nuevo backend de Fase 4, `solve_partition_ilp_phase4(..., backend="joint")`, que simplemente activa
  la flag y llama al MILP real -- **no modifica** `_solve_phase4_greedy` ni `_solve_phase4_exhaustive`,
  que seguirán existiendo exactamente como están, seleccionables como antes.
- `ILPSolution` necesita un campo opcional nuevo (`activation_strategy_bits: Optional[Dict[str, str]] = None`)
  para que `_solve_with_pulp_dual` pueda devolver qué estrategia ganó por nodo, sin romper ningún
  llamador existente (campo opcional, default `None`).

### Plan de verificación antes de considerar esto terminado

1. **Regresión dura:** con la flag en `False` (default), `pytest tests/ convergence_validation/tests/`
   debe seguir en el mismo conteo de hoy (136 passed / 7 skipped) -- cero capas de código nuevo deben
   ejecutarse en este modo.
2. **Caso sintético controlado** (como se hizo para validar el hallazgo de pesos compartidos): 2-3
   nodos con activación grande y costo CPU >> costo GPU, verificar que con la flag activa el
   solver elige `recompute` en vez de mandar el nodo a CPU, y que el objetivo resultante es menor.
3. **Caso real, en frío (solo CPU, sin GPU, cero riesgo):** resolver resnet50 batch_192 con la flag
   activa, comparar capas en GPU / memoria real (vía `correct_solution_memory`) contra el baseline ya
   medido en este documento (25 capas GPU, 2256.9 MB real) -- debería lograr más capas en GPU sin
   superar el presupuesto.
4. **Solo si (3) es prometedor:** una prueba real corta y monitoreada (10 pasos, igual que las
   anteriores) para confirmar la mejora de tiempo real -- esta vez sí se espera una baja medible,
   porque ahora el ahorro viene de evitar cómputo en CPU, no de reducir transferencias (que ya se
   demostró que no es el cuello de botella).

### Implementado y probado -- resultado: el diseño del MILP es correcto, pero choca con un bug de runtime no detectado antes

Estado: 🔴 **implementado en `src/ilp/solve.py`/`src/ilp/model_builder.py` (bandera
`enable_joint_activation_strategy`, apagada por defecto, cero riesgo para la campaña ya validada) --
pero bloqueado por un hallazgo nuevo en el runtime antes de poder usarse de verdad.** No se revirtió
el código porque es correcto y seguro (ver regresión abajo); solo no se activa en ningún script.

### Verificación 1 -- regresión (bandera apagada): sin cambios

`pytest tests/ convergence_validation/tests/`: 136 passed / 7 skipped, idéntico al estado antes de
este cambio.

### Verificación 2 -- caso sintético: el MILP decide exactamente lo esperado

Nodo con activación de 1900MB, 100x más lento en CPU que en GPU, presupuesto de solo 150MB (no cabe
reteniendo). Sin la nueva bandera: el MILP lo manda entero a CPU (objetivo=503). Con la bandera: lo
deja en GPU con `recompute` (objetivo=8.5, 59x mejor), exactamente la decisión que se buscaba.

### Verificación 3 -- caso real en frío (resnet50 batch_192, sin GPU, solo el solver)

Con la bandera activa, el MILP deja **126/126 capas en GPU** (vs 25/126 del baseline), 99 de ellas
con `recompute`. El validador de memoria (`topology_memory.py`, el mismo que usa el pipeline real)
predice un pico de solo **5789 MB** -- muy por debajo del presupuesto (6632 MB) y de la VRAM física
(8188 MB). Hasta aquí, todo parecía un éxito rotundo.

### Verificación 4 -- prueba real en GPU: **OOM, dos veces, con el mismo error exacto**

Se probó la prueba corta y monitoreada (10 pasos, misma metodología que las pruebas anteriores) dos
veces -- una con presupuesto 6632 MB (predicción 5789 MB) y otra, tras el primer OOM, con presupuesto
mucho más conservador de 2500 MB (predicción 1967 MB, ~3x menos). **Las dos corridas fallaron con el
mismo error exacto, en el mismo nodo exacto** (`add_2`, la suma residual de `layer1.2.bn3` +
`layer1.1.relu_2`), usando la misma cantidad real de memoria (~7.26 GiB) **sin importar cuánto más
bajo fuera el presupuesto dado al solver**. Que el resultado real fuera idéntico a pesar de una
predicción 3 veces menor fue la señal de que no era un problema de margen de seguridad -- era que
`recompute` casi no estaba ahorrando memoria real en absoluto, sin importar cuántas capas se le
asignaran.

### Causa raíz confirmada con un experimento aislado (sin tocar el modelo real, sin riesgo)

Se replicó, en una GPU real pero con un modelo sintético minúsculo (5 bloques conv+bn+relu, batch
192), la **misma granularidad exacta** que usa `hybrid_executor.py::call_module`: cada módulo
primitivo (una Conv2d, una BatchNorm2d, una ReLU) envuelto en su **propia** llamada separada a
`torch.utils.checkpoint.checkpoint`, nodo por nodo, igual que hace el intérprete FX real.

| | pico de memoria medido |
|---|---|
| Sin checkpoint | 2072.9 MB |
| Con checkpoint, cada módulo primitivo por separado | **2812.9 MB (36% PEOR, no mejor)** |

`torch.utils.checkpoint` ahorra memoria al descartar las activaciones **internas** de la región que
envuelve y recomputarlas en el backward -- pero un módulo primitivo (una sola Conv2d) prácticamente
no tiene activaciones internas: su único tensor caro **es su propia salida**, y esa salida sigue
viva igual (la necesita la siguiente capa) tanto con checkpoint como sin él. El resultado neto es
pagar el overhead de bookkeeping de `checkpoint` (guardar estado de RNG, reconstruir el grafo de
autograd, etc.) sin descartar nada real -- de ahí que termine siendo **peor**, no mejor.

**Esto es el mismo hallazgo que ya documentó `fix/06` para modelos decoder-only** ("checkpoint en una
capa aislada ahorra ~0%, protege las entradas pequeñas del nodo, no su salida cara, hace falta
agrupar un bloque completo") -- solo que nadie lo había confirmado para modelos de visión
(`call_module`) porque el heurístico greedy de Fase 4 existente casi nunca elegía `recompute` de
verdad para resnet50/resnet152 en ninguna campaña anterior (confirmado: `recompute_count: 0` en
los `hybrid_execution_summary.json` de la campaña de 4 modelos). El diseño nuevo del MILP conjunto,
al usar `recompute` mucho más agresivamente (99-111 capas en vez de 0), fue lo que finalmente lo
expuso.

### Qué significa para el diseño conjunto: el MILP está bien, el modelo de costo de memoria está mal

La parte que el usuario pidió explícitamente -- "debe ser una decisión del ILP" -- está verificada
como correcta (verificación 2). Lo que falla es un supuesto más profundo, compartido con Fase 4
tal como existe hoy: que `recompute[n]`/`checkpoint[n]` en un nodo aislado realmente libera
`gpu_act_mem[n]`. Eso es falso a esta granularidad. La corrección real necesaria, **no implementada
todavía**, es agrupar los candidatos de `recompute`/`checkpoint` por bloque (p.ej. todo un
`layer1.0` Bottleneck de resnet, no cada conv/bn/relu suelto) -- igual que `_group_nodes_into_
activation_candidates` ya hace para grafos decoder-only (`src/ilp/solve.py:33-61`), pero hoy
restringido únicamente a `data.graph_trace_source == "torch_export_decoder_only"`
(`src/ilp/solve.py`, dentro de `_solve_phase4_greedy`) -- el camino de visión usa `[[n] for n in
gpu_nodes]`, un grupo por nodo, el mismo patrón que acabamos de confirmar que no ahorra memoria.
Extender esa agrupación a grafos de visión, y hacer que `hybrid_executor.py::call_module` sepa
ejecutar un **grupo** de módulos consecutivos bajo un solo `torch.utils.checkpoint` (hoy solo
envuelve un módulo a la vez) es un cambio de runtime más grande, no solo una fórmula del ILP --
análogo al trabajo que `fix/06` ya hizo para decoder-only, pero del lado de visión.

**No se intentó una tercera corrida real en GPU** tras confirmar la causa con el experimento
aislado -- intentar de nuevo sin la agrupación por bloques solo repetiría el mismo OOM.

## Agrupación por bloque para grafos de visión (`call_module`) -- implementada, correcta, pero con un límite real sin resolver

Estado: 🔶 **implementada y verificada como correcta, pero con un hallazgo nuevo (otro, distinto del
anterior) que limita cuánto ahorro real se puede lograr hoy -- no lista para usarse en la corrida
real que motivó todo esto.**

### Qué se implementó

Análogo exacto, del lado de visión, de lo que `_detect_checkpoint_groups` ya hace para grafos
decoder-only (agrupar por bloque en vez de por nodo aislado):

- `_detect_checkpoint_groups_call_module` (`src/runtime/hybrid_executor.py`): nuevo método, llamado
  junto a `_detect_checkpoint_groups` en `__init__`, agrupa nodos `call_module` consecutivos que
  comparten estrategia Y `block_id` (misma función `_block_id_for_layer_name` ya existente) en un
  solo segmento -- reusa el mismo mecanismo de ejecución (`_checkpoint_group_start`/
  `_checkpoint_group_results`, consumido por `run_node` de forma genérica por nombre de nodo, sin
  importar el tipo de operación).
- `_prepare_segment_fn`'s `segment_fn` (función compartida con el camino decoder-only) extendida
  para ejecutar nodos `call_module` (antes solo sabía ejecutar `call_function`, cuyo `.target` ya es
  invocable -- `call_module`'s `.target` es un string, el nombre calificado del módulo, hace falta
  `self.fetch_attr(...)` primero).
- Corregido el comentario, ya desactualizado, en `_detect_checkpoint_groups` que afirmaba
  (incorrectamente, según el hallazgo de arriba) que los grafos de visión "ya tienen
  checkpoint/recompute correcto a granularidad de módulo completo".

### Dos bugs reales encontrados y corregidos en el camino (no en el diseño, en la ejecución)

1. **Mutación in-place dentro del grupo.** `submod(*resolved_args)` dentro de `segment_fn` no
   protegía contra un módulo `inplace=True` (p.ej. `ReLU(inplace=True)`, estándar en resnet de
   torchvision) mutando un tensor que el propio mecanismo de recompute de
   `torch.utils.checkpoint` todavía necesitaba intacto. **Corregido**: clonar el tensor de entrada
   antes de llamar a un módulo in-place, dentro de `segment_fn` -- el mismo truco que
   `_run_layer_with_dual_placement` ya usa (fix/02, bug 2).
2. **El mismo bug, pero preexistente, en el camino de UN SOLO nodo** (`call_module`, el código que
   ya existía antes de esta sesión para `strategy_name == "recompute"`/`"checkpoint"`): tampoco
   clonaba la entrada antes de pasarla a un módulo in-place. Nunca se había disparado en producción
   porque `recompute`/`checkpoint` casi nunca se usaban de verdad (ver Hallazgo C arriba). **Corregido**
   con el mismo patrón.
3. **El valor de SALIDA del grupo, mutado justo después por el consumidor externo.** El patrón
   `out += identity` de torchvision (suma residual **in-place**, literal en cada Bottleneck/
   BasicBlock) mutaba la salida del grupo (el resultado de `bn3`) inmediatamente después de que el
   grupo la devolviera -- corrompiendo la versión que `torch.utils.checkpoint` esperaba poder
   recomputar. **Corregido**: clonar el valor de retorno de `segment_fn` antes de devolverlo.

Verificado con una prueba aislada y controlada (resnet50 real, pesos y entrada fijos con semilla,
sin ambigüedad de módulo compartido -- grupo de 6 nodos: `conv1,bn1,conv2,bn2,conv3,bn3` de
`layer1.0`, excluyendo `relu`): **pérdida y norma del gradiente bit-idénticas** entre la versión con
y sin agrupación (diferencia exactamente `0.0` en ambas). La corrección es matemáticamente correcta.
Ahorro real medido en ese caso: **5.3%** (5572.3MB sin estrategia vs 5276.3MB con el grupo, batch 64).

### El límite real, no resuelto: un módulo compartido (mismo nombre, varios sitios de llamada) rompe el agrupamiento, y peor, puede hacer crashear el entrenamiento

`self.relu` en cada Bottleneck de torchvision se invoca **tres veces** (tras `bn1`, tras `bn2`, y
tras la suma residual) -- la misma instancia de módulo, tres nodos FX distintos, los tres resolviendo
al MISMO `layer_name` ("layer1.0.relu"). Esto tiene dos consecuencias:

- Si `relu` se incluye en el diccionario de estrategias del bloque, mi agrupador la agrupa dentro
  del bloque principal (`conv1..bn3`) **y también** dispara, por separado, el camino de UN SOLO nodo
  para la tercera llamada (la que sigue a la suma residual) -- porque esa tercera llamada cae fuera
  de cualquier grupo de >1 nodo (queda aislada tras el nodo `add`, que es `call_function`, no
  `call_module`, y rompe el agrupamiento). El resultado, confirmado empíricamente, es un choque real:
  `RuntimeError: one of the variables needed for gradient computation has been modified by an
  inplace operation ... is at version 1; expected version 0`. Aplicar los dos fixes de in-place
  arriba **no lo resuelve** -- se intentó, el error persiste exactamente igual.
- Si se **excluye** `relu` del diccionario de estrategias (lo que hice para validar el resto), el
  agrupamiento SÍ funciona y es correcto, pero el nodo `relu` (no marcado, `call_function`... no,
  `call_module` pero estrategia "retain") rompe el grupo en cada punto donde aparece -- en vez de UN
  grupo de 6 nodos (`conv1,bn1,conv2,bn2,conv3,bn3`), se obtienen **tres grupos de 2** (`conv1,bn1` /
  `conv2,bn2` / `conv3,bn3`), cada uno ahorrando mucho menos que el bloque completo. De ahí el 5.3%
  medido, muy por debajo del ~20%+ que se esperaría de un bloque realmente completo.

**No se intentó una tercera corrida real sobre resnet50/batch_192 con el solver conjunto.** Con el
agrupamiento actual (grupos de 2, no de 6-8, y sin poder tocar los `relu` compartidos sin arriesgar
el mismo crash), el ahorro real por capa sería mucho menor al ~100% que el modelo de costo del ILP
todavía asume -- repetir la corrida real solo repetiría una variante del mismo problema de fondo
(predicción de memoria optimista), solo que ahora por una razón distinta y ya identificada, no
arreglada.

### Qué haría falta para cerrar esto del todo (no implementado)

Tratar un nodo `call_module` cuyo `layer_name` ya apareció antes en el MISMO grupo en formación (un
módulo reusado) como una señal explícita de "no agrupar este bloque de forma simple" -- o, más
general, extender el agrupador para que los nodos `call_function` "transparentes" (como el `add`
residual) puedan quedar DENTRO de un grupo que abarque ambas ramas (igual que
`_detect_checkpoint_groups` ya hace para nodos sin `layer_name` en el camino decoder-only), en vez
de cortar el grupo en cada uno. Es un cambio de diseño más profundo al agrupador, no una línea suelta
-- se deja documentado aquí, sin implementar, para no seguir acumulando riesgo sin que el usuario
decida primero si vale la pena perseguirlo.

## El bug del módulo compartido -- corregido por completo, no con un parche

Estado: ✅ **corregido de forma completa y principiada, verificado sin regresión.**

### Diseño de la corrección

El problema real: `self.relu` de torchvision se invoca 3 veces dentro de un mismo Bottleneck (tras
`bn1`, tras `bn2`, tras la suma residual) -- la MISMA instancia de módulo, el MISMO `layer_name`,
pero 3 posiciones distintas del grafo. Toda decisión de estrategia de activación en este proyecto
(ILP y runtime) está indexada por `layer_name`, nunca por sitio de llamada -- así que un solo valor
de estrategia no puede describir 3 sitios distintos sin ambigüedad. No es un bug de un archivo, es
una tensión estructural entre "la unidad de decisión" (nombre de capa) y "la unidad de ejecución"
(nodo del grafo), presente en todo el proyecto desde antes de esta sesión, nunca antes expuesta
porque `recompute`/`checkpoint` casi nunca se usaban de verdad.

**Corrección en dos capas, no una sola:**

1. **Capa de datos del ILP** (`src/ilp/data_loader.py`): nuevo campo
   `ILPInputData.shared_call_site_layers` -- detecta, desde `graph_edges.csv` (contando cuántos
   `src_id`/`dst_id` distintos produce/consume cada `producer_name`/`consumer_name`), qué nombres de
   capa aparecen en más de una posición del grafo real. `merge_ilp_inputs_multi_hardware` lo
   propaga (unión entre perfiles de hardware).
2. **El MILP conjunto** (`src/ilp/solve.py`, `_solve_with_pulp_dual`): para cualquier capa en
   `shared_call_site_layers`, **nunca se crea** la variable `recompute[n]`/`checkpoint[n]` -- es
   directamente el entero `0`, no una variable de decisión. El solver JAMÁS puede ofrecer esa
   opción, por diseño, no por casualidad.
3. **El runtime** (`src/runtime/hybrid_executor.py`): nuevo método
   `_compute_shared_call_module_layer_names`, calculado directamente del grafo real que se está
   ejecutando (no confía en el campo del paso 1 -- defensa en profundidad, válido incluso si alguien
   construye un `DevicePlan` por otro camino). Dos usos:
   - `_detect_checkpoint_groups_call_module`: si una capa compartida aparece **mientras un grupo ya
     se está formando**, se absorbe como paso transparente (no rompe el grupo) -- así `relu` entre
     `bn1` y `conv2` ya no fragmenta el bloque en pedazos de 2 nodos.
   - `call_module` (camino de un solo nodo): fuerza `"retain"` para una capa compartida sin importar
     lo que diga `self._activation_strategies` -- respaldo defensivo.

Además, se corrigieron **dos bugs reales de mutación in-place** encontrados en el camino (uno nuevo
en el código de grupo, uno preexistente en el código de un solo nodo que nunca se había ejercido):
ni `segment_fn` ni el camino de un solo nodo clonaban la entrada antes de llamar a un módulo
`inplace=True`, y el valor de retorno de un grupo tampoco se clonaba antes de ser mutado
externamente por la suma residual in-place de torchvision (`out += identity`).

### Verificación: correctitud confirmada, sin ambigüedad

Prueba real sobre resnet50 (pesos y entrada fijos con semilla, batch 64), grupo completo de 8 nodos
(`conv1,bn1,relu,conv2,bn2,relu,conv3,bn3` de `layer1.0`, con `relu` compartido incluido esta vez):
**sin choque, grupo detectado como un solo bloque de 8 nodos, pérdida y gradiente bit-idénticos**
(diferencia `0.0`) contra la versión sin estrategia. `pytest tests/ convergence_validation/tests/`:
136 passed / 7 skipped, sin regresión.

## El hallazgo más profundo (lo que el usuario pidió investigar a fondo): el ahorro de memoria de recompute NO es aditivo, y se revierte

Estado: 🔴 **confirmado empíricamente. Esta es la verdadera causa de fondo -- no el módulo
compartido, que ya estaba resuelto. El problema de fondo sigue sin solución, y no es un parche
simple.**

Con el bug del módulo compartido ya corregido, medí el ahorro real de memoria agrupando bloques
completos de resnet50 (8 nodos cada uno, exactamente como debía ser desde el principio), variando
CUÁNTOS bloques se marcan con `recompute` a la vez (siempre los primeros N, de 16 totales), batch
64, mismos pesos/entrada:

| bloques con recompute | memoria pico | cambio vs. sin estrategia |
|---|---|---|
| 0 (baseline) | 5572.5 MB | -- |
| 1 | 5376.5 MB | **-3.5%** |
| 2 | 4987.5 MB | **-10.5%** |
| 4 | 4547.5 MB | **-18.4%** (el mejor punto) |
| 8 | 5280.9 MB | -5.2% (ya empeorando) |
| 16 (toda la red, menos el stem) | 5962.5 MB | **+7.0% -- PEOR que no hacer nada** |

La pérdida/gradiente siguen siendo correctos en los 5 casos (diferencia `<1e-4`, ruido de punto
flotante normal). **No es un problema de correctitud. Es que el ahorro de memoria no escala
linealmente con cuántos bloques se recomputan -- tiene un punto óptimo (4 de 16 bloques, ~18%) y
después de ese punto, recomputar MÁS bloques hace que el pico de memoria EMPEORE respecto a no
hacer nada.**

### Por qué esto es la causa de fondo real, no el módulo compartido

El modelo de costo del ILP (`gpu_act_mem[n] * (xb[n] - recompute[n] - checkpoint[n])`, ver la
sección de diseño más arriba) asume que el ahorro de cada nodo marcado `recompute` es:
(a) **independiente** de cuántos otros nodos también se marcan, y (b) **aditivo** -- sumar el
ahorro de 99 nodos predice 99 veces el ahorro de uno solo. Ambos supuestos son falsos, confirmado
con los números de la tabla. Ninguna cantidad de arreglar el agrupamiento por bloque cambia esto --
ya está bien agrupado (8 nodos por bloque, confirmado) y el problema persiste. La solución que la
campaña real necesitaba (99-111 nodos marcados recompute, ver verificación 3 y 4 más arriba en este
documento) está muy por encima del punto donde el ahorro real se revierte (4-8 bloques de 16, aquí).
Es decir: **incluso con todo corregido, el plan que generó el OOM original nunca iba a funcionar,
no por el módulo compartido, sino porque pedía muchísimo más recompute del que el mecanismo puede
dar sin perjudicarse a sí mismo.**

### Hipótesis sobre el mecanismo (no confirmada al nivel de causa raíz de PyTorch, pero consistente con la evidencia)

`torch.utils.checkpoint` con `use_reentrant=False` no envuelve el forward en un `no_grad()` simple
-- instala *hooks* de guardado/recuperación (`saved_tensors_hooks`) que mantienen un "frame" de
contabilidad vivo desde que el bloque corre en forward hasta que su propio recompute se dispara
durante el backward (en orden inverso). Con pocos bloques marcados, el ahorro de no retener las
activaciones internas supera el costo fijo de ese frame. Con muchos bloques marcados, estos frames
se acumulan simultáneamente (todos viven a la vez durante buena parte del backward, ya que el
backward recién empieza a "consumirlos" en orden inverso) y, a partir de cierto punto, su costo
acumulado supera el ahorro de activaciones que están evitando retener. No se confirmó esto diseccionando
los internos de PyTorch con un profiler de memoria por línea -- es la explicación más consistente
con la forma de la curva medida (ahorro que crece, llega a un máximo, y se revierte), ofrecida con
ese nivel de confianza, no como hecho verificado línea por línea.

### Qué significa esto para "una solución completa"

No hay una corrección de una sola línea para esto -- es una propiedad real y medida del mecanismo de
recompute de PyTorch a esta escala, no un bug de PRISM. Las rutas reales hacia una solución
completa, ninguna implementada todavía:

1. **Acotar el ILP con el límite real medido.** En vez de dejar que el MILP marque recompute para
   tantos nodos como "quiera" (optimizando con un modelo de costo que ya sabemos que está mal más
   allá de cierto punto), limitar explícitamente cuántos nodos/bloques pueden marcarse
   simultáneamente (p. ej. vía una restricción `sum(recompute[n]) <= K`), calibrando `K` de forma
   empírica por arquitectura/batch -- consistente con el patrón que `hybrid_overhead_calibration.py`
   ya usa para otro problema (medir, no asumir). Esto acotaría el daño pero no resuelve el modelo de
   costo en sí (seguiría siendo una aproximación lineal con un techo artificial).
2. **Modelar el costo real (no lineal) directamente**, parecido a cómo `topology_memory.py` ya hace
   una simulación más fiel que la suma lineal del MILP para la memoria base -- requeriría una
   función de costo convexa/cóncava por tramos para el término de recompute, lo cual complica la
   formulación MILP (deja de ser puramente lineal) o exige una heurística de búsqueda sobre el
   NÚMERO de bloques en vez de una decisión binaria directa por nodo.
3. **Replantear si recompute es la palanca correcta para el caso que motivó todo esto.** El hallazgo
   de la sección "Causa real" más arriba sigue siendo válido: para batch 192 de resnet50, correr
   todo en GPU necesitaría 33.5GB contra 8GB disponibles -- una brecha demasiado grande para que
   recompute (con un techo real de ~18% de ahorro medido aquí) la cierre por sí solo. La palanca que
   sí funciona hoy, sin tocar código, sigue siendo el tamaño de lote/régimen (ver la tabla de
   resnet152 batch_32 vs batch_96 en la sección de Hallazgo C).

### Confirmado también en modelos de lenguaje (no es un artefacto exclusivo de visión)

Repetí la misma prueba (misma metodología, cuántos bloques marcados con `recompute` vs. memoria
pico real) con `distilgpt2` real (batch 16, seq_length 128, 3 pasos reales de entrenamiento), usando
el mecanismo de agrupación que YA existía para decoder-only desde antes de esta sesión
(`fix/06`, no el que construí hoy para visión):

| bloques transformer con recompute (de 6) | memoria pico vs. sin estrategia |
|---|---|
| 1 | +1.3% (peor) |
| 2 | +7.5% (peor) |
| 3 | -8.6% |
| 4 | -8.5% |
| 5 | **-10.2% (el mejor punto)** |
| 6 (todos) | -3.8% (ya empeorando otra vez) |

Mismo patrón cualitativo que resnet50: pocos bloques marcados puede incluso empeorar memoria, el
ahorro mejora hasta un punto intermedio, y vuelve a empeorar en el extremo de marcar todo. Confirma
que esto **no es un artefacto de mi código nuevo para visión** -- es una propiedad real del
mecanismo de PyTorch, ya presente (sin que nadie lo hubiera medido así) en el camino de decoder-only
que llevaba semanas en el proyecto.

**No se intentó ninguna corrida real de la campaña con esto.** Dado que incluso la mejor
configuración medida (4 bloques) da ~18% de ahorro, muy por debajo del ~65% que el plan conjunto de
99-111 nodos necesitaría para evitar el OOM original, repetir esa prueba real solo repetiría el
mismo fallo, ahora por una razón bien entendida en vez de una desconocida.

## Fase 2 implementada: calibrar la curva real y dársela al MILP como datos

Estado: ✅ **implementado, validado con un caso real completo, con un resultado positivo
confirmado** (más uso de GPU, memoria real segura) -- después de encontrar y resolver un hallazgo
adicional sobre el CONTEXTO en que la curva debe medirse.

### Lo implementado

- `src/runner/recompute_curve_calibration.py` (nuevo): mide, para el modelo/batch real, el pico de
  memoria real en varios puntos de "cuántos bloques candidatos están marcados recompute" --
  reproduce con precisión la curva ya medida a mano (resnet50: pico en 4/16 bloques; distilgpt2:
  pico en 5/6). En el camino, se encontró y corrigió que medir varios puntos EN SECUENCIA en el
  mismo proceso de Python da resultados contaminados a menos que se llame `gc.collect()` entre cada
  medición (mismo patrón de fuga por ciclos de referencia que el ítem 42 del jurado) -- sin eso, el
  mismo punto exacto medido dos veces en el mismo proceso puede cambiar de signo.
- `src/ilp/model_builder.py`: `ILPConfig4` gana `recompute_block_order` (bloques candidatos,
  ordenados de mayor a menor ahorro teórico) y `recompute_curve_points` (puntos medidos
  `(n_bloques, pico_mb)`).
- `src/ilp/solve.py::_solve_with_pulp_dual`: una variable binaria por BLOQUE (no por nodo),
  restringida a elegir siempre un PREFIJO de `recompute_block_order` (restricciones
  `block_recompute[i] >= block_recompute[i+1]`) -- la curva se midió para prefijos, nunca para un
  subconjunto arbitrario, así que el MILP no puede elegir uno. Una codificación lineal por tramos
  estándar (variables indicadoras `is_k`, una por cada posible conteo total de bloques) traduce
  `Σ block_recompute[b]` al delta de memoria real interpolado de los puntos calibrados, sumado una
  sola vez al total -- nunca además del término por nodo (`gpu_act_coeff` excluye explícitamente a
  los nodos de bloques calibrados de la resta individual, para no contar el ahorro dos veces).

### Verificado: caso sintético y regresión

Caso sintético (16 bloques, curva con forma idéntica a la medida en resnet50 real: sube, pico en 4,
se revierte en 16): el MILP elige **exactamente 4/16**, el punto óptimo real, no 0 ni 16. Con
presupuesto generoso, elige 0 (no paga el costo de tiempo de recompute si no hace falta). `pytest
tests/ convergence_validation/tests/`: 136 passed / 7 skipped, sin regresión, bandera apagada por
defecto en todo momento.

### Hallazgo adicional durante la verificación con el caso real: la curva también depende del contexto de dispositivo, no solo de qué bloques

Al conectar todo al caso real que motivó esta investigación (resnet50 batch_192), el resultado fue
seguro pero inútil: el solver eligió 0 bloques de recompute. Investigando por qué: la curva se había
calibrado asumiendo **toda la red en GPU** (conveniente para aislar el efecto de recompute en el
experimento original), pero el problema real de ese caso pone solo 25/126 capas en GPU -- y cada
bloque candidato solo tenía 0-2 de sus 7 nodos en GPU en ese reparto real. Como la restricción del
MILP exige que el BLOQUE COMPLETO esté en GPU para poder recomputarlo
(`block_recompute[b] <= xb[n]` para cada nodo del bloque), usar recompute ahí hubiera exigido mover
5-6 nodos más de CPU a GPU por bloque -- un costo que correctamente no valía la pena para esa curva.
**No es un bug: para batch_192, la brecha real (33.5GB necesarios contra 6.6GB disponibles) es
demasiado grande para que recompute la cierre, sin importar qué tan bien calibrada esté la curva**
-- confirma, por otro camino, el hallazgo ya documentado en la sección "Causa real" de este mismo
documento.

**Verificado que el diseño SÍ funciona cuando el contexto es menos extremo:** se repitió todo con
resnet50 **batch_64** (reparto real: 79/126 capas en GPU, con `layer2.0`/`layer3.0`/`layer4.0`
*ya* 100% en GPU en ese reparto). Calibrando la curva en ESE contexto real (no "todo en GPU"):
baseline real 3352.3MB, mejor punto en 2/3 bloques candidatos (-14.0%). Alimentando esa curva al
MILP completo: **el solver eligió exactamente 2/3 bloques** (el óptimo real) y, más importante,
**encontró margen para subir el uso de GPU de 79 a 86/126 capas** -- más GPU, con la memoria real
verificada (`topology_memory.py`) en 3192.7MB, segura y factible, sin reparaciones necesarias. Es
el resultado que se buscaba desde el principio: más GPU, misma salvaguarda de memoria.

`measure_recompute_curve` ahora acepta `baseline_forward_assignment`/`baseline_backward_assignment`
opcionales (default: todo en GPU, para una prueba rápida aislada) y hay un nuevo helper,
`select_gpu_resident_candidate_blocks`, para filtrar los bloques candidatos a solo los que ya están
completamente en GPU en ese reparto base -- confirmado necesario, no cosmético: un bloque mayormente
en CPU no es un candidato real de recompute bajo la restricción `block_recompute[b] <= xb[n]`.

### El flujo de dos pasos que esto implica (documentado, no automatizado todavía)

1. Resolver el ILP real una vez SIN recompute (`enable_recompute=False`) para obtener el reparto
   CPU/GPU base real.
2. Filtrar los bloques candidatos a los que ya están completamente en GPU en ese reparto
   (`select_gpu_resident_candidate_blocks`).
3. Calibrar la curva real con `measure_recompute_curve`, pasando ESE reparto como
   `baseline_forward_assignment`/`baseline_backward_assignment`.
4. Resolver el ILP conjunto (`backend="joint"`) con `recompute_block_order`/`recompute_curve_points`
   ya calibrados.

**No se automatizó este flujo de 4 pasos en ningún script de producción todavía** (p. ej.
`validation/run_ilp_partition.py`) -- se verificó manualmente, paso por paso, para el caso real de
batch_64. Automatizarlo es el siguiente paso natural, pendiente de decisión del usuario.

### Automatización del flujo de 4 pasos -- implementada

Estado: ✅ **implementado en `validation/run_ilp_partition.py`, probado de punta a punta.**

- `src/runner/recompute_curve_calibration.py` gana dos funciones nuevas: `build_ordered_candidate_blocks`
  (agrupa y ordena los bloques candidatos por `gpu_act_mem` real, reusando exactamente lo que el
  MILP ya calcula vía `model_builder.py::_build_effective_memory`) y
  `select_gpu_resident_candidate_blocks` (filtra a los que ya están completamente en GPU en un
  reparto base real). `measure_recompute_curve` ahora acepta
  `baseline_forward_assignment`/`baseline_backward_assignment` opcionales (default: todo en GPU).
- `validation/run_ilp_partition.py`: nueva bandera `--phase4_calibrate_recompute_curve` (requiere
  `--phase4_backend joint` y `--calibration_batch_size`) que ejecuta los 4 pasos automáticamente:
  resuelve un baseline sin recompute, filtra bloques candidatos ya en GPU en ese baseline, calibra
  la curva real cargando el modelo real, y resuelve el MILP conjunto con esa curva. Nunca
  silenciosamente asume un beneficio no medido: si CUDA no está disponible, si ningún bloque
  candidato está completamente en GPU en el baseline, o si la calibración falla, cae de vuelta a la
  solución del baseline sin recompute, con un mensaje explícito -- nunca una excepción.

**Probado de punta a punta, dos veces:**
1. Con el presupuesto real de la campaña corregido por overhead (`run_ilp_partition.py`'s propia
   corrección ya existente, `hybrid_overhead_calibration.py`): el presupuesto efectivo resultó muy
   ajustado (4326.9 MB, solo 56/126 capas en GPU) y, correctamente, ningún bloque quedó
   completamente en GPU -- el flujo detectó esto y usó el baseline sin recompute, sin fallar.
2. Con un presupuesto generoso (10000 MB, corrección deshabilitada para aislar la prueba): 15 de
   16 bloques quedaron completamente en GPU en el baseline, la calibración corrió sobre el modelo
   real y produjo una curva (óptimo en 4 bloques, igual forma que antes), y el MILP conjunto
   resolvió exitosamente usándola -- `pytest tests/ convergence_validation/tests/` sigue en
   136 passed / 7 skipped, sin cambios al comportamiento por defecto (la bandera nueva nunca se
   activa a menos que se pida explícitamente).

### Validado con entrenamiento real en GPU -- no solo predicción de memoria

Estado: ✅ **confirmado con una corrida real completa, sin errores.**

Se regeneró, con el script real (`run_ilp_partition.py --phase4_calibrate_recompute_curve`), el
plan de resnet50/batch_64 que ya se había validado manualmente (presupuesto 6632.1 MB sin la
corrección de overhead, para reproducir el mismo caso): 85/126 capas en GPU, 2 de los 3 bloques
candidatos con `recompute` (`layer2.0`, `layer3.0` -- 14 capas ATen en total). Se corrió el mismo
protocolo cuidadoso de las pruebas anteriores de esta sesión (10 pasos reales, GPU monitoreada
antes/durante/después, nunca dejado sin supervisión):

- **Completó los 10 pasos sin ningún error.** `recompute_layers=14` confirma que el mecanismo se
  activó de verdad durante la ejecución real, no solo en la predicción.
- **Convergencia real sana**: pérdida 6.24→0.92, exactitud 6%→96.9% -- consistente con otras
  corridas ya validadas de resnet50.
- **Memoria real medida: 3885.4 MB** -- más alta que la predicción de `topology_memory.py`
  (3168.2 MB, un 22.6% de margen entre predicción y realidad, consistente con la imprecisión de
  memoria para resnet50 ya documentada en hallazgos anteriores de esta investigación) pero **muy
  por debajo del presupuesto (6632.1 MB) y de la VRAM física (8188 MB)** -- segura, sin OOM.
- GPU limpia antes y después (79 MiB), sin archivos de prueba dejados en el repositorio.

Esto confirma que el plan que produce el flujo automatizado no solo "parece seguro" en la
predicción de memoria -- entrena de verdad, de principio a fin, sin errores.

### Validado también con un LLM (distilgpt2) -- mismo flujo automatizado, mismo resultado positivo

Estado: ✅ **confirmado de punta a punta, incluyendo entrenamiento real, para un modelo de
lenguaje** -- no es un mecanismo exclusivo de visión.

Dos corridas a través del mismo script real (`run_ilp_partition.py --phase4_calibrate_recompute_curve`),
`distilgpt2` batch_16:

1. **Presupuesto generoso (10000 MB)**: los 6 de 6 bloques transformer quedaron completamente en
   GPU en el baseline; la calibración midió una curva real sobre el modelo real (óptimo en 4
   bloques, misma forma que la medida manualmente antes en esta investigación) y, correctamente,
   el MILP conjunto eligió **0 bloques de recompute** -- no hace falta pagar el costo de tiempo si
   no hay presión de memoria.
2. **Presupuesto ajustado (1900 MB)**: solo 2 de 6 bloques quedaron completamente en GPU en el
   baseline; la curva calibrada en ese contexto (0→1296, 1→1144, 2→1004 MB) llevó al MILP a elegir
   **ambos bloques candidatos** (`transformer.h.0`, `transformer.h.1`, 18 capas), con 57/59 capas
   en GPU.

**Prueba real de entrenamiento (10 pasos, mismo protocolo cuidadoso) con el plan del caso 2:**
completó sin ningún error. `recompute_layers=50` confirma que el mecanismo de agrupación por
bloque de `fix/06` (granularidad de operador ATen, no de módulo) se activó de verdad junto con la
exclusión de pesos compartidos (`lm_head`/`wte`) de esta sesión -- varios mecanismos de este
documento trabajando juntos, sin conflicto. Pérdida 5.92→5.71, exactitud de token subiendo
0.1125→0.15 -- convergencia sana. Memoria real medida: **1170.5 MB** (20.7% más que la predicción
de 970.1 MB -- mismo orden de subestimación ya documentado para LLM en hallazgos anteriores) pero
muy por debajo del presupuesto (1900 MB) y de la VRAM física (8188 MB). GPU limpia antes y después.

### Confirmado también con `gpt2_small`, y comparado en tiempo real contra no usar recompute

Estado: ✅ **tres modelos distintos validados de punta a punta (resnet50, distilgpt2,
gpt2_small), y la primera comparación directa de tiempo real confirma una ganancia neta, no solo
en memoria.**

`gpt2_small` (12 bloques transformer) batch_8, mismo script automatizado: con presupuesto generoso
(10000 MB), 0 bloques elegidos (correcto); con presupuesto ajustado (2600 MB), 5 de 12 bloques
quedaron completamente en GPU en el baseline, la curva calibrada (0→1509.5, 1→1448.6, 2→1369.5,
4→**1332.9 (óptimo)**, 5→1420.3) llevó al MILP a elegir 3 bloques (27 capas), 105/113 capas en GPU.

**Prueba real de entrenamiento:** completó sin errores, pérdida 5.95→5.31, memoria real
1223.9 MB (por debajo incluso de la predicción de 1258.6 MB -- un margen seguro en la dirección
correcta esta vez, a diferencia de resnet50/distilgpt2 donde la predicción subestimaba un poco).

**Comparación directa, mismo presupuesto (2600 MB), con y sin el mecanismo calibrado:**

| | memoria real pico | tiempo real por paso |
|---|---|---|
| Sin recompute (plan base, 103/113 capas en GPU) | 1315.5 MB | 1129.4 ms |
| Con recompute calibrado (105/113 capas en GPU) | **1223.9 MB (-7.0%)** | **1010.4 ms (-10.5%)** |

**El plan con recompute calibrado ganó en los dos ejes a la vez** -- no es un intercambio de
memoria por tiempo, es una mejora neta en ambos, para el mismo presupuesto exacto. Primera
confirmación directa (no solo inferida) de que el mecanismo vale la pena en tiempo real, no solo en
memoria.

### Integrado en los scripts de campaña reales

Estado: ✅ **disponible en `scripts/run_thesis_mode.sh` vía variables de entorno, apagado por
defecto, sin cambiar ningún comportamiento existente.**

- `scripts/run_ilp_partition.sh`: nuevas variables `PHASE4_CALIBRATE_RECOMPUTE_CURVE` (default
  `false`), `CALIBRATION_BATCH_SIZE`/`_PRECISION`/`_SEQ_LENGTH`/`_DATASETS_ROOT`/`_FRACTIONS`.
  Si se activa sin `CALIBRATION_BATCH_SIZE`, falla explícitamente (`exit 2`) en vez de calibrar con
  un batch size adivinado.
- `scripts/run_thesis_mode.sh`: las mismas variables, forwardeadas al paso de materialización real
  del plan (la búsqueda adaptativa de `safety_ratio`×`phase4_target`) -- `CALIBRATION_BATCH_SIZE` se
  deriva automáticamente de `probe_batch_size` (ya parseado del nombre del directorio de config,
  p. ej. ".../batch_128" -> 128), nunca un valor adivinado o default genérico. Si se activa sin
  también poner `PHASE4_BACKEND=joint`, `run_ilp_partition.py` lo rechaza explícitamente y la
  búsqueda adaptativa lo registra como `[WARN]` y sigue probando la siguiente combinación, en vez
  de fallar la campaña completa.

**Verificado:** `bash -n` confirma sintaxis válida en ambos scripts; una corrida real a través del
wrapper de shell (`run_ilp_partition.sh`, no solo el CLI de Python) con las variables nuevas
activadas llega correctamente hasta la calibración; una corrida sin las variables nuevas (el caso
de cualquier campaña ya existente) produce exactamente el mismo resultado que antes de este
cambio. `pytest tests/ convergence_validation/tests/`: 136 passed / 7 skipped, sin regresión.

### Advertencia honesta antes de usar esto en una campaña larga real

La calibración mide la curva real cargando el modelo real y corriendo pasadas reales de
forward+backward en GPU -- esto ocurre en **cada intento** del bucle de búsqueda adaptativa
(`safety_ratio` × `phase4_target`), que en campañas ya vistas en esta sesión necesitó hasta 7-8
intentos. Activar esta bandera en una campaña larga de varios modelos/batches multiplicará el
tiempo de ese paso -- no se ha medido ese costo agregado todavía. Para una primera prueba real de
campaña, se recomienda activarlo en un solo modelo/batch a la vez, no en una corrida masiva.

### Bug real encontrado al intentar el primer piloto real -- corregido

Al intentar lanzar un piloto real (`PHASE4_CALIBRATE_RECOMPUTE_CURVE=true` vía variable de entorno
de nivel superior, siguiendo exactamente el patrón de `launch_full_validation_campaign.sh`), se
encontraron dos problemas:

1. **Bug real, corregido**: `scripts/run_thesis_mode.sh` tiene DOS puntos donde llama a
   `scripts/run_ilp_partition.sh` (el barrido Pareto inicial, y la materialización final vía
   búsqueda adaptativa de `safety_ratio`×`phase4_target`). Solo el SEGUNDO reenvía
   `PHASE4_CALIBRATE_RECOMPUTE_CURVE` con un `CALIBRATION_BATCH_SIZE` correcto (derivado de
   `probe_batch_size`). El primero no la reenviaba en absoluto -- pero como la variable estaba
   exportada en el entorno de nivel superior, **se filtró igual** a ese primer punto (herencia de
   variables de entorno en bash, no necesita reenvío explícito para propagarse). Resultado: el
   primer punto fallaba (`exit 2`, sin `CALIBRATION_BATCH_SIZE`) para **cada** config descubierto,
   saltándose el barrido Pareto para todos ellos. **Corregido**: ese primer punto ahora fuerza
   `PHASE4_CALIBRATE_RECOMPUTE_CURVE=false` explícitamente (esa etapa es el barrido sobre múltiples
   presupuestos, no tiene un batch size único con sentido para calibrar). Verificado con `bash -n`
   en ambos scripts.
2. **No es un bug, pero hay que saberlo**: `run_thesis_mode.sh` (STEP 4) **siempre reprocesa TODOS
   los directorios de configuración que encuentra en disco** (`discover_config_dirs`), sin importar
   qué `MODELS_CSV`/`BATCH_SIZES_CSV` se usó para perfilar en *esta* corrida -- incluye cualquier
   config de una campaña anterior que siga presente. Un "piloto rápido" de un solo modelo/batch
   lanzado así reprocesa igual los 8 configs de la campaña de 4 modelos ya validada antes en esta
   sesión, tardando casi lo mismo que una campaña completa -- no es lo que se buscaba. **Para un
   piloto real y acotado**, usar `scripts/run_ilp_partition.sh` directamente sobre un solo
   `CONFIG_DIR` (como se hizo en todas las pruebas de esta investigación), no `run_thesis_mode.sh`
   completo.

Un intento de piloto real fue detenido a mitad de camino tras detectar esto (proceso de
entrenamiento real en GPU terminado limpiamente con `SIGTERM`, GPU verificada libre después) --
ningún archivo de la campaña anterior fue dañado, solo se reprocesaron de forma redundante.

### Pendiente, no implementado

- No se ha corrido ninguna campaña real completa (`run_thesis_mode.sh` de principio a fin) con
  esta bandera activada -- solo se verificó que el script la acepta y la reenvía correctamente
  (y, tras el bug de arriba, que ya no falla para los casos sin batch size definido).
- No se ha medido el costo agregado en tiempo de activar esto en el bucle de búsqueda adaptativa
  completo (ver advertencia arriba) -- el intento de medirlo fue el que expuso el bug.

## Limitación de diseño, aceptada deliberadamente: solo prefijos, no cualquier subconjunto de bloques

El MILP conjunto (`enable_joint_activation_strategy`) está restringido a elegir siempre un
**prefijo** de `recompute_block_order` (bloques ordenados de mayor a menor ahorro teórico) --
nunca una combinación arbitraria. Esto significa que PRISM **sí** intenta usar la mayor cantidad de
GPU posible dentro de lo que modela: la colocación de dispositivo (`xf[n]`/`xb[n]`) sigue siendo
una decisión POR CAPA, no por bloque, y el solver prefiere GPU para cada capa siempre que entre en
el presupuesto de memoria, porque GPU es casi siempre más barato en tiempo -- eso no depende de
cómo se agrupen los bloques de recompute. Lo que la restricción de prefijos sí puede dejar pasar es
una combinación de bloques a recomputar mejor que cualquier prefijo -- por ejemplo, si dos bloques
grandes juntos no entran en memoria pero un bloque chico más abajo en el orden sí habría entrado
solo, el MILP no puede descubrir esa combinación porque nunca la considera.

**Por qué se aceptó esta limitación en vez de corregirla:** quitar la restricción de prefijos
significa que el MILP tendría que decidir, para cada uno de los N bloques candidatos, si lo
recomputa o no, de forma independiente -- 2^N combinaciones posibles, en vez de las N+1 que permite
un prefijo. Para resnet50 (16 bloques candidatos en los casos probados) eso es 65,536 combinaciones
en vez de 17; para un modelo con más bloques (resnet152 tiene ~50), sería del orden de 10^15. El
problema de partición CPU/GPU que este MILP ya resuelve es NP-duro por sí solo (confirmado y
discutido en `PLAN_ACCION_JURADO.md`, ítem 8, observación de Schnorr); agregarle esta segunda
dimensión combinatoria completa, sin ninguna estructura que la acote, arriesgaría que el solver no
termine en un tiempo aceptable para antes de poder empezar el entrenamiento real -- exactamente lo
que señaló el usuario. La restricción de prefijos no es una limitación técnica que falte resolver:
es la forma deliberada de mantener el problema resoluble en segundos (confirmado: todos los casos
reales de esta investigación resolvieron en 1-3 segundos), a cambio de no garantizar el óptimo
global en la elección de qué bloques recomputar -- un trade-off consciente entre completitud y
tiempo de resolución, consistente con cómo el resto del ILP de este proyecto ya opera (ver
`PLAN_ACCION_JURADO.md`, ítem 8: "el coarsening reduce tamaño de instancia, no la clase de
complejidad").

## Qué NO se va a tocar

`_solve_phase4_greedy`, `_solve_phase4_exhaustive`, `training_profiler.py`, `topology_memory.py`
(salvo que su propia lógica de simulación post-hoc necesite enterarse de la nueva estrategia para
seguir siendo un chequeo de seguridad fiel -- por confirmar en el detalle de implementación), y
ningún script de lanzamiento (`run_thesis_mode.sh` seguiría sin pasar la nueva flag, por lo que el
comportamiento de producción no cambia hasta que se decida explícitamente activarla).

## Hallazgo de memoria, resuelto en la misma sesión (independiente de lo anterior)

Doble conteo de peso compartido (`lm_head.weight is transformer.wte.weight`, weight tying de GPT-2):
`src/ilp/topology_memory.py` sumaba el `params_mb`+`optimizer_states_mb` de ese tensor **dos veces**
(una por cada nombre de capa que lo referencia) tanto en el término estático como en la acumulación
de gradiente durante el barrido de backward. Corregido con un grupo de pesos compartidos
(`shared_weight_group`, detectado por identidad de objeto `id()` en `training_profiler.py`,
propagado por `stats_aggregator.py`, deduplicado en ambos puntos de `topology_memory.py`). Verificado
con una reproducción sintética (tensor compartido cobrado una sola vez: 400MB en vez de 600MB) y con
`pytest tests/ convergence_validation/tests/` (136 passed / 7 skipped, sin regresión). No se ha
vuelto a perfilar `distilgpt2`/`gpt2_small` para confirmar la caída del error de +69.3% con datos
reales todavía -- pendiente si se relanza esa campaña.
