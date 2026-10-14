# Ítem 3 — PRISM evitando OOM de verdad: Phase 4 nunca activada, bug en la validación, y búsqueda adaptativa de dos niveles

Relacionado con: ítem 1 (`fix/01_accuracy_convergence/README.md`), descubierto durante la prueba de
humo local de régimen B para `resnet50` batch 128.

Estado: ✅ **implementado y verificado con ejecución real, en ambos niveles de búsqueda.**

## El problema original

Al correr la convergencia real de régimen B para `resnet50` (batch 128, régimen B, presupuesto
reducido elegido por `scripts/run_ilp_pareto_sweep.sh`), el plan que el propio ILP resolvió como
"óptimo" y "factible" hacía **OOM real** en el primer paso de entrenamiento real:

```
ilp_plan régimen B (batch 128): status=oom
"PRISM training ran out of VRAM at step 1 (batch_size=128): CUDA out of memory..."
```

La pregunta del usuario fue directa: *"la idea de PRISM es evitar a toda costa el OOM, ¿qué y cómo
puedo solucionarlo?"* Esto llevó a una investigación en cuatro partes.

## Hallazgo 1 — Phase 4 (recomputación/checkpoint) nunca se activaba en ninguna campaña real

`src/ilp/solve.py` y `src/ilp/model_builder.py` ya implementan una segunda capa de optimización,
**Phase 4**: una vez que el ILP real (CBC, exacto) decide qué capas van a CPU/GPU, Phase 4 puede
además marcar capas ya asignadas a GPU como `recompute` (recalcular la activación en el backward
en vez de retenerla) o `checkpoint` (descargar la activación a CPU) — liberando memoria sin mover la
capa entera a CPU. Está probado (`tests/test_phase4_*.py`) y el runtime real ya lo implementa
genuinamente (`src/runtime/hybrid_executor.py`, usa `torch.utils.checkpoint` de verdad).

**Pero nunca se activaba.** `scripts/run_ilp_partition.sh` y `scripts/run_ilp_pareto_sweep.sh` jamás
pasaban `--phase4_activation` a `validation/run_ilp_partition.py` (ese flag es `action="store_true"`,
apagado por defecto). Resultado: en **todas** las campañas reales corridas hasta este punto —
locales y en los 3 hosts de clúster (`chuc-4`, `kinovis-2`, `paccaA100...`) — todo
`ilp_solution_summary.json` mostraba `"recompute": 0, "checkpoint": 0"`. La única palanca que el
ILP tenía para caber en presupuesto era mover capas enteras a CPU — nunca recomputación.

**Corrección:** se agregaron variables de entorno (`PHASE4_ACTIVATION`, `PHASE4_ENABLE_RECOMPUTE`,
`PHASE4_ENABLE_CHECKPOINT`, etc., todas con default que preserva el comportamiento anterior) a
`scripts/run_ilp_partition.sh` y se reenviaron desde `scripts/run_thesis_mode.sh` en los dos puntos
donde se materializa un plan (el ILP principal y la materialización del plan de régimen B).

## Hallazgo 2 — El estimado de memoria del simulador no sabe que existe `checkpoint`

Al revisar por qué el presupuesto "factible" seguía fallando, se encontró que
`simulate_plan_phase4()` (`src/runtime/simulator.py`) reutiliza `gpu_mem_used_mb` de la simulación
base **sin ajustarlo** — Phase 4 solo cambia el **tiempo** simulado (agrega una penalización), nunca
el estimado de memoria. Además, el mensaje de advertencia `"peak=X (forward=Y + backward=Z)"` es
engañoso con `MEMORY_MODEL=topological` (el usado en toda la tesis): `Y` y `Z` vienen de una rama de
cálculo que **no** es la que produce el total real bajo ese modelo — por eso `backward` aparecía
constante entre corridas con presupuestos distintos (el hallazgo de "memoria constante" ya anotado
mucho antes en `fix/02_dual_placement_verification_bugs/README.md` queda así root-causado). No se
corrigió esta parte (es más profunda, el modelo de memoria del simulador en sí) — se documenta como
limitación conocida; la mitigación real fue no depender de este número para decidir nada (ver
Hallazgo 4).

## Hallazgo 3 — Bug propio: la validación comparaba `all_gpu` antes de probar el plan real

Al construir la búsqueda adaptativa (ver Hallazgo 4) para probar un plan a su *batch size* real
(antes la validación de `run_thesis_mode.sh` solo probaba a `HYBRID_BATCH_SIZE=8`, nunca al batch
real del config — corregido también), **los primeros 6 intentos fallaron de forma idéntica**, byte
por byte en el mensaje de error. Investigando, se encontró que `HYBRID_COMPARE_BASELINES=true`
(default) hace que `validation/run_hybrid_execution.py` corra `all_cpu` y `all_gpu` **antes** de
`ilp_plan`, en el mismo proceso — y `all_gpu` a batch 128 en una tarjeta de 8GB **siempre** hace OOM
(es justamente el problema que se está rescatando). Como `_run_single_safe` solo atrapa
`HybridExecutionUnsupportedError` (no `OutOfMemoryError`), ese OOM de `all_gpu` — completamente
ajeno al plan bajo prueba — tumbaba el proceso entero antes de llegar a `ilp_plan`. Los 6 intentos
nunca probaron lo que se quería probar. **Corregido** excluyendo `--compare_baselines` de la
validación usada por la búsqueda adaptativa (no afecta el uso normal de `run_hybrid_execution.py`
fuera de esta búsqueda).

## Hallazgo 4 — El heurístico greedy de Phase 4 tenía un objetivo fijo de 60%, sin razón medida

Preocupación del usuario, confirmada en el código: `_solve_phase4_greedy`
(`src/ilp/solve.py`) tenía *hardcodeado* que, una vez activado (uso de GPU > 60% del presupuesto),
reduce el uso de GPU **hasta exactamente 60%** del presupuesto — sin importar cuánto margen hiciera
falta realmente. Esto marca `checkpoint`/`recompute` en más capas de las necesarias, pagando tiempo
de recómputo sin beneficio real.

## La solución: búsqueda adaptativa de dos niveles (sin adivinar nada)

En vez de fijar un margen de seguridad o un objetivo de Phase 4 a mano, `scripts/run_thesis_mode.sh`
ahora busca empíricamente, con ejecución real (no con el estimado del simulador, ver Hallazgo 2):

1. **Nivel interno (`PHASE4_TARGET_GPU_UTILIZATION_CSV`, default
   `0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60`):** con el presupuesto completo del *sweep*, prueba el
   objetivo de Phase 4 de más holgado a más conservador. Cada intento: re-resuelve el ILP (rápido,
   sin re-perfilar ni re-sweepear) y valida con una ejecución real corta (`HYBRID_STEPS`, por
   defecto 10 pasos) **al batch size real del config** (no a batch 8).
2. **Nivel externo (`GPU_VRAM_SAFETY_RATIO_CSV`, default `1.0,0.97,0.95,0.93,0.90,0.85`):** solo si
   NINGÚN objetivo de Phase 4 funciona con el presupuesto completo, recién ahí se reduce el
   presupuesto general y se repite el nivel interno.

La razón del orden: ajustar el objetivo de Phase 4 es una corrección más fina (solo decide cuántas
capas necesitan `checkpoint`); reducir el presupuesto general es una medida más drástica (fuerza más
capas enteras a CPU). Se prueba primero lo más fino.

Al encontrar la primera combinación que sobrevive sin OOM, queda como el plan usado — se guarda un
JSON de trazabilidad (`<config_dir>/safety_ratio_search_<model>.json`) con cuántos intentos hicieron
falta, cuánto tiempo tomó la búsqueda, y qué combinación final funcionó.

## Verificación con datos reales (no solo revisión de código)

- `resnet50` batch 128, régimen B, desde cero: la búsqueda encontró `ratio=1.0` (presupuesto
  completo del *sweep*, sin descuento) + `objetivo Phase4=0.65` — **7 capas con `checkpoint`**
  (antes, con el valor fijo 0.60 sin buscar, eran 9) — en **7 intentos, 248 segundos**. Nunca hizo
  falta el nivel externo.
- Confirmado con `pytest tests/` (95/99, mismos 4 fallos preexistentes de `seaborn`, sin
  regresiones) y `pytest convergence_validation/tests/` (6/6).
- El plan final real, entrenado con datos reales (no simulados), tanto en `resnet50` como en
  `resnet152` (batch 128 y 64 respectivamente), completó una época real sin OOM después de este
  arreglo — antes, ambos hacían OOM en el primer paso.

## Archivos modificados (todos con default que preserva el comportamiento anterior salvo activación explícita)

- `src/ilp/model_builder.py`: nuevo campo `ILPConfig4.target_gpu_utilization` (default 0.60).
- `src/ilp/solve.py`: `_solve_phase4_greedy` usa `cfg.target_gpu_utilization` en vez de `0.6`
  hardcodeado (2 sitios).
- `validation/run_ilp_partition.py`: nueva bandera `--phase4_target_gpu_utilization`.
- `scripts/run_ilp_partition.sh`: nueva variable `PHASE4_TARGET_GPU_UTILIZATION`, reenviada al CLI;
  también las variables `PHASE4_ACTIVATION`/`PHASE4_ENABLE_RECOMPUTE`/`PHASE4_ENABLE_CHECKPOINT`/etc.
  que antes no existían en este script en absoluto.
- `scripts/run_thesis_mode.sh`: variables `PHASE4_ACTIVATION` y relacionadas reenviadas en los dos
  puntos de materialización; `GPU_VRAM_SAFETY_RATIO`/`GPU_VRAM_SAFETY_RATIO_CSV`/
  `PHASE4_TARGET_GPU_UTILIZATION_CSV` nuevas; bucle de materialización+validación de régimen B
  reescrito como búsqueda anidada de dos niveles; validación ahora prueba al batch size real del
  config (antes usaba `HYBRID_BATCH_SIZE=8` fijo); excluye `--compare_baselines` específicamente en
  la validación de la búsqueda (ver Hallazgo 3).

`final_thesis/` no se tocó. `src/runtime/hybrid_executor.py` no se tocó en esta tarea (ya se había
corregido/extendido en `fix/02_...` en una tarea anterior).

## Pendiente

Decidir si el Hallazgo 2 (el simulador no refleja el ahorro de memoria de `checkpoint`/`recompute`,
y su desglose "forward+backward" es ruido bajo `MEMORY_MODEL=topological`) amerita una investigación
más profunda del modelo de memoria del simulador, o si queda documentado como limitación conocida —
la búsqueda adaptativa lo evita por completo (nunca depende de ese número para decidir nada), pero
el número en sí sigue siendo potencialmente engañoso si alguien lo lee directamente de los logs.
