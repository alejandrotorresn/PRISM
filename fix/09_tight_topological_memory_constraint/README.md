# Restricción de memoria "ajustada" (tight) en el núcleo del MILP

## Contexto

Toda la investigación de `fix/08` (expansión GPU + validación real) compensaba el mismo síntoma
una y otra vez: el MILP cree que necesita mucha más memoria GPU de la que en realidad necesita.
Ejemplo medido (resnet50/batch_192): el solver creía 6350.5MB, `topology_memory.py` medía 3291.7MB
reales -- más del doble de sobreestimación. La causa: la restricción de memoria que el solver usa
MIENTRAS decide (`src/ilp/solve.py`, rama `memory_model=="topological"`) es una única suma
agregada -- `Σ gpu_act_mem[n] * xb[n] <= budget` -- que trata la activación de TODAS las capas GPU
como si estuvieran vivas simultáneamente, ignorando que el orden de ejecución es secuencial y cada
activación se libera progresivamente.

## Validación previa (antes de tocar ningún código)

Si el orden topológico es fijo e independiente de qué capa va a CPU o GPU (confirmado:
`PreparedTopologyMemory.prepare()` construye el orden UNA vez, antes de conocer ninguna
asignación), el pico real se puede expresar como una familia de restricciones LINEALES -- una por
cada "punto crítico" donde cambia el conjunto de activaciones vivas (igual que `.evaluate()` ya
hace con `max()`, pero expresado como restricciones en vez de código Python).

Prototipado fuera del MILP y comparado contra `topology_memory.py`'s `.evaluate()` en 6 casos
reales (resnet50 batch_192/batch_64, resnet152 batch_32, distilgpt2 batch_32/batch_64, gpt2_small
batch_32) más un caso sintético con bloque `checkpoint` completo -- **todos dentro de ±0.02% de
error** (ruido de punto flotante), incluyendo el caso con 14 cruces de fase reales (distilgpt2
batch_64). En el camino se encontró y corrigió un bug real en el propio prototipo (la memoria
estática debe contar capas con backward en GPU aunque forward sea CPU, no solo forward).

**Corrección de alcance durante el diseño:** se descubrió que `_solve_with_pulp_dual` usa `xf[n]`
(forward) y `xb[n]` (backward) como variables INDEPENDIENTES (no una sola `xb[n]` como se asumió
inicialmente), y es la MISMA función que atiende Phase4 (`enable_joint_recompute`/`checkpoint`,
inertes por defecto). Decisión de alcance para esta V1: forzar `xf[n] == xb[n]` cuando
`memory_model=="topological_tight"` está activo, eliminando cruces de fase por construcción, y
rechazar explícitamente (error) la combinación con Phase4/curva calibrada -- modelar esos casos
correctamente requeriría linealizar productos de variables binarias, fuera de alcance aquí.

## Implementación

- `src/ilp/data_loader.py`: nuevo campo `ILPInputData.prepared_topology`.
- `src/ilp/model_builder.py`: `"topological_tight"` añadido a los valores válidos de `memory_model`.
- `src/ilp/solve.py`: nueva función `_build_tight_topological_gpu_constraints` (una restricción
  lineal por cada punto crítico forward/backward del recorrido topológico real, con deduplicación
  de pesos compartidos vía indicadores OR), restricción `xf[n]==xb[n]` cuando este memory_model
  está activo, error explícito si se combina con Phase4, y reporte de `gpu_mem_used_mb`/`cpu_mem_used_mb`
  directamente vía `PreparedTopologyMemory.evaluate()` (la misma fuente de verdad, no re-derivado).
- `validation/run_ilp_partition.py`: construye y asigna `data.prepared_topology` antes de resolver,
  cuando corresponda.

## Verificado

| modelo/caso | capas GPU | % del presupuesto usado | solver cree vs real |
|---|---|---|---|
| resnet50 batch_192 (budget 6632.1MB) | 126/126 | 99.8% | 6617.6MB == 6617.6MB (exacto) |
| resnet50 batch_64 (budget 6632.1MB) | 126/126 | 99.7% | 6610.0MB == 6610.0MB (exacto) |
| resnet152 batch_32 (budget 7369.0MB) | 364/364 | 99.99% | 7368.5MB == 7368.5MB (exacto) |
| distilgpt2 batch_32 (budget 2000.0MB) | 59/59 | 92.6% | 1852.1MB == 1852.1MB (exacto) |
| gpt2_small batch_32 (budget 2000.0MB) | 113/113 | ~98-101% | 2035.6MB creído -> repair existente corrigió a 1954.6MB real |

Tiempo de resolución: 0.4-0.5s por caso (resnet50/distilgpt2/gpt2_small), ~0.5s también para
resnet152 (364 nodos, ~928 restricciones adicionales) -- sin impacto perceptible frente al modelo
"topological" anterior. Regresión: `pytest tests/ convergence_validation/tests/` sigue en 136/143.

**Impacto real, sin necesitar `--enable_gpu_expansion` en absoluto:** el solver base, por sí solo,
ahora usa 92-100% del presupuesto en los 5 casos probados -- resnet50 batch_192 pasó de 25-90/126
capas en GPU (con el modelo "topological" viejo) a 126/126 directamente desde la primera solución.

### Nota sobre el caso gpt2_small: por qué el repair existente siguió siendo necesario

El solver creyó 2035.6MB (ligeramente sobre el presupuesto de 2000MB) para su solución "óptima" --
`correct_solution_memory` (el repair post-solve ya existente, sin cambios) lo corrigió a 1954.6MB
real con 1 movimiento. Esto es consistente con la tolerancia numérica normal de un solver MILP
(CBC acepta una solución dentro de un gap/tolerancia, no exactamente en el límite matemático) --
no se encontró evidencia de un error de fórmula (los otros 4 casos coincidieron exactos). El repair
existente sigue cumpliendo su rol de red de seguridad final, ahora con mucho menos trabajo que
antes.

## Qué no se intentó (alcance deliberado de esta V1)

- Phase4 (`_solve_phase4_greedy`, `_solve_phase4_joint`, calibración de curva de recompute) --
  rechazado explícitamente con un error claro. Requeriría linealizar productos de variables
  binarias para modelar correctamente cuándo una activación deja de estar retenida.
- `backend="exhaustive"` -- no participa, no usa `_solve_with_pulp_dual`.
- No se investigó a fondo el pequeño desajuste de gpt2_small (tolerancia de CBC) más allá de
  confirmar que el repair existente lo absorbe correctamente.
