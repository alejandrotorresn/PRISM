# Sobreestimación de memoria en bloques residuales: atribución incorrecta del segmento de backward

## Síntoma (hallazgo del usuario, no de este análisis)

Comparando, para el caso SIN cortes (100% GPU, confirmado 0 cut_edges en los 4 modelos), lo que
el solver/`topology_memory.py` CREE (ya auto-consistente, ver `fix/09`) contra lo que
`convergence_validation`'s corrida real de 200 pasos (`peak_gpu_mem_mb` vía
`torch.cuda.max_memory_allocated`) REPORTA:

| modelo (batch real usado) | solver cree (MB) | real medido (MB) | sobreestimación |
|---|---|---|---|
| resnet50 (batch 64) | 7198.06 | 5437.62 | +32.4% |
| resnet152 (batch 32) | 7275.06 | 5739.18 | +26.8% |
| gpt2_small (batch 8) | 2830.74 | 2468.77 | +14.7% |
| distilgpt2 (batch 16) | 3138.32 | 3091.70 | +1.5% |

Verificado cruzando batch sizes correctamente (el error inicial de este análisis fue comparar
gpt2_small contra su solución de batch_32, no la de batch_8 realmente usada por el estudio de
convergencia -- corregido). Confirmado también con una segunda medición real independiente
(`all_gpu`, ejecución nativa sin el intérprete FX): para resnet50/resnet152 coincide casi exacto
con `ilp_plan` (5436.2 vs 5437.6; 5723.9 vs 5739.2) -- descarta que el número "real" de referencia
esté mal medido por el propio hybrid executor.

## Hipótesis descartadas (con evidencia, no por intuición)

1. **Reescalado lineal por OOM-retry durante el perfilado** (`_current_batch_scale` en
   `src/runner/training_profiler.py`): revisando los logs de perfilado de esta campaña
   (`logs/zephyr/experiments/experiments_20261013_*.txt`), SÍ hubo eventos de OOM-retry, pero
   SIEMPRE en un batch size DISTINTO al realmente usado por cada modelo en el estudio de
   convergencia (resnet50 OOM solo en batch=192, no en batch=64 que es el usado; resnet152 OOM
   solo en batch=96, no en batch=32; distilgpt2 OOM solo en batch=64, no en batch=16; gpt2_small
   OOM solo en batch=32, no en batch=8). Descartado.
2. **Ruido/varianza en la medición por capa**: `gpu_bwd_mem_peak_mb_std` es 0.0 para TODAS las
   capas revisadas de resnet50/batch_64 (n_runs=2, perfectamente reproducible). Descartado --
   no es un artefacto de `max()` sobre muestras ruidosas.

## Causa raíz confirmada (reproducción directa, controlada)

El esquema de medición de memoria de backward por capa (`src/runner/training_profiler.py`,
`close_and_rearm_backward_segment` + `_arm_backward_marker`, líneas ~298-424) encadena eventos de
"gradiente de entrada listo" para delimitar segmentos: desde el evento anterior hasta el evento de
ESTE módulo, atribuyéndole esa ventana de memoria/tiempo. El diseño (ver comentario largo ya
existente en el código, líneas 298-333) es consciente de que PyTorch no permite usar
`register_full_backward_hook` en ResNet (el `out += identity` in-place de
`torchvision.models.resnet.Bottleneck` lo rompe) y por eso implementa esta alternativa basada en
`tensor.register_hook()` plano -- pero esa alternativa asume una dependencia de backward
estrictamente LINEAL (cada evento delimita limpiamente el segmento del módulo anterior).

**Un bloque residual real (bifurcación en la rama principal + rama de downsample, unidas en un
`add`) rompe ese supuesto**: el motor de autograd de PyTorch no garantiza que ambas ramas se
procesen en un orden "ordenado" relativo a los nombres de módulo -- puede intercalar/alternar el
orden real de ejecución entre la rama principal y la rama de downsample de forma distinta a la que
el esquema de "un evento cierra exactamente un segmento, siempre en el orden correcto" asume.

### Reproducción directa (no simulación aproximada -- usa literalmente `TrainingProfiler._register_hooks`/`_arm_backward_marker`/`_finalize_pending_backward_marker` del código real)

Bloque sintético con la MISMA estructura que `torchvision.models.resnet.Bottleneck` (confirmado via
`inspect.getsource`: conv1-bn1-relu, conv2-bn2-relu, conv3-bn3, rama downsample con conv+bn propios,
`out += identity` in-place, relu final -- y el `self.relu` es una ÚNICA instancia de `nn.ReLU`
reusada 3 veces, igual que el real):

```
=== medido por el esquema de hooks real del profiler ===
downsample_conv   bwd_mem_mb=0.516   (fwd=24.500)
downsample_bn     bwd_mem_mb=57.095  (fwd=24.501)   <- ~2.3x su propio tamaño de activación
conv1             bwd_mem_mb=0.000   (fwd=49.141)
bn1               bwd_mem_mb=24.501  (fwd=24.501)
relu (call 1)     bwd_mem_mb=24.500  (fwd=24.500)
conv2             bwd_mem_mb=81.595  (fwd=49.141)
bn2               bwd_mem_mb=24.501  (fwd=24.501)
relu (call 2)     bwd_mem_mb=24.500  (fwd=24.500)
conv3             bwd_mem_mb=81.595  (fwd=49.141)
bn3               bwd_mem_mb=24.501  (fwd=24.501)
relu_out (call 3) bwd_mem_mb=98.000  (fwd=24.500)   <- ~4x su propio tamaño de activación

=== referencia real aislada ===
Rama downsample (conv+bn) sola, backward aislado: 49.0 MB
```

`relu_out` (justo después del `add`) y `downsample_bn` muestran un costo de backward 2.3x-4x mayor
que su propio tamaño de activación -- desproporcionado para una capa de ese tamaño ejecutándose
sola. La magnitud exacta y QUÉ capa específica absorbe el exceso varía con la forma/canales (en
datos reales de resnet50 fue `layer2.0.bn3` quien mostró el pico extremo, 490.5MB contra sus 98MB
de activación propia -- 5x) porque depende de decisiones de scheduling del motor de autograd que no
están garantizadas por la API pública de PyTorch, no de un patrón fijo -- pero el mecanismo (una
ventana de medición que cruza la frontera de la bifurcación/unión y termina atribuyéndose a un nodo
que no la generó) es el mismo y está confirmado.

**Por qué esto explica el patrón observado:**
- resnet50/resnet152 (CNN, decenas de bloques residuales con rama downsample real -- compute de
  verdad, no solo una identidad) muestran la mayor sobreestimación (32%/27%).
- distilgpt2/gpt2_small (transformer, residual `x = x + sublayer(x)` -- la rama "atajo" es una
  identidad SIN módulo propio que hookear, nada que mal-atribuir desde ahí) muestran mucho menos
  efecto. gpt2_small (12 bloques) muestra más (14.7%) que distilgpt2 (6 bloques, 1.5%) -- consistente
  con que a más bloques, más oportunidades de que el orden de scheduling del `add` perturbe algún
  límite de segmento, aunque el efecto por bloque sea mucho más pequeño que en CNN al no haber una
  rama paralela con módulos propios.

## Decisión de cierre

Se evaluó reemplazar el mecanismo de medición por algo basado en `torch.profiler` (la vía "correcta"
en teoría). Probado directamente: `with_modules=True` está **deprecado y no hace nada en modo
eager** (confirmado con el propio warning de PyTorch); la alternativa real (`with_stack` +
números de secuencia forward/backward, o parsear snapshots del allocador CUDA vía
`torch.cuda.memory._record_memory_history`) es maquinaria nueva, no validada -- exactamente el tipo
de parche adicional que esta sesión decidió dejar de acumular.

**Decisión**: el hallazgo queda documentado como un sesgo CONSERVADOR conocido y acotado del
perfilado por capa (nunca produce subestimación ni OOM), no como un bug de seguridad. No se toca
`training_profiler.py`. En su lugar se verificó que la red de seguridad ya existente (`fix/08`:
expansión GPU + validación contra medición real) protege correctamente el único escenario donde
este sesgo podría importar en la práctica: presupuesto de GPU AJUSTADO (no el caso sin cortes que
mostró la tabla de arriba, donde el sesgo no tuvo efecto porque el plan óptimo ya cabía igual).

## Verificación final (presupuesto ajustado, con la red de seguridad activa)

Comando usado (`MEMORY_MODEL=topological_tight GPU_EXPANSION=true CALIBRATION_BATCH_SIZE=<batch>`,
presupuesto de 4094MB -- 50% de la GPU local de 8188MB, por debajo del pico real sin restricciones
de ambos modelos):

| modelo | solver cree (tras corrección) | **real medido, 10 pasos reales** | error final | capas GPU/total | OOM |
|---|---|---|---|---|---|
| resnet50 (budget efectivo 2671MB tras calibración de sobrecarga real) | 2086.4MB | **1990.1MB** | +4.8% | 58/126 (46%) | No |
| resnet152 (budget efectivo 3193MB tras calibración de sobrecarga real) | 2780.8MB | **2590.7MB** | +7.3% | 190/364 (52%) | No |

Nótese que en AMBOS casos `topology_memory`'s propia corrección (auto-consistente con el solver,
ver `fix/09`) seguía siendo optimista frente a una medición real de un solo pase (resnet50: creyó
2663.7MB, una sola pasada real midió 3760MB; resnet152: creyó 3189.3MB, midió 4124MB) -- la
validación contra memoria real de `fix/08` (`_validate_gpu_expansion_against_real_overhead`)
detectó esa brecha y recortó capas adicionales (20 y 55 respectivamente) ANTES de aceptar el plan
final. El error final reportado arriba (+4.8%/+7.3%) es contra una corrida real de 10 pasos
completa, no contra la sonda de calibración de un solo pase -- confirma que la red de seguridad
compensa de sobra el sesgo de este hallazgo (que por sí solo, sin corrección, habría sido
+32%/+27%). Las 4 metas quedan confirmadas con evidencia: objetivo ILP óptimo, sin OOM, error de
predicción final bajo (<8%), distribución CPU/GPU coherente y sin capas corruptas.

GPU vigilada con `nvidia-smi` antes/después de cada corrida (80MiB baseline, sin fugas).

## Qué no se intentó (alcance deliberado, no pendiente)

Reemplazar `training_profiler.py` por un mecanismo de atribución basado en `torch.profiler`
(`with_stack` + secuencia fwd/bwd, o snapshots del allocador CUDA) queda como trabajo futuro
explícito (ver capítulo de conclusiones de la tesis), no como una tarea abierta de esta
investigación -- la decisión de cierre arriba es definitiva para el alcance de esta tesis.
