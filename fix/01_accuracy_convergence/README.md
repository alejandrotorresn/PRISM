# Ítem 1 — Evidencia de accuracy/convergencia (PRISM vs All-GPU, FP32 y BF16)

Pedido por: **García #1**, **Schnorr #1**.

Nota de corrección: una versión anterior de este documento decía que este ítem fusionaba el ítem 31
(R² por familia de modelo). Eso fue un error — el ítem 31 es sobre la fidelidad predictiva del
*modelo de costo del ILP* (latencia estimada vs. medida), no sobre *accuracy* de entrenamiento; usa
los datos de perfilado ya existentes, no las corridas de `convergence_validation/`. Sigue fusionado
con el ítem 5, como ya indicaba `fix/PLAN_ACCION_JURADO.md`.

Estado: 🟡 **infraestructura completa y verificada — campaña completa pendiente de ejecutar**.

## El problema

La tesis afirma preservar la *accuracy* y la convergencia del entrenamiento (hipótesis 1.4,
objetivo específico 5, Cap. 8) pero no reporta ni una curva de pérdida ni un valor de accuracy en
ningún capítulo. Toda la evidencia existente es de **perfilado** (`src/profiler.py`,
`validation/run_hybrid_execution.py`): warmup + unas pocas iteraciones reutilizando el mismo
minibatch fijo, sin recorrer el dataset ni usar un split de validación real. Es correcto para medir
cuánto tarda una capa, pero no permite observar si el modelo converge.

## Qué se construyó

Código nuevo en [`convergence_validation/`](../../convergence_validation/) (ver su README para el
detalle técnico completo). En resumen:

- Dataloaders reales con *shuffle* y split train/val genuino (no el mismo batch repetido) para
  `imagenette2-160` (visión) y un split de `tiny_shakespeare` generado a propósito (lenguaje, no
  existía antes).
- Copia maestra de pesos en FP32 para BF16 (mixed-precision estándar), para que el entrenamiento
  real de varias épocas no sufra *weight-update underflow* — algo irrelevante en el perfilado corto
  actual pero real en un entrenamiento largo.
- Un bucle de entrenamiento PRISM que reutiliza (solo por `import`, sin modificar nada) el mismo
  motor de despacho CPU/GPU que usa el resto de la tesis, y un bucle All-GPU estándar aparte, con
  soporte de *gradient accumulation* para el régimen de rescate OOM.
- **Ningún archivo existente del repo fue modificado.** Verificado con `git status`/`git diff`
  sobre `src/`, `validation/`, `scripts/`, `tests/` antes y después del trabajo: cero cambios más
  allá de los que ya existían al empezar esta tarea.
- **No se relanzó perfilado ni resolución ILP.** El código lee directamente los
  `ilp_assignment.csv` ya calculados en campañas anteriores de la tesis (`data/<host>/.../batch_<N>/`),
  tanto el plan de presupuesto completo (régimen de suficiencia) como el de presupuesto reducido
  (régimen de rescate OOM), confirmados en disco para los 4 modelos del estudio antes de escribir
  una sola línea de código.

## Hallazgo nuevo durante la validación: learning rate por familia de modelo

Al probar el pipeline extremo a extremo (no solo en teoría — con datos y modelos reales), la
primera corrida de `distilgpt2` mostró la pérdida explotando de 6.23 (modelo preentrenado, sin
entrenar) a 10.88 (equivalente a predicción casi aleatoria, $\ln(\text{vocab\_size}) \approx 10.82$)
en solo 2 pasos de SGD con `lr=0.01` — el valor por defecto que sí funciona correctamente para las
CNN de visión (ver curvas de prueba abajo). `lr=0.01` con SGD es apropiado para afinar ligeramente
una CNN preentrenada, pero es catastróficamente agresivo para afinar un transformer preentrenado.

**Corrección aplicada:** `resnet50`/`resnet152` usan SGD con `lr=0.01` (sin cambios);
`distilgpt2`/`gpt2_small` usan AdamW con `lr=5e-5` (estándar para afinar transformers
preentrenados). Verificado que esto estabiliza la pérdida de `distilgpt2` (ver abajo). Esto no es
un ajuste cosmético: sin él, **las curvas de ambas familias de lenguaje habrían divergido
igual de mal en PRISM y en All-GPU**, haciendo que el experimento pareciera "no concluyente" por una
razón completamente ajena a la pregunta real (¿la partición CPU/GPU de PRISM afecta la
convergencia?).

## Verificación realizada (smoke tests, no la campaña completa)

Todo lo siguiente se corrió con datos reales (no sintéticos) contra planes ILP ya existentes en
disco, con pocos pasos (no la campaña completa de 5-10 épocas):

| Verificación | Resultado |
|---|---|
| `resnet50`, PRISM (`ilp_plan`), 3 pasos | pérdida 3.36→2.75, accuracy train 0.375→0.75, eval accuracy=0.875 sobre val real |
| `resnet50`, All-GPU, 3 pasos | pérdida 3.36→2.75, accuracy train 0.375→0.75, eval accuracy=0.875 — comparable a PRISM |
| `distilgpt2`, PRISM, preentrenado sin entrenar | pérdida 6.23 (confirma que el modelo preentrenado funciona correctamente antes de tocar el LR) |
| `distilgpt2`, PRISM, SGD lr=0.01, 2 pasos | pérdida 6.23→10.88 (diverge — motivó la corrección de LR) |
| `distilgpt2`, PRISM, AdamW lr=5e-5, 5 pasos | pérdida 6.33→5.82, estable y mejorando |
| `resnet50`, All-GPU + *gradient accumulation* (micro-batch 4, accum=2), 2 pasos | corre sin errores, eval accuracy=0.75 |
| Copia maestra FP32 para BF16 (test unitario) | los pesos cambian tras el paso, sin NaN, master en FP32 y modelo en BF16 tras la sincronización |
| `pytest convergence_validation/tests/` | 3/3 pasan (incluye verificación de que el trazado del grafo ocurre una sola vez por corrida, no una vez por paso) |
| `pytest tests/` (suite completa existente) | 95 pasan, 7 skip, 4 fallos — **los 4 fallos son por falta de `seaborn` en el entorno, no relacionados con este trabajo** (confirmado: mismo error en los 4, en un import de `validation/generate_ilp_report_assets.py` que ya fallaba antes) |
| `git status`/`git diff` sobre `src/`, `validation/`, `scripts/`, `tests/` | cero cambios nuevos — solo los 3 archivos que ya estaban modificados antes de empezar esta tarea |

## Actualización: `resnet152` y `gpt2_small` ya verificados (antes no se habían probado en absoluto)

| Modelo | Config | Resultado |
|---|---|---|
| `resnet152` batch 8, régimen A | PRISM, 5 pasos | loss 3.36→0.32, eval accuracy=0.975 |
| `resnet152` batch 8, régimen A | All-GPU, 5 pasos | loss→0.40, eval accuracy=0.95 — comparable a PRISM |
| `gpt2_small` batch 32, régimen A (presupuesto completo) | PRISM | **OOM** manejado limpiamente (`status=oom`, sin crash) — este modelo (12 capas, el doble que `distilgpt2`) no entra ni con el plan de presupuesto completo en la GPU local de 8GB |
| `gpt2_small` batch 32, régimen B (presupuesto reducido, `seq_length=64`) | PRISM, 5 pasos | loss→5.27, eval token_accuracy=0.186 — flujo dual real (Opción A) ejecutándose correctamente |

Con esto, los 4 modelos del estudio (`resnet50`, `resnet152`, `distilgpt2`, `gpt2_small`) tienen ya
al menos una verificación de convergencia extremo a extremo exitosa en esta máquina local.

**Incidente durante la verificación de `gpt2_small` régimen B (resuelto, documentado por su valor
como lección operativa):** el primer intento (batch 32, `seq_length=128`, presupuesto reducido)
terminó con `SIGKILL` del kernel (exit 137) por agotamiento de RAM del host — no un OOM de Python
manejado, sino el *out-of-memory killer* del sistema. Causa probable: `gpt2_small` tiene el doble de
capas que `distilgpt2` con el mismo tamaño oculto, y un presupuesto de GPU muy ajustado fuerza mucho
más cómputo (y recompute de verificación de flujo dual) a CPU, dentro de una máquina con solo 31GB
de RAM. Se repitió con `seq_length=64` y `eval_batches=2` (para bajar el pico de RAM), lanzando el
proceso en una terminal fuera de VSCode para que, si volvía a morir por falta de memoria, no se
llevara el editor con él — esa segunda corrida completó limpio. El sistema se recuperó solo en
ambos casos, sin dejar procesos huérfanos. No es un bug del mecanismo de flujo dual ni de
`convergence_validation/` — es, otra vez, un techo de RAM/VRAM de esta máquina de prueba local de
8GB/31GB, igual que el ya documentado para `resnet50` régimen B.

## Segunda fase: campaña real en clúster (PACCA + Grid5000), con greedy como tercer baseline

Con tiempo disponible para una corrida completa, se decidió relanzar desde cero en los clústers
reales (nodo `paccaA100.unicartagena.edu.co` en PACCA vía SLURM; algún nodo de `chuc`/`kinovis` en
Grid5000 vía OAR — exacto no determinable de antemano, pero misma configuración de hardware según
confirmó el usuario), en vez de seguir acotado a la GPU local de 8GB.

**Elección de batch/optimizador por régimen, basada en evidencia real (no en una suposición):** se
auditaron las columnas `all_gpu_status`/`all_gpu_gpu_mem_mb` ya presentes en los
`*_pareto_sweep.csv` que las campañas de perfilado previas dejaron en disco para `chuc-4`,
`kinovis-2` y `paccaA100...` (253 filas, los 4 modelos, batches 8–512). Resultado — **idéntico
umbral de OOM en los 3 hosts**, confirmando que tienen la misma configuración de hardware:

| Modelo | Optimizador | All-GPU cabe hasta | All-GPU hace OOM en |
|---|---|---|---|
| `resnet50` | SGD | batch 256 | batch 512 |
| `resnet152` | SGD | batch 128 | batch 256 |
| `distilgpt2` | AdamW | batch 512 (nunca hace OOM en el rango perfilado) | — |
| `gpt2_small` | AdamW | batch 512 (nunca hace OOM en el rango perfilado) | — |

De aquí: régimen A (resnet50=256, resnet152=128, LM=128) usa un batch donde All-GPU real funciona;
régimen B para visión (resnet50=512, resnet152=256) usa un batch donde All-GPU **realmente** hace
OOM (rescate genuino, no artificial); régimen B para los dos modelos de lenguaje sigue siendo vía
presupuesto artificialmente reducido + *gradient accumulation* (no hay OOM real que forzar en estas
GPUs de >32GB hasta batch 512). Estos valores quedaron como default en
`convergence_validation/run_convergence_study.sh` (`DEFAULT_BATCH_A`/`DEFAULT_BATCH_B`),
sobreescribibles con `BATCH_SIZE_<modelo>_<regimen>`.

**Tercer baseline añadido: greedy.** El heurístico greedy que ya existe en
`validation/sweep_ilp_pareto.py::_greedy_bits` (usado para las figuras del Cap. 6 de
latencia/energía) nunca persistía su asignación por capa a disco — solo agregados. Se añadió
`convergence_validation/build_greedy_plan.py`, que reutiliza por import esa misma función para
calcular la partición greedy al mismo presupuesto que el plan ILP de régimen B ya resuelto, y la
escribe con el mismo esquema de CSV que un `ilp_solution*/`. Esto permite correr `greedy_plan` a
través del mismo runtime de PRISM sin ningún camino de código nuevo en `hybrid_executor.py` —
verificado extremo a extremo localmente (`resnet152` batch 8: ILP 364/0 GPU/CPU, greedy 359/5 —
diverge donde se espera, confirma que el heurístico greedy es real, no un *stub*). Solo se corre en
régimen B (en régimen A, presupuesto completo, greedy y el ILP convergen trivialmente al mismo
"todo en GPU").

**Dos bugs encontrados y corregidos en `run_convergence_study.sh` durante esta auditoría** (archivo
propio, no se tocó nada existente):
1. El filtro de host (`CONFIG_HOST_FILTER`) usaba `find -path "*<host>*"` contra la ruta absoluta
   completa — pero el directorio home de esta máquina es literalmente `/home/zephyr/...`, el mismo
   nombre que el host `zephyr`, así que el filtro no excluía nada aquí y una corrida de prueba
   recogió sin querer un plan perfilado en `chuc-4`. Corregido restringiendo la búsqueda a los
   subdirectorios `data/<host>/` que coinciden con el filtro, no a toda la ruta absoluta.
2. `find_config_dir` tampoco filtraba por el optimizador de perfilado (`AdamW`/`SGD`/`RMSprop` en
   la ruta), pudiendo devolver un plan perfilado con un optimizador distinto al que realmente se
   entrena. Corregido exigiendo coincidencia exacta vía `optimizer_class_name()`.

**Datos antiguos archivados.** Todo lo que ya existía bajo `data/` y `reports/` (de corridas
previas de los 4 hosts) se movió a una única carpeta `data/legacy/` / `reports/legacy/` (mismo
contenido, solo reubicado — `data/` y `reports/` están en `.gitignore`, así que esto no afecta
control de versiones). Las campañas nuevas en PACCA/Grid5000 escriben en rutas limpias bajo
`data/<host>/`/`reports/<host>/` (recreadas desde cero, hermanas de `legacy/`) sin mezclarse con lo
anterior — `convergence_validation/run_convergence_study.sh::find_config_dir` excluye
explícitamente cualquier ruta bajo `legacy/` de su búsqueda, así que los datos archivados nunca
se usan automáticamente, solo quedan como referencia histórica.

**Scripts de lanzamiento v2.** `scripts/launch_slurm_v2.sh` (PACCA) y `scripts/launch_grid5k_v2.sh`
(Grid5000) son copias nuevas de las v1 (que quedan intactas) con una etapa adicional al final:
tras el perfilado+ILP de siempre, invocan `convergence_validation/run_convergence_study.sh` una
sola vez (no por cada semilla de perfilado — el plan ILP es una propiedad estructural del grafo,
no de la semilla de perfilado). `RUN_CONVERGENCE=false` reproduce el comportamiento exacto de la
v1 si se necesita solo perfilado.

## Qué falta

Lanzar las campañas v2 en los 3 hosts (`paccaA100.unicartagena.edu.co` vía
`launch_slurm_v2.sh`, un nodo `chuc`/`kinovis` vía `launch_grid5k_v2.sh`) y, cuando terminen,
incorporar las curvas de pérdida/accuracy reales (PRISM vs All-GPU vs Greedy, FP32) a
`reports/<host>/convergence_study/` al Capítulo 6 para cerrar este ítem como ✅ resuelto en
[`fix/PLAN_ACCION_JURADO.md`](../PLAN_ACCION_JURADO.md).
