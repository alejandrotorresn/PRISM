# Convergence Validation

Código para el ítem 1 de `fix/PLAN_ACCION_JURADO.md`: el jurado doctoral (García #1, Schnorr #1)
señaló que la tesis afirma preservar la *accuracy* y la convergencia del entrenamiento (hipótesis
1.4, objetivo específico 5) sin reportar evidencia — ni una curva de pérdida ni un valor de
accuracy. Este módulo entrena modelos reales, con datos reales que cambian en cada paso (no el
mismo minibatch repetido), para producir esa evidencia: PRISM vs. All-GPU, en FP32 y BF16.

La documentación de **qué se encontró y cómo se resolvió** vive en
[`fix/01_accuracy_convergence/README.md`](../fix/01_accuracy_convergence/README.md). Este archivo
documenta **cómo está construido el código**.

## Principio de diseño: no se modifica nada existente

Todo el código de este módulo es nuevo y vive aquí. **Ningún archivo bajo `src/`, `validation/`,
`scripts/` o `tests/` se edita.** Donde se reutiliza lógica ya validada (el intérprete que despacha
capas a CPU/GPU según el plan ILP, la función de pérdida/accuracy, el trazador de grafos), se hace
mediante `import` de solo lectura — exactamente igual que cualquier otro script del repo que ya
importa esos módulos (p. ej. `validation/run_hybrid_execution.py`). Importar un módulo de Python no
lo modifica ni lo ejecuta en modo escritura.

Esto es verificable: `git status`/`git diff` sobre `src/`, `validation/`, `scripts/`, `tests/` debe
mostrar cero cambios después de trabajar en este módulo, y la suite de tests existente
(`pytest tests/`) debe seguir pasando exactamente igual.

## Por qué no existe ya esta evidencia (resumen técnico)

El único camino del repo con pérdida/accuracy **reales** (no sintéticas) es
`src/runtime/hybrid_executor.py::run_hybrid_training`, pero fue diseñado para **perfilar** latencia,
no para entrenar: reutiliza el mismo minibatch fijo durante todos los `steps` de una llamada, sin
recorrer el dataset ni usar un split de validación. Ese diseño es correcto para su propósito (medir
cuánto tarda una capa), pero no permite observar si el modelo converge.

Este módulo añade lo que faltaba — ciclos reales de época, split train/val, pérdida/accuracy por
paso — **sin tocar ese archivo**, reconstruyendo solo el bucle externo (ver `prism_loop.py`) y
reutilizando por import las piezas internas ya probadas que deciden qué capa va a qué dispositivo.

## El pipeline no se repite desde cero

PRISM es *ahead-of-time*: el mapa de qué capa va a CPU/GPU se calcula una sola vez (perfilado + ILP)
y el entrenamiento corre sobre ese mapa ya fijo. Los 4 modelos de este estudio
(`resnet50`, `resnet152`, `distilgpt2`, `gpt2_small`) **ya tienen perfilado e ILP resueltos en
disco** de corridas previas de la tesis (`data/<host>/.../ilp_solution/` para el régimen de
presupuesto completo, `data/<host>/.../ilp_solution_pareto_best_budget_*/` para el régimen de
presupuesto reducido). Este módulo **lee esos `ilp_assignment.csv` existentes directamente**; no
relanza perfilado ni resolución ILP salvo que falte una combinación puntual, y en ese caso se
avisa antes de lanzar nada nuevo.

## Estructura

| Archivo | Qué hace |
|---|---|
| `datasets.py` | Dataloaders cíclicos reales (con *shuffle*, train/val separados) para `imagenette2-160` (visión) y `tiny_shakespeare` (lenguaje). Reutiliza por import los helpers de mapeo de clases de `src/data/dataset_registry.py` para que la definición de accuracy sea idéntica al resto de la tesis. |
| `prepare_tiny_shakespeare_split.py` | Script de una sola vez: baraja `datasets/tiny_shakespeare/train.txt` con semilla fija y separa un 10% como validación, sin tocar el dataset original. |
| `mixed_precision.py` | Copia maestra de los pesos en FP32 para BF16 (mixed-precision estándar): evita que el entrenamiento real de varias épocas sufra *weight-update underflow*, algo irrelevante en el perfilado corto actual pero real en un entrenamiento largo. |
| `all_gpu_loop.py` | Bucle de entrenamiento PyTorch estándar — All-GPU directo y All-GPU + *gradient accumulation* (para igualar el batch efectivo de PRISM en el régimen de rescate OOM). Sin dependencia de `hybrid_executor.py`. |
| `prism_loop.py` | Bucle de entrenamiento con el plan ILP: importa (solo lectura) el trazador de grafo y el intérprete de despacho CPU/GPU de `src/runtime/hybrid_executor.py`, traza **una sola vez** por corrida y entrena muchos pasos reales sobre ese grafo ya trazado. |
| `run_convergence_study.py` | CLI que orquesta una corrida: lee el `ilp_assignment.csv` correspondiente, llama al bucle adecuado, escribe los CSV de resultados. Modos: `ilp_plan`, `greedy_plan`, `all_gpu`, `all_gpu_gradaccum`. |
| `build_greedy_plan.py` | Construye un plan de partición CPU/GPU con el heurístico greedy que ya existe en `validation/sweep_ilp_pareto.py::_greedy_bits` (reutilizado por import, sin tocar ese archivo), al mismo presupuesto que el plan ILP de régimen B ya en disco, y lo escribe con el mismo esquema de CSV que un `ilp_solution*/` (`greedy_solution_budget_<N>/{ilp_assignment.csv,ilp_cut_edges.csv}`) para que `run_convergence_study.py --mode greedy_plan` lo lea sin ningún camino de código nuevo. |
| `run_convergence_study.sh` | Orquestador de la matriz completa (modelo × precisión × régimen × semilla), mismas convenciones de variables de entorno que `scripts/run_thesis_mode.sh`. Filtra la búsqueda del plan por host (`CONFIG_HOST_FILTER`, por defecto `HOST_TAG`) y por optimizador de perfilado, y usa batches **distintos por régimen** (`BATCH_SIZE_<modelo>_A` / `_B`, con defaults ya basados en evidencia real de OOM — ver `fix/01_accuracy_convergence/README.md`). Construye y corre `greedy_plan` automáticamente en régimen B (`RUN_GREEDY=true` por defecto). |
| `tests/test_convergence_smoke.py` | Test de humo: confirma que el dataloader realmente cicla (no repite el mismo batch) y que el grafo PRISM se traza una sola vez por corrida. |

Las campañas en clúster (PACCA vía SLURM, Grid5000) se lanzan con
`scripts/launch_slurm_v2.sh` / `scripts/launch_grid5k_v2.sh` — encadenan el perfilado+ILP ya
existente (`scripts/run_thesis_mode.sh`, sin cambios) con esta etapa de convergencia
(`RUN_CONVERGENCE=true` por defecto; `false` para el comportamiento idéntico a la v1, solo
perfilado). Las v1 (`launch_slurm.sh`, `launch_grid5k.sh`) quedan intactas.

## Cómo correr

```bash
# Una sola vez, antes de la primera corrida de lenguaje:
python convergence_validation/prepare_tiny_shakespeare_split.py

# Humo rápido (1 modelo, 1 época, 1 semilla) antes de lanzar la matriz completa:
MODELS_CSV=resnet50 EPOCHS=1 SEEDS_CSV=42 \
  bash convergence_validation/run_convergence_study.sh

# Matriz completa (ver scripts/run_thesis_mode.sh para el estilo de variables):
bash convergence_validation/run_convergence_study.sh
```

Salidas en `reports/<host>/convergence_study/<model>/<precision>/<regime>/seed_<N>/`:
`train_loss.csv` (pérdida por paso) y `eval_metrics.csv` (pérdida/accuracy periódica en el split de
validación real — esta es la curva de convergencia que cita el capítulo).

## Limitaciones metodológicas documentadas explícitamente

- **Régimen B, baseline secundario (`all_gpu_gradaccum`):** *gradient accumulation* no es
  numéricamente idéntico a un batch grande real cuando hay BatchNorm (normaliza por micro-batch, no
  por batch efectivo) — relevante para `resnet50`/`resnet152`, no para `distilgpt2`/`gpt2_small`
  (usan LayerNorm). Se reporta como comparación adicional, no como la única evidencia del régimen B.
- **BF16:** con copia maestra FP32 (mixed-precision estándar), no BF16 puro. Documentado en
  `mixed_precision.py`.
