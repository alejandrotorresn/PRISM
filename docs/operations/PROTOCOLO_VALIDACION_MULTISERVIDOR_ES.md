# Protocolo de Validacion Multi-Servidor de PRISM

## 1. Proposito y justificacion

El siguiente paso de PRISM no consiste principalmente en ampliar el codigo, sino en consolidar una base empirica que permita sostener con rigor doctoral las afirmaciones sobre robustez, transferibilidad y validez operacional del pipeline completo. En consecuencia, la tarea inmediata es ejecutar una campana controlada de profiling en multiples servidores heterogeneos, convertir esos artefactos en un dataset multi-hardware coherente y, a partir de dicho dataset, revalidar el flujo completo que enlaza medicion, agregacion, optimizacion ILP, barrido Pareto, ejecucion hibrida y reporte final.

Este protocolo pasa a ser el documento operativo maestro para la toma de datos real. Absorbe la verificacion previa a servidores y la lista de control por host, de modo que preparacion, despliegue y criterio de aceptacion queden concentrados en una sola pieza.

La razon metodologica es directa. Una tesis de sistemas que optimiza decisiones de colocacion CPU-GPU no puede apoyarse de manera suficiente en coeficientes obtenidos de una unica maquina. Si los costos de tiempo, memoria, energia y transferencia cambian con la arquitectura del host, entonces la credibilidad de la formulacion depende de demostrar que el modelo se mantiene interpretable y util cuando los perfiles provienen de hardware distinto. El objetivo de este protocolo es precisamente producir esa evidencia.

## 2. Objetivo cientifico de la campana

La campana debe demostrar cuatro propiedades. En primer lugar, que el sistema puede medir de forma reproducible las mismas configuraciones experimentales en servidores diferentes y conservar una estructura de artefactos compatible. En segundo lugar, que esos perfiles pueden fusionarse de forma controlada para construir una instancia ILP robusta multi-hardware. En tercer lugar, que el pipeline sigue siendo ejecutable de extremo a extremo cuando las soluciones se obtienen a partir de dicha agregacion. En cuarto lugar, que los resultados permiten comparar de manera defendible el comportamiento de `all_cpu`, `all_gpu`, `greedy` e `ilp_plan` no solo en tiempo, sino tambien en memoria GPU y calidad final de entrenamiento.

## 3. Pregunta operativa central

La respuesta corta a la pregunta del proyecto es afirmativa: ahora hace falta correr la toma de muestras con profiling en diferentes servidores y construir un dataset con todos los datos necesarios para validar. Sin embargo, ese dataset no debe entenderse como un simple repositorio de CSVs, sino como un corpus experimental normalizado cuyo contenido permita reejecutar y auditar el pipeline doctoral completo.

## 4. Diseno de la campana

### 4.1 Unidad experimental

La unidad experimental debe ser una configuracion comun definida por el cuarteto modelo-optimizador-precision-batch. Esa misma configuracion debe ejecutarse en todos los servidores que pertenezcan a una misma clase de comparacion. El repositorio ya impone una convencion de salida apta para este objetivo bajo el arbol `data/<hostname>/results/...`, tal como se documenta en [MULTI_NODE_ILP_RUNBOOK.md](MULTI_NODE_ILP_RUNBOOK.md) y [SERVER_LAUNCH_PROFILES.md](SERVER_LAUNCH_PROFILES.md).

### 4.2 Clases de servidor

La campana no debe tratar cada host como un caso aislado, sino agruparlos por clases razonables de hardware. Conviene distinguir, al menos, nodos GPU bien instrumentados con FP32 CPU+GPU, nodos GPU orientados a multiprecision donde el CPU no ofrezca soporte acelerado fiable, y nodos de memoria o VRAM restringida. Esta separacion ya esta anticipada por los perfiles de lanzamiento del repositorio y evita mezclar supuestos incompatibles en una misma tanda de resultados.

### 4.3 Modelos y configuraciones minimas

Para una validacion doctoral eficiente, la campana debe estructurarse en dos niveles. El primero es una malla canonica comun a todos los hosts, pensada para comparabilidad transversal. El segundo es una malla extendida por clase de hardware, pensada para aprovechar capacidades adicionales sin contaminar la linea base.

La malla canonica recomendada es la siguiente:

1. `simple_mlp`, `resnet50` y `vit_b16` como nucleo minimo transversal.
2. `bert_base`, `gpt2_small` y `distilgpt2` solo en nodos donde el preflight de NLP sea satisfactorio.
3. `fp32` como precision obligatoria de referencia.
4. `bf16` y `fp16` solo cuando la clase de servidor lo soporte de forma valida y reproducible.
5. `SGD` y `AdamW` como optimizadores obligatorios; `RMSprop` puede quedar como ampliacion de la linea base.
6. Batches `8,16,32` como conjunto comun; `64` solo cuando el host lo soporte sin introducir una tasa excesiva de OOM o descartes.

## 5. Secuencia de ejecucion por servidor

La ejecucion en cada servidor debe seguir siempre la misma secuencia, para evitar que el dataset final mezcle corridas de distinta calidad metodologica.

### 5.1 Etapa A: preflight ambiental

Antes de cualquier toma de muestras, cada servidor debe pasar por una validacion de entorno. El objetivo no es medir rendimiento, sino descartar fallos de interpretacion, CUDA, datasets, precision o instrumentacion. La referencia operativa del repositorio es el `Profile 0` y el `Profile 5` descritos en [SERVER_LAUNCH_PROFILES.md](SERVER_LAUNCH_PROFILES.md).

Comando recomendado de preflight real minimo:

```bash
MODELS_CSV=simple_mlp \
BATCH_SIZES_CSV=8 \
PRECISIONS_CSV=fp32 \
OPTIMIZERS_CSV=SGD \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
REPEATS=1 \
WARMUP=1 \
MEASURE=1 \
FAIL_FAST=true \
DATASETS_DIR=datasets \
DOWNLOAD_DATASETS=true \
PYTHON_CMD=python \
bash scripts/run_experiments.sh
```

### 5.2 Etapa B: profiling productivo por clase de servidor

Superado el preflight, cada servidor debe ejecutar un perfil estable de campana. La seleccion no debe improvisarse nodo a nodo; debe seguir las clases ya documentadas por el proyecto.

Para nodos bien instrumentados CPU+GPU orientados a baseline doctoral, la plantilla preferente es equivalente a `Profile 2`:

```bash
MODELS_CSV=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2 \
BATCH_SIZES_CSV=8,16,32,64 \
PRECISIONS_CSV=fp32 \
OPTIMIZERS_CSV=SGD,AdamW,RMSprop \
USE_SKIP_CPU=false \
ENABLE_RAPL=true \
FORCE_THREADS=16 \
REPEATS=5 \
WARMUP=3 \
MEASURE=10 \
FAIL_FAST=false \
DATASETS_DIR=datasets \
DOWNLOAD_DATASETS=true \
PYTHON_CMD=python \
bash scripts/run_experiments.sh
```

Para nodos GPU donde interese ampliar precision sin bloquearse por limitaciones del CPU, la plantilla debe aproximarse a `Profile 1`:

```bash
MODELS_CSV=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2 \
BATCH_SIZES_CSV=8,16,32,64 \
PRECISIONS_CSV=fp32,fp16,bf16 \
OPTIMIZERS_CSV=SGD,AdamW \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
FORCE_THREADS=8 \
REPEATS=5 \
WARMUP=3 \
MEASURE=10 \
FAIL_FAST=false \
DATASETS_DIR=datasets \
DOWNLOAD_DATASETS=true \
PYTHON_CMD=python \
bash scripts/run_experiments.sh
```

Para nodos de VRAM restringida, la campana debe reducirse en batches y modelos antes de sacrificar comparabilidad interna:

```bash
MODELS_CSV=simple_mlp,resnet50,vit_b16 \
BATCH_SIZES_CSV=8,16 \
PRECISIONS_CSV=fp32,fp16 \
OPTIMIZERS_CSV=SGD \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
FORCE_THREADS=4 \
REPEATS=3 \
WARMUP=2 \
MEASURE=5 \
FAIL_FAST=false \
DATASETS_DIR=datasets \
DOWNLOAD_DATASETS=true \
PYTHON_CMD=python \
bash scripts/run_experiments.sh
```

## 6. Contenido minimo del dataset multi-hardware

El dataset consolidado no debe limitarse a guardar el archivo agregado `metrics_stats.csv`. Debe preservar, por cada configuracion y por cada host, al menos los siguientes elementos:

1. Metricas forward y backward por capa, con medias y dispersion.
2. Energia por capa o al menos energia atribuida de forma consistente por dispositivo.
3. Memoria GPU y CPU por capa.
4. Artefactos de grafo `*_graph_edges.csv`.
5. Artefactos de transferencia `*_transfer_edges.csv`.
6. Metadatos del host: hostname, CPU, GPU, VRAM, RAM, CUDA, PyTorch, numero de hilos, politica de precision.
7. Indicadores de calidad muestral y procedencia estructural.
8. Registro del dataset real utilizado para alimentar profiling y runtime.

En la practica, el dataset debe poder reconstruirse recorriendo el arbol host-scoped de `data/<hostname>/results/...` y, cuando se haga una consolidacion externa, no se deben perder las referencias a los artefactos fuente.

## 7. Criterios de suficiencia muestral

Para uso doctoral, no conviene cerrar la campana con replicas demasiado escasas. La recomendacion operativa es exigir al menos cinco replicas por configuracion en la malla principal. Si una clase de servidor solo puede costear una malla reducida, debe reducir dimensiones experimentales antes que degradar severamente la calidad estadistica de las configuraciones que permanezcan.

La aceptacion minima por configuracion debe verificar:

1. Existencia de `metrics_stats.csv` con banderas de calidad auditables.
2. Existencia de `graph_edges` y `transfer_edges` validos.
3. Ausencia de fallback silencioso no declarado en precision o topologia.
4. Trazabilidad explicita del host origen.
5. Repetibilidad suficiente para que el ILP no se alimente de coeficientes espurios.

## 8. Construccion de la instancia multi-servidor

Una vez que varias maquinas han producido la misma configuracion experimental, la fusion debe realizarse con los mecanismos ya previstos por el proyecto. La ruta recomendada es descubrir directorios compatibles y resolver despues con agregacion multi-hardware.

Ejemplo de descubrimiento:

```bash
MODEL=resnet50 OPTIMIZER=AdamW PRECISION=fp32 BATCH=32 \
bash scripts/discover_ilp_config_dirs.sh
```

Ejemplo de particion ILP robusta con envolvente conservadora:

```bash
CONFIG_DIRS="data/nodeA/results/resnet50/AdamW/fp32/batch_32,data/nodeB/results/resnet50/AdamW/fp32/batch_32,data/nodeC/results/resnet50/AdamW/fp32/batch_32" \
MODEL=resnet50 \
HW_AGGREGATE=max \
HW_DISPERSION_K=0.0 \
bash scripts/run_ilp_partition.sh
```

Ejemplo de barrido Pareto sobre perfiles fusionados:

```bash
CONFIG_DIRS="data/nodeA/results/resnet50/AdamW/fp32/batch_32,data/nodeB/results/resnet50/AdamW/fp32/batch_32,data/nodeC/results/resnet50/AdamW/fp32/batch_32" \
MODEL=resnet50 \
GPU_BUDGETS_MB=400,800,1200,1600 \
HW_AGGREGATE=max \
HW_DISPERSION_K=0.0 \
bash scripts/run_ilp_pareto_sweep.sh
```

La politica inicial recomendable es `HW_AGGREGATE=max`, porque establece una envolvente conservadora defendible. En una segunda pasada, si la tesis quiere mostrar calibracion de robustez y no solo peor caso, puede repetirse la resolucion con `HW_AGGREGATE=mean` y un `HW_DISPERSION_K` positivo.

## 9. Validacion end-to-end posterior

El dataset multi-hardware cumple su funcion solo si activa la validacion posterior. Tras la fusion de perfiles, deben ejecutarse tres pasos adicionales sobre una seleccion representativa de configuraciones:

1. Resolver el ILP y el barrido Pareto con perfiles fusionados.
2. Ejecutar runtime hibrido con la solucion Pareto seleccionada, no con una solucion unrestricted por defecto.
3. Consolidar reportes finales que incluyan tiempo, memoria GPU, energia y calidad final del modelo.

Este tercer punto es especialmente importante. El repositorio ya puede registrar la procedencia del plan hibrido y priorizar filas `pareto_best`, por lo que la campana final debe explotar esa capacidad y no volver a una evidencia ambigua.

## 10. Criterios de cierre doctoral de la campana

La campana puede considerarse metodologicamente suficiente cuando se cumplan simultaneamente las condiciones siguientes. Debe existir una malla comun ejecutada en varias clases de servidor. Debe poder construirse al menos una instancia ILP robusta a partir de perfiles fusionados. Deben existir comparaciones explicitas frente a `all_cpu`, `all_gpu` y `greedy`. Deben haberse generado artefactos finales de reporte y tablas. Y, por ultimo, debe existir evidencia de calidad final del entrenamiento, no solo de coste de ejecucion.

Si alguno de estos puntos falta, el dataset podra ser util como prevalidacion, pero no como cierre empirico doctoral pleno.

## 11. Recomendacion operativa inmediata

La recomendacion mas pragmatica es comenzar con una campana transversal corta, pero estadisticamente limpia, en tres clases de servidor. Primero debe ejecutarse una malla canonica comun con `simple_mlp`, `resnet50` y `vit_b16`, `fp32`, `SGD` y `AdamW`, batches `8,16,32`, y al menos cinco replicas. Una vez consolidada esa base, deben seleccionarse dos o tres configuraciones representativas para resolver ILP multi-hardware, barrido Pareto y ejecucion hibrida alineada con el plan Pareto. Solo despues conviene ampliar a NLP o a precisiones avanzadas.

Ese orden maximiza valor cientifico por unidad de tiempo y evita producir una gran masa de datos que luego no pueda defenderse de manera integrada.

## 12. Estado operativo consolidado del repositorio

Al momento de iniciar la campana real, el repositorio se considera operativo y metodologicamente consistente para producir evidencia utilizable. Esa conclusion no descansa en una declaracion informal, sino en verificaciones efectivamente ejecutadas sobre codigo, modelos y orquestadores.

La base minima ya comprobada incluye validacion estructural integral mediante `validation/comprehensive_check.sh`, validacion de integridad del codigo con `validation/validate_code.py`, carga correcta de los siete modelos soportados con `validation/validate_all_models.py --preflight-scope fast`, y paso satisfactorio de la suite automatizada `validation/run_unit_tests.sh`. Adicionalmente, los orquestadores `scripts/run_experiments.sh` y `scripts/run_thesis_mode.sh` ya fueron verificados en modo seco, con construccion correcta de comandos, deteccion de entorno y secuencia de pasos coherente.

La implicacion practica es clara: el riesgo principal ya no reside en una carencia estructural del software, sino en la heterogeneidad real de los hosts y en la disciplina con la que se preserve el contrato experimental. Por ello, el control operativo debe desplazarse desde la pregunta de si el repositorio funciona hacia la pregunta de si cada servidor concreto respeta las condiciones bajo las cuales el repositorio fue validado.

## 13. Checklist minimo por servidor

Antes de incluir un host en la malla principal, debe verificarse lo siguiente:

- El hostname, la clase de servidor, CPU, GPU, RAM, VRAM, version de CUDA y version de PyTorch quedaron registrados.
- `PYTHON_CMD` apunta a un ejecutable simple y no a un comando con espacios.
- El entorno resuelve imports del proyecto sin errores y los datasets requeridos estan disponibles o pueden descargarse.
- El host aprobo `Profile 0` en modo seco.
- El host aprobo `Profile 5` como smoke real canonico de su clase.
- La salida se genero bajo `data/<hostname>/...` sin mezclar resultados con otros hosts.
- La configuracion productiva elegida coincide con un perfil documentado y no con una combinacion ad hoc.
- Los artefactos `*_metrics.csv`, `*_meta.json`, `*_graph_edges.csv`, `*_transfer_edges.csv` y `*_metrics_stats.csv` existen en las configuraciones retenidas.
- No hay fallback silencioso de precision ni topologia estructural degradada usada como evidencia principal.
- Las configuraciones destinadas a evidencia doctoral principal alcanzan una calidad muestral suficiente y comparable entre hosts.

Una plantilla resumida de registro por host puede mantenerse con los siguientes campos: hostname, clase de servidor, perfil elegido, estado de `Profile 0`, estado de `Profile 5`, disponibilidad de CPU profiling, disponibilidad de RAPL, estado BF16, integridad host-scoped de la salida, disponibilidad de artefactos ILP-ready, veredicto final Go/No-Go y observaciones.