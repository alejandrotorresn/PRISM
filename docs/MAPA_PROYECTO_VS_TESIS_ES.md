# Mapa del Proyecto frente al Documento Base de Tesis

## 1. Alcance del contraste

Este informe contrasta el contenido del documento adjunto `docs/A New Parallelization Approach in Deep Learning Using CPU.docx` con el estado real del repositorio a fecha 19 de marzo de 2026.

El objetivo no es resumir la tesis, sino responder tres preguntas practicas:

1. Que partes de la propuesta doctoral ya estan implementadas en el proyecto.
2. Que partes estan implementadas solo de forma parcial o bajo una aproximacion reducida.
3. Que elementos faltan todavia para poder afirmar que el proyecto realiza todo lo planteado en la tesis.

La comparacion se ha realizado contra el codigo ejecutable y los artefactos reales del repositorio, no solo contra la documentacion. Las evidencias principales provienen de `src/`, `validation/`, `scripts/` y `docs/`.

## 2. Lo que el documento de tesis exige realmente

El documento base plantea una propuesta doctoral mas amplia que un simple profiler o un ILP offline. La tesis define un marco metodologico completo para entrenamiento hibrido CPU-GPU con los siguientes compromisos nucleares:

- caracterizacion empirica de memoria, tiempos, energia y transferencias sobre hardware real;
- formulacion ILP sobre el grafo de entrenamiento;
- asignacion por capa entre CPU y GPU;
- tratamiento explicito de persistencia de activaciones;
- integracion de rematerializacion o checkpointing como decisiones del modelo;
- integracion de streaming y, cuando proceda, prefetching;
- resolucion del modelo y simulacion de planes hibridos;
- validacion experimental fisica de los planes optimos;
- comparacion con baselines tradicionales;
- comprobacion de que la exactitud final del modelo no se degrada.

Ademas, la introduccion del documento afirma algo especialmente exigente: la asignacion optima de capas en forward y backward de forma independiente. Ese punto eleva el alcance por encima de una formulacion binaria simple por capa.

## 3. Mapa estructural del proyecto actual

El repositorio real ya implementa un pipeline tecnico considerable. Su arquitectura funcional puede resumirse en cinco bloques.

### 3.1 Profiling y captura de artefactos

El punto de entrada `src/profiler.py` orquesta la ejecucion del profiling y delega el trabajo principal en `src/runner/training_profiler.py`. Este bloque genera metricas por capa, metadatos globales, artefactos de grafo y artefactos de transferencia. Tambien contiene politica de precision, configuracion del entorno y medidas de robustez operativa.

### 3.2 Extraccion de grafo y costos de transferencia

`src/core/graph_extractor.py` exporta nodos y aristas del DAG mediante `torch.fx` y usa una aproximacion de respaldo basada en modulos hoja cuando el trazado falla. `src/runner/training_profiler.py` mide y calibra costos PCIe, incluido un estimador de solapamiento basico.

### 3.3 Agregacion estadistica robusta

`src/core/stats_aggregator.py` agrega replicas y produce estadisticos robustos por capa. Esto supera incluso una version minima del proyecto, porque permite alimentar el ILP con medias, dispersion y percentiles en lugar de una sola corrida.

### 3.4 Carga de datos ILP, formulacion y resolucion

`src/ilp/data_loader.py`, `src/ilp/model_builder.py` y `src/ilp/solve.py` construyen un flujo coherente de optimizacion: carga de costos robustos, construccion de la funcion objetivo, restricciones de memoria CPU/GPU y resolucion por PuLP/CBC o, en pequeno tamano, por enumeracion exhaustiva.

### 3.5 Automatizacion, Pareto y reporting

`validation/run_ilp_partition.py`, `validation/sweep_ilp_pareto.py`, `validation/generate_ilp_report_assets.py`, `validation/export_ilp_tables_latex.py` y `scripts/run_thesis_smoke_workflow.sh` convierten el repositorio en un pipeline reproducible de perfilado, agregacion, resolucion ILP, barrido Pareto y generacion de activos para reporte.

## 4. Cobertura de la tesis: lo que ya esta implementado

La siguiente tabla resume las capacidades que si estan presentes en el codigo y que alinean claramente con la propuesta doctoral.

| Componente de la tesis | Evidencia en el repositorio | Estado |
| --- | --- | --- |
| Caracterizacion empirica por capa de tiempo, energia, memoria y FLOPs | `src/runner/training_profiler.py`, `src/core/energy.py`, `src/core/metrics.py` | Implementado |
| Politica de precision y control de ejecucion por dispositivo | `src/core/precision_policy.py`, `src/profiler.py` | Implementado |
| Exportacion de artefactos por corrida | `src/core/io_artifacts.py`, `src/runner/training_profiler.py` | Implementado |
| Extraccion del grafo de computo y exportacion de nodos/aristas | `src/core/graph_extractor.py` | Implementado |
| Costos de transferencia entre capas para el ILP | `src/runner/training_profiler.py`, `src/ilp/data_loader.py` | Implementado |
| Agregacion robusta de replicas | `src/core/stats_aggregator.py`, `validation/aggregate_metrics_stats.py` | Implementado |
| ILP de asignacion CPU/GPU por capa | `src/ilp/model_builder.py`, `src/ilp/solve.py` | Implementado |
| Restricciones de memoria CPU/GPU | `src/ilp/solve.py` | Implementado |
| Baselines `all_cpu` y `all_gpu` en barridos Pareto | `validation/sweep_ilp_pareto.py` | Implementado |
| Exportacion de resultados y tablas para reporte | `validation/generate_ilp_report_assets.py`, `validation/export_ilp_tables_latex.py` | Implementado |
| Flujo reproducible de smoke end-to-end | `scripts/run_thesis_smoke_workflow.sh` | Implementado |
| Fusion multi-hardware de perfiles para ILP robusto | `src/ilp/data_loader.py`, `validation/run_ilp_partition.py` | Implementado |

### 4.1 Fortalezas reales del proyecto

El proyecto actual es especialmente fuerte en dos dimensiones.

La primera es la capa de observacion. El repositorio ya dispone de una instrumentacion util para tesis, con artefactos por capa, salida estructurada, soporte de energia y una cadena de datos que llega de forma razonablemente limpia hasta la optimizacion.

La segunda es la formalizacion del problema de particion. El ILP existente no es una maqueta superficial: ya modela nodos, cortes de arista, presupuestos de memoria y barridos de presupuesto GPU, y produce salidas explotables para analisis y generacion de tablas.

En otras palabras, el repositorio ya resuelve bien la parte "profilar, construir costos y optimizar una asignacion offline".

## 5. Cobertura parcial: lo que existe, pero en una version mas debil que la tesis

Aqui aparecen los primeros desajustes entre la ambicion del documento base y lo que el codigo hace hoy.

| Requisito del documento | Situacion actual | Evaluacion |
| --- | --- | --- |
| Asignacion de cargas entre CPU y GPU | El proyecto perfila CPU y GPU y luego optimiza una asignacion offline por capa | Parcial |
| Robustez frente a heterogeneidad | Existe agregacion multi-hardware y `mu + k*sigma` en la carga ILP | Parcial, pero bien encaminado |
| Modelado de comunicacion y solapamiento | Se calibra PCIe y se estima un ratio de overlap | Parcial |
| Validacion experimental doctoral | Existe smoke workflow y activos de reporte | Parcial |

La razon por la cual estos puntos se clasifican como parciales es que todavia no cierran el bucle completo entre plan ILP y ejecucion fisica hibrida real. El repositorio genera el plan, pero no lo materializa aun como un motor de entrenamiento distribuido capa a capa sobre CPU y GPU.

## 6. Brechas criticas: lo que falta respecto al documento de tesis

Esta es la seccion decisiva. Si la exigencia es que el proyecto pueda hacer "todo lo planteado", estas capacidades faltantes no pueden considerarse accesorias; varias son nucleares.

### 6.1 No existe ejecucion hibrida real guiada por la solucion ILP

El repositorio optimiza una asignacion, pero no implementa un ejecutor que tome `ilp_assignment.csv` y entrene el modelo colocando realmente capas concretas en CPU o GPU bajo ese plan. No aparece un modulo de tipo `executor`, `runtime planner`, `hybrid trainer` o equivalente.

Consecuencia: hoy el proyecto demuestra capacidad de planificacion offline, no entrenamiento hibrido real validado en hardware.

### 6.2 No hay decisiones separadas para forward y backward

El documento base afirma asignacion optima de capas en forward y backward de forma independiente. Sin embargo, `src/ilp/solve.py` usa una sola variable binaria por nodo, y `src/ilp/model_builder.py` construye el costo de cada nodo sumando costo forward y backward por dispositivo. No existe un juego separado de variables para forward y backward.

Consecuencia: el codigo implementa una asignacion unica por capa, no una asignacion diferenciada por fase.

### 6.3 Rematerializacion y checkpointing no son variables de decision del ILP

La tesis los presenta como parte del nucleo del metodo. En el codigo actual no existen variables binarias o restricciones que decidan conservar activaciones, recomputarlas o checkpointarlas. Tampoco aparece un modulo que modele memoria temporal bajo esas decisiones.

Consecuencia: el ILP actual resuelve un problema mas simple que el propuesto en la tesis.

### 6.4 Streaming y prefetching no estan integrados en la optimizacion

En `src/runner/training_profiler.py` si existe una medicion de overlap y uso de `torch.cuda.Stream()` para calibracion. Sin embargo, eso no equivale a modelar streaming o prefetching como decisiones ILP ni a ejecutar planes asincronos reales. No se observan variables, restricciones ni politicas de scheduling que representen esas tecnicas.

Consecuencia: la tesis describe tecnicas de comunicacion asincrona que el proyecto aun no operacionaliza.

### 6.5 No existe simulador de ejecucion hibrida

El documento base habla de resolver instancias y luego simular planes de ejecucion para verificar topologia, penalizaciones temporales, presion de memoria y eficiencia de asignacion. En el repositorio no aparece un simulador dedicado que reconstruya la secuencia hibrida a partir de la solucion ILP.

Consecuencia: falta una capa intermedia clave entre la optimizacion y la validacion fisica.

### 6.6 No existe validacion fisica de los planes ILP

Los scripts actuales ejecutan profiling, agregacion, ILP, Pareto y reporting. Pero no existe un flujo que aplique un plan ILP sobre PyTorch y compare prediccion frente a observacion fisica. Tampoco existe el `validation/validate_ilp_pipeline.py` planteado en la hoja de ruta.

Consecuencia: no puede afirmarse todavia que el ILP predice o mejora la ejecucion real en maquina.

### 6.7 Falta comparacion frente a una heuristica greedy

`validation/sweep_ilp_pareto.py` compara ILP contra `all_cpu` y `all_gpu`, pero no implementa la heuristica greedy recomendada en la propia documentacion para medir la ganancia incremental del modelo.

Consecuencia: la validacion comparativa esta incompleta para una tesis.

### 6.8 Faltan estudios de ablacion

La documentacion metodologica propone ablaciones obligatorias: sin topologia, sin costos de transferencia por arista, sin robustez estadistica y modelo completo. En el codigo no aparece un harness de ablacion ni un script equivalente.

Consecuencia: aun no se puede demostrar experimentalmente la contribucion marginal de cada componente del metodo.

### 6.9 No se valida la exactitud final del modelo bajo ejecucion hibrida

La tesis exige verificar que la precision final no se degrade. El codigo actual mide profiling y resuelve ILP, pero no ejecuta un entrenamiento hibrido real con seguimiento de accuracy, loss final o metricas equivalentes de calidad del modelo.

Consecuencia: falta una de las condiciones centrales de la hipotesis doctoral.

### 6.10 La instrumentacion mencionada en el documento es mas amplia que la implementada

El documento base menciona PyTorch Profiler, Nsight Systems y NVTX como herramientas de instrumentacion. En el repositorio no aparece evidencia de integracion con `torch.profiler`, NVTX o Nsight. La instrumentacion se basa sobre todo en hooks, temporizacion propia y telemetria NVML/RAPL.

Consecuencia: la capa experimental del repositorio es util, pero no coincide exactamente con la promesa metodologica del documento.

## 7. Diagnostico sintetico

La mejor descripcion del estado actual es la siguiente:

El proyecto implementa de forma solida el subsistema de observacion empirica y una version funcional del subsistema de optimizacion offline. Sin embargo, todavia no implementa el subsistema de ejecucion hibrida real y validacion experimental completa que el documento doctoral presenta como parte esencial de la contribucion.

Por tanto, si el criterio es estricto, el proyecto hoy cubre bien la parte:

- medir;
- estructurar artefactos;
- construir costos robustos;
- resolver un ILP de particion;
- generar reportes.

Pero no cubre todavia, al menos no de manera completa, la parte:

- ejecutar el entrenamiento hibrido resultante;
- implementar y validar rematerializacion o checkpointing dentro del modelo;
- implementar y validar streaming o prefetching como decisiones operativas de scheduling;
- validar en hardware los planes ILP;
- demostrar preservacion de exactitud final;
- comparar contra una heuristica greedy;
- realizar ablaciones formales.

## 8. Prioridad de trabajo para que el proyecto cumpla la tesis

Si el objetivo es que el proyecto pueda hacer todo lo planteado, el orden recomendado de implementacion no deberia ser arbitrario. La prioridad tecnica razonable es la siguiente.

### Prioridad 1: cerrar la brecha funcional principal

1. Implementar un ejecutor de entrenamiento hibrido que consuma la solucion ILP y aplique asignacion real de capas a CPU y GPU.
2. Extender el ILP y el runtime para mantener la promesa original de decisiones independientes para forward y backward.
3. Incorporar variables de persistencia de activaciones, rematerializacion y checkpointing al modelo matematico y a la contabilidad de memoria.

### Prioridad 2: cerrar la brecha de validacion doctoral

1. Construir un simulador de planes hibridos que estime latencia, memoria y comunicacion a partir de la solucion.
2. Implementar validacion fisica de planes ILP en PyTorch y medir diferencia entre prediccion y observacion.
3. Medir accuracy, loss final o metricas equivalentes para demostrar que la exactitud no se degrada.

### Prioridad 3: cerrar la brecha comparativa y metodologica

1. Anadir baseline greedy.
2. Anadir harness de ablaciones.

## 9. Conclusion ejecutiva

El proyecto no esta vacio ni atrasado respecto a la tesis; al contrario, ya contiene una base tecnica fuerte y bastante madura para la mitad mas estructural del trabajo. Lo que ocurre es que el documento doctoral describe un sistema mas ambicioso que el software actualmente operativo.

La diferencia principal puede expresarse asi: el repositorio ya sabe medir y decidir; todavia no sabe ejecutar y validar plenamente la decision en el regimen hibrido que la tesis promete.

Dado que el alcance queda fijado en su formulacion fuerte (forward/backward independientes, persistencia de activaciones y scheduling asincrono), la ruta de cierre requiere completar todos los componentes esenciales del plan por fases antes de afirmar cumplimiento integral de la tesis.