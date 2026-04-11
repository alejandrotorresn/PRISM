# Estructura del Proyecto PRISM

## 1. Alcance de este documento

Este documento describe la organizacion efectiva del repositorio de PRISM en su estado actual y explica la responsabilidad metodologica de cada bloque. Su finalidad no es duplicar la documentacion global ni sustituir los runbooks operativos, sino ofrecer una cartografia estable del proyecto para que la lectura del codigo, la ejecucion experimental y la integracion con la tesis compartan una misma referencia estructural.

En consecuencia, la estructura se presenta aqui no solo como un arbol de directorios, sino como una descomposicion funcional del pipeline completo: preparacion de datos, profiling empirico, agregacion estadistica, particionamiento ILP, simulacion o ejecucion hibrida, generacion de reportes y consumo final en el manuscrito doctoral.

## 2. Vista general del repositorio

La raiz del proyecto se organiza alrededor de once bloques principales y dos archivos de control. `config/` contiene la definicion del entorno reproducible; `data/` preserva los resultados experimentales y fixtures de validacion; `datasets/` actua como deposito de datos persistentes para los modelos instrumentados; `docs/` concentra la documentacion tecnica y operativa; `logs/` conserva trazas de ejecucion; `reports/` almacena salidas agregadas y activos listos para reporte; `scripts/` contiene los puntos de entrada de automatizacion; `src/` implementa la logica central del sistema; `tests/` recoge la suite de pruebas automatizadas; `thesis/` integra el manuscrito LaTeX y sus artefactos; y `validation/` encapsula verificaciones, utilidades de ILP y exportaciones auxiliares. A esto se suman `pytest.ini`, que fija el comportamiento de prueba, y `README.md`, que cumple el papel de puerta de entrada operativa.

La disposicion sintetica del arbol es la siguiente:

```
.
├── config/       # Definiciones de entorno y dependencias
├── data/         # Salidas experimentales por host y fixtures de validacion
├── datasets/     # Datasets persistentes usados por profiling y runtime
├── docs/         # Documentacion tecnica, operativa y de soporte a tesis
├── logs/         # Trazas de ejecucion de scripts y workflows
├── reports/      # Resultados ILP y activos listos para reporte
├── scripts/      # Puntos de entrada shell/python para orquestacion
├── src/          # Implementacion central del sistema
├── tests/        # Suite pytest
├── thesis/       # Manuscrito LaTeX y PDF generado
├── validation/   # Validacion, auditoria, ILP y exportaciones
├── pytest.ini    # Configuracion de pruebas
└── README.md     # Vision general y quick start
```

## 3. Nucleo de implementacion en `src/`

El directorio `src/` constituye el centro tecnico del proyecto. Alli reside tanto el punto de entrada del profiling como los modulos que convierten medicion empirica en artefactos explotables por el modelo de optimizacion. `profiler.py` es la interfaz principal para lanzar corridas de profiling. Los documentos `profiler_en.md` y `profiler_es.md` complementan ese punto de entrada con guias de uso especificas.

Dentro de este nucleo, `core/` agrupa las primitivas compartidas del sistema: constantes, politicas de precision, captura de energia, extraccion de grafo, serializacion de artefactos, metrica derivada, agregacion estadistica y adaptacion al entorno de ejecucion. `data/` contiene el registro de datasets y la logica para resolver si una corrida debe apoyarse en entradas persistentes. `models/` encapsula la fabrica de modelos y de lotes de entrada con conocimiento de datasets y tareas instrumentadas. `runner/` implementa el pipeline efectivo de profiling capa a capa. `ilp/` materializa la transicion desde los datos robustos hasta la construccion, resolucion y exportacion del problema de particionamiento. Finalmente, `runtime/` representa la capa donde un plan de particion deja de ser solo una solucion abstracta y pasa a convertirse en simulacion determinista o en ejecucion hibrida observada.

La organizacion interna se resume asi:

```
src/
├── __init__.py
├── profiler.py                   # CLI principal para campañas de profiling
├── profiler_en.md                # Notas de uso en ingles
├── profiler_es.md                # Notas de uso en espanol
├── core/
│   ├── constants.py              # Constantes y defaults de ejecucion
│   ├── decoder_export_backend.py # Extraccion de grafo para modelos decoder-only
│   ├── energy.py                 # Monitoreo energetico CPU/GPU
│   ├── graph_extractor.py        # Exportacion de nodos y aristas del DAG
│   ├── io_artifacts.py           # Escritura de artefactos CSV/JSON
│   ├── loss_utils.py             # Objetivos de entrenamiento y auxiliares
│   ├── metrics.py                # Calculo y postproceso de metricas
│   ├── precision_policy.py       # Politica FP32/FP16/BF16 y preflight
│   ├── stats_aggregator.py       # Agregacion robusta entre replicas
│   └── system.py                 # Helpers del entorno de ejecucion
├── data/
│   └── dataset_registry.py       # Resolucion y descarga de datasets
├── ilp/
│   ├── data_loader.py            # Carga de metricas robustas y grafos
│   ├── export_solution.py        # Exportacion de soluciones ILP
│   ├── model_builder.py          # Construccion del modelo y restricciones
│   └── solve.py                  # Backend de resolucion
├── models/
│   └── factory.py                # Fabrica de modelos y entradas instrumentadas
├── runner/
│   └── training_profiler.py      # Pipeline de profiling de entrenamiento
└── runtime/
    ├── device_plan.py            # Normalizacion de planes de colocacion
    ├── hybrid_executor.py        # Ejecucion hibrida CPU/GPU observada
    ├── plan_representation.py    # Representacion canonica de plan
    └── simulator.py              # Simulacion determinista previa a ejecucion
```

Los modelos soportados en esta version son `resnet50`, `resnet152`, `vit_b16`, `bert_base`, `gpt2_small`, `distilgpt2` y `simple_mlp`. Esta seleccion no es solo una enumeracion de benchmarks; define el espacio experimental real para profiling, validacion de precision, construccion de artefactos ILP y evaluacion de portabilidad entre servidores heterogeneos.

## 4. Bloques operativos del repositorio

`config/` cumple una funcion de reproducibilidad. `environment.yml` fija el entorno Conda recomendado, mientras `requirements.txt` conserva una base de dependencias para instalaciones via pip. La responsabilidad de este bloque no es experimental sino basal: garantizar que el resto del repositorio pueda desplegarse de forma coherente.

`scripts/` concentra la orquestacion. `run_experiments.sh` gobierna campañas de profiling por grilla; `run_thesis_mode.sh` coordina el flujo doctoral extremo a extremo; `run_thesis_smoke_workflow.sh` permite una ejecucion reducida de comprobacion; `run_ilp_partition.sh` y `run_ilp_pareto_sweep.sh` envuelven la fase de optimizacion; `discover_ilp_config_dirs.sh` facilita ejecucion multi-host; `download_datasets.py` prepara la base de datos persistente requerida por modelos soportados; `generate_ilp_report_assets.sh` y `export_ilp_tables_latex.sh` conectan optimizacion con evidencia de reporte; `generate_thesis_figures.py` contribuye a la produccion grafica; `launch_grid5k.sh` atiende escenarios HPC; y `sanitize_cuda_env.sh` reduce fragilidad ambiental en lanzamientos reales.

`validation/` constituye la capa de aseguramiento tecnico. No se limita a pruebas unitarias: incluye auditoria estructural, validacion de integridad, chequeos de modelos, regresiones especificas sobre timeouts y zombies, herramientas de agregacion estadistica, lanzadores directos de ILP, barridos de sensibilidad, validacion del pipeline ILP, ejecucion hibrida real y exportacion de activos para reporte. En otras palabras, este directorio materializa el puente entre verificacion de software y verificacion metodologica.

`data/`, `datasets/`, `logs/` y `reports/` conforman el bloque de persistencia experimental. `data/` es la raiz de resultados observados y debe mantenerse con separacion por host para no mezclar evidencia de hardware disimilar. `datasets/` aloja los datos usados por las corridas instrumentadas y por el runtime hibrido cuando corresponde. `logs/` conserva la evidencia cronologica de ejecuciones de scripts. `reports/` agrega resultados ILP y productos listos para analisis o insercion documental.

`thesis/` representa el extremo de consumo academico del pipeline. Alli no solo reside el manuscrito LaTeX, sino tambien el punto en el que las salidas de profiling, ILP y reporting se transforman en tablas, figuras, capitulos y PDF final. Su presencia en la estructura no es ornamental: cierra el ciclo entre medicion computacional y expresion doctoral de resultados.

## 5. Contrato de artefactos

La estabilidad de este repositorio depende de un contrato de salidas que preserve trazabilidad entre etapas. Una corrida de profiling produce, por cada configuracion, un archivo de metricas por capa (`{model}_metrics.csv`), un archivo de metadatos (`{model}_meta.json`), dos archivos del grafo (`{model}_graph_nodes.csv` y `{model}_graph_edges.csv`) y un archivo de costos de transferencia (`{model}_transfer_edges.csv`). Sobre conjuntos repetidos de corridas, la agregacion robusta produce `{model}_metrics_stats.csv` o `metrics_stats.csv`, que actuan como insumo directo del bloque ILP.

La resolucion del particionamiento genera un subarbol `ilp_solution/` con asignaciones, aristas cortadas y resumenes estructurados. De forma complementaria, `reports/ilp_results*/` conserva colecciones orientadas a analisis, exportacion y sintesis para tesis. La regla operativa importante es que estas salidas deben permanecer host-scoped bajo `data/<hostname>/...`, porque la heterogeneidad del hardware forma parte del experimento y no puede ser borrada por una mezcla prematura de directorios.

## 6. Flujo end-to-end

El recorrido operacional del proyecto puede describirse de forma continua. Primero, `scripts/download_datasets.py` asegura que el entorno de datos exista y sea utilizable por los modelos soportados. Despues, `src/profiler.py` interpreta la configuracion experimental y delega el trabajo efectivo a `src/runner/training_profiler.py`. A continuacion, la capa de agregacion representada por `validation/aggregate_metrics_stats.py` y `src/core/stats_aggregator.py` transforma replicas en estadistica robusta. Con esa base observacional, `validation/run_ilp_partition.py` o `validation/sweep_ilp_pareto.py` construyen y resuelven planes de particion CPU/GPU, cuya persistencia final queda a cargo de `src/ilp/export_solution.py`. Cuando el flujo exige validacion mas fuerte, `validation/run_hybrid_execution.py` y el bloque `src/runtime/` permiten pasar desde la simulacion a la ejecucion hibrida efectiva. Finalmente, `validation/generate_ilp_report_assets.py` y `validation/export_ilp_tables_latex.py` convierten evidencia tecnica en activos reutilizables por `reports/` y `thesis/`.

## 7. Orden recomendado de lectura

Para una incorporacion rapida al proyecto conviene comenzar por [README.md](README.md), continuar con [GLOBAL_PROJECT_DOCUMENTATION_ES.md](GLOBAL_PROJECT_DOCUMENTATION_ES.md) como referencia tecnica canonica en espanol, contrastar con [GLOBAL_PROJECT_DOCUMENTATION.md](GLOBAL_PROJECT_DOCUMENTATION.md) cuando se necesite la version inglesa, y despues entrar en [PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md), [SERVER_LAUNCH_PROFILES.md](SERVER_LAUNCH_PROFILES.md) y [MULTI_NODE_ILP_RUNBOOK.md](MULTI_NODE_ILP_RUNBOOK.md) para la operacion real sobre servidores y escenarios multi-host. Este documento debe leerse como mapa estructural; la semantica detallada de CLI, formulas y protocolos vive en la documentacion global y en los runbooks especializados.

---

*Last Updated*: April 11, 2026
