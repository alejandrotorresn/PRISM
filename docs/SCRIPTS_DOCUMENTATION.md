# Documentación de Scripts de PRISM

Esta carpeta (`scripts/`) contiene los scripts principales encargados de la orquestación, ejecución de experimentos en la tesis, resolución del modelo ILP (Programación Lineal Entera), generación de reportes y análisis. Se dividen principalmente en scripts de **orquestación y validación (Bash)** y scripts de **análisis visual y métricas (Python)**.

## 1. Scripts de Orquestación y Ejecución Principal (Bash)

### `run_thesis_mode.sh`
**Propósito:** Es el orquestador principal (Master Orchestrator) de las campañas experimentales para la tesis. Ejecuta el pipeline completo de manera secuencial:
1. Campaña de perfilado (`run_experiments.sh`)
2. Resolución del modelo ILP + barrido de Pareto por configuración.
3. Ejecución híbrida opcional.
4. Consolidación de reportes y exportación a LaTeX.
**Parámetros / Variables de Entorno principales:**
- `PROFILE`: Perfil de ejecución de la campaña (ej. `doctoral_diagnostic`, `doctoral_full`, `custom`).
- `HOST_TAG`: Nombre del host o etiqueta del servidor en el que corre (por defecto toma `hostname`).
- `DOWNLOAD_DATASETS`: Si se establece en `true`, descarga los datasets requeridos antes de iniciar.

### `run_thesis_smoke_workflow.sh`
**Propósito:** Ejecuta un flujo de trabajo reducido pero completo (*smoke test*) en hardware real para verificar que toda la cadena técnica (perfilado, ILP, reportes) funciona correctamente antes de lanzar la campaña pesada completa de días.
**Parámetros:** No requiere argumentos CLI estrictos, funciona automáticamente para validación leyendo el entorno si se ajustan variables de rutas.

### `run_experiments.sh`
**Propósito:** Script maestro de perfilado. Realiza una búsqueda en cuadrícula (Grid Search) exhaustiva sobre Modelos, Tamaños de Batch, Precisiones y Optimizadores para generar la base de datos masiva de costos y latencias que el modelo matemático ILP necesita.
**Parámetros CLI principales:**
- `--skip_cpu`: Omite la ejecución y perfilado en la CPU (útil para aislar el experimento de la GPU y evitar problemas lentos o nulos de Float16 en ciertas CPUs).
- `--num_threads N`: Sobrescribe la afinidad de hilos de la CPU (limitando a `N` hilos). Muy útil en clústeres tipo SLURM u OAR.

### `run_thesis.sh`
**Propósito:** Wrapper o envoltorio de envío directo para el gestor de recursos OAR del clúster masivo Grid5000. Reserva los nodos computacionales y luego redirige la ejecución al nodo asignado.
**Parámetros CLI / Variables de Entorno:**
- `--profile`: Parámetro por consola para especificar el perfil (ej. `--profile doctoral_full`).
- `CAMPAIGN_PROFILE`: Perfil de la campaña a enviar mediante variable de entorno.
- `RUN_HYBRID`: Bandera (`true`/`false`) para ejecutar adicionalmente las verificaciones reales de tiempo de ejecución (runtime) de la estrategia elegida.

### `launch_grid5k.sh`
**Propósito:** Es el script "lanzador" que se ejecuta ya *dentro* del nodo asignado en Grid5000 (disparado por `run_thesis.sh`). Configura el entorno virtual Conda correcto de forma agresiva, ajusta parámetros del hardware subyacente y finalmente invoca el orquestador maestro (`run_thesis_mode.sh`).
**Parámetros / Variables de Entorno principales:**
- `CONDA_ENV_NAME`: Nombre del entorno de Conda que debe forzarse a activar.
- `CAMPAIGN_PROFILE`: Relevo del perfil que debe de ejecutar la campaña completa en dicho nodo.

### `sanitize_cuda_env.sh`
**Propósito:** Utilidad que limpia las variables de entorno relacionadas con CUDA (`LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`) eliminando librerías "stub" incompletas del sistema operativo anfitrión que podrían interrumpir el driver de PyTorch y causar segment fault.
**Uso:** Se importa desde los demás scripts vía `source sanitize_cuda_env.sh` y se ejecuta su función expuesta `sanitize_cuda_runtime_env()`.

## 2. Scripts del Solver ILP (Bash)

### `run_ilp_partition.sh`
**Propósito:** Ejecuta el núcleo del solver de particionamiento (Cálculo ILP matemático) para una configuración de modelo de hardware específico (ej. ResNet50 en FP32 sobre lotes de 8).
**Parámetros / Variables de Entorno:**
- `MODEL`: Identificador del Modelo a procesar (ej. `simple_mlp`, `resnet50`).
- `CONFIG_DIR`: Ruta estricta de la estructura de datos pre-perfilados (`data/.../batch_8`).
- `GPU_MEM_BUDGET_MB` / `CPU_MEM_BUDGET_MB`: Establece presupuestos de memoria obligatorios.
- `W_TIME`, `W_ENERGY`, `W_TRANSFER`: Define las ponderaciones multiobjetivo del solver.

### `run_ilp_pareto_sweep.sh`
**Propósito:** Realiza múltiples ejecuciones (Sweep) de `run_ilp_partition.sh` iterando y ajustando automáticamente el presupuesto restringido de memoria GPU. Con esto consigue construir la Frontera de Pareto del modelo (equilibrio Latencia vs Costo Memoria).
**Parámetros / Variables de Entorno:**
- `MODEL`, `CONFIG_DIR`: Exactamente igual que en partición simple.
- `GPU_BUDGETS_MB`: Recibe una lista delimitada por comas (`400,800,1200`) o la palabra `auto` para calcular el rango inteligente.

### `discover_ilp_config_dirs.sh`
**Propósito:** Escáner que "descubre" los directorios con los perfiles completados iterando recursivamente `data/` y luego dispara el modelo ILP masivo a través de múltiples hilos en todos los experimentos.
**Parámetros / Variables de Entorno:**
- `MODEL`, `OPTIMIZER`, `PRECISION`, `BATCH`: Utilizados como regex para filtrar resultados de la cuadrícula.
- `MODE`: Define que acción se tomará con el hallazgo (`partition` o `pareto`).

## 3. Scripts de Utilidad, Auditoría y Reportes (Bash & Python)

### `audit_experimental_grid.sh`
**Propósito:** Herramienta de auditoría para verificar si en una ejecución muy masiva de múltiples días faltó algún experimento (por fallo u OOM). Cruza todos los directorios reales contra todas las combinaciones esperadas.
**Parámetros / Variables de Entorno:**
- `INPUT_ROOT`: Raíz de los resultados (`data/host/...`).
- `EXPECTED_MODELS_CSV`, `EXPECTED_OPTIMIZERS_CSV`, `EXPECTED_PRECISIONS_CSV`, `EXPECTED_BATCHES_CSV`: Matrices de lo que se espera encontrar.

### `run_statistical_significance.sh`
**Propósito:** Ejecuta el cálculo estadístico validado (Ej. Cohen's d, Prueba Z, P-Values) de la solución lograda contra las líneas base.
**Parámetros / Variables de Entorno:**
- `CONSOLIDATED_CSV`: Archivo consolidado CSV con las soluciones.
- `OUTPUT_CSV`: Ubicación donde se dejará el análisis científico.

### `export_ilp_tables_latex.sh`
**Propósito:** Una vez analizados los CSVs, este script compila código fuente formal de tablas para el documento `.tex` de la tesis, de modo que las tablas se actualicen automáticamente con cada ejecución nueva.
**Parámetros / Variables de Entorno:** 
- `BEST_CSV`, `CONSOLIDATED_CSV`, `HYBRID_CSV`, `OUT_DIR`.

### `download_datasets.py`
**Propósito:** Descarga desde orígenes y registros como HuggingFace o PyTorch de manera desatendida los datasets (ej. imágenes pre-procesadas) antes de iniciar corridas para no ahogar la red a la mitad del experimento.
**Parámetros CLI:** 
- `--models`: Modelos a pre-descargar, acepta `all`.
- `--datasets_root`: Directorio base para alojar las cachés y los binarios.

### `find_free_gpus.py`
**Propósito:** Automatización interactiva para raspar (scraping) el inventario vivo del clúster masivo Grid5000 a través de túneles SSH remotos, buscando mostrarle al usuario que nodos/servidores GPU se encuentran actualmente sin reserva y libres de asignar. 
**Parámetros CLI:** Ninguno, usa la configuración predefinida de acceso con el usuario remoto.

## 4. Analíticas, Gráficos y Figuras (Python)

Estos scripts utilizan `seaborn` / `matplotlib` en formato vectorizado sin visualización de interfaz (`Agg`). Ninguno recibe parámetros estrictos por Bash sino que leen de las configuraciones de la tesis y los directorios de `data/`.

- **`generate_methodological_plots.py`**: Genera gráficos ilustrativos metodológicos usados en el Capítulo 1 o para explicar funcionamientos técnicos (e.g. Descomposición del Kernel contra el Dispatch de PyTorch).
- **`generate_multiserver_summary.py`**: Diseñado para el final del proyecto. Recolecta CSVs de las 10 o más granjas de servidores usados, los cruza entre sí y arroja la distribución histórica global de "Speedups" (Mejoras de Velocidad). *Requiere argumentos:* `--input_dir <directorio masivo>` y `--output_dir <reportes masivos>`.
- **`generate_thesis_figures.py`**: Recrea los gráficos oficiales del pipeline individual: tiempos, huellas de memoria e ineficiencias de manera estandarizada de acuerdo al formato APA/Tesis requerido.
- **`plot_all_energy_tradeoff.py` / `plot_all_oom_vs_ilp.py`**: Utilidades que se alimentan de la Frontera de Pareto para crear la visualización del impacto de la energía vs la memoria, mostrando claramente que ejecuciones habrían fallado sin el particionamiento (OOM).
- **`organize_plots.py`**: Utilidad funcional que renombra y reacomoda todos los miles de archivos `.png` o `.pdf` generados basándose en las palabras clave del título y moviéndolos a carpetas correctas por modelo y categoría (`memory`, `energy`, etc).
- **`plot_custom_user_figures.py`**: Excluido del pipeline automático; es un *script de usuario* que permite modificar en su propio código fuente una petición ad-hoc y forzar al motor a dibujar la gráfica solo para ese experimento específico en vez de graficar millones de combinaciones.
- **`scripts/generate_chapter6_prism_vs_allgpu_figures.py`**: Generador estadístico específico para el Capítulo 6, crea diagramas comparativos entre PRISM y las estrategias All-GPU basándose en la viabilidad (feasible regime).
- **`validation/generate_advanced_thesis_plots.py`**: Genera gráficos complejos integrados de Roofline, cascada de ablación, y mapas de calor (heatmaps) de afinidad, requeridos explícitamente para documentar el modelo teórico.
- **`validation/regenerate_chapter6_global_plots_no_mlp.py`**: Versión filtrada y robusta que extrae métricas estadísticas globales (Cohen's d, valores p) excluyendo modelos de juguete como MLP para enfocar el análisis en redes profundas válidas.
- **`validation/generate_vertex_distribution_oom_plot.py` / `generate_overhead_rescue_plot.py`**: Analíticas específicas de distribución de memoria y rescate en escenarios OOM, enfocándose en qué tan útil es la solución ILP ante estrés de memoria extrema.

## 5. Validación, Aseguramiento ILP y Ejecución (Carpeta `validation/`)

Esta carpeta es el puente entre el código fuente empírico y la rigurosidad académica requerida por la tesis, aislando las simulaciones, barridos y tests del orquestador principal.

- **`validation/sweep_ilp_pareto.py`**: Genera de forma sistemática la frontera de Pareto para un modelo, iterando automáticamente por los límites del presupuesto de VRAM de la GPU.
- **`validation/run_ilp_ablation_suite.py` / `run_ilp_sensitivity.py`**: Ejecutan análisis metodológicos rigurosos: la suite de ablación evalúa el impacto de quitar componentes de la función objetivo, mientras que los tests de sensibilidad evalúan cómo responde el modelo a los cambios en las restricciones.
- **`validation/run_hybrid_execution.py`**: Simulador / ejecutor híbrido que materializa las decisiones del modelo matemático (el plan de partición ILP) en hardware real, probando que el *speedup* matemático es realizable físicamente.
- **`validation/validate_ilp_pipeline.py` / `validate_all_models.py`**: Pruebas de integración continua que evalúan si los modelos pueden generar planes viables bajo condiciones controladas de estrés, previniendo regresiones o cuelgues del solver CBC.
- **`validation/comprehensive_check.sh` / `validate_zombie_fix.py`**: Scripts de resiliencia y limpieza técnica para el clúster. Previenen fugas de memoria (zombies) y validan el pipeline end-to-end de una forma más ágil que el flujo doctoral completo.

## 6. Scripts Históricos y Deprecados (Carpeta `scripts/archive/`)

Contiene utilidades de la primera fase experimental y migraciones históricas, como `fast_aggregate.sh` o `inject_gpu_metadata.py`. Se conservan por trazabilidad del repositorio doctoral, pero no son invocados por el pipeline automático moderno `run_thesis_mode.sh`.
