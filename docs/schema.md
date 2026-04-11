# Estructura de la Monografía Doctoral de PRISM

## Propósito del documento

Este documento fija la arquitectura narrativa de la monografía doctoral a partir del estado final de PRISM como sistema realmente implementado. Su función no es registrar hitos históricos del desarrollo, sino ordenar cómo debe presentarse, con coherencia académica, la evidencia que el repositorio ya produce.

Debe leerse en complementariedad con [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), que describe la organización del software; con [GLOBAL_PROJECT_DOCUMENTATION_ES.md](GLOBAL_PROJECT_DOCUMENTATION_ES.md), que desarrolla la semántica técnica del pipeline; y con [PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md), que establece las condiciones de la campaña empírica real sobre múltiples servidores.

La monografía debe preservar tres principios. Primero, cada capítulo debe tener valor argumentativo propio y no depender de explicaciones históricas sobre cómo se llegó al sistema actual. Segundo, toda afirmación relevante debe estar respaldada por artefactos o validaciones observables en el repositorio. Tercero, la tesis debe presentar el proyecto como un sistema completo de optimización del entrenamiento, no como una secuencia de prototipos desconectados.

## Capítulo 1: Introducción, problema y alcance científico

Este capítulo debe delimitar con precisión el problema doctoral: la optimización del entrenamiento profundo en arquitecturas heterogéneas CPU-GPU mediante decisiones de partición guiadas por evidencia empírica y formalizadas en un modelo ILP. El punto central es mostrar que la CPU deja de ser un recurso accesorio y pasa a convertirse en un actor activo dentro de una estrategia de entrenamiento orientada a reducir presión sobre la VRAM sin sacrificar ejecutabilidad ni validez experimental.

El contenido mínimo debe incluir contexto del problema, motivación científica, hipótesis, objetivo general y objetivos específicos, delimitación de alcance y organización del manuscrito. La formulación del alcance debe dejar explícito que la contribución considera separación forward/backward, persistencia de activaciones y planificación asíncrona de transferencia como parte del sistema final, no como extensiones opcionales.

## Capítulo 2: Fundamentos teóricos y estado del arte

Este capítulo debe construir el marco conceptual necesario para interpretar la propuesta. Debe cubrir entrenamiento profundo y grafos computacionales, jerarquías de memoria CPU-GPU, transferencia y solapamiento entre cómputo y comunicación, estrategias de gestión de memoria, programación lineal entera para sistemas, y estado del arte en entrenamiento distribuido y optimización de memoria.

La sección de vacíos de investigación debe conducir a una idea nítida: las soluciones existentes suelen privilegiar throughput o distribución, pero no formulan con la misma claridad una partición heterogénea offline, guiada por profiling real y validada con ejecución física y criterios de memoria.

## Capítulo 3: Arquitectura metodológica del sistema

Este capítulo debe presentar el sistema experimental como una unidad coherente. La narrativa recomendable parte del flujo completo: captura de datos, agregación robusta, construcción del modelo ILP, simulación o ejecución híbrida, consolidación de reportes y traducción final a evidencia doctoral.

Aquí conviene explicar con detalle la estructura modular del software y el contrato de artefactos. El lector debe entender cómo se relacionan `src/`, `validation/`, `scripts/`, `reports/`, `docs/` y `thesis/`, y por qué la trazabilidad host-scoped bajo `data/<hostname>/...` es metodológicamente obligatoria cuando existe heterogeneidad hardware.

## Capítulo 4: Profiling empírico para partición heterogénea CPU-GPU

Este capítulo debe recoger la construcción metodológica del dato. Su centro no es solo la descripción del instrumentador, sino la justificación de por qué el profiling por capa produce coeficientes útiles para optimización combinatoria.

El contenido mínimo debe incluir motivación de medición, arquitectura de instrumentación, modelo energético, política de precisión, extracción estructural del grafo, modelo de transferencia consciente de arista, agregación estadística robusta y amenazas a la validez experimental. La evidencia principal procede de `src/profiler.py`, `src/runner/training_profiler.py`, `src/core/graph_extractor.py`, `src/core/precision_policy.py`, `src/core/energy.py` y `src/core/stats_aggregator.py`, además de los artefactos `*_metrics.csv`, `*_meta.json`, `*_graph_nodes.csv`, `*_graph_edges.csv`, `*_transfer_edges.csv` y `*_metrics_stats.csv`.

## Capítulo 5: Formulación ILP robusta y validación comparativa

Este capítulo debe traducir la evidencia empírica en una formulación matemática ejecutable. Debe explicar conjuntos, parámetros, variables de decisión, función objetivo, linealizaciones, restricciones de memoria, robustificación estadística, integración multi-hardware, sensibilidad y comparación con baselines.

La redacción debe insistir en que el ILP no se presenta como abstracción aislada, sino como modelo alimentado por artefactos metodológicamente admisibles. La trazabilidad entre `*_metrics_stats.csv`, `*_graph_edges.csv`, `*_transfer_edges.csv` y las salidas `ilp_assignment.csv`, `ilp_cut_edges.csv`, `ilp_solution_summary.json` debe quedar conceptualmente cerrada.

## Capítulo 6: Simulación, ejecución híbrida y validación experimental

Este capítulo debe cerrar el bucle entre planificación offline y comportamiento observado. Debe explicar cómo una solución ILP se convierte en plan, cómo ese plan puede evaluarse en simulación y cómo puede después contrastarse mediante ejecución híbrida real.

El contenido mínimo debe abarcar representación del plan, simulador, ejecutor híbrido, contraste entre predicción y observación, comparación frente a `all_cpu`, `all_gpu` y `greedy`, así como análisis de exactitud final del entrenamiento y de estabilidad operacional. La evidencia proviene de `src/runtime/`, `validation/run_hybrid_execution.py`, `validation/validate_ilp_pipeline.py`, los artefactos del subárbol `ilp_solution/` y las salidas consolidadas de reportes.

## Capítulo 7: Conclusiones, límites y trabajo futuro

Este capítulo debe sintetizar el aporte doctoral completo: una metodología para optimizar el entrenamiento en CPU-GPU con base empírica, modelo ILP robusto y validación operacional. Debe separar con honestidad qué queda demostrado, bajo qué condiciones se sostiene esa demostración y cuáles son los límites de generalización.

Las líneas de trabajo futuro pueden cubrir ampliación a multi-GPU, escalamiento a modelos mayores, integración con orquestación de clúster, refinamientos de política de memoria y extensiones del modelo de costo. La clave es que esas extensiones aparezcan como prolongaciones naturales de una contribución ya cerrada, no como pendientes necesarios para legitimar el trabajo actual.

## Articulación de la evidencia

La escritura de la monografía debe mantenerse alineada con cuatro familias de evidencia del repositorio:

- cartografía estructural y responsabilidades del software en [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- semántica técnica y contratos de artefactos en [GLOBAL_PROJECT_DOCUMENTATION_ES.md](GLOBAL_PROJECT_DOCUMENTATION_ES.md)
- despliegue y validación multiservidor en [PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md)
- desarrollo monográfico específico en [CAPITULO_TESIS_PROFILING_ES.md](CAPITULO_TESIS_PROFILING_ES.md) y [CAPITULO_TESIS_ILP_ES.md](CAPITULO_TESIS_ILP_ES.md)

Cuando esa articulación se mantenga, la tesis podrá presentar el repositorio como un sistema científicamente coherente: mide, modela, optimiza, ejecuta, compara y documenta, todo dentro de una misma cadena de trazabilidad.
