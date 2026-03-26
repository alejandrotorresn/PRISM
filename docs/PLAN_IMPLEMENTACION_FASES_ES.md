# Plan de Implementacion por Fases para Cerrar la Brecha entre el Proyecto y la Tesis

## 1. Proposito del plan

Este documento funciona como pieza de gobierno metodologico del proyecto doctoral. Su papel no se limita a ordenar hitos de implementacion, sino a fijar la correspondencia entre brechas cientificas, decisiones de ingenieria, evidencia empirica y produccion monografica. En consecuencia, la hoja de ruta aqui expuesta debe leerse al mismo tiempo como plan de ejecucion, como registro de cierre por fases y como mecanismo de trazabilidad entre la propuesta original de tesis y el estado actualmente verificable del repositorio.

La idea rectora es que ninguna capacidad relevante del sistema cuenta por el solo hecho de existir en codigo. Para adquirir valor doctoral debe quedar formulada con alcance explicito, acompañada de criterios de aceptacion observables, validada mediante artefactos reproducibles y situada dentro de una narrativa cientifica coherente con los capitulos monograficos. Por ello, cada fase se define no como una lista de tareas sueltas, sino como el cierre de una brecha metodologica concreta cuya resolucion alimenta de forma directa una parte defendible de la tesis.

## 2. Principios de ejecucion

La ejecucion del plan se apoya en un conjunto de principios que ordenan las decisiones tecnicas y delimitan que tipo de avance puede considerarse cientificamente valido. El primero establece que no debe ampliarse la formulacion matematica antes de fijar con precision el alcance de la contribucion comprometida. En este proyecto, esa fijacion adopta una forma fuerte: separacion de asignaciones para forward y backward, persistencia de activaciones como decision explicita del ILP y compatibilidad operacional con transferencia asincrona y prefetching. Cualquier desarrollo posterior debe permanecer subordinado a ese contrato.

El segundo principio exige cerrar antes el bucle metodologico que el refinamiento ornamental del modelo. Una tesis de sistemas no gana solidez por acumular terminos en la funcion objetivo si todavia no puede contrastar sus decisiones con simulacion reproducible o con ejecucion observada. Por esa razon, el plan privilegia la continuidad entre profiling, ILP, simulacion y runtime por encima de extensiones elegantes pero prematuras.

El tercer principio afirma que toda nueva capacidad debe quedar sujeta a criterios de aceptacion observables. Una implementacion que no produce artefactos auditables, pruebas reproducibles o trazas interpretables puede ser util como exploracion interna, pero no como evidencia doctoral. El cuarto principio deriva de ello: documentacion, codigo y monografia deben evolucionar de manera acoplada, porque el mayor riesgo del proyecto no es solo tecnico, sino epistemico, a saber, sostener en la escritura afirmaciones mas fuertes que las demostradas por el sistema.

Finalmente, la secuencia de fases se diseña para preservar valor defendible incluso ante una detencion anticipada del proyecto. Cada fase debe dejar un cierre autonomo, con utilidad propia para la tesis, de modo que el progreso acumulado conserve legitimidad cientifica aun cuando el programa completo no hubiera alcanzado todavia su ultimo hito.

## 3. Vista general de fases

| Fase | Objetivo principal | Resultado de tesis habilitado |
| --- | --- | --- |
| Fase 0 | Congelar alcance y semantica del modelo | Coherencia entre tesis y software |
| Fase 1 | Cerrar validacion comparativa minima | ILP evaluable contra baselines y ablaciones |
| Fase 2 | Construir simulador hibrido | Prediccion reproducible de planes ILP |
| Fase 3 | Implementar ejecutor hibrido real | Ejecucion fisica CPU-GPU guiada por ILP |
| Fase 4 | Extender el modelo con memoria de activaciones | Tesis alineada con rematerializacion y persistencia |
| Fase 5 | Validacion doctoral completa | Resultados finales, analisis y capitulos cerrados |

### 3.1 Veredicto por fases (actualizado a 2026-03-26)

Al estado de corte actual, la Fase 1 debe considerarse funcional y metodologicamente recuperada tras la correccion de comparabilidad entre ILP y baseline `greedy` en `validation/sweep_ilp_pareto.py`, donde ambos pasan a evaluarse en el mismo espacio objetivo dual. Las Fases 2 y 3 se encuentran cerradas con validacion suficiente, mientras que la Fase 4 puede darse por plenamente operativa al integrar ya decision dual forward/backward, persistencia de activaciones, transferencia asincrona y prefetching explicito en runtime, todo ello con respaldo de pruebas y de ejecucion fisica controlada.

Sobre ese estado de cierre se añadió, a fecha 2026-03-26, un endurecimiento transversal del contrato metodologico entre profiling e ILP. Dicho endurecimiento excluye como vias normales de evidencia tanto las entradas sinteticas silenciosas como la topologia lineal de fallback, impone calibracion de transferencia medida y vuelve obligatoria la verificacion explicita de calidad muestral y de procedencia estructural antes de construir la instancia ILP. La consecuencia no es menor: la cadena completa que va desde la medicion empirica hasta la optimizacion deja de ser solo operativamente util y pasa a satisfacer un criterio mas estricto de admisibilidad doctoral de artefactos metodologicamente aceptables.

## 4. Fase 0: congelacion de alcance y contrato cientifico

### 4.1 Objetivo

Resolver las ambiguedades de diseno que hoy impiden avanzar sin riesgo de rehacer trabajo y formalizar contractualmente las decisiones ya adoptadas: asignacion independiente forward/backward, persistencia de activaciones como variable binaria ILP y scheduling asincrono en runtime.

### 4.2 Trabajo tecnico

1. Formalizar la semantica final del ILP con asignacion independiente por fase (forward y backward).
2. Formalizar persistencia de activaciones, rematerializacion y checkpointing como variables explicitas del modelo.
3. Formalizar transferencia asincrona CPU-GPU en el ejecutor hibrido e incorporar prefetching explicito como politica operativa auditable en ejecucion hibrida.
4. Actualizar la documentacion metodologica para que no existan afirmaciones incompatibles con estas decisiones.

### 4.3 Archivos afectados

- `docs/schema.md`
- `docs/PLAN_IMPLEMENTACION_FASES_ES.md`
- `docs/CAPITULO_TESIS_PROFILING_ES.md`
- `docs/CAPITULO_TESIS_ILP_ES.md`
- `docs/GLOBAL_PROJECT_DOCUMENTATION_ES.md`

### 4.4 Criterios de aceptacion

- Existe una definicion explicita del alcance del modelo y del runtime.
- La monografia, la documentacion tecnica y el codigo dejan de describir capacidades contradictorias.
- Las fases posteriores ya no dependen de supuestos tacitos.

### 4.5 Riesgo mitigado

Evita sobredimensionar la contribucion doctoral o implementar un runtime incompatible con el modelo matematico definitivo.

### 4.6 Acta de cierre de Fase 0 (2026-03-19)

La Fase 0 queda formalmente cerrada con las siguientes decisiones vinculantes para todo el proyecto:

1. La asignacion de dispositivo se define de forma independiente para forward y backward.
2. La persistencia de activaciones se modela como decision binaria explicita del ILP.
3. El ejecutor hibrido implementa transferencia asincrona CPU-GPU mediante `torch.cuda.Stream` y prefetching explicito de activaciones con look-ahead entre capas en ejecucion hibrida.

Estas decisiones dejan sin validez cualquier redaccion alternativa basada en bifurcacion de alcance. Desde esta fecha, las Fases 1-5 se ejecutan sobre la ruta unica completa y toda evidencia experimental debe reportarse bajo ese contrato metodologico.

## 5. Fase 1: validacion comparativa minima y robustecimiento experimental

### 5.1 Objetivo

Cerrar primero la parte de evaluacion que ya puede construirse sobre el ILP actual sin esperar al ejecutor hibrido completo. Esta fase debe convertir el planificador existente en un sistema comparativamente serio.

### 5.2 Trabajo tecnico

1. Implementar baseline `greedy` en `validation/sweep_ilp_pareto.py`.
2. Crear un harness de ablaciones con, al menos, cuatro variantes: sin topologia, sin costos de transferencia por arista, sin robustificacion estadistica y modelo completo.
3. Ampliar `validation/generate_ilp_report_assets.py` para consolidar resultados de ablacion y baseline.
4. Anadir chequeos de suficiencia muestral y banderas de calidad de coeficientes en `src/core/stats_aggregator.py` o en una capa de validacion dedicada.
5. Incorporar un reporte estructurado de dispersion y sensibilidad para `k_sigma`, `w_transfer` y `gpu_mem_budget_mb`.

### 5.3 Modulos probables

- `validation/sweep_ilp_pareto.py`
- `validation/generate_ilp_report_assets.py`
- `validation/export_ilp_tables_latex.py`
- `validation/aggregate_metrics_stats.py`
- nuevo `validation/run_ilp_ablation_suite.py`
- nuevo `validation/run_ilp_sensitivity.py`

### 5.4 Entregables

- CSV consolidado con `all_cpu`, `all_gpu`, `greedy` e `ilp`.
- CSV y figuras de ablacion.
- Tablas LaTeX comparativas listas para la tesis.
- Protocolo de sensibilidad reproducible (OAT sobre `k_sigma` y `w_transfer`).
- Columnas de calidad muestral (`quality_flag`, `max_cv_key_metrics`, CV por metrica) en `metrics_stats.csv`.

### 5.5 Criterios de aceptacion

- El baseline `greedy` se ejecuta desde CLI y aparece en los reportes finales.
- Las ablaciones pueden lanzarse sobre cualquier configuracion valida del pipeline.
- Los activos generados permiten sostener el argumento de ganancia incremental del ILP frente a alternativas mas simples.
- `aggregate_metrics_stats` emite advertencia auditada cuando alguna capa tiene `n_runs < MIN_RECOMMENDED_RUNS` o `CV > 0.30`.
- El script de sensibilidad produce un CSV estructurado con `delta_pct` vs. baseline para cada valor de parametro barrido.

### 5.6 Acta de cierre de Fase 1 (2026-03-19)

La Fase 1 queda formalmente cerrada. Todos los criterios de aceptacion han sido verificados sobre el conjunto de datos `data/zephyr/results_smoke/simple_mlp/SGD/fp32/batch_8` (3 replicas, 5 capas). Los resultados clave son los siguientes.

El ILP mejora el objetivo en un 93.33 % respecto al baseline all-CPU y en un 11.06 % respecto al baseline greedy al presupuesto de 64 MB de VRAM. El analisis de ablacion confirma que la robustificacion estadistica y la topologia de grafo contribuyen de forma diferenciada: suprimir la robustificacion reduce el objetivo en 0.50 unidades (escenario optimista irreal), mientras que suprimir la topologia o los costos de transferencia reduce 0.44 unidades.

El analisis de sensibilidad OAT muestra que `k_sigma` introduce una variacion lineal de +-13.8 % por unidad 0.5, lo que acredita su utilizacion como parametro de conservadurismo controlado. La sensibilidad a `w_transfer` es monotona pero no lineal: al valor 5.0 el objetivo sube 48.8 % mientras que al valor 0.0 el optimizador reordena capas (2 GPU, 3 cortes frente a 3 GPU, 1 corte en el baseline).

Las banderas de calidad muestral detectan alta dispersion (CV > 0.30) en todas las capas del conjunto smoke, lo que refleja la variabilidad esperada en un dataset de solo 3 ejecuciones y constituye el argumento para exigir al menos 5 replicas en la validacion doctoral final.

### 5.7 Capitulo de tesis habilitado

Esta fase cierra buena parte del material analitico de los capitulos de formulacion y resultados comparativos, incluso antes de tener el ejecutor hibrido final.

## 6. Fase 2: simulador de ejecucion hibrida

### 6.1 Objetivo

Crear una capa intermedia entre solucion ILP y ejecucion real. El simulador debe permitir evaluar la consistencia topologica de un plan, estimar tiempos agregados, memoria y costos de corte, y servir como banco de pruebas antes de desplegar el runtime fisico.

### 6.2 Trabajo tecnico

1. Definir una representacion formal del plan de ejecucion a partir de `ilp_assignment.csv` y `ilp_cut_edges.csv`.
2. Crear un modulo nuevo para simulacion dirigido por DAG y costos observados.
3. Estimar al menos: latencia total, memoria efectiva por dispositivo, numero de cortes, costo total de transferencia y descomposicion del objetivo.
4. Permitir modos de simulacion nominal y robusto.
5. Exportar un artefacto de simulacion auditable en JSON y CSV.

### 6.3 Modulos probables

- nuevo `src/runtime/plan_representation.py`
- nuevo `src/runtime/simulator.py`
- nuevo `validation/validate_ilp_pipeline.py`
- posible ajuste en `src/ilp/export_solution.py`

### 6.4 Entregables

- Simulador invocable por CLI.
- Reporte por plan con descomposicion del costo.
- Validacion de integridad topologica y de restricciones del plan.

### 6.5 Criterios de aceptacion

- Dado un conjunto de artefactos validos, el simulador produce un resumen determinista y reproducible.
- El simulador detecta incoherencias de plan, aristas sin costo o violaciones de presupuesto.
- Los reportes pueden citarse en la tesis como evidencia pre-ejecucion del comportamiento esperado.

### 6.6 Capitulo de tesis habilitado

Esta fase cierra el nucleo del capitulo de resolucion y simulacion, y reduce el salto conceptual entre formulacion matematica y experimento fisico.

## 6.7 Acta de cierre de Fase 2 (2026-03-19)

La Fase 2 queda formalmente cerrada tras verificar, sobre el pipeline activo del proyecto, que la solucion ILP puede transformarse en un plan ejecutable auditable y evaluarse de forma determinista antes de la ejecucion fisica. El cierre se sustenta en la incorporacion de una representacion formal del plan en `src/runtime/plan_representation.py`, un simulador de costos y restricciones en `src/runtime/simulator.py`, y una interfaz de validacion reproducible en `validation/validate_ilp_pipeline.py`, complementadas con su integracion opcional post-solve en `validation/run_ilp_partition.py`.

La verificacion funcional confirma el cumplimiento integral de los criterios de aceptacion. Para artefactos validos, el simulador produce resumen JSON y desglose CSV con estabilidad bit a bit en ejecuciones repetidas, lo que acredita reproducibilidad operativa. Para escenarios de inconsistencia, el sistema detecta y reporta de manera explicita incoherencias topologicas y de corte, ausencia de costos de transferencia cuando aplica modo estricto, y violaciones de presupuesto de memoria por dispositivo, emitiendo estado invalido y codigo de salida de error. En consecuencia, los reportes de simulacion quedan habilitados como evidencia pre-ejecucion citable en la tesis.

Durante el cierre se introdujo ademas un refuerzo metodologico en la carga de datos ILP (`src/ilp/data_loader.py`) para bloquear datasets degenerados con tiempos no positivos que inducen conclusiones espurias. Esta salvaguarda no altera el alcance de la fase, pero fortalece la validez interna de toda evidencia derivada de la simulacion y preserva la coherencia cientifica del programa doctoral.

## 7. Fase 3: ejecutor hibrido real guiado por ILP

### 7.1 Objetivo

Implementar la primera version funcional del entrenamiento hibrido real. Esta es la brecha mas importante del proyecto actual.

### 7.2 Alcance recomendado del MVP

El primer objetivo no debe ser soportar todos los modelos complejos del repositorio, sino demostrar un runtime correcto y medible sobre un subconjunto controlado, idealmente `simple_mlp` y `resnet50`. La ejecucion inicial debe respetar desde el diseno la semantica de asignacion independiente para forward y backward definida en Fase 0.

En la evolucion posterior del runtime se admite un backend aislado adicional para modelos causales decoder-only exportables, como la familia GPT-2. Este backend debe permanecer desacoplado de la ruta general basada en `torch.fx`, activarse solo por deteccion de capacidad y conservar la consistencia entre capas perfiladas, artefactos de grafo e instancia ILP. La regla de gobierno es conservadora: si el backend especializado no puede demostrar esa consistencia, el sistema debe registrar la incompatibilidad y no degradar silenciosamente a una topologia incorrecta.

### 7.3 Trabajo tecnico

1. Construir un cargador del plan ILP que convierta la asignacion en una politica de colocacion.
2. Implementar un runtime de entrenamiento que mueva modulos y tensores segun el plan.
3. Gestionar transferencias entre CPU y GPU con trazabilidad y medicion real.
4. Registrar tiempos, memoria y energia de la ejecucion hibrida completa.
5. Exportar resultados de ejecucion observada para compararlos con el simulador.

### 7.4 Modulos probables

- nuevo `src/runtime/hybrid_executor.py`
- nuevo `src/runtime/device_plan.py`
- nuevo `validation/run_hybrid_execution.py`
- nuevo `tests/test_hybrid_executor.py`

### 7.5 Entregables

- Ejecucion real de un plan ILP sobre hardware CPU-GPU.
- Artefactos observados del ejecutor hibrido.
- Scripts de prueba end-to-end para modelos piloto.

### 7.6 Criterios de aceptacion

- El runtime ejecuta al menos un modelo piloto sin violar integridad funcional.
- La ejecucion deja trazas suficientes para comparar prediccion y observacion.
- Se pueden medir al menos tiempo de paso, memoria y energia observada del plan hibrido.

### 7.7 Riesgo mitigado

Esta fase transforma la tesis de una propuesta de optimizacion offline a una contribucion de sistemas con evidencia operacional real.

## 7.8 Acta de cierre de Fase 3 (2026-03-19)

La Fase 3 queda formalmente cerrada tras validar, en hardware CPU-GPU del entorno de desarrollo, la ejecucion hibrida guiada por plan ILP sobre el modelo piloto `simple_mlp`. El cierre se sustenta en la incorporacion de `src/runtime/device_plan.py` y `src/runtime/hybrid_executor.py`, junto con la interfaz reproducible `validation/run_hybrid_execution.py` y su cobertura de pruebas en `tests/test_hybrid_executor.py`.

La validacion confirma cumplimiento de los criterios de aceptacion de la fase. El runtime ejecuta el plan sin violar integridad funcional, emite trazas por paso (forward, backward, optimizacion y transferencias), y reporta memoria maxima en GPU y eventos de cruce CPU-GPU. Adicionalmente, se cierra de forma explicita el requisito de energia observada: la ejecucion exporta potencia promedio y energia total medida en artefactos JSON/CSV, con fuente de medicion declarada (`nvml` para GPU o `rapl` para CPU cuando aplica).

Para preservar rigor metodologico, el CLI aborta por defecto si el plan requiere GPU y CUDA no esta disponible, evitando evidencia degradada no declarada. El fallback a CPU queda permitido solo bajo bandera explicita para diagnostico (`--allow_cpu_fallback`). En consecuencia, la Fase 3 queda habilitada como evidencia de ejecucion fisica auditable y comparable contra predicciones del simulador.

## 8. Fase 4: extensiones del modelo con persistencia de activaciones

### 8.1 Objetivo

Incorporar al modelo y al runtime las capacidades descritas en la tesis y ya fijadas en Fase 0: persistencia de activaciones, rematerializacion, checkpointing y transferencia asincrona CPU-GPU con trazabilidad operacional.

### 8.2 Dependencia critica

Esta fase solo debe arrancar despues de que el simulador y el runtime MVP existan. De otro modo, se corre el riesgo de anadir complejidad matematica sin capacidad de validacion.

### 8.3 Trabajo tecnico

1. Extender la representacion de estado por nodo para distinguir activaciones retenidas, recomputadas o checkpointadas.
2. Definir las nuevas variables del modelo y sus restricciones de memoria y tiempo.
3. Ajustar el simulador para incorporar los nuevos modos de ejecucion.
4. Ajustar el runtime, aunque inicialmente sea solo para un subconjunto de estrategias.
5. Implementar de forma operativa la asignacion independiente de forward y backward en el modelo, simulador y runtime.

### 8.4 Modulos probables

- `src/ilp/model_builder.py`
- `src/ilp/solve.py`
- nuevo `src/ilp/advanced_terms.py`
- `src/runtime/simulator.py`
- `src/runtime/hybrid_executor.py`

### 8.5 Entregables

- Nueva formulacion documentada del ILP.
- Artefactos de solucion ampliados.
- Comparacion entre modelo base y modelo extendido.

### 8.6 Criterios de aceptacion

- Las nuevas variables tienen semantica operacional clara y validacion estructural.
- El simulador y el runtime pueden consumir al menos una parte de la extension.
- La tesis puede afirmar con evidencia que la persistencia de activaciones no es solo una idea, sino una decision modelada y contrastada.

## 8.7 Avance operativo de Fase 4 (2026-03-20)

Esta seccion documenta un corte intermedio de la misma fecha (previo al acta formal de cierre incluida en la seccion 8.8). En ese corte, la Fase 4 ya no debe describirse como trabajo puramente prospectivo: se incorporo un tramo operativo verificable que extiende de forma coordinada el ILP, el simulador y el runtime.

En la capa de modelado se introdujo `src/ilp/advanced_terms.py`, junto con extensiones en `src/ilp/model_builder.py`, `src/ilp/data_loader.py` y `src/ilp/solve.py`, para representar estrategias de persistencia de activaciones bajo tres semanticas diferenciadas: retencion, recomputacion y checkpointing. Sobre esa base se habilito una solucion extendida de Fase 4 con heuristica inicial para decidir recomputacion bajo presion de memoria. Adicionalmente, el ILP ya soporta asignacion independiente de dispositivo para forward y backward, con exportacion de planes duales y costos de transferencia intra-fase y entre fases.

En la capa de simulacion, `src/runtime/simulator.py` ya consume tanto estrategias de activacion como asignaciones independientes de forward y backward, estimando el costo temporal y de transferencia de cortes en forward, cortes en backward y cruces entre fases. En la capa de ejecucion real, `src/runtime/hybrid_executor.py` soporta tres mecanismos operativos auditables: recomputacion por capa mediante `torch.utils.checkpoint`, checkpointing en memoria CPU de tensores guardados para backward mediante hooks de autograd y, cuando el plan dual realmente asigna dispositivos distintos y el entorno lo permite, una ruta de backward materializada en dispositivo distinto a forward mediante recomputacion autograd controlada por capa. Esta ampliacion conserva la trazabilidad previa de tiempo, memoria, energia y transferencias, y agrega contadores explicitos por paso para recomputacion, checkpointing y relocation de backward.

La verificacion automatizada ya acredita consistencia funcional del avance. La suite conjunta de runtime y Fase 4 ejecuta satisfactoriamente pruebas unitarias e integradas sobre estrategias de activacion sinteticas y sobre la compatibilidad con el ejecutor hibrido existente. Ademas, un smoke end-to-end sobre `data/zephyr/results_smoke/simple_mlp/SGD/fp32/batch_8` confirma que el solver dual (`pulp_cbc_dual`) exporta un plan consumible por `validation/validate_ilp_pipeline.py` y por `validation/run_hybrid_execution.py`, con objetivo reproducido de 2.031104 en simulacion y ejecucion fisica observada valida sobre hardware CPU-GPU. Debe advertirse, no obstante, que ese plan smoke concreto no activa relocation efectiva de backward porque su optimo asigna las mismas capas al mismo dispositivo en ambas fases; la evidencia de materializacion dual del runtime queda por ahora acreditada por pruebas controladas y por la capacidad implementada, no por ese caso smoke particular.

No obstante, este corte intermedio no constituye acta de cierre por si mismo. Los pendientes consignados aqui quedaron resueltos posteriormente dentro de la misma fecha y su formalizacion final se registra en la seccion 8.8.

## 8.8 Acta de cierre de Fase 4 (2026-03-20)

La Fase 4 queda formalmente cerrada al haberse satisfecho sus criterios de aceptacion con evidencia operacional y documental suficiente. El cierre se sustenta en tres bloques ya convergentes. Primero, el modelo extendido incorpora semantica explicita de persistencia de activaciones mediante retencion, recomputacion y checkpointing, junto con asignacion independiente de dispositivo para forward y backward, todo ello implementado en `src/ilp/advanced_terms.py`, `src/ilp/data_loader.py`, `src/ilp/model_builder.py` y `src/ilp/solve.py`. Segundo, el simulador y el runtime consumen esa ampliacion sin degradar el contrato previo de auditabilidad, trazando memoria, tiempo, energia, transferencias y ahora tambien recomputacion, checkpointing, prefetching y relocation de backward. Tercero, la cobertura automatizada ya valida consistencia funcional del modelo extendido mediante pruebas unitarias e integradas en `tests/test_phase4_activation.py`, `tests/test_phase4_synthetic.py`, `tests/test_phase4_comparison.py`, `tests/test_runtime_simulator.py` y `tests/test_hybrid_executor.py`.

La evidencia empirica de cierre no descansa unicamente en pruebas sinteticas. Sobre el dataset real `data/zephyr/results_smoke/simple_mlp/SGD/fp32/batch_8` se construyo y ejecuto un caso dual controlado versionado en `reports/ilp_results_phase4_controlled/simple_mlp_dual_runtime_evidence`. Dicho artefacto fija una colocacion no trivial donde `net.0` se ejecuta en GPU durante forward y en CPU durante backward, preservando el resto del modelo en CPU. El simulador valida topologia y costos del plan sin violaciones, con objetivo robusto de 19.884783, tiempo total estimado de 19.547401 ms, energia estimada de 0.145244 J y dos cortes contabilizados al considerar el cruce forward y el cruce entre fases. La ejecucion fisica mediante `validation/run_hybrid_execution.py` completa correctamente el paso de entrenamiento sobre hardware CPU-GPU y registra en sus artefactos observados `backward_relocation_layers = ["net.0"]` y `backward_relocation_count = 1`, lo que acredita materializacion efectiva de la asignacion dual en runtime sobre un caso real y no meramente simulado. Adicionalmente, en corrida con `--enable_async_transfer --enable_prefetch` sobre ese mismo plan dual se verifica activacion operacional de prefetch (`total_prefetch_events = 1`, `total_prefetch_mb = 0.015625`, `prefetch_layers = ["net.1"]`) en `/tmp/phase4_full_hybrid_prefetch_dual/hybrid_execution_summary.json`.

En consecuencia, la tesis ya puede afirmar con evidencia que la persistencia de activaciones y la separacion forward/backward no son una extension prospectiva, sino una capacidad implementada, simulable, ejecutable y auditable en el pipeline actual. Las comparaciones experimentales mas amplias frente a configuraciones base y la explotacion masiva de la matriz final dejan de ser condicion de cierre de Fase 4 y pasan a integrarse en la Fase 5 como trabajo de validacion doctoral integral y consolidacion monografica.

## 9. Fase 5: validacion doctoral integral y cierre de monografia

### 9.1 Objetivo

Convertir el sistema completo en evidencia doctoral final. Esta fase no introduce necesariamente nuevas capacidades fundamentales; su trabajo central es producir campanas, tablas, figuras, comparaciones y narrativas con calidad de defensa.

### 9.2 Trabajo tecnico y experimental

1. Ejecutar la matriz experimental final por modelos, precisiones, batches y optimizadores justificados.
1.1. Preparar al inicio de cada campana los datasets persistentes en `datasets/`, de modo que profiling y runtime consuman el mismo origen de datos y quede trazabilidad explicita del corpus efectivamente usado.
2. Comparar `all_cpu`, `all_gpu`, `greedy`, `ilp_base` y, si existe, `ilp_extendido`.
3. Comparar prediccion del simulador frente a observacion fisica del ejecutor hibrido.
4. Medir accuracy, loss final o metrica de calidad pertinente en los escenarios hibridos.
5. Consolidar figuras, tablas LaTeX y resumenes para los capitulos de resultados y conclusiones.

En el estado actual del proyecto, esta fase ya no parte de entradas sinteticas como supuesto operativo por defecto. La infraestructura incorpora una etapa previa de preparacion de datasets y un contrato de procedencia reproducible. El directorio `datasets/` actua como repositorio persistente comun para toda la campana, con mapeo metodologico fijado por familia de modelo: `simple_mlp` consume MNIST; `resnet50`, `resnet152` y `vit_b16` consumen Imagenette 160 remapeado a indices de clase de ImageNet-1K; `bert_base` consume AG News mediante cabeza explicita de clasificacion secuencial; y `gpt2_small` consume un corpus causal publico y estable (Tiny Shakespeare) mediante cabeza explicita de lenguaje causal. En consecuencia, la validez interna de la Fase 5 mejora en dos sentidos simultaneos: se elimina la divergencia entre origen de datos de profiling y origen de datos de runtime, y se registra en artefactos el nombre del dataset, split, ruta y fuente de targets.

Esta transicion no clausura por si sola la validacion doctoral final por tarea, porque todavia subsiste la necesidad de cerrar protocolos estadisticos y comparativos mas amplios por arquitectura. Sin embargo, si cierra una brecha metodologica sustantiva: el sistema deja de depender de batches sinteticos para la ejecucion ordinaria de la campana y pasa a operar sobre corpus versionables y auditables, con `accuracy` real para `bert_base` y `token_accuracy`/loss causal real para `gpt2_small`.

### 9.3 Entregables

- Dataset consolidado final.
- Graficas y tablas de la monografia.
- Resumen de amenazas a la validez y limites del sistema.
- Capitulo de resultados cerrado.

Como evidencia intermedia ya disponible tras esta consolidacion, el modo tesis en smoke con datasets reales produce un protocolo hibrido observacional que explicita `dataset_name`, `dataset_split`, `dataset_path`, `input_source` y `target_source`, ademas de `final_loss`, `quality_metric_name` y `final_quality_metric`. Dichos campos quedan consolidados tanto en los artefactos por configuracion como en el resumen agregado y en la exportacion LaTeX del mejor ejecutor hibrido por modelo.

### 9.4 Criterios de aceptacion

- Cada afirmacion fuerte de la tesis tiene una figura, tabla o artefacto que la respalda.
- Existe comparacion entre prediccion y ejecucion observada.
- Existe evidencia sobre memoria, tiempo, energia y exactitud final.

Adicionalmente, para que la aceptacion de Fase 5 sea metodologicamente consistente con la nueva infraestructura, cada artefacto clave debe preservar la procedencia del dato experimental: dataset usado, split, politica de targets y raiz de almacenamiento. Sin esa capa de procedencia, la comparacion entre profiling, simulacion y runtime volveria a quedar apoyada en supuestos no auditables.

## 10. Dependencias entre fases

La relacion entre fases no es meramente cronologica, sino logica. La Fase 0 resulta obligatoria antes de introducir nuevas variables o nuevas promesas de alcance en el ILP, porque fija el contrato cientifico que da sentido al resto del programa. La Fase 1 conserva cierta independencia relativa y puede avanzar en paralelo parcial con la Fase 2, pero solo en la medida en que su fortalecimiento experimental no presuponga todavia la existencia del simulador ni del runtime completo.

La Fase 2 debe preceder a la Fase 3 porque define la representacion del plan ejecutable y reduce el salto entre optimizacion y despliegue fisico. A su vez, la Fase 3 debe quedar cerrada antes de reclamar una validacion doctoral completa, ya que sin runtime real no existe contraste fisico suficiente entre prediccion y observacion. La Fase 4, por su parte, solo adquiere sentido una vez probado el circuito minimo de evidencia, pues de otro modo la tesis correria el riesgo de acumular complejidad semantica sin capacidad de materializacion. Finalmente, la Fase 5 depende del cierre sustantivo de las anteriores porque su tarea no es descubrir el sistema, sino consolidarlo como evidencia experimental, comparativa y monografica.

## 11. Orden recomendado si se busca la via mas pragmatica

Si el objetivo es alcanzar con la mayor economia de esfuerzo una tesis defendible, la secuencia mas eficaz sigue siendo la progresion lineal aqui consolidada. Primero debe cerrarse Fase 0, porque sin un contrato de alcance estable cualquier avance posterior queda expuesto a rehacerse. Despues conviene ejecutar Fase 1 para reforzar comparabilidad, sensibilidad y calidad de coeficientes del ILP ya existente. Sobre esa base, la construccion de la Fase 2 reduce incertidumbre antes de invertir en la complejidad del runtime. La Fase 3 convierte entonces la optimizacion en una evidencia de sistemas observable sobre modelos piloto. La Fase 4 completa las capacidades extendidas comprometidas por la tesis, y la Fase 5 queda reservada para la campana final, la consolidacion estadistica y la escritura monografica definitiva.

## 12. Decision estrategica final

La decision estrategica final queda fijada sin bifurcaciones: sostener la tesis en su formulacion fuerte y completar de forma secuencial el cierre de las Fases 0 a 5. Esta definicion descarta lecturas reduccionistas del proyecto como si se tratara solo de un profiler avanzado o de un ILP offline. La contribucion cientifica comprometida y ya parcialmente demostrada es la de un sistema de particion heterogenea que mide, robustifica, optimiza, simula y ejecuta planes CPU-GPU bajo restricciones fisicas, con extensiones explicitas de memoria y scheduling asincrono.

## 13. Anexo de trazabilidad historica proyecto-tesis

Este anexo absorbe el valor historico del antiguo documento de contraste entre proyecto y tesis. Se conserva aqui no como pieza paralela, sino como memoria auditada del punto de partida, de las brechas detectadas y de su cierre posterior.

### 13.1 Alcance del contraste original

El contraste inicial se realizo contra el documento base `docs/A New Parallelization Approach in Deep Learning Using CPU.docx` y contra el estado real del repositorio a 2026-03-19. La pregunta rectora era triple: que partes de la propuesta doctoral ya existian en el software, que partes estaban solo parcialmente cubiertas y que componentes faltaban todavia para sostener cumplimiento integral de la tesis.

### 13.2 Exigencias nucleares de la propuesta doctoral

La propuesta base no describia solo un profiler ni solo un ILP offline. Exigia una cadena metodologica completa compuesta por caracterizacion empirica por capa, formulacion ILP sobre grafo, asignacion CPU-GPU, persistencia de activaciones, rematerializacion o checkpointing, streaming/prefetching, simulacion de planes, validacion fisica de runtime, comparacion con baselines y verificacion de no degradacion de calidad final.

El punto doctrinal mas exigente del alcance inicial era la separacion de decisiones entre forward y backward. Esa condicion fue la que convirtio el problema de cierre en una cuestion no solo de optimizacion offline, sino de coherencia entre formulacion matematica, simulacion, runtime y narrativa monografica.

### 13.3 Mapa estructural del repositorio en el punto de partida

Al cierre de Fase 0, el repositorio ya mostraba cinco fortalezas estructurales verificables.

Primero, el bloque de profiling en `src/profiler.py` y `src/runner/training_profiler.py` ya producia metricas por capa, metadatos globales, artefactos de grafo y artefactos de transferencia. Segundo, la extraccion estructural del grafo y la calibracion PCIe ya estaban presentes en `src/core/graph_extractor.py` y `src/runner/training_profiler.py`. Tercero, la agregacion robusta de replicas en `src/core/stats_aggregator.py` ya permitia alimentar el ILP con medias, dispersion y cuantiles. Cuarto, el flujo de datos ILP en `src/ilp/data_loader.py`, `src/ilp/model_builder.py` y `src/ilp/solve.py` ya resolvia un problema coherente de particion offline. Quinto, el repositorio ya disponia de automatizacion reproducible para particion, Pareto y reporting mediante `validation/` y `scripts/`.

En otras palabras, el proyecto ya sabia medir y decidir. La brecha residia en ejecutar y validar plenamente la decision bajo el regimen hibrido prometido por la tesis.

### 13.4 Cobertura inicial y cobertura parcial

En el diagnostico original, las capacidades ya implementadas incluian caracterizacion empirica por capa de tiempo, energia, memoria y FLOPs; politica de precision; exportacion de artefactos; extraccion de grafo; costos de transferencia; agregacion robusta; ILP de asignacion CPU-GPU; restricciones de memoria; baselines `all_cpu` y `all_gpu`; exportacion de resultados y tablas; flujo smoke end-to-end; y fusion multi-hardware para ILP robusto.

Sin embargo, varias dimensiones seguian siendo parciales. Existia una asignacion offline por capa, pero aun no una materializacion de esa asignacion en runtime. Existia modelado de comunicacion, pero no simulacion ni ejecucion hibrida completa. Existia reporting, pero no todavia una validacion doctoral integral de prediccion frente a observacion fisica y de preservacion de calidad final.

### 13.5 Brechas historicas y estado actual de cierre

La siguiente tabla resume la trazabilidad completa entre brecha original y estado actual.

| Brecha historica | Diagnostico original | Estado actual | Evidencia principal |
| --- | --- | --- | --- |
| Ejecucion hibrida real guiada por ILP | Ausente | Cerrada | `src/runtime/hybrid_executor.py`, `validation/run_hybrid_execution.py` |
| Decisiones separadas forward/backward | Ausentes | Cerrada | `src/ilp/solve.py`, `src/runtime/device_plan.py`, `src/runtime/plan_representation.py` |
| Persistencia de activaciones, rematerializacion y checkpointing como decision | Ausentes en ILP | Cerrada | `src/ilp/advanced_terms.py`, `src/runtime/hybrid_executor.py` |
| Streaming y prefetching operativos | Parciales o no integrados | Cerrada operacionalmente | `src/runtime/hybrid_executor.py`, pruebas de runtime y evidencia controlada de Fase 4 |
| Simulador de planes hibridos | Ausente | Cerrada | `src/runtime/simulator.py`, `validation/validate_ilp_pipeline.py` |
| Validacion fisica de planes ILP | Ausente | Cerrada | `validation/validate_ilp_pipeline.py`, `validation/run_hybrid_execution.py` |
| Heuristica greedy | Ausente | Cerrada | `validation/sweep_ilp_pareto.py` |
| Estudios de ablacion | Ausentes | Cerrada | `validation/run_ilp_ablation_suite.py`, `validation/generate_ilp_report_assets.py` |
| Sensibilidad paramétrica reproducible | Ausente | Cerrada | `validation/run_ilp_sensitivity.py` |
| Validacion de calidad final bajo ejecutor hibrido | Ausente | Cubierta operacionalmente, pendiente de cierre doctoral amplio | `validation/run_hybrid_execution.py`, artefactos agregados de calidad |

### 13.6 Diagnostico sintetico vigente

La conclusion ejecutiva que antes estaba separada en un documento historico queda ahora integrada aqui: el proyecto ya no debe describirse como un sistema que solo sabe medir y decidir. Tras el cierre de Fases 2, 3 y 4, el repositorio mide, decide, simula y ejecuta; y tras el endurecimiento metodologico de 2026-03-26, lo hace ademas bajo un criterio mas estricto de admisibilidad de artefactos.

La unica brecha de gran escala que permanece abierta corresponde a Fase 5 en sentido doctoral fuerte: completar la matriz experimental final, consolidar comparaciones amplias por arquitectura y cerrar la narrativa monografica definitiva con soporte estadistico suficiente para defensa.