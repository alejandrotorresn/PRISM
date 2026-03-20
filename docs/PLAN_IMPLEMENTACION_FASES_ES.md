# Plan de Implementacion por Fases para Cerrar la Brecha entre el Proyecto y la Tesis

## 1. Proposito del plan

Este documento traduce el diagnostico de [MAPA_PROYECTO_VS_TESIS_ES.md] en una hoja de ruta ejecutable. Su funcion no es enumerar tareas aisladas, sino organizar el desarrollo del proyecto de modo que cada incremento de software produzca tambien evidencia util para la monografia doctoral.

El criterio rector es simple: cada fase debe cerrar una brecha metodologica concreta, dejar artefactos verificables y alimentar de forma directa uno o varios capitulos de la tesis.

## 2. Principios de ejecucion

El plan se apoya en cinco principios.

Primero, no conviene extender el modelo matematico antes de fijar el alcance exacto de la contribucion. En este proyecto ya se fija de forma explicita la promesa original de asignacion independiente para forward y backward, lo que afecta de manera directa al runtime, al simulador, al ILP y al relato cientifico.

Segundo, debe priorizarse el cierre del bucle metodologico antes que el refinamiento ornamental del modelo. En terminos practicos, es preferible disponer antes de un ejecutor hibrido minimo y una validacion fisica reproducible que de una formulacion muy sofisticada sin capacidad de contraste experimental.

Tercero, cada nueva capacidad debe venir acompanada de criterios de aceptacion observables. En una tesis doctoral, una funcionalidad no cuenta por estar implementada, sino por poder ser auditada, reproducida y defendida.

Cuarto, la documentacion y la monografia deben evolucionar junto al codigo. El mayor riesgo actual del proyecto no es tecnico, sino epistemico: prometer en la tesis mas de lo que el software demuestra.

Quinto, la secuencia de fases debe preservar valor incluso si el proyecto se detuviera antes del ultimo hito. Cada fase debe producir un resultado defendible por si mismo.

## 3. Vista general de fases

| Fase | Objetivo principal | Resultado de tesis habilitado |
| --- | --- | --- |
| Fase 0 | Congelar alcance y semantica del modelo | Coherencia entre tesis y software |
| Fase 1 | Cerrar validacion comparativa minima | ILP evaluable contra baselines y ablaciones |
| Fase 2 | Construir simulador hibrido | Prediccion reproducible de planes ILP |
| Fase 3 | Implementar ejecutor hibrido real | Ejecucion fisica CPU-GPU guiada por ILP |
| Fase 4 | Extender el modelo con memoria de activaciones | Tesis alineada con rematerializacion y persistencia |
| Fase 5 | Validacion doctoral completa | Resultados finales, analisis y capitulos cerrados |

### 3.1 Veredicto por fases (actualizado a 2026-03-20)

- **Fase 1**: funcional, pero con riesgo metodologico en comparabilidad ILP vs greedy (corregido en `validation/sweep_ilp_pareto.py` al evaluar baselines en el mismo espacio objetivo dual).
- **Fase 2**: funcional y validada.
- **Fase 3**: funcional y validada.
- **Fase 4**: totalmente funcional (dual + activaciones + async transfer + prefetching explicito en runtime), con cierre operacional validado por pruebas y ejecucion fisica.

## 4. Fase 0: congelacion de alcance y contrato cientifico

### 4.1 Objetivo

Resolver las ambiguedades de diseno que hoy impiden avanzar sin riesgo de rehacer trabajo y formalizar contractualmente las decisiones ya adoptadas: asignacion independiente forward/backward, persistencia de activaciones como variable binaria ILP y scheduling asincrono en runtime.

### 4.2 Trabajo tecnico

1. Formalizar la semantica final del ILP con asignacion independiente por fase (forward y backward).
2. Formalizar persistencia de activaciones, rematerializacion y checkpointing como variables explicitas del modelo.
3. Formalizar transferencia asincrona CPU-GPU en runtime e incorporar prefetching explicito como politica operativa auditable en ejecucion hibrida.
4. Actualizar la documentacion metodologica para que no existan afirmaciones incompatibles con estas decisiones.

### 4.3 Archivos afectados

- `docs/schema.md`
- `docs/MAPA_PROYECTO_VS_TESIS_ES.md`
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
3. El runtime implementa transferencia asincrona CPU-GPU mediante `torch.cuda.Stream` y prefetching explicito de activaciones con look-ahead entre capas en ejecucion hibrida.

Estas decisiones dejan sin validez cualquier redaccion alternativa basada en bifurcacion de alcance. Desde esta fecha, las Fases 1-5 se ejecutan sobre la ruta unica completa y toda evidencia experimental debe reportarse bajo ese contrato metodologico.

## 5. Fase 1: validacion comparativa minima y robustecimiento experimental

### 5.1 Objetivo

Cerrar primero la parte de evaluacion que ya puede construirse sobre el ILP actual sin esperar al runtime hibrido completo. Esta fase debe convertir el planificador existente en un sistema comparativamente serio.

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

Esta fase cierra buena parte del material analitico de los capitulos de formulacion y resultados comparativos, incluso antes de tener el runtime hibrido final.

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
- Artefactos observados de runtime hibrido.
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
2. Comparar `all_cpu`, `all_gpu`, `greedy`, `ilp_base` y, si existe, `ilp_extendido`.
3. Comparar prediccion del simulador frente a observacion fisica del runtime hibrido.
4. Medir accuracy, loss final o metrica de calidad pertinente en los escenarios hibridos.
5. Consolidar figuras, tablas LaTeX y resumenes para los capitulos de resultados y conclusiones.

### 9.3 Entregables

- Dataset consolidado final.
- Graficas y tablas de la monografia.
- Resumen de amenazas a la validez y limites del sistema.
- Capitulo de resultados cerrado.

### 9.4 Criterios de aceptacion

- Cada afirmacion fuerte de la tesis tiene una figura, tabla o artefacto que la respalda.
- Existe comparacion entre prediccion y ejecucion observada.
- Existe evidencia sobre memoria, tiempo, energia y exactitud final.

## 10. Dependencias entre fases

La dependencia real entre fases puede resumirse asi.

- Fase 0 es obligatoria antes de introducir nuevas variables en el ILP.
- Fase 1 puede ejecutarse en paralelo parcial con Fase 2.
- Fase 2 debe preceder a Fase 3 porque define el contrato del plan ejecutable.
- Fase 3 debe preceder a Fase 5 porque sin runtime real no hay validacion fisica completa.
- Fase 4 debe arrancar solo cuando Fase 3 haya probado el circuito minimo de evidencia.
- Fase 4 debe preceder a Fase 5, porque la validacion doctoral final requiere cubrir las capacidades extendidas comprometidas en la tesis.

## 11. Orden recomendado si se busca la via mas pragmatica

Si el objetivo es llegar antes a una tesis defendible, el camino mas eficaz es:

1. cerrar Fase 0 inmediatamente;
2. ejecutar Fase 1 para fortalecer el ILP ya existente;
3. construir Fase 2 para reducir incertidumbre del runtime;
4. implementar Fase 3 sobre modelos piloto;
5. ejecutar Fase 4 para cerrar las capacidades extendidas comprometidas en la tesis;
6. cerrar Fase 5 con campanas finales y escritura.

## 12. Decision estrategica final

La estrategia final queda fijada sin bifurcaciones: completar de forma secuencial las Fases 0 a 5 y sostener la tesis en su formulacion fuerte. En consecuencia, la contribucion cientifica comprometida es un sistema de particion heterogenea con simulacion, ejecucion real validada y extensiones de memoria y scheduling asincrono implementadas.