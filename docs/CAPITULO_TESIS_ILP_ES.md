# Capítulo 5
## Formulación ILP robusta para partición heterogénea CPU-GPU: fundamentos matemáticos, implementación y validez experimental

### Resumen

Este capítulo presenta una formulación de programación lineal entera mixta de variables binarias para asignar capas de un modelo de aprendizaje profundo a CPU o GPU bajo criterios combinados de tiempo, energía y transferencia, con restricciones de memoria y mecanismos de robustificación estadística. La propuesta no se limita a enumerar ecuaciones: establece el puente entre evidencia empírica y decisión óptima, justifica la linealización de términos de corte, discute sesgos potenciales de modelado y define protocolos de sensibilidad y validación para uso doctoral. En continuidad con el Capítulo 4, los coeficientes del modelo provienen de artefactos de profiling trazables y robustificados, de forma que la solución matemática sea operacionalmente ejecutable en infraestructura heterogénea real.

Dentro del sistema completo, este capítulo formaliza el núcleo de decisión de PRISM: Partitioning and Resource Intelligence for System Memory.

### 5.1 Introducción y alcance del problema

La asignación heterogénea de cómputo durante el entrenamiento profundo es un problema combinatorio con acoplamiento estructural. Cada capa puede ejecutarse en CPU o GPU, pero esa decisión no afecta solo su costo local: altera también el costo de comunicación con las capas adyacentes en el grafo de dependencias. En consecuencia, la política de asignación global no puede reducirse a una heurística miope basada únicamente en el throughput individual de cada operador. Una capa puede ser más rápida en GPU y, sin embargo, resultar subóptima en GPU si su ubicación induce múltiples cortes de arista con alto costo de transferencia.

En coherencia con el alcance metodológico consolidado del proyecto, este capítulo asume como contrato de alcance la versión fuerte del modelo: asignación independiente para forward y backward, persistencia de activaciones como decisión binaria explícita y compatibilidad operacional con planificación asíncrona para transferencia en flujo y precarga explícita. La formulación base presentada en las secciones iniciales debe interpretarse como núcleo matemático sobre el que se anclan dichas extensiones, no como una alternativa de alcance reducido.

La programación lineal entera (ILP) ofrece un marco natural para este escenario porque representa decisiones binarias ejecutables, incorpora restricciones físicas en forma lineal y habilita resolución exacta o cuasi exacta con solvers maduros. No obstante, la utilidad del ILP depende de la calidad de sus coeficientes: una formulación impecable alimentada con métricas sesgadas produce decisiones formalmente óptimas pero operacionalmente frágiles. Por ello, el alcance del presente capítulo presupone la metodología de profiling del Capítulo 4 y se centra en traducir evidencia empírica robusta en un problema de optimización resoluble, interpretable y auditable.

### 5.2 Notación formal y estructura de datos

Sea \(G=(V,E)\) un grafo dirigido acíclico donde \(V\) es el conjunto de nodos de capa y \(E\subseteq V\times V\) es el conjunto de aristas de dependencia tensorial. Cada arista \((u,v)\) codifica que la salida de \(u\) es insumo de \(v\). Para cada nodo \(v\in V\) se disponen costos robustos por dispositivo: \(T_{gpu}(v), T_{cpu}(v)\) para tiempo; \(E_{gpu}(v), E_{cpu}(v)\) para energía; y \(M_{gpu}(v), M_{cpu}(v)\) para memoria. Para cada arista \((u,v)\in E\), el costo de transferencia \(C_{tr}(u,v)\) se deriva del artefacto de transferencia consciente de arista construido en profiling.

Adicionalmente se consideran presupuestos de memoria \(B_{gpu}\) y \(B_{cpu}\), que representan límites físicos de viabilidad para la asignación. Estos presupuestos convierten la formulación en un modelo de optimización factible sobre hardware real, no solo en una minimización abstracta de costos.

### 5.3 Variables de decisión y semántica operacional

La variable de asignación por nodo se define como:

**Ecuación (I-1). Variable binaria de asignación por nodo.**

$$x_v\in\{0,1\},\qquad x_v=1\Rightarrow v\text{ en GPU},\quad x_v=0\Rightarrow v\text{ en CPU}$$

Para modelar la discontinuidad de dispositivo en cada arista se introduce:

**Ecuación (I-2). Variable binaria de corte de arista.**

$$y_{uv}\in\{0,1\},\qquad (u,v)\in E$$

El significado operacional es directo: \(y_{uv}=0\) implica que ambos extremos de la arista residen en el mismo dispositivo, mientras que \(y_{uv}=1\) indica un corte CPU-GPU con costo de transferencia asociado. La inclusión explícita de \(y_{uv}\) evita depender de penalizaciones implícitas y permite descomponer la función objetivo en componentes nodales y de comunicación con interpretación causal.

### 5.4 Función objetivo multicriterio escalarizada

La función objetivo combina tiempo, energía y transferencia mediante una escalarización ponderada. Para cada nodo se definen costos compuestos por dispositivo:

**Ecuación (I-3). Costo nodal ponderado en GPU.**

$$C_{gpu}(v)=w_tT_{gpu}(v)+w_eE_{gpu}(v)$$

**Ecuación (I-4). Costo nodal ponderado en CPU.**

$$C_{cpu}(v)=w_tT_{cpu}(v)+w_eE_{cpu}(v)$$

Para cada arista se define la penalización de comunicación:

**Ecuación (I-5). Costo de corte ponderado.**

$$C_{cut}(u,v)=w_{tr}C_{tr}(u,v)$$

La minimización global queda entonces:

**Ecuación (I-6). Objetivo ILP de partición heterogénea.**

$$
\min Z=
\sum_{v\in V}\left[x_vC_{gpu}(v)+(1-x_v)C_{cpu}(v)\right]
+\sum_{(u,v)\in E}y_{uv}C_{cut}(u,v)
$$

El primer término selecciona el costo nodal según el dispositivo asignado; el segundo penaliza la fragmentación topológica por cortes de dependencia. Los pesos \(w_t\), \(w_e\) y \(w_{tr}\) son hiperparámetros de política y, por tanto, deben documentarse en toda corrida reportada.

### 5.5 Linealización exacta del corte de arista

El comportamiento deseado para \(y_{uv}\) es \(y_{uv}=|x_u-x_v|\), expresión no lineal que debe transformarse para mantener un MILP lineal. Se impone el siguiente sistema de desigualdades:

**Ecuación (I-7). Cota inferior de diferencia directa.**

$$y_{uv}\ge x_u-x_v$$

**Ecuación (I-8). Cota inferior de diferencia inversa.**

$$y_{uv}\ge x_v-x_u$$

**Ecuación (I-9). Cota superior por suma.**

$$y_{uv}\le x_u+x_v$$

**Ecuación (I-10). Cota superior complementaria.**

$$y_{uv}\le 2-x_u-x_v$$

La equivalencia exacta puede demostrarse por exhaustión de los cuatro casos binarios posibles de \((x_u,x_v)\). Cuando ambos valen cero, las cotas superiores fuerzan \(y_{uv}=0\). Cuando ambos valen uno, la cota complementaria también fuerza \(y_{uv}=0\). Cuando \((x_u,x_v)=(1,0)\), la Ecuación (I-7) exige \(y_{uv}\ge1\), y la binariedad implica \(y_{uv}=1\). Cuando \((x_u,x_v)=(0,1)\), la Ecuación (I-8) fuerza análogamente \(y_{uv}=1\). Por tanto, el sistema lineal reproduce exactamente la norma absoluta de diferencia en dominio binario sin introducir aproximación.

### 5.6 Restricciones de memoria y viabilidad física

La factibilidad operativa se garantiza imponiendo presupuestos de memoria por dispositivo:

**Ecuación (I-11). Restricción de memoria GPU.**

$$\sum_{v\in V}M_{gpu}(v)x_v\le B_{gpu}$$

**Ecuación (I-12). Restricción de memoria CPU.**

$$\sum_{v\in V}M_{cpu}(v)(1-x_v)\le B_{cpu}$$

Estas restricciones impiden soluciones formalmente óptimas pero no ejecutables. Su interpretación experimental es inmediata: una reducción de \(B_{gpu}\) desplaza masa de asignación hacia CPU, mientras que incrementos de \(B_{gpu}\) habilitan regiones de menor latencia si los costos nodales favorecen GPU. El barrido controlado de \(B_{gpu}\) constituye, de hecho, una vía práctica para construir fronteras de compromiso entre factibilidad y desempeño.

### 5.7 Robustificación estadística de coeficientes

Los coeficientes del modelo no se toman de una única corrida, sino de estadísticos agregados en el Capítulo 4. Se define la robustificación genérica:

**Ecuación (I-13). Transformación robusta de métrica base.**

$$\hat{m}=\mu_m+k_\sigma\sigma_m$$

A partir de ella se construyen los costos robustos por nodo:

**Ecuación (I-14). Tiempo robusto en GPU.**

$$T_{gpu}(v)=\widehat{gpu\_fwd\_time}(v)+\widehat{gpu\_bwd\_time}(v)$$

**Ecuación (I-15). Tiempo robusto en CPU.**

$$T_{cpu}(v)=\widehat{cpu\_fwd\_time}(v)+\widehat{cpu\_bwd\_time}(v)$$

**Ecuación (I-16). Energía robusta en GPU.**

$$E_{gpu}(v)=\widehat{gpu\_fwd\_energy}(v)+\widehat{gpu\_bwd\_energy}(v)$$

**Ecuación (I-17). Energía robusta en CPU.**

$$E_{cpu}(v)=\widehat{cpu\_fwd\_energy}(v)+\widehat{cpu\_bwd\_energy}(v)$$

El parámetro \(k_\sigma\) controla explícitamente el conservadurismo de la política: \(k_\sigma=0\) corresponde a modo nominal, y \(k_\sigma>0\) incorpora margen por variabilidad observada. Esta decisión no es meramente numérica, sino metodológica: expresa cuánto riesgo operacional acepta el investigador en la traducción de evidencia empírica a decisión combinatoria.

### 5.8 Integración multi-hardware

Cuando se busca una política de partición transferible entre varios hosts, los coeficientes deben fusionarse antes de resolver el ILP. Se emplean dos estrategias principales. La primera es la agregación de peor caso:

**Ecuación (I-18). Agregación robusta de peor caso.**

$$\bar{c}=\max_i c_i$$

La segunda es una agregación por media y dispersión inter-host:

**Ecuación (I-19). Agregación media más heterogeneidad.**

$$\bar{c}=\mu(c)+k_d\sigma(c)$$

La estrategia de peor caso prioriza robustez extrema y minimiza degradación en el host más desfavorable. La estrategia media más dispersión busca equilibrio entre desempeño promedio y heterogeneidad del parque. La elección entre ambas debe depender del objetivo del despliegue: estabilidad garantizada en todo host, o rendimiento promedio alto con riesgo controlado.

### 5.9 Complejidad del modelo y consideraciones de resolución

Si \(|V|=n\) y \(|E|=m\), el modelo incluye \(n+m\) variables binarias y \(4m+2\) restricciones estructurales principales, sin contar restricciones de dominio que el solver maneja internamente. El crecimiento en \(m\) domina la complejidad práctica porque cada arista agrega cuatro restricciones de linealización. En grafos densos o con alta conectividad de saltos residuales, la dificultad combinatoria aumenta de forma marcada.

La implementación contempla tres motores de resolución: `pulp` con CBC para resolución MILP general, `exhaustive` para instancias pequeñas donde se desea verificación exacta por enumeración total, y `auto` para seleccionar el motor en función del tamaño. El modo exhaustivo no se usa para producción, sino como oráculo de corrección metodológica durante la validación del flujo experimental.

### 5.10 Trazabilidad con artefactos de entrada

La instancia ILP se construye a partir de tres artefactos principales. El archivo `*_metrics_stats.csv` aporta costos nodales robustos de tiempo, energía y memoria. El archivo `*_graph_edges.csv` define la estructura de dependencias y, por tanto, el conjunto de aristas \(E\). El archivo `*_transfer_edges.csv` aporta el costo de transferencia por arista, que se transforma en penalización de corte. Esta separación de fuentes no es accidental: garantiza que cada término de la función objetivo y de las restricciones sea trazable al artefacto que lo sustenta.

En la versión endurecida del flujo, esta trazabilidad no se limita a conocer el archivo de origen, sino que incluye criterios de admisibilidad metodológica antes de resolver. El cargador de datos ILP rechaza por defecto artefactos agregados que no acrediten calidad muestral suficiente, calibración de transferencia empíricamente medida y procedencia estructural válida del grafo. En términos operativos, ello significa que `quality_flag` debe indicar suficiencia estadística, `transfer_calibration_source` debe ser `measured`, y `graph_trace_source` debe corresponder a una ruta estructural admisible (`torch_fx` o `torch_export_decoder_only`). La topología lineal de respaldo solo puede consumirse bajo anulación diagnóstica explícita y debe considerarse fuera del protocolo doctoral principal. Con ello, la resolución ILP deja de apoyarse solo en artefactos disponibles y pasa a apoyarse en artefactos metodológicamente aceptables.

En la implementación, `src/ilp/data_loader.py` gestiona la lectura y normalización de insumos, `src/ilp/model_builder.py` construye las matrices y vectores de la instancia, `src/ilp/solve.py` invoca el backend seleccionado y `src/ilp/export_solution.py` produce los artefactos de salida (`ilp_assignment.csv`, `ilp_cut_edges.csv`, `ilp_solution_summary.json`). La trazabilidad completa permite auditar cualquier solución desde el valor de la variable binaria hasta la medición de profiling que generó su coeficiente asociado.

### 5.11 Sensibilidad, estabilidad y protocolo de evaluación

Una formulación doctoral no puede limitarse a reportar una solución puntual; debe demostrar estabilidad frente a variación de parámetros. Se propone un protocolo de sensibilidad en cuatro pasos: fijar una configuración base; variar un parámetro por vez en una rejilla controlada; resolver la instancia para cada punto de la rejilla; y cuantificar cambios en objetivo y estructura de asignación. Para comparar la sensibilidad de diferentes parámetros se define la elasticidad discreta:

**Ecuación (I-20). Elasticidad discreta del objetivo.**

$$\mathcal{E}_{Z,p}=\frac{\Delta Z/Z}{\Delta p/p}$$

donde \(p\in\{w_t,w_e,w_{tr},k_\sigma,B_{gpu}\}\).

La estabilidad estructural entre dos soluciones \(x^{(a)}\) y \(x^{(b)}\) se cuantifica mediante distancia de Hamming normalizada:

**Ecuación (I-21). Distancia de asignación entre soluciones.**

$$D_H\left(x^{(a)},x^{(b)}\right)=\frac{1}{|V|}\sum_{v\in V}\mathbf{1}\left[x_v^{(a)}\neq x_v^{(b)}\right]$$

Un valor bajo de \(D_H\) ante perturbaciones moderadas de parámetros sugiere una política estructuralmente robusta. Un valor alto indica que la solución es inestable y que la configuración de pesos o robustificación puede estar en una zona de transición sensible.

### 5.12 Validación de soluciones y comparación con baselines

La validación de cada corrida ILP debe cubrir tres dimensiones. La primera es factibilidad matemática, verificando binariedad de variables y cumplimiento de restricciones de memoria y linealización. La segunda es coherencia económica, comprobando que variaciones de parámetros produzcan tendencias esperadas: por ejemplo, aumentos de \(w_{tr}\) deberían reducir cortes, y reducciones de \(B_{gpu}\) deberían disminuir la fracción de nodos en GPU. La tercera es validez comparativa frente a baselines simples que permitan medir ganancia incremental del modelo.

Los baselines mínimos recomendados son `all_cpu`, `all_gpu` cuando sea factible por memoria, y una heurística greedy nodal sin penalización de corte. La comparación con estos baselines permite separar cuánto de la mejora proviene de la estructura del ILP y cuánto provendría de decisiones triviales. Si el ILP no supera claramente a la heurística greedy en escenarios con costos de transferencia altos, ello sugiere revisar pesos o calidad de coeficientes en los artefactos de profiling.

Para interpretación causal de resultados, conviene descomponer el objetivo total en tres componentes:

**Ecuación (I-22). Descomposición aditiva del objetivo.**

$$Z=Z_{node,gpu}+Z_{node,cpu}+Z_{cut}$$

con

$$
Z_{node,gpu}=\sum_{v\in V}x_vC_{gpu}(v),\quad
Z_{node,cpu}=\sum_{v\in V}(1-x_v)C_{cpu}(v),\quad
Z_{cut}=\sum_{(u,v)\in E}y_{uv}C_{cut}(u,v)
$$

Esta descomposición permite identificar si la mejora total proviene principalmente de mejor asignación nodal o de reducción de fragmentación comunicativa.

### 5.13 Amenazas a la validez del modelo ILP

Las amenazas a la validez se agrupan en tres clases. La primera es la amenaza de medición: si los coeficientes de entrada están sesgados por ruido o por mezclas de precisión ejecutada no controlada, la solución óptima del ILP optimiza un problema mal especificado. La segunda es la amenaza de modelado: la linealidad de la función objetivo y de las restricciones no captura interacciones no lineales de hardware, como saturación de buses o efectos de contención compartida. La tercera es la amenaza de transferibilidad: una política ajustada a un único host puede degradarse en infraestructuras heterogéneas de distinta generación.

Las mitigaciones se apoyan en el diseño de los capítulos 4 y 5 como bloque metodológico conjunto: robustificación de coeficientes por \(\mu+k_\sigma\sigma\), estrategias de fusión multi-hardware, análisis de sensibilidad con reporte de estabilidad estructural, validación cruzada con backend exhaustivo en subinstancias pequeñas y compuertas de admisibilidad previas a la construccion de la instancia. Estas compuertas reducen un riesgo antes subestimado: que el solver produzca una solucion formalmente correcta sobre artefactos metodologicamente degradados. Ninguna mitigación elimina totalmente el riesgo, pero en conjunto elevan sustancialmente la credibilidad empírica de la política obtenida y mantienen continuidad terminológica con el contrato de evidencia fijado en la documentación canónica.

### 5.14 Figuras y tablas recomendadas para manuscrito

**Figura I-1. Esquema de partición binaria sobre DAG.**
Representación del grafo de capas con nodos coloreados por dispositivo y aristas de corte destacadas para visualizar la relación entre decisión nodal y penalización comunicativa.

**Figura I-2. Flujo de construcción de instancia ILP desde profiling.**
Diagrama que conecta `*_metrics_stats.csv`, `*_graph_edges.csv` y `*_transfer_edges.csv` con el constructor de modelo y el solver.

**Figura I-3. Sensibilidad del número de cortes frente a \(w_{tr}\).**
Curva esperada decreciente que verifica semántica del peso de transferencia en la formulación.

**Figura I-4. Fracción de nodos en GPU frente a presupuesto \(B_{gpu}\).**
Curva de factibilidad-rendimiento para discutir límites físicos de memoria y comportamiento de la política.

**Figura I-5. Estabilidad estructural frente a \(k_\sigma\).**
Evolución de \(D_H\) respecto a una solución base para evaluar robustez de asignación ante mayor conservadurismo estadístico.

**Tabla I-1. Notación principal del modelo ILP.**

| Símbolo | Definición | Dominio |
|---|---|---|
| \(x_v\) | Asignación de nodo \(v\) (GPU=1, CPU=0) | Binario |
| \(y_{uv}\) | Indicador de corte en arista \((u,v)\) | Binario |
| \(C_{gpu}(v)\) | Costo nodal ponderado en GPU | Real no negativo |
| \(C_{cpu}(v)\) | Costo nodal ponderado en CPU | Real no negativo |
| \(C_{tr}(u,v)\) | Costo de transferencia de arista | Real no negativo |
| \(B_{gpu}\) | Presupuesto de memoria GPU | Real positivo |
| \(B_{cpu}\) | Presupuesto de memoria CPU | Real positivo |
| \(k_\sigma\) | Factor de robustificación estadística | Real no negativo |
| \(D_H\) | Distancia de Hamming normalizada entre políticas | [0,1] |

**Tabla I-2. Correspondencia entre artefactos y términos del modelo.**

| Artefacto | Campo relevante | Uso en ILP |
|---|---|---|
| `*_metrics_stats.csv` | tiempos, energías y memorias robustas | Costos nodales y restricciones |
| `*_graph_edges.csv` | `src_id`, `dst_id` | Definición del conjunto \(E\) |
| `*_transfer_edges.csv` | `transfer_sym_ms` | Penalización por corte de arista |
| Configuración experimental | pesos y presupuestos | Parametrización de objetivo y factibilidad |

**Tabla I-3. Checklist mínimo de validación por corrida ILP.**

| Validación | Criterio de aceptación |
|---|---|
| Factibilidad de memoria | Se cumplen (I-11) y (I-12) |
| Dominio de variables | \(x_v,y_{uv}\in\{0,1\}\) |
| Coherencia de corte | Se cumplen (I-7) a (I-10) |
| Integridad de salida | Existen `assignment`, `cut_edges` y `summary` |
| Comparación base | Mejora o justificación frente a `all_cpu` |

### 5.15 Tabla de referencia cruzada de ecuaciones

La siguiente tabla mapea cada ecuación del capítulo con sus variables clave y con el módulo de implementación asociado.

| Ecuación | Nombre | Variables principales | Módulo de implementación |
|---|---|---|---|
| (I-1) | Asignación por nodo | \(x_v\) | `src/ilp/model_builder.py` |
| (I-2) | Corte por arista | \(y_{uv}\) | `src/ilp/model_builder.py` |
| (I-3) | Costo nodal GPU | \(w_t,w_e,T_{gpu},E_{gpu}\) | `src/ilp/model_builder.py` |
| (I-4) | Costo nodal CPU | \(w_t,w_e,T_{cpu},E_{cpu}\) | `src/ilp/model_builder.py` |
| (I-5) | Costo de corte ponderado | \(w_{tr},C_{tr}\) | `src/ilp/model_builder.py` |
| (I-6) | Objetivo total | \(Z,x_v,y_{uv}\) | `src/ilp/model_builder.py` |
| (I-7) | Cota inferior directa | \(x_u,x_v,y_{uv}\) | `src/ilp/model_builder.py` |
| (I-8) | Cota inferior inversa | \(x_u,x_v,y_{uv}\) | `src/ilp/model_builder.py` |
| (I-9) | Cota superior por suma | \(x_u,x_v,y_{uv}\) | `src/ilp/model_builder.py` |
| (I-10) | Cota superior complementaria | \(x_u,x_v,y_{uv}\) | `src/ilp/model_builder.py` |
| (I-11) | Memoria GPU | \(M_{gpu},x_v,B_{gpu}\) | `src/ilp/model_builder.py` |
| (I-12) | Memoria CPU | \(M_{cpu},x_v,B_{cpu}\) | `src/ilp/model_builder.py` |
| (I-13) | Robustificación base | \(\hat{m},\mu_m,k_\sigma,\sigma_m\) | `src/ilp/data_loader.py` |
| (I-14) | Tiempo robusto GPU | \(T_{gpu}\) | `src/ilp/data_loader.py` |
| (I-15) | Tiempo robusto CPU | \(T_{cpu}\) | `src/ilp/data_loader.py` |
| (I-16) | Energía robusta GPU | \(E_{gpu}\) | `src/ilp/data_loader.py` |
| (I-17) | Energía robusta CPU | \(E_{cpu}\) | `src/ilp/data_loader.py` |
| (I-18) | Fusión de peor caso | \(\bar{c},c_i\) | `src/ilp/data_loader.py` |
| (I-19) | Fusión media+dispersión | \(\bar{c},k_d\) | `src/ilp/data_loader.py` |
| (I-20) | Elasticidad discreta | \(\mathcal{E}_{Z,p}\) | `validation/sweep_ilp_pareto.py` |
| (I-21) | Distancia de Hamming | \(D_H,x^{(a)},x^{(b)}\) | `validation/sweep_ilp_pareto.py` |
| (I-22) | Descomposición del objetivo | \(Z_{node,gpu},Z_{node,cpu},Z_{cut}\) | `src/ilp/export_solution.py` |

### 5.16 Conclusiones

El capítulo establece una formulación ILP robusta, interpretable y operacionalmente ejecutable para partición CPU-GPU de entrenamiento profundo. La contribución principal consiste en traducir costos empíricos trazables en una decisión binaria estructurada sobre grafo, manteniendo simultáneamente tractabilidad matemática y viabilidad física de despliegue. La linealización exacta del corte de arista, las restricciones explícitas de memoria y la robustificación estadística de coeficientes conforman un núcleo metodológico coherente que evita la disociación entre modelo formal y realidad experimental.

Desde la perspectiva doctoral, el aporte no reside solo en obtener una política de asignación eficiente para un caso particular, sino en proponer un marco reproducible para inferir políticas estables bajo incertidumbre y heterogeneidad de hardware. La articulación entre el Capítulo 4 y el presente capítulo completa la cadena metodológica: primero se construye evidencia empírica con control de calidad; después se formaliza la decisión combinatoria sobre esa evidencia. El capítulo de resultados que sigue debe evaluar, con base en esta formulación, no solo mejora promedio, sino también estabilidad estructural, sensibilidad paramétrica y transferibilidad entre plataformas.

### 5.17 Referencias de implementación

La implementación principal de este capítulo se localiza en los módulos `src/ilp/data_loader.py`, `src/ilp/model_builder.py`, `src/ilp/solve.py` y `src/ilp/export_solution.py`. Las utilidades de ejecución experimental y barrido paramétrico se encuentran en `validation/run_ilp_partition.py` y `validation/sweep_ilp_pareto.py`. Estas rutas constituyen la base verificable de las ecuaciones y protocolos presentados.
