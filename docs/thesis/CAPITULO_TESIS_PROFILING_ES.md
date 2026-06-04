# Capítulo 4
## Profiling empírico para la partición heterogénea CPU-GPU en entrenamiento de modelos de aprendizaje profundo

### Resumen

Este capítulo desarrolla la fundamentación metodológica, matemática y operacional del sistema de profiling empírico utilizado como base para la optimización de partición CPU-GPU. A diferencia de un enfoque descriptivo centrado en listados de métricas, el presente texto establece un marco de inferencia experimental: define unidades de observación, caracteriza fuentes de variabilidad, formaliza ecuaciones de costo y especifica protocolos de trazabilidad desde medición cruda hasta coeficiente robusto listo para optimización. La propuesta integra instrumentación por capa, diagnóstico de precisión en tiempo de ejecución, modelado de transferencia consciente de arista y agregación estadística de réplicas para reducir sensibilidad a ruido transitorio. Como resultado, el capítulo ofrece una metodología reproducible y defendible para construir evidencia cuantitativa utilizable en modelado ILP de despliegue heterogéneo, cuya formulación formal se desarrolla en el Capítulo 5.

En el marco del proyecto, este bloque metodológico constituye el fundamento empírico de PRISM: Partitioning and Resource Intelligence for System Memory.

La lectura de este capítulo se enmarca en el alcance metodológico consolidado del proyecto: los coeficientes aquí construidos alimentan una formulación ILP con separación forward/backward, persistencia de activaciones y ejecución con planificación asíncrona en el tiempo de ejecución.

### 4.1 Introducción, motivación científica y preguntas de investigación

La asignación de componentes de un modelo de aprendizaje profundo entre procesadores de propósito general y unidades de procesamiento gráfico constituye, en términos de ingeniería de sistemas, un problema de decisión multicriterio bajo incertidumbre operacional. En ausencia de una caracterización empírica detallada del comportamiento en tiempo de ejecución, las decisiones de optimización se apoyan inevitablemente en supuestos idealizados que ignoran latencias de despacho, fluctuaciones energéticas, restricciones de memoria y costos de comunicación entre dispositivos. Este desacople entre teoría y fenomenología operacional conduce con frecuencia a soluciones nominalmente óptimas pero inestables en despliegue real, cuyo rendimiento se degrada precisamente en las condiciones de carga y heterogeneidad que caracterizan los entornos de investigación y producción.

El problema metodológico central es, por tanto, construir un flujo de profiling que cumpla simultáneamente cinco propiedades estructurales. En primer lugar, debe poseer granularidad suficiente para representar la heterogeneidad de costo entre capas individuales, sin colapsar en un promedio que oculte las diferencias de comportamiento que determinan la bondad de la partición. En segundo lugar, debe ofrecer estabilidad estadística para sostener inferencia reproducible, lo que exige replicación controlada y mecanismos de agregación robusta. En tercer lugar, debe garantizar trazabilidad completa entre cada artefacto numérico y el contexto de ejecución que lo generó. En cuarto lugar, debe ser portable entre clases de hardware heterogéneo, permitiendo comparaciones válidas entre configuraciones de servidor distintas. Finalmente, debe producir coeficientes directamente compatibles con la construcción de instancias de optimización combinatoria.

Para estructurar el capítulo en términos de preguntas de investigación, se plantean cinco interrogantes de distinta naturaleza. El primero concierne a la obtención de costos por capa que mantengan fidelidad operacional en presencia de ruido de ejecución. El segundo se refiere a la separación, en la medida de lo posible, de cómputo efectivo respecto de overhead de orquestación. El tercero aborda la representación de transferencia inter-dispositivo de forma útil para penalización de cortes en un grafo de dependencias. El cuarto trata la robustificación de coeficientes de tiempo y energía sin perder sensibilidad a diferencias de arquitectura. El quinto examina cómo garantizar que los artefactos producidos sean auditables y reproducibles en campañas multi-servidor.

A partir de estos interrogantes se fijan tres hipótesis operativas que estructuran el diseño experimental. La primera sostiene que la observación por capa mejora la interpretabilidad y la utilidad para partición respecto a la observación agregada por modelo completo, en tanto expone heterogeneidad de costo invisible a nivel global. La segunda establece que la combinación de media, dispersión y cuantiles produce coeficientes más estables que los derivados de una corrida única, dado que las réplicas capturan variabilidad intrínseca del entorno de ejecución. La tercera propone que el modelado de transferencia consciente de arista reduce el sesgo optimista en decisiones de partición, al penalizar explícitamente los cortes que generan comunicación inter-dispositivo.

### 4.2 Marco conceptual: alcance, supuestos y unidades de observación

El alcance del capítulo comprende el entrenamiento supervisado bajo PyTorch con rutas de ejecución en CPU y GPU, con instrumentación por módulos hoja y exportación de artefactos tabulares procesables. La fase cubierta es la adquisición y consolidación de datos previa a la resolución del modelo de optimización, cuyo diseño se detalla en el Capítulo 5. Los límites explícitos del alcance son igualmente importantes para la correcta interpretación de los resultados: no se modelan efectos de red multi-nodo a nivel de micro-paquete, ni se realiza descomposición energética por subsistema de hardware, ni la heurística de backward por capa presupone separación completa de la fase de retropropagación.

Los supuestos centrales del marco son tres. Se asume que el grafo de dependencias del modelo puede obtenerse mediante trazado simbólico con `torch.fx` o, para ciertos modelos decoder-only no cubiertos por dicho trazado, mediante exportación estructural con `torch.export`. La ruta lineal sobre módulos hoja se conserva únicamente como mecanismo diagnóstico y no constituye evidencia estructural admisible para el protocolo doctoral principal. Se asume además que las métricas de energía disponibles en el entorno de ejecución reflejan la tendencia operativa del sistema aunque no constituyan instrumentación física perfecta, lo cual admite su uso comparativo sin atribuirles precisión energética absoluta. Finalmente, se asume que la variabilidad entre réplicas es modelable mediante estadísticos de primer y segundo orden más cuantiles de cola, premisa suficiente para robustecer coeficientes destinados a optimización.

La unidad primaria de observación es la capa, entendida como módulo hoja ejecutable en la jerarquía del modelo. Esta elección responde a un compromiso entre expresividad y tractabilidad: una granularidad más fina, a nivel de kernel individual, aumenta el volumen de ruido y la complejidad del mapeo semántico sin aportar información adicional relevante para la decisión de partición; una granularidad más gruesa, a nivel de bloques agregados, oculta precisamente la heterogeneidad de costo que hace valiosa la partición selectiva. Para cada capa, el pipeline captura cinco canales de observación complementarios: tiempo, energía, memoria, transferencia e intensidad de cómputo. Tiempo y energía aportan el costo nodal directo; memoria establece la factibilidad de la asignación física; transferencia penaliza los cortes de dependencia; e intensidad de cómputo contextualiza la eficiencia en términos de operaciones aritméticas ejecutadas por unidad de tiempo.

El principio de trazabilidad atraviesa transversalmente el diseño del pipeline. Todo valor numérico final debe ser rastreable, al menos, a tres niveles de especificidad: la condición de ejecución —precisión, tamaño de lote, semilla, hardware—, la medición cruda por corrida individual, y la transformación estadística aplicada durante la fase de agregación. Este principio determina la estructura de metadatos, el contrato de artefactos y los esquemas de nomenclatura del sistema.

### 4.3 Arquitectura de instrumentación

#### 4.3.1 Orquestación general del pipeline

El pipeline se estructura en cinco etapas lógicas secuenciales cuya separación facilita la verificación independiente de cada componente. La primera corresponde al parseo de argumentos y normalización del entorno, donde se establece el contexto de ejecución y se validan las dependencias de hardware disponibles. La segunda evalúa la política de precisión mediante sondeo de soporte ISA, determinando qué rutas de cómputo son válidas en la plataforma objetivo antes de iniciar cualquier medición. La tercera construye el modelo y resuelve las entradas experimentales, recurriendo por defecto a datasets reales y permitiendo entradas sintéticas solo bajo anulación diagnóstica explícita. La cuarta registra los hooks de instrumentación y ejecuta las iteraciones de entrenamiento, capturando mediciones por capa en tiempo real durante cada paso. La quinta consolida las métricas, calcula los estadísticos derivados y escribe los artefactos de salida.

La orquestación reside en `src/profiler.py`, que delega la captura de métricas en `src/runner/training_profiler.py` y utiliza los módulos de soporte en `src/core/` para políticas, utilidades de sistema y extracción de grafos. En la versión metodológicamente endurecida del flujo, la construcción de entradas usa datasets reales por defecto y exige anulación diagnóstica explícita para habilitar entradas sintéticas con fines de depuración. Esta separación de responsabilidades permite que cada componente sea testeable de forma independiente y que las políticas de precisión, el modelo de transferencia y la extracción estructural evolucionen sin acoplamiento rígido al bucle de medición principal.

#### 4.3.2 Instrumentación por módulo hoja y descomposición temporal

Se instrumentan exclusivamente los módulos hoja de la jerarquía del modelo para evitar doble conteo en estructuras anidadas: un bloque residual, por ejemplo, no se mide como unidad, sino a través de las capas convolucionales y de normalización que lo componen. Cada módulo recibe un pre-hook que marca el instante de inicio y, cuando el dispositivo destino es GPU, lanza un evento CUDA de inicio; y un post-hook que cierra la ventana temporal, calcula el tamaño de la activación de salida, cuenta los parámetros activos e invoca la estimación de FLOPs.

La descomposición temporal fundamental del profiling separa el tiempo de pared observado entre cómputo efectivo del kernel y overhead de despacho de las instrucciones:

**Ecuación (P-1). Descomposición temporal por capa.**

$$T^{wall}_{\ell} = T^{kernel}_{\ell} + T^{dispatch}_{\ell}$$

donde el término de despacho se obtiene como residuo no negativo:

**Ecuación (P-2). Overhead de despacho no negativo.**

$$T^{dispatch}_{\ell} = \max\left(0,\;T^{wall}_{\ell} - T^{kernel}_{\ell}\right)$$

Esta separación permite diagnosticar si la ineficiencia observada en una capa proviene de un kernel subóptimo o de una orquestación que impide el solapamiento de operaciones. Cuando el overhead de despacho domina sistemáticamente, la corrección adecuada no es migrar la capa a otro dispositivo sino revisar el patrón de lanzamiento; ambas causas quedan confundidas si solo se mide el tiempo de pared agregado.

El paso de entrenamiento instrumentado integra las tres etapas canónicas del aprendizaje supervisado:

**Ecuación (P-3). Tiempo de paso de entrenamiento.**

$$T^{step} = T^{fwd} + T^{bwd} + T^{opt}$$

El término $T^{opt}$ captura la contribución del optimizador, que puede ser significativa en escenarios con estados voluminosos como Adam con acumuladores de segundo momento por parámetro.

### 4.4 Modelo energético empírico y estimación de carga computacional

#### 4.4.1 Aproximación energética por potencia media

Ante la ausencia de instrumentación física por kernel en la mayoría de entornos de investigación, la energía se aproxima mediante la potencia promedio observada durante la ventana temporal de medición:

**Ecuación (P-4). Energía de fase.**

$$E = P_{avg} \cdot T$$

Para cada capa y dispositivo se reportan términos de fase forward y backward. En la versión metodológica actual, la fase backward se instrumenta de forma explícita por capa mediante hooks de retropropagación (`register_full_backward_pre_hook` y `register_full_backward_hook`) para capturar tiempo efectivo y overhead de despacho en backward. Esta medición directa es la fuente primaria de costo para la formulación ILP.

**Ecuación (P-5). Medición backward con fallback conservador.**

$$T_{\ell}^{bwd}=\begin{cases}T_{\ell,meas}^{bwd},&\text{si hay eventos backward válidos}\\\gamma T_{\ell}^{fwd},&\text{si no hay observabilidad suficiente}\end{cases},\quad E_{\ell}^{bwd}=\begin{cases}E_{\ell,meas}^{bwd},&\text{si hay eventos backward válidos}\\\gamma E_{\ell}^{fwd},&\text{si no hay observabilidad suficiente}\end{cases}$$

con $\gamma = 2.0$ reservado como mecanismo de continuidad para casos degradados de observabilidad. De esta manera, el pipeline prioriza evidencia medida y mantiene un respaldo explícito que evita huecos en los canales de costo cuando un operador no emite señal backward utilizable.

#### 4.4.2 FLOPs teóricos por operador y eficiencia relativa

Los FLOPs teóricos por capa se calculan a partir de fórmulas analíticas específicas para los operadores dominantes. Para una convolución bidimensional con $C_{out}$ canales de salida, altura $H_{out}$, anchura $W_{out}$, $C_{in}$ canales de entrada, $g$ grupos y kernel $K_x \times K_y$:

**Ecuación (P-6). FLOPs de convolución 2D.**

$$\mathrm{FLOPs}_{conv}=2\,C_{out}H_{out}W_{out}\left(\frac{C_{in}}{groups}K_xK_y\right)$$

Para una capa completamente conectada con $P$ ejemplos por lote:

**Ecuación (P-7). FLOPs de capa lineal.**

$$\mathrm{FLOPs}_{linear}=2\,P\,in\_features\,out\_features$$

Para mecanismos de atención multi-cabeza con $B$ ejemplos, secuencia de longitud $S$ y dimensión $D$:

**Ecuación (P-8). Aproximación de FLOPs en atención.**

$$\mathrm{FLOPs}_{attn}\approx4BSD^2+2BS^2D$$

Estas fórmulas no sustituyen la medición de tiempo, que captura efectos reales de pipeline y caché, pero proveen una escala de demanda computacional útil para el diagnóstico de eficiencia y la interpretación de disparidades entre capas. Para proporcionar un punto de referencia independiente del hardware, se estima un pico empírico de throughput mediante una operación matricial de referencia:

**Ecuación (P-9). Pico empírico de throughput.**

$$\mathrm{TFLOPS}_{peak}=\frac{2N^3}{10^{12}\Delta t}$$

La eficiencia relativa de cada capa se define como el cociente entre su throughput efectivo y el pico empírico:

**Ecuación (P-10). Eficiencia relativa de capa.**

$$\eta_{\ell}=\frac{\mathrm{TFLOPS}_{\ell}}{\mathrm{TFLOPS}_{peak}}$$

El cociente $\eta_{\ell}$ permite comparar eficiencia entre servidores heterogéneos normalizando por la capacidad local observada, lo que hace la métrica transferible entre plataformas con características de hardware distintas.

### 4.5 Política de precisión y compuertas de validez

#### 4.5.1 Sondeo ISA y política de ejecución efectiva

La precisión numérica con la que se ejecuta una capa no es necesariamente la precisión solicitada en la configuración del experimento. En arquitecturas x86 modernas, la ejecución en punto flotante de media precisión requiere soporte explícito de instrucciones ISA: `fp16` requiere la extensión `avx512_fp16`, mientras que `bf16` necesita `avx512_bf16` o la combinación `amx_bf16` con `amx_tile`. En ausencia de estas extensiones, el runtime puede silenciosamente degradar la precisión a `fp32` sin notificar el cambio, produciendo mediciones de tiempo y energía que corresponden a un régimen computacional distinto al configurado y, por tanto, incomparables con otras corridas de la campaña.

Para evitar este problema, el pipeline implementa un sondeo ISA explícito antes de ejecutar cualquier medición. Cuando el soporte requerido no está disponible, la corrida para esa precisión se marca como no ejecutada con su motivo documentado en los metadatos. La política de ejecución efectiva produce un estado con campos `allowed`, `cpu_precision_executed`, `reason` y `status`, que permiten trazabilidad completa de las decisiones de runtime. Esta salida tiene tres funciones operativas: habilita la exclusión controlada de corridas no válidas durante la agregación estadística, garantiza consistencia de auditoría entre los artefactos CSV y JSON, y preserva la integridad comparativa de la campaña al impedir que regímenes computacionales distintos queden mezclados bajo la misma etiqueta de precisión.

#### 4.5.2 Preflight CPU FP16 con timeout adaptativo

La ejecución de FP16 en CPU puede generar bloqueos indefinidos en modelos cuya propagación backward excede límites razonables de tiempo bajo condiciones de precisión reducida. El pipeline incorpora un mecanismo de preflight por etapas con un timeout adaptativo que escala con la complejidad del modelo:

**Ecuación (P-11). Timeout backward adaptativo.**

$$\tau_{bwd}=\max\left(10,\;T_{fwd}\cdot\gamma\cdot s\right)$$

donde $s$ es un factor de seguridad que amplía el margen en proporción al tiempo de forward observado. El diseño adaptativo tiene una ventaja doble: los falsos descartes se reducen porque el timeout escala con la complejidad real del modelo, evitando rechazar corridas válidas de modelos de alta latencia; los bloqueos reales se contienen porque el factor $\gamma \cdot s$ acota el tiempo de espera razonablemente para cada arquitectura. Un timeout fijo, en cambio, sería simultáneamente demasiado restrictivo para modelos complejos y demasiado permisivo para modelos simples.

### 4.6 Extracción estructural de grafo

#### 4.6.1 Ruta principal por torch.fx y topología lineal solo diagnóstica

La extracción de la topología de dependencias utiliza el trazado simbólico de `torch.fx` como ruta principal. El procedimiento `symbolic_trace` instrumenta el modelo para capturar el grafo de operaciones durante una ejecución con tensores simbólicos, generando nodos y aristas que representan el flujo de activaciones a través del modelo. Cuando `ShapeProp` es ejecutable, los nodos se enriquecen con metadatos de forma que permiten estimar el tamaño del tensor transmitido por cada arista de dependencia. Para ciertos modelos decoder-only, el pipeline admite una segunda ruta estructural basada en `torch.export`, igualmente preservada en metadatos. El campo `graph_trace_source` en los artefactos de salida documenta qué ruta de extracción fue utilizada en cada corrida, preservando la trazabilidad de la calidad estructural del grafo.

Cuando el trazado simbólico falla —situación que puede ocurrir con modelos que contienen flujo de control dinámico o llamadas a funciones no trazables bajo el mecanismo de `torch.fx`— el flujo intenta primero una ruta estructural alternativa para modelos decoder-only mediante `torch.export`. Solo si el operador ejecuta de forma explícita una anulación diagnóstica se permite una topología lineal de respaldo que construye una cadena de módulos hoja siguiendo el orden de registro en el modelo. Aunque dicha ruta puede ser útil para depuración, no es admisible como base de evidencia doctoral porque no captura aristas de dependencia no secuenciales —como conexiones residuales o atajos de omisión— y, por tanto, deja sin representar costos de corte relevantes para el modelo ILP.

#### 4.6.2 Esquema de nodos y aristas

El artefacto `*_graph_nodes.csv` contiene, para cada nodo, campos de identidad estructural (`node_id`, `node_name`, `op_type`, `topo_index`), campos de huella de cómputo (`params_mb`, `activ_out_mb`) y el campo de procedencia (`graph_trace_source`). Esta estructura permite realizar joins entre el grafo estructural y las métricas de rendimiento por nombre de capa, habilitando la construcción completa de la instancia ILP con información de topología y coste para cada nodo.

El artefacto `*_graph_edges.csv` codifica cada arista con los identificadores de nodo origen y destino (`src_id`, `dst_id`), el tamaño del tensor transmitido (`tensor_mb`), su forma (`tensor_shape`) y los nombres de productor y consumidor. Esta información es consumida directamente por el modelo de transferencia para calcular el costo de corte de cada arista de dependencia, y por el constructor de instancias ILP para definir el conjunto $E$ de la formulación.

### 4.7 Modelo de transferencia consciente de arista

#### 4.7.1 Calibración por tramos y rodilla de congestión

El modelo de transferencia mantiene la descomposición alpha-beta, pero en versión por tramos para capturar saturación práctica de PCIe bajo mensajes grandes. Para cada dirección de transferencia —`h2d` y `d2h`— se estiman dos regímenes: nominal y congestionado, separados por una rodilla de tamaño calibrada empíricamente.

**Ecuación (P-12). Transferencia direccional por tramos.**

$$t_{dir}(S)=\alpha_{dir}+\frac{S}{\beta_{dir}^{eff}(S)},\qquad \beta_{dir}^{eff}(S)=\begin{cases}\beta_{dir}^{nom},&S\le S_{knee,dir}\\\beta_{dir}^{cong},&S>S_{knee,dir}\end{cases}$$

donde $S$ es el tamaño del tensor en MB, $\alpha_{dir}$ es la latencia base y $\beta_{dir}^{nom},\beta_{dir}^{cong}$ representan anchos de banda efectivos en régimen nominal y congestionado, respectivamente. La diferenciación por dirección y por tramo reduce el sesgo optimista que aparece cuando se usa un único ancho de banda para todo el rango de tamaños.

#### 4.7.2 Atenuación por overlap y presión local de ramificación

En entornos con streams CUDA que permiten solapamiento cómputo-comunicación, el costo efectivo se atenúa por overlap, pero se incrementa por presión local de frontera cuando un productor alimenta múltiples consumidores. Esta presión se deriva de la estructura DAG y se interpreta como proxy local de contención.

**Ecuación (P-13). Factor de overlap.**

$$f_{ov}=1-0.5\sigma, \qquad \sigma\in[0,1]$$

de modo que el costo efectivo base en cada dirección se reduce en proporción al solapamiento:

**Ecuación (P-14). Costo efectivo con presión de ramificación.**

$$t_{edge}^{eff}=\frac{t_{h2d}^{eff}+t_{d2h}^{eff}}{2}\left(1+\kappa\,p_u\right)$$

donde $p_u$ es una medida local de presión de salida del nodo productor $u$ en el DAG y $\kappa$ controla la sensibilidad del recargo. El descuento máximo de overlap (50% cuando $\sigma\to1$) se mantiene deliberadamente conservador; el recargo por ramificación evita que el modelo subestime cortes en fronteras con alto fan-out, escenario frecuente de congestión efectiva en PCIe.

### 4.8 Contrato de artefactos y agregación estadística robusta

#### 4.8.1 Artefactos del pipeline

El pipeline produce cuatro categorías de artefactos con roles diferenciados en la cadena metodológica. El artefacto de medición cruda, `*_metrics.csv`, contiene una fila por capa por corrida con todas las columnas de identidad, tiempo, energía, memoria, transferencia, precisión ejecutada y estado de ejecución. El artefacto de metadatos, `*_meta.json`, encapsula el contexto completo de la corrida: hardware, calibración de transferencia, políticas de precisión aplicadas y rutas de artefactos generados. Los artefactos estructurales `*_graph_nodes.csv`, `*_graph_edges.csv` y `*_transfer_edges.csv` habilitan el modelado de la topología de dependencias y el costo de comunicación por arista. Finalmente, el artefacto agregado `*_metrics_stats.csv` concentra los estadísticos robustos por grupo experimental y constituye la entrada directa del constructor de instancias ILP.

#### 4.8.2 Claves de agrupamiento y estadísticos de tendencia y dispersión

La agregación de réplicas se realiza agrupando por la combinación completa de dimensiones experimentales: modelo, tamaño de lote, precisión solicitada, optimizador, nombre de capa, tipo de operador, y precisión ejecutada en cada dispositivo. Esta granularidad de agrupamiento garantiza que los estadísticos resultantes correspondan a condiciones computacionalmente homogéneas y no mezclen regímenes distintos bajo la misma etiqueta. Para cada canal numérico dentro de cada grupo se calculan la media y la desviación estándar muestral:

**Ecuación (P-15). Media muestral.**

$$\mu=\frac{1}{n}\sum_{i=1}^{n}x_i$$

**Ecuación (P-16). Desviación estándar muestral.**

$$\sigma=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\mu)^2}$$

Adicionalmente se calculan los cuantiles p50, p90 y p95, que capturan el comportamiento de cola del canal. Los cuantiles superiores son especialmente relevantes para coeficientes destinados a optimización: una capa con promedio favorable pero cola alta representa un riesgo de degradación en condiciones de carga no nominales, y ese riesgo debe quedar representado en el modelo de decisión. La lectura de resultados exige, por tanto, una interpretación jerárquica que va de media para tendencia central, a desviación estándar para volatilidad, a p90 y p95 para riesgo operacional.

#### 4.8.3 Robustificación para consumo de optimización

Los coeficientes que se entregan al modelo ILP no son medias brutas sino valores robustificados que incorporan explícitamente la dispersión observada durante la campaña experimental:

**Ecuación (P-17). Coeficiente robusto.**

$$\hat{m}=\mu_m+k_\sigma\sigma_m$$

El parámetro $k_\sigma$ regula el nivel de conservadurismo de los coeficientes. Cuando $k_\sigma = 0$ se opera en modo nominal, favoreciendo el rendimiento esperado bajo condiciones promedio. Para $k_\sigma > 0$ el coeficiente incorpora un margen sobre la media proporcional a la dispersión observada, reduciendo la probabilidad de que la solución ILP favorezca asignaciones que son óptimas en el promedio muestral pero degradadas bajo la variabilidad típica del entorno operacional. La elección de $k_\sigma$ es, en consecuencia, un parámetro de diseño del experimento doctoral con implicaciones directas sobre el conservadurismo de la política de partición resultante, y debe reportarse junto con todos los resultados del Capítulo 5.

### 4.9 Validez experimental y protocolo reproducible

#### 4.9.1 Dimensiones de validez y mitigaciones

La adopción de un marco de validez explícito es lo que distingue este trabajo de un ejercicio de benchmarking ad hoc. La validez interna se ve amenazada principalmente por tres fuentes: la interferencia de procesos de fondo del sistema operativo, que introduce ruido en las mediciones temporales de forma no controlada por el experimentador; las fluctuaciones térmicas del hardware, que afectan el throttling de frecuencia y alteran el rendimiento de forma correlacionada con el tiempo transcurrido desde el inicio del experimento; y las sincronizaciones implícitas del runtime de CUDA, que pueden concentrar latencias acumuladas en puntos imprevisibles del grafo de ejecución. El pipeline mitiga estas amenazas mediante réplicas por configuración, control de semillas de aleatoriedad, políticas de timeout y preflight temprano, y exclusión de artefactos parciales en la fase de agregación.

La validez de construcción exige que las métricas elegidas representen genuinamente el fenómeno de interés, que en este caso es el costo diferencial de asignar una capa a cada dispositivo bajo condiciones de entrenamiento reales. La inclusión simultánea de cinco canales complementarios y su contextualización mediante metadatos de hardware y política responde a esta exigencia, permitiendo interpretación causal en lugar de correlación superficial entre métricas. En particular, el riesgo de que las métricas elegidas no representen el fenómeno se mitiga combinando tiempo, energía, memoria, transferencia e intensidad de cómputo, de modo que ningún canal sea el único responsable de una decisión de asignación.

La validez externa se ve comprometida por el riesgo de sobreajuste a una configuración de hardware particular. Las mitigaciones incorporadas en el diseño son la normalización de eficiencia por pico local observado, la agregación de coeficientes multi-hardware con estrategias de peor caso o media más dispersión, y el reporte explícito de la dispersión entre hosts para que el lector pueda evaluar la transferibilidad de los resultados a hardware distinto al experimental.

#### 4.9.2 Protocolo experimental de nivel tesis y criterio de tamaño muestral

Para campañas de profiling con nivel de evidencia suficiente para defensa doctoral, se propone el siguiente protocolo en siete pasos. Primero, validar el entorno y las dependencias: versión de PyTorch, disponibilidad de drivers y soporte ISA. Segundo, ejecutar una corrida de smoke controlada por hardware para confirmar que el flujo produce artefactos bien formados antes de invertir tiempo de cómputo en la campaña completa. Tercero, ejecutar la campaña principal con el número de réplicas decidido según criterio de tamaño muestral. Cuarto, verificar completitud de artefactos comparando las rutas esperadas con las producidas y documentando toda corrida que haya fallecido. Quinto, ejecutar la agregación robusta para producir `*_metrics_stats.csv`. Sexto, auditar la consistencia de metadatos: verificar que los estados de precisión ejecutada coincidan con los solicitados o que las discrepancias estén documentadas con su razón. Séptimo, publicar el dataset consolidado con versionado de configuración, hash de entorno y lista de corridas excluidas con su justificación.

Para el diseño del tamaño muestral, el campo `n_runs` del artefacto agregado no es un dato accesorio sino un indicador de robustez inferencial. Como línea base para la fase exploratoria se recomiendan tres réplicas por configuración, suficientes para detectar inestabilidad grosera. Para la fase de resultados consolidados, cinco a siete réplicas ofrecen intervalos de cuantiles razonablemente estables. Cuando para una métrica clave $m$ se observa un coeficiente de variación elevado,

**Ecuación (P-18). Coeficiente de variación.**

$$CV_m=\frac{\sigma_m}{\mu_m}$$

se recomienda aumentar el número de réplicas hasta estabilizar los intervalos de cuantiles. Este criterio de escalado adaptativo permite concentrar los recursos computacionales en las configuraciones más variables, que son precisamente las que introducen mayor incertidumbre en los coeficientes del modelo ILP.

### 4.10 Lectura interpretativa e integración con el modelo de partición

#### 4.10.1 Patrones de diagnóstico y lectura cruzada de canales

La interpretación de resultados de profiling en contexto de partición heterogénea requiere leer los canales no de forma aislada sino en combinación. Existen patrones diagnósticos recurrentes que revelan distintas fuentes de ineficiencia. Cuando una capa presenta FLOPs teóricos altos combinados con eficiencia relativa $\eta_\ell$ baja, el fenómeno típicamente subyacente es un cuello de banda de memoria o el uso de kernels no optimizados para la arquitectura objetivo, más que una limitación genuina de capacidad de cómputo. Cuando el tiempo de pared es alto y el overhead de despacho domina una fracción significativa, el problema es probablemente de orquestación y sincronización del lanzamiento de kernels, e implica que simplemente reasignar la capa a otro dispositivo no resolverá la ineficiencia. Cuando los costos de transferencia de las aristas incidentes superan los costos de cómputo nodal, esa capa es un candidato de alta penalización en el modelo ILP, indicando que cualquier corte en sus dependencias será costoso independientemente de la calidad de las asignaciones individuales de los nodos adyacentes.

La combinación de tiempo GPU forward bajo y costo de transferencia en aristas incidentes alto es especialmente relevante para el análisis de partición: la ventaja computacional de la GPU podría quedar totalmente anulada por el costo de comunicación, haciendo subóptima la asignación a GPU en ausencia del contexto de topología que aporta el modelo ILP. Inversamente, una capa con overhead de despacho alto y throughput en TFLOPS moderado presenta una ineficiencia de orquestación que no desaparecería con una migración de dispositivo. Esta lectura cruzada es precisamente la que justifica tratar la partición como un problema de optimización estructurado sobre el grafo de dependencias, en lugar de un conjunto de decisiones locales miopes.

#### 4.10.2 Ejemplo metodológico integrado

Para ilustrar la función epistemológica del pipeline, considérese una configuración experimental con parámetros fijos $(modelo, batch, precision, optimizador)$ y $r$ réplicas ejecutadas. Para una capa $\ell$, el procedimiento de construcción de coeficientes procede como sigue: se miden $T_{\ell,i}^{fwd}$ y $E_{\ell,i}^{fwd}$ para cada réplica $i = 1, \ldots, r$; se miden además $T_{\ell,i}^{bwd}$ y $E_{\ell,i}^{bwd}$ con hooks backward, usando la rama de fallback de (P-5) solo cuando falta observabilidad; se agregan todas las réplicas para obtener $\mu$, $\sigma$ y cuantiles de cada canal; se aplica la robustificación de la Ecuación (P-17) con el parámetro $k_\sigma$ del experimento para producir $\hat{T}_\ell$ y $\hat{E}_\ell$; y se incorpora el campo `transfer_sym_ms` junto con los metadatos de tramo/rodilla en aristas incidentes como costo de corte. El resultado es un vector de coeficientes trazable desde el valor final hasta la medición cruda de cada réplica, pasando por el contexto de hardware y la política de precisión aplicada.

#### 4.10.3 Nexo con el Capítulo 5

El profiling define la calidad de la frontera de decisión que explota el modelo ILP. Los costos nodales provienen de los canales temporales y energéticos robustificados del artefacto de estadísticos; las restricciones de memoria dependen de los campos de huella de activación por capa; y las penalizaciones de corte dependen directamente de los costos de transferencia calibrados por arista. Si alguno de estos insumos es sistemáticamente sesgado, la solución ILP hereda ese sesgo: puede favorecer asignaciones que son óptimas bajo los coeficientes del experimento pero inestables en condiciones de hardware distintas. La robustez del pipeline de profiling es, en consecuencia, una condición necesaria aunque no suficiente para la validez del modelo de partición del Capítulo 5.

### 4.11 Figuras y tablas de referencia para manuscrito

**Figura P-1. Arquitectura causal del pipeline de profiling.**
Diagrama de bloques que muestra la secuencia completa desde la ejecución instrumentada hasta los artefactos robustos, con anotación de los módulos de software responsables de cada fase y los artefactos intermedios producidos.

**Figura P-2. Cronograma de captura por capa.**
Representación temporal de $T^{wall}$, $T^{kernel}$ y $T^{dispatch}$ para un subconjunto de capas representativas, ilustrando la separación de costos y la variabilidad del overhead bajo condiciones reales de ejecución.

**Figura P-3. Curvas por tramo y rodilla de congestión por dirección.**
Gráfico que relaciona tamaño de tensor con latencia calibrada en `h2d` y `d2h`, mostrando el cambio de pendiente en $S_{knee}$ y la diferencia entre régimen nominal y congestionado.

**Figura P-4. Efecto conjunto de overlap y presión de ramificación.**
Superficie o familia de curvas de $t_{edge}^{eff}$ frente a $\sigma$ y $p_u$, evidenciando la tensión entre atenuación por solapamiento y recargo por frontera ramificada.

**Figura P-5. Distribuciones de réplicas por canal de costo.**
Gráfico de violín o caja para los canales de tiempo y energía por capa, mostrando media, desviación estándar y cuantiles p50/p90/p95 y evidenciando la dispersión real del entorno experimental.

**Tabla P-1. Notación principal del capítulo.**

| Símbolo | Definición | Unidad |
|---|---|---|
| $T_{\ell}^{fwd}$ | Tiempo forward de la capa $\ell$ | ms |
| $T_{\ell}^{bwd}$ | Tiempo backward de la capa $\ell$ | ms |
| $T_{\ell}^{dispatch}$ | Overhead de despacho de la capa $\ell$ | ms |
| $E_{\ell}^{fwd}$ | Energía forward de la capa $\ell$ | J |
| $E_{\ell}^{bwd}$ | Energía backward de la capa $\ell$ | J |
| $\alpha_{dir}$ | Latencia base de transferencia por dirección | ms |
| $\beta_{dir}$ | Ancho de banda efectivo de transferencia por dirección | MB/ms |
| $\sigma$ | Proporción de overlap cómputo-comunicación | adimensional |
| $k_\sigma$ | Factor de robustificación estadística | adimensional |
| $\gamma$ | Factor de fallback backward/forward cuando no hay observabilidad | adimensional |
| $\eta_{\ell}$ | Eficiencia relativa de capa | adimensional |
| $\hat{m}$ | Coeficiente robusto de la métrica $m$ | según canal |

**Tabla P-2. Contrato de artefactos del pipeline de profiling.**

| Artefacto | Nivel | Rol en el pipeline |
|---|---|---|
| `*_metrics.csv` | capa-corrida | Medición cruda multicanal |
| `*_meta.json` | corrida | Contexto de hardware y trazabilidad |
| `*_graph_nodes.csv` | estructura | Nodos del grafo con huellas de cómputo |
| `*_graph_edges.csv` | estructura | Dependencias de datos entre capas |
| `*_transfer_edges.csv` | arista | Costos de comunicación calibrados |
| `*_metrics_stats.csv` | agregado | Coeficientes robustos para ILP |

**Tabla P-3. Dimensiones de validez: riesgos y mitigaciones.**

| Dimensión | Riesgo principal | Mitigación implementada |
|---|---|---|
| Interna | Interferencia de procesos de fondo | Réplicas y control de semillas |
| Interna | Fluctuaciones térmicas y throttling | Timeout y preflight por etapas |
| Interna | Sincronizaciones implícitas CUDA | Cuantiles de cola y exclusión de artefactos parciales |
| De construcción | Precisión ejecutada distinta a solicitada | Sondeo ISA y política de ejecución efectiva |
| De construcción | Uso de topología lineal de fallback fuera de diagnóstico | Rutas estructurales admisibles con documentación de `graph_trace_source` |
| Externa | Sobreajuste a un host específico | Normalización por pico local y fusión multi-hardware |
| Externa | Sesgo residual en congestión PCIe global | Modelo por tramos con rodilla, presión local de ramificación y reporte de calibración |

### 4.12 Discusión y conclusiones

La principal diferencia entre este enfoque y un benchmarking convencional reside en el objetivo inferencial. Un benchmark tradicional busca comparar plataformas bajo condiciones controladas y reportar valores de rendimiento absoluto; el pipeline aquí descrito busca construir coeficientes de decisión robustos para optimización estructurada de partición. Esta diferencia no es cosmética: exige integrar semántica de datos, mecanismos de validación, control de incertidumbre y trazabilidad documental en el diseño mismo del sistema, en lugar de añadirlos como consideraciones secundarias durante la fase de análisis. El profiling se convierte así en un ejercicio de ingeniería epistémica: el artefacto producido no es solo una tabla de tiempos, sino una estructura de conocimiento con garantías de interpretabilidad, reproducibilidad y auditabilidad.

Desde esta perspectiva, el valor del sistema de profiling no se mide únicamente por su precisión puntual en la estimación de tiempos de ejecución, sino por su capacidad para sostener decisiones estables de partición bajo variabilidad operacional real. Una política de asignación derivada de coeficientes bien calibrados y robustificados se comportará de manera predecible en hardware distinto al de entrenamiento; una derivada de coeficientes sesgados producirá optimismo in-sample e inestabilidad out-of-sample. La inclusión de políticas de precisión, calibración de transferencia y agregación estadística robusta responde a ese criterio de calidad del coeficiente, que es más exigente que el criterio habitual del benchmarking.

Las contribuciones principales de este capítulo se articulan en cuatro dimensiones. Primero, desde el punto de vista de la instrumentación, se propone una arquitectura que descompone el tiempo de ejecución por capa en cómputo del kernel y overhead de despacho, separando dos fuentes de ineficiencia con causas y remedios distintos. Segundo, desde el punto de vista energético, se formaliza una aproximación reproducible que, aunque no constituye instrumentación física perfecta, es consistente entre corridas y permite comparación relativa útil para la optimización. Tercero, desde el punto de vista estructural, la integración con `torch.fx` y, cuando corresponde, con `torch.export` produce artefactos de topología admisibles para el modelo ILP, transformando la transferencia de datos entre dispositivos en una magnitud optimizable en lugar de un efecto colateral ignorado. Cuarto, desde el punto de vista estadístico, el esquema de robustificación mediante $k_\sigma$ introduce un parámetro interpretable que controla el conservadurismo de los coeficientes y puede ajustarse en función del nivel de riesgo operacional admisible por el experimento.

El capítulo presenta también limitaciones metodológicas que deben reconocerse abiertamente en la presentación doctoral. Aunque la medición backward por hooks reduce sustancialmente el sesgo de una aproximación puramente heurística, persisten casos de observabilidad incompleta donde debe activarse el fallback con $\gamma$. El modelo de transferencia por tramos y presión local captura congestión de primer orden, pero no representa explícitamente contención multi-flujo global en el bus compartido. La medición energética mediante potencia promedio no distingue contribuciones de subsistemas de hardware, limitando la granularidad del diagnóstico energético. Estas limitaciones son reconocidas y delinean una agenda de investigación futura orientada a telemetría de bus de mayor resolución y desagregación energética por dominio funcional.

En conjunto, el profiling deja de ser una fase auxiliar de ingeniería y se consolida como el fundamento empírico del modelo de partición presentado en el Capítulo 5. Sin la base de métricas trazables que este capítulo produce, la formulación ILP carecería de los coeficientes necesarios para traducir sus ecuaciones formales en decisiones de partición válidas para hardware heterogéneo real.

### 4.13 Tabla de referencia cruzada de ecuaciones

La siguiente tabla mapea cada ecuación del capítulo con las variables que introduce o emplea, y con el módulo de código fuente donde se implementa la lógica correspondiente.

| Ecuación | Nombre | Variables principales | Módulo de implementación |
|---|---|---|---|
| (P-1) | Descomposición temporal | $T^{wall}_\ell$, $T^{kernel}_\ell$, $T^{dispatch}_\ell$ | `src/runner/training_profiler.py` |
| (P-2) | Dispatch no negativo | $T^{dispatch}_\ell$ | `src/runner/training_profiler.py` |
| (P-3) | Tiempo de paso | $T^{step}$, $T^{fwd}$, $T^{bwd}$, $T^{opt}$ | `src/runner/training_profiler.py` |
| (P-4) | Energía de fase | $E$, $P_{avg}$, $T$ | `src/core/metrics.py` |
| (P-5) | Backward medido con fallback | $T^{bwd}_\ell$, $E^{bwd}_\ell$, $\gamma$ | `src/runner/training_profiler.py` |
| (P-6) | FLOPs convolución 2D | $C_{out}$, $H_{out}$, $W_{out}$, $C_{in}$, $K_x$, $K_y$ | `src/core/metrics.py` |
| (P-7) | FLOPs capa lineal | $P$, $in\_features$, $out\_features$ | `src/core/metrics.py` |
| (P-8) | FLOPs atención | $B$, $S$, $D$ | `src/core/metrics.py` |
| (P-9) | Pico empírico de throughput | $N$, $\Delta t$ | `src/core/metrics.py` |
| (P-10) | Eficiencia relativa | $\eta_\ell$, $\mathrm{TFLOPS}_{peak}$ | `src/core/metrics.py` |
| (P-11) | Timeout backward adaptativo | $\tau_{bwd}$, $T_{fwd}$, $\gamma$, $s$ | `src/core/precision_policy.py` |
| (P-12) | Transferencia direccional por tramos | $t_{dir}$, $\alpha_{dir}$, $\beta^{nom}_{dir}$, $\beta^{cong}_{dir}$, $S_{knee}$ | `src/runner/training_profiler.py` |
| (P-13) | Factor de overlap | $f_{ov}$, $\sigma$ | `src/runner/training_profiler.py` |
| (P-14) | Costo efectivo con presión de ramificación | $t^{eff}_{edge}$, $f_{ov}$, $\kappa$, $p_u$ | `src/runner/training_profiler.py` |
| (P-15) | Media muestral | $\mu$ | `validation/aggregate_metrics_stats.py` |
| (P-16) | Desviación estándar muestral | $\sigma$ | `validation/aggregate_metrics_stats.py` |
| (P-17) | Coeficiente robusto | $\hat{m}$, $\mu_m$, $k_\sigma$, $\sigma_m$ | `validation/aggregate_metrics_stats.py` |
| (P-18) | Coeficiente de variación | $CV_m$, $\sigma_m$, $\mu_m$ | `validation/aggregate_metrics_stats.py` |

### 4.14 Referencias de implementación

Los módulos de código fuente relevantes para este capítulo se organizan por función en el pipeline. El punto de entrada de la orquestación es `src/profiler.py`. La captura de métricas, los hooks backward y el modelado de transferencia por tramos con presión de ramificación se implementan en `src/runner/training_profiler.py`. La política de precisión con sondeo ISA y preflight adaptativo reside en `src/core/precision_policy.py`. El cálculo de métricas derivadas —FLOPs, eficiencia, energía— se centraliza en `src/core/metrics.py`. La extracción de grafo por `torch.fx`, la ruta `torch.export` para modelos decoder-only y la topología lineal solo diagnóstica se implementan en `src/core/graph_extractor.py`. La agregación estadística robusta de réplicas se ejecuta mediante `validation/aggregate_metrics_stats.py`. Las validaciones de integridad de artefactos se encuentran en `validation/validate_all_models.py`.
