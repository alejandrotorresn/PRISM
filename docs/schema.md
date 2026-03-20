# Estructura de la Monografía Doctoral Post-Implementación

## Principios de Construcción

Esta estructura describe cómo quedará la monografía doctoral **después de completar el plan de implementación de fases** descrito en [PLAN_IMPLEMENTACION_FASES_ES.md](PLAN_IMPLEMENTACION_FASES_ES.md).

Cada capítulo se construye con evidencia tangible del software y cierra los gaps identificados en [MAPA_PROYECTO_VS_TESIS_ES.md](MAPA_PROYECTO_VS_TESIS_ES.md). La organización responde a tres principios doctorales:

1. **Valor independiente de cada capítulo**: Cada capítulo cierra un tramo metodológico específico y aporta evidencia verificable para la defensa doctoral integral.

2. **Alineación fase-a-capítulo**: Cada sección indica explícitamente en cuál fase del plan se genera su evidencia.

3. **Ruta única completa**: La tesis se ejecuta en la secuencia completa Fases 0-5, incluyendo modelo extendido con persistencia de activaciones y scheduling asíncrono.

---

## Capítulo 1: Introducción, Problema y Diseño de la Investigación

**Dependencia de fases**: Fase 0 (congelación de alcance)

Este capítulo presenta el marco doctoral con la máxima precisión. No se trata solo de motivar el problema, sino de delimitarlo explícitamente: qué implementa la tesis y qué deja para trabajo futuro.

Como resultado del cierre de Fase 0, el alcance queda fijado en su formulacion fuerte: asignacion independiente forward/backward, persistencia de activaciones como decision binaria ILP y scheduling asincrono para streaming/prefetching.

**Secciones**:

1.1 Contexto e introducción
- Saturación de memoria en entrenamiento profundo
- Heterogeneidad CPU-GPU como oportunidad estructural

1.2 Motivación técnica y relevancia científica
- Por qué el problema no se reduce a throughput
- Por qué la CPU es un recurso estructural, no marginal

1.3 Planteamiento formal del problema
- Definición del problema de partición heterogénea
- Objetivo: minimizar tiempo/memoria/energía bajo restricciones

1.4 Hipótesis de investigación
- Existe una asignación de capas óptima que mejora latencia sin degradar exactitud
- La asignación puede formularse como ILP y validarse experimentalmente

1.5 Objetivos general y específicos
- **General**: Diseñar y validar un sistema de partición CPU-GPU para entrenamientos profundos con modelo extendido y validación física integral
- **Específicos**:
  - Implementar asignación independiente para forward y backward en el ILP y en el runtime
  - Modelar persistencia de activaciones como decisión binaria
  - Implementar scheduling asíncrono para streaming/prefetching
  - Validar comparativamente contra baselines y verificar exactitud final

1.6 Alcance decisional fijado (resultado de Fase 0)
- Asignación forward/backward separada como compromiso obligatorio del modelo
- Persistencia de activaciones como variable binaria ILP obligatoria
- Streaming/prefetching implementados mediante scheduling asíncrono en runtime

1.7 Metodología general y validación
- Pipeline experimental: profiling → agregación → ILP → simulación → ejecución → medición
- Modelo experimental: redes piloto (simple_mlp, resnet50) con precisiones variadas
- Criterios de éxito: exactitud preservada, mejora de latencia/memoria demostrada

1.8 Contribución científica
- Sistema integral de partición heterogénea CPU-GPU con profiling robusto, ILP extendido, simulación reproducible y ejecución física validada
- Evidencia experimental de tiempo, memoria, energía y exactitud final bajo políticas de asignación y persistencia

1.9 Organización del documento

---

## Capítulo 2: Fundamentos Teóricos y Estado del Arte

**Dependencia de fases**: Ninguna (debe escribirse inmediatamente)

Proporciona la base académica para entender el problema y sitúa la contribución en el contexto global.

**Secciones**:

2.1 Entrenamiento de redes profundas y grafos computacionales
- Propagación forward/backward como DAG
- Descomposición por capas y módulos

2.2 Arquitecturas heterogéneas CPU-GPU y jerarquías de memoria
- Ancho de banda y latencia relativas
- Decisiones de data placement

2.3 Transferencia de datos, solapamiento y comunicación
- PCIe: costos de synchronous vs asynchronous transfers
- Solapamiento compute-communication

2.4 Estrategias de gestión de memoria
- Precision mixta (FP32, FP16, BF16)
- Checkpointing y rematerialización (técnicas inducidas)
- Entrenamiento distribuido y federado

2.5 Optimización declarativa: ILP en sistemas computacionales
- Revisión de Integer Linear Programming
- Aplicaciones previas en particionamiento de carga
- Robustificación estadística de parámetros

2.6 Estado del arte y vacios de investigación
- Soluciones existentes: distributed training (DDP, FSDP)
- Soluciones existentes: memory optimization techniques
- Brecha identificada: partición heterogénea offline con validación física

---

## Capítulo 3: Arquitectura Metodológica del Sistema

**Dependencia de fases**: Ninguna (debe escribirse inmediatamente)

Presenta el sistema experimental como un todo coherente, estableciendo la interconexión entre componentes antes de entrar en detalles técnicos.

**Secciones**:

3.1 Visión general del pipeline experimental
- Entrada: modelo PyTorch + configuración hardware
- Procesamiento: profiling → agregación → modelado → optimización → simulación → ejecución
- Salida: planes asignados, métricas observadas, evidencia de mejora

3.2 Estructura modular del software
- `src/`: núcleo (profiler, extractor de grafo, agregador, ILP, runtime)
- `validation/`: herramientas experimentales (sweep, simulador, executor, reporting)
- `scripts/`: orquestación de campañas

3.3 Contrato de artefactos y trazabilidad
- Artefactos por etapa: profiles.csv → aggregated_metrics.json → ilp_assignment.csv → execution_log.csv
- Reproducibilidad: cada artefacto incluye timestamp, seed, configuración

3.4 Definición de plan y su representación
- Plan = vector de asignaciones (nodo → dispositivo) + vector de cortes (arista → sí/no)
- Consumo de plan: simulador reconstruye secuencia temporal
- Ejecución de plan: runtime modifica colocación de módulos y tensores

3.5 Criterios de validez y riesgos metodológicos iniciales
- Suficiencia muestral en profiling
- Representatividad de modelos piloto
- Estabilidad de convergencia bajo ejecución hibrida

---

## Capítulo 4: Profiling Empírico para Partición Heterogénea CPU-GPU

**Dependencia de fases**: Ninguna (hoy está 85-90% completado)

Cierra la caracterización empírica del sistema. Los contenidos aquí no dependen de fases futuras; el capítulo es defensible inmediatamente.

**Secciones**:

4.1 Motivación empírica y preguntas de medición
- ¿Cuál es el tiempo/memoria/energía por capa en CPU vs GPU?
- ¿Cuánto cuesta transferir activaciones entre dispositivos?
- ¿Cómo cambian estos costos con precisión, batch size y arquitectura?

4.2 Arquitectura de instrumentación
- Hooks de PyTorch para capturar inicio/fin de operación por capa
- Decomposición temporal: forward, backward, optimizer step
- Medición de transferencias PCIe y solapamiento

4.3 Modelo de energía y carga computacional
- NVML para GPU power state y energy
- RAPL para CPU energy (si disponible)
- Estimación de FLOPs por capa desde descripción de operaciones

4.4 Política de precisión y validez
- Soporte para FP32, FP16, BF16 en perfilado
- Control de compuertas: número mínimo de réplicas, desviación máxima permitida

4.5 Extracción estructural del grafo
- Método 1 (preferido): `torch.fx.symbolic_trace()` para grafo exacto
- Método 2 (fallback): instrumentación module-wise basada en leaf modules
- Exportación: nodos (capas), aristas (activaciones), atributos (shape, dtype)

4.6 Modelo de transferencia consciente de arista
- Calibración de costo PCIe: latencia base + overhead tamaño
- Medición de overlap: ratio entre tiempo de compute y tiempo de transfer
- Factor de amortización para múltiples transferencias concurrentes

4.7 Agregación estadística robusta
- Estadísticas por capa: media, mediana, percentil 95, desviación estándar
- Estrategia `mu + k_sigma * sigma` para coeficientes robustos del ILP
- Validación de suficiencia muestral: mínimo de réplicas y coeficiente de variación

4.8 Validez experimental y reproducibilidad
- Controles: mismos seed, determinismo de PyTorch, fijación de frecuencia CPU/GPU si es posible
- Artefactos producidos: `{model}_{hardware}_{precision}_metrics.csv`, `{model}_graph.csv`, `{model}_transfers.csv`
- Validación: cross-checks entre métricas y razonabilidad de costos

---

## Capítulo 5: Formulación ILP Robusta para Partición Heterogénea CPU-GPU

**Dependencia de fases**: Fase 0 (congelación de alcance del modelo) + Fase 1 (validación comparativa y ablaciones)

Traduce los coeficientes empíricos en un problema de optimización formal. Tras Fase 0, el alcance queda fijado sobre el modelo extendido con asignación independiente por fase y persistencia de activaciones; tras Fase 1, se completa la validación comparativa.

**Secciones**:

5.1 Introducción y alcance del modelo (congelado en Fase 0)
- Declaración explícita del modelo extendido como alcance definitivo
- Asignación independiente para forward y backward con variables diferenciadas
- Integración de persistencia de activaciones y decisiones de recomputo/checkpointing

5.2 Notación formal
- Conjuntos: nodos V (capas), aristas E (transferencias), dispositivos D = {CPU, GPU}
- Parámetros: tiempo_cpu[v], tiempo_gpu[v], memoria_cpu[v], memoria_gpu[v], costo_transferencia[e], presupuestos

5.3 Variables de decisión
- x_fwd[v,d] ∈ {0,1}: asignación de la operación forward de la capa v al dispositivo d
- x_bwd[v,d] ∈ {0,1}: asignación de la operación backward de la capa v al dispositivo d
- p[v] ∈ {0,1}: decisión de persistencia de activación de la capa v
- y[e] ∈ {0,1}: corte de arista e cuando los extremos se ejecutan en dispositivos distintos

5.4 Función objetivo multicriterio
- Minimizar: α·latencia_total + β·memoria_max_gpu + γ·energia_total
- Latencia: suma de tiempos de ejecución en CPU, GPU y transferencias de cortes
- Memoria: max de presión sobre cada dispositivo

5.5 Linealizaciones exactas y restricciones
- Restricción de precedencia: si v → w es arista, latencia[w] ≥ latencia[v] + tiempo[v]
- Restricción de memoria: suma de memoria[dispositivo] ≤ presupuesto[dispositivo]
- Restricción de corte: y[v,w] ≥ χ(asignación[v] ≠ asignación[w])

5.6 Robustificación estadística
- Coeficientes base: MEAN(mediciones)
- Coeficientes robustos: μ + k_sigma·σ (percentil ~95 para k_sigma=2)
- Auditoría: tabla de variación por k_sigma para transparencia

5.7 Integración multi-hardware
- Fusión de perfiles: muestreo de múltiples GPUs (si disponible) con MAX o agregación conservadora
- Justificación: modelo predice worst-case en heterogeneidad

5.8 Complejidad y trazabilidad
- Complejidad: NP-hard (problema de particionamiento), solucionable exactamente para <50-100 nodos con CBC/PuLP
- Para instancias grandes: heurística greedy o relaxación
- Artefactos de traza: solver log, valor óptimo, gap, tiempo de resolución

5.9 Validación comparativa contra baselines (habilitada por Fase 1)
- Baseline all_cpu: todas las capas en CPU
- Baseline all_gpu: todas las capas en GPU
- Baseline greedy: asignación greedy por capa según criterio (e.g., tiempo_gpu < tiempo_cpu)
- Comparación: ILP vs baselines en latencia, memoria, energía

5.10 Estudios de sensibilidad (habilitados por Fase 1)
- Variación de k_sigma: cómo cambia solución con robustez estadística
- Variación de presupuesto GPU: curva de Pareto
- Variación de pesos (α, β, γ): sensibilidad de objetivo

5.11 Ablaciones formales (habilitadas por Fase 1)
- Ablación 1: sin topología (ignora dependencias de DAG)
- Ablación 2: sin costos de transferencia por arista
- Ablación 3: sin robustificación (usa media sin k_sigma)
- Ablación 4: modelo completo
- Comparación: ganancia marginal de cada componente

5.12 Extensión avanzada: persistencia de activaciones y rematerialización (Fase 4)
- Variables adicionales: p[v] ∈ {0,1} = "activación v es retenida en memoria"
- Restricciones: presupuesto de memoria incorpora decisión de retención vs recomputo
- Análisis: costo-beneficio de persistencia contra recomputo
- Esta sección es obligatoria dentro de la ruta completa de implementación y forma parte del núcleo de contribución doctoral

5.13 Amenazas a la validez
- Validez interna: presencia de variables ocultas que confunden asignación
- Validez externa: generalización de coeficientes a otros modelos/hardware
- Validez de constructo: ¿ILP realmente resuelve el problema declarado?

---

## Capítulo 6: Ejecución Híbrida, Simulación y Validación Experimental

**Dependencia de fases**: Fase 2 (simulador) + Fase 3 (ejecutor real) + Fase 5 (validación final)

Este es el capítulo que cierra la brecha entre planificación offline y operación real. Se construye en tres olas: predicción (Fase 2), ejecución (Fase 3), validación integral (Fase 5).

**Secciones**:

6.1 Introducción: cierre del bucle planificación-ejecución
- Descripción del desafío: transformar plan matemático en configuración de runtime real
- Arquitectura: representación del plan → simulación predictiva → ejecución física
- Validación: comparación predicción vs observación

6.2 Representación formal del plan de ejecución (Fase 2)
- Plan = (asignación: nodo→dispositivo, cortes: arista→bool, presupuestos observados)
- Contrato runtime: especificación de cómo el executor interpreta el plan
- Artefactos: `ilp_assignment.csv`, `ilp_cut_edges.csv`, `ilp_metrics_predicted.csv`

6.3 Simulador híbrido: arquitectura y validación topológica (Fase 2)
- Arquitectura del simulador: replay del DAG respetando precedencias
- Cálculo de latencia: suma de tiempos locales + penalizaciones de transferencia y serialización
- Validación de integridad: verificación de presupuestos, detecta violaciones topológicas
- Modos: nominal (media) vs robusto (percentil 95)

6.4 Estudio de predicción: latencia, memoria y costos de transferencia (Fase 2)
- Salida del simulador: desglose por dispositivo, por fase de ejecución, por coste de operación
- Comparación predicción intra-modelo: cómo varía predicción bajo cambios de plan
- Auditoría: artefactos exportados en JSON y CSV para inspección manual

6.5 Implementación del ejecutor híbrido en PyTorch (Fase 3)
- Cargador de plan: lee `ilp_assignment.csv` y construye política de colocación de módulos
- Colocación dinámica de módulos: `.to(device)` basado en plan tras primera pasada
- Gestión de transferencias: hooks para registrar tensores que cruzan dispositivos
- Trazabilidad: logging de activaciones transferidas, momento, tamaño, tiempo medido

6.6 Baselines reales y comparativos experimentales (Fase 3)
- Baseline all_cpu: todos los módulos en CPU
- Baseline all_gpu: todos los módulos en GPU
- Baseline greedy: asignación greedy implementada en runtime
- Plan ILP: asignación según solución optimizada obtenida en el pipeline ILP

6.7 Predicción frente a observación: contraste operacional (Fase 3)
- Métrica 1: Latencia predicha (simulador) vs latencia observada (executor)
- Métrica 2: Memoria predicha vs memoria high water mark observada
- Métrica 3: Número de cortes predichos vs cortes observados realmente
- Análisis: causas de divergencia (solapamiento, serialización, overhead)

6.8 Exactitud final del modelo y estabilidad de convergencia (Fase 3-5)
- Métrica: accuracy final, loss final, o métrica específica del dominio (e.g., AUC)
- Verificación: no hay degradación de exactitud bajo ejecución híbrida frente a all_gpu
- Estabilidad: convergencia suave sin oscilaciones anómalas por cambio de dispositivo

6.9 Extensiones con persistencia de activaciones y scheduling asíncrono (Fase 4)
- Nuevo baselines: all_cpu_with_remat, all_gpu_with_checkpoint
- Comparación: modelo base vs modelo con persistencia; trade-offs latencia vs memoria
- Auditoría: verificación de que rematprincipal no causa regresión de accuracy

6.10 Análisis integrado de resultados y limites experimentales (Fase 5)
- Síntesis: ¿el ILP predice mejora de latencia? ¿a qué costo de memoria?
- Límites observados: bajo qué configuraciones es válido el modelo
- Threats to validity: confusores potenciales, limitaciones de hardware, generalización
- Recomendaciones: cuándo aplicar la técnica, cuándo preferir all_gpu

---

## Capítulo 7: Conclusiones, Límites y Trabajo Futuro

**Dependencia de fases**: Fase 5 (redacción final)

Cierra la narrativa doctoral con honestidad metodológica.

**Secciones**:

7.1 Síntesis de resultados frente a la hipótesis
- Hipótesis: "existe asignación heterogénea óptima que mejora latencia sin degradar exactitud"
- Resultado esperado de la ruta completa: hipótesis totalmente contrastada con evidencia de simulación y ejecución física
- Evidencia: tablas, figuras, artefactos que sustentan la confirmación

7.2 Contribuciones metodológicas y de sistemas
- **Contribución 1**: Profiler robusto con agregación estadística para coeficientes de optimización
- **Contribución 2**: Formulación ILP extendida con validación comparativa y ablaciones
- **Contribución 3**: Runtime híbrido verificado experimentalmente
- **Contribución 4**: Extensión del modelo con persistencia de activaciones y scheduling asíncrono

7.3 Límites del modelo y del protocolo experimental
- Límite 1: Modelo base (asignación única) vs extendido (forward/backward separado)
- Límite 2: Rango de arquitecturas validadas (simple_mlp, resnet50 vs modelos muy grandes)
- Límite 3: Hardware específico (single GPU + CPU de referencia)
- Límite 4: Precisiones consideradas (FP32, FP16, BF16)

7.4 Líneas de trabajo futuro
- Distribución multi-GPU y escalamiento a modelos de mayor tamaño
- Integración con orquestación de clúster y scheduling multinodo
- Automatización de perfilado con modelos predictivos

---

## Resumen de Alineación Fase-Capítulo

| Capítulo | Dependencia de Fase | Estado de Completitud | Requisitos Textuales |
|----------|-------------------|----------------------|----------------------|
| 1 | Fase 0 | Tras Fase 0 | Decisiones de alcance |
| 2 | Ninguna | Inmediato | Revisión estándar de estado del arte |
| 3 | Ninguna | Inmediato | Descripción de arquitectura existente |
| 4 | Ninguna | Hoy 85% | Minor: ampliar 4.8 tras smoke test |
| 5 | Fase 0 + Fase 1 | Tras Fase 1 | Base + validación comparativa + ablaciones |
| 6 | Fase 2 + Fase 3 + Fase 5 | Tras Fase 5 | Simulador + ejecutor + validación final |
| 7 | Fase 5 | Tras Fase 5 | Síntesis de resultados y conclusiones |

---

## Ruta Doctoral Fijada

- **Alcance**: Ejecución completa de Fases 0-5 sin bifurcaciones de alcance.
- **Capítulos cerrados**: 1-7 con integración explícita de ILP extendido, simulador, runtime, persistencia y scheduling asíncrono.
- **Narración**: Sistema completo de partición heterogénea con validación operacional y evidencia experimental integral.
- **Riesgo principal**: exigencia de plazos por el carácter secuencial de Fases 2-5; mitigable con planificación cerrada de campañas.

---

## Mapeo de Gaps del MAPA a Capítulos

A modo de auditoría, la siguiente tabla conecta cada gap identificado en [MAPA_PROYECTO_VS_TESIS_ES.md](MAPA_PROYECTO_VS_TESIS_ES.md) con el capítulo y sección donde se cierra:

| Gap MAPA | Descripción | Capítulo | Sección | Fase |
|----------|------------|---------|---------|------|
| 6.1 | Sin ejecutor hibrido real | 6 | 6.5-6.6 | 3 |
| 6.2 | Sin forward/backward separado | 1, 5 | 1.6, 5.12 | 0 + 4 |
| 6.3 | Sin rematerializacion ILP | 5 | 5.12 | 4 |
| 6.4 | Sin streaming/prefetching | 1, 5 | 1.6, 5.12 | 0 + 4 |
| 6.5 | Sin simulador | 6 | 6.2-6.4 | 2 |
| 6.6 | Sin validacion fisica | 6 | 6.7-6.10 | 3 + 5 |
| 6.7 | Sin greedy baseline | 5 | 5.9 | 1 |
| 6.8 | Sin ablaciones | 5 | 5.11 | 1 |
| 6.9 | Sin exactitud final | 6 | 6.8 | 3 |
