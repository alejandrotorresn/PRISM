# Cruce: Plan de mejora PRISM vs. ítems del jurado

Cruza `PLAN_MEJORA_PRISM.md` (fases 0–8) con los 42 ítems de `PLAN_ACCION_JURADO.md`.
Leyenda de cobertura: **Cubre** = el plan lo resuelve con código/datos; **Habilita** = el plan produce los datos, pero falta redactar en la tesis; **Parcial** = el plan ayuda pero falta trabajo adicional; **Fuera** = es de redacción/bibliografía, el plan de código no lo toca.

| # | Ítem del jurado (resumen) | Fase/tarea del plan | Cobertura | Qué falta |
|---|---|---|---|---|
| 1 | Accuracy/convergencia PRISM vs All-GPU (FP32/BF16) | 5.4, 3.6, 8.1 | Habilita | BF16 no está en el plan: añadir como parámetro de dtype en 5.4 |
| 2 | Reconciliar cifras del Cap. 6 con datos crudos | 6.3, 8.1, 8.3 | Habilita | Revisar texto del capítulo con las cifras regeneradas |
| 3 | Protocolo experimental (VRAM, baseline, warm-up, repeticiones) | 3.4, 4.5, 5.3, `effective_config.json` (0.2) | Habilita | Redactar el protocolo con los parámetros guardados |
| 4 | Línea base externa (checkpointing, `save_on_cpu`, ZeRO/Checkmate/POET) | 5.5 | Parcial | 5.5 cubre checkpointing nativo, `save_on_cpu` y offload tipo ZeRO. Checkmate/POET quedan como comparación teórica |
| 5 / 31 | R²=0.000, Spearman/Kendall, R² por 5 familias | 2.1, 2.3, 5.1, 5.2 | Parcial | El plan mide error sim vs. real, pero **no** incluye el cálculo de Spearman/Kendall por modelo/familia. Añadir tarea 5.6 |
| 6 | Energía ≈ tiempo, limitación de Pareto | 4.2, 4.3, 2.1 | Cubre | Energía real GPU+CPU; redactar la limitación |
| 7 | Δt/Δe y brechas de optimalidad | 2.4, 5.1 | Cubre | El solver reporta `status` y gap reales; falta agregar Δt/Δe al resumen |
| 8 | Tratabilidad polinomial / convexidad | — | Fuera | Corrección de redacción |
| 9 | t-test/Cohen contra All-GPU y Greedy | 5.3, 5.4 | Cubre | Incluir comparaciones contra All-GPU y Greedy en 5.3 |
| 10 | Throughput y comparación con grad-accum | 5.5, 8.2(6), 8.3 | Cubre | Reportar muestras/s en el informe por paso |
| 11 | Intervalos de confianza | 5.3 | Cubre | — |
| 12 | Costo = suma de latencias vs. makespan | 2.1 | Cubre | Decidir y documentar: makespan o suma serial con su límite de validez |
| 13 | Coarsening conserva solo la activación del último vértice | 1.3, 2.3 | Parcial | El plan no revisa el coarsening. Añadir verificación en 2.3 |
| 14 | Copia de pesos actualizados cuando fwd/bwd están en dispositivos distintos | 2.5, 3.1 | Parcial | Entra al coste del dual placement; confirmar en 3.1.1 si el runtime la mide |
| 15 | Restricción de congestión mezcla unidades | 2.1, 4.4 | Parcial | 4.4 unifica MB/ms, pero el modelo con $S_{knee}$ requiere revisión explícita en 2.1 |
| 16 | $r_v$: ¿variable del ILP o heurística post-ILP? | 2.5 | Habilita | Decidir y documentar |
| 17 | Posicionar frente a Beaumont, POET, MONeT, Capuchin, ZeRO-Infinity | 5.5 | Fuera/Parcial | Solo redacción |
| 18, 36 | Bibliografía fabricada / auditar `references.bib` | — | Fuera | No es código. Requiere auditoría manual |
| 19 | Referencias no citadas | — | Fuera | — |
| 20, 21, 22 | Tono, notación, erratas | — | Fuera | — |
| 23 | Enviar paper | — | Fuera | Post-defensa |
| 24 | Diagrama de 5 fases | — | Fuera | — |
| 25 | Alcance 1 CPU + 1 GPU | — | Fuera | Justificación en el texto |
| 26 | Unified Memory | 5.5 (opcional) | Parcial | Se puede añadir `cudaMallocManaged` como baseline opcional |
| 27 | Validar γ=2.0 backward/forward | 4.1 | Cubre | El backward se mide por segmento; γ sale de datos |
| 28 | Validar imputación Roofline | 4.1, 2.6 | Parcial | Falta un test que compare Roofline vs. medición en capas que sí caben |
| 29 | % de instancias con gap 1 % vs. timeout | 2.4 | Cubre | Reportar el porcentaje en el resumen de la campaña |
| 30 | Crecimiento del ILP con recomputación | 2.5 | Parcial | Falta medir tamaño/tiempo del ILP conjunto Fase 4 |
| 32 | Multi-GPU, NVLink, escalabilidad | — | Fuera | Trabajo futuro; la escalabilidad del ILP más allá de 364 vértices puede medirse con 2.4 |
| 33 | Telemetric Feedback Loop vs. AOT | — | Fuera | Redacción |
| 34 | Generalización fuera de DL | — | Fuera | Redacción |
| 35 | distilgpt2 B=512 @ 20 GB, pasajes contradictorios | 6.3, 8.1, 8.3 | Habilita | Se resuelve al regenerar con datos medidos y releer ambos pasajes |
| 37 | Bugs de la verificación dual (ya [x]) | 3.1 | Cubre/amplía | El plan **encuentra más**: Dropout y BatchNorm en `call_module` (H5b/H5c) |
| 38 | Flujo dual decoder-only (ya [x]) | 3.1, 3.2 | Cubre | Medir el sobrecoste (H14) |
| 39, 41 | OOM real en régimen B (ya [x]) | 2.3, 3.4, 8.2(2,4) | Cubre | Criterios globales 8.2.2 y 8.2.4 los vuelven a verificar |
| 40 | `torch.export` en GPT-2 (ya [x]) | 1.5 | Cubre/amplía | Extiende a bert_base (H11) |
| 42 | Fase 4 decoder-only (ya [x]) | 2.5, 3.2 | Cubre/amplía | Agrupación por dispositivo (H6) |

## Resumen

- **Cubre o habilita (código/datos):** 1, 2, 3, 6, 7, 9–12, 16, 27, 29, 35, 37–42.
- **Parcial (faltan tareas):** 4, 5/31, 13, 14, 15, 26, 28, 30.
- **Fuera del plan de código (redacción/bibliografía):** 8, 17–25, 32–34, 36.

## Tareas nuevas que surgen del cruce (a añadir al plan)

- **[5.6]** Spearman/Kendall por modelo y por familia (ítems 5, 31), sobre la predicción del ILP vs. el tiempo medido.
- **[2.8]** Revisión del coarsening (ítem 13) y de las unidades de la restricción de congestión (ítem 15), con tests.
- **[4.6]** Test de Roofline vs. medición en capas que caben (ítem 28).
- **[5.7]** Medir crecimiento del ILP al incluir recomputación como variable (ítem 30).
- **[5.4b]** Ejecutar el caso BF16 (ítem 1) como parámetro de dtype.
- **[5.1b]** Reportar Δt/Δe y el porcentaje de instancias con gap alcanzado frente a timeout (ítems 7, 29).

## Observación importante

Los ítems 37–42 figuran como `[x]` (resueltos), pero la campaña nueva sigue mostrando planes LLM todo-CPU (H1), atención ausente en vit_b16 (H2), 7–18× más lento que grad-accum en el régimen B y un dual placement casi inexistente (H18). Conviene **reabrir** esos ítems como `[~]` hasta pasar los criterios globales 8.2.
