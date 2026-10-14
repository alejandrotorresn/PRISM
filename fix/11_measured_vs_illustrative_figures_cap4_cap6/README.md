# Datos hardcodeados vs. medidos en figuras de los Capítulos 4 y 6

Estado: **Capítulo 6 🔶 pendiente, aplazado a propósito** -- no se toca hasta tener datos reales de
pacca y grid5k. **Capítulo 4 ✅ resuelto** para `temporal_decomposition.png`/`congestion_knee.png`
(ver sección propia más abajo).

**Esto es exactamente el ítem #2 del jurado** (`PLAN_ACCION_JURADO.md`, pedido por García #3 y
Schnorr #2, 🔴): "Reconciliar las cifras del Cap. 6 contra los datos crudos... y marcar qué figuras
son medidas vs. ilustrativas". Esta investigación identifica con precisión CUÁLES figuras y
CUÁLES variables específicas, en ambos capítulos.

## Capítulo 6 -- cuatro scripts de figuras con datos fabricados (no leídos de ningún CSV/JSON real)

| script | severidad | variables hardcodeadas |
|---|---|---|
| `validation/generate_vertex_distribution_oom_plot.py` | 100% fabricado | `gpu_mem`, `cpu_mem`, `gpu_verts`, `greedy_time`, `prism_time`, `ilp_s`/`prof_s`, `vram_ratios`, `prism_vit`, `greedy_vit`, `prism_res`, `greedy_res` |
| `validation/generate_energy_gpu_prism_comparison.py` | 100% fabricado | `all_gpu_j`, `greedy_j`, `prism_j` (incluye el titular "-62.6% energía vs Greedy") |
| `validation/generate_overhead_rescue_plot.py` | 100% fabricado | `vertices`, `prof_time_s`, `ilp_time_s`, `total_campaign_s`, `latency_vit_b16`, `latency_resnet152` |
| `validation/generate_chapter6_energy_and_greedy_plots.py` | Mixto | `generate_overhead_breakdown_plot` (líneas 183,186: `ilp_solver_s`, `step_time_ms`) es 100% literal; las otras dos funciones del mismo archivo SÍ leen `reports/chuc-4/doctoral_full/csv/ilp_best_per_model.csv` -- pero ese CSV **ya no existe en el repo** |

Verificado (ver `fix/10`... no, ver investigación de esta sesión, no repetido aquí):
- Los PORCENTAJES de capas GPU/CPU citados (72% vit_b16, 54% resnet152) son plausibles y
  consistentes con `greedy_layers_gpu`/`greedy_layers_cpu` reales encontrados en
  `data/legacy/{chuc-4,kinovis-2,paccaA100...}/.../vit_b16_pareto_sweep.csv` (batch_512, 32GB VRAM) --
  varias corridas reales dan 70-75%.
- Los TIEMPOS específicos (172.4ms, 520.2ms, t_ILP, energía en Julios) **no coinciden con ninguna
  columna `sim_time_ms` de esos mismos CSV** (esas dan 900-2900ms) **ni aparecen en ningún otro
  CSV/JSON del repositorio**.
- `_greedy_bits` (`validation/sweep_ilp_pareto.py`, la función que de verdad decide la partición
  "greedy" en toda campaña real, incluida la de hoy) nunca ha hecho "apilar operadores
  secuencialmente hasta llenar VRAM" en ningún commit del historial de este repositorio -- confirmado
  con `git log -L` sobre la función. El texto del Capítulo 6 describe la heurística en esos términos;
  el algoritmo real es una asignación local por costo + reparación por relación costo/beneficio.

**Problema aparte, de reproducibilidad (no de fabricación):** 4 scripts adicionales SÍ leen datos
reales correctamente (`regenerate_chapter6_global_plots_no_mlp.py`,
`scripts/generate_chapter6_prism_vs_allgpu_figures.py`, `scripts/generate_pareto_curve_ch5.py`, y la
mitad real de `generate_chapter6_energy_and_greedy_plots.py`), pero sus CSV fuente bajo
`reports/{chuc-4,kinovis-2,paccaA100...}/` **ya no existen en ningún lado del repositorio** --
cualquier figura ya generada con ellos no se puede regenerar ni verificar hoy.

El resto de scripts de figuras auditados (`generate_ilp_report_assets.py`,
`generate_thesis_figures.py`, `plot_all_energy_tradeoff.py`, `plot_custom_user_figures.py`,
`generate_multiserver_summary.py`) leen datos reales correctamente -- sin fabricación.

**`plot_all_oom_vs_ilp.py` -- RESUELTO.** Tenía los límites de VRAM/DRAM fijos a `8192`/`32768`
(8GB/32GB), coincidencia con esta máquina local pero incorrectos si se corre sobre datos de
CHUC-4/KINOVIS-2 (32GB VRAM). Corregido: `src/core/system.py::get_hardware_metadata` ganó un campo
nuevo `cpu_ram_total_mb` (análogo al `gpu_vram_total_mb` ya existente, vía `psutil.virtual_memory()`),
así que cada `*_meta.json` ahora registra la VRAM/RAM reales de la máquina que lo generó. El script
lee esos dos campos reales por config (`<config_dir>/run_*/<model>_meta.json`) en vez de asumir un
valor fijo, y si faltan (datos de antes de este fix), **omite la línea en vez de mostrar un número
incorrecto**, con una advertencia explícita impresa. Verificado con datos reales de hoy: la línea de
VRAM ahora muestra "8.0GB" leído de `gpu_vram_total_mb=8188.0`, no un literal coincidente.

**`generate_advanced_thesis_plots.py` -- constante sin resolver, pero de bajo riesgo real.** La
figura roofline (`roofline_model.png`) tiene `vram_bandwidth_gb_s=760.0`/`dram_bandwidth_gb_s=50.0`
fijos, sin mecanismo de override (a diferencia de `peak_tflops_gpu/cpu`, que sí lee el `meta.json`
real correctamente -- confirmado con datos reales: 9.14/0.61 TFLOPS). Verificado que esta figura
**no está referenciada en ningún `.tex` de la tesis actualmente** -- la figura "roofline" que sí
aparece (Capítulo 2) es un diagrama TikZ dibujado a mano, explícitamente rotulado "conceptual",
sin relación con este script. Queda sin tocar por ahora al no afectar ninguna figura real del
documento; revisar si se decide usar esta figura en el futuro.

## Capítulo 4 -- tres figuras conceptuales (`scripts/generate_methodological_plots.py`), con un gap real entre el código y el texto de la tesis

El propio código SÍ documenta estos datos como ilustrativos/teóricos en sus comentarios -- pero
verificado contra el `.tex` real, **el texto de la tesis no siempre lo dice**:

1. **`temporal_decomposition.png`** (Fig.~\ref{fig:ch4_temporal_representation}, línea 190 de
   `chapter04_empirical_profiling_es.tex`): el código (`plot_temporal_decomposition`) comenta
   explícitamente "Illustrative data representing different layer types" y usa
   `t_kernel`/`t_dispatch` literales, sin ningún tipo de capa real. **El texto de la tesis (línea 185)
   dice "Como DEMUESTRA la Figura..."** y describe el fenómeno como si fuera una medición real, sin
   ninguna palabra "ilustrativo"/"conceptual" en el párrafo ni en el pie de figura. **Gap confirmado.**
2. **`congestion_knee.png`** (Fig.~\ref{fig:ch4_congestion_knee}, línea 345): el pie de figura dice
   literalmente **"el punto de inflexión EMPÍRICO"**. El código solo usa valores reales si recibe un
   `*_meta.json` con `transfer_alpha_h2d`/`transfer_beta_h2d` -- si no, cae a
   `alpha=0.05, beta_nom=12.0, beta_cong=6.0, knee=128.0` hardcodeados (con una advertencia impresa en
   consola, pero nada que llegue al PDF). **Busqué en todo el repositorio y no existe NINGÚN
   `*_meta.json` con esos campos** -- es decir, esta figura, tal como el repositorio existe hoy, solo
   pudo haberse generado con los valores de respaldo, nunca con datos reales, contradiciendo la
   palabra "empírico" de su propio pie de figura. **Gap confirmado, más serio que el anterior porque
   la palabra "empírico" está escrita explícitamente.**
3. **`overlap_tension.png`** (Fig.~\ref{fig:ch4_overlap_tension}, línea 361): el pie de figura la
   describe como una "superficie PARAMÉTRICA" del modelo ($\tau_{u,v}$ en función de $\omega$ y
   $p_u$) -- enmarcada como la visualización de una fórmula, no como una medición. El código coincide
   (comentario explícito "Teórico", constantes etiquetadas como "arbitrario"/"factor de
   penalización"). **Más bajo riesgo** -- el marco conceptual coincide razonablemente entre código y
   texto, aunque ninguno de los dos usa la palabra "ilustrativo" explícitamente.

## Decisión (del usuario)

No se necesitan los scripts del Capítulo 6 tal como están -- se necesita lo que HACEN: tomar los
datos de las ejecuciones finales reales (pacca/grid5k) y generar las gráficas correspondientes. Se
deja pendiente hasta tener esos datos; en ese momento se reescriben para leer de CSV/JSON reales en
vez de literales hardcodeados.

## Capítulo 4 -- RESUELTO para `temporal_decomposition.png` y `congestion_knee.png`

Estado: ✅ `scripts/generate_methodological_plots.py` reescrito para usar datos 100% reales, sin
ninguna medición nueva -- la instrumentación ya existía:

- `plot_temporal_decomposition`: reconstruye $t^{\mathrm{kernel}}$/$t^{\mathrm{dispatch}}$ reales
  por capa desde `*_metrics_stats.csv` (`gpu_fwd_time_ms_mean` × `dispatch_overhead_ratio_mean`),
  promediando TODAS las capas reales de cada tipo (Conv2d/BatchNorm2d/ReLU/Linear) -- no una sola
  capa elegida a mano, para no sesgar la figura. Verificado con datos reales de resnet50: ReLU
  muestra más sobrecarga de despacho relativa (0.084ms) que Conv2d/BatchNorm (~0.000ms), consistente
  con la narrativa teórica del capítulo.
- `plot_congestion_knee`: corregida la ruta por defecto (le faltaba el segmento `custom/`, por lo
  que nunca encontraba el `*_meta.json` real) y **eliminado un hack que forzaba artificialmente**
  `beta_cong = beta_nom * 0.6` cuando la calibración salía "demasiado plana" -- exactamente el tipo
  de distorsión que esta investigación busca eliminar. Con datos reales de resnet50 hoy
  (`transfer_calibration_source="measured"`), la rodilla de congestión casi no es visible
  (`beta_nom=12.44` vs `beta_cong=12.43` GB/s, prácticamente idénticos) -- resultado honesto, no un
  bug: el efecto de congestión medido en esta GPU/transferencias es mínimo. Si se quiere una rodilla
  más pronunciada en la figura final, hace falta un modelo/batch con transferencias más grandes, no
  forzar el número.
- Conectado a `scripts/run_thesis_mode.sh` (STEP 5, junto a los demás generadores de figuras) --
  se produce automáticamente con cada campaña, sin flags manuales adicionales.
- `plot_overlap_tension` se deja sin cambios a propósito: visualiza la FORMA de un modelo paramétrico
  ($\tau$ en función de $\omega$ y $p_u$), no una cantidad medida directamente -- el código y el pie
  de figura ya coinciden en enmarcarlo así.

Pendiente, menor: decidir si el pie de figura de `congestion_knee.png` en el `.tex` debe aclarar que
la rodilla es casi plana para esta plataforma específica (dato real, no ilustrativo, pero con una
forma distinta a la versión anterior).
