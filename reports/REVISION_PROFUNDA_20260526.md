# Revision profunda del proyecto (2026-05-26)

## Estado de la limpieza solicitada
- Resultado: no existen archivos `desktop.ini` en el repositorio.
- Verificacion: busqueda global por patron `**/desktop.ini` sin resultados.

## Perfil completo por archivo
- Perfil archivo-por-archivo (ruta, tamano en bytes, lineas para texto, MIME, hash SHA-256):
  - `reports/file_profile_full_20260526_183741.tsv`

## Hallazgos principales (ordenados por severidad)

### Alta
1. Ruta hardcodeada dependiente de entorno en `scripts/launch_grid5k.sh` (carga de conda fija bajo `/root/miniconda3`).
2. Uso de rutas relativas fragiles en `src/profiler.py` para crear `data/<host>` en funcion del CWD.
3. Duplicacion amplia de logica CLI/utilidades en scripts ILP bajo `validation/` (parseo de listas, carga de perfiles y datos).
4. Captura generica de excepciones sin suficiente contexto de logging en varios modulos (`src/ilp/solve.py`, `src/runtime/hybrid_executor.py`, `validation/generate_ilp_report_assets.py`).

### Media
1. Documentacion de estructura parcialmente desactualizada respecto a modulos ILP/Phase 4 recientes.
2. Heuristicos por defecto en `src/ilp/data_loader.py` con justificacion metodologica insuficiente en documentacion tecnica.
3. Parametrizacion de agregacion de hardware no totalmente consolidada entre scripts de validacion.

### Baja
1. Uso repetido de patrones de carga y parseo en `validation/` sin modulo compartido.
2. Algunos `pass` sin comentario contextual en runtime/core.

## Evaluacion de arquitectura
- El pipeline general esta bien estructurado por fases: profiling, agregacion, ILP, activacion/deuda, ejecucion hibrida y generacion de reportes.
- Fortalezas: separacion de responsabilidades entre `src/core`, `src/ilp`, `src/runtime`; base de tests extensa en `tests/`.
- Debilidades: dispersion de entradas CLI y reutilizacion limitada entre scripts de validacion.

## Riesgos de reproducibilidad
1. Dependencia del directorio de ejecucion para rutas relativas (riesgo alto).
2. Suposiciones de entorno no portables para activacion de conda en shell scripts (riesgo medio-alto).
3. Falta de pruebas automaticas especificas de determinismo cross-run/cross-host (riesgo medio).

## Estado de pruebas y calidad
- Diagnostico de errores de editor/analisis sobre `src/`, `tests/`, `scripts/`: sin errores reportados actualmente.
- Suite de tests amplia y segmentada por componentes (ILP, runtime, precision policy, phase4, etc.).

## Acciones priorizadas recomendadas
1. Extraer utilidades compartidas de CLI/parseo a un modulo comun (`validation/common.py` o `src/core/cli_utils.py`).
2. Normalizar todas las rutas I/O para que sean robustas al CWD (base project root explicita).
3. Eliminar hardcodes de conda/entorno en scripts shell y usar deteccion o variables de entorno.
4. Endurecer manejo de errores: incluir excepcion, contexto y traza donde aplique.
5. Actualizar `docs/PROJECT_STRUCTURE.md` para reflejar el estado actual de `src/ilp` y fases recientes.
6. Incorporar tests de reproducibilidad (mismo seed -> mismos artefactos/metricas).

## Archivos clave inspeccionados en la revision
- `src/profiler.py`
- `src/runner/training_profiler.py`
- `src/core/stats_aggregator.py`
- `src/ilp/data_loader.py`
- `src/ilp/solve.py`
- `src/runtime/hybrid_executor.py`
- `validation/sweep_ilp_pareto.py`
- `validation/run_ilp_partition.py`
- `validation/run_hybrid_execution.py`
- `tests/test_hybrid_executor.py`
- `tests/test_phase4_activation.py`
- `docs/PROJECT_STRUCTURE.md`
