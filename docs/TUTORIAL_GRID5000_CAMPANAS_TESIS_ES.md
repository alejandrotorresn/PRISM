# Tutorial de uso: campañas de tesis en Grid5000 (OAR + Kadeploy)

## 1. Objetivo
Este tutorial describe el uso de los scripts de lanzamiento para ejecutar campañas completas de tesis en Grid5000, con una politica de semillas orientada a eficiencia computacional:

- quick_smoke y doctoral_minimal: semilla unica (verificacion operativa)
- doctoral_full: barrido de semillas (robustez y reporte final)

Scripts principales:
- scripts/run_thesis.sh (wrapper OAR en frontend)
- scripts/launch_grid5k.sh (launcher en nodo desplegado)

## 2. Requisitos previos
1. Debe existir una reserva OAR valida y acceso al cluster objetivo.
2. Debe estar disponible el archivo de despliegue Kadeploy (por defecto `rocky9_profiling.yaml`).
3. El repositorio debe existir en el nodo desplegado en `/root/PRISM`.
4. El entorno conda `prism_env` debe existir en la imagen desplegada.
5. Debe existir el script local `scripts/launch_grid5k.sh` en la maquina desde donde se ejecuta `oarsub`.

## 3. Politica de semillas implementada
La interfaz canonica de ejecucion en frontend es por banderas de [scripts/run_thesis.sh](scripts/run_thesis.sh). La logica operacional equivalente en [scripts/launch_grid5k.sh](scripts/launch_grid5k.sh) es:

- Si `--profile doctoral_full`:
  - ejecuta una corrida por cada valor de `--full-seeds`
  - por defecto: `42,43,44`
  - repeticion por semilla controlada por `--full-repeats` (default 1)

- Si `--profile` es distinto de doctoral_full (`quick_smoke` o `doctoral_minimal`):
  - ejecuta una sola semilla `--single-seed` (default 42)
  - repeticion controlada por `--non-full-repeats` (default 1)

## 4. Ejecucion recomendada

### 4.1 Verificacion rapida (quick_smoke)
Comando:

```bash
oarsub -S "./scripts/run_thesis.sh --profile quick_smoke --single-seed 42 --non-full-repeats 1"
```

Uso: validar que despliegue, entorno, rutas, permisos, logs y artefactos funcionan.

### 4.2 Verificacion completa acotada (doctoral_minimal)
Comando:

```bash
oarsub -S "./scripts/run_thesis.sh --profile doctoral_minimal --single-seed 42 --non-full-repeats 1"
```

Uso: validar pipeline completo en escala intermedia antes de produccion final.

### 4.3 Campana final de tesis (doctoral_full + semillas)
Comando:

```bash
oarsub -S "./scripts/run_thesis.sh --profile doctoral_full --full-seeds 42,43,44,45,46 --full-repeats 1"
```

Uso: producir datos finales con estimacion de variabilidad y robustez.

## 5. Banderas canonicas (interfaz publica)
Usar siempre estas banderas en la invocacion OAR:

- `--profile`: `quick_smoke` | `doctoral_minimal` | `doctoral_full`
- `--single-seed`: semilla unica para quick_smoke/doctoral_minimal
- `--non-full-repeats`: replicas en quick_smoke/doctoral_minimal
- `--full-seeds`: lista CSV de semillas para doctoral_full
- `--full-repeats`: replicas por semilla en doctoral_full
- `--run-hybrid`: habilita o deshabilita etapa de ejecucion hibrida

Nota de estandarizacion: aunque el pipeline conserva variables de entorno por compatibilidad interna, la interfaz soportada para ejecucion de campanas en Grid5000 es por banderas para evitar ambiguedades de propagacion en OAR.

## 6. Overrides avanzados (solo operacion/integracion)
Variables de infraestructura de [scripts/run_thesis.sh](scripts/run_thesis.sh) y [scripts/launch_grid5k.sh](scripts/launch_grid5k.sh):

- `CONDA_ENV_NAME`: entorno conda a activar (default: `prism_env`)
- `PROJECT_ROOT`: ruta del repo en nodo remoto (default `/root/PRISM`)
- `LOCAL_PROJECT_ROOT`: ruta local del repositorio a sincronizar (default: raiz del repo deducida desde `scripts/run_thesis.sh`)
- `SYNC_PROJECT_BEFORE_RUN`: sincroniza el arbol del proyecto al nodo remoto antes de ejecutar (default: `true`)
- `SYNC_EXCLUDES`: lista CSV de rutas excluidas durante rsync (default: `/.git,/.venv,/logs,/reports,/data,/datasets,/books,/paper_thesis,/papers`)
  - Nota: usar rutas ancladas (con `/` al inicio) evita excluir por error `src/data`.
- `KADEPLOY_FILE`: manifest de despliegue kadeploy
  - Puede ser ruta absoluta o relativa al repositorio local.
  - Si no existe, `scripts/run_thesis.sh` aborta antes de invocar `kadeploy3`.

## 7. Salidas y trazabilidad
- Logs OAR: `thesis_job.<jobid>.output` y `thesis_job.<jobid>.error`
- Logs de launcher: carpeta `logs/` del proyecto remoto
- Resultados de datos (storage compartido) bajo ruta explicita por host y ejecucion:
  - `/root/PRISM/data/<host_tag>/thesis_runs/job_<oar_job_id>_<timestamp>/<profile>/seed_<N>/`
- En cada `seed_<N>` se genera `README_RUN_CONTENTS.txt` con el inventario de artefactos esperados y rutas de reportes.
- Reportes consolidados:
  - `/root/PRISM/reports/ilp_results/grid5k_<host_tag>_thesis_mode/<profile>/seed_<N>/`

## 8. Diagnostico rapido de fallos
1. Si falla kadeploy: revisar validez de `KADEPLOY_FILE` y acceso al sitio/cola.
2. Si falla conda: verificar que `prism_env` exista en la imagen desplegada.
3. Si aparecen errores `ModuleNotFoundError: No module named src.data`: verificar que no se haya desactivado `SYNC_PROJECT_BEFORE_RUN` y que `LOCAL_PROJECT_ROOT` apunte al repo correcto.
  - El wrapper ahora valida antes y despues del rsync que exista `src/data/__init__.py`.
4. Si faltan artefactos: revisar logs en `logs/` y confirmar que no se uso `DRY_RUN=true`.

## 9. Flujo operativo sugerido para campana final
1. Ejecutar quick_smoke (1 semilla).
2. Ejecutar doctoral_minimal (1 semilla).
3. Ejecutar doctoral_full con 3-5 semillas.
4. Consolidar resultados y generar reportes finales.
