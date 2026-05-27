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
La logica de [scripts/launch_grid5k.sh](scripts/launch_grid5k.sh) es:

- Si `CAMPAIGN_PROFILE=doctoral_full`:
  - ejecuta una corrida por cada valor de `FULL_SEEDS_CSV`
  - por defecto: `FULL_SEEDS_CSV=42,43,44`
  - repeticion por semilla controlada por `FULL_REPEATS_PER_SEED` (default 1)

- Si `CAMPAIGN_PROFILE` es distinto de doctoral_full (quick_smoke o doctoral_minimal):
  - ejecuta una sola semilla `SINGLE_SEED` (default 42)
  - repeticion controlada por `NON_FULL_REPEATS` (default 1)

## 4. Ejecucion recomendada

### 4.1 Verificacion rapida (quick_smoke)
Comando:

```bash
CAMPAIGN_PROFILE=quick_smoke \
SINGLE_SEED=42 \
NON_FULL_REPEATS=1 \
oarsub -S ./scripts/run_thesis.sh
```

Uso: validar que despliegue, entorno, rutas, permisos, logs y artefactos funcionan.

### 4.2 Verificacion completa acotada (doctoral_minimal)
Comando:

```bash
CAMPAIGN_PROFILE=doctoral_minimal \
SINGLE_SEED=42 \
NON_FULL_REPEATS=1 \
oarsub -S ./scripts/run_thesis.sh
```

Uso: validar pipeline completo en escala intermedia antes de produccion final.

### 4.3 Campana final de tesis (doctoral_full + semillas)
Comando:

```bash
CAMPAIGN_PROFILE=doctoral_full \
FULL_SEEDS_CSV=42,43,44,45,46 \
FULL_REPEATS_PER_SEED=1 \
oarsub -S ./scripts/run_thesis.sh
```

Uso: producir datos finales con estimacion de variabilidad y robustez.

## 5. Variables mas importantes
Variables de [scripts/run_thesis.sh](scripts/run_thesis.sh) y [scripts/launch_grid5k.sh](scripts/launch_grid5k.sh):

- `CAMPAIGN_PROFILE`: `quick_smoke` | `doctoral_minimal` | `doctoral_full`
- `CONDA_ENV_NAME`: entorno conda a activar (default: `prism_env`)
- `RUN_HYBRID`: habilita etapa de ejecucion hibrida
- `FULL_SEEDS_CSV`: lista CSV de semillas para doctoral_full
- `SINGLE_SEED`: semilla unica para quick_smoke/doctoral_minimal
- `FULL_REPEATS_PER_SEED`: replicas por semilla en doctoral_full
- `NON_FULL_REPEATS`: replicas en quick_smoke/doctoral_minimal
- `PROJECT_ROOT`: ruta del repo en nodo remoto (default `/root/PRISM`)
- `LOCAL_PROJECT_ROOT`: ruta local del repositorio a sincronizar (default: raiz del repo deducida desde `scripts/run_thesis.sh`)
- `SYNC_PROJECT_BEFORE_RUN`: sincroniza el arbol del proyecto al nodo remoto antes de ejecutar (default: `true`)
- `SYNC_EXCLUDES`: lista CSV de rutas excluidas durante rsync (default: `/.git,/.venv,/logs,/reports,/data,/datasets,/books,/paper_thesis,/papers`)
  - Nota: usar rutas ancladas (con `/` al inicio) evita excluir por error `src/data`.
- `KADEPLOY_FILE`: manifest de despliegue kadeploy
  - Puede ser ruta absoluta o relativa al repositorio local.
  - Si no existe, `scripts/run_thesis.sh` aborta antes de invocar `kadeploy3`.

## 6. Salidas y trazabilidad
- Logs OAR: `thesis_job.<jobid>.output` y `thesis_job.<jobid>.error`
- Logs de launcher: carpeta `logs/` del proyecto remoto
- Resultados: subdirectorios separados por perfil y semilla para evitar sobreescritura

## 7. Diagnostico rapido de fallos
1. Si falla kadeploy: revisar validez de `KADEPLOY_FILE` y acceso al sitio/cola.
2. Si falla conda: verificar que `prism_env` exista en la imagen desplegada.
3. Si aparecen errores `ModuleNotFoundError: No module named src.data`: verificar que no se haya desactivado `SYNC_PROJECT_BEFORE_RUN` y que `LOCAL_PROJECT_ROOT` apunte al repo correcto.
  - El wrapper ahora valida antes y despues del rsync que exista `src/data/__init__.py`.
4. Si faltan artefactos: revisar logs en `logs/` y confirmar que no se uso `DRY_RUN=true`.

## 8. Flujo operativo sugerido para campana final
1. Ejecutar quick_smoke (1 semilla).
2. Ejecutar doctoral_minimal (1 semilla).
3. Ejecutar doctoral_full con 3-5 semillas.
4. Consolidar resultados y generar reportes finales.
