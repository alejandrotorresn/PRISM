# Testing & Validation Map

## Objetivo
Este documento deja un mapa operativo del proyecto para ejecutar pruebas y validaciones de forma consistente (local y CI), con orden recomendado, criterios de éxito y puntos de riesgo conocidos.

## 1) Mapa del proyecto (vista de validación)

### Núcleo de ejecución
- `src/profiler.py`: CLI principal, parseo de argumentos, selección de precisión y orquestación del profiling.
- `src/runner/training_profiler.py`: bucle de medición/entrenamiento, control de timeouts y escritura de artefactos.
- `src/models/factory.py`: factoría de modelos soportados (`resnet50`, `resnet152`, `vit_b16`, `bert_base`, `gpt2_small`, `simple_mlp`).
- `src/core/precision_policy.py`: política FP32/FP16/BF16, detección ISA y preflight de FP16 en CPU.
- `src/core/io_artifacts.py`: persistencia de métricas y metadatos.
- `src/core/constants.py`: constantes compartidas.

### Pruebas y validación
- `tests/test_precision_policy_unit.py`: suite unitaria (pytest) de política de precisión y helpers críticos.
- `tests/test_timeout_validation.py`: validación enfocada en comportamiento de timeout/preflight.
- `validation/run_unit_tests.sh`: comando unificado para ejecutar pytest en entorno del proyecto.
- `validation/validate_code.py`: validación estructural/semántica del mecanismo de timeout en dos fases.
- `validation/validate_zombie_fix.py`: validación de flags/flujo para corrección de hilos zombie.
- `validation/comprehensive_check.sh`: auditoría grep de arquitectura modular y reglas críticas.
- `validation/validate_all_models.py`: validación amplia por modelo (alineada a imports modulares actuales).

### Orquestación de experimentos
- `scripts/run_experiments.sh`: barrido de combinaciones (modelo, optimizador, precisión, batch, hilos), con soporte `--skip_cpu`.

### Configuración de entorno
- `config/requirements.txt`: dependencias pip del proyecto.
- `config/environment.yml`: entorno conda opcional.

## 2) Estado de preparación actual

### Resultado de smoke de validación (entorno `.venv`)
- Unit tests: **12/12 passed** (`validation/run_unit_tests.sh`).
- Validación estructural: **13/13 passed** (`validation/validate_code.py`).
- Validación zombie-fix: **5/5 passed** (`validation/validate_zombie_fix.py`).

Advertencias observadas (no bloqueantes):
- `pynvml` deprecado (recomendado migrar a `nvidia-ml-py`).
- warning de `pymongo` por dependencia opcional de `MongoOutput`.

## 3) Runbook recomendado (rápido → profundo)

### Paso A: Unit tests (rápido)
```bash
bash validation/run_unit_tests.sh
```
Criterio de éxito: `N passed` y exit code 0.

### Paso B: Validación estructural de timeout
```bash
.venv/bin/python validation/validate_code.py
```
Criterio de éxito: `SUMMARY: 13/13 checks passed`.

### Paso C: Validación de zombie-thread fix
```bash
.venv/bin/python validation/validate_zombie_fix.py
```
Criterio de éxito: `5/5 checks passed`.

### Paso D: Auditoría integral modular
```bash
bash validation/comprehensive_check.sh
```
Criterio de éxito: `PASSED: 60` y `FAILED: 0`.

### Paso E: Smoke funcional del profiler
```bash
.venv/bin/python src/profiler.py --model simple_mlp --no_gpu --precision fp32 --warmup 1 --measure 1 --output_dir data/test-ci
```
Criterio de éxito: artefactos generados:
- `data/test-ci/simple_mlp_metrics.csv`
- `data/test-ci/simple_mlp_meta.json`

## 4) Matriz de cobertura (qué valida cada capa)

- **Unitaria (`pytest`)**: reglas de decisión y helpers puros de precisión/preflight.
- **Estructural (`validate_code.py`)**: presencia de flujo de timeout en dos fases y campos de metadata.
- **Flujo/flags (`validate_zombie_fix.py`)**: correcto uso de `--skip_cpu`, `--num_threads`, ubicación del preflight.
- **Arquitectura (`comprehensive_check.sh`)**: consistencia de organización modular y referencias obligatorias.
- **Integración runtime (smoke profiler)**: ejecución real end-to-end y escritura de artefactos.

## 5) Riesgos y ajustes recomendados

1. **Warnings de entorno no bloqueantes**
   - `pynvml` aparece como deprecado (recomendado migrar a `nvidia-ml-py`).
   - Warning de `pymongo` por dependencia opcional de `MongoOutput`.

2. **Costo de pruebas de modelos grandes**
   - La validación amplia por modelo puede requerir descargas y tiempo de cómputo significativo para `bert_base`/`gpt2_small`.
   - Recomendación: mantener smoke rápido en CI y ejecutar validación amplia en ventanas controladas.

## 6) Checklist operativo para validar un cambio

1. Activar entorno (`.venv` o conda).
2. Ejecutar `bash validation/run_unit_tests.sh`.
3. Ejecutar validadores Python (`validate_code.py`, `validate_zombie_fix.py`).
4. Ejecutar `bash validation/comprehensive_check.sh`.
5. Ejecutar smoke runtime (`simple_mlp`, CPU-only).
6. Confirmar artefactos y revisar warnings no bloqueantes.

Con estos pasos, el proyecto queda listo para pruebas repetibles y validación técnica consistente.
