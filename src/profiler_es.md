# Manual de usuario – Advanced hybrid profiler

## Propósito del script
Este profiler caracteriza arquitecturas de deep learning y genera métricas para integrarlas en un modelo de Programación Lineal Entera (ILP). Produce, por capa y por dispositivo (CPU/GPU), mediciones de tiempo, FLOPs, energía, memoria y transferencia, más metadatos globales que garantizan reproducibilidad. La instrumentación usa NVML/pyRAPL de forma segura, fuerza determinismo y ejecuta un benchmark GEMM para estimar el pico empírico de TFLOPS.

---

## Prerrequisitos

### Hardware
- GPU NVIDIA con soporte CUDA (métricas de kernel y energía vía NVML).
- CPU Linux con acceso a `/sys/class/powercap` para energía RAPL (opcional).
- Para fp16/bf16 en CPU: se sondea ISA acelerada. Requiere AVX512-FP16 (fp16) y AVX512-BF16 o AMX_BF16+AMX_TILE (bf16). Si no existe soporte acelerado, el perfilado se omite y se reporta en CSV/JSON.

### Software
- Python ≥ 3.9; PyTorch ≥ 2.0; TorchVision; Transformers (HuggingFace).
- pandas, NumPy, psutil.
- pynvml (`nvidia-ml-py`).
- pyRAPL (opcional, Linux) con permisos de lectura en `/sys/class/powercap/intel-rapl:0/energy_uj`.

Instalación típica:
```bash
conda install pytorch torchvision -c pytorch
pip install transformers pandas psutil nvidia-ml-py pyRAPL
```

---

## Ejecución

### Comando básico
```bash
python src/profiler.py --model resnet50
```

### Argumentos disponibles
- `--model`: resnet50, resnet152, vit_b16, bert_base, gpt2_small, simple_mlp
- `--precision`: fp32, fp16, bf16
- `--batch_size`: tamaño de batch (default: 8)
- `--warmup`: iteraciones de calentamiento (default: 5)
- `--measure`: iteraciones de medición (default: 15)
- `--output_dir`: directorio de salida para CSV/JSON (default: data)
- `--no_gpu`: fuerza ejecución solo en CPU
- `--gpu_id`: índice de GPU (default: 0)
- `--rapl`: habilita medición de energía CPU vía pyRAPL
- `--input_size`: tamaño de entrada para visión (default: 224)
- `--seq_length`: longitud de secuencia para NLP (default: 128)
- `--optimizer`: SGD, SGD_momentum, Adam, AdamW, RMSprop, Adagrad, Adadelta
- `--lr`: learning rate para metadatos (default: 0.01)
- `--momentum`: momentum (default: 0.9; aplica donde procede)

Ejemplos:
```bash
# GPU + CPU, bf16 en CPU con verificación de soporte, energía CPU
python src/profiler.py --model bert_base --batch_size 16 --precision bf16 --rapl

# Solo CPU, fp32
python src/profiler.py --model simple_mlp --no_gpu --precision fp32
```

---

## Flujo interno

- Determinismo: seeds globales y flags de PyTorch; fallback si no se puede forzar.
- Factory y casting: creación de modelo y datos sintéticos; antes de ejecutar, se evalúa la política de precisión según ISA CPU. Si la precisión solicitada no tiene ruta acelerada, no se ejecuta entrenamiento/perfilado y se guardan artefactos con estado de `skip`.
- Hooks por capa (solo hojas): pre/post hooks miden tiempo de kernel GPU (CUDA Events) o tiempo de pared CPU, overhead de despacho (`dispatch_ms = max(0, wall_ms - kernel_ms)`), memoria pico (proxy global) y tamaño de salida (payload PCIe), FLOPs teóricos por geometría (Conv, Linear, activaciones, norm; heurística para atención).
- Calibración PCIe: estima α/β para H2D y D2H; se usan parámetros como proxy de payload H2D y activaciones como proxy de payload D2H.
- Energía: NVML para GPU; pyRAPL opcional para CPU. Si RAPL no está disponible o falla, la energía CPU se reporta como `None` en metadatos y `0.0` en el CSV por capa.
- Benchmark TFLOPS: GEMM con dtype acorde a la precisión solicitada; reporta pico empírico por dispositivo.
- Salida: CSV por capa y JSON de metadatos globales.

---

## Salida y esquema

### CSV por capa
Ubicación: `data/{model_name}_metrics.csv`

### JSON de metadatos globales
Ubicación: `data/{model_name}_meta.json`

---

## Diccionario de datos

### CSV por capa (`{model_name}_metrics.csv`)

| Columna | Descripción |
|---------|-------------|
| **layer** | Nombre del módulo hoja (p. ej., `conv1`, `fc`) |
| **type** | Tipo de capa (`Conv2d`, `Linear`, `ReLU`, etc.) |
| **params_mb** | Tamaño de parámetros en MB |
| **grads_mb** | Tamaño de gradientes en MB (≈ params_mb) |
| **optimizer_states_mb** | Tamaño de estados del optimizador en MB (`params_mb × factor`) |
| **activations_mb** | Tamaño de activaciones de salida en MB |
| **theoretical_flops** | FLOPs teóricos (forward) |
| **tflops** | Rendimiento efectivo en TFLOPS |
| **efficiency_ratio** | tflops / pico medido |
| **gpu_fwd_time_ms** | Tiempo forward GPU (ms, CUDA Events) |
| **gpu_bwd_time_ms** | Tiempo backward GPU (ms, heurístico = 2× forward) |
| **gpu_fwd_energy_j** | Energía forward GPU (J, proporcional al tiempo) |
| **gpu_bwd_energy_j** | Energía backward GPU (J, heurístico = 2× forward) |
| **gpu_mem_peak_mb** | Memoria pico GPU (proxy global, MB) |
| **layer_j_per_tflop_gpu** | Energía por TFLOP en GPU (J/TFLOP) |
| **dispatch_overhead_ratio** | Overhead de framework GPU = dispatch_ms / fwd_ms |
| **cpu_fwd_time_ms** | Tiempo forward CPU (ms, wall-clock) |
| **cpu_bwd_time_ms** | Tiempo backward CPU (ms, heurístico = 2× forward) |
| **cpu_fwd_energy_j** | Energía forward CPU (J; 0.0 si RAPL no disponible) |
| **cpu_bwd_energy_j** | Energía backward CPU (J; 0.0 si RAPL no disponible) |
| **cpu_mem_mb** | Proxy de memoria CPU (activaciones MB) |
| **layer_j_per_tflop_cpu** | Energía por TFLOP en CPU (J/TFLOP; None si RAPL no disponible) |
| **transfer_h2d_ms** | Tiempo estimado Host→Device (α + params_mb / β) |
| **transfer_d2h_ms** | Tiempo estimado Device→Host (α + activations_mb / β) |
| **remat_penalty_ms** | Penalty de rematerialización (≈ tiempo forward GPU) |
| **precision_requested** | Precisión solicitada (`fp32`, `fp16`, `bf16`) |
| **cpu_precision_executed** | Precisión efectiva en CPU (incluye estados de no soporte, p. ej., `fp16_requested_isa_unsupported`) |
| **gpu_precision_executed** | Precisión efectiva en GPU |
| **run_executed** | Booleano: `true` si se ejecutó el perfilado, `false` si se omitió |
| **skip_unsupported_precision** | Booleano: `true` si se omitió por falta de ISA acelerada |
| **skip_reason** | Motivo detallado del `skip` |
| **optimizer** | Optimizador usado |
| **opt_step_time_ms** | Tiempo acumulado de `optimizer.step()` en la ventana de medición (ms) |

---

### JSON de metadatos globales (`{model_name}_meta.json`)

| Clave | Descripción |
|-------|-------------|
| **timestamp** | Fecha/hora de ejecución |
| **torch_version** | Versión de PyTorch |
| **os** | Plataforma/OS |
| **cpu_model** | Modelo de CPU detectado |
| **gpu_name** | Nombre de GPU detectado |
| **gpu_driver** | Versión de driver GPU |
| **rapl_available** | Booleano: disponibilidad de RAPL |
| **model** | Nombre del modelo perfilado |
| **layers_profiled_count** | Número de capas hoja perfiladas |
| **precision_mode** | Precisión solicitada |
| **execution_status** | Estado de ejecución (`completed` o `skipped_unsupported_precision`) |
| **execution_skip_reason** | Motivo del `skip` (si aplica) |
| **cpu_instruction_flags** | Flags ISA detectadas en CPU (`/proc/cpuinfo`) |
| **cpu_isa_probe** | Resultado estructurado del sondeo ISA para decisiones de precisión |
| **cpu_precision**, **gpu_precision** | Precisión efectiva en CPU/GPU |
| **gpu_total_layer_time_ms** | Suma de tiempos forward GPU por capa |
| **cpu_total_layer_time_ms** | Suma de tiempos forward CPU por capa |
| **gpu_step_time_ms**, **cpu_step_time_ms** | Tiempo promedio de paso GPU/CPU |
| **framework_overhead_gpu_ms**, **framework_overhead_cpu_ms** | Overhead global GPU/CPU |
| **framework_overhead_ratio_gpu**, **framework_overhead_ratio_cpu** | Ratios de overhead global |
| **framework_overhead_vector** | Vector de overhead por capa (semántica GPU) |
| **energy_total_gpu_j**, **energy_total_cpu_j** | Energía total GPU/CPU (CPU puede ser None) |
| **energy_avg_per_step_gpu_j**, **energy_avg_per_step_cpu_j** | Energía promedio por paso |
| **energy_distribution_vector** | Distribución de energía por capa (shares normalizados) |
| **gpu_mem_peak_mb_global** | Memoria pico global GPU |
| **gpu_mem_reserved_mb_global** | Memoria reservada global GPU |
| **cpu_uss_mb_global**, **cpu_pss_mb_global** | Memoria USS/PSS global CPU |
| **params_mb_total**, **grads_mb_total**, **activations_mb_total** | Totales de parámetros, gradientes y activaciones |
| **optimizer_state_mb_factor_fallback**, **optimizer_state_mb_factor_used** | Factor fallback y factor usado para estados del optimizador |
| **transfer_alpha_h2d**, **transfer_beta_h2d** | Parámetros α/β PCIe H2D |
| **transfer_alpha_d2h**, **transfer_beta_d2h** | Parámetros α/β PCIe D2H |
| **pcie_stats_raw** | Resultados crudos de calibración PCIe |
| **measured_peak_tflops_gpu**, **measured_peak_tflops_cpu** | Pico empírico TFLOPS GPU/CPU |
| **efficiency_ratio_avg**, **efficiency_ratio_vector** | Ratio de eficiencia promedio y vector por capa |
| **avg_tflops_per_layer**, **weighted_avg_tflops_per_layer** | Promedio simple y ponderado de TFLOPS |
| **energy_efficiency_j_per_tflop_gpu**, **energy_efficiency_j_per_tflop_cpu** | Eficiencia energética global (J/TFLOP) |
| **optimizer_used**, **optimizer_lr**, **optimizer_momentum** | Optimizador y parámetros |
| **optimizer_step_time_total_ms**, **optimizer_step_time_avg_ms** | Tiempo total y promedio de `optimizer.step()` |
| **total_model_flops**, **total_model_flops_per_step** | FLOPs totales acumulados y por paso (forward dividido por `measure`) |

---

## Mapeo al modelo ILP
- Costo computacional: usar `theoretical_flops`, `tflops`, `gpu_fwd_time_ms`/`cpu_fwd_time_ms` y backward heurístico (2×).
- Energía por operación: `layer_j_per_tflop_gpu` / `layer_j_per_tflop_cpu`; manejar `None` (no medido) vs `0.0` (registrado pero sin consumo) según política ILP.
- Memoria: `params_mb`, `grads_mb`, `activations_mb`, `optimizer_states_mb`; `gpu_mem_peak_mb` es un proxy conservador global.
- Transferencias: `transfer_h2d_ms` (payload params_mb) y `transfer_d2h_ms` (payload activations_mb) con α/β medidos.
- Overhead: `dispatch_overhead_ratio` captura el costo de orquestación en GPU.

---

## Validación y buenas prácticas
- Reproducibilidad: ejecutar en entornos controlados; usar seeds fijos.
- Política de precisión documentada: metadatos registran `cpu_precision_executed`, `execution_status`, `execution_skip_reason` y resultado del sondeo ISA.
- Sanidad de tiempos: variación <5% entre corridas para per-layer y step; overhead ratio estable ±5% absolutos.
- Energía: `energy_total_gpu_j` debe ser >0 si NVML está operativo; energía CPU será `None` si RAPL falla/no está.
- Preferir `weighted_avg_tflops_per_layer` como métrica de rendimiento sostenido.

---

## Solución de problemas
- Columnas de energía CPU en 0.0: RAPL no disponible/inicializado; instala pyRAPL y ejecuta con `--rapl` revisando `/sys/class/powercap`.
- Energía GPU en None/0: verifica NVML (driver, permisos, `nvidia-ml-py`).
- OOM en GEMM: reduce batch o precisión; en casos extremos ajusta `input_size`/`seq_length` o disminuye `measure`.
- Overhead negativo: el cálculo usa `max(0, wall_ms - kernel_ms)`; si ves 0 persistente, revisa sincronización CUDA.

---

## Ejemplos de uso
- ResNet50, fp16 en GPU, CPU fp32, con energía CPU:
```bash
python src/profiler.py --model resnet50 --batch_size 16 --precision fp16 --rapl
```

- ViT, bf16 solicitado; si CPU sin AVX512-BF16 y sin AMX_BF16/AMX_TILE, el perfilado se omite y queda registrado como `skip` en CSV/JSON:
```bash
python src/profiler.py --model vit_b16 --batch_size 8 --precision bf16 --rapl
```

- GPT-2 solo CPU:
```bash
python src/profiler.py --model gpt2_small --no_gpu --batch_size 4 --precision fp32 --rapl
```

- Smoke test rápido (solo CPU, salida en data/test):
```bash
python src/profiler.py --model simple_mlp --no_gpu --precision fp32 --warmup 1 --measure 2 --output_dir data/test
```
