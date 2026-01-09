# Manual de usuario – Advanced hybrid profiler

## Propósito del script
Este profiler caracteriza arquitecturas de deep learning y genera métricas para su integración en un modelo de Programación Lineal Entera (ILP). Produce, por capa y por dispositivo (CPU/GPU), mediciones de tiempo, FLOPs, energía, memoria y transferencia, junto con metadatos globales que garantizan reproducibilidad y trazabilidad. La instrumentación es segura e idempotente (NVML/RAPL), con determinismo activado y un benchmark GEMM para estimar el pico empírico de TFLOPS.

---

## Prerrequisitos

### Hardware
- GPU NVIDIA con soporte CUDA para métricas de kernel y energía GPU (NVML).
- CPU Linux con acceso a /sys/class/powercap para energía vía RAPL (opcional).
- Para bf16 en CPU: soporte AVX512-BF16; de lo contrario, se aplica fallback automático a fp32.

### Software
- Python ≥ 3.9; PyTorch ≥ 2.0; TorchVision; Transformers (HuggingFace).
- Pandas, NumPy, psutil.
- pynvml (nvidia-ml-py).
- pyRAPL (opcional, Linux). Requiere permisos de lectura en /sys/class/powercap/intel-rapl:0/energy_uj.

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
- --model: resnet50, resnet152, vit, bert, gpt2, mlp
- --batch-size: tamaño de batch (default: 8)
- --seq-len: longitud de secuencia para NLP (default: 128; aplica a modelos HuggingFace cuando prepares tu input)
- --precision: fp32, fp16, bf16
- --warmup: iteraciones de calentamiento (default: 5)
- --measure: iteraciones de medición (default: 15)
- --output-dir: directorio de salida para CSV/JSON (default: data)
- --gpu-id: índice de GPU (default: 0)
- --no-gpu: fuerza ejecución solo en CPU
- --rapl: habilita medición de energía CPU vía pyRAPL (si disponible)
- --nvml-sample-interval: intervalo de muestreo NVML en segundos (default: 0.05)
- --gpu-gemm-n, --cpu-gemm-n: tamaño N para benchmark GEMM de TFLOPS (default: 8192 / 2048)

Ejemplos:
```bash
# GPU + CPU, bf16 en CPU con verificación de soporte, medición de energía CPU
python src/profiler.py --model bert --batch-size 16 --precision bf16 --rapl

# Solo CPU, fp32
python src/profiler.py --model mlp --no-gpu --precision fp32
```

---

## Flujo interno

- Determinismo: seeds globales y flags de PyTorch; fallback si no se pueden forzar algoritmos deterministas.
- Factory y casting: creación del modelo y datos sintéticos; conversión de tensores flotantes a fp16/bf16 según precisión solicitada. Para bf16 en CPU se verifica AVX512-BF16; si no, se usa fp32.
- Hooks por capa (leaf-only): se instrumentan pre/post hooks para medir:
  - Tiempo de kernel en GPU (CUDA Events) y tiempo de pared (CPU).
  - Overhead de framework: dispatch_ms = max(0, wall_ms - kernel_ms).
  - Memoria pico (proxy global del allocator CUDA) y tamaño de salida (payload para PCIe).
  - FLOPs teóricos según geometría (Conv, Linear, activaciones, norm; heurística para atención).
- Calibración PCIe: estima α (latencia fija) y β (MB/ms) en ambos sentidos (H2D/D2H) con memoria pinned para H2D.
- Energía:
  - GPU: lecturas NVML periódicas, promedio de potencia (W) y energía total por ventana; shutdown idempotente.
  - CPU: pyRAPL (opcional) calcula potencia promedio a partir de energía µJ y duración µs; si falla o no está disponible, energía CPU se reporta como None en metadatos y 0.0 en columnas CSV por capa.
- Benchmark TFLOPS: GEMM con dtypes según precisión y fallback seguros; se reporta pico empírico por dispositivo.
- Output: CSV por capa y JSON con metadatos globales (hardware/software, tiempos, energía, memoria, TFLOPS medios y ponderados, distribución de energía y vector de overhead).

---

## Salida y esquema de datos

### CSV por capa
Ubicación: data/{model_name}_metrics.csv

Columnas principales:
- Identidad: layer, type
- Cómputo: theoretical_flops, tflops, efficiency_ratio
- Memoria:
  - params_mb, grads_mb
  - optimizer_states_mb = params_mb × factor (SGD=0, SGD_momentum=1, Adam/AdamW=2, RMSprop/Adagrad=1, Adadelta=2; fallback=2)
  - gpu_mem_peak_mb (proxy global del allocator en ese momento)
  - cpu_mem_mb (proxy basado en activaciones por capa)
- Tiempos y energía:
  - gpu_fwd_time_ms, gpu_bwd_time_ms (= fwd × 2)
  - gpu_fwd_energy_j, gpu_bwd_energy_j (= fwd × 2, None si energía GPU no está disponible)
  - cpu_fwd_time_ms, cpu_bwd_time_ms (= fwd × 2)
  - cpu_fwd_energy_j, cpu_bwd_energy_j (0.0 cuando RAPL no está disponible)
- Calidad energética:
  - layer_j_per_tflop_gpu (J/TFLOP por capa, si energía y trabajo > 0)
  - layer_j_per_tflop_cpu (None si no hay energía CPU)
- Transferencias:
  - transfer_h2d_ms = α_h2d + activations_mb / β_h2d
  - transfer_d2h_ms = α_d2h + activations_mb / β_d2h
- Overhead:
  - dispatch_overhead_ratio (GPU): g_dispatch / g_t_fwd (0 si g_t_fwd=0)
- Precisión y optimizador:
  - precision_requested, cpu_precision_executed, gpu_precision_executed
  - optimizer, opt_step_time_ms (promedio por step)

Nota sobre columnas vacías: Si el entorno no entrega energía CPU (RAPL no disponible), la energía por capa en CPU se reporta como 0.0 para el CSV (evitar celdas vacías) y como None en metadatos globales (para distinguir “no medido” de “cero”).

### JSON de metadatos globales
Ubicación: data/{model_name}_meta.json

Incluye:
- Hardware/software: timestamp, versiones, OS, cpu_model, gpu_name, gpu_driver, nvml_status, rapl_available.
- Contadores y precisión: layers_profiled_count, precision_mode, cpu_precision, gpu_precision.
- Tiempos: gpu_total_layer_time_ms, cpu_total_layer_time_ms, gpu_step_time_ms, cpu_step_time_ms.
- Overhead global: framework_overhead_gpu_ms, framework_overhead_cpu_ms y sus ratios; vector de overhead por capa (semántica GPU).
- Energía global:
  - energy_total_gpu_j, energy_total_cpu_j (CPU puede ser None)
  - energy_avg_per_step_gpu_j (None si GPU energía no disponible), energy_avg_per_step_cpu_j (None si CPU energía no disponible)
  - energy_distribution_vector (shares por tiempo; suma 1.0 por dispositivo)
- Memoria global: gpu_mem_peak_mb_global, gpu_mem_reserved_mb_global, cpu_uss_mb_global, cpu_pss_mb_global.
- PCIe: transfer_alpha/beta para H2D y D2H; pcie_stats_raw.
- Rendimiento: measured_peak_tflops_gpu, measured_peak_tflops_cpu; avg_tflops_per_layer (simple), weighted_avg_tflops_per_layer (ponderado).
- Eficiencia energética global (J/TFLOP): energy_efficiency_j_per_tflop_gpu, energy_efficiency_j_per_tflop_cpu (None si energía indisponible).
- Optimizador: optimizer_used, optimizer_lr, optimizer_momentum, optimizer_step_time_total_ms, optimizer_step_time_avg_ms.
- Factor de estados: optimizer_state_mb_factor_fallback y optimizer_state_mb_factor_used.

---

## Mapeo al modelo ILP

- Costo computacional: usar theoretical_flops y tflops por capa; para tiempos, gpu_fwd_time_ms/cpu_fwd_time_ms y sus backward estimados.
- Energía por operación: layer_j_per_tflop_gpu / layer_j_per_tflop_cpu; usar valores None/0.0 según disponibilidad y política de tu ILP.
- Memoria:
  - Activaciones y params/grads determinan restricciones de colocación.
  - optimizer_states_mb aporta el multiplicador por optimizador (registrado en metadatos).
  - gpu_mem_peak_mb (proxy global conservador) garantiza seguridad en límites de VRAM.
- Transferencias: transfer_h2d_ms y transfer_d2h_ms con α/β medidos para enlazar nodos CPU/GPU.
- Overhead: dispatch_overhead_ratio para capturar costo de orquestación en GPU (puede ajustarse como penalización en nodos GPU).

---

## Validación y buenas prácticas

- Reproducibilidad: ejecuta en entornos controlados (misma GPU/CPU/driver); usa seeds fijos.
- Fallbacks documentados: registra en metadatos CPU bf16→fp32 y disponibilidad RAPL/NVML.
- Sanidad de tiempos: variación <5% entre runs para per-layer y step; overhead ratio estable ±5% absolutos.
- Energía:
  - GPU: verificar que energy_total_gpu_j > 0; si 0, revisar NVML y permisos.
  - CPU: si usas --rapl, verifica que energy_total_cpu_j no sea None (si lo es, revisa pyRAPL y /sys/class/powercap).
- Coherencia ILP: weighted_avg_tflops_per_layer debe ser la métrica preferida de rendimiento sostenido.

---

## Solución de problemas

- Columnas CPU energía vacías en CSV:
  - Causa: RAPL no disponible/inicializado; el código escribe 0.0 en CSV (versión actual) y None en metadatos.
  - Acción: instala pyRAPL, ejecuta con --rapl y verifica /sys/class/powercap/intel-rapl:0/energy_uj.
- Energía GPU en None:
  - Causa: NVML no inicializa o falla lectura.
  - Acción: verifica instalación de nvidia-ml-py, driver, permisos; revisa nvml_status en metadatos.
- Out-of-memory en GEMM:
  - Acción: ajustar --gpu-gemm-n / --cpu-gemm-n; el benchmark reduce automáticamente N, pero puede requerir valores más pequeños.
- Dispatch overhead negativo:
  - El código usa max(0, wall_ms - kernel_ms); si observas 0 sistemático, revisa contexto CUDA y sincronización.

---

## Ejemplos de uso

- ResNet50, fp16 en GPU, CPU fp32, con energía CPU:
```bash
python src/profiler.py --model resnet50 --batch-size 16 --precision fp16 --rapl
```

- ViT, bf16 solicitado; si CPU sin AVX512-BF16, fallback a fp32_fallback registrado en metadatos:
```bash
python src/profiler.py --model vit --batch-size 8 --precision bf16 --rapl
```

- GPT-2 solo CPU:
```bash
python src/profiler.py --model gpt2 --no-gpu --batch-size 4 --precision fp32 --rapl
```

---

Claro, Luis. Aquí tienes el **Data Dictionary** actualizado para el profiler, que complementa el manual y describe cada columna del CSV y cada clave del JSON de salida. Esto te servirá para mapear directamente las métricas al modelo ILP y documentar con precisión:

---

# Data Dictionary – Advanced Hybrid Profiler

## 1. CSV por capa (`{model_name}_metrics.csv`)

| Columna | Descripción |
|---------|-------------|
| **layer** | Nombre del módulo hoja (ej. `conv1`, `fc`) |
| **type** | Tipo de capa (`Conv2d`, `Linear`, `ReLU`, etc.) |
| **params_mb** | Tamaño de parámetros de la capa en MB |
| **grads_mb** | Tamaño de gradientes en MB (≈ params_mb) |
| **optimizer_states_mb** | Tamaño de estados del optimizador en MB (`params_mb × factor`) |
| **theoretical_flops** | FLOPs teóricos de la capa (forward) |
| **tflops** | Rendimiento efectivo de la capa en TFLOPS |
| **efficiency_ratio** | Ratio de eficiencia = tflops / pico medido |
| **activations_mb** | Tamaño de activaciones de salida en MB |
| **gpu_fwd_time_ms** | Tiempo forward GPU (ms, CUDA Events) |
| **gpu_bwd_time_ms** | Tiempo backward GPU (ms, heurístico = 2× fwd) |
| **gpu_fwd_energy_j** | Energía forward GPU (J, proporcional al tiempo) |
| **gpu_bwd_energy_j** | Energía backward GPU (J, heurístico = 2× fwd) |
| **gpu_mem_peak_mb** | Memoria pico GPU (proxy global, MB) |
| **layer_j_per_tflop_gpu** | Energía por TFLOP en GPU (J/TFLOP) |
| **dispatch_overhead_ratio** | Overhead de framework GPU = dispatch_ms / fwd_ms |
| **cpu_fwd_time_ms** | Tiempo forward CPU (ms, wall-clock) |
| **cpu_bwd_time_ms** | Tiempo backward CPU (ms, heurístico = 2× fwd) |
| **cpu_fwd_energy_j** | Energía forward CPU (J, proporcional al tiempo; 0.0 si RAPL no disponible) |
| **cpu_bwd_energy_j** | Energía backward CPU (J, heurístico = 2× fwd; 0.0 si RAPL no disponible) |
| **cpu_mem_mb** | Proxy de memoria CPU (activaciones MB) |
| **layer_j_per_tflop_cpu** | Energía por TFLOP en CPU (J/TFLOP; None si RAPL no disponible) |
| **transfer_h2d_ms** | Tiempo estimado de transferencia Host→Device (α + MB/β) |
| **transfer_d2h_ms** | Tiempo estimado de transferencia Device→Host (α + MB/β) |
| **precision_requested** | Precisión solicitada (`fp32`, `fp16`, `bf16`) |
| **cpu_precision_executed** | Precisión efectiva en CPU (ej. `fp32_fallback`) |
| **gpu_precision_executed** | Precisión efectiva en GPU |
| **optimizer** | Optimizador usado (`SGD`, `Adam`, etc.) |
| **opt_step_time_ms** | Tiempo promedio de paso del optimizador (ms) |

---

## 2. JSON de metadatos globales (`{model_name}_meta.json`)

| Clave | Descripción |
|-------|-------------|
| **timestamp** | Fecha/hora de ejecución |
| **python_version** | Versión de Python |
| **torch_version** | Versión de PyTorch |
| **os** | Plataforma/OS |
| **cpu_model** | Modelo de CPU detectado |
| **gpu_name** | Nombre de GPU detectado |
| **gpu_driver** | Versión de driver GPU |
| **rapl_available** | Booleano: disponibilidad de RAPL |
| **nvml_status** | Estado NVML (`initialized`, `last_error`) |
| **model** | Nombre del modelo perfilado |
| **layers_profiled_count** | Número de capas hoja perfiladas |
| **precision_mode** | Precisión solicitada |
| **cpu_precision**, **gpu_precision** | Precisión efectiva en CPU/GPU |
| **gpu_total_layer_time_ms** | Suma de tiempos forward GPU por capa |
| **cpu_total_layer_time_ms** | Suma de tiempos forward CPU por capa |
| **gpu_step_time_ms**, **cpu_step_time_ms** | Tiempo promedio de paso GPU/CPU |
| **framework_overhead_gpu_ms**, **framework_overhead_cpu_ms** | Overhead global GPU/CPU |
| **framework_overhead_ratio_gpu**, **framework_overhead_ratio_cpu** | Ratio overhead global |
| **framework_overhead_vector** | Vector de overhead por capa (GPU) |
| **energy_total_gpu_j**, **energy_total_cpu_j** | Energía total GPU/CPU (CPU puede ser None) |
| **energy_avg_per_step_gpu_j**, **energy_avg_per_step_cpu_j** | Energía promedio por paso |
| **energy_distribution_vector** | Distribución de energía por capa (shares normalizados) |
| **gpu_mem_peak_mb_global** | Memoria pico global GPU |
| **gpu_mem_reserved_mb_global** | Memoria reservada global GPU |
| **cpu_uss_mb_global**, **cpu_pss_mb_global** | Memoria USS/PSS global CPU |
| **params_mb_total**, **grads_mb_total**, **activations_mb_total** | Totales de parámetros, gradientes y activaciones |
| **optimizer_state_mb_factor_fallback** | Factor fallback de estados del optimizador |
| **optimizer_state_mb_factor_used** | Factor usado para el optimizador actual |
| **transfer_alpha_h2d**, **transfer_beta_h2d** | Parámetros α/β PCIe H2D |
| **transfer_alpha_d2h**, **transfer_beta_d2h** | Parámetros α/β PCIe D2H |
| **pcie_stats_raw** | Resultados crudos de calibración PCIe |
| **measured_peak_tflops_gpu**, **measured_peak_tflops_cpu** | Pico empírico TFLOPS GPU/CPU |
| **efficiency_ratio_avg** | Promedio de ratios de eficiencia |
| **efficiency_ratio_vector** | Vector de eficiencia por capa |
| **avg_tflops_per_layer**, **weighted_avg_tflops_per_layer** | Promedio simple y ponderado de TFLOPS |
| **energy_efficiency_j_per_tflop_gpu**, **energy_efficiency_j_per_tflop_cpu** | Eficiencia energética global (J/TFLOP) |
| **optimizer_used**, **optimizer_lr**, **optimizer_momentum** | Optimizador y parámetros |
| **optimizer_step_time_total_ms**, **optimizer_step_time_avg_ms** | Tiempo total y promedio de paso del optimizador |

---

### Con este diccionario:
- Puedes mapear cada columna/clave directamente a variables de tu modelo ILP.  
- Sabes cuándo un valor puede ser `None` (no medido) o `0.0` (medido pero sin consumo).  
- Tienes claridad sobre proxies (memoria GPU/CPU) y heurísticas (backward = 2× forward).  

