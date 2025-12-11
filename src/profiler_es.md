# Manual de Usuario – Advanced Hybrid Profiler

## 1. Propósito del Script
Este profiler caracteriza arquitecturas de deep learning y genera métricas necesarias para la optimización mediante **Programación Lineal Entera (ILP)**.  
Captura por capa:

- **FLOPs** (complejidad computacional)  
- **Huella de memoria**  
  - **GPU VRAM**: snapshot global del *peak memory* vía CUDA (proxy conservador, incluye fragmentación y buffers)  
  - **CPU memoria**: aproximada usando activaciones como proxy y delta de RSS/USS del proceso  
  - **Estados del optimizador**: calculados explícitamente como `params_mb × factor` (ej. Adam m,v)  
- **Payload de transferencia**: tamaño exacto del tensor de salida (crítico para modelado PCIe)  
- **Tiempo de ejecución**  
  - **Tiempo de kernel GPU** (CUDA Events)  
  - **Tiempo de pared CPU** (perf_counter)  
  - **Overhead de framework**: vector por capa con `dispatch_overhead_ms` y `dispatch_overhead_ratio`  
- **Consumo energético**  
  - GPU vía NVML  
  - CPU vía RAPL (si disponible)  
  - Distribución de energía por capa proporcional al tiempo de cómputo, normalizada a 1.0 por dispositivo  
- **Eficiencia empírica**  
  - TFLOPS por capa y promedio ponderado global  
  - Ratio de eficiencia respecto al pico medido  
  - Energía por TFLOP (`layer_j_per_tflop_gpu`, `layer_j_per_tflop_cpu`)  

---

## 2. Prerrequisitos

### Hardware
- GPU NVIDIA con soporte CUDA (para métricas GPU y NVML)  
- CPU con acceso a `/sys/class/powercap` (Linux) para medición de energía vía RAPL  
- Para **bf16 en CPU**, soporte AVX512-BF16; de lo contrario, fallback automático a fp32  

### Software
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- TorchVision  
- Transformers (HuggingFace)  
- Pandas, NumPy, psutil  
- pynvml (`nvidia-ml-py`)  
- pyRAPL (opcional, Linux): sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj   

Instalación típica:
```bash
conda install pytorch torchvision -c pytorch
pip install transformers pandas psutil nvidia-ml-py pyRAPL
```

---

## 3. Ejecución

### Comando básico
```bash
python src/profiler.py --model resnet50
```

### Argumentos disponibles
- **`--model`**: arquitectura a perfilar. Opciones:  
  - `resnet50`, `resnet152`, `vit`, `bert`, `gpt2`, `mlp`
- **`--batch-size`**: tamaño de batch (default: 4)  
- **`--seq-len`**: longitud de secuencia (NLP, default: 128)  
- **`--precision`**: precisión numérica. Opciones: `fp32`, `fp16`, `bf16`  
- **`--warmup`**: iteraciones de calentamiento (default: 5)  
- **`--measure`**: iteraciones de medición (default: 15)  
- **`--output-dir`**: directorio de salida para CSV (default: `data`)  
- **`--gpu-id`**: índice de GPU (default: 0)  
- **`--no-gpu`**: fuerza ejecución solo en CPU  

Ejemplo:
```bash
python src/profiler.py --model bert --batch-size 16 --seq-len 256 --precision bf16
```

---

## 4. Flujo Interno

1. **Determinismo**: semillas y flags fijados para reproducibilidad  
2. **Factory**: construcción del modelo y datos sintéticos  
3. **Manejo de precisión**:  
   - Conversión a fp16/bf16 si se solicita  
   - Solo tensores flotantes convertidos (no `input_ids`)  
   - Fallback automático a fp32 si CPU no soporta bf16  
4. **Bucles de profiling**:  
   - Entrenamiento en GPU y CPU por separado  
   - Captura de tiempos, memoria y energía  
   - Estimación de FLOPs y payload de transferencia por capa  
   - Vector de overhead por capa (`dispatch_overhead_ms`, `dispatch_overhead_ratio`)  
5. **Calibración PCIe**: α y β medidos en tiempo de ejecución usando memoria *pinned* para mayor realismo  
6. **Salida CSV/JSON**: métricas por capa y metadatos globales  

---

## 5. Salida

### CSV
Generado en `data/{model_name}_metrics.csv`.  
Columnas principales:

- **layer**, **type**  
- **params_mb**, **grads_mb**, **optimizer_states_mb**  
- **theoretical_flops**, **tflops**, **efficiency_ratio**  
- **activations_mb**  
- **gpu_fwd_time_ms**, **gpu_bwd_time_ms**, **gpu_fwd_energy_j**, **gpu_bwd_energy_j**  
- **gpu_mem_peak_mb** (proxy global)  
- **cpu_fwd_time_ms**, **cpu_bwd_time_ms**, **cpu_fwd_energy_j**, **cpu_bwd_energy_j**  
- **cpu_mem_mb** (proxy activations)  
- **layer_j_per_tflop_gpu**, **layer_j_per_tflop_cpu**  
- **transfer_h2d_ms**, **transfer_d2h_ms**  
- **dispatch_overhead_ms**, **dispatch_overhead_ratio**  
- **precision_requested**, **cpu_precision_executed**, **gpu_precision_executed**

### JSON (metadatos globales)
Incluye:
- Hardware y software (`get_hardware_metadata()`)  
- Conteo de capas, precisión usada  
- Tiempos totales y overhead global  
- Energía total y promedio por paso  
- Distribución de energía normalizada  
- Memoria global (GPU reserved, CPU USS/PSS)  
- α y β de transferencias PCIe  
- TFLOPS promedio simple y ponderado  
- Eficiencia energética global (J/TFLOP)  
- Ratios de integridad (ej. suma de energía por capa vs total)  

---

## 6. Ejemplo CSV (comentado)

```
layer,type,params_mb,grads_mb,optimizer_states_mb,theoretical_flops,tflops,efficiency_ratio,
activations_mb,gpu_fwd_time_ms,gpu_bwd_time_ms,gpu_fwd_energy_j,gpu_bwd_energy_j,
gpu_mem_peak_mb,layer_j_per_tflop_gpu,dispatch_overhead_ms,dispatch_overhead_ratio,
cpu_fwd_time_ms,cpu_bwd_time_ms,cpu_fwd_energy_j,cpu_bwd_energy_j,cpu_mem_mb,layer_j_per_tflop_cpu,
transfer_h2d_ms,transfer_d2h_ms,precision_requested,cpu_precision_executed,gpu_precision_executed

conv1,Conv2d,0.0179,0.0179,0.0358,236027904,1.03,0.45,
6.125,1.03,2.06,0.024,0.048,
420.48,0.023,0.12,0.11,
143.48,286.96,10.20,20.40,1.416,0.071,
0.92,0.92,fp16,fp16,fp16
```

---

## 7. Limitaciones

- **CPU energía**: requiere `/sys/class/powercap`; si no, se reporta `NaN`  
- **Memoria GPU por capa**: proxy global, no estrictamente aislado  
- **Memoria CPU**: aproximada usando activaciones como proxy  
- **Backward**: estimado como `T_bwd = 2 × T_fwd` salvo que se activen hooks de backward  
- **Transferencias PCIe**: α y β dependen del hardware, calibrados en tiempo de ejecución  

---

## 8. Buenas Prácticas

- Ejecutar siempre en entorno controlado (misma GPU/CPU) para comparabilidad  
- Documentar cualquier fallback (ej. bf16→fp32 en CPU)  
- Usar batch sizes y secuencias representativas de cargas reales  
- Guardar CSV junto con metadatos de hardware/software  
- Validar integridad: suma de energía por capa ≈ energía total, TFLOPS ponderado consistente  

---
