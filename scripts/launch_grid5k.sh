#!/bin/bash
#OAR -n profiler_hybrid
#OAR -l /nodes=1
#OAR --stdout logs/profiler_%jobid%.out
#OAR --stderr logs/profiler_%jobid%.err

# ==============================================================================
# SCRIPT DE LANZAMIENTO PARA GRID5000 (OAR / KADEPLOY)
# Autodetecta Hardware (AMD/Intel) y maximiza paralelismo CPU.
# ==============================================================================

# 1. CARGAR ENTORNO
# Ajusta esta línea según dónde esté tu anaconda en la imagen kadeploy
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source activate hybrid-profiler
conda activate hybrid-profiler

# Crear directorios
mkdir -p data logs

# ==============================================================================
# 2. DETECCIÓN DE HARDWARE Y CONFIGURACIÓN OMP (CRÍTICO)
# ==============================================================================

echo ">>> Iniciando autodetección de hardware..."

# A. Detectar Vendor (Intel vs AMD)
CPU_VENDOR=$(lscpu | grep "Vendor ID" | awk '{print $3}')
MODEL_NAME=$(lscpu | grep "Model name" | head -n 1 | cut -d ':' -f 2 | xargs)

# B. Contar Cores Físicos (Evitar SMT/HyperThreading para DL)
# Contamos líneas únicas de par (Core, Socket)
PHY_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l)

echo "    - CPU Detectada: $MODEL_NAME ($CPU_VENDOR)"
echo "    - Cores Físicos Disponibles: $PHY_CORES"

# C. Configuración Agóstica
export OMP_NUM_THREADS=$PHY_CORES
export MKL_NUM_THREADS=$PHY_CORES
export TORCH_NUM_THREADS=$PHY_CORES

# D. Configuración Específica por Vendor
if [[ "$CPU_VENDOR" == "AuthenticAMD" ]]; then
    echo "    - Modo: OPTIMIZACIÓN AMD EPYC (ZEN)"
    
    # Fix para MKL en AMD (fuerza caminos AVX2/AVX512)
    export MKL_DEBUG_CPU_TYPE=5 
    
    # Binding para evitar saltos entre Chiplets (CCDs)
    export OMP_BIND_PROC=true
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close

elif [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
    echo "    - Modo: OPTIMIZACIÓN INTEL XEON"
    
    # Afinidad compacta para Intel MKL
    export KMP_AFFINITY=granularity=fine,compact,1,0
    # Deshabilitar espera activa para ahorrar ciclos si hay I/O
    export KMP_BLOCKTIME=0

else
    echo "    - ! Vendor desconocido. Usando configuración genérica."
fi

echo "------------------------------------------------------------------"
echo "ENV VARS APLICADAS:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_DEBUG_CPU_TYPE=${MKL_DEBUG_CPU_TYPE:-'(unset)'}"
echo "------------------------------------------------------------------"

# ==============================================================================
# 3. EJECUCIÓN DE EXPERIMENTOS
# ==============================================================================

# NOTA: Ajustamos batch sizes asumiendo una GPU decente en Grid5K (V100/A100/H100)
# Si usas GPUs antiguas (P100/K80), reduce el batch-size a la mitad.

# 1. ResNet-50 (Visión)
echo "[$(date)] Iniciando ResNet-50..."
python src/profiler.py \
    --model resnet50 \
    --batch-size 64 \
    --gpu-id 0 \
    --rapl \
    --output-dir data

# 2. BERT-Base (NLP)
echo "[$(date)] Iniciando BERT-Base..."
python src/profiler.py \
    --model bert \
    --batch-size 32 \
    --seq-len 128 \
    --gpu-id 0 \
    --rapl \
    --output-dir data

# 3. ViT-B/16 (Transformer Visión)
# Este es el que requiere el OMP_NUM_THREADS correcto para usar todos los cores.
echo "[$(date)] Iniciando ViT-B/16..."
python src/profiler.py \
    --model vit \
    --batch-size 64 \
    --gpu-id 0 \
    --rapl \
    --output-dir data

echo "[$(date)] Profiling Finalizado."