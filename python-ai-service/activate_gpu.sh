#!/bin/bash
# WSL2 GPU Activation Script
# 
# Source this before running training to enable GPU support:
# $ source activate_gpu.sh
# $ python training/train_1d_regressor_final.py AAPL --epochs 100
#
# This configures CUDA library paths for WSL2, which stores NVIDIA libraries
# in non-standard locations (/usr/lib/wsl/lib and conda site-packages/nvidia/*).

export CUDA_PATH=/usr/lib/wsl/lib
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# Build LD_LIBRARY_PATH with all NVIDIA libraries
NVIDIA_LIBS="/usr/lib/wsl/lib"

# Add conda environment NVIDIA libraries
CONDA_NVIDIA="/home/thunderboltdy/miniconda3/envs/ai-stocks/lib/python*/site-packages/nvidia"
for nvidia_pkg_lib in $(find /home/thunderboltdy/miniconda3/envs/ai-stocks/lib/python*/site-packages/nvidia -maxdepth 2 -name "lib" -type d 2>/dev/null); do
    NVIDIA_LIBS="${NVIDIA_LIBS}:${nvidia_pkg_lib}"
done

export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"

echo "✓ GPU environment configured"
echo "  CUDA_PATH=$CUDA_PATH"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NVIDIA libraries: $(echo $NVIDIA_LIBS | wc -w) paths configured"
