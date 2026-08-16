#!/bin/bash
#SBATCH --job-name=attnrct_base
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/base_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/base_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#
# Base environment ONLY. Runs on any partition, including stampede:
#   sbatch -p stampede cluster/00a_base_env.slurm
#
# flash-attn is deliberately NOT here. It is architecture-specific and can only be
# built on bigbatch (sm_86) -- see 00b_flash_attn.slurm. Splitting them means a
# stampede-only day still gets you a working environment.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
CONDA_DIR=$HOME_DIR/miniconda3
ENV_NAME=attnrct

mkdir -p "$PROJECT/logs"
echo "=== host: $(hostname) | $(date) ==="

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.0

echo "--- toolchain ---"
which nvcc && nvcc --version || echo "NOTE: no nvcc on this node"
ls -d /usr/local/cuda-* 2>/dev/null || echo "NOTE: no /usr/local/cuda-* on this node"

if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy wandb pytest tqdm

echo "--- verification ---"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("compiled for CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0))
    print("capability: sm_%d%d" % cap)
    print("FA-2 capable:" , "YES" if cap[0] >= 8 else "NO (needs sm_80+; this node cannot)")
PY

echo "=== base env done: $(date) ==="
