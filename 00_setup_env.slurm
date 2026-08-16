#!/bin/bash
#SBATCH --job-name=attnrct_setup
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/setup_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/setup_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
# If `sinfo -o "%P %G"` showed gpu GRES, uncomment:
# #SBATCH --gres=gpu:1

# Built as a batch job, not run on the login node: the flash-attn source build is
# CPU- and RAM-heavy and would affect every other user (guidelines 2.2.5).

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
CONDA_DIR=$HOME_DIR/miniconda3
ENV_NAME=attnrct

mkdir -p "$PROJECT/logs"
echo "=== host: $(hostname)  date: $(date) ==="

# --- CUDA 12.0, as recommended in the guidelines (9.4) ---
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.0
nvcc --version || echo "WARNING: nvcc not found; flash-attn source build will fail"

# --- Miniconda in the home dir, per guidelines 9.1 ---
if [ ! -d "$CONDA_DIR" ]; then
    echo "--- installing miniconda ---"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# --- PyTorch, CUDA 12.1 wheels (compatible with the 12.0 toolkit) ---
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy wandb pytest tqdm

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", "sm_%d%d" % torch.cuda.get_device_capability(0))
PY

# --- flash-attn: prebuilt wheel first, source build only as fallback ---
# A source build takes 30-90 min and a lot of RAM. Try not to.
echo "--- installing flash-attn ---"
if ! python -c "import flash_attn" 2>/dev/null; then
    pip install flash-attn --no-build-isolation || {
        echo "wheel install failed; falling back to a limited-parallelism source build"
        MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-binary :all:
    }
fi

python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" \
    || echo "FLASH-ATTN NOT AVAILABLE -- stage 2 will fail; open a ticket if the build errored"

echo "=== setup finished: $(date) ==="
