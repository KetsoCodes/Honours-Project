#!/bin/bash
#SBATCH --job-name=attnrct_flash
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/flash_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/flash_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
# #SBATCH --gres=gpu:1     <- uncomment if `sinfo -p bigbatch -o "%P %G"` shows gpu GRES
#
# flash-attn ONLY, and ONLY on bigbatch.
#
# Do NOT run this with `-p stampede`. flash-attn's build targets sm_80+; stampede is
# GTX 1060 (sm_61) and the build will fail during metadata preparation. biggpu is
# RTX 8000 (sm_75) and fails for the same reason. bigbatch (RTX 3090, sm_86) is the
# only partition where this can work.
#
# NOTE: no `set -e` around the build. A failure here should report diagnostics, not
# kill the job before it can tell you why.

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.0

echo "=== host: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version || { echo "FATAL: no nvcc -- cannot build flash-attn"; exit 1; }

CAP=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability(0))" 2>/dev/null || echo "00")
echo "compute capability: sm_${CAP}"
if [ "${CAP:0:1}" -lt 8 ] 2>/dev/null; then
    echo "FATAL: sm_${CAP} is below sm_80. flash-attn cannot build or run here."
    echo "  -> submit this script to bigbatch, without overriding -p."
    exit 1
fi

# Build kernels for THIS architecture only. The default builds every supported arch,
# which is where the multi-hour build times come from. MAX_JOBS caps RAM use.
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4

echo "--- attempting install (verbose, so failures are diagnosable) ---"
pip install flash-attn --no-build-isolation -v 2>&1 | tail -60

if python -c "import flash_attn" 2>/dev/null; then
    python -c "import flash_attn; print('SUCCESS flash_attn', flash_attn.__version__)"
else
    echo "FAILED. Diagnostics:"
    python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    python -c "import packaging, setuptools, wheel" 2>&1 || echo "  missing build deps"
    echo
    echo "Next options:"
    echo " 1. Prebuilt wheel matching torch/python/cxx11abi:"
    echo "    https://github.com/Dao-AILab/flash-attention/releases"
    echo " 2. pip install packaging ninja setuptools wheel  then retry"
    echo " 3. Ticket to support@wits-mss.supportsystem.com, subject 'TWK HPC Query'"
    exit 1
fi
echo "=== done: $(date) ==="
