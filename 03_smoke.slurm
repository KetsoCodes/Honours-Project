#!/bin/bash
#SBATCH --job-name=attnrct_smoke
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/smoke_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1

# Plumbing only. stampede is GTX 1060 (sm_61) and CANNOT run FA-2 -- the flash arm will
# take the SDPA fallback here, which is fine because this proves the env loads, the data
# reads, and a checkpoint writes and resumes. It proves NOTHING about efficiency.
#
# Guidelines 2.2.7: start on stampede, escalate only when it doesn't meet requirements.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct
[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets

cd "$PROJECT"

echo "--- unit tests ---"
python -m pytest attn_rct/tests/ -v

echo "--- 2-epoch run, checkpoint written ---"
python -m attn_rct.train \
    --manifest cluster/runs.csv --row 1 \
    --data-dir /datasets/fmnisi/lra/listops \
    --checkpoint-dir /tmp/smoke_ckpt \
    --result-file /tmp/smoke_result.json \
    --epochs 2 --limit-batches 20

echo "--- resume from that checkpoint (must NOT restart from epoch 0) ---"
python -m attn_rct.train \
    --manifest cluster/runs.csv --row 1 \
    --data-dir /datasets/fmnisi/lra/listops \
    --checkpoint-dir /tmp/smoke_ckpt \
    --result-file /tmp/smoke_result.json \
    --epochs 4 --limit-batches 20

echo "smoke test done"
