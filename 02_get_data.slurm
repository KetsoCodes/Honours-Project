#!/bin/bash
#SBATCH --job-name=attnrct_data
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/data_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/data_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1

# Official LRA ListOps, generated with the LRA authors' own generator.
#
# WHY NOT THE RELEASED ARCHIVE:
#   https://storage.googleapis.com/long-range-arena/lra_release.gz returns
#   403 AccessDenied ("Anonymous caller does not have storage.objects.get access"),
#   verified 2026-08-16 from two independent networks. Public access to the bucket has
#   been revoked. Generating the split with lra_benchmarks/data/listops.py is the
#   fallback the LRA README itself documents.
#
# WHAT THIS CHANGES FOR THE STUDY:
#   The generated split is NOT byte-identical to the released one, so absolute
#   accuracies are not directly comparable to published LRA ListOps numbers. Internal
#   validity is unaffected: every arm sees exactly the same generated data, which is all
#   the paired design requires. Say so plainly in the methodology.
#
#   Still a large improvement on the community mirror (fengyang0317/listops-1000), which
#   was raw generator output at lengths 1501-6000, filtered to ~21% survival and biased
#   towards the hardest examples.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
DATA_ROOT=/datasets/fmnisi/lra
REPO=$HOME_DIR/long-range-arena

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"

# SEPARATE ENV, deliberately.
# The LRA generator is TF/JAX-era code and needs tensorflow. TF and PyTorch pin
# conflicting numpy versions, so installing TF into `attnrct` risks breaking the
# training environment to produce a dataset exactly once. This env is disposable:
#   conda env remove -n lradata      # after the data is generated
if ! conda env list | grep -q "^lradata "; then
    conda create -y -n lradata python=3.11
fi
conda activate lradata

echo "=== host: $(hostname) | $(date) ==="

# /datasets/ may not exist for you yet; fall back to home. ListOps is tens of MB, so
# the 50GB home quota is not a concern -- guideline 7.4 targets 50GB+ datasets.
if ! mkdir -p "$DATA_ROOT" 2>/dev/null; then
    echo "NOTE: cannot write /datasets/fmnisi -- using home dir instead"
    DATA_ROOT=$HOME_DIR/data/lra
    mkdir -p "$DATA_ROOT"
fi
echo "data root: $DATA_ROOT"

if [ ! -d "$REPO" ]; then
    git clone --depth 1 https://github.com/google-research/long-range-arena.git "$REPO"
fi
cd "$REPO"
echo "LRA repo commit: $(git rev-parse HEAD)"    # record this in the write-up

# tensorflow-cpu: the generator only uses tf.io.gfile for file writing, so the GPU
# build would be wasted download. LRA's jax/flax deps are for their training code and
# are not needed here.
python -c "import tensorflow" 2>/dev/null || pip install -q "tensorflow-cpu" absl-py numpy
python -c "import tensorflow as tf; print('tensorflow:', tf.__version__)"

OUT=$DATA_ROOT/listops
mkdir -p "$OUT"

echo "--- generating ListOps ---"
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python -m lra_benchmarks.data.listops \
    --output_dir="$OUT" || {
        echo
        echo "Generator failed. Things to check:"
        echo "  * ls lra_benchmarks/data/listops.py   (module path correct?)"
        echo "  * pip install absl-py                 (flag parsing)"
        echo "  * available flags:"
        PYTHONPATH="$(pwd)" python -m lra_benchmarks.data.listops --help 2>&1 | head -40
        exit 1
    }

echo "--- produced ---"
ls -la "$OUT"

echo "--- schema and sizes ---"
for f in "$OUT"/*.tsv; do
    [ -e "$f" ] || continue
    printf '%s: %s rows | header: ' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    head -1 "$f"
done

echo
echo "Expect ~96k train / 2k val / 2k test, Source/Target columns, max length 2000."
echo "If filenames differ from basic_*.tsv, update --data-dir in train_array.slurm."
echo "=== done: $(date) ==="
