"""
Build the run manifest: one CSV row per (design, variant, seed) cell.

WHY A MANIFEST
--------------
The Slurm array index is just an integer. Mapping it to a configuration inside the job
script means the mapping lives in bash and cannot be inspected, diffed, or re-run. A CSV
written up front means:

  * you can see exactly what will run before submitting `wc -l` worth of GPU hours;
  * a failed task can be re-run by index, reproducing the identical configuration;
  * the analysis stage can join results back to designs without re-deriving anything.

DESIGN vs SEED (the distinction the whole study rests on)
--------------------------------------------------------
A *design* is a meaningful configuration -- depth, width, lr, batch. It is what we
generalise over, and it is the unit of observation in the Friedman test.
A *seed* is pure noise control. Seeds are averaged within a design x variant cell; they
are NOT independent observations. Treating them as such would inflate n and make the
statistics wrong.

COMPLETE MATRIX
---------------
Every design runs under every variant. The Demsar pipeline (Friedman omnibus, then
post-hoc) requires a complete design x variant matrix -- a missing cell drops the whole
design from the analysis. This is why the design space must be shared across arms rather
than capped per arm.

DESIGN SPACE FLOOR
------------------
d_model starts at 256, not 128. Linformer's E/F projections cost a fixed 2*k*max_len
parameters (1,024,000 at k=256, max_len=2000) regardless of depth, because of layerwise
sharing. Against a d_model=128, depth=2 model that is 2.95x the total parameter count;
against d_model=256, depth=4 it is 1.28x. Raising the floor closes most of the gap
without modifying any arm, without shrinking Linformer's k (which would handicap the
method we are trying to measure), and without breaking the complete matrix.

Run `python count_params.py` to regenerate those figures if the space changes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

# Arms. Order is fixed so array indices stay stable across regenerations.
VARIANTS = ["vanilla", "flash", "linformer", "linear", "sparse"]

# Design space. See the docstring on the d_model floor.
DESIGN_SPACE = {
    "d_model": [256, 384],
    "depth": [4, 6],
    "lr": [3e-4, 1e-3],
    "batch_size": [32],
}

SEEDS = [0, 1, 2]

# Held constant across every run -- part of the fixed frame, not the design space.
FIXED = {
    "max_len": 2000,
    "n_heads": 8,
    "linformer_k": 256,
    "linformer_sharing": "layerwise",
    "attn_dropout": 0.0,   # ADR-001
    "dropout": 0.1,
    "epochs": 20,
    "task": "listops",
}


def build_designs():
    keys = sorted(DESIGN_SPACE)
    for i, values in enumerate(itertools.product(*(DESIGN_SPACE[k] for k in keys))):
        design = dict(zip(keys, values))
        design["d_ff"] = 4 * design["d_model"]
        yield i, design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cluster/runs.csv")
    args = parser.parse_args()

    designs = list(build_designs())
    rows = []
    for (design_id, design), variant, seed in itertools.product(designs, VARIANTS, SEEDS):
        row = {
            "run_id": f"d{design_id:03d}_{variant}_s{seed}",
            "design_id": design_id,
            "variant": variant,
            "seed": seed,
            **design,
            **FIXED,
        }
        rows.append(row)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_designs, n_variants, n_seeds = len(designs), len(VARIANTS), len(SEEDS)
    print(f"designs        : {n_designs}")
    print(f"variants       : {n_variants}  {VARIANTS}")
    print(f"seeds          : {n_seeds}")
    print(f"total runs     : {len(rows)}  ({n_designs} x {n_variants} x {n_seeds})")
    print(f"analysis cells : {n_designs * n_variants}  (seeds averaged within cell)")
    print()
    print(f"wrote {args.out}")
    print(f"submit with: sbatch --array=1-{len(rows)}%8 cluster/train_array.slurm")
    print()
    print("design space:")
    print(json.dumps(DESIGN_SPACE, indent=2))


if __name__ == "__main__":
    main()
