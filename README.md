# Cluster runbook — mscluster

Ordered. Do not skip ahead; each stage's output is the next stage's precondition.

Guidelines this follows (mscluster Community Guidelines, Feb 2024):
- Never run compute on the login node (§2.2.5). Everything below that costs CPU/GPU
  goes through `sbatch`, including the flash-attn compile.
- Prefer `sbatch` over interactive `srun` (§6.2) — load shedding kills interactive jobs.
- Start on `stampede`, escalate to `bigbatch` only when needed (§2.2.7).
- MaxTime = 4320 min (72h) on every partition.
- Home dir is 50GB, and sideloading into it is discouraged (§8.3). Data goes to
  `/datasets/fmnisi/`.
- Acknowledge the cluster in the write-up (§2.3) — text at the bottom of this file.

---

## Stage 0 — access

```bash
ssh fmnisi@146.141.21.100
```

Access was previously blocked pending an MSS form. If this fails, open ONE ticket to
`support@wits-mss.supportsystem.com`, subject `TWK HPC Query`, from your Wits address.
Do not email staff directly (§2.2.1).

Sanity checks once in:

```bash
sinfo -o "%P %a %l %D %G"     # partitions, timelimits, and whether GRES/gpu is configured
squeue -u fmnisi
df -h /home-mscluster/fmnisi
```

Note whether the `%G` column shows `gpu:...`. If it does, every GPU job below needs
`#SBATCH --gres=gpu:1`. If it shows `(null)`, the partition hands you the whole node and
the flag is unnecessary. **Check this before submitting anything** — it is the single
most common reason a job runs on CPU and silently takes 40x longer.

## Stage 1 — environment

Submit, don't run on the login node:

```bash
cd /home-mscluster/fmnisi/attn_rct
sbatch cluster/00_setup_env.slurm
squeue -u fmnisi          # watch
cat logs/setup_*.out      # read when done
```

This installs Miniconda into your home dir, creates the `attnrct` env with PyTorch on
CUDA 12.0, and attempts flash-attn (prebuilt wheel first, source build as fallback).
The source build is slow (30–90 min) and memory-hungry, which is exactly why it is a
batch job on `bigbatch` rather than something you run at the login prompt.

## Stage 2 — verify FA-2 actually engages

```bash
sbatch cluster/01_probe.slurm
cat logs/probe_*.out
```

**This is a gate, not a formality.** The output must contain:

```
capability: sm_86
backend: flash-attn library varlen FA-2 (sm_86) -- VERIFIED
path taken: flash_varlen_fa2
```

If it says `sdpa_fallback_NOT_fa2` or reports anything below sm_80, stop. Every
efficiency number for the baseline would be measuring a different kernel. `biggpu`
(RTX 8000, Turing sm_75) cannot run FA-2 either — `bigbatch` is the only option.

## Stage 3 — official LRA data

```bash
sbatch cluster/02_get_data.slurm
```

Downloads and extracts ListOps into `/datasets/fmnisi/lra/listops/`, producing
`basic_train.tsv`, `basic_val.tsv`, `basic_test.tsv` with the `Source`/`Target` schema
your existing loader already expects.

Until this lands, every accuracy number you have is from the community mirror
(`fengyang0317/listops-1000`), which is raw generator output filtered to ~21% survival
at max_len=2000. Those numbers are not comparable to published LRA results and must not
be reported as such.

## Stage 4 — smoke test on stampede

Cheap plumbing check before spending `bigbatch` time. Per §2.2.7, this is what
`stampede` is for. It CANNOT run FA-2 (GTX 1060, sm_61) — it is only proving that the
env loads, the data reads, and a checkpoint writes and resumes.

```bash
sbatch cluster/03_smoke.slurm
```

## Stage 5 — the grid

```bash
# 1. build the manifest (cheap, login node is fine — it just writes a CSV)
python cluster/make_manifest.py --out cluster/runs.csv

# 2. check the size before you submit
wc -l cluster/runs.csv

# 3. submit, throttled to 8 concurrent so you don't flood the cluster (§2.2.8)
sbatch --array=1-$(( $(wc -l < cluster/runs.csv) - 1 ))%8 cluster/train_array.slurm
```

Monitoring and control:

```bash
squeue -u fmnisi
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
scancel <JOBID>              # whole array
scancel <JOBID>_7            # one task
scancel --user=fmnisi        # everything
```

---

## W&B

Never commit the key. On the login node, once:

```bash
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc_secrets
chmod 600 ~/.bashrc_secrets
echo '[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets' >> ~/.bashrc
```

The job scripts source this and pass it through. If a compute node has no outbound
network, set `WANDB_MODE=offline` in the job script and run `wandb sync logs/wandb/*`
from the login node afterwards — the probe job reports which case you're in.

## Required acknowledgement (§2.3)

> Computations were performed using High Performance Computing infrastructure provided
> by the Mathematical Sciences Support unit at the University of the Witwatersrand,
> Johannesburg.
