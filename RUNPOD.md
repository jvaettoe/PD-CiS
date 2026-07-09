# Running the detailed decomposition sweep on a GPU (runpod.io)

The faithful batch-top-k APD (and especially OracleAPD, ~C leave-one-out
forwards/step) is ~C× costlier than the old batch-shared mask. A full-config CPU
run is days-to-weeks; a single mid-range GPU does it in hours. This recipe runs
**only the decompositions** on the GPU by shipping the already-trained model
checkpoints up with the code — the models are device-agnostic and cached, so the
GPU never retrains anything.

## 0. Pre-stage the models locally (already in progress)

On this machine, warm the checkpoint cache for the whole grid (no decomposition):

```bash
./.venv/bin/python sweep.py --models-only --n 4 2 2.5 3
```

This writes 120 `checkpoints/*.pt` files. They are what you upload — the GPU
skips training entirely because `cc.load_or_train` finds them on disk.

## 1. Pick a pod

- GPU: anything with ≥16 GB (RTX 4090 / A5000 / A100 are all overkill-fast here;
  the models are tiny, the cost is the per-step batch-top-k). CPU/RAM modest.
- Image: a CUDA PyTorch template (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4`).
- Disk: 5–10 GB is plenty (checkpoints + results + figures are small).

## 2. Get the code + checkpoints onto the pod

The repo lives in `PD_CiS_CC/`. Ship the code **and** the `checkpoints/` dir
(results/ and figures/ are regenerable — no need to upload them):

```bash
# from your Mac, PD_CiS_CC/ as cwd; POD = user@host:port from the runpod SSH panel
rsync -avz --exclude '.venv' --exclude 'results' --exclude 'figures' \
      --exclude '__pycache__' ./ "$POD:/workspace/PD_CiS_CC/"
```

`checkpoints/` is git-ignored, so `rsync` of the working tree (not `git clone`)
is what carries the trained models up. Confirm on the pod:
`ls /workspace/PD_CiS_CC/checkpoints | wc -l` should show 120.

## 3. Install deps (CUDA torch — do NOT use the CPU numpy pin)

The `requirements.txt` pins `torch==2.2.2` / `numpy<2` for this Mac. On the GPU,
install a CUDA torch and let numpy float:

```bash
cd /workspace/PD_CiS_CC
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # matches the image's CUDA
pip install matplotlib einops jaxtyping tqdm pandas                    # everything except torch/numpy
```

`torch.linalg.svdvals` (the APD simplicity term) is supported on CUDA — that's
the whole reason for the GPU run; it is NOT supported on this Mac's MPS, which is
why local runs are cpu-only.

## 4. Run the full detailed sweep on the GPU

Full config — all methods incl. OracleAPD, all `d_emb` incl. 1000, full steps,
priority order n = 4 → 2 → 2.5 → 3:

```bash
python sweep.py --device cuda --n 4 2 2.5 3          # APD + OracleAPD + SPD, full grid
```

Or split cheapest-first so you get APD/SPD before the slow OracleAPD:

```bash
python sweep.py --device cuda --n 4 2 2.5 3 --methods APD SPD
python sweep.py --device cuda --n 4 2 2.5 3 --methods OracleAPD
```

Everything is resumable (`results_io.is_done` per run_id), so interrupting and
re-launching is free.

### Important: clear stale batch-shared APD rows first

The `run_id` slug does **not** encode `k` or the batch-top-k version, so any APD/
OracleAPD rows in `results/` from the *old batch-shared, fixed-k=50* code would be
silently treated as "done" and skipped. Before the detailed run, either:

- run the new APD/OracleAPD with `--force` (re-fits and row-replaces), or
- write to a clean namespace: add `--tag gpu` (outputs land in `results/gpu` and
  `figures/gpu`, leaving the old rows untouched).

SPD rows are unaffected (SPD never used top-k).

## 5. Bring the results back

```bash
rsync -avz "$POD:/workspace/PD_CiS_CC/results/" ./results/   # (or results/gpu/)
rsync -avz "$POD:/workspace/PD_CiS_CC/figures/" ./figures/
```

`results/summary.csv` is the tidy long table (one row per run×loss-term, with the
`k` column populated for APD/OracleAPD); `results/curves/<run_id>.json` holds the
full per-step curves and the identifying `config` (incl. `k`) for reconstruction.
