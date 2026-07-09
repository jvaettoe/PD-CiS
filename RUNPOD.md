# Running the detailed decomposition sweep on a GPU (runpod.io)

The faithful batch-top-k APD (and especially OracleAPD, ~C leave-one-out
forwards/step) is ~C× costlier than the old batch-shared mask. A full-config CPU
run is days-to-weeks; a single mid-range GPU does it in hours.

The device audit confirmed the whole path (training **and** decomposition) is
CUDA-clean: `load_or_train` puts both the model and the dataset on `--device`,
CPU-saved checkpoints map onto CUDA at load, and every tensor in `pd.py` is
created on the right device. So the GPU can do **either or both** of the two
jobs, and `cc.load_or_train` decides per-model from the cache:

- **Cached models** you upload → loaded from disk, reused, decomposed.
- **Uncached models** → **trained on the GPU on the spot** (tiny models, fast),
  then decomposed.

That means you do **not** have to finish the local pre-cache before going to the
GPU — ship whatever checkpoints exist and let the GPU fill the gaps (Strategy A,
§4). The local `--models-only` pre-caching is just a free head-start we get while
we're on CPU anyway.

## 0. (Optional) Pre-stage models locally for a head-start

On this machine, warm the checkpoint cache for the whole grid (no decomposition):

```bash
./.venv/bin/python sweep.py --models-only --n 4 2 2.5 3
```

This writes up to 120 `checkpoints/*.pt` files. Uploading them lets the GPU skip
that training. It is **not required** to be complete — the GPU trains any missing
model itself (Strategy A). Skip this step entirely for Strategy B.

## 1. Pick a pod

- GPU: anything with ≥16 GB (RTX 4090 / A5000 / A100 are all overkill-fast here;
  the models are tiny, the cost is the per-step batch-top-k). CPU/RAM modest.
- Image: a CUDA PyTorch template (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4`).
- Disk: 5–10 GB is plenty (checkpoints + results + figures are small).

## 2. Get the code + checkpoints onto the pod

`/workspace` is the pod's **persistent volume** (a RunPod convention, not a name
you chose) — it survives stop/start, so put the project there:
`/workspace/PD_CiS_CC/`. Ship the code **and** the `checkpoints/` dir (results/
and figures/ are regenerable — no need to upload them).

RunPod's Connect panel gives an SSH command like
`ssh root@194.68.x.x -p 22012 -i ~/.ssh/id_ed25519`. rsync needs the host and the
**port** given separately (a bare `host:port` token does NOT work — the colon is
rsync's host/path separator). Take HOST and PORT from that command:

```bash
# from your Mac, PD_CiS_CC/ as cwd
HOST=root@194.68.x.x        # from the Connect panel
PORT=22012                  # the -p value from the Connect panel
rsync -avz -e "ssh -p $PORT" \
      --exclude '.venv' --exclude 'results' --exclude 'figures' \
      --exclude '__pycache__' ./ "$HOST:/workspace/PD_CiS_CC/"
```

`checkpoints/` is git-ignored, so `rsync` of the working tree (not `git clone`)
is what carries the trained models up. Check the count on the pod with
`ls /workspace/PD_CiS_CC/checkpoints | wc -l` — however many you have is fine
(the GPU trains the rest). For Strategy B (GPU does everything) you can add
`--exclude 'checkpoints'` and upload no models at all.

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

This one command **is** both Strategy A and B: for each model it loads the
uploaded checkpoint if present, else trains it on the GPU, then decomposes. So
whether you uploaded all 120, some, or none, this fills in whatever's missing.

Or split cheapest-first so you get APD/SPD before the slow OracleAPD:

```bash
python sweep.py --device cuda --n 4 2 2.5 3 --methods APD SPD
python sweep.py --device cuda --n 4 2 2.5 3 --methods OracleAPD
```

Everything is resumable (`results_io.is_done` per run_id), so interrupting and
re-launching is free.

### Two gotchas from the device audit

- **Do NOT pass `--no-train` on the GPU.** That flag skips uncached models (it
  exists only so a local decomposition run won't fight the background trainer).
  On the GPU you *want* it to train the gaps, so leave it off.
- **`--device` defaults to `cpu`.** If you forget `--device cuda`, the sweep runs
  on the pod's CPU and you pay for an idle GPU. Verify with the pre-flight below.

### Pre-flight (run once, before the sweep)

```bash
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

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
rsync -avz -e "ssh -p $PORT" "$HOST:/workspace/PD_CiS_CC/results/" ./results/   # (or results/gpu/)
rsync -avz -e "ssh -p $PORT" "$HOST:/workspace/PD_CiS_CC/figures/" ./figures/
```

`results/summary.csv` is the tidy long table (one row per run×loss-term, with the
`k` column populated for APD/OracleAPD); `results/curves/<run_id>.json` holds the
full per-step curves and the identifying `config` (incl. `k`) for reconstruction.
