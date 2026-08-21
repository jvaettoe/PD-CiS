# RunPod runbook: trained VPD over the (F, N) grid

Trains the full VPD recipe (our port, tuned pricing imp 1e-4 / pnorm 2.0,
lr 3e-4, 25k steps, C = F + N) on every fitted grid cell, and scores the
learned solution's feature- vs neuron-alignment per matrix.

## Prerequisites

The repo checkout must include `results/fn_sweep/kfit/` (the per-cell K*
records — models are rebuilt deterministically from these; nothing else is
needed). Push the repo including results, or rsync that directory.

## Setup (once per pod)

```bash
git clone <repo> && cd PD_CiS_CC          # or rsync the working tree
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scipy tqdm jaxtyping
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

## Run

```bash
# everything (cheapest cells first, resumable — re-running skips done cells):
python -m biregular.fn_vpd_train --train-grid

# or restrict / parallelize across pods by cell:
python -m biregular.fn_vpd_train --train-cell 100 50
python -m biregular.fn_vpd_train --train-grid --max-f 300
```

Outputs land in `results/fn_sweep/vpd/F{F}_N{N}_K{K}_s{seed}.{pt,json}` —
checkpoint (pd_state + record) and a JSON record with: eval losses (same
harness as the oracle grid), per-matrix alignment (feature/neuron membership
scores, coverage, verdict), gate sign-awareness (+/- one-hot probe sums), and
VPD-native ci_stats. One-seed default (`--seed 42`); pass others for repeats.

## Bring results home + analyze

```bash
rsync -av <pod>:.../results/fn_sweep/vpd/ results/fn_sweep/vpd/
python -m biregular.fn_vpd_train --map        # vpd_preference_map.png
python -m biregular.fn_vpd_train --alignment  # recompute metrics from ckpts
```

`figures/fn_sweep/vpd_preference_map.png` renders next to the oracle
`basis_preference_map.png` for the trained-vs-oracle comparison. Verdicts:
feature / neuron / mixed (neither basis — the "third ontology" case) /
collapsed (alive components below half the smaller basis).

## Expected cost

Small models (up to 1000x500, C up to 1500): ~2-6 min per cell on any modern
GPU at 25k steps; full 64-cell grid in a few GPU-hours. CPU fallback works
(`--device cpu`) but is ~10-20x slower on the large cells.
