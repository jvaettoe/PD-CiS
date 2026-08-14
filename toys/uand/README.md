# uand — universal AND: all pairwise ANDs of 15 booleans

Self-contained toy model (no imports from the root PD_CiS_CC codebase).

**Model**: 15 → N (ReLU, **with hidden bias** — ReLU(x_i + x_j − 1) needs the −1)
→ 105 (linear, no bias). Targets = all C(15,2) = 105 pairwise ANDs.

**Data**: inputs iid Bernoulli(p) booleans, p ∈ {0.1, 0.2} (mean 1.5 / 3 active).

**Ground-truth question** (Vaintrob–Mendel–Hänni computation-in-superposition):
one neuron per gate suffices at N = 105 (the no-superposition anchor); does the
model compute more than N gates when N < 105? Gates computed above the diagonal
in the gates-vs-width plot = computation in superposition.

**Training**: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 gate fires); width sweep N ∈ {10, 20, 35, 50, 105} × p × losses
{MSE (primary), L4 (comparison)} = 20 runs.

**Census** (`census.py`): a gate counts as computed only if MSE|both-active < 0.1
AND MSE|one-active < 0.1 (fires on the pair, silent on singles; always-0 baseline
on both-active = 1.0). Plus hidden-activation counts on zero/one-hot/two-hot
probes and per-gate codeword sizes (neurons whose two-hot response exceeds both
one-hot responses by τ = 0.05).

```
python train.py && python census.py
```
