# hetK — heterogeneous codeword lengths via two-population sparsity

Self-contained toy model (no imports from the root PD_CiS_CC codebase).

**Model**: MLP 100 → 50 (ReLU) → 100, no biases; task y = ReLU(x) on U[−1,1]
values (the main-suite CleanDataset task).

**Data**: features 0–49 "rare" (p = 0.001), features 50–99 "common" (p = 0.05),
values U[−1,1] when active, target y = ReLU(x).

**Ground-truth expectation** (from the main suite's K(p) dependence): one model
containing two code populations — long codewords for rare features, short for
common — i.e. heterogeneous mechanism sizes for the later PD stage.

**Training**: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 feature active); losses: MSE (primary) and L4 (comparison).

**Census** (`census.py`): per-feature active-conditional MSE (baseline 1/3,
computed = < baseline/10), K(>0) and K(τ=0.05), all split by population; figures
for sorted MSE, K histograms, W_in heatmaps, training loss.

```
python train.py && python census.py
```
