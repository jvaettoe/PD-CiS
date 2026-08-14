# chain — two-layer chained computation

Self-contained toy model (no imports from the root PD_CiS_CC codebase).

**Model**: 100 → 50 (ReLU) → 50 (ReLU) → 100 (linear), no biases; task y = ReLU(x) on U[−1,1] values (the main-suite CleanDataset task).

**Data**: 100 iid Bernoulli(p) features, values U[−1,1], target y = ReLU(x); p ∈ {0.001, 0.01}.

**Ground-truth question**: does the model form per-feature *chains* — a layer-1
codeword feeding a layer-2 codeword — and how many of the 100 features are
computed through the double bottleneck? Expectation to test: CiS may NOT survive
under MSE in this model (the L4 twin is the comparison).

**Training**: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 active); losses: MSE (primary) and L4 (comparison).

**Census** (`census.py`): per-feature active-conditional MSE (baseline 1/3,
computed = < baseline/10); layer-wise codeword lengths K1/K2 (>0 and τ=0.05) on
one-hot probes; K1-vs-K2 scatter.

```
python train.py && python census.py
```
