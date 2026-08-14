# corr — correlated feature co-occurrence (ρ sweep)

Self-contained toy model (no imports from the root PD_CiS_CC codebase).

**Model**: MLP 100 → 50 (ReLU) → 100, no biases; task y = ReLU(x) on U[−1,1] values (the main-suite CleanDataset task).

**Data**: features (2i, 2i+1) form 50 disjoint pairs, marginal p = 0.01. Per
sample and pair, with prob ρ the pair shares one Bernoulli(p) occurrence draw
(linked), else two independent draws — marginal stays p, within-pair occurrence
correlation ≈ ρ. Values independent U[−1,1]; target y = ReLU(x). ρ ∈ {0, 0.25, 0.5, 0.75, 1}.

**Ground-truth question**: the weights should still implement 100 per-feature
mechanisms at any ρ — but does the *trained code* merge paired features
(within-pair codeword overlap rising with ρ)? ρ=0 must reproduce the main-suite
plain-MLP census (K(>0) ≈ 12 under L4 at p=0.01) — built-in sanity anchor.

**Training**: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 active); losses: MSE (primary) and L4 (comparison); one model per
(ρ, loss).

**Census** (`census.py`): features computed (< baseline/10), K(>0)/K(τ), and
within-pair vs between-pair codeword overlap vs ρ.

```
python train.py && python census.py
```
