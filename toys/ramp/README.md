# ramp — saturating-ramp target (two ReLUs per feature)

Self-contained toy model (no imports from the root PD_CiS_CC codebase or
other toys; code adapted from toys/chain per the deliberate-duplication
convention).

**Task**: y = ramp(x) = ReLU(x) − ReLU(x − θ), θ = 0.5, elementwise — zero
for x ≤ 0, diagonal rise on (0, 0.5), plateau at 0.5. A parallel
implementation needs 2 ReLU units per feature, i.e. 200 units for F = 100 —
above the total ReLU budget of either architecture, so the unit requirement
itself (not just interference) pressures superposition and possibly depth.

**Models** (kaiming): ChainMLP 100 → 50 (ReLU) → 50 (ReLU) → 100 and
SingleMLP 100 → 50 (ReLU) → 100 (control). Two variants each:

- **bias-free** (toys convention) — the homogeneity control: f(v·e_j) =
  v·f(e_j) for v > 0, so no genuine plateau is possible on singleton inputs
  and both models provably collapse to a linear compromise (verified:
  transfer curves are straight lines, slope ≈ 0.6).
- **`--bias`: hidden-layer biases, output layer still bias-free** — a
  deliberate departure from the toys convention, mirroring toys/uand (where
  ReLU(x_i + x_j − 1) needs the −1): the ramp's kink at θ = 0.5 requires
  ReLU(x − θ)-type units, unrepresentable without biases. Checkpoint names
  gain a `_bias` token.

**Data**: 100 iid Bernoulli(p) features, values U[−1,1], target ramp(x);
p = 0.01 only.

**Ground-truth questions** (posed by the biased variant; the bias-free runs
are the control): does the 50-unit single layer manage the
200-ReLU-equivalent load, and what does depth buy? Do plateaus actually get
learned? Contrast on record: with the plain ReLU target at p = 0.01 the
chain's second ReLU was dispensable (identity bypass 1.13x); prediction:
with the ramp it becomes load-bearing.

**Training**: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 active) = 15,773; loss L4 only.

**Census** (`census.py`): per-feature active-conditional MSE (ramp baseline
E[f²|x>0] = θ³/3 + θ²(1−θ) = 1/6, computed = < baseline/10 = 1/60);
codeword lengths K1 (+ K2 for the chain) (>0 and τ=0.05) on one-hot probes.
Caveat: the best purely linear singleton response already has
active-conditional MSE ≈ 9.1e-3 < 1/60, so "computed" does not certify a
plateau — the transfer curves do.

**Analysis** (`ramp_analysis.py`, runs both variants; `_bias`-suffixed
outputs for the biased models): transfer curves on one-hot value sweeps
(rise slope, plateau level/slope/flatness, kink positions, homogeneity
check); layer-2 bypass ablation (identity + fitted linear/affine map,
held-out L4 ratios vs the 1.13x/1.06x relu reference and the bias-free ramp
control); unit-role census (probes v = 0.25 vs 0.75, growth ratio
h(0.75)/h(0.25); < 1.5 = saturation-implementer, ≥ 2.5 = rise-implementer,
linear = 3); single-vs-chain held-out L4 + computed counts.

```
python train.py && python train.py --bias && python census.py && python ramp_analysis.py
```
