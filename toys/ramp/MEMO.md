# Memo: ramp — the diagonal-step target (first genuinely two-ReLU task)

## 1. Model architecture
Two architectures, trained on the same target: **ChainMLP** 100 → 50 (ReLU) →
50 (ReLU) → 100 and **SingleMLP** 100 → 50 (ReLU) → 100, both with linear
bias-free outputs. Each exists in two variants: the suite-standard **bias-free**
form, and a **hidden-bias** form (deliberate convention departure, as in
toys/uand: the target's kink at θ requires ReLU(x − θ)-type thresholds).

## 2. Task and training
Sparse inputs (Bernoulli(p = 0.01), values U[−1,1]); target the **saturating
ramp f(x) = ReLU(x) − ReLU(x − θ), θ = 0.5** — zero for x ≤ 0, diagonal rise
to 0.5, then plateau. Two kinks ⇒ two ReLU units per feature ⇒ a 200-unit
parallel requirement against a 50(+50)-unit budget: the first target in the
suite where the *unit count*, not just interference, forces superposition.
L4 loss, AdamW 3e-3/wd 1e-2/cosine, batch 2048, 15,773 steps, seed 42.
Census baseline: always-0 active-conditional MSE = E[f²|x>0] = 1/6 (analytic,
Monte-Carlo verified); "computed" < 1/60 (caveat: a linear response passes
this threshold, so transfer curves — not the census — carry the verdict here).

## 3. Findings

**Act 1 — the bias-free family provably cannot learn this target.** Bias-free
ReLU networks are positively homogeneous (f(v·x) = v·f(x), v > 0), so their
one-hot transfer curves are exactly linear: no plateau can form at any width
or depth. Measured: deviation from linearity ~2e-7, 0/100 kinks, zero
saturating units; both architectures converge to the same L4-optimal linear
compromise ≈ 0.6·ReLU(x), and the layer-2 bypass stays at the dispensable
1.13× because the effective task collapses back to scaled-ReLU. Lesson for
the suite: **check target–family representability before interpreting any
census** — the 100/100 "computed" count here certified nothing.

**Act 2 — with hidden biases, the chain learns the ramp and depth becomes
load-bearing; the single layer cannot.**
- The biased **chain** genuinely saturates: 99/100 features with a kink
  located at 0.505 [0.495, 0.525], plateau slope 0.206 (vs 0.598 linear
  control), rise slope 0.733. The biased **single** layer does not: plateau
  slope 0.567 — shape-indistinguishable from the linear control — despite
  having ~22 saturating *units* per feature. Unit-level saturation fails to
  compose into functional saturation in one layer at this width.
- **Division of labor across depth**: layer 1 carries the rise (7.2 riser
  units per feature, ~0.9 saturators), layer 2 is a **shared saturation
  bank** — 49.5 of 50 units saturating for every feature. The second layer's
  role is a collective clipping mechanism, not per-feature codewords.
- **The layer-2 bypass flips from dispensable to load-bearing**, as predicted
  on record: identity bypass costs 4.43× (vs 1.13× for both the ReLU target
  and the bias-free control at the same p), and even a fitted affine
  replacement costs 1.56× — the second nonlinearity now performs
  per-sample computation no linear map reproduces.
- **Depth wins under the 2-ReLU load**: biased chain held-out L4 3.88e-06 vs
  single 4.97e-06 (the control ordering was reversed). The 50-unit single
  layer does not manage the 200-unit-equivalent requirement.

**Takeaway for the PD stage.** The biased ramp chain is the suite's first
genuine multilayer mechanism object: ~100 per-feature rise mechanisms in
layer 1 chained through a *shared* 50-unit saturation bank in layer 2. The
correct cross-layer ground truth is therefore NOT 100 independent two-layer
chains but "100 read/rise mechanisms + 1 collective saturation mechanism" —
directly testing whether PD methods can represent a mechanism shared by all
features (the hard partition failure mode from the MDL-clustering result, now
as ground truth rather than pathology).
