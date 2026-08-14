# Memo: corr — correlated feature co-occurrence (ρ sweep)

## 1. Model architecture
Single-hidden-layer MLP, **100 → 50 (ReLU) → 100**, no biases (identical to the
main-suite MLP). One model per (ρ, loss); losses MSE (primary) and L4
(comparison).

## 2. Task and training
Main-suite CleanDataset task: sparse inputs with values U[−1,1], target
**y = ReLU(x)**, marginal p = 0.01 per feature. Features (2i, 2i+1) form 50
disjoint pairs; per sample and pair, with probability ρ the pair shares one
Bernoulli(p) occurrence draw, else two independent draws — marginal stays p,
within-pair occurrence correlation ≈ ρ (verified: measured 0.001 at ρ=0, 1.000
at ρ=1). ρ ∈ {0, 0.25, 0.5, 0.75, 1}. AdamW lr 3e-3, wd 1e-2, cosine, batch
2048, seed 42, steps = 10000/P(≥1 active). Census: active-conditional MSE
(baseline 1/3, computed = < 1/30), K(>0)/K(τ=0.05), and within-pair vs
between-pair codeword overlap (mean shared τ-neurons).
(First round used an easier copy task by mistake; archived in
`archive_copytask/`, superseded here.)

## 3. Findings

**Sanity anchor passes.** At ρ = 0 under L4, K(>0) = 11.5 ± 1.6 — matching the
main-suite plain MLP at p = 0.01 (11.6) — and 100/100 features computed. The
self-contained reimplementation reproduces the established regime.

**Correlation repels codes under L4.** All 100 features stay computed at every
ρ, but the within-pair codeword overlap *falls* with correlation — 1.60 (ρ=0,
≈ the between-pair level) → 1.14 → 0.10 → **0.00 at ρ ≥ 0.75** — while
between-pair overlap stays ≈ 1.1–1.6. Perfectly co-occurring features are
assigned **fully disjoint codewords**. Interpretation: features that fire
together interfere super-additively when sharing neurons, and L4 punishes those
compound errors hardest, so the code actively segregates correlated features.
The naive PD-motivated expectation (merging with ρ) is exactly wrong for L4.

**MSE does not merge — it triages and hedges.** MSE shows the familiar triage
(45/100 computed at ρ=0, K(τ) = 1.5 dedicated neurons). The apparent "merging"
signal at ρ>0 (within-pair overlap 2.1–2.5 vs 0.3–0.5 between) turns out on
closer inspection NOT to be two features sharing one mechanism: at every ρ>0,
**zero pairs have both members computed** — the overlap is between a computed
feature and its *abandoned* twin. Gain probes at ρ=1 show asymmetric pairs
(0.71 vs 0.32, with cross-gain ≈ 0.4 leaking onto the always-co-active twin):
the model computes one member and hedges the other onto the same neurons,
booking the confusion error linearly. Genuine merging was informationally
impossible anyway — values are independent even at ρ=1, so no shared code can
reconstruct both ReLUs.

**The two losses still respond to correlation in opposite directions** —
L4 segregates (overlap → 0), MSE co-locates computed features with their
hedged twins (overlap rises) — but only L4 remains full-CiS (100 features in
50 neurons) throughout; MSE's response is a refinement of triage, not merging.

**Confound + interpolation checks.** Retraining ρ=1 at the ρ=0 step count
(15,774, removing the exposure confound of the ρ-dependent step formula)
reproduces both results exactly (L4: 100/100, within-pair overlap 0.00; MSE:
25/100, within 2.78 vs between 1.01) — the step formula was not driving either
claim. An Ln mini-sweep at ρ=0 (the homogeneous main task) interpolates the
loss-shape transition outside uand: n=2 triages (45/100), n=2.5 is an
intermediate attenuated regime (0/100 strictly computed but median MSE halved,
dense K(>0)=31), n=3 and n=3.5 give full CiS (100/100, K ≈ 12-14) — the
triage→CiS transition passes through attenuation, not through a mixed
dedicated/superposed population, and its location (n between 2.5 and 3 at this
criterion) matches uand's N=50 window.

**Takeaway for the PD stage.** The ρ = 1 L4 model is the clean dissociation
target we wanted, in a stronger form than planned: 100 *parametrically disjoint*
(zero-overlap within pairs) mechanisms whose activations are *perfectly
correlated* in pairs. Any PD method that leans on co-activation statistics
(ablation-based minimality, VPD-style clustering) should be tempted to merge the
pair into one mechanism — while the weights say unambiguously that there are
two. The MSE ρ = 1 model is *not* the clean "genuinely shared mechanism"
converse it first appeared to be — it is one-computed-one-hedged triage, a
murkier ground truth. A value-correlated variant (pairs share the value draw
too, making a merged code informationally viable) would separate genuine
L4 repulsion-by-preference from repulsion-by-necessity — the natural next
corr experiment.
