# Memo: chain — two-layer chained computation

## 1. Model architecture
Two-hidden-layer MLP, **100 → 50 (ReLU) → 50 (ReLU) → 100 (linear)**, no biases —
the main-suite MLP with a second ReLU bottleneck inserted. Per-feature ground
truth must now be a *chain*: a layer-1 codeword feeding a layer-2 codeword.

## 2. Task and training
Main-suite CleanDataset task: sparse inputs (iid Bernoulli(p), values U[−1,1]),
target **y = ReLU(x)**; p ∈ {0.001, 0.01}. AdamW lr 3e-3, wd 1e-2, cosine,
batch 2048, seed 42, steps = 10000/P(≥1 active) (~105k / ~15.8k); losses MSE
(primary) and L4 (comparison). Census: active-conditional MSE (baseline 1/3,
computed = < 1/30); per-feature codeword lengths at both layers from one-hot
probes — K1 (w1 preacts) and K2 (h1·w2 preacts), >0 and τ=0.05 conventions.
(First round used an easier copy task by mistake; archived in
`archive_copytask/`, superseded here.)

## 3. Findings

**CiS survives the double bottleneck under L4 — fully.** 100/100 features
computed at both sparsities (median MSE 1.2–1.8e-2), so L4 threads all 100
ReLUs through *two* consecutive 50-neuron bottlenecks. The chain structure is
asymmetric: layer-1 codes stay sparse and suite-like (K1(τ) = 13.2 at p=0.001,
9.7 at p=0.01 — close to the single-layer K(p) values), while layer-2 codes are
much denser (K2(τ) = 23.6 / 30.9; K2(>0) = 32–41 of 50). The model compresses
into a sparse code first, then re-expands into a dense second-layer
representation before decoding.

**CiS does not survive under MSE — as predicted.** The expectation that MSE
would fail here is confirmed: only 44/100 (p=0.001) and 46/100 (p=0.01)
features are computed — the familiar ~min(N, F) triage from hetk/corr, now
through two layers — with small codes (K1(τ) = 3–5.5, K2(τ) = 4–5) for the
survivors and the rest left near the ignore baseline (median MSE ≈ 0.10 ≈
population mix of computed and abandoned).

**The second bottleneck is not what breaks MSE.** MSE's 44–46 computed features
match its single-layer behavior (45/100 in corr at ρ=0, same p=0.01 task) —
depth neither helps nor further hurts; the triage is a property of the loss,
not of the depth. Conversely, L4's full-CiS also carries over from one to two
layers without loss.

**Layer 2 is mostly linear at p = 0.01 (review check).** On active samples only
~10% of layer-2 preactivations are ReLU-clipped and a linear fit h2 ≈ A·h1
reaches R² = 0.992 — the p=0.01 "chain" is largely compress → (≈linear
re-expand) → decode, so its cross-layer PD difficulty partially collapses
(W2 nearly foldable). The p = 0.001 model is the genuinely nonlinear chain
(23% clipped, R² = 0.95) and should be the stage-2 cross-layer target.
Bypass ablation confirms both verdicts: dropping the layer-2 ReLU entirely
costs only +12% held-out L4 at p=0.01 (100/100 still computed; median
per-feature MSE even improves) but 3.2× at p=0.001. The p=0.001 nuance:
the bypass there *improves* the median per-feature MSE while tripling L4 —
the second ReLU's functional role is suppressing rare large interference
events (L4's tail), not improving typical reconstruction. The full p-sweep
(p ∈ {0.001…0.05}, results/bypass_sweep.md + figures/bypass_sweep.png) shows a
smooth crossover, not a sharp transition: the identity-bypass cost rises
monotonically as sparsity increases (1.00× at p ≥ 0.02, 1.13× at 0.01, 1.82×
at 0.005, 2.51× at 0.002, 3.23× at 0.001), tracking the layer-2 clipping
fraction (0.004 → 0.23) closely. The second nonlinearity is functionally dead
for p ≥ 0.02 and earns its keep gradually below p ≈ 0.01 — consistent with its
tail-cleanup role: sparser inputs mean rarer but relatively larger interference
events, exactly the regime L4 weights hardest.

**Designed-reference comparison (stacked codes).** Fitting a
decode-re-encode stack of two biregular codes (K1/K2 matched per p, 4-scalar
L4 fit, pinv decoders) gets nowhere near the trained chain: 20-36x worse L4
across p, and a plain code+least-squares-linear decoder beats the stacked
construction while still sitting ~10x above trained. Grafting the trained
model's fitted layer-2 linear map A onto a designed code is catastrophic
(~1e-1 L4) — A is co-adapted to the trained layer-1 basis. Unlike the
single-layer suite (designed codes within ~2x of trained), the chain's
solution is NOT well-approximated by stacked feature codes: its second layer
is not a re-encoded feature code. Caveat: the 4-scalar fit is single-start
(shallow-valley risk); multi-start refinement pending before strong claims.

**Takeaway for the PD stage.** The L4 chain models provide the first
ground-truth *cross-layer* PD target of the suite: 100 per-feature mechanisms,
each a (sparse layer-1 codeword) → (dense layer-2 codeword) chain, with the
layer-wise K statistics known per feature. The open stage-2 questions it was
built for are now well-posed: does a PD method stitch the two layer codewords
of a feature into one mechanism, and does the dense K2 layer decompose
per-feature at all?
