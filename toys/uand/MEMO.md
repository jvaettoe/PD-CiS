# Memo: uand — universal AND in superposition

## 1. Model architecture
Single-hidden-layer ReLU MLP: **15 → N (ReLU, with hidden bias) → 105 (linear,
no output bias)**, hidden width swept N ∈ {10, 20, 35, 50, 105}. The hidden bias
is a deliberate departure from the main suite: computing AND requires a threshold
(ReLU(x_i + x_j − 1)). One neuron per gate suffices at N = 105, making that width
the no-superposition anchor.

## 2. Task and training
Inputs: 15 iid Bernoulli(p) boolean features (p ∈ {0.1, 0.2}; mean 1.5 / 3 active).
Targets: **all C(15,2) = 105 pairwise ANDs**, y_(i,j) = x_i·x_j ∈ {0,1}.
Training: AdamW lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps =
10000 / P(≥1 gate fires); losses MSE (primary) and L4 (comparison); 20 runs total.
Census criterion: a gate is *computed* iff MSE|both-inputs-active < 0.1 AND
MSE|exactly-one-active < 0.1 (fires on the pair, stays silent on singles;
always-0 baseline on both-active = 1.0).

## 3. Findings
- **Strong computation in superposition under L4**: at N = 50 the model computes
  **105/105 gates** — 2.1 gates per neuron — at both sparsities. The transition
  is sharp: N = 35 is partial (11/105 at p=0.1, 69/105 at p=0.2 — the *denser*
  input regime does better at fixed width, consistent with abandonment cost
  scaling with the gate firing rate p²; caveat: the p comparison is confounded
  by the step formula giving p=0.2 ~2× the per-gate training exposure),
  N ≤ 20 computes zero gates.
- **No computation in superposition under MSE**: N = 50 computes only 10/105;
  N = 35 only 3–7. MSE stays far *below* even the one-gate-per-neuron diagonal
  and only succeeds with fully dedicated capacity (100–105/105 at N = 105).
  MSE models suppress false fires (MSE|single ~ 6e-3–2e-2) but simply fail to
  produce the both-active response (MSE|both 0.17–0.66): they under-fire rather
  than mis-fire — the same "cheap strategy" bias as hetK's rare-population
  abandonment, here starving the *task* rather than a feature population.
- **Gate codewords are multi-neuron and overlapping**: at L4 N = 50, each gate
  activates ~11 neurons beyond its inputs' one-hot activations (gate K ≈ 11 ± 1.3
  from a 50-neuron pool, 105 gates → heavy sharing). At N = 105 the codewords stay
  multi-neuron (K ≈ 14.5 under the superadditivity criterion, which excludes
  linear neurons — an earlier max-based criterion overcounted): even with
  dedicated capacity available, training prefers a distributed implementation
  over the obvious one-neuron-per-gate one. **Gate codewords are also
  input-factorized**: at N = 50 (L4), gates sharing an input overlap 4.6
  neurons vs 1.8 for input-disjoint gates (random ≈ 2.4) — the mechanism
  ontology is closer to "15 shared input-reads composed per gate" than 105
  monolithic circuits, which redefines the stage-2 PD target.
- **Hidden bias behaves**: no neuron is active on the zero input in any N ≤ 50
  model (#h>τ at 0 = 0) — the thresholding is used as designed.
- Sanity anchor holds: N = 105 computes (nearly) all gates under both losses.

- **Ln-exponent sweep (n ∈ {2…6} at p=0.1): the CiS onset is sharp in the
  census count, gradual in the underlying error, and width-dependent.** At
  N = 50 the gates-computed count goes 3 → 27 → 88 → 105 across n = 2.5 → 2.75
  → 3 → 3.25: a transition window of barely half an exponent unit centered at
  **n* ≈ 3**. At N = 35 the window shifts to **n* ≈ 4–5** (0 computed through
  n = 3.5, 11 at n = 4, 105 at n = 5): tighter widths need a harsher loss
  exponent, i.e. the critical exponent n*(N) falls as capacity grows. Caveat on
  "sudden": the per-gate both-active MSE median declines *smoothly* with n
  (0.24 → 0.089 at N=35; 0.16 → 0.064 at N=50) — the abruptness of the count is
  the tight per-gate error distribution crossing the 0.1 criterion nearly
  simultaneously, not a discontinuity in error. The regime change is real
  (all-or-most vs almost-none), but it is a population-level threshold
  crossing, not a per-gate jump.

**Takeaway for the PD stage**: the L4 N = 50 model is a genuine
105-mechanisms-in-50-neurons object with known ground truth (one mechanism per
input pair), overlapping multi-neuron codewords, and a sharp capacity transition
at N ≈ 35–50 — a harder and cleaner CiS target than the copy task, since its
mechanisms genuinely compute a nonlinear function. The MSE variant is the natural
"no-CiS control" of identical architecture.
