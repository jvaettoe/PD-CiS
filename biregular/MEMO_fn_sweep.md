# Memo: the (F, N) sweep program — designed code families and trained MLPs
*(2026-08-21 – 2026-08-23; results/fn_sweep/, results/mlp_sweep/, figures/…)*

## 1. Models

**Designed biregular codes** (three constructions, per-family subfolders):
MLP F → N (ReLU) → F, no biases, transpose decoder w_out = w_inᵀ,
w_in = a·M + b·(1−M) with only (a, b) fitted (L4, p = 0.001). Families:
- `greedy` — least-used-neuron near-biregular (this repo; no divisibility
  requirement). Grid: 8×8 base ∪ XL up to F = N = 10,000 (224 cells).
- `dfh` — Ferreira da Silva–Heimersheim's own stub-matching exact-biregular
  construction (their codes.py, verbatim in core/codes.py; requires N | F·K).
  Ratio grid F = r·N (r ∈ {2..40}, N ∈ {20..2000}; 70 cells) so every K is
  admissible and K* is unconstrained.
- `dfh_om` — dfh + their reduce_overlap swap pass (800k-swap budget,
  op-scaled; 57 cells with F·N ≤ 10⁶).

**Trained MLPs** (results/mlp_sweep/): same architecture, weights *trained*
(L4, AdamW 3e-3/wd 0.01/cosine, 20k steps, batch 4096, seed 42) on
CleanDataset(p). Grid: F ∈ {40..1000} × N ∈ {20..500} (N ≤ F) ×
p ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} — 495 models
(includes dFH's canonical p = 0.02). Census (features computed, K-dist,
held-out L4) stored per model; all verdicts must be read against it.

## 2. What is measured per cell

1. **K\*** — L4-optimal codeword length (argmin over log-spaced admissible K
   with refinement + blow-up early stop; per-K (a, b) rows stored).
2. **Oracle 2×2 basis preference** — the four (read, write) ∈
   {feature, neuron}² ideal PDs inserted into the exact VPD objective
   (stoch + PGD recon + importance, tuned pricing imp 1e-4/pnorm 2.0) with
   oracle gates (feature piece: x_j ≠ 0; neuron piece: h_n ≠ 0); mixed combos
   zero-padded to C = max(F, N). Cross-validated against Goodfire's own loss
   implementation (identical ordering and margins on the checked cell).
3. **Exact-solo leak/attenuation** — full model vs per-feature solo pieces on
   one-hot probes: attenuation = on-target error, leakage = off-target power;
   ratio per MSE and L4.
4. **Trained VPD** (greedy base grid 64 cells; mlp grid in progress) — full
   VPD training (C = F + N, warm-restart blocks until alive counts settle,
   ≤150k steps) + two-sided basis-membership alignment, gate sign-probes,
   alive counts.

## 3. Findings

### 3.1 Optimal fanning K*
K* ∈ [3, 6] over most of the compressed region, rising with cell scale
(K* = 12 at (4000, 1000); dfh reaches K* ≈ 10–23 in the F ≥ 8000 rows);
K* = 1 exactly (a = 1, b = 0, model bit-exact) whenever N ≥ F. K* is NOT a
function of F/N alone. The census "computed" count at threshold 1/30
undercounts designed codes near the loss-optimal self-attenuation (K·a² ≈
0.58–0.79) — a threshold artifact, not a computation failure.

### 3.2 Oracle basis preference: the two-mechanism map and the flip band
- **Read side (W_in)**: feature basis below the diagonal (margins ~1.8–1.9×),
  flipping to neuron basis in the K* = 1 region above it. Both read bases are
  recon-free everywhere, so the read verdict is purely an importance
  (gate-firing-rate) contest: feature gates fire at pF; neuron gates at ~N/2
  per event (b-background) in compressed cells but at pF/2 (ReLU
  sign-awareness, exactly factor 2 for U[−1,1] values) in dedicated cells —
  hence the flip at the diagonal, with margins exactly 3/2 (identity:
  (1+½)/(½+½)) in the K* = 1 cells.
- **Write side (W_out)**: neuron basis in high-compression cells (interference
  ownership: feature write pieces pay un-cheatable recon for masked leak;
  margins up to ~7.6×), BUT **flips to feature basis at scale** (greedy: 75 of
  224 cells; dfh: 46 of 70): once p·F ≳ 1, essentially every neuron fires on
  every sample, neuron-gate importance ≈ N·coeff overwhelms the (small)
  interference-recon cost, and per-feature writes become objectively cheaper
  (e.g. (10000, 4000): feature-write 3.0e-3 vs neuron-write 0.20, all
  importance). Caveats: boundary is pricing-dependent (raw terms stored for
  re-pricing) and confounded with the fixed-p convention (p·F grows with F).
- Trained VPD at small-margin flip cells still lands neuron-write (soft gates
  discount importance); the large-margin flip cells have no trained runs yet.

### 3.3 Ladder theory (complete chain, no fitted constants)
For one-hot probes (v > 0) the forward pass reduces exactly to the overlap
ladder: on-target K·a², leak on i: m_ij·a² + (K−m_ij)·a·b.
- **Ladder with actual overlaps**: matches measured leak/atten at 1.0000× in
  every nontrivial cell of every family (one identified exception:
  (10000, 6500) greedy, where fitted b > 0 breaks the ReLU-clipping premise).
- **Hypergeometric ensemble** (m ~ Hyp(N, K, K); instance-free): within
  0–17%; median deviation **greedy 1.023 ≈ dfh 1.024** (both ≈ random),
  **dfh_om 1.098** (overlap-min leaks ~10% less than random). Construction
  ordering: (greedy ≈ dfh ≈ random) > overlap-min.
- **Predicted (a\*, b\*)** = argmin of the TWO-BRANCH one-hot L4 ladder loss
  (v > 0 branch + v < 0 branch, where off-code neurons fire at b·v and
  discipline |b|): matches learned values to a ~2%, b ~4.5% mean. The
  one-branch version mispredicts b up to 3× — the negative-value branch is
  essential. Fully theoretical leak/atten curves vs K, F, N follow with no
  data input; reduced form ratio_p ≈ C·(F−1)·(K²/N)·[a^2p/(1−Ka²)^p] with
  computable C (R² 0.991 MSE with all exponents = 1). Family shapes: fixed-N
  ≈ linear in F; fixed-F quadratic (μ² rung); iso-(N, K) families are NOT
  iso-s families — the optimizer re-tunes a along them, cancelling or
  reversing the (F−1) growth (e.g. N=100/K*=5: measured ratio falls 23→17 as
  F rises 150→300).

### 3.4 Trained VPD on the designed codes (greedy base grid, 64 cells)
- Write side: neuron-aligned in 41 cells, never feature-aligned; K* = 1 cells
  learn perfectly sign-aware gates (probe ratio 0.00), as the oracle predicts.
- Read side: complete per-feature ontology at moderate compression
  ((100, 50): 100 one-hot reads covering 100/100 features, 93% of read mass)
  but a genuine **ontology transition at high compression**: at (200, 20)
  only 9/200 features keep dedicated reads (8% of mass); 74% of mass moves to
  ~O(N) dense/roster reads. "Mixed" verdicts at F/N ≳ 7 are this transition,
  not noise.
- **Read-side sharding**: with slack capacity (C = F + N) the read side
  saturates every available slot (alive = C) with mass-shards of the feature
  reads; stochastic masking *rewards* splitting (variance reduction), and the
  opposing importance differential (~p·coeff) is negligible. Neither 10×
  importance nor longer training removes shards; C = F exactly prevents them
  but *degrades* the recovered ontology (coverage 34/40 vs 40/40). Shards are
  not ablatable for free (greedy hard-ablation: no zero-cost subset; cost
  ∝ mass-share², cumulative ~14% of output power for 20 removals) but are
  functionally 40 units (PR of mean CI ≈ F; coverage F/F). Mass-alive counts
  measure C, not the ontology — use effective (PR/coverage) counts.
- Trained VPD also leans on the delta component (spillover entries up to
  ~5e-2): learned solutions do not exactly tile the weights.

### 3.5 Few-hot / all-hot composition (original (100, 50) code)
- The JOINT k-component forward (active components together) reproduces the
  full model's on-target responses exactly at every k (shared hidden state)
  and writes zero off-target: solos compose exactly under the shared ReLU.
  Summed *independent* solos instead overestimate on-target by the pairwise
  cross-talk (m a² + (K−m)ab per partner) — the multi-active term behind the
  two-branch b fix.
- Loss vs k: component-forward loss (pure collision) grows ~k²; full-model
  extra loss (leak) grows ~k; their ratio decays from the k = 1
  leak/attenuation value to 1 by k ≈ 40 (MSE) / k ≈ 10–15 (L4). Interference
  is a sparse-regime phenomenon; under the training loss it is negligible
  beyond ~10 active features.
- All-hot (x = 1): every preactivation d·a + (F−d)·b ≈ −0.64 < 0 → the model
  outputs exactly 0. The interference-cancelling b kills the network in the
  dense regime.

### 3.6 Trained-MLP sweep (first results; VPD stage in progress)
Oracle 2×2 on trained weights (census-gated to cells with ≥ F/2 computed):
reads feature-basis wherever CiS exists; write side leans **feature** at low p
(p = 0.001: 6 feature / 2 neuron / 1 tie) — diverging from designed codes —
and neuron at p = 0.5. Flagged before trusting: census threshold sensitivity
and possible undertraining of large cells at 20k steps.

## 4. Methods notes / cautions
- dfh construction: their stub-repair loop is single-core O(F·N)-per-collision
  — pathological at large K candidates (K = 63 at (8000, 2000) ⇒ hours);
  cells at that scale carry coarse-only K rows (documented per record).
- Overlap-min at 800k swaps is O(F·N)/swap; capped at F·N ≤ 10⁶.
- Pod-side eval RNG (CUDA) differs from local (CPU) at the ~1% pool-noise
  level; per-record configs identify provenance.
- p is fixed per sweep while F varies ⇒ p·F (collision rate) varies across
  each grid; the write-flip band partially reflects this.
