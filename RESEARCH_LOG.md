# Research log

Chronological overview of experiments in this repo. Per-model memos live next to
each model (`toys/<name>/MEMO.md`); this file records what was done and when.

## 2026-08-08 — Multi-seed statistics for the dFH-equivalence transition
- Built `embed_transition_stats.py` (train / battery / aggregate phases).
- Trained 80 new checkpoints (9 seeds 42–50 × {MLP, EmbedMLP d_emb ∈ 20…1000}
  at p=0.01), ran the dFH battery per seed with per-seed designed biregular
  codes, aggregated mean ± std.
- Results: all single-seed transition conclusions replicate with tight bands;
  trained-vs-designed L4 parity exactly at d_emb=40 (0.99±0.03); row-degree-σ
  peak at d_emb=N=50 confirmed; seed 42 typical. Convention fix: K_d = half-up
  round of mean K(>0). Outputs: `results|figures/embedded/equivalence_p0.01/stats/`.

## 2026-08-09 — Four new ground-truth toy models (`toys/`)
- Implemented four fully self-contained codebases (no root imports):
  `toys/hetk` (two-population sparsity), `toys/corr` (50 correlated pairs, ρ
  sweep), `toys/uand` (all 105 pairwise ANDs of 15 booleans, width sweep),
  `toys/chain` (two hidden ReLU layers). Loss is a sweep axis everywhere:
  MSE primary, L4 comparison. Stage 1 = feature-computation census (CiS check);
  PD comes later.
- **uand census (valid, task correct by design)**: L4 N=50 computes 105/105
  gates (2.1 gates/neuron — strong CiS); MSE never exceeds the one-gate-per-
  neuron diagonal (10/105 at N=50), only works with dedicated capacity N=105.
  Memo delivered: `toys/uand/MEMO.md`.
- hetk follow-ups implemented (wmse importance-weighted control, p_rare/n_rare
  abandonment-boundary sweep, `code_analysis.py` for code anatomy +
  2×2 interference).

## 2026-08-09 — Task correction in hetk/corr/chain (IMPORTANT)
- The corr ρ=0 sanity anchor failed (K(>0) 49.8 vs expected ≈12), which exposed
  a task divergence: the toy generators used values U[0,1] with y = x (easy copy
  task), while the main suite's CleanDataset uses values **U[−1,1] with
  y = ReLU(x)**. Fixed all three `data.py` generators to the ReLU task; census
  conventions unchanged (active mask x>0, baseline 1/3).
- First-round hetk/corr/chain results (incl. the initially reported hetk
  abandonment + code-anatomy findings and the corr merging pattern) are
  **copy-task artifacts pending re-verification**; archived under
  `toys/<name>/archive_copytask/`. All hetk/corr/chain trainings relaunched on
  the corrected task. uand unaffected.
- Also launched (user request): uand **Ln-exponent sweep** — generic l<n> losses,
  n ∈ {2.25…3.5, 5, 6} at N ∈ {35, 50}, p=0.1, to test whether CiS onset in the
  loss exponent is gradual or sudden.

## 2026-08-09 — hetk verdict on the corrected ReLU task (memo written)
- Retrained all 18 hetk runs; census + boundary + code anatomy + interference
  rerun. Headline: **MSE does no superposition at all** — it triages: exactly
  ~N=50 features get dedicated single neurons (K(τ)=1, a=1.000 identity code,
  MSE ~1e-9) and the rest are abandoned (gain exactly 0, MSE = 1/3 baseline);
  rule verified across population splits (100%/67%/56% ≈ 50/n_common). **L4
  superposes everything gradually** (common K(τ)=6.6 computed; rare K(τ)=21.3
  attenuated-linear, gain 0.55, 86% of its MSE is attenuation; smooth K ladder
  in p_rare, no sharp transition). **wMSE flips the triage** (rare computed
  43/50, common abandoned 3/50) — importance weighting selects the winner but
  does not buy coexistence; loss shape does. Full details: `toys/hetk/MEMO.md`.

## 2026-08-09 — corr verdict on the corrected ReLU task (memo written)
- Retrained 10 corr runs; census rerun. ρ=0 L4 anchor now matches the main
  suite exactly (K(>0) 11.5 vs 11.6). Headline: **correlation repels codes
  under L4** (within-pair overlap 1.60 → 0.00 by ρ=0.75; all 100 features stay
  computed — at ρ=1: perfectly co-activating pairs with fully disjoint
  codewords) but **attracts codes under MSE** (overlap 2.1–2.5 vs 0.3–0.5
  between; computed count falls 46 → 26; dedicated-neuron triage at ρ=0).
  Opposite responses of the two losses. The ρ=1 L4 model is the ideal
  activation-vs-parameter dissociation target for the PD stage.
  Details: `toys/corr/MEMO.md`.

## 2026-08-09 — chain verdict on the corrected ReLU task (memo written)
- Retrained 4 chain runs; census rerun. **L4 threads all 100 features through
  both 50-neuron bottlenecks** (100/100 computed; sparse layer-1 codes
  K1(τ)≈10-13 like the single-layer suite, dense layer-2 codes K2(τ)≈24-31 —
  compress-then-re-expand). **MSE fails CiS as the user predicted** (44-46/100
  ≈ the min(N,F) triage) — but matches its single-layer numbers, so the triage
  is a loss property, not a depth effect. Details: `toys/chain/MEMO.md`.

## 2026-08-09 — uand Ln-exponent sweep (memo updated)
- 16 new runs (n ∈ {2.25…3.5, 5, 6} × N ∈ {35, 50}, p=0.1). CiS onset vs loss
  exponent: sharp in the computed-count (N=50: 3→105 across n = 2.5→3.25,
  n* ≈ 3; N=35: window shifts to n* ≈ 4–5 — critical exponent falls with
  width), but the underlying per-gate error declines smoothly — a
  population-level threshold crossing, not a per-gate discontinuity.
  Details in `toys/uand/MEMO.md`. All stage-1 toy work is now complete;
  launching the three scheduled review subagent groups.

## 2026-08-09 — Subagent review round: two claims corrected, core dichotomies verified
- **Scrutinizer** (ran its own checks): (1) uand's n* is criterion-dependent —
  re-scoring at thresholds 0.05/0.15/0.2 moves the transition by >1 exponent
  unit; the robust facts are monotone improvement with n and the L4-above /
  MSE-below-diagonal ordering at every threshold. (2) corr's "MSE merging" was
  misdescribed: zero pairs have both members computed at any ρ>0 — it is
  triage-plus-hedging (asymmetric gains 0.71/0.32 with ~0.4 cross-leak onto the
  abandoned twin); genuine merging was informationally impossible (independent
  values). (3) hetk triage verified robust to wd=0 and seed 43; MSE-trained
  beats L4-trained under the MSE metric 12× — triage is the genuine optimum.
- **Code auditor**: generators/census conventions verified correct (corr
  correlation exact; 1/3 baseline exact; homogeneity validates probes); one
  metric bug fixed (uand gate-K criterion → superadditivity; N=105 gate-K
  21→14.5), regex guards + axes robustness fixed; step-formula exposure
  confounds flagged (corr ρ, uand p).
- **Findings-explorer**: unified two-strategy theory (abandon-vs-superpose cost
  crossing; loss exponent weights rare-large vs frequent-small errors);
  8-item plan; stage-2 recommendation = corr ρ=1 pair first.
- **Executed follow-ups** (P5/P6/P8): uand gate codewords are input-factorized
  (shared-input overlap 4.6 vs 1.8 disjoint — ontology ≈ 15 reads × 105 gates);
  hetk L4 rare population genuinely computes ReLU (output −0.03 at x=−1, not
  −0.55 passthrough); chain layer 2 is ~linear at p=0.01 (R²=0.99, 10% clipped)
  but genuinely nonlinear at p=0.001 (23%) — p=0.001 is the stage-2 chain
  target. All four MEMOs updated with corrections. Running: corr ρ=1 at fixed
  steps (exposure-confound check) + Ln mini-sweep on the main task (corr ρ=0,
  n ∈ {2.5, 3, 3.5}).

## 2026-08-09 — Final checks: exposure confound dead, main-task Ln interpolation
- corr ρ=1 at fixed 15,774 steps: identical results (L4 overlap 0.00, 100/100;
  MSE 25/100 hedged-triage) — step-formula confound eliminated.
- Ln mini-sweep on the main 100→50→100 ReLU task (corr ρ=0): n=2 triage
  (45/100) → n=2.5 intermediate attenuation (0/100 strict, median MSE halved,
  K(>0)=31) → n=3, 3.5 full CiS (100/100, K≈12-14). The triage→CiS transition
  interpolates through attenuation and sits at n ≈ 2.5-3 (criterion-dependent),
  consistent with uand N=50. Stage 1 + review round complete; next: stage-2 PD
  starting with the corr ρ=1 pair (per explorer recommendation) and the P1
  closed-form abandon-vs-superpose theory sketch.

## 2026-08-09 — chain bypass ablation (user request)
- Replaced layer-2 ReLU by identity / fitted linear map on both L4 chains:
  p=0.01 +12% L4 (fold justified — functionally single-layer); p=0.001 3.2×
  L4 with *improved* median per-feature MSE — the second ReLU does tail
  interference-cleanup, not typical-case computation. Memo updated.

## 2026-08-09 — chain bypass p-sweep (user request)
- 4 new L4 chains (p=0.002/0.005/0.02/0.05) + existing endpoints; three-variant
  ablation across all six p. Smooth crossover, no sharp transition: identity-
  bypass cost 1.00x (p>=0.02) -> 3.23x (p=0.001), tracking clipping fraction
  0.004 -> 0.23. Second ReLU functionally dead above p~0.02, gradually
  necessary below p~0.01. bypass_sweep.{md,png}; memo updated. Stacked-codes
  comparison plan delivered, awaiting user confirmation.

## 2026-08-09 — stacked-codes comparison (first pass; user-confirmed design)
- Fixed a dead-code bug in the local biregular generator (first run invalid).
- Result across p: designed two-layer references are FAR from the trained
  chain — stacked codes 20-36x worse L4, code+lstsq ~10x, code+trained-A
  catastrophic (co-adapted, as predicted); 0/100 features pass the strict
  computed criterion for every designed variant at every p. Contrast: the
  single-layer suite's designed codes reach within ~2x of trained. Reading:
  the trained chain's two-layer solution is much better than a
  decode-re-encode stack of codes — the layer-2 representation is not a
  re-encoded feature code (consistent with dense K2 + tail-cleanup ReLU
  finding). Caveat: 4-scalar fit may sit in a shallow-valley optimum (known
  issue); multi-start fit is the next refinement before strong conclusions.
  results/figures: toys/chain/stacked_codes.{md,png}.

## 2026-08-14 — Repo reorganized; biregular models ported into Goodfire's param-decomp
- Root modules reorganized into core/, biregular/, embed/ packages (git mv,
  package-absolute imports, shared core/paths.py anchors; ksweep output
  verified byte-identical post-move). Commit f1233c6.
- Ported both biregular targets (designed CodeAnsatz K=5 transpose p=0.001 and
  trained MLP p=0.001 s42) into Goodfire's param-decomp clone as a new
  experiments/biregular/ (local branch biregular-port; MIT). Exports with
  bit-exact forward equivalence (biregular/export_goodfire.py, commit c168dcf).
- Environment saga (Intel mac): repo pins py3.13/torch>=2.6 — no x86_64 wheels.
  Relaxed on the local branch to py3.12/torch 2.2.2 + torchvision>=0.17,
  numpy<2, wandb 0.21.4 (ships wandb_gql), numba dropped (unused). Goodfire's
  own TMS test suite passes 4/4 on this stack — environment validated.
- Smoke decompositions (250 steps) of both models ran clean end-to-end.
  Full 10k-step runs launched; readout (alive counts, write-side basis
  structure vs our port's 53-alive/per-neuron result) to follow.

## 2026-08-17 — Goodfire-native run on designed K=5 model: no-op at p=0.001
- Full 10k-step param-decomp run (67h CPU) on the designed model at data
  p=0.001 converged to a DEGENERATE solution: sum-of-components misses W by
  37-58% (runtime delta carries the model), flat mass spectrum, 0/200
  components axis-aligned in either layer; recon losses fine (trivially — ~90%
  of p=0.001 batches are all-zero, so importance-minimality gates everything
  off into the delta). Neither ExactPD's feature basis nor our port's
  neuron basis emerges. Diagnosis: canonical config assumes TMS-like p=0.05.
- Follow-ups running: trained model at 2,500 steps (user-requested shorter);
  designed model re-run at data p=0.05 (tests the sparsity diagnosis).

## 2026-08-17 — Degeneracy diagnosed: mis-priced objective, not a bug
- Two scrutiny subagents on the Goodfire-native degenerate run: (1) forensics —
  collapse completed in <100 steps, never any structure; measured economics:
  at p=0.001 coeff 3e-3, ideal-solution importance cost 6.0e-4 > max recon
  benefit 1.2e-4, so all-CIs-zero is the GLOBAL optimum; (2) integration audit
  — no bugs on any axis (dataset U[-1,1] verified, run_batch correct, modules
  attached, gradients flowed); hand-built ground-truth decomposition scores
  34x WORSE than degenerate at canonical coeffs, and predicted the p=0.05
  rerun would also collapse (~9x). Prediction CONFIRMED (p=0.05 short run:
  L0~0, flat mass, 0 axis-aligned).
- User decision: no FaithfulnessLoss (delta construction IS their faithfulness
  mechanism; adding a loss would deviate from the published method). Tuned run
  launched strictly within their config space: coeff 1e-4 + pnorm 2.0 (their
  own TMS 40-10 values, the correct donor for C=100) at p=0.001. Trained-model
  leg deferred until the designed-code story is complete (user request).

## 2026-08-17 — BREAKTHROUGH: tuned pricing rescues Goodfire-native decomposition
- p=0.05 short run: still degenerate (predicted). Tuned run (coeff 1e-4,
  pnorm 2.0 = their own TMS 40-10 values; p=0.001; canonical losses, NO
  FaithfulnessLoss per user decision): STRUCTURE FORMS at 2,500 steps —
  L0(0.1) 0.22/0.41, differentiated mass (8x range), 22/19 axis-aligned
  components, sum-vs-W err down to 0.26/0.15, linear1 covering 72 features,
  linear2 covering 44/50 neurons. Emerging ontology = feature-basis reads +
  NEURON-basis writes — corroborating our port's asymmetry in Goodfire's own
  unmodified pipeline. Full 10k-step tuned run launched for the definitive
  basis readout; trained-model leg restarts (tuned coeffs) after.

## 2026-08-17 — Exp 1: ExactPD inserted into the port's VPD objective (audited clean)
- New biregular/vpd_objective.py + append-only store (results/vpd_objective/).
- Headline (adversarially audited, no bugs, independently replicated): the
  exact solution costs 6.4x MORE total than the learned neuron-basis fit —
  entirely in reconstruction (9x stoch, 13x PGD), all on w_out (exact w_in
  recon is exactly 0 by math); exact is SPARSER on importance (0.200 = 2pF vs
  0.253). Interference ownership measured: masking inactive features' write
  pieces deletes the leak the model emits. Implication for Exp 2: imp-up/
  recon-down DOES favor ExactPD, break-even ~55x default ratio; sweep to
  straddle 55x and the collapse boundary above it. Caveats: faithfulness
  weight 0 (learned keeps a free 1.5e-4 delta); trained-CI variant deferred
  (analytic bound: cannot close gap below ~5.8x). Two stale-import fixes
  restored ExactPD/ci_stats on this branch (byte-identical to archive).

## 2026-08-17 — DEFINITIVE: Goodfire-native converged run corroborates the port
- 10k-step tuned run (coeff 1e-4, pnorm 2.0, p=0.001) converged: linear1 = 99
  alive, conc 0.86, 85 axis-aligned, covering 90 FEATURES (per-feature reads);
  linear2 = 62 alive, conc 0.83, 47 axis-aligned, covering 49/50 NEURONS
  (per-neuron writes); sum-vs-W residual 0.14/0.08. Goodfire's unmodified
  pipeline reproduces our port's asymmetric ontology: feature-basis reads,
  neuron-basis writes. Combined with Exp 1 (exact PD costs 6.4x, all in w_out
  interference ownership), the mechanism is fully accounted for. Next: --fit-ci
  variant, Exp-2 pricing sweep (~55x break-even), trained-model leg (tuned).

## 2026-08-17 — Warm-restart consolidation: Goodfire-native converges to 50 neuron writes
- +10k warm restart (env-hook resume in our experiment script): linear2 62->50
  alive (splits 10->1, conc 0.98, 49/50 axes), linear1 99 alive conc 0.95
  (95 feature axes). The 12 extras were unablated transients (earlier
  "stable row-split" inference retracted). FINAL designed-code corroboration:
  ~99 feature reads + 50 neuron writes — identical ontology to our port
  (53->50). Also established: scalar layerwise CI (components.py:435, 1->16->1
  per-component MLPs) is a design choice reinforcing the input-basis
  preference, but Exp-1's oracle-gate result shows the neuron-basis write
  optimum follows from the IDEAL objective (interference ownership) — gate
  architecture is separable. Candidate Exp-3: rerun under ci mode "global".

## 2026-08-17 — Exp 2 pricing sweep: no exact-basis window at 10k steps
- Five cells R in {10,30,55,100,300}x default importance. Sequence observed:
  structured neuron-leaning mixed (R=10, ci_l0 0.30, read conc 0.82) ->
  progressive gate starvation -> CI COLLAPSE at R=300 (imp mass 5.4e-5,
  ci_l0 0.07; mass-based verdict says "mixed" but gating is dead). ExactPD
  recovery NOT observed: dynamics skip the exact basin even where its total
  is statically cheaper — importance pressure kills CIs before rotating the
  write basis. Peak exact-direction signal at R=30 (write V-cos 0.70).
  Caveat: 10k-step fits vs 25k reference; warm-restart extensions of R in
  {10,30} are the natural follow-up. Phase diagram:
  figures/vpd_objective/phase_diagram.png. Trained-model Goodfire leg still
  running.

## 2026-08-17 — Trained-model Goodfire leg complete (Part A closed)
- 10k tuned + 10k warm restart. Read side: PERFECT per-feature basis (100
  alive, conc 0.99, 99/100 aligned, 100 axes, 0 splits). Write side: STABLY
  DISTRIBUTED — warm restart consolidated nothing (92->90 alive, conc 0.44
  unchanged, 45 neuron axes, 34 split slots, ~2 pieces/neuron). Unlike the
  designed model (62->50 clean neuron rows), the trained model's graded
  per-connection write weights do NOT reduce to 50 neuron slices: VPD's
  converged write ontology is ~90 sub-row pieces. The trained model is a
  measurably harder PD target than its designed twin — the grading (the very
  thing that separates trained from designed codes in the dFH battery) is
  what breaks clean neuron-basis writes. Part A closed.

## 2026-08-17 — Hybrid-oracle basis isolation (user-driven): the refined mechanism
- New hybrid_oracle variant (100 live neuron-basis write pieces, half-row
  each, oracle gates; reads/count/faithfulness identical to exact_oracle).
  Result: recon EXACTLY 0 (1e-18) — kernel self-masking confirmed at float
  precision — but imp w_out 4.78 (48x exact); at default pricing hard-gated
  neuron basis LOSES to hard-gated exact (4.88e-4 vs 3.31e-4). Learned
  reference (5.17e-5) beats both via SOFT gates (act_out 0.10, imp 0.187).
  REFINED CLAIM: recon+PGD is un-cheatable, importance is soft-discountable;
  training selects the basis whose cost lives in the discountable term
  (neurons) and discounts it. The naive "neuron basis wins under oracle
  gates" is FALSE — user skepticism exposed this. All legs now measured.

## 2026-08-17 — B-to-C recovery test complete (user-designed)
- Gates-only on frozen hybrid pieces: 2.615e-4 — does NOT recover C (5x gap);
  piece co-adaptation required (earlier gates-suffice claim corrected).
- FULL training from hybrid init: total 5.146e-5 = 0.995x C; write pieces
  collapse 100 half-rows -> 50 alive, matched to C at median cos 0.998,
  100% neuron-slot agreement; read side same basin (cos 0.878). C is the
  attractor from the neuron-basis configuration — recovery confirmed at loss
  AND component level. Controlled pair: exact-basis trained gates stall at
  3.582e-4. vpd_objective store now: 6 variants, one eval pool.

## 2026-08-17 — Embedded ideal-vs-VPD: Part 1 built, runs launched
- Designed sandwich W_e -> C(K=5, transpose) -> W_e^T at d_emb {200, 50};
  all asserts pass (d200 forward == plain code 8e-7; d50 eff == G*C, mean
  G_jj = 0.500 = d/F). Ideal PD stored (W_emb rows; W_in full SVD; tied
  transposes). SURPRISE: uniform/DC direction is an EXACT singular pair
  (closed form to 1e-16) but ranks 49/50 — negative b cancels DC mass.
  Loading matrix dense at d200 (max |L| 0.34): SVD frame vs embedded-feature
  frame maximally distinguishable. param-decomp 4-Linear experiment added,
  smokes clean (1.5/1.9 it/s); four-leg run chain launched (~7 h).

## 2026-08-18 — Embedded ideal-vs-VPD: all four legs + comparison complete
- (Leg-2 SIGKILL overnight; redone.) Consolidated verdicts:
  lin_in NOT SVD-aligned at either d_emb (Hungarian ~0.09-0.14, flat under
  consolidation) — the ideal SVD triplets are not what VPD finds. Strongest
  frame = embedded features: d200 0.633 median, 90/100 coverage
  (strengthening 10k->warm, below 0.8 threshold => verdict mixed); d50
  diluted (0.497, 58/100) as 100 feature directions crowd 50 dims.
  lin_emb -> ideal row pieces at d200 (conc 0.991, Hungarian 0.842);
  partial at d50. lin_out partial neuron structure. lin_unemb unstructured
  (delta carries it). OPEN FINDING: Sigma-residuals 0.21-0.85 — in the
  4-matrix sandwich the delta remains a major weight carrier even after
  consolidation, unlike the 2-matrix runs. Artifacts:
  results|figures/embedded/ideal_vs_vpd/; embed/MEMO.md being written.

## 2026-08-18 — CORRECTION + closure: VPD found the DUAL frame (causal-minimal PD)
- User question "what satisfies causal minimality?" -> answer: the dual-basis
  per-feature decomposition (W_in = sum_j dual_j x codeword_j): exact
  self-activation at d>=F (activation = x_j identically), least-squares-
  optimal below (pinv optimality; G quantifies degradation; trace theorem =
  the bound). Costs 2x-overcomplete pieces — causal minimality and
  count/mass minimality are incompatible on the rank-50 code layer.
- Re-scored learned lin_in against the dual frame: d200 median |cos| 0.911
  (77/100 > 0.7) vs 0.627 for W_e rows — the earlier "mixed" verdict was a
  wrong-reference artifact; VPD FOUND the dual-basis read ontology at d200,
  and its d50 dilution (0.564) matches the G-softening prediction. Summary/
  memo update needed to reflect dual-frame scoring as the primary reference.

## 2026-08-18 — Mechanism forward census: column (per-neuron) ontology wins at d200
- New embed/embed_mechanism_forward.py: full model vs strictly layer-local
  chain mechanisms (transpose decoder, tied W_out=W_in^T) under three W_in
  ontologies — per-emb-dim ROWS (200), per-neuron COLUMNS (50), SVD triplets
  (50). All tilings exact (row/column bitwise; SVD 1.4e-15); full-sum chain
  forwards == full model to <1e-15 on all 100 one-hot probes (identities, not
  findings). Option A (activation-direction projections) rejected: needs
  W_emb at inference; decomposition must be fixed before inference.
- Census (no cutoffs; PR + 50/90/99% counts): rows PR med 68.5, n50/90/99 =
  25/90/147 of 200; columns PR 6.6 (constant), 3/17/47 of 50; SVD PR 15.5,
  6/21/37 of 50.
- Truncated-forward fidelity (tied top-m): columns need EXACTLY m*=5 (=K, all
  features) for spike<=1% AND leak<=10% — machine-exact beyond (ReLU kills
  the b-floor: off-code pre-activations are negative, so the 45 census-heavy
  b-pieces are causally inert). SVD needs 40/43 (med spike/leak) of 50; rows
  176/198 of 200. VERDICT: per-neuron columns are the causally concentrated
  ontology; census energy and causal count DISAGREE for columns — truncated
  fidelity is the discriminator. Artifacts: results|figures/embedded/
  ideal_vs_vpd/mechanism_forward/.

## 2026-08-18 — Cut-side principle grid: confirmed on all 4 matrices (designed d200)
- New --grid mode in embed/embed_mechanism_forward.py: full 4-matrix x 2-cut-
  direction grid on the designed sandwich (y = relu(x W_e W_in) W_in^T W_e^T;
  W_emb F x d, W_in d x N, W_out = W_in^T, W_unemb = W_e^T). Each cell cuts one
  matrix into input-side rows or output-side columns (all 8 tilings bitwise),
  censuses activation coefficients on one-hot probes, and measures truncation
  fidelity with ONLY that matrix truncated to top-m pieces (untied; other three
  matrices full).
- Principle under test: decompose each layer on the side where its OWN weights
  re-concentrate the signal into discrete events; never cut a dense interface.
  Predictions {W_emb: input m*=1; W_in: output m*=5; W_out: input m*=5;
  W_unemb: output m*=1}.
- RESULT — winner sides 4/4 CONFIRMED, spike counts 4/4 at prediction:
  W_emb input 1/1 (spike/leak m*, errors ~1e-15 from m=1) vs output 162/110;
  W_in output 5/5 (regression: bitwise == existing census; m*=5 all features)
  vs input 162/110; W_out input 5/5 (post-ReLU h is exactly K-sparse) vs
  output 161/153; W_unemb output spike m*=1 (top piece IS the target readout,
  error 0 at m=1, every feature) vs input 161/153.
- One deviating number, understood: W_unemb output-cut leak m* = 98 (not 1) —
  at the FINAL matrix the coordinate cut is aligned with the error metric, so
  leak fidelity degenerates into a support census of y's leak (n99 = 97); the
  mechanism (target readout) is still exactly 1 piece. W_emb shows 1/1 only
  because the probe itself is 1-sparse.
- Structural finding: dense-interface cells are pairwise the SAME decomposition
  (W_emb output-cut == W_in input-cut, W_out output-cut == W_unemb input-cut;
  truncated forwards bitwise equal) — an interface cut is one object however
  booked, needing ~110-160 of 200 pieces. v-invariance of m* asserted (v=1 vs
  2.5; error curves can permute inside exactly-tied coefficient groups).
- VERDICT: principle survives unfalsified on the designed model — candidate
  general design rule for layer-local PD. Artifacts: results|figures/embedded/
  ideal_vs_vpd/mechanism_forward/cutside_grid_d200.{md,csv,png}.

## 2026-08-18 — d50 reruns: cut-side grid + mechanism solo spectra (G-noise regime)
- embed/embed_mechanism_forward.py gains --d-emb {200,50} (default 200): loads
  ideal_d{D}.pt, suffixes all outputs _d{D}; d200 outputs untouched. Regime at
  d50: W_e W_in = G C with G = W_e W_e^T a rank-50 projector (mean G_jj =
  0.500, max|W_e W_in - G C| = 3.9e-16); off-code preactivations go positive
  through G-noise — causal count #(pre>0) med 21 (14–26) per feature vs
  exactly K = 5 at d200 (100/100 deviate). G = I-dependent identities are
  REPORTED, not asserted; tiling (8/8 bitwise), duality (both interface pairs
  bitwise), W_emb-input identity and v-invariance stay hard asserts — all PASS.
- GRID at d50, m* spike/leak med (min–max), d200 in parens: W_emb input
  1 (1–1)/1 (1–1) [1/1]; W_emb output 43 (36–49)/40 (31–47) [162/110]; W_in
  input == W_emb output (duality); W_in output 34 (25–42)/33 (18–42) [5/5];
  W_out input 14 (9–20)/14 (9–20) [5/5]; W_out output 43 (33–49)/41 (35–46)
  [161/153]; W_unemb input == W_out output; W_unemb output spike 1 (1–1),
  leak 75 (67–82) [1/98]. Winner sides 3/4: W_unemb FLIPS to input by the
  combined median — the d200-diagnosed leak-criterion artifact now dominates
  (output-cut spike m* still 1 with the top piece = target readout for
  100/100; leak m* 75 ~ y-census n99 = 73, a support census of the broadband
  G-noise leak). The {1,5,5,1} knees soften to {1,34,14,1-spike}; note the
  |coefficient| ranking also spends slots on large-|negative| dead pieces in
  the W_in output cut, so m* (34) overshoots even the causal count (21).
- SPECTRA at d50 (1+5+5+1 winning-cut convention, features 0/61/1 as d200):
  full spike 0.2034/0.1768/0.1901; mech spike 0.1480/0.0828/0.1303 (spike err
  27/53/31% — top-5 misses causal mass; kept top-5 != on-code: 4/5, 3/5, 4/5
  neurons). Strict mech off-target EXACTLY 0 at any d (structural: the
  W_unemb column-j piece writes only coordinate j) — the predicted nonzero
  G-noise leak lives at the retained write z read through the FULL unembed
  bank: full-read leak max 0.1168/0.0878/0.1221 vs full-model off-max
  0.1145/0.1525/0.0963. Leak ladder SMEARED into overlapping bands:
  within-class std 1.4e-2–4.5e-2 vs the d200 exact ladder (~3e-16); class
  means shrink (m_ov=2: 0.034/0.023/0.048 vs d200 0.2138; m_ov=3: 0.096/0.061
  vs 0.3640) and bands overlap (probe 61: an m_ov=1 leak of 0.1525 exceeds
  its m_ov=3 leak 0.0956).
- Predictions on record, checked: (1) spike near K a^2 G_jj^2 with per-feature
  spread — CONFIRMED (mech-spike ratio med 1.061, range 0.447–1.920; full
  spike med 1.378, 0.805–2.470: extra positive G-noise writes); (2) m*=5 knee
  softened for some features — CONFIRMED and stronger: softened for ALL
  features (min 25); (3) leak ladder smeared into bands — CONFIRMED.
- Artifacts: results|figures/embedded/ideal_vs_vpd/mechanism_forward/
  {cutside_grid_d50.md/csv/png, mechanism_spectra_d50.md/png}; d200 files
  byte-untouched.

## 2026-08-18 — Ladder recovery sweep: discreteness returns EXACTLY at d_emb = F
- Sweep d_emb 50->200 (designed sandwich, same C, per-d random unit-row W_e):
  smearing shrinks slowly to d=90 then fast (class std 4.2e-2 at d50 ->
  1.3e-2 at d99) and drops ~4 orders of magnitude AT d=100 (1.9e-6, float
  residual only; separability inf for practical purposes). Sharp transition
  at d=F exactly, as G = W_e W_e+ = I once W_e has full row rank — NOT
  gradual/asymptotic. Conditioning of W_e spikes at the transition (3.8e2 at
  d=100, decaying both directions) — the exact-recovery point is also the
  numerically most fragile. Figure:
  mechanism_forward/ladder_recovery_vs_demb.png.
