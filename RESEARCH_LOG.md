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
