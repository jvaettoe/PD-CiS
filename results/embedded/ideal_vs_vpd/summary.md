# Ideal PD vs VPD — cross-regime summary (d_emb = 200 vs 50)

Designed sandwich `W_e -> C(K=5, transpose, fit_ab(p=0.001)) -> W_e^T`; VPD
decomposes all four matrices (C = 100 subcomponents each). Ideal PD: 100 row
pieces `e_j (x) W_e[j]` for W_emb (tied transpose for the unembedding), all 50
SVD triplets `s_k u_k v_k^T` for W_in (tied transpose for W_out).

Runs (all four legs complete):

| d_emb | stage 10k (unconsolidated) | stage warm (+10k warm restart) |
|---|---|---|
| 200 | `p-05431108` | `p-832f0310` |
| 50 | `p-f8070ad6` | `p-97d06281` |

Numbers below are **consolidated (warm)** with the 10k value in parentheses;
alive = mass > 10% of per-matrix max; thresholds verbatim: svd_aligned if
Hungarian median |cos| vs triplets > 0.8 (alive); feature_frame_aligned if
best-|cos| median vs W_e rows > 0.8 (alive) with coverage >= 60; collapsed if
alive < 25; else mixed.

## Cross-regime contrast table

| matrix | metric | d_emb = 200 | d_emb = 50 | ontology verdict |
|---|---|---|---|---|
| lin_emb | Hungarian med \|cos\| vs row pieces | **0.842** (0.535) | 0.445 (0.283) | rows-aligned / rows-partial |
| | frac > 0.8 | 0.55 (0.01) | 0.12 (0.02) | |
| | input-axis conc median (alive) | 0.991 (0.945) | 0.728 (0.593) | |
| | distinct-feature coverage | 91/100 (87) | 79/100 (75) | |
| lin_in | (i) Hungarian med \|cos\| vs SVD triplets | 0.091 (0.080) | 0.138 (0.110) | feature-frame-leaning / diluted (formal: mixed) |
| | (ii) frame best-\|cos\| median vs W_e rows | **0.633** (0.579) | 0.497 (0.452) | |
| | (ii) distinct-feature coverage | **90/100** (86) | 58/100 (58) | |
| | (iii) U-side neuron top-share median / coverage | 0.462, 43/50 | 0.468, 41/50 | |
| | alive | 100/100 | 66/100 (69) | |
| lin_out | neuron conc median (alive) / coverage | 0.492, 47/50 | 0.512, 37/50 | neuron-partial |
| | Hungarian med \|cos\| vs transposed triplets | 0.086 (0.086) | 0.120 (0.122) | |
| | alive | 70/100 (80) | 46/100 (50) | |
| lin_unemb | Hungarian med \|cos\| vs transposed row pieces | 0.065 (0.060) | 0.083 (0.081) | unstructured |
| | output-axis conc median / coverage | 0.304, 62/100 | 0.294, 53/100 | |
| | Sigma-subcomponent rel residual | 0.850 (0.999) | 0.560 (0.624) | |

## Verdict statement

- **lin_in is definitively NOT svd-aligned at either d_emb.** Hungarian
  median |cos| vs the ideal SVD triplets sits at 0.09–0.14 and is flat across
  consolidation — the hand-derived singular frame is not what VPD finds.
- **lin_in leans toward the embedded-feature frame at d200**: best-|cos| vs
  the (non-orthogonal) rows W_e[j] strengthens 0.579 -> 0.633 under
  consolidation with distinct-feature coverage 86 -> 90/100 — a real lean, but
  below the 0.8 threshold, so the formal verdict stays `mixed`.
- **The lean is diluted at d50** (0.497 median, coverage 58/100), as expected
  when 100 feature directions crowd a 50-dimensional embedding space.
- **lin_emb consolidates toward the ideal row pieces at d200** (concentration
  0.991, Hungarian 0.842, 55% of matches > 0.8) but only **partially at d50**
  (0.728 / 0.445).
- **lin_out shows partial neuron structure** (47/50 resp. 37/50 distinct-neuron
  coverage on the write side, concentration ~0.5); no triplet alignment.
- **lin_unemb is unstructured at both d_emb** — the delta carries it
  (Sigma-residual 0.56–0.85).
- **Open finding:** Sigma-subcomponent residuals remain high throughout
  (0.21–0.85 after consolidation) — in the 4-matrix sandwich the runtime delta
  component stays a major weight carrier, unlike the 2-matrix designed runs.

Figure: `figures/embedded/ideal_vs_vpd/summary_ontology.png` (verdict grid,
matrix x d_emb). Per-stage detail: `d200/comparison.md`, `d50/comparison.md`,
`d{200,50}/{per_matrix.csv, run_10k.json, run_warm.json}`.
