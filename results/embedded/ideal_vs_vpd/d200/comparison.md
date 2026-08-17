# Ideal PD vs VPD — embedded designed code, d_emb=200

Per-matrix structural comparison of learned VPD subcomponents (V/U rank-1 pieces) against the hand-derived ideal PD (row pieces for W_emb/W_embᵀ, SVD triplets for W_in/W_out).

## Stage 10k — **UNCONSOLIDATED** (10k, pre warm-restart — no basis verdicts final)

- run: `exports/goodfire/out/decompositions/p-05431108` (checkpoint `model_10000.pth`, step 10000)
- alive threshold: mass > 10% of per-matrix max mass

| matrix | alive/C | Σ-comp rel resid | conc median (alive) | conc coverage (alive) | Hungarian med \|cos\| (alive / all) | frac>0.8 (alive) |
|---|---|---|---|---|---|---|
| lin_emb | 100/100 | 0.981 | 0.945 | 87/100 | 0.535 / 0.535 | 0.010 |
| lin_in | 100/100 | 0.450 | 0.462 | 42/50 | 0.080 / 0.080 | 0.000 |
| lin_out | 80/100 | 0.566 | 0.484 | 46/50 | 0.086 / 0.080 | 0.000 |
| lin_unemb | 100/100 | 0.999 | 0.306 | 60/100 | 0.060 / 0.060 | 0.000 |

### lin_in three-frame readout (stage 10k)

- (i) SVD triplets: Hungarian full-piece median |cos| = 0.080 (alive) / 0.080 (all matched); V-side-only best |cos| vs u_k median = 0.294 (alive)
- (ii) embedded-feature frame {W_e[j]}: best-|cos| median = 0.579 (alive) / 0.579 (all); distinct-feature coverage 86/100 (alive), 86/100 (all)
- (iii) U-side neuron axes: top-share median = 0.462 (alive); distinct-neuron coverage 42/50
- **verdict: `mixed`** — thresholds: svd_aligned: Hungarian median |cos| vs triplets > 0.8 (alive-restricted); feature_frame_aligned: best-|cos| median vs W_e rows > 0.8 (alive) with distinct-feature coverage >= 60; collapsed: alive < 25; else: mixed
