# Ideal PD vs VPD — embedded designed code, d_emb=50

Per-matrix structural comparison of learned VPD subcomponents (V/U rank-1 pieces) against the hand-derived ideal PD (row pieces for W_emb/W_embᵀ, SVD triplets for W_in/W_out).

## Stage 10k — **UNCONSOLIDATED** (10k, pre warm-restart — no basis verdicts final)

- run: `exports/goodfire/out/decompositions/p-f8070ad6` (checkpoint `model_10000.pth`, step 10000)
- alive threshold: mass > 10% of per-matrix max mass

| matrix | alive/C | Σ-comp rel resid | conc median (alive) | conc coverage (alive) | Hungarian med \|cos\| (alive / all) | frac>0.8 (alive) |
|---|---|---|---|---|---|---|
| lin_emb | 100/100 | 0.656 | 0.593 | 75/100 | 0.283 / 0.283 | 0.020 |
| lin_in | 69/100 | 0.239 | 0.456 | 44/50 | 0.110 / 0.079 | 0.000 |
| lin_out | 50/100 | 0.283 | 0.447 | 40/50 | 0.122 / 0.101 | 0.000 |
| lin_unemb | 98/100 | 0.624 | 0.301 | 54/100 | 0.081 / 0.081 | 0.000 |

### lin_in three-frame readout (stage 10k)

- (i) SVD triplets: Hungarian full-piece median |cos| = 0.110 (alive) / 0.079 (all matched); V-side-only best |cos| vs u_k median = 0.473 (alive)
- (ii) embedded-feature frame {W_e[j]}: best-|cos| median = 0.452 (alive) / 0.387 (all); distinct-feature coverage 58/100 (alive), 68/100 (all)
- (iii) U-side neuron axes: top-share median = 0.456 (alive); distinct-neuron coverage 44/50
- **verdict: `mixed`** — thresholds: svd_aligned: Hungarian median |cos| vs triplets > 0.8 (alive-restricted); feature_frame_aligned: best-|cos| median vs W_e rows > 0.8 (alive) with distinct-feature coverage >= 60; collapsed: alive < 25; else: mixed

## Stage warm — consolidated (post warm-restart)

- run: `exports/goodfire/out/decompositions/p-97d06281` (checkpoint `model_10000.pth`, step 10000)
- alive threshold: mass > 10% of per-matrix max mass

| matrix | alive/C | Σ-comp rel resid | conc median (alive) | conc coverage (alive) | Hungarian med \|cos\| (alive / all) | frac>0.8 (alive) |
|---|---|---|---|---|---|---|
| lin_emb | 100/100 | 0.626 | 0.728 | 79/100 | 0.445 / 0.445 | 0.120 |
| lin_in | 66/100 | 0.207 | 0.468 | 41/50 | 0.138 / 0.092 | 0.000 |
| lin_out | 46/100 | 0.262 | 0.512 | 37/50 | 0.120 / 0.098 | 0.000 |
| lin_unemb | 92/100 | 0.560 | 0.294 | 53/100 | 0.083 / 0.079 | 0.000 |

### lin_in three-frame readout (stage warm)

- (i) SVD triplets: Hungarian full-piece median |cos| = 0.138 (alive) / 0.092 (all matched); V-side-only best |cos| vs u_k median = 0.456 (alive)
- (ii) embedded-feature frame {W_e[j]}: best-|cos| median = 0.497 (alive) / 0.407 (all); distinct-feature coverage 58/100 (alive), 71/100 (all)
- (iii) U-side neuron axes: top-share median = 0.468 (alive); distinct-neuron coverage 41/50
- **verdict: `mixed`** — thresholds: svd_aligned: Hungarian median |cos| vs triplets > 0.8 (alive-restricted); feature_frame_aligned: best-|cos| median vs W_e rows > 0.8 (alive) with distinct-feature coverage >= 60; collapsed: alive < 25; else: mixed
