### SVD-proposition test — components = SVD triplets of W_in/W_out (+ W_emb rows)

| d_emb | rank | faithfulness dev | cap95 r_j (W_in) | cap95 c_j (W_out) | PR comps/feat (in) | PR comps/feat (out) | PR feats/comp (in) | PR feats/comp (out) |
|---|---|---|---|---|---|---|---|---|
| 20 | 20 | 1.7e-15 | 17.1 | 17.7 | 6.8 | 7.9 | 37.5 | 38.2 |
| 50 | 50 | 2.6e-15 | 41.4 | 40.7 | 11.8 | 14.8 | 36.0 | 36.3 |
| 200 | 50 | 1.4e-15 | 41.5 | 41.7 | 12.3 | 16.4 | 36.3 | 35.8 |
| 1000 | 50 | 7.8e-16 | 41.4 | 42.0 | 12.4 | 15.7 | 35.2 | 36.4 |

Reference (per-feature chain decompositions, same checkpoints): 100 components; 1 component per feature; capture count = 1; feature-PR per component = 1; faithfulness equally exact.

Reading: SVD components are exactly faithful and rank-1 with NO overlap corrections (orthogonal factors) — but they are collective objects: every feature needs most of the spectrum (capture ~rank), every component serves many features (PR), and all components co-activate on every input. The weight-mass decomposition answers 'what directions dominate W_in' — not 'which part computes feature j'. Neither is wrong; they are different questions, and the d_emb regimes do not qualitatively change the SVD picture (it cannot see the superposition boundary that the chain decompositions expose).
