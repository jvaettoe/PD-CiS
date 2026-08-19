# Featbasis census — the top-1 feature-frame W_in ontology (d_emb = 200, designed sandwich)

Checkpoint: `checkpoints/embedded_ideal/ideal_d200.pt` (K=5, p=0.001, a=0.3646, b=-0.0476). Construction: 100 pieces P_j = W_e[j] (x) g_j with Gmat = pinv(W_e^T) W_in (rows g_j), sum_j P_j = W_in exactly; the activation coefficient of piece P_i on probe v e_j is v (W_e W_e^T)[j, i] — the Gram row. At d200, Gmat = (W_e W_e^T)^{-1} C: the output sides are the DUAL-frame codewords. CROSS-LAYER (pieces use W_emb weights) but PRE-INFERENCE (fixed before any input) — the layer-locality rule is deliberately relaxed here, per user instruction; the Option-A objection (inference-time projections) does not apply.

## Exactness asserts (identities, not findings)

```
tiling       max|W_e^T Gmat - W_in|: 1.859e-15  (tol 1e-12) PASS
activation   max|A W_e^T - v Gram| (v=1.0): 1.110e-15  (tol 1e-12) PASS
chain        max|(v Gram) Gmat - A W_in| (v=1.0): 2.720e-15  (tol 1e-12) PASS
activation   max|A W_e^T - v Gram| (v=2.5): 5.329e-15  (tol 1e-12) PASS
chain        max|(v Gram) Gmat - A W_in| (v=2.5): 6.550e-15  (tol 1e-12) PASS
piece i=0   max|W_e P_i - Gram[:,0] g_0|: 5.551e-16  (tol 1e-12) PASS
piece i=50  max|W_e P_i - Gram[:,50] g_50|: 6.106e-16  (tol 1e-12) PASS
piece i=99  max|W_e P_i - Gram[:,99] g_99|: 9.992e-16  (tol 1e-12) PASS
identity     max|Gmat - Gram^{-1} C|: 9.992e-15  (tol 1e-12) PASS
unit rows    max|diag(Gram) - 1|: 1.364e-07  (tol 1e-06) PASS
self-top-1   argmax_i |Gram[j,i]| = j for all 100 features (worst margin 1/max_offdiag = 3.96): PASS

regression: probe e_1 solo minimal k (columns) = 5 == mechanism_minimal_k_d200.png value: PASS
regression: probe e_1 solo minimal k (svd) = 27 == mechanism_minimal_k_d200.png value: PASS
regression: probe e_1 solo minimal k (rows) = 167 == mechanism_minimal_k_d200.png value: PASS
```

## Activity census — featbasis as a FOURTH ontology (no cutoffs: PR + multi-threshold counts)

Featbasis energies per feature j: (W_e W_e^T)[j, i]^2 over 100 pieces (self coefficient exactly 1; off-self = feature-frame Gram noise, max 0.253, mean |.| 0.056).

| ontology | pieces | PR median (min–max) | n50 med (min–max) | n90 med (min–max) | n99 med (min–max) |
|---|---|---|---|---|---|
| rows (per-emb-dim) | 200 | 68.5 (51.6–81.1) | 25 (20–30) | 90 (79–103) | 147 (138–156) |
| columns (per-neuron) | 50 | 6.6 (6.6–6.6) | 3 (3–3) | 17 (17–17) | 47 (47–47) |
| SVD (triplets) | 50 | 15.5 (5.3–21.5) | 6 (2–9) | 21 (14–26) | 37 (31–42) |
| featbasis (top-1 feature-frame) | 100 | 2.3 (1.8–2.7) | 1 (1–1) | 25 (16–33) | 63 (52–71) |

Top-1 dominance (featbasis only): |coef_j|/max_{i≠j}|coef_i| median 5.03 (3.96–6.56); |coef_j|/sum_{i≠j}|coef_i| median 0.177 (0.151–0.222) — the self coefficient beats every single rival but NOT their sum (ratio-vs-sum < 1 for 100/100 features).

## Truncated four-layer solo fidelity (spike within 1%)

Protocol: W_emb row j + top-m featbasis pieces (by |Gram[j, i]|; top-1 is always the self piece) + the tied transposed pieces for W_out = W_in^T + W_unemb column j. The strict column-j readout writes only coordinate j, so the criterion is the spike error |y_m[j] - y[j]| / |y[j]| <= 1%; m = 1..100, all 100 features.

| ontology (probe-e_1 legacy figure / all-feature featbasis) | minimal k (first crossing) | m* (persistent) |
|---|---|---|
| columns (probe e_1) | 5 | — |
| SVD (probe e_1) | 27 | — |
| rows (probe e_1) | 167 | — |
| featbasis (probe e_1) | 86 | 92 |
| **featbasis (all 100 features)** | **74 (33–94)** | 92 (82–98) |

Top-1 (m = 1) spike error: median 4.40 (1.32–13.76) relative — the m = 1 solo OVERSHOOTS the full spike several-fold. Cause: the output sides are dual-frame codewords g_j = (W_e W_e^T)^{-1} c_j with norms ||g_j||/||c_j|| median 2.82 (1.92–4.52); the piece-norm overcompleteness sum_j ||P_j||_F / ||W_in||_F = 20.31. The exact tiling lives on massive cross-piece cancellation, so truncation is catastrophic at every small m.

## Verdict: is top-1 sufficient for a 1% spike?

**NO.** Minimal k for spike within 1% is median 74 (33–94) of 100 pieces (first crossing; persistent m* median 92 (82–98)). The Gram-row census looks maximally concentrated (PR median 2.3, n50 median 1) but the causal story is the opposite: cross-layer knowledge of W_emb buys a nominally top-1 READ ontology whose WRITE sides are dual-amplified, and the four-layer solo needs almost all pieces to cancel the amplification back down. The five-ontology minimal-k comparison for probe e_1 (figure v2): full model, columns k = 5, SVD k = 27, rows k = 167, featbasis k = 86 — featbasis lands between SVD and rows, far from the per-neuron columns' k = 5, and nowhere near k = 1.

## Cross-check: learned VPD lin_in components vs this frame

`exports/goodfire/out/decompositions/p-832f0310/model_10000.pth`, key `_components.lin_in.V`, (d_in, C) orientation, columns normalized.

VPD lin_in best-|cos|: vs feature frame W_e rows median 0.6332 (0.1526–0.7263); vs dual rows pinv(W_e)^T median 0.9110 (0.2728–0.9621); same best-match feature for 91/100 components; frame-vs-dual cos(W_e[j], dual_j) median 0.7036 (0.6091–0.8086).

Expectation on record: at d200 the two frames should be near-identical. FINDING: they are NOT — cos(W_e[j], dual_j) median 0.7036 (0.6091–0.8086), because the feature Gram's off-diagonals (mean |.| 0.056, max 0.253) accumulate over 99 rivals and tilt each dual well away from its feature row. The VPD components separate the frames cleanly: they match the DUAL rows (median 0.91) far better than the feature rows (median 0.63) — an independent confirmation of the dual-frame re-scoring result (VPD's embedded read ontology is the dual basis), now visible as a raw best-|cos| gap without any re-scoring.
