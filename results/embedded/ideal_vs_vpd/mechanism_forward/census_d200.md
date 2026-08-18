# Mechanism forward-pass census — d_emb = 200 (designed sandwich, transpose decoder)

Checkpoint: `checkpoints/embedded_ideal/ideal_d200.pt`  (K=5, p=0.001, a=0.3646, b=-0.0476).  Probes: one-hot v·e_j, v=1 (errors are v-invariant).  Option A (activation-direction projections) rejected: it uses W_emb knowledge at inference time; the decomposition must be fixed before inference. All ontologies below are exact tilings of W_in driven only by chain activations.

## Exactness asserts (identities, not findings)

```
W_out == W_in.T: bitwise PASS
row tiling  sum_d e_d(x)W_in[d,:] == W_in: bitwise PASS
column tiling sum_n W_in[:,n](x)e_n == W_in: bitwise PASS
SVD tiling   max|U S V^T - W_in|: 1.384e-15  (tol 1e-12) PASS
loadings     max|L - W_e U|: 0.000e+00  (tol 1e-12) PASS
row chain    max|sum_d (a_j)_d R_d - a_j W_in|: 4.996e-16  (tol 1e-12) PASS
row chain    max|Y_rows - Y_full|: 3.331e-16  (tol 1e-12) PASS
svd chain    max|sum_k s_k L_jk v_k - a_j W_in|: 1.568e-15  (tol 1e-12) PASS
svd chain    max|Y_svd - Y_full|: 9.992e-16  (tol 1e-12) PASS
column chain: coefficients (a_j W_in)_n are the pre-activations (identical forward by construction)
context      max|W_e W_in - C| = 1.006e-15  (G = I at d_emb=200 >= F)
```

## Activity census (no cutoffs: PR + multi-threshold counts)

Energies per feature j: rows (W_e[j])_d^2 over 200 dims; columns ((a_j W_in))_n^2 over 50 neurons; SVD (s_k L_jk)^2 over 50 channels.

| ontology | pieces | PR median (min–max) | n50 med (min–max) | n90 med (min–max) | n99 med (min–max) |
|---|---|---|---|---|---|
| rows (per-emb-dim) | 200 | 68.5 (51.6–81.1) | 25 (20–30) | 90 (79–103) | 147 (138–156) |
| columns (per-neuron) | 50 | 6.6 (6.6–6.6) | 3 (3–3) | 17 (17–17) | 47 (47–47) |
| SVD (triplets) | 50 | 15.5 (5.3–21.5) | 6 (2–9) | 21 (14–26) | 37 (31–42) |

## Truncated-forward fidelity (tied-transpose top-m mechanism)

m* = smallest m with the error persistently below tolerance for all m' >= m (spike: |y_m[j]-y[j]|/|y[j]| <= 1%; leak: ||(y_m-y)_off||/||y_off|| <= 10%).

| ontology | m* spike<=1% med (min–max) | m* leak<=10% med (min–max) |
|---|---|---|
| rows (per-emb-dim) | 176 (140–192) | 198 (192–200) |
| columns (per-neuron) | 5 (5–5) | 5 (5–5) |
| SVD (triplets) | 40 (33–48) | 43 (34–50) |

## Verdict

Most causally concentrated ontology: **columns (per-neuron)** (median m* for faithful spike+leak: rows (per-emb-dim) = 198, columns (per-neuron) = 5, SVD (triplets) = 43). The column (per-neuron) census spreads pre-activation energy over the b-floor (n99 near all 50), but the ReLU kills every off-code (negative) pre-activation, so causally only the K on-code pieces carry the mechanism — the census and the causal count disagree, and the truncated-forward fidelity is the discriminator.
