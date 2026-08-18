# Mechanism solo spectra — d_emb = 50 (winning-cut convention, 1+5+5+1 pieces)

Checkpoint: `checkpoints/embedded_ideal/ideal_d50.pt` (K=5, p=0.001, a=0.3646, b=-0.0476). Convention (per probed feature j): W_emb row j + top-5-by-|coefficient| W_in column pieces + the same-neuron W_out row pieces (tied) + W_unemb column j. Top-5 kept for d200 comparability.

**Regime:** W_e W_in = G C, G a rank-50 projector (mean G_jj = 0.500); the true causal count #(pre > 0) is median 21 (14–26) over all 100 features — it differs from the kept 5.

Structural note: the strict convention's W_unemb piece (column j) writes ONLY output coordinate j, so the strict mechanism spectrum has off-target EXACTLY 0 by construction at any d_emb. The G-noise that flows through the retained pieces is therefore measured at the retained write z read through the FULL unembed bank ("full-read leak" below) — the leak the mechanism's pieces hand to the other readout columns, which the strict piece set discards.

## Per-probe numbers

| j | full spike | mech spike | spike err | mech off-max (strict) | mech off-max (full-read) | full off-max | causal count #(pre>0) | kept == on-code? |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.2034 | 0.1480 | 2.72e-01 | 0 (structural) | 0.1168 | 0.1145 | 19 | no (4/5 on-code) |
| 61 | 0.1768 | 0.0828 | 5.32e-01 | 0 (structural) | 0.0878 | 0.1525 | 20 | no (3/5 on-code) |
| 1 | 0.1901 | 0.1303 | 3.15e-01 | 0 (structural) | 0.1221 | 0.0963 | 20 | no (4/5 on-code) |

## Prediction check: spike ≈ K·a²·G_jj² (all features)

K a² = 0.6645. mech-spike / (K a² G_jj²): median 1.061 (0.447–1.920); full-spike / (K a² G_jj²): median 1.378 (0.805–2.470). mech spike median 0.1682 (0.0828–0.3483).

## Leak ladder (full model, off-target values by codeword overlap)

| j | m_ov | n | mean | std | min–max | d200 mean | d200 std |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 58 | -0.0252 | 3.53e-02 | -0.0808–0.0949 | -0.0867 | 2.65e-16 |
| 0 | 1 | 37 | 0.0194 | 4.09e-02 | -0.0424–0.1145 | 0.0635 | 2.80e-16 |
| 0 | 2 | 4 | 0.0340 | 2.04e-02 | 0.0088–0.0594 | 0.2138 | 4.29e-16 |
| 61 | 0 | 60 | -0.0215 | 3.35e-02 | -0.0877–0.0662 | -0.0867 | 2.38e-16 |
| 61 | 1 | 34 | 0.0260 | 4.49e-02 | -0.0617–0.1525 | 0.0635 | 2.69e-16 |
| 61 | 2 | 4 | 0.0234 | 1.77e-02 | 0.0047–0.0419 | 0.2138 | 2.31e-16 |
| 61 | 3 | 1 | 0.0956 | 0.00e+00 | 0.0956–0.0956 | 0.3640 | 0.00e+00 |
| 1 | 0 | 62 | -0.0186 | 3.11e-02 | -0.0902–0.0577 | -0.0867 | 3.11e-16 |
| 1 | 1 | 30 | 0.0154 | 4.03e-02 | -0.0741–0.0963 | 0.0635 | 3.89e-16 |
| 1 | 2 | 6 | 0.0483 | 1.39e-02 | 0.0256–0.0696 | 0.2138 | 3.25e-16 |
| 1 | 3 | 1 | 0.0614 | 0.00e+00 | 0.0614–0.0614 | 0.3640 | 0.00e+00 |

Smear verdict: max within-class std at d50 = 4.489e-02 vs the d200 exact ladder (per-class std ~1e-16, machine zero) — the ladder is SMEARED into bands.
