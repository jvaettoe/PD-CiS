# Cut-side principle test — 4 matrices x 2 cut directions (designed sandwich, d_emb = 50)

Checkpoint: `checkpoints/embedded_ideal/ideal_d50.pt` (K=5, p=0.001, a=0.3646, b=-0.0476). Model: y = relu(x W_e W_in) W_in^T W_e^T; matrices W_emb = W_e (F x d), W_in (d x N), W_out = W_in^T (N x d), W_unemb = W_e^T (d x F). Probes: one-hot v e_j (relative errors v-invariant, asserted). Each cell truncates ONLY its own matrix to the top-m pieces by |activation coefficient| — all other matrices stay full (untied; unlike the default run's tied truncation).

**Principle under test:** decompose each layer on the side where its own weights re-concentrate the signal into discrete events; never cut along a dense interface.

**Regime (d_emb = 50 < F = 100):** W_e W_in = G C with G = W_e W_e^T a rank-50 projector (mean G_jj = 0.500, max |W_e W_in - G C| = 3.9e-16). Off-code preactivations go positive through G-noise: the causal count #(pre > 0) per feature is median 21 (14–26) vs exactly K = 5 at d200, and 100/100 features deviate from K. The d200 counts {1,5,5,1} are the REFERENCE here, not asserted predictions — knees are expected to soften.

## Tiling + regression asserts (identities, not findings)

```
W_emb    input-side  per-feature rows     (100 pieces): tiling bitwise PASS
W_emb    output-side  per-emb-dim columns  ( 50 pieces): tiling bitwise PASS
W_in     input-side  per-emb-dim rows     ( 50 pieces): tiling bitwise PASS
W_in     output-side  per-neuron columns   ( 50 pieces): tiling bitwise PASS
W_out    input-side  per-neuron rows      ( 50 pieces): tiling bitwise PASS
W_out    output-side  per-emb-dim columns  ( 50 pieces): tiling bitwise PASS
W_unemb  input-side  per-emb-dim rows     ( 50 pieces): tiling bitwise PASS
W_unemb  output-side  per-feature columns  (100 pieces): tiling bitwise PASS

regression: W_in input/output cut census energies == existing rows/columns census: bitwise PASS
regime report (d_emb < F, no assert): W_in output-cut m* spike med 34 (25–42), leak med 33 (18–42) (d200 reference: 5/5 for every feature)
duality: W_emb output-cut == W_in input-cut truncated forwards (same interface): bitwise PASS
duality: W_out output-cut == W_unemb input-cut truncated forwards (same interface): bitwise PASS
near-identity: W_emb input-cut errors <= 7.4e-16 for ALL m >= 1 (the active feature's row is the whole read): PASS
regime report (d_emb < F, no assert): W_unemb output-cut top piece is the target readout for 100/100 features (spike error at m=1 is 0 for those; max 0.0e+00) (d200 reference: 100/100)
v-invariance: m* (spike and leak) identical at v=1 and v=2.5 (features [0, 50], all 8 cells): PASS
```

## Activity census (no cutoffs: PR + multi-threshold counts)

| matrix | cut side | pieces | coefficient | PR med (min–max) | n50 | n90 | n99 |
|---|---|---|---|---|---|---|---|
| W_emb | input (per-feature rows) | 100 | x_i (one-hot probe entry) | 1.0 (1.0–1.0) | 1 (1–1) | 1 (1–1) | 1 (1–1) |
| W_emb | output (per-emb-dim columns) | 50 | a_d = (x W_e)_d | 18.4 (11.6–29.8) | 7 (4–11) | 23 (17–31) | 38 (30–43) |
| W_in | input (per-emb-dim rows) | 50 | a_d (arriving embedding) | 18.4 (11.6–29.8) | 7 (4–11) | 23 (17–31) | 38 (30–43) |
| W_in | output (per-neuron columns) | 50 | pre_n = (a W_in)_n | 13.5 (7.2–19.5) | 5 (3–8) | 20 (15–26) | 36 (30–41) |
| W_out | input (per-neuron rows) | 50 | h_n = relu(pre)_n (post-ReLU) | 6.0 (3.2–8.7) | 3 (2–4) | 7 (4–10) | 14 (9–20) |
| W_out | output (per-emb-dim columns) | 50 | z_d = (h W_out)_d | 17.2 (9.3–23.9) | 7 (5–10) | 23 (18–29) | 37 (32–43) |
| W_unemb | input (per-emb-dim rows) | 50 | z_d (arriving write-out) | 17.2 (9.3–23.9) | 7 (5–10) | 23 (18–29) | 37 (32–43) |
| W_unemb | output (per-feature columns) | 100 | y_i = (z W_e^T)_i | 16.2 (8.1–26.5) | 9 (5–13) | 42 (34–51) | 73 (65–81) |

## Truncated-forward fidelity (only that matrix truncated)

m* = smallest m with the error persistently below tolerance for all m' >= m (spike <= 1%, leak <= 10%).

| matrix | cut side | pieces | m* spike med (min–max) | m* leak med (min–max) |
|---|---|---|---|---|
| W_emb | input (per-feature rows) | 100 | 1 (1–1) | 1 (1–1) |
| W_emb | output (per-emb-dim columns) | 50 | 43 (36–49) | 40 (31–47) |
| W_in | input (per-emb-dim rows) | 50 | 43 (36–49) | 40 (31–47) |
| W_in | output (per-neuron columns) | 50 | 34 (25–42) | 33 (18–42) |
| W_out | input (per-neuron rows) | 50 | 14 (9–20) | 14 (9–20) |
| W_out | output (per-emb-dim columns) | 50 | 43 (33–49) | 41 (35–46) |
| W_unemb | input (per-emb-dim rows) | 50 | 43 (33–49) | 41 (35–46) |
| W_unemb | output (per-feature columns) | 100 | 1 (1–1) | 75 (67–82) |

## Verdicts (d200-reference counts: W_emb input m*=1; W_in output m*=5; W_out input m*=5; W_unemb output m*=1)

- **W_emb** — d200-reference winner: input-side at m*=1. Actual winner: **input-side** (combined median m* 1 vs 43 for the loser); winner m* spike/leak = 1/1. Side prediction CONFIRMED; count prediction spike CONFIRMED, leak CONFIRMED.
- **W_in** — d200-reference winner: output-side at m*=5. Actual winner: **output-side** (combined median m* 34 vs 43 for the loser); winner m* spike/leak = 34/33. Side prediction CONFIRMED; count prediction spike DEVIATES from d200 (reported), leak DEVIATES from d200 (reported).
- **W_out** — d200-reference winner: input-side at m*=5. Actual winner: **input-side** (combined median m* 14 vs 43 for the loser); winner m* spike/leak = 14/14. Side prediction CONFIRMED; count prediction spike DEVIATES from d200 (reported), leak DEVIATES from d200 (reported).
- **W_unemb** — d200-reference winner: output-side at m*=1. Actual winner: **input-side** (combined median m* 43 vs 75 for the loser); winner m* spike/leak = 43/41. Side prediction DEVIATES from d200 (reported); count prediction spike DEVIATES from d200 (reported), leak DEVIATES from d200 (reported).

## Overall

Winner-side d200-reference counts confirmed: 3/4. Winner-count matches: spike 1/4, leak 1/4.

d50 contrast with the d200 grid: the exact d200 counts {1,5,5,1} are a G = I artifact; the re-concentration reading survives where the signal is genuinely evented (W_emb, W_out input; W_in output beats its dense-interface rival). The one side FLIP, W_unemb, is the d200-diagnosed criterion artifact now dominating the combined median: the output cut's spike m* is still 1 for every feature (the top piece IS the target readout, 100/100), but its leak m* (75, vs y-census n99 = 73) is a support census of the now-broadband G-noise leak, which at d50 exceeds the dense-interface cost. d200-reference median m* at d50: W_emb input 1/1 (d200: 1/1), W_in output 34/33 (d200: 5/5), W_out input 14/14 (d200: 5/5), W_unemb output 1/75 (d200: 1/98). W_emb input stays exact at any d (the probe is 1-sparse). For W_in/W_out the knee softens because G-noise sends off-code preactivations positive (causal count above; note the |coefficient| ranking also spends slots on large-|negative| dead pieces for the W_in output cut). Full per-feature spread is in the fidelity table above and the CSV.
