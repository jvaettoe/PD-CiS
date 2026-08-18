# Cut-side principle test — 4 matrices x 2 cut directions (designed sandwich, d_emb = 200)

Checkpoint: `checkpoints/embedded_ideal/ideal_d200.pt` (K=5, p=0.001, a=0.3646, b=-0.0476). Model: y = relu(x W_e W_in) W_in^T W_e^T; matrices W_emb = W_e (F x d), W_in (d x N), W_out = W_in^T (N x d), W_unemb = W_e^T (d x F). Probes: one-hot v e_j (relative errors v-invariant, asserted). Each cell truncates ONLY its own matrix to the top-m pieces by |activation coefficient| — all other matrices stay full (untied; unlike the default run's tied truncation).

**Principle under test:** decompose each layer on the side where its own weights re-concentrate the signal into discrete events; never cut along a dense interface.

## Tiling + regression asserts (identities, not findings)

```
W_emb    input-side  per-feature rows     (100 pieces): tiling bitwise PASS
W_emb    output-side  per-emb-dim columns  (200 pieces): tiling bitwise PASS
W_in     input-side  per-emb-dim rows     (200 pieces): tiling bitwise PASS
W_in     output-side  per-neuron columns   ( 50 pieces): tiling bitwise PASS
W_out    input-side  per-neuron rows      ( 50 pieces): tiling bitwise PASS
W_out    output-side  per-emb-dim columns  (200 pieces): tiling bitwise PASS
W_unemb  input-side  per-emb-dim rows     (200 pieces): tiling bitwise PASS
W_unemb  output-side  per-feature columns  (100 pieces): tiling bitwise PASS

regression: W_in input/output cut census energies == existing rows/columns census: bitwise PASS
regression: W_in output-cut m* spike/leak = 5 for all 100 features (== established per-neuron columns result): PASS
duality: W_emb output-cut == W_in input-cut truncated forwards (same interface): bitwise PASS
duality: W_out output-cut == W_unemb input-cut truncated forwards (same interface): bitwise PASS
near-identity: W_emb input-cut errors <= 1.3e-15 for ALL m >= 1 (the active feature's row is the whole read): PASS
near-identity: W_unemb output-cut top piece IS the target readout column j (spike error 0 at m=1) for all features: PASS
v-invariance: m* (spike and leak) identical at v=1 and v=2.5 (features [0, 50], all 8 cells): PASS
```

## Activity census (no cutoffs: PR + multi-threshold counts)

| matrix | cut side | pieces | coefficient | PR med (min–max) | n50 | n90 | n99 |
|---|---|---|---|---|---|---|---|
| W_emb | input (per-feature rows) | 100 | x_i (one-hot probe entry) | 1.0 (1.0–1.0) | 1 (1–1) | 1 (1–1) | 1 (1–1) |
| W_emb | output (per-emb-dim columns) | 200 | a_d = (x W_e)_d | 68.5 (51.6–81.1) | 25 (20–30) | 90 (79–103) | 147 (138–156) |
| W_in | input (per-emb-dim rows) | 200 | a_d (arriving embedding) | 68.5 (51.6–81.1) | 25 (20–30) | 90 (79–103) | 147 (138–156) |
| W_in | output (per-neuron columns) | 50 | pre_n = (a W_in)_n | 6.6 (6.6–6.6) | 3 (3–3) | 17 (17–17) | 47 (47–47) |
| W_out | input (per-neuron rows) | 50 | h_n = relu(pre)_n (post-ReLU) | 5.0 (5.0–5.0) | 3 (3–3) | 5 (5–5) | 5 (5–5) |
| W_out | output (per-emb-dim columns) | 200 | z_d = (h W_out)_d | 68.4 (52.5–85.3) | 26 (21–32) | 89 (73–100) | 148 (133–159) |
| W_unemb | input (per-emb-dim rows) | 200 | z_d (arriving write-out) | 68.4 (52.5–85.3) | 26 (21–32) | 89 (73–100) | 148 (133–159) |
| W_unemb | output (per-feature columns) | 100 | y_i = (z W_e^T)_i | 8.0 (6.2–10.5) | 6 (4–8) | 68 (66–73) | 97 (96–98) |

## Truncated-forward fidelity (only that matrix truncated)

m* = smallest m with the error persistently below tolerance for all m' >= m (spike <= 1%, leak <= 10%).

| matrix | cut side | pieces | m* spike med (min–max) | m* leak med (min–max) |
|---|---|---|---|---|
| W_emb | input (per-feature rows) | 100 | 1 (1–1) | 1 (1–1) |
| W_emb | output (per-emb-dim columns) | 200 | 162 (133–188) | 110 (82–138) |
| W_in | input (per-emb-dim rows) | 200 | 162 (133–188) | 110 (82–138) |
| W_in | output (per-neuron columns) | 50 | 5 (5–5) | 5 (5–5) |
| W_out | input (per-neuron rows) | 50 | 5 (5–5) | 5 (5–5) |
| W_out | output (per-emb-dim columns) | 200 | 161 (128–183) | 153 (142–165) |
| W_unemb | input (per-emb-dim rows) | 200 | 161 (128–183) | 153 (142–165) |
| W_unemb | output (per-feature columns) | 100 | 1 (1–1) | 98 (98–99) |

## Verdicts (predictions: W_emb input m*=1; W_in output m*=5; W_out input m*=5; W_unemb output m*=1)

- **W_emb** — predicted winner: input-side at m*=1. Actual winner: **input-side** (combined median m* 1 vs 162 for the loser); winner m* spike/leak = 1/1. Side prediction CONFIRMED; count prediction spike CONFIRMED, leak CONFIRMED.
- **W_in** — predicted winner: output-side at m*=5. Actual winner: **output-side** (combined median m* 5 vs 162 for the loser); winner m* spike/leak = 5/5. Side prediction CONFIRMED; count prediction spike CONFIRMED, leak CONFIRMED.
- **W_out** — predicted winner: input-side at m*=5. Actual winner: **input-side** (combined median m* 5 vs 161 for the loser); winner m* spike/leak = 5/5. Side prediction CONFIRMED; count prediction spike CONFIRMED, leak CONFIRMED.
- **W_unemb** — predicted winner: output-side at m*=1. Actual winner: **output-side** (combined median m* 98 vs 161 for the loser); winner m* spike/leak = 1/98. Side prediction CONFIRMED; count prediction spike CONFIRMED, leak FAILED.

## Overall

Winner-side predictions confirmed: 4/4. Winner-count predictions: spike 4/4, leak 3/4.

Reading the one deviating number (W_unemb output-cut leak m*): at the FINAL matrix the output-side cut is coordinate-aligned with the error metric itself — dropping a piece deletes that output coordinate outright, so leak fidelity degenerates into a census of the output vector's leak support (m* leak = 98 vs n99 = 97 of the y-census) rather than a measure of mechanism concentration. The mechanism the layer re-concentrates — the target readout <z, W_e[j]> — is exactly 1 piece (spike m* = 1, error 0 at m = 1 for every feature); the leak is residual noise smeared over all coordinates, which no coordinate cut can keep without keeping the coordinates. The mirror cell W_emb input-cut shows both criteria at 1 only because the probe is genuinely 1-sparse: there the input signal, not just the mechanism, is 1 piece.

The dense-interface cells are not merely diffuse — they are the SAME decomposition seen from either side (bitwise-equal truncated forwards, asserted above): a cut along the embedding interface is one object whether booked to the matrix writing it or the matrix reading it, and it needs ~110–160 of 200 pieces. The principle survives unfalsified: every matrix is causally minimal exactly on the side where its own weights re-concentrate the signal into discrete events (feature rows / neuron columns / neuron rows / feature columns), never along the dense embedding interface.
