### Sparsity-matched code density: K=20 vs K=5 vs trained models (p=0.001)

Motivation: trained MLPs at p=0.001 learn K_eff = 18-22, so K=20 is the
sparsity-matched designed code (neuron degree d=40; mean pairwise codeword
overlap K^2/N = 8, maxov 14).

| model | L4 (p=0.001) | n_soft | median R² |
|---|---|---|---|
| trained MLP d50 | 7.7e-07 | 96.4 | 0.964 |
| trained EmbedMLP d1000 | 7.2e-07 | 96.4 | 0.964 |
| designed K=5, pinv | 1.7e-06 | 94.0 | 0.943 |
| designed K=20, pinv | 1.8e-06 | 94.0 | 0.947 |
| designed K=5, transpose | 4.1e-06 | 88.7 | 0.886 |
| designed K=20, transpose | 5.9e-06 | 85.9 | 0.858 |

Findings:
- Matching the trained code DENSITY does not close the designed-vs-trained gap:
  pinv is K-insensitive (94.0 both), transpose gets slightly WORSE (88.7 -> 85.9).
  The trained model's advantage over the 2-scalar ansatz is in graded per-connection
  weights, not in codeword length.
- The overlap-level ladder still holds exactly at K=20 (bars on r(m)=m*a^2+(K-m)*ab
  to ~1e-8, rungs m=3..14) but becomes a near-continuum crossing ZERO at m ~ 7:
  with a^2=0.031 < 19*|ab|, a single shared neuron no longer flips the response
  positive (pos_m1 = -0.29) - dense codes semi-cancel their leakage instead of
  quantizing it.
- Fitted scalars scale as expected: transpose a 0.365 -> 0.177 (~1/sqrt(K)),
  b grows (-0.048 -> -0.097); interference slightly higher (0.671 -> 0.686).
