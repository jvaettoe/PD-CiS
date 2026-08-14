### Per-feature R^2 census — trained EmbedMLP, p=0.001 (F=100 ReLUs, N=50 neurons; G = W_e·W_e⁺ alignment)

| d_emb | mean G_jj | mean |G_ij| | n_soft | min R2 | median R2 | >0.5 | >0.8 | >0.9 | >0.95 | >0.99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.200 | 0.0319 | 81.1 | 0.726 | 0.811 | 100 | 62 | 0 | 0 | 0 |
| 50 | 0.500 | 0.0395 | 94.6 | 0.912 | 0.947 | 100 | 100 | 100 | 43 | 0 |
| 200 | 1.000 | 0.0000 | 96.4 | 0.952 | 0.964 | 100 | 100 | 100 | 100 | 0 |
| 1000 | 1.000 | 0.0000 | 96.4 | 0.953 | 0.964 | 100 | 100 | 100 | 100 | 0 |

(Non-embedded biregular baseline at p=0.001, transpose no-swap: n_soft 88.7, min 0.859, median 0.886 — see results/overview/biregular_r2_K5_p0.001.md)
