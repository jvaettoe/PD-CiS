# SVD truncation-denoising study (EmbedMLP, p=0.001)

leak_rel = off-target energy relative to the full model; sig_dev_rel = mean on-target deviation relative to mean |R_jj|. n95 = 95% Frobenius mass cutoff of W_in.

- d_emb=20 (rank 20, n95=16): at joint truncation n=n95: leak_rel=0.815, sig_dev_rel=0.361; n_soft 81.1 → 58.9
- d_emb=50 (rank 50, n95=31): at joint truncation n=n95: leak_rel=0.891, sig_dev_rel=0.584; n_soft 94.6 → 53.9
- d_emb=200 (rank 50, n95=41): at joint truncation n=n95: leak_rel=0.937, sig_dev_rel=0.327; n_soft 96.4 → 79.0
- d_emb=1000 (rank 50, n95=46): at joint truncation n=n95: leak_rel=0.969, sig_dev_rel=0.146; n_soft 96.4 → 90.3

Verdict: see curves — if the red (leak) and blue (signal-loss) curves track each other with no knee, mass-ordered truncation cannot denoise this model: interference is carried by the same components as the signal, at every d_emb.
