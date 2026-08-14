# Exact per-feature decomposition of trained EmbedMLP (dual-basis construction)

## d_emb = 1000  (F=100, N=50)  — regime: exact alignment (d_emb >= F)

- (a) faithfulness: pieces sum to P·W_in; +remainder = W_in/W_out exact (dev 1.9e-16/7.9e-16); G = I to 1.3e-15
- (b) remainders inert: projected model == original (max dev 1.4e-14); ||(I−P)W_in||/||W_in|| = 0.683, ||W_out(I−P)||/||W_out|| = 0.173 (frozen init noise)
- (c) reads: piece j of W_in contributes (x·G)_j · r_j to the preactivation exactly
- (d) writes: piece j of W_out writes (h·c_j)·G_col_j exactly (= only output j)
- (e) piece-usage count per input == #active features (read strengths reproduce x exactly, dev 6.7e-16)
- (f) ranks: pieces rank-1 by construction; remainder(W_in) rank = 50 (predicted min(d_emb − F, N) = 50)
- (g) SVD contrast: W_in's top singular direction reads 95.2 features at once (participation ratio; feature-aligned pieces read exactly 1) — SVD pieces are faithful and rank-1 but not feature-aligned
- behavioral: solo chain vs full model diag j=0: 0.822/0.822, j=33: 0.813/0.813, j=66: 0.816/0.816  (figure: embed_exact_chain_vs_full_demb1000.png)

## d_emb = 200  (F=100, N=50)  — regime: exact alignment (d_emb >= F)

- (a) faithfulness: pieces sum to P·W_in; +remainder = W_in/W_out exact (dev 7.8e-16/1.3e-15); G = I to 1.1e-15
- (b) remainders inert: projected model == original (max dev 1.0e-14); ||(I−P)W_in||/||W_in|| = 0.243, ||W_out(I−P)||/||W_out|| = 0.125 (frozen init noise)
- (c) reads: piece j of W_in contributes (x·G)_j · r_j to the preactivation exactly
- (d) writes: piece j of W_out writes (h·c_j)·G_col_j exactly (= only output j)
- (e) piece-usage count per input == #active features (read strengths reproduce x exactly, dev 1.2e-15)
- (f) ranks: pieces rank-1 by construction; remainder(W_in) rank = 50 (predicted min(d_emb − F, N) = 50)
- (g) SVD contrast: W_in's top singular direction reads 92.5 features at once (participation ratio; feature-aligned pieces read exactly 1) — SVD pieces are faithful and rank-1 but not feature-aligned
- behavioral: solo chain vs full model diag j=0: 0.823/0.823, j=33: 0.817/0.817, j=66: 0.815/0.815  (figure: embed_exact_chain_vs_full_demb200.png)

## d_emb = 50  (F=100, N=50)  — regime: soft alignment (d_emb < F)

- (a) faithfulness: P = I so pieces sum to W_in/W_out EXACTLY, ZERO remainder (dev 3.0e-15). G = rank-50 projector: mean G_jj = 0.500 (= d_emb/F = 0.50), mean |G_ij| offdiag = 0.0395
- (b) no remainder exists (P = I); nothing to project out (check dev 1.8e-14)
- (c) reads: piece j of W_in contributes (x·G)_j · r_j to the preactivation exactly — self term G_jj·x_j + embedding leak Σ G_ij·x_i
- (d) writes: piece j of W_out writes (h·c_j)·G_col_j exactly — G-shaped write leak
- (e) read strengths == x·G exactly (dev 5.6e-16); pieces are SOFT-gated — every piece receives leak, usage is graded not binary
- (f) ranks: pieces rank-1 by construction; remainder identically 0
- (g) SVD contrast: W_in's top singular direction reads 42.6 features at once (participation ratio; feature-aligned pieces read exactly 1) — SVD pieces are faithful and rank-1 but not feature-aligned
- behavioral: solo chain vs full model diag j=0: 0.790/0.221, j=33: 0.768/0.179, j=66: 0.813/0.261  (figure: embed_exact_chain_vs_full_demb50.png)

## d_emb = 20  (F=100, N=50)  — regime: soft alignment (d_emb < F)

- (a) faithfulness: P = I so pieces sum to W_in/W_out EXACTLY, ZERO remainder (dev 9.4e-16). G = rank-20 projector: mean G_jj = 0.200 (= d_emb/F = 0.20), mean |G_ij| offdiag = 0.0319
- (b) no remainder exists (P = I); nothing to project out (check dev 8.2e-15)
- (c) reads: piece j of W_in contributes (x·G)_j · r_j to the preactivation exactly — self term G_jj·x_j + embedding leak Σ G_ij·x_i
- (d) writes: piece j of W_out writes (h·c_j)·G_col_j exactly — G-shaped write leak
- (e) read strengths == x·G exactly (dev 1.1e-16); pieces are SOFT-gated — every piece receives leak, usage is graded not binary
- (f) ranks: pieces rank-1 by construction; remainder identically 0
- (g) SVD contrast: W_in's top singular direction reads 30.5 features at once (participation ratio; feature-aligned pieces read exactly 1) — SVD pieces are faithful and rank-1 but not feature-aligned
- behavioral: solo chain vs full model diag j=0: 0.557/0.020, j=33: 0.568/0.025, j=66: 0.550/0.015  (figure: embed_exact_chain_vs_full_demb20.png)

Remark (corrected): for d_emb < F the strict dual basis does not exist, but the pseudo-dual construction DEGRADES GRACEFULLY, not catastrophically: W_e has full column rank, so P = I and the pieces sum to W_in/W_out EXACTLY with zero remainder — faithfulness is cleaner than for d_emb >= F. Only feature alignment softens: reads/writes follow G = W_e W_e^+ (the rank-d_emb feature-space projector) — self strength G_jj (mean d_emb/F) plus embedding leak G_ij, the continuous analog of codeword-overlap noise.
