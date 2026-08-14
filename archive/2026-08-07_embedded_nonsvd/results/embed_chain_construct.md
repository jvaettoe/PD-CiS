# Closed-form 100-chain decomposition with output self-activation matching (no learned parameters)

## d_emb = 20  — primal contacts + kernel-corrected writes

- solve: D-system: 100x4000, lstsq residual 1.33e-15, ||D|| = 34.611
- (1) faithfulness EXACT, zero remainder: dev W_in 1.7e-15, W_out 2.6e-15
- (2) all pieces rank-1 by construction
- (3) chain_jj == R_jj: max dev 1.11e-15 over all 100 features (mean R_jj 0.572)
- hidden-pattern divergence: cos(a_j, r_j) mean 0.893 (min 0.726); cos(relu a_j, relu r_j) mean 0.917 — chains compute through DIFFERENT neuron patterns than the model
- chain off-target leak on e_j (j=[0, 33, 66]): 1.309, 1.192, 1.349
- cross-activation (|gate| > 0.1 on other features' inputs): j=0: 68/99, j=33: 67/99, j=66: 73/99  (Gram offdiag rms 0.223)
- figure: embed_construct_spectrum_demb20.png

## d_emb = 50  — primal contacts + kernel-corrected writes

- solve: D-system: 100x2500, lstsq residual 4.33e-15, ||D|| = 10.951
- (1) faithfulness EXACT, zero remainder: dev W_in 2.4e-15, W_out 2.9e-15
- (2) all pieces rank-1 by construction
- (3) chain_jj == R_jj: max dev 4.33e-15 over all 100 features (mean R_jj 0.776)
- hidden-pattern divergence: cos(a_j, r_j) mean 0.687 (min 0.331); cos(relu a_j, relu r_j) mean 0.828 — chains compute through DIFFERENT neuron patterns than the model
- chain off-target leak on e_j (j=[0, 33, 66]): 1.090, 1.091, 1.021
- cross-activation (|gate| > 0.1 on other features' inputs): j=0: 51/99, j=33: 45/99, j=66: 44/99  (Gram offdiag rms 0.140)
- figure: embed_construct_spectrum_demb50.png

## d_emb = 200  — dual (d_emb >= F: primal-contact system has no kernel freedom)

- solve: n/a
- (1) faithfulness up to the inert remainder: dev 7.8e-16 / 1.3e-15 (forward-exact as before)
- (2) all pieces rank-1 by construction
- (3) chain_jj == R_jj: max dev 1.22e-15 over all 100 features (mean R_jj 0.817)
- chain off-target leak on e_j (j=[0, 33, 66]): 0.000, 0.000, 0.000
- cross-activation (|gate| > 0.1 on other features' inputs): j=0: 0/99, j=33: 0/99, j=66: 0/99  (Gram offdiag rms 0.071)
- figure: embed_construct_spectrum_demb200.png

## d_emb = 1000  — dual (d_emb >= F: primal-contact system has no kernel freedom)

- solve: n/a
- (1) faithfulness up to the inert remainder: dev 1.9e-16 / 7.9e-16 (forward-exact as before)
- (2) all pieces rank-1 by construction
- (3) chain_jj == R_jj: max dev 1.89e-15 over all 100 features (mean R_jj 0.817)
- chain off-target leak on e_j (j=[0, 33, 66]): 0.000, 0.000, 0.000
- cross-activation (|gate| > 0.1 on other features' inputs): j=0: 0/99, j=33: 0/99, j=66: 0/99  (Gram offdiag rms 0.031)
- figure: embed_construct_spectrum_demb1000.png

Verdict: the user's conjecture holds — an exact 100-chain PD with (1) faithfulness, (2) rank-1, (3) output self-activation matching exists in closed form at every d_emb, including d_emb < F. The trace/rigidity obstruction applies only to hidden-level matching, and manifests here as the measured hidden-pattern divergence: the chains' neuron patterns differ from the model's own.
