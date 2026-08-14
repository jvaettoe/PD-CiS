### On/off-target loss decomposition over all components (K5, p=0.001)

| model | MSE on-target | MSE off-target | MSE total | L4 on-target | L4 off-target | L4 total |
|---|---|---|---|---|---|---|
| biregular pinv (no-swap) | 5.918e-02 | 5.837e-01 | 6.429e-01 | 4.531e-03 | 1.139e-02 | 1.592e-02 |
| biregular transpose (no-swap) | 2.064e-02 | 1.510e+00 | 1.530e+00 | 4.262e-04 | 6.867e-02 | 6.910e-02 |
| MLP (fully-trained d50) | 3.355e-02 | 6.781e-01 | 7.116e-01 | 1.126e-03 | 5.081e-03 | 6.207e-03 |
| ignore-50 (no superposition) | 5.000e-01 | 0.000e+00 | 5.000e-01 | 5.000e-01 | 0.000e+00 | 5.000e-01 |
