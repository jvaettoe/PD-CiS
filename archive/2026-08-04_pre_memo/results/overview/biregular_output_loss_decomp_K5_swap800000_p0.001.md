### On/off-target loss decomposition over all components (K5, p=0.001_swap800000)

| model | MSE on-target | MSE off-target | MSE total | L4 on-target | L4 off-target | L4 total |
|---|---|---|---|---|---|---|
| biregular pinv (overlap-min) | 4.266e-02 | 6.307e-01 | 6.734e-01 | 1.887e-03 | 7.052e-03 | 8.939e-03 |
| biregular transpose (overlap-min) | 2.620e-02 | 1.646e+00 | 1.672e+00 | 6.864e-04 | 3.188e-02 | 3.257e-02 |
| MLP (fully-trained d50) | 3.355e-02 | 6.781e-01 | 7.116e-01 | 1.126e-03 | 5.081e-03 | 6.207e-03 |
| ignore-50 (no superposition) | 5.000e-01 | 0.000e+00 | 5.000e-01 | 5.000e-01 | 0.000e+00 | 5.000e-01 |
