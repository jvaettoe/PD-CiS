### On/off-target loss decomposition over all components (K5, p=0.1_swap800000)

| model | MSE on-target | MSE off-target | MSE total | L4 on-target | L4 off-target | L4 total |
|---|---|---|---|---|---|---|
| biregular pinv (overlap-min) | 5.531e-02 | 5.858e-01 | 6.411e-01 | 3.141e-03 | 6.083e-03 | 9.223e-03 |
| biregular transpose (overlap-min) | 1.101e-02 | 9.610e-01 | 9.720e-01 | 1.212e-04 | 1.160e-02 | 1.172e-02 |
| MLP (fully-trained d50) | 8.271e-02 | 5.138e-01 | 5.965e-01 | 6.855e-03 | 7.798e-03 | 1.465e-02 |
| ignore-50 (no superposition) | 5.000e-01 | 0.000e+00 | 5.000e-01 | 5.000e-01 | 0.000e+00 | 5.000e-01 |
