### ramp census — target ramp(x) = ReLU(x) − ReLU(x−0.5), L4, p = 0.01, seed 42; computed = active-cond MSE < 0.0167 (baseline 0.167); bias = hidden-layer biases (no = homogeneity control)

| arch | bias | p | computed /100 | MSE median | K1(>0) | K1(tau) | K2(>0) | K2(tau) |
|---|---|---|---|---|---|---|---|---|
| chain | no | 0.01 | 100 | 1.46e-02 | 12.6±2.0 | 8.6 | 45.4±1.9 | 33.4 |
| chain | yes | 0.01 | 100 | 1.12e-02 | 38.6±2.8 | 9.3 | 42.6±1.2 | 41.3 |
| single | no | 0.01 | 100 | 1.45e-02 | 12.0±2.0 | 8.4 | — | — |
| single | yes | 0.01 | 100 | 1.38e-02 | 25.6±2.5 | 8.8 | — | — |
