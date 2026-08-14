### Biregular K5 (overlap-min): L4 reconstruction loss — naive MLP(d_mlp=50) + no-superposition baselines vs pinv/transpose decoder

| decoder | p=0.001 | p=0.01 | p=0.1 | p=0.5 | p=0.8 |
|---|---|---|---|---|---|
| naive MLP d50 | 5.952e-07 | 1.241e-05 | 7.010e-04 | 6.936e-03 | 9.197e-03 |
| ignore-50 (no superposition) | 4.623e-05 | 4.882e-04 | 4.993e-03 | 2.509e-02 | 4.009e-02 |
| pinv decoder | 8.213e-07 | 1.361e-05 | 7.550e-04 | 1.421e-02 | 2.898e-02 |
| transpose decoder | 6.165e-06 | 8.056e-05 | 2.097e-03 | 3.448e-02 | 7.527e-02 |
| ratio (transpose/pinv) | 7.51x | 5.92x | 2.78x | 2.43x | 2.60x |
| ratio (pinv/naive) | 1.38x | 1.10x | 1.08x | 2.05x | 3.15x |
| ratio (transpose/naive) | 10.36x | 6.49x | 2.99x | 4.97x | 8.18x |
| ratio (ignore50/naive) | 77.67x | 39.36x | 7.12x | 3.62x | 4.36x |
