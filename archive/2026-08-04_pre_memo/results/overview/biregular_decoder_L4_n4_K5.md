### Biregular K5 (no overlap-min): L4 reconstruction loss — naive MLP(d_mlp=50) + no-superposition baselines vs pinv/transpose decoder

| decoder | p=0.001 | p=0.01 | p=0.1 | p=0.5 | p=0.8 |
|---|---|---|---|---|---|
| naive MLP d50 | 5.952e-07 | 1.241e-05 | 7.010e-04 | 6.936e-03 | 9.197e-03 |
| ignore-50 (no superposition) | 4.623e-05 | 4.882e-04 | 4.993e-03 | 2.509e-02 | 4.009e-02 |
| pinv decoder | 1.439e-06 | 2.235e-05 | 1.007e-03 | 1.726e-02 | 3.450e-02 |
| transpose decoder | 1.626e-05 | 1.856e-04 | 3.353e-03 | 5.190e-02 | 1.143e-01 |
| ratio (transpose/pinv) | 11.30x | 8.30x | 3.33x | 3.01x | 3.31x |
| ratio (pinv/naive) | 2.42x | 1.80x | 1.44x | 2.49x | 3.75x |
| ratio (transpose/naive) | 27.31x | 14.96x | 4.78x | 7.48x | 12.43x |
| ratio (ignore50/naive) | 77.67x | 39.36x | 7.12x | 3.62x | 4.36x |
