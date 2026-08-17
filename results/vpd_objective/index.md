# VPD-objective runs (Experiment 1: does the truth cost more?)

Fixed eval pool: CleanDataset p, seed/batches/batch as per each run's config; term weights = the port's defaults. Raw term values; `total` is the weighted objective.

| tag | timestamp | steps | total | stoch | pgd | imp | faithful | stoch w_in | stoch w_out | imp w_in | imp w_out | act_in | act_out | ci_l0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_oracle | 2026-08-17T14:38:09 | 0 | 3.315e-04 | 3.900e-05 | 1.167e-04 | 1.998e-01 | 0.000e+00 | 0.000e+00 | 5.157e-05 | 9.990e-02 | 9.990e-02 | 1.00 | 1.00 | 0.10 |
| learned_reference | 2026-08-17T14:38:14 | 25000 | 5.173e-05 | 4.340e-06 | 8.879e-06 | 2.530e-01 | 1.464e-04 | 2.965e-06 | 3.428e-06 | 6.642e-02 | 1.866e-01 | 0.89 | 0.10 | 0.34 |
