# fn_sweep: per-cell K* and oracle basis preference

p = 0.001, decoder = transpose, code seed 0; totals are the weighted VPD objective (stoch+PGD recon + importance at imp 1e-4 / pnorm 2.0) with oracle gates; combo key = (read basis, write basis). Note: in K* = 1, b = 0 cells the nonzero weight pieces of the two bases coincide and recon is 0 for all combos -- the margin there is carried entirely by gate sharpness (h != 0 is sign-aware, x != 0 is not), i.e. by the indexing, not the piece shapes.

| F | N | K* | l4* | computed | (f,f) | (f,n) | (n,f) | (n,n) | W_in pref (margin) | W_out pref (margin) |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 20 | 5 | 6.46e-06 | 0 | 3.256e-04 | 4.311e-05 | 3.607e-04 | 7.824e-05 | feature (1.81x) | neuron (7.55x) |
| 40 | 30 | 4 | 3.19e-06 | 21 | 2.148e-04 | 6.262e-05 | 2.694e-04 | 1.173e-04 | feature (1.87x) | neuron (3.43x) |
| 40 | 50 | 1 | 0.00e+00 | 40 | 7.983e-06 | 6.000e-06 | 6.000e-06 | 4.017e-06 | neuron (1.49x) | neuron (1.49x) |
| 40 | 75 | 1 | 0.00e+00 | 40 | 7.983e-06 | 6.000e-06 | 6.000e-06 | 4.017e-06 | neuron (1.49x) | neuron (1.49x) |
| 40 | 100 | 1 | 0.00e+00 | 40 | 7.983e-06 | 6.000e-06 | 6.000e-06 | 4.017e-06 | neuron (1.49x) | neuron (1.49x) |
| 40 | 150 | 1 | 0.00e+00 | 40 | 7.983e-06 | 6.000e-06 | 6.000e-06 | 4.017e-06 | neuron (1.49x) | neuron (1.49x) |
| 60 | 20 | 4 | 8.41e-06 | 0 | 3.642e-04 | 6.549e-05 | 4.176e-04 | 1.188e-04 | feature (1.81x) | neuron (5.56x) |
| 60 | 30 | 3 | 4.69e-06 | 0 | 3.011e-04 | 9.538e-05 | 3.844e-04 | 1.786e-04 | feature (1.87x) | neuron (3.16x) |
| 60 | 50 | 4 | 2.75e-06 | 54 | 2.296e-04 | 1.550e-04 | 3.724e-04 | 2.977e-04 | feature (1.92x) | neuron (1.48x) |
| 60 | 75 | 1 | 0.00e+00 | 60 | 1.217e-05 | 9.103e-06 | 9.103e-06 | 6.031e-06 | neuron (1.51x) | neuron (1.51x) |
| 60 | 100 | 1 | 0.00e+00 | 60 | 1.217e-05 | 9.103e-06 | 9.103e-06 | 6.031e-06 | neuron (1.51x) | neuron (1.51x) |
| 100 | 20 | 4 | 1.09e-05 | 0 | 5.212e-04 | 1.054e-04 | 6.067e-04 | 1.909e-04 | feature (1.81x) | neuron (4.94x) |
| 100 | 30 | 4 | 7.09e-06 | 0 | 4.355e-04 | 1.532e-04 | 5.687e-04 | 2.865e-04 | feature (1.87x) | neuron (2.84x) |
| 100 | 50 | 5 | 3.68e-06 | 98 | 3.767e-04 | 2.488e-04 | 6.055e-04 | 4.777e-04 | feature (1.92x) | neuron (1.51x) |
| 150 | 20 | 4 | 1.39e-05 | 0 | 7.054e-04 | 1.550e-04 | 8.304e-04 | 2.800e-04 | feature (1.81x) | neuron (4.55x) |
| 150 | 30 | 4 | 9.11e-06 | 0 | 5.799e-04 | 2.251e-04 | 7.751e-04 | 4.203e-04 | feature (1.87x) | neuron (2.58x) |
| 200 | 20 | 3 | 1.63e-05 | 0 | 8.027e-04 | 2.012e-04 | 9.639e-04 | 3.624e-04 | feature (1.80x) | neuron (3.99x) |
