### ramp analysis — bias-free (homogeneity control); L4, p = 0.01, seed 42, theta = 0.5

## Transfer curves (one-hot probes, v in [-1, 1])

| arch | rise slope | plateau level | plateau slope | plateau span | kink found | kink position | homog. dev max | neg-side max |
|---|---|---|---|---|---|---|---|---|
| chain | 0.598 [0.596, 0.601] | 0.478 [0.477, 0.481] | 0.598 [0.596, 0.601] | 0.239 [0.238, 0.240] | 0/100 | — | 2.98e-07 | 0.031 [0.030, 0.034] |
| single | 0.598 [0.596, 0.602] | 0.479 [0.477, 0.481] | 0.598 [0.596, 0.602] | 0.239 [0.238, 0.241] | 0/100 | — | 1.79e-07 | 0.024 [0.022, 0.025] |

Ideal ramp: rise slope 1, plateau level 0.5, plateau slope 0, kink at v = 0.5. Bias-free nets are positively homogeneous (f(v e_j) = v f(e_j), v > 0): homog. dev max at float-noise level means plateaus are impossible on singletons; hidden biases make the theta-kink representable.

## Layer-2 bypass ablation (chain)

| target | clip frac | L4 original | identity ratio | fitted (linear) ratio |
|---|---|---|---|---|
| ramp (bias-free (homogeneity control)) | 0.053 | 5.27e-06 | 1.13 | 1.06 |
| relu (reference, toys/chain) | 0.103 | 1.33e-05 | 1.13 | 1.06 |

On-record prediction: the ramp makes layer 2 load-bearing (ratios well above the relu-target 1.13x / 1.06x).

## Unit-role census (one-hot probes at v = 0.25 vs 0.75)

Growth ratio h(0.75)/h(0.25) per active unit; linear scaling gives exactly 3. Saturation-implementer: ratio < 1.5; rise-implementer: ratio >= 2.5.

| arch | layer | active units/feature | saturators/feature | risers/feature |
|---|---|---|---|---|
| chain | L1 | 9.1 | 0.00 | 9.12 |
| chain | L2 | 35.3 | 0.00 | 35.28 |
| single | L1 | 9.0 | 0.00 | 8.98 |

## Single vs chain

| arch | held-out L4 | computed /100 |
|---|---|---|
| chain | 5.27e-06 | 100 |
| single | 5.17e-06 | 100 |

