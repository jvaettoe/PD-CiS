### ramp analysis — hidden-layer biases; L4, p = 0.01, seed 42, theta = 0.5

## Transfer curves (one-hot probes, v in [-1, 1])

| arch | rise slope | plateau level | plateau slope | plateau span | kink found | kink position | homog. dev max | neg-side max |
|---|---|---|---|---|---|---|---|---|
| chain | 0.733 [0.726, 0.743] | 0.429 [0.425, 0.433] | 0.206 [0.187, 0.223] | 0.083 [0.075, 0.089] | 99/100 | 0.505 [0.495, 0.525] | 1.45e-01 | 0.013 [0.011, 0.015] |
| single | 0.604 [0.601, 0.607] | 0.477 [0.474, 0.480] | 0.567 [0.562, 0.571] | 0.227 [0.225, 0.229] | 0/100 | — | 1.30e-02 | 0.061 [0.058, 0.064] |

Ideal ramp: rise slope 1, plateau level 0.5, plateau slope 0, kink at v = 0.5. Bias-free nets are positively homogeneous (f(v e_j) = v f(e_j), v > 0): homog. dev max at float-noise level means plateaus are impossible on singletons; hidden biases make the theta-kink representable.

## Layer-2 bypass ablation (chain)

| target | clip frac | L4 original | identity ratio | fitted (affine) ratio |
|---|---|---|---|---|
| ramp (hidden-layer biases) | 0.035 | 3.88e-06 | 4.43 | 1.56 |
| ramp bias-free control (measured) | 0.053 | 5.27e-06 | 1.13 | 1.06 |
| relu (reference, toys/chain) | 0.103 | 1.33e-05 | 1.13 | 1.06 |

On-record prediction: the ramp makes layer 2 load-bearing (ratios well above the relu-target 1.13x / 1.06x).

## Unit-role census (one-hot probes at v = 0.25 vs 0.75)

Growth ratio h(0.75)/h(0.25) per active unit; linear scaling gives exactly 3. Saturation-implementer: ratio < 1.5; rise-implementer: ratio >= 2.5.

| arch | layer | active units/feature | saturators/feature | risers/feature |
|---|---|---|---|---|
| chain | L1 | 15.8 | 0.93 | 7.21 |
| chain | L2 | 49.9 | 49.46 | 0.00 |
| single | L1 | 32.1 | 22.39 | 5.28 |

## Single vs chain

| arch | held-out L4 | computed /100 |
|---|---|---|
| chain | 3.88e-06 | 100 |
| single | 4.97e-06 | 100 |

