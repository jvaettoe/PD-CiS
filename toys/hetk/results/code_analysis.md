### hetK code anatomy + interference (default config: rare p=0.001 x50, common p=0.05 x50)

#### loss = mse

- trained mse loss (held-out): 8.356e-05
- ansatz refit on M-hat: transpose (a=1.000, b=0.000) ratio 1.00; pinv (a=1.000, b=0.000) ratio 1.00
- gains c: rare 0.000±0.000, common 1.000±0.000
- rare MSE median 3.345e-01 vs attenuation-only prediction 3.333e-01 (fraction explained 1.00)
- neuron sharing: per-neuron degree rare 0.0±0.0, common 1.0±0.0; overlap within-rare 0.00, rare-common 0.00 (hypergeometric ~0.00)
- within-rare swap test: nan
- interference (mean squared one-hot leakage): rare->rare 3.62e-12, rare->common 5.33e-08, common->rare 5.45e-08, common->common 3.12e-12

#### loss = wmse

- trained wmse loss (held-out): 1.627e-04
- ansatz refit on M-hat: transpose (a=0.058, b=-0.000) ratio 1.94; pinv (a=0.237, b=-0.000) ratio 1.79
- gains c: rare 0.831±0.171, common 0.172±0.231
- rare MSE median 4.953e-03 vs attenuation-only prediction 4.051e-03 (fraction explained 0.82)
- neuron sharing: per-neuron degree rare 20.6±7.4, common 0.9±1.0; overlap within-rare 9.36, rare-common 0.29 (hypergeometric ~2.32)
- within-rare swap test: 0.08
- interference (mean squared one-hot leakage): rare->rare 1.04e-03, rare->common 6.06e-02, common->rare 2.54e-05, common->common 5.91e-04

#### loss = l4

- trained l4 loss (held-out): 1.128e-05
- ansatz refit on M-hat: transpose (a=0.203, b=-0.025) ratio 97.49; pinv (a=1.727, b=-0.079) ratio 13.52
- gains c: rare 0.552±0.006, common 0.948±0.014
- rare MSE median 7.787e-02 vs attenuation-only prediction 6.690e-02 (fraction explained 0.86)
- neuron sharing: per-neuron degree rare 21.3±6.1, common 6.6±1.7; overlap within-rare 9.58, rare-common 2.74 (hypergeometric ~3.90)
- within-rare swap test: 0.47
- interference (mean squared one-hot leakage): rare->rare 1.52e-02, rare->common 2.06e-02, common->rare 1.91e-03, common->common 1.35e-03

