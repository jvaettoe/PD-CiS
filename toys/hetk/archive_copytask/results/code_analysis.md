### hetK code anatomy + interference (default config: rare p=0.001 x50, common p=0.05 x50)

#### loss = mse

- trained mse loss (held-out): 1.663e-04
- ansatz refit on M-hat: transpose (a=0.045, b=-0.008) ratio 46.16; pinv (a=1.000, b=0.000) ratio 1.00
- gains c: rare 0.000±0.000, common 1.000±0.000
- rare MSE median 3.319e-01 vs attenuation-only prediction 3.333e-01 (fraction explained 1.00)
- neuron sharing: per-neuron degree rare 0.0±0.0, common 28.0±2.6; overlap within-rare 0.00, rare-common 0.00 (hypergeometric ~3.92)
- within-rare swap test: nan
- interference (mean squared one-hot leakage): rare->rare 1.25e-11, rare->common 8.10e-08, common->rare 8.31e-08, common->common 1.87e-11

#### loss = l4

- trained l4 loss (held-out): 1.607e-05
- ansatz refit on M-hat: transpose (a=0.118, b=-0.158) ratio 85.41; pinv (a=1.356, b=0.001) ratio 6.97
- gains c: rare 0.532±0.004, common 0.908±0.002
- rare MSE median 7.063e-02 vs attenuation-only prediction 7.293e-02 (fraction explained 1.03)
- neuron sharing: per-neuron degree rare 36.3±2.4, common 30.3±2.8; overlap within-rare 26.26, rare-common 22.00 (hypergeometric ~22.18)
- within-rare swap test: 0.10
- interference (mean squared one-hot leakage): rare->rare 1.50e-02, rare->common 2.12e-02, common->rare 1.47e-03, common->common 1.18e-03

