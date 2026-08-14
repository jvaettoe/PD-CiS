### Component recovery vs ExactPD — biregular K=5 transpose no-swap, p=0.1, VPD 4000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.111 | 0/100 | 0.076 | 0.37 | 3.462 | 0 |
| w_out | 0.120 | 1/100 | 0.065 | 0.21 | 3.378 | 0 |

- leak alignment of matched write shards: 0.270
- index agreement (match_in == match_out): 0.010
- solo/exact self-activation ratio (pair): 0.012
- fingerprint purity (signed cosine): 0.054
- ablation agreement: 0.580
- dead components: 0

Reading: act high + mass high = ExactPD basis; act high + mass low + leak-align high = same mechanism in the leak-aligned basis; act low = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=36.82, n_alive_union=48.00, pr_mean_ci=43.20
- ci_stats[w_out]: ci_l0=30.00, n_alive_union=49.00, pr_mean_ci=45.52
- ci_stats[headline]: ci_l0=36.82, n_alive_union=49.00, pr_mean_ci=45.52
- loss importance: raw=3.138e+01 weighted=3.138e-03
- loss stoch_recon: raw=1.551e-04 weighted=3.101e-04
- loss pgd_recon: raw=4.456e-04 weighted=8.912e-04
- ||delta||/||W||: w_in=0.038, w_out=0.031
