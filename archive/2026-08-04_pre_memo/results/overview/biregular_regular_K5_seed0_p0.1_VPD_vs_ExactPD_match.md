### Component recovery vs ExactPD — biregular K=5 pinv no-swap, p=0.1, VPD 4000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.137 | 0/100 | 0.071 | 0.21 | 3.134 | 0 |
| w_out | 0.140 | 0/100 | 0.068 | 0.39 | 3.738 | 0 |

- leak alignment of matched write shards: 0.248
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.126
- fingerprint purity (signed cosine): 0.065
- ablation agreement: 0.360
- dead components: 0

Reading: act high + mass high = ExactPD basis; act high + mass low + leak-align high = same mechanism in the leak-aligned basis; act low = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=19.13, n_alive_union=50.00, pr_mean_ci=47.61
- ci_stats[w_out]: ci_l0=16.38, n_alive_union=50.00, pr_mean_ci=49.94
- ci_stats[headline]: ci_l0=19.13, n_alive_union=50.00, pr_mean_ci=49.94
- loss importance: raw=2.666e+01 weighted=2.666e-03
- loss stoch_recon: raw=5.265e-05 weighted=1.053e-04
- loss pgd_recon: raw=1.778e-04 weighted=3.556e-04
- ||delta||/||W||: w_in=0.018, w_out=0.016
