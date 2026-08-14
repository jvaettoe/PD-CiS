### Component recovery vs ExactPD — identity model (w_in=w_out=I), p=0.001, VPD 4000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.461 | 36/100 | 0.448 | 0.92 | 1.631 | 25 |
| w_out | 0.366 | 22/100 | 0.247 | 0.86 | 2.955 | 3 |

- leak alignment of matched write shards: 0.000
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.166
- fingerprint purity (signed cosine): 0.354
- ablation agreement: 0.560
- dead components: 0

Reading: act high + mass high = ExactPD basis; act high + mass low + leak-align high = same mechanism in the leak-aligned basis; act low = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.37, n_alive_union=98.00, pr_mean_ci=45.26
- ci_stats[w_out]: ci_l0=0.37, n_alive_union=100.00, pr_mean_ci=89.28
- ci_stats[headline]: ci_l0=0.37, n_alive_union=100.00, pr_mean_ci=89.28
- loss importance: raw=4.026e-01 weighted=4.026e-05
- loss stoch_recon: raw=2.869e-06 weighted=5.739e-06
- loss pgd_recon: raw=1.203e-05 weighted=2.406e-05
- ||delta||/||W||: w_in=0.281, w_out=0.091
