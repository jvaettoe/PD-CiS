### Component recovery vs ExactPD — identity model (w_in=w_out=I), p=0.001, VPD 25000 steps

| side | purity | coverage(>0.5) | split features | mean row entropy |
|---|---|---|---|---|
| mass_in | 0.926 | 92/100 | 2 | 0.202 |
| mass_out | 0.747 | 76/100 | 33 | 0.506 |

- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (mean): 0.736
- fingerprint purity (signed cosine): 0.812
- ablation agreement: 0.620
- dead components: 0

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.07, n_alive_union=85.00, pr_mean_ci=36.74
- ci_stats[w_out]: ci_l0=0.08, n_alive_union=94.00, pr_mean_ci=73.22
- ci_stats[headline]: ci_l0=0.08, n_alive_union=94.00, pr_mean_ci=73.22
- loss importance: raw=1.249e-01 weighted=1.249e-05
- loss stoch_recon: raw=1.251e-06 weighted=2.503e-06
- loss pgd_recon: raw=3.200e-06 weighted=6.400e-06
- ||delta||/||W||: w_in=0.163, w_out=0.091
