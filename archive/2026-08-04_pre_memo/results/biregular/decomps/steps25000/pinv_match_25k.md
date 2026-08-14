### Component recovery vs ExactPD — biregular K=5 pinv no-swap, p=0.001, VPD 25000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.769 | 80/100 | 0.788 | 1.00 | 0.561 | 20 |
| w_out | 0.140 | 0/100 | 0.058 | 0.31 | 3.716 | 0 |

- leak alignment of matched write shards: 0.220
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.103
- fingerprint purity (signed cosine): 0.185
- ablation agreement: 0.930
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.237, neurons covered=45
- neuron-basis[w_out]: alive=66, shard neuron-concentration=0.798, neurons covered=50
- union act_out (active-neuron shards per feature): 0.997

Reading: act high + mass high = ExactPD basis; act high + mass low + leak-align high = same mechanism in the leak-aligned basis; act low + neuron-concentration high + union act_out high = PER-NEURON basis (solo act is capped at ~1/K by construction); act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.16, n_alive_union=99.00, pr_mean_ci=75.92
- ci_stats[w_out]: ci_l0=0.25, n_alive_union=51.00, pr_mean_ci=47.58
- ci_stats[headline]: ci_l0=0.25, n_alive_union=99.00, pr_mean_ci=75.92
- loss importance: raw=2.475e-01 weighted=2.475e-05
- loss stoch_recon: raw=2.094e-06 weighted=4.187e-06
- loss pgd_recon: raw=6.347e-06 weighted=1.269e-05
- ||delta||/||W||: w_in=0.089, w_out=0.024
