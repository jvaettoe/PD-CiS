### Component recovery vs ExactPD — biregular K=5 transpose no-swap, p=0.001, VPD 25000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.796 | 85/100 | 0.787 | 0.89 | 0.558 | 16 |
| w_out | 0.106 | 0/100 | 0.067 | 0.19 | 3.401 | 0 |

- leak alignment of matched write shards: 0.213
- index agreement (match_in == match_out): 0.020
- solo/exact self-activation ratio (pair): 0.085
- fingerprint purity (signed cosine): 0.208
- ablation agreement: 0.930
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.268, neurons covered=47
- neuron-basis[w_out]: alive=91, shard neuron-concentration=0.611, neurons covered=50
- union act_out (active-neuron shards per feature): 0.967

Reading: act high + mass high = ExactPD basis; act high + mass low + leak-align high = same mechanism in the leak-aligned basis; act low + neuron-concentration high + union act_out high = PER-NEURON basis (solo act is capped at ~1/K by construction); act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.19, n_alive_union=96.00, pr_mean_ci=37.68
- ci_stats[w_out]: ci_l0=0.57, n_alive_union=54.00, pr_mean_ci=30.31
- ci_stats[headline]: ci_l0=0.57, n_alive_union=96.00, pr_mean_ci=37.68
- loss importance: raw=3.274e-01 weighted=3.274e-05
- loss stoch_recon: raw=6.175e-06 weighted=1.235e-05
- loss pgd_recon: raw=1.416e-05 weighted=2.832e-05
- ||delta||/||W||: w_in=0.138, w_out=0.028
