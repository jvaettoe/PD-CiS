### Component recovery vs ExactPD — biregular K=5 transpose no-swap, p=0.001, VPD 100000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.987 | 97/100 | 0.942 | 0.98 | 0.187 | 1 |
| w_out | 0.102 | 0/100 | 0.063 | 0.22 | 3.445 | 0 |

- leak alignment of matched write shards: 0.197
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.098
- fingerprint purity (signed cosine): 0.227
- ablation agreement: 0.960
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.204, neurons covered=43
- neuron-basis[w_out]: alive=53, shard neuron-concentration=0.909, neurons covered=50
- union act_out (active-neuron shards per feature): 0.957

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 5.30
- circuit solo/exact self-activation ratio: 0.949
- circuit fingerprint purity (signed cosine): 0.550
- circuit ablation agreement: 1.000  (spillover to code-neighbours: 0.720)

#### Mechanism census
- learned w_in shards: feature-aligned=97, mixed=3, zombie=0, dead=0
- learned w_out shards: neuron-aligned=46, split=4, mixed=1, zombie=2, dead=47
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 97/100 features 1:1, mean weight cosine 0.919
- write incidence: circuit covers 1.00 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=2.672e-04, ExactPD (all comps)=2.672e-04, VPD (all comps, no delta)=2.580e-04
- forward L2 [onehot]: full model=1.530e-02, ExactPD (single comp)=2.064e-04, VPD (circuit)=1.456e-02, VPD (pair, basis cap)=9.989e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.11, n_alive_union=92.00, pr_mean_ci=15.91
- ci_stats[w_out]: ci_l0=0.57, n_alive_union=53.00, pr_mean_ci=32.77
- ci_stats[headline]: ci_l0=0.57, n_alive_union=92.00, pr_mean_ci=32.77
- loss importance: raw=2.877e-01 weighted=2.877e-05
- loss stoch_recon: raw=4.819e-06 weighted=9.637e-06
- loss pgd_recon: raw=1.248e-05 weighted=2.495e-05
- ||delta||/||W||: w_in=0.123, w_out=0.029
