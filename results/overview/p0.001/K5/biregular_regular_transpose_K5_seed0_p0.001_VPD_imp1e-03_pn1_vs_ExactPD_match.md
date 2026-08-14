### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 4000 steps_imp1e-03_pn1

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.063 | 3/100 | 0.088 | 0.50 | 3.730 | 1 |
| w_out | 0.052 | 0/100 | 0.069 | 0.60 | 3.872 | 0 |

- leak alignment of matched write shards: 0.346
- index agreement (match_in == match_out): 0.010
- solo/exact self-activation ratio (pair): 0.003
- fingerprint purity (signed cosine): 0.068
- ablation agreement: 0.820
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.135, neurons covered=44
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.139, neurons covered=44
- union act_out (active-neuron shards per feature): 0.154

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.012
- circuit fingerprint purity (signed cosine): 0.075
- circuit ablation agreement: 0.400  (spillover to code-neighbours: 0.970)

#### Mechanism census
- learned w_in shards: feature-aligned=3, mixed=3, zombie=94, dead=0
- learned w_out shards: neuron-aligned=0, split=1, mixed=0, zombie=99, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 3/100 features 1:1, mean weight cosine 0.618
- write incidence: circuit covers 0.88 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=1.354e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=9.873e-03, VPD (pair, basis cap)=9.960e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.00, n_alive_union=5.00, pr_mean_ci=2.69
- ci_stats[w_out]: ci_l0=0.00, n_alive_union=1.00, pr_mean_ci=3.14
- ci_stats[headline]: ci_l0=0.00, n_alive_union=5.00, pr_mean_ci=3.14
- loss importance: raw=6.648e-04 weighted=6.648e-08
- loss stoch_recon: raw=8.325e-05 weighted=1.665e-04
- loss pgd_recon: raw=2.094e-04 weighted=4.187e-04
- ||delta||/||W||: w_in=0.373, w_out=0.341
