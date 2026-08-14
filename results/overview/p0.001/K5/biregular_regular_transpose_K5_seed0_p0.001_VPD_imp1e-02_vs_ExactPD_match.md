### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 4000 steps_imp1e-02

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.076 | 1/100 | 0.104 | 0.71 | 3.652 | 0 |
| w_out | 0.105 | 3/100 | 0.089 | 0.64 | 3.800 | 0 |

- leak alignment of matched write shards: 0.428
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.013
- fingerprint purity (signed cosine): 0.103
- ablation agreement: 0.760
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.156, neurons covered=44
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.164, neurons covered=46
- union act_out (active-neuron shards per feature): 0.303

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.032
- circuit fingerprint purity (signed cosine): 0.110
- circuit ablation agreement: 0.430  (spillover to code-neighbours: 0.951)

#### Mechanism census
- learned w_in shards: feature-aligned=1, mixed=25, zombie=74, dead=0
- learned w_out shards: neuron-aligned=0, split=22, mixed=5, zombie=73, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 1/100 features 1:1, mean weight cosine 0.555
- write incidence: circuit covers 0.92 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=1.848e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=9.730e-03, VPD (pair, basis cap)=9.908e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.00, n_alive_union=9.00, pr_mean_ci=45.30
- ci_stats[w_out]: ci_l0=0.02, n_alive_union=17.00, pr_mean_ci=30.93
- ci_stats[headline]: ci_l0=0.02, n_alive_union=17.00, pr_mean_ci=45.30
- loss importance: raw=1.342e-03 weighted=1.342e-07
- loss stoch_recon: raw=8.199e-05 weighted=1.640e-04
- loss pgd_recon: raw=1.980e-04 weighted=3.961e-04
- ||delta||/||W||: w_in=0.346, w_out=0.343
