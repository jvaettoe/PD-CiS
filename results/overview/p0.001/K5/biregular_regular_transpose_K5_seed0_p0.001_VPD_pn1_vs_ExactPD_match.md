### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 4000 steps_pn1

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.251 | 7/100 | 0.256 | 0.74 | 2.568 | 8 |
| w_out | 0.196 | 0/100 | 0.120 | 0.73 | 3.578 | 0 |

- leak alignment of matched write shards: 0.449
- index agreement (match_in == match_out): 0.010
- solo/exact self-activation ratio (pair): 0.056
- fingerprint purity (signed cosine): 0.185
- ablation agreement: 0.860
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.183, neurons covered=44
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.320, neurons covered=45
- union act_out (active-neuron shards per feature): 0.703

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.203
- circuit fingerprint purity (signed cosine): 0.267
- circuit ablation agreement: 0.940  (spillover to code-neighbours: 0.773)

#### Mechanism census
- learned w_in shards: feature-aligned=7, mixed=92, zombie=1, dead=0
- learned w_out shards: neuron-aligned=0, split=74, mixed=15, zombie=11, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 7/100 features 1:1, mean weight cosine 0.506
- write incidence: circuit covers 0.90 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=2.038e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=8.432e-03, VPD (pair, basis cap)=9.459e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.42, n_alive_union=98.00, pr_mean_ci=69.57
- ci_stats[w_out]: ci_l0=0.50, n_alive_union=87.00, pr_mean_ci=63.67
- ci_stats[headline]: ci_l0=0.50, n_alive_union=98.00, pr_mean_ci=69.57
- loss importance: raw=3.935e-01 weighted=3.935e-05
- loss stoch_recon: raw=1.416e-05 weighted=2.831e-05
- loss pgd_recon: raw=2.666e-05 weighted=5.332e-05
- ||delta||/||W||: w_in=0.281, w_out=0.164
