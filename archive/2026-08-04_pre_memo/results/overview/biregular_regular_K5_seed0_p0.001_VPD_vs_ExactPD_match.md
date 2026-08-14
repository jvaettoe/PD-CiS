### Component recovery vs ExactPD — biregular K=5 pinv no-swap, p=0.001, VPD 100000 steps

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.993 | 99/100 | 0.989 | 1.00 | 0.028 | 2 |
| w_out | 0.141 | 0/100 | 0.056 | 0.29 | 3.701 | 0 |

- leak alignment of matched write shards: 0.212
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.141
- fingerprint purity (signed cosine): 0.202
- ablation agreement: 0.990
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.203, neurons covered=44
- neuron-basis[w_out]: alive=50, shard neuron-concentration=0.999, neurons covered=50
- union act_out (active-neuron shards per feature): 0.997

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 5.00
- circuit solo/exact self-activation ratio: 0.991
- circuit fingerprint purity (signed cosine): 0.700
- circuit ablation agreement: 0.990  (spillover to code-neighbours: 0.703)

#### Mechanism census
- learned w_in shards: feature-aligned=99, mixed=1, zombie=0, dead=0
- learned w_out shards: neuron-aligned=50, split=0, mixed=0, zombie=0, dead=50
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 99/100 features 1:1, mean weight cosine 0.992
- write incidence: circuit covers 1.00 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.067e-04, ExactPD (all comps)=1.067e-04, VPD (all comps, no delta)=1.068e-04
- forward L2 [onehot]: full model=6.429e-03, ExactPD (single comp)=5.918e-04, VPD (circuit)=6.478e-03, VPD (pair, basis cap)=1.029e-02

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.06, n_alive_union=90.00, pr_mean_ci=69.71
- ci_stats[w_out]: ci_l0=0.25, n_alive_union=50.00, pr_mean_ci=47.93
- ci_stats[headline]: ci_l0=0.25, n_alive_union=90.00, pr_mean_ci=69.71
- loss importance: raw=2.172e-01 weighted=2.172e-05
- loss stoch_recon: raw=1.456e-06 weighted=2.912e-06
- loss pgd_recon: raw=3.115e-06 weighted=6.230e-06
- ||delta||/||W||: w_in=0.032, w_out=0.013
