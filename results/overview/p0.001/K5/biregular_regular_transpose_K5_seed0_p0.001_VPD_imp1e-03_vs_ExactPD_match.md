### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 25000 steps_imp1e-03

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.877 | 84/100 | 0.746 | 0.88 | 0.888 | 4 |
| w_out | 0.294 | 34/100 | 0.135 | 0.50 | 3.542 | 0 |

- leak alignment of matched write shards: 0.415
- index agreement (match_in == match_out): 0.010
- solo/exact self-activation ratio (pair): 0.291
- fingerprint purity (signed cosine): 0.274
- ablation agreement: 0.890
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.201, neurons covered=42
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.287, neurons covered=49
- union act_out (active-neuron shards per feature): 0.846

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.774
- circuit fingerprint purity (signed cosine): 0.418
- circuit ablation agreement: 0.910  (spillover to code-neighbours: 0.760)

#### Mechanism census
- learned w_in shards: feature-aligned=84, mixed=5, zombie=11, dead=0
- learned w_out shards: neuron-aligned=2, split=41, mixed=12, zombie=45, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 84/100 features 1:1, mean weight cosine 0.807
- write incidence: circuit covers 0.98 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=1.923e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=1.208e-02, VPD (pair, basis cap)=9.478e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.06, n_alive_union=78.00, pr_mean_ci=56.97
- ci_stats[w_out]: ci_l0=0.19, n_alive_union=55.00, pr_mean_ci=42.98
- ci_stats[headline]: ci_l0=0.19, n_alive_union=78.00, pr_mean_ci=56.97
- loss importance: raw=8.060e-02 weighted=8.060e-06
- loss stoch_recon: raw=1.482e-05 weighted=2.965e-05
- loss pgd_recon: raw=3.286e-05 weighted=6.572e-05
- ||delta||/||W||: w_in=0.236, w_out=0.123
