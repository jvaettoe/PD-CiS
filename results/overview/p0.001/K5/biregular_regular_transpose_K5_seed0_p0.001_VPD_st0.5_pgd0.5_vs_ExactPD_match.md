### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 4000 steps_st0.5_pgd0.5

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.262 | 12/100 | 0.229 | 0.68 | 2.830 | 4 |
| w_out | 0.194 | 5/100 | 0.108 | 0.68 | 3.627 | 0 |

- leak alignment of matched write shards: 0.420
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.058
- fingerprint purity (signed cosine): 0.164
- ablation agreement: 0.850
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.176, neurons covered=48
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.257, neurons covered=48
- union act_out (active-neuron shards per feature): 0.678

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.202
- circuit fingerprint purity (signed cosine): 0.247
- circuit ablation agreement: 0.860  (spillover to code-neighbours: 0.801)

#### Mechanism census
- learned w_in shards: feature-aligned=12, mixed=88, zombie=0, dead=0
- learned w_out shards: neuron-aligned=0, split=82, mixed=14, zombie=4, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 12/100 features 1:1, mean weight cosine 0.516
- write incidence: circuit covers 0.96 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=2.192e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=8.599e-03, VPD (pair, basis cap)=9.482e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.44, n_alive_union=99.00, pr_mean_ci=71.13
- ci_stats[w_out]: ci_l0=0.54, n_alive_union=87.00, pr_mean_ci=52.44
- ci_stats[headline]: ci_l0=0.54, n_alive_union=99.00, pr_mean_ci=71.13
- loss importance: raw=1.968e-01 weighted=1.968e-05
- loss stoch_recon: raw=2.224e-05 weighted=4.448e-05
- loss pgd_recon: raw=4.515e-05 weighted=9.031e-05
- ||delta||/||W||: w_in=0.327, w_out=0.225
