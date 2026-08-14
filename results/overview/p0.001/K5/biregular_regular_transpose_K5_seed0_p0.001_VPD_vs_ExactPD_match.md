### Component recovery vs ExactPD — biregular K=5 transpose no-swap (c-free), p=0.001, VPD 25000 steps (default knobs)

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 0.890 | 91/100 | 0.870 | 0.96 | 0.406 | 6 |
| w_out | 0.101 | 0/100 | 0.049 | 0.22 | 3.497 | 0 |

- leak alignment of matched write shards: 0.207
- index agreement (match_in == match_out): 0.000
- solo/exact self-activation ratio (pair): 0.078
- fingerprint purity (signed cosine): 0.168
- ablation agreement: 0.970
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.215, neurons covered=45
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.568, neurons covered=50
- union act_out (active-neuron shards per feature): 0.983

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 10.00
- circuit solo/exact self-activation ratio: 0.882
- circuit fingerprint purity (signed cosine): 0.527
- circuit ablation agreement: 1.000  (spillover to code-neighbours: 0.695)

#### Mechanism census
- learned w_in shards: feature-aligned=91, mixed=9, zombie=0, dead=0
- learned w_out shards: neuron-aligned=25, split=27, mixed=0, zombie=48, dead=0
- ExactPD (reference): 100 chain-glued per-feature components (sparse read row + dense write column each)
- read correspondence: 91/100 features 1:1, mean weight cosine 0.897
- write incidence: circuit covers 1.00 of active neurons (mean active = 5.0); features served per write shard = 10.0

- forward L2 [dist]: full model=1.750e-04, ExactPD (all comps)=1.750e-04, VPD (all comps, no delta)=1.756e-04
- forward L2 [onehot]: full model=1.013e-02, ExactPD (single comp)=1.125e-03, VPD (circuit)=9.558e-03, VPD (pair, basis cap)=9.687e-03

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.

#### Training-state diagnostics
- ci_stats[w_in]: ci_l0=0.12, n_alive_union=95.00, pr_mean_ci=66.55
- ci_stats[w_out]: ci_l0=0.34, n_alive_union=51.00, pr_mean_ci=48.49
- ci_stats[headline]: ci_l0=0.34, n_alive_union=95.00, pr_mean_ci=66.55
- loss importance: raw=2.443e-01 weighted=2.443e-05
- loss stoch_recon: raw=4.783e-06 weighted=9.566e-06
- loss pgd_recon: raw=8.774e-06 weighted=1.755e-05
- ||delta||/||W||: w_in=0.136, w_out=0.027
