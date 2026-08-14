### Component recovery vs ExactPD — ExactPD vs itself (MLP d50 p=0.1)

| side | act purity | act cov(>0.5) | mass purity (act match) | act-vs-mass match agree | mass row entropy | split |
|---|---|---|---|---|---|---|
| w_in | 1.000 | 100/100 | 1.000 | 1.00 | 0.000 | 0 |
| w_out | 1.000 | 100/100 | 1.000 | 1.00 | 0.000 | 0 |

- leak alignment of matched write shards: 0.000
- index agreement (match_in == match_out): 1.000
- solo/exact self-activation ratio (pair): 1.000
- fingerprint purity (signed cosine): 1.000
- ablation agreement: 1.000
- dead components: 0
- neuron-basis[w_in]: alive=100, shard neuron-concentration=0.394, neurons covered=41
- neuron-basis[w_out]: alive=100, shard neuron-concentration=0.296, neurons covered=43
- union act_out (active-neuron shards per feature): 1.000

#### Circuit level (mechanism comparison: read shard + union of active-neuron write shards)
- circuit size (write shards per feature): 1.00
- circuit solo/exact self-activation ratio: 1.000
- circuit fingerprint purity (signed cosine): 1.000
- circuit ablation agreement: 1.000  (spillover to code-neighbours: 0.088)

- forward L2 [dist]: full model=1.115e-02, ExactPD (all comps)=1.115e-02, ExactPD (all comps, no delta)=1.115e-02
- forward L2 [onehot]: full model=5.965e-03, ExactPD (single comp)=8.271e-04, ExactPD (circuit)=8.271e-04, ExactPD (pair, basis cap)=8.271e-04

Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron write basis caps solo act at ~1/K by construction); CIRCUIT probes measure mechanism recovery and are the apples-to-apples comparison against ExactPD's chain-glued components. act high + mass high = ExactPD basis; act low + neuron-concentration high + union/circuit act high = PER-NEURON write basis; act low otherwise = genuinely distributed.
