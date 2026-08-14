# Memo: hetk — heterogeneous codeword lengths via two-population sparsity

## 1. Model architecture
Single-hidden-layer MLP, **100 → 50 (ReLU) → 100**, no biases (identical to the
main-suite MLP). Trained variants differ only in the loss: MSE (primary), L4
(comparison), and wMSE (importance-weighted MSE, per-feature weight 1/p_j
normalized to mean 1 — equalizes the populations' expected loss contribution).

## 2. Task and training
Main-suite CleanDataset task: inputs sparse with values U[−1,1], target
**y = ReLU(x)** (the model must compute 100 ReLUs). Two sparsity populations:
features 0–49 "rare" (p = 0.001), features 50–99 "common" (p = 0.05). AdamW
lr 3e-3, wd 1e-2, cosine, batch 2048, seed 42, steps = 10000/P(≥1 active) ≈ 10.8k.
Sweeps: p_rare ∈ {0.001…0.05} and population split n_rare ∈ {10, 25, 50}.
Census: active-conditional per-feature MSE (always-0 baseline 1/3, "computed" =
< 1/30), K(>0)/K(τ=0.05) codeword lengths, code anatomy, 2×2 interference.
(A first round accidentally used an easier copy task — y = x on U[0,1] values;
those artifacts are archived in `archive_copytask/` and superseded by this memo.)

## 3. Findings

**MSE does no superposition at all — it triages.** The MSE model assigns
**dedicated neurons**: common features get exactly K(τ) = 1.0 ± 0.0 neuron each
with unit gain (extracted code refit: a = 1.000, b = 0.000, ansatz/trained ratio
1.00 — the model *is* a one-neuron-per-feature identity code), reconstructed to
MSE ≈ 9e-10, while the rare population is fully abandoned (gain exactly 0.000,
K(τ) = 0, MSE = the ignore-baseline 1/3, interference at float-noise 1e-12).
The boundary sweep nails the rule: at every population split, MSE computes
exactly ~N = 50 features — 100% of common when n_common = 50, 67% when
n_common = 75, 56% when n_common = 90, and a 52/48 split in the homogeneous
p = 0.05 case. **Pick min(N, ·) features, one neuron each, drop the rest.**

**L4 replaces abandonment with attenuation — a graded, lossy superposition.**
(By the strict computed criterion the rare population never passes at any
p_rare < 0.05; the honest dichotomy is gain-0 abandonment (MSE) vs gain-0.55
attenuation (L4).) Under L4 both
populations get multi-neuron codes (common K(τ) = 6.6 — consistent with the
main-suite K(p = 0.05); rare K(τ) = 21.3 — consistent with the K(p) trend).
Common is fully computed (MSE 4.4e-3). The rare population is *partially*
computed: a clean attenuated linear map with gain c = 0.552 ± 0.006 (attenuation
explains ~86% of its 0.078 MSE), never crossing the strict computed threshold at
any p_rare < 0.05. The within-rare swap test at 0.47 shows partial positional
code semantics; rare-code neuron sharing is near the hypergeometric random level.
The boundary sweep shows a smooth K(τ) ladder (21.3 → 5.4 as p_rare → 0.05) —
no sharp abandonment transition under L4, just graded attenuation.

**wMSE does not equalize — it flips the triage.** With 1/p importance weights,
MSE now computes the rare population (43/50, MSE 5e-3, gain 0.83) and abandons
the common one (3/50 computed, gain 0.17, with large rare→common interference
6e-2). Equalizing expected loss contribution does not buy coexistence: the
MSE-family objective still selects ~one population and drops the other. The
capacity conflict is resolved by *which* population wins, not by superposing —
loss shape (L4), not importance weighting, is what buys superposition.

**Robustness and anatomy checks (adversarial review).** The triage result is
not an optimizer artifact: retraining with weight decay 0 and with seed 43
reproduces it exactly, and cross-evaluation shows the MSE-trained model beats
the L4-trained model *under the MSE metric* by 12× — triage is the genuine
MSE optimum, unreachable by longer training. Negative one-hot probes show the
L4 rare population genuinely computes ReLU (output at x = −1 is −0.03 ≈ 0,
not the −0.55 a linear passthrough would give): attenuated but truly computed.
Scope: all claims are for this training family (AdamW 3e-3, wd 1e-2, cosine,
kaiming init) and the ReLU task — the archived copy-task round came out
differently, so the conclusions are task-specific by demonstration.

**Interference ledger.** MSE buys its 1e-10-clean reconstruction precisely by
abandonment (leakage ~1e-12); L4 pays ~1e-2–1e-3 leakage across all population
pairs as the price of carrying 100 features in 50 neurons.

**Takeaway for the PD stage.** The planned "two code populations with different
K in one model" ground truth exists only under L4 — and its rare-population
mechanisms are attenuated, dense-ish codes rather than clean biregular ones.
The MSE model is an even simpler PD target (50 disjoint one-neuron mechanisms +
50 empty ones); the wMSE model is its mirror image. The loss function does not
merely change code statistics — it selects between qualitatively different
computational strategies (triage vs superposition).
