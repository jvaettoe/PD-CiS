# Memo: embed — the embedding sandwich (EmbedMLP) and its parameter decompositions

## 1. Model architecture
EmbedMLP: **y = ReLU((x·W_e)·W_in)·W_out·W_eᵀ** with F = 100 features, N = 50
hidden ReLUs, W_e fixed random unit-norm rows (F × d_emb, never trained),
trainable W_in/W_out; d_emb swept 20…1000. Two object families: TRAINED
checkpoints (`train_embed_demb.py`, `embed_transition_stats.py`) and the
DESIGNED sandwich W_in = W_e⁺·C, W_out = W_inᵀ with C = a·M + b·(1−M) the K = 5
biregular transpose code (fit_ab(p = 0.001)) — effective read-in W_e·W_in = G·C
with G = W_e·W_e⁺: exactly C at d_emb = 200 (G = I), G-softened at d_emb = 50
(mean G_jj = d_emb/F = 0.500).

## 2. Task and training
Main-suite task: sparse inputs (values U(−1,1)), target y = ReLU(x); L4 loss,
AdamW 3e-3, wd 1e-2, cosine (p = 0.001 and p = 0.01 regimes; multi-seed n = 9,
seeds 42–50, at p = 0.01). VPD (param-decomp, branch biregular-port) decomposes
ALL FOUR matrices of the designed sandwich (100 subcomponents per matrix,
ImportanceMinimality 1e-4/pnorm 2, stochastic recon losses, delta component
on): 10k steps then +10k warm-restart consolidation, at d_emb ∈ {200, 50}.

## 3. Findings

**The exact per-matrix decomposition is a dual-basis object — and SVD cannot
be feature-aligned.** Demanding (pieces sum exactly, rank-1, each W_in piece
carries signal only when its feature is active) FORCES the input-side vectors
to be the dual basis w̃_j of W_e's rows (exists iff d_emb ≥ F): W_emb pieces
e_j ⊗ w_e_j; W_in pieces w̃_j ⊗ (w_e_j·W_in); plus one INERT remainder per
inner matrix ((I−P)·W_in, P = W_e⁺W_e) that no input ever touches — frozen init
noise, 68%/17% of W_in/W_out norm at d_emb = 1000. W_in's top singular
direction reads ~95 features at once: SVD pieces are faithful and rank-1 but
never feature-aligned.

**d_emb < F degrades gracefully (G-softening); hidden-level matching is
impossible, output-level matching has a closed form.** Below F the pieces sum
exactly with ZERO remainder; only alignment softens, following G (leak G_ij,
solo chain on-target = G_jj²·R_jj; trained R² census: d20 0.811, d50 0.947,
d200 0.964). The trace/rigidity impossibility applies only to HIDDEN-pattern
matching; with output-only self-activation matching an exact 100-chain PD
exists in CLOSED FORM at every d_emb (contacts u_j = v_j = w_e_j; all
conditions ~1e-15) — priced by hidden-pattern divergence (cos(a_j, r_j) 0.89
at d20 / 0.69 at d50) and off-target chain leak ~1.

**Adopted PD = SVD weight-mass components: faithful, rank-1, zero corrections
— but collective, and truncation cannot denoise.** Each feature is captured by
~rank-many components (17/41 per feature), loadings dense (participation 29–45
features), all co-activate; mass-ordered truncation drops signal and leak
together (no knee at any d_emb; at n95, 82–97% of leak retained while n_soft
falls 95→54 at d50). Leak-free solo pieces require feature-anchored partitions.

**dFH-equivalence dissolves smoothly with d_emb (n = 9 seeds).** cos(W_out_eff,
scaled pinv) 0.996 ± 0.000 (d1000) → 0.686 ± 0.009 (d20), still ≥ 0.99 down to
d60; trained/designed L4 parity EXACTLY at d40 (ratio 0.99 ± 0.03); row-degree
σ peaks at d_emb = N = 50; swap semantics 0.93–1.00 throughout. Performance
transitions at d_emb = N (rank cap); mechanism-decomposability at d_emb = F
(gate trace budget) — N ≤ d_emb < F is performance-equivalent but
mechanistically half-budgeted.

**Designed sandwich + ideal PD (new chapter).** Construction asserts pass
(d200 forward ≡ plain designed code, 8e-7; d50 effective ≡ G·C). Ideal PD:
100 row pieces for W_emb (tied transpose for the unembedding) + all 50 SVD
triplets for W_in (tied for W_out). SURPRISE: the uniform (DC) direction is an
EXACT singular pair of C (σ = 0.4493, closed form to 1e-16) but sits at rank
49/50 — the negative b cancels DC mass rather than dominating. The loading
matrix L_jk = ⟨w_e_j, u_k⟩ is dense (max |L| = 0.34): the SVD frame and the
embedded-feature frame are maximally distinguishable, making the VPD test
sharp.

**VPD does not find the SVD triplets — it half-finds features and neurons
(consolidated verdicts, 10k + 10k warm restart).** lin_in is definitively NOT
svd-aligned at either d_emb (Hungarian median |cos| vs triplets 0.09–0.14,
flat under consolidation). Its strongest frame is the embedded-feature frame
{W_e[j]}: d200 best-|cos| 0.579 → 0.633 with coverage 86 → 90/100
(strengthening under consolidation but below the 0.8 threshold ⇒ formal
verdict `mixed`); d50 diluted (0.497, coverage 58/100 — 100 feature directions
crowding a 50-dim space). lin_emb consolidates toward the ideal row pieces at
d200 (concentration 0.991, Hungarian 0.842, 55% > 0.8) but only partially at
d50 (0.728/0.445). lin_out shows partial neuron structure (write-side
distinct-neuron coverage 47/50 resp. 37/50, concentration ~0.5). lin_unemb is
unstructured at both (Hungarian 0.06–0.08). OPEN FINDING: Σ-subcomponent
residuals stay high after consolidation (0.21–0.85; lin_unemb carried almost
entirely by the runtime delta) — in the 4-matrix sandwich the delta remains a
major weight carrier, unlike the 2-matrix designed runs. Artifacts:
`results|figures/embedded/ideal_vs_vpd/` (per-stage comparison.md/CSV/JSON,
align figures, `summary.md`, `summary_ontology.png`).

## Addendum — mechanism forward-pass census (d200): which W_in ontology is causally concentrated?

Strictly layer-local chain mechanisms on one-hot probes (W_emb component
e_j ⊗ W_e[j] emits a_j = v·W_e[j]; W_in pieces activate with coefficients read
off the chain; tied transposes for W_out and the unembedding). Option A —
projecting hidden activations onto W_emb-derived directions — is REJECTED on
principle: the decomposition must be fixed before inference, and A smuggles
W_emb knowledge into the W_in ontology at inference time. Three exact tilings
compared instead (all full-sum forwards ≡ the full model to <1e-15; identity,
not finding): per-emb-dim ROWS e_d ⊗ W_in[d,:], per-neuron COLUMNS
W_in[:,n] ⊗ e_n, and the SVD triplets σ_k u_k v_kᵀ.

**Verdict: the per-neuron column ontology is the causally concentrated one —
exactly K = 5 components per feature suffice, machine-exactly.** Truncated
tied-transpose forwards (top-m by |coefficient|): columns reach spike ≤1% and
leak ≤10% at m* = 5 for every feature (errors ~1e-16 from m = 5 on); SVD needs
40/43 of 50 (median spike/leak; no knee, consistent with the earlier
truncation-cannot-denoise result); rows need 176/198 of 200. The census
(cutoff-free: PR + 50/90/99% energy counts) and the causal count DISAGREE for
columns — pre-activation energy is spread over the b-floor (PR 6.6, n99 = 47
of 50) but the ReLU kills every off-code (negative) pre-activation, so the 45
b-floor pieces are causally inert and only the K on-code pieces carry the
mechanism. Rows: PR 68.5, n50/90/99 = 25/90/147 (embedding-density diffuse);
SVD: PR 15.5, n99 = 37 (collective). Artifacts:
`results|figures/embedded/ideal_vs_vpd/mechanism_forward/` (census_d200.md,
activity.csv, activity-curve / truncation-fidelity / example-spectra figures).

## Addendum — cut-side principle grid (d200): where to cut each of the four matrices

Completing the census's open question: the full 4-matrix × 2-cut-direction grid
(`--grid` mode of `embed/embed_mechanism_forward.py`), each matrix of
y = relu(x W_e W_in) W_inᵀ W_eᵀ cut into input-side rows or output-side columns
(all 8 tilings bitwise), with truncation fidelity measured for ONLY that matrix
truncated to its top-m pieces by |activation coefficient| (untied — the other
three matrices stay full; the W_in output-cut coincides with the established
tied per-neuron result, m* = 5 for every feature, regression-asserted).

**Verdict: the cut-side principle survives unfalsified — every matrix is
causally minimal exactly on the side where its own weights re-concentrate the
signal into discrete events, never along the dense embedding interface.**
Winner sides 4/4 as predicted, spike-m* 4/4 at the predicted counts:
W_emb input-cut 1 (the active feature's row is the whole read; errors ~1e-15
from m = 1) vs output-cut 162/110 (spike/leak); W_in output-cut 5 vs input-cut
162/110; W_out input-cut 5 (post-ReLU h is exactly K-sparse) vs output-cut
161/153; W_unemb output-cut spike 1 (the top piece IS the target readout
⟨z, W_e[j]⟩, error 0 at m = 1 for every feature) vs input-cut 161/153. Two
structural readings: (1) the dense-interface cells are pairwise the SAME
decomposition (W_emb output-cut ≡ W_in input-cut and W_out output-cut ≡
W_unemb input-cut, truncated forwards bitwise equal) — an interface cut is one
object however booked, and it needs ~110–160 of 200 pieces; (2) the one
deviating number, W_unemb output-cut leak m* = 98, is a criterion artifact:
at the final matrix the coordinate cut is aligned with the error metric, so
leak fidelity degenerates into a support census of the output's leak
(n99 = 97) — the re-concentrated mechanism is still exactly 1 piece, and
W_emb's 1/1 differs only because the probe itself is 1-sparse. Artifacts:
`results|figures/embedded/ideal_vs_vpd/mechanism_forward/
cutside_grid_d200.{md,csv,png}`.

## Addendum — d50 reruns of the grid and the solo spectra: what survives without G = I

Both experiments repeated at d_emb = 50 (`--d-emb 50`; outputs suffixed
`_d50`, d200 artifacts untouched). The regime change is a single fact with
many faces: W_e W_in = G C with G = W_e W_eᵀ a rank-50 projector (mean
G_jj = 0.5), so off-code pre-activations go positive through G-noise and the
causal count #(pre > 0) is median 21 (14–26) per feature instead of exactly
K = 5. Everything that was an identity stays an identity (all 8 tilings
bitwise, interface-duality pairs bitwise, W_emb input-cut exact, m*
v-invariant); everything that was a G = I consequence is now reported, not
asserted.

**Grid:** winner sides 3/4; the {1,5,5,1} knees soften to {1, 34, 14,
1-spike} (winner-side median m*; dense-interface cells at 40–43 of 50). The
one flip, W_unemb, is the d200-diagnosed criterion artifact promoted to
decisive: the output cut's spike m* is still 1 with the top piece = the
target readout for 100/100 features, but its leak m* (75 ≈ y-census
n99 = 73) is a support census of the now-broadband G-noise leak. The W_in
output-cut m* (34) even overshoots the causal count (21) because the
|coefficient| ranking spends slots on large-|negative| dead pieces — a
ranking artifact worth remembering for any p>0-style importance ordering.

**Solo spectra (1+5+5+1, features 0/61/1):** mech spike 0.148/0.083/0.130 vs
full 0.203/0.177/0.190 (27–53% spike deficit — top-5 misses causal mass, and
the kept top-5 is no longer the on-code set: 4/5, 3/5, 4/5). The strict
mechanism's off-target is exactly 0 at any d (structural: the W_unemb
column-j piece writes only coordinate j); the predicted nonzero G-noise leak
appears at the retained write read through the full unembed bank (full-read
leak max 0.117/0.088/0.122, same order as the full model's 0.114/0.153/0.096).
The full model's leak ladder — exact rungs per overlap class at d200 —
smears into overlapping bands (within-class std up to 4.5e-2 vs ~3e-16;
probe 61 has an m_ov=1 leak above its m_ov=3 leak). Predictions on record:
spike ≈ K a² G_jj² confirmed with spread (mech ratio med 1.061, 0.447–1.920);
knee softening confirmed for ALL features; ladder smear confirmed. Artifacts:
`results|figures/embedded/ideal_vs_vpd/mechanism_forward/
{cutside_grid_d50.md/csv/png, mechanism_spectra_d50.md/png}`.

## Addendum — featbasis (top-1 feature-frame) W_in ontology at d200: exact tiling, Gram-limited causality

A fourth W_in ontology (`--featbasis` mode of `embed/embed_mechanism_forward.py`,
d200 only), built by deliberately relaxing the layer-locality rule: 100 pieces
P_j = W_e[j] ⊗ g_j with Gmat = pinv(W_eᵀ) W_in, CROSS-LAYER (the pieces use
W_emb's weights) but PRE-INFERENCE (fixed before any input — the Option-A
objection to inference-time projections does not apply). The construction is
exact and fully algebraic: Σ_j P_j = W_in to 1.9e-15; the activation
coefficient of piece P_i on probe v·e_j is v (W_e W_eᵀ)[j,i], the feature-frame
GRAM ROW; and at d200, Gmat = (W_e W_eᵀ)⁻¹ C — the output sides are the
dual-frame codewords, not the codewords.

**Verdict: the top-1 promise is a census illusion.** The Gram-row census is the
most concentrated of the four ontologies (PR median 2.3, n50 = 1 everywhere;
self coefficient beats every single rival by ≥ 3.96×) — but the self piece does
NOT beat the SUM of its 99 rivals (ratio median 0.177, < 1 for 100/100), and
the four-layer solo (W_emb row j + top-m pieces + tied transposes + W_unemb
column j) shows why the causal story inverts: g_j = Gram⁻¹ c_j is
dual-amplified (‖g_j‖/‖c_j‖ median 2.82, piece-norm overcompleteness
Σ‖P_j‖_F/‖W_in‖_F = 20.3), so the m = 1 solo OVERSHOOTS the full spike by
median 4.4× and the exact tiling survives only through massive cross-piece
cancellation. Minimal k for a 1% spike: median 74 (33–94) first-crossing / 92
(82–98) persistent, of 100. The five-ontology comparison on probe e_1
(`mechanism_minimal_k_d200_v2.png`, legacy convention regression-asserted:
columns 5 / SVD 27 / rows 167): featbasis k = 86 — between SVD and rows,
nowhere near k = 1; per-neuron columns keep the causal crown at 5.

Bonus falsification from the cheap VPD cross-check: at d200 the feature frame
and its dual are NOT near-identical (cos(W_e[j], dual_j) median 0.70 — Gram
off-diagonals, mean |·| 0.056, accumulate over 99 rivals), and the learned VPD
lin_in components separate them cleanly: best-|cos| vs the dual rows median
0.911 vs 0.633 for the feature rows (same best-match feature 91/100) — a raw,
re-scoring-free confirmation that VPD's embedded read ontology is the dual
basis. Artifacts: `results/embedded/ideal_vs_vpd/mechanism_forward/
featbasis_census_d200.md`; `figures/embedded/ideal_vs_vpd/mechanism_forward/
{featbasis_activity_d200.png, mechanism_minimal_k_d200_v2.png}`.

**Top-1 decomposition (2026-08-19, user-driven).** Slicing W_in along the
embedded feature directions requires a correction for frame non-orthogonality,
and its placement decides everything: on the slice contents (min-norm exact
tiling) pieces are Gram-amplified 2.8x and top-1 overshoots 4-5x (cancellation
ratio 20x — top-1 by activation is not top-1 by mechanism); with no correction
top-1 solos are exact but tiling fails (72% residual); with the correction on
the DIRECTIONS (dual frame: pieces w~_j x c_j) both are exact — tiling to 1e-6
and top-(1,1,1,1) mechanisms reproducing the full spike (0.6645) with
off-target exactly zero. This is the exact per-feature PD of the embedded code
(cross-layer, pre-inference), and the frame VPD's learned lin_in components
align with (0.911 vs 0.633 for raw feature rows).
