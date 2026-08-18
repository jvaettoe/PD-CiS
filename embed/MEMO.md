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
