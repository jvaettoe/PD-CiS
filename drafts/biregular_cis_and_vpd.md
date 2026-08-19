# The biregular code as a CiS testbed: exact parameter decompositions and where VPD departs from them

*Draft — 2026-08-19. Covers the designed biregular code (transpose decoder) with
and without embedding. Sources: biregular/MEMO.md, embed/MEMO.md,
results/vpd_objective/, results/embedded/ideal_vs_vpd/, RESEARCH_LOG.md.*

## 1. The models

**Plain code.** MLP 100 → 50 (ReLU) → 100, no biases, task y = ReLU(x) on
sparse inputs (Bernoulli(p), values U[−1,1]). Weights designed, not trained:
a biregular binary code M (each feature owns a K-neuron codeword; each neuron
serves 2K features) with the two-scalar ansatz **w_in = a·M + b·(1−M)** and
transpose decoder **w_out = w_inᵀ**. Workhorse: K = 5, p = 0.001, fitted
(a, b) = (0.3646, −0.0476).

**Embedded code.** The same code wrapped in a fixed random embedding:
y = ReLU((x·W_e)·W_in)·W_inᵀ·W_eᵀ with W_e ∈ R^{100×d_emb} (unit-norm rows)
and **W_in = W_e⁺·C**, C = a·M + b·(1−M). Two regimes: d_emb = 200 (≥ F, where
G = W_e·W_e⁺ = I) and d_emb = 50 (< F, where G is a rank-50 projector).

## 2. Evidence for computation in superposition

**The plain code computes 100 ReLUs in 50 neurons, and the interference is
quantized.** All 100 features are computed (per-feature active-conditional MSE
far below the do-nothing baseline of 1/3), at K codeword neurons per feature —
i.e., 2 features per neuron of capacity. The signature that this is a *code*
and not an accident is the **overlap ladder**: the response at feature i when
probing feature j takes only the discrete values m·a² + (K−m)·a·b, indexed by
the integer codeword overlap m, with rung occupancy following the
hypergeometric law across code draws. Self-response is K·a² = 0.6645 — a
loss-optimal self-attenuation, not a defect.

**The embedding preserves CiS exactly above rank and gracefully below it.**
At d_emb ≥ F the sandwich computes literally the same function as the plain
code (verified to 1e-15): same census, same exact ladder. Below F the ladder
smears — self-responses drop to ≈ K·a²·G_jj² with per-feature spread, and each
overlap class widens into a band (class std ≈ rung spacing at d = 50, so
classes overlap) — but computation persists: the smearing is *quenched
disorder* from cramming 100 feature directions into d dimensions, not noise.
A d_emb sweep shows the discreteness is recovered **exactly at d_emb = F**
(class std collapses four orders of magnitude at d = 100 precisely, because
G = I is a rank fact, not an asymptotic one), with the embedding at its most
ill-conditioned exactly at the transition. Trained EmbedMLPs track designed
codes closely down to d_emb ≈ 60 (multi-seed dFH battery: decoder-pseudoinverse
cosine 0.996 ± 0.000 at d = 1000 falling to 0.686 ± 0.009 at d = 20; trained
beats matched designed codes until exact parity at d = 40).

**Loss dependence (scope note).** CiS in this family is bought by the loss
exponent: under MSE the same architecture triages (~min(N, F) features with
dedicated neurons; verified as the global optimum, robust to weight decay and
seed), with full CiS appearing at Ln, n ≈ 2.5–3. All claims here are for the
L4-family regime in which the code exists.

## 3. Exact parameter decompositions

**Plain code (ExactPD).** 100 chain-glued per-feature components: component j
= (sparse read row: feature j's codeword row of w_in) ⊕ (write column j of
w_out). Faithfulness is exact by construction (components tile both matrices);
each piece is rank-1; minimality holds in the **self-activation sense**: the
solo forward of component j reproduces the model's on-target spike exactly and
is leak-free (component j writes only coordinate j). A consequence measured
mask-free (klfree eval): the union of active exact components therefore cannot
emit the interference the model produces on inactive coordinates — the
per-feature write basis is distributionally inexpressive off-target
(KL(full||exact) ≈ 7-10 on interference classes vs ≈ 1.1 for the learned
neuron-basis solution, which traces the full output distribution; the oracle
neuron-basis ensemble is distributionally exact). VPD's preference is thus
not only a pricing fact but an expressivity fact about output distributions.

**Embedded code (layer-local ideal PD).** Requiring each layer to be
decomposed using only its own weights (no cross-layer knowledge, fixed before
inference) still yields an exact mechanism ontology, located by the
**cut-side principle**: *decompose each matrix on the side where its own
weights re-concentrate the signal into discrete events; never cut along the
dense embedding interface.* Tested as a falsifiable 4×2 grid (each matrix cut
both ways), the predicted winners hold in every case, with mechanism sizes
{1, 5, 5, 1}: W_emb per-feature rows (1 active) → W_in **per-neuron columns**
(K = 5 active; the ReLU renders all off-code pieces exactly inert) → W_out
per-neuron rows (5) → W_embᵀ per-feature columns (1). The solo mechanism
reproduces the model's spike exactly and is leak-free — interference is owned
by the *receiving* feature's unembed piece. Dense-interface cuts fail
symmetrically (~160–200 pieces needed) and are literally the same
decomposition viewed from either adjacent matrix. At d_emb = 50 the mechanism
survives but inflates (~20 feature-dependent pieces, mostly G-noise-activated
off-code neurons) and loses codeword identity.

**Two ideals that provably cannot coincide.** The SVD of W_in is the canonical
*weight-mass* decomposition (fewest rank-1 pieces = rank; optimal mass
concentration) but is causally dense: reproducing one feature's mechanism
needs ~27–40 of its 50 channels (vs 5 columns; 167+ of 200 for
per-dimension rows), its channels load densely across features, and truncation
cannot denoise. Conversely, any *causally minimal* per-feature decomposition
(the dual-basis construction: W_in = Σ_j w̃_j ⊗ c_j, exact self-gating at
d ≥ F, least-squares optimal below) needs 100 overcomplete non-orthogonal
pieces in a rank-50 matrix. Count/mass minimality and causal minimality are
incompatible on the code layer by pigeonhole: 100 mechanisms, rank 50. "The"
exact PD is therefore purpose-relative; the causal one is what mechanism
recovery should target.

## 4. Where VPD departs — and why

**What VPD finds (two implementations, converged).** On the plain code: read
pieces on feature axes, write pieces on **neuron** axes — ~50 write parts
(our port: 53 → 50; Goodfire's unmodified param-decomp at their own TMS 40-10
pricing: 62 → 50 after a warm restart, concentration 0.98, 49/50 axes). On
the embedded code: W_emb rows recovered (conc 0.99); W_in components aligned
with the **dual frame** (median |cos| 0.911 at d200) and *not* the SVD frame
(0.09); diluted (0.564) at d50 where no cleanly gateable frame exists. So VPD
agrees with the exact PD wherever pieces are input-side gateable (reads,
embedding rows, dual directions) and re-bases the write side onto neurons —
50 parts serving 100 functions.

**Why — measured, not conjectured.**
1. *Inserting the exact PD into the VPD objective raises the loss 6.4×*
   (oracle gates, audited): the penalty is entirely masked-reconstruction on
   w_out (read side exactly 0), while the exact solution is actually *sparser*
   on importance (0.200 = 2pF vs 0.253). The cost is **interference
   ownership**: masking an inactive feature's write piece deletes leak the
   model genuinely emits, and the objective demands the full output —
   interference included — be preserved.
2. *The preference is a two-term asymmetry*: hard-gated neuron-basis writes
   have exactly zero reconstruction cost but 48× the importance bill — and
   lose at default pricing; the learned optimum uses neuron pieces with
   *soft* gates. Reconstruction is un-cheatable (the PGD adversary finds any
   deleted interference; no gate rescues the exact basis — trained gates on
   frozen exact pieces: 3.58e-4 vs oracle 3.31e-4), while importance is
   soft-discountable. Training selects the basis whose cost sits in the
   discountable term.
3. *The neuron-basis configuration is the attractor*: full training from it
   reproduces the learned solution exactly (total 0.995×, write pieces matched
   at cos 0.998, 100% slot agreement); no pricing in a 30× sweep recovers the
   exact basis (importance pressure collapses the gates before rotating the
   basis — degenerate collapse is the global optimum at canonical pricing,
   a prediction made and confirmed at two sparsities).
4. *The mechanism is partly ideal-algorithm, partly design*: activity of a
   rank-1 piece is an input-side (kernel) property, and VPD gates/masks/prices
   exactly that projection — so any ablation-based minimality with full-output
   faithfulness inherits the input-frame preference (the oracle results prove
   gate architecture is not the cause). The scalar layerwise CI networks
   reinforce it; their untested `mode: "global"` variant is the remaining
   architectural escape hatch.

**The upshot for PD evaluation.** VPD is not failing to optimize — it is
faithfully optimizing an objective whose minimality concept (ablate whatever
is not causally active, preserve everything else, including interference)
provably prefers 50 shared parts over 100 owned mechanisms. The disagreement
with ExactPD reduces to one commitment: *is interference output to be
preserved, or noise to be attributed?* Consequently: (i) read-side and
write-side recovery must be scored separately; (ii) alignment must be scored
against the causally gateable frame (duals, not raw feature rows, in embedded
models) or true mechanisms will be misclassified as "mixed"; (iii) pricing
must be validated against a ground-truth cost harness before any run is
interpreted — under-priced regimes collapse silently and read as "no structure
exists"; and (iv) activation-energy censuses overstate causal size — only
truncated-forward fidelity separates live pieces from ReLU-dead ones.
