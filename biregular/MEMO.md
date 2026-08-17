# Memo: the biregular code model (designed CiS ground truth)

## 1. Model architecture
Single-hidden-layer MLP, **100 features → 50 neurons (ReLU) → 100 outputs, no
biases**, with weights *designed*, not trained: a biregular binary code M
(each feature owns a K-neuron codeword; each neuron serves 2K features) and the
c-free 2-scalar ansatz **w_in = a·M + b·(1−M)**, decoder either transpose
(w_out = w_inᵀ) or pinv (w_out = pinv(Mᵀ)ᵀ). Workhorse instance: K = 5,
seed 0, no-swap, transpose. Only (a, b) are fitted (L4, Adam+cosine, 4000
steps) at the target sparsity; overlap-min variants (reduce_overlap) serve as
benchmarks.

## 2. What it is made to compute / trained on
The main-suite CleanDataset task: sparse inputs (Bernoulli(p), values U[−1,1]),
target y = ReLU(x) — 100 ReLUs computed in superposition through 50 neurons.
The designed code makes the ground-truth mechanism ontology exact and known:
**100 per-feature components**, each a sparse read row (K codeword entries)
chained to a write column, with interference quantized on the overlap ladder
(response of a feature sharing m neurons = m·a² + (K−m)·ab, transpose decoder).

## 3. Summary of findings

**Code structure (ExactPD reference).** The m-overlap ladder is exact at all K;
rung occupancy is hypergeometric across seeds. ExactPD's 100 chain-glued
per-feature components reproduce the model by construction (faithfulness and
per-feature self-activation exact); the designed-vs-trained performance gap is
driven by graded per-connection weights, not codeword length.

**Our VPD port (in-repo).** Converged fits recover per-feature READ pieces but
a per-NEURON write basis: 53 alive subcomponents trending to ~50, no knob
changes the write side. Goodfire's MDL clustering (hard partition) cannot
reassemble the 100 overlapping mechanisms. Solo activations of (read, write)
pairs are capped at a² ≈ 1/K of the spike by construction.

**Port into Goodfire's own codebase (param-decomp, 2026-08-17).** With their
canonical TMS coefficients the pipeline collapses to a no-op at p=0.001 AND
p=0.05 (all-CIs-zero is the global optimum — ground truth scores 34× worse;
prediction confirmed out-of-sample). Repriced within their own config space
(imp 1e-4, pnorm 2.0 = their TMS 40-10 values): structure forms and, after a
warm-restart cycle, converges to **~99 feature-aligned read components (conc
0.95) + 50 neuron-aligned write components (conc 0.98, 49/50 axes)** — the
same ontology as our port, with the same transient-overcount phenomenology
(their 62 → 50 mirrors our 53 → 50). Cross-implementation corroboration.

**Why the write side comes out in neurons (six-variant objective analysis, one
shared eval pool; results/vpd_objective/).**
- *exact_oracle*: ExactPD pieces + perfect gates cost **6.4× more** than the
  learned solution — entirely reconstruction (9× stoch, 13× PGD), all on
  w_out; read-side recon is exactly 0 (input-aligned pieces are self-masking).
  Exact is *sparser* on importance (0.200 = 2pF vs 0.253). The deficit is
  interference ownership: masking inactive features' write pieces deletes leak
  the model genuinely emits.
- *exact_trainedci*: learned gates cannot rescue the exact basis (3.58e-4).
- *hybrid_oracle* (100 half-row neuron-basis write pieces, oracle gates):
  recon exactly 0 (1e-18) but importance 48× — at default pricing it LOSES to
  exact_oracle. The selection mechanism is therefore a two-term asymmetry:
  **recon+PGD is un-cheatable, importance is soft-discountable**; training
  picks the basis whose cost sits in the discountable term and discounts it.
- *hybrid_trainedci*: gates-only on frozen neuron pieces reaches 2.62e-4 —
  better than all oracles, still 5× short of the optimum (piece co-adaptation
  required).
- *hybrid_fulltrain*: full training from the neuron-basis init reaches
  **0.995× the learned reference**, collapsing 100 half-rows to 50 alive write
  pieces matched to the reference at median cos 0.998 with 100% neuron-slot
  agreement. The learned solution is the attractor of the objective from the
  neuron-basis configuration.
- Mechanism note: the layerwise scalar CI (per-component 1→16→1 MLP on
  v_cᵀ·layer-input) is a *design choice* that reinforces input-basis pieces;
  the oracle results show the preference itself follows from the ideal
  objective (ablation-based minimality + full-output faithfulness). Untested
  alternative: their `mode: "global"` CI.
- Exp-2 pricing sweep (imp ×{10..300}): no exact-basis window opens —
  importance pressure starves the gates before rotating the write basis
  (structured → starvation → CI collapse at ×300). The exact decomposition is
  both statically dispreferred at default pricing and dynamically unreachable
  at re-tuned pricing (10k-step budget caveat).

**Takeaway.** On its own ground-truth model, VPD-family decomposition is
basis-correct on the read side and systematically re-bases the write side to
neurons, in both implementations, for measured objective-level reasons: the
choice "interference is output to be preserved" + "gating acts on input-side
projections" makes per-feature write mechanisms the expensive ontology. Any
PD evaluation on CiS models must therefore score write-side recovery
separately from read-side recovery, and the pricing regime (importance coeff
vs task scale) must be validated against a ground-truth cost harness before
runs are interpreted — degenerate collapse is silent and looks like "no
structure exists."
