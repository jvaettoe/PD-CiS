# Memo — PD terminology and the biregular worked example

**Date:** 2026-07-30 · **Re:** What "component" means in APD/SPD/VPD vs our ExactPD, and how VPD decomposes the biregular model
**Model:** y = relu(x·W_in)·W_out; F=100 features, N=50 neurons; codewords K=5, neuron degree d=10.

## Terminology

**Subcomponent** — a rank-1 slice of *one* weight matrix. Atomic learned object. Subcomponents must **tile** their matrix (sum to it, no overlap).

**Component (mechanism)** — subcomponents that *always co-activate*, merged: if no input dissociates them, they count as one (possibly rank > 1) mechanism.

**Circuit (chain)** — the input-conditional *path* through mechanisms across layers. Not a learned object; assembled at runtime by routing. Circuits may overlap in shared mechanisms.

**Convention split.** APD — and our ExactPD — glue chains into components *by construction*: one index owns a slice of every matrix (component j = feature j's read row + write column). SPD/VPD learn components **per matrix with independent indices**; cross-layer chains exist only as circuits. Hence: feature-indexing lives in the *parameters* for ExactPD, in the *routing* for VPD.

## Worked example (VPD on the biregular code, 100k steps, p = 0.001)

- **Read side:** 100 subcomponents = the codeword rows → recovered in ExactPD's basis (act purity 0.987, 97/100 coverage).
- **Write side:** 50 subcomponents = the neuron rows, e_n ⊗ W_out[n,:] (concentration 0.91, all 50 neurons, bijective). Solo act ≈ 0.10 ≈ 1/K — a structural cap, not failure: one neuron carries ~1/K of a feature's write.
- **No merging:** write shards fire in codeword patterns, never all together → 100 read + 50 write mechanisms, all dissociable. (Goodfire's dense resid-MLP is the degenerate case — all 50 co-fire on every input, collapsing to their "single rank-50 W_out component," SPD §3.3.)
- **Circuits:** feature j's chain = read component j + the 5 write shards of its codeword neurons; the union reconstructs its write at act 0.96. Each write shard serves d = 10 circuits; circuit membership = the code matrix M (check: 100·5 = 50·10). Ablating j's matched pair damages j specifically (agreement 0.96) — the two methods agree at circuit level while tiling W_out differently.

## Why not 100 write mechanisms?

Parameter pieces must tile. Per-feature pieces of 5 neuron-rows each would sum to 10·W_out (each row is shared by 10 features); rescaled 1/10 they can't implement their feature alone. The only valid tilings: per-feature *columns* (ExactPD — dense pinv columns that must all stay active to reproduce interference leak; minimality catastrophe) or per-neuron *rows* (VPD — 5 active per input, exact reconstruction). **The model has 100 write functions but only 50 write parts** (10 functions per part). VPD learns the parts; the functions live in the routing.

## Bottom line

"Component" without qualification is ambiguous across this literature. Compare methods at two well-defined levels: **per-matrix basis** (they provably differ on W_out: per-feature columns vs per-neuron rows — the latter being Goodfire's own post-SPD target, dense k = d_mlp) and **circuit** (they agree almost perfectly here). Our biregular code is the clean case that separates the two readings, refining SPD's "single rank-50 component" into 50 sparsely-activating neuron mechanisms.
