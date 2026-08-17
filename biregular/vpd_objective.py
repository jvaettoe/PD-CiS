"""Experiment 1 -- does the truth cost more? Score ExactPD under the ported VPD objective.

Builds a VPD state whose C=100 subcomponents are EXACTLY the ExactPD per-feature pieces
(vpd.py parametrization is V [d0,C], U [C,d1], piece c = outer(V[:,c], U[c,:]), so
    w_in  piece j = e_j (x) w_in[j,:]   ->  V_in  = I,      U_in  = w_in
    w_out piece j = w_out[:,j] (x) e_j  ->  V_out = w_out,  U_out = I
and the weight delta is 0 by construction), then evaluates three variants on ONE fixed
eval pool (CleanDataset p=0.001, seed 777, 25 x 8192) with the port's default term
weights (decomp_fit.loss_kwargs_for("VPD")):

  exact_oracle      -- exact pieces + ORACLE gates: ci_j = [x_j != 0] on both matrices
                       (component j == feature j; the CI net is bypassed analytically)
  exact_trainedci   -- exact pieces FROZEN + CI nets trained (--fit-ci; deferred, long)
  learned_reference -- the saved 25k-step baseline VPD fit re-evaluated on the pool
  hybrid_oracle     -- (--hybrid, eval-only) write-basis ablation: identical exact READ
                       pieces (V_in = I, U_in = w_in) but NEURON-basis writes with all
                       C=100 components live -- component c carries HALF of neuron row
                       (c mod d_mlp): V_out[:, c] = e_{c mod N}, U_out[c, :] =
                       0.5 * w_out[c mod N, :] (each neuron row split across exactly 2
                       components; sum of pieces == w_out exactly). Oracle gates:
                       w_in as exact_oracle; w_out ci_c = [h_{c mod N} != 0] with
                       h = relu(x @ w_in) of the frozen model (both halves of a neuron
                       gate together). Isolates the write-basis choice with reads, gate
                       quality and faithfulness held fixed.
  hybrid_trainedci  -- (--hybrid --fit-ci) the same frozen hybrid pieces with the CI
                       nets TRAINED (port optimizer, --ci-steps), parallel to
                       exact_trainedci: can learned gates price the neuron write basis
                       down to the learned_reference total?
  hybrid_fulltrain  -- (--hybrid-full) initialize AT the hybrid configuration but make
                       everything trainable (components + fresh CI nets) and run the
                       port's standard full recipe (--full-steps, default 25k, default
                       weights, METHOD_LR). The record carries recovery_vs_learned:
                       Hungarian |cos| matching of rank-1 pieces to learned_reference
                       per matrix, per-neuron write-slot agreement, and heuristic
                       pass/fail criteria (see recovery_criteria). --gates-compare
                       post-hoc compares hybrid_trainedci's learned CI behavior to
                       learned_reference's on single-feature probes.

Experiment 2 (pricing sweep) lives here too: --sweep-cell fits a FRESH VPD (C=100)
to the same frozen model with the importance-minimality weight multiplied by
--imp-mult R (recon weights unchanged -- only the ratio matters), using the port's
standard fit machinery (mirrors biregular_vpd_compare's decomp_fit driving, steps
overridable). The record is the standard JSON EXTENDED with basis-ontology metrics
(read/write concentrations vs the ExactPD and neuron bases, alive counts, and a
thresholded verdict: neuron_basis / exact_basis / collapsed / mixed), and
--phase-figure renders figures/vpd_objective/phase_diagram.png over all sweep_*
runs. NB the sweep's fit seed is --fit-seed (default 0); --seed stays the
EVAL-POOL seed so every cell is scored on the same fixed pool as the variants.

  ./.venv/bin/python -m biregular.vpd_objective              # oracle + learned evals
  ./.venv/bin/python -m biregular.vpd_objective --hybrid     # eval hybrid_oracle only
  ./.venv/bin/python -m biregular.vpd_objective --fit-ci     # deferred: train CI nets
  ./.venv/bin/python -m biregular.vpd_objective --hybrid --fit-ci  # hybrid_trainedci
  ./.venv/bin/python -m biregular.vpd_objective --index      # rebuild index.md only
  ./.venv/bin/python -m biregular.vpd_objective --sweep-cell --imp-mult 55
  ./.venv/bin/python -m biregular.vpd_objective --phase-figure
"""
import argparse
import glob
import json
import os
from datetime import datetime

import torch

import core.decomp_fit as decomp_fit
from biregular.biregular_ansatz import CodeAnsatz
from biregular.biregular_vpd_compare import DEFAULT_KW, FIT_SEED, _decomp_path
from biregular.component_match import (activation_matrices, hungarian_match, mass_matrices,
                                       neuron_basis_stats, score_sides)
from core.data import CleanDataset
from core.paths import CHECKPOINTS, FIGURES, RESULTS
from core.pd import ExactPD
from core.vpd import VPD

F = 100
P_DEFAULT = "0.001"
EVAL_SEED = 777
EVAL_BATCHES, EVAL_BATCH = 25, 8192
INIT_SEED = 0            # CI-net init (V/U are overwritten, so it only matters for --fit-ci)
CI_STEPS_DEFAULT = 4000  # CPU screening budget (cf. biregular_vpd_compare --steps)
SWEEP_STEPS_DEFAULT = 10_000
SWEEP_ALIVE_FRAC = 0.10  # alive = component Frobenius mass > 10% of the side max

RES_DIR = os.path.join(RESULTS, "vpd_objective")
RUNS_DIR = os.path.join(RES_DIR, "runs")
INDEX_PATH = os.path.join(RES_DIR, "index.md")
CKPT_DIR = os.path.join(CHECKPOINTS, "vpd_objective")
FIG_DIR = os.path.join(FIGURES, "vpd_objective")
PHASE_FIG = os.path.join(FIG_DIR, "phase_diagram.png")

# Ontology-verdict thresholds (stored verbatim in each sweep record so the JSON is
# self-documenting). conc/cos/cov are means over ALIVE components / fractions of
# the relevant basis size; collapsed fires first (too few alive pieces for even a
# neuron-basis tiling on the write side, or fewer read pieces than half the
# features), then exact_basis, then neuron_basis, else mixed.
VERDICT_THRESHOLDS = {
    "alive_frac": SWEEP_ALIVE_FRAC,
    "collapsed_alive_out_frac_dmlp": 0.5,   # alive_out < 0.5*d_mlp -> collapsed
    "collapsed_alive_in_frac_F": 0.5,       # alive_in  < 0.5*F     -> collapsed
    "conc_hi": 0.8,                         # concentration/cosine "high" cut
    "coverage_frac": 0.8,                   # distinct-argmax coverage cut
}

TERM_WEIGHT_KEY = {"importance": "alpha_importance", "stoch_recon": "alpha_stoch",
                   "pgd_recon": "alpha_pgd", "faithful": "alpha_faithfulness"}


class OracleVPD(VPD):
    """VPD with the CI net bypassed: importance_j = 1 iff feature j is active
    (x_j != 0) in the sample, identically on both matrices. Valid only for the
    exact per-feature pieces (component j == feature j, C == n_features) on a
    plain MLP (identity embedding, so e == x). lower == upper == the oracle gate."""

    def _ci(self, e, h):
        assert self.C == self.model.n_features, "oracle gates need component j == feature j"
        g = (e != 0).float()
        return {n: g for n in self.param_names}, {n: g for n in self.param_names}


def build_exact_vpd(model, oracle: bool = False, init_seed: int = INIT_SEED) -> VPD:
    """VPD state whose C=F subcomponents ARE the ExactPD pieces (frozen, delta == 0).
    Asserts the pieces tile each matrix exactly and match ExactPD's own components."""
    torch.manual_seed(init_seed)  # reproducible CI-net init (V/U overwritten below)
    cls = OracleVPD if oracle else VPD
    pd_v = cls(model, C=model.n_features, warmup_steps=0)
    eye = torch.eye(model.n_features, device=model.device)
    with torch.no_grad():
        pd_v.V["w_in"].copy_(eye)                    # (F, C):  V_in[:, j] = e_j
        pd_v.U["w_in"].copy_(model.w_in)             # (C, N):  U_in[j, :] = w_in[j, :]
        pd_v.V["w_out"].copy_(model.w_out)           # (N, C):  V_out[:, j] = w_out[:, j]
        pd_v.U["w_out"].copy_(eye)                   # (C, F):  U_out[j, :] = e_j
    for k in pd_v.V:
        pd_v.V[k].requires_grad_(False)
        pd_v.U[k].requires_grad_(False)
    for name in pd_v.param_names:                    # exact tiling: sum of pieces == matrix
        err = float((pd_v._vu(name) - pd_v.targets[name]).abs().max())
        assert err < 1e-6, f"exact tiling failed for {name}: max|sum - W| = {err:.3e}"
    pd_e = ExactPD(model)                            # cross-check against the pd framework
    for j in (0, 17, 99):
        for name in pd_v.param_names:
            assert torch.allclose(pd_v._component_weight(name, j),
                                  pd_e._component_weight(name, j), atol=1e-6), (name, j)
    return pd_v


class HybridOracleVPD(VPD):
    """Oracle gates for the HYBRID pieces (exact reads, split-neuron writes).
    w_in: as OracleVPD, ci_j = [x_j != 0] (component j == feature j).
    w_out: component c carries half of neuron row (c mod d_mlp), so its importance
    is [h_{c mod d_mlp} != 0] under the FROZEN model's hidden h = relu(x @ w_in);
    both halves of a neuron gate together. lower == upper == the oracle gate."""

    def _ci(self, e, h):
        assert self.C == self.model.n_features, "oracle read gates need component j == feature j"
        reps = self.C // h.shape[1]
        assert self.C == reps * h.shape[1], "hybrid write gates need C to be a multiple of d_mlp"
        g_in = (e != 0).float()
        g_out = (h != 0).float().repeat(1, reps)
        lower = {"w_in": g_in, "w_out": g_out}
        return lower, dict(lower)


def build_hybrid_vpd(model, oracle: bool = False, init_seed: int = INIT_SEED) -> VPD:
    """VPD state with the ExactPD READ pieces (V_in = I, U_in = w_in, identical to
    build_exact_vpd) but split-neuron WRITE pieces: all C = F components live,
    component c = outer(e_{c mod N}, 0.5 * w_out[c mod N, :]) in neuron space, so
    each of the N = d_mlp neuron rows is split across exactly C/N components and
    the pieces still tile w_out exactly (delta == 0). Components always frozen;
    oracle=True bypasses the CI nets, oracle=False leaves them trainable
    (the hybrid_trainedci setting, parallel to build_exact_vpd(oracle=False))."""
    torch.manual_seed(init_seed)  # reproducible CI-net init
    cls = HybridOracleVPD if oracle else VPD
    pd_v = cls(model, C=model.n_features, warmup_steps=0)
    n_feat, d_mlp = model.n_features, model.d_mlp
    reps = n_feat // d_mlp
    assert n_feat == reps * d_mlp, f"C={n_feat} must be a multiple of d_mlp={d_mlp}"
    eye_f = torch.eye(n_feat, device=model.device)
    eye_n = torch.eye(d_mlp, device=model.device)
    with torch.no_grad():
        pd_v.V["w_in"].copy_(eye_f)                          # (F, C):  V_in[:, j] = e_j
        pd_v.U["w_in"].copy_(model.w_in)                     # (C, N):  U_in[j, :] = w_in[j, :]
        pd_v.V["w_out"].copy_(eye_n.repeat(1, reps))         # (N, C):  V_out[:, c] = e_{c mod N}
        pd_v.U["w_out"].copy_(model.w_out.repeat(reps, 1) / reps)  # (C, F): half neuron rows
    for k in pd_v.V:
        pd_v.V[k].requires_grad_(False)
        pd_v.U[k].requires_grad_(False)
    for name in pd_v.param_names:                            # exact tiling: sum of pieces == W
        err = float((pd_v._vu(name) - pd_v.targets[name]).abs().max())
        assert err < 1e-6, f"hybrid tiling failed for {name}: max|sum - W| = {err:.3e}"
    pd_e = ExactPD(model)                                    # reads identical to ExactPD ...
    for j in (0, 17, 99):
        assert torch.allclose(pd_v._component_weight("w_in", j),
                              pd_e._component_weight("w_in", j), atol=1e-6), j
    for c in (0, 23, 49, 50, 73, 99):                        # ... writes = half neuron rows
        want = torch.outer(eye_n[:, c % d_mlp], model.w_out[c % d_mlp, :] / reps)
        assert torch.allclose(pd_v._component_weight("w_out", c), want, atol=1e-6), c
    return pd_v


def load_learned(code: CodeAnsatz, p: str, device: str) -> tuple:
    """(model, pd_v, payload) for the baseline 25k-step fit (the checkpoint that
    biregular_vpd_compare --reuse and cluster_vpd load by default: knob tag '')."""
    path = _decomp_path(code.name, p, "")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = code.model(tuple(payload["biregular"]["ab"]), device=device)
    pd_v = VPD(model, payload["C"], warmup_steps=0)
    pd_v.load_state_dict(payload["pd_state"], strict=False)
    return model, pd_v, payload


# ---------------------------------------------------------------- evaluation

def eval_objective(pd_v: VPD, *, p: float, seed: int = EVAL_SEED,
                   n_batches: int = EVAL_BATCHES, batch: int = EVAL_BATCH,
                   loss_kw: dict = DEFAULT_KW) -> dict:
    """Mean per-term losses over the fixed pool. The dataset owns its generator, so
    the pool is identical across variants; the stoch/PGD mask draws are seeded
    per batch via fork_rng, so those are identical across variants too. Reports the
    port's three method terms + faithfulness (raw and weighted at the port's default
    coefficients) plus per-matrix breakdowns: importance per matrix, and stochastic
    masked recon with the subset routing FORCED to a single matrix (the other runs
    the frozen target weights)."""
    model = pd_v.model
    pd_v.requires_grad_(False)  # eval-only; PGD's adversary is an independent leaf
    ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                      seed=seed)
    pnorm, eps = loss_kw["pnorm"], 1e-12
    acc: dict[str, float] = {}

    def add(key, val):
        acc[key] = acc.get(key, 0.0) + float(val)

    for i in range(n_batches):
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            target = model(x)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 100003 + i)
            terms = pd_v.method_loss_terms(
                x, target, alpha_importance=loss_kw["alpha_importance"],
                alpha_stoch=loss_kw["alpha_stoch"], alpha_pgd=loss_kw["alpha_pgd"],
                pnorm=pnorm)
            terms["faithful"] = (pd_v.faithfulness_loss(), loss_kw["alpha_faithfulness"])
            _, logs = pd_v._total_and_logs(terms)
            for k, v in logs.items():
                add(k, v)
            with torch.no_grad():  # per-matrix breakdowns (fresh seeded mask draws)
                e = model.embed_input(x)
                h = torch.relu(e @ pd_v.targets["w_in"])
                lower, upper = pd_v._ci(e, h)
                B = x.shape[0]
                for name in pd_v.param_names:
                    add(f"importance_{name}",
                        ((upper[name] + eps) ** pnorm).mean(dim=0).sum())
                    masks = {n: ci + (1 - ci) * torch.rand_like(ci)
                             for n, ci in lower.items()}
                    dmasks = {n: torch.rand(B, device=x.device) for n in lower}
                    routing = {n: (torch.ones(B, device=x.device) if n == name
                                   else torch.zeros(B, device=x.device)) for n in lower}
                    out = pd_v._forward_masked_vpd(x, masks, dmasks, routing)
                    add(f"stoch_recon_{name}", ((out - target) ** 2).mean())

    means = {k: v / n_batches for k, v in acc.items()}
    res = {"total": means.pop("total"),
           "terms": {}, "per_matrix": {"importance": {}, "stoch_recon": {}}}
    for term, wkey in TERM_WEIGHT_KEY.items():
        res["terms"][term] = {"raw": means[term],
                              "weighted": means[term] * loss_kw[wkey]}
    for name in pd_v.param_names:
        res["per_matrix"]["importance"][name] = means[f"importance_{name}"]
        res["per_matrix"]["stoch_recon"][name] = means[f"stoch_recon_{name}"]
    return res


def basis_metrics(pd_v: VPD, p: float) -> dict:
    """Reused component_match scoring: per-side activation/mass purity under the
    Hungarian activation match, neuron-basis stats, and VPD-native ci_stats."""
    mats = activation_matrices(pd_v)
    mats.update(mass_matrices(pd_v))
    match_in = hungarian_match(mats["act_in"])
    match_out = hungarian_match(mats["act_out"])
    s = score_sides(mats, match_in, match_out, pd_v.model.n_features)
    return {"in": s["in"], "out": s["out"], "index_agreement": s["index_agreement"],
            "dead_components": s["dead_components"],
            "neuron_basis": neuron_basis_stats(pd_v),
            "ci_stats": pd_v.ci_stats(float(p))}


# ------------------------------------------------- sweep basis-ontology metrics

def _stats(t: torch.Tensor) -> dict:
    """min/median/mean/max of a 1-D tensor (empty -> NaNs)."""
    if t.numel() == 0:
        return {k: float("nan") for k in ("min", "median", "mean", "max")}
    return {"min": float(t.min()), "median": float(t.median()),
            "mean": float(t.mean()), "max": float(t.max())}


def sweep_basis_metrics(pd_v: VPD) -> dict:
    """Which basis did the sweep cell land in? All statistics are over ALIVE
    components (Frobenius mass ||V[:,c]||*||U[c,:]|| > SWEEP_ALIVE_FRAC * side max).

      read.feature_conc      -- per-component max|V_in|/||V_in|| (MLP: identity embed,
                                V_in rows ARE features; ExactPD read piece = e_j -> 1.0)
      read.feature_coverage  -- distinct feature argmaxes among alive read pieces
      write_neuron           -- same on V_out columns against the NEURON axis
                                (the 2026-07-30 neuron-basis finding: conc ~1, coverage ~d_mlp)
      write_feature          -- alignment against the ExactPD write direction:
                                u_row_feature_conc = max|U_out[c,:]|/||U_out[c,:]||
                                (ExactPD: U row = e_j -> 1.0), vcol_wout_cos = best
                                |cos| of V_out[:,c] vs the w_out COLUMN basis
                                (ExactPD: V col = w_out[:,j] -> 1.0), plus distinct
                                feature-argmax coverage of the U rows
      verdict                -- neuron_basis / exact_basis / collapsed / mixed via
                                VERDICT_THRESHOLDS (stored alongside)."""
    model = pd_v.model
    n_feat, d_mlp = model.n_features, model.d_mlp
    eps = 1e-12
    with torch.no_grad():
        alive = {}
        for name in pd_v.param_names:
            key = pd_v._key(name)
            mass = pd_v.V[key].norm(dim=0) * pd_v.U[key].norm(dim=1)
            alive[name] = mass > SWEEP_ALIVE_FRAC * mass.max().clamp_min(eps)

        # read side: V_in columns in feature space
        vin = pd_v.V[pd_v._key("w_in")][:, alive["w_in"]]           # (F, A_in)
        rconc = vin.abs().max(dim=0).values / vin.norm(dim=0).clamp_min(eps)
        read = {"alive": int(alive["w_in"].sum()), "feature_conc": _stats(rconc),
                "feature_coverage": len(set(vin.abs().argmax(dim=0).tolist()))
                if vin.numel() else 0}

        # write side vs the NEURON basis: V_out columns in neuron space
        vout = pd_v.V[pd_v._key("w_out")][:, alive["w_out"]]        # (N, A_out)
        nconc = vout.abs().max(dim=0).values / vout.norm(dim=0).clamp_min(eps)
        write_neuron = {"alive": int(alive["w_out"].sum()), "neuron_conc": _stats(nconc),
                        "neuron_coverage": len(set(vout.abs().argmax(dim=0).tolist()))
                        if vout.numel() else 0}

        # write side vs the ExactPD (w_out COLUMN) basis
        urows = pd_v.U[pd_v._key("w_out")][alive["w_out"]]          # (A_out, F)
        uconc = urows.abs().max(dim=1).values / urows.norm(dim=1).clamp_min(eps)
        w_cols = pd_v.targets["w_out"]
        w_cols = w_cols / w_cols.norm(dim=0, keepdim=True).clamp_min(eps)   # (N, F)
        v_hat = vout / vout.norm(dim=0, keepdim=True).clamp_min(eps)
        cos = (w_cols.T @ v_hat).abs() if vout.numel() else torch.zeros(n_feat, 0)
        write_feature = {"u_row_feature_conc": _stats(uconc),
                         "vcol_wout_cos": _stats(cos.max(dim=0).values
                                                 if cos.numel() else cos.flatten()),
                         "feature_coverage": len(set(urows.abs().argmax(dim=1).tolist()))
                         if urows.numel() else 0}

    th = VERDICT_THRESHOLDS
    if (write_neuron["alive"] < th["collapsed_alive_out_frac_dmlp"] * d_mlp
            or read["alive"] < th["collapsed_alive_in_frac_F"] * n_feat):
        verdict = "collapsed"
    elif (write_feature["u_row_feature_conc"]["mean"] > th["conc_hi"]
          and write_feature["vcol_wout_cos"]["mean"] > th["conc_hi"]
          and write_feature["feature_coverage"] >= th["coverage_frac"] * n_feat
          and read["feature_conc"]["mean"] > th["conc_hi"]
          and read["feature_coverage"] >= th["coverage_frac"] * n_feat):
        verdict = "exact_basis"
    elif (write_neuron["neuron_conc"]["mean"] > th["conc_hi"]
          and write_neuron["neuron_coverage"] >= th["coverage_frac"] * d_mlp
          and read["feature_conc"]["mean"] > th["conc_hi"]):
        verdict = "neuron_basis"
    else:
        verdict = "mixed"
    return {"F": n_feat, "d_mlp": d_mlp, "alive_frac": SWEEP_ALIVE_FRAC,
            "read": read, "write_neuron": write_neuron, "write_feature": write_feature,
            "verdict": verdict, "verdict_thresholds": dict(th)}


# ------------------------------------------- endpoint-vs-learned recovery (B vs C)

RECOVERY_NOTE = ("learned_reference (C) came from a DIFFERENT init/seed, so 'recovers C' "
                 "means same basin/structure (matched rank-1 directions), not identical "
                 "tensors; criteria are heuristics, not gospel.")


def _alive_mask(pd_v: VPD, name: str) -> torch.Tensor:
    """Component Frobenius mass > SWEEP_ALIVE_FRAC * side max (as sweep_basis_metrics)."""
    key = pd_v._key(name)
    mass = pd_v.V[key].norm(dim=0) * pd_v.U[key].norm(dim=1)
    return mass > SWEEP_ALIVE_FRAC * mass.max().clamp_min(1e-12)


def _piece_cosines(pd_b: VPD, pd_c: VPD, name: str) -> torch.Tensor:
    """(C_b, C_c) |cos| between flattened rank-1 pieces of matrix `name`:
    cos(outer(vb,ub), outer(vc,uc)) = cos(vb,vc) * cos(ub,uc)."""
    eps = 1e-12
    kb, kc = pd_b._key(name), pd_c._key(name)
    vb = pd_b.V[kb] / pd_b.V[kb].norm(dim=0, keepdim=True).clamp_min(eps)
    vc = pd_c.V[kc] / pd_c.V[kc].norm(dim=0, keepdim=True).clamp_min(eps)
    ub = pd_b.U[kb] / pd_b.U[kb].norm(dim=1, keepdim=True).clamp_min(eps)
    uc = pd_c.U[kc] / pd_c.U[kc].norm(dim=1, keepdim=True).clamp_min(eps)
    return ((vb.T @ vc) * (ub @ uc.T)).abs()


def recovery_vs_learned(pd_b: VPD, pd_c: VPD) -> dict:
    """Component-level recovery of C's structure by the B endpoint: Hungarian match on
    the rank-1-piece |cos| matrix per matrix (read pieces = w_in, write pieces = w_out),
    matched-cosine stats over ALL pairs and over pairs alive on both sides, alive
    counts, and per-neuron write-slot agreement (argmax|V_out| of matched alive pairs)."""
    out = {"note": RECOVERY_NOTE, "alive_frac": SWEEP_ALIVE_FRAC}
    with torch.no_grad():
        for name in pd_b.param_names:
            cos = _piece_cosines(pd_b, pd_c, name)                    # (C_b, C_c)
            match = hungarian_match(cos.detach().cpu().numpy())       # {c_C -> c_B}
            alive_b, alive_c = _alive_mask(pd_b, name), _alive_mask(pd_c, name)
            pairs = [(match[j], j) for j in range(pd_c.C)]
            vals = torch.tensor([float(cos[b, c]) for b, c in pairs])
            both = torch.tensor([bool(alive_b[b] and alive_c[c]) for b, c in pairs])
            out[name] = {"matched_cos_all": _stats(vals),
                         "matched_cos_alive_both": _stats(vals[both]),
                         "alive_b": int(alive_b.sum()), "alive_c": int(alive_c.sum()),
                         "n_alive_both_pairs": int(both.sum())}
            if name == "w_out":  # per-neuron write-slot agreement over alive pairs
                nb = pd_b.V[pd_b._key(name)].abs().argmax(dim=0)
                nc = pd_c.V[pd_c._key(name)].abs().argmax(dim=0)
                agree = [int(nb[b]) == int(nc[c])
                         for (b, c), ok in zip(pairs, both.tolist()) if ok]
                out[name]["neuron_slot_agreement"] = (
                    float(sum(agree)) / max(len(agree), 1))
                out[name]["neuron_slots_b"] = len(
                    {int(n) for n, ok in zip(nb.tolist(), alive_b.tolist()) if ok})
                out[name]["neuron_slots_c"] = len(
                    {int(n) for n, ok in zip(nc.tolist(), alive_c.tolist()) if ok})
    return out


def recovery_criteria(total_b: float, total_c: float, sweep_b: dict, rec: dict) -> dict:
    """Heuristic pass/fail (stated verbatim in the record): total within ~15% of C;
    write pieces neuron-aligned (conc > 0.8, >= 45 neurons covered); matched write-piece
    |cos| median > 0.8 over pairs alive on both sides."""
    wn = sweep_b["write_neuron"]
    med = rec["w_out"]["matched_cos_alive_both"]["median"]
    return {
        "total_within_15pct_of_C": {"pass": bool(total_b <= 1.15 * total_c),
                                    "ratio_B_over_C": total_b / total_c},
        "write_neuron_aligned": {"pass": bool(wn["neuron_conc"]["mean"] > 0.8
                                              and wn["neuron_coverage"] >= 45),
                                 "neuron_conc_mean": wn["neuron_conc"]["mean"],
                                 "neuron_coverage": wn["neuron_coverage"]},
        "matched_write_cos_median": {"pass": bool(med > 0.8), "median": med},
        "note": RECOVERY_NOTE}


def ci_probe_comparison(pd_b: VPD, pd_c: VPD) -> dict:
    """Gates-only comparison on single-feature probes x = e_j (j = 0..F-1, value 1.0):
    per matrix, each component's CI pattern is the F-vector of its lower-CI over the
    probes; Hungarian match on pattern |cos| between B's and C's components, matched
    cosine stats (over pairs where both patterns are nonzero), and the mean Jaccard of
    the thresholded (ci > CI_ALIVE_THRESHOLD) patterns over those matched pairs."""
    from core.vpd import CI_ALIVE_THRESHOLD
    model = pd_b.model
    eps = 1e-12
    X = torch.eye(model.n_features, device=model.device)
    out = {"probe": "single-feature identity batch, value 1.0",
           "jaccard_threshold": CI_ALIVE_THRESHOLD}
    with torch.no_grad():
        e = model.embed_input(X)
        h = torch.relu(e @ pd_b.targets["w_in"])
        lb, _ = pd_b._ci(e, h)
        lc, _ = pd_c._ci(e, h)
        for name in pd_b.param_names:
            A, Bm = lb[name], lc[name]                                # (F, C)
            na, nb_ = A.norm(dim=0), Bm.norm(dim=0)
            cos = ((A / na.clamp_min(eps)).T @ (Bm / nb_.clamp_min(eps))).abs()
            match = hungarian_match(cos.detach().cpu().numpy())       # {c_C -> c_B}
            pairs = [(match[j], j) for j in range(pd_c.C)]
            ok = torch.tensor([bool(na[b] > eps and nb_[c] > eps) for b, c in pairs])
            vals = torch.tensor([float(cos[b, c]) for b, c in pairs])
            Ab, Bb = A > CI_ALIVE_THRESHOLD, Bm > CI_ALIVE_THRESHOLD
            jac = torch.tensor(
                [float((Ab[:, b] & Bb[:, c]).sum()) /
                 max(float((Ab[:, b] | Bb[:, c]).sum()), 1.0) for b, c in pairs])
            out[name] = {"matched_cos_nonzero_both": _stats(vals[ok]),
                         "n_nonzero_both_pairs": int(ok.sum()),
                         "jaccard_mean_nonzero_both": float(jac[ok].mean())
                         if bool(ok.any()) else float("nan")}
    return out


# ---------------------------------------------------------------- storage

def save_run(tag: str, record: dict, pd_v: VPD) -> None:
    """Append-only: one JSON per run tag + the component/CI tensors as a .pt payload
    (pd_state is load_state_dict(strict=False)-compatible, model excluded)."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
    state = {k: v.cpu() for k, v in pd_v.state_dict().items() if not k.startswith("model.")}
    torch.save({"tag": tag, "method": type(pd_v).__name__, "C": pd_v.C,
                "pd_state": state, "config": record["config"]}, ckpt)
    record["checkpoint"] = os.path.relpath(ckpt, RESULTS + "/..")
    with open(os.path.join(RUNS_DIR, f"{tag}.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"  saved {os.path.join(RUNS_DIR, tag + '.json')}  +  {ckpt}")
    write_index()


def write_index() -> None:
    """Regenerate results/vpd_objective/index.md as one table over all run JSONs."""
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.json")))
    lines = ["# VPD-objective runs (Experiment 1: does the truth cost more?)", "",
             "Fixed eval pool: CleanDataset p, seed/batches/batch as per each run's "
             "config; term weights = the port's defaults. Raw term values; `total` "
             "is the weighted objective.", "",
             "| tag | timestamp | steps | total | stoch | pgd | imp | faithful "
             "| stoch w_in | stoch w_out | imp w_in | imp w_out "
             "| act_in | act_out | ci_l0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        lo, pm, bm = r["losses"], r["losses"]["per_matrix"], r.get("basis_metrics", {})
        t = lo["terms"]
        ci_l0 = bm.get("ci_stats", {}).get("headline", {}).get("ci_l0", float("nan"))
        lines.append(
            f"| {r['tag']} | {r['timestamp']} | {r['steps']} | {lo['total']:.3e} "
            f"| {t['stoch_recon']['raw']:.3e} | {t['pgd_recon']['raw']:.3e} "
            f"| {t['importance']['raw']:.3e} | {t['faithful']['raw']:.3e} "
            f"| {pm['stoch_recon']['w_in']:.3e} | {pm['stoch_recon']['w_out']:.3e} "
            f"| {pm['importance']['w_in']:.3e} | {pm['importance']['w_out']:.3e} "
            f"| {bm.get('in', {}).get('act_purity', float('nan')):.2f} "
            f"| {bm.get('out', {}).get('act_purity', float('nan')):.2f} "
            f"| {ci_l0:.2f} |")
    os.makedirs(RES_DIR, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  index ({len(paths)} runs) -> {INDEX_PATH}")


def print_comparison(records: list[dict]) -> None:
    """Compact per-term side-by-side table for the runs just produced."""
    tags = [r["tag"] for r in records]
    rows = [("total (weighted)", lambda lo: lo["total"])]
    rows += [(f"{t} raw", lambda lo, t=t: lo["terms"][t]["raw"])
             for t in ("stoch_recon", "pgd_recon", "importance", "faithful")]
    rows += [(f"stoch {n}", lambda lo, n=n: lo["per_matrix"]["stoch_recon"][n])
             for n in ("w_in", "w_out")]
    rows += [(f"imp {n}", lambda lo, n=n: lo["per_matrix"]["importance"][n])
             for n in ("w_in", "w_out")]
    w = max(len(name) for name, _ in rows)
    print("\n" + " " * (w + 2) + "  ".join(f"{t:>18}" for t in tags))
    for name, fn in rows:
        print(f"  {name:<{w}}" + "  ".join(f"{fn(r['losses']):>18.4e}" for r in records))


# ---------------------------------------------------------------- phase figure

VERDICT_COLORS = {  # Okabe-Ito (CVD-safe); verdict is ALSO position-encoded on y
    "neuron_basis": "#0072B2", "exact_basis": "#009E73",
    "collapsed": "#D55E00", "mixed": "#999999"}
SERIES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # fixed order, never cycled
MARKERS = ["o", "s", "^", "D"]


def phase_figure() -> None:
    """Render figures/vpd_objective/phase_diagram.png from all sweep_* run JSONs:
    x = importance multiplier (log); panels for verdict, alive counts, read/write
    concentrations vs the ExactPD and neuron bases, and the eval recon terms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "sweep_*.json"))):
        with open(path) as f:
            r = json.load(f)
        sb = r.get("sweep_basis")
        if sb is None:
            continue
        rows.append({
            "mult": float(r["config"]["imp_mult"]), "seed": int(r["config"]["fit_seed"]),
            "steps": int(r["steps"]), "verdict": sb["verdict"],
            "F": sb["F"], "d_mlp": sb["d_mlp"],
            "alive_in": sb["read"]["alive"], "alive_out": sb["write_neuron"]["alive"],
            "read_conc": sb["read"]["feature_conc"]["mean"],
            "read_cov": sb["read"]["feature_coverage"] / sb["F"],
            "wn_conc": sb["write_neuron"]["neuron_conc"]["mean"],
            "wn_cov": sb["write_neuron"]["neuron_coverage"] / sb["d_mlp"],
            "wf_uconc": sb["write_feature"]["u_row_feature_conc"]["mean"],
            "wf_cos": sb["write_feature"]["vcol_wout_cos"]["mean"],
            "wf_cov": sb["write_feature"]["feature_coverage"] / sb["F"],
            "stoch": r["losses"]["terms"]["stoch_recon"]["raw"],
            "pgd": r["losses"]["terms"]["pgd_recon"]["raw"]})
    if not rows:
        print(f"  no sweep_* runs under {RUNS_DIR}; nothing to render")
        return
    rows.sort(key=lambda r: (r["mult"], r["seed"]))
    x = np.array([r["mult"] for r in rows])
    F_ref, N_ref = rows[0]["F"], rows[0]["d_mlp"]

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    def series(ax, keys_labels, hline=None):
        for i, (key, lbl) in enumerate(keys_labels):
            # odd series get OPEN markers so coincident points remain visible
            ax.plot(x, col(key), MARKERS[i] + "-", color=SERIES[i], ms=6, lw=1.5,
                    mfc=SERIES[i] if i % 2 == 0 else "none", mew=1.5, label=lbl)
        if hline:
            for y, lbl, c in hline:
                ax.axhline(y, color=c, lw=1.0, ls=":")
                ax.annotate(lbl, (x.min(), y), fontsize=7, color=c,
                            xytext=(2, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2), constrained_layout=True)

    ax = axes[0][0]  # verdict: categorical y + fixed colors (double-encoded)
    order = ["collapsed", "mixed", "neuron_basis", "exact_basis"]
    ypos = {v: i for i, v in enumerate(order)}
    for r in rows:
        ax.scatter(r["mult"], ypos[r["verdict"]], s=70,
                   color=VERDICT_COLORS[r["verdict"]], zorder=3)
        ax.annotate(f"s{r['seed']}", (r["mult"], ypos[r["verdict"]]), fontsize=6,
                    xytext=(4, 5), textcoords="offset points", color="#555555")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("ontology verdict (thresholded)", fontsize=10)

    ax = axes[0][1]
    series(ax, [("alive_in", "alive read pieces (w_in)"),
                ("alive_out", "alive write pieces (w_out)")],
           hline=[(F_ref, f"F={F_ref} (exact basis)", "#009E73"),
                  (N_ref, f"d_mlp={N_ref} (neuron basis)", "#0072B2")])
    ax.set_ylim(0, max(F_ref, col("alive_in").max(), col("alive_out").max()) * 1.12)
    ax.set_title(f"alive components (mass > {SWEEP_ALIVE_FRAC:.0%} of side max)",
                 fontsize=10)

    ax = axes[0][2]
    series(ax, [("read_conc", "feature conc. max|V_in|/‖V_in‖ (mean)"),
                ("read_cov", "distinct-feature coverage / F")])
    ax.set_ylim(0, 1.05)
    ax.set_title("read side vs feature basis (ExactPD read = 1.0)", fontsize=10)

    ax = axes[1][0]
    series(ax, [("wn_conc", "neuron conc. max|V_out|/‖V_out‖ (mean)"),
                ("wn_cov", "distinct-neuron coverage / d_mlp")])
    ax.set_ylim(0, 1.05)
    ax.set_title("write side vs NEURON basis", fontsize=10)

    ax = axes[1][1]
    series(ax, [("wf_uconc", "U-row feature conc. (mean)"),
                ("wf_cos", "best |cos|(V_out, w_out col) (mean)"),
                ("wf_cov", "distinct-feature coverage / F")])
    ax.set_ylim(0, 1.05)
    ax.set_title("write side vs ExactPD (w_out column) basis", fontsize=10)

    ax = axes[1][2]
    series(ax, [("stoch", "stoch recon (raw)"), ("pgd", "PGD recon (raw)")])
    ax.set_yscale("log")
    ax.set_title("eval recon terms on the fixed pool", fontsize=10)

    for ax in axes.flat:
        ax.set_xlabel("importance-weight multiplier R (× 1e-4)", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("VPD pricing sweep: importance-minimality weight vs learned basis "
                 f"(K=5 transpose, p={P_DEFAULT}; {len(rows)} cells)", fontsize=12)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(PHASE_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  phase figure ({len(rows)} cells) -> {PHASE_FIG}")


# ---------------------------------------------------------------- runners

def _base_config(code: CodeAnsatz, ab, p: str, a) -> dict:
    return {"K": code.K, "family": code.family, "code_seed": code.seed,
            "swaps": code.swaps, "decoder": code.decoder, "p": p,
            "ab": [float(v) for v in ab], "C": F, "loss_kwargs": DEFAULT_KW,
            "eval_pool": {"seed": a.seed, "n_batches": a.batches, "batch": a.batch},
            "init_seed": INIT_SEED}


def run_variant(tag: str, pd_v: VPD, code: CodeAnsatz, ab, p: str, a,
                steps: int, extra_cfg: dict | None = None,
                extra_record: dict | None = None) -> dict:
    losses = eval_objective(pd_v, p=float(p), seed=a.seed, n_batches=a.batches,
                            batch=a.batch)
    record = {"tag": tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "config": _base_config(code, ab, p, a) | (extra_cfg or {}),
              "steps": steps, "losses": losses,
              "basis_metrics": basis_metrics(pd_v, float(p))} | (extra_record or {})
    save_run(tag, record, pd_v)
    return record


def run_sweep_cell(a, code: CodeAnsatz, model, ab, ab_source: str) -> dict:
    """Experiment 2 cell: fresh VPD fit with alpha_importance scaled by --imp-mult.
    Fit machinery mirrors biregular_vpd_compare.fit_or_load_vpd (build_pd + the
    port's fit loop at METHOD_LR), with the steps override and fit seed S; the
    EVALUATION uses the port's DEFAULT term weights on the standard fixed pool, so
    raw terms and totals are comparable across cells and against the variants."""
    R, S = a.imp_mult, a.fit_seed
    fit_kw = DEFAULT_KW | {"alpha_importance": DEFAULT_KW["alpha_importance"] * R}
    tag = f"sweep_imp{R:g}_s{S}"
    decomp_fit.set_seed(S)  # reproducible V/U + CI init (incl. the data-free warmup)
    pd_v = decomp_fit.build_pd("VPD", model, F)
    curves = decomp_fit.fit_decomposition(
        pd_v, CleanDataset(n_features=F, p=float(a.p), device=a.device, seed=S),
        steps=a.steps, loss_kwargs=fit_kw, seed=S, desc=f"{tag} p={a.p}")
    print(f"  {tag} fit done: total {curves['total'][0]:.3e} -> {curves['total'][-1]:.3e}")
    sweep = sweep_basis_metrics(pd_v)
    print(f"  verdict={sweep['verdict']}  alive(in/out)="
          f"{sweep['read']['alive']}/{sweep['write_neuron']['alive']}  "
          f"read_conc={sweep['read']['feature_conc']['mean']:.2f}  "
          f"write_nconc={sweep['write_neuron']['neuron_conc']['mean']:.2f}  "
          f"write_uconc={sweep['write_feature']['u_row_feature_conc']['mean']:.2f}")
    return run_variant(
        tag, pd_v, code, ab, a.p, a, steps=a.steps,
        extra_cfg={"ab_source": ab_source, "components": "learned_sweep",
                   "gates": "learned_ci", "imp_mult": R, "fit_seed": S,
                   "fit_loss_kwargs": fit_kw},
        extra_record={"sweep_basis": sweep})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=EVAL_SEED, help="eval-pool seed")
    ap.add_argument("--batches", type=int, default=EVAL_BATCHES)
    ap.add_argument("--batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--hybrid", action="store_true",
                    help="EVAL-ONLY write-basis ablation: exact reads + split-neuron "
                         "writes (all C live) + oracle gates -> hybrid_oracle; prints "
                         "a 3-way table against the SAVED exact_oracle and "
                         "learned_reference rows")
    ap.add_argument("--hybrid-full", action="store_true", dest="hybrid_full",
                    help="hybrid_fulltrain: init AT the hybrid configuration, train "
                         "EVERYTHING (components + fresh CI nets) with the standard "
                         "full recipe (--full-steps), eval + component-level recovery "
                         "vs learned_reference")
    ap.add_argument("--full-steps", type=int, default=25_000, dest="full_steps",
                    help="hybrid_fulltrain fit steps (25000 = the full reference recipe)")
    ap.add_argument("--gates-compare", action="store_true", dest="gates_compare",
                    help="post-hoc: compare hybrid_trainedci's learned CI behavior to "
                         "learned_reference's on single-feature probes (no training); "
                         "patches the hybrid_trainedci JSON with ci_probe_vs_learned")
    ap.add_argument("--fit-ci", action="store_true", dest="fit_ci",
                    help="DEFERRED variant (b): freeze exact components, train the CI "
                         "nets with the port's optimizer, then eval -> exact_trainedci")
    ap.add_argument("--ci-steps", type=int, default=CI_STEPS_DEFAULT, dest="ci_steps")
    ap.add_argument("--sweep-cell", action="store_true", dest="sweep_cell",
                    help="Experiment 2: fit a FRESH VPD with alpha_importance "
                         "scaled by --imp-mult, then eval on the standard pool")
    ap.add_argument("--imp-mult", type=float, default=None, dest="imp_mult",
                    help="importance-weight multiplier R (sweep tag sweep_imp{R}_s{S})")
    ap.add_argument("--steps", type=int, default=SWEEP_STEPS_DEFAULT,
                    help="sweep-cell fit steps (25000 = the full reference recipe)")
    ap.add_argument("--fit-seed", type=int, default=0, dest="fit_seed",
                    help="sweep-cell fit seed S (--seed stays the eval-pool seed)")
    ap.add_argument("--phase-figure", action="store_true", dest="phase_figure",
                    help="render figures/vpd_objective/phase_diagram.png from all "
                         "sweep_* runs (no fitting)")
    ap.add_argument("--index", action="store_true", help="rebuild index.md only")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.index:
        write_index()
        return
    if a.phase_figure:
        phase_figure()
        return

    code = CodeAnsatz(5, family="regular", seed=0, swaps=0, decoder="transpose")
    # (a, b) from the saved baseline payload when present (guarantees the SAME frozen
    # model the learned VPD was fit to, at zero refit cost; fit_ab is deterministic
    # and produced that ab in the first place). Fresh fit_ab only as a fallback.
    learned_path = _decomp_path(code.name, a.p, "")
    if os.path.exists(learned_path):
        payload = torch.load(learned_path, map_location=a.device, weights_only=False)
        ab, ab_source = tuple(payload["biregular"]["ab"]), "learned_payload"
    else:
        payload = None
        ab, ab_source = code.fit_ab(float(a.p), device=a.device), "fit_ab"
    model = code.model(ab, device=a.device)
    print(f"[{code.name}] p={a.p} ab=({ab[0]:.4f},{ab[1]:.4f}) [{ab_source}] "
          f"pool: seed={a.seed} {a.batches}x{a.batch}")

    if a.sweep_cell:  # Experiment 2: pricing-sweep cell
        assert a.imp_mult is not None, "--sweep-cell requires --imp-mult R"
        rec = run_sweep_cell(a, code, model, ab, ab_source)
        print_comparison([rec])
        return

    if a.gates_compare:  # post-hoc gates-only probe: hybrid_trainedci CI vs learned CI
        ckpt = os.path.join(CKPT_DIR, "hybrid_trainedci.pt")
        assert os.path.exists(ckpt), f"no {ckpt}; run --hybrid --fit-ci first"
        pd_b = build_hybrid_vpd(model)
        pd_b.load_state_dict(torch.load(ckpt, map_location=a.device,
                                        weights_only=False)["pd_state"], strict=False)
        _, pd_l, _ = load_learned(code, a.p, a.device)
        probe = ci_probe_comparison(pd_b, pd_l)
        rec_path = os.path.join(RUNS_DIR, "hybrid_trainedci.json")
        with open(rec_path) as f:
            rec = json.load(f)
        rec["ci_probe_vs_learned"] = probe
        with open(rec_path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"  patched {rec_path} with ci_probe_vs_learned")
        for name in ("w_in", "w_out"):
            s = probe[name]
            print(f"  {name}: matched-CI-pattern |cos| (both nonzero, "
                  f"n={s['n_nonzero_both_pairs']}) median {s['matched_cos_nonzero_both']['median']:.3f} "
                  f"mean {s['matched_cos_nonzero_both']['mean']:.3f} "
                  f"min {s['matched_cos_nonzero_both']['min']:.3f}  "
                  f"jaccard@{probe['jaccard_threshold']} {s['jaccard_mean_nonzero_both']:.3f}")
        return

    if a.hybrid_full:  # hybrid init, everything trainable, standard full recipe
        pd_v = build_hybrid_vpd(model)  # frozen pieces + fresh CI init ...
        for k in pd_v.V:                # ... then unfreeze the components
            pd_v.V[k].requires_grad_(True)
            pd_v.U[k].requires_grad_(True)
        n_train = sum(p_.numel() for p_ in pd_v.parameters() if p_.requires_grad)
        print(f"  hybrid_fulltrain: training everything ({n_train} params, "
              f"{a.full_steps} steps, lr {decomp_fit.METHOD_LR['VPD']})")
        decomp_fit.fit_decomposition(
            pd_v, CleanDataset(n_features=F, p=float(a.p), device=a.device,
                               seed=FIT_SEED),
            steps=a.full_steps, loss_kwargs=DEFAULT_KW, seed=FIT_SEED,
            desc=f"hybrid_fulltrain p={a.p}")
        sweep = sweep_basis_metrics(pd_v)
        print(f"  endpoint verdict={sweep['verdict']}  alive(in/out)="
              f"{sweep['read']['alive']}/{sweep['write_neuron']['alive']}")
        rec = run_variant("hybrid_fulltrain", pd_v, code, ab, a.p, a,
                          steps=a.full_steps,
                          extra_cfg={"ab_source": ab_source,
                                     "components": "hybrid_init_fulltrain",
                                     "gates": "learned_ci", "fit_seed": FIT_SEED},
                          extra_record={"sweep_basis": sweep})
        _, pd_l, _ = load_learned(code, a.p, a.device)
        with open(os.path.join(RUNS_DIR, "learned_reference.json")) as f:
            rec_c = json.load(f)
        recov = recovery_vs_learned(pd_v, pd_l)
        recov["criteria"] = recovery_criteria(rec["losses"]["total"],
                                              rec_c["losses"]["total"], sweep, recov)
        rec["recovery_vs_learned"] = recov
        with open(os.path.join(RUNS_DIR, "hybrid_fulltrain.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print("  recovery criteria: " + "  ".join(
            f"{k}={'PASS' if v['pass'] else 'FAIL'}"
            for k, v in recov["criteria"].items() if isinstance(v, dict)))
        print_comparison([rec, rec_c])
        return

    if a.hybrid and not a.fit_ci:  # write-basis ablation, oracle gates (eval only)
        pd_h = build_hybrid_vpd(model, oracle=True)
        rec_h = run_variant("hybrid_oracle", pd_h, code, ab, a.p, a, steps=0,
                            extra_cfg={"ab_source": ab_source,
                                       "components": "hybrid_exactread_neuronwrite_frozen",
                                       "gates": "oracle_activity_neuronwrite"})
        recs = []
        for tag in ("exact_oracle", "learned_reference"):  # saved rows, same pool
            path = os.path.join(RUNS_DIR, f"{tag}.json")
            assert os.path.exists(path), f"missing saved row {path}; run the default evals first"
            with open(path) as f:
                r = json.load(f)
            assert r["config"]["eval_pool"] == rec_h["config"]["eval_pool"] \
                and r["config"]["p"] == rec_h["config"]["p"], f"{tag}: eval pool mismatch"
            recs.append(r)
        print_comparison([recs[0], rec_h, recs[1]])
        return

    if a.fit_ci:  # variant (b): frozen components (exact, or hybrid with --hybrid),
        # CI nets trained with the port's standard optimizer
        if a.hybrid:
            tag, comp_cfg = "hybrid_trainedci", "hybrid_exactread_neuronwrite_frozen"
            pd_v = build_hybrid_vpd(model)
        else:
            tag, comp_cfg = "exact_trainedci", "exact_frozen"
            pd_v = build_exact_vpd(model)
        n_train = sum(p_.numel() for p_ in pd_v.parameters() if p_.requires_grad)
        print(f"  {tag}: training CI nets only ({n_train} params, "
              f"{a.ci_steps} steps, lr {decomp_fit.METHOD_LR['VPD']})")
        decomp_fit.fit_decomposition(
            pd_v, CleanDataset(n_features=F, p=float(a.p), device=a.device,
                               seed=FIT_SEED),
            steps=a.ci_steps, loss_kwargs=DEFAULT_KW, seed=FIT_SEED,
            desc=f"{tag} p={a.p}")
        rec = run_variant(tag, pd_v, code, ab, a.p, a, steps=a.ci_steps,
                          extra_cfg={"ab_source": ab_source, "components": comp_cfg,
                                     "gates": "trained_ci", "ci_fit_seed": FIT_SEED})
        print_comparison([rec])
        return

    # variant (a): exact components + oracle gates (eval only)
    pd_o = build_exact_vpd(model, oracle=True)
    rec_a = run_variant("exact_oracle", pd_o, code, ab, a.p, a, steps=0,
                        extra_cfg={"ab_source": ab_source, "components": "exact_frozen",
                                   "gates": "oracle_activity"})

    # variant (c): the converged learned fit, re-evaluated on the identical pool
    assert payload is not None, f"no saved learned VPD fit at {learned_path}"
    _, pd_l, payload = load_learned(code, a.p, a.device)
    rec_c = run_variant("learned_reference", pd_l, code, ab, a.p, a,
                        steps=int(payload["biregular"]["steps"]),
                        extra_cfg={"ab_source": ab_source, "components": "learned",
                                   "gates": "learned_ci",
                                   "source_checkpoint": os.path.relpath(
                                       learned_path, RESULTS + "/..")})
    print_comparison([rec_a, rec_c])


if __name__ == "__main__":
    main()
