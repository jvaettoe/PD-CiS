"""Component-recovery comparison of a learned PD against ExactPD.

MATCHING IS FUNCTIONAL (self-activation overlap), basis description is separate.
"Learned component k == ExactPD component j" is decided by whether shard k CARRIES
feature j's signal, not by whether its weights sit in ExactPD's basis: a leak-aligned
write component (write vector = feature j's whole output spectrum) has its Frobenius
mass spread over many columns and scores at CHANCE on weight-mass overlap while being
exactly feature j's write mechanism. So:

  act_in / act_out       -- THE MATCHING CRITERION. act_in[k,j] = normalized output at
                            j on probe e_j with w_in = component k only and w_out = the
                            TARGET weight (and vice versa for act_out). ~1 <=> shard k
                            alone carries feature j's signal on that side, in ANY basis.
                            Hungarian per side (VPD indexes components independently
                            per matrix; index k is unrelated across w_in/w_out).
  mass_in / mass_out      -- basis description: fraction of the shard's weight mass in
                            ExactPD's support (row j / column j). Purity reported UNDER
                            the activation match: act high + mass high = ExactPD basis;
                            act high + mass low = same mechanism, rotated basis (e.g.
                            leak-aligned); act low = genuinely distributed.
  ci_in / ci_out          -- functional (firing): VPD CI vs feature activity correlation.
  fingerprint / ablation  -- behavioural / causal probes of the matched mechanism PAIR
                            {w_in: match_in[j], w_out: match_out[j]}.
  leak_align              -- for the matched write shard: cosine of its off-target write
                            profile on e_j against the full model's leak spectrum
                            R[j, != j] -- distinguishes leak-aligned from ExactPD-basis
                            writes when act_out is high.

NB act purities are GAUGE-sensitive: a mechanism pair rescaled (lambda on the read
shard, 1/lambda on the write shard) is the same mechanism but shifts act_in/act_out
away from 1 in opposite directions. The Hungarian MATCH is robust to this (row-wise
monotone); the purity NUMBERS are not -- read them jointly across sides.

Prototype / validation CLI (saved decomps of the TRAINED models; biregular driver is
biregular_vpd_compare.py):

  ./.venv/bin/python component_match.py --run-id MLP_dembNA_dmlp50_p0.1_n4_VPD
  ./.venv/bin/python component_match.py --self-test          # ExactPD vs itself -> identity
  ./.venv/bin/python component_match.py --identity-control   # VPD on w_in=w_out=I (port test)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call

from data import CleanDataset
from pd import PD, ExactPD

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "comparison")
RES_DIR = os.path.join(os.path.dirname(__file__), "results", "overview")
THETA = 0.5          # coverage threshold (act values are s_j-normalized; mass in [0,1])
SPLIT_MASS = 0.25    # a feature is "split" if >1 components carry this much of it


# ---------------------------------------------------------------- matrices

def activation_matrices(pd_m: PD) -> dict[str, np.ndarray]:
    """{'act_in', 'act_out'}: (C, F) normalized self-activations -- THE matching
    criterion. act_in[k, j] = output_j(e_j; w_in = component k only, w_out = TARGET
    weight) / s_j, and symmetrically for act_out. The un-tested side uses the target
    weight itself (not the component sum), isolating the tested shard from the
    delta/reconstruction error. s_j = the model's own self-response diag (== ExactPD's
    s_j by on-target agreement)."""
    model = pd_m.model
    F = model.n_features
    X = torch.eye(F, device=model.device)
    rows_in, rows_out = [], []
    with torch.no_grad():
        s = model(X).diagonal().clone()
        s = torch.where(s.abs() > 1e-9, s, torch.ones_like(s))
        for k in range(pd_m.C):
            oh = torch.zeros(pd_m.C, device=model.device)
            oh[k] = 1.0
            p_in = {"w_in": pd_m.masked_weight("w_in", oh), "w_out": pd_m.targets["w_out"]}
            rows_in.append((functional_call(model, p_in, (X,)).diagonal() / s).cpu().numpy())
            p_out = {"w_in": pd_m.targets["w_in"], "w_out": pd_m.masked_weight("w_out", oh)}
            rows_out.append((functional_call(model, p_out, (X,)).diagonal() / s).cpu().numpy())
    return {"act_in": np.stack(rows_in), "act_out": np.stack(rows_out)}


def mass_matrices(pd_m: PD) -> dict[str, np.ndarray]:
    """{'mass_in', 'mass_out'}: (C, F) fraction of component k's Frobenius mass in
    feature j's ExactPD support (row j of the feature-space read-in / column j of the
    read-out). Rows sum to 1 (0 for dead components). BASIS description only -- a
    leak-aligned mechanism scores at chance here while act_out scores ~1."""
    model = pd_m.model
    m_in, m_out = [], []
    with torch.no_grad():
        for k in range(pd_m.C):
            w_in_k, w_out_k = pd_m.component_matrices(k)
            rin = model.project_in(w_in_k)
            rout = model.project_out(w_out_k)
            row_mass = rin.pow(2).sum(1)
            col_mass = rout.pow(2).sum(0)
            m_in.append((row_mass / row_mass.sum().clamp_min(1e-12)).cpu().numpy())
            m_out.append((col_mass / col_mass.sum().clamp_min(1e-12)).cpu().numpy())
    return {"mass_in": np.stack(m_in), "mass_out": np.stack(m_out)}


def ci_coactivation(pd_v: PD, p: float, batch: int = 4096, seed: int = 0,
                    target_actives: int = 50, max_total: int = 262144) -> dict:
    """{'ci_in', 'ci_out'}: (C, F) Pearson correlation of the VPD CI (lower mask
    source, as in vpd.ci_stats) with feature activity over a CleanDataset(p) batch.
    Batch is scaled to ~target_actives active samples per feature (a Pearson
    correlation from ~4 positives, the old fixed batch at p=0.001, is noise).
    Empty dict for methods without a CI function (only VPD has one)."""
    if not hasattr(pd_v, "_ci"):
        return {}
    model = pd_v.model
    n = int(min(max_total, max(batch, round(target_actives / float(p)))))
    ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                      seed=seed)
    x, _ = ds.generate_batch(n)
    with torch.no_grad():
        e = model.embed_input(x)
        h = torch.relu(e @ pd_v.targets["w_in"])
        lower, _ = pd_v._ci(e, h)
        act = (x > 0).float()
        za = (act - act.mean(0)) / act.std(0).clamp_min(1e-12)
        out = {}
        for name, tag in (("w_in", "ci_in"), ("w_out", "ci_out")):
            ci = lower[name]
            zc = (ci - ci.mean(0)) / ci.std(0).clamp_min(1e-12)
            corr = zc.T @ za / x.shape[0]
            corr[ci.std(0) < 1e-8] = 0.0
            out[tag] = corr.cpu().numpy()
    return out


def _pair_mask(pd_m: PD, k_in: int, k_out: int) -> dict[str, torch.Tensor]:
    """Batch-shared masks selecting w_in component k_in and w_out component k_out --
    the mechanism PAIR (per-matrix indexing)."""
    dev = pd_m.model.device
    m_in = torch.zeros(pd_m.C, device=dev); m_in[k_in] = 1.0
    m_out = torch.zeros(pd_m.C, device=dev); m_out[k_out] = 1.0
    return {"w_in": m_in, "w_out": m_out}


def paired_fingerprint_matrix(pd_m: PD, pd_e: PD, match_in: dict, match_out: dict) -> np.ndarray:
    """(F, C_e) SIGNED cosine between the one-hot response fingerprint of feature j's
    matched mechanism pair (row j) and each ExactPD component's fingerprint."""
    F = pd_m.model.n_features
    X = torch.eye(F, device=pd_m.model.device)
    rows_m, rows_e = [], []
    with torch.no_grad():
        for j in range(F):
            rows_m.append(pd_m.forward_masked(
                X, _pair_mask(pd_m, match_in[j], match_out[j])).flatten())
        for c in range(pd_e.C):
            oh = torch.zeros(pd_e.C, device=X.device); oh[c] = 1.0
            rows_e.append(pd_e.forward_masked(X, {n: oh for n in pd_e.param_names}).flatten())
    fm = torch.stack(rows_m); fe = torch.stack(rows_e)
    fm = fm / fm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    fe = fe / fe.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (fm @ fe.T).cpu().numpy()


def _ablation_matrix(pd_m: PD, p: float, removal_masks: list[dict],
                     batch: int = 8192, seed: int = 7, target_actives: int = 50,
                     max_total: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    """((F, F), (F,)): row j = increase in each feature's active-conditional squared
    error under removal_masks[j] (a {w_in, w_out} dict of KEEP-masks with the target
    unit removed); second array = #active samples per feature. Baseline = all
    components active on both matrices.

    The error is ACTIVE-conditional, so the sample count is drawn to give every
    feature ~target_actives active examples (total = target_actives/p, chunked,
    capped at max_total) -- a single fixed batch is hopeless at small p (2048 samples
    at p=0.001 = ~2 actives/feature, ~13% of features with none). Features that still
    end with zero actives must be EXCLUDED from any agreement score (their row is
    zeros; argmax on it is noise)."""
    model = pd_m.model
    F = model.n_features
    total = int(min(max_total, max(batch, round(target_actives / float(p)))))
    ds = CleanDataset(n_features=F, p=float(p), device=model.device, seed=seed)
    ones = torch.ones(pd_m.C, device=model.device)
    err_abl = torch.zeros(F, F, device=model.device)   # [j, i]: sum err_i, unit j removed
    err_base = torch.zeros(F, device=model.device)
    cnt = torch.zeros(F, device=model.device)
    with torch.no_grad():
        done = 0
        while done < total:
            b = min(batch, total - done)
            x, _ = ds.generate_batch(b)
            active = (x > 0).float()
            cnt += active.sum(0)
            y = model(x)
            base = pd_m.forward_masked(x, {n: ones for n in pd_m.param_names})
            err_base += (((base - y) ** 2) * active).sum(0)
            for j in range(F):
                out = pd_m.forward_masked(x, removal_masks[j])
                err_abl[j] += (((out - y) ** 2) * active).sum(0)
            done += b
    denom = cnt.clamp_min(1)
    mat = (err_abl - err_base.unsqueeze(0)) / denom.unsqueeze(0)
    return mat.cpu().numpy(), cnt.cpu().numpy()


def paired_ablation_matrix(pd_m: PD, p: float, match_in: dict, match_out: dict,
                           **kw) -> tuple[np.ndarray, np.ndarray]:
    """Ablation of the matched PAIR (one read + one write shard) -- the per-matrix
    basis-level probe."""
    F = pd_m.model.n_features
    ones = torch.ones(pd_m.C, device=pd_m.model.device)
    masks = []
    for j in range(F):
        m_in = ones.clone(); m_in[match_in[j]] = 0.0
        m_out = ones.clone(); m_out[match_out[j]] = 0.0
        masks.append({"w_in": m_in, "w_out": m_out})
    return _ablation_matrix(pd_m, p, masks, **kw)


def circuit_ablation_matrix(pd_m: PD, p: float, match_in: dict, match_out: dict,
                            ao: tuple | None = None, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Ablation of feature j's full CIRCUIT (read shard + the union of its active
    neurons' write shards). NB write shards are SHARED (each serves ~d features), so
    circuit removal is expected to spill onto the code-neighbours -- the spillover
    pattern is the overlap structure of the code, not a probe defect."""
    F = pd_m.model.n_features
    ones = torch.ones(pd_m.C, device=pd_m.model.device)
    masks = []
    for j in range(F):
        cm, _ = circuit_mask(pd_m, j, match_in, match_out, ao=ao)
        masks.append({"w_in": ones - cm["w_in"], "w_out": ones - cm["w_out"]})
    return _ablation_matrix(pd_m, p, masks, **kw)


def circuit_fingerprint_matrix(pd_m: PD, pd_e: PD, match_in: dict, match_out: dict,
                               ao: tuple | None = None) -> np.ndarray:
    """(F, C_e) SIGNED cosine between the one-hot response fingerprint of feature j's
    CIRCUIT (row j) and each ExactPD component's fingerprint -- the mechanism-level
    counterpart of paired_fingerprint_matrix."""
    F = pd_m.model.n_features
    X = torch.eye(F, device=pd_m.model.device)
    rows_m, rows_e = [], []
    with torch.no_grad():
        for j in range(F):
            cm, _ = circuit_mask(pd_m, j, match_in, match_out, ao=ao)
            rows_m.append(pd_m.forward_masked(X, cm).flatten())
        for c in range(pd_e.C):
            oh = torch.zeros(pd_e.C, device=X.device); oh[c] = 1.0
            rows_e.append(pd_e.forward_masked(X, {n: oh for n in pd_e.param_names}).flatten())
    fm = torch.stack(rows_m); fe = torch.stack(rows_e)
    fm = fm / fm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    fe = fe / fe.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (fm @ fe.T).cpu().numpy()


def circuit_solo_responses(pd_m: PD, pd_e: PD, match_in: dict, match_out: dict,
                           ao: tuple | None = None) -> dict:
    """Self-activation of feature j's circuit on e_j vs ExactPD's own s_j, plus the
    circuit's write-shard count: {'solo', 'exact', 'n_shards'} (each (F,))."""
    F = pd_m.model.n_features
    X = torch.eye(F, device=pd_m.model.device)
    solo, exact, n_shards = [], [], []
    with torch.no_grad():
        for j in range(F):
            cm, n = circuit_mask(pd_m, j, match_in, match_out, ao=ao)
            solo.append(float(pd_m.forward_masked(X[j:j + 1], cm)[0, j]))
            n_shards.append(n)
            oh = torch.zeros(pd_e.C, device=X.device); oh[j] = 1.0
            exact.append(float(pd_e.forward_masked(X[j:j + 1],
                                                   {nm: oh for nm in pd_e.param_names})[0, j]))
    return {"solo": np.array(solo), "exact": np.array(exact),
            "n_shards": np.array(n_shards)}


def solo_responses(pd_m: PD, pd_e: PD, match_in: dict, match_out: dict) -> dict:
    """Per-feature self-activation of the matched mechanism pair on e_j, alongside
    ExactPD's own s_j: {'solo': (F,), 'exact': (F,)}."""
    F = pd_m.model.n_features
    X = torch.eye(F, device=pd_m.model.device)
    solo, exact = [], []
    with torch.no_grad():
        for j in range(F):
            out = pd_m.forward_masked(X[j:j + 1], _pair_mask(pd_m, match_in[j], match_out[j]))
            solo.append(float(out[0, j]))
            oh = torch.zeros(pd_e.C, device=X.device); oh[j] = 1.0
            oute = pd_e.forward_masked(X[j:j + 1], {n: oh for n in pd_e.param_names})
            exact.append(float(oute[0, j]))
    return {"solo": np.array(solo), "exact": np.array(exact)}


def leak_alignment(pd_m: PD, match_out: dict) -> np.ndarray:
    """(F,) cosine between the matched write shard's off-target profile on e_j
    (w_in = target, w_out = shard match_out[j] only) and the full model's leak
    spectrum R[j, != j]. High values with LOW mass_out purity = the shard is feature
    j's write mechanism in the leak-aligned basis rather than ExactPD's column basis."""
    model = pd_m.model
    F = model.n_features
    X = torch.eye(F, device=model.device)
    out = []
    with torch.no_grad():
        R = model(X)
        for j in range(F):
            oh = torch.zeros(pd_m.C, device=model.device); oh[match_out[j]] = 1.0
            params = {"w_in": pd_m.targets["w_in"], "w_out": pd_m.masked_weight("w_out", oh)}
            v = functional_call(model, params, (X[j:j + 1],))[0]
            mask = torch.ones(F, dtype=torch.bool, device=model.device); mask[j] = False
            a, b = v[mask], R[j][mask]
            out.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)))
    return np.array(out)


# ---------------------------------------------------------------- matching / scores

def hungarian_match(S: np.ndarray) -> dict[int, int]:
    """Optimal one-to-one assignment maximising total score: {feature j -> comp k}.
    Requires C >= F: with fewer components than features the assignment leaves
    features unmatched and every downstream probe KeyErrors."""
    C, F = S.shape
    if C < F:
        raise ValueError(f"hungarian_match needs C >= F (got C={C} components, "
                         f"F={F} features); fewer components than features leaves "
                         "features unmatched")
    rows, cols = linear_sum_assignment(-S)
    return {int(j): int(k) for k, j in zip(rows, cols)}


def _matched_vals(M: np.ndarray, match: dict, F: int) -> np.ndarray:
    return np.array([M[match[j], j] for j in range(F)])


def score_sides(mats: dict, match_in: dict, match_out: dict, F: int) -> dict:
    """Per side: act purity/coverage under the ACTIVATION match, mass purity under the
    SAME match (basis question), agreement with the mass-based match, mass row entropy."""
    scores = {}
    for side, match, act_key, mass_key in (("in", match_in, "act_in", "mass_in"),
                                           ("out", match_out, "act_out", "mass_out")):
        act_v = _matched_vals(mats[act_key], match, F)
        mass_v = _matched_vals(mats[mass_key], match, F)
        mass_match = hungarian_match(mats[mass_key])
        M = mats[mass_key]
        row_tot = M.sum(1)
        alive = row_tot > 1e-9
        P = M[alive] / row_tot[alive, None]
        ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
        scores[side] = {
            "act_purity": float(act_v.mean()),
            "act_coverage": int((act_v > THETA).sum()),
            "mass_purity": float(mass_v.mean()),
            "match_agree": float(np.mean([match[j] == mass_match[j] for j in range(F)])),
            "row_entropy": float(ent.mean()) if alive.any() else float("nan"),
            "split": int(((M > SPLIT_MASS).sum(0) > 1).sum()),
        }
    scores["index_agreement"] = float(np.mean([match_in[j] == match_out[j]
                                               for j in range(F)]))
    scores["dead_components"] = int((mats["mass_in"].sum(1) < 1e-9).sum())
    return scores


def neuron_basis_stats(pd_m: PD, alive_frac: float = 0.01) -> dict:
    """Detect a PER-NEURON decomposition (found 2026-07-30: VPD slices w_out into
    one shard per hidden neuron ≈ e_n ⊗ w_out[n,:]; the per-feature act probe caps
    such a basis at ~1/K, which the old rubric misread as 'distributed').

    Per side: over ALIVE shards (Frobenius norm > alive_frac * max), the mean
    concentration of shard mass on its argmax neuron slice (w_in: columns = neurons;
    w_out: rows = neurons), the number of distinct argmax neurons covered, and the
    alive count. Concentration ~1 + coverage ~d_mlp <=> per-neuron basis."""
    out = {}
    with torch.no_grad():
        for side, axis in (("in", 0), ("out", 1)):
            mats_k = [pd_m.component_matrices(k)[axis] for k in range(pd_m.C)]
            norms = torch.stack([m.pow(2).sum() for m in mats_k]).sqrt()
            alive = norms > alive_frac * norms.max()
            conc, owners = [], []
            for k in range(pd_m.C):
                if not alive[k]:
                    continue
                m = mats_k[k]
                # neuron axis: columns of w_in (d0,F->d_mlp is dim1), rows of w_out
                sl = m.pow(2).sum(dim=0) if side == "in" else m.pow(2).sum(dim=1)
                conc.append(float(sl.max() / sl.sum().clamp_min(1e-12)))
                owners.append(int(sl.argmax()))
            out[f"w_{side}"] = {"alive": int(alive.sum()),
                                "neuron_conc": float(np.mean(conc)) if conc else float("nan"),
                                "neurons_covered": len(set(owners))}
    return out


def _write_shard_owners(pd_m: PD) -> tuple[torch.Tensor, torch.Tensor]:
    """(alive[C] bool, owner[C] long) for the w_out shards: alive = Frobenius norm
    > 1% of the max; owner = argmax neuron (row) mass. The shared substrate of the
    neuron-basis stats, the union probe, and circuit selection."""
    with torch.no_grad():
        mats_k = [pd_m.component_matrices(k)[1] for k in range(pd_m.C)]
        norms = torch.stack([m.pow(2).sum() for m in mats_k]).sqrt()
        alive = norms > 0.01 * norms.max()
        owner = torch.tensor([int(m.pow(2).sum(dim=1).argmax()) for m in mats_k],
                             device=norms.device)
    return alive, owner


def circuit_mask(pd_m: PD, j: int, match_in: dict, match_out: dict,
                 ao: tuple | None = None) -> tuple[dict[str, torch.Tensor], int]:
    """Masks selecting feature j's CIRCUIT: read shard match_in[j] plus every alive
    w_out shard whose owner neuron is active on probe e_j. Selection is functional
    (neuron activity on the probe, as in union_act_out) -- no code knowledge.
    Returns (masks, n_write_shards).

    For methods without per-matrix indexing/CI (ExactPD & co) the chain is already
    glued into one component, so the circuit degenerates to the matched pair."""
    dev = pd_m.model.device
    if not hasattr(pd_m, "_ci"):
        return _pair_mask(pd_m, match_in[j], match_out[j]), 1
    model = pd_m.model
    F = model.n_features
    alive, owner = ao if ao is not None else _write_shard_owners(pd_m)
    with torch.no_grad():
        e = model.embed_input(torch.eye(F, device=dev)[j:j + 1])
        h = torch.relu(e @ pd_m.targets["w_in"])[0]
    m_out = (alive.to(dev) & (h[owner.to(dev)] > 1e-9)).float()
    m_in = torch.zeros(pd_m.C, device=dev)
    m_in[match_in[j]] = 1.0
    return {"w_in": m_in, "w_out": m_out}, int(m_out.sum())


def union_act_out(pd_m: PD, ao: tuple | None = None) -> float:
    """Mean over features j of the normalized self-response with the UNION of the
    alive w_out shards whose argmax neuron is active on probe e_j (w_in = target).
    ~1 <=> the write mechanism for each feature is exactly the union of its active
    neurons' shards -- the group-probe complement to the solo act_out matrix, which
    structurally caps a per-neuron basis at ~1/K."""
    model = pd_m.model
    F = model.n_features
    X = torch.eye(F, device=model.device)
    alive, owner = ao if ao is not None else _write_shard_owners(pd_m)
    with torch.no_grad():
        s = model(X).diagonal().clone()
        s = torch.where(s.abs() > 1e-9, s, torch.ones_like(s))
        h = torch.relu(model.embed_input(X) @ pd_m.targets["w_in"])   # (F, d_mlp)
        vals = []
        for j in range(F):
            active_n = h[j] > 1e-9
            mask = (alive & active_n[owner]).float()
            params = {"w_in": pd_m.targets["w_in"],
                      "w_out": pd_m.masked_weight("w_out", mask)}
            vals.append(float(functional_call(model, params, (X[j:j + 1],))[0, j] / s[j]))
    return float(np.mean(vals))


def mechanism_census(pd_m: PD, pd_e: PD, mats: dict, match_in: dict, match_out: dict,
                     p: float, ao: tuple | None = None,
                     batch: int = 16384, seed: int = 5) -> dict:
    """Classify EVERY subcomponent of both decompositions and measure the
    read/write correspondence (per-matrix-indexed methods only; ExactPD's census is
    trivial: 100 chain-glued per-feature components, rendered as a fixed line).

    Thresholds: aligned act > THETA (0.5); write neuron-concentration > 0.9; alive =
    norm > 1% of side max; zombie = alive but CI <= 0.1 on every probe sample.

    Correspondence: read level = 1:1 feature match rate + mean weight-matrix cosine
    of matched read shards vs ExactPD rows; write level = the bipartite incidence
    (circuit owners vs probe-active neurons, and features served per write shard --
    for the biregular code this should reproduce K and d = F*K/N)."""
    if not hasattr(pd_m, "_ci"):
        return {}
    model = pd_m.model
    F, C = model.n_features, pd_m.C
    alive_out, owner = ao if ao is not None else _write_shard_owners(pd_m)
    alive_out = alive_out.cpu(); owner = owner.cpu()
    with torch.no_grad():
        in_mats = [pd_m.component_matrices(k)[0] for k in range(C)]
        in_norms = torch.stack([m.pow(2).sum() for m in in_mats]).sqrt()
        alive_in = (in_norms > 0.01 * in_norms.max()).cpu()
        out_mats = [pd_m.component_matrices(k)[1] for k in range(C)]
        conc = torch.tensor([float(m.pow(2).sum(1).max() / m.pow(2).sum().clamp_min(1e-12))
                             for m in out_mats])
        ds = CleanDataset(n_features=F, p=float(p), device=model.device, seed=seed)
        x, _ = ds.generate_batch(batch)
        e = model.embed_input(x)
        h = torch.relu(e @ pd_m.targets["w_in"])
        lower, _ = pd_m._ci(e, h)
        fires_in = (lower["w_in"] > 0.1).any(0).cpu()
        fires_out = (lower["w_out"] > 0.1).any(0).cpu()

        # ---- read side ----
        matched = {k: j for j, k in match_in.items() if mats["act_in"][k, j] > THETA}
        read_cos = []
        for k, j in matched.items():
            pe = pd_e.component_matrices(j)[0]
            read_cos.append(float((pe * in_mats[k]).sum()
                                  / (pe.norm() * in_mats[k].norm()).clamp_min(1e-12)))
        read = {"aligned": len(matched),
                "zombie": int((alive_in & ~fires_in
                               & torch.tensor([k not in matched for k in range(C)])).sum()),
                "dead": int((~alive_in).sum())}
        read["mixed"] = C - read["aligned"] - read["zombie"] - read["dead"]

        # ---- write side ----
        owner_counts = np.bincount(owner[alive_out].numpy(), minlength=model.d_mlp) \
            if hasattr(model, "d_mlp") else np.bincount(owner[alive_out].numpy())
        wr = {"dead": int((~alive_out).sum())}
        zomb = alive_out & ~fires_out
        split = alive_out & ~zomb & torch.tensor(
            [owner_counts[int(owner[k])] > 1 for k in range(C)])
        aligned_w = alive_out & ~zomb & ~split & (conc > 0.9)
        wr["zombie"] = int(zomb.sum()); wr["split"] = int(split.sum())
        wr["neuron_aligned"] = int(aligned_w.sum())
        wr["mixed"] = C - wr["dead"] - wr["zombie"] - wr["split"] - wr["neuron_aligned"]

        # ---- write-level incidence: circuits vs active neurons; features/shard ----
        Xp = torch.eye(F, device=model.device)
        hF = torch.relu(model.embed_input(Xp) @ pd_m.targets["w_in"]).cpu()  # (F, d_mlp)
        feat_count = torch.zeros(C)
        cover = []
        for j in range(F):
            act_n = hF[j] > 1e-9
            sel = alive_out & act_n[owner]
            cover.append(len(set(owner[sel].tolist()) & set(torch.where(act_n)[0].tolist()))
                         / max(1, int(act_n.sum())))
            feat_count[sel] += 1
    return {"read": read, "write": wr,
            "read_match_cos": float(np.mean(read_cos)) if read_cos else float("nan"),
            "circuit_neuron_coverage": float(np.mean(cover)),
            "mean_active_neurons": float((hF > 1e-9).float().sum(1).mean()),
            "features_per_write_shard": float(feat_count[alive_out].mean()) if alive_out.any() else float("nan")}


# ---------------------------------------------------------------- diagnostics

def vpd_diagnostics(pd_m: PD, p: float, batch: int = 2048, seed: int = 123) -> dict:
    """Training-state diagnostics: ci_stats, method loss terms (raw + weighted),
    per-matrix ||delta||/||W|| (VPD-gated where the machinery is VPD-specific)."""
    out = {}
    if hasattr(pd_m, "ci_stats"):
        out["ci_stats"] = pd_m.ci_stats(p)
    if hasattr(pd_m, "weight_delta"):
        out["delta_ratio"] = {n: float(pd_m.weight_delta(n).norm() / pd_m.targets[n].norm())
                              for n in pd_m.param_names}
    if hasattr(pd_m, "_ci"):
        model = pd_m.model
        ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                          seed=seed)
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            y = model(x)
        # the stoch/PGD terms draw random masks internally -- seed them so the
        # reported diagnostics are reproducible run-to-run
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            terms = pd_m.method_loss_terms(x, y)
        out["loss_terms"] = {name: {"raw": float(v), "weighted": float(v) * a}
                             for name, (v, a) in terms.items()}
    return out


# ---------------------------------------------------------------- rendering

_PANELS = [  # (key, title, cmap, vmin, vmax fixed or None, row order side)
    ("act_in", "act_in: shard-only self-activation / s_j", "RdBu_r", -1.1, 1.1, "in"),
    ("act_out", "act_out: shard-only self-activation / s_j", "RdBu_r", -1.1, 1.1, "out"),
    ("mass_in", "w_in mass fraction on feature j", "magma", 0.0, 1.0, "in"),
    ("mass_out", "w_out mass fraction on feature j", "magma", 0.0, 1.0, "out"),
    ("ci_in", "CI(w_in) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0, "in"),
    ("ci_out", "CI(w_out) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0, "out"),
    ("fingerprint", "pair fingerprint cosine (signed)", "RdBu_r", -1.0, 1.0, None),
    ("ablation", "pair ablation: error rise on feature", "magma", 0.0, None, None),
    ("circuit_fingerprint", "CIRCUIT fingerprint cosine (signed)", "RdBu_r", -1.0, 1.0, None),
    ("circuit_ablation", "CIRCUIT ablation: error rise on feature", "magma", 0.0, None, None),
]


def plot_match_matrices(mats: dict, match_in: dict, match_out: dict, scores: dict,
                        tag: str, ctx: str) -> None:
    """Heatmap grid; every per-side panel is permuted by the ACTIVATION match of its
    side (row j = that side's matched shard for feature j)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = mats["mass_in"].shape[1]
    orders = {"in": np.array([match_in[j] for j in range(F)]),
              "out": np.array([match_out[j] for j in range(F)]),
              None: np.arange(F)}
    panels = [p for p in _PANELS if p[0] in mats]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (key, title, cmap, lo, hi, side) in zip(axes.flat, panels):
        M = mats[key][orders[side]]
        vmax = hi if hi is not None else max(float(np.abs(M).max()), 1e-12)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=lo, vmax=vmax,
                       interpolation="nearest")
        row_lbl = {"in": "act-match_in(j)", "out": "act-match_out(j)", None: "pair j"}[side]
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ExactPD feature j")
        ax.set_ylabel(f"row j = {row_lbl}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    s_i, s_o = scores["in"], scores["out"]
    fig.suptitle(
        f"Component recovery vs ExactPD — {ctx}\n"
        f"w_in: act={s_i['act_purity']:.2f} (cov {s_i['act_coverage']}/{F}) "
        f"mass={s_i['mass_purity']:.2f}   "
        f"w_out: act={s_o['act_purity']:.2f} (cov {s_o['act_coverage']}/{F}) "
        f"mass={s_o['mass_purity']:.2f}   "
        f"leak-align={scores.get('leak_align', float('nan')):.2f}   "
        f"solo/exact={scores.get('solo_ratio', float('nan')):.2f}   "
        f"abl={scores.get('ablation_agreement', float('nan')):.2f}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{tag}_match_matrices.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_spectra_comparison(model, pd_e: PD, pd_m: PD, match_in: dict, match_out: dict,
                            mats: dict, cs: list[int], tag: str, ctx: str,
                            m_label: str = "VPD", ao: tuple | None = None) -> None:
    """Rows = probe features, cols = full model | ExactPD component j | matched
    mechanism PAIR (basis-level; a per-neuron write basis caps this at ~1/K) |
    CIRCUIT (mechanism-level: read shard + union of active-neuron write shards) |
    learned dense. Pair titles carry per-side ACT match quality."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = model.n_features
    ones_m = torch.ones(pd_m.C, device=model.device)
    data = []
    with torch.no_grad():
        for j in cs:
            x = torch.eye(F, device=model.device)[j:j + 1]
            oh_e = torch.zeros(pd_e.C, device=model.device); oh_e[j] = 1.0
            ki, ko = match_in[j], match_out[j]
            q_in, q_out = mats["act_in"][ki, j], mats["act_out"][ko, j]
            cm, n_sh = circuit_mask(pd_m, j, match_in, match_out, ao=ao)
            data.append((j, [
                ("full model", model(x)[0].cpu().numpy()),
                (f"ExactPD comp {j}",
                 pd_e.forward_masked(x, {n: oh_e for n in pd_e.param_names})[0].cpu().numpy()),
                (f"{m_label} pair (in:{ki} act={q_in:.2f}, out:{ko} act={q_out:.2f})",
                 pd_m.forward_masked(x, _pair_mask(pd_m, ki, ko))[0].cpu().numpy()),
                (f"{m_label} circuit (in:{ki} + {n_sh} write shards)",
                 pd_m.forward_masked(x, cm)[0].cpu().numpy()),
                (f"{m_label} all components",
                 pd_m.forward_masked(x, {n: ones_m for n in pd_m.param_names})[0].cpu().numpy()),
            ]))
    ymax = max(float(np.abs(o).max()) for _, row in data for _, o in row) * 1.08
    fig, axes = plt.subplots(len(cs), 5, figsize=(23, 3.0 * len(cs)), squeeze=False,
                             constrained_layout=True, sharey=True)
    for r, (j, row) in enumerate(data):
        for cidx, (label, o) in enumerate(row):
            ax = axes[r][cidx]
            colors = ["indianred" if i == j else "steelblue" for i in range(F)]
            ax.bar(range(F), o, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            leak = float(np.sqrt(max(0.0, (o ** 2).sum() - o[j] ** 2)))
            ax.set_title(f"{label}\n s={o[j]:.3f}  off-target ‖·‖={leak:.3f}", fontsize=8)
            ax.set_ylim(-ymax, ymax)
            if r == len(cs) - 1:
                ax.set_xlabel("output feature")
        axes[r][0].set_ylabel(f"e_{j}: output value")
    fig.suptitle(f"Output spectra — full vs ExactPD vs {m_label} pair (basis) vs "
                 f"{m_label} circuit (mechanism)  (one-hot probes; red = target)  —  {ctx}",
                 fontsize=12)
    out = os.path.join(FIG_DIR, f"{tag}_match_spectra.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_pair_weights(pd_m: PD, pd_e: PD, j: int, match_in: dict, match_out: dict,
                      mats: dict, tag: str, ctx: str, m_label: str = "VPD",
                      ao: tuple | None = None) -> None:
    """Single-component visual for feature j: the matched mechanism pair
    {w_in: match_in[j], w_out: match_out[j]} next to ExactPD component j, at the
    WEIGHT level (heatmaps of the actual component matrices + gauge-invariant
    per-feature mass profiles) and the behavioural level (solo spectrum on e_j).

    The rank-1 shard MATRIX outer(V[:,k], U[k,:]) is unique (no V/U gauge freedom
    survives the outer product), and row/column norms are invariant profiles, so
    nothing here needs sign fixing. The row-2 profile is the decisive panel for
    'is the write decomposition really distributed': ExactPD's column mass is a
    spike at f=j; a distributed shard's is spread."""
    os.makedirs(FIG_DIR, exist_ok=True)
    model = pd_m.model
    F = model.n_features
    ki, ko = match_in[j], match_out[j]
    with torch.no_grad():
        m_in_k, m_out_k = pd_m.component_matrices(ki)
        m_in_k2, m_out_k2 = pd_m.component_matrices(ko)
        e_in_j, e_out_j = pd_e.component_matrices(j)
        # feature-space views (identity projection for MLP; folds w_e for EmbedMLP)
        v_in = model.project_in(m_in_k).cpu().numpy()        # (F, d_mlp) VPD read shard
        v_out = model.project_out(m_out_k2).cpu().numpy()    # (d_mlp, F) VPD write shard
        x_in = pd_e.model.project_in(e_in_j).cpu().numpy()
        x_out = pd_e.model.project_out(e_out_j).cpu().numpy()
        # solo behaviour on the probe e_j (pair AND circuit)
        probe = torch.eye(F, device=model.device)[j:j + 1]
        oh_e = torch.zeros(pd_e.C, device=model.device); oh_e[j] = 1.0
        spec_e = pd_e.forward_masked(probe, {n: oh_e for n in pd_e.param_names})[0].cpu().numpy()
        spec_m = pd_m.forward_masked(probe, _pair_mask(pd_m, ki, ko))[0].cpu().numpy()
        cmk, n_sh = circuit_mask(pd_m, j, match_in, match_out, ao=ao)
        spec_c = pd_m.forward_masked(probe, cmk)[0].cpu().numpy()
        spec_f = model(probe)[0].cpu().numpy()

    prof = {"in": (np.linalg.norm(x_in, axis=1), np.linalg.norm(v_in, axis=1)),
            "out": (np.linalg.norm(x_out, axis=0), np.linalg.norm(v_out, axis=0))}
    q = {"act_in": mats["act_in"][ki, j], "act_out": mats["act_out"][ko, j],
         "mass_in": mats["mass_in"][ki, j], "mass_out": mats["mass_out"][ko, j]}

    fig, axes = plt.subplots(3, 4, figsize=(21, 11.5), constrained_layout=True)
    axes[0][3].axis("off"); axes[1][3].axis("off")
    for r, (side, ex_M, vp_M, k, extent_lbl) in enumerate((
            ("in", x_in, v_in, ki, ("feature", "neuron")),
            ("out", x_out, v_out, ko, ("neuron", "feature")))):
        vmax = max(np.abs(ex_M).max(), np.abs(vp_M).max(), 1e-12)
        for cidx, (M, ttl) in enumerate(((ex_M, f"ExactPD comp {j}  (P_{side})"),
                                         (vp_M, f"{m_label} w_{side} shard {k}"))):
            ax = axes[r][cidx]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(ttl, fontsize=9)
            ax.set_ylabel(extent_lbl[0]); ax.set_xlabel(extent_lbl[1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[r][2]
        ex_p, vp_p = prof[side]
        ax.bar(range(F), vp_p, width=1.0, color="steelblue",
               label=f"{m_label} shard {k}")
        ax.bar(range(F), ex_p, width=0.45, color="indianred", alpha=0.75,
               label=f"ExactPD comp {j}")
        ax.axvline(j, color="k", lw=0.6, ls=":")
        which = "row" if side == "in" else "column"
        ax.set_title(f"per-feature {which} mass  "
                     f"(act_{side}={q[f'act_{side}']:.2f}, mass_{side}={q[f'mass_{side}']:.2f})",
                     fontsize=9)
        ax.set_xlabel("feature"); ax.set_ylabel(f"‖{which}‖")
        ax.legend(fontsize=7)
    ymax = max(np.abs(s).max() for s in (spec_e, spec_m, spec_c, spec_f)) * 1.08
    for cidx, (ttl, spec) in enumerate(((f"ExactPD comp {j} alone", spec_e),
                                        (f"{m_label} pair (in:{ki}, out:{ko}) alone", spec_m),
                                        (f"{m_label} circuit ({n_sh} write shards)", spec_c),
                                        ("full model", spec_f))):
        ax = axes[2][cidx]
        colors = ["indianred" if i == j else "steelblue" for i in range(F)]
        ax.bar(range(F), spec, width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f"{ttl}\n s={spec[j]:.3f}", fontsize=9)
        ax.set_xlabel("output feature")
    axes[2][0].set_ylabel(f"output on probe e_{j}")
    fig.suptitle(f"Feature {j}: matched {m_label} pair vs ExactPD component — {ctx}",
                 fontsize=12)
    out = os.path.join(FIG_DIR, f"{tag}_pair_j{j}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_matched_profiles(pd_m: PD, pd_e: PD, match_in: dict, match_out: dict,
                          tag: str, ctx: str, m_label: str = "VPD") -> None:
    """Full-coverage read/write weight comparison on MATCHED components: row j =
    the per-feature mass profile of feature j's matched shard. Left column =
    ExactPD component j (identity-like diagonal by construction); right = the
    matched learned shard. Top row = w_in (row mass over features: the read
    profile); bottom = w_out (column mass: the write profile). The all-features
    complement of plot_pair_weights' single-feature detail: a recovered basis
    shows a diagonal; a per-neuron write basis shows dense rows."""
    os.makedirs(FIG_DIR, exist_ok=True)
    model = pd_m.model
    F = model.n_features

    def profiles(pd_x, match, side):
        rows = []
        with torch.no_grad():
            for j in range(F):
                w_in_k, w_out_k = pd_x.component_matrices(match[j])
                if side == "in":
                    prof = pd_x.model.project_in(w_in_k).pow(2).sum(1).sqrt()
                else:
                    prof = pd_x.model.project_out(w_out_k).pow(2).sum(0).sqrt()
                rows.append(prof.cpu().numpy())
        return np.stack(rows)

    ident = {j: j for j in range(F)}
    panels = [("w_in read profiles (row mass)", profiles(pd_e, ident, "in"),
               profiles(pd_m, match_in, "in")),
              ("w_out write profiles (column mass)", profiles(pd_e, ident, "out"),
               profiles(pd_m, match_out, "out"))]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5), constrained_layout=True)
    for r, (title, ex_P, m_P) in enumerate(panels):
        vmax = max(ex_P.max(), m_P.max(), 1e-12)
        for cidx, (P, lbl) in enumerate(((ex_P, "ExactPD comp j"),
                                         (m_P, f"{m_label} matched shard"))):
            ax = axes[r][cidx]
            im = ax.imshow(P, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"{title} — {lbl}", fontsize=9)
            ax.set_xlabel("feature"); ax.set_ylabel("row j = shard matched to feature j")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Matched-component read/write profiles vs ExactPD — {ctx}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{tag}_matched_profiles.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_forward_losses(pd_m: PD, pd_e: PD, p: float, match_in: dict, match_out: dict,
                        tag: str, ctx: str, m_label: str = "VPD",
                        batch: int = 16384, seed: int = 11,
                        ao: tuple | None = None) -> dict:
    """L2 forward-pass losses vs the task target relu(x): full model | ExactPD |
    m_label, in two regimes.

    Left panel (on-distribution, CleanDataset at p): full model; ExactPD all
    components (== full model exactly, faithfulness is 0 by construction -- shown
    as the sanity bar); m_label all components WITHOUT the weight delta (how much
    of the function the learned shards carry; the delta closes the remainder by
    definition, so components+delta == full model trivially).

    Right panel (one-hot probes e_c, averaged over c): full model; ExactPD single
    component c; m_label matched pair (in: match_in[c], out: match_out[c]). The
    single-mechanism regime -- NB for a per-neuron write basis the pair is one
    neuron's slice, so its loss is structurally high; it measures basis alignment,
    not quality (cf. neuron-basis stats)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    model = pd_m.model
    F = model.n_features
    ds = CleanDataset(n_features=F, p=float(p), device=model.device, seed=seed)
    x, y = ds.generate_batch(batch)
    ones_m = torch.ones(pd_m.C, device=model.device)
    ones_e = torch.ones(pd_e.C, device=model.device)
    X = torch.eye(F, device=model.device)
    with torch.no_grad():
        dist = {
            "full model": float(((model(x) - y) ** 2).mean()),
            "ExactPD (all comps)": float(
                ((pd_e.forward_masked(x, {n: ones_e for n in pd_e.param_names}) - y) ** 2).mean()),
            f"{m_label} (all comps, no delta)": float(
                ((pd_m.forward_masked(x, {n: ones_m for n in pd_m.param_names}) - y) ** 2).mean()),
        }
        full_R = model(X)
        oh_losses_e, oh_losses_m, oh_losses_c = [], [], []
        for c in range(F):
            oh = torch.zeros(pd_e.C, device=model.device); oh[c] = 1.0
            out_e = pd_e.forward_masked(X[c:c + 1], {n: oh for n in pd_e.param_names})[0]
            oh_losses_e.append(float(((out_e - X[c]) ** 2).mean()))
            out_m = pd_m.forward_masked(X[c:c + 1],
                                        _pair_mask(pd_m, match_in[c], match_out[c]))[0]
            oh_losses_m.append(float(((out_m - X[c]) ** 2).mean()))
            cmk, _ = circuit_mask(pd_m, c, match_in, match_out, ao=ao)
            out_c = pd_m.forward_masked(X[c:c + 1], cmk)[0]
            oh_losses_c.append(float(((out_c - X[c]) ** 2).mean()))
        onehot = {
            "full model": float(((full_R - X) ** 2).mean()),
            "ExactPD (single comp)": float(np.mean(oh_losses_e)),
            f"{m_label} (circuit)": float(np.mean(oh_losses_c)),
            f"{m_label} (pair, basis cap)": float(np.mean(oh_losses_m)),
        }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, (series, ttl) in zip(axes, [
            (dist, f"on-distribution (CleanDataset p={p}, batch {batch})"),
            (onehot, "one-hot probes e_c (mean over c)")]):
        names = list(series)
        vals = [series[n] for n in names]
        bars = ax.bar(range(len(names)), vals,
                      color=["tab:red", "tab:green", "tab:purple", "tab:blue"][:len(names)])
        ax.bar_label(bars, fmt="%.2e", fontsize=8)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("L2 vs task target relu(x)")
    fig.suptitle(f"Forward-pass L2: full model vs ExactPD vs {m_label} — {ctx}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{tag}_forward_losses.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return {"dist": dist, "onehot": onehot}


def write_summary(scores: dict, diags: dict, mats: dict, tag: str, ctx: str) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    F = mats["mass_in"].shape[1]
    lines = [f"### Component recovery vs ExactPD — {ctx}", "",
             "| side | act purity | act cov(>0.5) | mass purity (act match) | "
             "act-vs-mass match agree | mass row entropy | split |",
             "|---|---|---|---|---|---|---|"]
    for side in ("in", "out"):
        s = scores[side]
        lines.append(f"| w_{side} | {s['act_purity']:.3f} | {s['act_coverage']}/{F} | "
                     f"{s['mass_purity']:.3f} | {s['match_agree']:.2f} | "
                     f"{s['row_entropy']:.3f} | {s['split']} |")
    lines += ["",
              f"- leak alignment of matched write shards: {scores.get('leak_align', float('nan')):.3f}",
              f"- index agreement (match_in == match_out): {scores['index_agreement']:.3f}",
              f"- solo/exact self-activation ratio (pair): {scores.get('solo_ratio', float('nan')):.3f}",
              f"- fingerprint purity (signed cosine): {scores.get('fingerprint_purity', float('nan')):.3f}",
              f"- ablation agreement: {scores.get('ablation_agreement', float('nan')):.3f}"
              + (f" (excl. {scores['ablation_unmeasured']} features w/o active samples)"
                 if scores.get("ablation_unmeasured", 0) else ""),
              f"- dead components: {scores['dead_components']}"]
    nb = scores.get("neuron_basis")
    if nb:
        for side in ("w_in", "w_out"):
            s = nb[side]
            lines.append(f"- neuron-basis[{side}]: alive={s['alive']}, "
                         f"shard neuron-concentration={s['neuron_conc']:.3f}, "
                         f"neurons covered={s['neurons_covered']}")
        lines.append(f"- union act_out (active-neuron shards per feature): "
                     f"{scores['union_act_out']:.3f}")
    if "circuit_solo_ratio" in scores:
        lines += ["", "#### Circuit level (mechanism comparison: read shard + union of "
                  "active-neuron write shards)",
                  f"- circuit size (write shards per feature): {scores['circuit_size']:.2f}",
                  f"- circuit solo/exact self-activation ratio: {scores['circuit_solo_ratio']:.3f}",
                  f"- circuit fingerprint purity (signed cosine): {scores['circuit_fingerprint_purity']:.3f}",
                  f"- circuit ablation agreement: {scores['circuit_ablation_agreement']:.3f}"
                  f"  (spillover to code-neighbours: {scores['circuit_ablation_spillover']:.3f})"]
    cen = scores.get("census")
    if cen:
        r, w = cen["read"], cen["write"]
        lines += ["", "#### Mechanism census",
                  f"- learned w_in shards: feature-aligned={r['aligned']}, mixed={r['mixed']}, "
                  f"zombie={r['zombie']}, dead={r['dead']}",
                  f"- learned w_out shards: neuron-aligned={w['neuron_aligned']}, "
                  f"split={w['split']}, mixed={w['mixed']}, zombie={w['zombie']}, dead={w['dead']}",
                  "- ExactPD (reference): 100 chain-glued per-feature components "
                  "(sparse read row + dense write column each)",
                  f"- read correspondence: {r['aligned']}/100 features 1:1, "
                  f"mean weight cosine {cen['read_match_cos']:.3f}",
                  f"- write incidence: circuit covers {cen['circuit_neuron_coverage']:.2f} "
                  f"of active neurons (mean active = {cen['mean_active_neurons']:.1f}); "
                  f"features served per write shard = {cen['features_per_write_shard']:.1f}"]
    fl = scores.get("forward_l2")
    if fl:
        lines.append("")
        for regime, series in fl.items():
            lines.append(f"- forward L2 [{regime}]: " + ", ".join(
                f"{n}={v:.3e}" for n, v in series.items()))
    lines += ["",
              "Reading: PAIR probes measure per-matrix BASIS alignment (a per-neuron "
              "write basis caps solo act at ~1/K by construction); CIRCUIT probes "
              "measure mechanism recovery and are the apples-to-apples comparison "
              "against ExactPD's chain-glued components. act high + mass high = "
              "ExactPD basis; act low + neuron-concentration high + union/circuit "
              "act high = PER-NEURON write basis; act low otherwise = genuinely "
              "distributed."]
    if diags:
        lines.append("")
        lines.append("#### Training-state diagnostics")
        if "ci_stats" in diags:
            for mat, st in diags["ci_stats"].items():
                lines.append(f"- ci_stats[{mat}]: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in st.items()))
        if "loss_terms" in diags:
            for name, t in diags["loss_terms"].items():
                lines.append(f"- loss {name}: raw={t['raw']:.3e} weighted={t['weighted']:.3e}")
        if "delta_ratio" in diags:
            lines.append("- ||delta||/||W||: " + ", ".join(
                f"{n}={v:.3f}" for n, v in diags["delta_ratio"].items()))
    path = os.path.join(RES_DIR, f"{tag}_match.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"  saved {path}")


# ---------------------------------------------------------------- orchestration

def run_suite(pd_m: PD, pd_e: PD, p: float, tag: str, ctx: str, cs: list[int],
              m_label: str = "VPD") -> dict:
    """Activation matrices -> per-side Hungarian match -> paired probes + basis
    characterization -> render + summarise. Returns scores."""
    F = pd_e.model.n_features
    mats = activation_matrices(pd_m)
    mats.update(mass_matrices(pd_m))
    mats.update(ci_coactivation(pd_m, p))
    match_in = hungarian_match(mats["act_in"])
    match_out = hungarian_match(mats["act_out"])
    ao = _write_shard_owners(pd_m) if hasattr(pd_m, "_ci") else None
    mats["fingerprint"] = paired_fingerprint_matrix(pd_m, pd_e, match_in, match_out)
    mats["ablation"], abl_counts = paired_ablation_matrix(pd_m, p, match_in, match_out)
    solo = solo_responses(pd_m, pd_e, match_in, match_out)
    mats["circuit_fingerprint"] = circuit_fingerprint_matrix(pd_m, pd_e, match_in,
                                                             match_out, ao=ao)
    mats["circuit_ablation"], cabl_counts = circuit_ablation_matrix(
        pd_m, p, match_in, match_out, ao=ao)
    csolo = circuit_solo_responses(pd_m, pd_e, match_in, match_out, ao=ao)

    scores = score_sides(mats, match_in, match_out, F)
    scores["leak_align"] = float(np.mean(leak_alignment(pd_m, match_out)))
    scores["solo_ratio"] = float(np.nanmean(solo["solo"] / np.where(
        np.abs(solo["exact"]) > 1e-9, solo["exact"], np.nan)))
    scores["fingerprint_purity"] = float(np.mean(np.diag(mats["fingerprint"])[:F]))
    measured = [j for j in range(F) if abl_counts[j] > 0]
    scores["ablation_agreement"] = float(np.mean(
        [int(np.argmax(mats["ablation"][j]) == j) for j in measured])) if measured else float("nan")
    scores["ablation_unmeasured"] = F - len(measured)
    scores["neuron_basis"] = neuron_basis_stats(pd_m)
    scores["union_act_out"] = union_act_out(pd_m, ao=ao)
    # circuit-level (mechanism) scores -- the apples-to-apples comparison; the pair
    # scores above are per-matrix BASIS diagnostics (capped at ~1/K for a
    # per-neuron write basis)
    scores["circuit_solo_ratio"] = float(np.nanmean(csolo["solo"] / np.where(
        np.abs(csolo["exact"]) > 1e-9, csolo["exact"], np.nan)))
    scores["circuit_size"] = float(csolo["n_shards"].mean())
    scores["circuit_fingerprint_purity"] = float(
        np.mean(np.diag(mats["circuit_fingerprint"])[:F]))
    cmeasured = [j for j in range(F) if cabl_counts[j] > 0]
    scores["circuit_ablation_agreement"] = float(np.mean(
        [int(np.argmax(mats["circuit_ablation"][j]) == j) for j in cmeasured])) \
        if cmeasured else float("nan")
    spill = []
    for j in cmeasured:
        pos = np.clip(mats["circuit_ablation"][j], 0.0, None)
        if pos.sum() > 0:
            spill.append(1.0 - pos[j] / pos.sum())
    scores["circuit_ablation_spillover"] = float(np.mean(spill)) if spill else float("nan")
    scores["census"] = mechanism_census(pd_m, pd_e, mats, match_in, match_out, p, ao=ao)

    diags = vpd_diagnostics(pd_m, p)
    plot_match_matrices(mats, match_in, match_out, scores, tag, ctx)
    model = pd_e.model if isinstance(pd_m, ExactPD) else pd_m.model
    plot_spectra_comparison(model, pd_e, pd_m, match_in, match_out, mats, cs, tag, ctx,
                            m_label=m_label, ao=ao)
    for j in cs:
        plot_pair_weights(pd_m, pd_e, j, match_in, match_out, mats, tag, ctx,
                          m_label=m_label, ao=ao)
    plot_matched_profiles(pd_m, pd_e, match_in, match_out, tag, ctx, m_label=m_label)
    scores["forward_l2"] = plot_forward_losses(pd_m, pd_e, p, match_in, match_out,
                                               tag, ctx, m_label=m_label, ao=ao)
    write_summary(scores, diags, mats, tag, ctx)
    return scores


def _identity_model(device="cpu"):
    """MLP(100, 100) with w_in = w_out = I: per-feature rank-1 components are
    trivially the objective's optimum. The decisive VPD port test."""
    from mlp import MLP
    model = MLP(n_features=100, d_mlp=100, device=device)
    with torch.no_grad():
        eye = torch.eye(100, device=device)
        model.w_in.copy_(eye)
        model.w_out.copy_(eye)
    return model


def main():
    import decomp_fit
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", dest="run_id", default="MLP_dembNA_dmlp50_p0.1_n4_VPD",
                    help="saved decomposition to score against ExactPD on its own model")
    ap.add_argument("--self-test", action="store_true", dest="self_test",
                    help="score ExactPD against itself (must come out a perfect diagonal)")
    ap.add_argument("--identity-control", action="store_true", dest="identity_control",
                    help="fit VPD on w_in=w_out=I at p=0.001 (positive control for the port)")
    ap.add_argument("--steps", type=int, default=4000,
                    help="VPD fit steps for --identity-control")
    ap.add_argument("--c", type=int, nargs="+", default=[0, 33, 66],
                    help="probe features for the spectra comparison")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.self_test:
        from overview import _load_trained
        model = _load_trained("MLP", None, 50, "0.1", 4.0, a.device)
        pd_e = ExactPD(model)
        scores = run_suite(pd_e, pd_e, 0.1, "selftest_ExactPD",
                           "ExactPD vs itself (MLP d50 p=0.1)", a.c, m_label="ExactPD")
        ok = (scores["in"]["act_purity"] > 0.999 and scores["out"]["act_purity"] > 0.999
              and scores["in"]["mass_purity"] > 0.999 and scores["out"]["mass_purity"] > 0.999
              and scores["in"]["match_agree"] > 0.999 and scores["index_agreement"] > 0.999)
        print(f"self-test {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    if a.identity_control:
        p = 0.001
        model = _identity_model(a.device)
        pd_v = decomp_fit.build_pd("VPD", model, 100)
        decomp_fit.fit_decomposition(
            pd_v, CleanDataset(n_features=100, p=p, device=a.device, seed=31337),
            steps=a.steps, loss_kwargs=decomp_fit.loss_kwargs_for("VPD", k=1),
            seed=31337, desc=f"VPD identity control ({a.steps} steps)")
        pd_e = ExactPD(model)
        scores = run_suite(pd_v, pd_e, p, f"identity_control_VPD_{a.steps}",
                           f"identity model (w_in=w_out=I), p={p}, VPD {a.steps} steps",
                           a.c, m_label="VPD")
        ok = scores["in"]["act_purity"] > 0.9 and scores["out"]["act_purity"] > 0.9
        print(f"identity control {'PASS' if ok else 'FAIL (port/config suspect if 25k also fails)'}")
        return

    p = float(a.run_id.split("_p")[1].split("_")[0])
    pd_m = decomp_fit.load_decomposition(a.run_id, device=a.device)
    pd_e = ExactPD(pd_m.model)
    method = a.run_id.rsplit("_", 1)[1]
    run_suite(pd_m, pd_e, p, f"{a.run_id}_matchsuite", f"{a.run_id} (p={p})", a.c,
              m_label=method)


if __name__ == "__main__":
    main()
