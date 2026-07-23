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


def ci_coactivation(pd_v: PD, p: float, batch: int = 4096, seed: int = 0) -> dict:
    """{'ci_in', 'ci_out'}: (C, F) Pearson correlation of the VPD CI (lower mask
    source, as in vpd.ci_stats) with feature activity over a CleanDataset(p) batch.
    Empty dict for methods without a CI function (only VPD has one)."""
    if not hasattr(pd_v, "_ci"):
        return {}
    model = pd_v.model
    ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                      seed=seed)
    x, _ = ds.generate_batch(batch)
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


def paired_ablation_matrix(pd_m: PD, p: float, match_in: dict, match_out: dict,
                           batch: int = 2048, seed: int = 7) -> np.ndarray:
    """(F, F): row j = increase in each feature's active-conditional squared error
    when feature j's matched mechanism pair is removed from the components-only
    forward. Baseline = all components active on both matrices."""
    model = pd_m.model
    F = model.n_features
    ds = CleanDataset(n_features=F, p=float(p), device=model.device, seed=seed)
    x, _ = ds.generate_batch(batch)
    active = (x > 0).float()
    denom = active.sum(0).clamp_min(1)
    ones = torch.ones(pd_m.C, device=model.device)
    with torch.no_grad():
        y = model(x)
        base = pd_m.forward_masked(x, {n: ones for n in pd_m.param_names})
        base_err = (((base - y) ** 2) * active).sum(0) / denom
        rows = []
        for j in range(F):
            m_in = ones.clone(); m_in[match_in[j]] = 0.0
            m_out = ones.clone(); m_out[match_out[j]] = 0.0
            out = pd_m.forward_masked(x, {"w_in": m_in, "w_out": m_out})
            err = (((out - y) ** 2) * active).sum(0) / denom
            rows.append((err - base_err).cpu().numpy())
    return np.stack(rows)


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
    """Optimal one-to-one assignment maximising total score: {feature j -> comp k}."""
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
                            m_label: str = "VPD") -> None:
    """Rows = probe features, cols = full model | ExactPD component j | matched
    mechanism PAIR | learned dense. Pair titles carry per-side ACT match quality."""
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
            data.append((j, [
                ("full model", model(x)[0].cpu().numpy()),
                (f"ExactPD comp {j}",
                 pd_e.forward_masked(x, {n: oh_e for n in pd_e.param_names})[0].cpu().numpy()),
                (f"{m_label} pair (in:{ki} act={q_in:.2f}, out:{ko} act={q_out:.2f})",
                 pd_m.forward_masked(x, _pair_mask(pd_m, ki, ko))[0].cpu().numpy()),
                (f"{m_label} all components",
                 pd_m.forward_masked(x, {n: ones_m for n in pd_m.param_names})[0].cpu().numpy()),
            ]))
    ymax = max(float(np.abs(o).max()) for _, row in data for _, o in row) * 1.08
    fig, axes = plt.subplots(len(cs), 4, figsize=(19, 3.0 * len(cs)), squeeze=False,
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
    fig.suptitle(f"Output spectra — full vs ExactPD vs act-matched {m_label} mechanism pair "
                 f"(one-hot probes; red = target)  —  {ctx}", fontsize=12)
    out = os.path.join(FIG_DIR, f"{tag}_match_spectra.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


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
              f"- ablation agreement: {scores.get('ablation_agreement', float('nan')):.3f}",
              f"- dead components: {scores['dead_components']}", "",
              "Reading: act high + mass high = ExactPD basis; act high + mass low + "
              "leak-align high = same mechanism in the leak-aligned basis; act low = "
              "genuinely distributed."]
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
    mats["fingerprint"] = paired_fingerprint_matrix(pd_m, pd_e, match_in, match_out)
    mats["ablation"] = paired_ablation_matrix(pd_m, p, match_in, match_out)
    solo = solo_responses(pd_m, pd_e, match_in, match_out)

    scores = score_sides(mats, match_in, match_out, F)
    scores["leak_align"] = float(np.mean(leak_alignment(pd_m, match_out)))
    scores["solo_ratio"] = float(np.nanmean(solo["solo"] / np.where(
        np.abs(solo["exact"]) > 1e-9, solo["exact"], np.nan)))
    scores["fingerprint_purity"] = float(np.mean(np.diag(mats["fingerprint"])[:F]))
    scores["ablation_agreement"] = float(np.mean(
        [int(np.argmax(mats["ablation"][j]) == j) for j in range(F)]))

    diags = vpd_diagnostics(pd_m, p)
    plot_match_matrices(mats, match_in, match_out, scores, tag, ctx)
    model = pd_e.model if isinstance(pd_m, ExactPD) else pd_m.model
    plot_spectra_comparison(model, pd_e, pd_m, match_in, match_out, mats, cs, tag, ctx,
                            m_label=m_label)
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
