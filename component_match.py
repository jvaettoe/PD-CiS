"""Four-matrix component-recovery comparison of a learned PD against ExactPD.

"Learned component k == ExactPD component j" means four different things, and under
interference they can come apart; each matrix answers one notion of sameness, and the
DISAGREEMENT pattern between them is the finding (e.g. a leak-aligned w_out basis reads
as: mass_in diagonal, ci diagonal, w_out mass off-diagonal):

  1. mass_in / mass_out   -- weight space: fraction of learned component k's Frobenius
                             mass in feature j's ExactPD support (row j of readin /
                             column j of readout). Exact (ExactPD supports are disjoint),
                             rows sum to 1. Reported PER MATRIX -- the w_in side is
                             expected to recover, the w_out side is where interference
                             forces a basis choice.
  2. ci_in / ci_out       -- functional (firing): Pearson correlation of the VPD CI mask
                             value for component k with feature j's activity on a sparse
                             batch. VPD-only (needs a CI function).
  3. fingerprint          -- behavioural: cosine between flattened one-hot response
                             matrices of the matched MECHANISM PAIR and each ExactPD
                             component. Signed.
  4. ablation             -- causal: rise in feature j's active-conditional error when
                             feature j's matched mechanism pair is ablated.

PER-MATRIX COMPONENT INDEXING. VPD (like Goodfire's per-module design) indexes
components independently per weight matrix: separate CI nets and separate [B, C] masks
for w_in and w_out -- NOTHING aligns index k across the two matrices. So matching is
Hungarian PER SIDE (match_in from mass_in, match_out from mass_out), and every
functional probe uses the matched PAIR for feature j:
    mask = {w_in: onehot(match_in[j]), w_out: onehot(match_out[j])}.
Applying one shared index to both matrices (the naive thing) pairs a read-shard with an
unrelated write-shard and reads as zero output even under perfect per-side recovery.
For shared-index methods (ExactPD / APD / SPD) match_in == match_out and everything
reduces to the naive comparison; `index_agreement` reports how aligned the two sides'
indexings happen to be.

Prototype / validation CLI (uses saved decomps of the TRAINED models; the biregular
driver is biregular_vpd_compare.py):

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

from data import CleanDataset
from pd import PD, ExactPD

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "comparison")
RES_DIR = os.path.join(os.path.dirname(__file__), "results", "overview")
THETA_MASS = 0.5     # coverage threshold on mass fractions (in [0, 1])
SPLIT_MASS = 0.25    # a feature is "split" if >1 components carry this much of it


# ---------------------------------------------------------------- matrices

def mass_matrices(pd_m: PD) -> dict[str, np.ndarray]:
    """{'mass_in', 'mass_out'}: (C, F) fraction of component k's Frobenius mass in
    feature j's ExactPD support -- row j of the feature-space read-in for w_in,
    column j of the feature-space read-out for w_out. Rows sum to 1 (0 for dead
    components). Projected through the model's embedding so it is exact for MLP
    and meaningful for EmbedMLP."""
    model = pd_m.model
    m_in, m_out = [], []
    with torch.no_grad():
        for k in range(pd_m.C):
            w_in_k, w_out_k = pd_m.component_matrices(k)
            rin = model.project_in(w_in_k)              # (F, d_mlp)
            rout = model.project_out(w_out_k)           # (d_mlp, F)
            row_mass = rin.pow(2).sum(1)                # (F,)
            col_mass = rout.pow(2).sum(0)               # (F,)
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
        za = (act - act.mean(0)) / act.std(0).clamp_min(1e-12)      # (B, F)
        out = {}
        for name, tag in (("w_in", "ci_in"), ("w_out", "ci_out")):
            ci = lower[name]
            zc = (ci - ci.mean(0)) / ci.std(0).clamp_min(1e-12)     # (B, C)
            corr = zc.T @ za / x.shape[0]                           # (C, F)
            dead = ci.std(0) < 1e-8                                 # constant CI -> corr 0
            corr[dead] = 0.0
            out[tag] = corr.cpu().numpy()
    return out


def _pair_mask(pd_m: PD, k_in: int, k_out: int) -> dict[str, torch.Tensor]:
    """Batch-shared masks selecting w_in component k_in and w_out component k_out --
    the mechanism PAIR (per-matrix indexing; see module docstring)."""
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
    ExactPD's own s_j: {'solo': (F,), 'exact': (F,)}. solo/exact ~ 1 everywhere means
    the paired components carry the on-target computation."""
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


# ---------------------------------------------------------------- matching / scores

def hungarian_match(S: np.ndarray) -> dict[int, int]:
    """Optimal one-to-one assignment maximising total score: {feature j -> comp k}.
    Handles rectangular S (C, F)."""
    rows, cols = linear_sum_assignment(-S)
    return {int(j): int(k) for k, j in zip(rows, cols)}


def score_side(M: np.ndarray, match: dict, F: int) -> dict:
    """Purity/coverage/split/entropy of one mass matrix under its OWN side's match."""
    matched = np.array([M[match[j], j] for j in range(F)])
    row_tot = M.sum(1)
    alive = row_tot > 1e-9
    P = M[alive] / row_tot[alive, None]
    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
    return {"purity": float(matched.mean()),
            "coverage": int((matched > THETA_MASS).sum()),
            "split": int(((M > SPLIT_MASS).sum(0) > 1).sum()),
            "row_entropy": float(ent.mean()) if alive.any() else float("nan")}


# ---------------------------------------------------------------- diagnostics

def vpd_diagnostics(pd_m: PD, p: float, batch: int = 2048, seed: int = 123) -> dict:
    """Training-state diagnostics (VPD-gated where the machinery is VPD-specific):
    ci_stats, method loss terms (raw + weighted), per-matrix ||delta||/||W||.
    Makes 'stopped mid-sparsification' visible in every report."""
    out = {}
    if hasattr(pd_m, "ci_stats"):
        out["ci_stats"] = pd_m.ci_stats(p)
    if hasattr(pd_m, "weight_delta"):
        out["delta_ratio"] = {n: float(pd_m.weight_delta(n).norm() / pd_m.targets[n].norm())
                              for n in pd_m.param_names}
    if hasattr(pd_m, "_ci"):   # VPD: loss-term breakdown on a fresh batch
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

_PANELS = [  # (key, title, cmap, vmin, vmax, which match orders the rows)
    ("mass_in", "w_in mass fraction on feature j", "magma", 0.0, 1.0, "in"),
    ("mass_out", "w_out mass fraction on feature j", "magma", 0.0, 1.0, "out"),
    ("ci_in", "CI(w_in) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0, "in"),
    ("ci_out", "CI(w_out) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0, "out"),
    ("fingerprint", "pair fingerprint cosine (signed)", "RdBu_r", -1.0, 1.0, None),
    ("ablation", "pair ablation: error rise on feature", "magma", 0.0, None, None),
]


def plot_match_matrices(mats: dict, match_in: dict, match_out: dict, scores: dict,
                        tag: str, ctx: str) -> None:
    """Up to 2x3 heatmaps. mass/ci panels are permuted by their OWN side's match
    (row j = that side's match of feature j); fingerprint/ablation rows are the
    mechanism pairs, already j-indexed."""
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
        row_lbl = {"in": "match_in(j)", "out": "match_out(j)", None: "pair j"}[side]
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ExactPD feature j")
        ax.set_ylabel(f"row j = {row_lbl}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    s_in, s_out = scores["mass_in"], scores["mass_out"]
    fig.suptitle(
        f"Component recovery vs ExactPD — {ctx}\n"
        f"w_in: purity={s_in['purity']:.2f} cov={s_in['coverage']}/{F}   "
        f"w_out: purity={s_out['purity']:.2f} cov={s_out['coverage']}/{F}   "
        f"idx-agree={scores['index_agreement']:.2f}   "
        f"solo/exact={scores['solo_ratio']:.2f}   "
        f"abl.agree={scores.get('ablation_agreement', float('nan')):.2f}   "
        f"dead={scores['dead_components']}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{tag}_match_matrices.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_spectra_comparison(model, pd_e: PD, pd_m: PD, match_in: dict, match_out: dict,
                            mats: dict, cs: list[int], tag: str, ctx: str,
                            m_label: str = "VPD") -> None:
    """Rows = probe features, cols = full model | ExactPD component j | matched
    mechanism PAIR (w_in comp match_in[j] + w_out comp match_out[j]) | learned dense
    (all components, no delta). Red bar = target feature. Pair panel titles carry the
    per-side match masses so chance-level matches cannot read as 'the correct
    component'."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = model.n_features
    ones_m = torch.ones(pd_m.C, device=model.device)
    data = []   # (j, [(label, out_vector)])
    with torch.no_grad():
        for j in cs:
            x = torch.eye(F, device=model.device)[j:j + 1]
            oh_e = torch.zeros(pd_e.C, device=model.device); oh_e[j] = 1.0
            ki, ko = match_in[j], match_out[j]
            q_in, q_out = mats["mass_in"][ki, j], mats["mass_out"][ko, j]
            data.append((j, [
                ("full model", model(x)[0].cpu().numpy()),
                (f"ExactPD comp {j}",
                 pd_e.forward_masked(x, {n: oh_e for n in pd_e.param_names})[0].cpu().numpy()),
                (f"{m_label} pair (in:{ki} m={q_in:.2f}, out:{ko} m={q_out:.2f})",
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
    fig.suptitle(f"Output spectra — full vs ExactPD vs matched {m_label} mechanism pair "
                 f"(one-hot probes; red = target)  —  {ctx}", fontsize=12)
    out = os.path.join(FIG_DIR, f"{tag}_match_spectra.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def write_summary(scores: dict, diags: dict, mats: dict, tag: str, ctx: str) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    F = mats["mass_in"].shape[1]
    lines = [f"### Component recovery vs ExactPD — {ctx}", "",
             "| side | purity | coverage(>0.5) | split features | mean row entropy |",
             "|---|---|---|---|---|"]
    for side in ("mass_in", "mass_out"):
        s = scores[side]
        lines.append(f"| {side} | {s['purity']:.3f} | {s['coverage']}/{F} | "
                     f"{s['split']} | {s['row_entropy']:.3f} |")
    lines += ["",
              f"- index agreement (match_in == match_out): {scores['index_agreement']:.3f}",
              f"- solo/exact self-activation ratio (mean): {scores['solo_ratio']:.3f}",
              f"- fingerprint purity (signed cosine): "
              f"{scores.get('fingerprint_purity', float('nan')):.3f}",
              f"- ablation agreement: {scores.get('ablation_agreement', float('nan')):.3f}",
              f"- dead components: {scores['dead_components']}"]
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
    """Mass matrices -> per-side Hungarian matching -> paired functional probes ->
    render + summarise. Returns scores."""
    F = pd_e.model.n_features
    mats = mass_matrices(pd_m)
    mats.update(ci_coactivation(pd_m, p))
    match_in = hungarian_match(mats["mass_in"])
    match_out = hungarian_match(mats["mass_out"])
    mats["fingerprint"] = paired_fingerprint_matrix(pd_m, pd_e, match_in, match_out)
    mats["ablation"] = paired_ablation_matrix(pd_m, p, match_in, match_out)
    solo = solo_responses(pd_m, pd_e, match_in, match_out)

    scores = {"mass_in": score_side(mats["mass_in"], match_in, F),
              "mass_out": score_side(mats["mass_out"], match_out, F),
              "index_agreement": float(np.mean([match_in[j] == match_out[j]
                                                for j in range(F)])),
              "solo_ratio": float(np.mean(solo["solo"] / np.where(
                  np.abs(solo["exact"]) > 1e-9, solo["exact"], np.nan))),
              "fingerprint_purity": float(np.mean(np.diag(mats["fingerprint"])[:F])),
              "ablation_agreement": float(np.mean(
                  [int(np.argmax(mats["ablation"][j]) == j) for j in range(F)])),
              "dead_components": int((mats["mass_in"].sum(1) < 1e-9).sum())}

    diags = vpd_diagnostics(pd_m, p)
    plot_match_matrices(mats, match_in, match_out, scores, tag, ctx)
    model = pd_e.model if isinstance(pd_m, ExactPD) else pd_m.model
    plot_spectra_comparison(model, pd_e, pd_m, match_in, match_out, mats, cs, tag, ctx,
                            m_label=m_label)
    write_summary(scores, diags, mats, tag, ctx)
    return scores


def _identity_model(device="cpu"):
    """MLP(100, 100) with w_in = w_out = I: no superposition, no interference --
    per-feature rank-1 components are trivially the objective's optimum. The
    decisive VPD port test: failure here is a bug, not task difficulty."""
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
        ok = (scores["mass_in"]["purity"] > 0.999 and scores["mass_out"]["purity"] > 0.999
              and scores["fingerprint_purity"] > 0.999
              and scores["index_agreement"] > 0.999)
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
        ok = scores["mass_in"]["purity"] > 0.9 and scores["mass_out"]["purity"] > 0.9
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
