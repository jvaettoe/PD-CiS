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
                             matrices (component-only forward on all e_j probes). Scores
                             read AND write behaviour jointly, signed.
  4. ablation             -- causal: rise in feature j's active-conditional error when
                             component k is ablated from the components-only forward.

Matching is Hungarian (scipy linear_sum_assignment) on the mean of the two mass
matrices -- greedy matching cascades errors exactly in the smeared regimes of interest.

Prototype / validation CLI (uses saved decomps of the TRAINED models; the biregular
driver is biregular_vpd_compare.py):

  ./.venv/bin/python component_match.py --run-id MLP_dembNA_dmlp50_p0.1_n4_VPD
  ./.venv/bin/python component_match.py --self-test        # ExactPD vs itself -> identity
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


def _fingerprints(pd_obj: PD, F: int) -> torch.Tensor:
    """(C, F*F) one-hot response fingerprints: row k = flattened output of the
    component-k-only forward on all F one-hot probes."""
    X = torch.eye(F, device=pd_obj.model.device)
    rows = []
    with torch.no_grad():
        for k in range(pd_obj.C):
            oh = torch.zeros(pd_obj.C, device=X.device)
            oh[k] = 1.0
            rows.append(pd_obj.forward_masked(X, {n: oh for n in pd_obj.param_names}).flatten())
    return torch.stack(rows)


def fingerprint_matrix(pd_m: PD, pd_e: PD) -> np.ndarray:
    """(C, C_e) SIGNED cosine between one-hot response fingerprints. ExactPD's
    fingerprint j is a single spike at (j, j), so column j scores how much of
    component k's behaviour is 'respond to e_j at output j' -- leak-aligned
    components score lower here than in mass_in even when correctly matched."""
    F = pd_m.model.n_features
    fm, fe = _fingerprints(pd_m, F), _fingerprints(pd_e, F)
    fm = fm / fm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    fe = fe / fe.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (fm @ fe.T).cpu().numpy()


def ablation_matrix(pd_m: PD, p: float, batch: int = 2048, seed: int = 7) -> np.ndarray:
    """(C, F): increase in feature j's active-conditional squared error when
    component k is removed from the components-only forward. Baseline = all
    components active (for VPD this excludes the weight delta on both sides, so
    the delta's contribution cancels out of the difference)."""
    model = pd_m.model
    ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                      seed=seed)
    x, _ = ds.generate_batch(batch)
    active = (x > 0).float()
    denom = active.sum(0).clamp_min(1)
    ones = torch.ones(pd_m.C, device=model.device)
    with torch.no_grad():
        y = model(x)
        base = pd_m.forward_masked(x, {n: ones for n in pd_m.param_names})
        base_err = (((base - y) ** 2) * active).sum(0) / denom
        rows = []
        for k in range(pd_m.C):
            m = ones.clone()
            m[k] = 0.0
            out = pd_m.forward_masked(x, {n: m for n in pd_m.param_names})
            err = (((out - y) ** 2) * active).sum(0) / denom
            rows.append((err - base_err).cpu().numpy())
    return np.stack(rows)


# ---------------------------------------------------------------- matching / scores

def hungarian_match(S: np.ndarray) -> dict[int, int]:
    """Optimal one-to-one assignment maximising total score: {feature j -> comp k}.
    Handles rectangular S (C, F)."""
    rows, cols = linear_sum_assignment(-S)
    return {int(j): int(k) for k, j in zip(rows, cols)}


def match_and_score(mats: dict[str, np.ndarray], C: int, F: int) -> tuple[dict, dict]:
    """Hungarian match on the mean of the two mass matrices, then per-side scores:
    purity (mean matched value), coverage (matched value > theta), split features,
    dead components, mean row entropy (smearing)."""
    S = 0.5 * (mats["mass_in"] + mats["mass_out"])
    match = hungarian_match(S)
    scores = {}
    for side in ("mass_in", "mass_out"):
        M = mats[side]
        matched = np.array([M[match[j], j] for j in range(F)])
        row_tot = M.sum(1)
        alive = row_tot > 1e-9
        P = M[alive] / row_tot[alive, None]
        ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
        scores[side] = {
            "purity": float(matched.mean()),
            "coverage": int((matched > THETA_MASS).sum()),
            "split": int(((M > SPLIT_MASS).sum(0) > 1).sum()),
            "row_entropy": float(ent.mean()) if alive.any() else float("nan"),
        }
    scores["dead_components"] = int((mats["mass_in"].sum(1) < 1e-9).sum())
    if "fingerprint" in mats:
        fp = np.array([mats["fingerprint"][match[j], j] for j in range(F)])
        scores["fingerprint_purity"] = float(fp.mean())
    if "ablation" in mats:
        A = mats["ablation"]
        agree = sum(int(np.argmax(A[match[j]]) == j) for j in range(F))
        scores["ablation_agreement"] = agree / F
    return match, scores


# ---------------------------------------------------------------- rendering

_PANELS = [  # (key, title, cmap, vmin, vmax fixed or None -> per-matrix max)
    ("mass_in", "w_in mass fraction on feature j", "magma", 0.0, 1.0),
    ("mass_out", "w_out mass fraction on feature j", "magma", 0.0, 1.0),
    ("ci_in", "CI(w_in) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0),
    ("ci_out", "CI(w_out) ~ feature activity (corr)", "RdBu_r", -1.0, 1.0),
    ("fingerprint", "one-hot fingerprint cosine (signed)", "RdBu_r", -1.0, 1.0),
    ("ablation", "ablation: error rise on feature j", "magma", 0.0, None),
]


def plot_match_matrices(mats: dict, match: dict, scores: dict, tag: str,
                        ctx: str) -> None:
    """One figure, up to 2x3 heatmaps, rows permuted so the match lies on the
    diagonal (row j = the learned component matched to feature j)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = mats["mass_in"].shape[1]
    order = np.array([match[j] for j in range(F)])
    panels = [(k, t, c, lo, hi) for k, t, c, lo, hi in _PANELS if k in mats]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (key, title, cmap, lo, hi) in zip(axes.flat, panels):
        M = mats[key][order]
        vmax = hi if hi is not None else max(float(np.abs(M).max()), 1e-12)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=lo, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ExactPD feature j")
        ax.set_ylabel("matched learned component (row j = match of j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    sc_in, sc_out = scores["mass_in"], scores["mass_out"]
    fig.suptitle(f"Component recovery vs ExactPD — {ctx}\n"
                 f"w_in: purity={sc_in['purity']:.2f} cov(>{THETA_MASS})={sc_in['coverage']}/{F}   "
                 f"w_out: purity={sc_out['purity']:.2f} cov={sc_out['coverage']}/{F}   "
                 f"abl.agree={scores.get('ablation_agreement', float('nan')):.2f}   "
                 f"dead={scores['dead_components']}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{tag}_match_matrices.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_spectra_comparison(model, pd_e: PD, pd_m: PD, match: dict, cs: list[int],
                            tag: str, ctx: str, m_label: str = "VPD") -> None:
    """Rows = probe features, cols = full model | ExactPD component j | matched
    learned component | learned dense (all components, no delta). Red bar = target
    feature. Panel 3 is the basis discriminator: clean spike => ExactPD basis;
    spike+leak reproducing panel 1 => leak-aligned basis. Panel 4 vs panel 1 gap =
    function living outside the components (VPD's weight delta)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = model.n_features
    ones_m = torch.ones(pd_m.C, device=model.device)
    data = []   # (j, [(label, out_vector)])
    with torch.no_grad():
        for j in cs:
            x = torch.eye(F, device=model.device)[j:j + 1]
            oh_e = torch.zeros(pd_e.C, device=model.device); oh_e[j] = 1.0
            k = match[j]
            oh_m = torch.zeros(pd_m.C, device=model.device); oh_m[k] = 1.0
            data.append((j, [
                ("full model", model(x)[0].cpu().numpy()),
                (f"ExactPD comp {j}",
                 pd_e.forward_masked(x, {n: oh_e for n in pd_e.param_names})[0].cpu().numpy()),
                (f"{m_label} comp {k} (matched)",
                 pd_m.forward_masked(x, {n: oh_m for n in pd_m.param_names})[0].cpu().numpy()),
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
    fig.suptitle(f"Output spectra — full vs ExactPD vs matched {m_label} component "
                 f"(one-hot probes; red = target)  —  {ctx}", fontsize=12)
    out = os.path.join(FIG_DIR, f"{tag}_match_spectra.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def write_summary(scores: dict, match: dict, mats: dict, tag: str, ctx: str) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    F = mats["mass_in"].shape[1]
    lines = [f"### Component recovery vs ExactPD — {ctx}", "",
             "| side | purity | coverage(>0.5) | split features | mean row entropy |",
             "|---|---|---|---|---|"]
    for side in ("mass_in", "mass_out"):
        s = scores[side]
        lines.append(f"| {side} | {s['purity']:.3f} | {s['coverage']}/{F} | "
                     f"{s['split']} | {s['row_entropy']:.3f} |")
    lines += ["", f"- fingerprint purity (signed cosine): "
              f"{scores.get('fingerprint_purity', float('nan')):.3f}",
              f"- ablation agreement: {scores.get('ablation_agreement', float('nan')):.3f}",
              f"- dead components: {scores['dead_components']}"]
    path = os.path.join(RES_DIR, f"{tag}_match.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"  saved {path}")


# ---------------------------------------------------------------- orchestration

def run_suite(pd_m: PD, pd_e: PD, p: float, tag: str, ctx: str, cs: list[int],
              m_label: str = "VPD") -> dict:
    """Compute all four matrices, match, render, summarise. Returns scores."""
    F = pd_e.model.n_features
    mats = mass_matrices(pd_m)
    mats.update(ci_coactivation(pd_m, p))
    mats["fingerprint"] = fingerprint_matrix(pd_m, pd_e)
    mats["ablation"] = ablation_matrix(pd_m, p)
    match, scores = match_and_score(mats, pd_m.C, F)
    plot_match_matrices(mats, match, scores, tag, ctx)
    plot_spectra_comparison(pd_m.model if not isinstance(pd_m, ExactPD) else pd_e.model,
                            pd_e, pd_m, match, cs, tag, ctx, m_label=m_label)
    write_summary(scores, match, mats, tag, ctx)
    return scores


def main():
    import decomp_fit
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", dest="run_id", default="MLP_dembNA_dmlp50_p0.1_n4_VPD",
                    help="saved decomposition to score against ExactPD on its own model")
    ap.add_argument("--self-test", action="store_true", dest="self_test",
                    help="score ExactPD against itself (must come out a perfect diagonal)")
    ap.add_argument("--c", type=int, nargs="+", default=[0, 33, 66],
                    help="probe features for the spectra comparison")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.self_test:
        from overview import _load_trained
        model = _load_trained("MLP", None, 50, "0.1", 4.0, a.device)
        pd_e = ExactPD(model)
        scores = run_suite(pd_e, pd_e, 0.1, "selftest_ExactPD", "ExactPD vs itself (MLP d50 p=0.1)",
                           a.c, m_label="ExactPD")
        ok = (scores["mass_in"]["purity"] > 0.999 and scores["mass_out"]["purity"] > 0.999
              and scores["fingerprint_purity"] > 0.999)
        print(f"self-test {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    p = float(a.run_id.split("_p")[1].split("_")[0])
    pd_m = decomp_fit.load_decomposition(a.run_id, device=a.device)
    pd_e = ExactPD(pd_m.model)
    method = a.run_id.rsplit("_", 1)[1]
    run_suite(pd_m, pd_e, p, f"{a.run_id}_matchsuite", f"{a.run_id} (p={p})", a.c,
              m_label=method)


if __name__ == "__main__":
    main()
