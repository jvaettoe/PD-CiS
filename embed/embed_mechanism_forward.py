"""Mechanism forward-pass comparison at d_emb = 200: full model vs the three
W_in ontologies of the strictly layer-local PD mechanism (transpose decoder,
tied W_out = W_in.T, unembed = W_e.T).

For the designed embedded biregular sandwich (checkpoints/embedded_ideal/
ideal_d200.pt) the mechanism on a one-hot probe v*e_j is: the W_emb component
e_j (x) W_e[j] emits a_j = v*W_e[j]; W_in is decomposed into pieces that each
activate with a coefficient READ OFF THE CHAIN (no W_e knowledge at inference
— Option A, activation-direction projections, is rejected for exactly that
reason: the decomposition must be fixed before inference). Three ontologies:

  rows    : per-embedding-dimension pieces R_d = e_d (x) W_in[d,:]  (200 pieces)
            coefficient of R_d on feature j = (a_j)_d
  columns : per-neuron pieces W_in[:,n] (x) e_n                     (50 pieces)
            coefficient = (a_j @ W_in)_n  (~ c_j[n] at d200 where G = I)
  svd     : triplets s_k u_k v_k^T                                  (50 pieces)
            coefficient = s_k (u_k . a_j) = v * s_k * L[j,k]

All three tile W_in exactly, so the full-sum mechanism forward is an IDENTITY
with the full model (asserted, not a finding). The content is the per-feature
causal-activity census (participation ratios, sorted-coefficient curves,
multi-threshold #components for 50/90/99% energy — no arbitrary cutoffs) and
the top-m truncated-forward fidelity (spike + leak error vs m, tied-transpose
truncation of both W_in and W_out).

  ./.venv/bin/python -m embed.embed_mechanism_forward

Outputs:
  results/embedded/ideal_vs_vpd/mechanism_forward/{census_d200.md, activity.csv}
  figures/embedded/ideal_vs_vpd/mechanism_forward/
      {activity_curves_d200.png, truncation_fidelity_d200.png,
       spectra_examples_d200.png}
"""
import os

from core.paths import CHECKPOINTS, FIGURES, RESULTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

D_EMB = 200
CKPT = os.path.join(CHECKPOINTS, "embedded_ideal", f"ideal_d{D_EMB}.pt")
RES_DIR = os.path.join(RESULTS, "embedded", "ideal_vs_vpd", "mechanism_forward")
FIG_DIR = os.path.join(FIGURES, "embedded", "ideal_vs_vpd", "mechanism_forward")

ONTOLOGIES = ("rows", "columns", "svd")
ONT_LABEL = {"rows": "rows (per-emb-dim)", "columns": "columns (per-neuron)",
             "svd": "SVD (triplets)"}
ONT_COLOR = {"rows": "tab:blue", "columns": "tab:orange", "svd": "tab:green"}
THRESHOLDS = (0.50, 0.90, 0.99)
SPIKE_TOL, LEAK_TOL = 0.01, 0.10          # 1% spike, 10% leak
M_SHOW = 5                                # equal-budget m for the spectra figure


# ---------------------------------------------------------------- load + model

def load():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    t = {k: ck[k].double().numpy() for k in ("W_e", "W_in", "W_out", "C",
                                             "U", "S", "V", "L", "G")}
    t["M"] = np.asarray(ck["M"]).astype(bool)
    t["cfg"] = ck["config"]
    return t


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def full_forward(We: np.ndarray, Win: np.ndarray) -> np.ndarray:
    """Full-model outputs on ALL one-hot probes (v=1): row j = y(e_j).
    y = relu(x W_e W_in) W_in^T W_e^T  (tied transposes)."""
    H = relu(We @ Win)                       # (F, N)
    return H @ Win.T @ We.T                  # (F, F)


# ------------------------------------------------------------ exactness asserts

def run_asserts(t: dict) -> list[str]:
    We, Win, Wout = t["W_e"], t["W_in"], t["W_out"]
    U, S, V, L = t["U"], t["S"], t["V"], t["L"]
    F, d = We.shape
    N = Win.shape[1]
    lines = []

    def rec(name, val, tol):
        ok = val <= tol
        lines.append(f"{name}: {val:.3e}  (tol {tol:.0e}) {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name} failed: {val:.3e} > {tol:.0e}"

    # tied transposes
    assert np.array_equal(Wout, Win.T), "W_out != W_in.T"
    lines.append("W_out == W_in.T: bitwise PASS")
    # row tiling: sum_d e_d (x) W_in[d,:]  — every entry appears exactly once
    row_sum = np.zeros_like(Win)
    for dd in range(d):
        piece = np.zeros_like(Win); piece[dd, :] = Win[dd, :]
        row_sum += piece
    assert np.array_equal(row_sum, Win), "row tiling not bitwise"
    lines.append("row tiling  sum_d e_d(x)W_in[d,:] == W_in: bitwise PASS")
    # column tiling
    col_sum = np.zeros_like(Win)
    for n in range(N):
        piece = np.zeros_like(Win); piece[:, n] = Win[:, n]
        col_sum += piece
    assert np.array_equal(col_sum, Win), "column tiling not bitwise"
    lines.append("column tiling sum_n W_in[:,n](x)e_n == W_in: bitwise PASS")
    # SVD reconstruction (float tolerance)
    rec("SVD tiling   max|U S V^T - W_in|", np.abs((U * S) @ V.T - Win).max(), 1e-12)
    # stored loading matrix
    rec("loadings     max|L - W_e U|", np.abs(L - We @ U).max(), 1e-12)

    # chain forwards on all one-hot probes vs full model
    Y = full_forward(We, Win)
    # row chain: pre_j = sum_d (a_j)_d * W_in[d,:]
    pre_rows = np.einsum("jd,dn->jn", We, Win)      # explicit weighted row sum
    rec("row chain    max|sum_d (a_j)_d R_d - a_j W_in|",
        np.abs(pre_rows - We @ Win).max(), 1e-12)
    Y_rows = relu(pre_rows) @ Win.T @ We.T
    rec("row chain    max|Y_rows - Y_full|", np.abs(Y_rows - Y).max(), 1e-12)
    # svd chain: pre_j = sum_k s_k (u_k.a_j) v_k  = sum_k s_k L_jk v_k
    pre_svd = (L * S) @ V.T
    rec("svd chain    max|sum_k s_k L_jk v_k - a_j W_in|",
        np.abs(pre_svd - We @ Win).max(), 1e-12)
    Y_svd = relu(pre_svd) @ Win.T @ We.T
    rec("svd chain    max|Y_svd - Y_full|", np.abs(Y_svd - Y).max(), 1e-12)
    # column chain coefficients ARE the pre-activations (tiling trivially exact)
    lines.append("column chain: coefficients (a_j W_in)_n are the pre-activations "
                 "(identical forward by construction)")
    # context: pre-activation vs designed code (G = I at d200)
    lines.append(f"context      max|W_e W_in - C| = {np.abs(We @ Win - t['C']).max():.3e}"
                 "  (G = I at d_emb=200 >= F)")
    return lines


# -------------------------------------------------------------- activity census

def census_energies(t: dict) -> dict[str, np.ndarray]:
    """Per-feature squared activation coefficients (v=1) for each ontology.
    rows: (F, d_emb); columns/svd: (F, N)."""
    We, Win, S, L = t["W_e"], t["W_in"], t["S"], t["L"]
    return {"rows": We ** 2,
            "columns": (We @ Win) ** 2,
            "svd": (L * S) ** 2}


def participation_ratio(E: np.ndarray) -> np.ndarray:
    """PR = (sum c^2)^2 / sum c^4 per feature; E holds c^2."""
    return E.sum(1) ** 2 / (E ** 2).sum(1)


def n_for_threshold(E: np.ndarray, frac: float) -> np.ndarray:
    """Per feature: smallest m s.t. the top-m energies cover >= frac of total."""
    srt = -np.sort(-E, axis=1)
    cum = np.cumsum(srt, axis=1) / srt.sum(1, keepdims=True)
    return (cum < frac).sum(1) + 1


# --------------------------------------------------- truncated-forward fidelity

def truncation_errors(t: dict) -> dict[str, dict[str, np.ndarray]]:
    """For each ontology: spike/leak relative error arrays (F, n_pieces),
    entry [j, m-1] = error of the top-m tied-transpose truncated mechanism on
    probe e_j vs the full model. Ranking by |activation coefficient|."""
    We, Win, U, S, V, L = (t[k] for k in ("W_e", "W_in", "U", "S", "V", "L"))
    F, d = We.shape
    N = Win.shape[1]
    Y = full_forward(We, Win)
    out = {}
    for ont in ONTOLOGIES:
        n_pieces = d if ont == "rows" else N
        e_spike = np.zeros((F, n_pieces))
        e_leak = np.zeros((F, n_pieces))
        for j in range(F):
            a = We[j]
            y = Y[j]
            off = np.ones(F, bool); off[j] = False
            leak_den = np.linalg.norm(y[off])
            spike_den = abs(y[j])
            if ont == "rows":
                order = np.argsort(-np.abs(a), kind="stable")
                pre_cum = np.cumsum(a[order, None] * Win[order, :], axis=0)
                mask = np.zeros(d)
                for m in range(1, n_pieces + 1):
                    mask[order[m - 1]] = 1.0
                    h = relu(pre_cum[m - 1])          # (N,)
                    z = (Win @ h) * mask              # W_out_m = Win_m.T
                    ym = z @ We.T
                    e_spike[j, m - 1] = abs(ym[j] - y[j]) / spike_den
                    e_leak[j, m - 1] = np.linalg.norm(ym[off] - y[off]) / leak_den
                mask[:] = 0.0
            elif ont == "columns":
                pre = a @ Win
                order = np.argsort(-np.abs(pre), kind="stable")
                h_full = relu(pre)
                z = np.zeros(d)
                for m in range(1, n_pieces + 1):
                    n = order[m - 1]
                    z = z + h_full[n] * Win[:, n]     # disjoint pieces: pure cumsum
                    ym = z @ We.T
                    e_spike[j, m - 1] = abs(ym[j] - y[j]) / spike_den
                    e_leak[j, m - 1] = np.linalg.norm(ym[off] - y[off]) / leak_den
            else:                                     # svd
                alpha = S * L[j]                      # s_k (u_k . a_j)
                order = np.argsort(-np.abs(alpha), kind="stable")
                pre_cum = np.cumsum(alpha[order, None] * V[:, order].T, axis=0)
                for m in range(1, n_pieces + 1):
                    ks = order[:m]
                    h = relu(pre_cum[m - 1])
                    z = U[:, ks] @ (S[ks] * (h @ V[:, ks]))
                    ym = z @ We.T
                    e_spike[j, m - 1] = abs(ym[j] - y[j]) / spike_den
                    e_leak[j, m - 1] = np.linalg.norm(ym[off] - y[off]) / leak_den
        out[ont] = {"spike": e_spike, "leak": e_leak}
    return out


def m_star(err: np.ndarray, tol: float) -> np.ndarray:
    """Per feature: smallest m s.t. err(m') <= tol for ALL m' >= m (persistent
    crossing, robust to ReLU-induced non-monotonicity). Sentinel n_pieces+1 if
    never satisfied."""
    F, M = err.shape
    bad = err > tol
    ms = np.full(F, 1, dtype=int)
    for j in range(F):
        idx = np.nonzero(bad[j])[0]
        ms[j] = (idx[-1] + 2) if idx.size else 1
    return ms


# ------------------------------------------------------------------- reporting

def fmt_int_stats(v: np.ndarray) -> str:
    return f"{int(np.median(v))} ({v.min()}–{v.max()})"


def write_results(t, assert_lines, E, pr, nth, trunc, mstats) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    cfg = t["cfg"]
    # ---- activity.csv (long format, per feature per ontology)
    csv_path = os.path.join(RES_DIR, "activity.csv")
    cols = ["ontology", "j", "pr", "n50", "n90", "n99", "m_spike_1pct", "m_leak_10pct"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for ont in ONTOLOGIES:
            for j in range(t["W_e"].shape[0]):
                f.write(f"{ont},{j},{pr[ont][j]:.4f},{nth[ont][0.50][j]},"
                        f"{nth[ont][0.90][j]},{nth[ont][0.99][j]},"
                        f"{mstats[ont]['spike'][j]},{mstats[ont]['leak'][j]}\n")
    # ---- census_d200.md
    md = [f"# Mechanism forward-pass census — d_emb = {D_EMB} (designed sandwich, "
          "transpose decoder)", "",
          f"Checkpoint: `checkpoints/embedded_ideal/ideal_d{D_EMB}.pt`  "
          f"(K={cfg['K']}, p={cfg['p']}, a={cfg['a']:.4f}, b={cfg['b']:.4f}).  "
          "Probes: one-hot v·e_j, v=1 (errors are v-invariant).  "
          "Option A (activation-direction projections) rejected: it uses W_emb "
          "knowledge at inference time; the decomposition must be fixed before "
          "inference. All ontologies below are exact tilings of W_in driven only "
          "by chain activations.", "",
          "## Exactness asserts (identities, not findings)", ""]
    md += ["```"] + assert_lines + ["```", ""]
    md += ["## Activity census (no cutoffs: PR + multi-threshold counts)", "",
           "Energies per feature j: rows (W_e[j])_d^2 over 200 dims; columns "
           "((a_j W_in))_n^2 over 50 neurons; SVD (s_k L_jk)^2 over 50 channels.",
           "",
           "| ontology | pieces | PR median (min–max) | n50 med (min–max) | "
           "n90 med (min–max) | n99 med (min–max) |",
           "|---|---|---|---|---|---|"]
    for ont in ONTOLOGIES:
        n_pieces = E[ont].shape[1]
        p = pr[ont]
        md.append(f"| {ONT_LABEL[ont]} | {n_pieces} | "
                  f"{np.median(p):.1f} ({p.min():.1f}–{p.max():.1f}) | "
                  + " | ".join(fmt_int_stats(nth[ont][th]) for th in THRESHOLDS)
                  + " |")
    md += ["", "## Truncated-forward fidelity (tied-transpose top-m mechanism)",
           "",
           "m* = smallest m with the error persistently below tolerance for all "
           "m' >= m (spike: |y_m[j]-y[j]|/|y[j]| <= 1%; leak: "
           "||(y_m-y)_off||/||y_off|| <= 10%).", "",
           "| ontology | m* spike<=1% med (min–max) | m* leak<=10% med (min–max) |",
           "|---|---|---|"]
    for ont in ONTOLOGIES:
        md.append(f"| {ONT_LABEL[ont]} | {fmt_int_stats(mstats[ont]['spike'])} | "
                  f"{fmt_int_stats(mstats[ont]['leak'])} |")
    # verdict
    med_leak = {ont: int(np.median(mstats[ont]["leak"])) for ont in ONTOLOGIES}
    best = min(med_leak, key=med_leak.get)
    md += ["", "## Verdict", "",
           f"Most causally concentrated ontology: **{ONT_LABEL[best]}** "
           f"(median m* for faithful spike+leak: "
           + ", ".join(f"{ONT_LABEL[o]} = {med_leak[o]}" for o in ONTOLOGIES)
           + "). The column (per-neuron) census spreads pre-activation energy "
             "over the b-floor (n99 near all 50), but the ReLU kills every "
             "off-code (negative) pre-activation, so causally only the K on-code "
             "pieces carry the mechanism — the census and the causal count "
             "disagree, and the truncated-forward fidelity is the discriminator.",
           ""]
    md_path = os.path.join(RES_DIR, "census_d200.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {md_path}\nsaved {csv_path}")
    print("\n".join(md))


# --------------------------------------------------------------------- figures

def plot_activity_curves(E, pr, nth) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    for c, ont in enumerate(ONTOLOGIES):
        srt = -np.sort(-E[ont], axis=1)
        share = srt / srt.sum(1, keepdims=True)
        cum = np.cumsum(share, axis=1)
        x = np.arange(1, share.shape[1] + 1)
        ax = axes[0, c]
        for j in range(share.shape[0]):
            ax.plot(x, share[j], color="0.6", lw=0.5, alpha=0.25)
        ax.plot(x, np.median(share, axis=0), color=ONT_COLOR[ont], lw=2,
                label="median feature")
        ax.set_yscale("log")
        ax.set_title(f"{ONT_LABEL[ont]} — sorted energy share\n"
                     f"PR median {np.median(pr[ont]):.1f} "
                     f"({pr[ont].min():.1f}–{pr[ont].max():.1f})", fontsize=10)
        ax.set_xlabel("component rank"); ax.legend(fontsize=8)
        if c == 0:
            ax.set_ylabel("energy share (log)")
        ax = axes[1, c]
        for j in range(cum.shape[0]):
            ax.plot(x, cum[j], color="0.6", lw=0.5, alpha=0.25)
        ax.plot(x, np.median(cum, axis=0), color=ONT_COLOR[ont], lw=2)
        for th in THRESHOLDS:
            ax.axhline(th, color="k", lw=0.6, ls="--")
            ax.annotate(f" {int(th*100)}%: n={int(np.median(nth[ont][th]))}",
                        (x[-1], th), ha="right", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("top-m components")
        if c == 0:
            ax.set_ylabel("cumulative energy fraction")
    fig.suptitle(f"Activity census — per-feature sorted activation-energy curves, "
                 f"all 100 features (d_emb={D_EMB}, one-hot probes)", fontsize=12)
    path = os.path.join(FIG_DIR, f"activity_curves_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_truncation_fidelity(trunc) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    floor = 1e-17
    for ax, kind, tol, tol_lbl in ((axes[0], "spike", SPIKE_TOL, "1% tol"),
                                   (axes[1], "leak", LEAK_TOL, "10% tol")):
        for ont in ONTOLOGIES:
            err = np.maximum(trunc[ont][kind], floor)
            x = np.arange(1, err.shape[1] + 1)
            ax.plot(x, np.median(err, axis=0), color=ONT_COLOR[ont], lw=2,
                    label=f"{ONT_LABEL[ont]} (median)")
            ax.plot(x, err.max(axis=0), color=ONT_COLOR[ont], lw=1, ls="--",
                    label=f"{ONT_LABEL[ont]} (worst feature)")
        ax.axhline(tol, color="k", lw=0.8, ls=":")
        ax.annotate(f" {tol_lbl}", (1, tol), va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("top-m components kept (tied W_in / W_out truncation)")
        ax.set_title(f"{kind} error vs m", fontsize=11)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("relative error (log; floored at 1e-17)")
    fig.suptitle(f"Truncated-mechanism fidelity vs full model — one-hot probes, "
                 f"all 100 features (d_emb={D_EMB})", fontsize=12)
    path = os.path.join(FIG_DIR, f"truncation_fidelity_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def truncated_output(t, ont: str, j: int, m: int) -> np.ndarray:
    """Top-m tied-transpose truncated mechanism output on e_j (v=1)."""
    We, Win, U, S, V, L = (t[k] for k in ("W_e", "W_in", "U", "S", "V", "L"))
    a = We[j]
    if ont == "rows":
        order = np.argsort(-np.abs(a), kind="stable")[:m]
        mask = np.zeros(We.shape[1]); mask[order] = 1.0
        h = relu((a * mask) @ Win)
        return ((Win @ h) * mask) @ We.T
    if ont == "columns":
        pre = a @ Win
        order = np.argsort(-np.abs(pre), kind="stable")[:m]
        h = relu(pre)
        return (Win[:, order] @ h[order]) @ We.T
    alpha = S * L[j]
    order = np.argsort(-np.abs(alpha), kind="stable")[:m]
    h = relu(alpha[order] @ V[:, order].T)
    return (U[:, order] @ (S[order] * (h @ V[:, order]))) @ We.T


def plot_spectra_examples(t, trunc, js: list[int]) -> None:
    """Established single-component spectra style: bar per output feature,
    target red with probe arrow, off-target bars colored by codeword overlap
    m_ov; per example feature one panel for the full model + one per ontology
    at the EQUAL budget m = M_SHOW (= K) so the ontologies contrast visually."""
    os.makedirs(FIG_DIR, exist_ok=True)
    We, Win, M = t["W_e"], t["W_in"], t["M"]
    F = We.shape[0]
    ov = (M.astype(int) @ M.astype(int).T)
    Y = full_forward(We, Win)
    m_colors = {0: "steelblue", 1: "tab:orange", 2: "tab:purple", 3: "tab:brown"}
    ncol = 1 + len(ONTOLOGIES)
    fig, axes = plt.subplots(len(js), ncol, figsize=(4.5 * ncol, 3.3 * len(js)),
                             constrained_layout=True, sharey="row")
    axes = np.atleast_2d(axes)
    for r, j in enumerate(js):
        colors = ["tab:red" if i == j else m_colors.get(int(ov[j, i]), "tab:green")
                  for i in range(F)]
        panels = [(Y[j], f"full model  (s = {Y[j, j]:.3f} + leak)")]
        for ont in ONTOLOGIES:
            ym = truncated_output(t, ont, j, M_SHOW)
            sp = trunc[ont]["spike"][j, M_SHOW - 1]
            lk = trunc[ont]["leak"][j, M_SHOW - 1]
            panels.append((ym, f"{ONT_LABEL[ont]}, top-{M_SHOW}\n"
                               f"spike err {sp:.1e}, leak err {lk:.1e}"))
        for c, (spec, ttl) in enumerate(panels):
            ax = axes[r, c]
            ax.bar(range(F), spec, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            ax.annotate(f"input\ne_{j}", (j, float(spec[j])), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="tab:red",
                        arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.8))
            ax.set_title(ttl, fontsize=9)
            if r == len(js) - 1:
                ax.set_xlabel("output feature")
            if c == 0:
                ax.set_ylabel("output value")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col)
               for col in ("tab:red", "steelblue", "tab:orange", "tab:purple")]
    axes[0, 0].legend(handles, ["target j", "overlap 0", "overlap 1", "overlap 2"],
                      loc="upper right", fontsize=7.5)
    fig.suptitle(f"Output spectra on one-hot probes — full model vs equal-budget "
                 f"top-{M_SHOW} truncated mechanisms (d_emb={D_EMB}; off-target "
                 f"bars colored by codeword overlap)", fontsize=12)
    path = os.path.join(FIG_DIR, f"spectra_examples_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# ------------------------------------------------------------------------ main

def main() -> None:
    t = load()
    print(f"loaded {CKPT}  cfg={t['cfg']}")
    print("\n== exactness asserts ==")
    assert_lines = run_asserts(t)
    print("\n".join(assert_lines))

    # census
    E = census_energies(t)
    pr = {ont: participation_ratio(E[ont]) for ont in ONTOLOGIES}
    nth = {ont: {th: n_for_threshold(E[ont], th) for th in THRESHOLDS}
           for ont in ONTOLOGIES}
    # cross-checks (verification step 2)
    alpha_direct = t["S"] * (t["W_e"] @ t["U"])        # s_k (u_k . a_j) directly
    pr_direct = participation_ratio(alpha_direct ** 2)
    assert np.abs(pr_direct - pr["svd"]).max() < 1e-9, "SVD PR cross-check failed"
    print("\ncross-check: SVD PR from stored L == PR from direct s_k(u_k.a_j): "
          f"max diff {np.abs(pr_direct - pr['svd']).max():.2e} PASS")
    print("cross-check: row PR == PR of W_e[j] entries (embedding density) by "
          f"construction; median {np.median(pr['rows']):.1f} of {D_EMB} dims")

    # truncation fidelity
    print("\ncomputing truncated-forward fidelity ...")
    trunc = truncation_errors(t)
    mstats = {ont: {"spike": m_star(trunc[ont]["spike"], SPIKE_TOL),
                    "leak": m_star(trunc[ont]["leak"], LEAK_TOL)}
              for ont in ONTOLOGIES}

    write_results(t, assert_lines, E, pr, nth, trunc, mstats)

    # figures
    plot_activity_curves(E, pr, nth)
    plot_truncation_fidelity(trunc)
    # example features: j=0; the max-overlap feature; the worst SVD-n99 feature
    M = t["M"].astype(int)
    ov = M @ M.T; np.fill_diagonal(ov, 0)
    j_ov = int(np.argmax(ov.max(axis=1)))
    j_svd = int(np.argmax(nth["svd"][0.99]))
    js = sorted({0, j_ov, j_svd})
    print(f"example features for spectra: {js} (0; max-overlap {j_ov}; "
          f"worst svd n99 {j_svd})")
    plot_spectra_examples(t, trunc, js)


if __name__ == "__main__":
    main()
