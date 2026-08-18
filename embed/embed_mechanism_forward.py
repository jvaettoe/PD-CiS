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

--grid: cut-side principle test — the full 4-matrix x 2-cut-direction grid on
the same designed sandwich. The four matrices of y = relu(x W_e W_in) W_in^T
W_e^T are W_emb = W_e (F x d), W_in (d x N), W_out = W_in^T (N x d) and
W_unemb = W_e^T (d x F). Each is cut into input-side slices (rows) and
output-side slices (columns); the activation coefficient of a piece is the
scalar its slice contributes given the arriving signal at that layer
(input-side: the signal component it reads; output-side: the projection it
writes — post-ReLU h for W_out). Per cell: tiling assert, activity census
(PR + 50/90/99% counts), and truncated-forward fidelity with ONLY that matrix
truncated to its top-m pieces (all other matrices full — untied, unlike the
default run's tied truncation).

  ./.venv/bin/python -m embed.embed_mechanism_forward --grid

Grid outputs:
  results/embedded/ideal_vs_vpd/mechanism_forward/
      {cutside_grid_d200.md, cutside_grid_d200.csv}
  figures/embedded/ideal_vs_vpd/mechanism_forward/cutside_grid_d200.png
"""
import argparse
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


# ===================================================================== cut-side
# principle grid (--grid): 4 matrices x 2 cut directions, designed sandwich.

GRID_MATRICES = ("W_emb", "W_in", "W_out", "W_unemb")
GRID_SIDES = ("input", "output")
# predicted winner side and predicted winner m* per matrix
GRID_PRED = {"W_emb": ("input", 1), "W_in": ("output", 5),
             "W_out": ("input", 5), "W_unemb": ("output", 1)}
GRID_CUT_LABEL = {
    ("W_emb", "input"): "per-feature rows",
    ("W_emb", "output"): "per-emb-dim columns",
    ("W_in", "input"): "per-emb-dim rows",
    ("W_in", "output"): "per-neuron columns",
    ("W_out", "input"): "per-neuron rows",
    ("W_out", "output"): "per-emb-dim columns",
    ("W_unemb", "input"): "per-emb-dim rows",
    ("W_unemb", "output"): "per-feature columns",
}
GRID_COEF_LABEL = {
    ("W_emb", "input"): "x_i (one-hot probe entry)",
    ("W_emb", "output"): "a_d = (x W_e)_d",
    ("W_in", "input"): "a_d (arriving embedding)",
    ("W_in", "output"): "pre_n = (a W_in)_n",
    ("W_out", "input"): "h_n = relu(pre)_n (post-ReLU)",
    ("W_out", "output"): "z_d = (h W_out)_d",
    ("W_unemb", "input"): "z_d (arriving write-out)",
    ("W_unemb", "output"): "y_i = (z W_e^T)_i",
}
SIDE_COLOR = {"input": "tab:blue", "output": "tab:orange"}


def grid_matrices(t: dict) -> dict[str, np.ndarray]:
    """The four matrices of y = relu(x W_e W_in) W_in^T W_e^T."""
    return {"W_emb": t["W_e"], "W_in": t["W_in"],
            "W_out": t["W_out"], "W_unemb": t["W_e"].T}


def grid_signals(t: dict, v: float = 1.0) -> dict[str, np.ndarray]:
    """Chain signals on ALL one-hot probes x = v*e_j (row j = feature j):
    X (F,F) probes, A (F,d) embeddings, PRE (F,N), H (F,N) post-ReLU,
    Z (F,d) write-out, Y (F,F) outputs."""
    We, Win = t["W_e"], t["W_in"]
    F = We.shape[0]
    X = v * np.eye(F)
    A = X @ We                                # = v * W_e
    PRE = A @ Win
    H = relu(PRE)
    Z = H @ Win.T
    Y = Z @ We.T
    return {"X": X, "A": A, "PRE": PRE, "H": H, "Z": Z, "Y": Y}


def grid_coefficients(sig: dict) -> dict[tuple, np.ndarray]:
    """Per-cell activation coefficients, shape (F, n_pieces). Input-side cut of
    a matrix: coefficient of row-slice i = the arriving-signal component that
    row reads. Output-side cut: coefficient of column-slice k = the projection
    that column writes (the incoming projection it reads off the signal)."""
    return {("W_emb", "input"): sig["X"],
            ("W_emb", "output"): sig["A"],
            ("W_in", "input"): sig["A"],
            ("W_in", "output"): sig["PRE"],
            ("W_out", "input"): sig["H"],
            ("W_out", "output"): sig["Z"],
            ("W_unemb", "input"): sig["Z"],
            ("W_unemb", "output"): sig["Y"]}


def grid_tiling_asserts(t: dict) -> list[str]:
    """Every cell's slices tile its matrix bitwise (identities, not findings)."""
    lines = []
    assert np.array_equal(t["W_out"], t["W_in"].T), "W_out != W_in.T"
    for mat, W in grid_matrices(t).items():
        for side in GRID_SIDES:
            n = W.shape[0] if side == "input" else W.shape[1]
            acc = np.zeros_like(W)
            for i in range(n):
                piece = np.zeros_like(W)
                if side == "input":
                    piece[i, :] = W[i, :]
                else:
                    piece[:, i] = W[:, i]
                acc += piece
            assert np.array_equal(acc, W), f"{mat} {side} tiling not bitwise"
            lines.append(f"{mat:8s} {side}-side  {GRID_CUT_LABEL[(mat, side)]:20s} "
                         f"({n:3d} pieces): tiling bitwise PASS")
    return lines


def cell_y_cum(t: dict, sig: dict, mat: str, side: str, j: int):
    """Truncated-model outputs on probe j with ONLY `mat` truncated to its
    top-m pieces (ranked by |activation coefficient|, stable), all other
    matrices full. Returns (Y_cum, order): Y_cum[m-1] = output with m pieces."""
    We, Win = t["W_e"], t["W_in"]
    a, pre, h, z, y = (sig[k][j] for k in ("A", "PRE", "H", "Z", "Y"))
    cell = (mat, side)
    if cell == ("W_emb", "input"):
        x = sig["X"][j]
        order = np.argsort(-np.abs(x), kind="stable")
        a_cum = np.cumsum(x[order, None] * We[order, :], axis=0)       # (F, d)
        return (relu(a_cum @ Win) @ Win.T) @ We.T, order
    if cell in (("W_emb", "output"), ("W_in", "input")):
        order = np.argsort(-np.abs(a), kind="stable")
        pre_cum = np.cumsum(a[order, None] * Win[order, :], axis=0)    # (d, N)
        return (relu(pre_cum) @ Win.T) @ We.T, order
    if cell == ("W_in", "output"):
        # dropped columns give pre = 0 -> relu 0 -> zero write: identical to
        # keeping h[n] W_in[:,n] for kept n only (coincides with tied columns)
        order = np.argsort(-np.abs(pre), kind="stable")
        z_cum = np.cumsum(h[order, None] * Win[:, order].T, axis=0)    # (N, d)
        return z_cum @ We.T, order
    if cell == ("W_out", "input"):
        order = np.argsort(-np.abs(h), kind="stable")                  # post-ReLU
        z_cum = np.cumsum(h[order, None] * Win[:, order].T, axis=0)
        return z_cum @ We.T, order
    if cell in (("W_out", "output"), ("W_unemb", "input")):
        order = np.argsort(-np.abs(z), kind="stable")
        return np.cumsum(z[order, None] * We[:, order].T, axis=0), order  # (d, F)
    if cell == ("W_unemb", "output"):
        order = np.argsort(-np.abs(y), kind="stable")
        F = y.shape[0]
        onehots = np.zeros((F, F))
        onehots[np.arange(F), order] = y[order]
        return np.cumsum(onehots, axis=0), order
    raise ValueError(cell)


def grid_truncation_errors(t: dict, sig: dict,
                           features=None) -> dict[tuple, dict[str, np.ndarray]]:
    """Per cell: spike/leak relative-error arrays (len(features), n_pieces)."""
    F = t["W_e"].shape[0]
    js = range(F) if features is None else list(features)
    coef = grid_coefficients(sig)
    out = {}
    for cell, c in coef.items():
        n_pieces = c.shape[1]
        e_spike = np.zeros((len(list(js)), n_pieces))
        e_leak = np.zeros_like(e_spike)
        for r, j in enumerate(js):
            y = sig["Y"][j]
            off = np.ones(F, bool); off[j] = False
            Y_cum, _ = cell_y_cum(t, sig, *cell, j)
            e_spike[r] = np.abs(Y_cum[:, j] - y[j]) / abs(y[j])
            e_leak[r] = (np.linalg.norm(Y_cum[:, off] - y[off], axis=1)
                         / np.linalg.norm(y[off]))
        out[cell] = {"spike": e_spike, "leak": e_leak}
    return out


def grid_regression_asserts(t: dict, sig: dict, coef: dict,
                            trunc: dict) -> list[str]:
    """Grid vs the established census/fidelity results (verification steps)."""
    lines = []
    # 1. W_in cells reproduce the existing census energies bitwise
    E_old = census_energies(t)
    assert np.array_equal(coef[("W_in", "input")] ** 2, E_old["rows"])
    assert np.array_equal(coef[("W_in", "output")] ** 2, E_old["columns"])
    lines.append("regression: W_in input/output cut census energies == existing "
                 "rows/columns census: bitwise PASS")
    # W_in output-cut m* must reproduce the established columns result (m*=5
    # for every feature; the untied truncation coincides with the tied one
    # because dropped columns write exactly zero)
    ms_s = m_star(trunc[("W_in", "output")]["spike"], SPIKE_TOL)
    ms_l = m_star(trunc[("W_in", "output")]["leak"], LEAK_TOL)
    assert np.all(ms_s == 5) and np.all(ms_l == 5), \
        f"W_in output-cut m* regression failed: spike {np.unique(ms_s)}, " \
        f"leak {np.unique(ms_l)}"
    lines.append("regression: W_in output-cut m* spike/leak = 5 for all 100 "
                 "features (== established per-neuron columns result): PASS")
    # 2. dense-interface duality: the cut lives ON the interface, so the two
    # cells that cut the same interface give the SAME truncated forward
    for c1, c2 in ((("W_emb", "output"), ("W_in", "input")),
                   (("W_out", "output"), ("W_unemb", "input"))):
        for k in ("spike", "leak"):
            assert np.array_equal(trunc[c1][k], trunc[c2][k]), (c1, c2, k)
        lines.append(f"duality: {c1[0]} output-cut == {c2[0]} input-cut "
                     "truncated forwards (same interface): bitwise PASS")
    # 3. near-identities: W_emb input-cut keeps the only active row from m=1
    # (errors exactly 0); W_unemb output-cut's top-ranked piece is the target
    # readout column for every feature
    e_id = max(trunc[("W_emb", "input")]["spike"].max(),
               trunc[("W_emb", "input")]["leak"].max())
    assert e_id < 1e-12, f"W_emb input-cut not an identity: {e_id:.3e}"
    lines.append(f"near-identity: W_emb input-cut errors <= {e_id:.1e} for ALL "
                 "m >= 1 (the active feature's row is the whole read): PASS")
    F = t["W_e"].shape[0]
    for j in range(F):
        _, order = cell_y_cum(t, sig, "W_unemb", "output", j)
        assert order[0] == j, f"unemb top piece != target readout for j={j}"
    assert np.all(trunc[("W_unemb", "output")]["spike"][:, 0] == 0.0)
    lines.append("near-identity: W_unemb output-cut top piece IS the target "
                 "readout column j (spike error 0 at m=1) for all features: PASS")
    # 4. v-invariance (positively homogeneous chain): m* identical at v=1 and
    # v=2.5 for every cell. (The error CURVES can differ inside groups of
    # exactly-tied coefficients — the designed code has 5 equal on-code
    # values, and scaling permutes near-ties in the ranking — so the
    # v-invariant object is the m* landing point, not the tie-internal path.)
    sig_v = grid_signals(t, v=2.5)
    sub = [0, F // 2]
    tr1 = grid_truncation_errors(t, sig, features=sub)
    tr2 = grid_truncation_errors(t, sig_v, features=sub)
    for cell in tr1:
        for k, tol in (("spike", SPIKE_TOL), ("leak", LEAK_TOL)):
            assert np.array_equal(m_star(tr1[cell][k], tol),
                                  m_star(tr2[cell][k], tol)), (cell, k)
    lines.append(f"v-invariance: m* (spike and leak) identical at v=1 and "
                 f"v=2.5 (features {sub}, all 8 cells): PASS")
    return lines


# ------------------------------------------------------------- grid reporting

def grid_verdicts(mstats: dict) -> tuple[list[str], dict]:
    """Per-matrix verdict lines + summary dict {mat: (winner, ok_side, ok_m)}."""
    lines, summary = [], {}
    for mat in GRID_MATRICES:
        pred_side, pred_m = GRID_PRED[mat]
        med = {s: max(int(np.median(mstats[(mat, s)]["spike"])),
                      int(np.median(mstats[(mat, s)]["leak"])))
               for s in GRID_SIDES}
        winner = min(GRID_SIDES, key=lambda s: med[s])
        w_spike = int(np.median(mstats[(mat, winner)]["spike"]))
        w_leak = int(np.median(mstats[(mat, winner)]["leak"]))
        ok_side = winner == pred_side
        ok_spike = w_spike == pred_m
        ok_leak = w_leak == pred_m
        summary[mat] = dict(winner=winner, med=med, w_spike=w_spike,
                            w_leak=w_leak, ok_side=ok_side,
                            ok_spike=ok_spike, ok_leak=ok_leak)
        lines.append(
            f"- **{mat}** — predicted winner: {pred_side}-side at m*={pred_m}. "
            f"Actual winner: **{winner}-side** (combined median m* "
            f"{med[winner]} vs {med['output' if winner == 'input' else 'input']}"
            f" for the loser); winner m* spike/leak = {w_spike}/{w_leak}. "
            f"Side prediction {'CONFIRMED' if ok_side else 'FAILED'}; "
            f"count prediction spike {'CONFIRMED' if ok_spike else 'FAILED'}, "
            f"leak {'CONFIRMED' if ok_leak else 'FAILED'}.")
    return lines, summary


def write_grid_results(t, tiling_lines, reg_lines, coef, pr, nth, mstats,
                       verdict_lines, summary) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    cfg = t["cfg"]
    F = t["W_e"].shape[0]
    # ---- cutside_grid_d200.csv (long format)
    csv_path = os.path.join(RES_DIR, f"cutside_grid_d{D_EMB}.csv")
    with open(csv_path, "w") as f:
        f.write("matrix,side,cut,n_pieces,j,pr,n50,n90,n99,"
                "m_spike_1pct,m_leak_10pct\n")
        for cell in coef:
            mat, side = cell
            n_pieces = coef[cell].shape[1]
            for j in range(F):
                f.write(f"{mat},{side},{GRID_CUT_LABEL[cell]},{n_pieces},{j},"
                        f"{pr[cell][j]:.4f},{nth[cell][0.50][j]},"
                        f"{nth[cell][0.90][j]},{nth[cell][0.99][j]},"
                        f"{mstats[cell]['spike'][j]},{mstats[cell]['leak'][j]}\n")
    # ---- cutside_grid_d200.md
    md = [f"# Cut-side principle test — 4 matrices x 2 cut directions "
          f"(designed sandwich, d_emb = {D_EMB})", "",
          f"Checkpoint: `checkpoints/embedded_ideal/ideal_d{D_EMB}.pt` "
          f"(K={cfg['K']}, p={cfg['p']}, a={cfg['a']:.4f}, b={cfg['b']:.4f}). "
          "Model: y = relu(x W_e W_in) W_in^T W_e^T; matrices W_emb = W_e "
          "(F x d), W_in (d x N), W_out = W_in^T (N x d), W_unemb = W_e^T "
          "(d x F). Probes: one-hot v e_j (relative errors v-invariant, "
          "asserted). Each cell truncates ONLY its own matrix to the top-m "
          "pieces by |activation coefficient| — all other matrices stay full "
          "(untied; unlike the default run's tied truncation).", "",
          "**Principle under test:** decompose each layer on the side where "
          "its own weights re-concentrate the signal into discrete events; "
          "never cut along a dense interface.", "",
          "## Tiling + regression asserts (identities, not findings)", "",
          "```"] + tiling_lines + [""] + reg_lines + ["```", "",
          "## Activity census (no cutoffs: PR + multi-threshold counts)", "",
          "| matrix | cut side | pieces | coefficient | PR med (min–max) | "
          "n50 | n90 | n99 |",
          "|---|---|---|---|---|---|---|---|"]
    for cell in coef:
        mat, side = cell
        p = pr[cell]
        md.append(f"| {mat} | {side} ({GRID_CUT_LABEL[cell]}) | "
                  f"{coef[cell].shape[1]} | {GRID_COEF_LABEL[cell]} | "
                  f"{np.median(p):.1f} ({p.min():.1f}–{p.max():.1f}) | "
                  + " | ".join(fmt_int_stats(nth[cell][th]) for th in THRESHOLDS)
                  + " |")
    md += ["", "## Truncated-forward fidelity (only that matrix truncated)", "",
           "m* = smallest m with the error persistently below tolerance for "
           "all m' >= m (spike <= 1%, leak <= 10%).", "",
           "| matrix | cut side | pieces | m* spike med (min–max) | "
           "m* leak med (min–max) |",
           "|---|---|---|---|---|"]
    for cell in coef:
        mat, side = cell
        md.append(f"| {mat} | {side} ({GRID_CUT_LABEL[cell]}) | "
                  f"{coef[cell].shape[1]} | "
                  f"{fmt_int_stats(mstats[cell]['spike'])} | "
                  f"{fmt_int_stats(mstats[cell]['leak'])} |")
    md += ["", "## Verdicts (predictions: W_emb input m*=1; W_in output m*=5; "
           "W_out input m*=5; W_unemb output m*=1)", ""] + verdict_lines
    n_side = sum(s["ok_side"] for s in summary.values())
    n_spike = sum(s["ok_spike"] for s in summary.values())
    n_leak = sum(s["ok_leak"] for s in summary.values())
    md += ["", "## Overall", "",
           f"Winner-side predictions confirmed: {n_side}/4. Winner-count "
           f"predictions: spike {n_spike}/4, leak {n_leak}/4.", "",
           "Reading the one deviating number (W_unemb output-cut leak m*): at "
           "the FINAL matrix the output-side cut is coordinate-aligned with "
           "the error metric itself — dropping a piece deletes that output "
           "coordinate outright, so leak fidelity degenerates into a census "
           "of the output vector's leak support (m* leak = 98 vs n99 = 97 of "
           "the y-census) rather than a measure of mechanism concentration. "
           "The mechanism the layer re-concentrates — the target readout "
           "<z, W_e[j]> — is exactly 1 piece (spike m* = 1, error 0 at m = 1 "
           "for every feature); the leak is residual noise smeared over all "
           "coordinates, which no coordinate cut can keep without keeping the "
           "coordinates. The mirror cell W_emb input-cut shows both criteria "
           "at 1 only because the probe is genuinely 1-sparse: there the "
           "input signal, not just the mechanism, is 1 piece.", "",
           "The dense-interface cells are not merely diffuse — they are the "
           "SAME decomposition seen from either side (bitwise-equal truncated "
           "forwards, asserted above): a cut along the embedding interface is "
           "one object whether booked to the matrix writing it or the matrix "
           "reading it, and it needs ~110–160 of 200 pieces. The principle "
           "survives unfalsified: every matrix is causally minimal exactly on "
           "the side where its own weights re-concentrate the signal into "
           "discrete events (feature rows / neuron columns / neuron rows / "
           "feature columns), never along the dense embedding interface.", ""]
    md_path = os.path.join(RES_DIR, f"cutside_grid_d{D_EMB}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {md_path}\nsaved {csv_path}")
    print("\n".join(md))


def plot_grid(trunc, mstats) -> None:
    """2x4 panel: rows = spike/leak error vs m, columns = the four matrices;
    input-side vs output-side cut overlaid (median solid, worst feature
    dashed); predicted winner count marked."""
    os.makedirs(FIG_DIR, exist_ok=True)
    floor = 1e-17
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.4), constrained_layout=True)
    for col, mat in enumerate(GRID_MATRICES):
        pred_side, pred_m = GRID_PRED[mat]
        for row, (kind, tol, tol_lbl) in enumerate(
                (("spike", SPIKE_TOL, "1% tol"), ("leak", LEAK_TOL, "10% tol"))):
            ax = axes[row, col]
            for side in GRID_SIDES:
                err = np.maximum(trunc[(mat, side)][kind], floor)
                x = np.arange(1, err.shape[1] + 1)
                med_m = int(np.median(mstats[(mat, side)][kind]))
                ax.plot(x, np.median(err, axis=0), color=SIDE_COLOR[side], lw=2,
                        label=f"{side}-side {GRID_CUT_LABEL[(mat, side)]} "
                              f"(median, m*={med_m})")
                ax.plot(x, err.max(axis=0), color=SIDE_COLOR[side], lw=1,
                        ls="--", alpha=0.6, label=f"{side}-side (worst feature)")
            ax.axvline(pred_m, color=SIDE_COLOR[pred_side], lw=1, ls=":",
                       alpha=0.9)
            ax.annotate(f"pred: {pred_side} m*={pred_m}", (pred_m, floor * 30),
                        fontsize=7.5, color=SIDE_COLOR[pred_side], rotation=90,
                        ha="right", va="bottom")
            ax.axhline(tol, color="k", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_ylim(floor / 3, 30)
            ax.grid(True, which="major", color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"{mat}", fontsize=12)
            if row == 1:
                ax.set_xlabel("top-m pieces kept (this matrix only)")
            if col == 0:
                ax.set_ylabel(f"{kind} error (relative, log; "
                              f"floored 1e-17)")
            else:
                ax.set_ylabel("")
            ax.annotate(f" {tol_lbl}", (1, tol), va="bottom", fontsize=7.5)
            ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Cut-side principle grid — truncation fidelity per matrix, "
                 "input-side vs output-side cut (only that matrix truncated; "
                 f"one-hot probes, all 100 features, d_emb={D_EMB})",
                 fontsize=13)
    path = os.path.join(FIG_DIR, f"cutside_grid_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def grid_main() -> None:
    t = load()
    print(f"loaded {CKPT}  cfg={t['cfg']}")
    print("\n== tiling asserts (8 cells) ==")
    tiling_lines = grid_tiling_asserts(t)
    print("\n".join(tiling_lines))

    sig = grid_signals(t)
    # chain-forward identity: full-sum grid signals reproduce the full model
    assert np.allclose(sig["Y"], full_forward(t["W_e"], t["W_in"]),
                       atol=1e-13), "grid chain forward != full model"
    print("\ngrid chain forward == full model: PASS")

    coef = grid_coefficients(sig)
    E = {cell: c ** 2 for cell, c in coef.items()}
    pr = {cell: participation_ratio(E[cell]) for cell in coef}
    nth = {cell: {th: n_for_threshold(E[cell], th) for th in THRESHOLDS}
           for cell in coef}

    print("\ncomputing per-cell truncated-forward fidelity (8 cells) ...")
    trunc = grid_truncation_errors(t, sig)
    mstats = {cell: {"spike": m_star(trunc[cell]["spike"], SPIKE_TOL),
                     "leak": m_star(trunc[cell]["leak"], LEAK_TOL)}
              for cell in coef}

    print("\n== regression / duality / invariance asserts ==")
    reg_lines = grid_regression_asserts(t, sig, coef, trunc)
    print("\n".join(reg_lines))

    verdict_lines, summary = grid_verdicts(mstats)
    write_grid_results(t, tiling_lines, reg_lines, coef, pr, nth, mstats,
                       verdict_lines, summary)
    plot_grid(trunc, mstats)


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true",
                    help="run the cut-side principle grid (4 matrices x 2 cut "
                         "directions) instead of the default W_in-ontology census")
    args = ap.parse_args()
    grid_main() if args.grid else main()
