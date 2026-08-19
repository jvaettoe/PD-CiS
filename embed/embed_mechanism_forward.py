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

--d-emb {200,50}: checkpoint selector (default 200). Loads ideal_d{D}.pt and
suffixes every output _d{D}. At d50 (d_emb < F) the effective code is
W_e W_in = G C with G = W_e W_e^T a rank-50 projector (mean G_jj = d/F = 0.5):
off-code pre-activations can go positive through G-noise, the causal count
(pre > 0) varies per feature and may exceed K, and every identity that
depends on G = I is REPORTED instead of asserted. Tiling/duality/homogeneity
asserts are d-independent and stay hard.

--spectra: mechanism solo spectra — winning-cut component convention
(W_emb row j + top-5-by-|coefficient| W_in column pieces + same-neuron W_out
row pieces + W_unemb column j; 1+5+5+1 pieces, top-5 kept for d200
comparability), example features (0, 61, 1) as in the d200 run: full-model vs
mechanism panels, per-probe spike/leak numbers, and the leak-ladder smear
analysis (off-target values grouped by codeword overlap; d50 bands vs the
d200 exact ladder, contrasted in-memory without touching d200 outputs).

  ./.venv/bin/python -m embed.embed_mechanism_forward --spectra --d-emb 50

Spectra outputs:
  results/embedded/ideal_vs_vpd/mechanism_forward/mechanism_spectra_d{D}.md
  figures/embedded/ideal_vs_vpd/mechanism_forward/mechanism_spectra_d{D}.png

--featbasis (d_emb = 200 ONLY): a FOURTH W_in ontology — the top-1 per-feature
decomposition built from the embedded feature frame. 100 pieces
P_j = W_e[j] (x) g_j with Gmat = pinv(W_e^T) W_in (rows g_j), so
sum_j P_j = W_e^T Gmat = W_in EXACTLY (W_in lies in the row space of W_e).
The activation coefficient of piece P_i on the chain signal a_j = v W_e[j] is
a_j . W_e[i] = v (W_e W_e^T)[j, i] — the GRAM ROW of the feature frame; at
d200 the algebraic identity Gmat = (W_e W_e^T)^{-1} C holds (via W_e W_in = C),
i.e. the output sides g_j are the DUAL-frame codewords, not the codewords c_j.

DELIBERATE RULE RELAXATION: unlike rows/columns/SVD, this decomposition is
CROSS-LAYER — the pieces of W_in are built from W_emb's weights — but it stays
PRE-INFERENCE: the pieces are fixed before any input is seen, so the earlier
objection to Option A (activation-direction projections computed at inference
time) does not apply. The layer-locality rule is relaxed here on purpose, per
user instruction, to test whether cross-layer knowledge buys a k=1 ontology.

Protocol: activity census in the established cutoff-free conventions (PR +
50/90/99% energy counts) plus the featbasis-specific top-1 dominance ratios
|coef_j|/max_{i!=j}|coef_i| and |coef_j|/sum_{i!=j}|coef_i|; truncated
FOUR-LAYER SOLO mechanism fidelity (W_emb row j + top-m pieces + tied
transposed pieces for W_out = W_in^T + W_unemb column j — strict readout, so
the criterion is the SPIKE within 1%), spike error vs m for m = 1..100, all
100 features; the five-panel minimal-k figure extends
mechanism_minimal_k_d200.png (same probe e_1, same style, same first-crossing
minimal-k convention — regression-asserted to reproduce its 5/27/167); cheap
cross-check of the learned VPD lin_in components against the feature frame
W_e rows vs the dual rows pinv(W_e)^T.

  ./.venv/bin/python -m embed.embed_mechanism_forward --featbasis

Featbasis outputs (append-only; nothing existing is touched):
  results/embedded/ideal_vs_vpd/mechanism_forward/featbasis_census_d200.md
  figures/embedded/ideal_vs_vpd/mechanism_forward/
      {featbasis_activity_d200.png, mechanism_minimal_k_d200_v2.png}
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


def set_demb(d: int) -> None:
    """Select the checkpoint / output suffix (called from __main__)."""
    global D_EMB, CKPT
    D_EMB = d
    CKPT = os.path.join(CHECKPOINTS, "embedded_ideal", f"ideal_d{d}.pt")


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

def load(ckpt: str | None = None):
    ck = torch.load(ckpt or CKPT, map_location="cpu", weights_only=False)
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
    # context: pre-activation vs designed (G-projected) code
    G = t["G"]
    regime = ("G = I at d_emb >= F"
              if d >= F else
              f"rank-{min(F, d)} projector — off-code preactivations carry G-noise")
    lines.append(f"context      max|W_e W_in - G C| = "
                 f"{np.abs(We @ Win - G @ t['C']).max():.3e}"
                 f"  (G = W_e W_e^T, mean G_jj = {np.diag(G).mean():.3f}; {regime})")
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
    # ---- activity CSV (long format, per feature per ontology; d200's original
    # run predates the suffix and lives at activity.csv — left untouched)
    csv_path = os.path.join(RES_DIR, "activity.csv" if D_EMB == 200
                            else f"activity_d{D_EMB}.csv")
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
           f"Energies per feature j: rows (W_e[j])_d^2 over {D_EMB} dims; columns "
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
    md_path = os.path.join(RES_DIR, f"census_d{D_EMB}.md")
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
    # W_in output-cut m*: at d_emb >= F (G = I) it must reproduce the
    # established columns result (m*=5 for every feature; the untied
    # truncation coincides with the tied one because dropped columns write
    # exactly zero). Below F this is a REGIME REPORT, not an assert: G-noise
    # sends off-code preactivations positive, so the causal count varies.
    F = t["W_e"].shape[0]
    exact_regime = t["W_e"].shape[1] >= F
    ms_s = m_star(trunc[("W_in", "output")]["spike"], SPIKE_TOL)
    ms_l = m_star(trunc[("W_in", "output")]["leak"], LEAK_TOL)
    if exact_regime:
        assert np.all(ms_s == 5) and np.all(ms_l == 5), \
            f"W_in output-cut m* regression failed: spike {np.unique(ms_s)}, " \
            f"leak {np.unique(ms_l)}"
        lines.append("regression: W_in output-cut m* spike/leak = 5 for all 100 "
                     "features (== established per-neuron columns result): PASS")
    else:
        lines.append(
            f"regime report (d_emb < F, no assert): W_in output-cut m* spike "
            f"med {int(np.median(ms_s))} ({ms_s.min()}–{ms_s.max()}), leak "
            f"med {int(np.median(ms_l))} ({ms_l.min()}–{ms_l.max()}) "
            f"(d200 reference: 5/5 for every feature)")
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
    top_is_target = np.zeros(F, bool)
    for j in range(F):
        _, order = cell_y_cum(t, sig, "W_unemb", "output", j)
        top_is_target[j] = order[0] == j
    if exact_regime:
        assert top_is_target.all(), \
            f"unemb top piece != target readout for j={np.nonzero(~top_is_target)[0]}"
        assert np.all(trunc[("W_unemb", "output")]["spike"][:, 0] == 0.0)
        lines.append("near-identity: W_unemb output-cut top piece IS the target "
                     "readout column j (spike error 0 at m=1) for all features: "
                     "PASS")
    else:
        n_top = int(top_is_target.sum())
        sp1 = trunc[("W_unemb", "output")]["spike"][top_is_target, 0]
        sp1_max = sp1.max() if sp1.size else float("nan")
        lines.append(
            f"regime report (d_emb < F, no assert): W_unemb output-cut top "
            f"piece is the target readout for {n_top}/{F} features (spike "
            f"error at m=1 is 0 for those; max {sp1_max:.1e}) "
            f"(d200 reference: {F}/{F})")
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
    """Per-matrix verdict lines + summary dict {mat: (winner, ok_side, ok_m)}.
    At d_emb < F the GRID_PRED counts are the d200 REFERENCE, not predictions
    — deviations are reported (expected knee-softening), not failures."""
    exact_regime = D_EMB >= 100
    miss = "FAILED" if exact_regime else "DEVIATES from d200 (reported)"
    pred_word = "predicted" if exact_regime else "d200-reference"
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
            f"- **{mat}** — {pred_word} winner: {pred_side}-side at m*={pred_m}. "
            f"Actual winner: **{winner}-side** (combined median m* "
            f"{med[winner]} vs {med['output' if winner == 'input' else 'input']}"
            f" for the loser); winner m* spike/leak = {w_spike}/{w_leak}. "
            f"Side prediction {'CONFIRMED' if ok_side else miss}; "
            f"count prediction spike {'CONFIRMED' if ok_spike else miss}, "
            f"leak {'CONFIRMED' if ok_leak else miss}.")
    return lines, summary


def write_grid_results(t, tiling_lines, reg_lines, coef, pr, nth, mstats,
                       verdict_lines, summary, npos=None) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    cfg = t["cfg"]
    F = t["W_e"].shape[0]
    exact_regime = t["W_e"].shape[1] >= F
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
          "never cut along a dense interface.", ""]
    if not exact_regime and npos is not None:
        G = t["G"]
        md += [
            f"**Regime (d_emb = {D_EMB} < F = {F}):** W_e W_in = G C with "
            f"G = W_e W_e^T a rank-{t['W_e'].shape[1]} projector (mean G_jj = "
            f"{np.diag(G).mean():.3f}, max |W_e W_in - G C| = "
            f"{np.abs(t['W_e'] @ t['W_in'] - G @ t['C']).max():.1e}). Off-code "
            "preactivations go positive through G-noise: the causal count "
            f"#(pre > 0) per feature is median {int(np.median(npos))} "
            f"({npos.min()}–{npos.max()}) vs exactly K = {cfg['K']} at d200, "
            f"and {int((npos != cfg['K']).sum())}/{F} features deviate from K. "
            "The d200 counts {1,5,5,1} are the REFERENCE here, not asserted "
            "predictions — knees are expected to soften.", ""]
    md += ["## Tiling + regression asserts (identities, not findings)", "",
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
    pred_word = "predictions" if exact_regime else "d200-reference counts"
    md += ["", f"## Verdicts ({pred_word}: W_emb input m*=1; W_in output m*=5; "
           "W_out input m*=5; W_unemb output m*=1)", ""] + verdict_lines
    n_side = sum(s["ok_side"] for s in summary.values())
    n_spike = sum(s["ok_spike"] for s in summary.values())
    n_leak = sum(s["ok_leak"] for s in summary.values())
    md += ["", "## Overall", "",
           f"Winner-side {pred_word} confirmed: {n_side}/4. Winner-count "
           f"matches: spike {n_spike}/4, leak {n_leak}/4.", ""]
    if not exact_regime:
        m8 = {(mat, s): (int(np.median(mstats[(mat, s)]["spike"])),
                         int(np.median(mstats[(mat, s)]["leak"])))
              for mat in GRID_MATRICES for s in GRID_SIDES}
        y_n99 = nth[("W_unemb", "output")][0.99]
        md += [
            "d50 contrast with the d200 grid: the exact d200 counts "
            "{1,5,5,1} are a G = I artifact; the re-concentration reading "
            "survives where the signal is genuinely evented (W_emb, W_out "
            "input; W_in output beats its dense-interface rival). The one "
            "side FLIP, W_unemb, is the d200-diagnosed criterion artifact "
            "now dominating the combined median: the output cut's spike m* "
            "is still 1 for every feature (the top piece IS the target "
            "readout, 100/100), but its leak m* "
            f"({int(np.median(mstats[('W_unemb', 'output')]['leak']))}, "
            f"vs y-census n99 = {int(np.median(y_n99))}) is a support census "
            "of the now-broadband G-noise leak, which at d50 exceeds the "
            "dense-interface cost. d200-reference median m* at d50: "
            f"W_emb input {m8[('W_emb', 'input')][0]}/{m8[('W_emb', 'input')][1]}"
            f" (d200: 1/1), W_in output {m8[('W_in', 'output')][0]}/"
            f"{m8[('W_in', 'output')][1]} (d200: 5/5), W_out input "
            f"{m8[('W_out', 'input')][0]}/{m8[('W_out', 'input')][1]} "
            f"(d200: 5/5), W_unemb output {m8[('W_unemb', 'output')][0]}/"
            f"{m8[('W_unemb', 'output')][1]} (d200: 1/98). W_emb input stays "
            "exact at any d (the probe is 1-sparse). For W_in/W_out the knee "
            "softens because G-noise sends off-code preactivations positive "
            "(causal count above; note the |coefficient| ranking also spends "
            "slots on large-|negative| dead pieces for the W_in output cut). "
            "Full per-feature spread is in the fidelity table above and the "
            "CSV.", ""]
        md_path = os.path.join(RES_DIR, f"cutside_grid_d{D_EMB}.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md))
        print(f"saved {md_path}\nsaved {csv_path}")
        print("\n".join(md))
        return
    md += ["Reading the one deviating number (W_unemb output-cut leak m*): at "
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
            pred_tag = "pred" if D_EMB >= 100 else "d200 ref"
            ax.annotate(f"{pred_tag}: {pred_side} m*={pred_m}", (pred_m, floor * 30),
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

    npos = (sig["PRE"] > 0).sum(1)
    print(f"\ncausal count #(pre>0) per feature: median {int(np.median(npos))} "
          f"({npos.min()}–{npos.max()}); K = {t['cfg']['K']}")

    verdict_lines, summary = grid_verdicts(mstats)
    write_grid_results(t, tiling_lines, reg_lines, coef, pr, nth, mstats,
                       verdict_lines, summary, npos=npos)
    plot_grid(trunc, mstats)


# ================================================================ solo spectra
# (--spectra): winning-cut component convention, 1 + M_SHOW + M_SHOW + 1 pieces
# per feature: W_emb row j, top-M_SHOW-by-|coefficient| W_in column pieces,
# the same-neuron W_out row pieces (tied), and W_unemb column j.

# Same example features as the d200 run: 0; 61 = a max-overlap partner of 0
# (the ov=2 tie-set of feature 0 is {2, 61, 78, 96}; 61 was the d200 pick,
# kept for comparability); 1 = the globally max-overlap feature (ov=3).
SPECTRA_FEATURES = (0, 61, 1)


def solo_mechanism(t: dict, j: int, m_keep: int = M_SHOW):
    """Winning-cut solo mechanism for feature j on probe e_j (v=1).
    Returns (keep, pre, z, y_strict, y_fullread): keep = kept neuron indices;
    y_strict = the mechanism's own spectrum (W_unemb column j writes ONLY
    output j, so off-target is structurally 0); y_fullread = the retained
    write z read through the FULL unembed bank (diagnostic: where G-noise
    through the retained pieces would land — NOT part of the mechanism)."""
    We, Win = t["W_e"], t["W_in"]
    a = We[j]                                     # W_emb row j piece
    pre = a @ Win                                 # column-piece coefficients
    keep = np.argsort(-np.abs(pre), kind="stable")[:m_keep]
    h = relu(pre)
    z = Win[:, keep] @ h[keep]                    # same-neuron W_out rows
    y_strict = np.zeros(We.shape[0])
    y_strict[j] = z @ We[j]                       # W_unemb column j
    y_fullread = z @ We.T
    return keep, pre, z, y_strict, y_fullread


def leak_ladder(t: dict, Y: np.ndarray, j: int) -> dict[int, np.ndarray]:
    """Off-target output values of the FULL model on probe e_j grouped by
    codeword overlap m_ov(j, i)."""
    M = t["M"].astype(int)
    ov = M @ M.T
    out = {}
    for m in sorted(set(int(v) for i, v in enumerate(ov[j]) if i != j)):
        idx = [i for i in range(Y.shape[1]) if i != j and ov[j, i] == m]
        out[m] = Y[j, idx]
    return out


def plot_spectra_solo(t, Y, js, mech) -> None:
    """d200-style panel: per feature a full-model spectrum (top) and the
    strict 1+5+5+1 mechanism spectrum (bottom)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    F = t["W_e"].shape[0]
    fig, axes = plt.subplots(2, len(js), figsize=(4.8 * len(js), 6.4),
                             constrained_layout=True, sharey="row")
    axes = np.atleast_2d(axes)
    for c, j in enumerate(js):
        colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
        _, _, _, y_strict, y_fullread = mech[j]
        off = np.ones(F, bool); off[j] = False
        ax = axes[0, c]
        ax.bar(range(F), Y[j], width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"e_{j} — full model", fontsize=10)
        ax = axes[1, c]
        ax.bar(range(F), y_strict, width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"e_{j} — mechanism (1+{M_SHOW}+{M_SHOW}+1 pieces)\n"
                     f"spike {y_strict[j]:.3f}; full-read leak max "
                     f"{np.abs(y_fullread[off]).max():.3f}", fontsize=9)
        ax.set_xlabel("output feature")
    axes[0, 0].set_ylabel("output")
    axes[1, 0].set_ylabel("output")
    fig.suptitle(f"Mechanism solo forward vs full model (designed sandwich "
                 f"d{D_EMB}, v=1; red = probed feature)", fontsize=12)
    path = os.path.join(FIG_DIR, f"mechanism_spectra_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def spectra_main() -> None:
    t = load()
    print(f"loaded {CKPT}  cfg={t['cfg']}")
    We, Win, G, cfg = t["W_e"], t["W_in"], t["G"], t["cfg"]
    F = We.shape[0]
    K, a_amp = cfg["K"], cfg["a"]
    exact_regime = We.shape[1] >= F
    Y = full_forward(We, Win)
    M = t["M"].astype(int)
    ov = M @ M.T; np.fill_diagonal(ov, 0)
    js = list(SPECTRA_FEATURES)
    assert ov[js[0], js[1]] == ov[js[0]].max(), "pick 2 not a max-overlap partner"
    assert ov[js[2]].max() == ov.max(), "pick 3 not globally max-overlap"

    # reference d200 model (in memory only — d200 outputs untouched)
    t_ref = None
    if not exact_regime:
        t_ref = load(os.path.join(CHECKPOINTS, "embedded_ideal", "ideal_d200.pt"))
        Y_ref = full_forward(t_ref["W_e"], t_ref["W_in"])

    mech = {j: solo_mechanism(t, j) for j in js}
    npos_all = (We @ Win > 0).sum(1)

    md = [f"# Mechanism solo spectra — d_emb = {D_EMB} (winning-cut convention, "
          f"1+{M_SHOW}+{M_SHOW}+1 pieces)", "",
          f"Checkpoint: `checkpoints/embedded_ideal/ideal_d{D_EMB}.pt` "
          f"(K={K}, p={cfg['p']}, a={a_amp:.4f}, b={cfg['b']:.4f}). "
          "Convention (per probed feature j): W_emb row j + top-"
          f"{M_SHOW}-by-|coefficient| W_in column pieces + the same-neuron "
          "W_out row pieces (tied) + W_unemb column j. Top-"
          f"{M_SHOW} kept for d200 comparability.", ""]
    if not exact_regime:
        md += [f"**Regime:** W_e W_in = G C, G a rank-{We.shape[1]} projector "
               f"(mean G_jj = {np.diag(G).mean():.3f}); the true causal count "
               f"#(pre > 0) is median {int(np.median(npos_all))} "
               f"({npos_all.min()}–{npos_all.max()}) over all {F} features — "
               f"it differs from the kept {M_SHOW}.", ""]
    md += ["Structural note: the strict convention's W_unemb piece (column j) "
           "writes ONLY output coordinate j, so the strict mechanism spectrum "
           "has off-target EXACTLY 0 by construction at any d_emb. The "
           "G-noise that flows through the retained pieces is therefore "
           "measured at the retained write z read through the FULL unembed "
           "bank (\"full-read leak\" below) — the leak the mechanism's pieces "
           "hand to the other readout columns, which the strict piece set "
           "discards.", "",
           "## Per-probe numbers", "",
           "| j | full spike | mech spike | spike err | mech off-max "
           "(strict) | mech off-max (full-read) | full off-max | causal count "
           "#(pre>0) | kept == on-code? |",
           "|---|---|---|---|---|---|---|---|---|"]
    for j in js:
        keep, pre, z, y_strict, y_fullread = mech[j]
        off = np.ones(F, bool); off[j] = False
        oncode = set(np.nonzero(M[j])[0])
        keepset = set(int(n) for n in keep)
        agree = ("yes" if keepset == oncode else
                 f"no ({len(keepset & oncode)}/{K} on-code)")
        md.append(f"| {j} | {Y[j, j]:.4f} | {y_strict[j]:.4f} | "
                  f"{abs(y_strict[j] - Y[j, j]) / abs(Y[j, j]):.2e} | 0 (structural) | "
                  f"{np.abs(y_fullread[off]).max():.4f} | "
                  f"{np.abs(Y[j, off]).max():.4f} | {int(npos_all[j])} | "
                  f"{agree} |")
    # ---- spike prediction: spike ~ K a^2 G_jj^2 (all features)
    spikes = np.array([solo_mechanism(t, j)[3][j] for j in range(F)])
    pred = K * a_amp ** 2 * np.diag(G) ** 2
    ratio = spikes / pred
    full_ratio = np.diag(Y) / pred
    md += ["", "## Prediction check: spike ≈ K·a²·G_jj² (all features)", "",
           f"K a² = {K * a_amp ** 2:.4f}. mech-spike / (K a² G_jj²): median "
           f"{np.median(ratio):.3f} ({ratio.min():.3f}–{ratio.max():.3f}); "
           f"full-spike / (K a² G_jj²): median {np.median(full_ratio):.3f} "
           f"({full_ratio.min():.3f}–{full_ratio.max():.3f}). "
           f"mech spike median {np.median(spikes):.4f} "
           f"({spikes.min():.4f}–{spikes.max():.4f}).", ""]
    # ---- leak ladder: full-model off-target values by overlap class
    md += ["## Leak ladder (full model, off-target values by codeword overlap)",
           "",
           "| j | m_ov | n | mean | std | min–max |"
           + (" d200 mean | d200 std |" if t_ref is not None else ""),
           "|---|---|---|---|---|---|" + ("---|---|" if t_ref is not None else "")]
    smear = []
    for j in js:
        lad = leak_ladder(t, Y, j)
        lad_ref = leak_ladder(t_ref, Y_ref, j) if t_ref is not None else {}
        for m, vals in sorted(lad.items()):
            row = (f"| {j} | {m} | {len(vals)} | {vals.mean():.4f} | "
                   f"{vals.std():.2e} | {vals.min():.4f}–{vals.max():.4f} |")
            if t_ref is not None and m in lad_ref:
                vr = lad_ref[m]
                row += f" {vr.mean():.4f} | {vr.std():.2e} |"
            md.append(row)
            smear.append(vals.std())
    if t_ref is not None:
        md += ["", f"Smear verdict: max within-class std at d{D_EMB} = "
               f"{max(smear):.3e} vs the d200 exact ladder (per-class std "
               "~1e-16, machine zero) — "
               + ("the ladder is SMEARED into bands."
                  if max(smear) > 1e-10 else "no smearing detected."), ""]
    md_path = os.path.join(RES_DIR, f"mechanism_spectra_d{D_EMB}.md")
    os.makedirs(RES_DIR, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {md_path}")
    print("\n".join(md))
    plot_spectra_solo(t, Y, js, mech)


# ================================================================== featbasis
# (--featbasis, d200 only): top-1 per-feature decomposition of W_in from the
# embedded feature frame. CROSS-LAYER (pieces use W_emb weights) but
# PRE-INFERENCE (fixed before any input) — layer-locality deliberately relaxed
# per user instruction; see module docstring.

FB_LABEL = "featbasis (top-1 feature-frame)"
FB_COLOR = "tab:purple"
# legacy figure regression targets: mechanism_minimal_k_d200.png, probe e_1,
# first-crossing minimal k for spike within 1%
FB_LEGACY_PROBE = 1
FB_LEGACY_K = {"columns": 5, "svd": 27, "rows": 167}


def featbasis_frames(t: dict):
    """Gram = W_e W_e^T (feature-frame Gram, NOT the checkpoint's stored G =
    W_e pinv(W_e) which is I at d200); Gmat = pinv(W_e^T) W_in, rows g_j =
    output sides of the pieces P_j = W_e[j] (x) g_j."""
    We, Win = t["W_e"], t["W_in"]
    Gram = We @ We.T
    Gmat = np.linalg.pinv(We.T) @ Win
    return Gram, Gmat


def featbasis_asserts(t: dict, Gram: np.ndarray, Gmat: np.ndarray) -> list[str]:
    """(1) exact tiling; (2) piece activation on one-hot probes = Gram row;
    (3) d200 identity Gmat = Gram^{-1} C; frame sanity (unit rows, self-top-1)."""
    We, Win, C = t["W_e"], t["W_in"], t["C"]
    F, d = We.shape
    lines = []

    def rec(name, val, tol):
        ok = val <= tol
        lines.append(f"{name}: {val:.3e}  (tol {tol:.0e}) {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name} failed: {val:.3e} > {tol:.0e}"

    # (1) tiling: sum_j W_e[j] (x) g_j = W_e^T Gmat == W_in (float tolerance —
    # W_in lies in the row space of W_e, so the projector W_e^T pinv(W_e^T)
    # fixes it)
    rec("tiling       max|W_e^T Gmat - W_in|", np.abs(We.T @ Gmat - Win).max(),
        1e-12)
    # (2) piece activation on one-hot probe a_j = v W_e[j]: coefficient of
    # piece P_i is a_j . W_e[i] = v Gram[j,i] (the Gram row), and the
    # coefficient-weighted piece sum reproduces the chain pre-activations
    for v in (1.0, 2.5):
        A = v * We                                    # all probes' embeddings
        rec(f"activation   max|A W_e^T - v Gram| (v={v})",
            np.abs(A @ We.T - v * Gram).max(), 1e-12)
        rec(f"chain        max|(v Gram) Gmat - A W_in| (v={v})",
            np.abs((v * Gram) @ Gmat - A @ Win).max(), 1e-12)
    # explicit outer-product spot check: a_j P_i = Gram[j,i] g_i for all j,
    # sampled pieces i
    for i in (0, F // 2, F - 1):
        piece = np.outer(We[i], Gmat[i])
        rec(f"piece i={i:<3d} max|W_e P_i - Gram[:,{i}] g_{i}|",
            np.abs(We @ piece - Gram[:, [i]] * Gmat[[i], :]).max(), 1e-12)
    # (3) d200 algebraic identity via W_e W_in = C
    assert d >= F, "featbasis Gram identity is d200-scoped (d_emb >= F)"
    rec("identity     max|Gmat - Gram^{-1} C|",
        np.abs(Gmat - np.linalg.solve(Gram, C)).max(), 1e-12)
    # frame sanity: unit rows (Gram diag = 1) and self-top-1 (|Gram[j,j]|
    # strictly dominates its row)
    rec("unit rows    max|diag(Gram) - 1|",
        np.abs(np.diag(Gram) - 1.0).max(), 1e-6)
    offmax = np.abs(Gram - np.diag(np.diag(Gram))).max(1)
    assert all(np.argmax(np.abs(Gram[j])) == j for j in range(F)), \
        "self piece not top-1 for some feature"
    lines.append(f"self-top-1   argmax_i |Gram[j,i]| = j for all {F} features "
                 f"(worst margin 1/max_offdiag = {(1.0 / offmax).min():.2f}): PASS")
    return lines


def featbasis_dominance(Gram: np.ndarray):
    """Top-1 dominance ratios per feature: |coef_j| vs the max and the sum of
    the off-self |coefficients| in its Gram row."""
    F = Gram.shape[0]
    diag = np.abs(np.diag(Gram))
    off = np.abs(Gram).copy()
    off[np.arange(F), np.arange(F)] = 0.0
    return diag / off.max(1), diag / off.sum(1)


def solo_spike_curve(t: dict, ont: str, j: int,
                     Gram: np.ndarray | None = None,
                     Gmat: np.ndarray | None = None) -> np.ndarray:
    """Four-layer SOLO mechanism spike vs m on probe e_j (v=1): W_emb row j +
    top-m pieces of the ontology (ranked by |activation coefficient|, stable)
    + the tied transposed pieces for W_out = W_in^T + W_unemb column j.
    The strict W_unemb column-j readout writes ONLY coordinate j, so the solo
    output is spike-only; returns spike[m-1] for m = 1..n_pieces."""
    We, Win, U, S, V, L = (t[k] for k in ("W_e", "W_in", "U", "S", "V", "L"))
    d = We.shape[1]
    N = Win.shape[1]
    a = We[j]
    if ont == "rows":
        order = np.argsort(-np.abs(a), kind="stable")
        pre = np.cumsum(a[order, None] * Win[order, :], axis=0)     # (d, N)
        h = relu(pre)
        return np.array([(h[m - 1] @ Win[order[:m]].T
                          * We[j, order[:m]]).sum() for m in range(1, d + 1)])
    if ont == "columns":
        pre = a @ Win
        order = np.argsort(-np.abs(pre), kind="stable")
        return np.cumsum(relu(pre)[order] * (Win[:, order].T @ We[j]))
    if ont == "svd":
        alpha = S * L[j]
        order = np.argsort(-np.abs(alpha), kind="stable")
        pre = np.cumsum(alpha[order, None] * V[:, order].T, axis=0)
        h = relu(pre)
        wj = We[j] @ U
        return np.array([(S[order[:m]] * (h[m - 1] @ V[:, order[:m]])
                          * wj[order[:m]]).sum() for m in range(1, N + 1)])
    assert ont == "featbasis" and Gram is not None and Gmat is not None
    g = Gram[j]
    order = np.argsort(-np.abs(g), kind="stable")
    pre = np.cumsum(g[order, None] * Gmat[order, :], axis=0)        # (F, N)
    h = relu(pre)
    return np.array([(h[m - 1] @ Gmat[order[:m]].T
                      * Gram[j, order[:m]]).sum()
                     for m in range(1, Gram.shape[0] + 1)])


def k_first(err: np.ndarray, tol: float) -> int:
    """Smallest m with err[m-1] <= tol (FIRST crossing — the legacy
    mechanism_minimal_k_d200.png convention). Sentinel n_pieces+1 if never."""
    idx = np.nonzero(err <= tol)[0]
    return int(idx[0] + 1) if idx.size else err.shape[0] + 1


def featbasis_vpd_crosscheck(t: dict, Gram: np.ndarray):
    """Best-|cos| of the learned VPD lin_in components (V side, (d_in, C))
    against the feature frame W_e rows vs the dual rows pinv(W_e)^T."""
    vpd_path = os.path.join("exports", "goodfire", "out", "decompositions",
                            "p-832f0310", "model_10000.pth")
    sd = torch.load(vpd_path, map_location="cpu")
    Vc = sd["_components.lin_in.V"].double().numpy()          # (d_in, C)
    Vc = Vc / np.linalg.norm(Vc, axis=0, keepdims=True)
    We = t["W_e"]
    feat = We / np.linalg.norm(We, axis=1, keepdims=True)     # (F, d)
    dual = np.linalg.solve(Gram, We)                          # pinv(W_e)^T rows
    dualn = dual / np.linalg.norm(dual, axis=1, keepdims=True)
    cos_f = np.abs(feat @ Vc)                                 # (F, C)
    cos_d = np.abs(dualn @ Vc)
    out = {}
    for name, cos in (("feature", cos_f), ("dual", cos_d)):
        best = cos.max(0)
        out[name] = {"best": best, "match": cos.argmax(0),
                     "med": float(np.median(best)),
                     "min": float(best.min()), "max": float(best.max())}
    out["same_match"] = int((out["feature"]["match"]
                             == out["dual"]["match"]).sum())
    out["frame_cos"] = np.abs((feat * dualn).sum(1))          # cos(W_e[j], dual_j)
    return out


def plot_featbasis_activity(E_fb, pr_fb, nth_fb, dom_max, dom_sum) -> None:
    """Census figure in the activity-curves style: sorted energy share,
    cumulative energy with 50/90/99 markers, and the top-1 dominance ratios."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    srt = -np.sort(-E_fb, axis=1)
    share = srt / srt.sum(1, keepdims=True)
    cum = np.cumsum(share, axis=1)
    x = np.arange(1, share.shape[1] + 1)
    ax = axes[0]
    for jrow in share:
        ax.plot(x, jrow, color="0.6", lw=0.5, alpha=0.25)
    ax.plot(x, np.median(share, axis=0), color=FB_COLOR, lw=2,
            label="median feature")
    ax.set_yscale("log")
    ax.set_title(f"{FB_LABEL} — sorted Gram-row energy share\n"
                 f"PR median {np.median(pr_fb):.1f} "
                 f"({pr_fb.min():.1f}–{pr_fb.max():.1f})", fontsize=10)
    ax.set_xlabel("piece rank"); ax.set_ylabel("energy share (log)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for jrow in cum:
        ax.plot(x, jrow, color="0.6", lw=0.5, alpha=0.25)
    ax.plot(x, np.median(cum, axis=0), color=FB_COLOR, lw=2)
    for th in THRESHOLDS:
        ax.axhline(th, color="k", lw=0.6, ls="--")
        ax.annotate(f" {int(th * 100)}%: n={int(np.median(nth_fb[th]))}",
                    (x[-1], th), ha="right", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("top-m pieces"); ax.set_ylabel("cumulative energy fraction")
    ax.set_title("cumulative Gram-row energy", fontsize=10)
    ax = axes[2]
    ax.scatter(dom_max, dom_sum, s=14, color=FB_COLOR, alpha=0.7)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.annotate(" ratio 1 (self = off-self sum)", (dom_max.max(), 1.0),
                ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("|coef_j| / max_{i≠j} |coef_i|")
    ax.set_ylabel("|coef_j| / sum_{i≠j} |coef_i|")
    ax.set_title("top-1 dominance per feature\n"
                 f"vs-max median {np.median(dom_max):.2f}, "
                 f"vs-sum median {np.median(dom_sum):.3f}", fontsize=10)
    fig.suptitle(f"Featbasis activity census — per-feature Gram-row "
                 f"coefficient distributions, all 100 features "
                 f"(d_emb={D_EMB}, one-hot probes)", fontsize=12)
    path = os.path.join(FIG_DIR, f"featbasis_activity_d{D_EMB}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_minimal_k_v2(t, Gram, Gmat, curves_j) -> None:
    """Five-panel extension of mechanism_minimal_k_d200.png: full model,
    columns, SVD, rows, featbasis — each ontology at its own MINIMAL k
    (first crossing of spike within 1%), same probe e_1, same style."""
    os.makedirs(FIG_DIR, exist_ok=True)
    j = FB_LEGACY_PROBE
    F = t["W_e"].shape[0]
    Y = full_forward(t["W_e"], t["W_in"])
    y = Y[j, j]
    panels = [("full model", None, Y[j])]
    for ont, name in (("columns", "columns"), ("svd", "SVD"),
                      ("rows", "rows"), ("featbasis", "featbasis")):
        curve = curves_j[ont]
        k = k_first(np.abs(curve - y) / abs(y), SPIKE_TOL)
        spec = np.zeros(F); spec[j] = curve[k - 1]
        panels.append((name, k, spec))
    fig, axes = plt.subplots(1, 5, figsize=(22.5, 4.2), constrained_layout=True,
                             sharey=False)
    for ax, (name, k, spec) in zip(axes, panels):
        colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
        ax.bar(range(F), spec, width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        if k is None:
            ax.set_title(f"full model — spike {y:.4f}", fontsize=10)
        else:
            ax.set_title(f"{name}: minimal k = {k} pieces — spike "
                         f"{spec[j]:.4f} ({100 * spec[j] / y:.1f}% of full)",
                         fontsize=10)
        ax.set_xlabel("output feature")
    axes[0].set_ylabel("output")
    fig.suptitle(r"Four-layer solo mechanism for probe $e_1$"
                 f" (d{D_EMB}): each W_in ontology at its own MINIMAL k for "
                 "spike within 1% — now including featbasis "
                 "(top-1 feature-frame, cross-layer pre-inference)",
                 fontsize=12)
    path = os.path.join(FIG_DIR, f"mechanism_minimal_k_d{D_EMB}_v2.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def featbasis_main() -> None:
    assert D_EMB == 200, "--featbasis is scoped to d_emb = 200 only"
    t = load()
    print(f"loaded {CKPT}  cfg={t['cfg']}")
    We, Win = t["W_e"], t["W_in"]
    F = We.shape[0]
    cfg = t["cfg"]
    Gram, Gmat = featbasis_frames(t)

    print("\n== featbasis exactness asserts ==")
    assert_lines = featbasis_asserts(t, Gram, Gmat)
    print("\n".join(assert_lines))

    # ---- census: fourth ontology beside rows/columns/svd
    E_old = census_energies(t)
    E = dict(E_old, featbasis=Gram ** 2)
    onts = ONTOLOGIES + ("featbasis",)
    labels = dict(ONT_LABEL, featbasis=FB_LABEL)
    pr = {o: participation_ratio(E[o]) for o in onts}
    nth = {o: {th: n_for_threshold(E[o], th) for th in THRESHOLDS}
           for o in onts}
    dom_max, dom_sum = featbasis_dominance(Gram)

    # ---- four-layer solo spike fidelity, featbasis all features
    print("\ncomputing four-layer solo spike curves ...")
    Y = full_forward(We, Win)
    fb_curves = np.stack([solo_spike_curve(t, "featbasis", j, Gram, Gmat)
                          for j in range(F)])                    # (F, F)
    fb_err = np.abs(fb_curves - np.diag(Y)[:, None]) / np.abs(np.diag(Y))[:, None]
    fb_kfirst = np.array([k_first(fb_err[j], SPIKE_TOL) for j in range(F)])
    fb_kpers = m_star(fb_err, SPIKE_TOL)

    # probe-e_1 curves for the legacy ontologies + regression vs the existing
    # figure's minimal-k values
    j1 = FB_LEGACY_PROBE
    curves_j = {ont: solo_spike_curve(t, ont, j1) for ont in ONTOLOGIES}
    curves_j["featbasis"] = fb_curves[j1]
    y1 = Y[j1, j1]
    reg_lines = []
    for ont, k_ref in FB_LEGACY_K.items():
        k = k_first(np.abs(curves_j[ont] - y1) / abs(y1), SPIKE_TOL)
        assert k == k_ref, f"legacy minimal-k regression {ont}: {k} != {k_ref}"
        reg_lines.append(f"regression: probe e_{j1} solo minimal k ({ont}) = "
                         f"{k} == mechanism_minimal_k_d{D_EMB}.png value: PASS")
    print("\n".join(reg_lines))
    fb_k1 = int(fb_kfirst[j1])

    # cancellation diagnostic: dual-amplified piece norms
    piece_norms = np.linalg.norm(Gmat, axis=1)          # ||P_j||_F = ||g_j||
    code_norms = np.linalg.norm(t["C"], axis=1)
    amp = piece_norms / code_norms
    cancel = piece_norms.sum() / np.linalg.norm(Win)
    spike1_err = fb_err[:, 0]

    # ---- VPD cross-check
    xc = featbasis_vpd_crosscheck(t, Gram)
    xline = (f"VPD lin_in best-|cos|: vs feature frame W_e rows median "
             f"{xc['feature']['med']:.4f} ({xc['feature']['min']:.4f}–"
             f"{xc['feature']['max']:.4f}); vs dual rows pinv(W_e)^T median "
             f"{xc['dual']['med']:.4f} ({xc['dual']['min']:.4f}–"
             f"{xc['dual']['max']:.4f}); same best-match feature for "
             f"{xc['same_match']}/100 components; frame-vs-dual "
             f"cos(W_e[j], dual_j) median {np.median(xc['frame_cos']):.4f} "
             f"({xc['frame_cos'].min():.4f}–{xc['frame_cos'].max():.4f})")
    print("\n" + xline)

    # ---- markdown
    os.makedirs(RES_DIR, exist_ok=True)
    md = [f"# Featbasis census — the top-1 feature-frame W_in ontology "
          f"(d_emb = {D_EMB}, designed sandwich)", "",
          f"Checkpoint: `checkpoints/embedded_ideal/ideal_d{D_EMB}.pt` "
          f"(K={cfg['K']}, p={cfg['p']}, a={cfg['a']:.4f}, b={cfg['b']:.4f}). "
          "Construction: 100 pieces P_j = W_e[j] (x) g_j with "
          "Gmat = pinv(W_e^T) W_in (rows g_j), sum_j P_j = W_in exactly; the "
          "activation coefficient of piece P_i on probe v e_j is "
          "v (W_e W_e^T)[j, i] — the Gram row. At d200, "
          "Gmat = (W_e W_e^T)^{-1} C: the output sides are the DUAL-frame "
          "codewords. CROSS-LAYER (pieces use W_emb weights) but "
          "PRE-INFERENCE (fixed before any input) — the layer-locality rule "
          "is deliberately relaxed here, per user instruction; the Option-A "
          "objection (inference-time projections) does not apply.", "",
          "## Exactness asserts (identities, not findings)", "",
          "```"] + assert_lines + [""] + reg_lines + ["```", "",
          "## Activity census — featbasis as a FOURTH ontology "
          "(no cutoffs: PR + multi-threshold counts)", "",
          "Featbasis energies per feature j: (W_e W_e^T)[j, i]^2 over 100 "
          "pieces (self coefficient exactly 1; off-self = feature-frame "
          "Gram noise, max "
          f"{np.abs(Gram - np.diag(np.diag(Gram))).max():.3f}, mean |.| "
          f"{np.abs(Gram - np.diag(np.diag(Gram))).mean():.3f}).", "",
          "| ontology | pieces | PR median (min–max) | n50 med (min–max) | "
          "n90 med (min–max) | n99 med (min–max) |",
          "|---|---|---|---|---|---|"]
    for o in onts:
        p = pr[o]
        md.append(f"| {labels[o]} | {E[o].shape[1]} | "
                  f"{np.median(p):.1f} ({p.min():.1f}–{p.max():.1f}) | "
                  + " | ".join(fmt_int_stats(nth[o][th]) for th in THRESHOLDS)
                  + " |")
    md += ["",
           "Top-1 dominance (featbasis only): |coef_j|/max_{i≠j}|coef_i| "
           f"median {np.median(dom_max):.2f} ({dom_max.min():.2f}–"
           f"{dom_max.max():.2f}); |coef_j|/sum_{{i≠j}}|coef_i| median "
           f"{np.median(dom_sum):.3f} ({dom_sum.min():.3f}–{dom_sum.max():.3f})"
           " — the self coefficient beats every single rival but NOT their "
           "sum (ratio-vs-sum < 1 for "
           f"{int((dom_sum < 1).sum())}/100 features).", "",
           "## Truncated four-layer solo fidelity (spike within 1%)", "",
           "Protocol: W_emb row j + top-m featbasis pieces (by |Gram[j, i]|; "
           "top-1 is always the self piece) + the tied transposed pieces for "
           "W_out = W_in^T + W_unemb column j. The strict column-j readout "
           "writes only coordinate j, so the criterion is the spike error "
           "|y_m[j] - y[j]| / |y[j]| <= 1%; m = 1..100, all 100 features.", "",
           "| ontology (probe-e_1 legacy figure / all-feature featbasis) | "
           "minimal k (first crossing) | m* (persistent) |",
           "|---|---|---|",
           f"| columns (probe e_1) | {FB_LEGACY_K['columns']} | — |",
           f"| SVD (probe e_1) | {FB_LEGACY_K['svd']} | — |",
           f"| rows (probe e_1) | {FB_LEGACY_K['rows']} | — |",
           f"| featbasis (probe e_1) | {fb_k1} | {int(fb_kpers[j1])} |",
           f"| **featbasis (all 100 features)** | "
           f"**{fmt_int_stats(fb_kfirst)}** | {fmt_int_stats(fb_kpers)} |",
           "",
           f"Top-1 (m = 1) spike error: median {np.median(spike1_err):.2f} "
           f"({spike1_err.min():.2f}–{spike1_err.max():.2f}) relative — the "
           "m = 1 solo OVERSHOOTS the full spike several-fold. Cause: the "
           "output sides are dual-frame codewords g_j = (W_e W_e^T)^{-1} c_j "
           "with norms ||g_j||/||c_j|| median "
           f"{np.median(amp):.2f} ({amp.min():.2f}–{amp.max():.2f}); the "
           f"piece-norm overcompleteness sum_j ||P_j||_F / ||W_in||_F = "
           f"{cancel:.2f}. The exact tiling lives on massive cross-piece "
           "cancellation, so truncation is catastrophic at every small m.", "",
           "## Verdict: is top-1 sufficient for a 1% spike?", "",
           f"**NO.** Minimal k for spike within 1% is median "
           f"{int(np.median(fb_kfirst))} ({fb_kfirst.min()}–{fb_kfirst.max()})"
           " of 100 pieces (first crossing; persistent m* median "
           f"{int(np.median(fb_kpers))} ({fb_kpers.min()}–{fb_kpers.max()})). "
           "The Gram-row census looks maximally concentrated (PR median "
           f"{np.median(pr['featbasis']):.1f}, n50 median "
           f"{int(np.median(nth['featbasis'][0.50]))}) but "
           "the causal story is the opposite: cross-layer knowledge of W_emb "
           "buys a nominally top-1 READ ontology whose WRITE sides are "
           "dual-amplified, and the four-layer solo needs almost all pieces "
           "to cancel the amplification back down. The five-ontology "
           f"minimal-k comparison for probe e_1 (figure v2): full model, "
           f"columns k = {FB_LEGACY_K['columns']}, SVD k = "
           f"{FB_LEGACY_K['svd']}, rows k = {FB_LEGACY_K['rows']}, featbasis "
           f"k = {fb_k1} — featbasis lands between SVD and rows, far from "
           "the per-neuron columns' k = 5, and nowhere near k = 1.", "",
           "## Cross-check: learned VPD lin_in components vs this frame", "",
           f"`exports/goodfire/out/decompositions/p-832f0310/model_10000.pth`"
           ", key `_components.lin_in.V`, (d_in, C) orientation, columns "
           "normalized.", "", xline + ".", "",
           "Expectation on record: at d200 the two frames should be "
           "near-identical. FINDING: they are NOT — cos(W_e[j], dual_j) "
           f"median {np.median(xc['frame_cos']):.4f} "
           f"({xc['frame_cos'].min():.4f}–{xc['frame_cos'].max():.4f}), "
           "because the feature Gram's off-diagonals (mean |.| "
           f"{np.abs(Gram - np.diag(np.diag(Gram))).mean():.3f}, max "
           f"{np.abs(Gram - np.diag(np.diag(Gram))).max():.3f}) accumulate "
           "over 99 rivals and tilt each dual well away from its feature "
           "row. The VPD components separate the frames cleanly: they match "
           "the DUAL rows (median 0.91) far better than the feature rows "
           "(median 0.63) — an independent confirmation of the dual-frame "
           "re-scoring result (VPD's embedded read ontology is the dual "
           "basis), now visible as a raw best-|cos| gap without any "
           "re-scoring.", ""]
    md_path = os.path.join(RES_DIR, f"featbasis_census_d{D_EMB}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {md_path}")
    print("\n".join(md))

    # ---- figures
    plot_featbasis_activity(E["featbasis"], pr["featbasis"], nth["featbasis"],
                            dom_max, dom_sum)
    plot_minimal_k_v2(t, Gram, Gmat, curves_j)


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
    ap.add_argument("--spectra", action="store_true",
                    help="run the mechanism solo spectra (winning-cut "
                         "1+5+5+1 convention, example features)")
    ap.add_argument("--featbasis", action="store_true",
                    help="run the featbasis (top-1 feature-frame) fourth "
                         "ontology: census + four-layer solo minimal-k + VPD "
                         "cross-check (d_emb = 200 only)")
    ap.add_argument("--d-emb", type=int, choices=(200, 50), default=200,
                    help="checkpoint d_emb; outputs are suffixed _d{D} "
                         "(default 200)")
    args = ap.parse_args()
    assert sum((args.grid, args.spectra, args.featbasis)) <= 1, \
        "--grid, --spectra and --featbasis are exclusive"
    set_demb(args.d_emb)
    if args.grid:
        grid_main()
    elif args.spectra:
        spectra_main()
    elif args.featbasis:
        featbasis_main()
    else:
        main()
