"""R^2 + output-spectra + piece-visualization suite for the embedded chain
decomposition, across d_emb regimes (style-matched to the biregular suite).

Chains: piece j of W_emb = row w_e_j; of W_in = w~_j (x) r_j; of W_out = c_j (x) w~_j
(duals w~ = columns of W_e^+; r = W_e W_in effective reads; c = W_out W_e^T effective
writes). Alignment matrix G = W_e W_e^+: identity for d_emb >= F; rank-d_emb
projector for d_emb < F, where the solo chain's on-target response obeys
chain_jj = G_jj^2 * R_jj exactly (read and write each attenuated by G_jj).

  ./.venv/bin/python embed_suite.py
"""
import os

from core.paths import REPO_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embed.embed_exact_decomposition import decompose, load_embed_mlp

RES_DIR = os.path.join(REPO_ROOT, "results", "embedded")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "embedded")
D_EMBS = (20, 50, 200, 1000)
R2_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)
P = 0.001
JS = (0, 33, 66)


def r2_census() -> dict:
    os.makedirs(RES_DIR, exist_ok=True)
    cols = (["d_emb", "mean G_jj", "mean |G_ij|", "n_soft", "min R2", "median R2"]
            + [f">{t:g}" for t in R2_THRESHOLDS])
    rows, curves = [], {}
    for d_emb in D_EMBS:
        model = load_embed_mlp(d_emb)
        d = decompose(model)
        G = d["G"]
        offG = G - torch.diag(torch.diag(G))
        r2 = model.per_feature_r2(P).clamp(min=0)
        curves[d_emb] = np.sort(r2.numpy())[::-1]
        rows.append([str(d_emb), f"{float(torch.diag(G).mean()):.3f}",
                     f"{float(offG.abs().mean()):.4f}",
                     f"{float(r2.sum()):.1f}", f"{float(r2.min()):.3f}",
                     f"{float(r2.median()):.3f}"]
                    + [str(int((r2 > t).sum())) for t in R2_THRESHOLDS])
    md = ["### Per-feature R^2 census — trained EmbedMLP, p=0.001 "
          "(F=100 ReLUs, N=50 neurons; G = W_e·W_e⁺ alignment)", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "(Non-embedded biregular baseline at p=0.001, transpose no-swap: "
           "n_soft 88.7, min 0.859, median 0.886 — see results/overview/"
           "biregular_r2_K5_p0.001.md)"]
    path = os.path.join(RES_DIR, "embed_r2_p0.001")
    with open(path + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(path + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for d_emb in D_EMBS:
        ys = curves[d_emb]
        ax.plot(range(1, len(ys) + 1), ys, lw=1.8,
                label=f"d_emb={d_emb}  (median {np.median(ys):.3f})")
    for t in R2_THRESHOLDS:
        ax.axhline(t, color="grey", ls=":", lw=0.7)
    ax.set_title("Per-feature R² (sorted) — trained EmbedMLP, p=0.001", fontsize=10)
    ax.set_xlabel("feature rank"); ax.set_ylabel("per-feature R²")
    ax.set_xlim(0, 103); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    out = os.path.join(FIG_DIR, "embed_r2_sorted_p0.001.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")
    return curves


def spectra(d_emb: int) -> None:
    """Two-panel full-model vs solo-chain spectra on e_j; for d_emb < F the chain
    panel carries the predicted on-target level G_jj^2 * R_jj."""
    model = load_embed_mlp(d_emb)
    F = model.n_features
    d = decompose(model)
    We, Win, Wout = d["We"], d["Win"], d["Wout"]
    duals, G, r, c = d["duals"], d["G"], d["r"], d["c"]
    X = torch.eye(F, dtype=torch.float64)
    e1 = X @ We
    R = torch.relu(e1 @ Win) @ Wout @ We.T
    for j in JS:
        piece_in = torch.outer(duals[:, j], r[j])
        piece_out = torch.outer(c[:, j], duals[:, j])
        solo = (torch.relu(e1[j:j + 1] @ piece_in) @ piece_out @ We.T)[0]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0), constrained_layout=True,
                                 sharey=True)
        for col, (spec, ttl) in enumerate((
                (R[j].numpy(), f"full model  (s = {R[j, j]:.3f} + leak)"),
                (solo.numpy(), f"chain {j} alone  (s = {solo[j]:.3f})"))):
            ax = axes[col]
            colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
            ax.bar(range(F), spec, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("output feature")
        if d_emb < F:
            pred = float(G[j, j] ** 2 * R[j, j])
            axes[1].axhline(pred, color="tab:red", ls="--", lw=1.0,
                            label=f"predicted G_jj²·s = {pred:.3f}")
            axes[1].legend(fontsize=8)
        axes[0].set_ylabel("output value")
        fig.suptitle(f"Output spectrum — full model vs per-feature chain  "
                     f"(input e_{j}; d_emb={d_emb}, p={P})", fontsize=11)
        out = os.path.join(FIG_DIR, f"embed_spectrum_demb{d_emb}_j{j}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")


def g_heatmap(d_emb: int) -> None:
    model = load_embed_mlp(d_emb)
    G = decompose(model)["G"].numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
    v = max(abs(G.min()), abs(G.max()))
    im = ax.imshow(G, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(f"Alignment matrix G = W_e·W_e⁺  (d_emb={d_emb}; rank-{min(d_emb, 100)} "
                 f"projector)\ndiag = self-read strength, off-diag = embedding leak",
                 fontsize=10)
    ax.set_xlabel("feature"); ax.set_ylabel("feature")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = os.path.join(FIG_DIR, f"embed_G_demb{d_emb}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def piece_figure(d_emb: int, j: int) -> None:
    """One figure per (d_emb, j): the three matrices' piece-j subcomponents.
    Row 1: W_emb piece (the embedding row w_e_j). Rows 2/3: W_in piece w~_j (x) r_j
    and W_out piece c_j (x) w~_j — full rank-1 heatmap plus BOTH factors."""
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    d = decompose(model)
    duals, r, c, We = d["duals"], d["r"], d["c"], d["We"]
    we_j = We[j].numpy()
    du_j = duals[:, j].numpy()
    r_j = r[j].numpy()
    c_j = c[:, j].numpy()
    M_in = np.outer(du_j, r_j)               # (d_emb, N)
    M_out = np.outer(c_j, du_j)              # (N, d_emb)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.bar(range(len(we_j)), we_j, width=1.0, color="dimgray")
    ax.set_title(f"W_emb piece {j} = e_{j} ⊗ w_e_{j}   (the embedding row; "
                 f"unit norm over {d_emb} dims)", fontsize=10)
    ax.set_xlabel("embedding dim"); ax.set_ylabel("value")

    for row, (Mat, f1, f1lbl, f2, f2lbl, name, shape_lbl) in enumerate((
            (M_in, du_j, f"dual w̃_{j} (embedding dims)", r_j,
             f"effective read r_{j} (neurons)", "W_in", "(d_emb × N)"),
            (M_out, c_j, f"effective write c_{j} (neurons)", du_j,
             f"dual w̃_{j} (embedding dims)", "W_out", "(N × d_emb)")), start=1):
        axm = fig.add_subplot(gs[row, 0])
        v = np.abs(Mat).max()
        im = axm.imshow(Mat, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                        interpolation="nearest")
        axm.set_title(f"{name} piece {j} {shape_lbl}, rank-1", fontsize=9)
        fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.bar(range(len(f1)), f1, width=1.0, color="tab:blue")
        ax1.set_title(f"factor 1: {f1lbl}", fontsize=9)
        ax1.axhline(0, color="k", lw=0.5)
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.bar(range(len(f2)), f2, width=1.0, color="tab:orange")
        ax2.set_title(f"factor 2: {f2lbl}", fontsize=9)
        ax2.axhline(0, color="k", lw=0.5)
    fig.suptitle(f"Per-matrix subcomponents of chain {j}  (d_emb={d_emb}; "
                 f"duals = columns of W_e⁺)", fontsize=12)
    out = os.path.join(FIG_DIR, f"embed_pieces_demb{d_emb}_j{j}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    r2_census()
    for d_emb in D_EMBS:
        spectra(d_emb)
        for j in JS:
            piece_figure(d_emb, j)
    for d_emb in (20, 50):
        g_heatmap(d_emb)


if __name__ == "__main__":
    main()
