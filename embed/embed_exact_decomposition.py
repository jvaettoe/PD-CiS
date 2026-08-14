"""Exact per-feature decomposition of the trained EmbedMLP, from first principles.

Model: y = relu((x @ W_e) @ W_in) @ W_out @ W_e.T with W_e (F, d_emb) FIXED
unit-norm random rows, W_in (d_emb, N), W_out (N, d_emb) trained.

Derivation (no decomposition framework needed). Ask for pieces of each matrix that
(1) sum to the matrix, (2) are rank-1, (3) each W_in piece carries signal only when
its feature is active. A piece u (x) v of W_in responds to input e = x @ W_e with
strength e.u = x.(W_e u); depending on x_j alone forces W_e u = e_j, i.e. u = the
DUAL VECTOR w~_j of W_e's rows (w_e_i . w~_j = delta_ij; exists iff rows are
independent, d_emb >= F). Faithfulness then forces the neuron-side factor:

    W_emb: piece j = e_j (x) w_e_j                    (rows; sum exact)
    W_in : piece j = w~_j (x) (w_e_j @ W_in)          (sum = P @ W_in)
    W_out: piece j = (W_out @ w_e_j) (x) w~_j         (sum = W_out @ P)

with P = W_e^+ @ W_e the projector onto span(rows of W_e). The remainders
(I-P) @ W_in and W_out @ (I-P) are touched by NO input (x @ W_e lies in the row
space identically), so they are behaviorally inert; in trained models they are
frozen initialization noise (gradients live in the row space). The number of
pieces carrying nonzero signal on an input is exactly the number of active
features. SVD relation: the duals span W_e's top-F right-singular subspace
(w~ = rows whitened by Sigma^-2), but the SVD of W_in itself fails (3): every
singular piece reads a dense mixture of features.

  ./.venv/bin/python embed_exact_decomposition.py            # d_emb 1000 + 200
"""
import os

from core.paths import REPO_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.mlp import EmbedMLP

CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
RES_DIR = os.path.join(REPO_ROOT, "results", "embedded")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "embedded")


def load_embed_mlp(d_emb: int, d_mlp: int = 50, p: str = "0.001", n: str = "4",
                   seed: int = 42) -> EmbedMLP:
    path = os.path.join(CKPT_DIR, f"ncc_EmbedMLP_d_emb{d_emb}_d_mlp{d_mlp}"
                                  f"_n_features100_p{p}_n{n}_s{seed}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = EmbedMLP(**ck["model_config"], device="cpu")
    model.load_state_dict(ck["state_dict"])
    return model


def decompose(model):
    """Shared decomposition data: duals = columns of W_e^+, pieces' factors r/c,
    projector P (embedding space) and biorthogonality matrix G = W_e W_e^+ (feature
    space). d_emb >= F: G = I (exact alignment). d_emb < F: P = I (exact
    faithfulness, NO remainder) and G = rank-d_emb projector — reads/writes soften
    to x·G / G-leaky writes."""
    We = model.w_e.double()                                   # (F, d_emb)
    Win = model.w_in.detach().double()                        # (d_emb, N)
    Wout = model.w_out.detach().double()                      # (N, d_emb)
    We_pinv = torch.linalg.pinv(We)                           # (d_emb, F); cols = duals
    return {"We": We, "Win": Win, "Wout": Wout, "duals": We_pinv,
            "P": We_pinv @ We, "G": We @ We_pinv,
            "r": We @ Win, "c": Wout @ We.T}


def analyse(d_emb: int, js=(0, 33, 66)) -> list[str]:
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    d = decompose(model)
    We, Win, Wout = d["We"], d["Win"], d["Wout"]
    duals, P, G, r, c = d["duals"], d["P"], d["G"], d["r"], d["c"]
    exact = d_emb >= F
    dev_biorth = float((G - torch.eye(F)).abs().max())

    sum_in = duals @ r                                        # = P @ W_in
    sum_out = c @ duals.T                                     # = W_out @ P
    rem_in = Win - P @ Win
    rem_out = Wout - Wout @ P

    lines = [f"## d_emb = {d_emb}  (F={F}, N={N})  — regime: "
             f"{'exact alignment (d_emb >= F)' if exact else 'soft alignment (d_emb < F)'}",
             ""]

    # (a) faithfulness: regime-aware
    dev_a2 = float((sum_in + rem_in - Win).abs().max())
    dev_a3 = float((sum_out + rem_out - Wout).abs().max())
    assert max(dev_a2, dev_a3) < 1e-9, (dev_a2, dev_a3)
    if exact:
        lines.append(f"- (a) faithfulness: pieces sum to P·W_in; +remainder = W_in/W_out "
                     f"exact (dev {dev_a2:.1e}/{dev_a3:.1e}); G = I to {dev_biorth:.1e}")
    else:
        dev_rem = float(max(rem_in.abs().max(), rem_out.abs().max()))
        assert dev_rem < 1e-9, dev_rem
        offG = G - torch.diag(torch.diag(G))
        lines.append(f"- (a) faithfulness: P = I so pieces sum to W_in/W_out EXACTLY, "
                     f"ZERO remainder (dev {dev_rem:.1e}). G = rank-{d_emb} projector: "
                     f"mean G_jj = {float(torch.diag(G).mean()):.3f} (= d_emb/F = "
                     f"{d_emb / F:.2f}), mean |G_ij| offdiag = {float(offG.abs().mean()):.4f}")

    # (b) remainder inertness on the full forward
    xs = torch.rand(256, F, dtype=torch.float64) * 2 - 1
    def fwd(win, wout):
        e = xs @ We
        return torch.relu(e @ win) @ wout @ We.T
    dev_b = float((fwd(Win, Wout) - fwd(P @ Win, Wout @ P)).abs().max())
    assert dev_b < 1e-9, dev_b
    if exact:
        lines.append(f"- (b) remainders inert: projected model == original "
                     f"(max dev {dev_b:.1e}); ||(I−P)W_in||/||W_in|| = "
                     f"{float(rem_in.norm() / Win.norm()):.3f}, "
                     f"||W_out(I−P)||/||W_out|| = {float(rem_out.norm() / Wout.norm()):.3f} "
                     f"(frozen init noise)")
    else:
        lines.append(f"- (b) no remainder exists (P = I); nothing to project out "
                     f"(check dev {dev_b:.1e})")

    # (c) reads: piece j's strength = (x @ G)_j in BOTH regimes (G = I when exact)
    e = xs @ We
    xg = xs @ G
    for j in (0, F // 2, F - 1):
        piece = torch.outer(duals[:, j], r[j])
        dev_c = float((e @ piece - torch.outer(xg[:, j], r[j])).abs().max())
        assert dev_c < 1e-8, (j, dev_c)
    lines.append("- (c) reads: piece j of W_in contributes (x·G)_j · r_j to the "
                 "preactivation exactly" + ("" if exact else
                 " — self term G_jj·x_j + embedding leak Σ G_ij·x_i"))

    # (d) writes: piece j writes (h·c_j) · G_{:,j} exactly (one-hot column when exact)
    h = torch.relu(e @ Win)
    for j in (0, F // 2, F - 1):
        piece = torch.outer(c[:, j], duals[:, j])
        out = h @ piece @ We.T
        pred = torch.outer(h @ c[:, j], G[:, j])
        assert float((out - pred).abs().max()) < 1e-8, j
    lines.append("- (d) writes: piece j of W_out writes (h·c_j)·G_col_j exactly"
                 + (" (= only output j)" if exact else " — G-shaped write leak"))

    # (e) pieces active per input = number of active features
    xs_sparse = torch.zeros(64, F, dtype=torch.float64)
    gen = torch.Generator().manual_seed(3)
    for row in xs_sparse:
        k = int(torch.randint(0, 4, (1,), generator=gen))
        idx = torch.randperm(F, generator=gen)[:k]
        row[idx] = torch.rand(k, generator=gen, dtype=torch.float64)
    strength = (xs_sparse @ We) @ duals                       # (64, F): = x @ G
    dev_e = float((strength - xs_sparse @ G).abs().max())
    assert dev_e < 1e-8, dev_e
    if exact:
        lines.append(f"- (e) piece-usage count per input == #active features "
                     f"(read strengths reproduce x exactly, dev {dev_e:.1e})")
    else:
        lines.append(f"- (e) read strengths == x·G exactly (dev {dev_e:.1e}); pieces are "
                     f"SOFT-gated — every piece receives leak, usage is graded not binary")

    # (f) ranks
    if exact:
        s_rem_in = torch.linalg.svdvals(rem_in)
        rank_rem = int((s_rem_in > 1e-8 * s_rem_in[0]).sum())
        pred = min(d_emb - F, N)
        lines.append(f"- (f) ranks: pieces rank-1 by construction; remainder(W_in) "
                     f"rank = {rank_rem} (predicted min(d_emb − F, N) = {pred})")
    else:
        lines.append("- (f) ranks: pieces rank-1 by construction; remainder identically 0")

    # (g) SVD contrast: top singular piece of W_in reads a dense feature mixture
    U, S, Vh = torch.linalg.svd(Win, full_matrices=False)
    read_svd = We @ U[:, 0]                                   # feature response of u_1
    pr = float((read_svd.pow(2).sum() ** 2) / read_svd.pow(4).sum())
    lines.append(f"- (g) SVD contrast: W_in's top singular direction reads "
                 f"{pr:.1f} features at once (participation ratio; feature-aligned "
                 f"pieces read exactly 1) — SVD pieces are faithful and rank-1 but "
                 f"not feature-aligned")

    # behavioral readout: solo chain j vs full model on e_j
    X = torch.eye(F, dtype=torch.float64)
    R = fwd(Win, Wout)                                        # placeholder; recompute on X
    e1 = X @ We
    R = torch.relu(e1 @ Win) @ Wout @ We.T                    # full model, (F, F)
    fig, axes = plt.subplots(len(js), 2, figsize=(13.5, 2.9 * len(js)),
                             squeeze=False, constrained_layout=True, sharey=True)
    diag_pairs = []
    for row_i, j in enumerate(js):
        piece_in = torch.outer(duals[:, j], r[j])
        piece_out = torch.outer(c[:, j], duals[:, j])
        solo = torch.relu(e1[j:j + 1] @ piece_in) @ piece_out @ We.T
        diag_pairs.append((float(R[j, j]), float(solo[0, j])))
        for col, (spec, ttl) in enumerate((
                (R[j].numpy(), f"full model  (s = {R[j, j]:.3f} + leak)"),
                (solo[0].numpy(), f"chain {j} alone  (s = {solo[0, j]:.3f})"))):
            ax = axes[row_i][col]
            colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
            ax.bar(range(F), spec, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"e_{j}: {ttl}", fontsize=9)
    axes[-1][0].set_xlabel("output feature"); axes[-1][1].set_xlabel("output feature")
    fig.suptitle(f"EmbedMLP d_emb={d_emb}: full model vs per-feature chain "
                 f"(W_emb row + dual-basis pieces of W_in/W_out)", fontsize=11)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, f"embed_exact_chain_vs_full_demb{d_emb}.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    lines.append(f"- behavioral: solo chain vs full model diag "
                 + ", ".join(f"j={j}: {a:.3f}/{b:.3f}" for j, (a, b) in zip(js, diag_pairs))
                 + f"  (figure: {os.path.basename(fig_path)})")
    lines.append("")
    return lines


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    out = ["# Exact per-feature decomposition of trained EmbedMLP "
           "(dual-basis construction)", ""]
    for d_emb in (1000, 200, 50, 20):
        print(f"analysing d_emb={d_emb}...")
        out += analyse(d_emb)
    out.append("Remark (corrected): for d_emb < F the strict dual basis does not exist, "
               "but the pseudo-dual construction DEGRADES GRACEFULLY, not catastrophically: "
               "W_e has full column rank, so P = I and the pieces sum to W_in/W_out "
               "EXACTLY with zero remainder — faithfulness is cleaner than for d_emb >= F. "
               "Only feature alignment softens: reads/writes follow G = W_e W_e^+ (the "
               "rank-d_emb feature-space projector) — self strength G_jj (mean d_emb/F) "
               "plus embedding leak G_ij, the continuous analog of codeword-overlap noise.")
    path = os.path.join(RES_DIR, "embed_exact_decomposition.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
