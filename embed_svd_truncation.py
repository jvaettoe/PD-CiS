"""Truncation-based denoising study for the SVD PD of EmbedMLP.

Question: can mass-ordered truncation (keep the top-n SVD components of W_in/W_out)
remove the model's interference leak without destroying the signal? Hypothesis from
the flat spectra + dense loadings: NO — leak is distributed across the whole
spectrum, so signal and leak degrade together and no sweet spot exists. A knee
(leak falling steeply before signal) would instead identify a genuine noise
subspace.

Outputs: results/embedded/truncation/ + figures/embedded/truncation/.

  ./.venv/bin/python embed_svd_truncation.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embed_exact_decomposition import decompose, load_embed_mlp

RES_DIR = os.path.join(os.path.dirname(__file__), "results", "embedded", "truncation")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "embedded", "truncation")
D_EMBS = (20, 50, 200, 1000)
JS = (0, 33, 66)
MODES = ("joint", "win-only", "wout-only")


def truncate(U, S, Vh, n):
    return (U[:, :n] * S[:n]) @ Vh[:n]


def analyse(d_emb: int):
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    dd = decompose(model)
    We, Win, Wout = dd["We"], dd["Win"], dd["Wout"]
    Ui, Si, Vhi = torch.linalg.svd(Win, full_matrices=False)
    Uo, So, Vho = torch.linalg.svd(Wout, full_matrices=False)
    rank = len(Si)
    mass = (Si ** 2).cumsum(0) / (Si ** 2).sum()
    n95 = int((mass < 0.95).sum()) + 1

    X = torch.eye(F, dtype=torch.float64)
    e1 = X @ We

    def spectra(win, wout):
        return torch.relu(e1 @ win) @ wout @ We.T

    R = spectra(Win, Wout)
    off_full = R - torch.diag(torch.diag(R))
    leak_full = float(off_full.norm())
    diag_mean = float(torch.diag(R).abs().mean())

    ns = sorted({1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, n95, rank})
    ns = [n for n in ns if 1 <= n <= rank]
    rows = []
    for mode in MODES:
        for n in ns:
            win = truncate(Ui, Si, Vhi, n) if mode != "wout-only" else Win
            wout = truncate(Uo, So, Vho, n) if mode != "win-only" else Wout
            Rt = spectra(win, wout)
            off = Rt - torch.diag(torch.diag(Rt))
            leak = float(off.norm()) / max(leak_full, 1e-12)
            sig_dev = float((torch.diag(Rt) - torch.diag(R)).abs().mean()) / diag_mean
            rows.append({"mode": mode, "n": n, "leak_rel": leak,
                         "sig_dev_rel": sig_dev})
    return {"d_emb": d_emb, "rank": rank, "n95": n95, "rows": rows,
            "R": R, "svd": (Ui, Si, Vhi, Uo, So, Vho), "e1": e1, "We": We,
            "Win": Win, "Wout": Wout, "model": model}


def r2_of_truncated(res, n):
    """per_feature_r2 of the jointly-truncated model (float32 EmbedMLP copy)."""
    Ui, Si, Vhi, Uo, So, Vho = res["svd"]
    model = res["model"]
    import copy
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.w_in.copy_(truncate(Ui, Si, Vhi, n).float())
        m2.w_out.copy_(truncate(Uo, So, Vho, n).float())
    r2 = m2.per_feature_r2(0.001, n_batches=10).clamp(min=0)
    return float(r2.sum())


def main():
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    all_res = []
    md = ["# SVD truncation-denoising study (EmbedMLP, p=0.001)", "",
          "leak_rel = off-target energy relative to the full model; sig_dev_rel = "
          "mean on-target deviation relative to mean |R_jj|. n95 = 95% Frobenius "
          "mass cutoff of W_in.", ""]
    csv = ["d_emb,mode,n,leak_rel,sig_dev_rel"]
    for d_emb in D_EMBS:
        print(f"analysing d_emb={d_emb}...")
        res = analyse(d_emb)
        all_res.append(res)
        for r in res["rows"]:
            csv.append(f"{d_emb},{r['mode']},{r['n']},{r['leak_rel']:.4f},"
                       f"{r['sig_dev_rel']:.4f}")
        at95 = [r for r in res["rows"] if r["mode"] == "joint" and r["n"] == res["n95"]]
        n_soft_95 = r2_of_truncated(res, res["n95"])
        n_soft_full = r2_of_truncated(res, res["rank"])
        md.append(f"- d_emb={d_emb} (rank {res['rank']}, n95={res['n95']}): at joint "
                  f"truncation n=n95: leak_rel={at95[0]['leak_rel']:.3f}, "
                  f"sig_dev_rel={at95[0]['sig_dev_rel']:.3f}; n_soft {n_soft_full:.1f} "
                  f"→ {n_soft_95:.1f}")

    # curves figure
    fig, axes = plt.subplots(1, len(D_EMBS), figsize=(4.6 * len(D_EMBS), 3.8),
                             constrained_layout=True, sharey=True)
    styles = {"joint": "-", "win-only": "--", "wout-only": ":"}
    for ax, res in zip(axes, all_res):
        for mode in MODES:
            rs = [r for r in res["rows"] if r["mode"] == mode]
            ns = [r["n"] for r in rs]
            ax.plot(ns, [r["leak_rel"] for r in rs], styles[mode], color="tab:red",
                    lw=1.5, label=f"leak ({mode})" if mode == "joint" else None)
            ax.plot(ns, [r["sig_dev_rel"] for r in rs], styles[mode],
                    color="tab:blue", lw=1.5,
                    label=f"signal loss ({mode})" if mode == "joint" else None)
        ax.axvline(res["n95"], color="grey", ls="--", lw=0.8)
        ax.text(res["n95"], 1.02, "n95", fontsize=7, color="grey", ha="center")
        ax.set_title(f"d_emb={res['d_emb']}", fontsize=10)
        ax.set_xlabel("kept components n")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("relative to full model")
    axes[0].legend(fontsize=8)
    fig.suptitle("Truncation curves: leak retained (red) vs signal lost (blue) — "
                 "line styles: joint / W_in-only / W_out-only", fontsize=11)
    out = os.path.join(FIG_DIR, "truncation_curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

    # spectra before/after for d_emb 20 and 1000
    for res in (all_res[0], all_res[-1]):
        d_emb = res["d_emb"]
        Ui, Si, Vhi, Uo, So, Vho = res["svd"]
        e1, We, R = res["e1"], res["We"], res["R"]
        F = R.shape[0]
        n_half = max(1, res["rank"] // 2)
        cases = [("full", res["Win"], res["Wout"]),
                 (f"top {res['n95']} (n95)",
                  truncate(Ui, Si, Vhi, res["n95"]), truncate(Uo, So, Vho, res["n95"])),
                 (f"top {n_half}",
                  truncate(Ui, Si, Vhi, n_half), truncate(Uo, So, Vho, n_half))]
        fig, axes = plt.subplots(len(JS), len(cases),
                                 figsize=(4.6 * len(cases), 2.7 * len(JS)),
                                 squeeze=False, constrained_layout=True, sharey=True)
        for row, j in enumerate(JS):
            for col, (lbl, win, wout) in enumerate(cases):
                spec = (torch.relu(e1[j:j + 1] @ win) @ wout @ We.T)[0].numpy()
                ax = axes[row][col]
                colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
                ax.bar(range(F), spec, width=1.0, color=colors)
                ax.axhline(0, color="k", lw=0.5)
                ax.set_title(f"e_{j}: {lbl}  (s={spec[j]:.3f})", fontsize=9)
        fig.suptitle(f"Output spectra under joint truncation (d_emb={d_emb})",
                     fontsize=11)
        out = os.path.join(FIG_DIR, f"truncation_spectra_demb{d_emb}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")

    md += ["", "Verdict: see curves — if the red (leak) and blue (signal-loss) "
           "curves track each other with no knee, mass-ordered truncation cannot "
           "denoise this model: interference is carried by the same components as "
           "the signal, at every d_emb."]
    with open(os.path.join(RES_DIR, "embed_svd_truncation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RES_DIR, "embed_svd_truncation.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
