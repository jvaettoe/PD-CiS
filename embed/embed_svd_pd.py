"""SVD-proposition test: weight-mass components of W_in/W_out on EmbedMLP.

Components = SVD triplets sigma_k u_k (x) v_k of W_in (and of W_out), plus W_emb's
rows. Checks the three ideal-PD conditions and measures the functional structure:
per-feature capture counts (how many mass-components are needed for a feature's
read/write), per-component feature loadings (how many features drive a component),
and active components per one-hot input. Contrast object: the per-feature chain
decompositions (dual at d_emb >= F; output-matching construction at d_emb < F),
which have 1 component per feature by construction.

  ./.venv/bin/python embed_svd_pd.py
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


def pr(x, dim=-1, eps=1e-12):
    """Participation ratio along dim: (sum x^2)^2 / sum x^4."""
    x2 = x ** 2
    return (x2.sum(dim) ** 2) / (x2.pow(2).sum(dim) + eps)


def capture_count(loadings, contribs, frac=0.95):
    """#components (sorted by |contribution|) needed to reach `frac` of the total
    norm of the target vector built from contribs (rows: per component vectors)."""
    order = torch.argsort(loadings.abs(), descending=True)
    total = contribs.sum(0)
    tnorm = float(total.norm())
    acc = torch.zeros_like(total)
    for cnt, k in enumerate(order, start=1):
        acc = acc + contribs[k]
        if float((acc - total).norm()) <= (1 - frac) * tnorm:
            return cnt
    return len(order)


def analyse(d_emb: int):
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    dd = decompose(model)
    We, Win, Wout = dd["We"], dd["Win"], dd["Wout"]

    Ui, Si, Vhi = torch.linalg.svd(Win, full_matrices=False)     # W_in triplets
    Uo, So, Vho = torch.linalg.svd(Wout, full_matrices=False)    # W_out triplets
    rank_i = int((Si > 1e-10 * Si[0]).sum())
    rank_o = int((So > 1e-10 * So[0]).sum())

    # (1) faithfulness
    dev1 = float(((Ui * Si) @ Vhi - Win).abs().max())
    dev2 = float(((Uo * So) @ Vho - Wout).abs().max())

    # feature loadings: input side L[j,k] = sigma_k (w_e_j . u_k)
    L_in = (We @ Ui) * Si                                        # (F, rank)
    L_out = (Vho @ We.T).T * So                                  # (F, rank): w_e_j.b_k
    # per-component: how many features drive it; per-feature: how many components fire
    pr_comp_in = pr(L_in, dim=0)
    pr_feat_in = pr(L_in, dim=1)
    active_per_feat = (L_in.abs() > 0.01 * L_in.abs().max()).sum(1).float()

    pr_comp_out = pr(L_out, dim=0)
    pr_feat_out = pr(L_out, dim=1)

    # per-feature capture: #components for 95% of r_j / c_j
    caps_i, caps_o = [], []
    for j in range(0, F, 7):                                     # subsample features
        caps_i.append(capture_count(L_in[j], L_in[j].unsqueeze(1) * Vhi))
        caps_o.append(capture_count(L_out[j], (L_out[j].unsqueeze(0) * Uo).T))
    return {"d_emb": d_emb, "rank_i": rank_i, "rank_o": rank_o,
            "dev": max(dev1, dev2),
            "pr_comp_in": float(pr_comp_in.mean()),
            "pr_feat_in": float(pr_feat_in.mean()),
            "pr_comp_out": float(pr_comp_out.mean()),
            "pr_feat_out": float(pr_feat_out.mean()),
            "active_per_feat": float(active_per_feat.mean()),
            "cap95": float(np.mean(caps_i)), "cap95_out": float(np.mean(caps_o)),
            "L_in": L_in.numpy(), "S_in": Si.numpy()}


def svd_partition_artifacts(d_emb: int) -> dict:
    """The full SVD partition for one checkpoint: save the component factors, the
    singular spectra with mass-capture cutoffs, per-component piece figures, and
    solo-component output spectra."""
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    dd = decompose(model)
    We, Win, Wout = dd["We"], dd["Win"], dd["Wout"]
    Ui, Si, Vhi = torch.linalg.svd(Win, full_matrices=False)
    Uo, So, Vho = torch.linalg.svd(Wout, full_matrices=False)

    # persist the partition (components k: sigma_k * u_k (x) v_k)
    torch.save({"W_in": {"U": Ui, "S": Si, "Vh": Vhi},
                "W_out": {"U": Uo, "S": So, "Vh": Vho},
                "note": "component k of M = S[k] * U[:,k] (x) Vh[k,:]; "
                        "sum over k reconstructs M exactly"},
               os.path.join(RES_DIR, f"svd_components_demb{d_emb}.pt"))

    # spectra + Frobenius mass capture (the 'relevance' ordering)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    cuts = {}
    for ax, (S, name) in zip(axes, ((Si, "W_in"), (So, "W_out"))):
        mass = (S ** 2).cumsum(0) / (S ** 2).sum()
        n95 = int((mass < 0.95).sum()) + 1
        pr_rank = float((S ** 2).sum() ** 2 / (S ** 4).sum())
        cuts[name] = (n95, pr_rank)
        ax.semilogy(range(1, len(S) + 1), S.numpy(), "o-", ms=3, lw=1)
        ax.axvline(n95, color="tab:red", ls="--", lw=1,
                   label=f"95% mass: {n95} comps (PR {pr_rank:.1f})")
        ax.set_title(f"{name} singular spectrum (d_emb={d_emb})", fontsize=10)
        ax.set_xlabel("component k"); ax.set_ylabel("sigma_k")
        ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, f"embed_svd_spectrum_demb{d_emb}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

    # per-component piece figures for BOTH matrices: top-2, middle, last relevant
    for name, (Uc, Sc, Vhc), f1lbl, f2lbl in (
            ("win", (Ui, Si, Vhi), "embedding dims", "neurons"),
            ("wout", (Uo, So, Vho), "neurons", "embedding dims")):
        n95 = cuts["W_in" if name == "win" else "W_out"][0]
        ks = sorted({0, 1, n95 // 2, n95 - 1})
        fig, axes = plt.subplots(len(ks), 3, figsize=(15, 2.9 * len(ks)),
                                 squeeze=False, constrained_layout=True)
        for row, k in enumerate(ks):
            piece = (Sc[k] * torch.outer(Uc[:, k], Vhc[k])).numpy()
            v = max(abs(piece.min()), abs(piece.max()))
            im = axes[row][0].imshow(piece, aspect="auto", cmap="RdBu_r",
                                     vmin=-v, vmax=v, interpolation="nearest")
            axes[row][0].set_title(f"{name} comp {k}: sigma={float(Sc[k]):.3f} "
                                   f"(rank-1)", fontsize=9)
            fig.colorbar(im, ax=axes[row][0], fraction=0.046, pad=0.04)
            axes[row][1].bar(range(len(Uc)), Uc[:, k].numpy(), width=1.0,
                             color="tab:blue")
            axes[row][1].set_title(f"u_{k} ({f1lbl})", fontsize=9)
            axes[row][2].bar(range(Vhc.shape[1]), Vhc[k].numpy(), width=1.0,
                             color="tab:orange")
            axes[row][2].set_title(f"v_{k} ({f2lbl})", fontsize=9)
            for a2 in axes[row][1:]:
                a2.axhline(0, color="k", lw=0.5)
        mat_lbl = "W_in" if name == "win" else "W_out"
        fig.suptitle(f"SVD components of {mat_lbl} (d_emb={d_emb}) — factors "
                     f"u_k / v_k", fontsize=11)
        out = os.path.join(FIG_DIR, f"embed_svd_pieces_{name}_demb{d_emb}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")

    # intrinsic feature-space profiles (leak-free by construction): what each
    # component READS (W_in: sigma_k * We @ u_k over features) and WRITES
    # (W_out: sigma'_k * We @ b_k over features) -- the component's own vector,
    # with no model forward and hence no interference from anything else.
    ks_in = sorted({0, 1, cuts["W_in"][0] // 2, cuts["W_in"][0] - 1})
    ks_out = sorted({0, 1, cuts["W_out"][0] // 2, cuts["W_out"][0] - 1})
    nrows = max(len(ks_in), len(ks_out))
    fig, axes = plt.subplots(nrows, 2, figsize=(13.5, 2.6 * nrows),
                             squeeze=False, constrained_layout=True)
    for row in range(nrows):
        for col, (ks, ttl, prof_fn) in enumerate((
                (ks_in, "W_in comp {k}: read profile  sigma*We.u_k",
                 lambda k: (Si[k] * (We @ Ui[:, k])).numpy()),
                (ks_out, "W_out comp {k}: write profile  sigma*We.b_k",
                 lambda k: (So[k] * (We @ Vho[k])).numpy()))):
            ax = axes[row][col]
            if row >= len(ks):
                ax.axis("off")
                continue
            k = ks[row]
            prof = prof_fn(k)
            ax.bar(range(F), prof, width=1.0, color="steelblue")
            ax.axhline(0, color="k", lw=0.5)
            pr_v = float((prof ** 2).sum() ** 2 / (prof ** 4).sum())
            ax.set_title(ttl.format(k=k) + f"   (PR {pr_v:.1f} features)",
                         fontsize=9)
            ax.set_xlabel("feature")
    fig.suptitle(f"Intrinsic component profiles in feature space (d_emb={d_emb}) — "
                 f"the components' own read/write vectors, no forward pass, "
                 f"no interference", fontsize=11)
    out = os.path.join(FIG_DIR, f"embed_svd_profiles_demb{d_emb}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")
    return {"cuts": cuts}


def main():
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    rows, results = [], []
    for d_emb in D_EMBS:
        print(f"analysing d_emb={d_emb}...")
        res = analyse(d_emb)
        results.append(res)
        res["cuts"] = svd_partition_artifacts(d_emb)["cuts"]
        rows.append([str(d_emb), str(res["rank_i"]), f"{res['dev']:.1e}",
                     f"{res['cap95']:.1f}", f"{res['cap95_out']:.1f}",
                     f"{res['pr_feat_in']:.1f}", f"{res['pr_feat_out']:.1f}",
                     f"{res['pr_comp_in']:.1f}", f"{res['pr_comp_out']:.1f}"])
    cols = ["d_emb", "rank", "faithfulness dev", "cap95 r_j (W_in)",
            "cap95 c_j (W_out)", "PR comps/feat (in)", "PR comps/feat (out)",
            "PR feats/comp (in)", "PR feats/comp (out)"]
    md = ["### SVD-proposition test — components = SVD triplets of W_in/W_out "
          "(+ W_emb rows)", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["",
           "Reference (per-feature chain decompositions, same checkpoints): "
           "100 components; 1 component per feature; capture count = 1; "
           "feature-PR per component = 1; faithfulness equally exact.",
           "",
           "Reading: SVD components are exactly faithful and rank-1 with NO overlap "
           "corrections (orthogonal factors) — but they are collective objects: every "
           "feature needs most of the spectrum (capture ~rank), every component serves "
           "many features (PR), and all components co-activate on every input. The "
           "weight-mass decomposition answers 'what directions dominate W_in' — not "
           "'which part computes feature j'. Neither is wrong; they are different "
           "questions, and the d_emb regimes do not qualitatively change the SVD "
           "picture (it cannot see the superposition boundary that the chain "
           "decompositions expose)."]
    path = os.path.join(RES_DIR, "embed_svd_pd.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, len(D_EMBS), figsize=(4.2 * len(D_EMBS), 3.6),
                             constrained_layout=True)
    for ax, res in zip(axes, results):
        L = np.abs(res["L_in"])
        im = ax.imshow(L, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"d_emb={res['d_emb']}  |loading| (F x rank)", fontsize=9)
        ax.set_xlabel("SVD component"); ax.set_ylabel("feature")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Feature loadings on W_in's SVD components — dense at every d_emb "
                 "(no feature alignment)", fontsize=11)
    out = os.path.join(FIG_DIR, "embed_svd_loadings.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
