"""Cross-model / cross-d_emb overviews of the sweep.

Two aggregate views that distil the per-cell figures + cached data:

  (a) ExactPD `effective_count` heatmap over (d_emb x p) per hidden width -- the
      participation-ratio "# features the trained model actually computes", read off the
      exact per-feature decomposition. Shows how compute capacity varies with embedding
      dim and sparsity.  -> figures/overview/exactpd_effective_count_n<n>.png

  (b) Readable model reconstruction-loss tables: L_n = mean(|model(x) - relu(x)|^n) on a
      held-out batch at each p, for LinearMLP, every EmbedMLP d_emb, and the biregular K
      code (refit per p) -- one table per hidden width, as markdown (printed + saved) + CSV.
      -> results/overview/model_L<n>_n<n>_dmlp<w>.{md,csv}

  ./.venv/bin/python overview.py --n 4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import cc
import results_io
import biregular_ansatz as bir
import codes
from data import CleanDataset
from mlp import LinearMLP, EmbedMLP
from pd import ExactPD
from sweep import N_FEATURES, D_EMBS, PS, D_MLP   # track the sweep grid

D_MLPS = [50, 100]
BIREG_K = 5                       # biregular codeword length used in the model loss table
SWAPS = 800_000                   # overlap-min iterations (the source repo's setting)
BIREG_VARIANTS = [("no-swap", 0), ("overlap-min", SWAPS)]
OVERVIEW_FIG = os.path.join(os.path.dirname(__file__), "figures", "overview")
OVERVIEW_RES = os.path.join(os.path.dirname(__file__), "results", "overview")


# --------------------------------------------------------------------------
# (b) model reconstruction-loss tables
# --------------------------------------------------------------------------

def eval_ln(model, p, n_exp, device="cpu", batch=16384, seed=12345):
    """L_n reconstruction loss = mean(|model(x) - relu(x)|**n) on a held-out batch."""
    ds = CleanDataset(n_features=N_FEATURES, p=float(p), device=device, seed=seed)
    x, y = ds.generate_batch(batch)
    with torch.no_grad():
        return float(((model(x) - y).abs() ** n_exp).mean().item())


def trained_l4(model_name, d_emb, d_mlp, str_p, n_exp, device="cpu"):
    """Load the cached n_exp checkpoint for (model, d_emb, d_mlp, p) and eval its L_n."""
    if model_name == "LinearMLP":
        cls, kwargs = LinearMLP, {"n_features": N_FEATURES, "d_mlp": d_mlp}
    else:
        cls, kwargs = EmbedMLP, {"n_features": N_FEATURES, "d_emb": d_emb, "d_mlp": d_mlp}
    path = cc.checkpoint_path(str_p, cls, kwargs, seed=42, n_exp=n_exp)
    if not os.path.exists(path):
        return float("nan")
    model = cc.load_checkpoint(path, device=device)["model"]
    return eval_ln(model, str_p, n_exp, device=device)


# Codes and per-p fits are cached so the model table, biregular table, gains figure
# and effective_count heatmap all share one fit per (K, swaps, p).
_CODE_CACHE: dict = {}
_FIT_CACHE: dict = {}


def _bireg_code(K, swaps, seed=0):
    """Biregular code, optionally overlap-minimised (source convention seed=10+seed)."""
    key = (K, swaps, seed)
    if key not in _CODE_CACHE:
        M = codes.regular_code(N_FEATURES, 50, K, seed)
        if swaps > 0:
            M = codes.reduce_overlap(M, swaps, seed=10 + seed)
        _CODE_CACHE[key] = M
    return _CODE_CACHE[key]


def bireg_fit_eval(str_p, K, n_exp, swaps=0, fit_steps=2500, device="cpu"):
    """Refit the (K, swaps) biregular ansatz at sparsity p; return
    {L4, ac, bc, eff_count} (gauge-invariant gains + ExactPD effective_count)."""
    key = (K, swaps, str_p, fit_steps)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    M = _bireg_code(K, swaps)
    a, b, c = bir.fit_abc(M, float(str_p), steps=fit_steps, device=device)
    model = bir.model_from_abc(M, (a, b, c), device=device)
    res = {"L4": eval_ln(model, str_p, n_exp, device=device),
           "ac": a * c, "bc": b * c,
           "eff_count": ExactPD(model).component_metrics()["effective_count"]}
    _FIT_CACHE[key] = res
    return res


def build_loss_table(n_exp, d_mlp, device="cpu", include_biregular=True):
    """DataFrame: rows = models, cols = p, cells = L_n reconstruction loss."""
    rows = {}
    rows["LinearMLP"] = {p: trained_l4("LinearMLP", None, d_mlp, p, n_exp, device) for p in PS}
    for demb in sorted(D_EMBS):
        rows[f"EmbedMLP d_emb={demb}"] = {
            p: trained_l4("EmbedMLP", demb, d_mlp, p, n_exp, device) for p in PS}
    if include_biregular:
        for vname, sw in BIREG_VARIANTS:
            label = f"biregular K{BIREG_K} {vname} (N=50)"
            rows[label] = {p: bireg_fit_eval(p, BIREG_K, n_exp, swaps=sw, device=device)["L4"]
                           for p in PS}
    df = pd.DataFrame(rows).T          # rows=model, cols=p
    df = df[PS]                         # column order = sweep p order
    return df


def to_markdown(df, title):
    """A hand-rolled, aligned markdown table (no tabulate dependency). Cells in %.2e."""
    cols = ["model"] + [f"p={c}" for c in df.columns]
    body = [[idx] + [f"{v:.2e}" if np.isfinite(v) else "—" for v in df.loc[idx]]
            for idx in df.index]
    widths = [max(len(cols[j]), *(len(r[j]) for r in body)) for j in range(len(cols))]
    def fmt(cells):
        return "| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(cols))) + " |"
    return "\n".join([f"### {title}", "", fmt(cols), sep, *(fmt(r) for r in body)])


def write_loss_tables(n_exp, device="cpu"):
    os.makedirs(OVERVIEW_RES, exist_ok=True)
    ntok = results_io._fmt_n(n_exp)
    for d_mlp in D_MLPS:
        df = build_loss_table(n_exp, d_mlp, device=device, include_biregular=(d_mlp == 50))
        md = to_markdown(df, f"Model reconstruction loss L{ntok} vs sparsity p  (d_mlp={d_mlp}, "
                             f"F={N_FEATURES})")
        stem = os.path.join(OVERVIEW_RES, f"model_L{ntok}_n{ntok}_dmlp{d_mlp}")
        df.to_csv(stem + ".csv")
        with open(stem + ".md", "w") as f:
            f.write(md + "\n")
        print("\n" + md + f"\n  (saved {stem}.md / .csv)")


# --------------------------------------------------------------------------
# (a) ExactPD effective_count heatmap over (d_emb x p)
# --------------------------------------------------------------------------

def _demb_label(v):
    s = str(v)
    return "LinearMLP" if s in ("", "nan", "None", "NA") else str(int(float(v)))


def effective_count_heatmap(n_exp, device="cpu"):
    path = results_io.component_metrics_path(n_exp)
    if not os.path.exists(path):
        print(f"  [skip heatmap] no {path}")
        return
    cm = pd.read_csv(path)
    cm = cm[(cm["method"] == "ExactPD") & (cm["n_exp"].astype(float) == float(n_exp))].copy()
    cm["demb"] = cm["d_emb"].map(_demb_label)
    row_order = [r for r in ["LinearMLP", "20", "50", "100", "200", "1000"]
                 if r in set(cm["demb"])]

    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    fig, axes = plt.subplots(1, len(D_MLPS), figsize=(5.5 * len(D_MLPS), 4.6),
                             squeeze=False, constrained_layout=True)
    for ax, d_mlp in zip(axes[0], D_MLPS):
        sub = cm[cm["d_mlp"] == d_mlp]
        piv = (sub.pivot_table(index="demb", columns="p", values="effective_count")
                  .reindex(index=row_order, columns=[float(p) for p in PS]))
        A = piv.to_numpy(dtype=float)
        im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=N_FEATURES)
        ax.set_xticks(range(len(PS)), PS)
        ax.set_yticks(range(len(row_order)), row_order)
        ax.set_xlabel("data sparsity p"); ax.set_title(f"d_mlp={d_mlp}")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center",
                            color="w" if A[i, j] < 0.6 * N_FEATURES else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0][0].set_ylabel("model (d_emb; LinearMLP = no embedding)")
    fig.suptitle(f"ExactPD effective_count  (# features the trained model computes, of {N_FEATURES})"
                 f"  — n={results_io._fmt_n(n_exp)}", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"exactpd_effective_count_n{results_io._fmt_n(n_exp)}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------
# (c) biregular benchmark overview: L4 table + gains figure + effective_count heatmap
# --------------------------------------------------------------------------

def biregular_overview(n_exp, device="cpu"):
    """Fit the (K x variant x p) biregular grid once (cached) and emit:
    a readable L4 table, an L4 + gauge-invariant-gains figure, and an ExactPD
    effective_count heatmap styled like the trained-model one."""
    ntok = results_io._fmt_n(n_exp)
    os.makedirs(OVERVIEW_RES, exist_ok=True)
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    K_LIST = list(bir.K_CHOICES)

    grid = {}   # (K, vname) -> {p: res}
    for K in K_LIST:
        for vname, sw in BIREG_VARIANTS:
            grid[(K, vname)] = {p: bireg_fit_eval(p, K, n_exp, swaps=sw, device=device)
                                for p in PS}
            print(f"  [bireg] K={K} {vname}: fitted {len(PS)} p")

    # table: rows = K x variant, cols = p
    df = pd.DataFrame({f"K{K} {vname}": {p: r["L4"] for p, r in cells.items()}
                       for (K, vname), cells in grid.items()}).T[PS]
    md = to_markdown(df, f"Biregular ansatz reconstruction loss L{ntok} vs p  "
                         f"(refit per p; overlap-min = {SWAPS} reduce_overlap swaps; N=50)")
    stem = os.path.join(OVERVIEW_RES, f"biregular_L{ntok}_n{ntok}")
    df.to_csv(stem + ".csv")
    with open(stem + ".md", "w") as f:
        f.write(md + "\n")
    print("\n" + md + f"\n  (saved {stem}.md / .csv)")

    # figure: L4 | a·c | b·c vs p (color per K; solid = no-swap, dashed = overlap-min)
    pf = [float(p) for p in PS]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for i, K in enumerate(K_LIST):
        for vname, _ in BIREG_VARIANTS:
            ls = "-" if vname == "no-swap" else "--"
            cells = grid[(K, vname)]
            axes[0].plot(pf, [cells[p]["L4"] for p in PS], ls, marker="o",
                         color=cmap(i), label=f"K{K} {vname}")
            axes[1].plot(pf, [cells[p]["ac"] for p in PS], ls, marker="o", color=cmap(i))
            axes[2].plot(pf, [cells[p]["bc"] for p in PS], ls, marker="o", color=cmap(i))
    axes[0].set_yscale("log"); axes[0].set_ylabel(f"L{ntok} (log)")
    axes[0].set_title("reconstruction loss"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("a·c"); axes[1].set_title("on-code effective gain a·c")
    axes[2].set_ylabel("b·c"); axes[2].set_title("off-code leak gain b·c")
    axes[2].axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("data sparsity p (log)")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"Biregular ansatz vs p  (solid = no-swap, dashed = overlap-min@{SWAPS}) — "
                 f"n={ntok}", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"biregular_L{ntok}_gains_n{ntok}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

    # effective_count heatmap, same style as the trained-model one
    row_labels = [f"K{K} {vname}" for K in K_LIST for vname, _ in BIREG_VARIANTS]
    A = np.array([[grid[(K, vname)][p]["eff_count"] for p in PS]
                  for K in K_LIST for vname, _ in BIREG_VARIANTS])
    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=N_FEATURES)
    ax.set_xticks(range(len(PS)), PS)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("data sparsity p")
    ax.set_ylabel("biregular code (N=50)")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center",
                    color="w" if A[i, j] < 0.6 * N_FEATURES else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"ExactPD effective_count — biregular benchmark codes (of {N_FEATURES})"
                 f"  — n={ntok}", fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"exactpd_effective_count_biregular_n{ntok}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, default=4.0, dest="n_exp")
    ap.add_argument("--no-biregular", action="store_true", dest="no_biregular",
                    help="skip the biregular K x variant grid (the slow part)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(f"=== overview (n={results_io._fmt_n(a.n_exp)}) ===")
    effective_count_heatmap(a.n_exp, device=a.device)
    if not a.no_biregular:
        biregular_overview(a.n_exp, device=a.device)   # fills the fit cache
    write_loss_tables(a.n_exp, device=a.device)        # reuses cached K5 fits


if __name__ == "__main__":
    main()
