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

import cc
import results_io
import biregular_ansatz as bir
from mlp import LinearMLP, EmbedMLP
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

def _load_trained(model_name, d_emb, d_mlp, str_p, n_exp, device="cpu"):
    """Load the cached n_exp checkpoint for (model, d_emb, d_mlp, p), or None."""
    if model_name == "LinearMLP":
        cls, kwargs = LinearMLP, {"n_features": N_FEATURES, "d_mlp": d_mlp}
    else:
        cls, kwargs = EmbedMLP, {"n_features": N_FEATURES, "d_emb": d_emb, "d_mlp": d_mlp}
    path = cc.checkpoint_path(str_p, cls, kwargs, seed=42, n_exp=n_exp)
    if not os.path.exists(path):
        return None
    return cc.load_checkpoint(path, device=device)["model"]


def trained_l4(model_name, d_emb, d_mlp, str_p, n_exp, device="cpu"):
    """L_n of the cached trained model (NaN if no checkpoint)."""
    model = _load_trained(model_name, d_emb, d_mlp, str_p, n_exp, device)
    return float("nan") if model is None else model.reconstruction_loss(float(str_p), n_exp)


# One CodeAnsatz per (K, swaps): each instance caches its code, pinv base and per-p
# fits, so the model table, biregular table, gains figure and effective_count heatmap
# all share one fit per (K, swaps, p).
_ANSATZE: dict = {}


def _ansatz(K, swaps, seed=0) -> "bir.CodeAnsatz":
    key = (K, swaps, seed)
    if key not in _ANSATZE:
        _ANSATZE[key] = bir.CodeAnsatz(K, swaps=swaps, seed=seed)
    return _ANSATZE[key]


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
            rows[label] = {p: _ansatz(BIREG_K, sw).fit_eval(float(p), n_exp, device=device)["L4"]
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
# (d) behavioural feature count + agreement with ExactPD effective_count
# --------------------------------------------------------------------------

def behavioural_count_heatmap(n_exp, device="cpu"):
    """Independent, MODEL-ONLY count of features effectively computed: soft count
    n_soft = sum(max(0, R2_i)) from MLP.per_feature_r2 at each model's own training
    sparsity p (interference included). Same grid/style as effective_count_heatmap
    so the two read side by side. Returns {(demb_label, d_mlp, p_float): n_soft}
    for the agreement scatter."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    ntok = results_io._fmt_n(n_exp)
    row_specs = [("LinearMLP", "LinearMLP", None)] + \
                [(str(d), "EmbedMLP", d) for d in sorted(D_EMBS)]
    behav = {}
    fig, axes = plt.subplots(1, len(D_MLPS), figsize=(5.5 * len(D_MLPS), 4.6),
                             squeeze=False, constrained_layout=True)
    for ax, d_mlp in zip(axes[0], D_MLPS):
        A = np.full((len(row_specs), len(PS)), np.nan)
        for i, (label, mname, demb) in enumerate(row_specs):
            for j, p in enumerate(PS):
                model = _load_trained(mname, demb, d_mlp, p, n_exp, device)
                if model is None:
                    continue
                r2 = model.per_feature_r2(float(p))
                A[i, j] = float(r2.clamp(min=0).sum())
                behav[(label, d_mlp, float(p))] = A[i, j]
        im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=N_FEATURES)
        ax.set_xticks(range(len(PS)), PS)
        ax.set_yticks(range(len(row_specs)), [r[0] for r in row_specs])
        ax.set_xlabel("data sparsity p"); ax.set_title(f"d_mlp={d_mlp}")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center",
                            color="w" if A[i, j] < 0.6 * N_FEATURES else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0][0].set_ylabel("model (d_emb; LinearMLP = no embedding)")
    fig.suptitle(f"Behavioural feature count  n_soft = Σ max(0, R²_i)  "
                 f"(active-conditional, at each model's training p; of {N_FEATURES})"
                 f"  — n={ntok}", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"behavioural_count_n{ntok}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")
    return behav


def _exactpd_counts(n_exp):
    """{(demb_label, d_mlp, p_float): effective_count} for ExactPD rows of
    component_metrics.csv (same source as effective_count_heatmap)."""
    path = results_io.component_metrics_path(n_exp)
    if not os.path.exists(path):
        return {}
    cm = pd.read_csv(path)
    cm = cm[(cm["method"] == "ExactPD") & (cm["n_exp"].astype(float) == float(n_exp))]
    return {(_demb_label(r["d_emb"]), int(r["d_mlp"]), float(r["p"])): float(r["effective_count"])
            for _, r in cm.iterrows()}


def count_agreement_scatter(n_exp, behav, device="cpu", include_biregular=True):
    """Do the two counts agree? x = behavioural n_soft (model-only), y = ExactPD
    effective_count (decomposition participation ratio), one point per trained
    (model, d_mlp, p) cell, plus the biregular benchmarks (their fits are cached
    by the CodeAnsatz registry). Points on y=x: the counts agree; above: mass
    spread exceeds computation (smeared/superposed); below: mass concentrated on
    fewer features than are behaviourally computed."""
    exact = _exactpd_counts(n_exp)
    ntok = results_io._fmt_n(n_exp)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    colors = {50: "tab:blue", 100: "tab:orange"}
    seen_lab = set()

    def scat(x, y, d_mlp, marker, label_key, label):
        lab = label if label_key not in seen_lab else None
        seen_lab.add(label_key)
        ax.scatter([x], [y], marker=marker, s=45, alpha=0.8,
                   color=colors[d_mlp], label=lab)

    for (label, d_mlp, pf), x in behav.items():
        y = exact.get((label, d_mlp, pf))
        if y is None:
            continue
        marker = "s" if label == "LinearMLP" else "o"
        scat(x, y, d_mlp, marker, (marker, d_mlp),
             f"{'LinearMLP' if marker == 's' else 'EmbedMLP'} d_mlp={d_mlp}")
        # annotate the strongly divergent points (|x - y| > 25)
        if abs(x - y) > 25:
            ax.annotate(f"{'Lin' if label == 'LinearMLP' else 'd_emb=' + label} p={pf:g}",
                        (x, y), fontsize=6, xytext=(4, 3), textcoords="offset points")

    if include_biregular:
        for key, code in sorted(_ANSATZE.items()):
            for (pf, _, _), res in code._eval_cache.items():
                if "n_soft" in res:
                    scat(res["n_soft"], res["eff_count"], 50, "^", ("^", 50, "bireg"),
                         "biregular codes (N=50)")

    lim = [0, N_FEATURES * 1.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="y = x (agreement)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("behavioural n_soft  (Σ max(0, R²) — what the model computes)")
    ax.set_ylabel("ExactPD effective_count  (weight-mass participation ratio)")
    ax.set_title(f"Feature-count agreement: behavioural vs ExactPD — n={ntok}\n"
                 f"above y=x: smeared mass ≠ computed;  below: concentrated mass, graceful degradation",
                 fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    out = os.path.join(OVERVIEW_FIG, f"count_agreement_n{ntok}.png")
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
            grid[(K, vname)] = {p: _ansatz(K, sw).fit_eval(float(p), n_exp, device=device)
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
    behav = behavioural_count_heatmap(a.n_exp, device=a.device)
    if not a.no_biregular:
        biregular_overview(a.n_exp, device=a.device)   # fills the fit cache (incl. n_soft)
    count_agreement_scatter(a.n_exp, behav, device=a.device,
                            include_biregular=not a.no_biregular)
    write_loss_tables(a.n_exp, device=a.device)        # reuses cached K5 fits


if __name__ == "__main__":
    main()
