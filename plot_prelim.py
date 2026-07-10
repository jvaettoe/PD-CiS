"""Composite loss-curve plot for one sweep cell: fix (model, d_emb, n_exp) and
overlay every available sparsity p, one coloured line per p, a panel per loss
term (APD row + SPD row). Reads the per-run curve JSONs written by the sweep and
reuses results_io/figutil for paths, so it tracks the same tag namespacing.

  ./.venv/bin/python plot_prelim.py                          # d_emb=200, n=2, tag=prelim
  ./.venv/bin/python plot_prelim.py --demb 50 --n 2.5
  ./.venv/bin/python plot_prelim.py --tag gpu --demb 1000 --n 4 --model EmbedMLP
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import results_io
import figutil

# Loss terms to panel, per method (skipped silently if a curve lacks them).
PANELS = {
    "APD": ["total", "faithful", "minimal", "simple"],
    "OracleAPD": ["total", "faithful", "minimal", "simple"],
    "SPD": ["total", "faithful", "recon", "importance"],
}


def _fmt_n(n_exp: float) -> str:
    return f"{float(n_exp):g}"


def discover_ps(model: str, demb, ntok: str, method: str) -> list[str]:
    """All sparsity tokens p for which a curve JSON exists for this cell."""
    demb_tok = "NA" if demb is None else str(demb)
    pat = os.path.join(results_io.CURVES_DIR,
                       f"{model}_demb{demb_tok}_p*_n{ntok}_{method}.json")
    ps = []
    for f in glob.glob(pat):
        m = re.search(rf"_demb{re.escape(demb_tok)}_p([0-9.]+)_n{re.escape(ntok)}_",
                      os.path.basename(f))
        if m:
            ps.append(m.group(1))
    return sorted(set(ps), key=float)


def load_curves(model: str, demb, p: str, ntok: str, method: str):
    run_id = f"{model}_demb{'NA' if demb is None else demb}_p{p}_n{ntok}_{method}"
    path = results_io.curve_path(run_id)
    if not os.path.exists(path):
        return None, None
    blob = json.load(open(path))
    return blob["curves"], blob.get("config", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="prelim", help="results/figures namespace (default: prelim)")
    ap.add_argument("--model", default="EmbedMLP", help="model class name")
    ap.add_argument("--demb", type=int, default=200,
                    help="embedding dim (ignored/NA for LinearMLP)")
    ap.add_argument("--n", type=float, default=2.0, dest="n_exp", help="loss exponent")
    ap.add_argument("--methods", nargs="+", default=["APD", "SPD"],
                    choices=list(PANELS), help="methods to row-stack")
    args = ap.parse_args()

    results_io.set_tag(args.tag)
    figutil.set_tag(args.tag)
    demb = None if args.model == "LinearMLP" else args.demb
    ntok = _fmt_n(args.n_exp)

    # Union of sparsities present across the requested methods -> stable colours.
    ps = sorted({p for m in args.methods for p in discover_ps(args.model, demb, ntok, m)},
                key=float)
    if not ps:
        raise SystemExit(f"No curve JSONs found for {args.model} d_emb={demb} n={ntok} "
                         f"under results/{args.tag}/curves. Nothing to plot.")
    cmap = matplotlib.colormaps["viridis"]
    colors = {p: cmap(i / max(1, len(ps) - 1)) for i, p in enumerate(ps)}

    ncol = max(len(PANELS[m]) for m in args.methods)
    fig, axes = plt.subplots(len(args.methods), ncol,
                             figsize=(4.2 * ncol, 4.0 * len(args.methods)),
                             squeeze=False, constrained_layout=True)
    for row, method in enumerate(args.methods):
        terms = PANELS[method]
        for col in range(ncol):
            ax = axes[row][col]
            if col >= len(terms):
                ax.axis("off")
                continue
            term = terms[col]
            drew = False
            for p in ps:
                curves, cfg = load_curves(args.model, demb, p, ntok, method)
                if not curves or term not in curves:
                    continue
                y = np.clip(np.asarray(curves[term], float), 1e-12, None)
                k = cfg.get("k")
                lbl = f"p={p}" + (f" (k={k})" if method in ("APD", "OracleAPD") and k else "")
                ax.plot(np.arange(len(y)), y, color=colors[p], lw=1.6, label=lbl)
                drew = True
            ax.set_yscale("log")
            ax.set_title(f"{method}: {term}", fontsize=11, fontweight="bold")
            ax.set_xlabel("fit step")
            ax.grid(True, alpha=0.3, which="both")
            if drew and col == 0:
                ax.legend(fontsize=8, title="sparsity")

    demb_lbl = "NA" if demb is None else demb
    fig.suptitle(f"Decomposition loss curves  |  {args.model}, d_emb={demb_lbl}, "
                 f"n_exp={ntok}  (color = data sparsity p)  [tag={args.tag}]", fontsize=13)

    out_dir = os.path.join(figutil.FIG_ROOT, figutil.n_seg(ntok), "composites")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f"loss_curves_{args.model}_demb{demb_lbl}_n{ntok}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}  ({len(ps)} sparsities: {', '.join(ps)})")


if __name__ == "__main__":
    main()
