"""Figure saving + plotters for the decomposition sweep.

Self-contained (no cross-project import). `save_fig` stamps the sweep/model
params into the sub-folder, the filename, AND the title, so any figure's
provenance is readable from the file browser and the image alike."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt

from results_io import _fmt_n

_BASE_FIG = os.path.join(os.path.dirname(__file__), "figures")
FIG_ROOT = _BASE_FIG


def set_tag(tag: str | None) -> None:
    """Namespace figures under figures/<tag> (tag=None -> figures/)."""
    global FIG_ROOT
    FIG_ROOT = os.path.join(_BASE_FIG, tag) if tag else _BASE_FIG


def n_seg(n_group) -> str:
    """Path segment grouping figures by loss exponent: 2.0 -> 'n2', 2.5 -> 'n2.5'.
    Empty string when n_group is None (no grouping)."""
    return f"n{_fmt_n(float(n_group))}" if n_group is not None else ""


def save_fig(fig, subdir: str, filename: str, title: str,
             *, n_group=None, dpi: int = 150, close: bool = True, **params) -> str:
    """Save `fig` to FIG_ROOT/[n<n_group>/]<subdir>/<filename>. The suptitle is
    `title` plus a second line `k=v | ...` from **params (None dropped). Pass
    n_group to file the figure under a per-loss-exponent subfolder."""
    out_dir = os.path.join(FIG_ROOT, n_seg(n_group), subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    param_line = " | ".join(f"{k}={v}" for k, v in params.items() if v is not None)
    fig.suptitle(f"{title}\n{param_line}" if param_line else title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_loss_curves(curves: dict[str, list[float]], *, method: str):
    """One semilog-y line per loss term (incl. total) over fit steps."""
    fig, ax = plt.subplots(constrained_layout=True)
    for term, values in curves.items():
        lw = 2.2 if term == "total" else 1.3
        ls = "-" if term == "total" else "--"
        ax.plot(range(len(values)), values, label=term, linewidth=lw, linestyle=ls)
    ax.set_yscale("log")
    ax.set_xlabel("fit step")
    ax.set_ylabel("loss (log)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    return fig


def plot_model_views(model, *, is_embed: bool) -> dict:
    """Reuse the model's own feature-space viz. Returns {name: Figure} for
    'weights' (feature-space read-in/read-out for EmbedMLP), 'effective'
    (feature->feature map), and 'bars' (per-neuron feature assignment)."""
    figs = {}

    figs["weights"] = model.plot_weights(project=is_embed)

    eff = model.effective_matrix()
    fig, ax = plt.subplots(constrained_layout=True)
    v = eff.abs().max().item()
    im = ax.imshow(eff.cpu().numpy(), cmap="RdBu", vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("effective feature -> feature map")
    figs["effective"] = fig

    ax_bars = model.plot_weight_bars(show_cbar=True)
    ax_bars.set_title("feature -> neuron assignment")
    figs["bars"] = ax_bars.get_figure()

    return figs


def plot_components(pd_obj, *, top: int = 16) -> dict:
    """Feature-space views of a fitted decomposition's learned components -- the
    per-component analog of plot_model_views. Returns {name: Figure}:
      'assignment' -- (C x n_features) heatmap of each component's mass per
                      feature (|diag of its effective map|); a near-permutation
                      means one clean component per feature.
      'effective'  -- grid of the top-`top` (by norm) components' feature->feature
                      maps; each ~rank-1 blob on one feature = a found mechanism.
      'spectrum'   -- per-component singular values (w_in) of those top
                      components; rank-1 = the intended minimal mechanism.
    Components are ranked by norm because a fit typically leaves most at ~0."""
    import math
    import numpy as np

    norms = pd_obj.component_norms().cpu().numpy()
    order = np.argsort(norms)[::-1]
    n_used = int((norms > 1e-6 * (float(norms.max()) + 1e-12)).sum())
    top_idx = [int(c) for c in order[:min(top, pd_obj.C)]]
    figs = {}

    # 1. component -> feature assignment (rows sorted by norm, most-used first)
    assign = pd_obj.component_feature_assignment().cpu().numpy()[order]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 6))
    im = ax.imshow(assign, aspect="auto", cmap="magma")
    ax.set_xlabel("feature index")
    ax.set_ylabel("component (sorted by norm)")
    ax.set_title(f"component -> feature assignment  "
                 f"({n_used}/{pd_obj.C} components used)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figs["assignment"] = fig

    # 2. per-component effective feature->feature maps (top components)
    effs = [pd_obj.component_effective(c).cpu().numpy() for c in top_idx]
    vmax = max((float(np.abs(e).max()) for e in effs), default=1.0) or 1.0
    ncol = min(4, len(top_idx)) or 1
    nrow = math.ceil(len(top_idx) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow),
                             constrained_layout=True, squeeze=False)
    for i, (c, eff) in enumerate(zip(top_idx, effs)):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(eff, cmap="RdBu", vmin=-vmax, vmax=vmax)
        f = int(np.abs(np.diag(eff)).argmax())
        ax.set_title(f"comp {c}: feat {f}  (‖·‖={norms[c]:.2g})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(top_idx), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"per-component effective feature -> feature maps (top {len(top_idx)})")
    figs["effective"] = fig

    # 3. singular-value spectrum of the top components (w_in)
    fig, ax = plt.subplots(constrained_layout=True)
    for c in top_idx:
        sv = pd_obj.component_singular_values(c, "w_in").cpu().numpy()
        ax.plot(range(1, len(sv) + 1), np.clip(sv, 1e-12, None),
                marker=".", lw=1, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("sigma (log)")
    ax.set_title(f"per-component singular values (w_in), top {len(top_idx)} by norm")
    ax.grid(True, alpha=0.3, which="both")
    figs["spectrum"] = fig

    return figs


def _as_records(summary):
    """Accept either a pandas.DataFrame or a list[dict] and return list[dict]."""
    if hasattr(summary, "to_dict"):
        return summary.to_dict("records")
    return list(summary)


def plot_sweep_summary(summary, *, method: str, n_exp: float, term: str = "total"):
    """Final decomposition loss vs d_emb, one series per sparsity p, for a fixed
    (method, n_exp, term). Uses only EmbedMLP rows (d_emb varies there);
    consumes the output of results_io.load_summary()."""
    rows = _as_records(summary)
    target_n = _fmt_n(float(n_exp))

    def _has_demb(r):
        d = str(r.get("d_emb"))
        return d not in ("", "NA", "nan", "None")

    by_p = defaultdict(list)
    for r in rows:
        if (str(r.get("model")) == "EmbedMLP"
                and str(r.get("method")) == method
                and str(r.get("term")) == term
                and _has_demb(r)
                and _fmt_n(float(r["n_exp"])) == target_n):
            by_p[str(r["p"])].append((float(r["d_emb"]), float(r["final"])))

    fig, ax = plt.subplots(constrained_layout=True)
    for p in sorted(by_p, key=float):
        pts = sorted(by_p[p])
        xs = [d for d, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", label=f"p={p}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d_emb (log)")
    ax.set_ylabel(f"final {term} loss (log)")
    if by_p:
        ax.legend(title="sparsity p", fontsize="small")
    ax.grid(True, alpha=0.3, which="both")
    return fig
