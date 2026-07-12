"""Figure saving + plotters for the decomposition sweep.

Self-contained (no cross-project import). `save_fig` stamps the sweep/model
params into the sub-folder, the filename, AND the title, so any figure's
provenance is readable from the file browser and the image alike."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt

from results_io import _fmt_n

FIG_ROOT = os.path.join(os.path.dirname(__file__), "figures")


def set_tag(tag: str | None) -> None:
    """No-op, kept for call-site compatibility. Figures are organised by TOPIC
    (loss exponent n -> category -> descriptive filename), not by run provenance:
    the filename already encodes model/d_emb/p/n/method/k, so a re-render from any
    results tag overwrites the same file in place instead of spawning a duplicate
    tree. Which run produced a figure lives in results/<tag>/summary.csv, not the
    path. (results_io.set_tag still namespaces the underlying data.)"""
    return None


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


def component_title(view: str, params: dict | None, diagnostic: str = "") -> str:
    """Two-line figure title identifying the decomposition's model:
        '<view> — <Model>  (d_emb=.., p=.., n=.., <method>)'
        'C=.. | k=.. | rank=..  |  <diagnostic>'
    A human-readable headline (view + model) over the fixed decomposition params,
    so the model a decomposition belongs to is obvious from the image."""
    p = params or {}
    bits = []
    if p.get("d_emb") not in (None, "NA"):
        bits.append(f"d_emb={p['d_emb']}")
    if p.get("p") is not None:
        bits.append(f"p={p['p']}")
    if p.get("n_exp") is not None:
        bits.append(f"n={_fmt_n(p['n_exp'])}")
    if p.get("method"):
        bits.append(str(p["method"]))
    model_line = str(p.get("model", ""))
    if bits:
        model_line += f"  ({', '.join(bits)})"
    head = f"{view} — {model_line}" if model_line else view
    fixed = " | ".join(f"{k}={p[k]}" for k in ("C", "k", "rank") if p.get(k) is not None)
    second = f"{fixed}  |  {diagnostic}" if fixed and diagnostic else (fixed or diagnostic)
    return f"{head}\n{second}" if second else head


def plot_components(pd_obj, *, top: int = 16, params: dict | None = None) -> dict:
    """Feature-space views of a fitted decomposition's learned components -- the
    per-component analog of plot_model_views. Returns {name: Figure}:
      'assignment' -- (C x n_features) heatmap of each component's |diagonal|
                      self-map mass (linear scale, magma), rows sorted by EFFECTIVE
                      norm, with a companion sidebar of each component's
                      feature-space magnitude ‖effective‖. Read together: a bright
                      row = a monosemantic component on one feature; a dark row next
                      to a TALL sidebar bar = active but off-diagonal (superposed); a
                      dark row + short bar = negligible.
      'effective'  -- grid of the top-`top` (by EFFECTIVE norm) components'
                      feature->feature maps (each panel scaled independently); a
                      rank-1 blob on the diagonal = a clean found mechanism,
                      off-diagonal structure = superposition.
      'spectrum'   -- per-component singular values (w_in) of those top components.
    Components are ranked by their feature-space effective norm, since raw weight
    norm can be near-uniform even when most components do no feature-space work.
    `params` (model/d_emb/p/n_exp/method/C/k/rank) stamps a provenance line on each
    figure so the decomposition's model is readable from the image itself."""
    import math
    import numpy as np

    eff_norm = pd_obj.component_effective_norms().cpu().numpy()
    order = np.argsort(eff_norm)[::-1]
    eff_sorted = eff_norm[order]
    denom = float(eff_norm.max()) + 1e-12
    n_used = int((eff_norm > 0.01 * denom).sum())
    top_idx = [int(c) for c in order[:min(top, pd_obj.C)]]
    figs = {}

    # 1. |diagonal| assignment (linear heatmap, rows by effective norm) + magnitude sidebar
    assign = pd_obj.component_feature_assignment().cpu().numpy()[order]
    floor = max(float(eff_sorted.max()) * 1e-4, 1e-12)
    fig, (axh, axb) = plt.subplots(
        1, 2, figsize=(10, 6.4), width_ratios=[8, 1.5], constrained_layout=True)
    im = axh.imshow(assign, aspect="auto", cmap="magma",
                    vmin=0.0, vmax=float(assign.max()) or 1.0)
    axh.set_xlabel("feature index")
    axh.set_ylabel("component (sorted by effective norm)")
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    y = np.arange(len(eff_sorted))
    axb.plot(np.clip(eff_sorted, floor, None), y, lw=1.4, color="steelblue")
    axb.fill_betweenx(y, floor, np.clip(eff_sorted, floor, None), alpha=0.25, color="steelblue")
    axb.set_xscale("log")
    axb.set_ylim(len(eff_sorted) - 0.5, -0.5)   # align row order with the heatmap
    axb.set_yticks([])
    axb.set_xlabel("‖effective‖")
    axb.set_title("feature-space\nmagnitude", fontsize=9)
    axb.grid(True, alpha=0.3, which="both")
    fig.suptitle(component_title(
        "Component assignment", params,
        f"{n_used}/{pd_obj.C} components active in feature space"), fontsize=11)
    figs["assignment"] = fig

    # 2. per-component effective feature->feature maps (top components by effective norm).
    # Each panel is scaled to ITS OWN max, not a shared vmax, so a dominant component
    # can't wash the weaker-but-real panels to blank; the per-panel ‖eff‖ title carries
    # the true magnitude that per-panel scaling drops.
    effs = [pd_obj.component_effective(c).cpu().numpy() for c in top_idx]
    ncol = min(4, len(top_idx)) or 1
    nrow = math.ceil(len(top_idx) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow + 0.4),
                             constrained_layout=True, squeeze=False)
    for i, (c, eff) in enumerate(zip(top_idx, effs)):
        ax = axes[i // ncol][i % ncol]
        v = float(np.abs(eff).max()) or 1.0
        ax.imshow(eff, cmap="RdBu", vmin=-v, vmax=v)
        f = int(np.abs(np.diag(eff)).argmax())
        ax.set_title(f"comp {c}: feat {f}  (‖eff‖={eff_norm[c]:.2g})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(top_idx), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(component_title(
        "Per-component effective maps", params,
        f"top {len(top_idx)} by effective norm, each panel scaled independently"),
        fontsize=11)
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
    ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(component_title(
        "Component singular-value spectra", params,
        f"top {len(top_idx)} by effective norm"), fontsize=11)
    figs["spectrum"] = fig

    return figs


def save_component_figs(figs: dict, run_id: str, *, n_group=None,
                        k=None, rank=None, dpi: int = 130, close: bool = True) -> str:
    """Save the plot_components() figures with descriptive, model-identifying file
    names (matching the model-view / loss-curve plots) under
    figures/[n<n>/]components/. File name:
        <run_id>[_k<k>][_rank<rank>]_<view>.png
    e.g. EmbedMLP_demb200_p0.1_n2_APD_k10_rank1_assignment.png. Returns the dir."""
    out_dir = os.path.join(FIG_ROOT, n_seg(n_group), "components")
    os.makedirs(out_dir, exist_ok=True)
    ktok = f"_k{k}" if k is not None else ""
    rtok = f"_rank{rank}" if rank is not None else ""
    for view, fig in figs.items():
        fig.savefig(os.path.join(out_dir, f"{run_id}{ktok}{rtok}_{view}.png"),
                    dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig)
    return out_dir


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
