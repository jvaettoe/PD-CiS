"""Figure saving + plotters for the decomposition sweep.

Self-contained (no cross-project import). `save_fig` stamps the sweep/model
params into the sub-folder, the filename, AND the title, so any figure's
provenance is readable from the file browser and the image alike."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt

from results_io import _fmt_n

FIG_ROOT = os.path.join(os.path.dirname(__file__), "figures")


def save_fig(fig, subdir: str, filename: str, title: str,
             *, dpi: int = 150, close: bool = True, **params) -> str:
    """Save `fig` to FIG_ROOT/<subdir>/<filename>. The suptitle is `title` plus a
    second line `k=v | k=v | ...` built from **params (None values dropped), so
    every figure self-documents the config it came from."""
    out_dir = os.path.join(FIG_ROOT, subdir)
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
