"""Figure saving + plotters for the decomposition sweep.

Self-contained (no cross-project import). `save_fig` stamps the sweep/model
params into the sub-folder, the filename, AND the title, so any figure's
provenance is readable from the file browser and the image alike."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt

from results_io import _fmt_n, parse_run_id
from pd_viz import component_title  # noqa: F401  (re-export; single-PD figures now live on PDViz)

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


def plot_components_vs_p(entries: list, *, params: dict | None = None):
    """Cross-sparsity comparison for ONE model width: how the decomposition's
    components change as the data sparsity p sweeps. ROWS = p (each `entries` item),
    COLUMNS = [reads |u| assignment | writes |v| assignment | usage selection-freq].

    Each entry is a dict {p, k, pd_obj, usage} (usage from pd_obj.usage_stats, or
    None). Reuses the SAME primitives as plot_components -- read_write_profiles(),
    component_effective_norms() (row sort key), component_metrics() (row scalars) --
    so the per-p rows read exactly like the per-cell 'assignment'/'usage' views,
    just stacked for comparison. Heatmap colour is normalised PER ROW (each p's own
    max), since read/write magnitudes vary by orders of magnitude across p; the
    STRUCTURE (which feature each component reads/writes), not the scale, is what
    changes with p. Returns the Figure."""
    import numpy as np

    nrow = len(entries)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 3.1 * nrow + 0.8),
                             squeeze=False, constrained_layout=True)
    for i, e in enumerate(entries):
        pd_obj, k = e["pd_obj"], e.get("k")
        # One SVD pass per row, reused for the heatmaps AND component_metrics (which
        # would otherwise recompute the same read/write profiles).
        reads_t, writes_t, _, _, _ = pd_obj.component_signatures()
        eff_norm_t = pd_obj.component_effective_norms()
        reads, writes = np.abs(reads_t.cpu().numpy()), np.abs(writes_t.cpu().numpy())
        eff_norm = eff_norm_t.cpu().numpy()
        order = np.argsort(eff_norm)[::-1]
        ro, wo = reads[order], writes[order]
        m = pd_obj.component_metrics(reads=reads_t, writes=writes_t, eff_norm=eff_norm_t)

        a_r, a_w, a_u = axes[i]
        im_r = a_r.imshow(ro, aspect="auto", cmap="magma", vmin=0.0,
                          vmax=float(ro.max()) or 1.0)
        fig.colorbar(im_r, ax=a_r, fraction=0.046, pad=0.04)
        im_w = a_w.imshow(wo, aspect="auto", cmap="magma", vmin=0.0,
                          vmax=float(wo.max()) or 1.0)
        fig.colorbar(im_w, ax=a_w, fraction=0.046, pad=0.04)

        # row identity + scalar decomposition metrics, on the leftmost y-label
        a_r.set_ylabel(f"p={e['p']}  (k={k})\n"
                       f"eff_count={m['effective_count']:.0f}\n"
                       f"mono={m['mean_monosemanticity']:.2f}\n"
                       f"cov={m['features_covered']}/{ro.shape[1]}",
                       fontsize=9)
        for ax in (a_r, a_w):
            ax.set_yticks([])

        # usage: batch-top-k selection frequency (APD's own importance), sorted
        usage = e.get("usage")
        if usage is not None:
            freq = np.sort(usage["selection_freq"].cpu().numpy())[::-1]
            a_u.bar(np.arange(len(freq)), freq, color="steelblue", width=1.0)
            a_u.set_ylabel("selection freq", fontsize=8)
            a_u.set_xlim(-0.5, len(freq) - 0.5)
            a_u.grid(True, alpha=0.3)
            ever = int((usage["selection_freq"].cpu().numpy() > 0).sum())
            a_u.text(0.97, 0.92, f"{ever}/{pd_obj.C} used", transform=a_u.transAxes,
                     ha="right", va="top", fontsize=8)
        else:
            a_u.axis("off")

        if i == 0:
            a_r.set_title("reads |u|  (input feature that activates the component)",
                          fontsize=10)
            a_w.set_title("writes |v|  (output feature it drives)", fontsize=10)
            a_u.set_title("batch-top-k selection frequency", fontsize=10)
        if i == nrow - 1:
            a_r.set_xlabel("feature index")
            a_w.set_xlabel("feature index")
            a_u.set_xlabel("component (sorted by selection freq)")

    fig.suptitle(component_title(
        "Components vs sparsity p  (rows = p, effective-norm-sorted)", params,
        "same model width; how read/write structure & usage change with p"),
        fontsize=12)
    return fig


def plot_faithfulness_vs_p(series: dict, *, params: dict | None = None):
    """How well the decomposition reconstructs the model's linearised feature->feature
    map, as a function of sparsity p, one line per model width. `series` maps
    d_mlp -> list of (p_float, raw_resid, rel_resid) where raw_resid = ‖W_model -
    W_recon‖_F on the effective matrix and rel_resid = raw / ‖W_model‖_F (the same
    residual the per-cell 'faithfulness' view shows, summarised across p). Two panels:
      left  = raw Frobenius residual ‖resid‖_F  (absolute reconstruction error)
      right = relative residual ‖resid‖_F / ‖W_model‖_F  (dimensionless, comparable
              across widths whose effective maps differ in magnitude).
    This is the WEIGHT-SPACE reconstruction of the linear map, distinct from the
    behavioural per-example 'faithful' loss term stored in summary.csv."""
    import numpy as np

    fig, (a_raw, a_rel) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for d_mlp in sorted(series):
        pts = sorted(series[d_mlp])
        ps = [p for p, _, _ in pts]
        raw = [r for _, r, _ in pts]
        rel = [q for _, _, q in pts]
        a_raw.plot(ps, raw, marker="o", label=f"d_mlp={d_mlp}")
        a_rel.plot(ps, rel, marker="o", label=f"d_mlp={d_mlp}")
    for ax, ylab, ttl in [
        (a_raw, "‖resid‖_F", "raw Frobenius residual"),
        (a_rel, "‖resid‖_F / ‖W_model‖_F", "relative Frobenius residual"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("data sparsity p (log)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(component_title(
        "Reconstruction faithfulness vs sparsity p", params,
        "effective feature->feature map residual (lower = more faithful)"),
        fontsize=12)
    return fig


def plot_losses_vs_p(summary, *, method: str, n_exp: float, model: str = "EmbedMLP",
                     d_emb=None, terms=None):
    """The decomposition's own loss terms as a function of sparsity p, one small
    log-log panel per term, one line per model width d_mlp. Consumes the tidy long
    summary (results_io.load_summary output) exactly like plot_sweep_summary, but
    pivots the axes to x=p, y=final loss. `final` is the recorded end-of-fit value.
    Filters to (model, method, n_exp, d_emb); default terms are APD's
    faithful/minimal/simple/total (absent terms are skipped)."""
    rows = _as_records(summary)
    terms = terms or ["faithful", "minimal", "simple", "total"]
    target_n = _fmt_n(float(n_exp))

    # d_emb match that survives pandas dtype coercion. Once any LinearMLP row (blank
    # d_emb) lands in summary.csv the whole column becomes float, so EmbedMLP's 200
    # stringifies as '200.0' -- an exact string compare then silently matches nothing.
    # Mirror plot_sweep_summary: treat blank/NA/nan/None as the no-d_emb (LinearMLP)
    # bucket, else compare numerically. d_emb=None selects the LinearMLP bucket.
    def _demb_match(r):
        d = str(r.get("d_emb"))
        has = d not in ("", "NA", "nan", "None")
        if d_emb is None:
            return not has
        return has and float(d) == float(d_emb)

    # term -> d_mlp -> list[(p_float, final)]
    by = {t: defaultdict(list) for t in terms}
    present = set()
    for r in rows:
        if (str(r.get("model")) == model
                and str(r.get("method")) == method
                and str(r.get("term")) in by
                and _demb_match(r)
                and _fmt_n(float(r["n_exp"])) == target_n):
            by[str(r["term"])][str(r.get("d_mlp"))].append(
                (float(r["p"]), float(r["final"])))
            present.add(str(r["term"]))

    shown = [t for t in terms if t in present]
    ncol = len(shown) or 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.0),
                             squeeze=False, constrained_layout=True)
    for ax, term in zip(axes[0], shown):
        for d_mlp in sorted(by[term], key=lambda s: int(s) if s.isdigit() else s):
            pts = sorted(by[term][d_mlp])
            ax.plot([p for p, _ in pts], [v for _, v in pts],
                    marker="o", label=f"d_mlp={d_mlp}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("data sparsity p (log)")
        ax.set_ylabel("final loss (log)")
        ax.set_title(term, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(component_title(
        f"{method} loss terms vs sparsity p", {"model": model, "d_emb": d_emb,
                                               "n_exp": n_exp, "method": method},
        "final decomposition loss per term (end of fit)"), fontsize=12)
    return fig


def plot_behavioural_faithfulness_vs_p(series: dict, *, params: dict | None = None):
    """How well the FULL decomposition (sum of all C components) reproduces the
    model's OUTPUTS, as a function of sparsity p, one line per width. This is the
    BEHAVIOURAL (output-space) faithfulness -- distinct from both the weight-space
    effective-map residual (plot_faithfulness_vs_p) and the stored `faithful` loss
    term (a raw-weight MSE). `series` maps d_mlp -> [(p_float, out_mse, rel_out_err)]
    where out_mse = mean((forward_dense(x) - model(x))^2) and rel_out_err =
    ‖forward_dense(x) - model(x)‖_F / ‖model(x)‖_F on a fresh batch. Two panels:
      left  = output MSE            (absolute behavioural error)
      right = relative output error (dimensionless, comparable across widths)."""
    fig, (a_mse, a_rel) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for d_mlp in sorted(series):
        pts = sorted(series[d_mlp])
        ps = [p for p, _, _ in pts]
        mse = [m for _, m, _ in pts]
        rel = [r for _, _, r in pts]
        a_mse.plot(ps, mse, marker="o", label=f"d_mlp={d_mlp}")
        a_rel.plot(ps, rel, marker="o", label=f"d_mlp={d_mlp}")
    for ax, ylab, ttl in [
        (a_mse, "mean (ŷ − y)²", "output MSE"),
        (a_rel, "‖ŷ − y‖_F / ‖y‖_F", "relative output error"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("data sparsity p (log)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(component_title(
        "Behavioural faithfulness vs sparsity p", params,
        "output of Σ-all-components forward vs model (lower = more faithful)"),
        fontsize=12)
    return fig


def save_component_figs(figs: dict, run_id: str, *, n_group=None,
                        k=None, rank=None, dpi: int = 130, close: bool = True) -> str:
    """Save the plot_components() figures with descriptive, model-identifying file
    names (matching the model-view / loss-curve plots) under
    figures/[n<n>/]components/<method>/. File name:
        <run_id>[_k<k>][_rank<rank>]_<view>.png
    e.g. EmbedMLP_demb200_p0.1_n2_APD_k10_rank1_assignment.png. The per-method subdir
    (parsed from the run_id) keeps each PD method's figures separate, matching the
    loss_curves/<method>/ layout. Returns the dir."""
    try:
        method = parse_run_id(run_id)["method"]
    except ValueError:
        method = ""     # unparseable run_id -> fall back to the flat components/ dir
    out_dir = os.path.join(FIG_ROOT, n_seg(n_group), "components", method)
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
