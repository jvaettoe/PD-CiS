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


def plot_components(pd_obj, *, top: int = 16, params: dict | None = None,
                    usage: dict | None = None) -> dict:
    """Views of a fitted decomposition built on the fact that each component is
    (at rank 1) a rank-1 object = (read profile u_c, write profile v_c, coupling),
    so it is shown as its read/write feature vectors rather than a rank-1 100x100
    map or a one-point 'spectrum'. Returns {name: Figure}:
      'profiles'     -- top-`top` components (by effective norm), each a bar panel
                        of the read profile |u_c| (blue, up: which input feature
                        switches it on) and write profile |v_c| (red, down: which
                        output feature it drives). Same peak up & down = a clean
                        self-feature mechanism; different = a cross-feature one.
      'assignment'   -- two (C x n_features) heatmaps, reads |u| and writes |v|,
                        rows by effective norm. A clean decomposition is a
                        permutation in BOTH with matching columns; a component
                        bright in reads at i and writes at j is a cross-feature
                        mechanism (which the old |diagonal| view erased).
      'usage'        -- (only if `usage` from APD.usage_stats is passed) APD's OWN
                        importance: per-component batch-top-k selection frequency,
                        and the distribution of components-active-per-example vs k.
      'faithfulness' -- model.effective_matrix() vs the reconstructed (Σ components)
                        effective map vs their residual: WHERE reconstruction fails.
    `params` stamps a model/param provenance title on each figure."""
    import math
    import numpy as np

    eff_norm = pd_obj.component_effective_norms().cpu().numpy()
    order = np.argsort(eff_norm)[::-1]
    denom = float(eff_norm.max()) + 1e-12
    n_used = int((eff_norm > 0.01 * denom).sum())
    top_idx = [int(c) for c in order[:min(top, pd_obj.C)]]
    reads, writes = pd_obj.read_write_profiles()
    reads, writes = reads.cpu().numpy(), writes.cpu().numpy()
    nf = reads.shape[1]
    figs = {}

    # 1. read/write profiles -- the exact rank-1 signature of each top component
    ncol = min(4, len(top_idx)) or 1
    nrow = math.ceil(len(top_idx) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.1 * nrow + 0.6),
                             constrained_layout=True, squeeze=False)
    for i, c in enumerate(top_idx):
        ax = axes[i // ncol][i % ncol]
        u, v = np.abs(reads[c]), np.abs(writes[c])
        fr, fw = int(u.argmax()), int(v.argmax())
        ax.bar(np.arange(nf), u, color="steelblue", width=1.0)
        ax.bar(np.arange(nf), -v, color="indianred", width=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"c{c}: read {fr}{' = ' if fr == fw else ' → '}write {fw}   "
                     f"‖e‖={eff_norm[c]:.2g}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(top_idx), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(component_title(
        "Read (blue ↑) / write (red ↓) feature profiles", params,
        f"top {len(top_idx)} by effective norm; aligned peaks = a clean self-map"),
        fontsize=11)
    figs["profiles"] = fig

    # 2. read/write assignment heatmaps (replaces the |diagonal| view)
    ro, wo = np.abs(reads[order]), np.abs(writes[order])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=True)
    im1 = a1.imshow(ro, aspect="auto", cmap="magma", vmin=0.0, vmax=float(ro.max()) or 1.0)
    a1.set_title("reads |u|  (input feature that activates the component)", fontsize=10)
    a1.set_xlabel("feature index")
    a1.set_ylabel("component (sorted by effective norm)")
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
    im2 = a2.imshow(wo, aspect="auto", cmap="magma", vmin=0.0, vmax=float(wo.max()) or 1.0)
    a2.set_title("writes |v|  (output feature the component drives)", fontsize=10)
    a2.set_xlabel("feature index")
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
    fig.suptitle(component_title(
        "Component read / write assignment", params,
        f"{n_used}/{pd_obj.C} active; clean = a permutation in BOTH with matching columns"),
        fontsize=11)
    figs["assignment"] = fig

    # 3. usage / attribution -- APD's own importance notion (only if provided)
    if usage is not None:
        freq = usage["selection_freq"].cpu().numpy()
        counts = usage["per_example_count"].cpu().numpy().astype(int)
        k = usage.get("k")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        a1.bar(np.arange(len(freq)), np.sort(freq)[::-1], color="steelblue")
        a1.set_xlabel("component (sorted by selection frequency)")
        a1.set_ylabel("selection frequency")
        a1.set_title(f"batch-top-k selection frequency "
                     f"({int((freq > 0).sum())}/{len(freq)} ever selected)", fontsize=10)
        a1.grid(True, alpha=0.3)
        a2.hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5,
                color="slategray")
        if k is not None:
            a2.axvline(k, color="crimson", ls="--", lw=1.5, label=f"target k={k}")
            a2.legend()
        a2.set_xlabel("components active per example")
        a2.set_ylabel("# examples")
        a2.set_title(f"components per example (mean {counts.mean():.1f})", fontsize=10)
        a2.grid(True, alpha=0.3)
        fig.suptitle(component_title(
            "Component usage (attribution / batch-top-k)", params,
            "APD's behavioural importance, distinct from weight-space effective norm"),
            fontsize=11)
        figs["usage"] = fig

    # 4. faithfulness -- where the decomposition reconstructs the model (feature space)
    model_eff = pd_obj.model.effective_matrix().cpu().numpy()
    w_in_hat = pd_obj.reconstructed_weight("w_in")
    w_out_hat = pd_obj.reconstructed_weight("w_out")
    recon_eff = (pd_obj.model.project_in(w_in_hat)
                 @ pd_obj.model.project_out(w_out_hat)).detach().cpu().numpy()
    resid = model_eff - recon_eff
    vmax = max(float(np.abs(model_eff).max()), float(np.abs(recon_eff).max())) or 1.0
    rmax = float(np.abs(resid).max()) or 1.0
    rel = float(np.linalg.norm(resid) / (np.linalg.norm(model_eff) + 1e-12))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for ax, M, ttl, vm in [(axs[0], model_eff, "model effective map", vmax),
                           (axs[1], recon_eff, "reconstructed (Σ components)", vmax),
                           (axs[2], resid, f"residual (max {rmax:.1e})", rmax)]:
        im = ax.imshow(M, cmap="RdBu", vmin=-vm, vmax=vm)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("feature")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axs[0].set_ylabel("feature")
    fig.suptitle(component_title(
        "Faithfulness: model vs reconstructed effective map", params,
        f"relative residual {rel:.1e}"), fontsize=11)
    figs["faithfulness"] = fig

    return figs


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
        reads, writes = pd_obj.read_write_profiles()
        reads, writes = np.abs(reads.cpu().numpy()), np.abs(writes.cpu().numpy())
        eff_norm = pd_obj.component_effective_norms().cpu().numpy()
        order = np.argsort(eff_norm)[::-1]
        ro, wo = reads[order], writes[order]
        m = pd_obj.component_metrics()

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
