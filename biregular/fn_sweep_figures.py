"""Figures for the (n_features, n_neurons) sweep (biregular/fn_sweep.py).

Grid convention: x = F (n_features), y = N (n_neurons), both on their index
positions with value tick labels (the grids are log-spaced lists, not linear).
Colors follow the repo-neutral dataviz palette: categorical blue/orange/aqua in
fixed roles, sequential single-hue blue ramp for magnitudes, gray for ties and
missing cells.
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"])
PREF_COLORS = {"feature": "#2a78d6", "neuron": "#eb6834",
               "tie": "#e1e0d9", "interaction": "#e1e0d9"}  # interaction shown as tie
TERM_COLORS = {"stoch_recon": "#2a78d6", "pgd_recon": "#eb6834",
               "importance": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MISSING = "#f0efec"
ANCHOR = (100, 50)


def _grid_axes(ax, f_grid, n_grid):
    ax.set_xticks(range(len(f_grid)))
    ax.set_xticklabels(f_grid, color=INK2)
    ax.set_yticks(range(len(n_grid)))
    ax.set_yticklabels(n_grid, color=INK2)
    ax.set_xlabel("n_features F", color=INK)
    ax.set_ylabel("n_neurons N", color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, len(f_grid) - 0.5)
    ax.set_ylim(-0.5, len(n_grid) - 0.5)


def _diag_and_anchor(ax, f_grid, n_grid):
    """Mark cells with N == F."""
    for i, F_ in enumerate(f_grid):
        for j, N_ in enumerate(n_grid):
            if N_ == F_:
                ax.plot(i, j, marker="x", ms=6, mew=1.2, color=INK2, zorder=4)


def _cell_rect(ax, i, j, color):
    if color == MISSING:  # pending cell: faint outline only, no fill
        ax.add_patch(Rectangle((i - 0.47, j - 0.47), 0.94, 0.94, fill=False,
                               edgecolor=GRID, lw=0.6, zorder=1))
        return
    ax.add_patch(Rectangle((i - 0.47, j - 0.47), 0.94, 0.94, facecolor=color,
                           edgecolor="white", lw=1.0, zorder=2))


def kstar_heatmap(rows, f_grid, n_grid, fig_dir):
    by_cell = {(r["F"], r["N"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    kmax = max(r["K_star"] for r in rows)
    L4_FLOOR = 1e-12  # overcomplete cells can be exactly 0 (dedicated neurons)
    l4s = [max(r["l4_star"], L4_FLOOR) for r in rows]
    lo, hi = min(l4s), max(l4s)
    if lo >= hi:  # single cell (or identical values): expand for a valid LogNorm
        lo, hi = lo * 0.5, hi * 2.0
    norm_l4 = LogNorm(vmin=lo, vmax=hi)
    for ax, what in zip(axes, ("K_star", "l4_star")):
        _grid_axes(ax, f_grid, n_grid)
        for i, F_ in enumerate(f_grid):
            for j, N_ in enumerate(n_grid):
                r = by_cell.get((F_, N_))
                if r is None:
                    _cell_rect(ax, i, j, MISSING)
                    continue
                if what == "K_star":
                    frac = r["K_star"] / kmax
                    _cell_rect(ax, i, j, SEQ_BLUE(frac))
                    ax.text(i, j, str(r["K_star"]), ha="center", va="center",
                            fontsize=9, color="white" if frac > 0.55 else INK,
                            zorder=3)
                else:
                    frac = norm_l4(max(r["l4_star"], L4_FLOOR))
                    _cell_rect(ax, i, j, SEQ_BLUE(frac))
                    ax.text(i, j, f"{r['l4_star']:.0e}".replace("e-0", "e-"),
                            ha="center", va="center", fontsize=7,
                            color="white" if frac > 0.55 else INK, zorder=3)
        _diag_and_anchor(ax, f_grid, n_grid)
    axes[0].set_title("L4-optimal codeword length K*", color=INK)
    axes[1].set_title("held-out L4 at K* (log color)", color=INK)
    fig.suptitle("biregular code, transpose decoder, p = 0.001 (x = N=F)",
                 color=INK2, fontsize=9, y=0.99)
    fig.tight_layout()
    path = os.path.join(fig_dir, "kstar_heatmap.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def basis_preference_map(rows, f_grid, n_grid, fig_dir, tie_ratio):
    by_cell = {(r["F"], r["N"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, matrix, title in zip(
            axes, ("w_in", "w_out"),
            ("W_in (read side): feature = rows, neuron = columns",
             "W_out (write side): neuron = rows, feature = columns")):
        _grid_axes(ax, f_grid, n_grid)
        for i, F_ in enumerate(f_grid):
            for j, N_ in enumerate(n_grid):
                r = by_cell.get((F_, N_))
                if r is None:
                    _cell_rect(ax, i, j, MISSING)
                    continue
                pref, margin = r[f"{matrix}_pref"], r[f"{matrix}_margin"]
                _cell_rect(ax, i, j, PREF_COLORS[pref])
                dark_bg = pref in ("feature", "neuron")
                ax.text(i, j, f"{margin:.2f}x", ha="center", va="center",
                        fontsize=6.5, color="white" if dark_bg else INK2, zorder=3)
        _diag_and_anchor(ax, f_grid, n_grid)
        ax.set_title(title, color=INK, fontsize=10)
    handles = [Patch(facecolor=PREF_COLORS[k], label=lbl) for k, lbl in
               (("feature", "feature basis wins"), ("neuron", "neuron basis wins"),
                ("tie", f"tie (< {tie_ratio:.2f}x)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8)
    fig.suptitle("Oracle-gated VPD-loss basis preference per matrix "
                 "(cell label = loss-ratio margin; x = N=F)",
                 color=INK2, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(fig_dir, "basis_preference_map.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def exact_solo_figures(recs, f_grid, n_grid, fig_dir):
    """Leak/attenuation ratio of the full model relative to its ExactPD solo
    mechanisms, for MSE and L4: heatmap over the (F, N) grid + scatter vs
    compression F/N. K* = 1 cells have leak = atten = 0.0 bit-exactly
    (a = 1, b = 0), so their ratio is 0/0: shown as '0/0' in the map and as a
    marked band at the bottom of the scatter."""
    by_cell = {(r["F"], r["N"]): r for r in recs}
    for loss in ("mse", "l4"):
        key = f"ratio_{loss}"
        label = loss.upper()
        ratios = [r[key] for r in recs if r[key]]
        lo, hi = min(ratios), max(ratios)
        if lo >= hi:
            lo, hi = lo * 0.5, hi * 2.0
        norm = LogNorm(vmin=lo, vmax=hi)

        fig, ax = plt.subplots(figsize=(6.4, 4.9))
        _grid_axes(ax, f_grid, n_grid)
        for i, F_ in enumerate(f_grid):
            for j, N_ in enumerate(n_grid):
                r = by_cell.get((F_, N_))
                if r is None:
                    _cell_rect(ax, i, j, MISSING)
                    continue
                if not r[key]:
                    _cell_rect(ax, i, j, "#e1e0d9")
                    ax.text(i, j, "0/0", ha="center", va="center", fontsize=6,
                            color=INK2, zorder=3)
                    continue
                frac = norm(r[key])
                _cell_rect(ax, i, j, SEQ_BLUE(frac))
                ax.text(i, j, f"{r[key]:.2g}", ha="center", va="center",
                        fontsize=6.5, color="white" if frac > 0.55 else INK,
                        zorder=3)
        _diag_and_anchor(ax, f_grid, n_grid)
        ax.set_title(f"ExactPD full-vs-solo: leakage / self-attenuation "
                     f"({label}, one-hot probes; 0/0 = both losses exactly 0)",
                     color=INK, fontsize=9.5)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"exact_solo_ratio_map_{loss}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {path}")

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        band = lo / 30.0  # display band for the exact 0/0 cells
        for r in recs:
            if r[key]:
                ax.scatter(r["F"] / r["N"], r[key], s=34, color="#2a78d6",
                           edgecolor="white", linewidth=0.8, zorder=3)
            else:
                ax.scatter(r["F"] / r["N"], band, s=30, color="#c3c2b7",
                           edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, ls="--", lw=0.8, color=MUTED)
        ax.axvline(1.0, ls="--", lw=0.8, color=MUTED)
        ax.axhline(band, ls=":", lw=0.8, color=MUTED)
        ax.text(ax.get_xlim()[0] * 1.15, band * 1.25,
                "0/0 (leak = atten = 0 exactly)", color=INK2, fontsize=7)
        ax.set_xlabel("compression F / N", color=INK)
        ax.set_ylabel(f"leakage / self-attenuation ({label})", color=INK)
        ax.set_title("Interference- vs attenuation-dominated regimes",
                     color=INK, fontsize=10)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"exact_solo_ratio_scatter_{loss}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {path}")
    exact_solo_families(recs, fig_dir)


K_COLORS = {1: "#c3c2b7", 2: "#e87ba4", 3: "#2a78d6", 4: "#eb6834",
            5: "#1baf7a", 6: "#4a3aa7", 7: "#eda100"}


def exact_solo_families(recs, fig_dir):
    """Family structure of the leak/attenuation ratio (compressed cells only,
    N < F): K*-colored scatter (no connecting lines) + a fit check for the
    reduced ladder form ratio_p ~ C * (F-1)^alpha * [a^(2p)/(1-K a^2)^p]^delta
    (p = 2 MSE / 4 L4). The learned (a, b) absorb most of the geometry; pure
    (F, N, K) power laws fit poorly."""
    recs = [r for r in recs if r["N"] < r["F"] and r["ratio_mse"]]
    if len(recs) < 5:
        return
    F = np.array([r["F"] for r in recs], float)
    N = np.array([r["N"] for r in recs], float)
    K = np.array([r["K"] for r in recs], float)
    a = np.array([r["a"] for r in recs])
    for tag, label, p in (("l4", "L4", 4), ("mse", "MSE", 2)):
        y = np.array([r[f"ratio_{tag}"] for r in recs])
        # zero-parameter theory: hypergeometric ensemble closed form with each
        # cell's learned (a, b) inserted -- nothing fitted
        fit = np.array([r.get(f"pred_ensemble_ratio_{tag}") or np.nan
                        for r in recs])
        ok = np.isfinite(fit)
        r2 = 1 - ((np.log(y[ok]) - np.log(fit[ok])) ** 2).sum() / \
            ((np.log(y[ok]) - np.log(y[ok]).mean()) ** 2).sum()

        fig, axes = plt.subplots(1, 4, figsize=(18.4, 4.4), sharey=False)
        ax = axes[0]
        for r in recs:
            ax.scatter(r["F"] / r["N"], r[f"ratio_{tag}"], s=32,
                       color=K_COLORS.get(r["K"], INK), edgecolor="white",
                       lw=0.6, zorder=3)
        for Kv in sorted({r["K"] for r in recs}):
            ax.scatter([], [], color=K_COLORS.get(Kv, INK), label=f"K* = {Kv}")
        ax.legend(fontsize=7, frameon=False)
        ax.set_ylabel(f"leak / atten ({label})", color=INK)
        ax.set_title("colored by K*", color=INK, fontsize=10)

        # perceptually-ordered multi-hue ramps: with 14-16 ordered values a
        # single-hue ramp is indistinguishable; viridis/plasma keep the order
        # readable while spreading hue
        for ax, axis_key, cmap in ((axes[1], "F", plt.get_cmap("viridis")),
                                   (axes[2], "N", plt.get_cmap("plasma"))):
            vals = sorted({r[axis_key] for r in recs})
            cmap_of = {v: cmap(0.03 + 0.9 * i / max(len(vals) - 1, 1))
                       for i, v in enumerate(vals)}
            for r in recs:
                ax.scatter(r["F"] / r["N"], r[f"ratio_{tag}"], s=32,
                           color=cmap_of[r[axis_key]], edgecolor="white",
                           lw=0.6, zorder=3)
            for v in vals:
                ax.scatter([], [], color=cmap_of[v], label=f"{axis_key}={v}")
            ax.legend(fontsize=5.5, frameon=False, ncol=2)
            ax.set_title(f"colored by {axis_key}", color=INK, fontsize=10)

        ax = axes[3]
        lo, hi = y.min() * 0.6, y.max() * 1.6
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=MUTED, zorder=1)
        for r, f in zip(recs, fit):
            if np.isfinite(f):
                ax.scatter(r[f"ratio_{tag}"], f, s=28,
                           color=K_COLORS.get(r["K"], INK), edgecolor="white",
                           lw=0.6, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured ratio ({label})", color=INK)
        ax.set_ylabel("theory (no fitted constants)", color=INK)
        ax.set_title("zero-parameter theory: (F-1) E[(level)^p] / (1-Ka^2)^p,\n"
                     f"m ~ Hypergeom(N, K, K); learned (a, b) inserted. "
                     f"R^2 = {r2:.3f}", color=INK, fontsize=8)
        for ax in axes[:3]:
            ax.set_xlabel("F / N", color=INK)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="major", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK2, labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle(f"ExactPD leak/attenuation families ({label}; N < F cells "
                     "only)", color=INK2, fontsize=9, y=0.99)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"exact_solo_families_{tag}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {path} (fit R^2 {r2:.3f})")


def _ratio_theory(F, N, K, p, a, b):
    """Ensemble ladder ratio with learned (a, b): leakage levels
    m a^2 + (K - m) a b under m ~ Hypergeom(N, K, K), attenuation (1 - K a^2)^p."""
    from scipy.stats import hypergeom
    ms = np.arange(0, K + 1)
    pmf = hypergeom(N, K, K).pmf(ms)
    lev = ms * a * a + (K - ms) * a * b
    att = (1.0 - K * a * a) ** p
    if att < 1e-15:
        return np.nan
    return (F - 1) * float((pmf * np.abs(lev) ** p).sum()) / att


MAX_FAMILIES = 8


def _family_groups(recs, keys, min_size):
    """Group measured cells by the fixed pair `keys`; keep groups with at least
    min_size members, largest first, capped at MAX_FAMILIES; returns
    [(pair, cells)] sorted by pair for stable colors."""
    groups = {}
    for r in recs:
        groups.setdefault((r[keys[0]], r[keys[1]]), []).append(r)
    keep = [(k, v) for k, v in groups.items() if len(v) >= min_size]
    keep = sorted(keep, key=lambda kv: -len(kv[1]))[:MAX_FAMILIES]
    return sorted(keep)


_AB_CACHE = {}


def _ab_theory(F, N, K):
    """Predicted (a*, b*) = argmin of the TWO-BRANCH one-hot L4 ladder loss
    (training sees v ~ U[-1,1]; the common E[v^4] factor drops):
      v > 0: K code neurons at a v ->
             (1 - K a^2)^4 + (F-1) E_hyp[(m a^2 + (K-m) a b)^4]
      v < 0: N - K off-code neurons at b v ->
             (b^2 (N-K))^4 + (F-1) E_hyp[(b ((K-m) a + (N-2K+m) b))^4]
    The negative branch is what disciplines |b| (off-code firing writes pure-b^2
    garbage). Validated vs learned: a* within 2% mean, b* within 4.5% mean."""
    key = (F, N, K)
    if key not in _AB_CACHE:
        from scipy.optimize import minimize
        from scipy.stats import hypergeom
        ms = np.arange(0, K + 1)
        pmf = hypergeom(N, K, K).pmf(ms)

        def loss(prm):
            a, b = prm
            lev_p = ms * a * a + (K - ms) * a * b
            lev_n = b * ((K - ms) * a + (N - 2 * K + ms) * b)
            return ((1 - K * a * a) ** 4
                    + (F - 1) * float((pmf * lev_p ** 4).sum())
                    + (b * b * (N - K)) ** 4
                    + (F - 1) * float((pmf * lev_n ** 4).sum()))

        res = minimize(loss, x0=(0.4, -0.05), method="Nelder-Mead")
        _AB_CACHE[key] = (float(res.x[0]), float(res.x[1]))
    return _AB_CACHE[key]


def exact_solo_theory_curves(recs, fig_dir):
    """Fully theoretical leak/attenuation curves vs K, F, N (other two fixed):
    at every point the (a, b) are PREDICTED by _ab_theory (argmin of the ladder
    L4 loss -- no data input), then fed to the ensemble ratio. Families are
    taken from the measured grid; dots = measured ratios with their learned
    (a, b). Dot-curve gaps therefore include both the (a*, b*) prediction error
    (a ~2%, b ~27%) and the greedy-vs-random code effect."""
    recs = [r for r in recs if r["ratio_mse"] and r["N"] < r["F"]]
    for tag, label, p in (("mse", "MSE", 2), ("l4", "L4", 4)):
        fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
        panels = (
            (axes[0], "K", _family_groups(recs, ("F", "N"), 1)),
            (axes[1], "F", _family_groups(recs, ("N", "K"), 2)),
            (axes[2], "N", _family_groups(recs, ("F", "K"), 2)),
        )
        for ax, mode, fams in panels:
            for ci, (fam, cells_) in enumerate(fams):
                col = SEQ_BLUE(0.2 + 0.8 * ci / max(len(fams) - 1, 1))
                if mode == "K":
                    F_, N_ = fam
                    xs = np.unique(np.round(np.geomspace(2, min(N_, 100), 25))
                                   .astype(int))
                    args = [(F_, N_, int(k)) for k in xs]
                    lbl, mx = f"F={F_}, N={N_}", [c["K"] for c in cells_]
                elif mode == "F":
                    N_, K_ = fam
                    xs = np.geomspace(20, 1e4, 40)
                    args = [(float(x), N_, K_) for x in xs]
                    lbl, mx = f"N={N_}, K*={K_}", [c["F"] for c in cells_]
                else:
                    F_, K_ = fam
                    xs = np.unique(np.round(np.geomspace(2 * fam[1], 1e4, 30))
                                   .astype(int))
                    args = [(F_, int(n), K_) for n in xs]
                    lbl, mx = f"F={F_}, K*={K_}", [c["N"] for c in cells_]
                # dotted: data-free argmin theory over the full range
                ys_th = [_ratio_theory(*ag, p, *_ab_theory(round(ag[0]), ag[1],
                                                           ag[2]))
                         for ag in args]
                ax.plot(xs, ys_th, color=col, lw=0.9, ls=":", alpha=0.7,
                        zorder=1)
                # solid: learned (a, b) -- per-K kfit rows / family interpolation
                if mode == "K":
                    # drop per-K fits with K a^2 -> 1: the ratio denominator
                    # vanishes there (fits at K > K* accept zero attenuation)
                    rows = sorted((c for c in cells_[0].get("_krows", [])
                                   if c["K"] > 1
                                   and 1 - c["K"] * c["a"] ** 2 > 0.15),
                                  key=lambda c: c["K"])
                    lx = [c["K"] for c in rows]
                    ly = [_ratio_theory(fam[0], fam[1], c["K"], p, c["a"],
                                        c["b"]) for c in rows]
                elif len(cells_) >= 2:
                    key0 = "F" if mode == "F" else "N"
                    order = sorted(cells_, key=lambda c: c[key0])
                    cx = np.log([c[key0] for c in order])
                    ca = [c["a"] for c in order]
                    cb = [c["b"] for c in order]
                    lx = np.geomspace(np.exp(cx[0]), np.exp(cx[-1]), 30)
                    la = np.interp(np.log(lx), cx, ca)
                    lb = np.interp(np.log(lx), cx, cb)
                    if mode == "F":
                        ly = [_ratio_theory(float(x), fam[0], fam[1], p, av, bv)
                              for x, av, bv in zip(lx, la, lb)]
                    else:
                        ly = [_ratio_theory(fam[0], int(round(x)), fam[1], p,
                                            av, bv)
                              for x, av, bv in zip(lx, la, lb)]
                else:
                    lx, ly = [], []
                if len(lx):
                    ax.plot(lx, ly, color=col, lw=1.4, label=lbl, zorder=2)
                else:
                    ax.plot([], [], color=col, lw=1.4, label=lbl)
                ax.scatter(mx, [c[f"ratio_{tag}"] for c in cells_], s=34,
                           color=col, edgecolor="white", lw=0.8, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(mode, color=INK)
            ax.set_title(f"vs {mode}", color=INK, fontsize=10)
            ax.legend(fontsize=7, frameon=False)
            ax.grid(True, which="major", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK2, labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel(f"leak / atten ({label}), theory", color=INK)
        fig.suptitle(f"Ensemble ladder curves ({label}). Solid: learned (a, b) "
                     "(per-K fit rows / family interpolation); dotted: fully "
                     "theoretical (a*, b*) = ladder-L4 argmin; dots = measured",
                     color=INK2, fontsize=9, y=0.99)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"exact_solo_theory_curves_{tag}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {path}")


def exact_solo_theory_figure(recs, fig_dir):
    """Measured leak/attenuation ratio vs the ladder theory with learned (a, b):
    exact-overlap prediction (blue) and hypergeometric ensemble (orange)."""
    if not any(r.get("ratio_mse") for r in recs):
        return  # no nontrivial cells yet -- don't write an empty figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, tag, label in zip(axes, ("mse", "l4"), ("MSE", "L4")):
        xs = [r[f"ratio_{tag}"] for r in recs if r[f"ratio_{tag}"]]
        if not xs:
            continue
        lo, hi = min(xs) * 0.6, max(xs) * 1.6
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=MUTED, zorder=1)
        for key, color, lbl in (
                (f"pred_ladder_ratio_{tag}", "#2a78d6", "ladder (exact overlaps)"),):
            pts = [(r[f"ratio_{tag}"], r.get(key)) for r in recs
                   if r[f"ratio_{tag}"] and r.get(key)]
            if pts:
                ax.scatter(*zip(*pts), s=26, color=color, edgecolor="white",
                           linewidth=0.7, zorder=3,
                           label=lbl if tag == "mse" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured leak/atten ({label})", color=INK)
        ax.set_ylabel(f"predicted ({label})", color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Ladder theory with learned (a, b) vs measured full-model "
                 "leak/attenuation", color=INK2, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(fig_dir, "exact_solo_theory_check.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


TRAINED_COLORS = {"feature": "#2a78d6", "neuron": "#eb6834",
                  "mixed": "#1baf7a", "collapsed": "#e1e0d9"}


def trained_preference_map(rows, f_grid, n_grid, fig_dir):
    """Trained-VPD counterpart of basis_preference_map: cell color = alignment
    verdict of the learned solution, label = preference score (feature-membership
    minus neuron-membership, mass-weighted)."""
    by_cell = {(r["F"], r["N"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, matrix, title in zip(
            axes, ("w_in", "w_out"),
            ("W_in (read side), trained VPD", "W_out (write side), trained VPD")):
        _grid_axes(ax, f_grid, n_grid)
        for i, F_ in enumerate(f_grid):
            for j, N_ in enumerate(n_grid):
                r = by_cell.get((F_, N_))
                if r is None:
                    _cell_rect(ax, i, j, MISSING)
                    continue
                a = r["alignment"][matrix]
                _cell_rect(ax, i, j, TRAINED_COLORS[a["verdict"]])
                dark_bg = a["verdict"] in ("feature", "neuron")
                ax.text(i, j, f"{a['preference']:+.2f}", ha="center", va="center",
                        fontsize=7, color="white" if dark_bg else INK2, zorder=3)
        _diag_and_anchor(ax, f_grid, n_grid)
        ax.set_title(title, color=INK, fontsize=10)
    handles = [Patch(facecolor=TRAINED_COLORS[k], label=lbl) for k, lbl in
               (("feature", "feature-aligned"), ("neuron", "neuron-aligned"),
                ("mixed", "mixed / neither"), ("collapsed", "collapsed"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8)
    fig.suptitle("Learned VPD basis alignment per matrix (cell label = "
                 "feature-minus-neuron membership score; x = N=F)",
                 color=INK2, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(fig_dir, "vpd_preference_map.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def trained_alive_map(rows, f_grid, n_grid, fig_dir):
    """Alive-component counts of the learned solution, per matrix. Color =
    alive / natural basis size (F for W_in, N for W_out), diverging around 1:
    blue = fewer pieces than the basis (pruned/collapsed), gray = exact match,
    red = overcounting."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "alive_div", ["#2a78d6", "#f0efec", "#e34948"])
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    by_cell = {(r["F"], r["N"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, matrix, title in zip(
            axes, ("w_in", "w_out"),
            ("W_in alive components (vs F)", "W_out alive components (vs N)")):
        _grid_axes(ax, f_grid, n_grid)
        for i, F_ in enumerate(f_grid):
            for j, N_ in enumerate(n_grid):
                r = by_cell.get((F_, N_))
                if r is None:
                    _cell_rect(ax, i, j, MISSING)
                    continue
                alive = r["alignment"][matrix]["alive"]
                base = F_ if matrix == "w_in" else N_
                ratio = min(alive / base, 2.0)
                _cell_rect(ax, i, j, cmap(norm(ratio)))
                ax.text(i, j, str(alive), ha="center", va="center", fontsize=7,
                        color="white" if abs(ratio - 1) > 0.55 else INK,
                        zorder=3)
        _diag_and_anchor(ax, f_grid, n_grid)
        ax.set_title(title, color=INK, fontsize=10)
    fig.suptitle("Learned VPD alive components (mass > 10% of side max; "
                 "blue = fewer than basis, gray = exact, red = overcount; "
                 "x = N=F)", color=INK2, fontsize=9, y=0.99)
    fig.tight_layout()
    path = os.path.join(fig_dir, "vpd_alive_map.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def trained_preference_scatter(rows, fig_dir):
    """Scatter version of the trained-VPD map: log-log (F, N) axes, color =
    alignment verdict, point size = |preference score|."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    hi = max(max(r["F"] for r in rows), max(r["N"] for r in rows)) * 1.4
    lims = (15, hi)
    for ax, matrix, title in zip(
            axes, ("w_in", "w_out"),
            ("W_in (read side), trained VPD", "W_out (write side), trained VPD")):
        ax.plot(lims, lims, ls="--", lw=0.8, color=MUTED, zorder=1)
        ax.text(lims[1] * 0.55, lims[1] * 0.72, "N = F", color=MUTED,
                fontsize=7, rotation=38)
        for r in rows:
            a = r["alignment"][matrix]
            size = 30 + 220 * min(abs(a["preference"]), 1.0)
            ax.scatter(r["F"], r["N"], s=size,
                       color=TRAINED_COLORS[a["verdict"]],
                       edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel("n_features F", color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("n_neurons N", color=INK)
    handles = [Patch(facecolor=TRAINED_COLORS[k], label=lbl) for k, lbl in
               (("feature", "feature-aligned"), ("neuron", "neuron-aligned"),
                ("mixed", "mixed / neither"), ("collapsed", "collapsed"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8)
    fig.suptitle("Learned VPD basis alignment (point size = |feature-minus-"
                 "neuron membership|)", color=INK2, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(fig_dir, "vpd_preference_scatter.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def basis_preference_scatter(rows, fig_dir, tie_ratio):
    """Scatter version of the preference map: true log-log (F, N) axes, point
    color = preferred basis, point size = loss-ratio margin."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    hi = max(max(r["F"] for r in rows), max(r["N"] for r in rows)) * 1.4
    lims = (15, hi)
    for ax, matrix, title in zip(
            axes, ("w_in", "w_out"),
            ("W_in (read side)", "W_out (write side)")):
        ax.plot(lims, lims, ls="--", lw=0.8, color=MUTED, zorder=1)
        ax.text(lims[1] * 0.55, lims[1] * 0.72, "N = F", color=MUTED,
                fontsize=7, rotation=38)
        for r in rows:
            pref, margin = r[f"{matrix}_pref"], r[f"{matrix}_margin"]
            size = 30 + 55 * min(np.log2(max(margin, 1.0)), 4)
            ax.scatter(r["F"], r["N"], s=size, color=PREF_COLORS[pref],
                       edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel("n_features F", color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("n_neurons N", color=INK)
    handles = [Patch(facecolor=PREF_COLORS[k], label=lbl) for k, lbl in
               (("feature", "feature basis wins"), ("neuron", "neuron basis wins"),
                ("tie", f"tie (< {tie_ratio:.2f}x)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8)
    fig.suptitle("Oracle-gated VPD-loss basis preference (point size = "
                 "loss-ratio margin, capped at 16x)", color=INK2, fontsize=9,
                 y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(fig_dir, "basis_preference_scatter.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def _load_terms(oracle_dir, F_, N_, K):
    out = {}
    for rb, wb in itertools.product(("feature", "neuron"), repeat=2):
        p = os.path.join(oracle_dir, f"F{F_}_N{N_}_K{K}_{rb}-{wb}.json")
        with open(p) as f:
            rec = json.load(f)
        out[(rb, wb)] = rec["losses"]
    return out


def loss_2x2_examples(rows, oracle_dir, fig_dir):
    """Full 2x2 weighted-term breakdown for the anchor + F/N-ratio extremes."""
    by_cell = {(r["F"], r["N"]): r for r in rows}
    picks = []
    if ANCHOR in by_cell:
        picks.append(ANCHOR)
    ranked = sorted(by_cell, key=lambda c: c[0] / c[1])
    for c in (ranked[0], ranked[-1], max(by_cell, key=lambda c: c[0] * c[1])):
        if c not in picks:
            picks.append(c)
    combos = list(itertools.product(("feature", "neuron"), repeat=2))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 4.2),
                             squeeze=False)
    for ax, cell in zip(axes[0], picks):
        r = by_cell[cell]
        terms = _load_terms(oracle_dir, *cell, r["K_star"])
        xs = np.arange(len(combos))
        bottom = np.zeros(len(combos))
        for term in ("stoch_recon", "pgd_recon", "importance"):
            vals = np.array([terms[c]["terms"][term]["weighted"] for c in combos])
            ax.bar(xs, vals, 0.62, bottom=bottom, color=TERM_COLORS[term],
                   edgecolor="white", linewidth=1.0,
                   label=term if cell == picks[0] else None)
            bottom += vals
        for x, c in zip(xs, combos):
            ax.text(x, bottom[x] * 1.03, f"{terms[c]['total']:.1e}",
                    ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"({rb[0]},{wb[0]})" for rb, wb in combos],
                           fontsize=8, color=INK2)
        ax.set_title(f"F={cell[0]}, N={cell[1]}, K*={r['K_star']}",
                     fontsize=9, color=INK)
        ax.tick_params(length=0, labelsize=7)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(0, float(bottom.max()) * 1.18)
    axes[0][0].set_ylabel("weighted VPD objective", color=INK)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Oracle 2x2 loss composition: (read, write) basis combos, "
                 "(f = feature, n = neuron)", color=INK2, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(fig_dir, "loss_2x2_examples.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")
