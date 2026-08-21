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


def basis_preference_scatter(rows, fig_dir, tie_ratio):
    """Scatter version of the preference map: true log-log (F, N) axes, point
    color = preferred basis, point size = loss-ratio margin."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    lims = (15, 1300)
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
