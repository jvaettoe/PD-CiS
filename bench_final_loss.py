"""Benchmark trained-model FINAL LOSS (and loss curves) across d_emb, for a given
loss exponent n:  EmbedMLP d_mlp=50  vs  EmbedMLP d_mlp=100  vs  LinearMLP d_mlp=50.

Reads cached checkpoints directly (each stores the full `losses` curve).
  ./.venv/bin/python bench_final_loss.py --n 4
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CKPT_DIR = "checkpoints"
OUT_DIR = "figures/model_benchmark"
PS = [0.001, 0.01, 0.1, 0.5, 0.8]
DEMBS = [20, 50, 100, 200, 1000]

SERIES = {  # label -> (model_class, d_mlp, colour, marker)
    "EmbedMLP d_mlp=50":  ("EmbedMLP", 50,  "#1f77b4", "o"),
    "EmbedMLP d_mlp=100": ("EmbedMLP", 100, "#ff7f0e", "s"),
}
BASELINES = {  # label -> (model_class, d_mlp, colour); LinearMLP has no d_emb -> horizontal
    "LinearMLP d_mlp=50 (baseline)":  ("LinearMLP", 50,  "#2ca02c"),
    "LinearMLP d_mlp=100 (baseline)": ("LinearMLP", 100, "#d62728"),
}


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["losses"], ck["model_config"], ck["dataset_config"]


def scan(n):
    """-> {(model_class, d_mlp, d_emb_or_None, p): (losses, final)}"""
    out = {}
    for path in glob.glob(os.path.join(CKPT_DIR, f"ncc_*_n{_ntok(n)}_s42.pt")):
        losses, mc, dc = load(path)
        cls = "EmbedMLP" if "EmbedMLP" in os.path.basename(path) else "LinearMLP"
        key = (cls, mc.get("d_mlp"), mc.get("d_emb"), round(dc["p"], 6))
        out[key] = (losses, losses[-1])
    return out


def _ntok(n):
    return str(int(n)) if float(n).is_integer() else str(n)


def plot_final_vs_demb(data, n):
    fig, axes = plt.subplots(1, len(PS), figsize=(4 * len(PS), 4.2), sharey=True)
    for ax, p in zip(axes, PS):
        # baselines (LinearMLP, no d_emb) -> horizontal references
        for blabel, (bcls, bdmlp, bcolour) in BASELINES.items():
            bl = data.get((bcls, bdmlp, None, p))
            if bl:
                ax.axhline(bl[1], ls="--", color=bcolour, lw=1.6, label=blabel)
        for label, (cls, dmlp, colour, marker) in SERIES.items():
            xs, ys = [], []
            for d in DEMBS:
                v = data.get((cls, dmlp, d, p))
                if v:
                    xs.append(d); ys.append(v[1])
            if xs:
                ax.plot(xs, ys, marker=marker, color=colour, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"p = {p}")
        ax.set_xlabel("d_emb")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("final training loss")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Final training loss vs embedding dim  (n={_ntok(n)}, n_features=100)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"final_loss_vs_demb_n{_ntok(n)}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_curves(data, n, demb=200):
    """Loss curves at one representative d_emb, per p; LinearMLP overlaid."""
    fig, axes = plt.subplots(1, len(PS), figsize=(4 * len(PS), 4.2), sharey=True)
    for ax, p in zip(axes, PS):
        for label, (cls, dmlp, colour, marker) in SERIES.items():
            v = data.get((cls, dmlp, demb, p))
            if v:
                ax.plot(v[0], color=colour, lw=1.3, label=f"{label}")
        for blabel, (bcls, bdmlp, bcolour) in BASELINES.items():
            bl = data.get((bcls, bdmlp, None, p))
            if bl:
                ax.plot(bl[0], color=bcolour, lw=1.3, ls="--",
                        label=blabel.replace(" (baseline)", ""))
        ax.set_yscale("log")
        ax.set_title(f"p = {p}")
        ax.set_xlabel("step")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("training loss")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Loss curves at d_emb={demb}  (n={_ntok(n)}, n_features=100)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f"loss_curves_demb{demb}_n{_ntok(n)}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def n_trainable(cls, dmlp, demb):
    """Trainable params = w_in + w_out only (EmbedMLP's w_e is a fixed buffer).
    EmbedMLP: w_in(d_emb x d_mlp) + w_out(d_mlp x d_emb) = 2*d_emb*d_mlp.
    LinearMLP: w_in(100 x d_mlp) + w_out(d_mlp x 100) = 2*100*d_mlp."""
    return 2 * (demb if cls == "EmbedMLP" else 100) * dmlp


def plot_loss_vs_params(data, n):
    """Final loss vs TRAINABLE parameter count -- collapses all four models onto a
    single capacity axis. EmbedMLP series sweep d_emb (params grow with d_emb);
    LinearMLP baselines are single points. At a given x, the lowest point wins per
    parameter -> isolates architecture from raw capacity."""
    fig, axes = plt.subplots(1, len(PS), figsize=(4 * len(PS), 4.2), sharey=True)
    for ax, p in zip(axes, PS):
        for label, (cls, dmlp, colour, marker) in SERIES.items():
            pts = sorted((n_trainable(cls, dmlp, d), data[(cls, dmlp, d, p)][1], d)
                         for d in DEMBS if (cls, dmlp, d, p) in data)
            if pts:
                xs, ys, ds = zip(*pts)
                ax.plot(xs, ys, marker=marker, color=colour, label=label)
                for x, y, d in pts:  # annotate each point with its d_emb
                    ax.annotate(f"{d}", (x, y), fontsize=6, color=colour,
                                xytext=(0, 5), textcoords="offset points", ha="center")
        for blabel, (bcls, bdmlp, bcolour) in BASELINES.items():
            if (bcls, bdmlp, None, p) in data:
                x = n_trainable(bcls, bdmlp, None)
                y = data[(bcls, bdmlp, None, p)][1]
                ax.plot([x], [y], marker="*", markersize=15, color=bcolour,
                        ls="none", label=blabel.replace(" (baseline)", ""))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"p = {p}")
        ax.set_xlabel("trainable params (w_in+w_out)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("final training loss")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Final loss vs trainable parameters  (n={_ntok(n)}, n_features=100; "
                 f"EmbedMLP points labelled by d_emb)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, f"loss_vs_params_n{_ntok(n)}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=4.0)
    ap.add_argument("--curve-demb", type=int, default=200)
    a = ap.parse_args()
    data = scan(a.n)
    # coverage report
    print(f"n={_ntok(a.n)}: {len(data)} checkpoints")
    coverage = [(lbl, cls, dmlp) for lbl, (cls, dmlp, *_1) in SERIES.items()]
    coverage += [(lbl, bcls, bdmlp) for lbl, (bcls, bdmlp, _c) in BASELINES.items()]
    for label, cls, dmlp in coverage:
        have = [k for k in data if k[0] == cls and k[1] == dmlp]
        print(f"  {label:30s}: {len(have)} runs")
    print("final loss table (rows=d_emb, cols=p):")
    for label, (cls, dmlp, *_2) in SERIES.items():
        print(f"  [{label}]")
        print("    d_emb  " + "  ".join(f"{p:>9}" for p in PS))
        for d in DEMBS:
            row = "  ".join(
                (f"{data[(cls,dmlp,d,p)][1]:9.2e}" if (cls,dmlp,d,p) in data else f"{'--':>9}")
                for p in PS)
            print(f"    {d:5d}  {row}")
    for blabel, (bcls, bdmlp, _c) in BASELINES.items():
        print(f"  [{blabel}]")
        print("    " + "  ".join(f"{p:>9}" for p in PS))
        print("    " + "  ".join(
            (f"{data[(bcls,bdmlp,None,p)][1]:9.2e}" if (bcls,bdmlp,None,p) in data else f"{'--':>9}")
            for p in PS))
    print("figures:")
    print(" ", plot_final_vs_demb(data, a.n))
    print(" ", plot_curves(data, a.n, a.curve_demb))
    print(" ", plot_loss_vs_params(data, a.n))


if __name__ == "__main__":
    main()
