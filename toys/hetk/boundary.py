"""Abandonment phase boundary: rare-population census vs p_rare and vs the
population split n_rare, per loss kind.

Reads every hetk checkpoint (default + _prare/_nrare sweep runs), computes the
rare-population metrics with the matching dataset config, and plots the boundary
curves.

  python boundary.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import HetKDataset, N_FEATURES, N_RARE, P_RARE
from model import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
BASELINE = 1 / 3
COMPUTED_AT = BASELINE / 10
EVAL_SEED = 777


def perfeature_mse(model, p_rare, n_rare, batches=80, batch=8192):
    ds = HetKDataset(seed=EVAL_SEED, p_rare=p_rare, n_rare=n_rare)
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "hetk_*.pt"))):
        name = os.path.basename(path)
        m = re.match(r"hetk(?:_prare([\d.]+))?(?:_nrare(\d+))?_(mse|wmse|l4)"
                     r"_s\d+\.pt", name)
        if not m:
            continue
        p_rare = float(m.group(1)) if m.group(1) else P_RARE
        n_rare = int(m.group(2)) if m.group(2) else N_RARE
        kind = m.group(3)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = MLP(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        mse = perfeature_mse(model, p_rare, n_rare)
        idx = np.arange(n_rare)
        W = model.w_in.detach()
        rows.append({
            "p_rare": p_rare, "n_rare": n_rare, "loss": kind,
            "computed_frac": float((mse[idx] < COMPUTED_AT).mean()),
            "mse_med": float(np.median(mse[idx])),
            "k_tau": float((W[idx] > TAU).sum(1).float().mean()),
            "common_computed_frac": float(
                (mse[n_rare:] < COMPUTED_AT).mean())})
        r = rows[-1]
        print(f"p_rare={p_rare:g} n_rare={n_rare} {kind}: rare computed "
              f"{r['computed_frac']:.0%}, MSE med {r['mse_med']:.2e}, "
              f"K(tau) {r['k_tau']:.1f}")
    rows.sort(key=lambda r: (r["loss"], r["n_rare"], r["p_rare"]))

    md = ["### hetK abandonment boundary — rare-population census vs p_rare and "
          f"n_rare (common p=0.05; computed = active-cond MSE < "
          f"{COMPUTED_AT:.3g})", "",
          "| loss | p_rare | n_rare | rare computed | rare MSE med | rare "
          "K(tau) | common computed |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['loss']} | {r['p_rare']:g} | {r['n_rare']} | "
                  f"{r['computed_frac']:.0%} | {r['mse_med']:.2e} | "
                  f"{r['k_tau']:.1f} | {r['common_computed_frac']:.0%} |")
    with open(os.path.join(RES, "boundary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    styles = {"mse": ("tab:blue", "-"), "wmse": ("tab:green", "-."),
              "l4": ("tab:orange", "--")}

    # p_rare sweep (n_rare = 50)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for kind, (col, ls) in styles.items():
        rs = [r for r in rows if r["loss"] == kind and r["n_rare"] == N_RARE]
        if not rs:
            continue
        xs = [r["p_rare"] for r in rs]
        axes[0].plot(xs, [r["computed_frac"] for r in rs], "o" + ls, color=col,
                     lw=2, label=kind)
        axes[1].plot(xs, [r["mse_med"] for r in rs], "o" + ls, color=col, lw=2,
                     label=kind)
        axes[2].plot(xs, [r["k_tau"] for r in rs], "o" + ls, color=col, lw=2,
                     label=kind)
    axes[0].set_ylabel("rare fraction computed")
    axes[1].axhline(BASELINE, color="grey", ls=":", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rare MSE median (grey = ignore baseline)")
    axes[2].set_ylabel("rare K(tau) mean")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("p_rare (log)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("hetK abandonment boundary vs p_rare (n_rare=50, common "
                 "p=0.05)", fontsize=11)
    fig.savefig(os.path.join(FIG, "boundary_prare.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # split sweep (p_rare = 0.001)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for kind, (col, ls) in styles.items():
        rs = [r for r in rows if r["loss"] == kind and r["p_rare"] == P_RARE]
        if not rs:
            continue
        xs = [r["n_rare"] for r in rs]
        axes[0].plot(xs, [r["computed_frac"] for r in rs], "o" + ls, color=col,
                     lw=2, label=kind)
        axes[1].plot(xs, [r["mse_med"] for r in rs], "o" + ls, color=col, lw=2,
                     label=kind)
        axes[2].plot(xs, [r["k_tau"] for r in rs], "o" + ls, color=col, lw=2,
                     label=kind)
    axes[0].set_ylabel("rare fraction computed")
    axes[1].axhline(BASELINE, color="grey", ls=":", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rare MSE median")
    axes[2].set_ylabel("rare K(tau) mean")
    for ax in axes:
        ax.set_xlabel("n_rare (population split)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("hetK abandonment vs population split (p_rare=0.001)",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "boundary_split.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
