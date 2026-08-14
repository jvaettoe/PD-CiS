"""Feature-computation census for the chain models (two hidden ReLU layers).

Per (p, loss): per-feature active-conditional MSE (baseline 1/3, computed =
< 1/30) and layer-wise codeword lengths on one-hot probes: K1 (h1 preacts) and
K2 (h2 preacts), both >0 and tau conventions, plus a K1-vs-K2 scatter.

  python census.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import N_FEATURES, SparseDataset
from model import ChainMLP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
BASELINE = 1 / 3
COMPUTED_AT = BASELINE / 10
EVAL_SEED = 777


def perfeature_mse(model, p: float, batches: int = 80, batch: int = 8192):
    ds = SparseDataset(p, seed=EVAL_SEED)
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def layer_ks(model):
    """Per-feature K at both layers from one-hot probes (unit value)."""
    with torch.no_grad():
        pre1 = model.w1.detach()                      # (F, d1): row j = e_j preacts
        h1 = torch.relu(pre1)
        pre2 = h1 @ model.w2.detach()                 # (F, d2)
    return {"k1_pos": (pre1 > 0).sum(1).numpy(),
            "k1_tau": (pre1 > TAU).sum(1).numpy(),
            "k2_pos": (pre2 > 0).sum(1).numpy(),
            "k2_tau": (pre2 > TAU).sum(1).numpy()}


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows, per_run = [], {}
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "chain_*.pt"))):
        m = re.match(r"chain_p([\d.]+)_(mse|l4)_s\d+\.pt", os.path.basename(path))
        if not m:
            continue
        p, kind = float(m.group(1)), m.group(2)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = ChainMLP(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        mse = perfeature_mse(model, p)
        ks = layer_ks(model)
        per_run[(p, kind)] = {"mse": mse, **ks}
        rows.append({
            "p": p, "loss": kind, "n_computed": int((mse < COMPUTED_AT).sum()),
            "mse_median": float(np.median(mse)),
            "k1": f"{ks['k1_pos'].mean():.1f}±{ks['k1_pos'].std():.1f}",
            "k1_tau": f"{ks['k1_tau'].mean():.1f}",
            "k2": f"{ks['k2_pos'].mean():.1f}±{ks['k2_pos'].std():.1f}",
            "k2_tau": f"{ks['k2_tau'].mean():.1f}"})
        print(f"p={p:g} {kind}: computed {rows[-1]['n_computed']}/100")
    rows.sort(key=lambda r: (r["loss"], r["p"]))

    md = ["### chain census — 100 -> 50 (ReLU) -> 50 (ReLU) -> 100, seed 42; "
          f"computed = active-cond MSE < {COMPUTED_AT:.3g}", "",
          "| loss | p | computed /100 | MSE median | K1(>0) | K1(tau) | K2(>0) | "
          "K2(tau) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['loss']} | {r['p']:g} | {r['n_computed']} | "
                  f"{r['mse_median']:.2e} | {r['k1']} | {r['k1_tau']} | "
                  f"{r['k2']} | {r['k2_tau']} |")
    with open(os.path.join(RES, "census.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    runs = sorted(per_run)
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.8),
                             constrained_layout=True, sharey=True)
    for ax, key in zip(np.atleast_1d(axes), runs):
        d = per_run[key]
        ax.plot(sorted(d["mse"], reverse=True), color="k", lw=2)
        ax.axhline(BASELINE, color="grey", ls=":", lw=0.8)
        ax.axhline(COMPUTED_AT, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"p={key[0]:g}, {key[1]}", fontsize=10)
        ax.set_xlabel("feature (sorted)")
    np.atleast_1d(axes)[0].set_ylabel("active-conditional MSE")
    fig.suptitle("chain: per-feature MSE (grey = baseline 1/3, dashed = computed "
                 "threshold)", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_mse_sorted.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.8),
                             constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), runs):
        d = per_run[key]
        ax.scatter(d["k1_pos"], d["k2_pos"], s=14, alpha=0.6, color="tab:blue")
        ax.set_title(f"p={key[0]:g}, {key[1]}", fontsize=10)
        ax.set_xlabel("K1(>0)")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("K2(>0)")
    fig.suptitle("chain: layer-1 vs layer-2 codeword lengths per feature",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "census_k1_vs_k2.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
