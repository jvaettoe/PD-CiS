"""Feature-computation census for the ramp models (chain and single).

Per arch: per-feature active-conditional MSE (ramp baseline E[f^2|x>0] =
theta^3/3 + theta^2(1-theta) = 1/6, computed = < 1/60) and layer-wise codeword
lengths on one-hot probes: K1 (h1 preacts) and, for the chain, K2 (h2
preacts), both >0 and tau conventions.

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

from data import N_FEATURES, THETA, SparseDataset
from model import load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
BASELINE = THETA ** 3 / 3 + THETA ** 2 * (1 - THETA)   # = 1/6
COMPUTED_AT = BASELINE / 10                            # = 1/60
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
    """Per-feature K at each hidden layer from one-hot probes (unit value)."""
    with torch.no_grad():
        pre1 = model.w1.detach()                      # (F, d1): row j = e_j preacts
        out = {"k1_pos": (pre1 > 0).sum(1).numpy(),
               "k1_tau": (pre1 > TAU).sum(1).numpy()}
        if hasattr(model, "w3"):                      # chain: second hidden layer
            pre2 = torch.relu(pre1) @ model.w2.detach()   # (F, d2)
            out["k2_pos"] = (pre2 > 0).sum(1).numpy()
            out["k2_tau"] = (pre2 > TAU).sum(1).numpy()
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows, per_run = [], {}
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "ramp_*.pt"))):
        m = re.match(r"ramp_(chain|single)_p([\d.]+)_l4_s\d+\.pt",
                     os.path.basename(path))
        if not m:
            continue
        arch, p = m.group(1), float(m.group(2))
        model, _ = load_checkpoint(path)
        mse = perfeature_mse(model, p)
        ks = layer_ks(model)
        per_run[arch] = {"mse": mse, **ks}
        row = {"arch": arch, "p": p,
               "n_computed": int((mse < COMPUTED_AT).sum()),
               "mse_median": float(np.median(mse)),
               "k1": f"{ks['k1_pos'].mean():.1f}±{ks['k1_pos'].std():.1f}",
               "k1_tau": f"{ks['k1_tau'].mean():.1f}"}
        if "k2_pos" in ks:
            row["k2"] = f"{ks['k2_pos'].mean():.1f}±{ks['k2_pos'].std():.1f}"
            row["k2_tau"] = f"{ks['k2_tau'].mean():.1f}"
        else:
            row["k2"] = row["k2_tau"] = "—"
        rows.append(row)
        print(f"{arch} p={p:g}: computed {row['n_computed']}/100")
    rows.sort(key=lambda r: r["arch"])

    md = ["### ramp census — target ramp(x) = ReLU(x) − ReLU(x−0.5), L4, "
          f"p = 0.01, seed 42; computed = active-cond MSE < {COMPUTED_AT:.3g} "
          f"(baseline {BASELINE:.3g})", "",
          "| arch | p | computed /100 | MSE median | K1(>0) | K1(tau) | K2(>0) | "
          "K2(tau) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arch']} | {r['p']:g} | {r['n_computed']} | "
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
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("feature (sorted)")
    np.atleast_1d(axes)[0].set_ylabel("active-conditional MSE")
    fig.suptitle("ramp: per-feature MSE (grey = baseline 1/6, dashed = computed "
                 "threshold)", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_mse_sorted.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.8),
                             constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), runs):
        d = per_run[key]
        if "k2_pos" in d:
            ax.scatter(d["k1_pos"], d["k2_pos"], s=14, alpha=0.6,
                       color="tab:blue")
            ax.set_xlabel("K1(>0)")
            ax.set_ylabel("K2(>0)")
        else:
            ax.hist(d["k1_pos"], bins=np.arange(-0.5, 51.5), color="tab:blue",
                    alpha=0.8)
            ax.set_xlabel("K1(>0)")
            ax.set_ylabel("features")
        ax.set_title(key, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("ramp: codeword lengths per feature", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_k.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
