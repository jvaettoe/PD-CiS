"""Layer-2 bypass ablation across the p sweep (L4 chains).

Per p: held-out L4 of (original, identity bypass = drop layer-2 ReLU,
fitted-A bypass = layer 2 replaced by least-squares linear map), the bypass/
original L4 ratios, and the on-distribution layer-2 clipping fraction.
Figure: ratios + clipping vs p — is there a transition where the second
nonlinearity becomes functionally necessary?

  python bypass_sweep.py
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
EVAL_SEED, FIT_SEED = 777, 12345


def held_out_l4(fwd, p, batches=40, batch=8192):
    ds = SparseDataset(p, seed=EVAL_SEED)
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            tot += float(((fwd(x) - y).abs() ** 4).mean())
    return tot / batches


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints",
                                              "chain_p*_l4_s42.pt"))):
        m = re.match(r"chain_p([\d.]+)_l4_s\d+\.pt", os.path.basename(path))
        if not m:
            continue
        p = float(m.group(1))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = ChainMLP(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        W1, W2, W3 = model.w1.detach(), model.w2.detach(), model.w3.detach()
        xs, _ = SparseDataset(p, seed=FIT_SEED).generate_batch(131072)
        with torch.no_grad():
            h1 = torch.relu(xs @ W1)
            live = h1.sum(1) > 0
            pre2 = h1[live] @ W2
            clip = float(((pre2 < 0) & (h1[live] @ torch.abs(W2) > 1e-6))
                         .float().mean())
            A = torch.linalg.lstsq(h1[live], torch.relu(pre2)).solution
        l4_orig = held_out_l4(
            lambda x: torch.relu(torch.relu(x @ W1) @ W2) @ W3, p)
        l4_id = held_out_l4(lambda x: (torch.relu(x @ W1) @ W2) @ W3, p)
        l4_A = held_out_l4(lambda x: (torch.relu(x @ W1) @ A) @ W3, p)
        rows.append({"p": p, "clip": clip, "l4_orig": l4_orig,
                     "r_id": l4_id / l4_orig, "r_A": l4_A / l4_orig})
        print(f"p={p:g}: clip {clip:.3f}, L4 orig {l4_orig:.2e}, "
              f"identity x{rows[-1]['r_id']:.2f}, fitted-A x{rows[-1]['r_A']:.2f}")
    rows.sort(key=lambda r: r["p"])

    md = ["### chain layer-2 bypass ablation vs p (L4 models, seed 42)", "",
          "| p | clip frac | L4 original | identity bypass ratio | "
          "fitted-A bypass ratio |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['p']:g} | {r['clip']:.3f} | {r['l4_orig']:.2e} | "
                  f"{r['r_id']:.2f} | {r['r_A']:.2f} |")
    with open(os.path.join(ROOT, "results", "bypass_sweep.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    ps = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(ps, [r["r_id"] for r in rows], "o-", color="tab:red", lw=2,
            label="identity bypass (drop ReLU2) / original")
    ax.plot(ps, [r["r_A"] for r in rows], "s-", color="tab:orange", lw=2,
            label="fitted linear map / original")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input sparsity p (log)")
    ax.set_ylabel("held-out L4 ratio vs original (log)")
    ax2 = ax.twinx()
    ax2.plot(ps, [r["clip"] for r in rows], "^--", color="steelblue", lw=1.5,
             label="layer-2 clipping fraction")
    ax2.set_ylabel("clipping fraction", color="steelblue")
    ax2.set_ylim(0, 0.35)
    ax.set_title("chain: cost of removing the second nonlinearity vs p",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")
    out = os.path.join(ROOT, "figures", "bypass_sweep.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
