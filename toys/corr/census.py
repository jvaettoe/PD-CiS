"""Feature-computation census for the correlated-pair models across the rho sweep.

Per (rho, loss): per-feature active-conditional MSE (baseline 1/3, computed =
< 1/30), K(>0)/K(tau), and within-pair vs between-pair codeword overlap (mean
shared neurons in the tau-codes) — the weight-space form of the merged-vs-separate
mechanisms question.

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

from data import CorrDataset, N_FEATURES, N_PAIRS
from model import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
BASELINE = 1 / 3
COMPUTED_AT = BASELINE / 10
EVAL_SEED = 777


def perfeature_mse(model, rho: float, batches: int = 50, batch: int = 8192):
    ds = CorrDataset(rho, seed=EVAL_SEED)
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def overlaps(W: torch.Tensor):
    """(within-pair mean shared neurons, between-pair mean) for tau-codes."""
    codes = (W > TAU).float()
    ov = codes @ codes.T                              # (F, F) shared-neuron counts
    within = float(torch.stack(
        [ov[2 * i, 2 * i + 1] for i in range(N_PAIRS)]).mean())
    mask = torch.ones_like(ov, dtype=torch.bool)
    mask.fill_diagonal_(False)
    for i in range(N_PAIRS):
        mask[2 * i, 2 * i + 1] = mask[2 * i + 1, 2 * i] = False
    return within, float(ov[mask].mean())


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "corr_*.pt"))):
        m = re.match(r"corr_rho([\d.]+)_(mse|l[\d.]+)(_st\d+)?_s\d+\.pt", os.path.basename(path))
        if not m:
            continue
        rho, kind = float(m.group(1)), m.group(2)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = MLP(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        mse = perfeature_mse(model, rho)
        W = model.w_in.detach()
        within, between = overlaps(W)
        rows.append({
            "rho": rho, "loss": kind,
            "n_computed": int((mse < COMPUTED_AT).sum()),
            "mse_median": float(np.median(mse)),
            "k_pos_m": float((W > 0).sum(1).float().mean()),
            "k_pos_s": float((W > 0).sum(1).float().std()),
            "k_tau_m": float((W > TAU).sum(1).float().mean()),
            "within": within, "between": between})
        print(f"rho={rho:g} {kind}: computed {rows[-1]['n_computed']}/100, "
              f"K(>0) {rows[-1]['k_pos_m']:.1f}, overlap w/b "
              f"{within:.2f}/{between:.2f}")
    rows.sort(key=lambda r: (r["loss"], r["rho"]))

    md = ["### corr census — 50 correlated pairs (p=0.01), MLP 100-50-100, seed "
          f"42; computed = active-cond MSE < {COMPUTED_AT:.3g}", "",
          "| loss | rho | computed /100 | MSE median | K(>0) | K(tau) mean | "
          "overlap within-pair | overlap between |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['loss']} | {r['rho']:g} | {r['n_computed']} | "
                  f"{r['mse_median']:.2e} | {r['k_pos_m']:.1f}±{r['k_pos_s']:.1f}"
                  f" | {r['k_tau_m']:.1f} | {r['within']:.2f} | "
                  f"{r['between']:.2f} |")
    with open(os.path.join(RES, "census.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for kind, col in (("mse", "tab:blue"), ("l4", "tab:orange")):
        rs = [r for r in rows if r["loss"] == kind]
        xs = [r["rho"] for r in rs]
        axes[0].plot(xs, [r["n_computed"] for r in rs], "o-", color=col,
                     label=kind)
        axes[1].plot(xs, [r["k_pos_m"] for r in rs], "o-", color=col, label=kind)
        axes[2].plot(xs, [r["within"] for r in rs], "o-", color=col,
                     label=f"{kind} within")
        axes[2].plot(xs, [r["between"] for r in rs], "s--", color=col,
                     label=f"{kind} between")
    axes[0].set_title("features computed /100", fontsize=10)
    axes[1].set_title("K(>0) mean", fontsize=10)
    axes[2].set_title(f"codeword overlap (tau={TAU})", fontsize=10)
    for ax in axes:
        ax.set_xlabel("rho")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("corr: census vs within-pair occurrence correlation", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_vs_rho.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
