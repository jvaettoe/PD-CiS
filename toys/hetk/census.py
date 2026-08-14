"""Feature-computation census for the hetK model, split by sparsity population.

Metrics per loss kind (mse, l4): per-feature active-conditional MSE (trivial
always-0 baseline = 1/3; computed = MSE < 1/30), K(>0) and K(tau=0.05) codeword
lengths from one-hot preactivations, all split rare vs common.

  python census.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import HetKDataset, N_FEATURES, N_RARE
from model import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
BASELINE = 1 / 3
COMPUTED_AT = BASELINE / 10
EVAL_SEED = 777


def perfeature_mse(model, batches: int = 80, batch: int = 8192) -> np.ndarray:
    ds = HetKDataset(seed=EVAL_SEED)
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def load(kind: str):
    ck = torch.load(os.path.join(ROOT, "checkpoints", f"hetk_{kind}_s42.pt"),
                    map_location="cpu", weights_only=False)
    m = MLP(**ck["model_config"])
    m.load_state_dict(ck["state_dict"])
    return m, ck


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    pops = {"rare": np.arange(N_RARE), "common": np.arange(N_RARE, N_FEATURES)}
    kinds = [k for k in ("mse", "wmse", "l4")
             if os.path.exists(os.path.join(ROOT, "checkpoints",
                                            f"hetk_{k}_s42.pt"))]
    rows, per_kind = [], {}
    for kind in kinds:
        model, ck = load(kind)
        mse = perfeature_mse(model)
        W = model.w_in.detach()                    # (F, N): row j = preacts on e_j
        k_pos = (W > 0).sum(1).numpy()
        k_tau = (W > TAU).sum(1).numpy()
        per_kind[kind] = {"mse": mse, "k_pos": k_pos, "k_tau": k_tau,
                          "losses": ck["losses"], "W": W}
        for pop, idx in pops.items():
            rows.append({
                "loss": kind, "population": pop,
                "n_computed": int((mse[idx] < COMPUTED_AT).sum()),
                "mse_median": float(np.median(mse[idx])),
                "k_pos": f"{k_pos[idx].mean():.1f}±{k_pos[idx].std():.1f}",
                "k_tau": f"{k_tau[idx].mean():.1f}±{k_tau[idx].std():.1f}"})

    md = ["### hetK census — two-population sparsity (rare p=0.001 x50, common "
          f"p=0.05 x50), MLP 100-50-100, seed 42; computed = active-cond MSE < "
          f"{COMPUTED_AT:.3g} (baseline/10)", "",
          "| loss | population | computed /50 | MSE median | K(>0) | K(tau) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['loss']} | {r['population']} | {r['n_computed']} | "
                  f"{r['mse_median']:.2e} | {r['k_pos']} | {r['k_tau']} |")
    with open(os.path.join(RES, "census.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # sorted per-feature MSE by population
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4),
                             constrained_layout=True, sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        d = per_kind[kind]
        for pop, idx, col in (("rare", pops["rare"], "tab:blue"),
                              ("common", pops["common"], "tab:red")):
            ax.plot(sorted(d["mse"][idx], reverse=True), color=col, lw=2,
                    label=pop)
        ax.axhline(BASELINE, color="grey", ls=":", lw=0.8)
        ax.axhline(COMPUTED_AT, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"loss = {kind}", fontsize=10)
        ax.set_xlabel("feature (sorted within population)")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("active-conditional MSE")
    fig.suptitle("hetK: per-feature MSE by population (grey = baseline 1/3, "
                 "dashed = computed threshold)", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_mse_sorted.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # K histograms by population
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4),
                             constrained_layout=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        d = per_kind[kind]
        bins = np.arange(-0.5, max(d["k_pos"].max(), 1) + 1.5)
        for pop, idx, col in (("rare", pops["rare"], "tab:blue"),
                              ("common", pops["common"], "tab:red")):
            ax.hist(d["k_pos"][idx], bins=bins, alpha=0.6, color=col, label=pop)
        ax.set_title(f"loss = {kind}", fontsize=10)
        ax.set_xlabel("K(>0) per feature")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("features")
    fig.suptitle("hetK: codeword-length distributions by population", fontsize=11)
    fig.savefig(os.path.join(FIG, "census_k_hist.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # W_in heatmaps + training loss
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.5),
                             constrained_layout=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        W = per_kind[kind]["W"].numpy()
        v = np.abs(W).max()
        ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.axhline(N_RARE - 0.5, color="k", lw=0.8)
        ax.set_title(f"W_in, loss = {kind} (line: rare|common)", fontsize=10)
        ax.set_xlabel("neuron")
    np.atleast_1d(axes)[0].set_ylabel("feature")
    fig.savefig(os.path.join(FIG, "census_w_in.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for kind, col in zip(kinds, ("tab:blue", "tab:green", "tab:orange")):
        ax.plot(per_kind[kind]["losses"], color=col, lw=0.8, label=kind)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.legend(fontsize=8)
    ax.set_title("hetK training loss", fontsize=10)
    fig.savefig(os.path.join(FIG, "census_train_loss.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
