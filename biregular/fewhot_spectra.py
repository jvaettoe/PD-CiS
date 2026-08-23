"""Few-hot forward spectra, previous-suite style: full model vs the JOINT
n-component ExactPD forward, side by side.

Original biregular code (CodeAnsatz K=5, seed 0, no-swap, transpose decoder,
(a, b) fitted at p = 0.001). For a k-hot input with active set S the masked
forward with exactly those k components enabled shares the full model's hidden
state (inactive read pieces see x_j = 0), so it reproduces the ON-TARGET
responses exactly -- including active-active collision suppression -- and
writes exactly zero off-target. The full-model panel's off-target bars are the
interference the k-component mechanism cannot express (shaded by aggregate
codeword overlap with S).

  ./.venv/bin/python -m biregular.fewhot_spectra
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from biregular.biregular_ansatz import CodeAnsatz
from core.paths import FIGURES
from core.pd import ExactPD

P = 0.001
KS = (1, 2, 3, 5)
PROBE_SEED = 7
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COMP_C = "#eb6834"
OUT_DIR = os.path.join(FIGURES, "biregular", "fewhot")
os.makedirs(OUT_DIR, exist_ok=True)


def main(ks=KS, fname="fewhot_full_vs_kcomponents_K5_p0.001.png"):
    ans = CodeAnsatz(5, swaps=0, decoder="transpose")
    a, b = ans.fit_ab(P)
    model = ans.model((a, b))
    pd_e = ExactPD(model)
    F = model.n_features
    M = torch.tensor(ans.M, dtype=torch.float32)
    rng = np.random.default_rng(PROBE_SEED)
    cmap = plt.get_cmap("Blues")

    fig, axes = plt.subplots(len(ks), 2, figsize=(13.5, 2.7 * len(KS)),
                             sharex=True, sharey="row")
    for row, k in enumerate(ks):
        S = (list(range(F)) if k == F
             else sorted(rng.choice(F, size=k, replace=False).tolist()))
        x = torch.zeros(1, F)
        x[0, list(S)] = 1.0
        mask = torch.zeros(pd_e.C)
        mask[list(S)] = 1.0
        with torch.no_grad():
            full = model(x)[0]
            joint = pd_e.forward_masked(x, {"w_in": mask, "w_out": mask})[0]
        agg_m = (M @ M[list(S)].sum(0)).numpy()
        off = np.array([i not in S for i in range(F)])
        mmax = max(agg_m[off].max(), 1.0) if off.any() else 1.0

        ax = axes[row, 0]
        cols = [cmap(0.25 + 0.7 * agg_m[i] / mmax) if off[i] else "#2a78d6"
                for i in range(F)]
        ax.bar(range(F), full.numpy(), width=1.0, color=cols, zorder=2)
        if k <= 10:
            ax.scatter(list(S), [1.0] * k, s=26, marker="_", color=INK, zorder=4)
        leak_pow = (float((full[torch.tensor(off)] ** 2).sum())
                    if off.any() else 0.0)
        s_lbl = "ALL" if k == F else str(S)
        ax.set_title(f"k = {k}, active {s_lbl}: FULL model "
                     f"(off-target power {leak_pow:.3f}; shading = aggregate "
                     "overlap with S)", fontsize=8.5, color=INK)

        ax = axes[row, 1]
        ax.bar(range(F), joint.numpy(), width=1.0, color=COMP_C, zorder=2)
        if k <= 10:
            ax.scatter(list(S), [1.0] * k, s=26, marker="_", color=INK, zorder=4)
        on_dev = float((joint[list(S)] - full[list(S)]).abs().max())
        off_max = (float(joint[torch.tensor(off)].abs().max())
                   if off.any() else 0.0)
        ax.set_title(f"{k}-COMPONENT forward (components {s_lbl}): max on-target "
                     f"|dev from full| = {on_dev:.1e}; max off-target = "
                     f"{off_max:.1e}", fontsize=8.5, color=INK)

        for ax in axes[row]:
            ax.axhline(0, lw=0.6, color=INK2)
            ax.tick_params(labelsize=7, colors=INK2)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("output coordinate (feature)", color=INK)
    fig.suptitle("Few-hot probes: full model vs joint k-component ExactPD "
                 f"forward (biregular K=5 transpose, p={P}, values +1; "
                 "black dash = target)", color=INK2, fontsize=10, y=0.995)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
    main(ks=(1, 10, 25, 100), fname="allhot_ramp_K5_p0.001.png")
    loss_curve()


def loss_curve(n_draws=100, seed=123,
               fname="fewhot_loss_vs_k_K5_p0.001.png"):
    """Full-model vs joint-k-component forward losses as a function of k (number
    of hot inputs, values +1, random active sets, mean over draws).

    The k-component forward is exact on-target and zero off-target, so its loss
    is the pure on-target (attenuation/collision) error, while the full model
    adds the interference field; the ratio full/component is the few-hot
    generalization of the leak/attenuation ratio (-> 13.7 at k = 1 for MSE,
    -> 1 as the dense regime kills both outputs equally)."""
    ans = CodeAnsatz(5, swaps=0, decoder="transpose")
    a, b = ans.fit_ab(P)
    model = ans.model((a, b))
    pd_e = ExactPD(model)
    F = model.n_features
    rng = np.random.default_rng(seed)
    ks = list(range(1, 21)) + list(range(22, 51, 2)) + list(range(55, 101, 5))
    res = {m: {"full": [], "comp": []} for m in ("mse", "l4")}
    with torch.no_grad():
        for k in ks:
            acc = {m: {"full": 0.0, "comp": 0.0} for m in ("mse", "l4")}
            for _ in range(n_draws):
                S = rng.choice(F, size=k, replace=False)
                x = torch.zeros(1, F)
                x[0, list(S)] = 1.0
                mask = torch.zeros(pd_e.C)
                mask[list(S)] = 1.0
                target = torch.relu(x)[0]
                full = model(x)[0]
                comp = pd_e.forward_masked(x, {"w_in": mask, "w_out": mask})[0]
                for m, p_ in (("mse", 2), ("l4", 4)):
                    acc[m]["full"] += float(((full - target).abs() ** p_).mean())
                    acc[m]["comp"] += float(((comp - target).abs() ** p_).mean())
            for m in ("mse", "l4"):
                res[m]["full"].append(acc[m]["full"] / n_draws)
                res[m]["comp"].append(acc[m]["comp"] / n_draws)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, m, lbl in ((axes[0], "mse", "MSE"), (axes[1], "l4", "L4")):
        ax.plot(ks, res[m]["full"], color="#2a78d6", lw=1.5, label="full model")
        ax.plot(ks, res[m]["comp"], color="#eb6834", lw=1.5,
                label="k-component forward")
        ax.set_yscale("log")
        ax.set_xlabel("k (number of hot inputs)", color=INK)
        ax.set_ylabel(f"{lbl} loss vs target ReLU(x)", color=INK)
        ax.set_title(f"{lbl}", color=INK, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    for m, c, lbl in (("mse", "#2a78d6", "MSE"), ("l4", "#eb6834", "L4")):
        ratio = [f / c_ if c_ > 0 else np.nan
                 for f, c_ in zip(res[m]["full"], res[m]["comp"])]
        ax.plot(ks, ratio, color=c, lw=1.5, label=lbl)
    ax.axhline(1.0, ls="--", lw=0.8, color=MUTED_ if (MUTED_ := "#898781") else None)
    ax.set_yscale("log")
    ax.set_xlabel("k (number of hot inputs)", color=INK)
    ax.set_ylabel("loss ratio full / k-component", color=INK)
    ax.set_title("relative loss (interference burden)", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Loss vs number of hot inputs: full model vs joint k-component "
                 f"ExactPD forward (biregular K=5 transpose, p={P}, values +1, "
                 f"{n_draws} draws/k)", color=INK2, fontsize=10, y=0.99)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    for ax in axes:
        ax.set_xscale("log")
    fig.savefig(os.path.join(OUT_DIR, fname.replace(".png", "_loglog.png")),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path} (+ _loglog)")
