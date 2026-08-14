"""K(p) sweep: learned codeword length of the trained MLP (F=100, N=50, L4) as a
function of input sparsity p, from 0.001 to 1.0.

Trains any missing checkpoints with the exact original recipe (AdamW 3e-3 wd 1e-2,
cosine, batch 2048, steps = 10000/(1-(1-p)^100), seed 42), then measures the
per-feature codeword length two ways:
  K_tau  = #(w_in[j,:] > 0.05)   (de Ferreira-Heimersheim's threshold count)
  K_pos  = #(w_in[j,:] > 0)      (all positive preactivations on e_j)
and plots median with min-max band across the 100 features.

Outputs: results/ksweep/ + figures/ksweep/.

  ./.venv/bin/python ksweep.py
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data import CleanDataset
from mlp import MLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RES_DIR = os.path.join(os.path.dirname(__file__), "results", "ksweep")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "ksweep")
PS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0)
SEED = 42
TAU = 0.05


def ckpt_path(p: float) -> str:
    return os.path.join(CKPT_DIR, f"ncc_MLP_d_mlp50_n_features100_p{p:g}_n4_s{SEED}.pt")


def n_steps(p: float) -> int:
    return int(10_000 / (1 - (1 - p) ** 100))


def train(p: float) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = MLP(n_features=100, d_mlp=50, device="cpu")
    dataset = CleanDataset(n_features=100, p=p, device="cpu", seed=SEED)
    steps = n_steps(p)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for _ in tqdm(range(steps), desc=f"MLP p={p:g}"):
        x, y = dataset.generate_batch(2048)
        opt.zero_grad()
        loss = (torch.abs(model(x) - y) ** 4).mean()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "MLP",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "dataset_class": "CleanDataset",
                "dataset_config": {"n_features": 100, "p": p, "seed": SEED,
                                   "exactly_one_active_feature": False},
                "losses": losses, "seed": SEED, "n_steps": steps, "n_exp": 4.0,
                "batch_size": 2048, "cosine_scheduler": True}, ckpt_path(p))
    print(f"  trained p={p:g}: final loss {losses[-1]:.3e}")


def main():
    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    stats = []
    for p in PS:
        if not os.path.exists(ckpt_path(p)):
            train(p)
        ck = torch.load(ckpt_path(p), map_location="cpu", weights_only=False)
        m = MLP(**ck["model_config"], device="cpu")
        m.load_state_dict(ck["state_dict"])
        W = m.w_in.detach()                       # (F, N): row j = preacts on e_j
        k_tau = (W > TAU).sum(1).float()
        k_pos = (W > 0).sum(1).float()
        stats.append({
            "p": p,
            "tau_med": float(k_tau.median()), "tau_lo": float(k_tau.min()),
            "tau_hi": float(k_tau.max()),
            "pos_med": float(k_pos.median()), "pos_lo": float(k_pos.min()),
            "pos_hi": float(k_pos.max())})
        print(f"p={p:g}: K(tau={TAU}) median {stats[-1]['tau_med']:.0f} "
              f"[{stats[-1]['tau_lo']:.0f}, {stats[-1]['tau_hi']:.0f}]; "
              f"K(>0) median {stats[-1]['pos_med']:.0f} "
              f"[{stats[-1]['pos_lo']:.0f}, {stats[-1]['pos_hi']:.0f}]")

    cols = ["p", "K_tau_median", "K_tau_min", "K_tau_max",
            "K_pos_median", "K_pos_min", "K_pos_max"]
    with open(os.path.join(RES_DIR, "K_of_p.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for s in stats:
            f.write(f"{s['p']},{s['tau_med']},{s['tau_lo']},{s['tau_hi']},"
                    f"{s['pos_med']},{s['pos_lo']},{s['pos_hi']}\n")
    md = ["### Learned codeword length K vs input sparsity p (trained MLP F=100 "
          "N=50, L4, seed 42)", "",
          "| p | K (tau=0.05) median [min, max] | K (>0) median [min, max] |",
          "|---|---|---|"]
    for s in stats:
        md.append(f"| {s['p']:g} | {s['tau_med']:.0f} [{s['tau_lo']:.0f}, "
                  f"{s['tau_hi']:.0f}] | {s['pos_med']:.0f} [{s['pos_lo']:.0f}, "
                  f"{s['pos_hi']:.0f}] |")
    with open(os.path.join(RES_DIR, "K_of_p.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ps = [s["p"] for s in stats]
    ax.fill_between(ps, [s["tau_lo"] for s in stats], [s["tau_hi"] for s in stats],
                    color="tab:blue", alpha=0.2, label=f"K(τ={TAU}) min-max")
    ax.plot(ps, [s["tau_med"] for s in stats], "o-", color="tab:blue", lw=2,
            label=f"K(τ={TAU}) median")
    ax.plot(ps, [s["pos_med"] for s in stats], "s--", color="tab:orange", lw=1.5,
            label="K(>0) median")
    ax.fill_between(ps, [s["pos_lo"] for s in stats], [s["pos_hi"] for s in stats],
                    color="tab:orange", alpha=0.12)
    ax.axhline(5, color="grey", ls=":", lw=0.8)
    ax.text(0.0011, 5.4, "K=5 (dFH designed optimum at p=0.02)", fontsize=7,
            color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("input sparsity p (log)")
    ax.set_ylabel("learned codeword length K (per-feature distribution)")
    ax.set_title("K(p): learned code density vs sparsity — trained MLP "
                 "(F=100 → N=50, L4)", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "K_of_p.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
