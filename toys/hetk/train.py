"""Train hetK models (two-population sparsity).

Loss kinds: mse (primary), l4 (comparison), wmse (importance-weighted MSE:
per-feature weight 1/p_j normalized to mean 1 — equalizes the populations'
expected loss contribution, disentangling loss SHAPE from effective importance).

Sweeps: --p-rare (abandonment boundary; common stays 0.05) and --n-rare
(population split). Default-config checkpoints keep the original names
(hetk_{loss}_s42.pt); sweep runs add _prare{p} / _nrare{n} tokens.

  python train.py                              # default config, mse + l4
  python train.py --loss wmse
  python train.py --p-rare 0.002 0.005 0.01 0.02 0.05 --loss mse l4
  python train.py --n-rare 10 25 --loss mse l4
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import HetKDataset, N_RARE, P_RARE
from model import MLP

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SEED = 42


def ckpt_name(kind: str, p_rare: float, n_rare: int) -> str:
    parts = ["hetk"]
    if p_rare != P_RARE:
        parts.append(f"prare{p_rare:g}")
    if n_rare != N_RARE:
        parts.append(f"nrare{n_rare}")
    parts.append(kind)
    return "_".join(parts) + f"_s{SEED}.pt"


def loss_fn(kind: str, out, y, weights=None):
    err = out - y
    if kind == "mse":
        return (err ** 2).mean()
    if kind == "wmse":
        return (weights * err ** 2).mean()
    return (err.abs() ** 4).mean()


def train(kind: str, p_rare: float = P_RARE, n_rare: int = N_RARE) -> None:
    path = os.path.join(CKPT, ckpt_name(kind, p_rare, n_rare))
    if os.path.exists(path):
        print(f"skip {path} (exists)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = MLP()
    ds = HetKDataset(seed=SEED, p_rare=p_rare, n_rare=n_rare)
    weights = (1 / ds.p)
    weights = weights / weights.mean()
    steps = round(10000 / ds.p_any_active())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    desc = f"hetk p_rare={p_rare:g} n_rare={n_rare} {kind}"
    for _ in tqdm(range(steps), desc=desc):
        x, y = ds.generate_batch(2048)
        opt.zero_grad()
        loss = loss_fn(kind, model(x), y, weights)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "MLP",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "loss_kind": kind, "p_rare": p_rare, "n_rare": n_rare,
                "losses": losses, "seed": SEED, "n_steps": steps,
                "batch_size": 2048}, path)
    print(f"{desc}: final loss {losses[-1]:.3e}; saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", nargs="+", choices=("mse", "l4", "wmse"),
                    default=["mse", "l4"])
    ap.add_argument("--p-rare", type=float, nargs="+", default=[P_RARE],
                    dest="p_rare")
    ap.add_argument("--n-rare", type=int, nargs="+", default=[N_RARE],
                    dest="n_rare")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    for p_rare in a.p_rare:
        for n_rare in a.n_rare:
            for kind in a.loss:
                train(kind, p_rare, n_rare)


if __name__ == "__main__":
    main()
