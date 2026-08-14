"""Train correlated-pair models over the rho sweep under MSE and/or L4.

  python train.py                       # all rho x both losses
  python train.py --rho 0.5 --loss mse
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import CorrDataset
from model import MLP

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SEED = 42
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def loss_exponent(kind: str) -> float:
    return 2.0 if kind == "mse" else float(kind[1:])


def loss_fn(kind: str, out, y):
    err = out - y
    n = loss_exponent(kind)
    return (err ** 2).mean() if n == 2.0 else (err.abs() ** n).mean()


def train(rho: float, kind: str, steps_override: int | None = None) -> None:
    tok = f"_st{steps_override}" if steps_override else ""
    path = os.path.join(CKPT, f"corr_rho{rho:g}_{kind}{tok}_s{SEED}.pt")
    if os.path.exists(path):
        print(f"skip {path} (exists)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = MLP()
    ds = CorrDataset(rho, seed=SEED)
    steps = steps_override or round(10000 / ds.p_any_active())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for _ in tqdm(range(steps), desc=f"corr rho={rho:g} {kind}"):
        x, y = ds.generate_batch(2048)
        opt.zero_grad()
        loss = loss_fn(kind, model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "MLP",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "rho": rho, "loss_kind": kind, "losses": losses, "seed": SEED,
                "n_steps": steps, "batch_size": 2048}, path)
    print(f"corr rho={rho:g} {kind}: final loss {losses[-1]:.3e}; saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, nargs="+", default=list(RHOS))
    ap.add_argument("--loss", nargs="+", default=["mse", "l4"],
                    help="mse, l4, or generic l<n>")
    ap.add_argument("--steps", type=int, default=None,
                    help="fixed step count (unconfounds the rho-dependent formula)")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    for rho in a.rho:
        for kind in (("mse", "l4") if a.loss == ["both"] else a.loss):
            train(rho, kind, a.steps)


if __name__ == "__main__":
    main()
