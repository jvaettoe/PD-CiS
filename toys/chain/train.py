"""Train two-hidden-layer chain models under MSE and/or L4.

  python train.py                      # p in {0.001, 0.01} x both losses
  python train.py --p 0.01 --loss mse
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import SparseDataset
from model import ChainMLP

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SEED = 42
PS = (0.001, 0.01)


def loss_fn(kind: str, out, y):
    err = out - y
    return (err ** 2).mean() if kind == "mse" else (err.abs() ** 4).mean()


def train(p: float, kind: str) -> None:
    path = os.path.join(CKPT, f"chain_p{p:g}_{kind}_s{SEED}.pt")
    if os.path.exists(path):
        print(f"skip {path} (exists)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = ChainMLP()
    ds = SparseDataset(p, seed=SEED)
    steps = round(10000 / ds.p_any_active())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for _ in tqdm(range(steps), desc=f"chain p={p:g} {kind}"):
        x, y = ds.generate_batch(2048)
        opt.zero_grad()
        loss = loss_fn(kind, model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "ChainMLP",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "p": p, "loss_kind": kind, "losses": losses, "seed": SEED,
                "n_steps": steps, "batch_size": 2048}, path)
    print(f"chain p={p:g} {kind}: final loss {losses[-1]:.3e}; saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=list(PS))
    ap.add_argument("--loss", choices=("mse", "l4", "both"), default="both")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    for p in a.p:
        for kind in (("mse", "l4") if a.loss == "both" else (a.loss,)):
            train(p, kind)


if __name__ == "__main__":
    main()
