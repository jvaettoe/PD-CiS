"""Train U-AND models (all 105 pairwise ANDs of 15 booleans) over the width and
sparsity sweep under MSE and/or L4.

  python train.py                              # full sweep x both losses
  python train.py --n-hidden 20 --p 0.1 --loss mse
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import N_OUT, UANDDataset
from model import UAND

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SEED = 42
WIDTHS = (10, 20, 35, 50, 105)
PS = (0.1, 0.2)


def loss_exponent(kind: str) -> float:
    """'mse' -> 2, 'l4' -> 4, generic 'l<n>' -> n (Ln loss = mean |err|^n)."""
    if kind == "mse":
        return 2.0
    return float(kind[1:])


def loss_fn(kind: str, out, y):
    err = out - y
    n = loss_exponent(kind)
    return (err ** 2).mean() if n == 2.0 else (err.abs() ** n).mean()


def train(n_hidden: int, p: float, kind: str) -> None:
    path = os.path.join(CKPT, f"uand_N{n_hidden}_p{p:g}_{kind}_s{SEED}.pt")
    if os.path.exists(path):
        print(f"skip {path} (exists)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = UAND(d_mlp=n_hidden, n_out=N_OUT)
    ds = UANDDataset(p, seed=SEED)
    steps = round(10000 / ds.p_any_gate())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for _ in tqdm(range(steps), desc=f"uand N={n_hidden} p={p:g} {kind}"):
        x, y = ds.generate_batch(2048)
        opt.zero_grad()
        loss = loss_fn(kind, model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "UAND",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "p": p, "loss_kind": kind, "losses": losses, "seed": SEED,
                "n_steps": steps, "batch_size": 2048}, path)
    print(f"uand N={n_hidden} p={p:g} {kind}: final loss {losses[-1]:.3e}; "
          f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hidden", type=int, nargs="+", default=list(WIDTHS),
                    dest="n_hidden")
    ap.add_argument("--p", type=float, nargs="+", default=list(PS))
    ap.add_argument("--loss", nargs="+", default=["mse", "l4"],
                    help="loss kinds: mse, l4, or generic l<n> (e.g. l2.5) for "
                         "Ln loss = mean |err|^n")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    for p in a.p:
        for n in a.n_hidden:
            for kind in (("mse", "l4") if a.loss == ["both"] else a.loss):
                train(n, p, kind)


if __name__ == "__main__":
    main()
