"""Train ramp-target models (saturating ramp, theta = 0.5) under L4 at p = 0.01.

  python train.py                  # both architectures, bias-free
  python train.py --bias           # both architectures, hidden-layer biases
  python train.py --arch chain
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import SparseDataset
from model import ChainMLP, SingleMLP

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SEED = 42
P = 0.01
ARCHS = ("chain", "single")


def loss_fn(out, y):
    return ((out - y).abs() ** 4).mean()


def train(arch: str, bias: bool) -> None:
    tok = "_bias" if bias else ""
    path = os.path.join(CKPT, f"ramp_{arch}{tok}_p{P:g}_l4_s{SEED}.pt")
    if os.path.exists(path):
        print(f"skip {path} (exists)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = (ChainMLP if arch == "chain" else SingleMLP)(bias=bias)
    ds = SparseDataset(P, seed=SEED)
    steps = round(10000 / ds.p_any_active())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for _ in tqdm(range(steps), desc=f"ramp {arch}{tok} p={P:g} l4"):
        x, y = ds.generate_batch(2048)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": type(model).__name__,
                "model_config": model.config(), "state_dict": model.state_dict(),
                "p": P, "loss_kind": "l4", "arch": arch, "bias": bias,
                "losses": losses, "seed": SEED, "n_steps": steps,
                "batch_size": 2048}, path)
    print(f"ramp {arch}{tok} p={P:g} l4: final loss {losses[-1]:.3e}; "
          f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("chain", "single", "both"),
                    default="both")
    ap.add_argument("--bias", action="store_true",
                    help="hidden-layer biases (output layer stays bias-free)")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    for arch in (ARCHS if a.arch == "both" else (a.arch,)):
        train(arch, a.bias)


if __name__ == "__main__":
    main()
