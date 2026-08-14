"""Train an EmbedMLP checkpoint matching the original sweep recipe exactly
(archived cc.py train(): AdamW lr=3e-3 wd=1e-2, cosine, batch 2048, 105033 steps,
L4, CleanDataset(100, p, seed 42), model seed 42), for d_emb values the sweep
skipped (e.g. 40 — the transition probe between d_emb=20 and d_emb=N=50).

  ./.venv/bin/python train_embed_demb.py --d-emb 40
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import CleanDataset
from mlp import EmbedMLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train(d_emb: int, p: float, steps: int, seed: int) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = EmbedMLP(n_features=100, d_emb=d_emb, d_mlp=50, device="cpu")
    dataset = CleanDataset(n_features=100, p=p, device="cpu", seed=seed)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for step in tqdm(range(steps), desc=f"EmbedMLP d_emb={d_emb} s{seed}"):
        x, y = dataset.generate_batch(2048)
        opt.zero_grad()
        loss = (torch.abs(model(x) - y) ** 4).mean()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

    payload = {"format_version": 2, "model_class": "EmbedMLP",
               "model_config": model.config(), "state_dict": model.state_dict(),
               "dataset_class": "CleanDataset",
               "dataset_config": {"n_features": 100, "p": p, "seed": seed,
                                  "exactly_one_active_feature": False},
               "losses": losses, "seed": seed, "n_steps": steps,
               "n_exp": 4.0, "batch_size": 2048, "cosine_scheduler": True}
    path = os.path.join(CKPT_DIR, f"ncc_EmbedMLP_d_emb{d_emb}_d_mlp50"
                                  f"_n_features100_p{p:g}_n4_s{seed}.pt")
    torch.save(payload, path)
    print(f"final loss {losses[-1]:.3e}; saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-emb", type=int, required=True, dest="d_emb")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=105033)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a.d_emb, a.p, a.steps, a.seed)


if __name__ == "__main__":
    main()
