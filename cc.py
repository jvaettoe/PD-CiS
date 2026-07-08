import math
import os
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import einops
from abc import ABC, abstractmethod
import numpy as np
import random

from jaxtyping import Float
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from mlp import MLP, LinearMLP, EmbedMLP 
from data import SparseDataset, CleanDataset
from pd import APD, OracleAPD, SPD

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model: MLP, dataset: SparseDataset, n: float = 2, batch_size: int = 2048,
          steps: int = 10_000, cosine_scheduler: bool = True) -> list[float]:
    """Trains MLP with L_n loss, set to MSE as default"""
    parameters = list(model.parameters())
    assert all(p.requires_grad for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps) if cosine_scheduler else None
    losses = []
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        inputs, labels = dataset.generate_batch(batch_size)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = (torch.abs(outputs - labels) ** n).mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if step % 100 == 0:
            pbar.set_postfix({"loss": loss.item()})
    return losses
    
# --------------------------------------------------------------------------
# Persistence: a checkpoint boundary that caches training runs, 
# the dataset config, loss curve, and metadata needed to reconstruct a run.
# --------------------------------------------------------------------------

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
_DATASET_REGISTRY = {"CleanDataset": CleanDataset, "SparseDataset": SparseDataset}


def compute_n_steps(p: float, n_features: int, base_steps: int = 10_000) -> int:
    """Step-scaling used by both scripts: scale up to compensate for all-zero
    batches (fraction (1-p)**n_features carry no signal)."""
    frac_zero = (1 - p) ** n_features
    return int(base_steps / (1 - frac_zero))


def checkpoint_path(str_p: str, n_features: int = 50, d_mlp: int = 50,
                    seed: int = 42, ckpt_dir: str = CKPT_DIR) -> str:
    """checkpoints/ncc_f{nf}_d{d}_p{str_p}_s{seed}.pt

    Keyed by the *string* p (reuses the exact tokens in str_train_ps and avoids
    0.001-vs-1e-3 float ambiguity); f/d/s make architecture/seed changes yield a
    distinct artifact instead of silently reloading a stale one."""
    fname = f"ncc_f{n_features}_d{d_mlp}_p{str_p}_s{seed}.pt"
    return os.path.join(ckpt_dir, fname)


def save_checkpoint(path: str, model: LinearMLP, dataset: SparseDataset,
                    losses: list[float], *, seed: int, n_steps: int,
                    batch_size: int = 2048, cosine_scheduler: bool = True) -> None:
    """Persist everything needed to reconstruct model AND dataset deterministically.
    device is deliberately not saved (machine-dependent; chosen at load time)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": {"n_features": model.n_features, "d_mlp": model.d_mlp},
        "state_dict": model.state_dict(),
        "dataset_class": type(dataset).__name__,
        "dataset_config": {
            "n_features": dataset.n_features,
            "p": dataset.p,
            "seed": dataset.seed,
            "exactly_one_active_feature": dataset.exactly_one_active_feature,
        },
        "losses": losses,
        "seed": seed,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "cosine_scheduler": cosine_scheduler,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: str | None = None) -> dict:
    """Reconstruct a run from a checkpoint. Returns
    {"model": MLP, "dataset": SparseDataset, "losses": list[float], "meta": payload}."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)

    model = LinearMLP(device=device, **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dataset_cls = _DATASET_REGISTRY[payload["dataset_class"]]
    dataset = dataset_cls(device=device, **payload["dataset_config"])

    return {"model": model, "dataset": dataset,
            "losses": payload["losses"], "meta": payload}


def load_or_train(str_p: str, *, n_features: int = 50, d_mlp: int = 50,
                  seed: int = 42, dataset_seed: int | None = None,
                  exactly_one_active_feature: bool = False,
                  base_steps: int = 10_000, batch_size: int = 2048,
                  cosine_scheduler: bool = True, retrain: bool = False,
                  ckpt_dir: str = CKPT_DIR, device: str | None = None) -> dict:
    """Loads a matching checkpoint if one exists, otherwise trains from scratch and saves it."""
    path = checkpoint_path(str_p, n_features=n_features, d_mlp=d_mlp,
                           seed=seed, ckpt_dir=ckpt_dir)
    if os.path.exists(path) and not retrain:
        return load_checkpoint(path, device=device)

    p = float(str_p)
    set_seed(seed)
    dataset = CleanDataset(n_features=n_features, p=p,
                           seed=dataset_seed if dataset_seed is not None else seed,
                           device=device,
                           exactly_one_active_feature=exactly_one_active_feature)
    model = LinearMLP(n_features=n_features, d_mlp=d_mlp, device=device)
    n_steps = compute_n_steps(p, n_features, base_steps)
    losses = train(model, dataset, batch_size=batch_size, steps=n_steps,
                   cosine_scheduler=cosine_scheduler)
    save_checkpoint(path, model, dataset, losses, seed=seed, n_steps=n_steps,
                    batch_size=batch_size, cosine_scheduler=cosine_scheduler)
    return load_checkpoint(path, device=device)
