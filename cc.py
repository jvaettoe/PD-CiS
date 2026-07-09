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
_MODEL_REGISTRY = {"LinearMLP": LinearMLP, "EmbedMLP": EmbedMLP}


def compute_n_steps(p: float, n_features: int, base_steps: int = 10_000) -> int:
    """Step-scaling used by both scripts: scale up to compensate for all-zero
    batches (fraction (1-p)**n_features carry no signal)."""
    frac_zero = (1 - p) ** n_features
    return int(base_steps / (1 - frac_zero))


def _kwargs_key(model_kwargs: dict) -> str:
    """Deterministic, human-readable key from a model's constructor kwargs.
    Sorted keys kill dict-ordering ambiguity. All current kwargs are ints so
    f'{k}{v}' is exact; if a float kwarg is ever added, append a short hash of
    repr(sorted(model_kwargs.items())) to avoid 0.1-vs-1e-1 collisions."""
    return "_".join(f"{k}{model_kwargs[k]}" for k in sorted(model_kwargs))


def _fmt_n(n_exp: float) -> str:
    """Format the loss exponent for filenames: 2.0 -> '2', 2.5 -> '2.5'."""
    return f"{n_exp:g}"


def checkpoint_path(str_p: str, model_cls: type, model_kwargs: dict,
                    seed: int = 42, ckpt_dir: str = CKPT_DIR,
                    n_exp: float = 2.0) -> str:
    """checkpoints/ncc_{ModelName}_{sortedkwargs}_p{str_p}_n{n_exp}_s{seed}.pt

    Keyed by the *string* p (reuses the exact tokens in str_train_ps and avoids
    0.001-vs-1e-3 float ambiguity); model class name + sorted kwargs + loss
    exponent + seed make any architecture/exponent/seed change yield a distinct
    artifact instead of silently reloading a stale one, and stop LinearMLP/
    EmbedMLP with matching n_features/d_mlp from colliding."""
    fname = (f"ncc_{model_cls.__name__}_{_kwargs_key(model_kwargs)}"
             f"_p{str_p}_n{_fmt_n(n_exp)}_s{seed}.pt")
    return os.path.join(ckpt_dir, fname)


def save_checkpoint(path: str, model: MLP, dataset: SparseDataset,
                    losses: list[float], *, seed: int, n_steps: int,
                    batch_size: int = 2048, cosine_scheduler: bool = True,
                    n_exp: float = 2.0) -> None:
    """Persist everything needed to reconstruct model AND dataset deterministically.
    device is deliberately not saved (machine-dependent; chosen at load time).
    Model is stored generically via its class name + config() so any MLP subclass
    (LinearMLP, EmbedMLP, ...) round-trips."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format_version": 2,
        "model_class": type(model).__name__,
        "model_config": model.config(),
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
        "n_exp": n_exp,
        "batch_size": batch_size,
        "cosine_scheduler": cosine_scheduler,
    }
    tmp = path + ".tmp"          # atomic write so a concurrent reader never sees a
    torch.save(payload, tmp)     # half-written checkpoint (e.g. a --no-train decomp
    os.replace(tmp, path)        # run alongside a background --models-only trainer)


def load_checkpoint(path: str, device: str | None = None) -> dict:
    """Reconstruct a run from a checkpoint. Returns
    {"model": MLP, "dataset": SparseDataset, "losses": list[float], "meta": payload}."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)

    model_cls = _MODEL_REGISTRY[payload["model_class"]]
    model = model_cls(device=device, **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dataset_cls = _DATASET_REGISTRY[payload["dataset_class"]]
    dataset = dataset_cls(device=device, **payload["dataset_config"])

    return {"model": model, "dataset": dataset,
            "losses": payload["losses"], "meta": payload}


def load_or_train(str_p: str, model_cls: type = LinearMLP,
                  model_kwargs: dict | None = None, *,
                  seed: int = 42, dataset_seed: int | None = None,
                  exactly_one_active_feature: bool = False,
                  base_steps: int = 10_000, batch_size: int = 2048,
                  cosine_scheduler: bool = True, retrain: bool = False,
                  ckpt_dir: str = CKPT_DIR, device: str | None = None,
                  n_exp: float = 2.0) -> dict:
    """Loads a matching checkpoint if one exists, otherwise trains from scratch and
    saves it. Generic over MLP subclass: the caller specializes by passing model_cls
    and model_kwargs, e.g. load_or_train("0.1", LinearMLP, {"n_features":50,"d_mlp":50})
    or load_or_train("0.1", EmbedMLP, {"n_features":50,"d_emb":16,"d_mlp":50}). The
    dataset shares n_features with the model (read from model_kwargs). n_exp is the
    training-loss exponent (loss = |out - labels|**n_exp); it is part of the
    checkpoint key so distinct exponents yield distinct artifacts."""
    model_kwargs = model_kwargs or {"n_features": 50, "d_mlp": 50}
    n_features = model_kwargs["n_features"]
    path = checkpoint_path(str_p, model_cls, model_kwargs, seed=seed,
                           ckpt_dir=ckpt_dir, n_exp=n_exp)
    if os.path.exists(path) and not retrain:
        return load_checkpoint(path, device=device)

    p = float(str_p)
    set_seed(seed)
    dataset = CleanDataset(n_features=n_features, p=p,
                           seed=dataset_seed if dataset_seed is not None else seed,
                           device=device,
                           exactly_one_active_feature=exactly_one_active_feature)
    model = model_cls(device=device, **model_kwargs)
    n_steps = compute_n_steps(p, n_features, base_steps)
    losses = train(model, dataset, n=n_exp, batch_size=batch_size, steps=n_steps,
                   cosine_scheduler=cosine_scheduler)
    save_checkpoint(path, model, dataset, losses, seed=seed, n_steps=n_steps,
                    batch_size=batch_size, cosine_scheduler=cosine_scheduler,
                    n_exp=n_exp)
    return load_checkpoint(path, device=device)


# --------------------------------------------------------------------------
# Demo: the generic train/save/reload path specialised to each MLP subclass.
# The checkpoint machinery is architecture-agnostic (model.config() + registry);
# here "main" picks LinearMLP and EmbedMLP concretely and renders the
# feature-space weight-space plots (projected through readin()/readout() for
# EmbedMLP).  Run:  ./.venv/bin/python cc.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # headless: write figures without a display

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    str_p = "0.05"
    specs = [
        (LinearMLP, {"n_features": 20, "d_mlp": 20}),
        (EmbedMLP, {"n_features": 20, "d_emb": 8, "d_mlp": 20}),
    ]

    for model_cls, model_kwargs in specs:
        name = model_cls.__name__
        run = load_or_train(str_p, model_cls, model_kwargs,
                            base_steps=1_500, device="cpu")
        model = run["model"]
        ckpt = os.path.basename(checkpoint_path(str_p, model_cls, model_kwargs))
        print(f"{name}: final loss {run['losses'][-1]:.4f} | ckpt {ckpt}")

        # Feature-space weight view (EmbedMLP: projected via readin()/readout()).
        fig = model.plot_weights(project=isinstance(model, EmbedMLP))
        fig.suptitle(f"{name}: read-in / read-out (feature space)")
        fig.savefig(os.path.join(fig_dir, f"{name}_weights.png"), dpi=90)
        plt.close(fig)

        # Effective feature -> feature map (readin() @ readout()).
        eff = model.effective_matrix()
        fig, ax = plt.subplots(constrained_layout=True)
        v = eff.abs().max().item()
        im = ax.imshow(eff.cpu().numpy(), cmap="RdBu", vmin=-v, vmax=v)
        ax.set_title(f"{name}: effective feature->feature map")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(fig_dir, f"{name}_effective.png"), dpi=90)
        plt.close(fig)

        # Per-feature neuron assignment (feature-space stacked bars).
        ax = model.plot_weight_bars(show_cbar=True)
        ax.set_title(f"{name}: feature -> neuron assignment")
        ax.get_figure().savefig(os.path.join(fig_dir, f"{name}_bars.png"), dpi=90)
        plt.close(ax.get_figure())

    print(f"figures written to {fig_dir}")
