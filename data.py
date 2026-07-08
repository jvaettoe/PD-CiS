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


class SparseDataset:
    """Dataset that generates inputs and labels for a sparse MLP; from previous paper's implementation."""

    def __init__(self, n_features: int, p: float, device: str = None, seed: int | None = None, exactly_one_active_feature: bool = False):
        self.n_features = n_features
        self.set_p(p, exactly_one_active_feature)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed if seed is not None else int(time.time())
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.relu = nn.ReLU()

    def set_p(self, p: float, exactly_one_active_feature: bool = False):
        self.p = p
        self.exactly_one_active_feature = exactly_one_active_feature

    def _generate_inputs(self, batch_size: int, p: float | None = None) -> Float[Tensor, "batch n_features"]:
        p = p or self.p
        batch = torch.zeros((batch_size, self.n_features), device=self.device)
        mask = torch.rand(batch.shape, device=self.device, generator=self.generator) < p
        if self.exactly_one_active_feature:
            active_features = torch.randint(0, self.n_features, (batch_size,), device=self.device, generator=self.generator).unsqueeze(1)
            values = torch.rand(batch_size, 1, device=self.device, generator=self.generator) * 2 - 1
            batch.scatter_(1, active_features, values)
        else:
            values = torch.rand(batch.shape, device=self.device, generator=self.generator) * 2 - 1
            batch = values * mask
        return batch

    @abstractmethod
    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        pass

    def generate_batch(self, batch_size: int, p: float | None = None) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        inputs = self._generate_inputs(batch_size, p)
        labels = self._generate_labels(inputs)
        return inputs, labels

    def plot_M(self, ax: plt.Axes | None = None) -> plt.Axes:
        ax = ax or plt.subplots(constrained_layout=True)[1]
        v = self.M.abs().max()
        ax.matshow(self.M.cpu().detach(), cmap="RdBu", vmin=-v, vmax=v)
        return ax

class CleanDataset(SparseDataset):
    """Dataset that generates inputs and labels = ReLU(inputs)."""

    def __init__(self, n_features: int, p: float, device: str = None, seed: int | None = None, exactly_one_active_feature: bool = False):
        super().__init__(n_features, p, device=device, seed=seed, exactly_one_active_feature=exactly_one_active_feature)

    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        return self.relu(inputs)

