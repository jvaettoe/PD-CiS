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

class PD(nn.Module, ABC):
    """Base class for paramter decomposition methods (APD, SPD, VPD)"""
    def __init__(self, model: nn.Module, C: int):
        super.__init__()
        self.model = model
        self.device = model.device
        self.C = C

        self.param_names = ["w_in", "w_out"]
        self.comps = nn.ParameterDict({
            name: nn.Parameter(torch.randn(C, *getattr(model, name).shape, device = self.device) * 0.01)
            for name in self.param_names
        })
        self.relu = nn.ReLU()

    def weight_sum(self, name: str):
        return self.comps[name].sum(dim=0)
    
    def faith_loss(self):
        losses = [
            F.mse_loss(self.weight_sum(name), getattr(self.model, name).detach())
            for name in self.param_names
        ]
        return torch.stack(losses)

    def forward_full(self, x: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        params = {name: self.weight_sum(name) for name in self.param_names}
    
    def forward_mask()


class APD(PD):
    """Class implementing APD (Braun et al., 2025)"""
    def __init__(self):
        super.__init__(model, C)

        