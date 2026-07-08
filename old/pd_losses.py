import math
import os
import time
import torch
import torch.nn as nn
from collections import defaultdict
from torch import Tensor
import einops
from abc import abstractmethod
import numpy as np
import random

from jaxtyping import Float
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

def faith_loss(comps: dict[str, LinearComponent | EmbeddingComponent], model: nn.Module, 
               ) -> Float[Tensor, ""]:
    
    model_params = defaultdict(str, Float[Tensor, "d_in d_out"])
    comp_params = defaultdict(str, Float[Tensor, "d_in d_out"])

    for comp_name, comp in comps.items():
        comp_params[comp_name] = comp.weight

    f_loss = 0

    return f_loss