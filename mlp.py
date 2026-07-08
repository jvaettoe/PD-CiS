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

from pd import PD


class MLP(nn.Module, ABC):
    "Base class for simple single-layer MLP modules, no biases"
    def __init__(self, d_in: int, d_mlp: int, d_out: int, device: str | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_mlp = d_mlp
        self.d_out = d_out
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.w_in: Float[Tensor, "d_in d_mlp"] = nn.Parameter(torch.empty(d_in, d_mlp, device=self.device))
        self.w_out: Float[Tensor, "d_mlp d_out"] = nn.Parameter(torch.empty(d_mlp, d_out, device=self.device))
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.w_in)
        nn.init.kaiming_uniform_(self.w_out)

    @abstractmethod
    def forward(self, x: Float[Tensor, "batch n_features"], pd: PD | None, mode: str = "err"
                ) -> Float[Tensor, "batch n_features"]:
        ...

    @abstractmethod
    def config(self) -> dict:
        """Constructor kwargs (excluding device) needed to rebuild this model.
        Consumed by cc.save_checkpoint so any MLP subclass round-trips generically.
        Abstract because the base has no n_features -- each subclass declares its own."""
        ...

    # Feature-space projection hooks. The effective read-in/read-out weights that map
    # the n_features input/output space through the hidden layer. Base defaults are
    # correct whenever d_in == d_out == n_features (LinearMLP); EmbedMLP overrides them
    # to fold in its embedding w_e so the weight-space plots stay in feature space.
    def readin(self) -> Float[Tensor, "n_features d_mlp"]:
        return self.w_in

    def readout(self) -> Float[Tensor, "d_mlp n_features"]:
        return self.w_out

    # Visualisation suite; common for all instances of MLP modules
    def plot_weights(self, axes: list[plt.Axes] | None = None, share_v: bool = True,
                     project: bool = False) -> plt.Figure:
        """Plot the weights of the MLP. By default shows the raw learned parameters
        (for EmbedMLP these genuinely live in d_emb space -- the low-rank bottleneck).
        project=True instead shows the feature-space read-in/read-out (readin()/readout()),
        which for EmbedMLP folds in the embedding w_e."""
        if axes is None:
            fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
        else:
            assert len(axes) == 2, "Must provide exactly 2 axes"
            fig = axes[0].get_figure()
        cmap = mpl.colormaps["RdBu"]

        w_in = self.readin() if project else self.w_in
        w_out = self.readout() if project else self.w_out
        suffix = " (feature-space)" if project else ""
        axes[0].set_title(("read-in" if project else "W_in") + suffix)
        axes[1].set_title(("read-out.T" if project else "W_out.T") + suffix)

        absmax = w_in.detach().abs().max()
        im0 = axes[0].imshow(w_in.detach().cpu().numpy(), cmap=cmap, vmin=-absmax, vmax=absmax)
        if not share_v:
            absmax = w_out.detach().abs().max()
        im1 = axes[1].imshow(w_out.T.detach().cpu().numpy(), cmap=cmap, vmin=-absmax, vmax=absmax)

        if not share_v:
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        return fig

    def effective_matrix(self) -> Float[Tensor, "n_features n_features"]:
        """Computes the linearised effective feature to feature mapping. Routed through
        readin()/readout() so it is feature-space for every subclass (EmbedMLP:
        w_e @ w_in @ w_out @ w_e.T rather than the raw d_emb-space w_in @ w_out)."""
        return (self.readin() @ self.readout()).detach()

    def empirical_matrix(self, value: float = 1.0) -> Float[Tensor, "n_features n_features"]:
        """Empirical (post-ReLU) response matrix."""
        with torch.no_grad():
            X = value * torch.eye(self.n_features, device=self.device)
            return self(X).detach()
    
    def plot_input_output_behaviour(self, ax: plt.Axes | None = None, palette: str = "viridis") -> plt.Figure:
        """Plot the input-output behaviour of the MLP."""
        fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
        cmap = mpl.colormaps[palette]
        for i in range(self.n_features):
            test_input = torch.zeros(1024, self.n_features, device=self.device)
            test_input[:, i] = torch.linspace(-1, 1, 1024, device=self.device)
            test_output = self(test_input).detach()
            c = cmap(i / self.n_features)
            ax.plot(test_input[:, i].cpu(), test_output[:, i].cpu(), color=c)
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=self.n_features), cmap=cmap), ax=ax)
        cbar.set_label("Feature index")
        fig.suptitle("Input-output behaviour for individual features")
        return fig

    # def plot_weight_bars(self, ax: plt.Axes | None = None, palette: str = "inferno") -> plt.Figure:
    #     """Per-feature stacked bars of the neuron contributions W[d, n] = w_in[n, d] *
    #     w_out[d, n] (each neuron's share of feature n's self-computation M_eff[n, n]).
    #     One dominant bar => a dedicated neuron; several bars summing to ~1 =>
    #     superposition across shared neurons; ~0 total => dropped feature. Ported from
    #     mlpinsoup.plot_weight_bars (seaborn palette replaced by a matplotlib cmap)."""
    #     from matplotlib.cm import ScalarMappable
    #     from matplotlib.colors import Normalize
    #     W = einops.einsum(self.w_in, self.w_out,
    #                       "n_features d_mlp, d_mlp n_features -> d_mlp n_features")
    #     W = W.cpu().detach().numpy()
    #     d_mlp, n_features = W.shape
    #     x = np.arange(n_features)
    #     cmap = plt.cm.get_cmap(palette)
    #     colors = [cmap(i / max(1, d_mlp - 1)) for i in range(d_mlp)]
    #     fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
    #     bottom_pos = np.zeros(n_features)
    #     bottom_neg = np.zeros(n_features)
    #     for i in range(d_mlp):
    #         mask_pos = W[i] >= 0
    #         mask_neg = W[i] < 0
    #         if np.any(mask_pos):
    #             ax.bar(x[mask_pos], W[i][mask_pos], bottom=bottom_pos[mask_pos], color=colors[i])
    #             bottom_pos[mask_pos] += W[i][mask_pos]
    #         if np.any(mask_neg):
    #             ax.bar(x[mask_neg], W[i][mask_neg], bottom=bottom_neg[mask_neg], color=colors[i])
    #             bottom_neg[mask_neg] += W[i][mask_neg]
    #     sm = ScalarMappable(norm=Normalize(vmin=0, vmax=d_mlp), cmap=cmap)
    #     sm.set_array([])
    #     fig.colorbar(sm, ax=ax).set_label("MLP neuron index")
    #     ax.set_xlim(-0.5, n_features - 0.5)
    #     ax.set_xlabel("feature index")
    #     ax.set_ylabel(r"neuron contribution $w_{in}\odot w_{out}$")
    #     ax.set_title("Feature -> neuron assignment")
    #     return fig
    
    def plot_weight_bars(self, bar_label: str = "MLP neuron index", show_cbar: bool = True, ax: plt.Axes | None = None, palette: str = "inferno") -> plt.Axes:
        """Plots weights for each feature / hidden neuron pair as stacked bars. Uses the
        feature-space read-in/read-out (readin()/readout()) so each bar is a neuron's share
        of feature n's self-computation for any subclass, EmbedMLP included."""
        W = einops.einsum(self.readin(), self.readout(), "n_features d_mlp, d_mlp n_features -> d_mlp n_features")
        W = W.cpu().detach().numpy()
        d_mlp, n_features = W.shape
        x = np.arange(n_features)
        cmap = mpl.colormaps[palette]
        colors = [cmap(i / max(1, d_mlp - 1)) for i in range(d_mlp)]
        fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
        bottom_pos = np.zeros(n_features)
        bottom_neg = np.zeros(n_features)
        for i in range(d_mlp):
            mask_pos = W[i] >= 0
            mask_neg = W[i] < 0
            if np.any(mask_pos):
                ax.bar(x[mask_pos], W[i][mask_pos], bottom=bottom_pos[mask_pos], label=f"{bar_label} {i}", color=colors[i])
                bottom_pos[mask_pos] += W[i][mask_pos]
            if np.any(mask_neg):
                ax.bar(x[mask_neg], W[i][mask_neg], bottom=bottom_neg[mask_neg], label=f"{bar_label} {i}", color=colors[i])
                bottom_neg[mask_neg] += W[i][mask_neg]
        if show_cbar:
            sm = ScalarMappable(norm=Normalize(vmin=1, vmax=d_mlp), cmap=mpl.colormaps[palette])
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax) 
            cbar.set_label(bar_label)
        ax.set_xlim(-0.5, n_features - 0.5)
        return ax
    

# class LinearMLP(MLP):
#     """simple single-layer MLP module, no biases"""
#     def __init__(self, n_features: int, d_mlp: int, device: str = None):
#         super().__init__(n_features, d_mlp, device)
        
#     def forward(self, x: Float[Tensor, "batch n_features"], pd: PD | None, mode: str = "err"
#                 ) -> Float[Tensor, "batch n_features"]:
#         if pd is None:
#             return F.relu(x @ self.w_in) @ self.w_out
#         else:
#             h = F.relu(pd.decomp_op("w_in", x, mode = mode))
#             return pd.decomp_op("w_out", h, mode = mode)
#         #mid_linear = einops.einsum(x, self.w_in, "batch n_features, n_features d_mlp -> batch d_mlp")
#         #mid = self.relu(mid_linear)
#         #out = einops.einsum(mid, self.w_out, "batch d_mlp, d_mlp n_features -> batch n_features")
#         #return out
    
    
# class EmbedMLP(MLP):
#     """simple single-layer embedded MLP module, no biases"""
#     def __init__(self, n_features: int, d_emb: int, d_mlp: int, device: str = None):
#         super().__init__(n_features, d_mlp, device)
#         self.d_emb = d_emb

#         self.w_e: Float[Tensor, "n_features d_emb"] = torch.randn(n_features, d_emb, device = self.device)
#         w_e = w_e / w_e.norm(dim=1, keepdim = True)
#         self.register_buffer("w_e", w_e)

#     def forward(self, x: Float[Tensor, "batch n_features"], pd: PD | None, mode: str = "err"
#                 ) -> Float[Tensor, "batch n_features"]:
#         e = x @ self.w_e
#         if pd is None:
#             return F.relu(x @ self.w_in) @ self.w_out
#         else:
#             h = F.relu(pd.decomp_op("w_in", x, mode = mode))
#             return pd.decomp_op("w_out", h, mode = mode)

#     #def forward(self, x: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
#     #    mid_linear = einops.einsum(x, self.w_in, "batch n_features, n_features d_mlp -> batch d_mlp")
#     #    mid = self.relu(mid_linear)
#     #    out = einops.einsum(mid, self.w_out, "batch d_mlp, d_mlp n_features -> batch n_features")
#     #    return out

class LinearMLP(MLP):
    """simple single-layer MLP module, no biases."""
 
    def __init__(self, n_features: int, d_mlp: int, device: str | None = None):
        super().__init__(d_in=n_features, d_mlp=d_mlp, d_out=n_features, device=device)
        self.n_features = n_features

    def config(self) -> dict:
        return {"n_features": self.n_features, "d_mlp": self.d_mlp}

    def forward(self, x: Float[Tensor, "batch n_features"], pd=None, mode: str = "real"
                ) -> Float[Tensor, "batch n_features"]:
        if pd is None:
            return F.relu(x @ self.w_in) @ self.w_out
        h = F.relu(pd.decomp_op("w_in", x, mode=mode))
        return pd.decomp_op("w_out", h, mode=mode)
 
 
class EmbedMLP(MLP):
    """simple single-layer embedded MLP module, no biases."""
 
    def __init__(self, n_features: int, d_emb: int, d_mlp: int, device: str | None = None):
        super().__init__(d_in=d_emb, d_mlp=d_mlp, d_out=d_emb, device=device)
        self.n_features, self.d_emb = n_features, d_emb
        w_e = torch.randn(n_features, d_emb, device=self.device)
        self.register_buffer("w_e", w_e / w_e.norm(dim=1, keepdim=True))
        self.w_e: Float[Tensor, "n_features d_emb"]

    def config(self) -> dict:
        return {"n_features": self.n_features, "d_emb": self.d_emb, "d_mlp": self.d_mlp}

    def readin(self) -> Float[Tensor, "n_features d_mlp"]:
        return self.w_e @ self.w_in          # (n_features,d_emb) @ (d_emb,d_mlp)

    def readout(self) -> Float[Tensor, "d_mlp n_features"]:
        return self.w_out @ self.w_e.T       # (d_mlp,d_emb) @ (d_emb,n_features)

    def forward(self, x: Float[Tensor, "batch n_features"], pd=None, mode: str = "real"
                ) -> Float[Tensor, "batch n_features"]:
        e = x @ self.w_e
        if pd is None:
            m = F.relu(e @ self.w_in) @ self.w_out
        else:
            h = F.relu(pd.decomp_op("w_in", e, mode=mode))
            m = pd.decomp_op("w_out", h, mode=mode)
        return m @ self.w_e.T


