import torch
import torch.nn as nn


class UAND(nn.Module):
    """n_in booleans -> d_mlp (ReLU, WITH hidden bias: ReLU(x_i + x_j - 1) needs
    the -1) -> n_out gate outputs (linear, no bias)."""

    def __init__(self, n_in: int = 15, d_mlp: int = 50, n_out: int = 105):
        super().__init__()
        self.n_in, self.d_mlp, self.n_out = n_in, d_mlp, n_out
        self.hidden = nn.Linear(n_in, d_mlp, bias=True)
        self.w_out = nn.Parameter(torch.empty(d_mlp, n_out))
        nn.init.kaiming_uniform_(self.w_out)

    def forward(self, x):
        return torch.relu(self.hidden(x)) @ self.w_out

    def config(self) -> dict:
        return {"n_in": self.n_in, "d_mlp": self.d_mlp, "n_out": self.n_out}
