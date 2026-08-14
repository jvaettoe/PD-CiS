import torch
import torch.nn as nn


class MLP(nn.Module):
    """n_features -> d_mlp (ReLU) -> n_features, no biases."""

    def __init__(self, n_features: int = 100, d_mlp: int = 50):
        super().__init__()
        self.n_features, self.d_mlp = n_features, d_mlp
        self.w_in = nn.Parameter(torch.empty(n_features, d_mlp))
        self.w_out = nn.Parameter(torch.empty(d_mlp, n_features))
        nn.init.kaiming_uniform_(self.w_in)
        nn.init.kaiming_uniform_(self.w_out)

    def forward(self, x):
        return torch.relu(x @ self.w_in) @ self.w_out

    def config(self) -> dict:
        return {"n_features": self.n_features, "d_mlp": self.d_mlp}
