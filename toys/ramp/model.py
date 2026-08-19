import torch
import torch.nn as nn


def _bias_param(d: int, bias: bool):
    """Trainable bias (zeros init) if bias, else a constant-zero plain tensor
    (not registered, so bias-free state_dicts keep their original keys)."""
    return nn.Parameter(torch.zeros(d)) if bias else torch.zeros(d)


class ChainMLP(nn.Module):
    """n_features -> d1 (ReLU) -> d2 (ReLU) -> n_features.

    bias=True adds HIDDEN-layer biases (output layer stays bias-free) — a
    deliberate departure from the toys convention, as in toys/uand (where
    ReLU(x_i + x_j - 1) needs the -1): the ramp target's kink at theta = 0.5
    requires ReLU(x - theta)-type units, unrepresentable in a bias-free
    (positively homogeneous) net. bias=False is the homogeneity control."""

    def __init__(self, n_features: int = 100, d1: int = 50, d2: int = 50,
                 bias: bool = False):
        super().__init__()
        self.n_features, self.d1, self.d2 = n_features, d1, d2
        self.has_bias = bias
        self.w1 = nn.Parameter(torch.empty(n_features, d1))
        self.w2 = nn.Parameter(torch.empty(d1, d2))
        self.w3 = nn.Parameter(torch.empty(d2, n_features))
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w)
        self.b1 = _bias_param(d1, bias)
        self.b2 = _bias_param(d2, bias)

    def forward(self, x):
        h1 = torch.relu(x @ self.w1 + self.b1)
        return torch.relu(h1 @ self.w2 + self.b2) @ self.w3

    def config(self) -> dict:
        return {"n_features": self.n_features, "d1": self.d1, "d2": self.d2,
                "bias": self.has_bias}


class SingleMLP(nn.Module):
    """n_features -> d (ReLU) -> n_features. bias=True adds the hidden-layer
    bias (output layer bias-free) — same justification as ChainMLP."""

    def __init__(self, n_features: int = 100, d: int = 50,
                 bias: bool = False):
        super().__init__()
        self.n_features, self.d = n_features, d
        self.has_bias = bias
        self.w1 = nn.Parameter(torch.empty(n_features, d))
        self.w2 = nn.Parameter(torch.empty(d, n_features))
        for w in (self.w1, self.w2):
            nn.init.kaiming_uniform_(w)
        self.b1 = _bias_param(d, bias)

    def forward(self, x):
        return torch.relu(x @ self.w1 + self.b1) @ self.w2

    def config(self) -> dict:
        return {"n_features": self.n_features, "d": self.d,
                "bias": self.has_bias}


MODELS = {"ChainMLP": ChainMLP, "SingleMLP": SingleMLP}


def load_checkpoint(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    cfg.setdefault("bias", False)   # pre-bias checkpoints
    model = MODELS[ck["model_class"]](**cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
