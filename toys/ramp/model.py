import torch
import torch.nn as nn


class ChainMLP(nn.Module):
    """n_features -> d1 (ReLU) -> d2 (ReLU) -> n_features, no biases."""

    def __init__(self, n_features: int = 100, d1: int = 50, d2: int = 50):
        super().__init__()
        self.n_features, self.d1, self.d2 = n_features, d1, d2
        self.w1 = nn.Parameter(torch.empty(n_features, d1))
        self.w2 = nn.Parameter(torch.empty(d1, d2))
        self.w3 = nn.Parameter(torch.empty(d2, n_features))
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w)

    def forward(self, x):
        return torch.relu(torch.relu(x @ self.w1) @ self.w2) @ self.w3

    def config(self) -> dict:
        return {"n_features": self.n_features, "d1": self.d1, "d2": self.d2}


class SingleMLP(nn.Module):
    """n_features -> d (ReLU) -> n_features, no biases."""

    def __init__(self, n_features: int = 100, d: int = 50):
        super().__init__()
        self.n_features, self.d = n_features, d
        self.w1 = nn.Parameter(torch.empty(n_features, d))
        self.w2 = nn.Parameter(torch.empty(d, n_features))
        for w in (self.w1, self.w2):
            nn.init.kaiming_uniform_(w)

    def forward(self, x):
        return torch.relu(x @ self.w1) @ self.w2

    def config(self) -> dict:
        return {"n_features": self.n_features, "d": self.d}


MODELS = {"ChainMLP": ChainMLP, "SingleMLP": SingleMLP}


def load_checkpoint(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MODELS[ck["model_class"]](**ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
