import torch

N_FEATURES = 100


class SparseDataset:
    """n_features iid Bernoulli(p) occurrences, values U[-1,1]; target
    y = ReLU(x) (the main-suite CleanDataset task)."""

    def __init__(self, p: float, seed: int = 42):
        self.p = p
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, n: int):
        active = (torch.rand(n, N_FEATURES, generator=self.gen) < self.p).float()
        x = active * (torch.rand(n, N_FEATURES, generator=self.gen) * 2 - 1)
        return x, torch.relu(x)

    def p_any_active(self) -> float:
        return 1.0 - (1 - self.p) ** N_FEATURES
