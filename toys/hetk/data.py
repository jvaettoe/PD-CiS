import torch

N_FEATURES = 100
N_RARE = 50          # features 0..49 are the rare population
P_RARE = 0.001
P_COMMON = 0.05


class HetKDataset:
    """Sparse features in two sparsity populations: features 0..n_rare-1 active
    with p_rare, the rest with p_common; values U[-1,1]; target y = ReLU(x)
    (the main-suite CleanDataset task: the model must compute 100 ReLUs)."""

    def __init__(self, seed: int = 42, p_rare: float = P_RARE,
                 p_common: float = P_COMMON, n_rare: int = N_RARE):
        self.p_rare, self.p_common, self.n_rare = p_rare, p_common, n_rare
        self.p = torch.cat([torch.full((n_rare,), p_rare),
                            torch.full((N_FEATURES - n_rare,), p_common)])
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, n: int):
        active = (torch.rand(n, N_FEATURES, generator=self.gen) < self.p).float()
        x = active * (torch.rand(n, N_FEATURES, generator=self.gen) * 2 - 1)
        return x, torch.relu(x)

    def p_any_active(self) -> float:
        return 1.0 - float(torch.prod(1 - self.p))
