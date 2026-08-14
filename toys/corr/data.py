import torch

N_FEATURES = 100
N_PAIRS = 50          # pairs are (2i, 2i+1)
P = 0.01


class CorrDataset:
    """100 sparse features in 50 disjoint pairs (2i, 2i+1). Per sample and pair:
    with prob rho the pair is 'linked' (one shared Bernoulli(p) occurrence draw),
    else two independent Bernoulli(p) draws — marginal stays p, within-pair
    occurrence correlation ~= rho. Values independent U[-1,1]; target
    y = ReLU(x) (the main-suite CleanDataset task)."""

    def __init__(self, rho: float, seed: int = 42):
        self.rho, self.p = rho, P
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, n: int):
        g = self.gen
        linked = torch.rand(n, N_PAIRS, generator=g) < self.rho
        shared = torch.rand(n, N_PAIRS, generator=g) < self.p
        ind = torch.rand(n, N_PAIRS, 2, generator=g) < self.p
        a = torch.where(linked.unsqueeze(-1),
                        shared.unsqueeze(-1).expand(-1, -1, 2), ind)
        active = a.reshape(n, N_FEATURES).float()
        x = active * (torch.rand(n, N_FEATURES, generator=g) * 2 - 1)
        return x, torch.relu(x)

    def p_any_active(self) -> float:
        pair_none = (self.rho * (1 - self.p)
                     + (1 - self.rho) * (1 - self.p) ** 2)
        return 1.0 - pair_none ** N_PAIRS
