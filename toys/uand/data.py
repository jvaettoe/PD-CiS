from itertools import combinations

import torch

N_IN = 15
PAIRS = list(combinations(range(N_IN), 2))     # all C(15,2) = 105 gates
N_OUT = len(PAIRS)
I_IDX = torch.tensor([i for i, _ in PAIRS])
J_IDX = torch.tensor([j for _, j in PAIRS])


class UANDDataset:
    """n_in iid Bernoulli(p) boolean inputs; targets = all pairwise ANDs."""

    def __init__(self, p: float, seed: int = 42):
        self.p = p
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, n: int):
        x = (torch.rand(n, N_IN, generator=self.gen) < self.p).float()
        y = x[:, I_IDX] * x[:, J_IDX]
        return x, y

    def p_any_gate(self) -> float:
        """P(at least one gate fires) = P(>= 2 inputs active)."""
        q = 1 - self.p
        return 1.0 - (q ** N_IN + N_IN * self.p * q ** (N_IN - 1))
