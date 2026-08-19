import torch

N_FEATURES = 100
THETA = 0.5


def ramp(x):
    """Saturating ramp: 0 for x <= 0, x on (0, THETA), THETA for x >= THETA."""
    return torch.relu(x) - torch.relu(x - THETA)


class SparseDataset:
    """n_features iid Bernoulli(p) occurrences, values U[-1,1]; target
    y = ramp(x) = ReLU(x) - ReLU(x - 0.5) elementwise."""

    def __init__(self, p: float, seed: int = 42):
        self.p = p
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, n: int):
        active = (torch.rand(n, N_FEATURES, generator=self.gen) < self.p).float()
        x = active * (torch.rand(n, N_FEATURES, generator=self.gen) * 2 - 1)
        return x, ramp(x)

    def p_any_active(self) -> float:
        return 1.0 - (1 - self.p) ** N_FEATURES


if __name__ == "__main__":
    # 1) labels match the analytic piecewise form on a test batch
    ds = SparseDataset(0.05, seed=0)
    x, y = ds.generate_batch(8192)
    ref = torch.clamp(x, min=0.0, max=THETA)
    assert torch.allclose(y, ref, atol=1e-7), "labels != clamp(x, 0, theta)"
    assert torch.equal(y[x <= 0], torch.zeros_like(y[x <= 0]))
    # 2) active-conditional baseline E[f^2 | x > 0] = theta^3/3 + theta^2(1-theta)
    analytic = THETA ** 3 / 3 + THETA ** 2 * (1 - THETA)   # = 1/6
    v = torch.rand(2_000_000, generator=torch.Generator().manual_seed(1))
    mc = float((ramp(v) ** 2).mean())
    assert abs(mc - analytic) < 1e-3, (mc, analytic)
    print(f"labels OK; baseline analytic {analytic:.6f} (=1/6), "
          f"Monte Carlo {mc:.6f}")
