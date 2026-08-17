"""VPD -- Goodfire's adVersarial Parameter Decomposition, ported to this repo's PD framework.

Port of https://github.com/goodfire-ai/param-decomp (MIT License, Goodfire AI; cloned at
tag `vpd-paper`, local reference copy in ../Others' code/param-decomp). Core sources:
`param_decomp/models/components.py` (LinearComponents V/U + weight delta),
`param_decomp/models/sigmoids.py` (leaky-hard pair), `param_decomp/metrics/
{importance_minimality_loss,stochastic_recon_subset_loss,pgd_utils}.py`, and the TOY
recipe from `param_decomp/experiments/resid_mlp/resid_mlp2_config.yaml` (the config that
trains a toy residual MLP WITH the adversarial PGD loss):
    ImportanceMinimalityLoss coeff 1e-4 (pnorm 2.0 fixed, beta 0)
    StochasticReconSubsetLoss coeff 2.0 (continuous sampling, 1 mask sample)
    PGDReconSubsetLoss coeff 2.0 (init random, step_size 1.0, n_steps 1,
                                  mask_scope shared_across_batch)
    FaithfulnessLoss coeff 0.0 + a 200-step faithfulness warmup (lr 0.01)
    constant lr 3e-4, batch 2048, use_delta_component true, ci_alive_threshold 0.1

Documented adaptations for this repo's toy MLPs (MLP / EmbedMLP, raw nn.Parameter
weights, regression):
  - Reconstruction loss is MSE on model outputs (their TMS/resid-MLP setting), not KL on
    logits; no sequence dimension anywhere (masks are [B, C]).
  - Matrices kept in this repo's x @ W orientation: V [d0, C], U [C, d1], component
    reconstruction V @ U ~ W (their nn.Linear convention stores the transpose).
  - CI function: one shared MLP per matrix (their VectorSharedMLPCiFn / fn_type
    "shared_mlp", hidden scaled to [64] from their [256] for these small models) on the
    matrix's input activation; leaky-hard lower/upper sigmoid pair. The lower sigmoid
    reproduces their custom autograd Function (revival gradient: alpha*grad for x<=0
    when grad<0). NO lower jitter: their 1.05x - 0.05*rand applies only under binomial
    sampling, and the toy recipe runs continuous.
  - The faithfulness warmup is data-free (pure ||W - VU|| matching) and runs in __init__.
  - PGD adversarial masks: shared-across-batch source [1, C+1] per matrix (last slot =
    weight-delta mask), 1 signed-gradient step of size 1.0, clamped to [0,1], fresh per
    loss() call (their non-persistent PGDReconSubsetLoss, as used by the toy config --
    NOT the LM PersistentPGD), evaluated under a shared UniformK subset-routing draw.
  - Subset routing: per example, a uniform-k random subset of the two matrices runs in
    component mode; the rest use the frozen target weights (their UniformKSubsetRouter).

`masked_weight(name, None)` returns the COMPONENT-ONLY sum V @ U (no delta), so the
inherited faithfulness_loss/plot_faithfulness report how much function lives in the
delta spillover rather than in components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from core.pd import PD

CI_ALIVE_THRESHOLD = 0.1   # their ci_alive_threshold


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Their LowerLeakyHardSigmoidFunction: forward clamp(x,0,1); backward leaks
    alpha*grad for x<=0 ONLY when the gradient is negative (the loss wants the mask
    larger) -- the revival gradient that lets a dead component come back. A plain
    clamp (our previous approximation) zeroes that gradient and dead components
    stay dead forever."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.clamp(x, min=0.0, max=1.0)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        grad_input = torch.where(
            x <= 0,
            torch.where(grad_output < 0, ctx.alpha * grad_output,
                        torch.zeros_like(grad_output)),
            torch.where(x <= 1, grad_output, torch.zeros_like(grad_output)),
        )
        return grad_input, None


def _leaky_hard_lower(x: Tensor, alpha: float = 0.01) -> Tensor:
    return _LowerLeakyHardSigmoid.apply(x, alpha)


def _leaky_hard_upper(x: Tensor, alpha: float = 0.01) -> Tensor:
    """upper_leaky_hard_sigmoid: >=0 always, leaky above 1."""
    return torch.where(x > 1, 1 + alpha * (x - 1), torch.clamp(x, min=0.0, max=1.0))


class _CiFn(nn.Module):
    """VectorSharedMLPCiFn: input activation -> C pre-sigmoid CI values."""

    def __init__(self, d_in: int, C: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, C))
        # their init_param_: Kaiming-normal (fan = input dim, gain 1), zero bias
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, m.in_features ** -0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch C"]:
        return self.net(x)


class VPD(PD):
    """Goodfire VPD on a single-hidden-layer MLP (decomposes w_in and w_out)."""

    def __init__(self, model: nn.Module, C: int, ci_hidden: int = 64,
                 warmup_steps: int = 200, warmup_lr: float = 0.01,
                 param_names: list[str] | None = None) -> None:
        super().__init__(model, C, param_names)
        assert set(self.param_names) == {"w_in", "w_out"}, (
            f"VPD port assumes a single-hidden-layer MLP (w_in/w_out); got {self.param_names}")
        self.V = nn.ParameterDict()
        self.U = nn.ParameterDict()
        self.ci_fns = nn.ModuleDict()
        for name, t in self.targets.items():
            d0, d1 = t.shape
            key = self._key(name)
            # their init_param_: Kaiming-NORMAL, gain 1 ("linear"), explicit fan
            # (V: fan = d_in = d0; U: fan = C)
            V = torch.empty(d0, C, device=t.device).normal_(0.0, d0 ** -0.5)
            U = torch.empty(C, d1, device=t.device).normal_(0.0, C ** -0.5)
            self.V[key] = nn.Parameter(V)
            self.U[key] = nn.Parameter(U)
            self.ci_fns[key] = _CiFn(d0, C, ci_hidden).to(t.device)
        self._faithfulness_warmup(warmup_steps, warmup_lr)

    # ---- parameterization ----

    def _vu(self, name: str) -> Float[Tensor, "d0 d1"]:
        key = self._key(name)
        return self.V[key] @ self.U[key]

    def weight_delta(self, name: str) -> Float[Tensor, "d0 d1"]:
        """Spillover residual: target minus component sum (their weight_delta)."""
        return self.targets[name] - self._vu(name)

    def _component_weight(self, name: str, c: int) -> Float[Tensor, "d0 d1"]:
        key = self._key(name)
        return torch.outer(self.V[key][:, c], self.U[key][c, :])

    def masked_weight(self, name: str, mask: Float[Tensor, "C"] | None) -> Tensor:
        key = self._key(name)
        if mask is None:
            return self._vu(name)
        return (self.V[key] * mask) @ self.U[key]

    def _faithfulness_warmup(self, steps: int, lr: float) -> None:
        """Their faithfulness warmup: push V@U toward the target weights before the
        main fit. Data-free (pure weight matching)."""
        if steps <= 0:
            return
        opt = torch.optim.AdamW([p for k in self.V for p in (self.V[k], self.U[k])],
                                lr=lr, weight_decay=0.0)  # their warmup wd is 0.0
        for _ in range(steps):
            loss = sum(((self.targets[n] - self._vu(n)) ** 2).mean() for n in self.param_names)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---- CI ----

    def _ci(self, e: Tensor, h: Tensor) -> tuple[dict, dict]:
        """(ci_lower, ci_upper) per matrix. Inputs: e = embedded input (feeds w_in),
        h = hidden activation (feeds w_out). NB their 1.05x-0.05*rand jitter applies
        only under binomial sampling; the toy recipe (and we) run continuous, so the
        pre-sigmoid CI feeds the lower sigmoid unmodified."""
        lower, upper = {}, {}
        for name, act in (("w_in", e), ("w_out", h)):
            pre = self.ci_fns[self._key(name)](act)
            lower[name] = _leaky_hard_lower(pre)   # in [0,1]; leaky revival backward
            upper[name] = _leaky_hard_upper(pre)
        return lower, upper

    # ---- masked forward (mirrors APD._component_forward + their LinearComponents) ----

    def _forward_masked_vpd(
        self, x: Float[Tensor, "batch n_features"],
        masks: dict[str, Tensor], delta_masks: dict[str, Tensor],
        routing: dict[str, Tensor] | None,
    ) -> Float[Tensor, "batch n_features"]:
        """Component-mode forward with per-example component masks [B,C], per-example
        delta masks [B], and per-example routing bools [B] (True = component mode,
        False = frozen target weights for that matrix)."""
        e = self.model.embed_input(x)
        outs = {}
        for name, act in (("w_in", e),):
            key = self._key(name)
            comp = ((act @ self.V[key]) * masks[name]) @ self.U[key] \
                + delta_masks[name].unsqueeze(-1) * (act @ self.weight_delta(name))
            tgt = act @ self.targets[name]
            r = routing[name].unsqueeze(-1) if routing is not None else torch.ones_like(comp[:, :1])
            outs[name] = torch.where(r.bool(), comp, tgt)
        h = F.relu(outs["w_in"])
        key = self._key("w_out")
        comp = ((h @ self.V[key]) * masks["w_out"]) @ self.U[key] \
            + delta_masks["w_out"].unsqueeze(-1) * (h @ self.weight_delta("w_out"))
        tgt = h @ self.targets["w_out"]
        r = routing["w_out"].unsqueeze(-1) if routing is not None else torch.ones_like(comp[:, :1])
        m = torch.where(r.bool(), comp, tgt)
        return self.model.unembed_output(m)

    @staticmethod
    def _uniform_k_subset_routing(B: int, device) -> dict[str, Tensor]:
        """UniformKSubsetRouter over the 2 matrices: per example draw k ~ U{1,2}, then a
        random k-subset runs in component mode."""
        k = torch.randint(1, 3, (B,), device=device)
        first = torch.randint(0, 2, (B,), device=device)     # which matrix if k == 1
        r_in = (k == 2) | (first == 0)
        r_out = (k == 2) | (first == 1)
        return {"w_in": r_in.float(), "w_out": r_out.float()}

    # ---- losses (their toy recipe) ----

    def method_loss_terms(
        self, x: Float[Tensor, "batch n_features"],
        target_output: Float[Tensor, "batch n_features"],
        alpha_importance: float = 1e-4, alpha_stoch: float = 2.0, alpha_pgd: float = 2.0,
        pnorm: float = 2.0, imp_eps: float = 1e-12,
        pgd_steps: int = 1, pgd_step_size: float = 1.0,
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        B = x.shape[0]
        device = x.device
        with torch.no_grad():
            e = self.model.embed_input(x)
            h = F.relu(e @ self.targets["w_in"])
        ci_lower, ci_upper = self._ci(e, h)

        # importance minimality: mean over batch of (ci_upper + eps)^p, summed over C
        imp = sum(((u + imp_eps) ** pnorm).mean(dim=0).sum() for u in ci_upper.values())

        # stochastic recon (subset-routed, continuous sampling): mask = ci + (1-ci)*U(0,1)
        masks = {n: ci + (1 - ci) * torch.rand_like(ci) for n, ci in ci_lower.items()}
        dmasks = {n: torch.rand(B, device=device) for n in ci_lower}
        routing = self._uniform_k_subset_routing(B, device)
        stoch = F.mse_loss(self._forward_masked_vpd(x, masks, dmasks, routing), target_output)

        # PGD adversarial recon: shared-across-batch source [1, C+1], signed steps.
        # Their PGDReconSubsetLoss evaluates the adversary UNDER subset routing --
        # one routing draw shared by the inner ascent steps and the final loss.
        pgd_routing = self._uniform_k_subset_routing(B, device)
        adv = {n: torch.rand(1, self.C + 1, device=device, requires_grad=True)
               for n in ci_lower}
        for _ in range(pgd_steps):
            am = {n: ci_lower[n].detach() + (1 - ci_lower[n].detach()) * a[:, :self.C]
                  for n, a in adv.items()}
            dm = {n: a[:, self.C].expand(B) for n, a in adv.items()}
            pl = F.mse_loss(self._forward_masked_vpd(x, am, dm, pgd_routing), target_output)
            grads = torch.autograd.grad(pl, list(adv.values()))
            with torch.no_grad():
                for a, g in zip(adv.values(), grads):
                    a.add_(pgd_step_size * g.sign())
                    a.clamp_(0.0, 1.0)
        am = {n: ci_lower[n] + (1 - ci_lower[n]) * adv[n][:, :self.C].detach()
              for n in ci_lower}
        dm = {n: adv[n][:, self.C].detach().expand(B) for n in ci_lower}
        pgd = F.mse_loss(self._forward_masked_vpd(x, am, dm, pgd_routing), target_output)

        return {"importance": (imp, alpha_importance),
                "stoch_recon": (stoch, alpha_stoch),
                "pgd_recon": (pgd, alpha_pgd)}

    # ---- VPD's own counts (for the comparison suite) ----

    def ci_stats(self, p: float, *, batch: int = 4096, seed: int = 0,
                 threshold: float = CI_ALIVE_THRESHOLD) -> dict:
        """VPD-native component counts on a fresh batch at sparsity p:
          ci_l0        -- mean #components with ci > threshold per example (per-input
                          active count, ~ their CI_L0 metric; compare to k = p*F)
          n_alive_union-- #components alive on >= 1 example (model-level capacity count)
          pr_mean_ci   -- participation ratio of per-component mean ci (dominance-robust
                          effective number of components)
        Each is reported per matrix and max-ed over matrices for the headline number."""
        from core.data import CleanDataset
        ds = CleanDataset(n_features=self.model.n_features, p=float(p),
                          device=self.model.device, seed=seed)
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            e = self.model.embed_input(x)
            h = F.relu(e @ self.targets["w_in"])
            lower, _ = self._ci(e, h)
            out = {}
            for name, ci in lower.items():
                alive = ci > threshold
                mean_ci = ci.mean(dim=0)
                pr = float((mean_ci.sum() ** 2 / (mean_ci.pow(2).sum() + 1e-12)).item())
                out[name] = {"ci_l0": float(alive.float().sum(-1).mean().item()),
                             "n_alive_union": int(alive.any(dim=0).sum().item()),
                             "pr_mean_ci": pr}
            out["headline"] = {k: max(out[n][k] for n in self.param_names)
                               for k in ("ci_l0", "n_alive_union", "pr_mean_ci")}
        return out
