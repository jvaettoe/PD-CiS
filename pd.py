from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from jaxtyping import Float, Int


class PD(nn.Module, ABC):
    """
    Base class for PD methods (APD, SPD, ...).
    Subclasses to implement:
        - `masked_weight(name, mask)` 
        - `method_loss_terms(...)`
    """
    init_scale: float = 0.01  # std of freshly initialized components

    def __init__(self, model: nn.Module, C: int, param_names: list[str] | None = None) -> None:
        super().__init__()
        self.model = model.requires_grad_(False) 
        self.C = C
        self.param_names = param_names or [
            name for name, p in model.named_parameters() if p.dim() >= 2
        ]
        all_params = dict(model.named_parameters())
        self.targets: dict[str, Tensor] = {name: all_params[name] for name in self.param_names}
        self.n_weight_params = sum(t.numel() for t in self.targets.values())

    @staticmethod
    def _key(name: str) -> str:
        """ParameterDict keys can't contain '.', so dotted paths are escaped."""
        return name.replace(".", "__")

    @abstractmethod
    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        """Combine only the masked-in components into a full weight matrix;
        mask=None means all components (APD: indexed sum of full-rank
        components; SPD: (U * mask) @ V)."""
        ...

    def reconstructed_weight(self, name: str) -> Float[Tensor, "dim0 dim1"]:
        """Full reconstruction == masked_weight with every component active."""
        return self.masked_weight(name, None)

    def forward_masked(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        mask: dict[str, Float[Tensor, "C"]],
    ) -> Float[Tensor, "batch ..."]:
        """Model's own forward with each param rebuilt from only its masked-in
        components. Batch-shared masks only (one mask for the whole batch):
        APD's training masks and any method's post-hoc component ablation.
        SPD's per-example stochastic training masks cannot use this route --
        they are applied inline via decomp_op."""
        params = {name: self.masked_weight(name, mask[name]) for name in self.param_names}
        return functional_call(self.model, params, (x,))

    @abstractmethod
    def method_loss_terms(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."], **kwargs
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        """Method-specific loss terms as {name: (term, weight)}. Faithfulness is
        added by the base class; subclasses supply everything else (APD:
        minimality + simplicity; SPD: recon + layerwise recon + importance)."""
        ...

    def loss(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
        alpha_faithfulness: float = 1.0, **kwargs
    ) -> tuple[Float[Tensor, ""], dict[str, float]]:
        """Template shared by all PD methods: faithfulness + method terms.
        Method-specific hyperparameters (APD's k, SPD's pnorm, all alphas)
        pass through **kwargs to method_loss_terms."""
        terms = {"faithful": (self.faithfulness_loss(), alpha_faithfulness)}
        terms |= self.method_loss_terms(x, target_output, **kwargs)
        return self._total_and_logs(terms)

    def faithfulness_loss(self) -> Float[Tensor, ""]:
        """Global MSE over all decomposed weights: sum of squared errors across
        ALL matrices / total weight-parameter count (spd-paper _calc_tensors_mse).
        One global mean, not a sum of per-matrix means -- invariant to how many
        matrices are decomposed and to their relative sizes."""
        sq_err = torch.stack([
            ((self.reconstructed_weight(name) - t) ** 2).sum() for name, t in self.targets.items()
        ]).sum()
        return sq_err / self.n_weight_params

    def forward_dense(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."]
    ) -> Float[Tensor, "batch ..."]:
        """Faithful forward: the model's own forward() with every decomposed
        parameter replaced by its full reconstruction. Works for any subclass;
        needs no pd-integration in the model."""
        params = {name: self.reconstructed_weight(name) for name in self.param_names}
        return functional_call(self.model, params, (x,))

    @staticmethod
    def _total_and_logs(
        weighted: dict[str, tuple[Float[Tensor, ""], float]]
    ) -> tuple[Float[Tensor, ""], dict[str, float]]:
        """Assemble total = sum(alpha_i * l_i) and a logs dict from
        {name: (loss_term, weight)}. Keeps log keys consistent across methods."""
        total = sum(w * l for l, w in weighted.values())
        logs = {k: l.item() for k, (l, _) in weighted.items()} | {"total": total.item()}
        return total, logs

    def decomp_op(
        self, name: str,
        x: Float[Tensor, "batch dim0"] | Int[Tensor, "batch"],
        mode: str = "masked",
    ) -> Float[Tensor, "batch dim1"]:
        """
        Inline per-operation interception point, called by the target model's
        forward() when pd is passed. Only methods with per-example masking
        (SPD) need this; APD intercepts at the parameter level via
        functional_call and never uses it. Overridden by SPD.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support decomp_op -- it substitutes whole "
            f"parameter tensors via functional_call instead. Call "
            f"{type(self).__name__.lower()}.forward_dense(x) / .forward_masked(x, mask) "
            f"directly rather than model(x, pd=...)."
        )


# ------------------------
# APD (Braun et al., 2025)
# ------------------------

class APD(PD):
    def __init__(self, model: nn.Module, C: int, param_names: list[str] | None = None) -> None:
        super().__init__(model, C, param_names)
        self.components = nn.ParameterDict({
            self._key(name): nn.Parameter(torch.randn(C, *t.shape, device=t.device) * self.init_scale)
            for name, t in self.targets.items()
        })

    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        comps = self.components[self._key(name)]
        if mask is None:
            return comps.sum(dim=0)  # theta_hat = sum_c P_c
        # Index only the k active components: k/C of the work of multiplying
        # all C by a mostly-zero mask, with identical gradients.
        return comps[mask.bool()].sum(dim=0)

    def simplicity_loss(self, mask: dict[str, Float[Tensor, "C"]], p: float = 1.0) -> Float[Tensor, ""]:
        terms = [
            (torch.linalg.svdvals(active).abs() ** p).sum()
            for name, m in mask.items()
            if (active := self.components[self._key(name)][m.bool()]).numel()
        ]
        return torch.stack(terms).sum() if terms else torch.zeros((), device=next(self.parameters()).device)

    def topk_mask(
        self, importance: dict[str, Float[Tensor, "C"]], k: int
    ) -> dict[str, Float[Tensor, "C"]]:
        # One shared mask: a 'component' is one index c across ALL params.
        combined = sum(importance.values())
        m = torch.zeros_like(combined)
        m[torch.topk(combined, k).indices] = 1.0
        return {name: m for name in self.param_names}

    def compute_importance(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
    ) -> dict[str, Float[Tensor, "C"]]:
        loss = F.mse_loss(self.forward_dense(x), target_output)
        grads = torch.autograd.grad(loss, [self.components[self._key(n)] for n in self.param_names])
        return {
            name: (grad * self.components[self._key(name)]).sum(dim=tuple(range(1, grad.dim()))).abs().detach()
            for name, grad in zip(self.param_names, grads)
        }

    def method_loss_terms(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
        k: int, alpha_minimality: float = 1.0, alpha_simplicity: float = 1.0, p: float = 1.0,
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        mask = self.topk_mask(self.compute_importance(x, target_output), k)
        return {
            "minimal": (F.mse_loss(self.forward_masked(x, mask), target_output), alpha_minimality),
            "simple": (self.simplicity_loss(mask, p=p), alpha_simplicity),
        }


class OracleAPD(APD):
    """APD with oracle (exact-ablation) attributions"""

    def compute_importance(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
    ) -> dict[str, Float[Tensor, "C"]]:
        with torch.no_grad():
            base = F.mse_loss(self.forward_dense(x), target_output)
            ones = torch.ones(self.C, device=base.device)
            scores = torch.zeros(self.C, device=base.device)
            for c in range(self.C):
                mask = ones.clone()
                mask[c] = 0.0
                out = self.forward_masked(x, {name: mask for name in self.param_names})
                scores[c] = (F.mse_loss(out, target_output) - base).abs()
        return {name: scores for name in self.param_names}


# -----------------------------------
# SPD (Bushnaq, Braun, Sharkey, 2025)
# -----------------------------------

def leaky_hard_sigmoid_lower(x: Float[Tensor, "..."], slope: float = 0.01) -> Float[Tensor, "..."]:
    return torch.where(x < 0, slope * x, x.clamp(max=1.0))


def leaky_hard_sigmoid_upper(x: Float[Tensor, "..."], slope: float = 0.01) -> Float[Tensor, "..."]:
    return torch.where(x > 1, 1 + slope * (x - 1), x.clamp(min=0.0))


def _init_param_(param: Tensor, fan_val: float, nonlinearity: str = "linear") -> Tensor:
    gain = nn.init.calculate_gain(nonlinearity)
    with torch.no_grad():
        param.normal_(0.0, gain * fan_val ** -0.5)
    return param


class AffineGate(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(_init_param_(torch.empty(C), fan_val=1))
        self.bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        return x * self.weight + self.bias


class PerComponentGateMLP(nn.Module):
    def __init__(self, C: int, n_neurons: int) -> None:
        super().__init__()
        self.mlp_in = nn.Parameter(_init_param_(torch.empty(C, n_neurons), fan_val=1, nonlinearity="relu"))
        self.in_bias = nn.Parameter(torch.zeros(C, n_neurons))
        self.mlp_out = nn.Parameter(_init_param_(torch.empty(C, n_neurons), fan_val=n_neurons))
        self.out_bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        hidden = F.gelu(x.unsqueeze(-1) * self.mlp_in + self.in_bias)      # ... C n
        return (hidden * self.mlp_out).sum(dim=-1) + self.out_bias          # ... C


class SPD(PD):
    """
    Stochastic Parameter Decomposition (Bushnaq, Braun, Sharkey, 2025),
    cross-checked against github.com/goodfire-ai/spd.
    """

    def __init__(self, model: nn.Module, C: int, param_specs: dict[str, str],
                 gate_type: str = "per_component_mlp", gate_hidden: int = 16) -> None:
        super().__init__(model, C, param_names=list(param_specs))
        self.param_specs = param_specs

        self.U, self.V, self.gate = nn.ParameterDict(), nn.ParameterDict(), nn.ModuleDict()
        for name, t in self.targets.items():
            key = self._key(name)
            self.U[key] = nn.Parameter(
                _init_param_(torch.empty(t.shape[0], C, device=t.device), fan_val=t.shape[1]))
            self.V[key] = nn.Parameter(
                _init_param_(torch.empty(C, t.shape[1], device=t.device), fan_val=C))
            gate = (PerComponentGateMLP(C, gate_hidden) if gate_type == "per_component_mlp"
                    else AffineGate(C) if gate_type == "affine"
                    else None)
            if gate is None:
                raise ValueError(f"Unknown gate_type '{gate_type}'")
            self.gate[key] = gate.to(t.device)
        self._logits: dict[str, Float[Tensor, "batch C"]] = {}
        self._active: set[str] | None = None

    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        """W^l ~= sum_c m_c U_c V_c = (U * m) @ V. Not used in SPD training
        (whose masks are per-example, applied inline in decomp_op) -- this
        serves faithfulness and batch-shared post-hoc component ablation."""
        key = self._key(name)
        U = self.U[key] if mask is None else self.U[key] * mask
        return U @ self.V[key]

    def decomp_op(
        self, name: str,
        x: Float[Tensor, "batch dim0"] | Int[Tensor, "batch"],
        mode: str = "masked",
    ) -> Float[Tensor, "batch dim1"]:
        key = self._key(name)
        op = self.param_specs[name]

        if mode == "masked" and self._active is not None and name not in self._active:
            t = self.targets[name]
            return x @ t if op == "matmul" else t[x]

        U, V = self.U[key], self.V[key]
        if op == "matmul":
            inner = x @ U
        elif op == "embedding":
            inner = U[x]
        else:
            raise ValueError(f"Unknown op '{op}' for parameter '{name}'")

        if mode == "dense":
            return inner @ V
        elif mode == "masked":
            logit = self.gate[key](inner)
            self._logits[name] = logit
            g = leaky_hard_sigmoid_lower(logit)
            mask = g + (1 - g) * torch.rand_like(g)
            return (inner * mask) @ V
        raise ValueError(
            f"decomp_op got unexpected mode '{mode}' -- did you mean 'dense' or 'masked'? "
            f"('err' should never reach decomp_op -- call model(x) without pd instead.)"
        )

    def _masked_forward(self, x, active: set[str] | None) -> Float[Tensor, "batch ..."]:
        self._active = active
        try:
            return self.model(x, pd=self, mode="masked")
        finally:
            self._active = None

    def method_loss_terms(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
        alpha_recon: float = 1.0,
        alpha_recon_layerwise: float = 1.0,
        alpha_importance: float = 1e-5,
        pnorm: float = 2.0,
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        self._logits = {}
        out = self._masked_forward(x, active=None)
        l_recon = F.mse_loss(out, target_output)
        l_importance = torch.stack([
            (leaky_hard_sigmoid_upper(logit) ** pnorm).sum(dim=-1).mean()
            for logit in self._logits.values()
        ]).sum()

        l_layerwise = torch.stack([
            F.mse_loss(self._masked_forward(x, active={name}), target_output)
            for name in self.param_names
        ]).mean()

        return {
            "recon": (l_recon, alpha_recon),
            "recon_layerwise": (l_layerwise, alpha_recon_layerwise),
            "importance": (l_importance, alpha_importance),
        }
