"""Fit a parameter decomposition (APD / OracleAPD / SPD) to a frozen MLP.

pd.py supplies the decomposition math (each class exposes
`loss(x, target_output, **kwargs) -> (total, logs)`), but no optimiser loop.
This module is that loop plus the per-method plumbing (which PD to build, which
loss kwargs to pass). Pure compute: it returns per-term loss curves and does no
file I/O (see results_io.py for storage)."""

from collections import defaultdict

import torch
from tqdm import tqdm

from cc import set_seed
from data import SparseDataset
from mlp import MLP
from pd import PD, APD, OracleAPD, SPD

METHODS = ("APD", "OracleAPD", "SPD")

# Per-method step budgets. OracleAPD recomputes importance with C leave-one-out
# forwards per step, so it gets far fewer steps than the gradient-attribution
# methods. SPD's masks are stochastic and need longer to converge.
METHOD_STEPS = {"APD": 3000, "OracleAPD": 500, "SPD": 5000}
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 2048
ORACLE_BATCH = 1024

# For SPD, which target params get decomposed and how they are applied. Both
# LinearMLP and EmbedMLP route w_in/w_out through decomp_op as plain matmuls
# (EmbedMLP's embedding w_e is applied outside the pd path, so it is not
# decomposed).
SPD_PARAM_SPECS = {"w_in": "matmul", "w_out": "matmul"}


def build_pd(method: str, model: MLP, C: int) -> PD:
    """Instantiate the decomposition for `method`. The model is frozen inside
    PD.__init__ (requires_grad_(False)), so only the decomposition's own params
    are trainable."""
    if method == "APD":
        return APD(model, C)
    if method == "OracleAPD":
        return OracleAPD(model, C)
    if method == "SPD":
        return SPD(model, C, param_specs=SPD_PARAM_SPECS)
    raise ValueError(f"Unknown method '{method}' (expected one of {METHODS})")


def loss_kwargs_for(method: str, k: int, lp: float = 1.0) -> dict:
    """Method-specific kwargs forwarded verbatim into PD.loss(**kwargs).

    NOTE: APD's singular-value exponent is also named `p` in pd.py -- we call it
    `lp` here so it never collides with the data sparsity `p`, and map lp -> 'p'."""
    if method in ("APD", "OracleAPD"):
        return {"alpha_faithfulness": 1.0, "k": k,
                "alpha_minimality": 1.0, "alpha_simplicity": 1.0, "p": lp}
    if method == "SPD":
        return {"alpha_faithfulness": 1.0, "alpha_recon": 1.0,
                "alpha_recon_layerwise": 1.0, "alpha_importance": 1e-5,
                "pnorm": 2.0}
    raise ValueError(f"Unknown method '{method}' (expected one of {METHODS})")


def fit_decomposition(
    pd_obj: PD,
    dataset: SparseDataset,
    *,
    steps: int,
    loss_kwargs: dict,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    seed: int | None = None,
    log_every: int = 1,
    progress: bool = True,
    desc: str = "decomp",
) -> dict[str, list[float]]:
    """Optimise the trainable PD parameters against the FROZEN model's own output.

    Each step draws inputs from the dataset (labels ignored), takes the frozen
    model's prediction as the reconstruction target, and steps Adam on the
    decomposition's loss. Returns curves {term_name: [value per logged step]},
    whose keys are exactly the log keys pd_obj.loss emits (APD/OracleAPD:
    faithful/minimal/simple/total; SPD: faithful/recon/recon_layerwise/
    importance/total)."""
    if seed is not None:
        set_seed(seed)  # reproducible PD init + SPD stochastic masks (global RNG)

    params = [p for p in pd_obj.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    curves: dict[str, list[float]] = defaultdict(list)
    pbar = tqdm(range(steps), desc=desc, disable=not progress)
    for step in pbar:
        x, _ = dataset.generate_batch(batch_size)
        with torch.no_grad():
            target = pd_obj.model(x)  # frozen model output (pd=None dense path)
        opt.zero_grad()
        total, logs = pd_obj.loss(x, target, **loss_kwargs)
        total.backward()
        opt.step()
        if step % log_every == 0:
            for name, value in logs.items():
                curves[name].append(value)
            if progress and step % (max(1, steps // 20)) == 0:
                pbar.set_postfix({"total": logs.get("total", float("nan"))})
    return dict(curves)
