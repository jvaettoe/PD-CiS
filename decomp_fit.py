"""Fit a parameter decomposition (APD / OracleAPD / SPD) to a frozen MLP.

pd.py supplies the decomposition math (each class exposes
`loss(x, target_output, **kwargs) -> (total, logs)`), but no optimiser loop.
This module is that loop plus the per-method plumbing (which PD to build, which
loss kwargs to pass). Pure compute: it returns per-term loss curves and does no
file I/O (see results_io.py for storage)."""

import os
from collections import defaultdict

import torch
from tqdm import tqdm

import random

import numpy as np

from data import SparseDataset
from mlp import MLPBase
from pd import PD, APD, OracleAPD, SPD, ExactPD
from vpd import VPD

METHODS = ("APD", "OracleAPD", "SPD", "ExactPD", "VPD")


def set_seed(seed: int) -> None:
    """Inlined from the archived cc.py (memo iteration: training stack not restored)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Per-method step budgets. OracleAPD recomputes importance with C leave-one-out
# forwards per step, so it gets far fewer steps than the gradient-attribution
# methods. SPD's masks are stochastic and need longer to converge. ExactPD is the
# handcoded ground truth -- its components are set analytically and frozen, so it
# has no trainable params and needs a single eval pass to log its loss terms.
METHOD_STEPS = {"APD": 3000, "OracleAPD": 500, "SPD": 5000, "ExactPD": 1,
                "VPD": 25_000}  # VPD: Goodfire resid_mlp2 reference (GPU-scale;
                                # on CPU pass --steps-scale 0.16 for the old 4k budget)
DEFAULT_LR = 1e-3
# Per-method LR overrides: VPD's reference (resid_mlp2 config) trains at a constant
# 3e-4 (AdamW, weight_decay 0 == our Adam), not the harness default 1e-3.
METHOD_LR = {"VPD": 3e-4}
DEFAULT_BATCH = 2048
ORACLE_BATCH = 1024

# Optional global gradient-norm clip. Off by default (as in Apollo APD): the
# svdvals-backward instability that motivated clipping is gone now that APD uses
# factored components + a closed-form Schatten (pd.APD.simplicity_loss), so the
# gradient is finite under rank collapse. Kept as an opt-in backstop.
GRAD_CLIP = None

# Rank budget of each APD component (P_c = A_c @ B_c, inner dim RANK). 1 => rank-1
# components (like SPD's U@V); a rank-1 component cannot absorb the whole model,
# which is what structurally prevents the single-component collapse.
APD_RANK = 1

# Simplicity (Schatten-p) coefficient for APD/OracleAPD. Apollo APD's
# calc_schatten_loss is normalised /n_weight_params/batch (as ours now is) and run
# at schatten_coeff = 1.0 -- reconstruction leads not by a tiny coefficient but by
# the matched normalisation. (An earlier un-normalised svdvals simplicity needed
# ~1e-5 to avoid dominating; that ad-hoc value starved the decomposition pressure
# and caused collapse -- see the factored refactor.) Tunable/sweepable.
ALPHA_SIMPLICITY = 1.0

# For SPD, which target params get decomposed and how they are applied. Both
# MLP and EmbedMLP route w_in/w_out through decomp_op as plain matmuls
# (EmbedMLP's embedding w_e is applied outside the pd path, so it is not
# decomposed).
SPD_PARAM_SPECS = {"w_in": "matmul", "w_out": "matmul"}


def build_pd(method: str, model: MLPBase, C: int, rank: int = APD_RANK) -> PD:
    """Instantiate the decomposition for `method`. The model is frozen inside
    PD.__init__ (requires_grad_(False)), so only the decomposition's own params
    are trainable. `rank` is the APD/OracleAPD component rank budget (ignored by
    SPD, whose components are rank-1 by construction)."""
    if method == "APD":
        return APD(model, C, rank=rank)
    if method == "OracleAPD":
        return OracleAPD(model, C, rank=rank)
    if method == "SPD":
        return SPD(model, C, param_specs=SPD_PARAM_SPECS)
    if method == "ExactPD":
        # C is ignored: ExactPD forces C = n_features (one component per feature).
        return ExactPD(model)
    if method == "VPD":
        return VPD(model, C)
    raise ValueError(f"Unknown method '{method}' (expected one of {METHODS})")


def loss_kwargs_for(method: str, k: int, lp: float = 1.0,
                    alpha_simplicity: float = ALPHA_SIMPLICITY) -> dict:
    """Method-specific kwargs forwarded verbatim into PD.loss(**kwargs).

    NOTE: APD's singular-value exponent is also named `p` in pd.py -- we call it
    `lp` here so it never collides with the data sparsity `p`, and map lp -> 'p'.
    `alpha_simplicity` downweights the Schatten-p term so reconstruction leads
    (see ALPHA_SIMPLICITY); pass a different value to sweep the balance."""
    if method in ("APD", "OracleAPD", "ExactPD"):
        return {"alpha_faithfulness": 1.0, "k": k,
                "alpha_minimality": 1.0, "alpha_simplicity": alpha_simplicity, "p": lp}
    if method == "VPD":
        # Goodfire resid_mlp2 toy recipe; faithfulness coeff 0 (delta component + warmup).
        return {"alpha_faithfulness": 0.0, "alpha_importance": 1e-4,
                "alpha_stoch": 2.0, "alpha_pgd": 2.0, "pnorm": 2.0}
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
    lr: float | None = None,
    batch_size: int = DEFAULT_BATCH,
    seed: int | None = None,
    log_every: int = 1,
    progress: bool = True,
    desc: str = "decomp",
    grad_clip: float | None = GRAD_CLIP,
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

    # lr=None resolves per method (VPD's reference recipe is 3e-4, not the default)
    if lr is None:
        lr = METHOD_LR.get(type(pd_obj).__name__, DEFAULT_LR)

    params = [p for p in pd_obj.parameters() if p.requires_grad]
    # ExactPD (and any analytic decomposition) has no trainable params: skip the
    # optimiser entirely and just evaluate the loss once per step to log its terms.
    opt = torch.optim.Adam(params, lr=lr) if params else None

    curves: dict[str, list[float]] = defaultdict(list)
    pbar = tqdm(range(steps), desc=desc, disable=not progress)
    for step in pbar:
        x, _ = dataset.generate_batch(batch_size)
        with torch.no_grad():
            target = pd_obj.model(x)  # frozen model output (pd=None dense path)
        if opt is not None:
            opt.zero_grad()
            total, logs = pd_obj.loss(x, target, **loss_kwargs)
            total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
        else:
            # No trainable params, but APD's batch-top-k attribution still needs
            # autograd (grad wrt the per-example gates), so DON'T wrap in no_grad.
            _, logs = pd_obj.loss(x, target, **loss_kwargs)
        if step % log_every == 0:
            for name, value in logs.items():
                curves[name].append(value)
            if progress and step % (max(1, steps // 20)) == 0:
                pbar.set_postfix({"total": logs.get("total", float("nan"))})
    return dict(curves)


# --------------------------------------------------------------------------
# Persist / reload a fitted decomposition so its components can be inspected.
# We save ONLY the decomposition's own params (not the frozen model, which
# lives in the checkpoint cache and is reloaded from `ckpt`), keyed by run_id.
# --------------------------------------------------------------------------
# save_decomposition / load_decomposition (checkpoint-backed) are NOT restored in
# the memo iteration -- they need the archived cc.py training stack. See
# archive/2026-08-04_pre_memo/code/decomp_fit.py.
