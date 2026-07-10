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

import cc
import results_io
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


# --------------------------------------------------------------------------
# Persist / reload a fitted decomposition so its components can be inspected.
# We save ONLY the decomposition's own params (not the frozen model, which
# lives in the checkpoint cache and is reloaded from `ckpt`), keyed by run_id.
# --------------------------------------------------------------------------

def save_decomposition(run_id: str, pd_obj: PD, ckpt_meta: dict, method: str) -> str:
    """Write results/<tag>/decomps/<run_id>.pt = the fitted component tensors
    plus a pointer to the model checkpoint needed to reload+project them.
    Atomic (tmp + replace). `ckpt_meta` = {str_p, model_class, model_kwargs,
    n_exp, seed} -- enough for load_decomposition to find the model."""
    state = {k: v for k, v in pd_obj.state_dict().items() if not k.startswith("model.")}
    payload = {"method": method, "C": pd_obj.C,
               "param_names": list(pd_obj.param_names),
               "pd_state": state, "ckpt": ckpt_meta}
    path = results_io.decomp_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_decomposition(run_id: str, device: str = "cpu") -> PD:
    """Rebuild a fitted decomposition for inspection: reload its model from the
    checkpoint cache, rebuild the PD, load the saved component params. Requires
    both the decomp .pt and the model checkpoint to exist."""
    path = results_io.decomp_path(run_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no saved decomposition for {run_id}: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    ck = payload["ckpt"]
    model_cls = cc._MODEL_REGISTRY[ck["model_class"]]
    ckpt = cc.checkpoint_path(ck["str_p"], model_cls, ck["model_kwargs"],
                              seed=ck["seed"], n_exp=ck["n_exp"])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"model checkpoint for {run_id} not found: {ckpt} "
            f"(pull the checkpoints, or re-train that model)")
    model = cc.load_checkpoint(ckpt, device=device)["model"]
    pd_obj = build_pd(payload["method"], model, payload["C"])
    pd_obj.load_state_dict(payload["pd_state"], strict=False)
    return pd_obj
