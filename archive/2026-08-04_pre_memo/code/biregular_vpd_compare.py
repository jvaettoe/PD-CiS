"""Fit VPD to a biregular ansatz model and score component recovery vs ExactPD.

Builds the CodeAnsatz model for (K, swaps=0, decoder) at sparsity p, fits VPD to it
(the model has no checkpoint, so the decomposition is persisted with the ansatz
parameters instead of a checkpoint pointer), then runs the four-matrix recovery
suite from component_match.py plus the output-spectra comparison.

The scientific question (see component_match.py): ExactPD's w_in components are
disjoint rows and should be recoverable; its w_out column basis needs ~all
components active per input to reproduce the interference leak, so VPD may prefer a
leak-aligned write basis instead — the transpose decoder (leak ~ signal) stresses
this hardest.

  ./.venv/bin/python biregular_vpd_compare.py                      # transpose, p=0.1
  ./.venv/bin/python biregular_vpd_compare.py --decoder pinv
  ./.venv/bin/python biregular_vpd_compare.py --reuse              # skip refit if saved
"""
import argparse
import os

import torch

import decomp_fit
from biregular_ansatz import CodeAnsatz
from component_match import run_suite
from data import CleanDataset
from pd import ExactPD
from vpd import VPD

DECOMP_DIR = os.path.join(os.path.dirname(__file__), "results", "biregular", "decomps")
F = 100
FIT_SEED = 31337


def _decomp_path(name: str, p: str) -> str:
    return os.path.join(DECOMP_DIR, f"{name}_p{p}_VPD.pt")


def fit_or_load_vpd(code: CodeAnsatz, p: str, steps: int, device: str,
                    reuse: bool) -> tuple:
    """Return (model, pd_v, steps_used). Persists the fit with the ansatz params
    (K, decoder, abc, ...) so --reuse rebuilds the identical model without
    refitting anything. steps_used is the PAYLOAD's step count under --reuse (the
    CLI --steps does not describe a reloaded fit)."""
    path = _decomp_path(code.name, p)
    if reuse and os.path.exists(path):
        payload = torch.load(path, map_location=device, weights_only=False)
        model = code.model(tuple(payload["biregular"]["abc"]), device=device)
        pd_v = VPD(model, payload["C"], warmup_steps=0)
        pd_v.load_state_dict(payload["pd_state"], strict=False)
        print(f"  reusing saved VPD fit: {path}")
        return model, pd_v, int(payload["biregular"]["steps"])

    abc = code.fit_abc(float(p), device=device)
    model = code.model(abc, device=device)
    pd_v = decomp_fit.build_pd("VPD", model, F)
    k = max(1, min(F - 1, round(float(p) * F)))
    curves = decomp_fit.fit_decomposition(
        pd_v, CleanDataset(n_features=F, p=float(p), device=device, seed=FIT_SEED),
        steps=steps, loss_kwargs=decomp_fit.loss_kwargs_for("VPD", k=k),
        seed=FIT_SEED, desc=f"VPD {code.name} p={p}")
    print(f"  VPD fit done: total {curves['total'][0]:.3e} -> {curves['total'][-1]:.3e}")

    state = {k_: v for k_, v in pd_v.state_dict().items() if not k_.startswith("model.")}
    os.makedirs(DECOMP_DIR, exist_ok=True)
    payload = {"method": "VPD", "C": pd_v.C, "pd_state": state,
               "curves_total": curves["total"],
               "biregular": {"K": code.K, "family": code.family, "seed": code.seed,
                             "swaps": code.swaps, "decoder": code.decoder,
                             "p": p, "abc": [float(v) for v in abc], "steps": steps}}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  saved {path}")
    return model, pd_v, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--decoder", default="transpose", choices=["transpose", "pinv"])
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--steps", type=int, default=4000,
                    help="VPD fit steps (4000 = the CPU budget; 25000 = full recipe)")
    ap.add_argument("--c", type=int, nargs="+", default=[0, 33, 66],
                    help="probe features for the spectra comparison")
    ap.add_argument("--reuse", action="store_true",
                    help="load the saved VPD fit for this cell instead of refitting")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    code = CodeAnsatz(a.K, swaps=0, decoder=a.decoder)
    print(f"[{code.name}] p={a.p} steps={a.steps}")
    model, pd_v, steps_used = fit_or_load_vpd(code, a.p, a.steps, a.device, a.reuse)
    pd_e = ExactPD(model)
    tag = f"{code.name}_p{a.p}_VPD_vs_ExactPD"
    ctx = f"biregular K={a.K} {a.decoder} no-swap, p={a.p}, VPD {steps_used} steps"
    run_suite(pd_v, pd_e, float(a.p), tag, ctx, a.c, m_label="VPD")


if __name__ == "__main__":
    main()
