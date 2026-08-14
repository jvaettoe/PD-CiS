"""Fit VPD to the c-free biregular ansatz and score component recovery vs ExactPD.

Memo-iteration question: does ANY setting of VPD's own loss parameters make it learn
ExactPD's per-feature W_out decomposition, or does it only ever learn the neuron-basis
one? VPD has no top-k; the per-example active-write-shard count (~K at defaults) is
shaped by the importance-minimality term (`--alpha-importance` coeff, `--pnorm`
exponent), while the stochastic/PGD reconstruction terms (`--alpha-stoch`,
`--alpha-pgd`) are the pressure DEFENDING the neuron basis (a per-feature write
basis cannot reproduce the interference leak without ~all components active).
Literature prior (SPD sec 3.3 / Fig 8): cranking importance yields per-feature
W_out only as an over-sparsification failure mode (CIs collapse, recon degrades).

  ./.venv/bin/python biregular_vpd_compare.py --p 0.001 --steps 25000       # baseline
  ./.venv/bin/python biregular_vpd_compare.py --p 0.001 --steps 4000 \\
      --alpha-importance 1e-3                                               # sweep cell
"""
import argparse
import os

from core.paths import REPO_ROOT

import torch

import core.decomp_fit as decomp_fit
from biregular.biregular_ansatz import CodeAnsatz
from biregular.component_match import run_suite
from core.data import CleanDataset
from core.pd import ExactPD
from core.vpd import VPD

DECOMP_DIR = os.path.join(REPO_ROOT, "results", "biregular", "decomps")
F = 100
FIT_SEED = 31337

DEFAULT_KW = {"alpha_faithfulness": 0.0, "alpha_importance": 1e-4,
              "alpha_stoch": 2.0, "alpha_pgd": 2.0, "pnorm": 2.0}


def knob_tag(kw: dict) -> str:
    """Filename token for non-default loss knobs ('' at defaults), e.g.
    '_imp1e-03_pn1.0_st0.5_pgd0.5'."""
    parts = []
    if kw["alpha_importance"] != DEFAULT_KW["alpha_importance"]:
        parts.append(f"imp{kw['alpha_importance']:.0e}")
    if kw["pnorm"] != DEFAULT_KW["pnorm"]:
        parts.append(f"pn{kw['pnorm']:g}")
    if kw["alpha_stoch"] != DEFAULT_KW["alpha_stoch"]:
        parts.append(f"st{kw['alpha_stoch']:g}")
    if kw["alpha_pgd"] != DEFAULT_KW["alpha_pgd"]:
        parts.append(f"pgd{kw['alpha_pgd']:g}")
    return ("_" + "_".join(parts)) if parts else ""


def _decomp_path(name: str, p: str, tag: str) -> str:
    return os.path.join(DECOMP_DIR, f"{name}_p{p}_VPD{tag}.pt")


def fit_or_load_vpd(code: CodeAnsatz, p: str, steps: int, device: str,
                    reuse: bool, loss_kw: dict, tag: str) -> tuple:
    """Return (model, pd_v, steps_used). Persists the fit with the c-free ansatz
    params (K, decoder, ab, loss knobs); --reuse rebuilds the identical model
    without refitting."""
    path = _decomp_path(code.name, p, tag)
    if reuse and os.path.exists(path):
        payload = torch.load(path, map_location=device, weights_only=False)
        model = code.model(tuple(payload["biregular"]["ab"]), device=device)
        pd_v = VPD(model, payload["C"], warmup_steps=0)
        pd_v.load_state_dict(payload["pd_state"], strict=False)
        print(f"  reusing saved VPD fit: {path}")
        return model, pd_v, int(payload["biregular"]["steps"])

    ab = code.fit_ab(float(p), device=device)
    model = code.model(ab, device=device)
    pd_v = decomp_fit.build_pd("VPD", model, F)
    curves = decomp_fit.fit_decomposition(
        pd_v, CleanDataset(n_features=F, p=float(p), device=device, seed=FIT_SEED),
        steps=steps, loss_kwargs=loss_kw,
        seed=FIT_SEED, desc=f"VPD {code.name} p={p}{tag}")
    print(f"  VPD fit done: total {curves['total'][0]:.3e} -> {curves['total'][-1]:.3e}")

    state = {k_: v for k_, v in pd_v.state_dict().items() if not k_.startswith("model.")}
    os.makedirs(DECOMP_DIR, exist_ok=True)
    payload = {"method": "VPD", "C": pd_v.C, "pd_state": state,
               "curves_total": curves["total"],
               "loss_kwargs": loss_kw,
               "biregular": {"K": code.K, "family": code.family, "seed": code.seed,
                             "swaps": code.swaps, "decoder": code.decoder,
                             "p": p, "ab": [float(v) for v in ab], "steps": steps}}
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
    ap.add_argument("--p", default="0.001")
    ap.add_argument("--steps", type=int, default=4000,
                    help="VPD fit steps (4000 = CPU screening budget; 25000 = full recipe)")
    ap.add_argument("--c", type=int, nargs="+", default=[0, 33, 66],
                    help="probe FEATURE indices for the spectra comparison")
    ap.add_argument("--alpha-importance", type=float, default=DEFAULT_KW["alpha_importance"],
                    dest="alpha_importance", help="importance-minimality coefficient")
    ap.add_argument("--pnorm", type=float, default=DEFAULT_KW["pnorm"],
                    help="importance-minimality exponent (p<=1 is sparsity-inducing)")
    ap.add_argument("--alpha-stoch", type=float, default=DEFAULT_KW["alpha_stoch"],
                    dest="alpha_stoch", help="stochastic recon coefficient")
    ap.add_argument("--alpha-pgd", type=float, default=DEFAULT_KW["alpha_pgd"],
                    dest="alpha_pgd", help="adversarial (PGD) recon coefficient")
    ap.add_argument("--reuse", action="store_true",
                    help="load the saved VPD fit for this cell instead of refitting")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    loss_kw = {"alpha_faithfulness": 0.0, "alpha_importance": a.alpha_importance,
               "alpha_stoch": a.alpha_stoch, "alpha_pgd": a.alpha_pgd,
               "pnorm": a.pnorm}
    tag = knob_tag(loss_kw)
    code = CodeAnsatz(a.K, swaps=0, decoder=a.decoder)
    print(f"[{code.name}] p={a.p} steps={a.steps} knobs={tag or 'defaults'}")
    model, pd_v, steps_used = fit_or_load_vpd(code, a.p, a.steps, a.device, a.reuse,
                                              loss_kw, tag)
    pd_e = ExactPD(model)
    rtag = f"{code.name}_p{a.p}_VPD{tag}_vs_ExactPD"
    ctx = (f"biregular K={a.K} {a.decoder} no-swap (c-free), p={a.p}, "
           f"VPD {steps_used} steps{tag or ' (default knobs)'}")
    run_suite(pd_v, pd_e, float(a.p), rtag, ctx, a.c, m_label="VPD")


if __name__ == "__main__":
    main()
