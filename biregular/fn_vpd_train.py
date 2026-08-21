"""Trained VPD over the (n_features, n_neurons) grid + basis-alignment scoring.

Counterpart to biregular/fn_sweep.py's oracle grid: per cell, rebuild the
designed model at its L4-optimal K* (from results/fn_sweep/kfit/) and run the
FULL VPD training recipe (our port: core/vpd.py, tuned pricing imp 1e-4 /
pnorm 2.0, lr 3e-4, 25k steps), then score how feature- vs neuron-aligned the
learned solution is, per matrix.

Alignment metric (two-sided basis membership, per alive component):
  W_in  feature score_c = max_j |cos(V_c, e_j)|      * |cos(U_c, w_in[j, :])|
  W_in  neuron  score_c = max_n |cos(V_c, w_in[:,n])| * |cos(U_c, e_n)|
  W_out neuron  score_c = max_n |cos(V_c, e_n)|      * |cos(U_c, w_out[n, :])|
  W_out feature score_c = max_j |cos(V_c, w_out[:,j])| * |cos(U_c, e_j)|
Per matrix: mass-weighted mean of each score over alive components, distinct-
argmax coverage of each basis, preference = mean_feature - mean_neuron, and a
verdict (feature / neuron / mixed / collapsed). Gate sign-awareness is probed
with +/- one-hot inputs (the discriminating observable where the two bases'
weight pieces nearly coincide, e.g. K* = 1 cells).

Designed for RunPod: cells are independent commands; results are per-cell
files that rsync back into results/fn_sweep/vpd/.

  python -m biregular.fn_vpd_train --train-grid [--max-f 300] [--seed 42]
  python -m biregular.fn_vpd_train --train-cell 100 50
  python -m biregular.fn_vpd_train --alignment            # recompute from ckpts
  python -m biregular.fn_vpd_train --map                  # trained preference map
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from biregular.biregular_vpd_compare import DEFAULT_KW, FIT_SEED
from biregular.fn_sweep import P, CODE_SEED, DECODER, cells, kfit_path
from biregular.vpd_objective import eval_objective
from core.codes import greedy_regular_code
from core.data import CleanDataset
from core.decomp_fit import fit_decomposition
from core.paths import FIGURES, RESULTS
from core.vpd import CI_ALIVE_THRESHOLD, VPD
from embed.embed_equivalence import model_from_code

VPD_DIR = os.path.join(RESULTS, "fn_sweep", "vpd")
FIG_DIR = os.path.join(FIGURES, "fn_sweep")
os.makedirs(VPD_DIR, exist_ok=True)
ALIVE_FRAC = 0.10          # component alive = Frobenius mass > 10% of side max
SCORE_HI, SCORE_GAP = 0.6, 0.1   # verdict thresholds on membership scores
STEPS_DEFAULT = 25_000
PROBE_V = 0.5


def cell_model(F_, N_, device="cpu"):
    with open(kfit_path(F_, N_)) as f:
        kf = json.load(f)
    M = torch.tensor(greedy_regular_code(F_, N_, kf["K_star"], CODE_SEED))
    model = model_from_code(M, kf["a"], kf["b"], DECODER)
    return model.to(device), kf


def ckpt_path(F_, N_, K, seed):
    return os.path.join(VPD_DIR, f"F{F_}_N{N_}_K{K}_s{seed}.pt")


# ------------------------------------------------------------- alignment

def _cosmat(A, B, eps=1e-12):
    """|cos| matrix between columns of A (d, n) and columns of B (d, m)."""
    A = A / A.norm(dim=0, keepdim=True).clamp_min(eps)
    B = B / B.norm(dim=0, keepdim=True).clamp_min(eps)
    return (A.T @ B).abs()


def _side_scores(V, U, basis_V, basis_U):
    """Two-sided membership: per component c, max_k cos(V_c, basis_V_k) *
    cos(U_c, basis_U_k). V (d0, C), U (C, d1); basis_V (d0, K), basis_U (K, d1).
    Returns (score per component, argmax basis element per component)."""
    prod = _cosmat(V, basis_V) * _cosmat(U.T, basis_U.T)   # (C, K)
    return prod.max(dim=1).values, prod.argmax(dim=1)


def basis_membership(pd_v: VPD) -> dict:
    """Per-matrix feature/neuron membership of the learned components."""
    model = pd_v.model
    w_in, w_out = pd_v.targets["w_in"], pd_v.targets["w_out"]
    F_, N_ = model.n_features, model.d_mlp
    eye_f = torch.eye(F_, device=w_in.device)
    eye_n = torch.eye(N_, device=w_in.device)
    bases = {
        "w_in": {"feature": (eye_f, w_in), "neuron": (w_in, eye_n)},
        "w_out": {"neuron": (eye_n, w_out), "feature": (w_out, eye_f)},
    }
    out = {}
    with torch.no_grad():
        for name in pd_v.param_names:
            V, U = pd_v.V[name], pd_v.U[name]
            mass = V.norm(dim=0) * U.norm(dim=1)
            alive = mass > ALIVE_FRAC * mass.max().clamp_min(1e-12)
            Va, Ua, w = V[:, alive], U[alive, :], mass[alive]
            rec = {"alive": int(alive.sum()), "C": pd_v.C}
            for basis, (bV, bU) in bases[name].items():
                if Va.numel() == 0:
                    rec[basis] = {"mean": 0.0, "coverage": 0}
                    continue
                s, idx = _side_scores(Va, Ua, bV, bU)
                rec[basis] = {
                    "mean": float((s * w).sum() / w.sum()),
                    "min": float(s.min()), "median": float(s.median()),
                    "coverage": len(set(idx.tolist())),
                    "basis_size": bV.shape[1]}
            mf = rec["feature"]["mean"] if Va.numel() else 0.0
            mn = rec["neuron"]["mean"] if Va.numel() else 0.0
            rec["preference"] = mf - mn
            n_expect = {"w_in": F_, "w_out": N_}[name]
            if rec["alive"] < 0.5 * min(N_, n_expect):
                rec["verdict"] = "collapsed"
            elif mf > SCORE_HI and mf - mn > SCORE_GAP:
                rec["verdict"] = "feature"
            elif mn > SCORE_HI and mn - mf > SCORE_GAP:
                rec["verdict"] = "neuron"
            else:
                rec["verdict"] = "mixed"
            out[name] = rec
    return out


def gate_sign_stats(pd_v: VPD, v: float = PROBE_V) -> dict:
    """Sum of gates on +v vs -v one-hot probes, per matrix. Sign-aware gating
    (the h != 0 ontology) shows ratio_minus/plus ~ 0 on both matrices."""
    model = pd_v.model
    F_ = model.n_features
    out = {}
    with torch.no_grad():
        for sign in (+1.0, -1.0):
            x = sign * v * torch.eye(F_, device=pd_v.targets["w_in"].device)
            e = model.embed_input(x)
            h = torch.relu(e @ pd_v.targets["w_in"])
            lower, _ = pd_v._ci(e, h)
            for name, ci in lower.items():
                out.setdefault(name, {})["plus" if sign > 0 else "minus"] = \
                    float(ci.sum())
    for name, d in out.items():
        d["minus_over_plus"] = d["minus"] / max(d["plus"], 1e-12)
    return out


# -------------------------------------------------------------- training

def train_cell(F_, N_, *, seed, steps, device):
    model, kf = cell_model(F_, N_, device)
    K = kf["K_star"]
    path = ckpt_path(F_, N_, K, seed)
    if os.path.exists(path):
        print(f"skip F={F_} N={N_} (exists)", flush=True)
        return
    C = F_ + N_
    torch.manual_seed(seed)
    pd_v = VPD(model, C=C)   # includes their 200-step faithfulness warmup
    ds = CleanDataset(n_features=F_, p=P, device=device, seed=FIT_SEED)
    loss_kw = dict(DEFAULT_KW)
    curves = fit_decomposition(pd_v, ds, steps=steps, loss_kwargs=loss_kw,
                               seed=FIT_SEED, desc=f"VPD F{F_}N{N_}K{K}")
    pd_v.to("cpu")
    model.to("cpu")
    alignment = basis_membership(pd_v)
    gates = gate_sign_stats(pd_v)
    losses = eval_objective(pd_v, p=P)
    ci = pd_v.ci_stats(P)
    rec = {"F": F_, "N": N_, "K": K, "a": kf["a"], "b": kf["b"], "p": P,
           "decoder": DECODER, "code_seed": CODE_SEED, "C": C, "seed": seed,
           "steps": steps, "loss_kw": loss_kw,
           "final_total": curves["total"][-1],
           "losses": losses, "alignment": alignment, "gate_sign": gates,
           "ci_stats": ci}
    torch.save({"pd_state": pd_v.state_dict(), "record": rec}, path)
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"F={F_} N={N_} K={K}: total={losses['total']:.4e} "
          f"in={alignment['w_in']['verdict']} ({alignment['w_in']['preference']:+.2f}) "
          f"out={alignment['w_out']['verdict']} ({alignment['w_out']['preference']:+.2f}) "
          f"sign(w_out)={gates['w_out']['minus_over_plus']:.2f}", flush=True)


def recompute_alignment():
    for fn in sorted(os.listdir(VPD_DIR)):
        if not fn.endswith(".pt"):
            continue
        payload = torch.load(os.path.join(VPD_DIR, fn), map_location="cpu",
                             weights_only=False)
        rec = payload["record"]
        model, _ = cell_model(rec["F"], rec["N"])
        torch.manual_seed(rec["seed"])
        pd_v = VPD(model, C=rec["C"], warmup_steps=0)
        pd_v.load_state_dict(payload["pd_state"])
        rec["alignment"] = basis_membership(pd_v)
        rec["gate_sign"] = gate_sign_stats(pd_v)
        payload["record"] = rec
        torch.save(payload, os.path.join(VPD_DIR, fn))
        with open(os.path.join(VPD_DIR, fn.replace(".pt", ".json")), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{fn}: in={rec['alignment']['w_in']['verdict']} "
              f"out={rec['alignment']['w_out']['verdict']}", flush=True)


def make_map():
    rows = []
    for fn in sorted(os.listdir(VPD_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(VPD_DIR, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        print("no trained cells yet")
        return
    from biregular.fn_sweep import F_GRID, N_GRID
    from biregular.fn_sweep_figures import trained_preference_map
    trained_preference_map(rows, F_GRID, N_GRID, FIG_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-grid", action="store_true")
    ap.add_argument("--train-cell", type=int, nargs=2, metavar=("F", "N"))
    ap.add_argument("--alignment", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--max-f", type=int, default=None)
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()
    if args.train_cell:
        train_cell(*args.train_cell, seed=args.seed, steps=args.steps,
                   device=args.device)
    if args.train_grid:
        todo = [c for c in cells(args.max_f, args.max_n)
                if os.path.exists(kfit_path(*c))]
        print(f"train-grid: {len(todo)} cells with kfit records "
              f"(device {args.device})", flush=True)
        for F_, N_ in sorted(todo, key=lambda c: c[0] * c[1]):
            train_cell(F_, N_, seed=args.seed, steps=args.steps,
                       device=args.device)
    if args.alignment:
        recompute_alignment()
    if args.map:
        make_map()


if __name__ == "__main__":
    main()
