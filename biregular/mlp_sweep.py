"""Trained-MLP sweep: (F, N) x sparsity grid with the fn_sweep analysis suite.

Per cell (F, N, p): train the suite MLP (F -> N ReLU -> F, no biases, L4)
on CleanDataset(p), then
  --oracle : oracle-gated 2x2 basis comparison (feature/neuron per matrix) of
             ideal PDs built from the TRAINED weights (w_out != w_in^T here;
             tiling still exact by construction), eval at the cell's p
  --vpd    : full trained VPD (C = F + N, convergence blocks, tuned pricing)
             + basis-membership alignment vs the trained model's own bases
  --solo   : full vs per-feature solo decomposition (leak-free for ANY weights;
             on-target exact on one-hot probes) -> leak/attenuation ratios
Census (features computed, held-out L4) recorded with every model; verdicts
must be read against it (cells without CiS are not basis findings).

Grid: F in {40,100,200,400,1000} x N in {20,50,100,250,500}, N <= F;
p in {0.001, 0.01, 0.05, 0.1, 0.2, 0.5}.

  python -m biregular.mlp_sweep --train-models [--device cuda]
  python -m biregular.mlp_sweep --oracle --vpd --solo --figures
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from biregular.biregular_vpd_compare import DEFAULT_KW, FIT_SEED
from biregular.fn_sweep import build_combo_vpd, BASES, TIE_RATIO, classify_matrix
from biregular.fn_vpd_train import (MAX_STEPS_DEFAULT, STABLE_BLOCKS,
                                    alive_counts, basis_membership,
                                    gate_sign_stats)
from biregular.vpd_objective import EVAL_SEED, eval_objective
from core.data import CleanDataset
from core.decomp_fit import fit_decomposition
from core.mlp import MLP
from core.paths import FIGURES, RESULTS
from core.vpd import VPD

F_GRID = [40, 100, 200, 400, 1000]
N_GRID = [20, 50, 100, 250, 500]
P_GRID = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
# --fine (phase 2, supersets): denser F/N + dFH's canonical p = 0.02 and more
F_GRID_FINE = [40, 70, 100, 140, 200, 300, 400, 700, 1000]
N_GRID_FINE = [20, 35, 50, 70, 100, 150, 250, 350, 500]
P_GRID_FINE = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
TRAIN_STEPS, TRAIN_BATCH, TRAIN_SEED = 20_000, 4096, 42
VPD_BLOCK = 25_000

RES_DIR = os.path.join(RESULTS, "mlp_sweep")
MODEL_DIR = os.path.join(RES_DIR, "models")
ORACLE_DIR = os.path.join(RES_DIR, "oracle")
VPD_DIR = os.path.join(RES_DIR, "vpd")
SOLO_DIR = os.path.join(RES_DIR, "solo")
FIG_DIR = os.path.join(FIGURES, "mlp_sweep")
for _d in (MODEL_DIR, ORACLE_DIR, VPD_DIR, SOLO_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def cells():
    for p in P_GRID:
        for F_, N_ in itertools.product(F_GRID, N_GRID):
            if N_ <= F_:
                yield F_, N_, p


def tag(F_, N_, p):
    return f"F{F_}_N{N_}_p{p:g}"


def census(model, p, device):
    """Held-out L4 + features-computed count (active-conditional MSE < 1/30
    against the always-0 baseline 1/3) + read-side K distribution (tau=0.05)."""
    F_ = model.n_features
    ds = CleanDataset(n_features=F_, p=p, device=device, seed=EVAL_SEED)
    l4, err, cnt = 0.0, torch.zeros(F_, device=device), torch.zeros(F_, device=device)
    with torch.no_grad():
        for _ in range(25):
            x, y = ds.generate_batch(8192)
            out = model(x)
            l4 += float(((out - y).abs() ** 4).mean())
            a = (x > 0).float()
            err += (((out - y) ** 2) * a).sum(0)
            cnt += a.sum(0)
        pf = (err / cnt.clamp(min=1)).cpu().numpy()
        Ks = (model.w_in.detach().abs() > 0.05).sum(1).cpu().numpy()
    return {"l4": l4 / 25, "computed": int((pf < 1 / 30).sum()),
            "K_median": float(np.median(Ks)), "K_mean": float(Ks.mean())}


def train_models(device):
    for F_, N_, p in cells():
        path = os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt")
        if os.path.exists(path):
            continue
        torch.manual_seed(TRAIN_SEED)
        model = MLP(n_features=F_, d_mlp=N_, device=device)
        ds = CleanDataset(n_features=F_, p=p, device=device, seed=TRAIN_SEED)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
        for _ in range(TRAIN_STEPS):
            x, y = ds.generate_batch(TRAIN_BATCH)
            loss = ((model(x) - y).abs() ** 4).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        cen = census(model, p, device)
        torch.save({"model_config": model.config(), "p": p,
                    "state_dict": {k: v.cpu() for k, v in
                                   model.state_dict().items()},
                    "train": {"steps": TRAIN_STEPS, "batch": TRAIN_BATCH,
                              "seed": TRAIN_SEED, "loss": "L4",
                              "lr": 3e-3, "wd": 0.01},
                    "census": cen}, path)
        print(f"{tag(F_, N_, p)}: l4={cen['l4']:.3e} computed={cen['computed']}/"
              f"{F_} K_med={cen['K_median']:.0f}", flush=True)


def load_model(F_, N_, p, device="cpu"):
    ck = torch.load(os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt"),
                    map_location=device, weights_only=False)
    model = MLP(**ck["model_config"], device=device)
    model.load_state_dict(ck["state_dict"])
    model.device = device
    return model, ck["census"]


def run_oracle(device):
    for F_, N_, p in cells():
        mp = os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt")
        if not os.path.exists(mp):
            continue
        combos = [(rb, wb) for rb, wb in itertools.product(BASES, BASES)
                  if not os.path.exists(os.path.join(
                      ORACLE_DIR, f"{tag(F_, N_, p)}_{rb}-{wb}.json"))]
        if not combos:
            continue
        model, cen = load_model(F_, N_, p, device)
        ekw = ({"batch": 2048, "n_batches": 100} if max(F_, N_) > 2000 else {})
        for rb, wb in combos:
            pd_v = build_combo_vpd(model, rb, wb)
            res = eval_objective(pd_v, p=p, **ekw)
            rec = {"F": F_, "N": N_, "p": p, "read_basis": rb, "write_basis": wb,
                   "C": pd_v.C, "census": cen, "loss_kw": dict(DEFAULT_KW),
                   "losses": res}
            with open(os.path.join(ORACLE_DIR,
                                   f"{tag(F_, N_, p)}_{rb}-{wb}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            print(f"{tag(F_, N_, p)} ({rb},{wb}): total={res['total']:.4e}",
                  flush=True)


def run_vpd(device):
    for F_, N_, p in cells():
        mp = os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt")
        vp = os.path.join(VPD_DIR, tag(F_, N_, p) + "_s42.pt")
        if not os.path.exists(mp) or os.path.exists(vp):
            continue
        model, cen = load_model(F_, N_, p, device)
        C = F_ + N_
        torch.manual_seed(TRAIN_SEED)
        pd_v = VPD(model, C=C)
        ds = CleanDataset(n_features=F_, p=p, device=device, seed=FIT_SEED)
        alive_hist, total_steps = [], 0
        while True:
            block = len(alive_hist)
            curves = fit_decomposition(
                pd_v, ds, steps=VPD_BLOCK, loss_kwargs=dict(DEFAULT_KW),
                seed=FIT_SEED + 97 * block,
                desc=f"VPD {tag(F_, N_, p)} b{block}")
            total_steps += VPD_BLOCK
            alive_hist.append(alive_counts(pd_v))
            settled = (len(alive_hist) > STABLE_BLOCKS
                       and all(alive_hist[-1] == alive_hist[-1 - i]
                               for i in range(1, STABLE_BLOCKS + 1)))
            if settled or total_steps >= MAX_STEPS_DEFAULT:
                break
        pd_v.to("cpu")
        model.to("cpu")
        model.device = "cpu"
        rec = {"F": F_, "N": N_, "p": p, "C": C, "seed": TRAIN_SEED,
               "steps": total_steps, "alive_history": alive_hist,
               "census": cen, "final_total": curves["total"][-1],
               "losses": eval_objective(pd_v, p=p),
               "alignment": basis_membership(pd_v),
               "gate_sign": gate_sign_stats(pd_v),
               "ci_stats": pd_v.ci_stats(p)}
        torch.save({"pd_state": pd_v.state_dict(), "record": rec}, vp)
        with open(vp.replace(".pt", ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        a = rec["alignment"]
        print(f"{tag(F_, N_, p)}: total={rec['losses']['total']:.3e} "
              f"in={a['w_in']['verdict']}({a['w_in']['preference']:+.2f}) "
              f"out={a['w_out']['verdict']}({a['w_out']['preference']:+.2f})",
              flush=True)


def run_solo(device):
    for F_, N_, p in cells():
        mp = os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt")
        sp = os.path.join(SOLO_DIR, tag(F_, N_, p) + ".json")
        if not os.path.exists(mp) or os.path.exists(sp):
            continue
        model, cen = load_model(F_, N_, p, device)
        w_in = model.w_in.detach()
        w_out = model.w_out.detach()
        with torch.no_grad():
            R = (torch.relu(torch.eye(F_, device=w_in.device) @ w_in)
                 @ w_out).cpu()
            solo_on = (torch.relu(w_in) * w_out.T).sum(dim=1).cpu()
        dev = float((solo_on - R.diagonal()).abs().max())
        dfull = R - torch.eye(F_)
        e2, e4 = dfull ** 2, dfull ** 4
        mse_on = float(e2.diagonal().mean() / F_)
        mse_off = float((e2.sum(1) - e2.diagonal()).mean() / F_)
        l4_on = float(e4.diagonal().mean() / F_)
        l4_off = float((e4.sum(1) - e4.diagonal()).mean() / F_)
        rec = {"F": F_, "N": N_, "p": p, "census": cen,
               "solo_on_target_max_dev": dev,
               "atten_mse": mse_on, "leak_mse": mse_off,
               "atten_l4": l4_on, "leak_l4": l4_off,
               "ratio_mse": mse_off / mse_on if mse_on > 1e-15 else None,
               "ratio_l4": l4_off / l4_on if l4_on > 1e-15 else None}
        with open(sp, "w") as f:
            json.dump(rec, f, indent=1)
        r = rec["ratio_mse"]
        print(f"{tag(F_, N_, p)}: leak/atten (mse) = "
              f"{'exact' if r is None else f'{r:.3g}'}", flush=True)


def make_figures():
    from biregular.fn_sweep_figures import basis_preference_map, kstar_heatmap
    for p in P_GRID:
        fdir = os.path.join(FIG_DIR, f"p{p:g}")
        os.makedirs(fdir, exist_ok=True)
        rows = []
        for F_, N_ in itertools.product(F_GRID, N_GRID):
            if N_ > F_:
                continue
            totals = {}
            ok = True
            for rb, wb in itertools.product(BASES, BASES):
                op = os.path.join(ORACLE_DIR, f"{tag(F_, N_, p)}_{rb}-{wb}.json")
                if not os.path.exists(op):
                    ok = False
                    break
                with open(op) as f:
                    totals[(rb, wb)] = json.load(f)["losses"]["total"]
            if not ok:
                continue
            v_in, m_in = classify_matrix(totals, "w_in")
            v_out, m_out = classify_matrix(totals, "w_out")
            with open(os.path.join(MODEL_DIR, tag(F_, N_, p) + ".pt"), "rb"):
                pass
            cen = load_model(F_, N_, p)[1]
            rows.append({"F": F_, "N": N_, "K_star": round(cen["K_median"]),
                         "l4_star": cen["l4"],
                         "w_in_pref": v_in, "w_in_margin": m_in,
                         "w_out_pref": v_out, "w_out_margin": m_out})
        if rows:
            kstar_heatmap(rows, F_GRID, N_GRID, fdir)
            basis_preference_map(rows, F_GRID, N_GRID, fdir, TIE_RATIO)
        vrows = []
        for fjson in sorted(os.listdir(VPD_DIR)):
            if fjson.endswith(".json"):
                with open(os.path.join(VPD_DIR, fjson)) as f:
                    r = json.load(f)
                if r["p"] == p:
                    vrows.append(r)
        if vrows:
            from biregular.fn_sweep_figures import trained_preference_map
            trained_preference_map(vrows, F_GRID, N_GRID, fdir)


def main():
    ap = argparse.ArgumentParser()
    for flag in ("train-models", "oracle", "vpd", "solo", "figures"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--fine", action="store_true",
                    help="phase-2 grids (finer F/N, more p incl. dFH's 0.02)")
    ap.add_argument("--p-only", default=None,
                    help="comma-separated p subset (worker sharding)")
    args = ap.parse_args()
    if args.fine:
        global F_GRID, N_GRID, P_GRID
        F_GRID, N_GRID, P_GRID = F_GRID_FINE, N_GRID_FINE, P_GRID_FINE
    if args.p_only:
        keep = {float(v) for v in args.p_only.split(",")}
        P_GRID = [q for q in P_GRID if q in keep]
    if args.train_models:
        train_models(args.device)
    if args.oracle:
        run_oracle(args.device)
    if args.vpd:
        run_vpd(args.device)
    if args.solo:
        run_solo(args.device)
    if args.figures:
        make_figures()


if __name__ == "__main__":
    main()
