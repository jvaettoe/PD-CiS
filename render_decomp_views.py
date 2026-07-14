"""Render the component-visualisation suite for the saved n=4 APD decompositions
of EmbedMLP (d_emb=200) at BOTH hidden widths, across all sparsities p, so you can
see the components and how they change as p changes.

Two outputs, both under figures/n4/components/:
  1. Per-cell suite (delegates to plot_components.py): for each (d_mlp, p) the
     profiles / assignment / usage / faithfulness / permutation-sorted views, named
     ...dmlp<d_mlp>...  -- distinct per width (they used to collide).
  2. Cross-p comparison (figutil.plot_components_vs_p): ONE figure per d_mlp with
     rows = p, columns = [reads |u| | writes |v| | usage], titled with each row's
     k + effective_count / monosemanticity / coverage. Filename:
       EmbedMLP_demb200_dmlp<d_mlp>_n4_APD_vs_p.png

Note k = round(p * n_features) is clamped to [1, C-1], so k=1 for BOTH p=0.001 and
p=0.01 (the two sparsest rows share a budget).

  ./.venv/bin/python render_decomp_views.py
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

import decomp_fit
import figutil
import results_io
from data import CleanDataset

MODEL = "EmbedMLP"
DEMB = 200
N_EXP = 4.0
METHOD = "APD"
DMLPS = [50, 100]
PS = ["0.001", "0.01", "0.1", "0.5", "0.8"]


def k_for(p: str, C: int) -> int:
    """Batch-top-k budget = round(p * n_features), clamped to [1, C-1]
    (matches sweep.k_for / plot_components)."""
    return max(1, min(C - 1, round(float(p) * 100)))


def residual_frobenius(pd_obj):
    """Frobenius norm of the effective feature->feature map residual (same quantity
    the per-cell 'faithfulness' view renders): raw ‖W_model - W_recon‖_F and the
    relative ‖resid‖_F / ‖W_model‖_F. W_recon sums ALL components' weights and
    projects them the same way the model does (project_in/out fold in w_e)."""
    model_eff = pd_obj.model.effective_matrix().detach().cpu().numpy()
    w_in_hat = pd_obj.reconstructed_weight("w_in")
    w_out_hat = pd_obj.reconstructed_weight("w_out")
    recon_eff = (pd_obj.model.project_in(w_in_hat)
                 @ pd_obj.model.project_out(w_out_hat)).detach().cpu().numpy()
    resid = model_eff - recon_eff
    raw = float(np.linalg.norm(resid))
    rel = raw / (float(np.linalg.norm(model_eff)) + 1e-12)
    return raw, rel


def build_usage(pd_obj, p: str, k: int, device="cpu"):
    """APD behavioural usage on a fresh batch (same recipe as plot_components)."""
    ds = CleanDataset(n_features=pd_obj.model.n_features, p=float(p),
                      device=device, seed=0)
    xb, _ = ds.generate_batch(1024)
    with torch.no_grad():
        tb = pd_obj.model(xb)
    return pd_obj.usage_stats(xb, tb, k)


def render_per_cell():
    """Delegate each (d_mlp, p) cell to plot_components.py -- the existing per-cell
    render path (no new plotting logic), now collision-free thanks to d_mlp in the
    run_id."""
    n_ok = 0
    for d_mlp in DMLPS:
        for p in PS:
            cmd = [sys.executable, "plot_components.py",
                   "--model", MODEL, "--demb", str(DEMB), "--d-mlp", str(d_mlp),
                   "--p", p, "--n", results_io._fmt_n(N_EXP), "--method", METHOD]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                n_ok += 1
            else:
                print(f"  [FAIL] d_mlp={d_mlp} p={p}: {r.stderr.strip()[:300]}")
    print(f"[1/2] per-cell suites rendered: {n_ok}/{len(DMLPS) * len(PS)}")


def render_cross_p(device="cpu"):
    out_dir = os.path.join(figutil.FIG_ROOT, figutil.n_seg(N_EXP), "components")
    os.makedirs(out_dir, exist_ok=True)
    for d_mlp in DMLPS:
        entries = []
        for p in PS:
            run_id = results_io.make_run_id(model=MODEL, d_emb=DEMB, d_mlp=d_mlp,
                                            p=p, n_exp=N_EXP, method=METHOD)
            pd_obj = decomp_fit.load_decomposition(run_id, device=device)
            k = k_for(p, pd_obj.C)
            entries.append({"p": p, "k": k, "pd_obj": pd_obj,
                            "usage": build_usage(pd_obj, p, k, device)})
        params = {"model": MODEL, "d_emb": DEMB, "d_mlp": d_mlp,
                  "n_exp": N_EXP, "method": METHOD, "C": entries[0]["pd_obj"].C}
        fig = figutil.plot_components_vs_p(entries, params=params)
        out = os.path.join(out_dir,
                           f"{MODEL}_demb{DEMB}_dmlp{d_mlp}_n{results_io._fmt_n(N_EXP)}"
                           f"_{METHOD}_vs_p.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        figutil.plt.close(fig)
        print(f"  saved {out}")
    print(f"[2/2] cross-p figures: {len(DMLPS)}")


def render_faithfulness_vs_p(device="cpu"):
    """One figure, both widths: effective-map Frobenius residual vs p."""
    out_dir = os.path.join(figutil.FIG_ROOT, figutil.n_seg(N_EXP), "components")
    os.makedirs(out_dir, exist_ok=True)
    series = {}
    for d_mlp in DMLPS:
        pts = []
        for p in PS:
            run_id = results_io.make_run_id(model=MODEL, d_emb=DEMB, d_mlp=d_mlp,
                                            p=p, n_exp=N_EXP, method=METHOD)
            pd_obj = decomp_fit.load_decomposition(run_id, device=device)
            raw, rel = residual_frobenius(pd_obj)
            pts.append((float(p), raw, rel))
        series[d_mlp] = pts
    params = {"model": MODEL, "d_emb": DEMB, "n_exp": N_EXP, "method": METHOD}
    fig = figutil.plot_faithfulness_vs_p(series, params=params)
    out = os.path.join(out_dir,
                       f"{MODEL}_demb{DEMB}_n{results_io._fmt_n(N_EXP)}"
                       f"_{METHOD}_faithfulness_vs_p.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    figutil.plt.close(fig)
    print(f"[3/3] faithfulness-vs-p figure: {out}")


def main():
    render_per_cell()
    render_cross_p()
    render_faithfulness_vs_p()
    print(f"\nAll component figures under "
          f"{os.path.join(figutil.FIG_ROOT, figutil.n_seg(N_EXP), 'components')}")


if __name__ == "__main__":
    main()
