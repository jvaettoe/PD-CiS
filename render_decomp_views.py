"""Render the component-visualisation suite for the saved n=4 decompositions
of EmbedMLP (d_emb=200) at BOTH hidden widths, across all sparsities p, so you can
see the components and how they change as p changes.

Defaults to APD; pass --method to run the same suite for any fitted method (e.g.
--method ExactPD for the handcoded per-feature ground truth). Every output filename
carries the method, so methods never overwrite one another.

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


def demb_arg():
    """Embedding dim for make_run_id: None for LinearMLP (its run_ids have no d_emb),
    else DEMB. Keeps this suite runnable for either architecture."""
    return None if MODEL == "LinearMLP" else DEMB


def demb_tok() -> str:
    """Filename token for the embedding dim: 'NA' for LinearMLP, else the number."""
    return "NA" if MODEL == "LinearMLP" else str(DEMB)


def components_dir() -> str:
    """Per-method output dir: figures/n<n>/components/<method>/ (matches the
    loss_curves/<method>/ layout so each PD method's figures stay separate)."""
    d = os.path.join(figutil.FIG_ROOT, figutil.n_seg(N_EXP), "components", METHOD)
    os.makedirs(d, exist_ok=True)
    return d


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


def behavioural_faithfulness(pd_obj, p: str, device="cpu", batch=4096):
    """Output-space faithfulness of the FULL decomposition: run the model's forward
    with every decomposed weight replaced by the sum of ALL C components
    (pd_obj.forward_dense) and compare to the model's own output on a fresh batch.
    Returns (out_mse, rel_out_err). Fresh compute (seed=0), not the noisy last
    training minibatch. Distinct from residual_frobenius (weight-space effective map)."""
    ds = CleanDataset(n_features=pd_obj.model.n_features, p=float(p),
                      device=device, seed=0)
    xb, _ = ds.generate_batch(batch)
    with torch.no_grad():
        target = pd_obj.model(xb)
        pred = pd_obj.forward_dense(xb)
        diff = (pred - target)
        out_mse = float((diff ** 2).mean().item())
        rel = float((diff.norm() / (target.norm() + 1e-12)).item())
    return out_mse, rel


def build_usage(pd_obj, p: str, k: int, device="cpu"):
    """Batch-top-k behavioural usage on a fresh batch (same recipe as
    plot_components). Only APD/OracleAPD/ExactPD expose usage_stats; SPD (stochastic
    gating, no k) has none, so return None and let the usage panel be omitted."""
    if not hasattr(pd_obj, "usage_stats"):
        return None
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
            # --demb is a valid int (ignored for LinearMLP: build_run_id maps it to None)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                n_ok += 1
            else:
                print(f"  [FAIL] d_mlp={d_mlp} p={p}: {r.stderr.strip()[:300]}")
    print(f"[1/2] per-cell suites rendered: {n_ok}/{len(DMLPS) * len(PS)}")


def render_cross_p(device="cpu"):
    out_dir = components_dir()
    for d_mlp in DMLPS:
        entries = []
        for p in PS:
            run_id = results_io.make_run_id(model=MODEL, d_emb=demb_arg(), d_mlp=d_mlp,
                                            p=p, n_exp=N_EXP, method=METHOD)
            pd_obj = decomp_fit.load_decomposition(run_id, device=device)
            k = k_for(p, pd_obj.C)
            entries.append({"p": p, "k": k, "pd_obj": pd_obj,
                            "usage": build_usage(pd_obj, p, k, device)})
        params = {"model": MODEL, "d_emb": demb_arg(), "d_mlp": d_mlp,
                  "n_exp": N_EXP, "method": METHOD, "C": entries[0]["pd_obj"].C}
        fig = figutil.plot_components_vs_p(entries, params=params)
        out = os.path.join(out_dir,
                           f"{MODEL}_demb{demb_tok()}_dmlp{d_mlp}_n{results_io._fmt_n(N_EXP)}"
                           f"_{METHOD}_vs_p.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        figutil.plt.close(fig)
        print(f"  saved {out}")
    print(f"[2/2] cross-p figures: {len(DMLPS)}")


def render_faithfulness_vs_p(device="cpu"):
    """One figure, both widths: effective-map Frobenius residual vs p."""
    out_dir = components_dir()
    series = {}
    for d_mlp in DMLPS:
        pts = []
        for p in PS:
            run_id = results_io.make_run_id(model=MODEL, d_emb=demb_arg(), d_mlp=d_mlp,
                                            p=p, n_exp=N_EXP, method=METHOD)
            pd_obj = decomp_fit.load_decomposition(run_id, device=device)
            raw, rel = residual_frobenius(pd_obj)
            pts.append((float(p), raw, rel))
        series[d_mlp] = pts
    params = {"model": MODEL, "d_emb": demb_arg(), "n_exp": N_EXP, "method": METHOD}
    fig = figutil.plot_faithfulness_vs_p(series, params=params)
    out = os.path.join(out_dir,
                       f"{MODEL}_demb{demb_tok()}_n{results_io._fmt_n(N_EXP)}"
                       f"_{METHOD}_faithfulness_vs_p.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    figutil.plt.close(fig)
    print(f"[3/3] faithfulness-vs-p figure: {out}")


def render_behavioural_faithfulness_vs_p(device="cpu"):
    """One figure, both widths: output-space error of the Σ-all-components forward
    vs the model, as a function of p."""
    out_dir = components_dir()
    series = {}
    for d_mlp in DMLPS:
        pts = []
        for p in PS:
            run_id = results_io.make_run_id(model=MODEL, d_emb=demb_arg(), d_mlp=d_mlp,
                                            p=p, n_exp=N_EXP, method=METHOD)
            pd_obj = decomp_fit.load_decomposition(run_id, device=device)
            mse, rel = behavioural_faithfulness(pd_obj, p, device)
            pts.append((float(p), mse, rel))
        series[d_mlp] = pts
    params = {"model": MODEL, "d_emb": demb_arg(), "n_exp": N_EXP, "method": METHOD}
    fig = figutil.plot_behavioural_faithfulness_vs_p(series, params=params)
    out = os.path.join(out_dir,
                       f"{MODEL}_demb{demb_tok()}_n{results_io._fmt_n(N_EXP)}"
                       f"_{METHOD}_behavioural_faithfulness_vs_p.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    figutil.plt.close(fig)
    print(f"[4/4] behavioural-faithfulness-vs-p figure: {out}")


def render_losses_vs_p():
    """One figure: the APD loss terms (faithful/minimal/simple/total) vs p, one line
    per width, straight from the tidy summary (no decomp reload needed)."""
    out_dir = components_dir()
    summary = results_io.load_summary(N_EXP)
    fig = figutil.plot_losses_vs_p(summary, method=METHOD, n_exp=N_EXP,
                                   model=MODEL, d_emb=demb_arg())
    out = os.path.join(out_dir,
                       f"{MODEL}_demb{demb_tok()}_n{results_io._fmt_n(N_EXP)}"
                       f"_{METHOD}_losses_vs_p.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    figutil.plt.close(fig)
    print(f"[5/5] {METHOD}-losses-vs-p figure: {out}")


def main():
    render_per_cell()
    render_cross_p()
    render_faithfulness_vs_p()
    render_behavioural_faithfulness_vs_p()
    render_losses_vs_p()
    print(f"\nAll component figures under {components_dir()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=METHOD,
                    choices=["APD", "OracleAPD", "SPD", "ExactPD"],
                    help="which fitted decomposition to visualise (default APD)")
    ap.add_argument("--model", default=MODEL, choices=["EmbedMLP", "LinearMLP"],
                    help="model class (default EmbedMLP; LinearMLP has no d_emb)")
    ap.add_argument("--demb", type=int, default=DEMB, help="embedding dim (default 200)")
    ap.add_argument("--n", type=float, default=N_EXP, dest="n_exp",
                    help="loss exponent / topical group (default 4)")
    ap.add_argument("--dmlps", type=int, nargs="+", default=DMLPS,
                    help="hidden widths to render (default 50 100)")
    a = ap.parse_args()

    # Reassign the module-level config the render functions read at call time.
    METHOD, MODEL, DEMB, N_EXP, DMLPS = a.method, a.model, a.demb, a.n_exp, a.dmlps
    main()
