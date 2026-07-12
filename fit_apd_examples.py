"""Generate a few example APD decompositions with the full visualisation suite,
to sanity-check before the large PD runs. Fits APD on a spread of cached models
(sparse/dense x low/high embedding dim), saves each decomposition + metrics via
the real pipeline, and renders loss curves + component views under tag 'apdcheck'.
Model-level views for these cells are already under figures/models/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import torch

import cc
import decomp_fit
import figutil
import results_io
from mlp import EmbedMLP, LinearMLP

TAG = "apdcheck"
N_EXP = 2.0
C = 100
STEPS = 2500
results_io.set_tag(TAG)
figutil.set_tag(TAG)

# (model, kwargs, str_p) -- spread across sparsity and embedding dim.
CELLS = [
    ("EmbedMLP", {"n_features": 100, "d_emb": 1000, "d_mlp": 50}, "0.01"),  # high-dim, sparse
    ("EmbedMLP", {"n_features": 100, "d_emb": 200, "d_mlp": 50}, "0.1"),    # canonical
    ("EmbedMLP", {"n_features": 100, "d_emb": 20, "d_mlp": 50}, "0.5"),     # low-dim, dense
    ("LinearMLP", {"n_features": 100, "d_mlp": 50}, "0.1"),                 # baseline
]


def k_for(str_p):
    return max(1, min(C - 1, round(float(str_p) * 100)))


for model_name, kw, str_p in CELLS:
    cls = EmbedMLP if model_name == "EmbedMLP" else LinearMLP
    run = cc.load_or_train(str_p, cls, kw, seed=42, n_exp=N_EXP, device="cpu")
    model, dataset = run["model"], run["dataset"]
    demb = kw.get("d_emb")
    run_id = results_io.make_run_id(model=model_name, d_emb=demb, p=str_p,
                                    n_exp=N_EXP, method="APD")
    k = k_for(str_p)
    lk = decomp_fit.loss_kwargs_for("APD", k=k, lp=1.0)

    pd_obj = decomp_fit.build_pd("APD", model, C)
    curves = decomp_fit.fit_decomposition(pd_obj, dataset, steps=STEPS, loss_kwargs=lk,
                                          batch_size=512, seed=42, progress=False, desc=run_id)
    ckpt_meta = {"str_p": str_p, "model_class": model_name,
                 "model_kwargs": kw, "n_exp": N_EXP, "seed": 42}
    metrics = decomp_fit.save_decomposition(run_id, pd_obj, ckpt_meta, "APD")
    config = {"model": model_name, "d_emb": demb, "p": str_p, "n_exp": N_EXP,
              "method": "APD", "C": C}
    results_io.append_component_metrics(run_id, config, metrics)

    ntok = results_io._fmt_n(N_EXP)
    fig = figutil.plot_loss_curves(curves, method="APD")
    figutil.save_fig(fig, os.path.join("loss_curves", "APD"), f"{run_id}_loss.png",
                     "APD loss curves", n_group=N_EXP,
                     method="APD", k=k, d_emb=demb, p=str_p, n_exp=N_EXP)

    params = {"model": model_name, "d_emb": demb, "p": str_p, "n_exp": N_EXP,
              "method": "APD", "C": C, "k": k, "rank": getattr(pd_obj, "rank", None)}
    xb, _ = dataset.generate_batch(1024)
    with torch.no_grad():
        tb = model(xb)
    usage = pd_obj.usage_stats(xb, tb, k)
    figs = figutil.plot_components(pd_obj, top=16, params=params, usage=usage)
    figutil.save_component_figs(figs, run_id, n_group=N_EXP, k=k,
                                rank=getattr(pd_obj, "rank", None))

    print(f"{run_id}: mean_mono={metrics['mean_monosemanticity']:.3f} "
          f"used={metrics['components_used']}/{C} covered={metrics['features_covered']} "
          f"final_total={curves['total'][-1]:.3e}")

print("APD_EXAMPLES_DONE")
