"""Sweep driver: for each (n_exp, p, d_emb) train an MLP, fit APD/OracleAPD/SPD,
record the decomposition losses, and render param-stamped figures.

Resumable: trained models are cached by cc.load_or_train; each decomposition is
skipped when its results are already on disk (results_io.is_done). Safe to
interrupt and re-run. Run a fast check first with `--smoke`.

  ./.venv/bin/python sweep.py --smoke                 # seconds, tiny sub-grid
  ./.venv/bin/python sweep.py --methods APD SPD       # cheap methods first
  ./.venv/bin/python sweep.py --methods OracleAPD     # the slow one, last
"""

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")  # headless: render figures without a display

import cc
from mlp import LinearMLP, EmbedMLP
import decomp_fit
import results_io
import figutil

# ---- fixed sweep configuration ----
N_FEATURES = 100
D_MLP = 50            # hidden width < n_features: the compressed-computation regime
SEED = 42
C = 100              # decomposition components (~one per feature)
K = 50               # APD/OracleAPD top-k active components; must be < C
LP = 1.0             # APD singular-value (simplicity) exponent

N_EXPS = [2, 2.5, 3, 4]
PS = ["0.001", "0.01", "0.1", "0.5", "0.8"]     # kept as strings for keying
D_EMBS = [1000, 200, 100, 50, 20]

# Smoke overrides: a tiny sub-grid + reduced steps, for a seconds-long check.
SMOKE_N_EXPS = [2]
SMOKE_PS = ["0.1"]
SMOKE_D_EMBS = [20, 100]
SMOKE_STEPS = {"APD": 30, "OracleAPD": 15, "SPD": 30}
SMOKE_BASE_STEPS = 200


def _demb_tok(d_emb: int | None) -> str:
    return "NA" if d_emb is None else str(d_emb)


def iter_model_specs(smoke: bool = False):
    """Yield one spec dict per model to train: EmbedMLP over n_exp x p x d_emb,
    plus a LinearMLP baseline over n_exp x p (d_emb inapplicable -> None)."""
    n_exps = SMOKE_N_EXPS if smoke else N_EXPS
    ps = SMOKE_PS if smoke else PS
    d_embs = SMOKE_D_EMBS if smoke else D_EMBS

    for n_exp, str_p, d_emb in itertools.product(n_exps, ps, d_embs):
        yield {
            "model": "EmbedMLP", "model_cls": EmbedMLP, "d_emb": d_emb,
            "model_kwargs": {"n_features": N_FEATURES, "d_emb": d_emb, "d_mlp": D_MLP},
            "str_p": str_p, "n_exp": n_exp,
        }
    for n_exp, str_p in itertools.product(n_exps, ps):
        yield {
            "model": "LinearMLP", "model_cls": LinearMLP, "d_emb": None,
            "model_kwargs": {"n_features": N_FEATURES, "d_mlp": D_MLP},
            "str_p": str_p, "n_exp": n_exp,
        }


def _emit_model_figures(model, spec):
    """Feature-space views of the trained model (method-independent, one set per
    model). Filenames + titles carry the model params."""
    is_embed = spec["model"] == "EmbedMLP"
    figs = figutil.plot_model_views(model, is_embed=is_embed)
    demb = _demb_tok(spec["d_emb"])
    n_tok = results_io._fmt_n(spec["n_exp"])
    for view, fig in figs.items():
        fname = f"{spec['model']}_demb{demb}_p{spec['str_p']}_n{n_tok}_{view}.png"
        figutil.save_fig(
            fig, os.path.join("model_views", spec["model"]), fname,
            f"{spec['model']} {view} (feature space)",
            model=spec["model"], n_features=N_FEATURES, d_emb=spec["d_emb"],
            p=spec["str_p"], n_exp=spec["n_exp"],
        )


def _run_method(model, dataset, spec, method, *, smoke, force):
    """Fit one decomposition for one trained model; log curves + summary rows +
    a loss-curve figure. Skips if already done (unless force)."""
    d_emb = spec["d_emb"]
    run_id = results_io.make_run_id(
        model=spec["model"], d_emb=d_emb, p=spec["str_p"],
        n_exp=spec["n_exp"], method=method,
    )
    if results_io.is_done(run_id) and not force:
        print(f"  [skip] {run_id}")
        return
    if force:
        results_io.delete_summary_rows(run_id)

    steps = SMOKE_STEPS[method] if smoke else decomp_fit.METHOD_STEPS[method]
    batch = decomp_fit.ORACLE_BATCH if method == "OracleAPD" else decomp_fit.DEFAULT_BATCH
    k = K if method in ("APD", "OracleAPD") else None
    loss_kwargs = decomp_fit.loss_kwargs_for(method, k=K, lp=LP)

    print(f"  [fit ] {run_id}  (steps={steps}, batch={batch})")
    pd_obj = decomp_fit.build_pd(method, model, C)
    curves = decomp_fit.fit_decomposition(
        pd_obj, dataset, steps=steps, loss_kwargs=loss_kwargs,
        lr=decomp_fit.DEFAULT_LR, batch_size=batch, seed=SEED, desc=run_id,
    )
    finals = {term: values[-1] for term, values in curves.items()}

    config = {
        "run_id": run_id, "model": spec["model"], "n_features": N_FEATURES,
        "d_emb": d_emb, "d_mlp": D_MLP, "p": spec["str_p"], "n_exp": spec["n_exp"],
        "method": method, "C": C, "k": k, "steps": steps,
        "lr": decomp_fit.DEFAULT_LR, "batch_size": batch, "seed": SEED,
        "loss_kwargs": loss_kwargs,
    }
    results_io.write_curve_json(run_id, config, curves)
    results_io.append_summary_rows(run_id, config, finals)

    demb = _demb_tok(d_emb)
    n_tok = results_io._fmt_n(spec["n_exp"])
    fig = figutil.plot_loss_curves(curves, method=method)
    figutil.save_fig(
        fig, os.path.join("loss_curves", method),
        f"{spec['model']}_demb{demb}_p{spec['str_p']}_n{n_tok}_{method}_loss.png",
        f"{method} loss curves",
        model=spec["model"], d_emb=d_emb, p=spec["str_p"], n_exp=spec["n_exp"],
        method=method, C=C, k=k,
    )


def run_one_model(spec, *, methods, base_steps, device, force, smoke):
    tag = f"{spec['model']} d_emb={spec['d_emb']} p={spec['str_p']} n={spec['n_exp']}"
    print(f"[model] {tag}")
    run = cc.load_or_train(
        spec["str_p"], spec["model_cls"], spec["model_kwargs"],
        seed=SEED, base_steps=base_steps, device=device, n_exp=spec["n_exp"],
    )
    model, dataset = run["model"], run["dataset"]
    _emit_model_figures(model, spec)
    for method in methods:
        _run_method(model, dataset, spec, method, smoke=smoke, force=force)


def _emit_sweep_summaries(methods, smoke):
    """Sweep-level figures: final loss vs d_emb, one series per p, per (method,
    n_exp). Regenerated from the full summary.csv each time."""
    summary = results_io.load_summary()
    n_exps = SMOKE_N_EXPS if smoke else N_EXPS
    for method in methods:
        for n_exp in n_exps:
            fig = figutil.plot_sweep_summary(summary, method=method, n_exp=n_exp)
            n_tok = results_io._fmt_n(n_exp)
            figutil.save_fig(
                fig, "sweep_summary",
                f"final_total_vs_demb__method-{method}_n-{n_tok}.png",
                f"{method}: final total loss vs d_emb",
                method=method, n_exp=n_exp, n_features=N_FEATURES,
            )


def main(*, smoke=False, methods=decomp_fit.METHODS, device="cpu",
         base_steps=10_000, force=False):
    assert K < C, f"K ({K}) must be < C ({C}) or minimality is trivial"
    if smoke:
        base_steps = SMOKE_BASE_STEPS
    specs = list(iter_model_specs(smoke))
    print(f"===== decomposition sweep ({'SMOKE' if smoke else 'FULL'}) =====")
    print(f"  models={len(specs)}  methods={list(methods)}  device={device}  "
          f"C={C} K={K}\n")
    for spec in specs:
        run_one_model(spec, methods=methods, base_steps=base_steps,
                      device=device, force=force, smoke=smoke)
    _emit_sweep_summaries(methods, smoke)
    print(f"\ndone. results -> {results_io.summary_path()} ; figures -> {figutil.FIG_ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny sub-grid, reduced steps")
    ap.add_argument("--methods", nargs="+", default=list(decomp_fit.METHODS),
                    choices=decomp_fit.METHODS, help="which decompositions to fit")
    ap.add_argument("--device", default="cpu", help="torch device (cpu recommended)")
    ap.add_argument("--base-steps", type=int, default=10_000,
                    help="MLP training steps before p-scaling")
    ap.add_argument("--force", action="store_true",
                    help="re-run and replace completed decompositions")
    args = ap.parse_args()
    main(smoke=args.smoke, methods=args.methods, device=args.device,
         base_steps=args.base_steps, force=args.force)
