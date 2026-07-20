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
SMOKE_STEPS = {"APD": 30, "OracleAPD": 15, "SPD": 30, "ExactPD": 1, "VPD": 30}
SMOKE_BASE_STEPS = 200


def _demb_tok(d_emb: int | None) -> str:
    return "NA" if d_emb is None else str(d_emb)


def iter_model_specs(smoke: bool = False, n_exps=None, drop_demb=None, d_mlp=D_MLP,
                     no_linear=False):
    """Yield one spec dict per model to train: EmbedMLP over n_exp x p x d_emb,
    plus a LinearMLP baseline over n_exp x p (d_emb inapplicable -> None).
    n_exps overrides the exponent list AND its order (used to reprioritise which
    rows run first); drop_demb excludes those embedding dims (e.g. the expensive
    d_emb=1000 in a fast preliminary); d_mlp overrides the hidden width (now
    first-class in the run_id, so distinct widths never collide -- no --tag needed).
    Everything is resumable so done cells skip."""
    if n_exps is None:
        n_exps = SMOKE_N_EXPS if smoke else N_EXPS
    ps = SMOKE_PS if smoke else PS
    d_embs = SMOKE_D_EMBS if smoke else D_EMBS
    if drop_demb:
        d_embs = [d for d in d_embs if d not in set(drop_demb)]

    for n_exp, str_p, d_emb in itertools.product(n_exps, ps, d_embs):
        yield {
            "model": "EmbedMLP", "model_cls": EmbedMLP, "d_emb": d_emb,
            "model_kwargs": {"n_features": N_FEATURES, "d_emb": d_emb, "d_mlp": d_mlp},
            "str_p": str_p, "n_exp": n_exp,
        }
    if no_linear:
        return
    for n_exp, str_p in itertools.product(n_exps, ps):
        yield {
            "model": "LinearMLP", "model_cls": LinearMLP, "d_emb": None,
            "model_kwargs": {"n_features": N_FEATURES, "d_mlp": d_mlp},
            "str_p": str_p, "n_exp": n_exp,
        }


def _emit_model_figures(model, spec):
    """Feature-space views of the trained model (method-independent, one set per
    model). Filenames + titles carry the model params."""
    figs = model.plot_views()
    demb = _demb_tok(spec["d_emb"])
    d_mlp = spec["model_kwargs"]["d_mlp"]
    n_tok = results_io._fmt_n(spec["n_exp"])
    for view, fig in figs.items():
        fname = f"{spec['model']}_demb{demb}_dmlp{d_mlp}_p{spec['str_p']}_n{n_tok}_{view}.png"
        figutil.save_fig(
            fig, os.path.join("model_views", spec["model"]), fname,
            f"{spec['model']} {view} (feature space)",
            n_group=spec["n_exp"],
            model=spec["model"], n_features=N_FEATURES, d_emb=spec["d_emb"],
            d_mlp=d_mlp, p=spec["str_p"], n_exp=spec["n_exp"],
        )


def k_for(str_p: str) -> int:
    """Batch-top-k budget = expected number of active features per example =
    round(p * n_features), clamped to [1, C-1] (k=0 invalid, k=C trivial)."""
    return max(1, min(C - 1, round(float(str_p) * N_FEATURES)))


def _run_method(model, dataset, spec, method, *, smoke, force,
                steps_scale=1.0, batch_override=None):
    """Fit one decomposition for one trained model; log curves + summary rows +
    a loss-curve figure. Skips if already done (unless force)."""
    d_emb = spec["d_emb"]
    run_id = results_io.make_run_id(
        model=spec["model"], d_emb=d_emb, d_mlp=spec["model_kwargs"]["d_mlp"],
        p=spec["str_p"], n_exp=spec["n_exp"], method=method,
    )
    if results_io.is_done(run_id) and not force:
        print(f"  [skip] {run_id}")
        return
    if force:
        results_io.delete_summary_rows(run_id)
        results_io.delete_component_metrics_rows(run_id)

    base = SMOKE_STEPS[method] if smoke else decomp_fit.METHOD_STEPS[method]
    steps = max(1, round(base * steps_scale))
    if method == "OracleAPD":
        batch = min(decomp_fit.ORACLE_BATCH, batch_override or decomp_fit.ORACLE_BATCH)
    else:
        batch = batch_override or decomp_fit.DEFAULT_BATCH
    # APD/OracleAPD/ExactPD use batch-top-k with k = expected active features
    # (p*n_features); SPD has no k (stochastic gating).
    k = k_for(spec["str_p"]) if method in ("APD", "OracleAPD", "ExactPD") else None
    loss_kwargs = decomp_fit.loss_kwargs_for(method, k=(k or 1), lp=LP)

    print(f"  [fit ] {run_id}  (steps={steps}, batch={batch})")
    pd_obj = decomp_fit.build_pd(method, model, C)
    curves = decomp_fit.fit_decomposition(
        pd_obj, dataset, steps=steps, loss_kwargs=loss_kwargs,
        lr=decomp_fit.DEFAULT_LR, batch_size=batch, seed=SEED, desc=run_id,
    )
    finals = {term: values[-1] for term, values in curves.items()}

    config = {
        "run_id": run_id, "model": spec["model"], "n_features": N_FEATURES,
        "d_emb": d_emb, "d_mlp": spec["model_kwargs"]["d_mlp"],
        "p": spec["str_p"], "n_exp": spec["n_exp"],
        # pd_obj.C, not the module C: ExactPD forces C = n_features.
        "method": method, "C": pd_obj.C, "k": k, "steps": steps,
        "lr": decomp_fit.DEFAULT_LR, "batch_size": batch, "seed": SEED,
        "loss_kwargs": loss_kwargs,
    }
    metrics = pd_obj.component_metrics()
    results_io.write_curve_json(run_id, config, curves)
    results_io.append_summary_rows(run_id, config, finals)
    results_io.append_component_metrics(run_id, config, metrics)
    # Decomp .pt is written LAST: is_done() keys on its existence (the completion
    # sentinel), so a crash before this point re-runs the cell rather than leaving
    # a 'done' run with no .pt for load_decomposition/plot_components.
    decomp_fit.save_decomposition(
        run_id, pd_obj,
        {"str_p": spec["str_p"], "model_class": spec["model"],
         "model_kwargs": spec["model_kwargs"], "n_exp": spec["n_exp"], "seed": SEED},
        method,
    )

    demb = _demb_tok(d_emb)
    d_mlp = spec["model_kwargs"]["d_mlp"]
    n_tok = results_io._fmt_n(spec["n_exp"])
    k_tok = f"_k{k}" if k is not None else ""  # APD/OracleAPD self-identify k in the filename
    fig = figutil.plot_loss_curves(curves, method=method)
    figutil.save_fig(
        fig, os.path.join("loss_curves", method),
        f"{spec['model']}_demb{demb}_dmlp{d_mlp}_p{spec['str_p']}_n{n_tok}_{method}{k_tok}_loss.png",
        f"{method} loss curves",
        n_group=spec["n_exp"],
        model=spec["model"], d_emb=d_emb, d_mlp=d_mlp, p=spec["str_p"], n_exp=spec["n_exp"],
        method=method, C=pd_obj.C, k=k,   # pd_obj.C (ExactPD: n_features), not module C
    )


def _ckpt_exists(spec) -> bool:
    path = cc.checkpoint_path(spec["str_p"], spec["model_cls"], spec["model_kwargs"],
                              seed=SEED, n_exp=spec["n_exp"])
    return os.path.exists(path)


def run_one_model(spec, *, methods, base_steps, device, force, smoke,
                  no_train=False, steps_scale=1.0, batch_override=None):
    tag = f"{spec['model']} d_emb={spec['d_emb']} p={spec['str_p']} n={spec['n_exp']}"
    if no_train and not _ckpt_exists(spec):
        # decompose only already-cached models (safe to run alongside a
        # background --models-only trainer); the trainer will fill this in.
        print(f"[skip-uncached] {tag}")
        return
    print(f"[model] {tag}")
    run = cc.load_or_train(
        spec["str_p"], spec["model_cls"], spec["model_kwargs"],
        seed=SEED, base_steps=base_steps, device=device, n_exp=spec["n_exp"],
    )
    model, dataset = run["model"], run["dataset"]
    _emit_model_figures(model, spec)
    for method in methods:
        _run_method(model, dataset, spec, method, smoke=smoke, force=force,
                    steps_scale=steps_scale, batch_override=batch_override)


def _emit_sweep_summaries(methods, smoke, n_exps=None):
    """Sweep-level figures: final loss vs d_emb, one series per p, per (method,
    n_exp). Regenerated from each exponent's summary.csv."""
    if n_exps is None:
        n_exps = SMOKE_N_EXPS if smoke else N_EXPS
    for method in methods:
        for n_exp in n_exps:
            summary = results_io.load_summary(n_exp)
            fig = figutil.plot_sweep_summary(summary, method=method, n_exp=n_exp)
            n_tok = results_io._fmt_n(n_exp)
            figutil.save_fig(
                fig, "sweep_summary",
                f"final_total_vs_demb__method-{method}_n-{n_tok}.png",
                f"{method}: final total loss vs d_emb",
                n_group=n_exp,
                method=method, n_exp=n_exp, n_features=N_FEATURES,
            )


def main(*, smoke=False, methods=decomp_fit.METHODS, device="cpu",
         base_steps=10_000, force=False, n_exps=None, models_only=False,
         no_train=False, steps_scale=1.0, batch_override=None,
         drop_demb=None, d_mlp=D_MLP, no_linear=False):
    assert K < C, f"K ({K}) must be < C ({C}) or minimality is trivial"
    if smoke:
        base_steps = SMOKE_BASE_STEPS
    specs = list(iter_model_specs(smoke, n_exps=n_exps, drop_demb=drop_demb, d_mlp=d_mlp,
                                  no_linear=no_linear))
    order = n_exps if n_exps is not None else (SMOKE_N_EXPS if smoke else N_EXPS)
    mode = "MODELS-ONLY" if models_only else ("NO-TRAIN" if no_train else "")
    print(f"===== decomposition sweep ({'SMOKE' if smoke else 'FULL'}) =====")
    print(f"  models={len(specs)}  methods={list(methods)}  device={device}  "
          f"C={C}  n_order={list(order)}  steps_scale={steps_scale}  "
          f"batch={batch_override or 'default'}  drop_demb={drop_demb or '-'}  "
          f"d_mlp={d_mlp}  {mode}\n")

    if models_only:
        # Train + cache every model (device-agnostic); no decomposition. Warms
        # the checkpoint cache so a later GPU run only does the decompositions.
        for i, spec in enumerate(specs, 1):
            mtag = f"{spec['model']} d_emb={spec['d_emb']} p={spec['str_p']} n={spec['n_exp']}"
            print(f"[train {i}/{len(specs)}] {mtag}")
            cc.load_or_train(spec["str_p"], spec["model_cls"], spec["model_kwargs"],
                             seed=SEED, base_steps=base_steps, device=device,
                             n_exp=spec["n_exp"])
        print(f"\nmodels cached -> {cc.CKPT_DIR}")
        return

    for spec in specs:
        run_one_model(spec, methods=methods, base_steps=base_steps,
                      device=device, force=force, smoke=smoke, no_train=no_train,
                      steps_scale=steps_scale, batch_override=batch_override)
    _emit_sweep_summaries(methods, smoke, n_exps=n_exps)
    print(f"\ndone. results -> {results_io.RESULTS_DIR}/n*/ ; figures -> {figutil.FIG_ROOT}")


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
    ap.add_argument("--n", nargs="+", type=float, default=None, dest="n_exps",
                    help="loss exponents to run, in priority order "
                         "(e.g. --n 2.5 4 3 2); default is the full grid order")
    ap.add_argument("--models-only", action="store_true", dest="models_only",
                    help="train + cache all models, run no decompositions "
                         "(warms the cache for a later GPU decomposition run)")
    ap.add_argument("--no-train", action="store_true", dest="no_train",
                    help="only decompose already-cached models; skip uncached "
                         "(safe to run alongside a background --models-only trainer)")
    ap.add_argument("--steps-scale", type=float, default=1.0, dest="steps_scale",
                    help="scale every method's fit steps (e.g. 0.2 for a fast preliminary)")
    ap.add_argument("--batch", type=int, default=None, dest="batch_override",
                    help="override decomposition batch size (APD/SPD; smaller = faster batch-top-k)")
    ap.add_argument("--drop-demb", nargs="+", type=int, default=None, dest="drop_demb",
                    help="exclude these embedding dims (e.g. --drop-demb 1000 for preliminary)")
    ap.add_argument("--d-mlp", type=int, default=D_MLP, dest="d_mlp",
                    help=f"hidden width (default {D_MLP}); first-class in the run_id, so "
                         f"distinct widths land in distinct files (e.g. --d-mlp 100)")
    ap.add_argument("--no-linear", action="store_true", dest="no_linear",
                    help="skip the LinearMLP baselines (decompose only EmbedMLP cells)")
    args = ap.parse_args()
    main(smoke=args.smoke, methods=args.methods, device=args.device,
         base_steps=args.base_steps, force=args.force, n_exps=args.n_exps,
         models_only=args.models_only, no_train=args.no_train,
         steps_scale=args.steps_scale, batch_override=args.batch_override,
         drop_demb=args.drop_demb, d_mlp=args.d_mlp, no_linear=args.no_linear)
