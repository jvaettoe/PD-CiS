"""Regenerate all derived figures straight from the topical results tree's curve
JSONs + summary.csv -- no pod / no models needed (figures are derived data).

Produces, under figures/n<n>/:
  - loss_curves/<method>/...   one per run (from each curve JSON)
  - sweep_summary/...          final-loss-vs-d_emb, per (method, n_exp), from summary.csv
  - composites/...             per (model, d_emb, d_mlp, n_exp) cell, all methods
                               present (delegates to plot_prelim.py)

Model-view figures are NOT regenerated here -- they need the trained model, not
just curves. Rerun the sweep (or a model-figure pass) once checkpoints are local.

  ./.venv/bin/python regen_figures.py            # all exponents
  ./.venv/bin/python regen_figures.py --n 4      # only results/n4
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")

import results_io
import figutil


def _demb_tok(d_emb) -> str:
    if d_emb in (None, "", "NA"):
        return "NA"
    return str(int(d_emb)) if str(d_emb).replace(".", "").isdigit() else str(d_emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=None, dest="n_exp",
                    help="restrict to one loss exponent (default: every results/n*)")
    args = ap.parse_args()

    glob_root = (results_io._n_dir(args.n_exp) if args.n_exp is not None
                 else os.path.join(results_io.RESULTS_DIR, "n*"))
    curve_files = sorted(glob.glob(os.path.join(glob_root, "curves", "*.json")))
    if not curve_files:
        sys.exit(f"No curve JSONs under {glob_root}/curves — nothing to do.")

    # ---- 1. per-run loss-curve figures ----
    cells = {}  # (model, demb_tok, d_mlp, ntok) -> set(methods)
    n_loss = 0
    for f in curve_files:
        blob = json.load(open(f))
        cfg, curves = blob["config"], blob["curves"]
        model, method = cfg["model"], cfg["method"]
        demb, dmlp, p = cfg.get("d_emb"), cfg.get("d_mlp"), cfg["p"]
        ntok = results_io._fmt_n(float(cfg["n_exp"]))
        k = cfg.get("k")
        dtok = _demb_tok(demb)

        fig = figutil.plot_loss_curves(curves, method=method)
        k_tok = f"_k{k}" if k not in (None, "") and method in ("APD", "OracleAPD") else ""
        figutil.save_fig(
            fig, os.path.join("loss_curves", method),
            f"{model}_demb{dtok}_dmlp{dmlp}_p{p}_n{ntok}_{method}{k_tok}_loss.png",
            f"{method} loss curves",
            n_group=ntok,
            model=model, d_emb=demb, d_mlp=dmlp, p=p, n_exp=cfg["n_exp"],
            method=method, C=cfg.get("C"), k=k,
        )
        n_loss += 1
        cells.setdefault((model, dtok, dmlp, ntok), set()).add(method)
    print(f"[1/3] per-run loss-curve PNGs: {n_loss}")

    # ---- 2. sweep-summary figures (final loss vs d_emb, per method x n_exp) ----
    ntoks = sorted({nt for (_, _, _, nt) in cells}, key=float)
    methods = sorted({m for s in cells.values() for m in s})
    n_sweep = 0
    for ntok in ntoks:
        summary = results_io.load_summary(ntok)
        for method in methods:
            fig = figutil.plot_sweep_summary(summary, method=method, n_exp=float(ntok))
            figutil.save_fig(
                fig, "sweep_summary",
                f"final_total_vs_demb__method-{method}_n-{ntok}.png",
                f"{method}: final total loss vs d_emb",
                n_group=ntok,
                method=method, n_exp=ntok,
            )
            n_sweep += 1
    print(f"[2/3] sweep-summary PNGs: {n_sweep}")

    # ---- 3. per-cell composites (delegate to plot_prelim.py) ----
    n_comp = 0
    for (model, dtok, dmlp, ntok), meths in sorted(cells.items(), key=lambda kv: str(kv[0])):
        cmd = [sys.executable, "plot_prelim.py",
               "--model", model, "--d-mlp", str(dmlp), "--n", ntok,
               "--methods", *sorted(meths)]
        if model != "LinearMLP":
            cmd += ["--demb", dtok]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            n_comp += 1
        else:
            print(f"   composite {model} demb={dtok} dmlp={dmlp} n={ntok} "
                  f"FAILED: {r.stderr.strip()[:200]}")
    print(f"[3/3] composite PNGs: {n_comp}")
    print(f"\nAll figures under {figutil.FIG_ROOT}")


if __name__ == "__main__":
    main()
