"""Regenerate all derived figures for a results tag straight from its curve
JSONs + summary.csv -- no pod / no models needed (figures are derived data).

Produces, under figures/<tag>/:
  - loss_curves/<method>/...   one per run (from each curve JSON)
  - sweep_summary/...          final-loss-vs-d_emb, per (method, n_exp), from summary.csv
  - composites/...             per (model, d_emb, n_exp) cell, all methods present
                               (delegates to plot_prelim.py)

Model-view figures are NOT regenerated here -- they need the trained model, not
just curves. Rerun the sweep (or a model-figure pass) once checkpoints are local.

  ./.venv/bin/python regen_figures.py --tag gpu
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
    ap.add_argument("--tag", default="gpu", help="results/figures namespace")
    args = ap.parse_args()
    results_io.set_tag(args.tag)
    figutil.set_tag(args.tag)

    curve_files = sorted(glob.glob(os.path.join(results_io.CURVES_DIR, "*.json")))
    if not curve_files:
        sys.exit(f"No curve JSONs under results/{args.tag}/curves — nothing to do.")

    # ---- 1. per-run loss-curve figures ----
    cells = {}  # (model, demb_tok, ntok) -> set(methods)
    n_loss = 0
    for f in curve_files:
        blob = json.load(open(f))
        cfg, curves = blob["config"], blob["curves"]
        model, method = cfg["model"], cfg["method"]
        demb, p = cfg.get("d_emb"), cfg["p"]
        ntok = results_io._fmt_n(float(cfg["n_exp"]))
        k = cfg.get("k")
        dtok = _demb_tok(demb)

        fig = figutil.plot_loss_curves(curves, method=method)
        k_tok = f"_k{k}" if k not in (None, "") and method in ("APD", "OracleAPD") else ""
        figutil.save_fig(
            fig, os.path.join("loss_curves", method),
            f"{model}_demb{dtok}_p{p}_n{ntok}_{method}{k_tok}_loss.png",
            f"{method} loss curves",
            model=model, d_emb=demb, p=p, n_exp=cfg["n_exp"],
            method=method, C=cfg.get("C"), k=k,
        )
        n_loss += 1
        cells.setdefault((model, dtok, ntok), set()).add(method)
    print(f"[1/3] per-run loss-curve PNGs: {n_loss}")

    # ---- 2. sweep-summary figures (final loss vs d_emb, per method x n_exp) ----
    summary = results_io.load_summary()
    methods = sorted({m for s in cells.values() for m in s})
    ntoks = sorted({nt for (_, _, nt) in cells}, key=float)
    n_sweep = 0
    for method in methods:
        for ntok in ntoks:
            fig = figutil.plot_sweep_summary(summary, method=method, n_exp=float(ntok))
            figutil.save_fig(
                fig, "sweep_summary",
                f"final_total_vs_demb__method-{method}_n-{ntok}.png",
                f"{method}: final total loss vs d_emb",
                method=method, n_exp=ntok,
            )
            n_sweep += 1
    print(f"[2/3] sweep-summary PNGs: {n_sweep}")

    # ---- 3. per-cell composites (delegate to plot_prelim.py) ----
    n_comp = 0
    for (model, dtok, ntok), meths in sorted(cells.items()):
        cmd = [sys.executable, "plot_prelim.py", "--tag", args.tag,
               "--model", model, "--n", ntok,
               "--methods", *sorted(meths)]
        if model != "LinearMLP":
            cmd += ["--demb", dtok]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            n_comp += 1
        else:
            print(f"   composite {model} demb={dtok} n={ntok} FAILED: {r.stderr.strip()[:200]}")
    print(f"[3/3] composite PNGs: {n_comp}")
    print(f"\nAll figures under {figutil.FIG_ROOT}")


if __name__ == "__main__":
    main()
