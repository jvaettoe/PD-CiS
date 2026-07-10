"""Render the component views for a SAVED decomposition (results/<tag>/decomps/
<run_id>.pt, written by the sweep). Feature-space per-component maps + a
component->feature assignment heatmap (norm-sorted and permutation-sorted) +
singular-value spectrum, plus scalar purity/coverage metrics.

  ./.venv/bin/python plot_components.py --tag gpu --demb 200 --p 0.1 --n 4 --method APD
  ./.venv/bin/python plot_components.py --tag gpu --run EmbedMLP_demb200_p0.1_n4_APD
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import results_io
import figutil
import decomp_fit


def _fmt_n(n: float) -> str:
    return f"{float(n):g}"


def build_run_id(a) -> str:
    demb = "NA" if a.model == "LinearMLP" else str(a.demb)
    return f"{a.model}_demb{demb}_p{a.p}_n{_fmt_n(a.n_exp)}_{a.method}"


def permutation_order(assign: np.ndarray) -> np.ndarray:
    """Row order that clusters components by the feature they most own, so a
    clean (permutation-like) decomposition lands on the diagonal."""
    peak = assign.max(axis=1)
    argf = assign.argmax(axis=1)
    return np.lexsort((-peak, argf))   # primary: owned feature; tiebreak: peak


def purity_metrics(assign: np.ndarray) -> dict:
    total = assign.sum(axis=1) + 1e-12
    peak = assign.max(axis=1)
    used = total > 1e-6 * float(total.max())
    mono = (peak / total)[used]                        # top-feature share per used comp
    covered = len(set(assign.argmax(axis=1)[used].tolist()))
    return {"components_used": int(used.sum()),
            "features_covered": covered,
            "mean_monosemanticity": float(mono.mean()) if used.any() else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gpu")
    ap.add_argument("--run", default=None, help="explicit run_id (else built from the flags below)")
    ap.add_argument("--model", default="EmbedMLP")
    ap.add_argument("--demb", type=int, default=200)
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--n", type=float, default=2.0, dest="n_exp")
    ap.add_argument("--method", default="APD", choices=["APD", "OracleAPD", "SPD"])
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    results_io.set_tag(a.tag)
    figutil.set_tag(a.tag)
    run_id = a.run or build_run_id(a)
    pd_obj = decomp_fit.load_decomposition(run_id, device=a.device)

    figs = figutil.plot_components(pd_obj, top=a.top)

    # extra: permutation-sorted assignment (see docstring)
    assign = pd_obj.component_feature_assignment().cpu().numpy()
    order = permutation_order(assign)
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 6))
    im = ax.imshow(assign[order], aspect="auto", cmap="magma")
    ax.set_xlabel("feature index")
    ax.set_ylabel("component (permutation-sorted)")
    ax.set_title("component -> feature assignment (permutation-sorted)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figs["assignment_sorted"] = fig

    out_dir = os.path.join(figutil.FIG_ROOT, "components", run_id)
    os.makedirs(out_dir, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    m = purity_metrics(assign)
    print(f"{run_id}")
    print(f"  components used:               {m['components_used']}/{pd_obj.C}")
    print(f"  features covered (owned by >=1): {m['features_covered']}/{assign.shape[1]}")
    print(f"  mean monosemanticity:          {m['mean_monosemanticity']:.3f}  (1.0 = each used component sits on one feature)")
    print(f"  figures -> {out_dir}")


if __name__ == "__main__":
    main()
