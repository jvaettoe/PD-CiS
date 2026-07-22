"""Render the component views for a SAVED decomposition (results/n<n>/decomps/
<run_id>.pt, written by the sweep). Feature-space per-component maps + a
component->feature assignment heatmap (norm-sorted and permutation-sorted) +
singular-value spectrum, plus scalar purity/coverage metrics.

  ./.venv/bin/python plot_components.py --demb 200 --d-mlp 100 --p 0.1 --n 4 --method APD
  ./.venv/bin/python plot_components.py --run EmbedMLP_demb200_dmlp100_p0.1_n4_APD
"""
import argparse

import matplotlib
matplotlib.use("Agg")

import results_io
import figutil
import decomp_fit
from pd import APD


def build_run_id(a) -> str:
    demb = None if a.model == "MLP" else a.demb
    return results_io.make_run_id(model=a.model, d_emb=demb, d_mlp=a.d_mlp, p=a.p,
                                  n_exp=a.n_exp, method=a.method)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="explicit run_id (else built from the flags below)")
    ap.add_argument("--model", default="EmbedMLP")
    ap.add_argument("--demb", type=int, default=200)
    ap.add_argument("--d-mlp", type=int, default=50, dest="d_mlp",
                    help="hidden width (first-class in the run_id)")
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--n", type=float, default=2.0, dest="n_exp")
    ap.add_argument("--method", default="APD", choices=["APD", "OracleAPD", "SPD", "ExactPD", "VPD"])
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run_id = a.run or build_run_id(a)
    pd_obj = decomp_fit.load_decomposition(run_id, device=a.device)

    try:
        info = results_io.parse_run_id(run_id)
    except ValueError:
        info = {}
    # k = round(p * n_features) is a pure function of p (APD/OracleAPD/ExactPD, which
    # all use batch-top-k; SPD has none)
    k = None
    if info.get("p") and info.get("method") in ("APD", "OracleAPD", "ExactPD"):
        k = max(1, min(pd_obj.C - 1, round(float(info["p"]) * 100)))
    params = {**info, "C": pd_obj.C, "k": k, "rank": getattr(pd_obj, "rank", None)}
    # behavioural usage: the APD family exposes batch-top-k attributions (SPD does not)
    usage = None
    if k is not None and isinstance(pd_obj, APD):
        usage = pd_obj.sample_usage_stats(float(info["p"]), k)

    figs = pd_obj.plot_components(top=a.top, params=params, usage=usage)
    figs["assignment_sorted"] = pd_obj.plot_component_assignment_sorted(params=params)

    ntok = params.get("n_exp")
    out_dir = figutil.save_component_figs(figs, run_id, n_group=ntok,
                                          k=params.get("k"), rank=params.get("rank"))

    m = pd_obj.component_metrics()
    print(f"{run_id}")
    print(f"  components active (feature space): {m['components_used']}/{pd_obj.C}")
    print(f"  features covered (owned by >=1):   {m['features_covered']}/{pd_obj.model.n_features}")
    print(f"  mean monosemanticity:             {m['mean_monosemanticity']:.3f}  (1.0 = each used component sits on one feature)")
    print(f"  mean diagonal share:              {m['mean_diagonal_share']:.3f}  (1.0 = pure self-maps; low = off-diagonal/superposed)")
    print(f"  figures -> {out_dir}")


if __name__ == "__main__":
    main()
