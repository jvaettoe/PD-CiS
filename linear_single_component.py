"""ExactPD single-component vs full-model one-hot response on the trained LinearMLP.

Same analysis + viz suite as biregular_single_component.py, driven on the fully-trained
LinearMLP (d_mlp=50, n_exp=4.0 checkpoints) instead of the analytic biregular codes.
ExactPD gives one rank-1 component per feature; on the one-hot probe e_c the single
component c and the full model share the hidden activation relu(w_in[c,:]) so they agree
exactly on-target and differ only off-target — the interference the trained superposition
introduces. Outputs use linear_* filename stems (figures/overview, results/overview).

  ./.venv/bin/python linear_single_component.py                 # p=0.001 loss bars etc.
  ./.venv/bin/python linear_single_component.py --overview      # p-sweep over all PS
  ./.venv/bin/python linear_single_component.py --sf-loss       # loss spectrum + on/off profile
  ./.venv/bin/python linear_single_component.py --io-spectrum   # raw-output spectrum view
"""
import argparse

from biregular_single_component import (_analyse_model, plot_loss_bars, plot_io_response,
                                        write_ratios, plot_single_vs_full_overview,
                                        plot_io_spectrum_style, plot_sf_loss_spectrum,
                                        plot_sf_loss_decomp)
from overview import _load_trained

D_MLP = 50
LABEL = f"LinearMLP d{D_MLP}"


def _linear_model(p: str, device="cpu"):
    """The fully-trained LinearMLP(d_mlp=50, n_exp=4.0) checkpoint at sparsity p."""
    m = _load_trained("LinearMLP", None, D_MLP, p, 4.0, device)
    if m is None:
        raise SystemExit(f"missing checkpoint: LinearMLP d_mlp={D_MLP} p={p} n_exp=4.0 "
                         f"(expected under checkpoints/)")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", default="0.001")
    ap.add_argument("--overview", action="store_true",
                    help="render the single-vs-full p-sweep overview across all PS instead")
    ap.add_argument("--sf-loss", action="store_true", dest="sf_loss",
                    help="full-model vs single-PD-component loss spectrum (per-c) + on/off profile")
    ap.add_argument("--io-spectrum", action="store_true", dest="io_spectrum",
                    help="single vs full raw-output I/O response in the output-spectrum panel style")
    ap.add_argument("--c", type=int, default=0, help="feature index for the one-hot probe")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ctx = f"LinearMLP d_mlp={D_MLP}, p={a.p}"

    if a.overview:
        plot_single_vs_full_overview(
            0, device=a.device,
            providers=[(LABEL, lambda p, d: _analyse_model(_linear_model(p, d), p, d))],
            stem=f"linear_single_vs_full_overview_d{D_MLP}",
            ctx=f"LinearMLP d_mlp={D_MLP}, fully trained")
        return

    res = {LABEL: _analyse_model(_linear_model(a.p, a.device), a.p, a.device)}
    d = res[LABEL]
    print(f"[{LABEL}] MSE single={d['l2_single']:.3e} full={d['l2_full']:.3e} "
          f"(ratio {d['l2_single']/d['l2_full']:.3f})  |  "
          f"L4 single={d['l4_single']:.3e} full={d['l4_full']:.3e} "
          f"(ratio {d['l4_single']/d['l4_full']:.3f})  |  "
          f"interference={100*d['interference']:.1f}% "
          f"(comp off-target max={d['comp_off_target_max']:.1e})")
    entries = [(LABEL, d)]

    if a.sf_loss:
        plot_sf_loss_spectrum(0, a.p, a.c, device=a.device, entries=entries,
                              stem=f"linear_sf_loss_spectrum_d{D_MLP}_p{a.p}_c{a.c}", ctx=ctx)
        plot_sf_loss_decomp(0, a.p, device=a.device, entries=entries,
                            stem=f"linear_sf_loss_decomp_d{D_MLP}_p{a.p}", ctx=ctx)
        return
    if a.io_spectrum:
        plot_io_spectrum_style(0, a.p, a.c, device=a.device, entries=entries,
                               stem=f"linear_io_spectrum_d{D_MLP}_p{a.p}_c{a.c}", ctx=ctx)
        return
    # default run: single-vs-full bar/ratio/io-response figures
    plot_loss_bars(res, 0, a.p, stat="mean", stem=f"linear_avg_loss_d{D_MLP}_p{a.p}", ctx=ctx)
    plot_loss_bars(res, 0, a.p, stat="max", stem=f"linear_max_loss_d{D_MLP}_p{a.p}", ctx=ctx)
    plot_io_response(res, 0, a.p, stem=f"linear_io_response_d{D_MLP}_p{a.p}", ctx=ctx)
    write_ratios(res, 0, a.p, "", LABEL, stem=f"linear_single_vs_full_d{D_MLP}_p{a.p}", ctx=ctx)


if __name__ == "__main__":
    main()
