"""Render every model-level view for all cached trained models -- no
decomposition involved, reads checkpoints/ directly. Per model:
  - weights   : read-in / read-out (feature space; EmbedMLP folds w_e)
  - effective : linearised feature -> feature map
  - bars      : per-feature neuron assignment
  - io        : per-feature input->output behaviour

  ./.venv/bin/python plot_models.py             # -> figures/models/model_views/...
  ./.venv/bin/python plot_models.py --tag gpu   # -> figures/gpu/model_views/...
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import cc
import figutil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="models", help="figures/<tag>/model_views (default: models)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    figutil.set_tag(a.tag)

    ckpts = sorted(glob.glob(os.path.join(cc.CKPT_DIR, "*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoints in {cc.CKPT_DIR}")

    n = 0
    for ck in ckpts:
        base = os.path.basename(ck)
        m = re.search(r"_p([0-9.]+)_n([0-9.]+)_s\d+\.pt$", base)
        if not m:
            continue
        p_tok, n_tok = m.group(1), m.group(2)
        run = cc.load_checkpoint(ck, device=a.device)
        model, meta = run["model"], run["meta"]
        model_cls = meta["model_class"]
        is_embed = model_cls == "EmbedMLP"
        d_emb = meta["model_config"].get("d_emb")
        d_mlp = meta["model_config"].get("d_mlp")
        demb_tok = str(d_emb) if is_embed else "NA"

        figs = model.plot_views()   # weights/effective/bars
        figs["io"] = model.plot_input_output_behaviour()
        for view, fig in figs.items():
            figutil.save_fig(
                fig, os.path.join("model_views", model_cls),
                f"{model_cls}_demb{demb_tok}_dmlp{d_mlp}_p{p_tok}_n{n_tok}_{view}.png",
                f"{model_cls} {view} (feature space)",
                n_group=n_tok,
                model=model_cls, d_emb=(d_emb if is_embed else None),
                d_mlp=d_mlp, p=p_tok, n_exp=n_tok,
            )
        n += 1
        print(f"  {base}")
    print(f"\nrendered model views for {n} checkpoints -> {figutil.FIG_ROOT}/model_views")


if __name__ == "__main__":
    main()
