"""Compare the biregular code (no overlap-min) with two decoders:
  - pinv:      w_out = c * pinv(M.T).T   (scaled pseudoinverse — the least-squares decoder)
  - transpose: w_out = c * w_in.T        (tied weights — decoder is the encoder transpose)

Same code M, same encoder w_in = a*M + b*(1-M); gains (a,b,c) refit to L4 at each p for
each decoder. Reports the model's L4 reconstruction loss (on sparse data at p, the training
objective) side by side, and stores the table under results/overview/.

  ./.venv/bin/python biregular_decoder_compare.py --K 5
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

import biregular_ansatz as bir
import results_io
from mlp import MLP
from overview import OVERVIEW_FIG, OVERVIEW_RES, trained_l4
from sweep import PS

N_EXP = 4.0


def ignore_half_model(F=100, d_mlp=50, device="cpu") -> MLP:
    """No-superposition baseline: dedicate one neuron each to the first d_mlp features
    (monosemantic identity -> perfect relu reconstruction) and output 0 for the rest.
    w_in rows 0..d_mlp-1 = identity, others 0; w_out cols 0..d_mlp-1 = identity, others 0.
    forward -> out[j<d_mlp] = relu(x_j) = target, out[j>=d_mlp] = 0."""
    m = MLP(n_features=F, d_mlp=d_mlp, device=device)
    idx = range(d_mlp)
    with torch.no_grad():
        m.w_in.zero_(); m.w_in[idx, idx] = 1.0      # feature n -> neuron n (n < d_mlp)
        m.w_out.zero_(); m.w_out[idx, idx] = 1.0     # neuron n -> feature n
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--swaps", type=int, default=0,
                    help="overlap-min swaps for the biregular decoders (0 = no-swap)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ntok = results_io._fmt_n(N_EXP)
    swap_tok = "" if a.swaps == 0 else f"_swap{a.swaps}"
    code_lbl = "no overlap-min" if a.swaps == 0 else "overlap-min"

    rows = {}   # decoder -> {p: L4}
    gains = {}  # decoder -> {p: (a*c, b*c)}
    for dec in ("pinv", "transpose"):
        ans = bir.CodeAnsatz(a.K, swaps=a.swaps, decoder=dec)
        rows[dec], gains[dec] = {}, {}
        for p in PS:
            r = ans.fit_eval(float(p), N_EXP, device=a.device)
            rows[dec][p] = r["L4"]
            gains[dec][p] = (r["ac"], r["bc"])
            print(f"[{dec:9}] p={p:6}  L4={r['L4']:.3e}  a*c={r['ac']:.3f}  b*c={r['bc']:.3f}")

    # naive fully-trained MLP(d_mlp=50) baseline (the free model of the same width;
    # same reconstruction_loss settings as fit_eval, so directly comparable)
    rows["naive"] = {}
    for p in PS:
        rows["naive"][p] = trained_l4("MLP", None, 50, p, N_EXP, device=a.device)
        print(f"[naive-d50] p={p:6}  L4={rows['naive'][p]:.3e}  (fully-trained MLP)")

    # no-superposition baseline: compute 50 features perfectly, ignore (zero) the other 50
    ig = ignore_half_model(device=a.device)
    rows["ignore50"] = {}
    for p in PS:
        rows["ignore50"][p] = ig.reconstruction_loss(float(p), N_EXP)
        analytic = 0.05 * float(p)   # E[relu(u)^4]=1/10 per ignored active feature, 50/100
        print(f"[ignore-50] p={p:6}  L4={rows['ignore50'][p]:.3e}  (analytic 0.05*p={analytic:.3e})")

    # --- store table (md + csv) ---
    os.makedirs(OVERVIEW_RES, exist_ok=True)
    stem = os.path.join(OVERVIEW_RES, f"biregular_decoder_L{ntok}_n{ntok}_K{a.K}{swap_tok}")
    header = ["decoder"] + [f"p={p}" for p in PS]
    all_rows = ["naive", "ignore50", "pinv", "transpose"]
    lines = [f"### Biregular K{a.K} ({code_lbl}): L{ntok} reconstruction loss — "
             f"naive MLP(d_mlp=50) + no-superposition baselines vs pinv/transpose decoder",
             "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    labels = {"naive": "naive MLP d50", "ignore50": "ignore-50 (no superposition)",
              "pinv": "pinv decoder", "transpose": "transpose decoder"}
    for dec in all_rows:
        lines.append("| " + " | ".join([labels[dec]] + [f"{rows[dec][p]:.3e}" for p in PS]) + " |")
    lines.append("| " + " | ".join(["ratio (transpose/pinv)"] +
                 [f"{rows['transpose'][p] / rows['pinv'][p]:.2f}x" for p in PS]) + " |")
    lines.append("| " + " | ".join(["ratio (pinv/naive)"] +
                 [f"{rows['pinv'][p] / rows['naive'][p]:.2f}x" for p in PS]) + " |")
    lines.append("| " + " | ".join(["ratio (transpose/naive)"] +
                 [f"{rows['transpose'][p] / rows['naive'][p]:.2f}x" for p in PS]) + " |")
    lines.append("| " + " | ".join(["ratio (ignore50/naive)"] +
                 [f"{rows['ignore50'][p] / rows['naive'][p]:.2f}x" for p in PS]) + " |")
    md = "\n".join(lines) + "\n"
    with open(stem + ".md", "w") as f:
        f.write(md)
    with open(stem + ".csv", "w") as f:
        f.write("model," + ",".join(PS) + "\n")
        for dec in all_rows:
            f.write(dec + "," + ",".join(f"{rows[dec][p]:.6e}" for p in PS) + "\n")
    print("\n" + md + f"  (saved {stem}.md / .csv)")

    # --- figure: L4 vs p, naive baseline + both decoders ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    xs = [float(p) for p in PS]
    for dec, col, style in [("naive", "k", "o--"), ("ignore50", "tab:gray", "s:"),
                            ("pinv", "tab:blue", "o-"), ("transpose", "tab:red", "o-")]:
        ax.plot(xs, [rows[dec][p] for p in PS], style, color=col, label=labels[dec])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("data sparsity p"); ax.set_ylabel(f"L{ntok} reconstruction loss")
    ax.set_title(f"Biregular K{a.K} ({code_lbl}) vs naive MLP(d_mlp=50)")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    out = os.path.join(OVERVIEW_FIG, f"biregular_decoder_compare_n{ntok}_K{a.K}{swap_tok}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
