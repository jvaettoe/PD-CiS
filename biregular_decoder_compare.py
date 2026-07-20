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

import biregular_ansatz as bir
import results_io
from overview import OVERVIEW_FIG, OVERVIEW_RES
from sweep import PS

N_EXP = 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ntok = results_io._fmt_n(N_EXP)

    rows = {}   # decoder -> {p: L4}
    gains = {}  # decoder -> {p: (a*c, b*c)}
    for dec in ("pinv", "transpose"):
        ans = bir.CodeAnsatz(a.K, swaps=0, decoder=dec)
        rows[dec], gains[dec] = {}, {}
        for p in PS:
            r = ans.fit_eval(float(p), N_EXP, device=a.device)
            rows[dec][p] = r["L4"]
            gains[dec][p] = (r["ac"], r["bc"])
            print(f"[{dec:9}] p={p:6}  L4={r['L4']:.3e}  a*c={r['ac']:.3f}  b*c={r['bc']:.3f}")

    # --- store table (md + csv) ---
    os.makedirs(OVERVIEW_RES, exist_ok=True)
    stem = os.path.join(OVERVIEW_RES, f"biregular_decoder_L{ntok}_n{ntok}_K{a.K}")
    header = ["decoder"] + [f"p={p}" for p in PS]
    lines = [f"### Biregular K{a.K} (no overlap-min): L{ntok} reconstruction loss — "
             f"pinv vs transpose decoder", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for dec in ("pinv", "transpose"):
        lines.append("| " + " | ".join([dec] + [f"{rows[dec][p]:.3e}" for p in PS]) + " |")
    lines.append("| " + " | ".join(["ratio (transpose/pinv)"] +
                 [f"{rows['transpose'][p] / rows['pinv'][p]:.2f}x" for p in PS]) + " |")
    md = "\n".join(lines) + "\n"
    with open(stem + ".md", "w") as f:
        f.write(md)
    with open(stem + ".csv", "w") as f:
        f.write("decoder," + ",".join(PS) + "\n")
        for dec in ("pinv", "transpose"):
            f.write(dec + "," + ",".join(f"{rows[dec][p]:.6e}" for p in PS) + "\n")
    print("\n" + md + f"  (saved {stem}.md / .csv)")

    # --- figure: L4 vs p, both decoders ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    xs = [float(p) for p in PS]
    for dec, col in [("pinv", "tab:blue"), ("transpose", "tab:red")]:
        ax.plot(xs, [rows[dec][p] for p in PS], "o-", color=col, label=f"{dec} decoder")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("data sparsity p"); ax.set_ylabel(f"L{ntok} reconstruction loss")
    ax.set_title(f"Biregular K{a.K} (no overlap-min): pinv vs transpose decoder")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    out = os.path.join(OVERVIEW_FIG, f"biregular_decoder_compare_n{ntok}_K{a.K}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
