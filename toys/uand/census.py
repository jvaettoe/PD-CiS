"""Gate-computation census for the U-AND models.

Per (N, p, loss): per-gate MSE conditional on both inputs active (trivial
always-0 baseline = 1.0) AND conditional on exactly one input active (a gate
counts as computed only if BOTH are < 0.1 — firing on both-active, silent on
singles); gates-computed vs width N is the CiS headline. Also hidden-activation
counts on zero/one-hot/two-hot probes and per-gate codeword sizes (neurons whose
two-hot activation exceeds both one-hot activations by tau).

  python census.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import I_IDX, J_IDX, N_IN, N_OUT, UANDDataset
from model import UAND

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
COMPUTED_AT = 0.1          # vs always-0 baseline 1.0 on both-active events
EVAL_SEED = 777


def pergate_mse(model, p: float, batches: int = 50, batch: int = 8192):
    """(MSE | both active, MSE | exactly one of the pair active) per gate."""
    ds = UANDDataset(p, seed=EVAL_SEED)
    err_b = torch.zeros(N_OUT)
    cnt_b = torch.zeros(N_OUT)
    err_s = torch.zeros(N_OUT)
    cnt_s = torch.zeros(N_OUT)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            out = model(x)
            both = (x[:, I_IDX] * x[:, J_IDX])
            single = (x[:, I_IDX] + x[:, J_IDX] == 1).float()
            err = (out - y) ** 2
            err_b += (err * both).sum(0)
            cnt_b += both.sum(0)
            err_s += (err * single).sum(0)
            cnt_s += single.sum(0)
    return ((err_b / cnt_b.clamp(min=1)).numpy(),
            (err_s / cnt_s.clamp(min=1)).numpy())


def probe_stats(model):
    """Hidden activity on zero, one-hot, two-hot probes + per-gate codeword size."""
    with torch.no_grad():
        h0 = torch.relu(model.hidden(torch.zeros(1, N_IN)))[0]
        eye = torch.eye(N_IN)
        h1 = torch.relu(model.hidden(eye))                     # (15, N)
        two = eye[I_IDX] + eye[J_IDX]
        h2 = torch.relu(model.hidden(two))                     # (105, N)
    # superadditivity criterion: a neuron belongs to a gate's codeword only if
    # its two-hot response exceeds the SUM of one-hot responses (linear neurons
    # satisfy h2 = h1_i + h1_j - h0 exactly and are excluded)
    gate_code = (h2 > h1[I_IDX] + h1[J_IDX] - h0 + TAU)
    return {"n_h0": int((h0 > TAU).sum()),
            "n_h1_mean": float((h1 > TAU).sum(1).float().mean()),
            "n_h2_mean": float((h2 > TAU).sum(1).float().mean()),
            "gate_k_mean": float(gate_code.sum(1).float().mean()),
            "gate_k_std": float(gate_code.sum(1).float().std())}


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "uand_*.pt"))):
        m = re.match(r"uand_N(\d+)_p([\d.]+)_(mse|l[\d.]+)_s\d+\.pt",
                     os.path.basename(path))
        if not m:
            continue
        N, p, kind = int(m.group(1)), float(m.group(2)), m.group(3)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = UAND(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        mse_b, mse_s = pergate_mse(model, p)
        computed = (mse_b < COMPUTED_AT) & (mse_s < COMPUTED_AT)
        ps = probe_stats(model)
        rows.append({"N": N, "p": p, "loss": kind,
                     "n_computed": int(computed.sum()),
                     "mse_b_med": float(np.median(mse_b)),
                     "mse_s_med": float(np.median(mse_s)), **ps})
        print(f"N={N} p={p:g} {kind}: computed {rows[-1]['n_computed']}/105")
    rows.sort(key=lambda r: (r["loss"], r["p"], r["N"]))

    md = [f"### uand census — all {N_OUT} pairwise ANDs of {N_IN} booleans; "
          "computed = MSE|both-active < 0.1 AND MSE|one-active < 0.1 "
          "(always-0 baseline on both-active = 1.0)", "",
          "| loss | p | N | computed /105 | MSE|both med | MSE|single med | "
          "#h>tau at 0 | #h>tau one-hot | #h>tau two-hot | gate K |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['loss']} | {r['p']:g} | {r['N']} | {r['n_computed']} | "
                  f"{r['mse_b_med']:.2e} | {r['mse_s_med']:.2e} | {r['n_h0']} | "
                  f"{r['n_h1_mean']:.1f} | {r['n_h2_mean']:.1f} | "
                  f"{r['gate_k_mean']:.1f}±{r['gate_k_std']:.1f} |")
    with open(os.path.join(RES, "census.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for kind, ls in (("mse", "-"), ("l4", "--")):
        for p, col in ((0.1, "tab:blue"), (0.2, "tab:red")):
            rs = [r for r in rows if r["loss"] == kind and r["p"] == p]
            if rs:
                ax.plot([r["N"] for r in rs], [r["n_computed"] for r in rs],
                        "o" + ls, color=col, lw=2, label=f"{kind}, p={p:g}")
    ax.plot([0, 110], [0, 110], color="grey", ls=":", lw=0.8)
    ax.text(60, 52, "computed = N (no superposition)", fontsize=7, color="grey",
            rotation=38)
    ax.axhline(N_OUT, color="k", lw=0.5)
    ax.set_xlabel("hidden width N")
    ax.set_ylabel(f"gates computed / {N_OUT}")
    ax.set_title("uand: gates computed vs width — above the diagonal = "
                 "computation in superposition", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "census_gates_vs_width.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # Ln-exponent sweep: gates computed vs loss exponent n at fixed (N, p)
    from train import loss_exponent
    sweep = {}
    for r in rows:
        sweep.setdefault((r["N"], r["p"]), []).append(
            (loss_exponent(r["loss"]), r["n_computed"]))
    sweep = {k: sorted(v) for k, v in sweep.items() if len(v) >= 3}
    if sweep:
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        for (N, p), pts in sorted(sweep.items()):
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", lw=2,
                    label=f"N={N}, p={p:g}")
        ax.axhline(N_OUT, color="k", lw=0.5)
        ax.set_xlabel("training-loss exponent n (Ln = mean |err|^n)")
        ax.set_ylabel(f"gates computed / {N_OUT}")
        ax.set_title("uand: CiS onset vs loss exponent — gradual or sudden?",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(FIG, "census_gates_vs_exponent.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
