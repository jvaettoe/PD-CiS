"""ExactPD single-component vs full-model one-hot response on the biregular code.

The biregular code models are analytic LinearMLPs: w_in = a*M + b*(1-M) (rows =
features), w_out = c*pinv(M.T).T. ExactPD on one gives one rank-1 component per
feature = row c of readin / column c of readout. We probe the interference the code
introduces by feeding the one-hot input e_c and comparing:
  - the FULL model output  model(e_c)                  (the whole 100-vector)
  - the SINGLE component-c output on e_c                (nonzero only at position c)

Key fact: on e_c both share the hidden activation relu(readin[c,:]), so they AGREE
exactly at output c (self-response s_c = relu(readin[c,:]) . readout[:,c]) and differ
ONLY off-target -- the full model leaks into outputs j != c, the single component is
0 there. Hence single-component response == the diagonal of the full one-hot response
matrix R (R[c,:] = model(e_c)); R's off-diagonal = the interference the per-feature
components discard. Overlap minimisation exists to shrink that leakage, so we render
the no-swap and overlap-min codes side by side.

  ./.venv/bin/python biregular_single_component.py            # K=5, p=0.1, both variants
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import biregular_ansatz as bir
from pd import ExactPD
from overview import BIREG_K, SWAPS, OVERVIEW_FIG

VARIANTS = [("no-swap", 0), ("overlap-min", SWAPS)]


def analyse(K: int, swaps: int, p: str, device="cpu", decoder: str = "pinv") -> dict:
    """Build the biregular LinearMLP, run ExactPD, return the full one-hot response
    matrix R and derived per-feature / global interference metrics."""
    ans = bir.CodeAnsatz(K, swaps=swaps, decoder=decoder)
    model = ans.model(ans.fit_abc(float(p), device=device), device=device)
    pd_e = ExactPD(model)
    F = model.n_features
    X = torch.eye(F, device=device)                 # probe c = one-hot e_c (row c)
    I = torch.eye(F, device=device)                 # target: relu(e_c) = e_c
    with torch.no_grad():
        R = model(X).detach()                       # (F, F): R[c,:] = FULL model output on e_c

        # EXPLICIT single-component forwards -- no diagonal shortcut. For each probe e_c,
        # run ONLY component c (one-hot mask at c selects it via ExactPD's boolean
        # masked_weight) and keep its FULL output vector. S[c,:] = component-c output on e_c.
        S = torch.zeros(F, F, device=device)
        for c in range(F):
            oh = torch.zeros(pd_e.C, device=device); oh[c] = 1.0   # select component c
            S[c] = pd_e.forward_masked(X[c:c + 1], {"w_in": oh, "w_out": oh})[0]

        # sanity: the selected component reproduces the full model on its own output and
        # is localized there (both MEASURED now, not assumed).
        diag_gap = (S.diagonal() - R.diagonal()).abs().max().item()
        S_off = S - torch.diag(S.diagonal())        # component output off its own feature
        assert diag_gap < 1e-4, f"component != full model on-target (gap {diag_gap:.1e})"
        assert (pd_e.forward_dense(X) - R).abs().max().item() < 1e-4, "faithfulness"

    s = S.diagonal().clone()                         # single-component self-responses
    L = S_off.norm(dim=1)                            # component's OWN off-target leakage (~0)
    off = R - torch.diag(R.diagonal())               # full model's off-target leakage
    interference = float((off.pow(2).sum() / R.pow(2).sum()).item())

    # Per-probe forward-pass loss vs target e_c, over the FULL 100-d output, averaged
    # over probes. Single-component loss now comes from the explicit forward S, so any
    # off-target leakage the component DID produce is honestly counted.
    dfull, dsingle = R - I, S - I
    l2_full, l4_full = dfull.pow(2).mean(1), dfull.abs().pow(4).mean(1)
    l2_single, l4_single = dsingle.pow(2).mean(1), dsingle.abs().pow(4).mean(1)
    return {"R": R.cpu().numpy(), "S": S.cpu().numpy(), "s": s.cpu().numpy(),
            "L": L.cpu().numpy(), "interference": interference, "F": F,
            "comp_off_target_max": float(S_off.abs().max().item()),
            "l2_full": float(l2_full.mean()), "l2_single": float(l2_single.mean()),
            "l4_full": float(l4_full.mean()), "l4_single": float(l4_single.mean()),
            "sq_overlap": ans.sq_overlap()}


def plot_avg_loss(res: dict[str, dict], K: int, p: str, tag: str = "") -> None:
    """The simple headline: average forward-pass loss, single component vs full model,
    per variant. Two panels (MSE and the training L4)."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    names = [n for n, _ in VARIANTS]
    x = np.arange(len(names)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, (key_s, key_f, ttl) in zip(axes, [
            ("l2_single", "l2_full", "average MSE  (mean_j (out−target)²)"),
            ("l4_single", "l4_full", "average L4 loss  (mean_j |out−target|⁴, the training loss)")]):
        vs = [res[n][key_s] for n in names]
        vf = [res[n][key_f] for n in names]
        b1 = ax.bar(x - w / 2, vs, w, label="single component", color="tab:blue")
        b2 = ax.bar(x + w / 2, vf, w, label="full model", color="tab:red")
        ax.bar_label(b1, fmt="%.4f", fontsize=8); ax.bar_label(b2, fmt="%.4f", fontsize=8)
        ax.set_xticks(x, names); ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("loss (averaged over 100 one-hot probes)")
        ax.legend(fontsize=9)
    fig.suptitle(f"Single ExactPD component vs full model — average forward-pass loss on "
                 f"one-hot inputs e_c  (K={K}, p={p}{tag and ', ' + tag})", fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"biregular_avg_loss_K{K}_p{p}{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_io_response(res: dict[str, dict], K: int, p: str, tag: str = "") -> None:
    """I/O response: the same one-hot e_c through its single component vs the full
    model, for a few example features. Single = clean spike at c; full = spike + leakage."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    names = [n for n, _ in VARIANTS]
    F = res[names[0]]["F"]
    examples = [0, F // 3, 2 * F // 3]
    fig, axes = plt.subplots(len(examples), len(names),
                             figsize=(7.0 * len(names), 2.4 * len(examples)),
                             squeeze=False, constrained_layout=True)
    for col, n in enumerate(names):
        R = res[n]["R"]; S = res[n]["S"]
        for row, c in enumerate(examples):
            ax = axes[row][col]
            ax.bar(range(F), R[c], width=1.0, color="tab:red", alpha=0.7,
                   label="full model output")
            # explicit single-component output over ALL outputs (localized on c)
            ax.bar(range(F), S[c], width=1.6, color="tab:blue", label="single component c")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{n}  —  input e_{c}", fontsize=9)
            if row == len(examples) - 1:
                ax.set_xlabel("output feature j")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"I/O response — one-hot e_c through its component vs the full model  "
                 f"(single = self-response at c only; full = self + leakage)  "
                 f"(K={K}, p={p}{tag and ', ' + tag})", fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"biregular_io_response_K{K}_p{p}{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=BIREG_K)
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--decoder", default="pinv", choices=["pinv", "transpose"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tag = "" if a.decoder == "pinv" else f"_{a.decoder}"
    res = {}
    for name, sw in VARIANTS:
        print(f"[{name}] building K={a.K} swaps={sw} decoder={a.decoder} ...")
        res[name] = analyse(a.K, sw, a.p, device=a.device, decoder=a.decoder)
        d = res[name]
        print(f"  MSE single={d['l2_single']:.4f} full={d['l2_full']:.4f}  |  "
              f"L4 single={d['l4_single']:.4f} full={d['l4_full']:.4f}  "
              f"(interference={100*d['interference']:.1f}%, "
              f"component off-target max={d['comp_off_target_max']:.1e})")
    plot_avg_loss(res, a.K, a.p, tag=tag)
    plot_io_response(res, a.K, a.p, tag=tag)


if __name__ == "__main__":
    main()
