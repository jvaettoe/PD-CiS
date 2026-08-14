"""Trained chain vs designed two-layer references, per p (L4 chains):
(1) stacked biregular codes (decode-re-encode, 4-scalar L4 fit, pinv decoders)
(2a) designed code layer-1 + TRAINED chain's fitted A and W3 (transplant)
(2b) designed code layer-1 + optimal linear decoder (lstsq)
plus the single-hidden-layer biregular pinv ansatz as anchor.
K1/K2 per p measured from the trained chain (tau=0.05, half-up).

  python stacked_codes.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import N_FEATURES, SparseDataset
from model import ChainMLP

ROOT = os.path.dirname(os.path.abspath(__file__))
TAU, EVAL_SEED, FIT_SEED = 0.05, 777, 12345
N = 50


def biregular(K: int, seed: int) -> torch.Tensor:
    """(F, N) boolean code, row degree exactly K, column degrees balanced
    (= 2K up to greedy ties): each row takes the K least-used neurons with
    random tie-breaking."""
    rng = np.random.default_rng(seed)
    deg = np.zeros(N)
    M = np.zeros((N_FEATURES, N), dtype=bool)
    for j in range(N_FEATURES):
        order = np.argsort(deg + rng.random(N) * 1e-3)
        M[j, order[:K]] = True
        deg[order[:K]] += 1
    assert (M.sum(1) == K).all() and M.sum(0).max() - M.sum(0).min() <= 1
    return torch.tensor(M)


def held_out(fwd, p, batches=30, batch=8192):
    ds = SparseDataset(p, seed=EVAL_SEED)
    l4 = 0.0
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            out = fwd(x)
            l4 += float(((out - y).abs() ** 4).mean())
            a = (x > 0).float()
            err += (((out - y) ** 2) * a).sum(0)
            cnt += a.sum(0)
    pf = (err / cnt.clamp(min=1)).numpy()
    return l4 / batches, int((pf < 1 / 30).sum())


def fit_stacked(M1, M2, p, steps=4000, batch=4096):
    ds = SparseDataset(p, seed=31337)
    xp, yp = ds.generate_batch(131072)
    gen = torch.Generator().manual_seed(31337)
    M1f, M2f = M1.double(), M2.double()
    prm = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
           for v in (0.3, 0.0, 0.3, 0.0)]
    opt = torch.optim.Adam(prm, lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xp, yp = xp.double(), yp.double()
    for _ in range(steps):
        i = torch.randint(0, 131072, (batch,), generator=gen)
        a1, b1, a2, b2 = prm
        W1 = a1 * M1f + b1 * (1 - M1f)
        R = torch.linalg.pinv(W1.T).T
        C2 = a2 * M2f + b2 * (1 - M2f)
        W2 = R @ C2
        W3 = torch.linalg.pinv(C2.T).T
        out = torch.relu(torch.relu(xp[i] @ W1) @ W2) @ W3
        loss = ((out - yp[i]).abs() ** 4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return [float(v) for v in prm]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints",
                                              "chain_p*_l4_s42.pt"))):
        m = re.match(r"chain_p([\d.]+)_l4_s\d+\.pt", os.path.basename(path))
        if not m:
            continue
        p = float(m.group(1))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tm = ChainMLP(**ck["model_config"])
        tm.load_state_dict(ck["state_dict"])
        W1t, W2t, W3t = tm.w1.detach(), tm.w2.detach(), tm.w3.detach()
        h1p = torch.relu(W1t)
        K1 = int(np.floor(float((W1t > TAU).sum(1).float().mean()) + 0.5))
        K2 = int(np.floor(float(((h1p @ W2t) > TAU).sum(1).float().mean()) + 0.5))
        M1, M2 = biregular(K1, 0), biregular(K2, 7)

        a1, b1, a2, b2 = fit_stacked(M1, M2, p)
        M1f, M2f = M1.double(), M2.double()
        W1 = (a1 * M1f + b1 * (1 - M1f)).float()
        R = torch.linalg.pinv(W1.double().T).T.float()
        C2 = (a2 * M2f + b2 * (1 - M2f)).float()
        W2, W3 = R @ C2, torch.linalg.pinv(C2.double().T).T.float()

        xs, ys = SparseDataset(p, seed=FIT_SEED).generate_batch(131072)
        with torch.no_grad():
            h1t = torch.relu(xs @ W1t)
            A = torch.linalg.lstsq(h1t, torch.relu(h1t @ W2t)).solution
            h1d = torch.relu(xs @ W1)
            D = torch.linalg.lstsq(h1d, ys).solution

        variants = {
            "trained": lambda x: torch.relu(torch.relu(x @ W1t) @ W2t) @ W3t,
            "stacked": lambda x: torch.relu(torch.relu(x @ W1) @ W2) @ W3,
            "code+trainedA": lambda x: (torch.relu(x @ W1) @ A) @ W3t,
            "code+lstsq": lambda x: torch.relu(x @ W1) @ D,
        }
        r = {"p": p, "K1": K1, "K2": K2,
             "ab": (round(a1, 3), round(b1, 3), round(a2, 3), round(b2, 3))}
        for name, fwd in variants.items():
            l4, nc = held_out(fwd, p)
            r[name] = l4
            r[name + "_nc"] = nc
        rows.append(r)
        print(f"p={p:g} K1={K1} K2={K2} ab={r['ab']}: " + "  ".join(
            f"{k} {r[k]:.2e}({r[k + '_nc']})" for k in variants))
    rows.sort(key=lambda r: r["p"])

    md = ["### chain: trained vs stacked designed codes (pinv decoders, "
          "4-scalar L4 fit; parentheses = features computed)", "",
          "| p | K1 | K2 | trained | stacked codes | code+trained A | "
          "code+lstsq |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['p']:g} | {r['K1']} | {r['K2']} | "
                  f"{r['trained']:.2e} ({r['trained_nc']}) | "
                  f"{r['stacked']:.2e} ({r['stacked_nc']}) | "
                  f"{r['code+trainedA']:.2e} ({r['code+trainedA_nc']}) | "
                  f"{r['code+lstsq']:.2e} ({r['code+lstsq_nc']}) |")
    with open(os.path.join(ROOT, "results", "stacked_codes.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    ps = [r["p"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for k, col in (("trained", "k"), ("stacked", "tab:blue"),
                   ("code+trainedA", "tab:red"), ("code+lstsq", "tab:orange")):
        axes[0].plot(ps, [r[k] for r in rows], "o-", color=col, lw=2, label=k)
        if k != "trained":
            axes[1].plot(ps, [r[k] / r["trained"] for r in rows], "o-",
                         color=col, lw=2, label=k)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(1, color="k", lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("p (log)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("held-out L4")
    axes[1].set_ylabel("ratio to trained chain")
    fig.suptitle("chain: trained vs designed two-layer references", fontsize=11)
    fig.savefig(os.path.join(ROOT, "figures", "stacked_codes.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print("saved figures/stacked_codes.png")


if __name__ == "__main__":
    main()
