"""Rare-code characterization + interference accounting for the default hetK
checkpoints (one section per loss kind present: mse, wmse, l4).

(4) code anatomy: extracted code M-hat per population; local 2-scalar (a, b)
    ansatz refit (under the checkpoint's own loss) and ansatz/trained loss ratio;
    per-feature gain (attenuation) decomposition of the rare MSE; output spectra;
    within-rare swap test; neuron sharing between the populations.
(6) interference: 2x2 source-population -> target-population leakage matrix from
    one-hot probes, plus on-target gains.

  python code_analysis.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import HetKDataset, N_FEATURES, N_RARE
from model import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAU = 0.05
EVAL_SEED = 777
FIT_SEED = 31337


def eval_loss(model, kind: str, weights, batches: int = 25, batch: int = 8192):
    ds = HetKDataset(seed=EVAL_SEED)
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            err = model(x) - y
            if kind == "mse":
                tot += float((err ** 2).mean())
            elif kind == "wmse":
                tot += float((weights * err ** 2).mean())
            else:
                tot += float((err.abs() ** 4).mean())
    return tot / batches


def fit_ab(M: torch.Tensor, kind: str, weights, decoder: str = "transpose",
           steps: int = 4000, batch: int = 4096):
    """Local 2-scalar ansatz fit on boolean code M under the given loss."""
    Mf = M.double()
    ds = HetKDataset(seed=FIT_SEED)
    gen = torch.Generator().manual_seed(FIT_SEED)
    x_pool, y_pool = ds.generate_batch(131072)
    x_pool, y_pool = x_pool.double(), y_pool.double()
    w = weights.double()
    PINV = torch.linalg.pinv(Mf.T)
    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, 131072, (batch,), generator=gen)
        x, y = x_pool[idx], y_pool[idx]
        w_in = a * Mf + b * (1 - Mf)
        w_out = w_in.T if decoder == "transpose" else PINV.T
        err = torch.relu(x @ w_in) @ w_out - y
        if kind == "mse":
            loss = (err ** 2).mean()
        elif kind == "wmse":
            loss = (w * err ** 2).mean()
        else:
            loss = (err.abs() ** 4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return float(a), float(b)


def ansatz_model(M: torch.Tensor, a: float, b: float, decoder: str) -> MLP:
    Mf = M.double()
    m = MLP()
    with torch.no_grad():
        w_in = (a * Mf + b * (1 - Mf)).float()
        m.w_in.copy_(w_in)
        m.w_out.copy_(w_in.T.clone() if decoder == "transpose"
                      else torch.linalg.pinv(Mf.T).T.float())
    return m


def gains(model) -> np.ndarray:
    """Per-feature gain c_j from one-hot probes over a value grid."""
    vs = torch.linspace(0.05, 1.0, 20)
    c = np.zeros(N_FEATURES)
    with torch.no_grad():
        for j in range(N_FEATURES):
            x = vs.unsqueeze(1) * torch.nn.functional.one_hot(
                torch.tensor(j), N_FEATURES).float()
            out = model(x)[:, j]
            c[j] = float((vs * out).sum() / (vs ** 2).sum())
    return c


def perfeature_mse(model, batches: int = 80, batch: int = 8192):
    ds = HetKDataset(seed=EVAL_SEED)
    err = torch.zeros(N_FEATURES)
    cnt = torch.zeros(N_FEATURES)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def swap_test_rare(model, max_pairs: int = 2000, seed: int = 5) -> float:
    W = model.w_in.detach()
    codes = [torch.where(W[j] > TAU)[0] for j in range(N_RARE)]
    ks = np.array([len(c) for c in codes])
    pairs = [(i, j) for i in range(N_RARE) for j in range(N_RARE)
             if i != j and ks[i] == ks[j] and ks[i] > 0]
    if not pairs:
        return float("nan")
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    hits = 0
    with torch.no_grad():
        H = torch.relu(W)
        for i, j in pairs:
            h = torch.zeros(W.shape[1])
            h[codes[j]] = H[i][codes[i]]
            hits += int(torch.argmax(h @ model.w_out.detach())) == j
    return hits / len(pairs)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    ds = HetKDataset()
    weights = (1 / ds.p)
    weights = weights / weights.mean()
    kinds = [k for k in ("mse", "wmse", "l4")
             if os.path.exists(os.path.join(ROOT, "checkpoints",
                                            f"hetk_{k}_s42.pt"))]
    md = ["### hetK code anatomy + interference (default config: rare p=0.001 "
          "x50, common p=0.05 x50)", ""]
    inter = {}
    for kind in kinds:
        ck = torch.load(os.path.join(ROOT, "checkpoints", f"hetk_{kind}_s42.pt"),
                        map_location="cpu", weights_only=False)
        model = MLP(**ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        W = model.w_in.detach()
        M_hat = W > TAU

        # ansatz refit under the checkpoint's own loss
        l_trained = eval_loss(model, kind, weights)
        ratios = {}
        for dec in ("transpose", "pinv"):
            a, b = fit_ab(M_hat, kind, weights, decoder=dec)
            l_ans = eval_loss(ansatz_model(M_hat, a, b, dec), kind, weights)
            ratios[dec] = (a, b, l_ans / l_trained)

        # attenuation decomposition
        c = gains(model)
        mse = perfeature_mse(model)
        attn_pred = (1 - c) ** 2 / 3
        rare, common = np.arange(N_RARE), np.arange(N_RARE, N_FEATURES)

        # neuron sharing
        col_rare = M_hat[:N_RARE].sum(0).float()
        col_common = M_hat[N_RARE:].sum(0).float()
        codes = M_hat.float()
        ov = codes @ codes.T
        k_mean = codes.sum(1).mean()
        within_rare = float(ov[:N_RARE, :N_RARE][
            ~torch.eye(N_RARE, dtype=torch.bool)].mean())
        cross = float(ov[:N_RARE, N_RARE:].mean())
        hyper = float(k_mean ** 2 / W.shape[1])

        swap = swap_test_rare(model)

        # interference 2x2 from one-hot probes
        with torch.no_grad():
            OUT = model(torch.eye(N_FEATURES))
        L = np.zeros((2, 2))
        pops = [rare, common]
        for si, src in enumerate(pops):
            for ti, tgt in enumerate(pops):
                acc = []
                for j in src:
                    others = tgt[tgt != j]
                    acc.append(float((OUT[j, others] ** 2).mean()))
                L[si, ti] = np.mean(acc)
        inter[kind] = L

        md += [f"#### loss = {kind}", "",
               f"- trained {kind} loss (held-out): {l_trained:.3e}",
               f"- ansatz refit on M-hat: transpose (a={ratios['transpose'][0]:.3f}, "
               f"b={ratios['transpose'][1]:.3f}) ratio "
               f"{ratios['transpose'][2]:.2f}; pinv (a={ratios['pinv'][0]:.3f}, "
               f"b={ratios['pinv'][1]:.3f}) ratio {ratios['pinv'][2]:.2f}",
               f"- gains c: rare {c[rare].mean():.3f}±{c[rare].std():.3f}, "
               f"common {c[common].mean():.3f}±{c[common].std():.3f}",
               f"- rare MSE median {np.median(mse[rare]):.3e} vs "
               f"attenuation-only prediction {np.median(attn_pred[rare]):.3e} "
               f"(fraction explained "
               f"{np.median(attn_pred[rare]) / max(np.median(mse[rare]), 1e-12):.2f})",
               f"- neuron sharing: per-neuron degree rare "
               f"{col_rare.mean():.1f}±{col_rare.std():.1f}, common "
               f"{col_common.mean():.1f}±{col_common.std():.1f}; overlap "
               f"within-rare {within_rare:.2f}, rare-common {cross:.2f} "
               f"(hypergeometric ~{hyper:.2f})",
               f"- within-rare swap test: {swap:.2f}",
               f"- interference (mean squared one-hot leakage): "
               f"rare->rare {L[0, 0]:.2e}, rare->common {L[0, 1]:.2e}, "
               f"common->rare {L[1, 0]:.2e}, common->common {L[1, 1]:.2e}", ""]

        # spectra figure
        picks = [0, 1, 2, N_RARE]
        fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.4),
                                 constrained_layout=True)
        with torch.no_grad():
            for ax, j in zip(axes, picks):
                out = model(torch.nn.functional.one_hot(
                    torch.tensor(j), N_FEATURES).float().unsqueeze(0))[0]
                ax.bar(range(N_FEATURES), out.numpy(), width=1.0,
                       color=["tab:red" if k == j else "steelblue"
                              for k in range(N_FEATURES)])
                pop = "rare" if j < N_RARE else "common"
                ax.set_title(f"e_{j} ({pop})", fontsize=9)
                ax.set_xlabel("output feature")
        axes[0].set_ylabel("output")
        fig.suptitle(f"hetK output spectra (one-hot probes, v=1) — loss = "
                     f"{kind}; red = probed feature", fontsize=11)
        fig.savefig(os.path.join(FIG, f"code_spectra_{kind}.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(RES, "code_analysis.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, len(kinds), figsize=(4.4 * len(kinds), 3.8),
                             constrained_layout=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        L = inter[kind]
        im = ax.imshow(np.log10(np.maximum(L, 1e-12)), cmap="viridis",
                       vmin=-9, vmax=-2)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{L[i, j]:.1e}", ha="center", va="center",
                        color="w", fontsize=9)
        ax.set_xticks([0, 1], ["rare", "common"])
        ax.set_yticks([0, 1], ["rare", "common"])
        ax.set_xlabel("target population")
        ax.set_title(f"loss = {kind}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("source population")
    fig.colorbar(im, ax=np.atleast_1d(axes)[-1], label="log10 leakage")
    fig.suptitle("hetK interference: mean squared one-hot leakage by population "
                 "pair", fontsize=11)
    fig.savefig(os.path.join(FIG, "interference_matrix.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
