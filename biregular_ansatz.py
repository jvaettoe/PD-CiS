"""Reconstruct the biregular, no-overlap-min "3-parameter ansatz" model and run
our ExactPD on it -- refitting the model at each sparsity p.

Ported from the adjacent CiS repo (Others' code/toy-model-cis-code-master,
sweep_K_codes.py --swaps 0): a designed biregular code M = regular_code(F, N, K)
(NO reduce_overlap) defines a hand-built single-hidden-layer model whose only free
parameters are three scalars (a, b, c) fit to L4 loss:

    W_in  = a * M.T + b * (1 - M.T)      # (N, F): a on-code, b off-code   (encoder)
    W_out = c * pinv(M.T)                # (F, N): scaled pseudoinverse     (decoder)

The DECODER is tied to a scaled pseudoinverse of the code: pinv(M.T) is fixed and
only the scale c (with a, b) is learned. The code M is p-free, but (a, b, c) are fit
to data at a chosen sparsity -- so the model DOES depend on p through them. We refit
(a, b, c) at each swept p, so every p is its own optimal model and the full ExactPD
suite genuinely varies with p.

Their SimpleMLP is y = relu(x @ W_in.T) @ W_out.T = our LinearMLP's relu(x @ w_in) @ w_out
under the transpose w_in = W_in.T = a*M + b*(1-M) (F, N), w_out = W_out.T = (c*pinv(M.T)).T
(N, F). So we build a LinearMLP, inject those weights, and reuse the ExactPD / component-viz
stack unchanged.

  ./.venv/bin/python biregular_ansatz.py --K 5                  # single cell, refit at p=0.02
  ./.venv/bin/python biregular_ansatz.py --K 5 --p-sweep        # full suite per p (refit each)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import codes
import decomp_fit
import figutil
from data import CleanDataset
from mlp import LinearMLP
from pd import ExactPD

F, N, P = 100, 50, 0.02   # feature count, neuron count, default/design sparsity (as in the paper)
PS_SWEEP = [0.001, 0.01, 0.1, 0.5, 0.8]   # the standard sparsity sweep (matches the main grid)

K_CHOICES = (4, 5, 6, 7)   # codeword lengths swept in the source paper

# REFERENCE ONLY (not used by the code): the (a, b, c) the adjacent repo fit per K at
# p=0.02 (data/sweep_K_codes_noswap.json, regular, --swaps 0). `fit_abc` re-derives them
# from scratch here (and their p-dependence). NB only the products a*c, b*c are
# determined -- these particular a/c values are one point on the ReLU-rescaling gauge orbit.
REF_ABC_P002 = {4: (0.8689, -0.0230, 1.7413), 5: (0.8804, -0.0264, 1.7333),
                6: (0.8830, -0.0339, 1.7469), 7: (0.8856, -0.0357, 1.7409)}

FIG_ROOT = os.path.join(os.path.dirname(__file__), "figures", "biregular")


def _pinv_wout(M):
    """Our-convention decoder base (unscaled): w_out = (pinv(M.T)).T, shape (N, F)."""
    return np.linalg.pinv(M.T.astype(np.float64)).T.astype(np.float32)


def model_from_abc(M, abc, device="cpu"):
    """Build a LinearMLP(F, N) from a code M and scalars (a, b, c): w_in = a*M + b*(1-M),
    w_out = c * (pinv(M.T)).T (the scaled-pseudoinverse decoder)."""
    a, b, c = abc
    Mf = M.astype(np.float64)
    w_in = (a * Mf + b * (1.0 - Mf)).astype(np.float32)     # (F, N)
    w_out = (c * _pinv_wout(M)).astype(np.float32)           # (N, F)
    model = LinearMLP(n_features=F, d_mlp=N, device=device)
    with torch.no_grad():
        model.w_in.copy_(torch.from_numpy(w_in).to(device))
        model.w_out.copy_(torch.from_numpy(w_out).to(device))
    return model


def fit_abc(M, p, steps=4000, batch=4096, pool=131072, lr=0.01, device="cpu", seed=31337):
    """Fit (a, b, c) to L4 loss for the FIXED code M at sparsity p (port of the adjacent
    train_3param). The decoder base pinv(M.T) is fixed -- only the 3 scalars are learned,
    so W_out stays a scaled pseudoinverse of the code. Adam + cosine LR.

    Pre-generates ONE data pool and samples minibatches by index (as the source does),
    instead of regenerating a random batch every step -- the batch generation, not the
    tiny matmul, was the dominant per-step cost."""
    Mf = torch.tensor(M.astype(np.float32), device=device)               # (F, N)
    PINV = torch.tensor(_pinv_wout(M), device=device)                    # (N, F) decoder base
    ds = CleanDataset(n_features=F, p=p, device=device, seed=seed)
    x_pool, y_pool = ds.generate_batch(pool)                             # generated ONCE
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.tensor(0.85, device=device, requires_grad=True)
    b = torch.tensor(-0.04, device=device, requires_grad=True)
    c = torch.tensor(1.9, device=device, requires_grad=True)
    opt = torch.optim.Adam([a, b, c], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, pool, (batch,), device=device, generator=gen)
        x, y = x_pool[idx], y_pool[idx]
        w_in = a * Mf + b * (1 - Mf)
        loss = ((torch.relu(x @ w_in) @ (c * PINV) - y).abs() ** 4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return float(a), float(b), float(c)


def l4_loss(model, p=P, seed=9999, batch=4096, device="cpu"):
    """L4 reconstruction loss on a fresh sparse batch at sparsity p."""
    ds = CleanDataset(n_features=F, p=p, device=device, seed=seed)
    x, y = ds.generate_batch(batch)
    with torch.no_grad():
        return ((model(x) - y).abs() ** 4).mean().item()


def render_cell(model, M, base, family, p, k, top, out_dir, device="cpu"):
    """Render the FULL viz suite for one (model, p) cell and return its loss terms.

    Weight pictures (model.plot_weights, code + effective map) + the ExactPD component
    suite incl. the p-dependent `usage` view (batch-top-k at k). Every filename is
    stamped `{base}_p{p}_k{k}_<view>.png`."""
    cell = f"{base}_p{p:g}_k{k}"

    fig = model.plot_weights()
    fig.suptitle(f"Biregular ansatz weights  ({family}, p={p:g}) — w_in = a*M + b*(1-M)")
    fig.savefig(os.path.join(out_dir, f"{cell}_model_weights.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axs[0].imshow(M.astype(float), aspect="auto", cmap="Greys", interpolation="nearest")
    axs[0].set_title(f"code M (F x N): {int(M.sum(1)[0])} neurons/feature")
    axs[0].set_xlabel("neuron"); axs[0].set_ylabel("feature")
    eff = model.effective_matrix().cpu().numpy()
    v = float(np.abs(eff).max()) or 1.0
    im = axs[1].imshow(eff, cmap="RdBu", vmin=-v, vmax=v)
    axs[1].set_title("effective feature->feature map  (w_in @ w_out)")
    axs[1].set_xlabel("feature"); axs[1].set_ylabel("feature")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(out_dir, f"{cell}_code_and_effective.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    pd_obj = ExactPD(model)
    ds = CleanDataset(n_features=F, p=p, device=device, seed=0)
    xb, _ = ds.generate_batch(1024)
    with torch.no_grad():
        tb = model(xb)
    usage = pd_obj.usage_stats(xb, tb, k)   # batch-top-k selection at this p/k (p-dependent)
    params = {"model": f"biregular-{family}", "p": f"{p:g}", "method": "ExactPD",
              "C": pd_obj.C, "k": k}
    figs = figutil.plot_components(pd_obj, top=top, params=params, usage=usage)
    for name, f in figs.items():
        f.savefig(os.path.join(out_dir, f"{cell}_ExactPD_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(f)

    x2, _ = ds.generate_batch(4096)
    with torch.no_grad():
        t2 = model(x2)
    _, logs = pd_obj.loss(x2, t2, **decomp_fit.loss_kwargs_for("ExactPD", k=k))
    logs["L4"] = l4_loss(model, p=p, device=device)   # held-out (seed 9999) batch
    return logs


def plot_summaries(rows, base, out_dir):
    """Two cross-p summary figures from the per-p rows: loss-vs-p (L4 + ExactPD terms)
    and the gauge-invariant effective through-gains a*c / b*c vs p.

    NB the raw (a,b,c) are NOT individually meaningful: the model is invariant under the
    exact ReLU rescaling (a,b,c) -> (lambda*a, lambda*b, c/lambda), so L4 is flat along
    that orbit and the optimiser lands on an arbitrary point of it. Only the products
    a*c (on-code effective gain) and b*c (off-code leak gain) are determined -- so we
    plot those, not a/c separately."""
    ps = [r["p"] for r in rows]
    ptok = "-".join(f"{p:g}" for p in ps)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    styles = {"L4": dict(color="k", lw=2), "total": dict(color="tab:purple"),
              "minimal": dict(color="tab:blue"), "simple": dict(color="tab:green"),
              "faithful": dict(color="tab:red", ls="--")}
    for t in ("L4", "total", "minimal", "simple", "faithful"):
        ys = [r[t] if r[t] > 0 else np.nan for r in rows]   # log axis drops faithful~0
        ax.plot(ps, ys, marker="o", label=t, **styles[t])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("data sparsity p (log)"); ax.set_ylabel("loss (log)")
    ax.set_title(f"{base}: loss vs p  (model refit at each p; ExactPD terms + L4)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    out = os.path.join(out_dir, f"{base}_loss_vs_p_{ptok}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(ps, [r["a"] * r["c"] for r in rows], marker="o", color="tab:blue",
            label="a·c  (on-code effective gain)")
    ax.plot(ps, [r["b"] * r["c"] for r in rows], marker="o", color="tab:red",
            label="b·c  (off-code leak gain)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("data sparsity p (log)"); ax.set_ylabel("effective through-gain")
    ax.set_title(f"{base}: effective gains vs p\n"
                 f"(gauge-invariant a·c, b·c; raw a,b,c undefined up to ReLU rescaling)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    out = os.path.join(out_dir, f"{base}_gains_vs_p_{ptok}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=5, choices=K_CHOICES,
                    help="codeword length (per-feature degree); default 5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="regular", choices=["regular", "random"])
    ap.add_argument("--p", type=float, default=P, help="sparsity for a single run (default 0.02)")
    ap.add_argument("--p-sweep", action="store_true", dest="p_sweep",
                    help="refit + render the full suite at each p in --ps")
    ap.add_argument("--ps", type=float, nargs="+", default=PS_SWEEP,
                    help=f"p values for the sweep (default {PS_SWEEP})")
    ap.add_argument("--top", type=int, default=16, help="# top components in the detail views")
    ap.add_argument("--fit-steps", type=int, default=4000, dest="fit_steps")
    ap.add_argument("--a", type=float, default=None, help="fix a (skip refit; also needs --b --c)")
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.a is not None and (a.b is None or a.c is None):
        ap.error("--a requires --b and --c (fixing the ansatz scalars skips the refit)")
    fixed_abc = None if a.a is None else (a.a, a.b, a.c)
    M = (codes.regular_code if a.family == "regular" else codes.random_code)(F, N, a.K, a.seed)
    base = f"biregular_{a.family}_K{a.K}_seed{a.seed}"
    os.makedirs(FIG_ROOT, exist_ok=True)

    d = M.sum(0)
    print(f"biregular ansatz  F={F} N={N} K={a.K} seed={a.seed} family={a.family}")
    print(f"  code: feature deg = {int(M.sum(1)[0])} (all K); neuron deg range = "
          f"[{int(d.min())}, {int(d.max())}]  (biregular => uniform)")
    print("  decoder W_out = c * pinv(M.T) (scaled pseudoinverse of the code; only a,b,c fit)")

    ps = a.ps if a.p_sweep else [a.p]
    rows = []
    for p in ps:
        abc = fixed_abc if fixed_abc is not None else fit_abc(M, p, steps=a.fit_steps, device=a.device)
        model = model_from_abc(M, abc, device=a.device)
        k = max(1, min(F - 1, round(p * F)))
        logs = render_cell(model, M, base, a.family, p, k, a.top, FIG_ROOT, device=a.device)
        rows.append({"p": p, "k": k, "a": abc[0], "b": abc[1], "c": abc[2], **logs})
        tag = "[fit]" if fixed_abc is None else "[fixed]"
        print(f"  p={p:<6} k={k:<3} {tag} (a,b,c)=({abc[0]:.4f},{abc[1]:.4f},{abc[2]:.4f})  "
              f"L4={logs['L4']:.2e}  minimal={logs['minimal']:.2e}  faithful={logs['faithful']:.1e}")

    if len(rows) > 1:
        plot_summaries(rows, base, FIG_ROOT)
    print(f"  figures -> {FIG_ROOT}  (base: {base})")


if __name__ == "__main__":
    main()
