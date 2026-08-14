"""The biregular "2-parameter ansatz" model and our ExactPD on it (memo iteration).

A designed biregular code M = regular_code(F, N, K) defines a hand-built
single-hidden-layer model whose only free parameters are TWO scalars (a, b) fit
to L4 loss:

    w_in  = a * M + b * (1 - M)          # (F, N): a on-code, b off-code (encoder)
    w_out = pinv(M.T).T   [pinv]         # (N, F): pseudoinverse decoder, OR
    w_out = w_in.T        [transpose]    # (N, F): tied decoder

The former third scalar c (a decoder scale) was REMOVED: under the exact ReLU
rescaling (a, b, c) -> (lambda*a, lambda*b, c/lambda) only the products were ever
determined, so c is pure gauge. With c gone the determined quantities are:
    pinv:      a  (on-code through-gain),      b  (off-code leak gain)
    transpose: a^2 (self-attenuation),         a*b (negative noise)
and, for the transpose decoder, the codeword-overlap noise levels on a unit
one-hot probe e_j: a feature j' sharing m codeword neurons with j responds at
    r_{j'} = m*a^2 + (K-m)*a*b
(m=0: K*a*b < 0 the negative noise floor; m>=1: positive, because each shared
neuron reads AND writes with the on-code weight a; j itself: K*a^2).

  ./.venv/bin/python biregular_ansatz.py --decoder transpose --p 0.001
  ./.venv/bin/python biregular_ansatz.py --decoder pinv --p 0.001
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import codes
from data import CleanDataset
from mlp import MLP
from pd import ExactPD

F, N = 100, 50            # feature count, neuron count
P_DEFAULT = 0.001         # the memo's sparsity
K_CHOICES = (4, 5, 6, 7)

FIG_BASE = os.path.join(os.path.dirname(__file__), "figures", "biregular")
RES_DIR = os.path.join(os.path.dirname(__file__), "results", "overview")

# ExactPD loss kwargs (formerly decomp_fit.loss_kwargs_for("ExactPD", k)):
EXACTPD_LOSS_KW = dict(alpha_faithfulness=1.0, alpha_minimality=1.0,
                       alpha_simplicity=1.0, p=1.0)


class CodeAnsatz:
    """A designed binary code plus its 2-scalar ansatz.

    State: the (F, N) boolean code `M` (regular_code / random_code, optionally
    overlap-minimised via codes.reduce_overlap), its identity (family, K, seed,
    swaps), the fixed decoder base pinv(M.T) (cached), and per-p (a, b) fits
    (cached). The model is
        w_in = a*M + b*(1-M),   w_out = pinv(M.T).T  or  w_in.T
    -- no decoder scale: the former c was gauge freedom and has been removed.
    `model()` returns a plain MLP, so this class sits BESIDE the MLP hierarchy
    (a factory + bookkeeping), not inside it."""

    def __init__(self, K: int, *, family: str = "regular", seed: int = 0, swaps: int = 0,
                 decoder: str = "pinv", n_features: int = F, n_neurons: int = N):
        assert decoder in ("pinv", "transpose"), decoder
        self.K, self.family, self.seed, self.swaps = K, family, seed, swaps
        self.decoder = decoder
        self.F, self.N = n_features, n_neurons
        code_fn = codes.regular_code if family == "regular" else codes.random_code
        M = code_fn(n_features, n_neurons, K, seed)
        self.sq_overlap_pre = self._sq_overlap(M)
        if swaps > 0:
            M = codes.reduce_overlap(M, swaps, seed=10 + seed)  # source convention
        self.M = M
        self._pinv = None            # decoder base, computed once on demand
        self._ab_cache: dict = {}    # (p, steps) -> (a, b)
        self.fit_curves: dict = {}   # (p, steps) -> [per-step L4 batch loss]

    @property
    def name(self) -> str:
        """Filename base token; stamps every identifying param."""
        swap_tok = f"_swap{self.swaps}" if self.swaps > 0 else ""
        dec_tok = "" if self.decoder == "pinv" else f"_{self.decoder}"
        return f"biregular_{self.family}{swap_tok}{dec_tok}_K{self.K}_seed{self.seed}"

    @staticmethod
    def _sq_overlap(M) -> tuple[int, int]:
        """(sum(overlap^2), max overlap) over feature pairs."""
        ov = M.astype(int) @ M.astype(int).T
        np.fill_diagonal(ov, 0)
        return int((ov ** 2).sum()), int(ov.max())

    def sq_overlap(self) -> tuple[int, int]:
        return self._sq_overlap(self.M)

    def overlap_matrix(self) -> np.ndarray:
        """(F, F) int codeword-overlap counts m_{jj'} = |code_j intersect code_j'|
        (diagonal = K). Drives the transpose noise levels m*a^2 + (K-m)*a*b."""
        return (self.M.astype(int) @ self.M.astype(int).T)

    def pinv_wout(self) -> np.ndarray:
        """Decoder base: w_out = (pinv(M.T)).T, shape (N, F)."""
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.M.T.astype(np.float64)).T.astype(np.float32)
        return self._pinv

    def model(self, ab, device="cpu") -> MLP:
        """Build the MLP(F, N) for scalars (a, b)."""
        a, b = ab
        Mf = self.M.astype(np.float64)
        w_in = (a * Mf + b * (1.0 - Mf)).astype(np.float32)      # (F, N)
        if self.decoder == "transpose":
            w_out = w_in.T.copy()                                 # (N, F) tied decoder
        else:
            w_out = self.pinv_wout().copy()                       # (N, F) pseudoinverse
        model = MLP(n_features=self.F, d_mlp=self.N, device=device)
        with torch.no_grad():
            model.w_in.copy_(torch.from_numpy(w_in).to(device))
            model.w_out.copy_(torch.from_numpy(w_out).to(device))
        return model

    def gains(self, ab) -> dict:
        """Decoder-aware determined quantities. pinv: the through-gains are a and b
        directly (decoder fixed). transpose: self-attenuation a^2, negative noise a*b,
        and the m=1 positive-overlap level a^2 + (K-1)*a*b."""
        a, b = ab
        if self.decoder == "transpose":
            return {"a2": a * a, "ab": a * b,
                    "pos_m1": a * a + (self.K - 1) * a * b}
        return {"a": a, "b": b}

    def noise_level(self, ab, m: int) -> float:
        """Transpose-decoder response at overlap m on a unit one-hot probe:
        r = m*a^2 + (K-m)*a*b  (m = K gives the on-target self-attenuation)."""
        a, b = ab
        return m * a * a + (self.K - m) * a * b

    def fit_ab(self, p: float, *, steps: int = 4000, batch: int = 4096, pool: int = 131072,
               lr: float = 0.01, device: str = "cpu", seed: int = 31337,
               init: tuple | None = None):
        """Fit (a, b) to L4 loss for the FIXED code at sparsity p. The decoder has no
        free parameters (pinv base fixed; transpose tied to the encoder). Adam +
        cosine LR; ONE pre-generated data pool sampled by index. Cached by
        (p, steps, init); default init = near the known optimum, `init=(1.0, 0.0)`
        etc. gives a generic start (used by the convergence figure)."""
        key = (float(p), steps, init)
        if key in self._ab_cache:
            return self._ab_cache[key]
        Mf = torch.tensor(self.M.astype(np.float32), device=device)      # (F, N)
        PINV = torch.tensor(self.pinv_wout(), device=device)             # (N, F)
        ds = CleanDataset(n_features=self.F, p=p, device=device, seed=seed)
        x_pool, y_pool = ds.generate_batch(pool)                         # generated ONCE
        gen = torch.Generator(device=device).manual_seed(seed)
        # default inits at the old fits' gauge-invariant products (c==1 gauge)
        if init is not None:
            a0, b0 = init
        else:
            a0, b0 = (0.4, -0.04) if self.decoder == "transpose" else (1.5, -0.05)
        a = torch.tensor(a0, device=device, requires_grad=True)
        b = torch.tensor(b0, device=device, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        curve = []
        for _ in range(steps):
            idx = torch.randint(0, pool, (batch,), device=device, generator=gen)
            x, y = x_pool[idx], y_pool[idx]
            w_in = a * Mf + b * (1 - Mf)
            w_out = w_in.T if self.decoder == "transpose" else PINV
            loss = ((torch.relu(x @ w_in) @ w_out - y).abs() ** 4).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            curve.append(float(loss))
        ab = (float(a), float(b))
        self._ab_cache[key] = ab
        self.fit_curves[key] = curve
        return ab


def render_cell(model, code: CodeAnsatz, p, k, top, out_dir, device="cpu"):
    """Render the FULL viz suite for one (model, p) cell and return its loss terms."""
    M, base, family = code.M, code.name, code.family
    cell = f"{base}_p{p:g}_k{k}"

    # share_v=False: per-panel colour scales — the pinv decoder's W_out (max ~0.19)
    # is invisible on W_in's scale (max ~1.5)
    fig = model.plot_weights(share_v=False)
    fig.suptitle(f"Biregular ansatz weights  ({family}, {code.decoder}, p={p:g}) — "
                 f"w_in = a*M + b*(1-M)")
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
    fig.savefig(os.path.join(out_dir, f"{cell}_code_and_effective.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    pd_obj = ExactPD(model)
    ds = CleanDataset(n_features=F, p=p, device=device, seed=0)
    xb, _ = ds.generate_batch(1024)
    with torch.no_grad():
        tb = model(xb)
    usage = pd_obj.usage_stats(xb, tb, k)
    params = {"model": f"biregular-{family}", "p": f"{p:g}", "method": "ExactPD",
              "C": pd_obj.C, "k": k}
    figs = pd_obj.plot_components(top=top, params=params, usage=usage)
    for name, f in figs.items():
        f.savefig(os.path.join(out_dir, f"{cell}_ExactPD_{name}.png"), dpi=130,
                  bbox_inches="tight")
        plt.close(f)

    x2, _ = ds.generate_batch(4096)
    with torch.no_grad():
        t2 = model(x2)
    _, logs = pd_obj.loss(x2, t2, k=k, **EXACTPD_LOSS_KW)
    logs["L4"] = model.reconstruction_loss(p, 4.0, batch=4096, seed=9999)  # held-out
    return logs


def write_learning_table(rows, code: CodeAnsatz, path: str) -> None:
    """CSV learning table: fitted (a, b) plus the decoder's determined quantities.
    transpose: a2 (self-attenuation), ab (negative noise), pos_m1 = a^2+(K-1)ab (the
    m=1 codeword-overlap positive-noise level); pinv: a, b are the gains directly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gain_keys = ["a2", "ab", "pos_m1"] if code.decoder == "transpose" else []
    fields = ["p", "k", "a", "b"] + gain_keys + ["L4", "total", "minimal", "simple",
                                                 "faithful"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  learning table -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=5,
                    help="codeword length; F*K must be divisible by N "
                         f"(source paper swept {K_CHOICES}; K=20 matches the "
                         "learned code density at p=0.001)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="regular", choices=["regular", "random"])
    ap.add_argument("--decoder", default="transpose", choices=["transpose", "pinv"],
                    help="transpose = memo main text; pinv = appendix section")
    ap.add_argument("--p", type=float, default=P_DEFAULT)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=0)
    ap.add_argument("--fit-steps", type=int, default=4000, dest="fit_steps")
    ap.add_argument("--a", type=float, default=None, help="fix a (skip refit; needs --b)")
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if (a.a is None) != (a.b is None):
        ap.error("--a and --b must be given together (fixing the scalars skips the refit)")
    fixed_ab = None if a.a is None else (a.a, a.b)
    code = CodeAnsatz(a.K, family=a.family, seed=a.seed, swaps=a.swaps, decoder=a.decoder)
    if a.swaps > 0:
        ov0, ov1 = code.sq_overlap_pre, code.sq_overlap()
        print(f"  overlap-min ({a.swaps} swaps): sum(ov^2) {ov0[0]} -> {ov1[0]}, "
              f"maxov {ov0[1]} -> {ov1[1]}")
    fig_root = os.path.join(FIG_BASE, a.decoder)
    os.makedirs(fig_root, exist_ok=True)

    d = code.M.sum(0)
    print(f"biregular ansatz  F={F} N={N} K={a.K} seed={a.seed} family={a.family} "
          f"swaps={a.swaps} decoder={a.decoder}")
    print(f"  code: feature deg = {int(code.M.sum(1)[0])} (all K); neuron deg range = "
          f"[{int(d.min())}, {int(d.max())}]  (biregular => uniform)")
    dec_desc = ("w_out = w_in.T (tied)" if a.decoder == "transpose"
                else "w_out = pinv(M.T).T (fixed)")
    print(f"  decoder {dec_desc}; only (a, b) fit — c removed (was pure gauge)")

    p = a.p
    ab = fixed_ab if fixed_ab is not None else code.fit_ab(p, steps=a.fit_steps,
                                                           device=a.device)
    model = code.model(ab, device=a.device)
    k = max(1, min(F - 1, round(p * F)))
    logs = render_cell(model, code, p, k, a.top, fig_root, device=a.device)
    g = code.gains(ab)
    row = {"p": p, "k": k, "a": ab[0], "b": ab[1], **g, **logs}
    tag = "[fit]" if fixed_ab is None else "[fixed]"
    gain_str = "  ".join(f"{k_}={v:.4f}" for k_, v in g.items())
    print(f"  p={p:<6} k={k:<3} {tag} (a,b)=({ab[0]:.4f},{ab[1]:.4f})  {gain_str}")
    print(f"  L4={logs['L4']:.2e}  minimal={logs['minimal']:.2e}  "
          f"faithful={logs['faithful']:.1e}")
    write_learning_table([row], code,
                         os.path.join(RES_DIR, f"{code.name}_p{p:g}_learning.csv"))
    print(f"  figures -> {fig_root}  (base: {code.name})")


if __name__ == "__main__":
    main()
