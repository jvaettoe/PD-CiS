"""ExactPD single-component vs full-model one-hot response on the biregular code
(memo iteration: c-free 2-parameter ansatz, self-contained — no overview/sweep deps).

The biregular models are analytic MLPs: w_in = a*M + b*(1-M); decoder = w_in.T
(transpose, memo main text) or pinv(M.T).T (pinv, appendix). ExactPD gives one
rank-1 component per feature. We probe with one-hot inputs e_j and compare the
FULL model output (spike + interference leak) against the SINGLE component-j
output (leak-free spike).

For the transpose decoder the off-target leak is EXACTLY quantized by codeword
overlap m = |code_j ∩ code_j'|:  r_{j'} = m*a^2 + (K-m)*a*b   (unit probe), so
`plot_output_spectrum_gains` colors bars by m and overlays these predicted levels
— the appendix verification figure.

  ./.venv/bin/python biregular_single_component.py                 # full memo set, p=0.001
  ./.venv/bin/python biregular_single_component.py --p 0.001 --js 0 33 66
"""
import argparse
import os

from core.paths import REPO_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import biregular.biregular_ansatz as bir
from core.pd import ExactPD

BIREG_K = 5
SWAPS = 800000                      # overlap-min iteration budget (source repo value)
VARIANTS = [("no-swap", 0), ("overlap-min", SWAPS)]
FIG_BASE = os.path.join(REPO_ROOT, "figures", "biregular")
RES_DIR = os.path.join(REPO_ROOT, "results", "overview")

_CODE_CACHE: dict = {}


def _code(K: int, swaps: int, decoder: str, seed: int = 0) -> "bir.CodeAnsatz":
    """Memoised CodeAnsatz so the expensive overlap-min build (reduce_overlap 800k)
    and its per-p fit_ab cache are reused within a process."""
    key = (K, swaps, decoder, seed)
    if key not in _CODE_CACHE:
        _CODE_CACHE[key] = bir.CodeAnsatz(K, swaps=swaps, decoder=decoder, seed=seed)
    return _CODE_CACHE[key]


def _swap_tok(swaps: int) -> str:
    return "" if swaps == 0 else f"_swap{swaps}"


def _fig_dir(decoder: str) -> str:
    d = os.path.join(FIG_BASE, decoder)
    os.makedirs(d, exist_ok=True)
    return d


def analyse(K: int, swaps: int, p: str, device="cpu", decoder: str = "transpose") -> dict:
    """Build the biregular MLP for (K, swaps, decoder) at p, run the single-component
    analysis. Adds the biregular `sq_overlap`."""
    ans = _code(K, swaps, decoder)
    model = ans.model(ans.fit_ab(float(p), device=device), device=device)
    d = _analyse_model(model, p, device=device)
    d["sq_overlap"] = ans.sq_overlap()
    return d


def _analyse_model(model, p: str, device="cpu") -> dict:
    """Model-agnostic core: run ExactPD on any MLP, return the full one-hot response
    matrix R, the single-component matrix S (explicit forward_masked), and derived
    per-feature / global interference + per-probe loss metrics."""
    pd_e = ExactPD(model)
    F = model.n_features
    X = torch.eye(F, device=device)                 # probe j = one-hot e_j (row j)
    I = torch.eye(F, device=device)                 # target: relu(e_j) = e_j
    with torch.no_grad():
        R = model(X).detach()                       # (F, F): R[j,:] = FULL output on e_j

        S = torch.zeros(F, F, device=device)        # single-component outputs
        for j in range(F):
            oh = torch.zeros(pd_e.C, device=device); oh[j] = 1.0
            S[j] = pd_e.forward_masked(X[j:j + 1], {"w_in": oh, "w_out": oh})[0]

        diag_gap = (S.diagonal() - R.diagonal()).abs().max().item()
        S_off = S - torch.diag(S.diagonal())
        assert diag_gap < 1e-4, f"component != full model on-target (gap {diag_gap:.1e})"
        assert (pd_e.forward_dense(X) - R).abs().max().item() < 1e-4, "faithfulness"

    s = S.diagonal().clone()
    L = S_off.norm(dim=1)
    off = R - torch.diag(R.diagonal())
    interference = float((off.pow(2).sum() / R.pow(2).sum()).item())

    dfull, dsingle = R - I, S - I
    # per-probe L4 split into ON-target (output j vs target 1) and OFF-target (leak)
    # contributions: l4 = l4_on + l4_off with l4_on = |out_j - 1|^4 / F.
    e4f, e4s = dfull.abs().pow(4), dsingle.abs().pow(4)
    l4_full_on = e4f.diagonal() / F                     # (F,) per probe
    l4_full_off = (e4f.sum(1) - e4f.diagonal()) / F
    l4_single_on = e4s.diagonal() / F
    l4_single_off = (e4s.sum(1) - e4s.diagonal()) / F
    l4_full = l4_full_on + l4_full_off
    l4_single = l4_single_on + l4_single_off
    jf, js = int(l4_full.argmax()), int(l4_single.argmax())   # worst probes
    return {"R": R.cpu().numpy(), "S": S.cpu().numpy(), "s": s.cpu().numpy(),
            "L": L.cpu().numpy(), "interference": interference, "F": F,
            "comp_off_target_max": float(S_off.abs().max().item()),
            "l4_full": float(l4_full.mean()), "l4_single": float(l4_single.mean()),
            "l4_full_on": float(l4_full_on.mean()), "l4_full_off": float(l4_full_off.mean()),
            "l4_single_on": float(l4_single_on.mean()),
            "l4_single_off": float(l4_single_off.mean()),
            "l4_full_max": float(l4_full.max()), "l4_single_max": float(l4_single.max()),
            # worst-probe decomposition (split AT the argmax probe, so parts sum to max)
            "l4_full_max_on": float(l4_full_on[jf]), "l4_full_max_off": float(l4_full_off[jf]),
            "l4_single_max_on": float(l4_single_on[js]),
            "l4_single_max_off": float(l4_single_off[js])}


def plot_loss_bars(res: dict[str, dict], K: int, p: str, decoder: str,
                   stat: str = "mean", tag: str = "") -> None:
    """Forward-pass L4 loss (the training loss), single ExactPD component vs full
    model, per variant. Single panel — the former MSE panel is dropped (memo uses
    L4 only). stat="mean" averages over the 100 one-hot probes; "max" takes the
    worst probe per series. Pass a filtered `res` (e.g. no-swap only) plus a `tag`
    for a variant-subset figure with its own filename."""
    out_dir = _fig_dir(decoder)
    sfx = "" if stat == "mean" else "_max"
    agg = "average" if stat == "mean" else "worst-case (max over 100 one-hot probes)"
    reduce_word = "averaged over" if stat == "mean" else "max over"
    stem = f"biregular_{'avg' if stat == 'mean' else 'max'}_loss_K{K}_p{p}_{decoder}{tag}"
    names = list(res)
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    # stacked: solid = on-target contribution (|out_j - 1|^4 / F), hatched = the
    # off-target leak contribution (sum_{i != j} |out_i|^4 / F); single component's
    # off-target part is exactly 0 by construction.
    for off, base, color, lbl in ((-w / 2, "l4_single", "tab:blue", "single component"),
                                  (+w / 2, "l4_full", "tab:red", "full model")):
        on = [res[n][base + sfx + "_on"] for n in names]
        offv = [res[n][base + sfx + "_off"] for n in names]
        b_on = ax.bar(x + off, on, w, color=color, label=f"{lbl} — on-target")
        b_off = ax.bar(x + off, offv, w, bottom=on, color=color, alpha=0.45,
                       hatch="//", label=f"{lbl} — off-target (leak)")
        tot = [o + f for o, f in zip(on, offv)]
        ax.bar_label(b_off, labels=[f"{t:.2e}" for t in tot], fontsize=8)
    ax.set_xticks(x, names)
    ax.set_title(f"{agg} L4 loss, split on/off-target  (mean_j |out−target|⁴)",
                 fontsize=10)
    ax.set_ylabel(f"L4 ({reduce_word} 100 one-hot probes)")
    ax.legend(fontsize=8)
    fig.suptitle(f"Single ExactPD component vs full model — {agg} forward-pass L4 "
                 f"(K={K}, p={p}, {decoder})", fontsize=11)
    out = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_io_response(res: dict[str, dict], K: int, p: str, decoder: str) -> None:
    """I/O response: one-hot e_j through its single component vs the full model, for
    example features. Single = clean spike at j; full = spike + leakage."""
    out_dir = _fig_dir(decoder)
    stem = f"biregular_io_response_K{K}_p{p}_{decoder}"
    names = list(res)
    F = res[names[0]]["F"]
    examples = [0, F // 3, 2 * F // 3]
    fig, axes = plt.subplots(len(examples), len(names),
                             figsize=(7.0 * len(names), 2.4 * len(examples)),
                             squeeze=False, constrained_layout=True)
    for col, n in enumerate(names):
        R = res[n]["R"]; S = res[n]["S"]
        for row, j in enumerate(examples):
            ax = axes[row][col]
            ax.bar(range(F), R[j], width=1.0, color="tab:red", alpha=0.6,
                   label="full model output")
            ax.bar(range(F), S[j], width=1.0, color="tab:blue", alpha=0.6,
                   label="single component j")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{n}  —  input e_{j}", fontsize=9)
            if row == len(examples) - 1:
                ax.set_xlabel("output feature")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"I/O response — one-hot e_j through its component vs the full model "
                 f"(K={K}, p={p}, {decoder})", fontsize=11)
    out = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def write_ratios(res: dict[str, dict], K: int, p: str, decoder: str) -> None:
    """Print + save the single-component / full-model L4 ratio per variant (mean and
    worst-case aggregation). MSE columns dropped with the MSE panel."""
    os.makedirs(RES_DIR, exist_ok=True)
    stem = f"biregular_single_vs_full_K{K}_p{p}_{decoder}"
    ctx = f"Biregular K{K} p={p} ({decoder})"
    cols = ["variant", "stat", "L4 single", "L4 full", "L4 single/full"]
    rows = []
    for name in list(res):
        d = res[name]
        for stat, sfx in (("mean", ""), ("max", "_max")):
            l4s, l4f = d["l4_single" + sfx], d["l4_full" + sfx]
            rows.append([name, stat, f"{l4s:.3e}", f"{l4f:.3e}", f"{l4s/l4f:.3f}"])
    md = [f"### {ctx}: single-component / full-model forward-pass L4 ratio", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    path = os.path.join(RES_DIR, stem)
    with open(path + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(path + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    print("\n" + "\n".join(md) + f"\n  (saved {path}.md / .csv)")


def plot_component_vs_full(K: int, p: str, j: int, decoder: str, swaps: int = 0,
                           seed: int = 0, device: str = "cpu") -> None:
    """Two-panel comparison on one-hot probe e_j: LEFT = the full model's output
    spectrum (spike + interference leak); RIGHT = the single-component forward pass
    of feature j's ExactPD component alone (leak-free spike; on-target agreement is
    exact by construction). Shared y-limits so the panels compare directly."""
    out_dir = _fig_dir(decoder)
    ans = _code(K, swaps, decoder, seed)
    model = ans.model(ans.fit_ab(float(p), device=device), device=device)
    pd_e = ExactPD(model)
    F = ans.F
    probe = torch.eye(F, device=device)[j:j + 1]
    with torch.no_grad():
        full = model(probe)[0].cpu().numpy()
        oh = torch.zeros(pd_e.C, device=device); oh[j] = 1.0
        single = pd_e.forward_masked(probe, {"w_in": oh, "w_out": oh})[0].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), constrained_layout=True,
                             sharey=True)
    for ax, (spec, ttl) in zip(axes, (
            (full, f"full model  (s = {full[j]:.3f} + leak)"),
            (single, f"ExactPD component {j} alone  (s = {single[j]:.3f}, leak-free)"))):
        colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
        ax.bar(range(F), spec, width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("output feature")
    axes[0].set_ylabel("output value")
    seed_ttl = "" if seed == 0 else f", seed={seed}"
    fig.suptitle(f"Output spectrum — full model vs single-component forward pass  "
                 f"(input e_{j}; K={K}{_swap_tok(swaps)}{seed_ttl}, p={p}, {decoder})",
                 fontsize=11)
    seed_tok = "" if seed == 0 else f"_seed{seed}"
    out_path = os.path.join(
        out_dir,
        f"biregular_component_vs_full_K{K}{_swap_tok(swaps)}{seed_tok}_p{p}_j{j}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


def plot_output_spectrum(K: int, p: str, j: int, decoder: str, swaps: int = 0,
                         device: str = "cpu") -> None:
    """Memo main-body spectrum: full-model output on one-hot probe e_j (bar per
    output feature, target red). No gain lines — see plot_output_spectrum_gains."""
    out_dir = _fig_dir(decoder)
    ans = _code(K, swaps, decoder)
    model = ans.model(ans.fit_ab(float(p), device=device), device=device)
    F = ans.F
    with torch.no_grad():
        out = model(torch.eye(F, device=device)[j:j + 1])[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
    ax.bar(range(F), out, width=1.0, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    leak = float(np.sqrt(max(0.0, (out ** 2).sum() - out[j] ** 2)))
    ax.set_title(f"Output spectrum on e_{j}  (K={K}{_swap_tok(swaps)}, p={p}, {decoder})"
                 f"  —  s={out[j]:.3f}, off-target ‖·‖={leak:.3f}", fontsize=10)
    ax.set_xlabel("output feature"); ax.set_ylabel("output value")
    out_path = os.path.join(out_dir,
                            f"biregular_output_spectrum_K{K}{_swap_tok(swaps)}_p{p}_j{j}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


def plot_output_spectrum_gains(K: int, p: str, j: int, decoder: str, swaps: int = 0,
                               device: str = "cpu", seed: int = 0) -> dict:
    """APPENDIX figure: output spectrum on e_j with off-target bars colored by
    codeword overlap m = |code_j ∩ code_j'| and, for the TRANSPOSE decoder, dashed
    reference lines at the predicted levels r(m) = m*a^2 + (K-m)*a*b for
    m = 0..maxov and the on-target level K*a^2. This verifies the overlap-noise
    mechanism: m=0 features sit at the negative floor K*a*b; any shared neuron
    contributes +a^2 (reads AND writes on-code), flipping the sign for m>=1.

    For pinv the leak is the rank-deficiency error of the pseudoinverse (no exact
    per-m levels); bars are still colored by m, lines omitted.

    Returns {m: max |bar - r(m)|} for the transpose verification."""
    out_dir = _fig_dir(decoder)
    ans = _code(K, swaps, decoder, seed)
    ab = ans.fit_ab(float(p), device=device)
    model = ans.model(ab, device=device)
    F = ans.F
    ov = ans.overlap_matrix()[j]                    # (F,) overlap with probe feature
    with torch.no_grad():
        out = model(torch.eye(F, device=device)[j:j + 1])[0].cpu().numpy()

    maxov = int(max(ov[i] for i in range(F) if i != j))
    m_colors = {0: "steelblue", 1: "tab:orange", 2: "tab:purple", 3: "tab:brown"}
    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
    colors = ["tab:red" if i == j else m_colors.get(int(ov[i]), "tab:green")
              for i in range(F)]
    ax.bar(range(F), out, width=1.0, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    # unambiguous marker for the probe feature (a bar at small j hugs the y-axis)
    ax.annotate(f"input\ne_{j}", (j, float(out[j])), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=8, color="tab:red",
                arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.0))

    dev = {}
    if decoder == "transpose":
        for m in range(0, maxov + 1):
            r = ans.noise_level(ab, m)
            n_m = int(sum(1 for i in range(F) if i != j and ov[i] == m))
            ax.axhline(r, color=m_colors.get(m, "tab:green"), ls="--", lw=1.0,
                       label=f"shares {m} of {K} neurons:  {m}·a² + {K - m}·ab "
                             f"= {r:+.3f}   ({n_m} features)")
            bars_m = [out[i] for i in range(F) if i != j and ov[i] == m]
            if bars_m:
                dev[m] = float(max(abs(v - r) for v in bars_m))
        r_self = ans.noise_level(ab, K)
        ax.axhline(r_self, color="tab:red", ls="--", lw=1.0,
                   label=f"target j itself (all {K} neurons):  {K}·a² = {r_self:+.3f}")
        dev["self"] = float(abs(out[j] - r_self))
        leg = ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0.0, frameon=True, fancybox=False,
                        edgecolor="black", framealpha=1.0,
                        title=r"$\bf{Legend}$: predicted response by number of shared"
                              "\ncodeword neurons "
                              f"(a={ab[0]:.3f}, b={ab[1]:.3f})",
                        title_fontsize=8)
        leg.get_title().set_ha("left")
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=m_colors.get(m, "tab:green"))
                   for m in range(0, maxov + 1)]
        leg = ax.legend(handles, [f"shares {m} of {K} neurons"
                                  for m in range(0, maxov + 1)],
                        fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0.0, frameon=True, fancybox=False,
                        edgecolor="black", framealpha=1.0,
                        title=r"$\bf{Legend}$: codeword overlap with probe j",
                        title_fontsize=8)
        leg.get_title().set_ha("left")

    seed_ttl = "" if seed == 0 else f", seed={seed}"
    ax.set_title(f"Output spectrum  (input e_{j}; K={K}{_swap_tok(swaps)}{seed_ttl}, "
                 f"p={p}, {decoder})", fontsize=10)
    ax.set_xlabel("output feature"); ax.set_ylabel("output value")
    seed_tok = "" if seed == 0 else f"_seed{seed}"
    out_path = os.path.join(
        out_dir,
        f"biregular_output_spectrum_gains_K{K}{_swap_tok(swaps)}{seed_tok}_p{p}_j{j}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}" + (f"  (max |bar - level|: "
          + ", ".join(f"m={k}: {v:.1e}" for k, v in dev.items()) if dev else ""))
    return dev


def plot_w_in(K: int, p: str, decoder: str, swaps: int = 0, device: str = "cpu") -> None:
    """Standalone W_in heatmap: the fitted encoder a*M + b*(1-M) (100x50), diverging
    colormap centred at 0 so on-code entries (a > 0) read against the off-code
    background (b < 0). Marginal annotations carry the fitted scalars."""
    out_dir = _fig_dir(decoder)
    ans = _code(K, swaps, decoder)
    a, b = ans.fit_ab(float(p), device=device)
    w_in = a * ans.M.astype(float) + b * (1.0 - ans.M.astype(float))   # (F, N)
    v = max(abs(a), abs(b))
    fig, ax = plt.subplots(figsize=(6.4, 9.2), constrained_layout=True)
    im = ax.imshow(w_in, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                   interpolation="nearest")
    ax.set_title(f"W_in = a·M + b·(1−M)   (K={K}{_swap_tok(swaps)}, p={p}, {decoder})\n"
                 f"a = {a:.4f} (on-code, {K}/row)   b = {b:.4f} (off-code)", fontsize=10)
    ax.set_xlabel("neuron"); ax.set_ylabel("feature")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(out_dir, f"biregular_w_in_K{K}{_swap_tok(swaps)}_p{p}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


def plot_fit_loss(K: int, p: str, decoders: list[str], device: str = "cpu",
                  smooth_w: int = 51) -> None:
    """Training-loss curves of the (a, b) fits from a GENERIC init (a=1, b=0):
    per-step L4 batch loss (moving average) for each (decoder, variant), log-y.
    The convergence check for the memo — descent to the optimum well inside the
    4000-step budget. (The production fits start near the optimum, so their curves
    are flat by construction; the generic init shows the actual dynamics. The L4
    valley is SHALLOW: the generic-init fit plateaus at the same loss scale but can
    land at a slightly different (a, b) than the warm-started production fit — both
    held-out L4 values are reported in the legend.)"""
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    kern = np.ones(smooth_w) / smooth_w
    for decoder in decoders:
        for name, sw in VARIANTS:
            ans = _code(K, sw, decoder)
            ab_ref = ans.fit_ab(float(p), device=device)
            ab_gen = ans.fit_ab(float(p), device=device, init=(1.0, 0.0))
            l4_ref = ans.model(ab_ref, device=device).reconstruction_loss(
                float(p), 4.0, batch=8192, seed=777)
            l4_gen = ans.model(ab_gen, device=device).reconstruction_loss(
                float(p), 4.0, batch=8192, seed=777)
            curve = ans.fit_curves[(float(p), 4000, (1.0, 0.0))]
            s = np.convolve(np.array(curve), kern, mode="valid")
            ax.plot(np.arange(len(s)) + smooth_w // 2, s, lw=1.5,
                    label=f"{decoder} / {name}  (generic-init L4 {l4_gen:.2e}; "
                          f"production {l4_ref:.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("fit step"); ax.set_ylabel(f"L4 batch loss ({smooth_w}-step mov. avg)")
    ax.set_title(f"(a, b) fit convergence from generic init (a=1, b=0) — "
                 f"K={K}, p={p}, Adam + cosine LR, 4000 steps", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    os.makedirs(FIG_BASE, exist_ok=True)
    out = os.path.join(FIG_BASE, f"biregular_fit_loss_K{K}_p{p}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def plot_code_matrix(K: int, seed: int = 0, swaps: int = 0) -> None:
    """The codeword matrix M (F x N binary) with degree marginals: every row sums
    to K, every column to d = F*K/N — the biregularity at a glance. Decoder-
    independent, so it lives at the figures/biregular root."""
    ans = _code(K, swaps, "transpose", seed)     # decoder irrelevant for M
    M = ans.M
    Fn, Nn = M.shape
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 10.4),
        gridspec_kw={"width_ratios": [7, 1.1], "height_ratios": [10, 1.1],
                     "wspace": 0.06, "hspace": 0.05})
    ax, axr, axc, axoff = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    axoff.axis("off")
    ax.imshow(M, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_title(f"Biregular code M  (K={K}{_swap_tok(swaps)}, seed={seed})\n"
                 f"F={Fn} features × N={Nn} neurons", fontsize=11, loc="left", pad=10)
    ax.set_ylabel("feature index")
    ax.tick_params(labelsize=8, length=0, labelbottom=False)
    axr.barh(np.arange(Fn), M.sum(1), height=0.7, color="dimgray")
    axr.set_ylim(ax.get_ylim()); axr.set_xlim(0, K + 2)
    axr.set_xticks([K]); axr.set_title("row sum\n(= K)", fontsize=8)
    d = Fn * K // Nn
    axc.bar(np.arange(Nn), M.sum(0), width=0.7, color="dimgray")
    axc.set_xlim(ax.get_xlim()); axc.set_ylim(0, d + 3)
    axc.set_yticks([d]); axc.set_ylabel("col sum\n(= d)", fontsize=8)
    axc.set_xlabel("neuron index")
    for a2 in (axr, axc):
        a2.tick_params(labelsize=8, length=0)
        for s in a2.spines.values():
            s.set_visible(False)
    axr.tick_params(labelleft=False)
    os.makedirs(FIG_BASE, exist_ok=True)
    out = os.path.join(FIG_BASE, f"biregular_code_matrix_K{K}{_swap_tok(swaps)}_seed{seed}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def plot_w_out(K: int, p: str, decoder: str, swaps: int = 0, device: str = "cpu") -> None:
    """Standalone W_out heatmap (N x F). pinv: the FIXED parameter-free decoder
    pinv(M.T).T — dense, carries the interference cancellation. transpose: the tied
    decoder w_in.T (a on-code, b off-code, same scalars as the encoder)."""
    out_dir = _fig_dir(decoder)
    ans = _code(K, swaps, decoder)
    a, b = ans.fit_ab(float(p), device=device)
    w_out = ans.model((a, b), device=device).w_out.detach().cpu().numpy()   # (N, F)
    v = float(np.abs(w_out).max())
    desc = ("pinv(Mᵀ)ᵀ — fixed, no learned parameters"
            if decoder == "pinv" else f"w_inᵀ (tied; a = {a:.4f}, b = {b:.4f})")
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
    # symmetric log colour scale: the pinv decoder's mass sits at small magnitudes
    # under a few large entries, so a linear scale washes it out near white
    norm = matplotlib.colors.SymLogNorm(linthresh=0.1 * v, vmin=-v, vmax=v)
    im = ax.imshow(w_out, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")
    ax.set_title(f"W_out = {desc}   (K={K}{_swap_tok(swaps)}, p={p}, {decoder})",
                 fontsize=10)
    ax.set_xlabel("feature"); ax.set_ylabel("neuron")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(out_dir, f"biregular_w_out_K{K}{_swap_tok(swaps)}_p{p}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


R2_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def r2_report(K: int, p: str, decoders: list[str], device: str = "cpu") -> None:
    """Per-feature R^2 census of the learned computation: for each (decoder, variant)
    cell, the behavioural per-feature R^2 (mlp.per_feature_r2 at p: active-conditional,
    interference included) tabulated as threshold counts, alongside the fitted
    scalars a, b and the products a^2, a*b. Table -> results/overview; one figure per
    decoder (sorted R^2 curves, threshold lines) -> figures/biregular/<decoder>/.

    Thresholds 0.5/0.8/0.9/0.95/0.99 plus the threshold-free soft count
    n_soft = sum(max(0, R^2)) and min/median: the interesting structure at p=0.001
    is entirely in the high tail (all features clear 0.8), so the fine-grained
    top thresholds do the discriminating."""
    os.makedirs(RES_DIR, exist_ok=True)
    cols = (["decoder", "variant", "a", "b", "a^2", "a*b", "n_soft", "min R2",
             "median R2"] + [f">{t:g}" for t in R2_THRESHOLDS])
    rows = []
    curves = {}
    for decoder in decoders:
        for name, sw in VARIANTS:
            ans = _code(K, sw, decoder)
            a, b = ans.fit_ab(float(p), device=device)
            model = ans.model((a, b), device=device)
            r2 = model.per_feature_r2(float(p)).clamp(min=0)
            curves[(decoder, name)] = np.sort(r2.cpu().numpy())[::-1]
            rows.append([decoder, name, f"{a:.4f}", f"{b:.4f}", f"{a*a:.4f}",
                         f"{a*b:.4f}", f"{float(r2.sum()):.1f}",
                         f"{float(r2.min()):.3f}", f"{float(r2.median()):.3f}"]
                        + [str(int((r2 > t).sum())) for t in R2_THRESHOLDS])
    stem = f"biregular_r2_K{K}_p{p}"
    md = [f"### Per-feature R^2 census — biregular K={K}, p={p} "
          f"(F=100 target ReLUs on N=50 neurons)", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    path = os.path.join(RES_DIR, stem)
    with open(path + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(path + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    print("\n" + "\n".join(md) + f"\n  (saved {path}.md / .csv)")

    for decoder in decoders:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for (dec, name), ys in curves.items():
            if dec != decoder:
                continue
            ax.plot(range(1, len(ys) + 1), ys, lw=1.8,
                    label=f"{name}  (min {ys[-1]:.3f}, median {np.median(ys):.3f})")
        for t in R2_THRESHOLDS:
            ax.axhline(t, color="grey", ls=":", lw=0.7)
            ax.text(101.5, t, f"{t:g}", va="center", fontsize=7, color="grey")
        row = next(r for r in rows if r[0] == decoder)   # no-swap row annotates a,b
        ax.set_title(f"Per-feature R² (sorted) — K={K}, p={p}, {decoder}\n"
                     f"no-swap fit: a={row[2]}, b={row[3]}, a²={row[4]}, ab={row[5]}",
                     fontsize=10)
        ax.set_xlabel("feature rank"); ax.set_ylabel("per-feature R²")
        ax.set_xlim(0, 106); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower left")
        out = os.path.join(_fig_dir(decoder), f"{stem}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=BIREG_K)
    ap.add_argument("--p", default="0.001")
    ap.add_argument("--js", type=int, nargs="+", default=[0, 33, 66],
                    help="probe FEATURE indices for the spectrum figures")
    ap.add_argument("--decoders", nargs="+", default=["transpose", "pinv"],
                    choices=["transpose", "pinv"],
                    help="transpose = memo main text; pinv = appendix")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    for decoder in a.decoders:
        print(f"[{decoder}] single-component suite, K={a.K}, p={a.p}")
        res = {name: analyse(a.K, sw, a.p, device=a.device, decoder=decoder)
               for name, sw in VARIANTS}
        for name in res:
            so = res[name]["sq_overlap"]
            print(f"  {name}: interference={res[name]['interference']:.4f}  "
                  f"sum(ov^2)={so[0]}  maxov={so[1]}")
        plot_loss_bars(res, a.K, a.p, decoder, stat="mean")
        plot_loss_bars(res, a.K, a.p, decoder, stat="max")
        plot_io_response(res, a.K, a.p, decoder)
        write_ratios(res, a.K, a.p, decoder)
        plot_w_in(a.K, a.p, decoder, device=a.device)
        plot_w_out(a.K, a.p, decoder, device=a.device)
        for j in a.js:
            plot_output_spectrum(a.K, a.p, j, decoder, device=a.device)
            plot_output_spectrum_gains(a.K, a.p, j, decoder, device=a.device)
            plot_component_vs_full(a.K, a.p, j, decoder, device=a.device)
    r2_report(a.K, a.p, a.decoders, device=a.device)
    plot_fit_loss(a.K, a.p, a.decoders, device=a.device)
    plot_code_matrix(a.K)


if __name__ == "__main__":
    main()
