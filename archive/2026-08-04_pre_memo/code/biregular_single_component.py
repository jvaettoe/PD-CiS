"""ExactPD single-component vs full-model one-hot response on the biregular code.

The biregular code models are analytic MLPs: w_in = a*M + b*(1-M) (rows =
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
from overview import BIREG_K, SWAPS, OVERVIEW_FIG, OVERVIEW_RES, _load_trained
from biregular_decoder_compare import ignore_half_model
from sweep import PS

VARIANTS = [("no-swap", 0), ("overlap-min", SWAPS)]

_CODE_CACHE: dict = {}


def _code(K: int, swaps: int, decoder: str, seed: int = 0) -> "bir.CodeAnsatz":
    """Memoised CodeAnsatz so the expensive overlap-min build (reduce_overlap 800k) and its
    per-p fit_abc cache are reused within a process."""
    key = (K, swaps, decoder, seed)
    if key not in _CODE_CACHE:
        _CODE_CACHE[key] = bir.CodeAnsatz(K, swaps=swaps, decoder=decoder, seed=seed)
    return _CODE_CACHE[key]


def _swap_tok(swaps: int) -> str:
    """Filename token distinguishing overlap-min outputs from no-swap ('' when swaps==0)."""
    return "" if swaps == 0 else f"_swap{swaps}"


def analyse(K: int, swaps: int, p: str, device="cpu", decoder: str = "pinv") -> dict:
    """Biregular wrapper: build the biregular MLP for (K, swaps, decoder) at p, then
    run the model-agnostic single-component analysis. Adds the biregular `sq_overlap`."""
    ans = _code(K, swaps, decoder)
    model = ans.model(ans.fit_abc(float(p), device=device), device=device)
    d = _analyse_model(model, p, device=device)
    d["sq_overlap"] = ans.sq_overlap()
    return d


def _analyse_model(model, p: str, device="cpu") -> dict:
    """Model-agnostic core: run ExactPD on any MLP, return the full one-hot
    response matrix R, the single-component matrix S (explicit forward_masked), and derived
    per-feature / global interference + per-probe loss metrics. Works for the biregular codes
    and the trained MLP alike (ExactPD component c writes only to output c → single =
    on-target only)."""
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
    l2_full, l4_full = dfull.pow(2).mean(1), dfull.abs().pow(4).mean(1)      # per-probe (F,)
    l2_single, l4_single = dsingle.pow(2).mean(1), dsingle.abs().pow(4).mean(1)
    return {"R": R.cpu().numpy(), "S": S.cpu().numpy(), "s": s.cpu().numpy(),
            "L": L.cpu().numpy(), "interference": interference, "F": F,
            "comp_off_target_max": float(S_off.abs().max().item()),
            # mean over the 100 one-hot probes
            "l2_full": float(l2_full.mean()), "l2_single": float(l2_single.mean()),
            "l4_full": float(l4_full.mean()), "l4_single": float(l4_single.mean()),
            # worst case: max over probes (independent per series)
            "l2_full_max": float(l2_full.max()), "l2_single_max": float(l2_single.max()),
            "l4_full_max": float(l4_full.max()), "l4_single_max": float(l4_single.max())}


def plot_loss_bars(res: dict[str, dict], K: int, p: str, tag: str = "",
                   stat: str = "mean", stem: str | None = None,
                   ctx: str | None = None) -> None:
    """Forward-pass loss, single ExactPD component vs full model, per variant, in two
    panels (MSE and the training L4). stat="mean" averages over the 100 one-hot probes;
    stat="max" takes the worst probe per series (worst single component vs the full
    model's worst probe -- independent maxes). `stem` overrides the output filename stem
    (no extension), `ctx` the title context; both default to the biregular naming."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    sfx = "" if stat == "mean" else "_max"
    agg = "average" if stat == "mean" else "worst-case (max over 100 one-hot probes)"
    reduce_word = "averaged over" if stat == "mean" else "max over"
    if stem is None:
        base = "biregular_avg_loss" if stat == "mean" else "biregular_max_loss"
        stem = f"{base}_K{K}_p{p}{tag}"
    if ctx is None:
        ctx = f"K={K}, p={p}{tag and ', ' + tag}"
    names = list(res)
    x = np.arange(len(names)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, (key_s, key_f, ttl) in zip(axes, [
            ("l2_single", "l2_full", f"{agg} MSE  (mean_j (out−target)²)"),
            ("l4_single", "l4_full", f"{agg} L4 loss  (mean_j |out−target|⁴, the training loss)")]):
        vs = [res[n][key_s + sfx] for n in names]
        vf = [res[n][key_f + sfx] for n in names]
        b1 = ax.bar(x - w / 2, vs, w, label="single component", color="tab:blue")
        b2 = ax.bar(x + w / 2, vf, w, label="full model", color="tab:red")
        ax.bar_label(b1, fmt="%.2e", fontsize=8); ax.bar_label(b2, fmt="%.2e", fontsize=8)
        ax.set_xticks(x, names); ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(f"loss ({reduce_word} 100 one-hot probes)")
        ax.legend(fontsize=9)
    fig.suptitle(f"Single ExactPD component vs full model — {agg} forward-pass loss on "
                 f"one-hot inputs e_c  ({ctx})", fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_io_response(res: dict[str, dict], K: int, p: str, tag: str = "",
                     stem: str | None = None, ctx: str | None = None) -> None:
    """I/O response: the same one-hot e_c through its single component vs the full
    model, for a few example features. Single = clean spike at c; full = spike + leakage."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    if stem is None:
        stem = f"biregular_io_response_K{K}_p{p}{tag}"
    if ctx is None:
        ctx = f"K={K}, p={p}{tag and ', ' + tag}"
    names = list(res)
    F = res[names[0]]["F"]
    examples = [0, F // 3, 2 * F // 3]
    fig, axes = plt.subplots(len(examples), len(names),
                             figsize=(7.0 * len(names), 2.4 * len(examples)),
                             squeeze=False, constrained_layout=True)
    for col, n in enumerate(names):
        R = res[n]["R"]; S = res[n]["S"]
        for row, c in enumerate(examples):
            ax = axes[row][col]
            ax.bar(range(F), R[c], width=1.0, color="tab:red", alpha=0.6,
                   label="full model output")
            # explicit single-component output over ALL outputs (localized on c);
            # translucent + matched width so the overlap at c blends instead of hiding red
            ax.bar(range(F), S[c], width=1.0, color="tab:blue", alpha=0.6,
                   label="single component c")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{n}  —  input e_{c}", fontsize=9)
            if row == len(examples) - 1:
                ax.set_xlabel("output feature j")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"I/O response — one-hot e_c through its component vs the full model  "
                 f"(single = self-response at c only; full = self + leakage)  "
                 f"({ctx})", fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def write_ratios(res: dict[str, dict], K: int, p: str, tag: str, decoder: str,
                 stem: str | None = None, ctx: str | None = None) -> None:
    """Print + save the single-component / full-model loss ratio (full model = benchmark
    denominator), per variant, for both mean and worst-case (max) aggregation."""
    os.makedirs(OVERVIEW_RES, exist_ok=True)
    if stem is None:
        stem = f"biregular_single_vs_full_K{K}_p{p}{tag}"
    if ctx is None:
        ctx = f"Biregular K{K} p={p} ({decoder})"
    cols = ["variant", "stat", "MSE single", "MSE full", "MSE single/full",
            "L4 single", "L4 full", "L4 single/full"]
    rows = []
    for name in list(res):
        d = res[name]
        for stat, sfx in (("mean", ""), ("max", "_max")):
            l2s, l2f = d["l2_single" + sfx], d["l2_full" + sfx]
            l4s, l4f = d["l4_single" + sfx], d["l4_full" + sfx]
            rows.append([name, stat, f"{l2s:.3e}", f"{l2f:.3e}", f"{l2s/l2f:.3f}",
                         f"{l4s:.3e}", f"{l4f:.3e}", f"{l4s/l4f:.3f}"])
    md = [f"### {ctx}: single-component / full-model "
          f"forward-pass loss ratio (full model = benchmark)", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    path = os.path.join(OVERVIEW_RES, stem)
    with open(path + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(path + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    print("\n" + "\n".join(md) + f"\n  (saved {path}.md / .csv)")


def plot_single_vs_full_overview(K: int, device="cpu", swaps: int = 0,
                                 providers=None, tag=None, stem: str | None = None,
                                 ctx: str | None = None) -> None:
    """Overview across the whole p range: single-component vs full-model forward-pass loss
    (and their ratio). Series = biregular pinv/transpose by default, or the passed
    `providers` = [(label, analysis_fn(p, device) -> dict), ...]."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True); os.makedirs(OVERVIEW_RES, exist_ok=True)
    tok = _swap_tok(swaps) if tag is None else tag
    if stem is None:
        stem = f"biregular_single_vs_full_overview_K{K}{tok}"
    if ctx is None:
        descr = tag.lstrip("_") if tag else ("overlap-min" if swaps else "no-swap")
        ctx = f"K={K}, {descr}"
    if providers is None:
        providers = [(dec, (lambda p, d, dec=dec: analyse(K, swaps, p, device=d, decoder=dec)))
                     for dec in ("pinv", "transpose")]
    labels = [lab for lab, _ in providers]
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    colr = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}
    data = {lab: {k: [] for k in ("l2_single", "l2_full", "l4_single", "l4_full")}
            for lab in labels}
    for lab, fn in providers:
        for p in PS:
            r = fn(p, device)
            for k in data[lab]:
                data[lab][k].append(r[k])
            print(f"[overview {lab:14} p={p:6}] MSE single={r['l2_single']:.3e} "
                  f"full={r['l2_full']:.3e} ratio={r['l2_single']/r['l2_full']:.3f}")
    xs = [float(p) for p in PS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, key_s, key_f, ttl in [(axes[0], "l2_single", "l2_full", "forward-pass MSE"),
                                   (axes[1], "l4_single", "l4_full", "forward-pass L4")]:
        for lab in labels:
            ax.plot(xs, data[lab][key_f], "o-", color=colr[lab], label=f"{lab} full model")
            ax.plot(xs, data[lab][key_s], "o--", color=colr[lab], alpha=0.7,
                    label=f"{lab} single component")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("data sparsity p"); ax.set_ylabel("loss (mean over 100 one-hot probes)")
        ax.set_title(ttl, fontsize=10); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
    for lab in labels:
        r2 = [s / f for s, f in zip(data[lab]["l2_single"], data[lab]["l2_full"])]
        r4 = [s / f for s, f in zip(data[lab]["l4_single"], data[lab]["l4_full"])]
        axes[2].plot(xs, r2, "o-", color=colr[lab], label=f"{lab} MSE ratio")
        axes[2].plot(xs, r4, "s--", color=colr[lab], alpha=0.7, label=f"{lab} L4 ratio")
    axes[2].set_xscale("log"); axes[2].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel("data sparsity p"); axes[2].set_ylabel("single / full  (full = benchmark)")
    axes[2].set_title("single-component / full-model ratio", fontsize=10)
    axes[2].grid(True, alpha=0.3, which="both"); axes[2].legend(fontsize=7)
    fig.suptitle(f"Single ExactPD component vs full model across p — one-hot forward-pass loss "
                 f"({ctx})", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")

    # results table
    cols = ["series", "p", "MSE single", "MSE full", "MSE ratio",
            "L4 single", "L4 full", "L4 ratio"]
    lines = [f"### {ctx}: single-vs-full forward-pass loss across p", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for lab in labels:
        for i, p in enumerate(PS):
            s2, f2 = data[lab]["l2_single"][i], data[lab]["l2_full"][i]
            s4, f4 = data[lab]["l4_single"][i], data[lab]["l4_full"][i]
            lines.append("| " + " | ".join([lab, p, f"{s2:.3e}", f"{f2:.3e}", f"{s2/f2:.3f}",
                         f"{s4:.3e}", f"{f4:.3e}", f"{s4/f4:.3f}"]) + " |")
    path = os.path.join(OVERVIEW_RES, stem)
    with open(path + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}.md")


def _spectrum_models(K: int, p: str, device="cpu", swaps: int = 0) -> list:
    """The four models compared in the output-spectrum views (name, model), all
    MLP instances so model(one-hot) gives the 100-d output. MLP entry may
    be None if its checkpoint is missing. `swaps` selects no-swap (0) or overlap-min biregular
    code; MLP/ignore-50 are swap-independent."""
    tag = "no-swap" if swaps == 0 else "overlap-min"
    ap = _code(K, swaps, "pinv")
    at = _code(K, swaps, "transpose")
    return [
        (f"biregular pinv ({tag})", ap.model(ap.fit_abc(float(p), device=device), device=device)),
        (f"biregular transpose ({tag})", at.model(at.fit_abc(float(p), device=device), device=device)),
        ("MLP (fully-trained d50)", _load_trained("MLP", None, 50, p, 4.0, device)),
        ("ignore-50 (no superposition)", ignore_half_model(device=device)),
    ]


def plot_output_spectrum(K: int, p: str, c: int, device="cpu", swaps: int = 0) -> None:
    """Output spectrum for a one-hot input e_c: raw signed output value vs output-feature
    index j, for four models side by side (biregular pinv / transpose, fully-trained
    MLP d50, and the ignore-50 no-superposition baseline). Reveals how each model's
    response concentrates on the target vs leaks into other outputs."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    F = 100
    models = _spectrum_models(K, p, device, swaps)
    x = torch.eye(F, device=device)[c:c + 1]              # one-hot e_c, shape (1, F)
    outs = []
    for name, m in models:
        if m is None:
            print(f"  [skip] {name}: no checkpoint"); continue
        with torch.no_grad():
            outs.append((name, m(x)[0].detach().cpu().numpy()))
    ymax = max(float(np.abs(o).max()) for _, o in outs) * 1.08

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True, sharey=True)
    for ax, (name, o) in zip(axes.flat, outs):
        colors = ["indianred" if j == c else "steelblue" for j in range(F)]
        ax.bar(range(F), o, width=1.0, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        leak = float(np.sqrt(max(0.0, (o ** 2).sum() - o[c] ** 2)))
        ax.set_title(f"{name}\n s_c=out[{c}]={o[c]:.3f}   off-target leak ‖·‖={leak:.3f}", fontsize=9)
        ax.set_xlabel("output feature j"); ax.set_ylim(-ymax, ymax)
    axes[0][0].set_ylabel("output value"); axes[1][0].set_ylabel("output value")
    fig.suptitle(f"Output spectrum for one-hot input e_{c}  (red = target feature {c}; "
                 f"target relu(e_c)=e_c)  —  K={K}, p={p}{_swap_tok(swaps) and ', overlap-min'}",
                 fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"biregular_output_spectrum_K{K}{_swap_tok(swaps)}_p{p}_c{c}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_output_spectrum_summary(K: int, p: str, device="cpu", swaps: int = 0) -> None:
    """Aggregate the per-component output spectra over ALL 100 one-hot inputs into summary
    bars per model: mean self-response s̄ = mean_c R[c,c], off-target leakage
    L̄ = mean_c ‖R[c,≠c]‖, and the leakage/self ratio L̄/s̄. Averaging over all components
    (incl. ignore-50's dropped half) makes ignore-50 read s̄=0.5 — its half-coverage — vs the
    superposition models' full-coverage-but-attenuated s̄ with nonzero leakage."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True); os.makedirs(OVERVIEW_RES, exist_ok=True)
    F = 100
    X = torch.eye(F, device=device)
    stats = []   # (name, s_bar, L_bar, ratio)
    for name, m in _spectrum_models(K, p, device, swaps):
        if m is None:
            print(f"  [skip] {name}: no checkpoint"); continue
        with torch.no_grad():
            R = m(X).detach()
        self_r = R.diagonal()
        off = R - torch.diag(self_r)
        s_bar = float(self_r.mean())
        L_bar = float(off.norm(dim=1).mean())        # mean per-input off-target ‖leak‖
        stats.append((name, s_bar, L_bar, L_bar / s_bar if s_bar else float("nan")))
        print(f"  {name:32} self̄={s_bar:.3f}  leak̄={L_bar:.3f}  ratio={stats[-1][3]:.3f}")

    names = [s[0].split(" (")[0] for s in stats]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:gray"][:len(stats)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, idx, ttl in [(axes[0], 1, "mean self-response  s̄ = mean_c R[c,c]"),
                         (axes[1], 2, "off-target leakage  L̄ = mean_c ‖R[c,≠c]‖"),
                         (axes[2], 3, "leakage / self ratio  L̄ / s̄")]:
        vals = [s[idx] for s in stats]
        b = ax.bar(range(len(stats)), vals, color=colors)
        ax.bar_label(b, fmt="%.3f", fontsize=9)
        ax.set_xticks(range(len(stats)), names, rotation=20, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=10)
    axes[2].axhline(1.0, color="k", lw=0.8, ls=":")   # leak == signal
    tok = _swap_tok(swaps)
    fig.suptitle(f"Output-spectrum summary averaged over all 100 components  (K={K}, p={p}"
                 f"{tok and ', overlap-min'}; ignore-50 = no-superposition reference)", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"biregular_output_spectrum_summary_K{K}{tok}_p{p}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")

    cols = ["model", "mean self-response", "off-target leak", "leak/self ratio"]
    lines = [f"### Output-spectrum summary over all components (K{K}, p={p}{tok})", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += [f"| {n} | {s:.4f} | {L:.4f} | {r:.4f} |" for n, s, L, r in stats]
    stem = os.path.join(OVERVIEW_RES, f"biregular_output_spectrum_summary_K{K}{tok}_p{p}")
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(stem + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(f"{n},{s:.6e},{L:.6e},{r:.6e}\n" for n, s, L, r in stats)
    print(f"  saved {stem}.md / .csv")


def plot_output_loss_decomposition(K: int, p: str, device="cpu", swaps: int = 0) -> None:
    """Decompose the average one-hot squared error into on-target (error at the target
    feature c = attenuation) and off-target (interference) parts, averaged over all 100
    components, for the 4 models. SSE scale (no /F) -- raw sum of squared errors over outputs:
        on  = mean_c (R[c,c]-1)^2 ,  off = mean_c Σ_{j≠c} R[c,j]^2   (MSE; L4 analog with |·|^4)
        on + off == mean_c Σ_j (R[c,j]-target)^2  (= F × the mean forward-pass loss).
    Separates the failure modes: ignore-50 -> 100% on-target (drops half, no interference);
    superposition -> off-target-dominated (interference). The on/off SPLIT (%) is scale-free."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True); os.makedirs(OVERVIEW_RES, exist_ok=True)
    F = 100
    X = torch.eye(F, device=device)
    I = torch.eye(F, device=device)
    stats = []   # (name, on_mse, off_mse, on_l4, off_l4)
    for name, m in _spectrum_models(K, p, device, swaps):
        if m is None:
            print(f"  [skip] {name}: no checkpoint"); continue
        with torch.no_grad():
            R = m(X).detach()
        d = R - I                                          # error vs target e_c per row
        diag = d.diagonal()
        on2 = (diag ** 2)                                  # (F,) on-target squared error
        off2 = (d ** 2).sum(dim=1) - on2                   # Σ_{j≠c} R[c,j]^2 per component
        on4 = (diag ** 4)
        off4 = (d.abs() ** 4).sum(dim=1) - on4
        on_mse, off_mse = float(on2.mean()), float(off2.mean())        # SSE scale (no /F)
        on_l4, off_l4 = float(on4.mean()), float(off4.mean())
        stats.append((name, on_mse, off_mse, on_l4, off_l4))
        tot = on_mse + off_mse
        print(f"  {name:32} MSE on={on_mse:.3e} off={off_mse:.3e} total={tot:.3e} "
              f"(on {100*on_mse/tot:.1f}%)  |  L4 on={on_l4:.3e} off={off_l4:.3e}")

    names = [s[0].split(" (")[0] for s in stats]
    x = np.arange(len(stats))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, (io, ioff, ttl) in zip(axes, [(1, 2, "forward-pass MSE"),
                                          (3, 4, "forward-pass L4 (training loss)")]):
        on = np.array([s[io] for s in stats]); off = np.array([s[ioff] for s in stats])
        b1 = ax.bar(x, on, color="tab:orange", label="on-target (attenuation)")
        b2 = ax.bar(x, off, bottom=on, color="tab:purple", label="off-target (interference)")
        for xi, (o, f) in enumerate(zip(on, off)):
            tot = o + f
            ax.text(xi, tot, f"{tot:.1e}\n(on {100*o/tot:.0f}%)" if tot else "0",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("Σ squared error over outputs (SSE, no /F)")
        ax.legend(fontsize=8)
    tok = _swap_tok(swaps)
    fig.suptitle(f"On-target vs off-target squared-error decomposition (avg over all 100 "
                 f"components; SSE scale)  —  K={K}, p={p}{tok and ', overlap-min'}", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"biregular_output_loss_decomp_K{K}{tok}_p{p}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")

    cols = ["model", "MSE on-target", "MSE off-target", "MSE total",
            "L4 on-target", "L4 off-target", "L4 total"]
    lines = [f"### On/off-target loss decomposition over all components (K{K}, p={p}{tok})", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for n, om, ofm, ol, ofl in stats:
        lines.append(f"| {n} | {om:.3e} | {ofm:.3e} | {om+ofm:.3e} | "
                     f"{ol:.3e} | {ofl:.3e} | {ol+ofl:.3e} |")
    stem = os.path.join(OVERVIEW_RES, f"biregular_output_loss_decomp_K{K}{tok}_p{p}")
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(stem + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(f"{n},{om:.6e},{ofm:.6e},{om+ofm:.6e},{ol:.6e},{ofl:.6e},{ol+ofl:.6e}\n"
                     for n, om, ofm, ol, ofl in stats)
    print(f"  saved {stem}.md / .csv")


ON_COLOR, OFF_COLOR = "tab:orange", "tab:purple"


def plot_io_spectrum_style(K: int, p: str, c: int, device="cpu", swaps: int = 0,
                           entries=None, tag=None, stem: str | None = None,
                           ctx: str | None = None) -> None:
    """Single-component vs full-model I/O response in the output-spectrum panel style: raw
    SIGNED output vs feature index j, self/target bar (j==c) red, off-target steelblue,
    shared y. Rows = models (biregular pinv/transpose by default, or the passed `entries`),
    cols = full model | single ExactPD component of feature c. Single = one red spike at c."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    F = 100
    tok = _swap_tok(swaps) if tag is None else tag
    if stem is None:
        stem = f"biregular_io_spectrum_K{K}{tok}_p{p}_c{c}"
    if ctx is None:
        descr = f", {tag.lstrip('_')}" if tag else (", overlap-min" if swaps else "")
        ctx = f"K={K}, p={p}{descr}"
    if entries is None:
        entries = [(dec, analyse(K, swaps, p, device=device, decoder=dec))
                   for dec in ("pinv", "transpose")]
    data = [(lab, r["R"][c], r["S"][c]) for lab, r in entries]
    ymax = max(float(np.abs(v).max()) for _, o, s in data for v in (o, s)) * 1.08

    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.2 * len(data)), squeeze=False,
                             constrained_layout=True, sharey=True)
    for row, (lab, full, single) in enumerate(data):
        for col, (which, o) in enumerate((("full model", full), ("single component c", single))):
            ax = axes[row][col]
            colors = ["indianred" if j == c else "steelblue" for j in range(F)]
            ax.bar(range(F), o, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            leak = float(np.sqrt(max(0.0, (o ** 2).sum() - o[c] ** 2)))   # clamp fp noise >=0
            ax.set_title(f"{lab} — {which}:  s_c={o[c]:.3f}   off-target ‖·‖={leak:.3f}", fontsize=9)
            ax.set_ylim(-ymax, ymax)
            if row == len(data) - 1:
                ax.set_xlabel("output feature j")
        axes[row][0].set_ylabel("output value")
    fig.suptitle(f"I/O response, single component vs full model (one-hot input e_{c}; "
                 f"red = target feature {c})  —  {ctx}", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_sf_loss_spectrum(K: int, p: str, c: int, device="cpu", swaps: int = 0,
                          entries=None, tag=None, stem: str | None = None,
                          ctx: str | None = None) -> None:
    """Per-c loss spectrum: per-feature squared error for the FULL model vs the SINGLE
    ExactPD component of the one-hot input feature c. Rows = models (biregular pinv/transpose
    by default, or the passed `entries`). Bars coloured by kind: on-target attenuation (j==c,
    orange) vs off-target leakage (j!=c, purple). Single = only the on-target bar (off == 0)."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True)
    F = 100
    tok = _swap_tok(swaps) if tag is None else tag
    if stem is None:
        stem = f"biregular_sf_loss_spectrum_K{K}{tok}_p{p}_c{c}"
    if ctx is None:
        descr = f", {tag.lstrip('_')}" if tag else (", overlap-min" if swaps else "")
        ctx = f"K={K}, p={p}{descr}"
    if entries is None:
        entries = [(dec, analyse(K, swaps, p, device=device, decoder=dec))
                   for dec in ("pinv", "transpose")]
    tgt = np.zeros(F); tgt[c] = 1.0                           # relu(e_c) = e_c
    data = [(lab, (r["R"][c] - tgt) ** 2, (r["S"][c] - tgt) ** 2) for lab, r in entries]
    ymax = max(v.max() for _, ef, es in data for v in (ef, es)) * 1.1

    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.2 * len(data)), squeeze=False,
                             constrained_layout=True, sharey=True)
    for row, (lab, errF, errS) in enumerate(data):
        for col, (which, err) in enumerate((("full model", errF), ("single component c", errS))):
            ax = axes[row][col]
            colors = [ON_COLOR if j == c else OFF_COLOR for j in range(F)]
            ax.bar(range(F), err, width=1.0, color=colors)
            on = float(err[c]); off = float(err.sum() - err[c])
            ax.set_title(f"{lab} — {which}\n on={on:.2e}  off={off:.2e}  total={on+off:.2e}",
                         fontsize=9)
            ax.set_ylim(0, ymax)
            if row == len(data) - 1:
                ax.set_xlabel("output feature j")
        axes[row][0].set_ylabel("squared error")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ON_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=OFF_COLOR)]
    axes[0][0].legend(handles, ["on-target (attenuation)", "off-target (leakage)"], fontsize=7)
    fig.suptitle(f"Loss spectrum — full model vs single ExactPD component, one-hot input e_{c}  "
                 f"({ctx})", fontsize=12)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")


def plot_sf_loss_decomp(K: int, p: str, device="cpu", swaps: int = 0,
                        entries=None, tag=None, stem: str | None = None,
                        ctx: str | None = None) -> None:
    """Aggregated on/off-target squared-error profile (SSE, no /F), FULL model vs SINGLE
    ExactPD component, averaged over all 100 components. Models = biregular pinv/transpose by
    default, or the passed `entries`. Single = on-target only (off-target == 0)."""
    os.makedirs(OVERVIEW_FIG, exist_ok=True); os.makedirs(OVERVIEW_RES, exist_ok=True)
    F = 100
    I = np.eye(F)
    tok = _swap_tok(swaps) if tag is None else tag
    if stem is None:
        stem = f"biregular_sf_loss_decomp_K{K}{tok}_p{p}"
    if ctx is None:
        descr = f", {tag.lstrip('_')}" if tag else (", overlap-min" if swaps else "")
        ctx = f"K={K}, p={p}{descr}"
    if entries is None:
        entries = [(dec, analyse(K, swaps, p, device=device, decoder=dec))
                   for dec in ("pinv", "transpose")]
    rows = []   # (label, on_mse, off_mse, on_l4, off_l4)
    for dec, r in entries:
        for which, M in (("full", r["R"]), ("single", r["S"])):
            d = M - I
            diag = np.diagonal(d)
            on2 = diag ** 2
            off2 = (d ** 2).sum(1) - on2
            on4 = diag ** 4
            off4 = (np.abs(d) ** 4).sum(1) - on4
            rows.append((f"{dec}\n{which}", float(on2.mean()), float(off2.mean()),
                         float(on4.mean()), float(off4.mean())))
            print(f"  {dec:10} {which:7} MSE on={on2.mean():.3e} off={off2.mean():.3e} "
                  f"| L4 on={on4.mean():.3e} off={off4.mean():.3e}")

    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, (io, ioff, ttl) in zip(axes, [(1, 2, "forward-pass MSE"),
                                          (3, 4, "forward-pass L4 (training loss)")]):
        on = np.array([r[io] for r in rows]); off = np.array([r[ioff] for r in rows])
        ax.bar(x, on, color=ON_COLOR, label="on-target (attenuation)")
        ax.bar(x, off, bottom=on, color=OFF_COLOR, label="off-target (interference)")
        for xi, (o, f) in enumerate(zip(on, off)):
            tot = o + f
            ax.text(xi, tot, f"{tot:.1e}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("Σ squared error over outputs (SSE, no /F)")
        ax.legend(fontsize=8)
    fig.suptitle(f"On/off-target loss profile — full model vs single ExactPD component "
                 f"(avg over 100 components; single = on-target only)  —  {ctx}",
                 fontsize=11)
    out = os.path.join(OVERVIEW_FIG, f"{stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"  saved {out}")

    cols = ["decoder/which", "MSE on", "MSE off", "MSE total", "L4 on", "L4 off", "L4 total"]
    lines = [f"### Full vs single on/off-target loss profile ({ctx}, SSE scale)", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for lab, om, ofm, ol, ofl in rows:
        lines.append(f"| {lab.replace(chr(10),' ')} | {om:.3e} | {ofm:.3e} | {om+ofm:.3e} | "
                     f"{ol:.3e} | {ofl:.3e} | {ol+ofl:.3e} |")
    path = os.path.join(OVERVIEW_RES, stem)
    with open(path + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=BIREG_K)
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--decoder", default="pinv", choices=["pinv", "transpose"])
    ap.add_argument("--overview", action="store_true",
                    help="render the single-vs-full p-sweep overview (both decoders) instead")
    ap.add_argument("--spectrum", action="store_true",
                    help="render the output-spectrum comparison (4 models, one-hot input) instead")
    ap.add_argument("--spectrum-summary", action="store_true", dest="spectrum_summary",
                    help="render the output-spectrum summary bars aggregated over all components")
    ap.add_argument("--spectrum-loss", action="store_true", dest="spectrum_loss",
                    help="render the on/off-target squared-error decomposition over all components")
    ap.add_argument("--sf-loss", action="store_true", dest="sf_loss",
                    help="full-model vs single-PD-component loss spectrum (per-c) + on/off profile")
    ap.add_argument("--io-spectrum", action="store_true", dest="io_spectrum",
                    help="single vs full raw-output I/O response in the output-spectrum panel style")
    ap.add_argument("--c", type=int, default=0, help="feature index for the one-hot probe (--spectrum)")
    ap.add_argument("--swaps", type=int, default=0,
                    help="biregular code overlap-min swaps (0 = no-swap; SWAPS for overlap-min)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.overview:
        plot_single_vs_full_overview(a.K, device=a.device, swaps=a.swaps)
        return
    if a.spectrum:
        plot_output_spectrum(a.K, a.p, a.c, device=a.device, swaps=a.swaps)
        return
    if a.spectrum_summary:
        plot_output_spectrum_summary(a.K, a.p, device=a.device, swaps=a.swaps)
        return
    if a.spectrum_loss:
        plot_output_loss_decomposition(a.K, a.p, device=a.device, swaps=a.swaps)
        return
    if a.sf_loss:
        plot_sf_loss_spectrum(a.K, a.p, a.c, device=a.device, swaps=a.swaps)
        plot_sf_loss_decomp(a.K, a.p, device=a.device, swaps=a.swaps)
        return
    if a.io_spectrum:
        plot_io_spectrum_style(a.K, a.p, a.c, device=a.device, swaps=a.swaps)
        return
    # default run: single-vs-full bar/ratio/io-response figures for both code variants
    tag = "" if a.decoder == "pinv" else f"_{a.decoder}"
    res = {}
    for name, sw in VARIANTS:
        print(f"[{name}] building K={a.K} swaps={sw} decoder={a.decoder} ...")
        res[name] = analyse(a.K, sw, a.p, device=a.device, decoder=a.decoder)
        d = res[name]
        print(f"  MSE single={d['l2_single']:.3e} full={d['l2_full']:.3e} "
              f"(ratio {d['l2_single']/d['l2_full']:.3f})  |  "
              f"L4 single={d['l4_single']:.3e} full={d['l4_full']:.3e} "
              f"(ratio {d['l4_single']/d['l4_full']:.3f})  |  "
              f"interference={100*d['interference']:.1f}%")
    plot_loss_bars(res, a.K, a.p, tag=tag, stat="mean")
    plot_loss_bars(res, a.K, a.p, tag=tag, stat="max")
    plot_io_response(res, a.K, a.p, tag=tag)
    write_ratios(res, a.K, a.p, tag, a.decoder)


if __name__ == "__main__":
    main()
