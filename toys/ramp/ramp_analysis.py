"""Ramp-specific analyses for the trained models (chain + single, L4, p=0.01).

1. Transfer curves: per-feature output on one-hot probes v in [-1, 1];
   rise slope, plateau level/slope/flatness, kink position, and the exact
   homogeneity check (bias-free nets satisfy f(v e_j) = v f(e_j) for v > 0,
   so a genuine plateau is impossible on singletons — measure how the model
   compromises).
2. Bypass ablation of the chain's layer 2 (identity + lstsq-fitted linear
   map), held-out L4 ratios vs the relu-target reference (1.13x / 1.06x at
   p = 0.01, toys/chain/results/bypass_sweep.md). On-record prediction: the
   ramp makes layer 2 load-bearing.
3. Unit-role census: one-hot probes at v = 0.25 (mid-rise) vs v = 0.75
   (plateau); units whose activation saturates between the probes (growth
   ratio < 1.5 vs 3 for linear) are saturation-implementers.
4. Single-vs-chain: held-out L4 + computed counts.

  python ramp_analysis.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from census import COMPUTED_AT, perfeature_mse
from data import N_FEATURES, THETA, SparseDataset, ramp
from model import load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
P = 0.01
SEED = 42
EVAL_SEED, FIT_SEED = 777, 12345
REF_RELU = {"r_id": 1.13, "r_A": 1.06}   # relu target, p=0.01 (toys/chain)
ACT_TAU = 0.01                            # unit counts as active above this
V_RISE, V_PLAT = 0.25, 0.75              # unit-role probe values


def transfer_curves(model, n_v: int = 401):
    vs = torch.linspace(-1, 1, n_v)
    rows = []
    with torch.no_grad():
        for v in vs:
            rows.append(torch.diagonal(model(v * torch.eye(N_FEATURES))))
    return vs.numpy(), torch.stack(rows, 1).numpy()   # (F, V)


def curve_stats(vs, curves):
    """Per-feature rise slope, plateau level/slope/flatness, kink, and
    deviations from exact homogeneity / from the ideal ramp."""
    rise = (vs >= 0.05) & (vs <= 0.45)
    plat = (vs >= 0.6) & (vs <= 1.0)
    neg = vs <= 0.0
    pos = vs > 0.0
    F = curves.shape[0]
    out = {k: np.zeros(F) for k in
           ("rise_slope", "plateau_level", "plateau_slope", "plateau_span",
            "kink", "neg_max", "homog_dev")}
    v1 = curves[:, -1]                                # value at v = 1
    for j in range(F):
        c = curves[j]
        s = np.polyfit(vs[rise], c[rise], 1)[0]
        out["rise_slope"][j] = s
        out["plateau_level"][j] = c[plat].mean()
        out["plateau_slope"][j] = np.polyfit(vs[plat], c[plat], 1)[0]
        out["plateau_span"][j] = c[plat].max() - c[plat].min()
        out["neg_max"][j] = np.abs(c[neg]).max()
        # deviation from f(v e_j) = v * f(e_j) on v > 0
        out["homog_dev"][j] = np.abs(c[pos] - vs[pos] * v1[j]).max()
        # first v >= 0.1 where local slope falls below half the rise slope
        d = np.gradient(c, vs)
        cand = np.where((vs >= 0.1) & (vs <= 1.0) & (d < 0.5 * s))[0]
        out["kink"][j] = vs[cand[0]] if len(cand) else np.nan
    return out


def med_iqr(a):
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "—"
    return (f"{np.median(a):.3f} "
            f"[{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]")


def held_out_l4(fwd, batches: int = 40, batch: int = 8192):
    ds = SparseDataset(P, seed=EVAL_SEED)
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            tot += float(((fwd(x) - y).abs() ** 4).mean())
    return tot / batches


def bypass(chain):
    W1, W2, W3 = chain.w1.detach(), chain.w2.detach(), chain.w3.detach()
    xs, _ = SparseDataset(P, seed=FIT_SEED).generate_batch(131072)
    with torch.no_grad():
        h1 = torch.relu(xs @ W1)
        live = h1.sum(1) > 0
        pre2 = h1[live] @ W2
        clip = float(((pre2 < 0) & (h1[live] @ torch.abs(W2) > 1e-6))
                     .float().mean())
        A = torch.linalg.lstsq(h1[live], torch.relu(pre2)).solution
    l4_orig = held_out_l4(lambda x: torch.relu(torch.relu(x @ W1) @ W2) @ W3)
    l4_id = held_out_l4(lambda x: (torch.relu(x @ W1) @ W2) @ W3)
    l4_A = held_out_l4(lambda x: (torch.relu(x @ W1) @ A) @ W3)
    return {"clip": clip, "l4_orig": l4_orig, "l4_id": l4_id, "l4_A": l4_A,
            "r_id": l4_id / l4_orig, "r_A": l4_A / l4_orig}


def unit_roles(model):
    """Growth ratios h(0.75)/h(0.25) on one-hot probes; per-feature counts of
    saturation-implementers (ratio < 1.5) and rise-implementers (>= 2.5;
    linear scaling gives exactly 3)."""
    layers = {}
    with torch.no_grad():
        for v, tag in ((V_RISE, "lo"), (V_PLAT, "hi")):
            h1 = torch.relu(v * model.w1.detach())            # (F, d1)
            layers.setdefault("L1", {})[tag] = h1
            if hasattr(model, "w3"):
                layers.setdefault("L2", {})[tag] = \
                    torch.relu(h1 @ model.w2.detach())        # (F, d2)
    out = {}
    for name, d in layers.items():
        act = d["lo"] > ACT_TAU
        g = torch.where(act, d["hi"] / d["lo"].clamp(min=1e-9),
                        torch.nan).numpy()
        out[name] = {
            "growth": g,
            "n_active": act.sum(1).numpy(),
            "n_sat": ((d["lo"] > ACT_TAU) & (d["hi"] < 1.5 * d["lo"]))
                     .sum(1).numpy(),
            "n_rise": ((d["lo"] > ACT_TAU) & (d["hi"] >= 2.5 * d["lo"]))
                      .sum(1).numpy()}
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    models = {}
    for arch in ("chain", "single"):
        path = os.path.join(ROOT, "checkpoints",
                            f"ramp_{arch}_p{P:g}_l4_s{SEED}.pt")
        models[arch], _ = load_checkpoint(path)
    md = [f"### ramp analysis — L4, p = {P:g}, seed {SEED}, theta = {THETA}",
          ""]

    # -- 1. transfer curves ------------------------------------------------
    md += ["## Transfer curves (one-hot probes, v in [-1, 1])", "",
           "| arch | rise slope | plateau level | plateau slope | plateau "
           "span | kink found | homog. dev max | neg-side max |",
           "|---|---|---|---|---|---|---|---|"]
    tc = {}
    for arch, model in models.items():
        vs, curves = transfer_curves(model)
        st = curve_stats(vs, curves)
        tc[arch] = (vs, curves, st)
        n_kink = int(np.isfinite(st["kink"]).sum())
        md.append(f"| {arch} | {med_iqr(st['rise_slope'])} | "
                  f"{med_iqr(st['plateau_level'])} | "
                  f"{med_iqr(st['plateau_slope'])} | "
                  f"{med_iqr(st['plateau_span'])} | {n_kink}/100 | "
                  f"{st['homog_dev'].max():.2e} | "
                  f"{med_iqr(st['neg_max'])} |")
        print(f"{arch}: rise slope {med_iqr(st['rise_slope'])}, plateau slope "
              f"{med_iqr(st['plateau_slope'])}, kinks {n_kink}/100, "
              f"max homogeneity deviation {st['homog_dev'].max():.2e}")
    md += ["", "Ideal ramp: rise slope 1, plateau level 0.5, plateau slope 0, "
           "kink at v = 0.5. Bias-free nets are positively homogeneous "
           "(f(v e_j) = v f(e_j), v > 0): homog. dev max is the largest "
           "deviation of any curve from a straight line through the origin — "
           "if it is at float-noise level, plateaus are impossible on "
           "singletons and the fitted rise slope is the model's linear "
           "compromise.", ""]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True,
                             sharex=True, sharey=True)
    ideal = np.clip(np.linspace(-1, 1, 401), 0, THETA)
    for row, arch in enumerate(("chain", "single")):
        vs, curves, st = tc[arch]
        ax = axes[row, 0]
        for j in range(0, 100, 17):
            ax.plot(vs, curves[j], lw=1.2, alpha=0.8, label=f"f{j}")
        ax.plot(vs, ideal, "k--", lw=1.5, label="ideal ramp")
        ax.set_title(f"{arch}: example features", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        ax = axes[row, 1]
        med = np.median(curves, 0)
        q25, q75 = np.percentile(curves, 25, 0), np.percentile(curves, 75, 0)
        ax.plot(vs, med, color="tab:blue", lw=2, label="median")
        ax.fill_between(vs, q25, q75, color="tab:blue", alpha=0.25,
                        label="IQR")
        ax.plot(vs, ideal, "k--", lw=1.5, label="ideal ramp")
        ax.set_title(f"{arch}: population", fontsize=10)
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.axvline(THETA, color="grey", ls=":", lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("one-hot value v")
    for ax in axes[:, 0]:
        ax.set_ylabel("output at own feature")
    fig.suptitle("ramp: transfer curves on one-hot probes (dashed = ideal, "
                 "dotted = theta)", fontsize=11)
    fig.savefig(os.path.join(FIG, "transfer_curves.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # -- 2. bypass ablation (chain layer 2) --------------------------------
    b = bypass(models["chain"])
    md += ["## Layer-2 bypass ablation (chain)", "",
           "| target | clip frac | L4 original | identity ratio | fitted-A "
           "ratio |", "|---|---|---|---|---|",
           f"| ramp | {b['clip']:.3f} | {b['l4_orig']:.2e} | "
           f"{b['r_id']:.2f} | {b['r_A']:.2f} |",
           f"| relu (reference, toys/chain) | 0.103 | 1.33e-05 | "
           f"{REF_RELU['r_id']:.2f} | {REF_RELU['r_A']:.2f} |", "",
           "On-record prediction: the ramp makes layer 2 load-bearing "
           "(ratios well above the relu-target 1.13x / 1.06x).", ""]
    print(f"bypass: clip {b['clip']:.3f}, L4 orig {b['l4_orig']:.2e}, "
          f"identity x{b['r_id']:.2f}, fitted-A x{b['r_A']:.2f} "
          f"(relu ref x{REF_RELU['r_id']:.2f} / x{REF_RELU['r_A']:.2f})")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    xs_pos = np.arange(2)
    ax.bar(xs_pos - 0.18, [b["r_id"], b["r_A"]], 0.36, color="tab:red",
           label="ramp target")
    ax.bar(xs_pos + 0.18, [REF_RELU["r_id"], REF_RELU["r_A"]], 0.36,
           color="tab:orange", label="relu target (reference)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(xs_pos)
    ax.set_xticklabels(["identity bypass", "fitted-A bypass"])
    ax.set_ylabel("held-out L4 ratio vs original")
    ax.set_title(f"chain layer-2 bypass cost at p = {P:g}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(FIG, "bypass.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # -- 3. unit-role census ----------------------------------------------
    md += ["## Unit-role census (one-hot probes at v = 0.25 vs 0.75)", "",
           "Growth ratio h(0.75)/h(0.25) per active unit; linear scaling "
           "gives exactly 3. Saturation-implementer: ratio < 1.5; "
           "rise-implementer: ratio >= 2.5.", "",
           "| arch | layer | active units/feature | saturators/feature | "
           "risers/feature |", "|---|---|---|---|---|"]
    roles = {}
    for arch, model in models.items():
        roles[arch] = unit_roles(model)
        for name, r in roles[arch].items():
            md.append(f"| {arch} | {name} | {r['n_active'].mean():.1f} | "
                      f"{r['n_sat'].mean():.2f} | {r['n_rise'].mean():.2f} |")
            print(f"{arch} {name}: active {r['n_active'].mean():.1f}, "
                  f"saturators {r['n_sat'].mean():.2f}, "
                  f"risers {r['n_rise'].mean():.2f} per feature")
    md.append("")

    panels = [(a, n) for a in roles for n in roles[a]]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4),
                             constrained_layout=True, sharey=True)
    for ax, (arch, name) in zip(np.atleast_1d(axes), panels):
        g = roles[arch][name]["growth"]
        g = g[np.isfinite(g)]
        ax.hist(g, bins=np.linspace(0, 4, 41), color="tab:blue", alpha=0.8)
        ax.axvline(3.0, color="k", ls="--", lw=1, label="linear (3)")
        ax.axvline(1.5, color="tab:red", ls=":", lw=1, label="saturation (<1.5)")
        ax.set_title(f"{arch} {name}", fontsize=10)
        ax.set_xlabel("h(0.75) / h(0.25)")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("active (feature, unit) pairs")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("ramp: unit growth ratios between mid-rise and plateau probes",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "unit_roles.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # -- 4. single vs chain ------------------------------------------------
    md += ["## Single vs chain", "",
           "| arch | held-out L4 | computed /100 |", "|---|---|---|"]
    for arch, model in models.items():
        l4 = held_out_l4(model)
        n_comp = int((perfeature_mse(model, P) < COMPUTED_AT).sum())
        md.append(f"| {arch} | {l4:.2e} | {n_comp} |")
        print(f"{arch}: held-out L4 {l4:.2e}, computed {n_comp}/100")
    md.append("")

    with open(os.path.join(RES, "ramp_analysis.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"results -> {os.path.join(RES, 'ramp_analysis.md')}; "
          f"figures -> {FIG}")


if __name__ == "__main__":
    main()
