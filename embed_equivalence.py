"""Pairwise equivalence battery (de Ferreira-Heimersheim metrics): trained models vs
matched designed biregular codes at p = 0.01.

Pairs: plain MLP d50 (K_d=12) and EmbedMLP d_emb in {1000, 100, 50, 20} (K_d = 10,
13, 19, 23 -- the >0 preactivation counts from the activation sweep, NOT tau-counts).
Each trained model is bracketed by its matched designed code in BOTH swap variants
(no-swap primary, overlap-min benchmark) x BOTH decoders (transpose, pinv).

dFH battery (their repo, on effective weights W_in_eff = W_e W_in, W_out_eff =
W_out W_e^T): per-feature active-conditional MSE overlay; encoder value histogram;
K-dist both axes (tau=0.05); on/off-code means; decoder frobenius-cosines (pinv vs
transpose convention); ansatz-refit ratio on the EXTRACTED code M-hat (their ~1.13
headline); swap test; held-out L4 comparison vs the designed codes.

  ./.venv/bin/python embed_equivalence.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from biregular_ansatz import CodeAnsatz
from data import CleanDataset
from embed_exact_decomposition import load_embed_mlp
from mlp import MLP

RES_DIR = os.path.join(os.path.dirname(__file__), "results", "embedded", "equivalence_p0.01")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "embedded", "equivalence_p0.01")
P = 0.01
TAU = 0.05
SEED_EVAL = 777
K_D = {"MLP": 12, 1000: 10, 100: 13, 50: 19, 20: 23}
SWAPS_OM = 800000


def load_models():
    out = {}
    ck = torch.load("checkpoints/ncc_MLP_d_mlp50_n_features100_p0.01_n4_s42.pt",
                    map_location="cpu", weights_only=False)
    m = MLP(**ck["model_config"], device="cpu")
    m.load_state_dict(ck["state_dict"])
    out["MLP"] = (m, m.w_in.detach().double(), m.w_out.detach().double())
    for d in (1000, 100, 50, 20):
        m = load_embed_mlp(d, p="0.01")
        We = m.w_e.double()
        w_in_eff = We @ m.w_in.detach().double()          # (F, N)
        w_out_eff = m.w_out.detach().double() @ We.T      # (N, F)
        out[d] = (m, w_in_eff, w_out_eff)
    return out


def fit_ab_on_code(M: torch.Tensor, decoder: str, p: float = P, steps: int = 4000,
                   batch: int = 4096, pool: int = 131072, lr: float = 0.01,
                   seed: int = 31337):
    """(a, b) L4 fit for an ARBITRARY boolean code M (F, N) -- mirror of
    CodeAnsatz.fit_ab, needed for extracted codes M-hat."""
    F = M.shape[0]
    Mf = M.double()
    PINV = torch.linalg.pinv(Mf.T)
    ds = CleanDataset(n_features=F, p=p, device="cpu", seed=seed)
    x_pool, y_pool = ds.generate_batch(pool)
    x_pool, y_pool = x_pool.double(), y_pool.double()
    gen = torch.Generator().manual_seed(seed)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, pool, (batch,), generator=gen)
        x, y = x_pool[idx], y_pool[idx]
        w_in = a * Mf + b * (1 - Mf)
        w_out = w_in.T if decoder == "transpose" else PINV.T
        loss = ((torch.relu(x @ w_in) @ w_out - y).abs() ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return float(a), float(b)


def model_from_code(M: torch.Tensor, a: float, b: float, decoder: str) -> MLP:
    Mf = M.double()
    w_in = (a * Mf + b * (1 - Mf)).float()
    w_out = (w_in.T.clone() if decoder == "transpose"
             else torch.linalg.pinv(Mf.T).T.float())
    m = MLP(n_features=M.shape[0], d_mlp=M.shape[1], device="cpu")
    with torch.no_grad():
        m.w_in.copy_(w_in); m.w_out.copy_(w_out)
    return m


def perfeature_mse(model, p: float = P, batches: int = 25, batch: int = 8192,
                   seed: int = 9999) -> np.ndarray:
    """dFH's active-conditional per-feature MSE."""
    F = model.n_features
    ds = CleanDataset(n_features=F, p=p, device="cpu", seed=seed)
    err = torch.zeros(F); cnt = torch.zeros(F)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            active = (x > 0).float()
            err += (((model(x) - y) ** 2) * active).sum(0)
            cnt += active.sum(0)
    return (err / cnt.clamp(min=1)).numpy()


def frob_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A * B).sum() / (A.norm() * B.norm()).clamp_min(1e-12))


def swap_test(w_in_eff: torch.Tensor, w_out_eff: torch.Tensor, tau: float = TAU,
              max_pairs: int = 2000, seed: int = 5) -> float:
    """dFH Fig 5: move feature i's hidden values onto feature j's codeword positions
    (equal-K pairs), decode, count argmax == j."""
    F = w_in_eff.shape[0]
    codes_ = [torch.where(w_in_eff[j] > tau)[0] for j in range(F)]
    ks = np.array([len(c) for c in codes_])
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(F) for j in range(F)
             if i != j and ks[i] == ks[j] and ks[i] > 0]
    if not pairs:
        return float("nan")
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    hits = 0
    with torch.no_grad():
        H = torch.relu(w_in_eff)                       # hidden on one-hot inputs
        for i, j in pairs:
            h = torch.zeros(w_in_eff.shape[1], dtype=torch.float64)
            h[codes_[j]] = H[i][codes_[i]]
            y = h @ w_out_eff
            hits += int(torch.argmax(y)) == j
    return hits / len(pairs)


def battery(name, model, w_in_eff, w_out_eff, K_d, refs):
    """All metrics for one trained model against its matched designed refs."""
    r = {"name": str(name), "K_d": K_d}
    # measured K conventions
    r["K_pos"] = float((w_in_eff > 0).sum(1).float().mean())
    r["K_tau"] = float((w_in_eff > TAU).sum(1).float().mean())
    kr = (w_in_eff > TAU).sum(1).float()
    kc = (w_in_eff > TAU).sum(0).float()
    r["kdist_row"] = (float(kr.mean()), float(kr.std()))
    r["kdist_col"] = (float(kc.mean()), float(kc.std()))
    on = w_in_eff[w_in_eff > TAU]; off = w_in_eff[w_in_eff <= TAU]
    r["a_hat"], r["b_hat"] = float(on.mean()), float(off.mean())
    # decoder structure
    r["cos_pinv"] = frob_cos(w_out_eff, torch.linalg.pinv(w_in_eff.T).T)
    r["cos_transpose"] = frob_cos(w_out_eff, w_in_eff.T)
    # L4 of trained model
    r["L4_trained"] = model.reconstruction_loss(P, 4.0, batch=8192, seed=SEED_EVAL)
    # ansatz refit on the EXTRACTED code
    M_hat = (w_in_eff > TAU)
    r["Mhat_rowdeg"] = (float(M_hat.sum(1).float().mean()),
                        float(M_hat.sum(1).float().std()))
    ratios = {}
    for dec in ("transpose", "pinv"):
        a, b = fit_ab_on_code(M_hat, dec)
        l4 = model_from_code(M_hat, a, b, dec).reconstruction_loss(
            P, 4.0, batch=8192, seed=SEED_EVAL)
        ratios[dec] = l4 / r["L4_trained"]
    r["ansatz_ratio"] = ratios
    # swap test
    r["swap"] = swap_test(w_in_eff, w_out_eff)
    # designed-code L4s (both variants x decoders) from refs
    r["designed_L4"] = {k: v["L4"] for k, v in refs.items()}
    r["designed_ab"] = {k: (v["a"], v["b"]) for k, v in refs.items()}
    r["pf_mse"] = perfeature_mse(model)
    return r


def table_md(rows, note=""):
    """Markdown battery table for a list of battery() rows."""
    cols = ["model", "K_d", "K(>0)", "K(tau)", "row-deg m±s", "col-deg m±s",
            "a_hat", "b_hat", "cos pinv", "cos transp", "L4 trained",
            "L4 designed ns-T", "L4 designed om-T", "L4 designed ns-P",
            "L4 designed om-P", "ansatz ratio T", "ansatz ratio P", "swap"]
    md = ["### dFH equivalence battery — trained models vs matched designed codes "
          f"(p={P}, tau={TAU}; ns=no-swap, om=overlap-min, T=transpose, P=pinv)"
          + note, "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([
            r["name"], str(r["K_d"]), f"{r['K_pos']:.1f}", f"{r['K_tau']:.1f}",
            f"{r['kdist_row'][0]:.1f}±{r['kdist_row'][1]:.1f}",
            f"{r['kdist_col'][0]:.1f}±{r['kdist_col'][1]:.1f}",
            f"{r['a_hat']:.3f}", f"{r['b_hat']:.3f}",
            f"{r['cos_pinv']:.3f}", f"{r['cos_transpose']:.3f}",
            f"{r['L4_trained']:.2e}",
            f"{r['designed_L4']['noswap_transpose']:.2e}",
            f"{r['designed_L4']['om_transpose']:.2e}",
            f"{r['designed_L4']['noswap_pinv']:.2e}",
            f"{r['designed_L4']['om_pinv']:.2e}",
            f"{r['ansatz_ratio']['transpose']:.2f}",
            f"{r['ansatz_ratio']['pinv']:.2f}",
            f"{r['swap']:.2f}"]) + " |")
    return "\n".join(md) + "\n"


def make_battery_figures(models, rows, refs_all, fig_dir, tag="", suffix=""):
    """The five battery figures for `rows` (one per model in `models`); `tag`
    suffixes filenames, `suffix` suffixes titles (e.g. " (seed 43)")."""
    os.makedirs(fig_dir, exist_ok=True)
    rows = rows[:len(models)]

    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.6),
                             constrained_layout=True, sharey=True)
    for ax, r in zip(np.atleast_1d(axes), rows):
        ax.plot(sorted(r["pf_mse"], reverse=True), color="k", lw=2, label="trained")
        refs = refs_all[r["name"] if r["name"] == "MLP" else int(r["name"])]
        for k, style, col in (("noswap_transpose", "-", "tab:blue"),
                              ("om_transpose", "--", "tab:blue"),
                              ("noswap_pinv", "-", "tab:orange"),
                              ("om_pinv", "--", "tab:orange")):
            ax.plot(sorted(refs[k]["pf_mse"], reverse=True), style, color=col,
                    lw=1.2, label=k)
        ax.axhline(1 / 3, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{r['name']} vs designed K={r['K_d']}", fontsize=9)
        ax.set_xlabel("feature (sorted)")
    np.atleast_1d(axes)[0].set_ylabel("per-feature MSE (active-cond.)")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.suptitle("Per-feature MSE: trained vs matched designed codes (p=0.01); "
                 f"grey = output-zero (1/3){suffix}", fontsize=11)
    fig.savefig(os.path.join(fig_dir, f"perfeature_mse_pairs{tag}.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.2),
                             constrained_layout=True)
    for ax, r in zip(np.atleast_1d(axes), rows):
        key = r["name"] if r["name"] == "MLP" else int(r["name"])
        W = models[key][1].flatten().numpy()
        bins = np.arange(np.floor(W.min() / TAU) * TAU, W.max() + TAU, TAU)
        ax.hist(W, bins=bins, color="steelblue", log=True)
        ab = refs_all[key]["noswap_transpose"]
        ax.axvline(ab["a"], color="tab:red", ls="--", lw=1, label=f"a_d={ab['a']:.2f}")
        ax.axvline(ab["b"], color="tab:purple", ls="--", lw=1, label=f"b_d={ab['b']:.2f}")
        ax.set_title(f"{r['name']} W_in_eff values", fontsize=9)
        ax.legend(fontsize=6)
    fig.suptitle("Encoder value histograms vs matched designed (a, b) — transpose "
                 f"fit{suffix}", fontsize=11)
    fig.savefig(os.path.join(fig_dir, f"value_hists{tag}.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 5.6),
                             constrained_layout=True)
    axes = np.array(axes).reshape(2, -1)
    for c, r in enumerate(rows):
        key = r["name"] if r["name"] == "MLP" else int(r["name"])
        W = models[key][1]
        for rr, (deg, exact, lbl) in enumerate((
                ((W > TAU).sum(1).numpy(), r["K_d"], "per-feature degree"),
                ((W > TAU).sum(0).numpy(), 2 * r["K_d"], "per-neuron degree"))):
            ax = axes[rr][c]
            ax.hist(deg, bins=np.arange(deg.min() - 0.5, deg.max() + 1.5),
                    color="steelblue")
            ax.axvline(exact, color="tab:red", ls="--", lw=1,
                       label=f"designed = {exact}")
            ax.set_title(f"{r['name']}: {lbl}", fontsize=9)
            ax.legend(fontsize=6)
    fig.suptitle(f"Degree distributions (tau={TAU}) vs matched designed biregular "
                 f"(K_d, 2K_d){suffix}", fontsize=11)
    fig.savefig(os.path.join(fig_dir, f"kdist{tag}.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for dec, off, col in (("transpose", -0.18, "tab:blue"), ("pinv", 0.18, "tab:orange")):
        axes[0].bar(x + off, [r["ansatz_ratio"][dec] for r in rows],
                    0.36, color=col, label=dec)
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].axhline(1.13, color="grey", ls=":", lw=0.8)
    axes[0].text(len(names) - 0.4, 1.15, "dFH ~1.13", fontsize=7, color="grey")
    axes[0].set_xticks(x, names); axes[0].set_ylabel("L4 ansatz(M̂)/L4 trained")
    axes[0].set_title(f"Ansatz-refit ratio on extracted code{suffix}", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["swap"] for r in rows], 0.5, color="steelblue")
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_xticks(x, names); axes[1].set_ylabel("swap-test pass rate")
    axes[1].set_title(f"Swap test (dFH Fig 5){suffix}", fontsize=10)
    fig.savefig(os.path.join(fig_dir, f"ansatz_and_swap{tag}.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    series = [("L4_trained", None, "k"),
              ("noswap_transpose", "designed ns-T", "tab:blue"),
              ("om_transpose", "designed om-T", "tab:cyan"),
              ("noswap_pinv", "designed ns-P", "tab:orange"),
              ("om_pinv", "designed om-P", "tab:red")]
    w = 0.16
    for i, (k, lbl, col) in enumerate(series):
        vals = [r["L4_trained"] if k == "L4_trained" else r["designed_L4"][k]
                for r in rows]
        ax.bar(x + (i - 2) * w, vals, w, color=col, label=lbl or "trained")
    ax.set_yscale("log"); ax.set_xticks(x, names)
    ax.set_ylabel("held-out L4 (p=0.01)")
    ax.set_title("L4: trained vs matched designed codes (no-swap / overlap-min, "
                 f"both decoders){suffix}", fontsize=10)
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(fig_dir, f"l4_comparison{tag}.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)


def load_models_subset(d_embs):
    out = {}
    for d in d_embs:
        m = load_embed_mlp(d, p="0.01")
        We = m.w_e.double()
        out[d] = (m, We @ m.w_in.detach().double(),
                  m.w_out.detach().double() @ We.T)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-embs", type=int, nargs="+", default=None, dest="d_embs",
                    help="run only these EmbedMLP d_embs (K_d measured in-script "
                         "as round(mean #preacts>0)); default = full original set")
    ap.add_argument("--tag", default="", help="suffix for the output md")
    args = ap.parse_args()

    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    if args.d_embs:
        models = load_models_subset(args.d_embs)
        for d, (_, w_in_eff, _) in models.items():
            if d not in K_D:
                K_D[d] = round(float((w_in_eff > 0).sum(1).float().mean()))
            print(f"d_emb={d}: K_d = {K_D[d]} (measured K>0 = "
                  f"{float((w_in_eff > 0).sum(1).float().mean()):.1f})")
    else:
        models = load_models()

    # designed references per K_d x swaps x decoder
    refs_all = {}
    for key, K in K_D.items():
        refs = {}
        for swaps, sw_lbl in ((0, "noswap"), (SWAPS_OM, "om")):
            for dec in ("transpose", "pinv"):
                ans = CodeAnsatz(K, swaps=swaps, decoder=dec)
                a, b = ans.fit_ab(P)
                dm = ans.model((a, b))
                refs[f"{sw_lbl}_{dec}"] = {
                    "a": a, "b": b,
                    "L4": dm.reconstruction_loss(P, 4.0, batch=8192, seed=SEED_EVAL),
                    "pf_mse": perfeature_mse(dm)}
                print(f"designed K={K} {sw_lbl} {dec}: a={a:.4f} b={b:.4f} "
                      f"L4={refs[f'{sw_lbl}_{dec}']['L4']:.2e}")
        refs_all[key] = refs

    rows = []
    for key, (model, w_in_eff, w_out_eff) in models.items():
        print(f"battery: {key}...")
        rows.append(battery(key, model, w_in_eff, w_out_eff, K_D[key], refs_all[key]))

    if not args.d_embs:
        ans = CodeAnsatz(10, swaps=0, decoder="transpose")
        a, b = ans.fit_ab(P)
        dm = ans.model((a, b))
        rows.append(battery("designed_K10_selftest", dm, dm.w_in.detach().double(),
                            dm.w_out.detach().double(), 10, refs_all[1000]))

    # ---- report ----
    md = table_md(rows)
    out_name = f"equivalence{args.tag}.md"
    with open(os.path.join(RES_DIR, out_name), "w") as f:
        f.write(md)
    print(md)

    # figures
    make_battery_figures(models, rows, refs_all, FIG_DIR, tag=args.tag)
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
