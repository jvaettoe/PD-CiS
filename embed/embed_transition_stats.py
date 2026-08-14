"""Multi-seed statistics for the dFH-equivalence transition curves (p = 0.01).

Replicates the 9-point transition sweep (EmbedMLP d_emb in {20..100, 1000} + plain
MLP d50 anchor) across training seeds 42-50 and reports sample mean +- sample std
(ddof=1) of every battery metric vs d_emb. Each seed gets its OWN matched designed
biregular codes: K_d is measured per seed (>0 preactivation convention) and the
code matrix is drawn with code seed = training seed, then its 2-scalar ansatz is
trained (fit_ab) for both swap variants x both decoders.

Phases (shardable by --seeds; all idempotent):
  train      train any missing checkpoints (exact original recipe, 15,773 steps)
  battery    per-seed metric rows + per-seed md table + per-seed battery figures,
             everything seed-stamped in path, filename, and title
  aggregate  raw CSV + mean+-std table + transition_stats figures across all seeds

  ./.venv/bin/python embed_transition_stats.py --phase train --seeds 43 44
  ./.venv/bin/python embed_transition_stats.py --phase battery --seeds 43 44
  ./.venv/bin/python embed_transition_stats.py --phase aggregate
"""
import argparse
import glob
import json
import os

from core.paths import REPO_ROOT
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from biregular.biregular_ansatz import CodeAnsatz
from core.data import CleanDataset
from embed.embed_exact_decomposition import load_embed_mlp
from embed.embed_equivalence import (P, SEED_EVAL, SWAPS_OM, TAU, battery,
                               make_battery_figures, perfeature_mse, table_md)
from core.mlp import MLP
from embed.train_embed_demb import train as train_embed

ROOT = REPO_ROOT
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RES_STATS = os.path.join(ROOT, "results", "embedded", "equivalence_p0.01", "stats")
FIG_STATS = os.path.join(ROOT, "figures", "embedded", "equivalence_p0.01", "stats")
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50)
D_EMBS = (20, 30, 40, 50, 60, 70, 80, 100, 1000)
STEPS = 15773                       # = 10000 / (1 - 0.99^100), the original recipe
D_ORDER = [20, 30, 40, 50, 60, 70, 80, 100, 1000]


def mlp_ckpt(seed: int) -> str:
    return os.path.join(CKPT_DIR, f"ncc_MLP_d_mlp50_n_features100_p{P:g}_n4_s{seed}.pt")


def embed_ckpt(d_emb: int, seed: int) -> str:
    return os.path.join(CKPT_DIR, f"ncc_EmbedMLP_d_emb{d_emb}_d_mlp50"
                                  f"_n_features100_p{P:g}_n4_s{seed}.pt")


def train_mlp(seed: int) -> None:
    """Plain-MLP twin of train_embed_demb.train (ksweep recipe, seed param)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MLP(n_features=100, d_mlp=50, device="cpu")
    dataset = CleanDataset(n_features=100, p=P, device="cpu", seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    losses = []
    for _ in tqdm(range(STEPS), desc=f"MLP s{seed}"):
        x, y = dataset.generate_batch(2048)
        opt.zero_grad()
        loss = (torch.abs(model(x) - y) ** 4).mean()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    torch.save({"format_version": 2, "model_class": "MLP",
                "model_config": model.config(), "state_dict": model.state_dict(),
                "dataset_class": "CleanDataset",
                "dataset_config": {"n_features": 100, "p": P, "seed": seed,
                                   "exactly_one_active_feature": False},
                "losses": losses, "seed": seed, "n_steps": STEPS, "n_exp": 4.0,
                "batch_size": 2048, "cosine_scheduler": True}, mlp_ckpt(seed))
    print(f"MLP s{seed}: final loss {losses[-1]:.3e}; saved {mlp_ckpt(seed)}")


def phase_train(seeds, d_embs, skip_mlp):
    for seed in seeds:
        if not skip_mlp:
            if os.path.exists(mlp_ckpt(seed)):
                print(f"skip MLP s{seed} (exists)")
            else:
                train_mlp(seed)
        for d in d_embs:
            if os.path.exists(embed_ckpt(d, seed)):
                print(f"skip EmbedMLP d{d} s{seed} (exists)")
            else:
                train_embed(d, P, STEPS, seed)


def load_seed_models(seed, d_embs, skip_mlp):
    """{key: (model, w_in_eff, w_out_eff)} for one seed; keys 'MLP' + ints."""
    out = {}
    if not skip_mlp:
        ck = torch.load(mlp_ckpt(seed), map_location="cpu", weights_only=False)
        m = MLP(**ck["model_config"], device="cpu")
        m.load_state_dict(ck["state_dict"])
        out["MLP"] = (m, m.w_in.detach().double(), m.w_out.detach().double())
    for d in d_embs:
        m = load_embed_mlp(d, p=f"{P:g}", seed=seed)
        We = m.w_e.double()
        out[d] = (m, We @ m.w_in.detach().double(),
                  m.w_out.detach().double() @ We.T)
    return out


def designed_refs(K: int, seed: int, cache: dict) -> dict:
    """Per-seed matched designed codes: code drawn with code seed = training seed,
    (a, b) TRAINED via fit_ab; cached by (seed, K)."""
    if (seed, K) in cache:
        return cache[(seed, K)]
    refs = {}
    for swaps, sw_lbl in ((0, "noswap"), (SWAPS_OM, "om")):
        for dec in ("transpose", "pinv"):
            ans = CodeAnsatz(K, seed=seed, swaps=swaps, decoder=dec)
            a, b = ans.fit_ab(P)
            dm = ans.model((a, b))
            refs[f"{sw_lbl}_{dec}"] = {
                "a": a, "b": b,
                "L4": dm.reconstruction_loss(P, 4.0, batch=8192, seed=SEED_EVAL),
                "pf_mse": perfeature_mse(dm)}
            print(f"  designed s{seed} K={K} {sw_lbl} {dec}: a={a:.4f} b={b:.4f} "
                  f"L4={refs[f'{sw_lbl}_{dec}']['L4']:.2e}")
    cache[(seed, K)] = refs
    return refs


def row_to_json(r: dict) -> dict:
    j = dict(r)
    j["pf_mse"] = [float(v) for v in r["pf_mse"]]
    return j


def phase_battery(seeds, d_embs, skip_mlp):
    for seed in seeds:
        res_dir = os.path.join(RES_STATS, f"seed_s{seed}")
        fig_dir = os.path.join(FIG_STATS, f"seed_s{seed}")
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        models = load_seed_models(seed, d_embs, skip_mlp)
        cache = {}
        refs_all, rows = {}, []
        for key, (model, w_in_eff, w_out_eff) in models.items():
            # half-up, not banker's round: K(>0)=18.5 must match the established
            # K_d=19 at d50 (the sweep-derived values 10/13/19/23 are all half-up)
            K_d = int(np.floor(float((w_in_eff > 0).sum(1).float().mean()) + 0.5))
            print(f"battery s{seed} {key}: K_d = {K_d} (K>0 = "
                  f"{float((w_in_eff > 0).sum(1).float().mean()):.1f})")
            refs_all[key] = designed_refs(K_d, seed, cache)
            rows.append(battery(key, model, w_in_eff, w_out_eff, K_d,
                                refs_all[key]))
        with open(os.path.join(res_dir, f"rows_s{seed}.json"), "w") as f:
            json.dump({"seed": seed, "rows": [row_to_json(r) for r in rows]}, f)
        md = table_md(rows, note=f" — seed {seed}")
        with open(os.path.join(res_dir, f"equivalence_s{seed}.md"), "w") as f:
            f.write(md)
        print(md)
        make_battery_figures(models, rows, refs_all, fig_dir,
                             tag=f"_s{seed}", suffix=f" (seed {seed})")
        print(f"seed {seed}: results -> {res_dir}, figures -> {fig_dir}")


# ---------------------------------------------------------------- aggregate ----

CSV_COLS = ["model", "seed", "K_d", "K_pos", "K_tau", "rowdeg_m", "rowdeg_s",
            "coldeg_m", "coldeg_s", "a_hat", "b_hat", "cos_pinv", "cos_transpose",
            "L4_trained", "L4_designed_ns_T", "L4_designed_om_T",
            "L4_designed_ns_P", "L4_designed_om_P", "ansatz_ratio_T",
            "ansatz_ratio_P", "swap", "l4_ratio_ns_P"]


def flat_row(r: dict, seed: int) -> dict:
    return {"model": r["name"], "seed": seed, "K_d": r["K_d"],
            "K_pos": r["K_pos"], "K_tau": r["K_tau"],
            "rowdeg_m": r["kdist_row"][0], "rowdeg_s": r["kdist_row"][1],
            "coldeg_m": r["kdist_col"][0], "coldeg_s": r["kdist_col"][1],
            "a_hat": r["a_hat"], "b_hat": r["b_hat"],
            "cos_pinv": r["cos_pinv"], "cos_transpose": r["cos_transpose"],
            "L4_trained": r["L4_trained"],
            "L4_designed_ns_T": r["designed_L4"]["noswap_transpose"],
            "L4_designed_om_T": r["designed_L4"]["om_transpose"],
            "L4_designed_ns_P": r["designed_L4"]["noswap_pinv"],
            "L4_designed_om_P": r["designed_L4"]["om_pinv"],
            "ansatz_ratio_T": r["ansatz_ratio"]["transpose"],
            "ansatz_ratio_P": r["ansatz_ratio"]["pinv"],
            "swap": r["swap"],
            "l4_ratio_ns_P": r["L4_trained"] / r["designed_L4"]["noswap_pinv"]}


def collect(metric, flats, key):
    return np.array([f[metric] for f in flats if f["model"] == str(key)])


def band(ax, xs, means, stds, color, label, marker="o"):
    means, stds = np.asarray(means), np.asarray(stds)
    ax.plot(xs, means, marker + "-", color=color, lw=2, label=label)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18)


def strip(ax, mean, std, color, label):
    ax.axhspan(mean - std, mean + std, color=color, alpha=0.12)
    ax.axhline(mean, color=color, ls="--", lw=1, label=label)


def phase_aggregate():
    files = sorted(glob.glob(os.path.join(RES_STATS, "seed_s*", "rows_s*.json")))
    if not files:
        raise SystemExit("no rows_s*.json found — run --phase battery first")
    flats, seeds = [], []
    for path in files:
        with open(path) as f:
            payload = json.load(f)
        seeds.append(payload["seed"])
        flats.extend(flat_row(r, payload["seed"]) for r in payload["rows"])
    seeds = sorted(seeds)
    note = f"n={len(seeds)} seeds (s={seeds[0]}-{seeds[-1]})"
    print(f"aggregating {len(flats)} rows from {note}")

    with open(os.path.join(RES_STATS, "transition_seeds_raw.csv"), "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for fl in sorted(flats, key=lambda r: (str(r["model"]), r["seed"])):
            f.write(",".join(str(fl[c]) for c in CSV_COLS) + "\n")

    # stats per model key across seeds
    keys = ["MLP"] + [d for d in D_ORDER
                      if any(f["model"] == str(d) for f in flats)]
    stat_metrics = ["K_d", "a_hat", "b_hat", "cos_pinv", "cos_transpose",
                    "L4_trained", "L4_designed_ns_P", "L4_designed_om_P",
                    "ansatz_ratio_P", "l4_ratio_ns_P", "rowdeg_s", "swap"]
    stats = {}
    for key in keys:
        if not any(f["model"] == str(key) for f in flats):
            continue
        s = {}
        for m in stat_metrics:
            v = collect(m, flats, key)
            s[m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    len(v))
        kd = collect("K_d", flats, key)
        s["K_d_range"] = (int(kd.min()), int(kd.max()))
        stats[key] = s

    md = [f"### Multi-seed transition statistics — mean ± sample std, {note} "
          f"(p={P}, tau={TAU}; designed codes per-seed at matched K_d)", "",
          "| model | n | K_d mean [range] | a_hat | b_hat | cos pinv | "
          "L4 trained | L4 designed ns-P | ansatz ratio P | "
          "L4 trained/designed ns-P | row-deg σ | swap |",
          "|" + "---|" * 12]
    for key in keys:
        s = stats[key]
        md.append("| " + " | ".join([
            str(key), str(s["cos_pinv"][2]),
            f"{s['K_d'][0]:.1f} [{s['K_d_range'][0]}, {s['K_d_range'][1]}]",
            f"{s['a_hat'][0]:.3f}±{s['a_hat'][1]:.3f}",
            f"{s['b_hat'][0]:.3f}±{s['b_hat'][1]:.3f}",
            f"{s['cos_pinv'][0]:.3f}±{s['cos_pinv'][1]:.3f}",
            f"{s['L4_trained'][0]:.2e}±{s['L4_trained'][1]:.1e}",
            f"{s['L4_designed_ns_P'][0]:.2e}±{s['L4_designed_ns_P'][1]:.1e}",
            f"{s['ansatz_ratio_P'][0]:.2f}±{s['ansatz_ratio_P'][1]:.2f}",
            f"{s['l4_ratio_ns_P'][0]:.2f}±{s['l4_ratio_ns_P'][1]:.2f}",
            f"{s['rowdeg_s'][0]:.1f}±{s['rowdeg_s'][1]:.1f}",
            f"{s['swap'][0]:.2f}±{s['swap'][1]:.2f}"]) + " |")
    with open(os.path.join(RES_STATS, "transition_stats.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- transition_stats.png: 4 panels, mean +- std bands over d_emb ----
    ds = [d for d in D_ORDER if d in stats]
    have_mlp = "MLP" in stats

    def m_(metric):
        return [stats[d][metric][0] for d in ds]

    def s_(metric):
        return [stats[d][metric][1] for d in ds]

    def scatter(ax, metric, color):
        for d in ds:
            v = collect(metric, flats, d)
            ax.plot([d] * len(v), v, ".", color=color, ms=3, alpha=0.35)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.axvline(50, color="grey", ls=":", lw=0.8)

    ax = axes[0][0]
    band(ax, ds, m_("cos_pinv"), s_("cos_pinv"), "tab:blue",
         "cos(W_out_eff, pinv)")
    scatter(ax, "cos_pinv", "tab:blue")
    if have_mlp:
        strip(ax, stats["MLP"]["cos_pinv"][0], stats["MLP"]["cos_pinv"][1],
              "grey", "MLP anchor")
    ax.set_title("decoder ≈ scaled pseudoinverse", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    band(ax, ds, m_("a_hat"), s_("a_hat"), "tab:red", "â (on-code mean)")
    band(ax, ds, m_("b_hat"), s_("b_hat"), "tab:purple", "b̂ (off-code mean)",
         marker="s")
    scatter(ax, "a_hat", "tab:red")
    scatter(ax, "b_hat", "tab:purple")
    if have_mlp:
        strip(ax, stats["MLP"]["a_hat"][0], stats["MLP"]["a_hat"][1],
              "grey", "MLP â")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("effective code values", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    band(ax, ds, m_("ansatz_ratio_P"), s_("ansatz_ratio_P"), "tab:orange",
         "L4 ansatz(M̂)/L4 trained (pinv)")
    band(ax, ds, m_("l4_ratio_ns_P"), s_("l4_ratio_ns_P"), "tab:green",
         "L4 trained / L4 designed (ns-pinv)", marker="s")
    scatter(ax, "ansatz_ratio_P", "tab:orange")
    scatter(ax, "l4_ratio_ns_P", "tab:green")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_title("loss ratios (1 = parity)", fontsize=10)
    ax.set_xlabel("d_emb (log)")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    band(ax, ds, m_("rowdeg_s"), s_("rowdeg_s"), "tab:brown", "row-degree σ")
    scatter(ax, "rowdeg_s", "tab:brown")
    ax.set_xlabel("d_emb (log)")
    ax2 = ax.twinx()
    band(ax2, ds, m_("swap"), s_("swap"), "steelblue", "swap rate", marker="s")
    ax2.set_ylim(0.5, 1.02)
    ax.set_title("biregularity spread + swap robustness", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("dFH-equivalence transition, multi-seed mean ± std — " + note +
                 " (p=0.01; dotted line = d_emb = N = 50)", fontsize=12)
    out = os.path.join(FIG_STATS, "transition_stats.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- transition_stats_L4.png: absolute L4 bands ----
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.axvline(50, color="grey", ls=":", lw=0.8)
    for metric, color, label in (
            ("L4_trained", "k", "trained"),
            ("L4_designed_ns_P", "tab:orange", "designed no-swap pinv"),
            ("L4_designed_om_P", "tab:red", "designed overlap-min pinv")):
        band(ax, ds, m_(metric), s_(metric), color, label)
        scatter(ax, metric, color)
    if have_mlp:
        strip(ax, stats["MLP"]["L4_trained"][0], stats["MLP"]["L4_trained"][1],
              "grey", "MLP trained")
    ax.set_xlabel("d_emb (log)")
    ax.set_ylabel("held-out L4 (p=0.01, log)")
    ax.set_title("Held-out L4, multi-seed mean ± std — trained vs per-seed "
                 "matched designed codes — " + note, fontsize=11)
    ax.legend(fontsize=8)
    out = os.path.join(FIG_STATS, "transition_stats_L4.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("train", "battery", "aggregate"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--d-embs", type=int, nargs="+", default=list(D_EMBS),
                    dest="d_embs")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads (shards run 4-way on 8 logical cores)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(RES_STATS, exist_ok=True)
    os.makedirs(FIG_STATS, exist_ok=True)
    if args.phase == "train":
        phase_train(args.seeds, args.d_embs, args.skip_mlp)
    elif args.phase == "battery":
        phase_battery(args.seeds, args.d_embs, args.skip_mlp)
    else:
        phase_aggregate()


if __name__ == "__main__":
    main()
