"""Component-level comparison of a learned decomposition against ExactPD ground truth.

ExactPD's components are LABELED (component i = feature i's mechanism), so a learned
decomposition (APD / SPD / VPD) can be scored component-by-component:

  1. match heatmap  -- cosine similarity between each learned component's rank-1
                       feature-space signature (read u_c (+) write v_c) and each
                       ExactPD component's; greedy one-to-one matching.
  2. purity/coverage -- purity = mean matched cosine over features; coverage(theta) =
                       #features matched above theta; split = #features claimed (argmax)
                       by more than one learned component.
  3. ablation agreement -- ablate each well-matched learned component (batch-shared
                       mask, components-only forward) and check the feature whose
                       active-conditional error rises MOST is the weight-matched one.
  4. recon decomposition -- per-feature captured fraction: projection of the method's
                       summed read-in reconstruction onto ExactPD's rows of readin.

  ./.venv/bin/python compare_exactpd.py --method VPD --model EmbedMLP --demb 200 --p 0.1
  ./.venv/bin/python compare_exactpd.py --method VPD --all --d-mlp 50   # every cell with both decomps
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import decomp_fit
import results_io
from data import CleanDataset

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "comparison")
THETA = 0.7


def _signatures(pd_obj) -> np.ndarray:
    """(C, 2F) L2-normalised |read| (+) |write| feature-space signatures."""
    reads, writes, _, _, _ = pd_obj.component_signatures()
    sig = torch.cat([reads.abs(), writes.abs()], dim=1).cpu().numpy()
    n = np.linalg.norm(sig, axis=1, keepdims=True)
    return sig / np.maximum(n, 1e-12)


def greedy_match(S: np.ndarray) -> dict[int, int]:
    """One-to-one matching maximising total cosine: {feature j -> learned c}."""
    pairs = [(S[c, j], c, j) for c in range(S.shape[0]) for j in range(S.shape[1])]
    pairs.sort(reverse=True)
    used_c, used_j, match = set(), set(), {}
    for s, c, j in pairs:
        if c not in used_c and j not in used_j:
            match[j] = c
            used_c.add(c)
            used_j.add(j)
    return match


def compare_cell(method: str, run_id_m: str, device="cpu") -> dict | None:
    run_id_e = run_id_m.rsplit("_", 1)[0] + "_ExactPD"
    try:
        pd_m = decomp_fit.load_decomposition(run_id_m, device=device)
        pd_e = decomp_fit.load_decomposition(run_id_e, device=device)
    except FileNotFoundError:
        return None
    info = results_io.parse_run_id(run_id_m)
    F = pd_e.model.n_features

    # 1. similarity + matching
    sig_m, sig_e = _signatures(pd_m), _signatures(pd_e)
    S = sig_m @ sig_e.T                                  # (C_m, F) cosines
    match = greedy_match(S)
    matched_cos = np.array([S[match[j], j] if j in match else 0.0 for j in range(F)])

    # 2. scores
    purity = float(matched_cos.mean())
    coverage = int((matched_cos > THETA).sum())
    claims = np.argmax(S, axis=1)                        # each learned comp's favourite feature
    strong = S[np.arange(S.shape[0]), claims] > THETA
    uniq, counts = np.unique(claims[strong], return_counts=True)
    split = int((counts > 1).sum())

    # 3. ablation agreement over well-matched pairs
    ds = CleanDataset(n_features=F, p=float(info["p"]), device=device, seed=7)
    x, _ = ds.generate_batch(2048)
    active = (x > 0).float()
    with torch.no_grad():
        y = pd_m.model(x)
        ones = torch.ones(pd_m.C, device=device)
        base = pd_m.forward_masked(x, {n: ones for n in pd_m.param_names})
        base_err = (((base - y) ** 2) * active).sum(0) / active.sum(0).clamp_min(1)
        agree = n_checked = 0
        for j in range(F):
            if j not in match or matched_cos[j] <= THETA:
                continue
            m = ones.clone()
            m[match[j]] = 0.0
            out = pd_m.forward_masked(x, {n: m for n in pd_m.param_names})
            err = (((out - y) ** 2) * active).sum(0) / active.sum(0).clamp_min(1)
            attributed = int((err - base_err).argmax().item())
            agree += int(attributed == j)
            n_checked += 1
    agreement = agree / n_checked if n_checked else float("nan")

    # 4. recon decomposition: per-feature captured fraction of readin rows
    readin = pd_e.model.readin().detach()                # (F, N) ground-truth rows
    recon_in = pd_m.model.project_in(pd_m.reconstructed_weight("w_in")).detach()
    num = (recon_in * readin).sum(1)
    captured = (num / (readin * readin).sum(1).clamp_min(1e-12)).cpu().numpy()

    # figures
    os.makedirs(FIG_DIR, exist_ok=True)
    order = np.array([match.get(j, -1) for j in range(F)])
    S_perm = np.stack([S[order[j]] if order[j] >= 0 else np.zeros(F) for j in range(F)])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    im = axs[0].imshow(S_perm, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axs[0].set_xlabel("ExactPD feature"), axs[0].set_ylabel("matched learned component (row j = match of feature j)")
    axs[0].set_title(f"signature cosine (matched-permuted)  purity={purity:.2f} "
                     f"coverage(>{THETA})={coverage}/{F}", fontsize=9)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].bar(range(F), captured, width=1.0, color="steelblue")
    axs[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axs[1].set_xlabel("feature"), axs[1].set_ylabel("captured fraction of readin row")
    axs[1].set_title(f"per-feature weight reconstruction  (ablation agreement="
                     f"{agreement:.2f} over {n_checked} matched)", fontsize=9)
    fig.suptitle(f"{method} vs ExactPD — {run_id_m.replace('_' + method, '')}", fontsize=11)
    out = os.path.join(FIG_DIR, f"{run_id_m}_vs_ExactPD.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {"cell": run_id_m.replace(f"_{method}", ""), "purity": purity,
            "coverage": coverage, "split": split, "agreement": agreement,
            "captured_mean": float(np.mean(captured))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="VPD", choices=["APD", "OracleAPD", "SPD", "VPD", "ExactPD"])
    ap.add_argument("--model", default="EmbedMLP")
    ap.add_argument("--demb", type=int, default=200)
    ap.add_argument("--d-mlp", type=int, default=50, dest="d_mlp")
    ap.add_argument("--p", default="0.1")
    ap.add_argument("--n", type=float, default=4.0, dest="n_exp")
    ap.add_argument("--all", action="store_true", help="every cell with both decomps present")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.all:
        pat = os.path.join("results", f"n{results_io._fmt_n(a.n_exp)}", "decomps",
                           f"*dmlp{a.d_mlp}*_{a.method}.pt")
        run_ids = [os.path.basename(p)[:-3] for p in sorted(glob.glob(pat))]
    else:
        demb = None if a.model == "MLP" else a.demb
        run_ids = [results_io.make_run_id(model=a.model, d_emb=demb, d_mlp=a.d_mlp,
                                          p=a.p, n_exp=a.n_exp, method=a.method)]

    rows = []
    for rid in run_ids:
        r = compare_cell(a.method, rid, device=a.device)
        if r:
            rows.append(r)
            print(f"{r['cell']:42} purity={r['purity']:.3f} coverage={r['coverage']:>3} "
                  f"split={r['split']:>2} agreement={r['agreement']:.2f} "
                  f"captured={r['captured_mean']:.2f}")
    print(f"\n{len(rows)} cells -> {FIG_DIR}")


if __name__ == "__main__":
    main()
