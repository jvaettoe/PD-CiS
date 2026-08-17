"""Designed embedded biregular code: build + ideal per-matrix PD (vs VPD later).

Sandwich model (all four matrices, to be decomposed per-matrix by VPD):

    y = relu(((x @ W_e) @ W_in)) @ W_out @ W_e.T

with C = a*M + b*(1-M) the designed transpose-decoder code (CodeAnsatz K=5,
seed 0, no swaps, fit_ab(p=0.001)), W_e the EmbedMLP fixed random unit-norm
rows (model seed 42, EmbedMLP init convention), and

    W_in  = pinv(W_e) @ C        (d_emb, N)
    W_out = W_in.T               (N, d_emb)

so W_e @ W_in = G @ C with G = W_e @ pinv(W_e): exactly C at d_emb=200 >= F
(G = I, forward == the plain designed-code MLP) and G-softened at d_emb=50.

Ideal per-matrix PD (the hand-derived decomposition VPD is tested against):
    W_emb : 100 row pieces e_j (x) W_e[j]            (tiling exact)
    W_in  : all min(d_emb, N)=50 SVD triplets s_k u_k v_k.T  (reconstruction exact)
    W_out = W_in.T and unembed = W_emb.T             (tied transposes, not duplicated)
plus the loading matrix L_jk = <W_e[j], u_k> (how each embedded feature reads
the singular frame) and the DC/bulk check on C: biregularity makes the uniform
feature/neuron directions an EXACT singular pair of C (verified numerically).

  ./.venv/bin/python -m embed.embed_ideal_vs_vpd --build
  ./.venv/bin/python -m embed.embed_ideal_vs_vpd --compare <run_dir> --d-emb 200 --stage 10k
"""
import argparse
import csv
import glob
import json
import os
import re

from core.paths import CHECKPOINTS, FIGURES, REPO_ROOT, RESULTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from biregular.biregular_ansatz import CodeAnsatz
from core.mlp import EmbedMLP

F, N, K = 100, 50, 5      # feature count, neuron count, codeword length
P = 0.001                 # the memo's sparsity (fit_ab target)
D_EMBS = (200, 50)        # exact regime (G=I) and G-softened regime
WE_SEED = 42              # EmbedMLP model seed (checkpoint convention)

CKPT_DIR = os.path.join(CHECKPOINTS, "embedded_ideal")
RES_DIR = os.path.join(RESULTS, "embedded", "ideal_vs_vpd")
FIG_DIR = os.path.join(FIGURES, "embedded", "ideal_vs_vpd")
EXPORT_DIR = os.path.join(REPO_ROOT, "exports", "goodfire")


def sparse_batch(n: int = 256, seed: int = 0) -> torch.Tensor:
    """Random sparse batch in feature space (export_goodfire convention)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, F, generator=gen) * 2 - 1
    return x * (torch.rand(n, F, generator=gen) < 0.05).float()


def designed_code():
    """The designed code and its fitted scalars: C = a*M + b*(1-M), float64."""
    code = CodeAnsatz(K, swaps=0, decoder="transpose")
    a, b = code.fit_ab(P)
    Mf = code.M.astype(np.float64)
    C = a * Mf + b * (1.0 - Mf)                               # (F, N)
    return code, (a, b), C


def build_model(d_emb: int, C: np.ndarray):
    """EmbedMLP with the designed sandwich: W_e from the model-seed-42 init
    convention (fixed unit-norm rows), W_in = pinv(W_e) @ C, W_out = W_in.T."""
    torch.manual_seed(WE_SEED)
    model = EmbedMLP(n_features=F, d_emb=d_emb, d_mlp=N, device="cpu")
    We = model.w_e.double().numpy()                           # (F, d_emb)
    We_pinv = np.linalg.pinv(We)                              # (d_emb, F)
    Win = We_pinv @ C                                         # (d_emb, N)
    G = We @ We_pinv                                          # (F, F) biorthogonality
    with torch.no_grad():
        model.w_in.copy_(torch.from_numpy(Win).float())
        model.w_out.copy_(torch.from_numpy(Win.T.copy()).float())
    return model, We, Win, G


def verify_build(d_emb: int, model: EmbedMLP, code: CodeAnsatz, ab, C, G) -> dict:
    """Construction asserts on the FLOAT32 model (the object that ships):
    effective read-in == C (d_emb >= F) or == G @ C (d_emb < F); forward of the
    d_emb=200 sandwich == the plain designed-code MLP."""
    We32 = model.w_e.detach()
    eff = (We32 @ model.w_in.detach()).double().numpy()       # (F, N)
    dev_gc = float(np.abs(eff - G @ C).max())
    assert dev_gc < 1e-5, f"d_emb={d_emb}: W_e@W_in vs G@C dev {dev_gc}"
    out = {"dev_eff_vs_GC": dev_gc}
    if d_emb >= F:
        dev_c = float(np.abs(eff - C).max())
        dev_g = float(np.abs(G - np.eye(F)).max())
        assert dev_c < 1e-5, f"d_emb={d_emb}: W_e@W_in vs C dev {dev_c}"
        assert dev_g < 1e-5, f"d_emb={d_emb}: G vs I dev {dev_g}"
        plain = code.model(ab)                                # MLP with w_in=C, w_out=C.T
        x = sparse_batch()
        with torch.no_grad():
            dev_fwd = float((model(x) - plain(x)).abs().max())
        assert dev_fwd < 1e-5, f"d_emb={d_emb}: forward vs plain code dev {dev_fwd}"
        out.update({"dev_eff_vs_C": dev_c, "dev_G_vs_I": dev_g,
                    "dev_forward_vs_plain": dev_fwd})
        print(f"  [assert] W_e@W_in == C   (dev {dev_c:.2e}); G == I (dev {dev_g:.2e})")
        print(f"  [assert] forward == plain designed code (max |dz| {dev_fwd:.2e})")
    else:
        dev_c = float(np.abs(eff - C).max())                  # softening, NOT asserted
        out.update({"dev_eff_vs_C_unasserted": dev_c,
                    "G_diag_mean": float(np.diag(G).mean())})
        print(f"  [assert] W_e@W_in == G@C (dev {dev_gc:.2e}); softening |eff - C| "
              f"max {dev_c:.3f}, mean G_jj = {np.diag(G).mean():.3f} (= d_emb/F)")
    return out


def dc_bulk_check(C: np.ndarray, ab) -> dict:
    """Is the uniform feature-space direction an exact singular direction of C?
    Biregular prediction: yes — C @ 1_N = (K a + (N-K) b) 1_F and
    C.T @ 1_F = (FK/N a + (F - FK/N) b) 1_N, so (1_F/sqrt(F), 1_N/sqrt(N)) is an
    exact singular pair with sigma_dc = sqrt(rowsum * colsum) (= both closed forms,
    since F*rowsum = N*colsum = total sum of C)."""
    a, b = ab
    u = np.ones(F) / np.sqrt(F)
    v = np.ones(N) / np.sqrt(N)
    Cv, Ctu = C @ v, C.T @ u
    resid_left = float(np.linalg.norm(Cv - (u @ Cv) * u))     # off-uniform part
    resid_right = float(np.linalg.norm(Ctu - (v @ Ctu) * v))
    sigma_dc = float(np.linalg.norm(Cv))
    rowsum, colsum = K * a + (N - K) * b, (F * K / N) * a + (F - F * K / N) * b
    sigma_pred = float(np.sqrt(abs(rowsum * colsum)))
    Uc, Sc, _ = np.linalg.svd(C, full_matrices=False)
    k_dc = int(np.argmax(np.abs(Uc.T @ u)))                   # index of the DC direction
    align = float(np.abs(Uc[:, k_dc] @ u))
    return {"sigma_dc": sigma_dc, "sigma_dc_closed_form": sigma_pred,
            "resid_left": resid_left, "resid_right": resid_right,
            "dc_rank": k_dc + 1, "dc_alignment": align,
            "sigma_top": float(Sc[0]), "sigma_second": float(Sc[1]),
            "sigma_min": float(Sc[-1]), "C_singvals": Sc.tolist()}


def ideal_pd(d_emb: int, code: CodeAnsatz, ab, C, model, We, Win, G,
             build_devs: dict) -> dict:
    """Construct + store the ideal per-matrix PD. W_emb = row pieces (tiling
    assert), W_in = full SVD triplets (reconstruction assert); W_out / unembed are
    tied transposes, noted not duplicated. Also L_jk = <W_e[j], u_k> and DC/bulk."""
    a, b = ab
    # W_emb: 100 row pieces e_j (x) W_e[j] — tile the matrix exactly
    tiling = np.zeros_like(We)
    for j in range(F):
        tiling[j] += We[j]                                    # e_j (x) W_e[j] has row j only
    dev_tile = float(np.abs(tiling - We).max())
    assert dev_tile == 0.0, f"d_emb={d_emb}: row-piece tiling dev {dev_tile}"

    # W_in: all min(d_emb, N) SVD triplets — reconstruct exactly
    U, S, Vh = np.linalg.svd(Win, full_matrices=False)        # (d_emb,50), (50,), (50,50)
    dev_svd = float(np.abs((U * S) @ Vh - Win).max())
    assert dev_svd < 1e-9, f"d_emb={d_emb}: SVD reconstruction dev {dev_svd}"
    print(f"  [assert] W_emb tiling exact (dev {dev_tile:.1e}); W_in SVD "
          f"reconstruction (dev {dev_svd:.2e}, {len(S)} triplets)")

    L = We @ U                                                # (F, 50) loadings
    dc = dc_bulk_check(C, ab)
    print(f"  [dc/bulk] uniform pair exact: resid L/R {dc['resid_left']:.2e}/"
          f"{dc['resid_right']:.2e}; sigma_dc = {dc['sigma_dc']:.4f} "
          f"(closed form {dc['sigma_dc_closed_form']:.4f}) at rank "
          f"{dc['dc_rank']}/{N} — top sigma = {dc['sigma_top']:.4f}")

    cfg = {"n_features": F, "d_mlp": N, "d_emb": d_emb, "K": K, "p": P,
           "code": code.name, "we_seed": WE_SEED, "a": a, "b": b}
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"ideal_d{d_emb}.pt")
    torch.save({"format_version": 1, "config": cfg, "M": code.M,
                "C": torch.from_numpy(C), "W_e": torch.from_numpy(We),
                "W_in": torch.from_numpy(Win), "W_out": torch.from_numpy(Win.T.copy()),
                "G": torch.from_numpy(G), "U": torch.from_numpy(U),
                "S": torch.from_numpy(S), "V": torch.from_numpy(Vh.T.copy()),
                "L": torch.from_numpy(L), "dc_bulk": dc,
                "state_dict": model.state_dict()}, ckpt_path)

    payload = {
        "config": cfg,
        "build_asserts": build_devs,
        "ideal_pd": {
            "w_emb": {"type": "row_pieces", "n_pieces": F, "tiling_dev": dev_tile},
            "w_in": {"type": "svd_triplets", "n_pieces": len(S),
                     "reconstruction_dev": dev_svd, "singvals": S.tolist()},
            "w_out": "tied: W_in triplets transposed (not duplicated)",
            "unembed": "tied: W_emb row pieces transposed (not duplicated)"},
        "loading_matrix": L.tolist(),                         # L_jk = <W_e[j], u_k>
        "dc_bulk": dc,
    }
    os.makedirs(RES_DIR, exist_ok=True)
    json_path = os.path.join(RES_DIR, f"ideal_pd_d{d_emb}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  ideal PD -> {ckpt_path}")
    print(f"           -> {json_path}")

    plot_spectrum_dcbulk(d_emb, S, dc)
    plot_loading_matrix(d_emb, L)
    return payload


def plot_spectrum_dcbulk(d_emb: int, S_win: np.ndarray, dc: dict) -> None:
    Sc = np.asarray(dc["C_singvals"])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    idx = np.arange(1, len(Sc) + 1)
    ax.plot(idx, Sc, "o-", ms=4, lw=1, color="steelblue",
            label="C = a·M + b·(1−M)  (feature space)")
    ax.plot(np.arange(1, len(S_win) + 1), S_win, "s--", ms=4, lw=1,
            color="darkorange", label=f"W_in = W_e⁺·C  (d_emb={d_emb})")
    k = dc["dc_rank"]
    ax.plot([k], [Sc[k - 1]], "o", ms=11, mfc="none", mec="tab:red", mew=1.8)
    ax.annotate(f"DC (uniform pair): σ = {dc['sigma_dc']:.3f}, rank {k}/{len(Sc)}",
                xy=(k, Sc[k - 1]), xytext=(k + 3, Sc[k - 1] + 0.25 * Sc[0]),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("singular index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"DC/bulk singular spectrum, designed C vs embedded W_in "
                 f"(d_emb={d_emb})", fontsize=10)
    d = os.path.join(FIG_DIR, f"d{d_emb}")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "spectrum_dcbulk.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_loading_matrix(d_emb: int, L: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 8), constrained_layout=True)
    v = float(np.abs(L).max())
    im = ax.imshow(L, cmap="RdBu", vmin=-v, vmax=v, aspect="auto",
                   interpolation="nearest")
    ax.set_xlabel("W_in left-singular direction k (u_k)")
    ax.set_ylabel("embedded feature j (W_e row)")
    ax.set_title(f"loading matrix  L_jk = ⟨W_e[j], u_k⟩  (d_emb={d_emb})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    d = os.path.join(FIG_DIR, f"d{d_emb}")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "loading_matrix.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def wrapper_forward(sd: dict, x: torch.Tensor) -> torch.Tensor:
    """Reimplementation of the 4-Linear export (nn.Linear computes x @ W.T)."""
    e = x @ sd["lin_emb.weight"].T
    h = torch.relu(e @ sd["lin_in.weight"].T)
    m = h @ sd["lin_out.weight"].T
    return m @ sd["lin_unemb.weight"].T


def export_goodfire(d_emb: int, model: EmbedMLP, ab) -> None:
    """4-Linear state_dict for param-decomp (weights = transposes of the math
    matrices, nn.Linear stores (out, in)) + sidecar YAML; forward-equivalence
    assert of the wrapper vs the built model."""
    a, b = ab
    We32 = model.w_e.detach()
    sd = {"lin_emb.weight": We32.T.contiguous(),              # (d_emb, F): e = x @ W_e
          "lin_in.weight": model.w_in.detach().T.contiguous(),    # (N, d_emb)
          "lin_out.weight": model.w_out.detach().T.contiguous(),  # (d_emb, N)
          "lin_unemb.weight": We32.contiguous()}              # (F, d_emb): y = m @ W_e.T
    x = sparse_batch(seed=1)
    with torch.no_grad():
        dev = float((model(x) - wrapper_forward(sd, x)).abs().max())
    assert dev < 1e-5, f"embedded_d{d_emb}: export wrapper forward mismatch {dev}"
    d = os.path.join(EXPORT_DIR, f"embedded_d{d_emb}")
    os.makedirs(d, exist_ok=True)
    torch.save(sd, os.path.join(d, "biregular.pth"))
    with open(os.path.join(d, "embedded_train_config.yaml"), "w") as f:
        f.write("embed_biregular_model_config:\n"
                f"  n_features: {F}\n"
                f"  d_emb: {d_emb}\n"
                f"  d_mlp: {N}\n"
                f"provenance: \"designed embedded biregular: CodeAnsatz K={K} seed0 "
                f"noswap transpose fit_ab(p={P}) a={a:.4f} b={b:.4f}; W_e = EmbedMLP "
                f"seed {WE_SEED} unit-norm rows; W_in = pinv(W_e)@C, W_out = W_in.T\"\n")
    print(f"  [assert] export wrapper forward == model (max |dz| {dev:.2e}) -> {d}")


def build(d_embs) -> None:
    code, ab, C = designed_code()
    print(f"designed code {code.name}: fit_ab(p={P}) -> a = {ab[0]:.4f}, "
          f"b = {ab[1]:.4f}")
    for d_emb in d_embs:
        regime = "exact (G=I)" if d_emb >= F else "G-softened"
        print(f"\nd_emb = {d_emb}  ({regime})")
        model, We, Win, G = build_model(d_emb, C)
        devs = verify_build(d_emb, model, code, ab, C, G)
        ideal_pd(d_emb, code, ab, C, model, We, Win, G, devs)
        export_goodfire(d_emb, model, ab)


# ---------------------------------------------------------------------------
# Part 3: structural comparison of a VPD run against the stored ideal PD
# ---------------------------------------------------------------------------

MODULES = ("lin_emb", "lin_in", "lin_out", "lin_unemb")
STAGES = ("10k", "warm")
ALIVE_FRAC = 0.10             # alive := mass > 10% of the per-matrix max mass
THRESHOLDS = {                # verdict thresholds, stored verbatim in every JSON
    "svd_aligned": "Hungarian median |cos| vs triplets > 0.8 (alive-restricted)",
    "feature_frame_aligned": "best-|cos| median vs W_e rows > 0.8 (alive) "
                             "with distinct-feature coverage >= 60",
    "collapsed": "alive < 25",
    "else": "mixed",
    "alive": "component mass ||V_c||*||U_c|| > 0.1 * per-matrix max mass",
}


def _unit_cols(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=0, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def _stats(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)),
            "mean": float(np.mean(v)), "min": float(v.min()), "max": float(v.max()),
            "frac_gt_0.8": float((v > 0.8).mean()),
            "frac_gt_0.5": float((v > 0.5).mean())}


def _coverage(argmax_idx, mask=None) -> int:
    idx = np.asarray(argmax_idx)
    if mask is not None:
        idx = idx[np.asarray(mask)]
    return int(len(np.unique(idx)))


def _hungarian(score: np.ndarray):
    """Max-|cos| assignment on a (C, K) score matrix -> (rows, cols).
    scipy if available (verified in this venv), greedy fallback otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(score, maximize=True)
    except ImportError:
        rows, cols = [], []
        used_r, used_c = set(), set()
        for flat in np.argsort(score, axis=None)[::-1]:
            r, c = divmod(int(flat), score.shape[1])
            if r in used_r or c in used_c:
                continue
            rows.append(r); cols.append(c)
            used_r.add(r); used_c.add(c)
            if len(cols) == min(score.shape):
                break
        return np.asarray(rows), np.asarray(cols)


def _match_block(score: np.ndarray, alive: np.ndarray) -> dict:
    """Hungarian match on a (C, K) rank-1 |cos| score; stats over all matched
    pairs AND restricted to matched pairs whose component is alive."""
    rows, cols = _hungarian(score)
    cos = score[rows, cols]
    m_alive = alive[rows]
    return {"n_matched": int(len(cos)),
            "matched": [[int(r), int(k), float(s)]
                        for r, k, s in zip(rows, cols, cos)],
            "all": _stats(cos), "alive": _stats(cos[m_alive]),
            "matched_cos_alive": cos[m_alive].tolist()}


def _mass_block(V: np.ndarray, U: np.ndarray):
    mass = np.linalg.norm(V, axis=0) * np.linalg.norm(U, axis=1)
    alive = mass > ALIVE_FRAC * mass.max()
    block = {"C": int(mass.size), "n_alive": int(alive.sum()),
             "max": float(mass.max()), "median": float(np.median(mass)),
             "alive_idx": np.flatnonzero(alive).tolist()}
    return mass, alive, block


def _conc_block(A: np.ndarray, alive: np.ndarray, n_axes: int) -> dict:
    """Axis concentration of unit rows A (C, dim): per-component top-axis share
    |A_c|_inf / |A_c|_2 plus distinct-argmax coverage (alive and all-C)."""
    An = _unit_rows(A)
    share = np.abs(An).max(axis=1)
    arg = np.abs(An).argmax(axis=1)
    return {"top_share": share.tolist(), "argmax": arg.tolist(),
            "alive": _stats(share[alive]), "all": _stats(share),
            "coverage_alive": _coverage(arg, alive), "coverage_all": _coverage(arg),
            "n_axes": n_axes}


def load_run(run_dir: str) -> dict:
    """Load the latest model_<step>.pth of a param-decomp run. Orientation
    (verified empirically vs target_model + param-decomp LinearComponents source):
    V is (d_in, C), U is (C, d_out); learned piece c = outer(V[:,c], U[c,:]) in
    math (in, out) orientation; sum_c V@U approximates target.weight.T only up
    to the runtime delta component (residual recorded, NOT asserted small)."""
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pth"))
    assert ckpts, f"no model_*.pth in {run_dir}"
    step, path = max((int(re.search(r"model_(\d+)\.pth", p).group(1)), p)
                     for p in ckpts)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    out = {"ckpt": path, "step": step, "modules": {}}
    for m in MODULES:
        W = sd[f"target_model.{m}.weight"].double().numpy()   # (out, in)
        V = sd[f"_components.{m}.V"].double().numpy()         # (d_in, C)
        U = sd[f"_components.{m}.U"].double().numpy()         # (C, d_out)
        assert V.shape[0] == W.shape[1] and U.shape[1] == W.shape[0], \
            f"{m}: V{V.shape}/U{U.shape} inconsistent with weight {W.shape}"
        rel = float(np.linalg.norm(V @ U - W.T) / np.linalg.norm(W))
        out["modules"][m] = {"V": V, "U": U, "W_math": W.T.copy(),
                             "sum_rel_residual": rel}
    return out


def load_ideal(d_emb: int) -> dict:
    path = os.path.join(CKPT_DIR, f"ideal_d{d_emb}.pt")
    assert os.path.exists(path), f"run --build first ({path} missing)"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: ck[k].double().numpy() for k in ("W_e", "W_in", "U", "S", "V", "L")}


def analyze_run(run: dict, ideal: dict, d_emb: int) -> dict:
    We = ideal["W_e"]                                         # (F, d_emb)
    Wen = _unit_rows(We)
    Ui, Vi = _unit_cols(ideal["U"]), _unit_cols(ideal["V"])   # u_k (d_emb,50), v_k (N,50)

    # sanity: the run's target model IS the exported designed model
    dev_emb = float(np.abs(run["modules"]["lin_emb"]["W_math"] - We).max())
    dev_in = float(np.abs(run["modules"]["lin_in"]["W_math"] - ideal["W_in"]).max())
    assert dev_emb < 1e-5 and dev_in < 1e-5, \
        f"run target != ideal build (dev emb {dev_emb}, in {dev_in})"

    out = {}

    # ---- lin_emb: components vs ideal row pieces e_j (x) W_e[j] ----
    V, U = run["modules"]["lin_emb"]["V"], run["modules"]["lin_emb"]["U"]
    mass, alive, mb = _mass_block(V, U)
    cosV = np.abs(_unit_cols(V).T)                            # (C, F): |cos(V_c, e_j)|
    cosU = np.abs(_unit_rows(U) @ Wen.T)                      # (C, F): |cos(U_c, W_e[j])|
    out["lin_emb"] = {
        "mass": mb, "sum_rel_residual": run["modules"]["lin_emb"]["sum_rel_residual"],
        "input_concentration": _conc_block(V.T, alive, n_axes=F),
        "hungarian_row_pieces": _match_block(cosV * cosU, alive)}

    # ---- lin_in (headline): V-side vs three frames ----
    V, U = run["modules"]["lin_in"]["V"], run["modules"]["lin_in"]["U"]
    mass, alive, mb = _mass_block(V, U)
    Vn, Un = _unit_cols(V), _unit_rows(U)
    # (i) ideal SVD triplets: full rank-1 pieces and V-side only
    cosV = np.abs(Vn.T @ Ui)                                  # (C, 50) vs u_k
    cosU = np.abs(Un @ Vi)                                    # (C, 50) vs v_k
    svd_full = _match_block(cosV * cosU, alive)
    vside_best = cosV.max(axis=1)
    svd_vside = {"best_cos": vside_best.tolist(),
                 "best_k": cosV.argmax(axis=1).tolist(),
                 "alive": _stats(vside_best[alive]), "all": _stats(vside_best)}
    # (ii) embedded-feature frame rows of W_e (non-orthogonal: best match, no Hungarian)
    frame_cos = np.abs(Vn.T @ Wen.T)                          # (C, F)
    fb, fa = frame_cos.max(axis=1), frame_cos.argmax(axis=1)
    frame = {"best_cos": fb.tolist(), "best_feature": fa.tolist(),
             "alive": _stats(fb[alive]), "all": _stats(fb),
             "coverage_alive": _coverage(fa, alive), "coverage_all": _coverage(fa)}
    # (iii) U-side (output, neuron space) axis concentration
    uside = _conc_block(U, alive, n_axes=N)
    # verdict (thresholds verbatim in THRESHOLDS)
    hung_med = svd_full["alive"].get("median", float("nan"))
    frame_med = frame["alive"].get("median", float("nan"))
    if hung_med > 0.8:
        verdict = "svd_aligned"
    elif frame_med > 0.8 and frame["coverage_alive"] >= 60:
        verdict = "feature_frame_aligned"
    elif int(mb["n_alive"]) < 25:
        verdict = "collapsed"
    else:
        verdict = "mixed"
    out["lin_in"] = {
        "mass": mb, "sum_rel_residual": run["modules"]["lin_in"]["sum_rel_residual"],
        "svd_full_hungarian": svd_full, "svd_vside": svd_vside,
        "feature_frame": frame, "uside_neuron_concentration": uside,
        "verdict": verdict, "thresholds": THRESHOLDS}

    # ---- lin_out: input side = neuron space (write-side story) ----
    V, U = run["modules"]["lin_out"]["V"], run["modules"]["lin_out"]["U"]
    mass, alive, mb = _mass_block(V, U)
    cosV = np.abs(_unit_cols(V).T @ Vi)                       # (C, 50) vs v_k (input)
    cosU = np.abs(_unit_rows(U) @ Ui)                         # (C, 50) vs u_k (output)
    out["lin_out"] = {
        "mass": mb, "sum_rel_residual": run["modules"]["lin_out"]["sum_rel_residual"],
        "neuron_concentration": _conc_block(V.T, alive, n_axes=N),
        "hungarian_triplets_T": _match_block(cosV * cosU, alive)}

    # ---- lin_unemb: transposed row pieces W_e[j] (x) e_j (mirror of lin_emb) ----
    V, U = run["modules"]["lin_unemb"]["V"], run["modules"]["lin_unemb"]["U"]
    mass, alive, mb = _mass_block(V, U)
    cosV = np.abs(_unit_cols(V).T @ Wen.T)                    # (C, F): vs W_e[j] (input)
    cosU = np.abs(_unit_rows(U))                              # (C, F): |cos(U_c, e_j)|
    out["lin_unemb"] = {
        "mass": mb, "sum_rel_residual": run["modules"]["lin_unemb"]["sum_rel_residual"],
        "output_concentration": _conc_block(U, alive, n_axes=F),
        "hungarian_row_pieces_T": _match_block(cosV * cosU, alive)}
    return out


def _fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{nd}f}"


def _stage_rows(stage: str, r: dict):
    """per_matrix.csv rows for one stored run JSON."""
    rows = []
    for m in MODULES:
        d = r["matrices"][m]
        conc_key = {"lin_emb": "input_concentration", "lin_in": "uside_neuron_concentration",
                    "lin_out": "neuron_concentration", "lin_unemb": "output_concentration"}[m]
        hung_key = {"lin_emb": "hungarian_row_pieces", "lin_in": "svd_full_hungarian",
                    "lin_out": "hungarian_triplets_T", "lin_unemb": "hungarian_row_pieces_T"}[m]
        conc, hung = d[conc_key], d[hung_key]
        row = {"stage": stage, "matrix": m, "C": d["mass"]["C"],
               "n_alive": d["mass"]["n_alive"],
               "sum_rel_residual": _fmt(d["sum_rel_residual"]),
               "conc_median_alive": _fmt(conc["alive"].get("median")),
               "conc_coverage_alive": conc["coverage_alive"],
               "hung_median_cos_alive": _fmt(hung["alive"].get("median")),
               "hung_frac_gt0.8_alive": _fmt(hung["alive"].get("frac_gt_0.8")),
               "frame_best_median_alive": "", "frame_coverage_alive": "",
               "svd_vside_median_alive": "", "verdict": ""}
        if m == "lin_in":
            row.update({
                "frame_best_median_alive": _fmt(d["feature_frame"]["alive"].get("median")),
                "frame_coverage_alive": d["feature_frame"]["coverage_alive"],
                "svd_vside_median_alive": _fmt(d["svd_vside"]["alive"].get("median")),
                "verdict": d["verdict"]})
        rows.append(row)
    return rows


def _stage_md(stage: str, r: dict) -> str:
    tag = ("**UNCONSOLIDATED** (10k, pre warm-restart — no basis verdicts final)"
           if stage == "10k" else "consolidated (post warm-restart)")
    lines = [f"## Stage {stage} — {tag}", "",
             f"- run: `{r['run_dir']}` (checkpoint `{os.path.basename(r['ckpt'])}`, "
             f"step {r['step']})",
             f"- alive threshold: mass > {ALIVE_FRAC:.0%} of per-matrix max mass", "",
             "| matrix | alive/C | Σ-comp rel resid | conc median (alive) | "
             "conc coverage (alive) | Hungarian med \\|cos\\| (alive / all) | "
             "frac>0.8 (alive) |",
             "|---|---|---|---|---|---|---|"]
    for row, m in zip(_stage_rows(stage, r), MODULES):
        d = r["matrices"][m]
        hung_key = {"lin_emb": "hungarian_row_pieces", "lin_in": "svd_full_hungarian",
                    "lin_out": "hungarian_triplets_T", "lin_unemb": "hungarian_row_pieces_T"}[m]
        conc_key = {"lin_emb": "input_concentration", "lin_in": "uside_neuron_concentration",
                    "lin_out": "neuron_concentration", "lin_unemb": "output_concentration"}[m]
        hung, conc = d[hung_key], d[conc_key]
        lines.append(
            f"| {m} | {row['n_alive']}/{row['C']} | {row['sum_rel_residual']} | "
            f"{row['conc_median_alive']} | {row['conc_coverage_alive']}/{conc['n_axes']} | "
            f"{row['hung_median_cos_alive']} / {_fmt(hung['all'].get('median'))} | "
            f"{row['hung_frac_gt0.8_alive']} |")
    din = r["matrices"]["lin_in"]
    lines += ["",
        f"### lin_in three-frame readout (stage {stage})", "",
        f"- (i) SVD triplets: Hungarian full-piece median |cos| = "
        f"{_fmt(din['svd_full_hungarian']['alive'].get('median'))} (alive) / "
        f"{_fmt(din['svd_full_hungarian']['all'].get('median'))} (all matched); "
        f"V-side-only best |cos| vs u_k median = "
        f"{_fmt(din['svd_vside']['alive'].get('median'))} (alive)",
        f"- (ii) embedded-feature frame {{W_e[j]}}: best-|cos| median = "
        f"{_fmt(din['feature_frame']['alive'].get('median'))} (alive) / "
        f"{_fmt(din['feature_frame']['all'].get('median'))} (all); distinct-feature "
        f"coverage {din['feature_frame']['coverage_alive']}/{F} (alive), "
        f"{din['feature_frame']['coverage_all']}/{F} (all)",
        f"- (iii) U-side neuron axes: top-share median = "
        f"{_fmt(din['uside_neuron_concentration']['alive'].get('median'))} (alive); "
        f"distinct-neuron coverage "
        f"{din['uside_neuron_concentration']['coverage_alive']}/{N}",
        f"- **verdict: `{din['verdict']}`** — thresholds: "
        + "; ".join(f"{k}: {v}" for k, v in THRESHOLDS.items() if k != "alive"),
        ""]
    return "\n".join(lines)


CSV_FIELDS = ["stage", "matrix", "C", "n_alive", "sum_rel_residual",
              "conc_median_alive", "conc_coverage_alive", "hung_median_cos_alive",
              "hung_frac_gt0.8_alive", "frame_best_median_alive",
              "frame_coverage_alive", "svd_vside_median_alive", "verdict"]


def write_tables(d_emb: int) -> None:
    """Regenerate comparison.md + per_matrix.csv from ALL stage JSONs present."""
    res_d = os.path.join(RES_DIR, f"d{d_emb}")
    stored = [(s, os.path.join(res_d, f"run_{s}.json")) for s in STAGES]
    stored = [(s, p) for s, p in stored if os.path.exists(p)]
    rows, md = [], [f"# Ideal PD vs VPD — embedded designed code, d_emb={d_emb}", "",
                    "Per-matrix structural comparison of learned VPD subcomponents "
                    "(V/U rank-1 pieces) against the hand-derived ideal PD "
                    "(row pieces for W_emb/W_embᵀ, SVD triplets for W_in/W_out).", ""]
    for stage, path in stored:
        with open(path) as f:
            r = json.load(f)
        rows += _stage_rows(stage, r)
        md.append(_stage_md(stage, r))
    with open(os.path.join(res_d, "per_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(res_d, "comparison.md"), "w") as f:
        f.write("\n".join(md))
    print(f"  tables ({', '.join(s for s, _ in stored)}) -> "
          f"{res_d}/{{comparison.md, per_matrix.csv}}")


def _hist(ax, vals, color, title, xlabel, xlim=(0, 1)):
    v = np.asarray(vals, dtype=float)
    ax.hist(v, bins=np.linspace(*xlim, 21), color=color, edgecolor="white", lw=0.5)
    if v.size:
        ax.axvline(np.median(v), color="0.25", lw=1.2, ls="--")
        ax.text(0.03, 0.95, f"med {np.median(v):.3f}\nn = {v.size}",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_compare(d_emb: int, stage: str, mx: dict) -> None:
    fig_d = os.path.join(FIG_DIR, f"d{d_emb}")
    os.makedirs(fig_d, exist_ok=True)
    suff = f"_{stage}"
    # align_wemb: lin_emb + lin_unemb (tied transpose) matched |cos| vs row pieces
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    _hist(axes[0], mx["lin_emb"]["hungarian_row_pieces"]["matched_cos_alive"],
          "steelblue", "lin_emb vs row pieces e_j⊗W_e[j]",
          "Hungarian matched |cos| (alive)")
    _hist(axes[1], mx["lin_unemb"]["hungarian_row_pieces_T"]["matched_cos_alive"],
          "steelblue", "lin_unemb vs transposed row pieces",
          "Hungarian matched |cos| (alive)")
    axes[0].set_ylabel("components")
    fig.suptitle(f"W_emb / W_embᵀ alignment with ideal row pieces "
                 f"(d_emb={d_emb}, stage {stage})", fontsize=10)
    fig.savefig(os.path.join(fig_d, f"align_wemb{suff}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    # align_win_svd_vs_frame: the three alignment measures side by side
    din = mx["lin_in"]
    alive = np.zeros(din["mass"]["C"], dtype=bool)
    alive[din["mass"]["alive_idx"]] = True
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    _hist(axes[0], din["svd_full_hungarian"]["matched_cos_alive"], "steelblue",
          "(i) SVD triplets σ_k u_k v_kᵀ", "Hungarian matched |cos| (alive)")
    _hist(axes[1], np.asarray(din["feature_frame"]["best_cos"])[alive], "darkorange",
          "(ii) feature frame {W_e[j]}", "best |cos|, V-side (alive)")
    _hist(axes[2], np.asarray(din["uside_neuron_concentration"]["top_share"])[alive],
          "seagreen", "(iii) neuron axes, U-side", "top-axis share (alive)")
    axes[0].set_ylabel("components")
    fig.suptitle(f"lin_in (W_in) V-side vs three frames — verdict: {din['verdict']} "
                 f"(d_emb={d_emb}, stage {stage})", fontsize=10)
    fig.savefig(os.path.join(fig_d, f"align_win_svd_vs_frame{suff}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    # align_wout: matched |cos| vs transposed triplets + write-side neuron share
    dout = mx["lin_out"]
    alive = np.zeros(dout["mass"]["C"], dtype=bool)
    alive[dout["mass"]["alive_idx"]] = True
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    _hist(axes[0], dout["hungarian_triplets_T"]["matched_cos_alive"], "steelblue",
          "lin_out vs transposed triplets σ_k v_k u_kᵀ",
          "Hungarian matched |cos| (alive)")
    _hist(axes[1], np.asarray(dout["neuron_concentration"]["top_share"])[alive],
          "seagreen", "input side = neuron space (write-side)",
          "top-neuron share (alive)")
    axes[0].set_ylabel("components")
    fig.suptitle(f"W_out alignment (d_emb={d_emb}, stage {stage})", fontsize=10)
    fig.savefig(os.path.join(fig_d, f"align_wout{suff}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  figures -> {fig_d}/align_{{wemb,win_svd_vs_frame,wout}}{suff}.png")


def compare(run_dir: str, d_emb: int, stage: str) -> None:
    """Part 3: structural comparison of a VPD run against the stored ideal PD.
    Per matrix: alive counts, Σ-component residual, axis concentration/coverage,
    Hungarian |cos| vs the ideal pieces; lin_in additionally against the three
    frames (SVD u_k / feature frame W_e[j] / U-side neuron axes) with a verdict.
    Writes results/embedded/ideal_vs_vpd/d{d_emb}/{run_<stage>.json, comparison.md,
    per_matrix.csv} (tables regenerated over all stages present) + align figures."""
    ideal = load_ideal(d_emb)
    run = load_run(run_dir)
    print(f"compare d_emb={d_emb} stage={stage}: {run['ckpt']} (step {run['step']})")
    mx = analyze_run(run, ideal, d_emb)
    for m in MODULES:
        print(f"  [{m}] alive {mx[m]['mass']['n_alive']}/{mx[m]['mass']['C']}, "
              f"sum-of-components rel residual {mx[m]['sum_rel_residual']:.3f}")
    print(f"  [lin_in] verdict: {mx['lin_in']['verdict']}")
    res_d = os.path.join(RES_DIR, f"d{d_emb}")
    os.makedirs(res_d, exist_ok=True)
    payload = {"d_emb": d_emb, "stage": stage,
               "consolidation": ("UNCONSOLIDATED (10k, pre warm-restart)"
                                 if stage == "10k" else "post warm-restart"),
               "run_dir": run_dir, "ckpt": run["ckpt"], "step": run["step"],
               "alive_frac": ALIVE_FRAC, "thresholds": THRESHOLDS,
               "orientation": "V (d_in, C), U (C, d_out); piece_c = V[:,c] ⊗ U[c,:] "
                              "(math in×out); Σ pieces ≈ weight.T up to the runtime "
                              "delta component (residual recorded per matrix)",
               "matrices": mx}
    json_path = os.path.join(res_d, f"run_{stage}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  metrics -> {json_path}")
    write_tables(d_emb)
    plot_compare(d_emb, stage, mx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="build the designed embedded models + ideal PD + exports")
    ap.add_argument("--compare", metavar="RUN_DIR", default=None,
                    help="structural comparison vs a VPD run dir (Part 3)")
    ap.add_argument("--d-emb", type=int, nargs="*", default=list(D_EMBS),
                    dest="d_embs",
                    help="d_emb values; --compare requires exactly one (200 or 50)")
    ap.add_argument("--stage", choices=STAGES, default="10k",
                    help="run stage label for --compare outputs (10k = "
                         "unconsolidated, warm = post warm-restart)")
    a = ap.parse_args()
    if not a.build and a.compare is None:
        ap.error("choose --build and/or --compare RUN_DIR")
    if a.build:
        build(a.d_embs)
    if a.compare is not None:
        if len(a.d_embs) != 1:
            ap.error("--compare requires exactly one --d-emb (200 or 50)")
        compare(a.compare, a.d_embs[0], a.stage)


if __name__ == "__main__":
    main()
