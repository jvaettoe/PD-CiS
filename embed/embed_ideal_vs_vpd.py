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
  ./.venv/bin/python -m embed.embed_ideal_vs_vpd --compare <run_dir> --d-emb 200   # Part 3 stub
"""
import argparse
import json
import os

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


def compare(run_dir: str, d_embs) -> None:
    """Part 3 (stub): structural comparison of a VPD run against the stored ideal
    PD, per (d_emb, matrix). W_emb components vs ideal row pieces (input-side
    concentration, distinct-feature coverage, Hungarian |cos|); W_in learned
    V-side vs THREE frames — (i) SVD u_k, (ii) embedded-feature frame {W_e[j]},
    (iii) neuron axes on the U-side — with verdict svd_aligned /
    feature_frame_aligned / mixed (thresholds stored verbatim); W_out / W_emb.T
    the same metrics transposed. Writes results/embedded/ideal_vs_vpd/d{d_emb}/
    {comparison.md, per_matrix.csv, run_10k.json, run_warm.json} + align figures.
    Lands with plan Part 3 once the VPD runs (Part 2) produce run_dir."""
    for d_emb in d_embs:
        ideal_path = os.path.join(CKPT_DIR, f"ideal_d{d_emb}.pt")
        assert os.path.exists(ideal_path), f"run --build first ({ideal_path} missing)"
    raise NotImplementedError(
        "--compare is implemented in plan Part 3 (needs the VPD runs from Part 2); "
        f"got run_dir = {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="build the designed embedded models + ideal PD + exports")
    ap.add_argument("--compare", metavar="RUN_DIR", default=None,
                    help="structural comparison vs a VPD run dir (Part 3, stub)")
    ap.add_argument("--d-emb", type=int, nargs="*", default=list(D_EMBS),
                    dest="d_embs")
    a = ap.parse_args()
    if not a.build and a.compare is None:
        ap.error("choose --build and/or --compare RUN_DIR")
    if a.build:
        build(a.d_embs)
    if a.compare is not None:
        compare(a.compare, a.d_embs)


if __name__ == "__main__":
    main()
