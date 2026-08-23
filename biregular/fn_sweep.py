"""(n_features, n_neurons) sweep for the plain biregular code (transpose decoder):
per-cell L4-optimal codeword length K*, then oracle-gated 2x2 basis preference
under the exact VPD losses.

Basis convention (w_in is (F, N); w_out = w_in^T is (N, F)):
  W_in  feature basis = ROWS    (per-feature codeword reads, F pieces)
  W_in  neuron  basis = COLUMNS (per-neuron read pieces, N pieces)
  W_out neuron  basis = ROWS    (per-neuron write pieces, N pieces)
  W_out feature basis = COLUMNS (per-feature write pieces, F pieces)
ExactPD = (feature, feature); the learned-VPD ontology at (100, 50) = (feature,
neuron). Mixed combos zero-pad to C = max(F, N) with all-zero pieces whose
oracle gates are pinned to 0 (no recon effect; importance ~ (1e-12)^pnorm).

Oracle gates: feature piece j fires iff x_j != 0; neuron piece n fires iff
h_n != 0 under the frozen model. For W_in neuron columns this is still exact:
a masked column only changes a preactivation the ReLU clips anyway.

  ./.venv/bin/python -m biregular.fn_sweep --fit-grid   [--max-f 300]
  ./.venv/bin/python -m biregular.fn_sweep --oracle-grid [--max-f 300]
  ./.venv/bin/python -m biregular.fn_sweep --figures
"""
import argparse
import itertools
import json
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from biregular.biregular_vpd_compare import DEFAULT_KW
from biregular.vpd_objective import EVAL_SEED, eval_objective
from core.codes import greedy_regular_code
from core.data import CleanDataset
from core.mlp import MLP
from core.paths import FIGURES, RESULTS
from core.vpd import VPD
from embed.embed_equivalence import fit_ab_on_code, model_from_code

F_GRID = [40, 60, 100, 150, 200, 300, 500, 1000]
N_GRID = [20, 30, 50, 75, 100, 150, 250, 500]
# XL grids (supersets of the base grids; --xl): finer + up to 10,000 both axes
F_GRID_XL = [40, 60, 100, 150, 200, 300, 500, 700, 1000, 1500, 2500, 4000,
             6500, 10000]
N_GRID_XL = [20, 30, 50, 75, 100, 150, 250, 350, 500, 750, 1000, 1500, 2500,
             4000, 6500, 10000]
K_COARSE_POINTS = 12   # log-spaced over the FULL range [1, N], endpoints included
K_REFINE_ROUNDS = 3    # geometric-midpoint refinement around the argmin
P = 0.001
CODE_SEED = 0
DECODER = "transpose"
BASES = ("feature", "neuron")
TIE_RATIO = 1.05
HELD_OUT_BATCHES, HELD_OUT_BATCH = 30, 8192

RES_DIR = os.path.join(RESULTS, "fn_sweep")

# --- code-construction families (per-family result/figure subfolders) ---
#   greedy  : least-used-neuron near-biregular (this repo; no divisibility req)
#   dfh     : Ferreira da Silva-Heimersheim-style exact biregular (stub
#             matching, core.codes.regular_code; requires N | F*K)
#   dfh_om  : dfh + reduce_overlap swap pass (overlap-min benchmark variant)
FAMILIES = ("greedy", "dfh", "dfh_om")
FAMILY = None
OM_SWAP_BUDGET = 4e10       # ~op budget for reduce_overlap; iters = budget/(F*N)
OM_MAX_SWAPS = 800_000      # the established (100, 50) swap count
OM_MAX_CELL = 1_000_000     # skip overlap-min for F*N above this
_CODE_CACHE = {}


def set_family(family: str) -> None:
    """Point all result/figure paths at the family's subfolder."""
    global FAMILY, RES_FAM_DIR, KFIT_DIR, ORACLE_DIR, EXACT_SOLO_DIR, FIG_DIR
    assert family in FAMILIES, family
    FAMILY = family
    RES_FAM_DIR = os.path.join(RES_DIR, family)
    KFIT_DIR = os.path.join(RES_FAM_DIR, "kfit")
    ORACLE_DIR = os.path.join(RES_FAM_DIR, "oracle")
    EXACT_SOLO_DIR = os.path.join(RES_FAM_DIR, "exact_solo")
    FIG_DIR = os.path.join(FIGURES, "fn_sweep", family)
    for d in (KFIT_DIR, ORACLE_DIR, EXACT_SOLO_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def build_code(F_, N_, K):
    """Code matrix for the current family (cached; dfh_om's swap pass is
    expensive). dfh families require N | F*K (exact biregularity)."""
    key = (FAMILY, F_, N_, K)
    if key not in _CODE_CACHE:
        if FAMILY == "greedy":
            M = greedy_regular_code(F_, N_, K, CODE_SEED)
        else:
            from core import codes as _codes
            M = _codes.regular_code(F_, N_, K, CODE_SEED)
            if FAMILY == "dfh_om":
                iters = int(min(OM_MAX_SWAPS, OM_SWAP_BUDGET / (F_ * N_)))
                M = _codes.reduce_overlap(M, iters, seed=10 + CODE_SEED)
        if len(_CODE_CACHE) > 40:
            _CODE_CACHE.clear()
        _CODE_CACHE[key] = M
    return _CODE_CACHE[key]


set_family("greedy")


# dfh grid: F = ratio * N so gcd(F, N) = N and the admissible-K step is 1 --
# every K is available and K* is the unconstrained optimum (comparable to
# greedy). ~71 cells spanning the same compression range as the XL grid.
DFH_RATIOS = (2, 3, 4, 6, 10, 16, 25, 40)
DFH_N = (20, 30, 50, 75, 100, 150, 250, 500, 1000, 2000)
DFH_CELLS = sorted((r * n, n) for n in DFH_N for r in DFH_RATIOS if r * n <= 10_000)


def cells(max_f=None, max_n=None):
    pool = (DFH_CELLS if FAMILY in ("dfh", "dfh_om")
            else itertools.product(F_GRID, N_GRID))
    for F_, N_ in pool:
        if (max_f is None or F_ <= max_f) and (max_n is None or N_ <= max_n):
            yield F_, N_


def grid_axes():
    """Figure axes = the distinct F and N values of the current family's grid."""
    cs = list(cells())
    return sorted({c[0] for c in cs}), sorted({c[1] for c in cs})


# ------------------------------------------------------------- stage 1: K*

def fit_budget(F_, N_=0):
    """Scale-adaptive (steps, batch, pool, dtype) for fit_ab_on_code: the
    2-parameter fit needs no more; large cells dominate wall-clock and memory.
    fp32 for big cells (GPU fp64 is 1/64 throughput on consumer cards; the fit
    is a 2-scalar optimization and fp32 suffices -- recorded per cell)."""
    if F_ * N_ >= 1_000_000 or F_ >= 2000:
        return 3000, 2048, 32768, "float32"
    if F_ >= 500:
        return 3000, 2048, 65536, "float64"
    return 4000, 4096, 131072, "float64"


def held_out_l4(model, p=P, batches=HELD_OUT_BATCHES, batch=HELD_OUT_BATCH,
                seed=EVAL_SEED, device="cpu"):
    """Multi-batch held-out L4 + features-computed count (per-feature
    active-conditional MSE < 1/30; always-0 baseline is 1/3)."""
    F_ = model.n_features
    ds = CleanDataset(n_features=F_, p=p, device=device, seed=seed)
    l4 = 0.0
    err, cnt = torch.zeros(F_, device=device), torch.zeros(F_, device=device)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            out = model(x)
            l4 += float(((out - y).abs() ** 4).mean())
            a = (x > 0).float()
            err += (((out - y) ** 2) * a).sum(0)
            cnt += a.sum(0)
    pf = (err / cnt.clamp(min=1)).cpu().numpy()
    return l4 / batches, int((pf < 1 / 30).sum())


def kfit_path(F_, N_):
    return os.path.join(KFIT_DIR, f"F{F_}_N{N_}.json")


def k_step(F_, N_):
    """Admissible-K step for exact biregularity: N | F*K <=> K = 0 mod
    N/gcd(F, N). The greedy family has no such constraint (step 1)."""
    if FAMILY == "greedy":
        return 1
    return N_ // math.gcd(F_, N_)


def k_candidates(N_, F_=None, points=K_COARSE_POINTS):
    """Log-spaced fannings over the admissible range: [1, N] for greedy; the
    multiples of k_step for the exact-biregular (dfh) families. Endpoints of
    the admissible set always included."""
    step = 1 if F_ is None else k_step(F_, N_)
    if step == 1:
        return sorted(set(int(round(g)) for g in np.geomspace(1, N_, num=points)))
    admissible = list(range(step, N_ + 1, step))
    if len(admissible) <= points:
        return admissible
    idx = sorted(set(int(round(g)) for g in
                     np.geomspace(1, len(admissible), num=points)))
    return [admissible[i - 1] for i in idx]


def fit_cell(F_, N_, device="cpu"):
    """K sweep for one cell: coarse log-spaced pass over [1, N], then geometric
    midpoint refinement around the argmin; per K: greedy code -> (a, b) L4 fit
    -> held-out L4. K* = final argmin."""
    steps, batch, pool, dtype = fit_budget(F_, N_)
    step = k_step(F_, N_)
    rows, tried = [], set()

    def run_k(K):
        M = torch.tensor(build_code(F_, N_, K))
        a, b = fit_ab_on_code(M, DECODER, p=P, steps=steps, batch=batch,
                              pool=pool, device=device,
                              dtype=getattr(torch, dtype))
        model = model_from_code(M, a, b, DECODER).to(device)
        l4, computed = held_out_l4(model, device=device)
        rows.append({"K": K, "a": a, "b": b, "l4": l4, "computed": computed})
        tried.add(K)
        print(f"  F={F_} N={N_} K={K}: l4={l4:.3e} computed={computed} "
              f"a={a:.4f} b={b:.4f}", flush=True)

    blown = 0
    for K in k_candidates(N_, F_):  # ascending; stop once the tail is blown up
        run_k(K)
        best_l4 = min(r["l4"] for r in rows)
        blown = blown + 1 if rows[-1]["l4"] > 50 * max(best_l4, 1e-12) else 0
        if blown >= 2:
            print(f"  F={F_} N={N_}: early stop at K={K} (L4 > 50x best twice)",
                  flush=True)
            break
    for _ in range(K_REFINE_ROUNDS):  # geometric midpoints beside the argmin
        best_k = min(rows, key=lambda r: r["l4"])["K"]
        ts = sorted(tried)
        i = ts.index(best_k)
        new = [step * max(1, int(round(math.sqrt(best_k * ts[j]) / step)))
               for j in (i - 1, i + 1) if 0 <= j < len(ts)]
        new = [k for k in set(new) if k not in tried and 1 <= k <= N_]
        if not new:
            break
        for K in sorted(new):
            run_k(K)

    best = min(rows, key=lambda r: r["l4"])
    rec = {"F": F_, "N": N_, "p": P, "decoder": DECODER, "code_seed": CODE_SEED,
           "family": FAMILY, "k_step": step,
           "fit": {"steps": steps, "batch": batch, "pool": pool, "dtype": dtype,
                   "k_rule": f"geomspace[1,N]x{K_COARSE_POINTS}"
                             f"+refine{K_REFINE_ROUNDS}"},
           "held_out": {"batches": HELD_OUT_BATCHES, "batch": HELD_OUT_BATCH,
                        "seed": EVAL_SEED},
           "rows": rows, "K_star": best["K"], "a": best["a"], "b": best["b"],
           "l4_star": best["l4"], "computed_star": best["computed"]}
    with open(kfit_path(F_, N_), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def run_fit_grid(max_f, max_n, device="cpu"):
    todo = [(F_, N_) for F_, N_ in cells(max_f, max_n)
            if not os.path.exists(kfit_path(F_, N_))]
    if FAMILY == "dfh_om":   # reduce_overlap is O(F*N) per swap; cap cell size
        skipped = [c for c in todo if c[0] * c[1] > OM_MAX_CELL]
        todo = [c for c in todo if c[0] * c[1] <= OM_MAX_CELL]
        if skipped:
            print(f"fit-grid[{FAMILY}]: skipping {len(skipped)} cells with "
                  f"F*N > {OM_MAX_CELL:g} (overlap-min cost)", flush=True)
    print(f"fit-grid[{FAMILY}]: {len(todo)} cells to fit "
          f"(of {sum(1 for _ in cells(max_f, max_n))}, device {device})", flush=True)
    for F_, N_ in sorted(todo, key=lambda c: c[0] * c[1]):  # cheap cells first
        print(f"[cell F={F_} N={N_}]", flush=True)
        rec = fit_cell(F_, N_, device=device)
        print(f"  -> K*={rec['K_star']} l4*={rec['l4_star']:.3e} "
              f"computed={rec['computed_star']}", flush=True)


# ------------------------------------------- stage 2: oracle 2x2 basis combos

class ComboOracleVPD(VPD):
    """Oracle gates for any (read_basis, write_basis) combo on a plain MLP
    (identity embedding, e == x). Feature pieces gate on x_j != 0, neuron
    pieces on h_n != 0 (frozen-model hidden); zero-padded slots gate at 0.
    lower == upper == the oracle gate (no CI net)."""

    def __init__(self, model, C, combo):
        self.combo = dict(zip(("w_in", "w_out"), combo))
        super().__init__(model, C, warmup_steps=0)

    def _ci(self, e, h):
        gates = {}
        for name in self.param_names:
            g = (e != 0).float() if self.combo[name] == "feature" else (h != 0).float()
            if g.shape[1] < self.C:
                pad = torch.zeros(g.shape[0], self.C - g.shape[1], device=g.device)
                g = torch.cat([g, pad], dim=1)
            gates[name] = g
        return gates, dict(gates)


def build_combo_vpd(model, read_basis, write_basis):
    """Frozen VPD state for one basis combo. Row pieces: piece i = e_i (x) W[i, :]
    (V = I, U = W); column pieces: piece j = W[:, j] (x) e_j (V = W, U = I).
    Genuine pieces first, then all-zero padding up to C = max piece count."""
    n_feat, d_mlp = model.n_features, model.d_mlp
    cnt = {"w_in": n_feat if read_basis == "feature" else d_mlp,
           "w_out": d_mlp if write_basis == "neuron" else n_feat}
    C = max(cnt.values())
    torch.manual_seed(0)  # CI-net init (unused: _ci bypasses it)
    pd_v = ComboOracleVPD(model, C, (read_basis, write_basis))

    def vu(W, rows):
        d0, d1 = W.shape
        if rows:
            V, U = torch.eye(d0), W.clone()
        else:
            V, U = W.clone(), torch.eye(d1)
        k = V.shape[1]
        Vp = torch.zeros(d0, C)
        Up = torch.zeros(C, d1)
        Vp[:, :k], Up[:k, :] = V, U
        return Vp, Up

    with torch.no_grad():
        for name, rows in (("w_in", read_basis == "feature"),
                           ("w_out", write_basis == "neuron")):
            V, U = vu(pd_v.targets[name], rows)
            pd_v.V[name].copy_(V)
            pd_v.U[name].copy_(U)
    for k in pd_v.V:
        pd_v.V[k].requires_grad_(False)
        pd_v.U[k].requires_grad_(False)
    for name in pd_v.param_names:  # exact tiling incl. zero padding
        err = float((pd_v._vu(name) - pd_v.targets[name]).abs().max())
        assert err < 1e-6, f"tiling failed for {name}: max|sum - W| = {err:.3e}"
    return pd_v


def oracle_path(F_, N_, K, rb, wb):
    return os.path.join(ORACLE_DIR, f"F{F_}_N{N_}_K{K}_{rb}-{wb}.json")


def run_oracle_grid(max_f, max_n, device="cpu"):
    for F_, N_ in cells(max_f, max_n):
        kp = kfit_path(F_, N_)
        if not os.path.exists(kp):
            continue
        with open(kp) as f:
            kf = json.load(f)
        K, a, b = kf["K_star"], kf["a"], kf["b"]
        combos = [(rb, wb) for rb, wb in itertools.product(BASES, BASES)
                  if not os.path.exists(oracle_path(F_, N_, K, rb, wb))]
        if not combos:
            continue
        M = torch.tensor(build_code(F_, N_, K))
        model = model_from_code(M, a, b, DECODER).to(device)
        model.device = device  # MLPBase caches device at __init__; keep it honest
        # memory guard for huge cells: smaller batches, same 200k-sample pool
        ekw = ({"batch": 2048, "n_batches": 100} if max(F_, N_) > 2000 else {})
        for rb, wb in combos:
            pd_v = build_combo_vpd(model, rb, wb)
            res = eval_objective(pd_v, p=P, **ekw)
            rec = {"F": F_, "N": N_, "K": K, "a": a, "b": b, "p": P,
                   "decoder": DECODER, "code_seed": CODE_SEED,
                   "read_basis": rb, "write_basis": wb,
                   "C": pd_v.C, "loss_kw": dict(DEFAULT_KW),
                   "eval": {"seed": EVAL_SEED}, "losses": res}
            with open(oracle_path(F_, N_, K, rb, wb), "w") as f:
                json.dump(rec, f, indent=1)
            print(f"F={F_} N={N_} K={K} ({rb},{wb}): total={res['total']:.4e}",
                  flush=True)


# ------------------------------------- exact-PD solo vs full: leak / attenuation



def exact_solo_cell(F_, N_, device="cpu"):
    """Full model vs ExactPD solo mechanisms, via the established (100, 50)
    machinery: biregular_single_component._analyse_model (one-hot e_j probes,
    ExactPD forward_masked solos, per-probe on-/off-target loss split with the
    per-coordinate /F normalization; solo-vs-full on-target and faithfulness
    asserted inside). By positive homogeneity the leak/atten ratio is exactly
    value-independent, so v = 1 probes fully determine it.

      self-attenuation = on-target error (shared exactly with the solo)
      leakage          = off-target power (what no union of solos can express)
    Reported ratio = leakage / self-attenuation (MSE headline, L4 alongside)."""
    with open(kfit_path(F_, N_)) as f:
        kf = json.load(f)
    K, a, b = kf["K_star"], kf["a"], kf["b"]
    if K == 1 and abs(b) < 1e-12 and abs(a - 1.0) < 1e-12:
        # dedicated-neuron cells are exact (proven bit-exact): leak = atten = 0
        rec = {"F": F_, "N": N_, "K": K, "a": a, "b": b, "p": P,
               "decoder": DECODER, "code_seed": CODE_SEED,
               "interference": 0.0, "atten_mse": 0.0, "leak_mse": 0.0,
               "atten_l4": 0.0, "leak_l4": 0.0, "ratio_mse": None,
               "ratio_l4": None, "solo_off_target_max": 0.0,
               "solo_check": "analytic (K=1, a=1, b=0: model exact)"}
        with open(os.path.join(EXACT_SOLO_DIR, f"F{F_}_N{N_}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        return rec
    M = torch.tensor(build_code(F_, N_, K))
    model = model_from_code(M, a, b, DECODER).to(device)
    model.device = device
    if F_ <= 1500:  # established machinery incl. full per-component solo loop
        from biregular.biregular_single_component import _analyse_model
        res = _analyse_model(model, p=str(P), device=device)
        R = torch.tensor(res["R"])                   # (probe j, coord i), v = 1
        extra = {"solo_off_target_max": res["comp_off_target_max"],
                 "l4_single_on": res["l4_single_on"],
                 "l4_single_off": res["l4_single_off"]}
    else:  # big-F fast path: one matmul for R; solo on-target check vectorized
        w_in, w_out = model.w_in.detach(), model.w_out.detach()
        with torch.no_grad():
            R = (torch.relu(torch.eye(F_, device=w_in.device) @ w_in)
                 @ w_out).cpu()
            solo_on = (torch.relu(w_in) * w_out.T).sum(dim=1).cpu()
        dev = float((solo_on - R.diagonal()).abs().max())
        assert dev < 1e-3, f"solo on-target mismatch {dev:.1e}"
        # solo off-target is structurally 0: the write piece is column j only
        extra = {"solo_off_target_max": 0.0, "solo_check": "vectorized",
                 "solo_on_target_max_dev": dev}
    dfull = R - torch.eye(F_)
    e2, e4 = dfull ** 2, dfull ** 4
    mse_on = float((e2.diagonal()).mean() / F_)
    mse_off = float((e2.sum(1) - e2.diagonal()).mean() / F_)
    l4_on = float((e4.diagonal()).mean() / F_)
    l4_off = float((e4.sum(1) - e4.diagonal()).mean() / F_)
    off = R - torch.diag(R.diagonal())
    rec = {"F": F_, "N": N_, "K": K, "a": a, "b": b, "p": P,
           "decoder": DECODER, "code_seed": CODE_SEED,
           "interference": float((off.pow(2).sum() / R.pow(2).sum())),
           "atten_mse": mse_on, "leak_mse": mse_off,
           "atten_l4": l4_on, "leak_l4": l4_off,
           "ratio_mse": mse_off / mse_on if mse_on > 1e-15 else None,
           "ratio_l4": l4_off / l4_on if l4_on > 1e-15 else None,
           **extra}
    with open(os.path.join(EXACT_SOLO_DIR, f"F{F_}_N{N_}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def run_exact_solo(max_f=None, max_n=None, device="cpu"):
    recs = []
    for F_, N_ in cells(max_f, max_n):
        pth = os.path.join(EXACT_SOLO_DIR, f"F{F_}_N{N_}.json")
        if os.path.exists(pth):
            with open(pth) as f:
                recs.append(json.load(f))
            continue
        if not os.path.exists(kfit_path(F_, N_)):
            continue
        rec = exact_solo_cell(F_, N_, device=device)
        r = rec["ratio_mse"]
        print(f"F={F_} N={N_} K={rec['K']}: leak/atten (mse) = "
              f"{'exact (0/0)' if r is None else f'{r:.3g}'}  "
              f"solo_off_max={rec['solo_off_target_max']:.1e}", flush=True)
        recs.append(rec)
    if recs:
        from biregular.fn_sweep_figures import exact_solo_figures
        fg, ng = grid_axes()
        exact_solo_figures(recs, fg, ng, FIG_DIR)
    return recs


def run_exact_solo_theory():
    """Theoretical leak/attenuation from the overlap ladder, with the LEARNED
    (a, b) inserted. For a one-hot probe of feature j (v > 0, transpose
    decoder) only j's K code neurons fire (b*v < 0 is ReLU-clipped), so
      on-target:  out_j = K a^2          -> atten = (1 - K a^2)^p
      leak on i:  out_i = m_ij a^2 + (K - m_ij) a b   (m_ij = codeword overlap)
    Two prediction levels per cell and loss (p = 2 MSE, p = 4 L4):
      ladder  -- exact overlaps m_ij from the cell's own code M (validates the
                 formula; should match the measured forward to float precision)
      ensemble-- a-priori from (a, b, K, F, N) alone with m ~ Hypergeom(N, K, K)
    Results appended into the exact_solo JSONs + a measured-vs-predicted check
    figure."""
    from scipy.stats import hypergeom
    recs = []
    for F_, N_ in cells():
        pth = os.path.join(EXACT_SOLO_DIR, f"F{F_}_N{N_}.json")
        if not os.path.exists(pth):
            continue
        with open(pth) as f:
            rec = json.load(f)
        K, a, b = rec["K"], rec["a"], rec["b"]
        atten = {p: (1.0 - K * a * a) ** p for p in (2, 4)}
        # ladder-exact needs the code instance; dfh's stub-repair construction
        # is single-core O(F*N)-per-collision -- skip it for big edge counts
        # (the ensemble prediction below needs no code matrix)
        if FAMILY == "dfh_om" or (FAMILY != "greedy"
                                  and F_ * K > 100_000):
            m_off = None   # code (re)construction too costly for ladder-exact
        else:
            M = build_code(F_, N_, K).astype(float)
            ov = M @ M.T                               # (F, F) overlaps
            i, j = np.meshgrid(np.arange(F_), np.arange(F_), indexing="ij")
            m_off = ov[i != j]                         # all m_ij, i != j
            lev_off = m_off * a * a + (K - m_off) * a * b
        ms = np.arange(0, K + 1)
        pmf = hypergeom(N_, K, K).pmf(ms)
        lev_ens = ms * a * a + (K - ms) * a * b
        for p, tag in ((2, "mse"), (4, "l4")):
            leak_ladder = (float((lev_off ** p).sum() / F_) / F_
                           if m_off is not None else None)
            leak_ens = (F_ - 1) * float((pmf * lev_ens ** p).sum()) / F_
            att = atten[p] / F_
            rec[f"pred_ladder_ratio_{tag}"] = (
                leak_ladder / att if leak_ladder is not None and att > 1e-15
                else None)
            rec[f"pred_ensemble_ratio_{tag}"] = (leak_ens / att if att > 1e-15
                                                 else None)
        with open(pth, "w") as f:
            json.dump(rec, f, indent=1)
        recs.append(rec)
        r_meas, r_lad, r_ens = (rec["ratio_mse"], rec["pred_ladder_ratio_mse"],
                                rec["pred_ensemble_ratio_mse"])
        if r_meas:
            lad = f"{r_lad:.4g} ({r_lad/r_meas:.4f}x)" if r_lad else "skipped"
            print(f"F={F_} N={N_} K={K}: measured {r_meas:.4g}  "
                  f"ladder {lad}  ensemble {r_ens:.4g} ({r_ens/r_meas:.3f}x)",
                  flush=True)
    from biregular.fn_sweep_figures import exact_solo_theory_figure
    exact_solo_theory_figure(recs, FIG_DIR)


# --------------------------------------------------- summary + classification

def load_cell(F_, N_):
    kp = kfit_path(F_, N_)
    if not os.path.exists(kp):
        return None
    with open(kp) as f:
        kf = json.load(f)
    K = kf["K_star"]
    totals = {}
    for rb, wb in itertools.product(BASES, BASES):
        op = oracle_path(F_, N_, K, rb, wb)
        if not os.path.exists(op):
            return None
        with open(op) as f:
            totals[(rb, wb)] = json.load(f)["losses"]["total"]
    return kf, totals


def classify_matrix(totals, matrix):
    """Preference for one matrix: compare its two bases at BOTH settings of the
    other matrix. Consistent + margin > TIE_RATIO in at least one column ->
    winner; margins under TIE_RATIO everywhere -> tie; sign flip -> interaction.
    Returns (verdict, margin) with margin = max losing/winning ratio."""
    if matrix == "w_in":
        ratios = [totals[("neuron", wb)] / totals[("feature", wb)] for wb in BASES]
    else:
        ratios = [totals[(rb, "neuron")] / totals[(rb, "feature")] for rb in BASES]
    # ratio > 1 -> feature basis wins that column
    if all(r < TIE_RATIO and 1 / r < TIE_RATIO for r in ratios):
        return "tie", max(max(r, 1 / r) for r in ratios)
    feat_wins = [r > 1 for r in ratios]
    if all(feat_wins):
        return "feature", max(ratios)
    if not any(feat_wins):
        return "neuron", max(1 / r for r in ratios)
    return "interaction", max(max(r, 1 / r) for r in ratios)


def write_kstar_table():
    """Pivot table of the L4-optimal fanning K* per (F, N) cell, from the kfit
    records alone (fills in ahead of the oracle combos). Also written per-cell
    with (a, b) and l4* in kstar_table.md."""
    kf = {}
    for F_, N_ in cells():
        p = kfit_path(F_, N_)
        if os.path.exists(p):
            with open(p) as f:
                kf[(F_, N_)] = json.load(f)
    path = os.path.join(RES_FAM_DIR, "kstar_table.md")
    with open(path, "w") as f:
        f.write(f"# Optimal fanning K* per (F, N)\n\np = {P}, decoder = {DECODER}, "
                f"code seed {CODE_SEED}; K* = argmin held-out L4 over the K grid "
                f"({len(kf)}/{sum(1 for _ in cells())} cells fitted).\n\n")
        f.write("## K* pivot (rows = N, columns = F)\n\n")
        f.write("| N \\ F | " + " | ".join(str(F_) for F_ in F_GRID) + " |\n")
        f.write("|" + "---|" * (len(F_GRID) + 1) + "\n")
        for N_ in N_GRID:
            vals = [str(kf[(F_, N_)]["K_star"]) if (F_, N_) in kf else "."
                    for F_ in F_GRID]
            f.write(f"| **{N_}** | " + " | ".join(vals) + " |\n")
        f.write("\n## Per-cell detail\n\n| F | N | F/N | K* | a | b | l4* "
                "| computed |\n|" + "---|" * 8 + "\n")
        for (F_, N_), r in sorted(kf.items()):
            f.write(f"| {F_} | {N_} | {F_/N_:.2f} | {r['K_star']} | {r['a']:.4f} "
                    f"| {r['b']:.4f} | {r['l4_star']:.2e} "
                    f"| {r['computed_star']}/{F_} |\n")
    print(f"kstar table ({len(kf)} cells) -> {path}")
    return kf


def write_summary():
    write_kstar_table()
    rows = []
    for F_, N_ in cells():
        got = load_cell(F_, N_)
        if got is None:
            continue
        kf, totals = got
        v_in, m_in = classify_matrix(totals, "w_in")
        v_out, m_out = classify_matrix(totals, "w_out")
        rows.append({"F": F_, "N": N_, "K_star": kf["K_star"], "a": kf["a"],
                     "b": kf["b"], "l4_star": kf["l4_star"],
                     "computed": kf["computed_star"],
                     "total_ff": totals[("feature", "feature")],
                     "total_fn": totals[("feature", "neuron")],
                     "total_nf": totals[("neuron", "feature")],
                     "total_nn": totals[("neuron", "neuron")],
                     "w_in_pref": v_in, "w_in_margin": m_in,
                     "w_out_pref": v_out, "w_out_margin": m_out})
    if not rows:
        print("summary: no complete cells yet")
        return rows
    keys = list(rows[0].keys())
    with open(os.path.join(RES_FAM_DIR, "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float)
                             else str(r[k]) for k in keys) + "\n")
    with open(os.path.join(RES_FAM_DIR, "index.md"), "w") as f:
        f.write("# fn_sweep: per-cell K* and oracle basis preference\n\n"
                f"p = {P}, decoder = {DECODER}, code seed {CODE_SEED}; totals are "
                "the weighted VPD objective (stoch+PGD recon + importance at "
                "imp 1e-4 / pnorm 2.0) with oracle gates; combo key = "
                "(read basis, write basis). Note: in K* = 1, b = 0 cells the "
                "nonzero weight pieces of the two bases coincide and recon is 0 "
                "for all combos -- the margin there is carried entirely by gate "
                "sharpness (h != 0 is sign-aware, x != 0 is not), i.e. by the "
                "indexing, not the piece shapes.\n\n")
        f.write("| F | N | K* | l4* | computed | (f,f) | (f,n) | (n,f) | (n,n) "
                "| W_in pref (margin) | W_out pref (margin) |\n")
        f.write("|" + "---|" * 11 + "\n")
        for r in rows:
            f.write(f"| {r['F']} | {r['N']} | {r['K_star']} | {r['l4_star']:.2e} "
                    f"| {r['computed']} | {r['total_ff']:.3e} | {r['total_fn']:.3e} "
                    f"| {r['total_nf']:.3e} | {r['total_nn']:.3e} "
                    f"| {r['w_in_pref']} ({r['w_in_margin']:.2f}x) "
                    f"| {r['w_out_pref']} ({r['w_out_margin']:.2f}x) |\n")
    print(f"summary: {len(rows)} complete cells -> summary.csv + index.md")
    return rows


# ------------------------------------------------------------------- figures

def make_figures():
    from biregular.fn_sweep_figures import (basis_preference_map,
                                            basis_preference_scatter,
                                            kstar_heatmap, loss_2x2_examples)
    fg, ng = grid_axes()
    # K* heatmap needs only the kfit records (renders before oracle data lands)
    kf_rows = []
    for F_, N_ in cells():
        if os.path.exists(kfit_path(F_, N_)):
            with open(kfit_path(F_, N_)) as f:
                kf = json.load(f)
            kf_rows.append({"F": F_, "N": N_, "K_star": kf["K_star"],
                            "l4_star": kf["l4_star"]})
    if kf_rows:
        kstar_heatmap(kf_rows, fg, ng, FIG_DIR)
    rows = write_summary()
    if not rows:
        return
    basis_preference_map(rows, fg, ng, FIG_DIR, TIE_RATIO)
    basis_preference_scatter(rows, FIG_DIR, TIE_RATIO)
    loss_2x2_examples(rows, ORACLE_DIR, FIG_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-grid", action="store_true")
    ap.add_argument("--oracle-grid", action="store_true")
    ap.add_argument("--exact-solo", action="store_true")
    ap.add_argument("--exact-solo-theory", action="store_true")
    ap.add_argument("--theory-curves", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--max-f", type=int, default=None)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--device", default="cpu",
                    help="device for the (a,b) fits + held-out evals")
    ap.add_argument("--xl", action="store_true",
                    help="use the XL grids (up to 10,000 in F and N)")
    ap.add_argument("--family", choices=FAMILIES, default="greedy",
                    help="code construction (per-family result subfolders)")
    args = ap.parse_args()
    set_family(args.family)
    if args.xl:
        global F_GRID, N_GRID
        F_GRID, N_GRID = F_GRID_XL, N_GRID_XL
    if args.fit_grid:
        run_fit_grid(args.max_f, args.max_n, device=args.device)
    if args.oracle_grid:
        run_oracle_grid(args.max_f, args.max_n, device=args.device)
    if args.exact_solo:
        run_exact_solo(args.max_f, args.max_n, device=args.device)
    if args.exact_solo_theory:
        run_exact_solo_theory()
    if args.theory_curves:
        from biregular.fn_sweep_figures import exact_solo_theory_curves
        recs = []
        for F_, N_ in cells():
            pth = os.path.join(EXACT_SOLO_DIR, f"F{F_}_N{N_}.json")
            if not os.path.exists(pth):
                continue
            with open(pth) as f:
                rec = json.load(f)
            with open(kfit_path(F_, N_)) as f:   # per-K learned (a, b)
                rec["_krows"] = json.load(f)["rows"]
            recs.append(rec)
        exact_solo_theory_curves(recs, FIG_DIR)
    if args.summary and not args.figures:
        write_summary()
    if args.figures:
        make_figures()


if __name__ == "__main__":
    main()
