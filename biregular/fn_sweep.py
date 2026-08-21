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
K_COARSE_POINTS = 12   # log-spaced over the FULL range [1, N], endpoints included
K_REFINE_ROUNDS = 3    # geometric-midpoint refinement around the argmin
P = 0.001
CODE_SEED = 0
DECODER = "transpose"
BASES = ("feature", "neuron")
TIE_RATIO = 1.05
HELD_OUT_BATCHES, HELD_OUT_BATCH = 30, 8192

RES_DIR = os.path.join(RESULTS, "fn_sweep")
KFIT_DIR = os.path.join(RES_DIR, "kfit")
ORACLE_DIR = os.path.join(RES_DIR, "oracle")
FIG_DIR = os.path.join(FIGURES, "fn_sweep")
for _d in (KFIT_DIR, ORACLE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def cells(max_f=None, max_n=None):
    for F_, N_ in itertools.product(F_GRID, N_GRID):
        if (max_f is None or F_ <= max_f) and (max_n is None or N_ <= max_n):
            yield F_, N_


# ------------------------------------------------------------- stage 1: K*

def fit_budget(F_):
    """Scale-adaptive (steps, batch, pool) for fit_ab_on_code: the 2-parameter
    fit needs no more; large-F cells dominate wall-clock and pool memory."""
    if F_ >= 500:
        return 3000, 2048, 65536
    return 4000, 4096, 131072


def held_out_l4(model, p=P, batches=HELD_OUT_BATCHES, batch=HELD_OUT_BATCH,
                seed=EVAL_SEED):
    """Multi-batch held-out L4 + features-computed count (per-feature
    active-conditional MSE < 1/30; always-0 baseline is 1/3)."""
    F_ = model.n_features
    ds = CleanDataset(n_features=F_, p=p, device="cpu", seed=seed)
    l4 = 0.0
    err, cnt = torch.zeros(F_), torch.zeros(F_)
    with torch.no_grad():
        for _ in range(batches):
            x, y = ds.generate_batch(batch)
            out = model(x)
            l4 += float(((out - y).abs() ** 4).mean())
            a = (x > 0).float()
            err += (((out - y) ** 2) * a).sum(0)
            cnt += a.sum(0)
    pf = (err / cnt.clamp(min=1)).numpy()
    return l4 / batches, int((pf < 1 / 30).sum())


def kfit_path(F_, N_):
    return os.path.join(KFIT_DIR, f"F{F_}_N{N_}.json")


def k_candidates(N_, points=K_COARSE_POINTS):
    """Log-spaced fannings over the FULL admissible range [1, N] (K is bounded
    by the neuron count, not by any small-integer convention); endpoints always
    included."""
    return sorted(set(int(round(g)) for g in np.geomspace(1, N_, num=points)))


def fit_cell(F_, N_):
    """K sweep for one cell: coarse log-spaced pass over [1, N], then geometric
    midpoint refinement around the argmin; per K: greedy code -> (a, b) L4 fit
    -> held-out L4. K* = final argmin."""
    steps, batch, pool = fit_budget(F_)
    rows, tried = [], set()

    def run_k(K):
        M = torch.tensor(greedy_regular_code(F_, N_, K, CODE_SEED))
        a, b = fit_ab_on_code(M, DECODER, p=P, steps=steps, batch=batch, pool=pool)
        model = model_from_code(M, a, b, DECODER)
        l4, computed = held_out_l4(model)
        rows.append({"K": K, "a": a, "b": b, "l4": l4, "computed": computed})
        tried.add(K)
        print(f"  F={F_} N={N_} K={K}: l4={l4:.3e} computed={computed} "
              f"a={a:.4f} b={b:.4f}", flush=True)

    blown = 0
    for K in k_candidates(N_):  # ascending; stop once the tail is clearly blown up
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
        new = [int(round(math.sqrt(best_k * ts[j])))
               for j in (i - 1, i + 1) if 0 <= j < len(ts)]
        new = [k for k in set(new) if k not in tried and 1 <= k <= N_]
        if not new:
            break
        for K in sorted(new):
            run_k(K)

    best = min(rows, key=lambda r: r["l4"])
    rec = {"F": F_, "N": N_, "p": P, "decoder": DECODER, "code_seed": CODE_SEED,
           "fit": {"steps": steps, "batch": batch, "pool": pool,
                   "k_rule": f"geomspace[1,N]x{K_COARSE_POINTS}"
                             f"+refine{K_REFINE_ROUNDS}"},
           "held_out": {"batches": HELD_OUT_BATCHES, "batch": HELD_OUT_BATCH,
                        "seed": EVAL_SEED},
           "rows": rows, "K_star": best["K"], "a": best["a"], "b": best["b"],
           "l4_star": best["l4"], "computed_star": best["computed"]}
    with open(kfit_path(F_, N_), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def run_fit_grid(max_f, max_n):
    todo = [(F_, N_) for F_, N_ in cells(max_f, max_n)
            if not os.path.exists(kfit_path(F_, N_))]
    print(f"fit-grid: {len(todo)} cells to fit "
          f"(of {sum(1 for _ in cells(max_f, max_n))})", flush=True)
    for F_, N_ in sorted(todo, key=lambda c: c[0] * c[1]):  # cheap cells first
        print(f"[cell F={F_} N={N_}]", flush=True)
        rec = fit_cell(F_, N_)
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


def run_oracle_grid(max_f, max_n):
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
        M = torch.tensor(greedy_regular_code(F_, N_, K, CODE_SEED))
        model = model_from_code(M, a, b, DECODER)
        for rb, wb in combos:
            pd_v = build_combo_vpd(model, rb, wb)
            res = eval_objective(pd_v, p=P)
            rec = {"F": F_, "N": N_, "K": K, "a": a, "b": b, "p": P,
                   "decoder": DECODER, "code_seed": CODE_SEED,
                   "read_basis": rb, "write_basis": wb,
                   "C": pd_v.C, "loss_kw": dict(DEFAULT_KW),
                   "eval": {"seed": EVAL_SEED}, "losses": res}
            with open(oracle_path(F_, N_, K, rb, wb), "w") as f:
                json.dump(rec, f, indent=1)
            print(f"F={F_} N={N_} K={K} ({rb},{wb}): total={res['total']:.4e}",
                  flush=True)


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
    path = os.path.join(RES_DIR, "kstar_table.md")
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
    with open(os.path.join(RES_DIR, "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float)
                             else str(r[k]) for k in keys) + "\n")
    with open(os.path.join(RES_DIR, "index.md"), "w") as f:
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
    rows = write_summary()
    if not rows:
        return
    from biregular.fn_sweep_figures import (basis_preference_map,
                                            basis_preference_scatter,
                                            kstar_heatmap, loss_2x2_examples)
    kstar_heatmap(rows, F_GRID, N_GRID, FIG_DIR)
    basis_preference_map(rows, F_GRID, N_GRID, FIG_DIR, TIE_RATIO)
    basis_preference_scatter(rows, FIG_DIR, TIE_RATIO)
    loss_2x2_examples(rows, ORACLE_DIR, FIG_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-grid", action="store_true")
    ap.add_argument("--oracle-grid", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--max-f", type=int, default=None)
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()
    if args.fit_grid:
        run_fit_grid(args.max_f, args.max_n)
    if args.oracle_grid:
        run_oracle_grid(args.max_f, args.max_n)
    if args.summary and not args.figures:
        write_summary()
    if args.figures:
        make_figures()


if __name__ == "__main__":
    main()
