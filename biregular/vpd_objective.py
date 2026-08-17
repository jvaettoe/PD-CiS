"""Experiment 1 -- does the truth cost more? Score ExactPD under the ported VPD objective.

Builds a VPD state whose C=100 subcomponents are EXACTLY the ExactPD per-feature pieces
(vpd.py parametrization is V [d0,C], U [C,d1], piece c = outer(V[:,c], U[c,:]), so
    w_in  piece j = e_j (x) w_in[j,:]   ->  V_in  = I,      U_in  = w_in
    w_out piece j = w_out[:,j] (x) e_j  ->  V_out = w_out,  U_out = I
and the weight delta is 0 by construction), then evaluates three variants on ONE fixed
eval pool (CleanDataset p=0.001, seed 777, 25 x 8192) with the port's default term
weights (decomp_fit.loss_kwargs_for("VPD")):

  exact_oracle      -- exact pieces + ORACLE gates: ci_j = [x_j != 0] on both matrices
                       (component j == feature j; the CI net is bypassed analytically)
  exact_trainedci   -- exact pieces FROZEN + CI nets trained (--fit-ci; deferred, long)
  learned_reference -- the saved 25k-step baseline VPD fit re-evaluated on the pool

Storage is append-only: results/vpd_objective/runs/<tag>.json per run, component
tensors at checkpoints/vpd_objective/<tag>.pt, and --index regenerates
results/vpd_objective/index.md from all run JSONs.

  ./.venv/bin/python -m biregular.vpd_objective              # oracle + learned evals
  ./.venv/bin/python -m biregular.vpd_objective --fit-ci     # deferred: train CI nets
  ./.venv/bin/python -m biregular.vpd_objective --index      # rebuild index.md only
"""
import argparse
import glob
import json
import os
from datetime import datetime

import torch

import core.decomp_fit as decomp_fit
from biregular.biregular_ansatz import CodeAnsatz
from biregular.biregular_vpd_compare import DEFAULT_KW, FIT_SEED, _decomp_path
from biregular.component_match import (activation_matrices, hungarian_match, mass_matrices,
                                       neuron_basis_stats, score_sides)
from core.data import CleanDataset
from core.paths import CHECKPOINTS, RESULTS
from core.pd import ExactPD
from core.vpd import VPD

F = 100
P_DEFAULT = "0.001"
EVAL_SEED = 777
EVAL_BATCHES, EVAL_BATCH = 25, 8192
INIT_SEED = 0            # CI-net init (V/U are overwritten, so it only matters for --fit-ci)
CI_STEPS_DEFAULT = 4000  # CPU screening budget (cf. biregular_vpd_compare --steps)

RES_DIR = os.path.join(RESULTS, "vpd_objective")
RUNS_DIR = os.path.join(RES_DIR, "runs")
INDEX_PATH = os.path.join(RES_DIR, "index.md")
CKPT_DIR = os.path.join(CHECKPOINTS, "vpd_objective")

TERM_WEIGHT_KEY = {"importance": "alpha_importance", "stoch_recon": "alpha_stoch",
                   "pgd_recon": "alpha_pgd", "faithful": "alpha_faithfulness"}


class OracleVPD(VPD):
    """VPD with the CI net bypassed: importance_j = 1 iff feature j is active
    (x_j != 0) in the sample, identically on both matrices. Valid only for the
    exact per-feature pieces (component j == feature j, C == n_features) on a
    plain MLP (identity embedding, so e == x). lower == upper == the oracle gate."""

    def _ci(self, e, h):
        assert self.C == self.model.n_features, "oracle gates need component j == feature j"
        g = (e != 0).float()
        return {n: g for n in self.param_names}, {n: g for n in self.param_names}


def build_exact_vpd(model, oracle: bool = False, init_seed: int = INIT_SEED) -> VPD:
    """VPD state whose C=F subcomponents ARE the ExactPD pieces (frozen, delta == 0).
    Asserts the pieces tile each matrix exactly and match ExactPD's own components."""
    torch.manual_seed(init_seed)  # reproducible CI-net init (V/U overwritten below)
    cls = OracleVPD if oracle else VPD
    pd_v = cls(model, C=model.n_features, warmup_steps=0)
    eye = torch.eye(model.n_features, device=model.device)
    with torch.no_grad():
        pd_v.V["w_in"].copy_(eye)                    # (F, C):  V_in[:, j] = e_j
        pd_v.U["w_in"].copy_(model.w_in)             # (C, N):  U_in[j, :] = w_in[j, :]
        pd_v.V["w_out"].copy_(model.w_out)           # (N, C):  V_out[:, j] = w_out[:, j]
        pd_v.U["w_out"].copy_(eye)                   # (C, F):  U_out[j, :] = e_j
    for k in pd_v.V:
        pd_v.V[k].requires_grad_(False)
        pd_v.U[k].requires_grad_(False)
    for name in pd_v.param_names:                    # exact tiling: sum of pieces == matrix
        err = float((pd_v._vu(name) - pd_v.targets[name]).abs().max())
        assert err < 1e-6, f"exact tiling failed for {name}: max|sum - W| = {err:.3e}"
    pd_e = ExactPD(model)                            # cross-check against the pd framework
    for j in (0, 17, 99):
        for name in pd_v.param_names:
            assert torch.allclose(pd_v._component_weight(name, j),
                                  pd_e._component_weight(name, j), atol=1e-6), (name, j)
    return pd_v


def load_learned(code: CodeAnsatz, p: str, device: str) -> tuple:
    """(model, pd_v, payload) for the baseline 25k-step fit (the checkpoint that
    biregular_vpd_compare --reuse and cluster_vpd load by default: knob tag '')."""
    path = _decomp_path(code.name, p, "")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = code.model(tuple(payload["biregular"]["ab"]), device=device)
    pd_v = VPD(model, payload["C"], warmup_steps=0)
    pd_v.load_state_dict(payload["pd_state"], strict=False)
    return model, pd_v, payload


# ---------------------------------------------------------------- evaluation

def eval_objective(pd_v: VPD, *, p: float, seed: int = EVAL_SEED,
                   n_batches: int = EVAL_BATCHES, batch: int = EVAL_BATCH,
                   loss_kw: dict = DEFAULT_KW) -> dict:
    """Mean per-term losses over the fixed pool. The dataset owns its generator, so
    the pool is identical across variants; the stoch/PGD mask draws are seeded
    per batch via fork_rng, so those are identical across variants too. Reports the
    port's three method terms + faithfulness (raw and weighted at the port's default
    coefficients) plus per-matrix breakdowns: importance per matrix, and stochastic
    masked recon with the subset routing FORCED to a single matrix (the other runs
    the frozen target weights)."""
    model = pd_v.model
    pd_v.requires_grad_(False)  # eval-only; PGD's adversary is an independent leaf
    ds = CleanDataset(n_features=model.n_features, p=float(p), device=model.device,
                      seed=seed)
    pnorm, eps = loss_kw["pnorm"], 1e-12
    acc: dict[str, float] = {}

    def add(key, val):
        acc[key] = acc.get(key, 0.0) + float(val)

    for i in range(n_batches):
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            target = model(x)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 100003 + i)
            terms = pd_v.method_loss_terms(
                x, target, alpha_importance=loss_kw["alpha_importance"],
                alpha_stoch=loss_kw["alpha_stoch"], alpha_pgd=loss_kw["alpha_pgd"],
                pnorm=pnorm)
            terms["faithful"] = (pd_v.faithfulness_loss(), loss_kw["alpha_faithfulness"])
            _, logs = pd_v._total_and_logs(terms)
            for k, v in logs.items():
                add(k, v)
            with torch.no_grad():  # per-matrix breakdowns (fresh seeded mask draws)
                e = model.embed_input(x)
                h = torch.relu(e @ pd_v.targets["w_in"])
                lower, upper = pd_v._ci(e, h)
                B = x.shape[0]
                for name in pd_v.param_names:
                    add(f"importance_{name}",
                        ((upper[name] + eps) ** pnorm).mean(dim=0).sum())
                    masks = {n: ci + (1 - ci) * torch.rand_like(ci)
                             for n, ci in lower.items()}
                    dmasks = {n: torch.rand(B, device=x.device) for n in lower}
                    routing = {n: (torch.ones(B, device=x.device) if n == name
                                   else torch.zeros(B, device=x.device)) for n in lower}
                    out = pd_v._forward_masked_vpd(x, masks, dmasks, routing)
                    add(f"stoch_recon_{name}", ((out - target) ** 2).mean())

    means = {k: v / n_batches for k, v in acc.items()}
    res = {"total": means.pop("total"),
           "terms": {}, "per_matrix": {"importance": {}, "stoch_recon": {}}}
    for term, wkey in TERM_WEIGHT_KEY.items():
        res["terms"][term] = {"raw": means[term],
                              "weighted": means[term] * loss_kw[wkey]}
    for name in pd_v.param_names:
        res["per_matrix"]["importance"][name] = means[f"importance_{name}"]
        res["per_matrix"]["stoch_recon"][name] = means[f"stoch_recon_{name}"]
    return res


def basis_metrics(pd_v: VPD, p: float) -> dict:
    """Reused component_match scoring: per-side activation/mass purity under the
    Hungarian activation match, neuron-basis stats, and VPD-native ci_stats."""
    mats = activation_matrices(pd_v)
    mats.update(mass_matrices(pd_v))
    match_in = hungarian_match(mats["act_in"])
    match_out = hungarian_match(mats["act_out"])
    s = score_sides(mats, match_in, match_out, pd_v.model.n_features)
    return {"in": s["in"], "out": s["out"], "index_agreement": s["index_agreement"],
            "dead_components": s["dead_components"],
            "neuron_basis": neuron_basis_stats(pd_v),
            "ci_stats": pd_v.ci_stats(float(p))}


# ---------------------------------------------------------------- storage

def save_run(tag: str, record: dict, pd_v: VPD) -> None:
    """Append-only: one JSON per run tag + the component/CI tensors as a .pt payload
    (pd_state is load_state_dict(strict=False)-compatible, model excluded)."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
    state = {k: v.cpu() for k, v in pd_v.state_dict().items() if not k.startswith("model.")}
    torch.save({"tag": tag, "method": type(pd_v).__name__, "C": pd_v.C,
                "pd_state": state, "config": record["config"]}, ckpt)
    record["checkpoint"] = os.path.relpath(ckpt, RESULTS + "/..")
    with open(os.path.join(RUNS_DIR, f"{tag}.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"  saved {os.path.join(RUNS_DIR, tag + '.json')}  +  {ckpt}")
    write_index()


def write_index() -> None:
    """Regenerate results/vpd_objective/index.md as one table over all run JSONs."""
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.json")))
    lines = ["# VPD-objective runs (Experiment 1: does the truth cost more?)", "",
             "Fixed eval pool: CleanDataset p, seed/batches/batch as per each run's "
             "config; term weights = the port's defaults. Raw term values; `total` "
             "is the weighted objective.", "",
             "| tag | timestamp | steps | total | stoch | pgd | imp | faithful "
             "| stoch w_in | stoch w_out | imp w_in | imp w_out "
             "| act_in | act_out | ci_l0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        lo, pm, bm = r["losses"], r["losses"]["per_matrix"], r.get("basis_metrics", {})
        t = lo["terms"]
        ci_l0 = bm.get("ci_stats", {}).get("headline", {}).get("ci_l0", float("nan"))
        lines.append(
            f"| {r['tag']} | {r['timestamp']} | {r['steps']} | {lo['total']:.3e} "
            f"| {t['stoch_recon']['raw']:.3e} | {t['pgd_recon']['raw']:.3e} "
            f"| {t['importance']['raw']:.3e} | {t['faithful']['raw']:.3e} "
            f"| {pm['stoch_recon']['w_in']:.3e} | {pm['stoch_recon']['w_out']:.3e} "
            f"| {pm['importance']['w_in']:.3e} | {pm['importance']['w_out']:.3e} "
            f"| {bm.get('in', {}).get('act_purity', float('nan')):.2f} "
            f"| {bm.get('out', {}).get('act_purity', float('nan')):.2f} "
            f"| {ci_l0:.2f} |")
    os.makedirs(RES_DIR, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  index ({len(paths)} runs) -> {INDEX_PATH}")


def print_comparison(records: list[dict]) -> None:
    """Compact per-term side-by-side table for the runs just produced."""
    tags = [r["tag"] for r in records]
    rows = [("total (weighted)", lambda lo: lo["total"])]
    rows += [(f"{t} raw", lambda lo, t=t: lo["terms"][t]["raw"])
             for t in ("stoch_recon", "pgd_recon", "importance", "faithful")]
    rows += [(f"stoch {n}", lambda lo, n=n: lo["per_matrix"]["stoch_recon"][n])
             for n in ("w_in", "w_out")]
    rows += [(f"imp {n}", lambda lo, n=n: lo["per_matrix"]["importance"][n])
             for n in ("w_in", "w_out")]
    w = max(len(name) for name, _ in rows)
    print("\n" + " " * (w + 2) + "  ".join(f"{t:>18}" for t in tags))
    for name, fn in rows:
        print(f"  {name:<{w}}" + "  ".join(f"{fn(r['losses']):>18.4e}" for r in records))


# ---------------------------------------------------------------- runners

def _base_config(code: CodeAnsatz, ab, p: str, a) -> dict:
    return {"K": code.K, "family": code.family, "code_seed": code.seed,
            "swaps": code.swaps, "decoder": code.decoder, "p": p,
            "ab": [float(v) for v in ab], "C": F, "loss_kwargs": DEFAULT_KW,
            "eval_pool": {"seed": a.seed, "n_batches": a.batches, "batch": a.batch},
            "init_seed": INIT_SEED}


def run_variant(tag: str, pd_v: VPD, code: CodeAnsatz, ab, p: str, a,
                steps: int, extra_cfg: dict | None = None) -> dict:
    losses = eval_objective(pd_v, p=float(p), seed=a.seed, n_batches=a.batches,
                            batch=a.batch)
    record = {"tag": tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "config": _base_config(code, ab, p, a) | (extra_cfg or {}),
              "steps": steps, "losses": losses,
              "basis_metrics": basis_metrics(pd_v, float(p))}
    save_run(tag, record, pd_v)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=EVAL_SEED, help="eval-pool seed")
    ap.add_argument("--batches", type=int, default=EVAL_BATCHES)
    ap.add_argument("--batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--fit-ci", action="store_true", dest="fit_ci",
                    help="DEFERRED variant (b): freeze exact components, train the CI "
                         "nets with the port's optimizer, then eval -> exact_trainedci")
    ap.add_argument("--ci-steps", type=int, default=CI_STEPS_DEFAULT, dest="ci_steps")
    ap.add_argument("--index", action="store_true", help="rebuild index.md only")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.index:
        write_index()
        return

    code = CodeAnsatz(5, family="regular", seed=0, swaps=0, decoder="transpose")
    # (a, b) from the saved baseline payload when present (guarantees the SAME frozen
    # model the learned VPD was fit to, at zero refit cost; fit_ab is deterministic
    # and produced that ab in the first place). Fresh fit_ab only as a fallback.
    learned_path = _decomp_path(code.name, a.p, "")
    if os.path.exists(learned_path):
        payload = torch.load(learned_path, map_location=a.device, weights_only=False)
        ab, ab_source = tuple(payload["biregular"]["ab"]), "learned_payload"
    else:
        payload = None
        ab, ab_source = code.fit_ab(float(a.p), device=a.device), "fit_ab"
    model = code.model(ab, device=a.device)
    print(f"[{code.name}] p={a.p} ab=({ab[0]:.4f},{ab[1]:.4f}) [{ab_source}] "
          f"pool: seed={a.seed} {a.batches}x{a.batch}")

    if a.fit_ci:  # variant (b): exact components frozen, CI nets trained
        pd_v = build_exact_vpd(model)
        n_train = sum(p_.numel() for p_ in pd_v.parameters() if p_.requires_grad)
        print(f"  exact_trainedci: training CI nets only ({n_train} params, "
              f"{a.ci_steps} steps, lr {decomp_fit.METHOD_LR['VPD']})")
        decomp_fit.fit_decomposition(
            pd_v, CleanDataset(n_features=F, p=float(a.p), device=a.device,
                               seed=FIT_SEED),
            steps=a.ci_steps, loss_kwargs=DEFAULT_KW, seed=FIT_SEED,
            desc=f"exact_trainedci p={a.p}")
        rec = run_variant("exact_trainedci", pd_v, code, ab, a.p, a, steps=a.ci_steps,
                          extra_cfg={"ab_source": ab_source, "components": "exact_frozen",
                                     "gates": "trained_ci", "ci_fit_seed": FIT_SEED})
        print_comparison([rec])
        return

    # variant (a): exact components + oracle gates (eval only)
    pd_o = build_exact_vpd(model, oracle=True)
    rec_a = run_variant("exact_oracle", pd_o, code, ab, a.p, a, steps=0,
                        extra_cfg={"ab_source": ab_source, "components": "exact_frozen",
                                   "gates": "oracle_activity"})

    # variant (c): the converged learned fit, re-evaluated on the identical pool
    assert payload is not None, f"no saved learned VPD fit at {learned_path}"
    _, pd_l, payload = load_learned(code, a.p, a.device)
    rec_c = run_variant("learned_reference", pd_l, code, ab, a.p, a,
                        steps=int(payload["biregular"]["steps"]),
                        extra_cfg={"ab_source": ab_source, "components": "learned",
                                   "gates": "learned_ci",
                                   "source_checkpoint": os.path.relpath(
                                       learned_path, RESULTS + "/..")})
    print_comparison([rec_a, rec_c])


if __name__ == "__main__":
    main()
