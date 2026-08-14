"""Goodfire's MDL subcomponent-clustering, ported and run on our biregular VPD fit.

Port of param-decomp/param_decomp/clustering/ (VPD paper appendix "Clustering
Subcomponents into Components"): binarize per-subcomponent causal importances at
tau=0.01, build the co-activation matrix s_ij = sum_b g_i g_j, then agglomeratively
merge the pair with the lowest MDL merge cost

    dL(i,j) = (s_tot - s_i - s_j) log2((k-1)/k)
              + s_{i OR j} log2(k-1) - s_i log2(k) - s_j log2(k)
              + alpha (s_{i OR j} (r_i + r_j) - s_i r_i - s_j r_j)

(their compute_costs.compute_merge_costs), sampling among near-minimal pairs
(range_sampler, threshold 0.05), until the chosen dL turns positive. The result is
a HARD PARTITION: each subcomponent belongs to exactly one component.

The scientific point on the biregular model: ExactPD's 100 feature mechanisms
OVERLAP in the 50 per-neuron write shards (each serves d=10 features), which no
partition can represent. Expected outcomes: SPD's resolution (per-feature reads +
one dense write blob), one global blob, or arbitrary cuts -- but never the 100
chain-glued mechanisms. The overlapping alternative is component_match.circuit_mask.

  ./.venv/bin/python cluster_vpd.py                    # baseline fit, alpha sweep
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from biregular_ansatz import CodeAnsatz
from biregular_vpd_compare import _decomp_path
from component_match import _write_shard_owners, activation_matrices, hungarian_match
from data import CleanDataset
from vpd import VPD

RES_DIR = os.path.join(os.path.dirname(__file__), "results", "overview")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "comparison")
TAU = 0.01          # their clustering-appendix binarization threshold
RANGE_THR = 0.05    # their range_sampler threshold


def load_fit(p: str, decoder: str, tag: str, device: str = "cpu"):
    code = CodeAnsatz(5, swaps=0, decoder=decoder)
    path = _decomp_path(code.name, p, tag)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = code.model(tuple(payload["biregular"]["ab"]), device=device)
    pd_v = VPD(model, payload["C"], warmup_steps=0)
    pd_v.load_state_dict(payload["pd_state"], strict=False)
    return code, model, pd_v


def harvest_binarized_cis(pd_v, p: float, total: int = 50000, batch: int = 8192,
                          seed: int = 17, device: str = "cpu", tau: float = TAU):
    """[samples, 2C] boolean CI activity (their tau=0.01 binarization), plus the
    per-sample feature-activity matrix for sanity checks."""
    model = pd_v.model
    F = model.n_features
    ds = CleanDataset(n_features=F, p=p, device=device, seed=seed)
    acts, feats = [], []
    done = 0
    with torch.no_grad():
        while done < total:
            b = min(batch, total - done)
            x, _ = ds.generate_batch(b)
            e = model.embed_input(x)
            h = torch.relu(e @ pd_v.targets["w_in"])
            lower, _ = pd_v._ci(e, h)
            acts.append(torch.cat([lower["w_in"] > tau, lower["w_out"] > tau], dim=1))
            feats.append(x > 0)
            done += b
    return torch.cat(acts).bool(), torch.cat(feats).bool()


def merge_costs(coact: torch.Tensor, ranks: torch.Tensor, alpha: float) -> torch.Tensor:
    """Their compute_merge_costs (bits; negative = merging i,j lowers the MDL)."""
    k = coact.shape[0]
    s = torch.diag(coact)
    coact_or = s.view(-1, 1) + s.view(1, -1) - coact          # s_{i OR j}
    rank_sum = ranks.view(-1, 1) + ranks.view(1, -1)
    s_other = (s.sum() - s.view(-1, 1) - s.view(1, -1)) * math.log2((k - 1) / k)
    bits_local = (coact_or * math.log2(k - 1)
                  - s.view(-1, 1) * math.log2(k) - s.view(1, -1) * math.log2(k))
    si_ri = s * ranks
    penalty = coact_or * rank_sum - si_ri.view(-1, 1) - si_ri.view(1, -1)
    return s_other + bits_local + alpha * penalty


def mdl_total(coact: torch.Tensor, ranks: torch.Tensor, alpha: float) -> float:
    """Their compute_mdl_cost: sum_i s_i (log2 k + alpha r_i)."""
    k = coact.shape[0]
    s = torch.diag(coact)
    return float((s * (math.log2(k) + alpha * ranks)).sum())


def _self_test():
    """Hand-checkable 3-group case: perfect co-activation pair (0,1) vs independent 2."""
    A = torch.zeros(10, 3, dtype=torch.bool)
    A[:4, 0] = A[:4, 1] = True          # 0 and 1 always co-fire
    A[4:7, 2] = True
    coact = (A.float().T @ A.float()).double()
    ranks = torch.ones(3, dtype=torch.float64)
    dl = merge_costs(coact, ranks, alpha=0.1)
    # merging the perfectly co-active pair must be the cheapest and MDL-improving
    dl.fill_diagonal_(float("inf"))
    assert dl.argmin() in (1, 3), dl        # flat index of (0,1)/(1,0)
    assert dl[0, 1] < 0, dl[0, 1]
    # manual check of the (0,1) entry: s_i=s_j=s_or=4, s_tot=11, k=3, ranks 1+1
    exp = ((11 - 4 - 4) * math.log2(2 / 3) + 4 * math.log2(2) - 8 * math.log2(3)
           + 0.1 * (4 * 2 - 4 - 4))
    assert abs(float(dl[0, 1]) - exp) < 1e-9, (float(dl[0, 1]), exp)


def cluster(act: torch.Tensor, alpha: float, seed: int = 0, k_floor: int = 2):
    """Greedy agglomerative merge with their range sampler. Returns the partition at
    the MDL stop point (first chosen dL > 0) as a list of member-index lists."""
    rng = np.random.default_rng(seed)
    A = act.double()
    n = A.shape[1]
    groups = [[i] for i in range(n)]
    coact = A.T @ A
    ranks = torch.ones(n, dtype=torch.float64)
    history = []
    while len(groups) > k_floor:
        dl = merge_costs(coact, ranks, alpha)
        dl.fill_diagonal_(float("inf"))
        lo = float(dl.min())
        if lo > 0:                       # their stop point: no merge lowers the MDL
            break
        hi = float(dl[dl != float("inf")].max())
        cand = torch.nonzero(dl <= lo + RANGE_THR * max(hi - lo, 1e-12))
        i, j = cand[rng.integers(len(cand))].tolist()
        i, j = min(i, j), max(i, j)
        # merge j into i: union membership, recompute row i exactly
        A[:, i] = ((A[:, i] + A[:, j]) > 0).double()
        keep = [c for c in range(len(groups)) if c != j]
        A = A[:, keep]
        groups[i] = groups[i] + groups[j]
        groups.pop(j)
        ranks = torch.tensor([len(g) for g in groups], dtype=torch.float64)
        coact = A.T @ A
        history.append((len(groups), lo))
    return groups, history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", default="0.001")
    ap.add_argument("--decoder", default="transpose", choices=["transpose", "pinv"])
    ap.add_argument("--tag", default="", help="knob tag of the payload ('' = baseline)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="CI binarization threshold (0.01 = their clustering appendix; "
                         "0.1 = their ci_alive_threshold, robust to fit noise)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    _self_test()
    print("MDL cost self-test PASS")

    code, model, pd_v = load_fit(a.p, a.decoder, a.tag, a.device)
    F, C = model.n_features, pd_v.C
    act, feats = harvest_binarized_cis(pd_v, float(a.p), total=a.samples,
                                       device=a.device, tau=a.tau)
    alive = act.any(0)
    idx_alive = torch.nonzero(alive).flatten().tolist()
    labels = [f"w_in:{i}" if i < C else f"w_out:{i - C}" for i in idx_alive]
    act = act[:, alive]
    print(f"alive subcomponents: {len(labels)} of {2 * C} "
          f"({sum(1 for l in labels if l.startswith('w_in'))} w_in, "
          f"{sum(1 for l in labels if l.startswith('w_out'))} w_out)")

    # owner maps for the readout
    ao_alive, owner = _write_shard_owners(pd_v)
    mats = activation_matrices(pd_v)
    match_in = hungarian_match(mats["act_in"])
    read_feature = {k: j for j, k in match_in.items()}
    ov_codes = code.M                                     # (F, N) codewords

    # sanity spot-check: co(write shard | read shard) for codeword shards
    col = {l: c for c, l in enumerate(labels)}
    for j in (0, 33, 66):
        kin = match_in[j]
        li = f"w_in:{kin}"
        if li not in col:
            continue
        r = act[:, col[li]]
        neurons = np.where(ov_codes[j])[0]
        shards = [int(k) for k in range(C)
                  if bool(ao_alive[k]) and int(owner[k]) in neurons]
        cond = [float((act[:, col[f'w_out:{k}']] & r).sum() / r.sum().clamp(min=1))
                for k in shards if f"w_out:{k}" in col]
        print(f"  sanity j={j}: read fires {int(r.sum())}x; "
              f"co(write|read) over codeword shards: "
              + ", ".join(f"{v:.2f}" for v in cond))

    os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    for alpha in a.alphas:
        groups, history = cluster(act, alpha)
        sizes = sorted((len(g) for g in groups), reverse=True)
        lines = [f"### Goodfire MDL clustering on VPD fit — {code.name} p={a.p}"
                 f"{a.tag or ' (baseline)'}, alpha={alpha:g}",
                 "",
                 f"- subcomponents: {len(labels)} alive; final clusters: {len(groups)}; "
                 f"sizes: {sizes[:12]}{'...' if len(sizes) > 12 else ''}",
                 f"- merges executed: {len(history)}", ""]
        mapping = {}
        for g_id, g in enumerate(sorted(groups, key=len, reverse=True)):
            n_in = sum(1 for m in g if labels[m].startswith("w_in"))
            n_out = len(g) - n_in
            feats_g = sorted(read_feature[int(labels[m].split(':')[1])]
                             for m in g if labels[m].startswith("w_in")
                             and int(labels[m].split(':')[1]) in read_feature)
            neur_g = sorted({int(owner[int(labels[m].split(':')[1])])
                             for m in g if labels[m].startswith("w_out")})
            desc = []
            if feats_g:
                desc.append(f"features {feats_g[:10]}{'...' if len(feats_g) > 10 else ''}")
            if neur_g:
                desc.append(f"neurons {neur_g[:10]}{'...' if len(neur_g) > 10 else ''}")
            if len(g) > 1 or n_out:
                lines.append(f"- cluster {g_id}: {n_in} w_in + {n_out} w_out "
                             f"({'; '.join(desc)})")
            for m in g:
                mapping[labels[m]] = g_id
        n_single_in = sum(1 for g in groups
                          if len(g) == 1 and labels[g[0]].startswith("w_in"))
        lines.append(f"- singleton w_in clusters: {n_single_in}")
        md = "\n".join(lines) + "\n"
        s_tau = "" if a.tau == TAU else f"_tau{a.tau:g}"
        stem = f"vpd_cluster_mapping{a.tag}_{a.decoder}{s_tau}_alpha{alpha:g}"
        with open(os.path.join(RES_DIR, stem + ".md"), "w") as f:
            f.write(md)
        with open(os.path.join(RES_DIR, stem + ".json"), "w") as f:
            json.dump(mapping, f, indent=1)
        print("\n" + md + f"  saved {stem}.md/.json")


if __name__ == "__main__":
    main()
