"""Closed-form 100-chain decomposition of EmbedMLP with OUTPUT self-activation
matching (user's conjecture) — no learned parameters, linear solves only.

Conditions: (1) pieces sum to each matrix (faithfulness 0; at d_emb > F modulo the
provably-inert null remainder), (2) rank-1 pieces, (3) chain j's forward on e_j has
the SAME on-target output as the model: chain_jj = R_jj.

Construction, d_emb < F (the superposition regime):
  contacts u_j = v_j = w_e_j (rows of W_emb; gates w_e_j.w_e_j = 1)
  read coefficients:  A0 = minimum-norm solution of  W_e^T A = W_in   (rows a_j)
  write coefficients: B  = B0 + D K^T, B0 = W_out We^+, K = basis of ker(We^T);
                      D solves the F LINEAR equations  relu(a_j).b_j = R_jj
                      (each linear in D; minimum-norm solve)
  pieces P_j^in = w_e_j (x) a_j,  P_j^out = b_j (x) w_e_j.
At d_emb >= F there is no kernel freedom (K empty) and the primal-contact chain
cannot be corrected; there the DUAL construction (embed_exact_decomposition) already
satisfies all three conditions exactly, so we use it and report both.

The rigidity lemma implies the price at d_emb < F: chain hidden patterns a_j cannot
be parallel to the model's r_j — chains compute their feature through DIFFERENT
neuron patterns than the model uses. Measured below, not assumed.

  ./.venv/bin/python embed_chain_construct.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embed_exact_decomposition import decompose, load_embed_mlp

RES_DIR = os.path.join(os.path.dirname(__file__), "results", "embedded")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "embedded")
JS = (0, 33, 66)


def construct(d_emb: int):
    """Returns dict with pieces' factors and diagnostics for one checkpoint."""
    model = load_embed_mlp(d_emb)
    F, N = model.n_features, model.d_mlp
    dd = decompose(model)
    We, Win, Wout = dd["We"], dd["Win"], dd["Wout"]
    X = torch.eye(F, dtype=torch.float64)
    e1 = X @ We
    R = torch.relu(e1 @ Win) @ Wout @ We.T
    r = We @ Win                                     # model's effective reads

    if d_emb >= F:
        # dual construction already satisfies all three conditions here
        duals = dd["duals"]
        A = We @ Win                                 # a_j = r_j (dual chains)
        U = duals                                    # contacts w~_j
        Bmat = (Wout @ We.T)                         # b_j = c_j
        V = duals
        mode = "dual (d_emb >= F: primal-contact system has no kernel freedom)"
        solve_info = "n/a"
    else:
        U = We.T                                     # contacts u_j = w_e_j
        V = We.T
        A = torch.linalg.lstsq(We.T, Win, driver="gelsd").solution  # min-norm
        assert float((We.T @ A - Win).abs().max()) < 1e-9, "read solve failed"
        B0 = Wout @ torch.linalg.pinv(We)            # min-norm: B0 @ We = W_out
        # kernel of We^T (vectors k in R^F with k^T We = 0)
        Uf, Sf, Vhf = torch.linalg.svd(We, full_matrices=True)
        K = Uf[:, d_emb:]                            # (F, F-d_emb)
        H = torch.relu(A)                            # h_j = relu(a_j) rows
        # actual gates: |w_e_j|^2 (unit only to float32 precision in the checkpoint)
        gate = (We * We).sum(1)
        g = torch.diag(R) / gate**2 - (H * B0.T).sum(1)  # gate hits both contacts
        # constraint row j: vec coeff = kron(K[j,:], h_j)  -> M (F, N*(F-d))
        M = torch.einsum("jm,jn->jnm", K, H).reshape(F, -1)
        # exact min-norm solution via the (well-conditioned) 100x100 normal system
        lam = torch.linalg.solve(M @ M.T, g)
        Dvec = M.T @ lam
        D = Dvec.reshape(N, F - d_emb)
        Bmat = B0 + D @ K.T
        mode = "primal contacts + kernel-corrected writes"
        res = float((M @ Dvec - g).abs().max())
        solve_info = (f"D-system: {F}x{M.shape[1]}, lstsq residual {res:.2e}, "
                      f"||D|| = {float(D.norm()):.3f}")
    return {"model": model, "F": F, "N": N, "We": We, "Win": Win, "Wout": Wout,
            "U": U, "A": A, "Bmat": Bmat, "V": V, "R": R, "r": r, "e1": e1,
            "mode": mode, "solve_info": solve_info}


def chain_forward(c, j, x_row):
    """Solo chain j on a single input row (1, F): through W_emb piece j, W_in piece
    j, relu, W_out piece j, unembed."""
    We = c["We"]
    e = x_row[0, j] * We[j]                          # W_emb piece j output
    s = float(e @ c["U"][:, j])
    h = torch.relu(s * c["A"][j])
    t_vec = c["Bmat"][:, j].unsqueeze(1) @ c["V"][:, j].unsqueeze(0)  # (N, d)
    return (h @ t_vec) @ We.T                        # (F,) 1-D output spectrum


def analyse(d_emb: int) -> list[str]:
    c = construct(d_emb)
    F, N = c["F"], c["N"]
    We, Win, Wout = c["We"], c["Win"], c["Wout"]
    R, r = c["R"], c["r"]
    lines = [f"## d_emb = {d_emb}  — {c['mode']}", "",
             f"- solve: {c['solve_info']}"]

    # (1) faithfulness
    sum_in = c["U"] @ c["A"]
    sum_out = c["Bmat"] @ c["V"].T
    if d_emb < F:
        dev1 = float((sum_in - Win).abs().max())
        dev2 = float((sum_out - Wout).abs().max())
        lines.append(f"- (1) faithfulness EXACT, zero remainder: dev W_in {dev1:.1e}, "
                     f"W_out {dev2:.1e}")
        assert max(dev1, dev2) < 1e-8, (dev1, dev2)
    else:
        P = torch.linalg.pinv(We) @ We
        dev1 = float((sum_in - P @ Win).abs().max())
        dev2 = float((sum_out - Wout @ P).abs().max())
        lines.append(f"- (1) faithfulness up to the inert remainder: dev {dev1:.1e} / "
                     f"{dev2:.1e} (forward-exact as before)")
        assert max(dev1, dev2) < 1e-8, (dev1, dev2)

    # (2) rank-1 by construction (outer products) — nothing to test numerically
    lines.append("- (2) all pieces rank-1 by construction")

    # (3) output self-activation match
    X = torch.eye(F, dtype=torch.float64)
    devs, chain_diag = [], []
    for j in range(F):
        out = chain_forward(c, j, X[j:j + 1])
        chain_diag.append(float(out[j]))
        devs.append(abs(float(out[j]) - float(R[j, j])))
    max3 = max(devs)
    lines.append(f"- (3) chain_jj == R_jj: max dev {max3:.2e} over all {F} features "
                 f"(mean R_jj {float(torch.diag(R).mean()):.3f})")
    assert max3 < 1e-7, max3

    # characterization: hidden-pattern divergence + leak + cross-activation
    if d_emb < F:
        cos_pre = torch.nn.functional.cosine_similarity(c["A"], r, dim=1)
        cos_post = torch.nn.functional.cosine_similarity(
            torch.relu(c["A"]), torch.relu(r), dim=1)
        lines.append(f"- hidden-pattern divergence: cos(a_j, r_j) mean "
                     f"{float(cos_pre.mean()):.3f} (min {float(cos_pre.min()):.3f}); "
                     f"cos(relu a_j, relu r_j) mean {float(cos_post.mean()):.3f} — "
                     f"chains compute through DIFFERENT neuron patterns than the model")
    leaks = []
    for j in JS:
        out = chain_forward(c, j, X[j:j + 1])
        off = out.clone(); off[j] = 0
        leaks.append(float(off.norm()))
    lines.append(f"- chain off-target leak on e_j (j={list(JS)}): "
                 + ", ".join(f"{v:.3f}" for v in leaks))
    # cross-activation: chain j responds to e_i with gate Gram_ij
    Gram = We @ We.T
    cross = []
    for j in JS:
        acts = [abs(float((X[i:i + 1] @ We @ c["U"][:, j]))) for i in range(F) if i != j]
        cross.append(sum(1 for a_ in acts if a_ > 0.1))
    lines.append(f"- cross-activation (|gate| > 0.1 on other features' inputs): "
                 + ", ".join(f"j={j}: {n}/99" for j, n in zip(JS, cross))
                 + f"  (Gram offdiag rms {float((Gram - torch.eye(F)).pow(2).mean().sqrt()):.3f})")

    # spectra figure: full vs chain for JS
    fig, axes = plt.subplots(len(JS), 2, figsize=(13.5, 2.9 * len(JS)),
                             squeeze=False, constrained_layout=True, sharey=True)
    for row_i, j in enumerate(JS):
        chain_spec = chain_forward(c, j, X[j:j + 1]).numpy()
        for col, (spec, ttl) in enumerate((
                (R[j].numpy(), f"full model (s = {R[j, j]:.3f})"),
                (chain_spec, f"chain {j} (s = {chain_spec[j]:.3f} = R_jj)"))):
            ax = axes[row_i][col]
            colors = ["tab:red" if i == j else "steelblue" for i in range(F)]
            ax.bar(range(F), spec, width=1.0, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"e_{j}: {ttl}", fontsize=9)
    fig.suptitle(f"Output-matching chains vs full model (d_emb={d_emb}) — "
                 f"closed-form, gates=1, chain_jj=R_jj exact", fontsize=11)
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f"embed_construct_spectrum_demb{d_emb}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    lines.append(f"- figure: {os.path.basename(out_path)}")
    lines.append("")
    return lines


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    out = ["# Closed-form 100-chain decomposition with output self-activation "
           "matching (no learned parameters)", ""]
    for d_emb in (20, 50, 200, 1000):
        print(f"constructing d_emb={d_emb}...")
        out += analyse(d_emb)
    out.append("Verdict: the user's conjecture holds — an exact 100-chain PD with "
               "(1) faithfulness, (2) rank-1, (3) output self-activation matching "
               "exists in closed form at every d_emb, including d_emb < F. The "
               "trace/rigidity obstruction applies only to hidden-level matching, "
               "and manifests here as the measured hidden-pattern divergence: the "
               "chains' neuron patterns differ from the model's own.")
    path = os.path.join(RES_DIR, "embed_chain_construct.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
