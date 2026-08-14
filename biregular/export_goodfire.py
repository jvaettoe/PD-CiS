"""Export the biregular models for Goodfire's param-decomp repo.

Writes exports/goodfire/{designed,trained}/biregular.pth (state_dict with
linear1.weight = w_in.T, linear2.weight = w_out.T -- nn.Linear stores
(out, in)) plus the sidecar biregular_train_config.yaml that
BiregularTargetRunInfo expects. Verifies forward equivalence of the exported
wrapper against the source model before writing.

  ./.venv/bin/python -m biregular.export_goodfire
"""
import os

import torch
import torch.nn.functional as F

from biregular.biregular_ansatz import CodeAnsatz
from core.mlp import MLP
from core.paths import CHECKPOINTS, REPO_ROOT

OUT = os.path.join(REPO_ROOT, "exports", "goodfire")
P = 0.001


def wrapper_forward(sd, x):
    return F.relu(x @ sd["linear1.weight"].T) @ sd["linear2.weight"].T


def export(name: str, model: MLP, provenance: str) -> None:
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    sd = {"linear1.weight": model.w_in.detach().T.contiguous(),
          "linear2.weight": model.w_out.detach().T.contiguous()}
    x = torch.rand(256, model.n_features) * 2 - 1
    x *= (torch.rand(256, model.n_features) < 0.05).float()
    with torch.no_grad():
        diff = float((model(x) - wrapper_forward(sd, x)).abs().max())
    assert diff < 1e-5, f"{name}: forward mismatch {diff}"
    torch.save(sd, os.path.join(d, "biregular.pth"))
    with open(os.path.join(d, "biregular_train_config.yaml"), "w") as f:
        f.write("biregular_model_config:\n"
                f"  n_features: {model.n_features}\n"
                f"  d_mlp: {model.d_mlp}\n"
                f"provenance: \"{provenance}\"\n")
    print(f"{name}: exported (forward max |diff| {diff:.2e}) -> {d}")


def main():
    ans = CodeAnsatz(5, swaps=0, decoder="transpose")
    a, b = ans.fit_ab(P)
    export("designed", ans.model((a, b)),
           f"CodeAnsatz K=5 seed0 noswap transpose, fit_ab(p={P}) a={a:.4f} b={b:.4f}")

    ck = torch.load(os.path.join(
        CHECKPOINTS, "ncc_MLP_d_mlp50_n_features100_p0.001_n4_s42.pt"),
        map_location="cpu", weights_only=False)
    m = MLP(**ck["model_config"], device="cpu")
    m.load_state_dict(ck["state_dict"])
    export("trained", m, "trained MLP d50 p=0.001 n4 seed42 (ncc checkpoint)")


if __name__ == "__main__":
    main()
