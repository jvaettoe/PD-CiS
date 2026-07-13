"""Train the LinearMLP d_mlp=100 baseline grid (n x p, no d_emb axis) for the
final-loss benchmark. cc.load_or_train caches -> resumable / skip-if-done."""
import cc
from mlp import LinearMLP
PS = ["0.8","0.5","0.1","0.01","0.001"]   # fast first
NS = [2.0, 2.5, 3.0, 4.0]
for p in PS:
    for n in NS:
        run = cc.load_or_train(p, LinearMLP, {"n_features":100,"d_mlp":100},
                               seed=42, n_exp=n, device="cpu")
        print(f"n{n} Linear100 p{p}: final {run['losses'][-1]:.3e} ({len(run['losses'])} steps)", flush=True)
print("TRAIN_LINEAR100_DONE", flush=True)
