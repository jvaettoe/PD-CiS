import cc
from mlp import EmbedMLP
PS = ["0.8","0.5","0.1","0.01","0.001"]   # fast (few steps) first
NS = [2.0, 2.5, 3.0]
DEMBS = [20,50,100,200,1000]              # emb1000 (slow) last within each row
for p in PS:
    for n in NS:
        for d in DEMBS:
            run = cc.load_or_train(p, EmbedMLP, {"n_features":100,"d_emb":d,"d_mlp":100},
                                   seed=42, n_exp=n, device="cpu")
            print(f"n{n} emb{d} p{p}: final {run['losses'][-1]:.3e} ({len(run['losses'])} steps)", flush=True)
print("TRAIN_MULTI_DONE", flush=True)
