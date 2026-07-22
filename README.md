# PD_CiS_CC — Parameter Decomposition of toy sparse MLPs

Research code for **Parameter Decomposition** (PD) of small single-hidden-layer ReLU MLPs
trained on sparse features (`y = relu(x)`). Methods: **APD**, **OracleAPD**, **SPD** (learned),
and **ExactPD** (the analytic, per-feature ground truth). A second line of work builds
**biregular codeword ansatz** models and runs ExactPD on them.

## Core library (imported everywhere; keep flat)

| File | Role |
|---|---|
| `pd.py` | PD base class + `APD` / `OracleAPD` / `SPD` / `ExactPD`; per-component metrics & rank-1 signatures |
| `mlp.py` | `MLP` / `EmbedMLP` models; weight, effective-map, input–output plots |
| `data.py` | `CleanDataset` — sparse Bernoulli–uniform features, target `relu(x)` |
| `cc.py` | Train an MLP + checkpoint cache (`load_or_train`, save/load) |
| `decomp_fit.py` | Optimiser loop that fits a decomposition; `build_pd`, save/load a decomposition |
| `results_io.py` | `run_id`, paths, and IO for `results/n<n>/{summary,component_metrics}.csv`, curves, decomps |
| `figutil.py` | All figure builders (loss curves, component views, sweep / vs-p summaries) + saving |
| `codes.py` | Binary codeword generators (`regular_code`, `random_code`) + overlap reduction |

## Entrypoints (run directly)

| Script | What it does | Example |
|---|---|---|
| `sweep.py` | Main driver: train + fit each method across `(n_exp, p, d_emb)` | `python sweep.py --methods ExactPD --n 4` |
| `biregular_ansatz.py` | Build the biregular codeword ansatz (refit per `p`) + run ExactPD + full viz | `python biregular_ansatz.py --K 5 --p-sweep` |
| `render_decomp_views.py` | Render the component-viz suite (per-cell + cross-p + vs-p) for a method/model | `python render_decomp_views.py --method ExactPD --demb 200` |
| `plot_components.py` | Render one saved decomposition's component views | `python plot_components.py --run <run_id>` |
| `regen_figures.py` | Rebuild all derived figures from stored curves + summary | `python regen_figures.py` |
| `plot_prelim.py` | Composite loss-curve plot for one sweep cell | `python plot_prelim.py …` |
| `plot_models.py` | Trained-model feature-space weight / effective-map views | `python plot_models.py …` |
| `bench_final_loss.py` | Trained-model final-loss benchmark (EmbedMLP vs neuron/embedding controls) | `python bench_final_loss.py` |
| `bench_train_d100.py`, `bench_train_linear100.py` | Tiny checkpoint-seeders for the benchmark | — |

## Layout

- `checkpoints/`, `results/`, `figures/` — generated artifacts (git-ignored).
  Results are organised by loss exponent: `results/n<n>/…`; figures by
  `figures/n<n>/{loss_curves,components}/<method>/…`; biregular figures under `figures/biregular/`.
- `RUNPOD.md` + `start_pod.sh` / `pull_pod1.sh` / `save_and_stop_pod1.sh` — running the heavy
  sweep on a RunPod GPU.
- `archive/` — retired one-off scripts (see `archive/README.md`).
