"""User-friendly storage for decomposition-sweep results.

Two artefacts, both under `results/`:

  - `summary.csv`   -- a TIDY LONG table: one row per (run x loss-term). Every
                       method's differing term names land in a single `term`
                       column, so there are no method-specific empty columns and
                       it pivots cleanly:
                         df.pivot_table(index=['d_emb','p'],
                                        columns=['method','term'], values='final')
  - `curves/<run_id>.json` -- the full per-step loss curves for one run, plus a
                       self-describing `config` block.

`run_id` is a pure slug of the identifying config, so a completed run is
detectable (`is_done`) and re-runs are idempotent."""

import csv
import json
import os
from datetime import datetime

_BASE_RESULTS = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = _BASE_RESULTS
CURVES_DIR = os.path.join(RESULTS_DIR, "curves")


def set_tag(tag: str | None) -> None:
    """Namespace all outputs under results/<tag> (tag=None -> results/). Lets a
    cheap preliminary pass write to results/prelim without touching the full run."""
    global RESULTS_DIR, CURVES_DIR
    RESULTS_DIR = os.path.join(_BASE_RESULTS, tag) if tag else _BASE_RESULTS
    CURVES_DIR = os.path.join(RESULTS_DIR, "curves")

SUMMARY_COLUMNS = [
    "run_id", "timestamp", "model", "n_features", "d_emb", "p", "n_exp",
    "method", "C", "k", "steps", "lr", "seed", "term", "final",
]


def _fmt_n(n_exp: float) -> str:
    """2.0 -> '2', 2.5 -> '2.5' (matches cc._fmt_n so run_ids and checkpoint
    filenames agree on the exponent token)."""
    return f"{float(n_exp):g}"


def make_run_id(*, model: str, d_emb: int | None, p: str, n_exp: float,
                method: str) -> str:
    """Stable, human-readable slug from the identifying config. d_emb=None
    (LinearMLP) -> 'NA'. Example: 'EmbedMLP_demb200_p0.1_n2_APD'.

    C/k/seed/steps are held fixed across a sweep, so they are intentionally NOT
    in the slug; if you later sweep them, append a short hash of the full config."""
    demb = "NA" if d_emb is None else str(d_emb)
    return f"{model}_demb{demb}_p{p}_n{_fmt_n(n_exp)}_{method}"


def summary_path() -> str:
    return os.path.join(RESULTS_DIR, "summary.csv")


def curve_path(run_id: str) -> str:
    return os.path.join(CURVES_DIR, f"{run_id}.json")


def _summary_run_ids() -> set[str]:
    """run_ids already present in summary.csv (empty if the file is absent)."""
    path = summary_path()
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def is_done(run_id: str) -> bool:
    """True iff BOTH the curve JSON exists AND >=1 summary row carries this
    run_id -- so a half-written run (one artefact but not the other) re-runs."""
    return os.path.exists(curve_path(run_id)) and run_id in _summary_run_ids()


def write_curve_json(run_id: str, config: dict,
                     curves: dict[str, list[float]]) -> str:
    """results/curves/<run_id>.json = {"config": config, "curves": curves}.
    Written atomically (tmp file + os.replace) so a crash can't leave a partial
    JSON that would fool is_done."""
    os.makedirs(CURVES_DIR, exist_ok=True)
    path = curve_path(run_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"config": config, "curves": curves}, f, indent=2)
    os.replace(tmp, path)
    return path


def delete_summary_rows(run_id: str) -> int:
    """Remove all rows for `run_id` from summary.csv (used by --force so a
    re-run replaces rather than duplicates). Returns the number removed."""
    path = summary_path()
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    kept = [r for r in rows if r["run_id"] != run_id]
    removed = len(rows) - len(kept)
    if removed:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
    return removed


def append_summary_rows(run_id: str, config: dict,
                        finals: dict[str, float]) -> None:
    """Append one long row per (term, final_value). Header is written once. The
    row's config fields come from `config` (missing keys -> ''); `d_emb` is blank
    for LinearMLP and `k` blank for SPD by simply being absent/None in config."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = summary_path()
    write_header = not os.path.exists(path)
    stamp = datetime.now().isoformat(timespec="seconds")
    base = {**config, "run_id": run_id, "timestamp": stamp}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for term, value in finals.items():
            w.writerow({**base, "term": term, "final": value})


def load_summary():
    """Return the summary as a pandas.DataFrame if pandas is importable, else a
    list[dict] via csv.DictReader. Never hard-fails on a pandas-less venv."""
    path = summary_path()
    try:
        import pandas as pd
        return pd.read_csv(path)
    except ImportError:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
