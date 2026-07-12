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
import re
from datetime import datetime

_BASE_RESULTS = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = _BASE_RESULTS
CURVES_DIR = os.path.join(RESULTS_DIR, "curves")
DECOMPS_DIR = os.path.join(RESULTS_DIR, "decomps")


def set_tag(tag: str | None) -> None:
    """Namespace all outputs under results/<tag> (tag=None -> results/). Lets a
    cheap preliminary pass write to results/prelim without touching the full run."""
    global RESULTS_DIR, CURVES_DIR, DECOMPS_DIR
    RESULTS_DIR = os.path.join(_BASE_RESULTS, tag) if tag else _BASE_RESULTS
    CURVES_DIR = os.path.join(RESULTS_DIR, "curves")
    DECOMPS_DIR = os.path.join(RESULTS_DIR, "decomps")


def decomp_path(run_id: str) -> str:
    """results/<tag>/decomps/<run_id>.pt -- the fitted decomposition's component
    tensors, saved so components can be visualised after the fit (see
    decomp_fit.save_decomposition / load_decomposition)."""
    return os.path.join(DECOMPS_DIR, f"{run_id}.pt")

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


_RUN_ID_RE = re.compile(
    r"^(?P<model>.+?)_demb(?P<demb>NA|\d+)_p(?P<p>[0-9.eE+-]+)"
    r"_n(?P<n>[0-9.]+)_(?P<method>APD|OracleAPD|SPD)$"
)


def parse_run_id(run_id: str) -> dict:
    """Inverse of make_run_id: split a slug back into its identifying fields.
    Returns {model, d_emb (int|None), p (str), n_exp (str token), method}. The
    single source of truth for reading a run_id, so plotters don't each re-invent
    a regex. Raises ValueError on a slug that doesn't match."""
    m = _RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"unparseable run_id: {run_id!r}")
    demb = m.group("demb")
    return {"model": m.group("model"),
            "d_emb": None if demb == "NA" else int(demb),
            "p": m.group("p"), "n_exp": m.group("n"), "method": m.group("method")}


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
    """True iff ALL THREE final artefacts exist: the curve JSON, the decomp .pt,
    AND >=1 summary row for this run_id. A crash between any two writes therefore
    re-runs the cell rather than leaving a 'done' run whose .pt never landed (which
    would make load_decomposition/plot_components raise). The .pt is written last
    (see sweep.py), so its presence is the completion sentinel."""
    return (os.path.exists(curve_path(run_id))
            and os.path.exists(decomp_path(run_id))
            and run_id in _summary_run_ids())


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


def _delete_rows(path: str, columns: list[str], run_id: str) -> int:
    """Remove every row whose run_id == `run_id` from a tidy CSV, rewritten
    atomically (tmp + os.replace) so an interrupted --force can't truncate the
    file. Returns the number of rows removed. Shared by the summary and
    component-metrics deleters."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["run_id"] != run_id]
    removed = len(rows) - len(kept)
    if removed:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
        os.replace(tmp, path)
    return removed


def delete_summary_rows(run_id: str) -> int:
    """Remove all summary.csv rows for `run_id` (used by --force so a re-run
    replaces rather than duplicates). Returns the number removed."""
    return _delete_rows(summary_path(), SUMMARY_COLUMNS, run_id)


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


COMPONENT_METRICS_COLUMNS = [
    "run_id", "timestamp", "model", "d_emb", "p", "n_exp", "method", "C",
    "components_used", "effective_count", "features_covered",
    "mean_monosemanticity", "mean_diagonal_share",
]


def component_metrics_path() -> str:
    return os.path.join(RESULTS_DIR, "component_metrics.csv")


def delete_component_metrics_rows(run_id: str) -> int:
    """Remove all component_metrics.csv rows for `run_id` (mirrors
    delete_summary_rows; call it alongside on --force so a re-run replaces rather
    than duplicates the per-run metrics row). Returns the number removed."""
    return _delete_rows(component_metrics_path(), COMPONENT_METRICS_COLUMNS, run_id)


def append_component_metrics(run_id: str, config: dict, metrics: dict) -> None:
    """One tidy row per run capturing how cleanly it decomposed (components_used,
    features_covered, mean_monosemanticity), for cross-sweep comparison. The
    per-component monosemanticity array lives in the decomp .pt, not here."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = component_metrics_path()
    write_header = not os.path.exists(path)
    row = {**config, "run_id": run_id,
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "components_used": metrics["components_used"],
           "effective_count": metrics.get("effective_count"),
           "features_covered": metrics["features_covered"],
           "mean_monosemanticity": metrics["mean_monosemanticity"],
           "mean_diagonal_share": metrics.get("mean_diagonal_share")}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPONENT_METRICS_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


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
