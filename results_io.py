"""User-friendly storage for decomposition-sweep results.

Results are organised TOPICALLY by loss exponent n (mirroring figures/), NOT by
training-run provenance. Everything for one exponent lives under `results/n<n>/`:

  - `results/n<n>/summary.csv`            -- a TIDY LONG table: one row per
                       (run x loss-term). Every method's differing term names land
                       in a single `term` column, so there are no method-specific
                       empty columns and it pivots cleanly:
                         df.pivot_table(index=['d_emb','d_mlp','p'],
                                        columns=['method','term'], values='final')
  - `results/n<n>/component_metrics.csv`  -- one tidy row per run: how cleanly it
                       decomposed (components_used, monosemanticity, ...).
  - `results/n<n>/curves/<run_id>.json`   -- full per-step loss curves + config.
  - `results/n<n>/decomps/<run_id>.pt`    -- the fitted component tensors.

`run_id` is a pure slug of the identifying config (model, d_emb, d_mlp, p, n,
method), so a completed run is detectable (`is_done`) and re-runs are idempotent.
Because the slug encodes n and d_mlp, every path is DERIVED from the run_id -- no
per-run tag directory is needed."""

import csv
import glob
import json
import os
import re
from datetime import datetime

_BASE_RESULTS = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = _BASE_RESULTS   # base root; the per-n topical subdirs derive from run_id


def set_tag(tag: str | None) -> None:
    """No-op, kept for call-site compatibility (exactly like figutil.set_tag).
    Results are organised by TOPIC (loss exponent n) now, so every path is derived
    from the run_id (which encodes n and d_mlp) -- there is no per-run tag directory
    to namespace. Left as a stub so existing --tag call sites don't break."""
    return None


def _n_dir(n_exp: float | str) -> str:
    """results/n<token> for a loss exponent (2.0 -> results/n2, 2.5 -> results/n2.5),
    mirroring figutil.n_seg so results and figures share the same per-exponent tree."""
    return os.path.join(_BASE_RESULTS, f"n{_fmt_n(float(n_exp))}")


def _n_dir_for(run_id: str) -> str:
    """Topical dir for a run, parsed from its run_id's n token."""
    return _n_dir(parse_run_id(run_id)["n_exp"])


def decomp_path(run_id: str) -> str:
    """results/n<n>/decomps/<run_id>.pt -- the fitted decomposition's component
    tensors, saved so components can be visualised after the fit (see
    decomp_fit.save_decomposition / load_decomposition)."""
    return os.path.join(_n_dir_for(run_id), "decomps", f"{run_id}.pt")

SUMMARY_COLUMNS = [
    "run_id", "timestamp", "model", "n_features", "d_emb", "d_mlp", "p", "n_exp",
    "method", "C", "k", "steps", "lr", "seed", "term", "final",
]


def _fmt_n(n_exp: float) -> str:
    """2.0 -> '2', 2.5 -> '2.5' (matches cc._fmt_n so run_ids and checkpoint
    filenames agree on the exponent token)."""
    return f"{float(n_exp):g}"


def make_run_id(*, model: str, d_emb: int | None, d_mlp: int, p: str,
                n_exp: float, method: str) -> str:
    """Stable, human-readable slug from the identifying config. d_emb=None
    (LinearMLP) -> 'NA'. Example: 'EmbedMLP_demb200_dmlp100_p0.1_n4_APD',
    'LinearMLP_dembNA_dmlp50_p0.1_n4_APD'.

    d_mlp is now first-class in the slug so the d_mlp=50 and d_mlp=100 sweeps get
    DISTINCT run_ids (they previously collided, distinguished only by a tag dir).
    C/k/seed/steps are held fixed across a sweep, so they are intentionally NOT
    in the slug; if you later sweep them, append a short hash of the full config."""
    demb = "NA" if d_emb is None else str(d_emb)
    return f"{model}_demb{demb}_dmlp{d_mlp}_p{p}_n{_fmt_n(n_exp)}_{method}"


_RUN_ID_RE = re.compile(
    r"^(?P<model>.+?)_demb(?P<demb>NA|\d+)_dmlp(?P<dmlp>\d+)_p(?P<p>[0-9.eE+-]+)"
    r"_n(?P<n>[0-9.]+)_(?P<method>APD|OracleAPD|SPD|ExactPD|VPD)$"
)


def parse_run_id(run_id: str) -> dict:
    """Inverse of make_run_id: split a slug back into its identifying fields.
    Returns {model, d_emb (int|None), d_mlp (int), p (str), n_exp (str token),
    method}. The single source of truth for reading a run_id, so plotters don't
    each re-invent a regex. Raises ValueError on a slug that doesn't match."""
    m = _RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"unparseable run_id: {run_id!r}")
    demb = m.group("demb")
    return {"model": m.group("model"),
            "d_emb": None if demb == "NA" else int(demb),
            "d_mlp": int(m.group("dmlp")),
            "p": m.group("p"), "n_exp": m.group("n"), "method": m.group("method")}


def summary_path(n_exp: float | str) -> str:
    return os.path.join(_n_dir(n_exp), "summary.csv")


def curve_path(run_id: str) -> str:
    return os.path.join(_n_dir_for(run_id), "curves", f"{run_id}.json")


def _summary_run_ids(n_exp: float | str) -> set[str]:
    """run_ids already present in results/n<n>/summary.csv (empty if absent)."""
    path = summary_path(n_exp)
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
    n_exp = parse_run_id(run_id)["n_exp"]
    return (os.path.exists(curve_path(run_id))
            and os.path.exists(decomp_path(run_id))
            and run_id in _summary_run_ids(n_exp))


def write_curve_json(run_id: str, config: dict,
                     curves: dict[str, list[float]]) -> str:
    """results/curves/<run_id>.json = {"config": config, "curves": curves}.
    Written atomically (tmp file + os.replace) so a crash can't leave a partial
    JSON that would fool is_done."""
    path = curve_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    n_exp = parse_run_id(run_id)["n_exp"]
    return _delete_rows(summary_path(n_exp), SUMMARY_COLUMNS, run_id)


def append_summary_rows(run_id: str, config: dict,
                        finals: dict[str, float]) -> None:
    """Append one long row per (term, final_value). Header is written once. The
    row's config fields come from `config` (missing keys -> ''); `d_emb` is blank
    for LinearMLP and `k` blank for SPD by simply being absent/None in config."""
    path = summary_path(parse_run_id(run_id)["n_exp"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    "run_id", "timestamp", "model", "d_emb", "d_mlp", "p", "n_exp", "method", "C",
    "components_used", "effective_count", "features_covered",
    "mean_monosemanticity", "mean_diagonal_share",
]


def component_metrics_path(n_exp: float | str) -> str:
    return os.path.join(_n_dir(n_exp), "component_metrics.csv")


def delete_component_metrics_rows(run_id: str) -> int:
    """Remove all component_metrics.csv rows for `run_id` (mirrors
    delete_summary_rows; call it alongside on --force so a re-run replaces rather
    than duplicates the per-run metrics row). Returns the number removed."""
    n_exp = parse_run_id(run_id)["n_exp"]
    return _delete_rows(component_metrics_path(n_exp), COMPONENT_METRICS_COLUMNS, run_id)


def append_component_metrics(run_id: str, config: dict, metrics: dict) -> None:
    """One tidy row per run capturing how cleanly it decomposed (components_used,
    features_covered, mean_monosemanticity), for cross-sweep comparison. The
    per-component monosemanticity array lives in the decomp .pt, not here."""
    path = component_metrics_path(parse_run_id(run_id)["n_exp"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def _all_summary_paths() -> list[str]:
    """Every results/n*/summary.csv, sorted (the topical per-exponent tables)."""
    return sorted(glob.glob(os.path.join(_BASE_RESULTS, "n*", "summary.csv")))


def load_summary(n_exp: float | str | None = None):
    """Return the summary as a pandas.DataFrame if pandas is importable, else a
    list[dict] via csv.DictReader. Never hard-fails on a pandas-less venv.

    n_exp selects one exponent's table (results/n<n>/summary.csv); n_exp=None
    concatenates every results/n*/summary.csv into a single frame/list, so a
    cross-exponent plotter sees the whole sweep."""
    paths = [summary_path(n_exp)] if n_exp is not None else _all_summary_paths()
    paths = [p for p in paths if os.path.exists(p)]
    try:
        import pandas as pd
        if not paths:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    except ImportError:
        rows: list[dict] = []
        for p in paths:
            with open(p, newline="") as f:
                rows.extend(csv.DictReader(f))
        return rows
