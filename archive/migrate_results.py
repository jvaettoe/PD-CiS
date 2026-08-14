"""One-off, re-runnable migration to the topical (model-ordered) results layout.

Before: results were namespaced by TRAINING-RUN tag, and the run_id omitted d_mlp
(so the d_mlp=50 and d_mlp=100 n=4 sweeps had IDENTICAL run_ids, distinguished
only by their tag directory: results/decomp_n4_dmlp50 vs results/decomp_n4_dmlp100).

After: results are organised by TOPIC (loss exponent n), mirroring figures/. d_mlp
is first-class in the run_id, so both widths coexist under results/n4/ with distinct
filenames -- e.g. EmbedMLP_demb200_dmlp50_p0.1_n4_APD vs ..._dmlp100_....

What this does (idempotent -- safe to re-run):
  1. COPY each decomp_n4_dmlp{50,100} curve JSON + decomp .pt into results/n4/
     under its NEW d_mlp-keyed run_id (the .pt payload is byte-identical -- it keys
     on ckpt.model_kwargs.d_mlp, not the run_id; only the FILENAME changes. The JSON
     gets config.run_id / config.d_mlp rewritten).
  2. MERGE both tags' summary.csv + component_metrics.csv into
     results/n4/{summary,component_metrics}.csv, injecting the d_mlp column.
  3. MOVE the stale pre-refactor tags results/gpu + results/apdcheck into
     results/_archive/ (kept, not deleted).
The source decomp_n4_dmlp{50,100} dirs are LEFT IN PLACE as a backup (results/ is
gitignored); archive/delete them once you've verified results/n4/.

  ./.venv/bin/python migrate_results.py
"""
import csv
import json
import os
import re
import shutil

import core.results_io as results_io

BASE = results_io.RESULTS_DIR
N_DIR = results_io._n_dir(4)   # results/n4

# (tag directory under results/, the d_mlp those runs used)
SOURCES = [("decomp_n4_dmlp50", 50), ("decomp_n4_dmlp100", 100)]
ARCHIVE_STALE = ["gpu", "apdcheck"]

# Legacy run_id (pre-d_mlp): {model}_demb{demb}_p{p}_n{n}_{method}.
_OLD_RE = re.compile(
    r"^(?P<model>.+?)_demb(?P<demb>NA|\d+)_p(?P<p>[0-9.eE+-]+)"
    r"_n(?P<n>[0-9.]+)_(?P<method>APD|OracleAPD|SPD)$"
)


def _new_run_id(old_run_id: str, d_mlp: int) -> str:
    """Map a legacy (no-d_mlp) run_id to the new d_mlp-keyed slug."""
    m = _OLD_RE.match(old_run_id)
    if not m:
        raise ValueError(f"unparseable legacy run_id: {old_run_id!r}")
    demb = m.group("demb")
    return results_io.make_run_id(
        model=m.group("model"),
        d_emb=None if demb == "NA" else int(demb),
        d_mlp=d_mlp, p=m.group("p"), n_exp=float(m.group("n")),
        method=m.group("method"),
    )


def _copy_curves(src_dir: str, d_mlp: int) -> int:
    out = os.path.join(N_DIR, "curves")
    os.makedirs(out, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(os.path.join(src_dir, "curves"))):
        if not fn.endswith(".json"):
            continue
        old = fn[:-len(".json")]
        new = _new_run_id(old, d_mlp)
        blob = json.load(open(os.path.join(src_dir, "curves", fn)))
        cfg = blob.setdefault("config", {})
        cfg["run_id"] = new
        cfg["d_mlp"] = d_mlp
        with open(os.path.join(out, f"{new}.json"), "w") as f:
            json.dump(blob, f, indent=2)
        n += 1
    return n


def _copy_decomps(src_dir: str, d_mlp: int) -> int:
    out = os.path.join(N_DIR, "decomps")
    os.makedirs(out, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(os.path.join(src_dir, "decomps"))):
        if not fn.endswith(".pt"):
            continue
        old = fn[:-len(".pt")]
        new = _new_run_id(old, d_mlp)
        shutil.copy2(os.path.join(src_dir, "decomps", fn),
                     os.path.join(out, f"{new}.pt"))   # payload byte-identical
        n += 1
    return n


def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _migrate_csv(src_name: str, columns: list[str]) -> list[dict]:
    """Read every source tag's <src_name>, remap run_id -> new + inject d_mlp,
    and return the merged rows (written once by the caller)."""
    merged = []
    for tag, d_mlp in SOURCES:
        rows = _read_csv_rows(os.path.join(BASE, tag, src_name))
        for row in rows:
            row = dict(row)
            row["run_id"] = _new_run_id(row["run_id"], d_mlp)
            row["d_mlp"] = d_mlp
            merged.append(row)
    return merged


def _write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    print(f"migrating into {N_DIR}")
    total_c = total_d = 0
    for tag, d_mlp in SOURCES:
        src = os.path.join(BASE, tag)
        if not os.path.isdir(src):
            print(f"  [skip] {tag} not present (already migrated?)")
            continue
        c = _copy_curves(src, d_mlp)
        d = _copy_decomps(src, d_mlp)
        total_c += c
        total_d += d
        print(f"  {tag} (d_mlp={d_mlp}): copied {c} curves, {d} decomps")

    summary_rows = _migrate_csv("summary.csv", results_io.SUMMARY_COLUMNS)
    _write_csv(results_io.summary_path(4), results_io.SUMMARY_COLUMNS, summary_rows)
    metric_rows = _migrate_csv("component_metrics.csv", results_io.COMPONENT_METRICS_COLUMNS)
    _write_csv(results_io.component_metrics_path(4),
               results_io.COMPONENT_METRICS_COLUMNS, metric_rows)
    print(f"  summary.csv: {len(summary_rows)} rows ; "
          f"component_metrics.csv: {len(metric_rows)} rows")

    # archive stale pre-refactor tags (kept, not deleted)
    archive = os.path.join(BASE, "_archive")
    os.makedirs(archive, exist_ok=True)
    for tag in ARCHIVE_STALE:
        src = os.path.join(BASE, tag)
        dst = os.path.join(archive, tag)
        if os.path.isdir(src):
            if os.path.exists(dst):
                print(f"  [skip archive] {tag}: {dst} already exists")
                continue
            shutil.move(src, dst)
            print(f"  archived results/{tag} -> results/_archive/{tag}")
        else:
            print(f"  [skip archive] results/{tag} not present")

    # verify: every migrated run is 'done' under the new tree
    print("\nverifying is_done for migrated runs:")
    ok = bad = 0
    for tag, d_mlp in SOURCES:
        src = os.path.join(BASE, tag)
        if not os.path.isdir(src):
            continue
        for fn in sorted(os.listdir(os.path.join(src, "decomps"))):
            if not fn.endswith(".pt"):
                continue
            new = _new_run_id(fn[:-3], d_mlp)
            if results_io.is_done(new):
                ok += 1
            else:
                bad += 1
                print(f"  [NOT done] {new}")
    print(f"  {ok} runs done, {bad} incomplete")
    print(f"\ndone. new tree: {N_DIR}")


if __name__ == "__main__":
    main()
