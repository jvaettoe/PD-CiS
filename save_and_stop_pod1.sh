#!/usr/bin/env bash
# Rescue pod 1's PARTIAL data (no done-check) before stopping it. Priority is the
# trained checkpoints, so a restart only redoes decompositions, not training.
set -uo pipefail
cd "/Users/jmvatto/Documents/Research Projects/PIBBSS 2026/PD_CiS_CC"

SSH="ssh -p 18898 -i ~/.ssh/id_ed25519"
POD1=root@213.173.109.159
BASE=/workspace/PD_CiS_CC

echo ">>> pulling pod 1 checkpoints (the models — most important) ..."
rsync -avz -e "$SSH" "$POD1:$BASE/checkpoints/" ./checkpoints/

echo ">>> pulling whatever APD/SPD curves + figures exist so far ..."
rsync -avz -e "$SSH" "$POD1:$BASE/results/gpu/curves/" ./results/gpu/curves/ || true
rsync -avz -e "$SSH" "$POD1:$BASE/figures/gpu/"        ./figures/gpu/        || true

# NON-DESTRUCTIVE: pod 1's partial APD/SPD summary goes to its OWN file.
# results/gpu/summary.csv (the complete OracleAPD summary) is NOT touched.
echo ">>> saving pod 1's partial APD/SPD summary to a SEPARATE file ..."
rsync -avz -e "$SSH" "$POD1:$BASE/results/gpu/summary.csv" \
      ./results/gpu/summary_pod1_apdspd_partial.csv 2>/dev/null || true

echo ">>> checkpoints now local: $(ls checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ') / 120"
python3 - <<'PY'
import csv, collections, os
def summ(p, label):
    if not os.path.exists(p):
        print(f"{label}: (none)"); return
    r = list(csv.DictReader(open(p)))
    print(f"{label}: methods={dict(collections.Counter(x['method'] for x in r))} "
          f"| unique fits={len(set(x['run_id'] for x in r))}")
summ('results/gpu/summary.csv', 'OracleAPD summary (UNCHANGED)')
summ('results/gpu/summary_pod1_apdspd_partial.csv', 'pod-1 partial APD/SPD (separate)')
PY
echo
echo ">>> DATA SAVED. Now STOP the pod in the RunPod console to end GPU billing."
echo "    (Use STOP if restarting within the hour — keeps the volume for a fast"
echo "     restart. Use TERMINATE for zero further charge; you have the data locally.)"
