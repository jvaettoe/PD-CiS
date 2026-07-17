#!/usr/bin/env bash
# Pull pod 1 (APD + SPD) results, merge with pod 2's (OracleAPD) already on disk,
# pull the trained checkpoints, and rebuild every figure from the combined curves.
# Safe to re-run: the summary is rebuilt from staged parts, not blindly appended.
set -euo pipefail
cd "/Users/jmvatto/Documents/Research Projects/PIBBSS 2026/PD_CiS_CC"

SSH="ssh -p 18898 -i ~/.ssh/id_ed25519"
POD1=root@213.173.109.159
BASE=/workspace/PD_CiS_CC
LOG=/workspace/sweep_gpu.log

# 1. Refuse to pull unless pod 1's sweep has actually finished.
echo ">>> checking pod 1 has finished..."
STATE=$($SSH $POD1 "grep -aq '^done\.' $LOG && echo YES || echo NO" 2>/dev/null || echo UNREACHABLE)
if [ "$STATE" != "YES" ]; then
  echo "!! pod 1 not finished (state=$STATE). Current line:"
  $SSH $POD1 "grep -aE '^\[(model|train)' $LOG | tail -1" 2>/dev/null || true
  echo "!! aborting — re-run this script once it prints 'done.'"
  exit 1
fi
echo ">>> pod 1 done. pulling..."

# 2. Curves + figures + checkpoints. Trailing slashes on BOTH sides => contents
#    merge into the target dir (no nested curves/curves this time).
rsync -avz -e "$SSH" "$POD1:$BASE/results/gpu/curves/" ./results/gpu/curves/
rsync -avz -e "$SSH" "$POD1:$BASE/figures/gpu/"        ./figures/gpu/
rsync -avz -e "$SSH" "$POD1:$BASE/checkpoints/"        ./checkpoints/

# 3. Summary merge, idempotently. Stage pod 2's OracleAPD summary once (that is
#    what results/gpu/summary.csv currently holds), then always rebuild
#    combined = OracleAPD (header + rows) + pod-1 APD/SPD (rows only).
[ -f results/gpu/_summary_oracle.csv ] || cp results/gpu/summary.csv results/gpu/_summary_oracle.csv
rsync -avz -e "$SSH" "$POD1:$BASE/results/gpu/summary.csv" /tmp/s1_apdspd.csv
{ cat results/gpu/_summary_oracle.csv; tail -n +2 /tmp/s1_apdspd.csv; } > results/gpu/summary.csv

# 4. Rebuild all loss-curve / composite / sweep-summary figures from the now-
#    combined curves, so composites show APD + SPD + OracleAPD together.
./.venv/bin/python regen_figures.py --tag gpu

# 5. Report what we ended up with.
python3 - <<'PY'
import csv
from collections import Counter
r = list(csv.DictReader(open('results/gpu/summary.csv')))
print("\n=== combined results/gpu/summary.csv ===")
print("rows:", len(r), "| unique fits:", len(set(x['run_id'] for x in r)))
print("by method:", dict(Counter(x['method'] for x in r)))
print("unique models:", len(set(x['run_id'].rsplit('_',1)[0] for x in r)))
PY
echo ">>> done. checkpoints + all three methods merged. Safe to terminate pod 1."
