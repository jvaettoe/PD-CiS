#!/usr/bin/env bash
# Bring a fresh pod up to speed and start caching all remaining models.
# Usage:  bash start_pod.sh root@<IP> <PORT>
#   (take <IP> and <PORT> from the new pod's Connect panel: ssh root@<IP> -p <PORT>)
set -euo pipefail

HOST=${1:?usage: bash start_pod.sh root@IP PORT}
PORT=${2:?usage: bash start_pod.sh root@IP PORT}
KEY=~/.ssh/id_ed25519
cd "/Users/jmvatto/Documents/Research Projects/PIBBSS 2026/PD_CiS_CC"

echo ">>> [1/3] uploading code + checkpoints to $HOST:$PORT ..."
rsync -avz -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" \
      --exclude '.venv' --exclude 'results' --exclude 'figures' \
      --exclude '__pycache__' --exclude '.git' \
      ./ "$HOST:/workspace/PD_CiS_CC/"

echo ">>> [2/3] installing deps + verifying CUDA on the pod ..."
ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST" 'bash -s' <<'REMOTE'
set -e
cd /workspace/PD_CiS_CC
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.__version__, torch.cuda.get_device_name(0))"
pip install -q matplotlib einops jaxtyping tqdm pandas
echo "checkpoints already present: $(ls checkpoints/*.pt 2>/dev/null | wc -l)/120"
REMOTE

echo ">>> [3/3] launching models-only training (caches all remaining models) ..."
ssh -p "$PORT" -i "$KEY" "$HOST" 'bash -s' <<'REMOTE'
cd /workspace/PD_CiS_CC
pkill -f sweep.py 2>/dev/null || true
sleep 2
nohup python -u sweep.py --device cuda --models-only --n 4 2 2.5 3 \
      > /workspace/train_models.log 2>&1 &
sleep 3
echo "launched. current checkpoints: $(ls checkpoints/*.pt 2>/dev/null | wc -l)/120"
REMOTE

echo
echo ">>> training running on the pod. Watch it with:"
echo "    ssh -p $PORT -i $KEY $HOST 'tail -f /workspace/train_models.log'"
echo ">>> when it prints 'models cached ->', pull the full set with:"
echo "    rsync -avz -e \"ssh -p $PORT -i $KEY\" $HOST:/workspace/PD_CiS_CC/checkpoints/ ./checkpoints/"
