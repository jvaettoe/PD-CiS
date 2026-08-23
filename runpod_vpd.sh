#!/usr/bin/env bash
# RunPod driver for the trained-VPD grid (biregular/fn_vpd_train.py).
#
#   ./runpod_vpd.sh launch <HOST> <PORT>   # upload code + kfit records, start run
#   ./runpod_vpd.sh status <HOST> <PORT>   # progress + log tail
#   ./runpod_vpd.sh fetch  <HOST> <PORT>   # pull results back, render the map
#
# HOST/PORT = the pod's SSH-over-exposed-TCP endpoint (user root).
# launch is idempotent: re-running uploads new kfit cells and the train grid
# skips already-finished cells.
set -euo pipefail

CMD="${1:?usage: runpod_vpd.sh launch|status|fetch HOST PORT}"
HOST="${2:?missing HOST}"
PORT="${3:?missing PORT}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
RSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
REMOTE=/workspace/PD_CiS_CC
LOCAL="$(cd "$(dirname "$0")" && pwd)"

case "$CMD" in
launch)
    echo "== uploading code + kfit records to $HOST:$REMOTE"
    $SSH "mkdir -p $REMOTE"
    rsync -rltz -e "$RSH" \
        --include='core/***' --include='biregular/***' --include='embed/***' \
        --include='results/' --include='results/fn_sweep/' \
        --include='results/fn_sweep/kfit/***' \
        --exclude='*' \
        "$LOCAL/" "root@$HOST:$REMOTE/"
    echo "== installing deps + launching train grid"
    $SSH "cd $REMOTE \
        && pip install -q --break-system-packages --root-user-action=ignore \
               numpy matplotlib scipy tqdm jaxtyping einops \
        && python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA on pod\"' \
        && if [ -f vpd.pid ] && kill -0 \$(cat vpd.pid) 2>/dev/null; then \
               echo 'already running (pid '\$(cat vpd.pid)')'; \
           else nohup python -m biregular.fn_vpd_train --train-grid \
                    > vpd_grid.log 2>&1 < /dev/null & \
                echo \$! > vpd.pid; echo 'launched (pid '\$(cat vpd.pid)')'; fi"
    echo "== done; monitor with: ./runpod_vpd.sh status $HOST $PORT"
    ;;
status)
    $SSH "cd $REMOTE 2>/dev/null \
        && echo \"cells done: \$(ls results/fn_sweep/vpd/*.json 2>/dev/null | wc -l) \
of \$(ls results/fn_sweep/kfit/*.json 2>/dev/null | wc -l) kfit records; \
running: \$([ -f vpd.pid ] && kill -0 \$(cat vpd.pid) 2>/dev/null && echo yes || echo no)\" \
        && tail -5 vpd_grid.log"
    ;;
fetch)
    echo "== pulling results/fn_sweep/vpd/ from pod"
    rsync -rltz -e "$RSH" "root@$HOST:$REMOTE/results/fn_sweep/vpd/" \
        "$LOCAL/results/fn_sweep/vpd/"
    echo "== rendering trained preference map"
    (cd "$LOCAL" && ./.venv/bin/python -m biregular.fn_vpd_train --map)
    ;;
*)
    echo "unknown command: $CMD (use launch|status|fetch)" >&2
    exit 1
    ;;
esac
