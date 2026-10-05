#!/bin/bash
# OOM watchdog for a running retrain screening (retrain_2026 stage).
#
# Every minute (WATCHDOG_INTERVAL seconds) it counts out-of-memory failures recorded in retrain_failures_<tag>.csv of the
# case's run folders. On the first new one it stops the launcher (whole process tree) and relaunches
# the same screening with 4 concurrent trainings (--jobs-per-gpu 2 --workers-per-job 2); candidates
# already OK are skipped and the failed ones are retrained. It reduces only once: later OOMs are
# logged, not acted on. It exits when the launcher has finished.
#
# Usage (from the MoQ-NAS repo root, detached):
#   setsid nohup bash scripts/retrain_oom_watchdog.sh <name> <runs_subdir> <gpu> <launcher args...> \
#       > retrain_2026/watchdog/<name>.out 2>&1 < /dev/null &
# Example:
#   ... oom_watchdog.sh AF retrain_2026/runs/experiment_cifar10_acc_flops 1 \
#       --cases AF_std_biobj --roles all --seeds 1 --tag F13v1c
cd "$(dirname "$0")/.." || exit 1
NAME=$1; RUNS=$2; GPU=$3; shift 3; ARGS=("$@")
PY=${WATCHDOG_PYTHON:-$HOME/miniconda3/envs/moqnas/bin/python}
INTERVAL=${WATCHDOG_INTERVAL:-60}
TAG=$(printf '%s\n' "${ARGS[@]}" | grep -A1 -x -- '--tag' | tail -1)
CASE=$(printf '%s\n' "${ARGS[@]}" | grep -A1 -x -- '--cases' | tail -1)
LOG=retrain_2026/watchdog/$NAME.log
mkdir -p retrain_2026/watchdog
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

oom_count() {
  $PY - "$RUNS" "$TAG" <<'PY'
import csv, glob, sys
n = 0
for f in glob.glob(f'{sys.argv[1]}/**/retrain_failures_{sys.argv[2]}.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        msg = (r.get('message') or '').lower()
        if r.get('status') == 'FAILED_OOM' or 'out of memory' in msg or 'terminated abruptly' in msg:
            n += 1
print(n)
PY
}
launcher_pid() { pgrep -u "$USER" -f "^[^ ]*python[^ ]* launch_retrain_protocol.py --cases $CASE" | head -1; }
descendants() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; descendants "$c"; done; }
ours_on_gpu() {
  local u; u=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$u" |
    while IFS=, read -r p _; do ps -o user= -p "$p"; done | grep -c "^$USER$"
}

stop_launcher() {
  local l=$1 all p alive i
  all="$l $(descendants "$l")"
  kill -TERM $all 2>/dev/null
  for i in $(seq 1 30); do
    alive=0; for p in $all; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 2
  done
  for p in $all; do kill -0 "$p" 2>/dev/null && kill -9 "$p"; done
  for i in $(seq 1 30); do [ "$(ours_on_gpu)" = 0 ] && break; sleep 2; done
  log "stopped launcher $l ($(echo $all | wc -w) processes); ours on GPU$GPU now: $(ours_on_gpu)"
}

base=$(oom_count); reduced=0; beat=0
log "start: case=$CASE tag=$TAG gpu=$GPU launcher=$(launcher_pid) baseline OOM rows=$base"
while true; do
  sleep "$INTERVAL"
  l=$(launcher_pid)
  if [ -z "$l" ]; then
    log "launcher finished (no process). OOM rows=$(oom_count). Exiting."
    exit 0
  fi
  n=$(oom_count)
  if [ "$n" -gt "$base" ]; then
    if [ $reduced = 0 ]; then
      log "OOM detected ($base -> $n rows). Reducing to 4 concurrent trainings."
      stop_launcher "$l"
      setsid nohup $PY launch_retrain_protocol.py "${ARGS[@]}" --gpus "$GPU" --jobs-per-gpu 2 --workers-per-job 2 \
        > "retrain_2026/watchdog/${NAME}_master_4.log" 2>&1 < /dev/null &
      sleep 20
      log "relaunched with 2x2: launcher=$(launcher_pid)"
      reduced=1
    else
      log "OOM again at 4 concurrent ($base -> $n rows); not acting, check manually."
    fi
    base=$n
  fi
  beat=$((beat + 1))
  if [ $((beat % 30)) = 0 ]; then log "alive: launcher=$l OOM rows=$n ours on GPU$GPU=$(ours_on_gpu)"; fi
done
