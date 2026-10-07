#!/bin/bash
# Unattended continuation of one case: wait for its 1-seed screening to finish, check it, run the
# stage 2-3 selection on the server, commit/push it, and launch the stage 4 confirmation (seeds 11 12 13)
# with its OOM watchdog. Every step is written to retrain_2026/auto/<name>.status so a later session can
# tell what happened (the Mac download of the results stays pending: run scripts/sync_retrain_results.sh).
#
# Usage (from the MoQ-NAS repo root, detached):
#   setsid nohup bash scripts/retrain_auto_confirm.sh <name> <case> <expected_screened> <gpu> <jobs_per_gpu> \
#       <workers_per_job> > retrain_2026/auto/<name>.out 2>&1 < /dev/null &
# Example (Case 1 on dualgpu1):
#   ... retrain_auto_confirm.sh C1 C1_triobj 90 1 3 2
# Set AUTO_DRY_RUN=1 to test: selection with --partial into /tmp and a launcher --dry-run, no commit/launch.
cd "$(dirname "$0")/.." || exit 1
NAME=$1; CASE=$2; EXPECTED=$3; GPU=$4; JPG=$5; WPJ=$6
TAG=F13v1c; INTERVAL=${AUTO_INTERVAL:-300}
PY=${AUTO_PYTHON:-$( [ -x "$HOME/miniconda3/envs/moqnas/bin/python" ] && echo "$HOME/miniconda3/envs/moqnas/bin/python" || echo "$HOME/miniforge3/envs/moqnas/bin/python" )}
OUT=retrain_matrices/confirm_${CASE}_${TAG}.csv
mkdir -p retrain_2026/auto retrain_2026/confirm_${NAME}
STATUS=retrain_2026/auto/${NAME}.status
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
screen_launcher() { pgrep -u "$USER" -f "^[^ ]*python[^ ]* launch_retrain_protocol.py --cases $CASE --seeds 1 " | head -1; }

counts() {  # "<ok> <failure_rows>" for the screening (seed_1) of the case
  $PY - "$CASE" "$TAG" <<'PY'
import csv, glob, json, os, sys
case, tag = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open('retrain_matrices/literature_retrain_candidates.csv')) if r['case'] == case]
if case == 'C1_triobj':
    rows = [r for r in rows if any(x in r['role'].split('+') for x in ('best_acc', 'knee', 'compact', 'strat10'))]
ok = 0; fails = 0; seen = set()
for r in rows:
    f = os.path.join('retrain_2026/runs', r['local_dir'], f'retrain_results_{tag}.txt')
    if (r['local_dir'], r['id']) in seen:
        continue
    seen.add((r['local_dir'], r['id']))
    if os.path.isfile(f):
        ok += json.load(open(f)).get(r['id'], {}).get('seed_1', {}).get('status') == 'OK'
for ld in {r['local_dir'] for r in rows}:
    f = os.path.join('retrain_2026/runs', ld, f'retrain_failures_{tag}.csv')
    if os.path.isfile(f):
        fails += sum(1 for x in csv.DictReader(open(f)) if x.get('seed') in ('1', ''))
print(ok, fails)
PY
}

log "start: case=$CASE expected=$EXPECTED gpu=$GPU ${JPG}x${WPJ} dry_run=${AUTO_DRY_RUN:-0}"
if [ -z "$AUTO_DRY_RUN" ]; then
  while [ -n "$(screen_launcher)" ]; do sleep "$INTERVAL"; done
  log "screening launcher finished"
fi
read -r OK FAILS <<< "$(counts)"
log "screening: OK=$OK/$EXPECTED failure_rows=$FAILS"
if [ -z "$AUTO_DRY_RUN" ] && { [ "$OK" -lt "$EXPECTED" ] || [ "$FAILS" -gt 0 ]; }; then
  log "STOP: screening incomplete or with failures; nothing selected or launched (check manually)"
  exit 1
fi

if [ -n "$AUTO_DRY_RUN" ]; then SEL_OUT=/tmp/auto_${NAME}_confirm.csv; PARTIAL=--partial; else SEL_OUT=$OUT; PARTIAL=; fi
if ! $PY scripts/select_retrain_representatives.py --case "$CASE" --runs-root retrain_2026/runs --tag "$TAG" \
     --out "$SEL_OUT" $PARTIAL >> "$STATUS" 2>&1; then
  log "STOP: selection refused or failed (see above)"; exit 1
fi
REPORT="${SEL_OUT%.csv}_report.md"
$PY - "$SEL_OUT" "$REPORT" >> "$STATUS" 2>&1 <<'PY'
import collections, csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print('representatives by algorithm:', dict(collections.Counter(r['algo'] for r in rows)), '| total', len(rows),
      '| trainings with 3 seeds:', 3 * len(rows))
for line in open(sys.argv[2]):
    if line.startswith('Estabilidad media por regla') or line.startswith('| moqnas |') or line.startswith('| nsga'):
        print(line.rstrip())
PY

if [ -n "$AUTO_DRY_RUN" ]; then
  $PY launch_retrain_protocol.py --cases "$CASE" --candidates "$SEL_OUT" --roles all --seeds 11 12 13 --tag "$TAG" \
     --gpus $GPU --jobs-per-gpu "$JPG" --workers-per-job "$WPJ" --dry-run 2>&1 | grep "^#" >> "$STATUS"
  log "dry run done"; exit 0
fi

git add "$OUT" "${OUT%.csv}_report.md" "${OUT%.csv}"_analysis_*.csv
git commit -q -m "retrain: ${CASE} screening done (${OK}/${EXPECTED}); stage 2-3 selection (run on $(hostname))" \
  && log "committed $(git log --oneline -1 | cut -c1-12)" || log "WARN: commit failed"
git push -q origin retrain-2026 >> "$STATUS" 2>&1 && log "pushed to GitHub" || log "WARN: push failed (push manually)"

(setsid nohup $PY launch_retrain_protocol.py --cases "$CASE" --candidates "$OUT" --roles all --seeds 11 12 13 \
   --tag "$TAG" --gpus $GPU --jobs-per-gpu "$JPG" --workers-per-job "$WPJ" \
   > "retrain_2026/confirm_${NAME}/confirm_${NAME}_master.log" 2>&1 < /dev/null &)
sleep 15
(setsid nohup bash scripts/retrain_oom_watchdog.sh "${NAME}conf" retrain_2026/runs "$GPU" --cases "$CASE" \
   --candidates "$OUT" --roles all --seeds 11 12 13 --tag "$TAG" \
   > "retrain_2026/watchdog/${NAME}conf.out" 2>&1 < /dev/null &)
sleep 60
L=$(pgrep -u "$USER" -f "^[^ ]*python[^ ]* launch_retrain_protocol.py --cases $CASE --candidates" | wc -l)
W=$(pgrep -u "$USER" -f "^bash scripts/retrain_oom_watchdog.sh ${NAME}conf" | wc -l)
log "confirmation launched: launchers=$L watchdog=$W; $(grep '^# protocol' "retrain_2026/confirm_${NAME}/confirm_${NAME}_master.log")"
log "PENDING: download results to the Mac (scripts/sync_retrain_results.sh) and record in the strategy doc"
