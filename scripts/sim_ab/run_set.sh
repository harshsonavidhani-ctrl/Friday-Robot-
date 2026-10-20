#!/bin/bash
# Usage: run_set.sh <tag> <secs> <world...>   (env MODE=plain|body)
# Runs 4 Sept (dbc0c8c), today, and today capped at 0.125 m/s (body only),
# one after another, each with that version's real launch flags.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="$ROOT/state/sim_ab"; O="$W/out"; mkdir -p "$O"
PY="$ROOT/venv/bin/python"
TAG=$1; SECS=$2; shift 2
MASK="71:79,88:100,221:237,319:334"
OLD=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK")
NEW=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK" --no-cameras --unstick-keep-trying --reverse-floor 0.20 --no-way-back)
[ "${MODE:-body}" = "body" ] && export BODY=1 && NEW+=(--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard)
for w in "$@"; do
  timeout $((SECS+160)) $PY $HERE/run_one.py $W/wt_0904 $w $O/${TAG}_0904_$w.json $SECS "${OLD[@]}" > $O/${TAG}_0904_$w.log 2>&1; echo "0904 $w exit $?"
  timeout $((SECS+160)) $PY $HERE/run_one.py $W/wt_today $w $O/${TAG}_today_$w.json $SECS "${NEW[@]}" > $O/${TAG}_today_$w.log 2>&1; echo "today $w exit $?"
  if [ -n "${BODY:-}" ]; then
    CAP_V=0.125 timeout $((SECS+160)) $PY $HERE/run_one.py $W/wt_today $w $O/${TAG}_cap125_$w.json $SECS "${NEW[@]}" > $O/${TAG}_cap125_$w.log 2>&1; echo "cap125 $w exit $?"
  fi
done
echo ALL DONE
