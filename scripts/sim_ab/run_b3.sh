#!/bin/bash
# Drivetrain sim with the FIXED motor model (right side wired reversed):
# 4 Sept (dbc0c8c), today BEFORE stage 4e (wt_today, 18:38 snapshot), today WITH
# stage 4e (wt_careful). Room by room, so partial results are usable.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="$ROOT/state/sim_ab"; O="$W/out"; PY="$ROOT/venv/bin/python"
MASK="71:79,88:100,221:237,319:334"
OLD=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK")
NEW=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK" --no-cameras --unstick-keep-trying
     --reverse-floor 0.20 --no-way-back --motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard)
export BODY=1
for w in chairs popup two_rooms low_things corridor; do
  timeout 280 $PY $HERE/run_one.py $W/wt_0904 $w $O/b3_0904_$w.json 120 "${OLD[@]}" > $O/b3_0904_$w.log 2>&1; echo "0904 $w exit $?"
  timeout 280 $PY $HERE/run_one.py $W/wt_today $w $O/b3_today_$w.json 120 "${NEW[@]}" > $O/b3_today_$w.log 2>&1; echo "today $w exit $?"
  timeout 280 $PY $HERE/run_one.py $W/wt_careful $w $O/b3_careful_$w.json 120 "${NEW[@]}" --max-forward-speed 0.125 --strict-reverse > $O/b3_careful_$w.log 2>&1; echo "careful $w exit $?"
done
echo ALL DONE
