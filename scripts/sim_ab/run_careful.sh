#!/bin/bash
# Drivetrain sim (BODY=1) of TODAY'S WORKING TREE with stage 4e on:
# --max-forward-speed 0.125 --strict-reverse (2026-10-20). Five rooms, 120 s
# each, results in state/sim_ab/out/b2_careful_*.json -- compare with
# `summarise.py b1 0904 today` (the run before the change).
#
#   setsid nohup scripts/sim_ab/run_careful.sh > state/sim_ab/out/b2.out 2>&1 < /dev/null &
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="$ROOT/state/sim_ab"; O="$W/out"; mkdir -p "$O"
PY="$ROOT/venv/bin/python"
cd "$ROOT" || exit 1
SHA="$(git stash create)"; [ -n "$SHA" ] || SHA=HEAD
git worktree remove --force "$W/wt_careful" 2>/dev/null
git worktree prune
git worktree add -q --detach "$W/wt_careful" "$SHA" || exit 1
echo "code under test: $SHA"
MASK="71:79,88:100,221:237,319:334"
NEW=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK" --no-cameras
     --unstick-keep-trying --reverse-floor 0.20 --no-way-back
     --motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard
     --max-forward-speed 0.125 --strict-reverse)
export BODY=1
for w in chairs popup two_rooms corridor low_things; do
  timeout 280 "$PY" "$HERE/run_one.py" "$W/wt_careful" "$w" "$O/b2_careful_$w.json" 120 "${NEW[@]}" \
      > "$O/b2_careful_$w.log" 2>&1
  echo "careful $w exit $?"
done
echo ALL DONE
