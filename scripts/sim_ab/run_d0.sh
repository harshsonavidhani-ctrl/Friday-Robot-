#!/bin/bash
# 2026-10-31, before round 2. BODY sim, five rooms, 120 s each, as confident_explore.env
# drives her: head (HEAD) against new (the working tree).
#   scripts/sim_ab/run_d0.sh ; scripts/sim_ab/summarise.py d0 head new
#   ROOMS="chairs low_things" NEW_ONLY=1 scripts/sim_ab/run_d0.sh    # after a change
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="$ROOT/state/sim_ab"; O="$W/out"; mkdir -p "$O"
PY="$ROOT/venv/bin/python"
cd "$ROOT" || exit 1
SHA="$(git stash create)"; [ -n "$SHA" ] || SHA=HEAD
for wt in wt_head wt_new; do git worktree remove --force "$W/$wt" 2>/dev/null; done
git worktree prune
git worktree add -q --detach "$W/wt_head" HEAD || exit 1
git worktree add -q --detach "$W/wt_new" "$SHA" || exit 1
echo "new code: $SHA"
MASK="71:79,88:100,221:237,319:334"
BASE=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK" --no-cameras
      --unstick-keep-trying --no-way-back --strict-reverse --bumper-escape
      --keep-distance 0.14 --reverse-floor 0.20 --max-forward-speed 0.125
      --motor-turn-min-duty 190 --motor-turn-kick-duty 0.85
      --motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard)
export BODY=1
one() {  # tag worktree room
  timeout 400 "$PY" "$HERE/run_one.py" "$W/$2" "$3" "$O/d0_$1_$3.json" 120 \
      "${BASE[@]}" > "$O/d0_$1_$3.log" 2>&1
  echo "$1 $3 rc=$?"
}
for room in ${ROOMS:-chairs low_things two_rooms popup corridor}; do
  [ -n "$NEW_ONLY" ] || one head wt_head "$room" &
  one new wt_new "$room" &
  wait
done
echo ALL DONE
