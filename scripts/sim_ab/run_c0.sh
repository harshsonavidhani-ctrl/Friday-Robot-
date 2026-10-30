#!/bin/bash
# 2026-10-30, before the hackathon's round 2. BODY sim, five rooms, 120 s each:
#   head   = what she ran in round 1 (HEAD, cap 0.125, reverse floor 0.20)
#   new16  = turn-out + cap 0.16, reverse floor 0.28, pivot duty 215
#   new20  = the same at cap 0.20
# The three run side by side, so each stays slowed the same.
#   scripts/sim_ab/run_c0.sh ; scripts/sim_ab/summarise.py c0 head new16 new20
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
      --unstick-keep-trying --no-way-back --strict-reverse
      --motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard)
export BODY=1
one() {  # tag worktree room flags...
  tag=$1; wt=$2; room=$3; shift 3
  timeout 290 "$PY" "$HERE/run_one.py" "$W/$wt" "$room" "$O/c0_${tag}_$room.json" 120 \
      "${BASE[@]}" "$@" > "$O/c0_${tag}_$room.log" 2>&1
  echo "$tag $room rc=$?"
}
for room in chairs low_things two_rooms popup corridor; do
  one head wt_head "$room" --max-forward-speed 0.125 --reverse-floor 0.20 &
  one new16 wt_new "$room" --max-forward-speed 0.16 --reverse-floor 0.28 --motor-turn-min-duty 215 &
  one new20 wt_new "$room" --max-forward-speed 0.20 --reverse-floor 0.28 --motor-turn-min-duty 215 &
  wait
done
echo ALL DONE
