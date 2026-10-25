#!/bin/bash
# 2026-10-25, the fine-tune A/B (Harsh: fine-tune the PPO, it replaces today's
# ACT only if it wins in the simulator): the SAME live flags (modes/
# confident_explore.env) and the drivetrain body, room by room, for
#   act      today's policy (friday_localnav_act.onnx via friday_localnav_deploy)
#   ppo_old  his PPO as it is (rl/exploration_model.onnx via pi_deploy)
#   ppo_ft   the fine-tuned PPO (the .onnx given as $2)
# Usage: run_ft.sh <tag> [fine-tuned.onnx] [codes...]    (codes default: act ppo_old [ppo_ft])
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
O="$ROOT/state/sim_ab/out"; PY="$ROOT/venv/bin/python"
TAG="${1:?tag}"; FT="${2:-}"; shift 2 2>/dev/null || shift $#
CODES=("$@"); [ ${#CODES[@]} -eq 0 ] && CODES=(act ppo_old) && [ -n "$FT" ] && CODES+=(ppo_ft)
MASK="71:79,88:100,221:237,319:334"
FLAGS=(--navigator-speed-authority 0.6 --lidar-self-mask "$MASK" --no-cameras --unstick-keep-trying
       --reverse-floor 0.20 --no-way-back --motor-kick-rest 0 --motor-pulse-burst 1
       --no-motor-plug-guard --max-forward-speed 0.125 --strict-reverse)
export BODY=1
for w in chairs popup two_rooms low_things corridor; do
  for code in "${CODES[@]}"; do
    case "$code" in
      act) POL="" ;;
      ppo_old) POL="$ROOT/rl/exploration_model.onnx" ;;
      ppo_ft) POL="$FT" ;;
    esac
    POLICY="$POL" nice -n 10 timeout 300 $PY $HERE/run_one.py "$ROOT" $w $O/${TAG}_${code}_$w.json 120 "${FLAGS[@]}" \
      > $O/${TAG}_${code}_$w.log 2>&1
    echo "$code $w exit $?"
  done
done
echo ALL DONE
