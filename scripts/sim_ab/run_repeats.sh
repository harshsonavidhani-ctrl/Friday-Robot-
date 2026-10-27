#!/bin/bash
# 2026-10-28: one 2-minute run per room is too noisy to judge doorways -- the same
# two_rooms test went ACT 31.7 / round 3 21.6 once and ACT 18.2 / round 3 30.5
# the next time. Three more of each, every room, the masked harness.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export SKIP_DONE=1          # a stopped batch picks up where it was (10-28: stopped after 7 of 30)
for k in 1 2 3; do
  "$HERE/run_ft.sh" rep$k "$ROOT/state/sim_ab/models/exploration_model_ft3.onnx" act ppo_ft
done
echo REPEATS DONE
