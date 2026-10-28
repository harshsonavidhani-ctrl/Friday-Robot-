#!/bin/bash
# 2026-10-28: round 3, continued -- wait for the laptop's training, fetch its
# model, wait for the ACT vs round 3 repeats, then test it 3x in every room.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
O="$ROOT/state/sim_ab/out"; M="$ROOT/state/sim_ab/models/exploration_model_ft3b.onnx"
L=$(python3 -c "import json;print(json.load(open('$ROOT/secrets/laptop_agent.json'))['url'].split('//')[1].split(':')[0])")
SSH=(ssh -o BatchMode=yes -o HostKeyAlias=friday-laptop -o ConnectTimeout=15 "Harsh Vidhani@$L")
until "${SSH[@]}" 'findstr /c:"Saved ONNX" "%USERPROFILE%\FridayML\explore\full3b.log"' 2>/dev/null | grep -q "Saved ONNX"; do sleep 60; done
RUN=$("${SSH[@]}" 'findstr /c:"run folder" "%USERPROFILE%\FridayML\explore\full3b.log"' | sed 's/.*run folder: //' | tr -d '\r')
"${SSH[@]}" "powershell -NoProfile -Command [Convert]::ToBase64String([IO.File]::ReadAllBytes('$RUN\\exploration_model.onnx'))" | tr -d '\r\n ' | base64 -d > "$M"
echo "fetched $RUN -> $M ($(stat -c %s "$M") bytes)"
until grep -q "REPEATS DONE" "$O/rep_progress.txt"; do sleep 30; done
export SKIP_DONE=1
for k in 1 2 3; do "$HERE/run_ft.sh" b$k "$M" ppo_ft; done
echo CHAIN DONE
