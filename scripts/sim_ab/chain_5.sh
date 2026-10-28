#!/bin/bash
# 2026-10-28: round 5 (door practice) -- wait for the laptop's training, fetch
# its model, score it against round 3 on the laptop (door crossings), and test
# it 3x in every Pi sim room (tags r5a r5b r5c, code ppo_ft).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
O="$ROOT/state/sim_ab/out"; M="$ROOT/state/sim_ab/models/exploration_model_ft5.onnx"
L=$(python3 -c "import json;print(json.load(open('$ROOT/secrets/laptop_agent.json'))['url'].split('//')[1].split(':')[0])")
SSH=(ssh -o BatchMode=yes -o HostKeyAlias=friday-laptop -o ConnectTimeout=15 "Harsh Vidhani@$L")
until "${SSH[@]}" 'findstr /c:"Saved ONNX" "%USERPROFILE%\FridayML\explore\full5.log"' 2>/dev/null | grep -q "Saved ONNX"; do sleep 60; done
RUN=$("${SSH[@]}" 'findstr /c:"run folder" "%USERPROFILE%\FridayML\explore\full5.log"' | sed 's/.*run folder: //' | tr -d '\r')
"${SSH[@]}" "powershell -NoProfile -Command [Convert]::ToBase64String([IO.File]::ReadAllBytes('$RUN\\exploration_model.onnx'))" | tr -d '\r\n ' | base64 -d > "$M"
echo "fetched $RUN -> $M ($(stat -c %s "$M") bytes)"
R3='%USERPROFILE%\FridayML\explore_runs\run-20261027-2239\ppo_exploration_model.zip'
"${SSH[@]}" "cd /d \"%USERPROFILE%\\FridayML\\explore\" && \"%USERPROFILE%\\FridayML\\py\\python.exe\" -W ignore finetune.py --env 5 --episodes 40 --compare \"$RUN\" --against \"$R3\" > compare5.log 2>&1" &
export SKIP_DONE=1
for k in a b c; do "$HERE/run_ft.sh" r5$k "$M" ppo_ft; done
wait
echo CHAIN DONE
