#!/usr/bin/env bash
# FRIDAY MODE: confident_explore -- one command, the best run she has done.
#
#     rl/modes/explore.sh                 # the frozen 20-minute run
#     rl/modes/explore.sh --duration 600  # same everything, shorter leash
#
# Harsh, 2026-09-04: "save this as a MODE of friday ... friday explore will
# trigger the saved autonomous run".
#
# ⚠ THIS CANNOT BE CALLED FROM INSIDE THE MCP SERVER, and that is the whole
# reason the systemd unit next to it exists.
#
# friday_localnav_deploy needs EXCLUSIVE access to the LiDAR and motor serial
# ports, so run_localnav.sh stops friday-mcp for the duration and restores it
# on exit. An MCP tool that called this directly would therefore kill the
# server it was answering from, and the tool call would die mid-flight with
# the run orphaned. So the brain must not run this script; it should start the
# UNIT, which returns immediately:
#
#     systemctl --user start friday-explore.service     # begins the run
#     systemctl --user stop  friday-explore.service     # stops her early
#     systemctl --user is-active friday-explore.service # still going?
#     cat ~/friday/state/explore_mode.json              # what happened
#
# Wiring that into mcp_server/tools/ is one decorated function -- see
# mcp_server/registry.py -- and it is deliberately left for Harsh to do.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL="$(cd "$HERE/.." && pwd)"
STATE="$RL/../state"
MODE_FILE="$HERE/confident_explore.env"
STATUS="$STATE/explore_mode.json"

# shellcheck source=/dev/null
. "$MODE_FILE"

# THE MAP IS THE POINT OF A MODE THAT RUNS AGAIN AND AGAIN. Copy it aside
# before every run: relocalisation can legitimately reject a map (she was
# carried, the room changed) and then run() archives it and rebuilds, which is
# correct but irreversible from inside the run. The golden copy from the 21:18
# run is state/rl_map-GOLDEN-20260904-2118.* and is never overwritten by this.
if [ -f "$STATE/rl_map.npz" ]; then
    cp -f "$STATE/rl_map.npz"        "$STATE/rl_map.prev.npz"        2>/dev/null || true
    cp -f "$STATE/rl_map.pose.json"  "$STATE/rl_map.prev.pose.json"  2>/dev/null || true
    cp -f "$STATE/rl_map.visits.npz" "$STATE/rl_map.prev.visits.npz" 2>/dev/null || true
fi

started="$(date -Is)"
write_status() {
    # Machine-readable, for the brain and the dashboard. Written on every exit
    # path including a kill, so "is she still going" is never inferred from the
    # absence of a file.
    cat > "$STATUS" <<JSON
{
  "mode": "confident_explore",
  "state": "$1",
  "started": "$started",
  "ended": "$(date -Is)",
  "exit_code": ${2:-null},
  "log": "$(ls -1t "$STATE/runs"/run-*.log 2>/dev/null | head -1)",
  "code_tag": "run-20260904-2118-best",
  "previous_map": "state/rl_map.prev.npz"
}
JSON
}
trap 'write_status interrupted 130' INT TERM

write_status running

ARGS=(--motor "$MODE_MOTOR"
      --i-am-watching
      --duration "$MODE_DURATION"
      --navigator-speed-authority "$MODE_NAV_AUTHORITY")
[ "${MODE_VOICE:-0}" = "1" ] && ARGS+=(--voice)
# The other half of the 21:18 run: face, brain and dashboard down for the
# duration. run_localnav.sh has always had the flag; it was never passed from
# here, which is why every mode run since has driven with a 130-200 ms loop
# against a 100 ms deadline. See confident_explore.env for the measurements.
[ "${MODE_QUIET_SERVICES:-0}" = "1" ] && ARGS+=(--quiet-services)
# Reverse escape is ON in the loop; this is how it gets turned OFF from the mode
# file without touching code. Same shape as every other MODE_ flag here.
[ "${MODE_REVERSE_ESCAPE:-1}" = "0" ] && ARGS+=(--no-reverse-escape)

# Caller's arguments go LAST so they win: run_localnav.sh forwards them to
# argparse, which takes the final occurrence. That is what makes
# `explore.sh --duration 600` work without a second mode file.
"$RL/run_localnav.sh" "${ARGS[@]}" "$@"
rc=$?

# The log path is resolved at write time by taking the newest file in
# state/runs -- run_localnav.sh owns RUNLOG and does not export it, and reading
# a variable that is only set in a child shell is how a status file ends up
# confidently pointing at nothing.

# 134 is SIGABRT from the known `FATAL: exception not rethrown` at teardown. It
# happens AFTER the map, the black box and the summary are all written, so the
# run's output is complete and calling it a failure would be wrong. Reported as
# its own state rather than silently mapped to success.
case "$rc" in
    0)   write_status completed 0 ;;
    134) write_status completed_with_teardown_abort 134 ;;
    *)   write_status failed "$rc" ;;
esac
exit "$rc"
