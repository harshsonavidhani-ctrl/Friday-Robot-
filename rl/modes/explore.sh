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
# What the way back's tracker saw, for replaying a real room offline (way_back._serve).
export FRIDAY_TRACK_RECORD="$STATE/runs/tracker-$(date +%Y%m%d-%H%M%S).jsonl"
# ...and what the bumper ToF saw, raw: a bump can then be replayed frame by frame.
export FRIDAY_TOF_RECORD="$STATE/runs/tof-$(date +%Y%m%d-%H%M%S).jsonl"

write_status running

ARGS=(--motor "$MODE_MOTOR"
      --i-am-watching
      --duration "$MODE_DURATION"
      --navigator-speed-authority "$MODE_NAV_AUTHORITY")
# Her body as the LiDAR sees it TODAY; see confident_explore.env for why the
# restored 21:18 code cannot use its own default.
[ -n "${MODE_LIDAR_SELF_MASK:-}" ] && ARGS+=(--lidar-self-mask "$MODE_LIDAR_SELF_MASK")
[ "${MODE_VOICE:-0}" = "1" ] && ARGS+=(--voice)
# Which services go quiet for the run. This was face + brain + dashboard (the
# 21:18 run); it is the face alone since 2026-09-08, so the dashboard's stop
# button stays reachable while she drives. See confident_explore.env, which is
# the single source of truth for the list and carries the measurements.
# MODE_QUIET_UNITS names the units to stop; MODE_QUIET_SERVICES is the older
# all-three switch. The named list wins, because it is the more specific one.
if [ -n "${MODE_QUIET_UNITS:-}" ]; then
    ARGS+=(--quiet-units "$MODE_QUIET_UNITS")
elif [ "${MODE_QUIET_SERVICES:-0}" = "1" ]; then
    ARGS+=(--quiet-services)
fi
# Reverse escape is ON in the loop; this is how it gets turned OFF from the mode
# file without touching code. Same shape as every other MODE_ flag here.
[ "${MODE_REVERSE_ESCAPE:-1}" = "0" ] && ARGS+=(--no-reverse-escape)

# Brain-guided targeting, same shape as every other MODE_ flag: the env file is
# the only place it is written down. Default 0 keeps a run without the mode file
# identical to the frozen configuration.
[ "${MODE_BRAIN_TARGET:-0}" = "1" ] && ARGS+=(--brain-target)

# Cameras off, same shape as every other MODE_ flag. Default 1 (cameras on)
# keeps a run without the mode file identical to the frozen configuration.
[ "${MODE_CAMERAS:-1}" = "0" ] && ARGS+=(--no-cameras)

# Harsh, 2026-10-02: "bring back the active exploration" (short bursts, very
# slow, did not reverse out when blocked). Both switches select behaviour that
# already exists in the code; default 1 keeps a run without the mode file as
# it has been since 09-28/29.
#   MODE_QUIET_MOTORS=0 -> the drive before a58b9e3 ("quieter gears"): kick on
#     every start, 1-tick creep pulses, no coast before reversing.
#   MODE_UNSTICK=0      -> the blocked behaviour before 200769c: the escape +
#     recovery reverse out, no undo-one-move / "please move me".
if [ "${MODE_QUIET_MOTORS:-1}" = "0" ]; then
    ARGS+=(--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard)
fi
[ "${MODE_UNSTICK:-1}" = "0" ] && ARGS+=(--no-unstick)
# MODE_UNSTICK_KEEP_TRYING=1 (2026-10-02): the undo stays, the "please move me"
# wait goes -- after the tries she rests 3 s and tries again.
[ "${MODE_UNSTICK_KEEP_TRYING:-0}" = "1" ] && ARGS+=(--unstick-keep-trying)
# MODE_STEADY_TURNS=1 (2026-10-04, "noise more than movement ... it should be
# smooth"): a turn in place runs steadily at the pivot floor instead of being
# pulsed on and off, and holds through a same-direction dip. Stops unchanged.
[ "${MODE_STEADY_TURNS:-0}" = "1" ] && ARGS+=(--motor-steady-pivots)
# MODE_START_DEBOUNCE=N (2026-10-04): from a standstill, start only after motion
# has been asked for N ticks running; stops are never delayed.
[ -n "${MODE_START_DEBOUNCE:-}" ] && ARGS+=(--motor-start-debounce "$MODE_START_DEBOUNCE")
# MODE_TURN_COMMIT=S (2026-10-04): a turn taken while blocked is kept S seconds
# against a request to turn the other way -- no left/right dithering.
[ -n "${MODE_TURN_COMMIT:-}" ] && ARGS+=(--turn-commit "$MODE_TURN_COMMIT")
[ -n "${MODE_REVERSE_FLOOR:-}" ] && ARGS+=(--reverse-floor "$MODE_REVERSE_FLOOR")
# MODE_BUMPER_ESCAPE=1 (2026-10-30): one committed way out of a tight spot.
[ "${MODE_BUMPER_ESCAPE:-0}" = "1" ] && ARGS+=(--bumper-escape)
# MODE_TURN_MIN_DUTY=N (2026-10-30): the duty a turn in place runs at. Must stay
# at or under MOTOR_MAX_DUTY * 255 = 216, the ceiling Harsh approved.
[ -n "${MODE_TURN_MIN_DUTY:-}" ] && ARGS+=(--motor-turn-min-duty "$MODE_TURN_MIN_DUTY")
# MODE_MAX_FORWARD_SPEED / MODE_STRICT_REVERSE (2026-10-20): the 09-04 03:52
# speed, and reverse only into space the LiDAR sees empty (pi_deploy stage 4e).
[ -n "${MODE_MAX_FORWARD_SPEED:-}" ] && ARGS+=(--max-forward-speed "$MODE_MAX_FORWARD_SPEED")
[ "${MODE_STRICT_REVERSE:-0}" = "1" ] && ARGS+=(--strict-reverse)

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
