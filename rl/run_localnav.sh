#!/usr/bin/env bash
# Run the LeRobot/ACT local-nav policy with the two things that are easy to get
# wrong already handled. Both of these have actually bitten this project:
#
#   1. friday-mcp holds /dev/ttyUSB0, ttyUSB1 AND ttyACM0 permanently via its
#      background pollers. pi_deploy opens the same LiDAR with exclusive=True, so
#      two readers split frames and one wedges MID-FRAME. It survives a while
#      first -- a 60 s run passes with stale=False and proves nothing. This stops
#      the service and, via the trap, ALWAYS restarts it: a crash, a timeout or a
#      Ctrl-C must never leave Friday without her serial doorway.
#
#   2. The hand-written example command carried --motor-max-duty 0.4. That is
#      well under the approved ceiling (friday/config.py MOTOR_MAX_DUTY = 0.85), and
#      0.4 x 255 = 102 sits BELOW the measured 190 pivot break-away -- so pivots
#      got their 2-tick kick, fell under the threshold and hummed. Defaults here
#      are the approved ones; pass flags to override deliberately, not by
#      inheriting a stale paste.
#
# Raising the ceiling ABOVE 0.75 needs Harsh's explicit approval: 4 motors on 2
# L298N channels is ~2x the per-channel load, and a brownout mid-drive resets the
# Pi -- which is the failure this whole project is built to avoid.
#
# Usage:
#   ./run_localnav.sh --simulate --duration 60        # no hardware
#   ./run_localnav.sh --no-cameras --duration 60      # real sensors, wheels dead
#   ./run_localnav.sh --motor friday --i-am-watching --duration 30   # wheels live
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../venv/bin/python"

# Only bounce the service if it was actually up, so a run started while it is
# already stopped does not silently START it on exit.
MCP_WAS_UP=0
if systemctl --user is-active --quiet friday-mcp.service; then
    MCP_WAS_UP=1
fi

restore_mcp() {
    if [ "$MCP_WAS_UP" = "1" ]; then
        systemctl --user start friday-mcp.service || true
        echo "friday-mcp restored: $(systemctl --user is-active friday-mcp.service)"
    fi
}
trap restore_mcp EXIT INT TERM

port_holders() {
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        for f in /proc/"$p"/fd/*; do
            t="$(readlink "$f" 2>/dev/null || true)"
            case "$t" in /dev/ttyUSB*|/dev/ttyACM*) echo "$t <- pid $p";; esac
        done
    done 2>/dev/null | sort -u
}

if [ "$MCP_WAS_UP" = "1" ]; then
    echo "Stopping friday-mcp to release the serial ports..."
    systemctl --user stop friday-mcp.service
    # `systemctl stop` returns when the job is queued, NOT when the process has
    # actually closed its file descriptors -- measured at ~5 s here. A fixed
    # `sleep 3` therefore aborted a real run against ports that were about to be
    # free. Poll for the release instead, and keep a ceiling so a genuinely wedged
    # holder still trips the guard below rather than hanging.
    for _ in $(seq 1 20); do
        [ -z "$(port_holders)" ] && break
        sleep 1
    done
fi

# Fail loudly rather than fighting for frames: if anything still holds a serial
# port, the LiDAR contention above is about to happen and the run is worthless.
holders="$(port_holders)"
if [ -n "$holders" ]; then
    echo "ABORT: serial ports are still held; the LiDAR would lose frames:" >&2
    echo "$holders" >&2
    exit 1
fi

# NOT exec: exec REPLACES this shell, which discards the EXIT trap above and
# leaves friday-mcp stopped after the run. That happened. Run it as a child, keep
# the exit status, and let the trap put the service back.
rc=0
"$PY" "$HERE/friday_localnav_deploy.py" \
    --motor-max-duty 0.85 \
    --motor-min-duty 80 \
    "$@" || rc=$?
exit "$rc"
