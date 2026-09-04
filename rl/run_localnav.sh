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

# THE FACE COSTS A CORE, AND THE CONTROL LOOP PAYS FOR IT. Measured with the robot
# standing still: friday_face.py at 84.5% of a core, load average 2.40 on 4 cores
# before a single wheel turns. A 269 s run with the face up MISSED ITS 100 ms
# DEADLINE ON 597 OF 1361 TICKS -- mean 106.7 ms, p95 266 ms -- so the loop ran at
# 6.9 Hz, not 10.
#
# That is not merely "slower". Every rate-limited quantity in the chain is applied
# per tick and sized for 100 ms: the slew limiter, dead reckoning, and the frontier
# planner's heading bias. A 0.55 rad/s bias held for 266 ms instead of 100 is a 2.7x
# oversteer -- she swings past the heading, the next tick corrects, and that
# overshoot-correct cycle is the ZIG-ZAG Harsh watched. omega_out flipped sign 15
# times a minute.
#
# These are Friday's own units, stopped the same way friday-mcp already is and
# restored by the same trap, so a crash or a Ctrl-C cannot leave her faceless.
# --keep-services opts out when you want the face up for a demo and can accept the
# steering cost.
# friday-dashboard is here for a different reason from the other two: it is not the CPU,
# it is that its ffmpeg children hold BOTH /dev/video nodes exclusively whenever anyone
# has the camera panel open, and a V4L2 device does not survive two streaming readers.
# The script used to only WARN about that, which meant a run could go camera-blind --
# and a camera-blind run looks exactly like a working one in the log. Measured
# 2026-09-04: with the panel open, both cameras returned zero frames to the robot.
# HER SERVICES STAY UP BY DEFAULT NOW. Harsh, 2026-09-05: "all the mcp tools
# dashboard face and everything will continuously work while exploring".
#
# They used to be stopped for the run, and the reason was real: measured with the
# robot standing still, friday_face.py alone held 84.5% of a core and a run with
# the face up missed its 100 ms deadline on 597 of 1361 ticks. What changed is
# that the brain was found burning 137-151% of CPU while idle -- a lock convoy in
# stt.Subscription.frames(), which yielded while holding the microphone lock -- and
# fixing that returned a whole core. There is now room to run them.
#
# --quiet-services puts the old behaviour back, and it is what to use when the
# question is "how well can she navigate" rather than "how well does she demo".
CPU_UNITS=""
QUIET_SERVICES=0
for a in "$@"; do
    [ "$a" = "--quiet-services" ] && QUIET_SERVICES=1
done
[ "$QUIET_SERVICES" = "1" ] && \
    CPU_UNITS="friday-face.service friday-brain.service friday-dashboard.service"
STOPPED_UNITS=""
KEEP_SERVICES=0
for a in "$@"; do
    [ "$a" = "--keep-services" ] && KEEP_SERVICES=1
done

restore_units() {
    if [ "$MCP_WAS_UP" = "1" ]; then
        systemctl --user start friday-mcp.service || true
        echo "friday-mcp restored: $(systemctl --user is-active friday-mcp.service)"
    fi
    for u in $STOPPED_UNITS; do
        systemctl --user start "$u" || true
        echo "$u restored: $(systemctl --user is-active "$u")"
    done
}
trap restore_units EXIT INT TERM

if [ "$KEEP_SERVICES" = "0" ]; then
    for u in $CPU_UNITS; do
        if systemctl --user is-active --quiet "$u"; then
            systemctl --user stop "$u" || true
            STOPPED_UNITS="$STOPPED_UNITS $u"
        fi
    done
    [ -n "$STOPPED_UNITS" ] && echo "Stopped for the run (restored on exit):$STOPPED_UNITS"
fi

# The dashboard streams both cameras through ffmpeg whenever someone has its camera
# panel open, and a V4L2 device is exclusive -- so CameraThread cannot open what
# ffmpeg is holding and the run goes camera-blind. That is NOT an abort: a camera is
# advisory and a blind run is still a real run, unlike the LiDAR contention above.
# It is worth saying out loud, because a camera-blind run looks exactly like a
# working one in the logs, and for months it WAS every run (cv2 was missing from the
# venv entirely -- see friday_freespace.py).
camera_holders() {
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        for f in /proc/"$p"/fd/*; do
            t="$(readlink "$f" 2>/dev/null || true)"
            case "$t" in /dev/video*) echo "$t <- pid $p $(tr '\0' ' ' < /proc/"$p"/cmdline 2>/dev/null | cut -c1-40)";; esac
        done
    done 2>/dev/null | sort -u
}

port_holders() {
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        for f in /proc/"$p"/fd/*; do
            t="$(readlink "$f" 2>/dev/null || true)"
            case "$t" in /dev/ttyUSB*|/dev/ttyACM*) echo "$t <- pid $p";; esac
        done
    done 2>/dev/null | sort -u
}

# THE LEASE COMES FIRST. When the run was started through the MCP tool
# (start_exploration), the server has ALREADY put the serial devices down and is
# refusing to reopen them -- see mcp_server/hardware.py. It is still running, still
# answering tools, still driving the dashboard and the face. There is nothing to
# stop, and stopping it would throw away the whole point.
LEASE_FILE="$HERE/../state/hardware.lease"
LEASED=0
if [ -f "$LEASE_FILE" ]; then
    now=$(date +%s)
    exp=$(awk '{print int($1)}' "$LEASE_FILE" 2>/dev/null || echo 0)
    [ "$exp" -gt "$now" ] 2>/dev/null && LEASED=1
fi
if [ "$LEASED" = "1" ]; then
    echo "friday-mcp has leased the hardware to this run and stays UP."
    MCP_WAS_UP=0
elif [ "$MCP_WAS_UP" = "1" ]; then
    echo "Stopping friday-mcp to release the serial ports (no lease held)..."
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

if ! "$PY" -c "import cv2" 2>/dev/null; then
    echo "WARNING: cv2 is not importable from $PY -- both cameras will be dead and" >&2
    echo "         the free-space navigator disabled. Fix with:" >&2
    echo "         $PY -m pip install opencv-python-headless" >&2
fi

cams="$(camera_holders)"
if [ -n "$cams" ]; then
    echo "WARNING: something already holds a camera; this run may be camera-blind." >&2
    echo "$cams" >&2
    echo "         (the dashboard streams them while its camera panel is open)" >&2
fi

# NOT exec: exec REPLACES this shell, which discards the EXIT trap above and
# leaves friday-mcp stopped after the run. That happened. Run it as a child, keep
# the exit status, and let the trap put the service back.
# THE BLACK BOX IS A RING BUFFER THAT DEFAULTS TO 30 SECONDS. Six consecutive 360 s
# autonomy runs were each recorded down to their last 8%, so the question they were
# run to answer -- why one explored well and the next did not -- could not be asked
# of the files at all: 5.5 minutes of every run was overwritten before it was ever
# written out. The flag exists and works (friday_localnav_deploy.py applies it); it
# was simply never passed. Size it from --duration here so the recording covers the
# run by default. ~140 bytes per tick at 10 Hz: a 780 s ring is about a megabyte.
bb_arg=()
want_bb=1
duration=""
prev=""
for a in "$@"; do
    case "$a" in
        --blackbox-seconds|--blackbox-seconds=*) want_bb=0 ;;
        --duration=*) duration="${a#--duration=}" ;;
    esac
    [ "$prev" = "--duration" ] && duration="$a"
    prev="$a"
done
if [ "$want_bb" = "1" ]; then
    # No --duration means the run is open-ended; cap the ring rather than let it grow
    # without bound, but keep it far above the 30 s that lost the six runs above.
    bb=900
    [ -n "$duration" ] && bb="$(awk -v d="$duration" 'BEGIN{printf "%d", d + 60}')"
    bb_arg=(--blackbox-seconds "$bb")
    echo "Black box sized to ${bb}s (the default is 30, which keeps only the tail of a long run)."
fi

FWD_ARGS=()
for a in "$@"; do
    case "$a" in --keep-services|--quiet-services) continue ;; esac
    FWD_ARGS+=("$a")
done

# KEEP THE LOG WITH THE RUN, not wherever the caller happened to redirect it. A
# 20-minute run's log was written to a session scratchpad that was cleared before it
# could be read, so the summary block -- object counts, cliff hits, reverse lifts, the
# per-stage budget -- was gone while the black box survived. The black box has the
# numbers but not the narrative.
LOGDIR="$HERE/../state/runs"
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"
echo "Log: $RUNLOG"

rc=0
"$PY" "$HERE/friday_localnav_deploy.py" \
    --motor-max-duty 0.85 \
    --motor-min-duty 80 \
    "${bb_arg[@]}" \
    "${FWD_ARGS[@]}" > >(tee "$RUNLOG") 2>&1 || rc=$?
# Process substitution, NOT `| tee`: a pipeline's $? is tee's, so `|| rc=$?` would
# report the logger's success and hide a crashed run. This keeps python's own status.
wait
exit "$rc"
