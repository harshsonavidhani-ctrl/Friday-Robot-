#!/usr/bin/env bash
# Run Friday's saved exploration mode for N MINUTES.
#
#     scripts/explore_for.sh 10        # ten minutes
#
# This exists so `friday-explore@.service` can take a human-sized instance name
# -- `friday-explore@10.service` is ten minutes -- while modes/explore.sh keeps
# taking seconds. systemd cannot do arithmetic in a unit file, and putting
# `$((%i * 60))` straight into an ExecStart shell line would hand an unvalidated
# instance name to the shell.
#
# ⚠ IT ADDS NOTHING ELSE. Harsh, 2026-09-08: "no changes in exploration mode pls
# it will be untoched only it will be wired up". Every flag still comes from
# rl/modes/confident_explore.env via rl/modes/explore.sh; the only argument
# passed here is --duration, which explore.sh has always accepted and which the
# documented `explore.sh --duration 600` form already used.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINUTES="${1:-}"
# A positive integer and nothing else. Anything the dashboard did not send --
# a stray flag, an empty instance, a decimal -- stops here with a message
# rather than reaching a robot that can drive.
if ! [[ "$MINUTES" =~ ^[0-9]+$ ]] || [ "$MINUTES" -lt 1 ]; then
    echo "usage: explore_for.sh <minutes>   (a whole number, 1 or more)" >&2
    exit 64
fi
# An upper bound because this drives real motors on a battery. 120 minutes is
# far beyond any run yet attempted (the longest is 20) and is here to catch a
# typo like 6000, not to express a policy about battery life.
if [ "$MINUTES" -gt 120 ]; then
    echo "refusing $MINUTES minutes: the ceiling is 120." >&2
    exit 64
fi

# HER FACE AND HER BRAIN STAY UP FOR THE RUN. Harsh, 2026-09-08: "during run
# the face plus brain will be working because brain will help in targetting an
# object and move towards it for now enable the face plus brain during run".
#
# --keep-services is run_localnav.sh's own opt-out and it is passed HERE, in
# the dashboard's wrapper, rather than by editing MODE_QUIET_UNITS: the mode
# file stays byte-for-byte as measured. Caller arguments are appended last and
# run_localnav.sh strips this one before argparse sees it.
#
# ⚠ IT COSTS STEERING, AND THE NUMBER IS MEASURED. friday_face.py alone holds
# 84.5% of a core standing still. With the face up, a 269 s run missed its
# 100 ms deadline on 597 of 1361 ticks -- mean 106.7 ms, p95 266 ms -- so the
# loop ran at 6.9 Hz, not 10. Every rate-limited quantity (slew limiter, dead
# reckoning, the planner's heading bias) is sized for 100 ms, so a bias held
# for 266 ms is a 2.7x oversteer: that is the ZIG-ZAG of 2026-09-06. Delete
# this flag to get the quiet run back; nothing else has to change.
#
# ⚠ AND IT IS NOT YET ENOUGH FOR BRAIN-GUIDED TARGETING. run_localnav.sh still
# stops friday-mcp for the run (exclusive LiDAR and motor ports), and the MCP
# server is the brain's ONLY route to the body -- so the brain can see, speak
# and drive the face during a run, but every move() it asks for has nothing to
# answer it. Keeping friday-mcp up needs the hardware LEASE in
# mcp_server/hardware.py (run_localnav.sh's LEASED path), which is how
# autonomy.start_exploration() does it. That is the next piece, not this one.
# AND SHE MAY DRIVE AT WHAT SHE SEES. --brain-target lets the brain name one
# object from the run's own detector and nudge her toward it; the frontier
# planner still steers whenever nothing is in view, which is most of the time.
#
# ⚠ INERT WITHOUT AN API KEY, WHICH IS THIS PI'S NORMAL STATE. There is no .env
# here -- CLAUDE.md keeps ANTHROPIC_API_KEY unset while Friday is being built so
# Claude Code bills the subscription rather than the API -- so brain_router
# returns "skipped", no order is ever written, and the run is exactly what it
# would have been. The flag is passed now so that exporting a key is the only
# thing needed to turn this on, rather than a code change nobody remembers.
#
# Every call is Haiku through brain_router, under DAILY_SPEND_CAP_USD, logged to
# logs/spend.jsonl. An empty room costs nothing: no sighting, no call.
exec "$HERE/../rl/modes/explore.sh" --duration "$(( MINUTES * 60 ))" \
     --keep-services --brain-target
