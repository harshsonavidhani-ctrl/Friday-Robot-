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

# THE DASHBOARD RUNS THE 21:18 RUN, EXACTLY AS THE MODE FILE SAYS, AND NOTHING
# ELSE. Harsh, 2026-09-17: "i want this run behaviour back and also will be
# triggered by dashboard same this run".
#
# This wrapper used to add two flags of its own on top of the mode:
#   --keep-services  kept face + brain up (2026-09-08), which cost the loop its
#                    deadline: face alone 84.5% of a core, 597 of 1361 ticks late
#                    in a 269 s run, and a late tick is an oversteer.
#   --brain-target   brain-guided targeting (2026-09-09).
# Both are gone. The restored tag's parser does not know --brain-target at all
# and would abort before she moved, and which services go quiet is written in
# ONE place, MODE_QUIET_UNITS in rl/modes/confident_explore.env. So a run from
# the phone and `rl/modes/explore.sh` from a terminal are the same run, differing
# only in --duration.
exec "$HERE/../rl/modes/explore.sh" --duration "$(( MINUTES * 60 ))"
