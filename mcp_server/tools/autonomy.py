"""autonomy.py — "Friday, explore" starts the saved autonomous run.

Harsh, 2026-09-04: *"the wakeword file will be like switch like hey friday
explore then the autonoumus run then like this every voice commnds and every
tasks will be fired by wake word"*.

THIS IS NOT `explore_room()`. There are two exploration systems in this
project and they are not interchangeable:

  * `tools/explore.py explore_room()` -- the step-9 SLAM mission. Discrete
    moves through movement.turn_degrees / move_distance, this server's own
    safety interrupt, and a map that starts empty every time.
  * THIS ONE -- the RL/DWA continuous run in rl/, driven at 10 Hz by
    friday_localnav_deploy.py, with the persistent map, the near-field
    barrier, cliff detection and object detection. It is the configuration
    measured as the best run of the project (git tag run-20260904-2118-best)
    and the one Harsh asked to be able to fire by voice.

IT RUNS AS A SYSTEMD UNIT AND NOT AS A SUBPROCESS OF THIS SERVER, and that is
load-bearing rather than tidy. The run needs the LiDAR, ToF and motor serial
ports; this server owns them. Shelling out to the script directly would mean
this server's own hardware being taken away underneath the tool call that
started it. So:

  1. the hardware is LEASED OUT -- see hardware.lease_out(). The serial devices
     are released and this server refuses to reopen them until the lease ends.
     Every other tool, the dashboard, the face, the voice and the brain keep
     working, which is exactly what was asked for;
  2. `systemctl --user start friday-explore.service` returns immediately and
     the run outlives this call;
  3. the lease expires on its own, so a crashed run cannot leave her unable to
     open her own motors.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import config
from mcp_server import hardware
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.autonomy")

# A run is started as friday-explore@<minutes>.service and found by prefix --
# see _active_unit(). The plain `friday-explore.service` still exists for a run
# started by hand and is matched by that same prefix, so nothing here names one
# unit and misses the other.
#
# scripts/explore_for.sh refuses anything above this, and the unit name IS the
# minutes, so clamp here rather than handing systemd an instance that can only
# fail after the run has already been announced.
EXPLORE_MINUTES_MAX = 120
# The mode's own default is 1200 s. The lease is deliberately longer: it has to
# outlast the run plus the teardown that restores services, and an over-long
# lease costs nothing (returning it is explicit, and expiry is the backstop for
# a crash) while a short one would hand the ports back mid-drive.
LEASE_MARGIN_S = 300.0
DEFAULT_RUN_S = 1200.0


# The Tier-2 planner's unit. Started beside a run when thinking is asked for,
# and ALWAYS stopped when the run stops -- see stop_exploration().
COGNITIVE_UNIT = "friday-cognitive.service"


def _systemctl(*args: str) -> tuple[int, str]:
    done = subprocess.run(("systemctl", "--user", *args),
                          capture_output=True, text=True, timeout=20)
    return done.returncode, (done.stdout + done.stderr).strip()


# The run's own process name. A run started by hand -- `./run_localnav.sh ...`
# or rl/modes/explore.sh from a terminal -- is not owned by the unit, so
# `systemctl is-active` says nothing about it. "Stop" has to reach that one too.
RUN_PROCESS = "friday_localnav_deploy.py"


def _active_unit() -> str | None:
    """Whichever exploration unit is out right now, or None.

    A run reaches systemd by more than one name: `friday-explore.service` (the
    frozen 20-minute default, started by hand) and
    `friday-explore@<minutes>.service` (this tool and the dashboard's button).
    Matching only the first meant "stop" could not reach a run started from the
    phone, and `_running()` reported a driving robot as idle -- which is how a
    second run gets started on top of the first, fighting it for the ports.

    `activating` counts as much as `active`: these are Type=oneshot units, so
    systemd reports a run IN PROGRESS as "activating (start)", and a check for
    "active" alone calls every live run stopped.
    """
    try:
        done = subprocess.run(
            ("systemctl", "--user", "list-units", "--no-legend", "--no-pager",
             "--state=active", "--state=activating", "friday-explore*"),
            capture_output=True, text=True, timeout=20)
    except Exception:                               # noqa: BLE001
        return None
    for line in done.stdout.splitlines():
        fields = line.replace("●", " ").replace("*", " ").split()
        if fields and fields[0].startswith("friday-explore"):
            return fields[0]
    return None


def _run_pids() -> list[int]:
    """PIDs of any autonomous run, however it was started.

    Matched on the SCRIPT PATH rather than with a bare pkill pattern, because a
    loose pattern here would match this very server's own command line and the
    shell running it -- which is the mistake that has killed a working session
    on this machine before.
    """
    pids: list[int] = []
    try:
        done = subprocess.run(["pgrep", "-f", RUN_PROCESS],
                              capture_output=True, text=True, timeout=10)
    except Exception:                               # noqa: BLE001
        return pids
    for token in done.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        # CONFIRM IT FROM /proc BEFORE IT COUNTS AS A RUN. pgrep -f matches any
        # command line CONTAINING the string, which includes the shell that
        # launched it, an editor with the file open, and a grep for it -- the
        # first version of this returned two PIDs where one run existed. This
        # function's result gets SIGINT, so a loose match here is how the wrong
        # process gets signalled.
        try:
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue                                # exited between pgrep and here
        if len(argv) >= 2 and argv[0].endswith(b"python") \
                and argv[1].endswith(RUN_PROCESS.encode()):
            pids.append(pid)
    return pids


def _running() -> bool:
    return _active_unit() is not None or bool(_run_pids())


@tool
def start_exploration(minutes: float | None = None,
                      thinking: bool | None = None) -> str:
    """Go and explore on my own, driving myself, until I am done or you stop me.

    Use this when Harsh says explore, go explore, map the house, look around on
    your own, or asks me to go off and discover the place by myself. I keep
    talking and everything else keeps working while I do it.

    This is the full autonomous rover run -- I drive continuously at 10 Hz,
    steer round things I can see, avoid drops and stairs, name what I look at,
    and add what I find to the map I already have.

    Args:
        minutes: how long to explore. Defaults to 20.
        thinking: True to ALSO think about where to go while I drive -- my
            Tier-2 planner reads my LiDAR, ToF and my own account of the run
            every few seconds and nudges me toward somewhere new instead of
            re-driving ground I already have. Use it when Harsh says think,
            thinking, smart, intelligent, LLM, AI or "properly" about
            exploring. It needs the internet and it costs a little per
            decision, so it is off unless asked for.

    Returns JSON: whether it started, how long for, and where the log is.
    """
    if _running():
        return json.dumps({"started": False,
                           "why": "I am already exploring. Say stop to end it."})

    # MINUTES HAS TO REACH THE RUN, NOT JUST THE LEASE. This used to start
    # `friday-explore.service`, whose duration is MODE_DURATION=1200 in the
    # mode file -- so "explore for five minutes" leased the hardware for five
    # and then drove for twenty. The instance name IS the minutes, and
    # scripts/explore_for.sh turns it into --duration, which argparse takes
    # last and therefore wins over the mode's own value.
    mins = 20 if minutes is None else int(round(max(1.0, float(minutes))))
    mins = max(1, min(mins, EXPLORE_MINUTES_MAX))
    run_s = mins * 60.0
    unit = f"friday-explore@{mins}.service"

    # LEASE FIRST, START SECOND. The other order races: the run would try to
    # open the LiDAR while this process still had it, and lose. Leasing here --
    # inside the server that HOLDS the ports -- is the whole mechanism: it
    # closes them, so rl/run_localnav.sh finds nothing holding /dev/ttyUSB*
    # and leaves this server up instead of stopping it.
    hardware.lease_out(run_s + LEASE_MARGIN_S, holder="friday-explore")
    # --no-block: a Type=oneshot unit's `start` does not return until the whole
    # run has finished, so a plain start would block this tool call for the
    # entire drive and then fail on the subprocess timeout.
    code, out = _systemctl("start", "--no-block", unit)
    if code != 0:
        hardware.lease_return()
        log.error("could not start %s: %s", unit, out)
        return json.dumps({"started": False, "why": out or "systemctl refused"})

    # THINKING IS A SEPARATE UNIT, STARTED SECOND AND NEVER ALLOWED TO FAIL THE
    # RUN. friday-cognitive only reads the run's published state and writes
    # heading orders, so a run without it is exactly the run that has always
    # existed -- which is why a failure here is reported and not raised.
    thought = "off (ask for thinking to turn it on)"
    if thinking:
        tcode, tout = _systemctl("start", "--no-block", COGNITIVE_UNIT)
        if tcode == 0:
            thought = "on -- I will think about where to go while I drive"
            log.warning("cognitive planner started alongside the run")
        else:
            thought = f"asked for, but it would not start ({tout or 'systemctl refused'})"
            log.error("could not start %s: %s", COGNITIVE_UNIT, tout)

    log.warning("autonomous exploration started for %.0fs (%s, thinking=%s)",
                run_s, unit, bool(thinking))
    return json.dumps({
        "started": True,
        "minutes": mins,
        "thinking": thought,
        "unit": unit,
        "status_file": str(config.ROOT / "state" / "explore_mode.json"),
        "note": "my motors and range sensors belong to the run until it ends; "
                "every other tool still works",
    })


@tool
def stop_exploration() -> str:
    """Stop exploring and give me my wheels back.

    Use this when Harsh says stop while I am off exploring on my own. The
    ordinary stop() only reaches moves this server is making, and an autonomous
    run is a separate process, so it needs this.
    """
    # THE PLANNER GOES DOWN FIRST AND UNCONDITIONALLY. It is stopped before the
    # run so it cannot write one more heading order into a run that is already
    # tearing down, and it is stopped even when no run is out -- otherwise a
    # planner left up from a previous run keeps issuing orders at the next one.
    # A standing order expires on its own 15 s TTL anyway; this makes it
    # immediate. Never allowed to fail the stop.
    try:
        _systemctl("stop", "--no-block", COGNITIVE_UNIT)
    except Exception as exc:                        # noqa: BLE001
        log.error("could not stop %s: %s", COGNITIVE_UNIT, exc)

    if not _running():
        hardware.lease_return()
        return "I was not exploring. Wheels are mine again anyway."

    # The unit first, if that is how it was started -- its teardown stops the
    # motors and puts her services back. Whichever unit is actually out: a run
    # started from the phone is friday-explore@<minutes>.service, and stopping
    # the plain name would report success while she kept driving.
    # --no-block: the EXIT trap has up to 90 s to restore her services, and a
    # tool call that waited for it would time out long before.
    out_unit = _active_unit()
    if out_unit:
        _systemctl("stop", "--no-block", out_unit)

    # ...and then any run started by hand, which the unit knows nothing about.
    # SIGINT, not SIGKILL: run_localnav.sh traps it to restore friday-mcp,
    # friday-face and the rest, and friday_localnav_deploy uses it to stop the
    # motors, save the map and write the black box. Killing outright would leave
    # the wheels commanded and her services down.
    left = _run_pids()
    for pid in left:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception as exc:                    # noqa: BLE001
            log.error("could not signal run pid %d: %s", pid, exc)
    for _ in range(60):                             # up to ~15 s to shut down
        if not _run_pids():
            break
        time.sleep(0.25)

    # The lease is returned whatever happened above, because a lease that
    # outlives its run is indistinguishable from broken hardware.
    hardware.lease_return()
    still = _run_pids()
    if still:
        return ("I have asked the run to stop and released my wheels, but it is "
                f"still shutting down (pid {still[0]}).")
    return "Stopped exploring."


@tool
def exploration_status() -> str:
    """Am I exploring right now, and how did the last run go?

    Returns JSON: whether the run is active, plus the last run's outcome, log
    path and how long the hardware is still leased for.
    """
    state: dict = {"exploring": _running(),
                   "hardware_leased_for_s": round(hardware.lease_active(), 1)}
    try:
        state["last_run"] = json.loads(
            (config.ROOT / "state" / "explore_mode.json").read_text())
    except Exception:                               # noqa: BLE001
        state["last_run"] = None
    return json.dumps(state, indent=2)
