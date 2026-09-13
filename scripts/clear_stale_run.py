#!/usr/bin/env python3
"""Mark an exploration run that died without writing its own ending as ended.

rl/modes/explore.sh writes state/explore_mode.json as "running" when a run
starts and rewrites it on every exit path it can see -- but a POWER CUT is not
one of them (see friday-crashes-are-power-cuts). The file then says "running"
forever, and everything that reads it believes her:

  - main.py keeps the wake word asleep ("exploring -- wake word off until she
    parks"), so she boots deaf. 2026-09-13 16:03: the Pi came back from a cut
    at 15:51 mid-run and ignored "hey Friday" until the file was corrected.
  - the dashboard refuses manual drive ("no hands until she parks").

WHY A PROCESS CHECK AND NOT A BOOT-TIME CHECK: this Pi has no RTC. Right after
a power cut the clock comes back from fake-hwclock, possibly an hour behind,
so "started before boot" cannot be trusted until NTP syncs. Every run -- the
units, the dashboard button, a run started by hand -- goes through
rl/modes/explore.sh, which stays alive for the whole run. No such process
means no run, whatever the file says. That also makes this safe to run while
a real run is out: it finds the process and leaves the file alone.

Run from friday-mcp.service's ExecStartPre; harmless to run by hand any time.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "state" / "explore_mode.json"
TELEMETRY = ROOT / "state" / "rl_autonomy.json"
RUN_MARKER = "explore.sh"


def run_process_alive() -> bool:
    """Is a process EXECUTING explore.sh? Only argv[0] (run directly) and
    argv[1] (`bash explore.sh`) count. Matching the string anywhere in the
    command line matched shells that merely MENTION the path -- a
    `bash -c "... modes/explore.sh ..."` from a Claude session or a grep --
    and would have left a stale file in place.
    """
    me = os.getpid()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            argv = Path(entry.path, "cmdline").read_bytes().split(b"\0")[:2]
        except OSError:
            continue
        if any(os.path.basename(a.decode(errors="replace")) == RUN_MARKER for a in argv):
            return True
    return False


def main() -> int:
    try:
        status = json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return 0
    if status.get("state") != "running":
        return 0
    if run_process_alive():
        print(f"explore_mode.json says running and {RUN_MARKER} is alive: left alone")
        return 0

    # The last telemetry the run wrote is the best record of when it died.
    ended = status.get("started")
    try:
        started = datetime.fromisoformat(status["started"]).timestamp()
        last = TELEMETRY.stat().st_mtime
        if last > started:
            ended = datetime.fromtimestamp(last).astimezone().isoformat(timespec="seconds")
    except (KeyError, TypeError, ValueError, OSError):
        pass

    status["state"] = "interrupted"
    status["ended"] = ended
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, STATUS)
    print(f"explore_mode.json said running with no run alive: marked interrupted (ended {ended})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
