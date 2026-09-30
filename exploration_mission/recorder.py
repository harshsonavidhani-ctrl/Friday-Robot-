"""Record what an exploration run publishes, for replay (File 21).

    ./venv/bin/python -m exploration_mission.recorder          # then say "explore"

The RL run's tracker publishes its latest fix once a second
(state/exploration_mission/live.json: the start tracker's pose, whether the
sweep matched, the run's dead-reckoned pose, and the sweep itself) and each
one overwrites the last. This keeps every one of them, one JSON line each,
with the run's own status beside it, in

    state/exploration_mission/recordings/run-YYYYmmdd-HHMMSS.jsonl

WHY. The start tracker matches every sweep against the START's sweep only, so
its pose is good in sight of the start and dead reckoning beyond it. Which way
of building the mission's map from that holds up in a real room has to be
MEASURED on a real run before rooms are segmented on it, not guessed -- and a
replay needs the sweeps.

It only reads files. It never moves the robot, holds the run, or talks to the
LiDAR. It waits for a run to start publishing, records until the run has been
silent for `idle_stop_s`, and exits. Nothing is ever deleted.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from exploration_mission import adapters


def record(run=None, out_dir: Path | None = None, poll_s: float = 0.25,
           idle_stop_s: float = 15.0, wait_s: float | None = None,
           fresh_s: float = 5.0, sleep: Callable[[float], None] = time.sleep,
           say: Callable[[str], None] = print) -> Path | None:
    """Wait for a run, record it, return the file (None if none started)."""
    run = run or adapters.RunAdapter()
    clock = run.clock
    out_dir = Path(out_dir) if out_dir else run.state_dir / "recordings"
    waited = 0.0
    say("waiting for an exploration run to publish...")
    while run.live(fresh_s) is None:
        if wait_s is not None and waited >= wait_s:
            say("no run started; nothing recorded")
            return None
        sleep(poll_s)
        waited += poll_s

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / time.strftime("run-%Y%m%d-%H%M%S.jsonl",
                                   time.localtime(clock.wall()))
    last_t, last_new, rows, last_sync = None, clock.monotonic(), 0, clock.monotonic()
    say(f"recording to {path}")
    with open(path, "a", encoding="utf-8") as out:
        while True:
            fix = run.live()
            now = clock.monotonic()
            if fix is not None and fix.get("t") != last_t:
                last_t, last_new = fix.get("t"), now
                fix.pop("age_s", None)
                fix["scan"] = {str(k): v for k, v in fix["scan"].items()}
                fix["status"] = run.status()
                out.write(json.dumps(fix) + "\n")
                out.flush()
                rows += 1
            if now - last_sync >= 10.0:
                os.fsync(out.fileno())
                last_sync = now
            if now - last_new >= idle_stop_s:
                break
            sleep(poll_s)
        out.flush()
        os.fsync(out.fileno())
    say(f"run went quiet: {rows} fixes recorded in {path.name}")
    return path


def load(path: Path) -> list[dict]:
    """A recording back as a list of fixes (scan keys as ints)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue                      # a torn last line after a power cut
        row["scan"] = {int(k): int(v) for k, v in (row.get("scan") or {}).items()}
        rows.append(row)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wait", type=float, default=None,
                    help="give up if no run starts within this many seconds")
    ap.add_argument("--idle", type=float, default=15.0,
                    help="stop after the run has published nothing for this long")
    args = ap.parse_args()
    try:
        record(wait_s=args.wait, idle_stop_s=args.idle)
    except KeyboardInterrupt:
        print("stopped")
