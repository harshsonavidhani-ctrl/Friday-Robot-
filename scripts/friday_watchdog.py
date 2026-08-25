#!/usr/bin/env python3
"""friday_watchdog.py — when Friday crashes, try to work out why.

Watches Friday's own services. When one starts failing, it collects the logs
from around the failure and hands them to dev_task, which diagnoses and fixes
on a branch, proves the fix against the smoke tests, and only then lets it go
live. A fix that touches motors, power or safety stops and asks Harsh instead.

    ./venv/bin/python scripts/friday_watchdog.py            # run it
    ./venv/bin/python scripts/friday_watchdog.py --once     # one pass, for testing
    ./venv/bin/python scripts/friday_watchdog.py --dry-run  # diagnose, change nothing

RESTRAINT IS THE HARD PART. An automatic fixer that reacts to everything is
worse than none: it burns money, it churns the code, and it can turn one broken
service into a broken robot. So:

  * A service must fail REPEATEDLY (config.WATCHDOG_FAILURES_BEFORE_FIX) before
    anything is attempted. Most crashes are a one-off -- a USB device that
    dropped, a camera that was busy -- and systemd's own restart already fixed
    them by the time anyone looked.
  * At most config.WATCHDOG_FIXES_PER_DAY attempts a day, over and above the
    spend cap dev_task already enforces.
  * The same service is not retried within the cooldown, so a fix that did not
    work cannot become a loop.
  * If the fix does not hold, the watchdog STOPS TRYING and says so. Two failed
    attempts at the same service means the problem is not one it understands,
    and continuing would just be damage with a schedule.

It never touches a unit it did not create, never edits anything outside the
project, and cannot bypass any of dev_task's rails -- it is only a caller.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                               # noqa: E402

log = logging.getLogger("friday.watchdog")

UNITS = ("friday-mcp", "friday-dashboard", "friday-face")
STATE = ROOT / "state" / "watchdog.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2))
    except OSError as exc:
        log.warning("could not save watchdog state: %s", exc)


def _unit_health(unit: str) -> dict:
    """What systemd thinks of one of Friday's services."""
    out = subprocess.run(
        ["systemctl", "--user", "show", f"{unit}.service",
         "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
         "-p", "ExecMainStatus"],
        capture_output=True, text=True).stdout
    fields = dict(line.split("=", 1) for line in out.strip().splitlines()
                  if "=" in line)
    return {
        "unit": unit,
        "active": fields.get("ActiveState", "unknown"),
        "sub": fields.get("SubState", "unknown"),
        "restarts": int(fields.get("NRestarts", "0") or 0),
        "exit_status": fields.get("ExecMainStatus", "0"),
    }


def _recent_logs(unit: str, lines: int = 80) -> str:
    return subprocess.run(
        ["journalctl", "--user", "-u", f"{unit}.service", "-n", str(lines),
         "--no-pager"], capture_output=True, text=True).stdout.strip()


def _is_failing(health: dict, previous: dict) -> bool:
    """Failing means failed outright, or restarting more than it was."""
    if health["active"] == "failed":
        return True
    if health["sub"] == "auto-restart":
        return True
    return health["restarts"] > previous.get("restarts", health["restarts"])


def _diagnose(unit: str, health: dict, logs: str, dry_run: bool) -> str:
    instruction = (
        f"Friday's {unit}.service is failing and needs fixing.\n\n"
        f"systemd reports: active={health['active']} sub={health['sub']} "
        f"restarts={health['restarts']} exit={health['exit_status']}\n\n"
        f"The last log lines are below. Work out the ROOT CAUSE and fix it in "
        f"Friday's code. Do not paper over it by catching and ignoring the "
        f"error, and do not change any systemd unit -- the bug is almost "
        f"certainly in the Python.\n\n"
        f"--- journalctl -u {unit}.service ---\n{logs[-4000:]}\n")

    if dry_run:
        return f"[dry run] would have asked dev_task to fix {unit}:\n{instruction[:500]}…"

    from mcp_server.tools import dev
    return dev.dev_task(instruction)


def check_once(dry_run: bool = False) -> int:
    state = _load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("date") != today:
        state = {"date": today, "fixes_today": 0, "units": {}}
    units_state = state.setdefault("units", {})

    acted = 0
    for unit in UNITS:
        health = _unit_health(unit)
        previous = units_state.get(unit, {})
        record = dict(previous)
        record["restarts"] = health["restarts"]
        record["last_seen"] = time.time()

        if not _is_failing(health, previous):
            record["consecutive_failures"] = 0
            units_state[unit] = record
            continue

        failures = previous.get("consecutive_failures", 0) + 1
        record["consecutive_failures"] = failures
        units_state[unit] = record
        print(f"{unit}: {health['active']}/{health['sub']} — "
              f"failure {failures} of {config.WATCHDOG_FAILURES_BEFORE_FIX} "
              f"before a fix is attempted")

        if failures < config.WATCHDOG_FAILURES_BEFORE_FIX:
            continue
        if previous.get("attempts", 0) >= config.WATCHDOG_MAX_ATTEMPTS_PER_UNIT:
            print(f"  {unit} has already been fixed "
                  f"{previous['attempts']} time(s) without holding. "
                  f"NOT trying again — this needs Harsh.")
            continue
        if state.get("fixes_today", 0) >= config.WATCHDOG_FIXES_PER_DAY:
            print(f"  at the daily limit of {config.WATCHDOG_FIXES_PER_DAY} "
                  f"fix attempts. Waiting until tomorrow.")
            continue
        since = time.time() - previous.get("last_attempt", 0)
        if since < config.WATCHDOG_COOLDOWN_S:
            print(f"  cooling down — {config.WATCHDOG_COOLDOWN_S - since:.0f}s "
                  f"before {unit} may be retried")
            continue

        print(f"  asking dev_task to diagnose {unit}…")
        result = _diagnose(unit, health, _recent_logs(unit), dry_run)
        print("  " + result.replace("\n", "\n  "))
        if not dry_run:
            record["attempts"] = previous.get("attempts", 0) + 1
            record["last_attempt"] = time.time()
            record["consecutive_failures"] = 0
            units_state[unit] = record
            state["fixes_today"] = state.get("fixes_today", 0) + 1
        acted += 1

    _save_state(state)
    return acted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="diagnose but change nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    if args.once or args.dry_run:
        check_once(dry_run=args.dry_run)
        return 0

    print(f"watching {', '.join(UNITS)} every {config.WATCHDOG_INTERVAL_S}s")
    while True:
        try:
            check_once()
        except Exception as exc:                # noqa: BLE001
            log.error("watchdog pass failed: %s", exc)
        time.sleep(config.WATCHDOG_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
