"""rate_limit.py — File 08 step 10: nothing that costs money may run in a loop.

    ok, why = rate_limit.allow("vision_describe")
    if not ok:
        log.info("skipping: %s", why)

WHY A CEILING IS NOT ENOUGH. config.DAILY_SPEND_CAP_USD already stops the day
at $2 and every paying path checks it (brain.ask, brain.remark, idle_study,
narrate). But a cap is a wall at the end of the corridor, not a speed limit: a
tool called in a loop -- a model that decides to look again, and again, and
again -- spends the whole day's money in a minute and the only sign is that
Friday goes quiet at lunchtime. Harsh's step 10: "no runaway cost from a loop".

So each paying tool also gets three numbers here: how close together two calls
may be, how many in an hour, how many in a day. They live in config.RATE_LIMITS
so there is ONE place to read what Friday is allowed to spend money on, which
is the same reason every threshold in this project lives in config.py.

NOT A SAFETY MECHANISM, and deliberately not built like one. The counters are a
small JSON file written the way face_control writes the emotion (temp file,
rename), shared by every process that calls a paying tool -- main.py and the
MCP server both do. Two processes asking in the same millisecond can both be
told yes, because the alternative is a lock file that can wedge and leave
Friday unable to look at anything. One extra Haiku call costs $0.005; a wedged
lock costs an evening. Safety limits -- clearance, cliffs, the duration cap --
live in mcp_server/safety.py and are enforced in the server, never here.

RECORDING IS SEPARATE FROM ASKING. allow() only reads; used() writes the
timestamp. A tool that asks and then fails to start has not used its budget.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

import config

log = logging.getLogger("friday.rate_limit")

_lock = threading.Lock()

HOUR_S = 3600.0
DAY_S = 86400.0


def limits(name: str) -> dict[str, Any]:
    """The three numbers for `name`, or {} when it is not limited."""
    return dict(getattr(config, "RATE_LIMITS", {}).get(name) or {})


def _read() -> dict[str, list[float]]:
    try:
        raw = json.loads(config.RATE_LIMIT_FILE.read_text())
        return {k: [float(t) for t in v] for k, v in raw.items()
                if isinstance(v, list)}
    except FileNotFoundError:
        return {}
    except Exception as exc:                    # noqa: BLE001
        # A corrupt counter file must not stop Friday working. Starting the
        # day again is the cheap failure; refusing every paying call is not.
        log.warning("could not read the rate-limit counters (%s) — starting "
                    "them again", exc)
        return {}


def _write(state: dict[str, list[float]]) -> None:
    try:
        path = config.RATE_LIMIT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ratelimit.",
                                   suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle)
        os.replace(tmp, path)
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not write the rate-limit counters: %s", exc)


def _recent(state: dict[str, list[float]], name: str, now: float) -> list[float]:
    """This tool's calls within the last day, oldest first."""
    return sorted(t for t in state.get(name, []) if now - t < DAY_S)


def allow(name: str, now: float | None = None) -> tuple[bool, str]:
    """May `name` run right now? Returns (ok, why not).

    An unlimited name is always allowed: adding a tool without a limit is a
    decision, not an accident, and check_limits() in the smoke test is what
    makes it a visible one.
    """
    rule = limits(name)
    if not rule:
        return True, ""
    now = time.time() if now is None else now
    with _lock:
        recent = _recent(_read(), name, now)

    gap = float(rule.get("min_interval_s", 0.0))
    if gap and recent and now - recent[-1] < gap:
        return False, (f"{name}: {gap - (now - recent[-1]):.0f}s left of its "
                       f"{gap:.0f}s gap")
    per_hour = int(rule.get("per_hour", 0))
    if per_hour:
        count = sum(1 for t in recent if now - t < HOUR_S)
        if count >= per_hour:
            return False, f"{name}: {count} calls this hour, limit {per_hour}"
    per_day = int(rule.get("per_day", 0))
    if per_day and len(recent) >= per_day:
        return False, f"{name}: {len(recent)} calls today, limit {per_day}"
    return True, ""


def used(name: str, now: float | None = None) -> None:
    """Record that `name` actually ran. Never raises."""
    if not limits(name):
        return
    now = time.time() if now is None else now
    with _lock:
        state = _read()
        state[name] = _recent(state, name, now) + [now]
        _write(state)


def count(name: str, window_s: float = DAY_S) -> int:
    """How many times `name` has run inside `window_s`. For status pages."""
    now = time.time()
    with _lock:
        return sum(1 for t in _recent(_read(), name, now) if now - t < window_s)


def status() -> dict[str, dict[str, Any]]:
    """Every limited tool: its rule, and what it has used. Never raises."""
    report: dict[str, dict[str, Any]] = {}
    try:
        now = time.time()
        with _lock:
            state = _read()
        for name, rule in getattr(config, "RATE_LIMITS", {}).items():
            recent = _recent(state, name, now)
            report[name] = {
                "rule": rule,
                "last_hour": sum(1 for t in recent if now - t < HOUR_S),
                "today": len(recent),
                "allowed_now": allow(name, now)[0],
            }
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not build the rate-limit status: %s", exc)
    return report


def main() -> int:
    """What Friday has spent her allowances on today."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for name, row in sorted(status().items()):
        rule = row["rule"]
        print(f"{name:22s} {row['last_hour']:3d}/h  {row['today']:4d}/day   "
              f"limit {rule.get('min_interval_s', 0):.0f}s gap, "
              f"{rule.get('per_hour', 0)}/h, {rule.get('per_day', 0)}/day"
              f"{'' if row['allowed_now'] else '   <== BLOCKED NOW'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
