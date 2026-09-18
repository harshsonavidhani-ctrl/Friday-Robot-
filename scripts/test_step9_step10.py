#!/usr/bin/env python3
"""File 08 steps 9 and 10: the gates and the limits. No motors, no money.

    ./venv/bin/python scripts/test_step9_step10.py

⚠ NOTHING HERE DRIVES THE ROBOT. Step 9's whole risk is that it moves with
nobody asking, so every case below stops at the decision -- _why_not_now() and
the gate helpers -- and the one test that goes further hands it a FAKE MCP
session that records the calls instead of making them. The real outing is
tested on the floor, with Harsh watching, by turning the flag on.

The cases that matter are the ones that must say NO.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

config.RATE_LIMIT_FILE = ROOT / "state" / "rate_limits.test.json"
config.RATE_LIMIT_FILE.unlink(missing_ok=True)

import physical_curiosity as pc                         # noqa: E402
import rate_limit                                       # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


class FakeSession:
    """An MCP session that records calls instead of making them."""

    def __init__(self, battery: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.battery = battery if battery is not None else {"power": {}}

    async def call_tool(self, name: str, args: dict | None = None):
        self.calls.append((name, args or {}))
        text = json.dumps(self.battery) if name == "get_battery" else "{}"
        return type("R", (), {"content": [type("T", (), {"text": text})()]})()


def main() -> int:
    import asyncio

    print("STEP 10 — the rate limiter")
    config.RATE_LIMITS["_test"] = {"min_interval_s": 60.0, "per_hour": 2,
                                   "per_day": 3}
    now = time.time()
    check("a tool with no rule is always allowed",
          rate_limit.allow("_unlisted")[0])
    check("the first call is allowed", rate_limit.allow("_test", now)[0])
    rate_limit.used("_test", now)
    ok, why = rate_limit.allow("_test", now + 5)
    check("a second call inside the gap is refused", not ok, why)
    check("...and it says how long is left", "left of its" in why, why)
    rate_limit.used("_test", now + 61)
    ok, why = rate_limit.allow("_test", now + 122)
    check("the third call this hour is refused", not ok, why)
    ok, _ = rate_limit.allow("_test", now + 3700)
    check("an hour later it is allowed again", ok)
    rate_limit.used("_test", now + 3700)
    ok, why = rate_limit.allow("_test", now + 7400)
    check("but the daily limit still holds", not ok, why)
    ok, _ = rate_limit.allow("_test", now + 90000)
    check("tomorrow it is allowed again", ok)

    config.RATE_LIMIT_FILE.write_text("{ not json at all")
    ok, _ = rate_limit.allow("_test", now)
    check("a corrupt counter file fails OPEN, never closed", ok,
          "a broken counter must not stop Friday working")
    config.RATE_LIMIT_FILE.unlink(missing_ok=True)

    print("\nSTEP 9 — the gates that must say no")
    config.PHYSICAL_CURIOSITY = False
    check("off by default in the committed config",
          "\nPHYSICAL_CURIOSITY = False" in (ROOT / "config.py").read_text())
    check("switched off means switched off",
          pc._why_not_now() == "switched off", pc._why_not_now() or "")

    config.PHYSICAL_CURIOSITY = True
    pc.turn_ended()
    # Pin the clock to the afternoon: the daylight gate is checked first (it
    # is the cheapest), so at 2 am it would answer for the quiet gate.
    daylight = pc._daylight
    pc._daylight = lambda now=None: True
    try:
        why = pc._why_not_now()
        check("she will not go the moment a turn ends",
              why is not None and "quiet for" in why, why or "went anyway")
        pc._last_activity = time.monotonic() - 99999
        why = pc._why_not_now()
        check("after a long quiet, nothing else is in the way",
              why is None, why or "")
    finally:
        pc._daylight = daylight

    pc._last_activity = time.monotonic() - 99999
    check("daylight only: 03:00 is refused",
          not pc._daylight(datetime.datetime(2026, 9, 19, 3, 0)))
    check("daylight only: 14:00 is fine",
          pc._daylight(datetime.datetime(2026, 9, 19, 14, 0)))

    state = ROOT / "state" / "explore_mode.json"
    saved = state.read_text() if state.exists() else None
    try:
        state.write_text(json.dumps({"state": "running"}))
        check("she stands down while something else is driving",
              pc._already_driving())
        state.write_text("{ broken")
        check("an unreadable explore state counts as BUSY, not free",
              pc._already_driving(),
              "not knowing whether something is driving is not permission")
        state.write_text(json.dumps({"state": "completed"}))
        check("a finished run does not block her", not pc._already_driving())
    finally:
        if saved is not None:
            state.write_text(saved)

    print("\nSTEP 9 — power, and stopping")
    brownout = FakeSession({"power": {"under_voltage_now": True}})
    check("no driving while the supply is browning out",
          not asyncio.run(pc._power_is_fine(brownout)))
    throttled = FakeSession({"power": {"throttled_now": True}})
    check("no driving while throttled",
          not asyncio.run(pc._power_is_fine(throttled)))
    healthy = FakeSession({"power": {"under_voltage_now": False,
                                     "throttled_now": False}})
    check("healthy power is fine", asyncio.run(pc._power_is_fine(healthy)))

    class Broken(FakeSession):
        async def call_tool(self, name, args=None):
            raise RuntimeError("server is gone")

    check("unreadable power is NOT permission",
          not asyncio.run(pc._power_is_fine(Broken())))

    check("heard() is a no-op when she is not out — and never raises",
          pc.heard() is None)

    # A REAL RUNNING LOOP IN ANOTHER THREAD, because that is what heard() is
    # talking to in production: the curiosity thread, busy awaiting the
    # outing. run_coroutine_threadsafe onto a loop nobody is running schedules
    # a task that never executes -- which is what the first version of this
    # test proved about the test, not about the code.
    import threading
    stopped = FakeSession()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def spin() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    ready.wait(2.0)
    pc._loop, pc._session, pc._out = loop, stopped, True
    pc.heard()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and ("stop", {}) not in stopped.calls:
        time.sleep(0.02)
    loop.call_soon_threadsafe(loop.stop)
    spinner.join(timeout=2.0)
    loop.close()
    pc._loop, pc._session, pc._out = None, None, False
    check("a wake word calls stop() on the live session",
          ("stop", {}) in stopped.calls, str(stopped.calls))
    check("...and the wake also restarts the quiet clock",
          pc.quiet_for() < 5.0)

    config.PHYSICAL_CURIOSITY = False
    config.RATE_LIMITS.pop("_test", None)
    config.RATE_LIMIT_FILE.unlink(missing_ok=True)

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
