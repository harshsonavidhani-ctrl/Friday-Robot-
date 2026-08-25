#!/usr/bin/env python3
"""test_reflex.py — prove the fast loop stops the motors, and that nothing can stop IT.

Run with the service stopped (it opens the serial ports itself):

    systemctl --user stop friday-mcp
    ./venv/bin/python scripts/test_reflex.py

The hazards are FAKED, deliberately. Testing a cliff detector by driving the
real robot at a real staircase proves the same thing once and then you have no
robot. Fake sensors let every branch run, including the ones you would never
dare stage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                               # noqa: E402
from mcp_server import hardware, reflex                     # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail:
        print(f"        {detail}")


class FakeMotors:
    """Records stop() calls instead of driving anything."""
    def __init__(self):
        self.stops = 0
    def stop(self):
        self.stops += 1


def main() -> int:
    print("REFLEX — the fast loop, tested with fake hazards\n")

    fake = FakeMotors()
    hardware.motors_if_present = lambda: fake                # noqa: E731

    # --- 1. a cliff fires the reflex ---------------------------------------
    reflex._cliff_gap_mm = lambda: config.CLIFF_DROP_MM + 200
    hazard = reflex._hazard()
    check("a drop ahead is a hazard", hazard is not None and hazard[0] == "cliff",
          hazard[1] if hazard else "no hazard reported")

    # --- 2. solid floor is not a hazard ------------------------------------
    reflex._cliff_gap_mm = lambda: 120
    hardware.ranges = lambda: _FakeRanges(800)               # noqa: E731
    check("solid floor and clear ahead is not a hazard",
          reflex._hazard() is None)

    # --- 3. an obstacle fires ----------------------------------------------
    hardware.ranges = lambda: _FakeRanges(config.SAFETY_STOP_MM - 50)
    hazard = reflex._hazard()
    check("an obstacle inside the stop distance is a hazard",
          hazard is not None and hazard[0] == "obstacle",
          hazard[1] if hazard else "")

    # --- 4. UNKNOWN IS NOT CLEAR -------------------------------------------
    hardware.ranges = lambda: _FakeRanges(None)
    hazard = reflex._hazard()
    check("losing the sensors is a hazard, not permission",
          hazard is not None and hazard[0] == "blind",
          hazard[1] if hazard else "")

    # --- 5. the loop actually stops the motors -----------------------------
    hardware.ABORT.clear()
    before = fake.stops
    reflex.start()
    hardware.MOVING.set()                       # pretend the wheels are turning
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not hardware.ABORT.is_set():
        time.sleep(0.02)
    fired = hardware.ABORT.is_set()
    hardware.MOVING.clear()
    check("the running loop set ABORT within 3s", fired,
          f"ABORT={hardware.ABORT.is_set()}")
    check("and it stopped the motors", fake.stops > before,
          f"stop() called {fake.stops - before} time(s)")

    # --- 6. it does NOT act while parked -----------------------------------
    hardware.ABORT.clear()
    parked = fake.stops
    time.sleep(0.4)                             # MOVING is clear
    check("it leaves a parked robot alone",
          not hardware.ABORT.is_set() and fake.stops == parked,
          "a wall in front of a stationary robot is not an emergency")

    # --- 7. a trip is remembered and reportable ----------------------------
    status = reflex.status()
    check("the trip is reported in status",
          status["trips_since_start"] > 0 and "last_trip" in status,
          f"{status.get('last_trip', {}).get('reason')}: "
          f"{status.get('last_trip', {}).get('detail', '')[:60]}")

    # --- 8. it survives a sensor that throws -------------------------------
    def explode():
        raise RuntimeError("sensor on fire")
    reflex._cliff_gap_mm = explode
    time.sleep(0.3)
    check("a throwing sensor does not kill the loop",
          reflex.status()["running"],
          "a reflex that can die is not a reflex")

    reflex.stop()
    hardware.ABORT.clear()

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


class _FakeRanges:
    def __init__(self, forward_mm):
        self._f = forward_mm
    def forward_clearance_mm(self):
        return self._f, "fake"


if __name__ == "__main__":
    raise SystemExit(main())
