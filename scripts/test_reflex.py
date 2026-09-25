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

    # --- 3b. but only for driving AT it (2026-09-25 22:43) ----------------
    # Pinned 15 cm from a box, every turn and reverse was killed after 0.2 s.
    for away in ("backward", "left", "right"):
        check(f"a box ahead does not stop her going {away}",
              reflex._hazard(away) is None)
    check("it still stops her driving forward into it",
          (reflex._hazard("forward") or ("",))[0] == "obstacle")
    check("and an unknown direction still gets the full checks",
          (reflex._hazard("") or ("",))[0] == "obstacle"
          and (reflex._hazard("sideways") or ("",))[0] == "obstacle")

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

    # --- 5b. the running loop lets her back away ----------------------------
    hardware.ranges = lambda: _FakeRanges(config.SAFETY_STOP_MM - 50)
    reflex._cliff_gap_mm = lambda: 120
    hardware.ABORT.clear()
    backing = fake.stops
    hardware._last_drive = (-120.0, -120.0)     # both wheels backward
    hardware.MOVING.set()
    time.sleep(0.4)
    hardware.MOVING.clear()
    check("the running loop leaves a reverse away from the box alone",
          not hardware.ABORT.is_set() and fake.stops == backing)
    hardware._last_drive = (120.0, 120.0)       # both wheels forward
    hardware.MOVING.set()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not hardware.ABORT.is_set():
        time.sleep(0.02)
    hardware.MOVING.clear()
    check("and stops a forward move into it", hardware.ABORT.is_set())
    hardware._last_drive = None

    # The direction is read off the command, pre-inversion.
    for command, want in (((120, 120), "forward"), ((-120, -120), "backward"),
                          ((120, -120), "right"), ((-120, 120), "left"),
                          ((0, 120), ""), (None, "")):
        hardware._last_drive = command
        check(f"command {command} reads as {want or 'unknown'}",
              hardware.moving_direction() == want)
    hardware._last_drive = None

    class Board:
        def __init__(self):
            self.calls = []
        def drive(self, left, right):
            self.calls.append(("drive", left, right))
        def stop(self):
            self.calls.append(("stop",))
        serial = "port"
    board = Board()
    wrapped = hardware._Commanded(board)
    wrapped.drive(-100, -100)
    check("the wrapper passes the command through and remembers it",
          board.calls == [("drive", -100, -100)]
          and hardware.moving_direction() == "backward"
          and wrapped.serial == "port")
    wrapped.stop()
    check("and forgets it on stop, so the next move starts as unknown",
          board.calls[-1] == ("stop",) and hardware.moving_direction() == "")

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
