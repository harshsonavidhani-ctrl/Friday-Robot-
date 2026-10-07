#!/usr/bin/env python3
"""turn_degrees measured by the gyro (movement._gyro_turn + safety.guarded_move's
done check). No hardware: a fake board whose yaw moves only while a pivot is
commanded. Nothing is written to state/.

    ./venv/bin/python scripts/test_gyro_turn.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402
from mcp_server import embodiment, hardware, safety     # noqa: E402
from mcp_server.tools import movement                   # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if not ok and detail else ""))


class Board:
    """The motor board: yaw (+ = left) integrates at rate_dps while a pivot is
    commanded, plus `coast` degrees after the stop (momentum)."""

    def __init__(self, rate_dps=75.0, stalled=False, coast=2.0, gyro=True):
        self.rate, self.stalled, self.coast, self.gyro = rate_dps, stalled, coast, gyro
        self.yaw, self.cmd, self.t, self.stops, self.drives = 0.0, (0, 0), None, 0, []

    def _advance(self):
        now = time.monotonic()
        if self.t is not None and self.cmd != (0, 0) and not self.stalled:
            left, right = self.cmd
            sense = 1.0 if right > left else -1.0          # right wheel faster = left turn
            self.yaw += sense * self.rate * (now - self.t)
        self.t = now

    def drive(self, left, right):
        self._advance()
        self.cmd = (left, right)
        self.drives.append((left, right))

    def stop(self):
        self._advance()
        if self.cmd != (0, 0) and not self.stalled:
            left, right = self.cmd
            self.yaw += (1.0 if right > left else -1.0) * self.coast
        self.cmd = (0, 0)
        self.stops += 1

    def keepalive(self):
        self._advance()

    def imu(self):
        if not self.gyro:
            return None
        self._advance()
        rate = 0.0 if (self.cmd == (0, 0) or self.stalled) else self.rate
        return {"status": "ok", "ok": True, "yaw_deg": self.yaw, "rate_dps": rate,
                "gaps": 0, "errors": 0}


board = Board()
hardware.motors = lambda: board
embodiment.check = lambda direction: None
embodiment.note_moved = lambda: None
movement.GYRO_TURN_SETTLE_S = 0.05
_real_cal = config.load_calibration
config.load_calibration = lambda: {**_real_cal(), "turn_deg_per_s": 0.0}

print("--- turn_degrees, closed on the gyro ---")
out = movement.turn_degrees(90)
check("90 right: turns until the gyro says ~90 (coast included), then reports the "
      "MEASURED angle", "measured by the gyro" in out and 84 <= -board.yaw <= 96
      and board.stops >= 1, f"{out!r} yaw {board.yaw:.1f}")
check("...the pivot was a right turn (left wheel forward, right back)",
      board.drives and board.drives[0][0] > 0 > board.drives[0][1], str(board.drives[:1]))
board.__init__()
out = movement.turn_degrees(-45)
check("45 left: + = left on the board's yaw, stops near 45",
      40 <= board.yaw <= 51 and "left" in out.lower(), f"{out!r} yaw {board.yaw:.1f}")
board.__init__(stalled=True)
t0 = time.monotonic()
out = movement.turn_degrees(60)
check("a turn that does not happen (wheels slip): runs out its time budget and SAYS "
      "it only turned 0 -- no more pretending", "Only turned 0 of 60" in out
      and time.monotonic() - t0 < 6.0, out)
board.__init__(gyro=False)
out = movement.turn_degrees(30)
check("no gyro answering: the old timed pivot (and its honest refusal while the "
      "turn rate is unmeasured)", "cannot do angles yet" in out, out)
board.__init__()
out = movement.turn_degrees(0.4)
check("under a degree: nothing to do", "nothing to turn" in out, out)

print("--- guarded_move's done check can only end a move sooner ---")
board.__init__()
r = safety.guarded_move("left", 2.0, done=lambda: "enough")
check("done() says finished at once: complete, not 'stopped early', motors stopped",
      not r.stopped_early and r.reason == "enough" and r.actual_s < 0.3
      and board.stops == 1, str(r))
board.__init__()
calls = []
r = safety.guarded_move("left", 0.3, done=lambda: (calls.append(1), 1 / 0)[1])
check("a done() that raises is ignored: the move runs its time, as without it",
      not r.stopped_early and r.actual_s >= 0.28 and calls, str(r))
board.__init__()
r = safety.guarded_move("left", 0.4, done=lambda: None, poll_s=5.0)
check("poll_s can never LENGTHEN a pass (5 s asked, 0.1 s kept): the move still "
      "ends on time", 0.35 <= r.actual_s < 0.6, str(r))
board.__init__()
threading.Timer(0.15, hardware.ABORT.set).start()
r = safety.guarded_move("left", 2.0, done=lambda: None)
check("stop() still wins (ABORT), with a done check in place",
      r.stopped_early and r.reason == "stop was called", str(r))
hardware.ABORT.clear()

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
