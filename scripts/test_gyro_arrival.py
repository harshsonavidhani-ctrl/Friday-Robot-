#!/usr/bin/env python3
"""The gyro at the arrival (2026-10-09): her heading narrows the looks.

    ./venv/bin/python scripts/test_gyro_arrival.py

No robot, no motor board, network blocked. 2026-10-07 21:34 the way home
"arrived" and the start's view then searched EVERY heading: the look said
"turned 137 deg" and the approach saw "two places" and refused. The run had
tracked her heading with the gyro all along. These checks pin:
  * global_localise / from_start search only near a given heading, which
    settles a look-alike turned round, and is faster;
  * the signature's rotation search keeps to the gyro's window;
  * GyroHeading latches once, follows the board's yaw, and drops out on a
    reconnect (the Uno resets when its port is opened) -- never a guess;
  * the mission latches from the run's pose only when the run had a gyro
    and File 14's engine has not moved her since;
  * the default reader never opens the motor link.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import gyro_heading as gh      # noqa: E402
from exploration_mission import home_anchor as ha       # noqa: E402
from exploration_mission import mission_manager as mmg  # noqa: E402
from exploration_mission import rth_ladder as rl        # noqa: E402
import make_maps as mm                                  # noqa: E402
import synth                                            # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


# =============================================================================
print("--- the start's view, searched near the gyro's heading ---")
occ = mm.closed_room().grid.occupied()
c, s = math.cos(0.3), math.sin(0.3)
# The start at the dead centre of a symmetric box: turned round, it looks the same.
mid = synth.grid_ray_cast(occ, (0.0, 0.0), 0.05, (3.0, 2.5, 0.3))
TRUE = (-0.7, 0.7, 1.71)
off = synth.grid_ray_cast(occ, (0.0, 0.0), 0.05,
                          (3.0 + c * TRUE[0] - s * TRUE[1], 2.5 + s * TRUE[0] + c * TRUE[1],
                           0.3 + TRUE[2]))
sig = ha.signature([mid])

t0 = time.perf_counter()
every = rl.from_start([off], sig)
t_every = time.perf_counter() - t0
check("every heading (no gyro): two places look the same -> refused, as before",
      every["pose"] is None and "two places" in every["why"], every["why"][:90])

t0 = time.perf_counter()
near = rl.from_start([off], sig, heading_rad=TRUE[2] + math.radians(12))
t_near = time.perf_counter() - t0
got = near["pose"]
check("the gyro's heading (12 deg out): the right one of the two, within 15 cm / 6 deg",
      got is not None and math.hypot(got[0] - TRUE[0], got[1] - TRUE[1]) < 0.15
      and abs(rl._wrap(got[2] - TRUE[2])) < math.radians(6), str(got))
check("...and quicker than searching every heading", t_near < t_every,
      f"{t_near:.2f} s vs {t_every:.2f} s")
far = rl.from_start([off], sig, heading_rad=TRUE[2] + math.radians(rl.GYRO_WINDOW_DEG + 25))
check("a heading further out than the window never yields the true pose "
      "(the window is a hard limit, so the heading must come from the gyro)",
      far["pose"] is None or abs(rl._wrap(far["pose"][2] - TRUE[2])) > math.radians(
          rl.GYRO_WINDOW_DEG), str(far["pose"]))

P0 = (2.0, 2.0, 0.3)


def sweep_at(dx, dy, dth):
    cc, ss = math.cos(P0[2]), math.sin(P0[2])
    return synth.grid_ray_cast(occ, (0.0, 0.0), 0.05,
                               (P0[0] + cc * dx - ss * dy, P0[1] + ss * dx + cc * dy,
                                P0[2] + dth))


start_sig = ha.signature([sweep_at(0, 0, 0)])
for true in [(0.0, 0.0, 0.0), (0.6, -0.4, math.radians(-40)),
             (-0.5, 0.3, math.radians(150))]:
    got = rl.from_start([sweep_at(*true)], start_sig,
                        heading_rad=true[2] - math.radians(8))["pose"]
    check(f"off-centre start, her pose {tuple(round(v, 2) for v in true)} with the gyro "
          "8 deg out: found", got is not None
          and math.hypot(got[0] - true[0], got[1] - true[1]) < 0.2
          and abs(rl._wrap(got[2] - true[2])) < math.radians(6), str(got))

# =============================================================================
print("--- the signature's turn, kept to the gyro's window ---")
ref = np.full(360, 3.0, dtype=np.float32)
ref[10:20] = 1.0                                   # one near thing ...
ref[190:200] = 1.0                                 # ... and its twin opposite
cur = np.roll(ref, -40)                            # turned +40 deg (cur[b] = ref[b + 40])
free = ha.similarity(cur, ref)
check("every shift: the twin makes +40 and -140 equally good (a coin toss)",
      free.match and free.shift_deg in (40.0, -140.0), str(free))
for h in (40.0, 55.0, 25.0):
    got = ha.similarity(cur, ref, heading_deg=h)
    check(f"gyro says {h:+.0f}: the shift is +40, never the twin", got.shift_deg == 40.0,
          str(got))
got = ha.similarity(cur, ref, heading_deg=-140.0 + 10.0)
check("gyro says -130: the twin (-140) is the only shift it allows", got.shift_deg == -140.0,
      str(got))
got = ha.similarity(np.roll(ref, 90), ref, heading_deg=0.0, window_deg=30.0)
check("nothing near the gyro's heading matches -> no match (not a far shift)",
      not got.match or abs(got.shift_deg) <= 30.0, str(got))
v = rl.verify([{float(b): 3000 if not (10 <= b < 20 or 190 <= b < 200) else 1000
                for b in range(360)}], [float(x) for x in ref], heading_rad=0.0)
check("verify() passes the heading through (standing on the start, gyro 0 -> 0)",
      v["verified"] and v["shift_deg"] == 0.0, str(v))

# =============================================================================
print("--- GyroHeading: latched once, then the board's yaw ---")


class Board:
    def __init__(self):
        self.drive, self.yaw, self.status, self.calls = object(), 0.0, "ok", 0

    def read(self):
        self.calls += 1
        return self.drive, {"status": self.status, "ok": self.status == "ok",
                            "yaw_deg": self.yaw}


clock = [0.0]


def fake_sleep(s):
    clock[0] += s


b = Board()
g = gh.GyroHeading(read=b.read, sleep=fake_sleep, monotonic=lambda: clock[0])
check("nothing latched -> None, and the board is not even asked",
      g.now() is None and b.calls == 0)
check("the run had no gyro -> no latch, never a guess, board not asked",
      g.latch(None) is False and g.now() is None and b.calls == 0 and "no gyro" in g.why)
b.yaw = 12.0
check("latch at +100 deg (board yaw 12)", g.latch(math.radians(100.0)) and g.latched)
b.yaw = 12.0 + 45.0                                   # she turned 45 left
check("she turned 45 left -> 145", abs(math.degrees(g.now()) - 145.0) < 1e-6)
b.yaw = 12.0 + 45.0 - 200.0                           # then 200 right
check("...then 200 right -> -55 (wrapped)", abs(math.degrees(g.now()) + 55.0) < 1e-6)
b.status = "cal"
t = clock[0]
check("a reading that is not 'ok' -> None after waiting (keeps the latch)",
      g.now() is None and g.latched and clock[0] - t >= gh.READY_WAIT_S - 0.21)
b.status = "ok"
b.drive = object()                                    # friday-mcp reconnected: board reset
check("the motor board was reconnected (its yaw restarted) -> None, latch dropped",
      g.now() is None and not g.latched and "reconnected" in g.why)


class Late(Board):
    """After the run, the link is reopened and the board measures its bias."""
    def __init__(self):
        super().__init__()
        self.seq = [None, None, "cal", "cal", "ok"]

    def read(self):
        st = self.seq.pop(0) if self.seq else "ok"
        if st is None:
            return None, None
        self.status = st
        return super().read()


lb = Late()
g = gh.GyroHeading(read=lb.read, sleep=fake_sleep, monotonic=lambda: clock[0])
check("the link not open yet, then 'cal', then 'ok': latch waits for it",
      g.latch(0.5) and g.latched)
nb = Board()
nb.status = "nogyro"
g = gh.GyroHeading(read=nb.read, sleep=fake_sleep, monotonic=lambda: clock[0])
t = clock[0]
check("'nogyro' that never clears -> no latch, after READY_WAIT_S (not forever)",
      not g.latch(0.5) and gh.READY_WAIT_S - 0.5 <= clock[0] - t <= gh.READY_WAIT_S + 0.5,
      f"waited {clock[0] - t:.1f} s")


class Reset(Late):
    """2026-10-19 17:58: just reconnected, the board's first WHO_AM_I failed --
    'nogyro' until its 2 s retry, then 'cal', then 'ok'."""
    def __init__(self):
        super().__init__()
        self.seq = ["nogyro"] * 10 + ["cal"] * 5 + ["ok"]


g = gh.GyroHeading(read=Reset().read, sleep=fake_sleep, monotonic=lambda: clock[0])
check("'nogyro' right after a reset, then 'cal', then 'ok': latch waits it out",
      g.latch(0.5) and g.latched)

saved = sys.modules.pop("mcp_server.hardware", None)
check("the default reader outside friday-mcp: no link, nothing opened",
      gh._hardware_imu() == (None, None))
opened = []
sys.modules["mcp_server.hardware"] = SimpleNamespace(
    lease_active=lambda: 0.0, _motors=None,
    motors=lambda: opened.append(1), motors_if_present=lambda: opened.append(1))
check("...inside friday-mcp with the link not open: (None, None), and it is NOT opened "
      "(opening the port resets the Uno)", gh._hardware_imu() == (None, None) and not opened)
drive = SimpleNamespace(serial=object(), imu=lambda: {"status": "ok", "ok": True,
                                                      "yaw_deg": 3.0})
sys.modules["mcp_server.hardware"] = SimpleNamespace(lease_active=lambda: 30.0,
                                                     _motors=drive)
check("...a run still holds the lease: not read", gh._hardware_imu() == (None, None))
sys.modules["mcp_server.hardware"] = SimpleNamespace(lease_active=lambda: 0.0,
                                                     _motors=drive)
d, imu = gh._hardware_imu()
check("...the link friday-mcp already holds: read", d is drive and imu["yaw_deg"] == 3.0)
sys.modules.pop("mcp_server.hardware")
if saved is not None:
    sys.modules["mcp_server.hardware"] = saved

# =============================================================================
print("--- the mission latches from the run's pose only when it can ---")


def mission(run_gyro, fix_moves, engine_moves, pose_deg=130.0, start_deg=10.0):
    board = Board()
    board.yaw = 7.0
    me = SimpleNamespace(
        heading=gh.GyroHeading(read=board.read, sleep=fake_sleep,
                               monotonic=lambda: clock[0]),
        _run_gyro=run_gyro, _fix_moves=fix_moves,
        mission={"start_theta": math.radians(start_deg)},
        pose=(0.4, -0.2, math.radians(pose_deg)),
        _engine_moves=lambda: engine_moves)
    return me, board


me, board = mission(run_gyro=3, fix_moves=2, engine_moves=2)
h = mmg.MissionManager._gyro_heading(me)
check("the run had a gyro, the engine has not moved her since its last pose: heading "
      "= pose - start (130 - 10 = 120)", h is not None
      and abs(math.degrees(h) - 120.0) < 1e-6, str(h))
board.yaw += 30.0
check("...then the engine turns her 30 left: the board's yaw carries it (150)",
      abs(math.degrees(mmg.MissionManager._gyro_heading(me)) - 150.0) < 1e-6)
me, board = mission(run_gyro=3, fix_moves=2, engine_moves=3)
check("File 14's engine moved her after the run's last pose -> None (that pose is "
      "stale), and it is not retried", mmg.MissionManager._gyro_heading(me) is None
      and mmg.MissionManager._gyro_heading(me) is None and board.calls == 0
      and "File 14" in me.heading.why)
me, board = mission(run_gyro=None, fix_moves=2, engine_moves=2)
check("the run had no gyro -> None, board not asked",
      mmg.MissionManager._gyro_heading(me) is None and board.calls == 0)
me, board = mission(run_gyro=0, fix_moves=0, engine_moves=0, pose_deg=-170.0,
                    start_deg=175.0)
check("heading wraps (-170 - 175 = +15)",
      abs(math.degrees(mmg.MissionManager._gyro_heading(me)) - 15.0) < 1e-6)

# =============================================================================
print("--- the run publishes whether it had a gyro ---")
import json                                              # noqa: E402
import tempfile                                          # noqa: E402
sys.path.insert(0, str(ROOT / "rl"))
import way_back                                          # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "live.json"
    ok = way_back._publish({0: 1000}, (0, 0, 0), None, (0, 0, 0),
                           {"pose": (0.1, 0.2, 0.3), "reason": "matched"}, 8.0,
                           path=out, gyro_gaps=4)
    pub = json.loads(out.read_text())
    ok2 = way_back._publish({0: 1000}, (0, 0, 0), None, (0, 0, 0),
                            {"pose": (0.1, 0.2, 0.3), "reason": "matched"}, 8.0, path=out)
    pub2 = json.loads(out.read_text())
check("live.json carries gyro_gaps (4), and None from a run without a gyro",
      ok and ok2 and pub["gyro_gaps"] == 4 and pub2["gyro_gaps"] is None, str(pub))

check("the network guard saw no attempt", not netguard.attempts)
print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
