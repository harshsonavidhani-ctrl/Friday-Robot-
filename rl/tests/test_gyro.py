#!/usr/bin/env python3
"""The gyro in the RL run (firmware GYRO 2+, Harsh 2026-10-07). No hardware.

  * FridayMotorDriver: G asked each tick, read on the next drain, whole lines,
    sign applied ONCE (motor.parse_imu), last_wire untouched.
  * ControlLoop: dead reckoning takes the turn she MADE; a commanded turn that
    did not happen is not counted; gaps fall back without counting twice; a
    stalled pivot is flagged; picked up mid-run = event published + wheels still.
  * StartTracker: a turn the gyro measured is searched narrowly round ALL of
    it -- on the real room, with turns that go 40% further than commanded
    (the dead reckoning's blind side), it still ends near the start.

    ./venv/bin/python rl/tests/test_gyro.py        (~2 min: the real-room runs)
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE_T = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_T.parent))
sys.path.insert(0, str(HERE_T.parent.parent))

import config as friday_config                           # noqa: E402  (the project's)
TMP = Path(tempfile.mkdtemp(prefix="friday-gyro-"))
friday_config.BODY_FILE = TMP / "body.json"

import way_back as wb                                    # noqa: E402
from pi_deploy import (ControlLoop, FridayMotorDriver, HERE,  # noqa: E402
                       NullMotorDriver, PolicyRunner, RobotConfig, SensorSlot)

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    if not ok:
        fails += 1


rcfg = RobotConfig()
rcfg.tof_enabled = False
rcfg.cam_enabled = False

print("--- the run's motor driver asks the board for G ---")


class FakeSerial:
    def __init__(self):
        self.written, self.inbox, self.is_open = [], b"", True

    def write(self, data):
        self.written.append(data)

    @property
    def in_waiting(self):
        return len(self.inbox)

    def read(self, n):
        out, self.inbox = self.inbox[:n], self.inbox[n:]
        return out


drv = FridayMotorDriver.__new__(FridayMotorDriver)
drv._lock = threading.Lock()
drv._ser, drv._closed = FakeSerial(), False
drv._rx, drv.last_imu, drv.last_imu_t = b"", None, 0.0
drv.imu_replies = drv.tilt_replies = drv._err_count = drv._cmds = 0
drv._imu_parse, drv.last_wire = None, "M120 120"
drv.request_imu()
check("G goes on the wire, and last_wire / the command count are untouched",
      drv._ser.written == [b"G\n"] and drv.last_wire == "M120 120" and drv._cmds == 0)
drv._ser.inbox = b"OK\nG 9000 -1500 ok 0 0 64 3 9 -25 0 0\nG 9"
drv.request_imu()
imu = drv.imu()
check("the reply is read on the next drain; the sign is config.GYRO_SIGN's "
      "(+1: + = left, measured with her LiDAR)",
      imu is not None and imu["yaw_deg"] == 90.0 * friday_config.GYRO_SIGN
      and imu["rate_dps"] == -15.0 * friday_config.GYRO_SIGN
      and drv.imu_replies == 1, str(imu))
check("...the half line waits for the rest", drv._rx == b"G 9")
drv._ser.inbox = b"100 0 ok 0 0 64 3 9 -25 0 0\nTILT\nERR\n"
drv.request_imu()
check("...completed next time; TILT and ERR counted as before",
      drv.last_imu["yaw_deg"] == 91.0 * friday_config.GYRO_SIGN and drv.tilt_replies == 1
      and drv._err_count == 1, str(drv.last_imu))
drv.last_imu_t -= 1.0
check("a reading older than 0.5 s is not used", drv.imu() is None)
check("NullMotorDriver (--simulate, the tests) has no gyro: the loop never asks it",
      not hasattr(NullMotorDriver(rcfg), "request_imu"))


print("--- the loop: the turn she made, not the turn she was told ---")


class GyroDriver(NullMotorDriver):
    """NullMotorDriver + a scripted board: next_yaw is what the next G says."""

    def __init__(self, c):
        super().__init__(c)
        self.yaw, self.imu_replies, self.extra = 0.0, 0, {}
        self.fresh, self.ok = True, True

    def request_imu(self):
        pass

    def imu(self, max_age_s=0.5):
        if not self.fresh:
            return None
        return {"status": "ok" if self.ok else "cal", "ok": self.ok,
                "yaw_deg": self.yaw, "rate_dps": 0.0, "gaps": 0, "errors": 0,
                "tilt_deg": 6.5, "shake_dps": 0.2, "hpeak_g": 0.01,
                "vpeak_g": -0.02, "tilted": False, "bumps": 0, **self.extra}

    def reply(self, yaw):
        self.yaw = yaw
        self.imu_replies += 1


def new_loop():
    slots = {"lidar": SensorSlot("lidar", rcfg.lidar_deadline, True),
             "tof": SensorSlot("tof", rcfg.tof_deadline, True)}
    loop = ControlLoop(rcfg, PolicyRunner(HERE / "exploration_model.onnx", rcfg, 1),
                       GyroDriver(rcfg), slots)
    loop.hold.path = TMP / f"hold_{id(loop)}.json"
    loop.hold.pause_path = TMP / f"pause_{id(loop)}.json"
    return loop, slots


loop, slots = new_loop()
m = loop.motors
out = []
m.reply(0.0)
out.append(loop._gyro_dtheta())          # first reading: no difference yet
m.reply(9.0)
out.append(loop._gyro_dtheta())          # +9 deg
out.append(loop._gyro_dtheta())          # no new reply this tick
m.reply(27.0)
out.append(loop._gyro_dtheta())          # +18: both ticks' turn
check("differences of the board's yaw: first None, then +9, 0 (no reply yet), "
      "+18 (that tick and the one before) -- every degree counted once",
      out[0] is None and abs(math.degrees(out[1]) - 9) < 1e-6 and out[2] == 0.0
      and abs(math.degrees(out[3]) - 18) < 1e-6, str(out))
gaps0 = loop.gyro_gaps
m.fresh = False
a = loop._gyro_dtheta()
m.fresh = True
m.reply(40.0)
b = loop._gyro_dtheta()
m.reply(45.0)
c = loop._gyro_dtheta()
check("a gap falls back (None) and the chain RESTARTS: the 13 deg turned during "
      "the gap is not counted again by the gyro (dead reckoning had it)",
      a is None and b is None and abs(math.degrees(c) - 5) < 1e-6
      and loop.gyro_gaps == gaps0 + 2, f"{a} {b} {c} gaps {loop.gyro_gaps - gaps0}")
m.reply(170.0)
check("a jump over 90 deg between replies is a glitch: fall back",
      loop._gyro_dtheta() is None)
m.ok = False
m.reply(171.0)
check("bias not yet measured (status cal): fall back", loop._gyro_dtheta() is None)
m.ok = True

# The tick itself: a pivot commanded, the board says she did not turn.
seen = []
real_dr = loop.map.dead_reckon
loop.map.dead_reckon = lambda v, om, dt: (seen.append(om), real_dr(v, om, dt))[1]
m.executed = lambda: (0.0, 1.2)          # what the wheels were told: a pivot
open_scan = np.full(rcfg.n_beams, 3.0, np.float32)
th0 = None
for k in range(10):
    m.reply(46.0)                         # she is NOT turning
    slots["lidar"].publish(open_scan)
    loop.tick()
    if th0 is None:
        th0 = float(loop.map.pose[2])
check("a commanded pivot that is not happening: dead reckoning turns her by 0, "
      "not by the command (the 22% of turns that execute as stops)",
      all(abs(o) < 1e-9 for o in seen[2:]) and abs(float(loop.map.pose[2]) - th0) < 1e-6,
      str([round(o, 3) for o in seen]))
check("...and after 0.6 s it is flagged as a stalled turn",
      loop.stats.get("pivot_stalls", 0) >= 1 and loop.pivot_stalled,
      str(loop.stats.get("pivot_stalls")))
seen.clear()
yaw = 46.0
for k in range(5):
    yaw += 6.0
    m.reply(yaw)
    slots["lidar"].publish(open_scan)
    loop.tick()
check("a turn that IS happening: counted at the rate measured (6 deg a tick)",
      all(abs(math.degrees(o * rcfg.dt) - 6.0) < 1e-6 for o in seen), str(seen))

print("--- picked up while she explores ---")
loop2, slots2 = new_loop()
m2 = loop2.motors
m2.reply(0.0)
m2.extra = {"tilted": True, "tilt_deg": 50.0}
for _ in range(3):
    slots2["lidar"].publish(open_scan)
    row = loop2.tick()
import body_sense                                          # noqa: E402
ev = body_sense.read()
check("the run's own detector publishes 'lifted' (friday-brain reacts mid-run too)",
      ev is not None and ev["event"] == "lifted" and ev.get("during") == "run", str(ev))
check("...and the wheels are held still while she is up",
      row["v"] == 0.0 and row["omega"] == 0.0 and loop2.stats.get("lifted_ticks", 0) >= 1,
      f"{row['gate']} {row['v']} {row['omega']}")


print("--- the start tracker: a measured turn, searched narrowly ---")
try:
    import path_planner
    from slam.grid import OccupancyGrid
    grid = OccupancyGrid.load(HERE.parent / "state" / "map.npz")
except Exception as exc:                                  # noqa: BLE001
    grid = None
    print(f"  [SKIP] real-room tracker check ({exc})")

if grid is not None:
    occ = grid.occupied_mask()
    nx, ny = occ.shape
    ox, oy = grid.origin
    plan = path_planner.preprocess(grid).plannable

    def cell(x, y):
        return int((x - ox) / grid.cell), int((y - oy) / grid.cell)

    def open_floor(x, y):
        cx, cy = cell(x, y)
        return 0 <= cx < nx and 0 <= cy < ny and plan[cx, cy]

    def sweep(pose, rng):
        out = {}
        for deg in range(360):
            a = pose[2] + math.radians(deg)
            dx, dy = math.cos(a), math.sin(a)
            r = 0.1
            while r < 6.0:
                cx, cy = cell(pose[0] + dx * r, pose[1] + dy * r)
                if not (0 <= cx < nx and 0 <= cy < ny):
                    break
                if occ[cx, cy]:
                    out[deg] = int(1000 * (r + rng.gauss(0, 0.01)))
                    break
                r += 0.02
        return out

    def over_turning_run(seed, gyro, secs=60.0):
        """Turns go 40% FURTHER than commanded half the time (the side the old
        window cannot see) and a quarter not at all. With the gyro, the dead
        reckoning's heading is the truth plus 0.5 deg/s of drift."""
        rng = random.Random(seed)
        cells = np.argwhere(plan)
        c = cells[rng.randrange(len(cells))]
        truth = [(c[0] + .5) * grid.cell + ox, (c[1] + .5) * grid.cell + oy,
                 rng.uniform(-3, 3)]
        start = tuple(truth)
        dead = [0.0, 0.0, 0.0]
        tr = wb.StartTracker(8.0)
        om, ms = 0.0, []
        for k in range(int(secs * 10)):
            dt, v = 0.1, 0.2
            if rng.random() < 0.05:
                om = rng.choice([-1.2, -0.6, 0, 0, 0.6, 1.2])
            r = rng.random()
            t_om = om * (0.0 if r < 0.25 else 1.4 if r < 0.75 else 1.0)
            nxp = truth[0] + v * math.cos(truth[2]) * dt
            nyp = truth[1] + v * math.sin(truth[2]) * dt
            if open_floor(nxp, nyp):
                truth[0], truth[1] = nxp, nyp
            else:
                om = 1.2
            truth[2] = wb._wrap(truth[2] + t_om * dt)
            d_om = (t_om + math.radians(rng.gauss(0, 0.5))) if gyro else om
            dead[0] += v * math.cos(dead[2]) * dt
            dead[1] += v * math.sin(dead[2]) * dt
            dead[2] = wb._wrap(dead[2] + d_om * dt)
            if k % 3 == 0:
                tr.submit(sweep(truth, rng), dead, gyro_gaps=0 if gyro else None)
                t0 = time.perf_counter()
                tr._fold(*tr._next_job())
                ms.append((time.perf_counter() - t0) * 1000.0)
        rel_true = wb._compose(wb._inverse(tuple(truth)), start)
        rel_est = wb._compose(wb._inverse(tuple(dead)), tr.home_rl())
        err = math.hypot(rel_est[0] - rel_true[0], rel_est[1] - rel_true[1])
        return err, tr.matched, tr.updates, float(np.median(ms)), tr._folder.known_turns

    for seed in (1, 4):
        e_g, m_g, u_g, ms_g, known = over_turning_run(seed, gyro=True)
        e_n, m_n, u_n, ms_n, _ = over_turning_run(seed, gyro=False)
        # 0.35 m, not test_way_back's 0.30: these runs ALSO over-turn by 40%,
        # and ~22% of their ticks drive into a wall while she does not move --
        # forward distance, which no gyro measures (seed 4's residual is that;
        # it matched 167 of 200 sweeps with the gyro, 144 without). The way
        # home's last step (the start's own view, 2fcfd48) takes it from there.
        check(f"real room, over-turning run {seed}: WITH the gyro the start is found "
              f"within 0.35 m, and better than without ({e_g:.2f} m; without: "
              f"{e_n:.2f} m)", e_g <= 0.35 and e_g < e_n and known > 0,
              f"known turns {known}")
        check(f"...and each fold is faster (median {ms_g:.0f} ms vs {ms_n:.0f} ms "
              f"without)", ms_g < ms_n)

    # The tracker PROCESS understands both job shapes.
    tr = wb.StartTracker(8.0)
    tr.start()
    rng = random.Random(7)
    cells = np.argwhere(plan)
    c = cells[len(cells) // 2]
    pose = [(c[0] + .5) * grid.cell + ox, (c[1] + .5) * grid.cell + oy, 0.3]
    for k in range(6):
        tr.submit(sweep(pose, rng), (0.0, 0.0, 0.0), gyro_gaps=0)
        time.sleep(0.4)
    tr.stop()
    check("the tracker process takes the new job (with the gyro's gap count): "
          "folds, no error", tr.updates >= 2 and not tr.error, f"{tr.updates} {tr.error}")

print("--- PASSED ---" if not fails else f"--- {fails} FAILED ---")
sys.exit(1 if fails else 0)
