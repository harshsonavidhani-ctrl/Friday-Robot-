"""The way back: explore exactly as before, then drive to where the run started.

Harsh, 2026-09-26: "just adding the feature where it knows where it started like
should return to starting position immediately after exploration" and "don't
change the behaviour, just add". So three things have to hold:

  1. While EXPLORING, a loop with the way back on commands exactly what a loop
     without it commands, tick for tick.
  2. When the time is up, the planner's target is the start (and nothing
     replaces it), the pull toward it is labelled, and arrival ends the run.
  3. The start tracker knows where the start is after a run on this drivetrain,
     where dead reckoning does not. On the REAL room (state/map.npz), ray-cast,
     with turns that execute short or not at all and forward moves that stall
     against walls: measured 2026-09-26, six 2-minute runs ended 5-23 cm from
     the true start (dead reckoning: 0.96-4.04 m).
"""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE_T = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_T.parent))
sys.path.insert(0, str(HERE_T.parent.parent))

import way_back as wb  # noqa: E402
from explorer import ExplorationPlanner  # noqa: E402
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot)

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class FakeTracker:
    def __init__(self, home):
        self._home = home
        self.submitted = 0

    def submit(self, *_a, **_k):
        self.submitted += 1

    def home_rl(self):
        return self._home

    def stop(self):
        pass

    def snapshot(self):
        return {"home_rl": self._home}


def make_loop(with_way_back, home=(1.0, -1.0, 0.0)):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    loop = ControlLoop(cfg, PolicyRunner(HERE / "exploration_model.onnx", cfg),
                       NullMotorDriver(cfg), slots)
    loop.map_enabled = False
    pl = ExplorationPlanner(loop.world, period_s=999.0)
    loop.planner = pl
    if with_way_back:
        loop.way_back = wb.WayBack(FakeTracker(home), budget_s=60.0)
    return loop, slots


print("--- way back ---")

# 1. Frames: the start in the run's frame is R . T^-1.
t = wb.StartTracker(8.0)
with t._lock:
    t._fix_t, t._fix_rl = (1.0, 0.0, math.pi / 2), (5.0, 5.0, 0.0)
h = t.home_rl()
check("the start is placed in the run's frame from one fix",
      abs(h[0] - 5.0) < 1e-9 and abs(h[1] - 6.0) < 1e-9
      and abs(wb._wrap(h[2] + math.pi / 2)) < 1e-9, f"{h}")

# 2. Exploring is untouched: identical commands with and without the way back.
def run(with_way_back, n=30):
    loop, slots = make_loop(with_way_back)
    out = []
    for i in range(n):
        slots["lidar"].publish(np.full(cfg.n_beams, 3.0 + 0.01 * i, np.float32))
        row = loop.tick()
        out.append((row["v"], row["omega"]))
    return np.array(out), loop


off, _ = run(False)
on, loop_on = run(True)
check("while exploring, the way back changes no command",
      np.allclose(off, on, atol=1e-12), f"max delta {np.abs(off - on).max():.2e}")
check("...but the start tracker is fed every tick",
      loop_on.way_back.tracker.submitted == 30)

# 3. Time up: the start becomes the target and stays it.
loop, slots = make_loop(True, home=(1.0, -1.0, 0.0))
loop._begin_way_back("time")
check("the phase is returning", loop.way_back.phase == "returning")
check("the planner's target is the start",
      loop.planner.target is loop._home_target
      and loop._home_target.approach == (1.0, -1.0))
loop.planner._plan_once()
check("a planning cycle does not replace it", loop.planner.target is loop._home_target)
loop.planner.note_progress(np.array([1.0, -1.0, 0.0]))
check("standing on it does not retire it (the loop decides arrival)",
      loop.planner.target is loop._home_target)
for i in range(10):
    slots["lidar"].publish(np.full(cfg.n_beams, 3.0, np.float32))
    loop._steer_home()
    loop.tick()
ctx = loop.last_record.planner or {}
check("the pull toward the start is labelled as the way back",
      str(ctx.get("reason", "")).startswith("way back"), f"{ctx.get('reason')}")
check("the state published for the dashboard carries it",
      (loop.autonomy_state() or {}).get("way_back", {}).get("phase") == "returning")

# 3b. The way back drives HOME, not wherever the policy fancies: on the robot
#     (23:12) the policy reversed on 34% of the way back and she never got there.
class WheelsObey(NullMotorDriver):
    """Does what it is told, so dead reckoning moves (the null driver never does)."""

    def drive(self, v, omega):
        super().drive(v, omega)
        self.last_executed = (float(v), float(omega))


loop3, slots3 = make_loop(True)
loop3.motors = WheelsObey(cfg)
p0 = np.array(loop3.map.pose, float)
goal = (float(p0[0]) + 1.2, float(p0[1]) + 0.9, 0.0)
loop3.way_back = wb.WayBack(FakeTracker(goal), budget_s=60.0)
loop3._begin_way_back("time")
d0 = loop3.way_back.distance(loop3.map.pose)
reversed_ticks = 0
for i in range(150):
    slots3["lidar"].publish(np.full(cfg.n_beams, 3.0, np.float32))
    loop3._steer_home()
    row = loop3.tick()
    reversed_ticks += row["v"] < -1e-6
    if loop3.way_back.check(loop3.map.pose) is not None:
        break
check("in open space she turns to the start and drives to it",
      loop3.way_back.outcome == "arrived",
      f"{d0:.2f} m -> {loop3.way_back.distance(loop3.map.pose):.2f} m in {i + 1} ticks")
check("...without reversing on the way", reversed_ticks == 0, f"{reversed_ticks} ticks")
check("far off to one side: turn in place first",
      wb.homing_command(math.radians(120), 2.0, 1.6)[:2] == (0.0, 1.6))
check("roughly facing it: drive, steering toward it",
      wb.homing_command(math.radians(-20), 2.0, 1.6)[0] > 0.2
      and wb.homing_command(math.radians(-20), 2.0, 1.6)[1] < 0)
check("while exploring there is no homing command at all",
      make_loop(True)[0].way_back.phase == "exploring")

check("still far away: keep driving", loop.way_back.check((0.0, 0.0, 0.0)) is None)
check("within tolerance: arrived",
      loop.way_back.check((0.9, -0.9, 0.0)) == "arrived")
loop._finish_way_back("arrived")        # narrator off: must not wait or raise
loop2, _ = make_loop(True)
loop2.way_back.budget_s = 0.0
loop2._begin_way_back("time")
check("out of time ends it too", loop2.way_back.check((5.0, 5.0, 0.0)) == "out of time")

# 4. Early end only when frontiers stay exhausted, and never in the first seconds.
w = wb.WayBack(FakeTracker((0, 0, 0)))
check("exhausted once is not enough", not w.exhausted_long_enough(True, 60.0, now=100.0))
check("...held 8 s is", w.exhausted_long_enough(True, 60.0, now=108.5))
w2 = wb.WayBack(FakeTracker((0, 0, 0)))
w2.exhausted_long_enough(True, 5.0, now=100.0)
check("...but not 13 s into a run", not w2.exhausted_long_enough(True, 13.0, now=108.5))

# 5. The whole run(): the timer starts the way back; a person's stop (SIGINT,
#    which is how stop_exploration and the unit end a run) never does, and
#    stops a way back already under way.
import tempfile  # noqa: E402

from world import WorldMap  # noqa: E402


def whole_run(duration, interrupt_at=None):
    tmp = tempfile.TemporaryDirectory()
    loop, slots = make_loop(False)
    loop.planner = None                       # run() builds its own
    loop.map_enabled = True
    loop.world = WorldMap(map_file=Path(tmp.name) / "map.npz",
                          max_range=cfg.lidar_max_range)
    loop.state_file = None
    loop.blackbox.dump = lambda _p: None
    loop.report = lambda: None
    real_tick, n = loop.tick, [0]

    def tick():
        n[0] += 1
        if interrupt_at is not None and n[0] >= interrupt_at:
            raise KeyboardInterrupt
        slots["lidar"].publish(np.full(cfg.n_beams, 2.5, np.float32))
        return real_tick()
    loop.tick = tick
    loop.run(duration=duration, verbose=False)
    tmp.cleanup()
    return loop


stopped = whole_run(duration=30.0, interrupt_at=5)
check("stopped by a person while exploring: no way back",
      stopped.way_back is not None and stopped.way_back.phase == "exploring")
home_now = whole_run(duration=0.5, interrupt_at=60)
check("the timer starts the way back; never having left, she is back at once",
      home_now.way_back is not None and home_now.way_back.why == "time"
      and home_now.way_back.outcome == "arrived", str(home_now.way_back.snapshot()))


class FarTracker(FakeTracker):
    """A start 3 m away, so the way back is still under way when stopped."""

    def __init__(self, _max_range):
        super().__init__((15.0, 12.0, 0.0))

    def start(self):
        pass


real_tracker = wb.StartTracker
wb.StartTracker = FarTracker
try:
    timed = whole_run(duration=0.5, interrupt_at=40)
finally:
    wb.StartTracker = real_tracker
check("a stop during the way back ends it where she is",
      timed.way_back is not None and timed.way_back.phase == "returning"
      and timed.way_back.why == "time", str(timed.way_back.snapshot()))
check("the planner was steering for the start when she stopped",
      timed.planner is not None and timed.planner.home is not None
      and timed.planner.target.approach == (15.0, 12.0))

# 5b. On the robot the matching runs in its OWN PROCESS: on a thread it took
#     the GIL from the control loop and the reflex (262 overruns, two
#     collisions, 2026-09-26 22:44).
tr = wb.StartTracker(8.0, hz=20.0)
tr.start()
room = {d: 1500 + 400 * int(60 <= d < 150) for d in range(0, 360)}
deadline = time.monotonic() + 30.0
while tr.updates < 3 and time.monotonic() < deadline and not tr.error:
    tr.submit(room, (1.0, 2.0, 0.0))      # the loop submits every tick
    time.sleep(0.1)
check("the tracker process folds sweeps and reports fixes",
      tr.updates >= 3 and not tr.error, str(tr.snapshot()))
check("...and a robot that never moved is at its start",
      tr.home_rl() is not None and math.hypot(tr.home_rl()[0] - 1.0,
                                              tr.home_rl()[1] - 2.0) < 0.06,
      str(tr.home_rl()))
check("the matching is not in this process",
      tr._folder.matcher is None and tr._proc is not None)
tr.stop()
tr.join(timeout=5.0)
check("stopping ends the process", tr._proc.poll() is not None)

# 6. The tracker on the real room, with this drivetrain's slips.
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

    def slipping_run(seed, secs=60.0):
        rng = random.Random(seed)
        cells = np.argwhere(plan)
        c = cells[rng.randrange(len(cells))]
        truth = [(c[0] + .5) * grid.cell + ox, (c[1] + .5) * grid.cell + oy,
                 rng.uniform(-3, 3)]
        start = tuple(truth)
        dead = [0.0, 0.0, 0.0]
        tr = wb.StartTracker(8.0)
        om = 0.0
        for k in range(int(secs * 10)):
            dt, v = 0.1, 0.2
            if rng.random() < 0.05:
                om = rng.choice([-1.2, -0.6, 0, 0, 0.6, 1.2])
            # Turns execute short, and a quarter of them not at all.
            t_om = om * 0.8 * (0.0 if rng.random() < 0.25 else 1.0)
            nxp = truth[0] + v * math.cos(truth[2]) * dt
            nyp = truth[1] + v * math.sin(truth[2]) * dt
            if open_floor(nxp, nyp):
                truth[0], truth[1] = nxp, nyp
            else:
                om = 1.2           # against a wall: the wheels "drove", she did not
            truth[2] = wb._wrap(truth[2] + t_om * dt)
            dead[0] += v * math.cos(dead[2]) * dt
            dead[1] += v * math.sin(dead[2]) * dt
            dead[2] = wb._wrap(dead[2] + om * dt)
            if k % 3 == 0:
                tr.submit(sweep(truth, rng), dead)
                tr._fold(*tr._next_job())
        rel_true = wb._compose(wb._inverse(tuple(truth)), start)
        rel_est = wb._compose(wb._inverse(tuple(dead)), tr.home_rl())
        rel_dead = wb._compose(wb._inverse(tuple(dead)), (0.0, 0.0, 0.0))
        return (math.hypot(rel_est[0] - rel_true[0], rel_est[1] - rel_true[1]),
                math.hypot(rel_dead[0] - rel_true[0], rel_dead[1] - rel_true[1]),
                tr.matched, tr.updates)

    for seed in (1, 4):
        t0 = time.monotonic()
        err, dead_err, matched, updates = slipping_run(seed)
        check(f"real room, slipping run {seed}: the start is found within 0.30 m",
              err <= 0.30 and err < dead_err,
              f"(tracker {err:.2f} m, dead reckoning {dead_err:.2f} m, matched "
              f"{matched}/{updates}, {time.monotonic() - t0:.0f} s)")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
