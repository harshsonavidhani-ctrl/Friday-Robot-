#!/usr/bin/env python3
"""A turn in place that swings her front INTO something is stopped (2026-10-10).

Harsh: "reflex safety add it". 2026-10-10 13:46 (rl/blackbox-20261010-134713.csv,
t=4.0-5.0 s): forward was already cut, she pivoted right (wire M190 190) and the
bumper ToF went 0.40 -> 0.29 -> 0.16 -> 0.02 m -- a bump -- because a pivot was
allowed down to tof_pivot_min_clearance (0.12 m). Checks:
  * the rule: turning in place, ToF under PIVOT_WATCH_M and down PIVOT_CLOSE_M
    within PIVOT_WINDOW_S -> that turn stops, that direction refused PIVOT_BLOCK_S;
  * the fast 30 Hz thread stops it too (it used to allow any turn above 0.12 m);
  * REVERSE and the OTHER direction still pass -- it can never box her in by itself;
  * a pivot beside something that is NOT getting nearer still passes (the
    2026-09-02 bug: 316 ticks frozen 0.20 m from a wall);
  * unstick's retrace is exempt.

    ./venv/bin/python rl/tests/test_pivot_reflex.py      (seconds; no hardware)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE_T = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_T.parent))
sys.path.insert(0, str(HERE_T.parent.parent))

import config as friday_config                           # noqa: E402  (the project's)
TMP = Path(tempfile.mkdtemp(prefix="friday-pivot-"))
friday_config.BODY_FILE = TMP / "body.json"

import pi_deploy as pd                                   # noqa: E402
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot)

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    if not ok:
        fails += 1


def new_loop(tof=True):
    c = RobotConfig()
    c.cam_enabled = False
    c.tof_enabled = tof
    slots = {"lidar": SensorSlot("lidar", c.lidar_deadline, True),
             "tof": SensorSlot("tof", c.tof_deadline, True)}
    loop = ControlLoop(c, PolicyRunner(HERE / "exploration_model.onnx", c, 1),
                       NullMotorDriver(c), slots)
    loop.hold.path = TMP / f"hold_{id(loop)}.json"
    loop.hold.pause_path = TMP / f"pause_{id(loop)}.json"
    return loop, slots, c


print("--- the rule ---")
loop, slots, c = new_loop()
t = 100.0
for d in (0.40, 0.36, 0.33):                     # turning right (-1), nearer each sample
    loop.reflex.last_min = d
    loop._note_pivot(0.0, -0.96, t)
    t += 0.1
check("turning right, the bumper 0.40 -> 0.33 m within 0.2 s: CLOSING",
      loop._pivot_closing(-1, t))
check("...the other way (left) is not", not loop._pivot_closing(+1, t))
loop2, _, _ = new_loop()
t = 100.0
for d in (0.20, 0.205, 0.198, 0.202):            # beside a wall, not getting nearer
    loop2.reflex.last_min = d
    loop2._note_pivot(0.0, -0.96, t)
    t += 0.1
check("turning beside a wall 0.20 m away that does NOT get nearer: allowed "
      "(the 316-tick freeze of 2026-09-02 stays fixed)", not loop2._pivot_closing(-1, t))
loop3, _, _ = new_loop()
t = 100.0
for d in (0.80, 0.70, 0.62):                     # nearer, but nowhere near
    loop3.reflex.last_min = d
    loop3._note_pivot(0.0, -0.96, t)
    t += 0.1
check(f"nearer but still over {pd.PIVOT_WATCH_M} m: allowed", not loop3._pivot_closing(-1, t))
loop4, _, _ = new_loop()
t = 100.0
for d in (0.40, 0.36, 0.33):
    loop4.reflex.last_min = d
    loop4._note_pivot(0.25, -0.96, t)            # driving an arc, not turning in place
    t += 0.1
check("an arc while driving forward is the forward reflex's job, not this rule",
      not loop4._pivot_closing(-1, t))

print("--- the 30 Hz reflex thread ---")
loop._pivot_samples.clear()
loop.motors.last_cmd = (0.0, -0.96)
for d in (0.40, 0.36, 0.33):                     # what the thread samples, real clock
    loop.reflex.last_min = d
    loop._note_pivot(*loop.motors.last_cmd)
check("the fast thread STOPS a closing turn (it allowed any turn above 0.12 m)",
      loop._fast_reflex_should_stop() is True)
check("...and that direction is refused for a while",
      loop._pivot_blocked(-1) and not loop._pivot_blocked(+1))
loop.motors.last_cmd = (-0.15, 0.0)
check("...a straight reverse is never stopped by it", loop._fast_reflex_should_stop() is False)
loop.motors.last_cmd = (0.0, 0.96)
check("...nor the turn the other way", loop._fast_reflex_should_stop() is False)
loop._undo_active = True
loop.motors.last_cmd = (0.0, -0.96)
loop._pivot_block = (0, 0.0)
check("...nor unstick's retrace", loop._fast_reflex_should_stop() is False)
loop._undo_active = False

print("--- whole ticks: the 13:46 bump replayed ---")
loop, slots, c = new_loop()
action_want = {"v": 0.0, "w": -0.96}
loop.policy.infer = lambda obs: np.asarray(c.to_normalised(action_want["v"],
                                                           action_want["w"]),
                                           dtype=np.float32)
loop.navigator_enabled = False
open_scan = np.full(c.n_beams, 3.0, np.float32)
rows = []
for d in (0.40, 0.40, 0.36, 0.33, 0.29, 0.26, 0.24, 0.22):
    slots["lidar"].publish(open_scan)
    slots["tof"].publish(np.full((8, 8), d, np.float32))
    row = loop.tick()
    rows.append((d, row["gate"], round(row["omega"], 2)))
turned_at = [d for d, g, w in rows if w < -0.1]
stopped = [d for d, g, w in rows if g == "pivot_closing"]
check("the right turn is stopped while the bumper is still well clear (by 0.30 m)",
      bool(stopped) and max(stopped) >= 0.29 and min(d for d, _, w in rows
                                                 if w < -0.1) >= 0.30,
      str(rows))
action_want.update(v=-0.15, w=0.0)
slots["lidar"].publish(open_scan)
slots["tof"].publish(np.full((8, 8), 0.22, np.float32))
row = loop.tick()
check("...backing away is still allowed while that turn is refused",
      row["v"] < 0.0, f"{row['gate']} v={row['v']}")
action_want.update(v=0.0, w=0.96)
slots["lidar"].publish(open_scan)
slots["tof"].publish(np.full((8, 8), 0.30, np.float32))
row = loop.tick()
check("...and so is turning the OTHER way", row["omega"] > 0.1 and row["gate"] != "pivot_closing",
      f"{row['gate']} w={row['omega']}")

print("--- PASSED ---" if not fails else f"--- {fails} FAILED ---")
sys.exit(1 if fails else 0)
