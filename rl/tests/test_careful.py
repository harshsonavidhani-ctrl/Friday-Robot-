"""Stage 4e (2026-10-20): the 09-04 03:52 speed, and reverse only into space the
LiDAR SEES is empty. Harsh: "on 4 september at 3:52 am was the exploration I
want back" -- every real run that night drove at exactly 0.125 m/s -- and the
drivetrain sim (scripts/sim_ab) put 21 of today's 52 bumps on REVERSING.

No hardware: strict_rear_refusal is a pure function, and ControlLoop._careful
runs on a stand-in self.
"""
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy                                       # noqa: E402
from pi_deploy import RobotConfig, strict_rear_refusal  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


cfg = RobotConfig()
cfg.strict_reverse = True
cfg.max_forward_speed = 0.125
N = cfg.n_beams
STEP = 360.0 / N
NEED = cfg.rear_stop + cfg.strict_rear_extra


def beam(deg):
    """Index of the beam at `deg` (0 = her nose, 180 = straight behind)."""
    return int(round((deg % 360.0) / STEP)) % N


def open_room(r=3.0):
    return np.full(N, r, np.float32)


print("strict_rear_refusal: what counts as 'seen empty'")
check("open room behind -> reverse allowed", strict_rear_refusal(open_room(), [], cfg) == "")
s = open_room()
s[beam(180)] = NEED - 0.05
check("ONE beam (a chair leg) straight behind -> refused",
      "behind" in strict_rear_refusal(s, [], cfg), strict_rear_refusal(s, [], cfg))
s = open_room()
s[beam(180)] = NEED + 0.10
check("the same leg further than rear_stop + extra -> allowed",
      strict_rear_refusal(s, [], cfg) == "")
s = open_room()
s[beam(90)] = 0.30           # a wall BESIDE her, not behind
s[beam(270)] = 0.30
check("walls beside her (not in the corridor she backs through) -> allowed",
      strict_rear_refusal(s, [], cfg) == "", strict_rear_refusal(s, [], cfg))
s = open_room()
s[beam(150)] = 0.40          # 30 deg off the tail at 0.40 m: lateral 0.20 m, in path
check("something 30 deg off the tail, inside her width -> refused",
      strict_rear_refusal(s, [], cfg) != "", strict_rear_refusal(s, [], cfg))
s = open_room()
s[beam(180)] = 0.15          # inside her own body: no information, not an obstacle
check("a return inside her own body is ignored, not believed",
      strict_rear_refusal(s, [], cfg) == "")

# Today's self-mask, mapped to beams the way the run does it.
blind = pi_deploy.masked_beam_indices(
    pi_deploy.parse_self_mask("71:79,88:100,221:237,319:334"), -51.8, N)
check("today's mask blinds beams straight behind her (raw 221-237)",
      any(abs(((int(b) * STEP) % 360) - 180) <= 10 for b in blind), str(sorted(blind)))
s = open_room()
for b in blind:
    s[b] = 0.25              # whatever is there: the run publishes lidar_max_range
check("an object hidden in a BLIND beam is not 'seen', and the rest is open -> allowed "
      "(the blind wedge is narrow; most of the rear is seen)",
      strict_rear_refusal(s, blind, cfg) == "", strict_rear_refusal(s, blind, cfg))
many = [beam(d) for d in range(120, 241, int(STEP))]
check("most of the rear blind -> refused as UNKNOWN, never 'clear'",
      "unknown" in strict_rear_refusal(open_room(), many, cfg),
      strict_rear_refusal(open_room(), many, cfg))
check("no sweep -> refused", strict_rear_refusal(None, [], cfg) != "")


print("ControlLoop._careful: the cap and the reverse budget")


class Rec:
    def __init__(self):
        self.rows = []

    def add(self, stage, v, w, why=""):
        self.rows.append((stage, v, w, why))


def loop_stub():
    return types.SimpleNamespace(cfg=cfg, stats={}, map=types.SimpleNamespace(blind_beams=[]),
                                 _rev_since=None, _rev_rest_until=0.0)


careful = pi_deploy.ControlLoop._careful
L = loop_stub()
v, gate = careful(L, 0.45, 0.0, "navigator", open_room(), Rec())
check("forward 0.45 -> 0.125 (the 03:52 speed)", abs(v - 0.125) < 1e-9 and gate == "navigator")
v, gate = careful(L, 0.10, 0.3, "navigator", open_room(), Rec())
check("slower than the cap is untouched", v == 0.10)
cfg.max_forward_speed = 0.0
v, _ = careful(L, 0.45, 0.0, "ok", open_room(), Rec())
check("cap 0 = off", v == 0.45)
cfg.max_forward_speed = 0.125

L = loop_stub()
s = open_room()
s[beam(180)] = 0.30
v, gate = careful(L, -0.20, 0.0, "REFLEX_BACKOFF", s, Rec())
check("the back-off (issued AFTER the rear gate) into a leg behind -> 0, 'reverse_refused'",
      v == 0.0 and gate == "reverse_refused")
v, gate = careful(L, -0.20, 0.8, "undo", s, Rec())
check("...the undo too: no stage gets round it", v == 0.0)

L = loop_stub()
cfg.reverse_max_s, cfg.reverse_rest_s = 0.3, 0.4
v, _ = careful(L, -0.20, 0.0, "recovery", open_room(), Rec())
check("rear seen empty -> she may back up", v == -0.20)
time.sleep(0.35)
v, gate = careful(L, -0.20, 0.0, "recovery", open_room(), Rec())
check(f"after reverse_max_s of backing up -> stop backing", v == 0.0 and gate == "reverse_refused")
v, _ = careful(L, -0.20, 0.0, "recovery", open_room(), Rec())
check("...and turning only while it rests", v == 0.0)
v, _ = careful(L, 0.0, 1.2, "escape_turn", open_room(), Rec())
check("...a turn in place still goes", v == 0.0)
time.sleep(0.45)
v, _ = careful(L, -0.20, 0.0, "recovery", open_room(), Rec())
check("after the rest she may back up again", v == -0.20)
cfg.reverse_max_s, cfg.reverse_rest_s = 1.2, 1.0

cfg.strict_reverse = False
L = loop_stub()
s = open_room()
s[beam(180)] = 0.30
v, _ = careful(L, -0.20, 0.0, "recovery", s, Rec())
check("--strict-reverse off = the old behaviour exactly", v == -0.20)

src = (Path(__file__).resolve().parent.parent / "pi_deploy.py").read_text()
i_careful = src.index("v, gate = self._careful(v, omega, gate, scan, rec)")
i_limiter = src.index("v, omega = self.limiter(v, omega, dt_real, emergency=emergency,")
i_undo = src.index("# ---- 4b. UNDO THE LAST MOVE")
check("stage 4e runs after the undo and before the limiter", i_undo < i_careful < i_limiter)

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
