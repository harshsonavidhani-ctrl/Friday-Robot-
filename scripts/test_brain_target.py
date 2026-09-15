#!/usr/bin/env python3
"""test_brain_target.py — the targeting rules, with no robot and no run.

    ./venv/bin/python scripts/test_brain_target.py

WHY THIS EXISTS AND WHY IT IS COMPLETE. rl/brain_target.py deliberately imports
nothing from the control loop, so all of it can be exercised here against fake
detections and a fake planner. That is the whole point of the separate file: a
sign error in the nudge means she drives AWAY from the thing she announced, and
finding that on the robot costs a hardware run every attempt.

The one thing this cannot cover is whether the run actually calls
`bearing_error()` -- that is the wiring, and it is checked on the robot.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl import brain_target as bt                       # noqa: E402
import rl_control                                       # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


class Det:
    """Stands in for rl.friday_vision.Detection: same duck type, nothing more."""

    def __init__(self, label, score, box, area=0.05):
        self.label, self.score, self.box, self.area = label, score, box, area


class FakePlanner:
    """Stands in for ExplorationPlanner, including the bits recovery reaches for."""

    def __init__(self, bearing=0.5):
        self._lock = object()
        self.target = "a-frontier"
        self.blacklist = set()
        self.targets_abandoned = 0
        self.progress_calls = 0
        self._bearing = bearing

    def bearing_error(self, pose):
        return self._bearing

    def note_progress(self, pose):
        self.progress_calls += 1


# --------------------------------------------------------------------------
print("\nWHERE — the sign convention, which is the thing worth being sure of\n")

# positive is LEFT, matching omega, bearing_error() and Detection.bearing_deg.
left = bt.where_of((0.10, 0.4, 0.30, 0.6))      # centre 0.20, left of frame
right = bt.where_of((0.70, 0.4, 0.90, 0.6))     # centre 0.80, right of frame
middle = bt.where_of((0.45, 0.4, 0.55, 0.6))    # centre 0.50

check("an object LEFT of centre gives a POSITIVE where", left > 0, f"got {left}")
check("an object RIGHT of centre gives a NEGATIVE where", right < 0, f"got {right}")
check("dead centre is 0", abs(middle) < 1e-9, f"got {middle}")
check("the edges saturate at +-1",
      abs(bt.where_of((0.0, 0, 0.0, 1)) - 1.0) < 1e-9
      and abs(bt.where_of((1.0, 0, 1.0, 1)) + 1.0) < 1e-9)
check("a left object nudges LEFT (positive rad)", bt.nudge_rad(left) > 0)
check("a right object nudges RIGHT (negative rad)", bt.nudge_rad(right) < 0)
check("the nudge is capped at the swing",
      abs(bt.nudge_rad(5.0, 25.0) - bt.nudge_rad(1.0, 25.0)) < 1e-12,
      "a where beyond +-1 must not turn further than the swing")

print("\n  a malformed box is None, never a confident 'dead ahead'\n")
for bad, why in [(None, "None"), ((0.1, 0.2), "too short"),
                 (("a", "b", "c", "d"), "not numbers"),
                 ((0.9, 0, 0.1, 1), "xmax < xmin"), ((-0.5, 0, 0.5, 1), "out of range")]:
    check(f"where_of({why}) is None", bt.where_of(bad) is None)


# --------------------------------------------------------------------------
print("\nPICK — the best sighting of the label we were told to find\n")

dets = [Det("chair", 0.60, (0.1, 0, 0.2, 1)),
        Det("chair", 0.90, (0.7, 0, 0.8, 1)),
        Det("person", 0.99, (0.4, 0, 0.5, 1)),
        Det("bottle", 0.20, (0.0, 0, 0.1, 1))]

check("picks the HIGHEST-SCORING match, not the first",
      bt.pick(dets, "chair").score == 0.90)
check("is case- and space-insensitive", bt.pick(dets, "  Chair ") is not None)
check("a label that is not there is None", bt.pick(dets, "sofa") is None)
check("below min_score is not a sighting", bt.pick(dets, "bottle") is None)
check("an empty label is None", bt.pick(dets, "") is None)
check("no detections at all is None", bt.pick([], "chair") is None)


# --------------------------------------------------------------------------
print("\nSEEN — what the brain is shown\n")

pub = bt.seen_from(dets)
check("one row per usable detection", len(pub) == 4, f"got {len(pub)}")
check("carries label, score, where, area",
      set(pub[0]) == {"label", "score", "where", "area"}, f"got {set(pub[0])}")
check("publishes NO bearing/degrees field — it is not geometry",
      not any("bearing" in k or "deg" in k for row in pub for k in row))
check("publishes no distance field — there is no distance",
      not any("dist" in k or "range" in k for row in pub for k in row))
check("a detection with a broken box is dropped, not published as 0",
      len(bt.seen_from([Det("chair", 0.9, None)])) == 0)


# --------------------------------------------------------------------------
print("\nTHE COMMAND CHANNEL — latest wins, acted on once per unseen token\n")

tmp = Path(tempfile.mkdtemp())
cmd = tmp / "rl_command.json"

check("a missing file is no order, not a crash", bt.read_command(cmd) is None)
cmd.write_text("{not json")
check("a half-written file is no order", bt.read_command(cmd) is None)
cmd.write_text('["a list"]')
check("a non-object is no order", bt.read_command(cmd) is None)
cmd.write_text('{"label": "", "token": 5}')
check("an empty label is no order", bt.read_command(cmd) is None)
cmd.write_text('{"label": "chair", "token": "not-a-number"}')
check("a bad token is no order", bt.read_command(cmd) is None)
cmd.write_text('{"label": "chair", "token": 7, "ttl_s": 30}')
got = bt.read_command(cmd)
check("a good order reads back",
      got == {"kind": "label", "label": "chair", "heading_deg": None,
              "token": 7, "ttl_s": 30.0},
      f"got {got}")

target = bt.BrainTarget(command_file=cmd)
check("first poll adopts the order", target.poll(now=100.0) is True)
check("  and the label is live", target.label == "chair")
check("re-polling the SAME token changes nothing (free to poll every tick)",
      target.poll(now=101.0) is False)
cmd.write_text('{"label": "person", "token": 8, "ttl_s": 30}')
check("a NEW token replaces the order", target.poll(now=102.0) is True)
check("  and the target switched", target.label == "person")
check("  and the clock restarted", abs(target.expires_at - 132.0) < 1e-9)


# --------------------------------------------------------------------------
print("\nTHE CLOSED LOOP — re-aimed on every sighting, quiet when blind\n")

t = bt.BrainTarget(command_file=cmd, swing_deg=25.0, ttl_s=90.0, stale_s=3.0)
t.poll(now=0.0, command={"label": "chair", "token": 1, "ttl_s": 90.0})

check("no sighting yet, so no steering", t.nudge(now=0.0) is None)
n1 = t.update([Det("chair", 0.9, (0.1, 0, 0.3, 1))], now=1.0)
check("sees it on the left -> nudges left", n1 is not None and n1 > 0)
n2 = t.update([Det("chair", 0.9, (0.7, 0, 0.9, 1))], now=2.0)
check("it moves right -> nudge follows it right", n2 is not None and n2 < 0,
      "this is the closed loop: an under-turn is corrected, not accumulated")
n3 = t.update([Det("chair", 0.9, (0.45, 0, 0.55, 1))], now=3.0)
check("centred -> nudge goes to ~0", n3 is not None and abs(n3) < 1e-9)

t.update([], now=4.0)
check("one missed frame still steers on the last sighting",
      t.nudge(now=4.5) is not None)
check("a STALE sighting stops steering",
      t.nudge(now=9.0) is None,
      "driving at where something used to be is worse than not driving at it")
check("a different label in view is not a sighting",
      t.update([Det("sofa", 0.99, (0.1, 0, 0.2, 1))], now=10.0) is None)

check("the order expires on its OWN clock",
      t.update([Det("chair", 0.9, (0.1, 0, 0.3, 1))], now=1000.0) is None)
check("  and expiry clears the label", t.label is None)


# --------------------------------------------------------------------------
print("\nTHE PLANNER WRAPPER — transparent, and safe for recovery to reach into\n")

# A clock the test drives, because the wrapper asks the target for a nudge with
# no `now` -- exactly as the control loop does. Without this the wrapper would be
# comparing a fake sighting time against the real wall clock and every sighting
# would look stale.
class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


clock = Clock(0.0)
inner = FakePlanner(bearing=0.5)
live = bt.BrainTarget(command_file=cmd, clock=clock)
live.poll(command={"label": "chair", "token": 1, "ttl_s": 90.0})
wrapped = bt.TargetedPlanner(inner, live)

check("with no sighting it is the FRONTIER's bearing",
      wrapped.bearing_error(None) == 0.5)
clock.t = 1.0
live.update([Det("chair", 0.9, (0.1, 0, 0.3, 1))])
check("with a sighting it is the BRAIN's nudge",
      wrapped.bearing_error(None) == live.nudge())
check("  and the frontier bearing is untouched underneath",
      inner.bearing_error(None) == 0.5)

wrapped.note_progress(None)
check("note_progress is held back while the brain steers",
      inner.progress_calls == 0,
      "arrival is measured against the frontier, not the object")
live.clear()
wrapped.note_progress(None)
check("  and runs normally again once it stops", inner.progress_calls == 1)

check("reads forward to the real planner", wrapped.target == "a-frontier")
wrapped.target = None
check("WRITES forward too (recovery sets .target = None)", inner.target is None)
wrapped.targets_abandoned += 1
check("  including read-modify-write (recovery bumps the counter)",
      inner.targets_abandoned == 1,
      "if this lands on the wrapper the two objects disagree about the run")
wrapped.blacklist.add("k")
check("  and the blacklist is the real one", "k" in inner.blacklist)
check("the lock recovery takes is the real one", wrapped._lock is inner._lock)

live.poll(command={"label": "chair", "token": 2, "ttl_s": 90.0})
live.update([Det("chair", 0.9, (0.1, 0, 0.3, 1))])
check("the brain's target is NOT in .target, so recovery cannot blacklist it",
      wrapped.target is None and live.label == "chair")


# --------------------------------------------------------------------------
print("\nrl_control — the writer half\n")

rl_control.COMMAND_FILE = tmp / "written.json"
rl_control.STATE_DIR = tmp
msg = rl_control.target("Chair ")
check("writes a lowercased label", msg["label"] == "chair")
check("  with a monotonic token", msg["token"] > 0)
check("  readable by the run's reader",
      bt.read_command(rl_control.COMMAND_FILE)["label"] == "chair")
try:
    rl_control.target("")
    check("an empty label raises", False)
except ValueError:
    check("an empty label raises", True)

cleared = rl_control.clear()
check("clear writes an empty label", cleared["label"] == "")
check("  which the run reads as no order",
      bt.read_command(rl_control.COMMAND_FILE) is None)

rl_control.AUTONOMY_FILE = tmp / "rl_autonomy.json"
check("seen() is empty with no run out", rl_control.seen() == [])
check("describe() is empty with no run out", rl_control.describe() == "")
rl_control.AUTONOMY_FILE.write_text(json.dumps(
    {"seen": [{"label": "chair", "score": 0.9, "where": 0.4, "area": 0.1},
              {"label": "person", "score": 0.7, "where": -0.5, "area": 0.2}]}))
check("seen() reads the run's published detections", len(rl_control.seen()) == 2)
line = rl_control.describe()
check("describe() puts a positive where on the LEFT", "chair on my left" in line, line)
check("  and a negative where on the RIGHT", "person on my right" in line, line)
rl_control.AUTONOMY_FILE.write_text('{"seen": "not a list"}')
check("a malformed seen field is empty, not a crash", rl_control.seen() == [])


# --------------------------------------------------------------------------
print("\nTHE HEADING ORDER — Tier 2 steering, and the sign that costs a run\n")

import math                                             # noqa: E402

hcmd = tmp / "rl_heading.json"

hcmd.write_text('{"heading_deg": 45, "token": 1, "ttl_s": 20}')
got = bt.read_command(hcmd)
check("a heading order reads back as kind=heading",
      got == {"kind": "heading", "label": None, "heading_deg": 45.0,
              "token": 1, "ttl_s": 20.0}, f"got {got}")
hcmd.write_text('{"heading_deg": 200, "token": 2}')
check("a heading beyond +-180 is no order", bt.read_command(hcmd) is None)
hcmd.write_text('{"heading_deg": "left a bit", "token": 3}')
check("a non-numeric heading is no order", bt.read_command(hcmd) is None)
hcmd.write_text('{"heading_deg": 30, "label": "chair", "token": 4}')
check("a heading BEATS a label in the same message",
      bt.read_command(hcmd)["kind"] == "heading")

# -- the sign, stated the only way that matters: against a real pose ---------
h = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h.poll(now=0.0, command={"kind": "heading", "label": None,
                         "heading_deg": 90.0, "token": 10, "ttl_s": 90.0})
err = h.heading_error(0.0, now=0.0)          # facing theta=0
check("a POSITIVE wire heading gives a POSITIVE error (turn LEFT)",
      err is not None and err > 0, f"got {err}")
h2 = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h2.poll(now=0.0, command={"kind": "heading", "label": None,
                          "heading_deg": -90.0, "token": 11, "ttl_s": 90.0})
check("a NEGATIVE wire heading gives a NEGATIVE error (turn RIGHT)",
      (h2.heading_error(0.0, now=0.0) or 0) < 0)

# -- it decays as she turns, which is what stops her spinning ----------------
h3 = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h3.poll(now=0.0, command={"kind": "heading", "label": None,
                          "heading_deg": 90.0, "token": 12, "ttl_s": 90.0})
e0 = h3.heading_error(0.0, now=1.0)                  # armed here: target = +90 deg
e1 = h3.heading_error(math.radians(30), now=2.0)     # she has turned 30 left
e2 = h3.heading_error(math.radians(60), now=3.0)     # ...and 30 more
check("the error SHRINKS as she turns toward it",
      e0 > e1 > e2 > 0, f"got {e0:.3f} {e1:.3f} {e2:.3f}")
check("  it is the REMAINING turn, not the original order",
      abs(e1 - math.radians(60)) < 1e-6, f"got {math.degrees(e1):.1f} deg")
check("ARRIVING hands the wheel back (None, so the frontier planner resumes)",
      h3.heading_error(math.radians(88), now=4.0) is None)
check("  and it stays handed back — no re-trigger on drift",
      h3.heading_error(math.radians(70), now=5.0) is None,
      "without the latch she would hunt around the heading until the TTL ran out")

# -- wrap, expiry, and staying out of the label path -------------------------
h4 = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h4.poll(now=0.0, command={"kind": "heading", "label": None,
                          "heading_deg": 170.0, "token": 13, "ttl_s": 90.0})
h4.heading_error(math.radians(170), now=1.0)         # target wraps past +180
e_wrap = h4.heading_error(math.radians(-150), now=2.0)
check("the target wraps across +-180 instead of unwinding the long way",
      e_wrap is not None and abs(e_wrap) <= math.pi, f"got {e_wrap}")

h5 = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h5.poll(now=0.0, command={"kind": "heading", "label": None,
                          "heading_deg": 90.0, "token": 14, "ttl_s": 10.0})
check("a heading order expires on its TTL", h5.heading_error(0.0, now=11.0) is None)
check("  and expiry clears it", h5.kind is None)

h6 = bt.BrainTarget(command_file=hcmd, ttl_s=90.0)
h6.poll(now=0.0, command={"kind": "label", "label": "chair",
                          "heading_deg": None, "token": 15, "ttl_s": 90.0})
check("a LABEL order never answers heading_error",
      h6.heading_error(0.0, now=1.0) is None)

# -- through the planner wrapper, which is where the loop actually calls it --
fp = FakePlanner(bearing=0.5)
# THE CLOCK IS INJECTED HERE FOR THE REASON BrainTarget's OWN DOCSTRING GIVES:
# TargetedPlanner.bearing_error() calls heading_error() with no `now`, so it
# reads the clock itself. Polling at a fake now=0.0 against the wall clock makes
# the order look 60-odd days expired and the wrapper correctly falls through to
# the frontier planner -- a green test for broken code, or here a red one for
# working code. One clock, used by both.
fake_now = [1000.0]
hb = bt.BrainTarget(command_file=hcmd, ttl_s=90.0, clock=lambda: fake_now[0])
wrapped_h = bt.TargetedPlanner(fp, hb)
check("with no order the frontier planner answers, pose untouched",
      wrapped_h.bearing_error(None) == 0.5)
hb.poll(command={"kind": "heading", "label": None,
                 "heading_deg": 90.0, "token": 16, "ttl_s": 90.0})
be = wrapped_h.bearing_error([0.0, 0.0, 0.0])
check("a heading order OUTRANKS the frontier planner",
      be is not None and abs(be - math.radians(90)) < 1e-6, f"got {be}")
check("  and note_progress is held back while it steers",
      (wrapped_h.note_progress([0.0, 0.0, 0.0]), fp.progress_calls)[1] == 0)
check("once arrived the frontier planner gets the wheel back",
      wrapped_h.bearing_error([0.0, 0.0, math.radians(90)]) == 0.5)

# -- the writer does the negation exactly once ------------------------------
rl_control.COMMAND_FILE = tmp / "rl_command.json"
msg = rl_control.heading(-170.0)                     # "turn LEFT 170"
check("rl_control.heading negates: turn_right -170 -> wire +170 (LEFT)",
      msg["heading_deg"] == 170.0, f"got {msg['heading_deg']}")
check("  and it keeps what the caller actually asked for",
      msg["requested_turn_right_deg"] == -170.0)
for bad in (181.0, -181.0, "left"):
    try:
        rl_control.heading(bad)
        check(f"heading({bad!r}) is refused", False, "no exception raised")
    except ValueError:
        check(f"heading({bad!r}) is refused", True)


print(f"\n{PASS} passed, {FAIL} failed\n")
sys.exit(1 if FAIL else 0)
