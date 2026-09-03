"""Friday's face must say what she is actually doing.

Harsh, 2026-09-03: "the facial expressions are not matching the tasks it is
performing." They were not matching because nothing in the autonomous path ever
set them -- tts.speak() was the only writer in the whole stack.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import face_control
import friday_exploration as fe
import friday_expression as fx
from friday_expression import FaceExpression, expression_for

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


print("--- exploration expressions ---")

# 1. THE VOCABULARY IS NOT OURS TO INVENT. friday_face.py refuses to start unless it
#    has geometry for every name in face_control.EMOTIONS, so a name this module makes
#    up would be silently ignored at best.
phases = (fx.BOOTSTRAP, fx.SELECT, fx.NAVIGATE, fx.OBSERVE, fx.RECOVER, fx.COMPLETE)
gates = ("ok", "governor", "planner", "tof_slow", "REFLEX_STOP", "visual_stop",
         "stale_lidar", "stale_tof", "recovery", "escape_turn", "no_reverse_turn",
         "reverse_lifted", "rear_blocked")
produced = {expression_for(p, g, fwd_min=d, just_reached=r, just_found=f)
            for p in phases for g in gates for d in (None, 0.1, 0.5)
            for r in (False, True) for f in (False, True)}
check("every expression is a real emotion",
      produced <= set(face_control.EMOTIONS),
      f"{len(produced)} used, unknown: {sorted(produced - set(face_control.EMOTIONS))}")

# 2. NEVER "speaking". tts._Mouth.start() saves the current emotion to restore after
#    the sentence, so a second writer setting "speaking" makes tts save and restore
#    INTO it -- Friday then talks forever with nothing to say. main.py carries the
#    same warning because it is the bug that actually happened.
check("never sets the speaking face -- tts owns it", "speaking" not in produced)

# 3. The phase names must be the supervisor's. They are duplicated in friday_expression
#    to keep it dependency-free (the supervisor imports IT), so drift is possible and
#    would silently fall through to "neutral" for every tick of a whole phase.
check("phase names match friday_exploration.Phase",
      (fx.BOOTSTRAP, fx.SELECT, fx.NAVIGATE, fx.OBSERVE, fx.RECOVER, fx.COMPLETE) ==
      (fe.Phase.BOOTSTRAP, fe.Phase.SELECT, fe.Phase.NAVIGATE,
       fe.Phase.OBSERVE, fe.Phase.RECOVER, fe.Phase.COMPLETE))

# 4. Danger outranks everything, and how close it is changes the answer. The safety
#    chain has already acted by the time these gates appear; the face reports it.
check("a reflex stop reads as alarm",
      expression_for(fx.NAVIGATE, "REFLEX_STOP", fwd_min=0.5) == "surprised")
check("...and nearly touching reads as fear",
      expression_for(fx.NAVIGATE, "REFLEX_STOP", fwd_min=0.10) == "scared")
check("danger outranks a just-reached target",
      expression_for(fx.NAVIGATE, "REFLEX_STOP", fwd_min=0.5, just_reached=True)
      == "surprised")

# 5. Blindness is not fear. A stale sensor means she cannot see, which is a different
#    thing to report than an obstacle she can.
for gate in ("stale_lidar", "stale_tof"):
    check(f"{gate} reads as confusion, not alarm",
          expression_for(fx.NAVIGATE, gate) == "confused")

# 6. Being stuck. "dizzy" specifically for the spin-in-place recovery, because that is
#    what it looks like from outside and the face should agree with the wheels.
check("the recovery spin reads as dizzy",
      expression_for(fx.NAVIGATE, "recovery") == "dizzy")
check("being unable to progress reads as confused",
      expression_for(fx.RECOVER, "ok") == "confused")

# 7. Ordinary work still looks like something. This is the common case -- most ticks of
#    a good run are exactly this -- and a blank face through it is the original bug.
check("driving to a frontier reads as curious",
      expression_for(fx.NAVIGATE, "ok") == "curious")
check("braking for something reads as thinking, not alarm",
      expression_for(fx.NAVIGATE, "governor") == "thinking")
check("choosing a target reads as thinking",
      expression_for(fx.SELECT, "ok") == "thinking")
check("a finished map reads as proud", expression_for(fx.COMPLETE, "ok") == "proud")

# 8. The writer must not hammer the face. It is a separate process on a CPU-bound Pi
#    reading a state file; ten writes a second for an unchanged value is pure waste.
class _Rec:
    def __init__(self): self.writes = []
    def set_emotion(self, name): self.writes.append(name)


rec = _Rec()
sys.modules["face_control"] = rec           # FaceExpression imports it per call
face = FaceExpression()
for _ in range(20):
    face.update(fx.NAVIGATE, "ok")
check("an unchanged expression is written once, not every tick",
      len(rec.writes) == 1, f"{len(rec.writes)} writes over 20 ticks")
face.update(fx.NAVIGATE, "REFLEX_STOP", fwd_min=0.5)
check("...but a change is written immediately",
      rec.writes[-1] == "surprised", str(rec.writes))

# 9. A missing face must never cost a control tick. It is a separate process that may
#    not be running at all -- the same contract tts._Mouth works to.
class _Broken:
    @staticmethod
    def set_emotion(name): raise RuntimeError("no face process")


sys.modules["face_control"] = _Broken
broken = FaceExpression()
try:
    got = broken.update(fx.NAVIGATE, "ok")
    check("a dead face process does not raise", True, f"returned {got!r}")
    check("...and it stops trying rather than throwing every tick", not broken.enabled)
except Exception as exc:                                # noqa: BLE001
    check("a dead face process does not raise", False, repr(exc))

# 10. The face must not strobe. In a tight space the gate changes several times a
#     second, and mapping that straight through flickered faster than the face's own
#     280 ms cross-fade could finish -- a transition that never completes is not an
#     expression. But ALARM must never wait behind a calmer face.
sys.modules["face_control"] = (rec2 := _Rec())
dwell = FaceExpression()
dwell.update(fx.NAVIGATE, "ok")                 # curious
dwell.update(fx.SELECT, "ok")                   # thinking, too soon
dwell.update(fx.NAVIGATE, "recovery")           # dizzy, too soon
check("a calm expression is held long enough to be seen",
      rec2.writes == ["curious"], str(rec2.writes))
check("...and the suppressed changes are counted", dwell.suppressed == 2,
      str(dwell.suppressed))
dwell.update(fx.NAVIGATE, "REFLEX_STOP", fwd_min=0.10)
check("alarm never waits behind a calmer face",
      rec2.writes[-1] == "scared", str(rec2.writes))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
