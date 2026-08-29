"""Voice: it must never block the control loop, and never say anything it cannot back up.

`friday/tts.py speak()` BLOCKS until Friday stops talking -- a second or more against a
100 ms control budget. Called from the loop it would blow the deadline by an order of
magnitude, the LiDAR slot would go stale, and the fail-safe would stop the robot;
speaking would be indistinguishable from a sensor fault. So the loop only enqueues, and
`say()` must be non-blocking even when the speaker is wedged, the queue is full, or the
TTS layer raises.

The second requirement is grounding. Every line must be built from a field that was
actually measured. Nothing here names an object, because no object detection is
implemented and a plausible sentence is worse than silence -- a listener cannot tell an
invented observation from a real one.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from narrator import (COOLDOWNS, MIN_GAP_S, Narrator, P_CHATTER, P_EXPLORE,
                      P_RECOVERY, P_SAFETY, QUEUE_MAX)

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def state(**over):
    base = {
        "CURRENT_MODE": "Autonomous exploration",
        "REASON_CODE": "ok",
        "SENSOR_STATE": {"lidar": {"min": 1.2}, "tof": {"nearest": 0.8}, "cameras": {}},
        "MAP_STATE": {"explored_m2": 10.0},
        "EXPLORATION_TARGET": {},
        "PLANNER_CONTRIBUTION": {},
        "RECOVERY_STATUS": {"state": "MONITOR"},
    }
    base.update(over)
    return base


print("--- narrator ---")

# 1. NON-BLOCKING, even against a speaker that never returns. This is the property the
#    control loop's timing depends on.
stuck = threading_ok = None
n = Narrator(speak=lambda t: time.sleep(30), enabled=True)
n.start()
t0 = time.perf_counter()
for i in range(50):
    n.observe(state(REASON_CODE="REFLEX_STOP" if i % 2 else "ok"))
elapsed = (time.perf_counter() - t0) * 1000
check("50 observe() calls against a WEDGED speaker stay fast",
      elapsed < 50.0, f"{elapsed:.1f} ms total")
check("...and the queue is bounded, not growing", n._q.qsize() <= QUEUE_MAX,
      f"{n._q.qsize()} queued")
n._stop.set()

# 2. A speaker that RAISES must not kill the thread or the run.
def boom(text):
    raise RuntimeError("speaker on fire")


n2 = Narrator(speak=boom, enabled=True)
n2.start()
n2.observe(state())
time.sleep(0.5)
check("a raising speaker does not kill the narrator thread", n2._thread.is_alive())
n2.observe(state(REASON_CODE="REFLEX_STOP"))
time.sleep(0.5)
check("...and it keeps accepting events afterwards", len(n2.log) >= 2, f"{len(n2.log)}")
n2.stop()

# 3. DISABLED means silent. Not quieter -- silent.
said = []
n3 = Narrator(speak=said.append, enabled=False)
n3.start()
for i in range(20):
    n3.observe(state(REASON_CODE="REFLEX_STOP" if i % 2 else "ok"))
time.sleep(0.3)
check("a disabled narrator says nothing at all", said == [] and n3.log == [])

# 4. TRANSITIONS, not levels. A condition that persists must be announced ONCE.
said = []
n4 = Narrator(speak=said.append, enabled=True)
n4.start()
for _ in range(30):
    n4.observe(state(REASON_CODE="REFLEX_STOP"))       # same condition, 30 ticks
time.sleep(0.6)
stops = [t for t in said if "Stopping" in t]
check("a persisting stop is announced once, not per tick", len(stops) == 1,
      f"{len(stops)} announcements")
n4.stop()

# 5. GROUNDING. The numbers spoken must be the numbers measured.
said = []
n5 = Narrator(speak=said.append, enabled=True)
n5.start()
n5.observe(state())
n5.observe(state(REASON_CODE="REFLEX_STOP",
                 SENSOR_STATE={"lidar": {"min": 1.2}, "tof": {"nearest": 0.37},
                               "cameras": {}}))
time.sleep(0.6)
check("the spoken distance is the measured one",
      any("0.37" in t for t in said), str(said))

said2 = []
n6 = Narrator(speak=said2.append, enabled=True)
n6.start()
n6.observe(state())
time.sleep(MIN_GAP_S + 0.1)
n6.observe(state(EXPLORATION_TARGET={"target_xy": (1.0, 2.0), "target_distance": 3.42},
                 PLANNER_CONTRIBUTION={"bearing_error": 0.9}))
time.sleep(0.6)
check("a target's distance and side are the planner's own",
      any("3.4" in t and "left" in t for t in said2), str(said2))
n6.stop()

# 6. NO INVENTED PERCEPTION. Nothing may name an object; none is detected.
banned = ("chair", "table", "person", "door", "wall ahead", "i see a", "looks like")
everything = " ".join(t.lower() for t in said + said2)
check("nothing spoken names an object", not any(b in everything for b in banned),
      everything[:100])

# 7. THROTTLING. The same kind cannot repeat inside its cooldown.
n7 = Narrator(speak=lambda t: None, enabled=True)
n7.start()
ok1 = n7.say("obstacle", "first", P_EXPLORE)
ok2 = n7.say("obstacle", "second", P_EXPLORE)
check("the first of a kind is accepted", ok1)
check("a second of the same kind inside its cooldown is suppressed", not ok2)
check("suppression is counted", n7.suppressed >= 1)
check("every event kind has a cooldown defined",
      all(k in COOLDOWNS for k in
          ("start", "target", "region", "obstacle", "safety_stop", "stuck",
           "recovered", "exhausted", "camera", "done")))

# 8. PRIORITY. Safety and recovery bypass the global gap; commentary does not. This
#    exists because a stop at t=0 was measured swallowing the recovery, the new target
#    AND the map milestone that followed within three seconds.
n8 = Narrator(speak=lambda t: None, enabled=True)
n8.start()
n8.say("safety_stop", "stopping", P_SAFETY)
check("a recovery event is NOT silenced by a stop moments earlier",
      n8.say("stuck", "stuck now", P_RECOVERY))
check("...but low-priority commentary IS held back",
      not n8.say("region", "mapped some", P_CHATTER))

# 9. Identical text is dropped even across kinds -- a rephrasing of a fact already
#    spoken is noise.
n9 = Narrator(speak=lambda t: None, enabled=True)
n9.start()
n9.say("stuck", "same words", P_RECOVERY)
check("identical text is not repeated", not n9.say("recovered", "same words", P_RECOVERY))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
