"""The director must notice that something else moved the face.

Harsh, 2026-09-03: "while tinkering between the different facial expressions
the screen goes blank." It was the idle drift reaching "sleeping" -- closed
eyes on a black panel -- over the top of what he had just set, because the
director could not see any writer but itself. The timings are shrunk here so
the whole 300 s drift plays out in a couple of seconds.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
config.FACE_BORED_AFTER_S = 0.3
config.FACE_SLEEPY_AFTER_S = 0.6
config.FACE_SLEEPING_AFTER_S = 0.9
config.FACE_TICK_S = 0.05
import face_control, face_director

fails = 0
def check(n, ok, d=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {n} {d}")
    if not ok: fails += 1

d = face_director.FaceDirector().start()
time.sleep(0.2)
check("starts on neutral", d.state()["face"] == "neutral", str(d.state()))

# Left alone, it must still drift all the way down -- that behaviour is wanted.
time.sleep(1.3)
check("still drifts to sleeping when nothing writes",
      d.state()["face"] == "sleeping", str(d.state()))

# Now somebody else writes the face, the way face_control.py or the exploration
# supervisor does. The drift must restart from that, not run over it.
face_control.set_emotion("curious")
time.sleep(0.2)
check("adopts a face written by someone else",
      d.state()["face"] == "curious", str(d.state()))
check("...at idle priority, so the voice still outranks it",
      d.state()["priority"] == face_director.IDLE, str(d.state()))

# Keep writing, the way FaceExpression re-asserts every few seconds. Through a
# whole run's worth of drift windows the screen must never go to sleeping.
blanked = []
for _ in range(30):                       # 30 * 0.08 = 2.4 s, past SLEEPING_AFTER
    face_control.set_emotion("thinking")
    time.sleep(0.08)
    if d.state()["face"] in ("sleepy", "sleeping"):
        blanked.append(d.state()["face"])
check("a writer that keeps re-asserting is never blanked", not blanked,
      str(blanked))

# And the voice pipeline still wins over an adopted face.
d.pipeline("listening")
time.sleep(0.15)
check("the voice pipeline outranks an adopted face",
      d.state()["face"] == "listening", str(d.state()))

# Stop writing and the drift comes back -- the idle life is not lost.
time.sleep(0.1)
d.rest()
time.sleep(1.4)
check("drift resumes once the other writer stops",
      d.state()["face"] == "sleeping", str(d.state()))
d.close()
print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
