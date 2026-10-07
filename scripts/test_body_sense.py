#!/usr/bin/env python3
"""She feels being picked up (body_sense.py + motor.parse_imu). No hardware, temp files.

    ./venv/bin/python scripts/test_body_sense.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-body-"))
config.BODY_FILE = TMP / "body.json"

import body_sense as bs                                 # noqa: E402
import motor                                            # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if not ok and detail else ""))


print("--- the board's G reply ---")
g = motor.parse_imu("G 9000 -1500 ok 0 0 64 3 9 -25 0 7")
check("+90 deg on the board is +90 here (+ = left; measured with her LiDAR, "
      "GYRO_SIGN +1)", g["yaw_deg"] == 90.0 and g["rate_dps"] == -15.0
      and config.GYRO_SIGN == 1, str(g))
check("...tilt, shake, jolts, tilted and bumps parsed",
      (g["tilt_deg"], g["shake_dps"], g["hpeak_g"], g["vpeak_g"], g["tilted"], g["bumps"])
      == (6.4, 0.3, 0.009, -0.025, False, 7), str(g))
check("GYRO 1's short reply still parses (no accelerometer fields)",
      motor.parse_imu("G 0 0 cal 0 0") == {"status": "cal", "ok": False, "yaw_deg": 0.0,
                                         "rate_dps": 0.0, "gaps": 0, "errors": 0})
check("old firmware (ERR), garbage, None -> None",
      motor.parse_imu("ERR") is None and motor.parse_imu("G x y") is None
      and motor.parse_imu(None) is None)
def _cs(line):
    x = 0
    for ch in line.encode():
        x ^= ch
    return f"{line}*{x:02X}"


_good = _cs("G 9000 -1500 ok 0 0 64 3 9 -25 0 0")
check("GYRO 6's checksummed line parses", motor.parse_imu(_good) is not None
      and motor.parse_imu(_good)["yaw_deg"] == 90.0)
check("...a digit lost on the wire (the 2026-10-07 '-24.6 read as -2.4'): refused",
      motor.parse_imu(_good.replace("9000", "900")) is None)
check("...a dropped field (11 of 12) or a cut status ('o'): refused, with or "
      "without a checksum",
      motor.parse_imu("G 9000 -1500 ok 0 64 3 9 -25 0 0") is None
      and motor.parse_imu("G 9000 -1500 o 0 0 64 3 9 -25 0 0") is None)
check("ok only when the gyro answers AND its bias is measured",
      motor.parse_imu("G 0 0 ok 0 0")["ok"] and not motor.parse_imu("G 0 0 cal 0 0")["ok"]
      and not motor.parse_imu("G 0 0 nogyro 0 0")["ok"])


def imu(tilt=6.5, shake=0.25, rate=0.0, tilted=False, bumps=0, status="ok"):
    return {"status": status, "ok": status == "ok", "yaw_deg": 0.0, "rate_dps": rate,
            "gaps": 0, "errors": 0, "tilt_deg": tilt, "shake_dps": shake,
            "hpeak_g": 0.01, "vpeak_g": -0.02, "tilted": tilted, "bumps": bumps}


def feed(det, sample, seconds, t0, driving=False, dt=0.1):
    out, t = [], t0
    while t < t0 + seconds - 1e-9:
        out += det.update(sample, t, driving)
        t += dt
    return out, t


print("--- picked up, put down ---")
d = bs.Detector()
ev, t = feed(d, imu(), 5.0, 0.0)
check("on the floor, still (measured: shake 0.2-0.3 deg/s): nothing", ev == [] and not d.lifted)
ev, t = feed(d, imu(shake=3.0), 0.4, t)
check("a hand's wobble for under 0.5 s: not yet (a knock is not a lift)", ev == [])
ev, t = feed(d, imu(shake=3.0), 0.3, t)
check("...held over 0.5 s (his hands measured 2.3-7.5 deg/s): LIFTED, once",
      ev == ["lifted"] and d.lifted, str(ev))
ev, t = feed(d, imu(shake=3.0), 3.0, t)
check("...no repeat while she stays up", ev == [])
ev, t = feed(d, imu(), 1.0, t)
check("level and still for under 1.2 s: still lifted (a pause in his hands)", ev == [] and d.lifted)
ev, t = feed(d, imu(), 0.5, t)
check("...1.2 s level and still: PUT DOWN", ev == ["put_down"] and not d.lifted, str(ev))

d = bs.Detector()
_, t = feed(d, imu(shake=6.0), 1.0, 0.0, driving=True)
ev, t = feed(d, imu(shake=20.0), 1.2, t)
check("her own rocking right after a hard pivot (shake 20 deg/s for 1.2 s after "
      "the wheels stop): NOT a lift -- the 2026-10-07 run said 'Whoa!' a dozen times",
      ev == [] and not d.lifted, str(ev))
ev, t = feed(d, imu(shake=3.0), 1.0, t)
check("...a hand's wobble once she has been still 1.5 s: lifted", ev == ["lifted"], str(ev))
run_d = bs.Detector(shake_lift=False, tilt_lift=False, bumps=False)
ev, _ = feed(run_d, imu(shake=28.0, tilt=29.0, bumps=1), 5.0, 0.0)
ev2, _ = feed(run_d, imu(shake=28.0, tilt=29.0, bumps=2), 1.0, 5.0)
check("the RUN's detector: shake 28 deg/s, a 29 deg tip over a threshold, bumps: "
      "nothing (the run rocks and tips itself)", ev == [] and ev2 == [], str(ev + ev2))
ev, _ = feed(run_d, imu(tilted=True, tilt=50.0), 0.2, 10.0)
check("...the board's own TILTED flag (past 35 deg): lifted, even in a run",
      ev == ["lifted"], str(ev))
d = bs.Detector()
ev, t = feed(d, imu(shake=6.0), 3.0, 0.0, driving=True)
check("driving shakes her: no lift from shake alone while the wheels drive",
      ev == [] and not d.lifted, str(ev))
ev, t = feed(d, imu(tilted=True), 0.1, t, driving=True)
check("...but the board saying TILTED is a lift even while driving",
      ev == ["lifted"], str(ev))

d = bs.Detector()
ev, t = feed(d, imu(tilt=40.0), 0.4, 0.0)
check("tipped past 25 deg for 0.3 s: lifted", ev == ["lifted"], str(ev))

print("--- while she is up ---")
d = bs.Detector()
_, t = feed(d, imu(shake=3.0), 0.7, 0.0)
ev, t = feed(d, imu(tilt=160.0, shake=3.0), 0.7, t)
check("upside down for 0.5 s: 'upside_down', once", ev == ["upside_down"], str(ev))
ev, t = feed(d, imu(tilt=160.0, shake=3.0), 5.0, t)
check("...not again while it lasts (at most every 8 s, and only once per spell)",
      ev.count("upside_down") <= 1, str(ev))
ev, t = feed(d, imu(shake=40.0), 0.7, t)
check("shaken hard: 'shaken'", "shaken" in ev, str(ev))
ev, t = feed(d, imu(shake=3.0, rate=200.0), 0.7, t)
check("spun round in his hands: 'spun'", "spun" in ev, str(ev))
d = bs.Detector()
_, t = feed(d, imu(shake=3.0), 0.7, 0.0)
ev, t = feed(d, imu(shake=3.0), 12.5, t)
check("held 12 s: 'held_long' (\"put me down, please?\"), once", ev.count("held_long") == 1, str(ev))

print("--- on the floor ---")
d = bs.Detector()
feed(d, imu(bumps=3), 1.0, 0.0)
ev, t = feed(d, imu(bumps=4), 0.1, 1.0)
check("a new jolt counted by the board: 'bump'", ev == ["bump"], str(ev))
ev, t = feed(d, imu(bumps=5), 0.1, 2.0)
check("...another within 8 s: not reacted to again", ev == [], str(ev))
d = bs.Detector()
ev, t = feed(d, imu(bumps=9), 1.0, 0.0)
check("the bump count she booted with is not a bump", ev == [], str(ev))
d = bs.Detector()
ev, _ = feed(d, imu(shake=2.5, status="cal"), 15.0, 0.0)
check("still measuring its bias after a reset (status cal, shake reads ~2.5 on a "
      "still floor): NO lift -- she said 'put me down' on the floor after two "
      "restarts", ev == [] and not d.lifted, str(ev))
check("no gyro / old firmware: no events, ever",
      bs.Detector().update(imu(status="nogyro", tilted=True), 0.0) == []
      and bs.Detector().update({"status": "ok", "ok": True, "yaw_deg": 0, "rate_dps": 0,
                                "gaps": 0, "errors": 0}, 0.0) == [])

print("--- the file between friday-mcp and friday-brain ---")
w = bs.BodyWatch(path=config.BODY_FILE)
bs.publish("lifted", True, 1)
check("a new event -> 'lifted' (AnyOf adds 'body: ')", w.step() == "lifted")
check("...read once", w.step() is None)
bs.publish("put_down", False, 2)
check("put down right after: not swallowed by the gap between reactions",
      w.step() == "put_down")
bs.publish("shaken", True, 3)
check("any other event inside 2.5 s of the last: skipped", w.step() is None)
w._last_react = 0.0
import json as _json                                       # noqa: E402
config.BODY_FILE.write_text(_json.dumps({"seq": 4, "event": "lifted",
                                         "t": time.time() - 10.0, "lifted": True}))
check("an event older than 3 s: not reacted to (she reacts in the moment)",
      w.step() is None)
bs.publish("spun", True, 5)
w.pause()
check("paused (she is mid-turn): dropped", w.step() is None)
w.resume()
bs.publish("spun", True, 6)
w2 = bs.BodyWatch(path=config.BODY_FILE)
check("a watch started now ignores the event already in the file", w2.step() is None)
import wake_word                                         # noqa: E402
w3 = bs.BodyWatch(path=config.BODY_FILE)
w3.poll_s = 0.02
any_of = wake_word.AnyOf(w3)
got_trig = []
threading_ = __import__("threading")
def _first():
    for trig in any_of.listen():
        got_trig.append(trig)
        break
th_ = threading_.Thread(target=_first, daemon=True)
th_.start()
time.sleep(0.2)
bs.publish("lifted", True, 99)
th_.join(3)
w3.close()
check("THROUGH wake_word.AnyOf, as main.py sees it: exactly 'body: lifted', which "
      "parse_trigger reads (it once came out 'body: body: lifted' and she did nothing)",
      got_trig == ["body: lifted"] and bs.parse_trigger(got_trig[0]) == "lifted",
      str(got_trig))
check("parse_trigger: ours and known only",
      bs.parse_trigger("body: lifted") == "lifted" and bs.parse_trigger("body: fly") is None
      and bs.parse_trigger("cpu_heat: 61") is None)
import face_director                                     # noqa: E402
faces = face_director.EMOTION_FACES | face_director.SAFETY_FACES
check("every reaction face is one the face director accepts as an emotion",
      all(f in faces for f, _ in config.BODY_REACTIONS.values()),
      str([f for f, _ in config.BODY_REACTIONS.values()]))
check("every reaction has lines, all short (EMO says a few words)",
      all(lines and all(len(x) <= 50 for x in lines)
          for _, lines in config.BODY_REACTIONS.values()))
check("the BodyWatch never sleeps: picking her up wakes her",
      w.sleep(600) == 0.0 and w.sleeping_for() == 0.0)

print("--- friday-mcp's reader, hardware faked ---")
seq = [imu()] * 5 + [imu(shake=3.0)] * 40 + [imu()] * 80     # 0.02 s polls


class FakeDrive:
    def __init__(self): self.i = 0
    def imu(self):
        s = seq[min(self.i, len(seq) - 1)]
        self.i += 1
        return s


from mcp_server import hardware                          # noqa: E402
from mcp_server.tools import movement                    # noqa: E402
fake = FakeDrive()
leased = [0.0]
driving = [""]
stops = []
hardware.lease_active = lambda: leased[0]
hardware.motors_if_present = lambda: fake
hardware.moving_direction = lambda: driving[0]
movement.stop = lambda: stops.append(1) or "stopped"
config.BODY_FILE = TMP / "body2.json"
s = bs.Sense(poll_s=0.02)
s.start()
time.sleep(3.5)
s.stop()
s.join(2)
got = bs.read()
check("the thread polls the board and writes the events (lifted, then put down)",
      got is not None and got["event"] == "put_down" and got["seq"] >= 2
      and got["lifted"] is False, str(got))
check("...and never stopped the wheels while they were not driving", stops == [])
fake2 = FakeDrive()
seq[:] = [imu(tilted=True)] * 5
hardware.motors_if_present = lambda: fake2
driving[0] = "forward"
s = bs.Sense(poll_s=0.02)
s.start()
time.sleep(0.3)
s.stop()
s.join(2)
check("picked up MID-MOVE: the move is stopped, as 'Friday, stop' would",
      stops != [] and bs.read()["event"] == "lifted", str(stops))
leased[0] = 30.0
fake3 = FakeDrive()
hardware.motors_if_present = lambda: fake3
s = bs.Sense(poll_s=0.02)
s.start()
time.sleep(0.3)
s.stop()
s.join(2)
check("a run holds the port (hardware lease): the board is not asked at all",
      fake3.i == 0, str(fake3.i))

print("--- wired in ---")
main_src = (ROOT / "main.py").read_text()
check("friday-brain: the BodyWatch is a wake source and _body_turn answers it",
      "body_sense.BodyWatch()" in main_src and "await _body_turn(" in main_src)
check("friday-mcp starts the reader", "body_sense.start()" in
      (ROOT / "mcp_server" / "server.py").read_text())
check("the reader only ever STOPS the wheels (no drive/move call in body_sense.py)",
      ".drive(" not in (ROOT / "body_sense.py").read_text()
      and "guarded_move" not in (ROOT / "body_sense.py").read_text())

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
