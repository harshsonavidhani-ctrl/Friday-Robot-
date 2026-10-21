#!/usr/bin/env python3
"""The laptop helper's autopilot (helper 10, 2026-10-21) -- the fast half of
playing a fast game. No Windows, no laptop: a pretend car on an S-shaped road
draws a pretend GTA minimap from the keys the autopilot presses, in real time.

    ./venv/bin/python scripts/test_autopilot.py      # ~30 s
"""
from __future__ import annotations

import http.client
import json
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import friday_laptop_agent as fa                          # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


GAME = "Grand Theft Auto V Enhanced"
MW, MH = 160, 110                    # the pretend minimap
M_PER_PX = 0.6
ROAD_HALF = 4.5                      # metres


def road_y(x: float) -> float:
    return 30.0 * math.sin(x / 45.0)


class Car:
    """x along the road, y across; heading 0 = +x. Moves by the keys held."""

    def __init__(self, blocked_at: float | None = None) -> None:
        self.x, self.y, self.h, self.v = 0.0, road_y(0.0), math.atan2(30 / 45, 1), 0.0
        self.down: set[str] = set()
        self.t = time.monotonic()
        self.worst = 0.0
        self.blocked_at = blocked_at
        self.lock = threading.Lock()
        self.reversed = False

    def press(self, key: str, up: bool) -> None:
        with self.lock:
            self._advance()
            (self.down.discard if up else self.down.add)(key)
            if not up and key == "s":
                self.reversed = True

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self.t = now - self.t, now
        target = 9.0 if "w" in self.down else (-3.0 if "s" in self.down else 0.0)
        self.v += (target - self.v) * min(1.0, dt * 1.5)
        steer = (1 if "d" in self.down else 0) - (1 if "a" in self.down else 0)
        self.h -= steer * 0.9 * dt * max(-1.0, min(1.0, self.v / 4.0))
        nx = self.x + math.cos(self.h) * self.v * dt
        if self.blocked_at is not None and nx > self.blocked_at:
            nx, self.v = self.blocked_at, 0.0
        self.y += math.sin(self.h) * self.v * dt
        self.x = nx
        self.worst = max(self.worst, abs(self.y - road_y(self.x)))

    def grab(self, box, max_w):
        with self.lock:
            self._advance()
            if box == (0.0, 0.0, 1.0, 1.0):          # the stuck test's thumbnail
                w, h = 24, 14
                shade = int(self.x * 7) % 200
                return w, h, bytes([shade, shade, shade, 255]) * (w * h)
            out = bytearray(MW * MH * 4)
            ax, ay = fa.ARROW_AT[0] * MW, fa.ARROW_AT[1] * MH
            fx, fy = math.cos(self.h), math.sin(self.h)       # forward
            rx, ry = -fy, fx                                  # right (y is "left" here)
            for py in range(MH):
                fwd = (ay - py) * M_PER_PX
                for px in range(MW):
                    side = (px - ax) * M_PER_PX
                    wx = self.x + fwd * fx - side * rx
                    wy = self.y + fwd * fy - side * ry
                    road = abs(wy - road_y(wx)) < ROAD_HALF
                    g = 130 if road else 55
                    i = (py * MW + px) * 4
                    out[i:i + 4] = bytes((g, g, g, 255))
            for dy in range(7):                               # the white arrow, tip up
                for dx in range(-(dy // 2), dy // 2 + 1):
                    x, y = int(ax) + dx, int(ay) - 3 + dy
                    i = (y * MW + x) * 4
                    out[i:i + 4] = bytes((255, 255, 255, 255))
            return MW, MH, bytes(out)


print("reading the minimap")
car = Car()
w, h, bgra = car.grab(fa.MAP_BOX, 320)
nav = fa.read_map(w, h, bgra)
check("finds the arrow pointing up the map", nav is not None and abs(nav["heading"]) <= 10, str(nav))
check("...and sees road straight ahead", nav and nav["ahead"] > 20, str(nav))
check("parse_box", fa.parse_box("0,0.78,0.16,0.99") == (0.0, 0.78, 0.16, 0.99)
      and fa.parse_box("0.5,0.2,0.1,0.9") is None and fa.parse_box("x") is None)

print("driving the S-bend (12 s, real time)")
TITLE = {"t": "Grand Theft Auto V"}
ap = fa.Autopilot(lambda: TITLE["t"], grabber=car.grab, presser=car.press)
fa_cfg = {"game": GAME, "mode": "road", "keys": dict(fa.AUTO_KEYS), "hold": [],
          "box": fa.MAP_BOX, "bias": 0.0, "speed": 1.0, "for_s": 60}
ap.set(fa_cfg)
end = time.monotonic() + 12
while time.monotonic() < end:
    ap.renew()
    time.sleep(0.5)
rep = ap.report()
check("drove on: the car went a long way", car.x > 60, f"x={car.x:.1f} m, report {rep}")
check(f"...and stayed on the road (worst {car.worst:.1f} m off centre, road half {ROAD_HALF} m)",
      car.worst < ROAD_HALF, f"x={car.x:.1f}")
check("...ticking ~10 times a second", rep.get("ticks", 0) > 60, str(rep.get("ticks")))

print("safety")
TITLE["t"] = "Command Prompt"
time.sleep(0.4)
check("the game leaves the front -> it stops", not ap.running(), str(ap.report()))
check("...and every key is up", not car.down, str(car.down))
check("...and says why", "front" in ap.report().get("why", ""), str(ap.report()))
TITLE["t"] = "Grand Theft Auto V"
fa.AUTO_LEASE_S, lease = 1.0, fa.AUTO_LEASE_S
ap.set(dict(fa_cfg))
time.sleep(1.6)
check("no word from the Pi for AUTO_LEASE_S -> it stops, keys up",
      not ap.running() and not car.down and "checking in" in ap.report().get("why", ""),
      str(ap.report()))
fa.AUTO_LEASE_S = lease
ap.set(dict(fa_cfg, mode="cruise", hold=["right"]))
time.sleep(0.4)
check("cruise holds exactly the keys asked for", car.down == {"right"}, str(car.down))
ap.stop("test")
check("stop -> keys up", not car.down and not ap.running(), str(car.down))

print("stuck on a wall")
wall = Car(blocked_at=6.0)
ap2 = fa.Autopilot(lambda: "Grand Theft Auto V", grabber=wall.grab, presser=wall.press)
ap2.set(dict(fa_cfg))
end = time.monotonic() + 6
while time.monotonic() < end and not ap2.report().get("stucks"):
    ap2.renew()
    time.sleep(0.3)
check("gas on, picture frozen -> it backs out (brake + opposite steer)",
      ap2.report().get("stucks", 0) >= 1 and wall.reversed, str(ap2.report()))
ap2.stop("test")

print("through the helper's HTTP routes")
held = []
agent = fa.Agent({"token": "t" * 40, "port": 0, "bind": "127.0.0.1"},
                 titler=lambda: "Grand Theft Auto V", grabber=car.grab, autokey=car.press,
                 holder=lambda k, ms, f: held.append(k) or "", shooter=lambda *a: b"\x89PNG")
srv = ThreadingHTTPServer(("127.0.0.1", 0), agent.handler())
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method: str, path: str, body: dict | None = None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    data = json.dumps(body) if body is not None else None
    hdr = {"X-Friday-Token": "t" * 40}
    if data:
        hdr["Content-Length"] = str(len(data))
    c.request(method, path, data, hdr)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


st, got = call("POST", "/autopilot", {"game": GAME, "mode": "road", "for_s": 30})
check("POST /autopilot road -> running", st == 200 and got.get("running"), str(got))
st, got = call("GET", "/autopilot")
check("GET /autopilot reports it", st == 200 and got.get("running") and "ticks" in got, str(got))
st, _ = call("POST", "/step", {"game": GAME, "keys": ["f"], "ms": 100})
time.sleep(0.2)
check("a manual /step switches it off first", not agent.autopilot.running()
      and agent.autopilot.report().get("why") == "a manual move", str(agent.autopilot.report()))
for bad in ({"game": GAME, "mode": "fly"}, {"game": GAME, "mode": "road", "keys": {"gas": "alt"}},
            {"game": GAME, "mode": "cruise"}, {"game": GAME, "mode": "cruise", "hold": ["f4"]},
            {"game": GAME, "mode": "road", "keys": {"nitro": "n"}},
            {"game": GAME, "mode": "road", "box": "1,1,0,0"}):
    st, _ = call("POST", "/autopilot", bad)
    check(f"refused: {json.dumps(bad)[len(GAME) + 14:]}", st == 400, str(st))
st, got = call("POST", "/autopilot", {"game": GAME, "mode": "road",
                                       "tune": {"lookahead": 0.5, "road_lo": 100}})
check("tune: numbers in range are taken (lookahead held to its 0.3 top)",
      st == 200 and agent.autopilot.cfg["tune"]["lookahead"] == 0.3
      and agent.autopilot.cfg["tune"]["road_lo"] == 100, str(agent.autopilot.cfg.get("tune")))
call("POST", "/autopilot", {"mode": "off"})
st, _ = call("POST", "/autopilot", {"game": GAME, "mode": "road", "tune": {"exec": 1}})
check("tune: an unknown setting -> 400", st == 400)
st, _ = call("POST", "/autopilot", {"game": "Solitaire", "mode": "road"})
check("a game that is not in front -> 409", st == 409)
call("POST", "/autopilot", {"game": GAME, "mode": "road"})
st, got = call("POST", "/autopilot", {"mode": "off"})
check("mode off -> stopped, keys up", st == 200 and not got.get("running") and not car.down,
      str(got))
st, got = call("GET", "/screen?game=" + GAME.replace(" ", "+") + "&box=0,0.78,0.16,0.99")
check("/screen with a box -> a picture", st == 200 and got == b"\x89PNG", str(st))
st, _ = call("GET", "/screen?game=" + GAME.replace(" ", "+") + "&box=0,0,2,1")
check("/screen with a bad box -> 400", st == 400)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
