"""Friday's BODY for the A/B sim: a stand-in motor board on the serial line, and a
drivetrain model built from the numbers measured on the real robot.

The control loop runs exactly as on the robot (the real-hardware branch, the
version's own FridayMotorDriver, kick/pulse/floors, unstick, gyro). Only the far
end of the serial cable is simulated. Nothing here can open a real port: the
`serial.Serial` the driver constructs IS FakeBoard, and FakeBoard refuses any
port name but FAKE_PORT.

Drivetrain (memory: friday-drivetrain-calibration / -execution-floors, 10-02
reverse floor test):
  straight: ~45 cm/s near full duty; a wheel at rest needs duty 65 to break away,
            a turning one keeps going down to 50
  pivot:    4 wheels scrubbing sideways. Duty 117 is a coin toss, 153 turns 71
            deg/s. Break-away 110..130 (random per attempt), keeps going to 105
  reverse:  needs more: break-away 90, keeps going to 60
  right side 7% weaker (the lopsided turns); 0.12 s motor lag
  firmware watchdog: no M for longer than W ms -> stop
  THE RIGHT MOTOR IS WIRED REVERSED (the driver's invert_right defaults True):
  forward goes out as "M208 -208". The first version of this model read the
  wire literally -- forward became a spin and a spin a straight drive -- and
  every BODY result before 2026-10-20 20:05 (b1, the first b2) is wrong.
"""
from __future__ import annotations

import math
import random
import threading
import time

FAKE_PORT = "/dev/FAKE-FRIDAY-MOTORS"
VMAX = 0.55          # m/s per side at duty 255
B_EFF = 0.42         # effective track for skid steer, m
TAU = 0.12
RIGHT_WIRED_REVERSED = True   # a negative duty on the wire turns the right side FORWARD

BOARD = [None]
PHYS = {"v": 0.0, "omega": 0.0, "yaw_unwrapped": 0.0}


class FakeBoard:
    def __init__(self, port=None, baud=115200, timeout=1.0, write_timeout=None, **kw):
        if port != FAKE_PORT:
            raise RuntimeError(f"A/B sim refused to open {port!r} -- only {FAKE_PORT}")
        self.timeout = timeout
        self.is_open = True
        self._rx = bytearray()
        self._lock = threading.Lock()
        self.duties = (0, 0)
        self.last_m = time.monotonic()
        self.watchdog_s = 0.5
        self.world = None                      # set by the physics thread
        BOARD[0] = self

    # -- pyserial surface --------------------------------------------------
    @property
    def in_waiting(self):
        with self._lock:
            return len(self._rx)

    def read(self, n=1):
        with self._lock:
            out = bytes(self._rx[:n])
            del self._rx[:n]
            return out

    def readline(self):
        end = time.monotonic() + (self.timeout or 0)
        while True:
            with self._lock:
                i = self._rx.find(b"\n")
                if i >= 0:
                    out = bytes(self._rx[:i + 1])
                    del self._rx[:i + 1]
                    return out
            if time.monotonic() >= end:
                return b""
            time.sleep(0.002)

    def reset_input_buffer(self):
        with self._lock:
            self._rx.clear()

    def flush(self):
        pass

    def close(self):
        self.duties = (0, 0)
        self.is_open = False

    def _say(self, line):
        with self._lock:
            self._rx += (line + "\n").encode()

    def write(self, data):
        for raw in bytes(data).decode(errors="replace").split("\n"):
            line = raw.strip()
            if not line:
                continue
            c = line[0]
            if c == "P":
                self._say("FRIDAY-MOTORS 1 GYRO 7 (A/B sim)")
            elif c == "W":
                try:
                    self.watchdog_s = int(line[1:]) / 1000.0
                except ValueError:
                    pass
                self._say("OK")
            elif c == "M":
                try:
                    l, r = (int(x) for x in line[1:].split())
                except ValueError:
                    self._say("ERR")
                    continue
                self.duties = (max(-255, min(255, l)), max(-255, min(255, r)))
                self.last_m = time.monotonic()
                self._say("OK")
            elif c == "S":
                self.duties = (0, 0)
                self.last_m = time.monotonic()
                self._say("OK")
            elif c == "G":
                self._say(self._gyro_line())
            else:
                self._say("OK")
        return len(data)

    def _gyro_line(self):
        w = self.world
        if w is None:
            return "G 0 0 cal 0 0 80 1 0 0 0 0"
        yaw_cdeg = int(round(math.degrees(PHYS["yaw_unwrapped"]) * 100))
        rate = int(round(math.degrees(PHYS["omega"]) * 100))
        bumps = int(w.score.get("bumps", 0))
        body = f"G {yaw_cdeg} {rate} ok 0 0 80 1 0 0 0 {bumps}"
        cs = 0
        for ch in body.encode():
            cs ^= ch
        return f"{body}*{cs:02X}"


def _side(d, moving, mode, pivot_start, gain):
    """Wheel-side speed (m/s) for duty d. mode: 'pivot' | 'reverse' | 'drive'."""
    a = abs(d)
    if mode == "pivot":
        start, keep = pivot_start, 105
    elif mode == "reverse":
        start, keep = 90, 60
    else:
        start, keep = 65, 50
    if a < (keep if moving else start):
        return 0.0
    if mode == "pivot":
        # 153 -> 71 deg/s: omega = 2*s/B_EFF -> s = 1.24*0.42/2 = 0.26 m/s at 153;
        # it levels off above: pwm 189 measured ~96 deg/s (104 right, 88 left).
        s = (0.26 * (a - 105) / (153 - 105) if a <= 153
             else 0.26 + 0.26 * (1.68 / 1.24 - 1.0) * (a - 153) / (189 - 153))
    else:
        s = VMAX * (a - 50) / (255 - 50)
    return math.copysign(s * gain, d)


def start_physics(world, hz=50.0):
    """Integrate the board's duties into the world at `hz`, forever (daemon)."""
    rng = random.Random(7)
    state = {"v": 0.0, "w": 0.0, "moving_l": False, "moving_r": False,
             "pivot_start": None}

    def loop():
        dt = 1.0 / hz
        last = time.monotonic()
        while True:
            time.sleep(dt)
            now = time.monotonic()
            h = now - last
            last = now
            b = BOARD[0]
            if b is None or not b.is_open:
                l = r = 0
            else:
                b.world = world
                l, r = b.duties
                if RIGHT_WIRED_REVERSED:
                    r = -r
                if now - b.last_m > b.watchdog_s:          # firmware watchdog
                    l = r = 0
            pivot = (l > 0 > r) or (r > 0 > l)
            rev = l < 0 and r < 0
            # one random break-away threshold per pivot attempt, not per step
            if pivot and state["pivot_start"] is None:
                state["pivot_start"] = rng.uniform(110, 130)
            if not pivot:
                state["pivot_start"] = None
            mode = "pivot" if pivot else ("reverse" if rev else "drive")
            sl = _side(l, state["moving_l"], mode, state["pivot_start"], 1.0)
            sr = _side(r, state["moving_r"], mode, state["pivot_start"], 0.93)
            state["moving_l"], state["moving_r"] = sl != 0.0, sr != 0.0
            v_t = (sl + sr) / 2.0
            w_t = (sr - sl) / B_EFF
            k = min(1.0, h / TAU)
            state["v"] += (v_t - state["v"]) * k
            state["w"] += (w_t - state["w"]) * k
            if abs(state["v"]) < 1e-3:
                state["v"] = 0.0
            if abs(state["w"]) < 1e-3:
                state["w"] = 0.0
            PHYS["v"], PHYS["omega"] = state["v"], state["w"]
            PHYS["yaw_unwrapped"] += state["w"] * h
            world.step(state["v"], state["w"], h)

    t = threading.Thread(target=loop, name="ab-physics", daemon=True)
    t.start()
    return t
