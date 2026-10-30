"""She feels her body: picked up, put down, upside down, shaken, spun, bumped.

Harsh, 2026-10-07: "if I pick it up like it reacts accordingly add like the
emo robot does". The motor board reads an MPU-6500 (firmware GYRO 2+, see
motor.parse_imu); this file turns its numbers into EVENTS and her reactions.

    friday-mcp:   Sense (thread)  -- 10x a second, motor.Motors.imu()
                     Detector     -- numbers -> events      (pure, tested)
                     publish()    -- state/body.json         (atomic)
    friday-brain: BodyWatch       -- a wake source: "body: <event>"
                     main._body_turn -- the face + one fixed line, no model

Only while no run holds the motors: a run owns the motor port (the hardware
lease) and friday-mcp cannot ask the board anything then.

SAFETY. Being picked up while a move is running stops the move (stop() is the
only thing this file ever sends to the wheels). The board stops the motors
itself when she is tilted past 35 deg (firmware); a level lift is caught here.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import config

log = logging.getLogger("friday.body")

NAME = "body"


# --- numbers -> events ---------------------------------------------------------

class Detector:
    """One IMU reading at a time -> a list of events (usually empty).

    LIFTED: the board says tilted, or tilt past BODY_LIFT_TILT_DEG for
    BODY_LIFT_TILT_S, or -- only while the wheels are not driving, because
    driving shakes her too -- roll/pitch "shake" over BODY_LIFT_SHAKE_DPS for
    BODY_LIFT_SHAKE_S. A hand holding her is never as still as the floor.
    PUT DOWN: level and still for BODY_DOWN_S. While lifted: upside_down,
    shaken, spun (each once per BODY_EVENT_S of it, at most every
    BODY_REPEAT_S) and held_long once. On the ground: bump when the board
    counts a new horizontal jolt.
    """

    def __init__(self, cfg=config, shake_lift: bool = True, tilt_lift: bool = True,
                 bumps: bool = True):
        """shake_lift / tilt_lift / bumps off = the RL run's detector: there
        the robot rocks and tips itself all the time, so only the board's own
        TILTED flag (past 35 deg) says lifted, and bumps are the run's to
        announce (2026-10-07: false lifts every 8-15 s mid-run)."""
        self.c = cfg
        self.shake_lift, self.tilt_lift, self.bumps_on = shake_lift, tilt_lift, bumps
        self._last_drive = -1e9
        self.lifted = False
        self.lifted_at = 0.0
        self._shake_since = None
        self._tilt_since = None
        self._down_since = None
        self._since: dict[str, float | None] = {}
        self._last: dict[str, float] = {}
        self._held_said = False
        self._bumps = None
        self.fault = False              # the chip is talking nonsense: see _impossible
        self._recent: deque = deque()

    def _impossible(self, tilt: float, shake: float, now: float) -> bool:
        """A reading no body can produce: hard shaking that never varies, on a tilt
        that never moves. 2026-10-30: tilt 80, shake 52 for two minutes while the
        ToF and LiDAR stood perfectly still; 2026-10-19: a steady 71.4. Both a
        half-connected chip after a knock. She said "dizzy" every 5 s, and a real
        hand is never that steady."""
        c = self.c
        self._recent.append((now, tilt, shake))
        while now - self._recent[0][0] > c.BODY_FAULT_S:
            self._recent.popleft()
        if now - self._recent[0][0] < c.BODY_FAULT_S * 0.9:
            return False
        tilts = [r[1] for r in self._recent]
        shakes = [r[2] for r in self._recent]
        return (min(shakes) >= c.BODY_SHAKEN_DPS
                and max(shakes) - min(shakes) <= c.BODY_FAULT_SHAKE_SPAN
                and max(tilts) - min(tilts) <= c.BODY_FAULT_TILT_SPAN)

    def _held(self, key: str, cond: bool, now: float, need_s: float) -> bool:
        if not cond:
            self._since[key] = None
            return False
        if self._since.get(key) is None:
            self._since[key] = now
        return now - self._since[key] >= need_s

    def _once(self, key: str, now: float) -> bool:
        if now - self._last.get(key, -1e9) < self.c.BODY_REPEAT_S:
            return False
        self._last[key] = now
        return True

    def update(self, imu: dict | None, now: float, driving: bool = False) -> list[str]:
        # Only once the board says "ok": while it is still measuring its bias
        # ("cal", ~1-6 s after every reset -- each friday-mcp start resets it)
        # the roll/pitch "shake" is not zeroed and reads ~2.5 deg/s on a still
        # floor, and she announced a lift and "put me down" after the 21:02 and
        # 21:16 restarts (2026-10-07) sitting on the floor.
        if not imu or "tilt_deg" not in imu or imu.get("status") != "ok":
            return []
        c = self.c
        tilt, shake = imu["tilt_deg"], imu["shake_dps"]
        self.fault = self._impossible(tilt, shake, now)
        if self.fault:
            self.lifted = False         # she was never lifted
            self._since.clear()
            return []
        out: list[str] = []
        bumps = imu.get("bumps", 0)
        new_bump = self._bumps is not None and bumps > self._bumps
        self._bumps = bumps

        if driving:
            self._last_drive = now
        if not self.lifted:
            tilted = bool(imu.get("tilted")) or (self.tilt_lift and self._held(
                "tilt", tilt > c.BODY_LIFT_TILT_DEG, now, c.BODY_LIFT_TILT_S))
            settled = now - self._last_drive >= getattr(c, "BODY_STILL_AFTER_DRIVE_S", 0.0)
            shaky = self.shake_lift and settled and self._held(
                "shake", shake > c.BODY_LIFT_SHAKE_DPS, now, c.BODY_LIFT_SHAKE_S)
            if tilted or shaky:
                self.lifted, self.lifted_at, self._held_said = True, now, False
                self._since.clear()
                out.append("lifted")
            elif new_bump and self.bumps_on and self._once("bump", now):
                out.append("bump")
            return out

        # -- lifted ------------------------------------------------------------
        # The run's detector (shake_lift off) lifts on the board's TILTED flag alone,
        # so level and no longer tilted is what releases it. Asking it for stillness
        # too left the wheels frozen 701 of 1454 ticks on 2026-10-30, tilt 1-4 deg and
        # shake 4-8: her own LiDAR and rocking never let it read under 0.8.
        calm = shake < c.BODY_DOWN_SHAKE_DPS or not self.shake_lift
        if self._held("down", tilt < c.BODY_LEVEL_DEG and calm
                      and not imu.get("tilted"), now, c.BODY_DOWN_S):
            self.lifted = False
            self._since.clear()
            out.append("put_down")
            return out
        if (self._held("upside", tilt > c.BODY_UPSIDE_DOWN_DEG, now, c.BODY_EVENT_S)
                and self._once("upside_down", now)):
            out.append("upside_down")
        # shake_lift off = the run's detector: nobody is shaking a robot that is
        # driving itself, and this line is what said "dizzy" every 5 s on 10-30.
        if (self.shake_lift
                and self._held("shaken", shake > c.BODY_SHAKEN_DPS, now, c.BODY_EVENT_S)
                and self._once("shaken", now)):
            out.append("shaken")
        if (self._held("spun", abs(imu.get("rate_dps", 0.0)) > c.BODY_SPUN_DPS,
                       now, c.BODY_EVENT_S) and self._once("spun", now)):
            out.append("spun")
        if not self._held_said and now - self.lifted_at >= c.BODY_HELD_LONG_S:
            self._held_said = True
            out.append("held_long")
        return out


# --- the file between the two processes ------------------------------------------

def publish(event: str | None, lifted: bool, seq: int, path: Path | None = None,
            **detail) -> None:
    """Atomic write of the latest event (or just the state, event=None)."""
    path = Path(path or config.BODY_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seq": seq, "event": event, "t": time.time(),
                   "lifted": bool(lifted), **detail}
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".body.", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception as exc:                            # noqa: BLE001
        log.debug("could not write %s: %s", path, exc)


def read(path: Path | None = None) -> dict | None:
    try:
        return json.loads(Path(path or config.BODY_FILE).read_text())
    except Exception:                                   # noqa: BLE001
        return None


# --- friday-mcp: the reader ------------------------------------------------------

class Sense(threading.Thread):
    """Polls the board while friday-mcp holds the motors; never raises."""

    def __init__(self, poll_s: float | None = None):
        super().__init__(name="body-sense", daemon=True)
        self.poll_s = float(poll_s or config.BODY_POLL_S)
        self.detector = Detector()
        self.seq = (read() or {}).get("seq", 0)
        self._stop = threading.Event()
        self._last_state = 0.0
        self._last_said = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from mcp_server import hardware
        log.info("body sense running: %.0f Hz on the motor board's IMU", 1 / self.poll_s)
        while not self._stop.wait(self.poll_s):
            try:
                if hardware.lease_active() > 0.0:
                    continue                    # a run holds the port
                drive = hardware.motors_if_present()
                imu = drive.imu() if drive is not None else None
                if imu is None:
                    continue
                now = time.monotonic()
                driving = hardware.moving_direction() != ""
                for event in self.detector.update(imu, now, driving):
                    self.seq += 1
                    log.info("body: %s (tilt %.0f deg, shake %.1f deg/s)", event,
                             imu.get("tilt_deg", 0.0), imu.get("shake_dps", 0.0))
                    if event == "lifted" and driving:
                        self._stop_moving()
                    publish(event, self.detector.lifted, self.seq,
                            tilt_deg=imu.get("tilt_deg"))
                if time.monotonic() - self._last_state > 2.0:
                    self._last_state = time.monotonic()
                    # A put-down that is never seen leaves her "lifted", and
                    # then picking her up again says nothing (2026-10-19).
                    if self.detector.lifted and now - self._last_said > 5.0:
                        self._last_said = now
                        log.info("body: still lifted -- tilt %.1f deg, shake %.2f deg/s"
                                 "%s (put down = level < %.0f and shake < %.1f for %.1f s)",
                                 imu.get("tilt_deg", 0.0), imu.get("shake_dps", 0.0),
                                 ", board TILTED" if imu.get("tilted") else "",
                                 config.BODY_LEVEL_DEG, config.BODY_DOWN_SHAKE_DPS,
                                 config.BODY_DOWN_S)
                    cur = read() or {}
                    if cur.get("lifted") != self.detector.lifted:
                        publish(cur.get("event"), self.detector.lifted, self.seq)
            except Exception as exc:                    # noqa: BLE001
                log.debug("body sense poll failed: %s", exc)

    @staticmethod
    def _stop_moving() -> None:
        """Picked up mid-move: the move ends, as "Friday, stop" would end it."""
        try:
            from mcp_server.tools import movement
            movement.stop()
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not stop the move she was lifted out of: %s", exc)


_SENSE: Sense | None = None


def start() -> Sense | None:
    """friday-mcp's one call. Off unless config.BODY_SENSE."""
    global _SENSE
    if not getattr(config, "BODY_SENSE", False):
        return None
    if _SENSE is None:
        _SENSE = Sense()
        _SENSE.start()
    return _SENSE


# --- friday-brain: the reaction --------------------------------------------------

class BodyWatch:
    """A wake source (duck-typed like cpu_heat.HeatWatch). It yields the bare
    EVENT; wake_word.AnyOf prefixes the source's name, so main sees
    "body: <event>". (Yielding "body: lifted" itself made "body: body:
    lifted", which nothing recognised -- 2026-10-07 20:36, a real lift and no
    reaction.)

    New events only (by seq), fresh ones only (BODY_STALE_S), at most one
    reaction per BODY_REACT_GAP_S. It reacts asleep too -- picking her up
    wakes her, as it does EMO -- and main decides about quiet mode.
    """

    name = NAME
    stream = None

    def __init__(self, path: Path | None = None, clock=time.time, poll_s: float = 0.1):
        self.path = path
        self.clock = clock
        self.poll_s = poll_s
        self._closed = threading.Event()
        self._paused = False
        self._seen = (read(self.path) or {}).get("seq", 0)   # not what came before us
        self._last_react = 0.0

    def step(self) -> str | None:
        data = read(self.path)
        if not data or data.get("seq", 0) <= self._seen:
            return None
        self._seen = data["seq"]
        event = data.get("event")
        now = self.clock()
        if (event not in config.BODY_REACTIONS
                or now - float(data.get("t", 0.0)) > config.BODY_STALE_S):
            return None
        if self._paused or now - self._last_react < config.BODY_REACT_GAP_S:
            # A put-down is never swallowed by the lift just before it.
            if event != "put_down" or self._paused:
                return None
        self._last_react = now
        return event

    def listen(self):
        log.info("body watch ready: reacts to %s", ", ".join(config.BODY_REACTIONS))
        while not self._closed.wait(self.poll_s):
            try:
                trig = self.step()
            except Exception as exc:                    # noqa: BLE001
                log.debug("body watch poll failed: %s", exc)
                trig = None
            if trig:
                yield trig

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self, seconds: float) -> float:
        return 0.0                  # picking her up wakes her

    def wake_up(self) -> None:
        pass

    def sleeping_for(self) -> float:
        return 0.0

    def close(self) -> None:
        self._closed.set()


def parse_trigger(trigger: str) -> str | None:
    """"body: lifted" -> "lifted" (None if it is not one of ours)."""
    if not trigger.startswith(NAME + ": "):
        return None
    event = trigger.split(": ", 1)[1].strip()
    return event if event in config.BODY_REACTIONS else None


def reaction(event: str, rng=random) -> tuple[str, str]:
    """(face, line) for an event."""
    face, lines = config.BODY_REACTIONS[event]
    return face, rng.choice(lines)
