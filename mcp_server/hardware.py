"""hardware.py — the single owner of every device Friday has.

One process, one connection per device. Tools ask this module for hardware;
they never open a port themselves. That matters for the serial devices in
particular: two objects with the motor port open would fight over replies, and
the ToF and LiDAR pollers must be started once and shared, not restarted per
call.

Everything here is LAZY and FORGIVING. A device that is not plugged in is a
normal state — the accessor raises a readable error the tool turns into a
sentence, and the rest of the robot keeps working. A missing camera must never
cost you motor control.
"""

from __future__ import annotations

import logging
import threading

import config
import sensors
import tts
from motor import MotorError, Motors

log = logging.getLogger("friday.mcp.hardware")

_lock = threading.RLock()
_motors: Motors | None = None
_ranges: sensors.Rangefinders | None = None

# Set by stop(). Any movement loop watches this and bails out immediately.
# An Event rather than a flag so a waiting move wakes the moment it is set,
# and module-level so stop() never has to find the object that is moving.
ABORT = threading.Event()

# Speed is a percentage rather than raw PWM because the brain should not know
# or care about PWM, and because it lets config.MAX_SPEED_PERCENT cap it in one
# place.
#
# ⚠ IT LIVES IN THE CALIBRATION FILE, NOT IN A MODULE GLOBAL. It used to be a
# global, which meant every restart of friday-mcp silently reset whatever Harsh
# had set from the dashboard back to the default -- the setting appeared to
# work, then quietly undid itself. Read per call for the same reason
# load_calibration() is: a dashboard edit should take effect without a restart.


# --- motors --------------------------------------------------------------

def motors() -> Motors:
    """The connected drive, opening the link on first use.

    Raises MotorError with a readable reason if the board is absent or is not
    running Friday's firmware. That refusal is deliberate: an unidentified
    board is not driven, ever.
    """
    global _motors
    with _lock:
        if _motors is not None and _motors.serial is not None:
            return _motors
        drive = Motors()
        drive.connect()                     # raises MotorError if not ours
        _motors = drive
        log.info("motors connected: %s on %s", drive.firmware, drive.port)
        return _motors


def motors_if_present() -> Motors | None:
    """The drive, or None. For status checks that must not raise."""
    try:
        return motors()
    except Exception:                       # noqa: BLE001 - status must not raise
        return None


def release_motors() -> None:
    """Close the motor link, stopping the motors on the way out."""
    global _motors
    with _lock:
        if _motors is not None:
            try:
                _motors.close()
            except Exception:               # noqa: BLE001 - shutting down
                pass
            _motors = None


# --- speed ---------------------------------------------------------------

def speed_percent() -> int:
    """The current drive speed percentage, from the calibration file."""
    stored = config.load_calibration().get(
        "speed_percent", config.DEFAULT_SPEED_PERCENT)
    return max(1, min(int(stored), config.MAX_SPEED_PERCENT))


def set_speed_percent(percent: int) -> int:
    """Clamp and persist the drive speed. Returns what was actually set.

    Persisted, not held in memory: this survives a restart of friday-mcp,
    which is the whole point -- the dashboard's setting used to vanish every
    time the service came back.
    """
    value = max(1, min(int(percent), config.MAX_SPEED_PERCENT))
    with _lock:
        config.save_calibration({"speed_percent": value})
    return value


def speed_pwm(kind: str = "move") -> float:
    """The current speed as the 0-255 number the firmware wants.

    Scaled onto the calibrated base speed, so the dashboard's calibration
    still sets the character of the movement and the percentage just trims it.
    motor.Motors caps the result at config.MOTOR_MAX_DUTY regardless, so this
    cannot be used to exceed the hard limit.
    """
    calibration = config.load_calibration()
    base = calibration["turn_speed" if kind == "turn" else "move_speed"]
    return base * (speed_percent() / 100.0)


# --- range sensors -------------------------------------------------------

def ranges() -> sensors.Rangefinders:
    """The LiDAR + ToF pair, pollers already running.

    Started once, on first use, and left running. The safety interrupt needs
    a fresh reading every 100 ms while moving; connecting on demand would be
    far too slow for that.
    """
    global _ranges
    with _lock:
        if _ranges is None:
            _ranges = sensors.Rangefinders().start()
            log.info("range sensors started")
        return _ranges


def release_ranges() -> None:
    global _ranges
    with _lock:
        if _ranges is not None:
            _ranges.stop()
            _ranges = None


# --- lifecycle -----------------------------------------------------------

def startup() -> dict[str, str]:
    """Bring up what can be brought up, and report the rest honestly.

    Deliberately does NOT fail on missing hardware. The server must start with
    a half-connected robot and say which half, because that is the normal
    state of this machine.
    """
    report: dict[str, str] = {}

    # wait_ready matters: the pollers open their ports on their own threads,
    # so status() read immediately after start() reports a present sensor as
    # absent. Waiting a moment makes the start-up line honest.
    report.update(ranges().wait_ready().status())

    # Start loading the Piper voice NOW, in the background. It takes ~10 s,
    # and paying that in front of the first thing Harsh says is the difference
    # between a robot that answers and one that looks broken.
    tts.preload()
    report["voice"] = tts.engine_name()

    try:
        drive = motors()
        report["motors"] = f"connected: {drive.firmware} on {drive.port}"
    except Exception as exc:                        # noqa: BLE001
        report["motors"] = f"NOT available — {exc}"
    return report


def shutdown() -> None:
    """Stop everything. Motors first — that is the part that can hurt."""
    ABORT.set()
    try:
        drive = _motors
        if drive is not None:
            drive.stop()
    except Exception:                               # noqa: BLE001
        pass
    release_motors()
    release_ranges()
