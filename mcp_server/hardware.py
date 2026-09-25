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
import time

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

# Set while the wheels are actually turning, cleared the moment they stop. The
# reflex loop watches this so it only intervenes when there is something to
# intervene in -- a hazard in front of a stationary robot is not an emergency,
# it is just a wall. guarded_move owns it, in a finally, so it cannot be left
# set by an exception.
MOVING = threading.Event()


# Speed is a percentage rather than raw PWM because the brain should not know
# or care about PWM, and because it lets config.MAX_SPEED_PERCENT cap it in one
# place.
#
# ⚠ IT LIVES IN THE CALIBRATION FILE, NOT IN A MODULE GLOBAL. It used to be a
# global, which meant every restart of friday-mcp silently reset whatever Harsh
# had set from the dashboard back to the default -- the setting appeared to
# work, then quietly undid itself. Read per call for the same reason
# load_calibration() is: a dashboard edit should take effect without a restart.


# --- the hardware lease --------------------------------------------------
#
# WHY THIS EXISTS. Harsh, 2026-09-04: "all the mcp tools dashboard face and
# everything will continuously work while exploring".
#
# The RL autonomous run (rl/friday_localnav_deploy.py) is a SEPARATE PROCESS
# that needs the LiDAR, ToF and motor serial ports exclusively -- two objects
# with the motor port open fight over replies, which is the reason this module
# exists at all. So rl/run_localnav.sh used to `systemctl stop friday-mcp` for
# the duration, and with the server gone every tool, the dashboard's controls
# and the brain's hands went with it.
#
# It does not have to. The server can put the DEVICES down without going away:
# release them, refuse to reopen them while the lease is out, and pick them up
# again afterwards. Everything that is not a serial port -- the tools, the
# camera views, the voice, the face, the decision log -- keeps working.
#
# A FILE, not an in-process flag, because the two sides are different processes
# and the run script is bash. It carries an expiry so a crashed run cannot
# leave the robot permanently unable to open its own motors: the longest it can
# cost is the lease it asked for.
LEASE_FILE = config.STATE_DIR / "hardware.lease" if hasattr(config, "STATE_DIR") \
    else config.ROOT / "state" / "hardware.lease"


class HardwareLeased(RuntimeError):
    """Raised instead of opening a device another process is currently using."""


def lease_active() -> float:
    """Seconds of lease remaining, or 0.0 when the hardware is ours."""
    try:
        raw = LEASE_FILE.read_text().strip()
    except (OSError, ValueError):
        return 0.0
    try:
        expires = float(raw.split()[0])
    except (IndexError, ValueError):
        return 0.0
    left = expires - time.time()
    if left <= 0.0:
        # Expired. Remove it so a stale file cannot outlive the run that wrote
        # it -- a lease that never ends is indistinguishable from broken
        # hardware, and would be diagnosed as broken hardware.
        try:
            LEASE_FILE.unlink()
        except OSError:
            pass
        return 0.0
    return left


def lease_out(seconds: float, holder: str = "autonomy") -> float:
    """Hand the serial devices to another process for `seconds`.

    Releases them here first, so the caller does not have to race us for the
    port. Idempotent: leasing while already leased just extends it.
    """
    LEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEASE_FILE.write_text(f"{time.time() + float(seconds):.0f} {holder}\n")
    release_motors()
    release_ranges()
    log.warning("hardware leased to %s for %.0fs -- motors and range sensors "
                "released; every other tool keeps working", holder, seconds)
    return float(seconds)


def lease_return() -> None:
    """Take the devices back. They reopen lazily on the next use."""
    try:
        LEASE_FILE.unlink()
        log.info("hardware lease returned")
    except OSError:
        pass


def _check_lease(what: str) -> None:
    left = lease_active()
    if left > 0.0:
        raise HardwareLeased(
            f"the {what} are in use by an autonomous run for another "
            f"{left:.0f}s. Everything else still works; say stop, or wait.")


# --- motors --------------------------------------------------------------

def motors() -> Motors:
    """The connected drive, opening the link on first use.

    Raises MotorError with a readable reason if the board is absent or is not
    running Friday's firmware. That refusal is deliberate: an unidentified
    board is not driven, ever.
    """
    global _motors
    with _lock:
        _check_lease("motors")
        if _motors is not None and _motors.serial is not None:
            return _motors
        drive = Motors()
        drive.connect()                     # raises MotorError if not ours
        _motors = _Commanded(drive)
        log.info("motors connected: %s on %s", drive.firmware, drive.port)
        return _motors


# --- which way the wheels were last told to go ------------------------------
#
# The reflex needs it. Every sensor it reads looks FORWARD, so a box 15 cm
# ahead is a reason to stop driving at it and no reason at all to stop backing
# away from it or turning to face elsewhere (2026-09-25 22:43: pinned 15 cm
# from a box, twenty turns and reverses in a row were killed after 0.2 s).
# Read off the command itself, because that is what the wheels are doing --
# and safety.py and motor.py stay byte-for-byte as they are. Cleared on every
# stop, so between moves, and in the instant before a move's first command,
# the direction is UNKNOWN, and unknown gets the full forward checks.

_last_drive: tuple[float, float] | None = None


def moving_direction() -> str:
    """"forward", "backward", "left", "right", or "" when not known."""
    command = _last_drive
    if command is None:
        return ""
    left, right = command
    if left > 0 and right > 0:
        return "forward"
    if left < 0 and right < 0:
        return "backward"
    if left > 0 > right:
        return "right"
    if right > 0 > left:
        return "left"
    return ""


class _Commanded:
    """The motor board, remembering the last command it was given.

    Everything passes straight through to motor.Motors; the only addition is
    noting (left, right) on drive() -- before the wiring inversion, so
    positive is that side forward -- and forgetting it on stop().
    """

    def __init__(self, inner: Motors) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def drive(self, left: float, right: float) -> None:
        global _last_drive
        _last_drive = (float(left), float(right))
        return self._inner.drive(left, right)

    def stop(self) -> None:
        global _last_drive
        _last_drive = None
        return self._inner.stop()


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
        _check_lease("range sensors")
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
