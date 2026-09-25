"""reflex.py — the fast loop. Friday's spinal cord, not her brain.

THE SPLIT. Friday thinks in two places at once, and they are not peers:

  SLOW LOOP   Sonnet 5, behind the Anthropic API. Reasoning, planning, memory,
              conversation. Hundreds of milliseconds at best, seconds when the
              network is poor, and NOTHING at all when the network is gone.
  FAST LOOP   this file, plus offline.py and wake_word.py. Runs on the Pi, in
              process, at REFLEX_HZ. No model, no network, no tokens. It knows
              almost nothing -- and it is the one that gets to stop the motors.

WHY THE STUPID ONE WINS. Everything that makes the slow loop good at planning
makes it unfit to catch a table edge: it is far away, it is not always
reachable, and it can be wrong in fluent and convincing ways. A reflex must
fire in tens of milliseconds and must not depend on a network. So the reflex
does not ask the brain, does not tell the brain first, and cannot be overruled
by it. There is no argument, no tool, and no config flag that turns this off,
because a safety system Friday can talk her way out of is not a safety system.

WHAT IT WATCHES, at config.REFLEX_HZ, whenever the wheels are driving her
FORWARD (or in a direction it was not told):

  obstacle   forward clearance inside config.SAFETY_STOP_MM
  cliff      the ToF's bottom rows stop seeing floor -- the floor ENDED. This
             is the one no other layer catches: the LiDAR is mounted on the
             top dome and sees a stair as clear air, and a cheerful drive
             forward is how a robot ends up on the floor below.
  blind      the range sensors stop answering entirely while moving

BELT AND BRACES, DELIBERATELY. guarded_move already re-checks clearance every
SAFETY_POLL_S from inside its own loop. This is a SECOND, INDEPENDENT watcher
in its own thread, and the duplication is the point: if the move loop wedges on
a slow serial write, blocks, or throws somewhere unexpected, the thing that
stops the motors must not be inside it. Two loops, one job, no shared failure.

⚠ NOT WHEN SHE IS BACKING AWAY OR TURNING. Every sensor here looks ahead, so
every hazard here is a hazard to driving forward -- the rule safety.py has
always stated ("turning and reversing are not blocked by the forward
clearance ... refusing to turn would leave it stuck there forever"). Until
2026-09-25 this loop fired on ANY motion: parked 15 cm from a box, every turn
and every reverse was killed after 0.2 s, twenty in a row, and she sat there
asking for help with clear floor behind her. Reverse stays time-capped
(MAX_REVERSE_SECONDS) and LiDAR-bump-guarded in guarded_move. The direction
comes from the motor command itself (hardware.moving_direction).

HONEST ABOUT WHAT IT CANNOT SEE. Backward has no sensor at all on this robot,
so reversing is time-limited rather than watched (see safety.py). The reflex
says so rather than implying it is guarding a direction it is blind to.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import config
from mcp_server import hardware

log = logging.getLogger("friday.mcp.reflex")


@dataclass
class Trip:
    """One time the reflex fired, kept so status() can report the last one."""
    at: float
    reason: str
    detail: str

    def age_s(self) -> float:
        return time.monotonic() - self.at


_thread: threading.Thread | None = None
_stop = threading.Event()
_lock = threading.Lock()
_last_trip: Trip | None = None
_trip_count = 0


def _cliff_gap_mm() -> int | None:
    """How far down the floor is just ahead, or None if it cannot be seen."""
    try:
        tof = hardware.ranges()
        reader = getattr(tof, "tof", None) or getattr(tof, "_tof", None)
        if reader is None or not hasattr(reader, "floor_gap_mm"):
            return None
        return reader.floor_gap_mm()
    except Exception:                           # noqa: BLE001
        return None


# Moves the forward-looking checks do not apply to. See the module docstring.
AWAY_FROM_AHEAD = ("backward", "left", "right")


def _hazard(direction: str = "") -> tuple[str, str] | None:
    """(reason, detail) if the wheels must stop now, else None.

    `direction` is the way the wheels were last told to go
    (hardware.moving_direction()). Empty -- not known, or the instant before a
    move's first command -- is treated as forward: unknown gets the checks.
    """
    if direction in AWAY_FROM_AHEAD:
        return None
    # 1. A drop ahead. Checked first: it is the only hazard that is invisible
    #    to every other layer, and the most expensive to get wrong.
    if config.CLIFF_DROP_MM:
        gap = _cliff_gap_mm()
        if gap is not None and gap > config.CLIFF_DROP_MM:
            return ("cliff", f"the floor falls away {gap} mm ahead — past the "
                             f"{config.CLIFF_DROP_MM} mm drop limit")

    # 2. Something in front, closer than the stop distance.
    try:
        distance, source = hardware.ranges().forward_clearance_mm()
    except Exception as exc:                    # noqa: BLE001
        return ("blind", f"the range sensors stopped answering ({exc})")

    if distance is None:
        return ("blind", "no range sensor can see ahead any more")
    if distance < config.SAFETY_STOP_MM:
        return ("obstacle", f"{distance} mm ahead ({source}), inside the "
                            f"{config.SAFETY_STOP_MM} mm stop distance")
    return None


def _fire(reason: str, detail: str) -> None:
    """Stop the motors. ABORT first, hardware second, everything else after."""
    global _last_trip, _trip_count

    # ABORT before touching serial, exactly as emergency_stop does: a move
    # already running bails out on the flag within SAFETY_POLL_S even if the
    # serial write below is slow or the link is wedged.
    hardware.ABORT.set()
    log.warning("REFLEX STOP — %s: %s", reason, detail)

    try:
        drive = hardware.motors_if_present()
        if drive is not None:
            drive.stop()
    except Exception as exc:                    # noqa: BLE001
        log.error("reflex could not stop the motors: %s — the Arduino "
                  "watchdog cuts them within %ss of silence",
                  exc, config.MOTOR_WATCHDOG_S)

    with _lock:
        _last_trip = Trip(time.monotonic(), reason, detail)
        _trip_count += 1

    # Decoration, and last, and never allowed to matter: the stop is the point.
    try:
        import face_director
        face_director.get().safety("scared")
    except Exception:                           # noqa: BLE001
        pass


def _run() -> None:
    period = 1.0 / max(1.0, config.REFLEX_HZ)
    log.info("reflex loop running at %g Hz", config.REFLEX_HZ)
    while not _stop.is_set():
        try:
            if hardware.MOVING.is_set():
                hazard = _hazard(hardware.moving_direction())
                if hazard:
                    _fire(*hazard)
        except Exception as exc:                # noqa: BLE001
            # A reflex that can die is not a reflex. Log it and keep looping.
            log.error("reflex iteration failed: %s", exc)
        _stop.wait(period)
    log.info("reflex loop stopped")


def start() -> None:
    """Start the reflex. Called by the MCP server at startup; idempotent."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="friday-reflex", daemon=True)
    _thread.start()


def stop() -> None:
    """Stop the reflex thread. Shutdown only -- never as a way to disable it."""
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=2.0)


def status() -> dict:
    """What the reflex is doing, for get_status and the tests."""
    with _lock:
        trip = _last_trip
        count = _trip_count
    running = _thread is not None and _thread.is_alive()
    out = {
        "running": running,
        "hz": config.REFLEX_HZ,
        "watching": ["obstacle", "cliff", "sensor loss"],
        "blind_to": "backward — no rear sensor on this robot",
        "trips_since_start": count,
    }
    if trip is not None:
        out["last_trip"] = {"reason": trip.reason, "detail": trip.detail,
                            "seconds_ago": round(trip.age_s(), 1)}
    return out
