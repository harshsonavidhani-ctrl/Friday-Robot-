"""safety.py — the limits that live in code, not in a prompt.

THIS IS THE POINT OF THE WHOLE MCP SERVER. A system prompt that says "stop
before you hit things" is a suggestion: the model can misread it, be talked out
of it, or simply be wrong. This file is not a suggestion. Every movement the
brain can command goes through guarded_move(), and guarded_move() stops the
motors whether or not the brain wanted it to.

WHAT IS ENFORCED, all from config so none of it is scattered:

  * Duration is clamped to config.MAX_MOVE_SECONDS (reverse:
    MAX_REVERSE_SECONDS, shorter because nothing on this robot looks backwards).
  * Speed is clamped to config.MAX_SPEED_PERCENT, and motor.Motors clamps
    again to config.MOTOR_MAX_DUTY underneath. Two independent ceilings.
  * Clearance is checked BEFORE the move starts, and again every
    config.SAFETY_POLL_S while it runs. Below config.SAFETY_STOP_MM the motors
    stop that instant and the reason is returned.
  * Below config.SAFETY_CAUTION_MM a forward move is allowed but forced slow.
  * Every move ends in an explicit stop, including when it raises, including on
    Ctrl-C, including when the clearance check itself throws.

WHAT IS DELIBERATELY NOT ENFORCED: turning and reversing are not blocked by the
forward clearance. If Friday has stopped with a wall in front, refusing to turn
would leave it stuck there forever with no way out. Forward is the direction
that gets blocked, because forward is the direction that hits things.

HONESTY. A move that was cut short returns a result that says so, in words. The
brain must be able to tell "I moved 30 cm" from "I meant to move 30 cm and
stopped after 4 because of a chair", because it says one of those out loud.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import config
from mcp_server import embodiment, hardware

log = logging.getLogger("friday.mcp.safety")

# Directions that drive the robot into space it cannot see. Forward is checked
# against the range sensors; the others are time-limited instead.
FORWARD_WORDS = ("forward", "ahead", "front")
REVERSE_WORDS = ("back", "backward", "backwards", "reverse")


class SafetyStop(RuntimeError):
    """Raised when the interrupt fires. Carries a sentence, not a code."""


@dataclass
class MoveResult:
    """What actually happened, as opposed to what was asked for."""
    direction: str
    requested_s: float
    actual_s: float
    stopped_early: bool
    reason: str
    clearance_mm: int | None

    def as_text(self) -> str:
        head = (f"{self.direction} for {self.actual_s:.1f}s"
                if not self.stopped_early else
                f"{self.direction} STOPPED after {self.actual_s:.1f}s of "
                f"{self.requested_s:.1f}s")
        if self.reason:
            head += f" — {self.reason}"
        if self.clearance_mm is not None:
            head += f" (clearance {self.clearance_mm} mm)"
        return head


def clamp_duration(direction: str, seconds: float) -> tuple[float, str | None]:
    """Clamp a requested duration to the configured ceiling."""
    if seconds <= 0:
        return 0.0, "duration was zero"
    ceiling = (config.MAX_REVERSE_SECONDS
               if direction in REVERSE_WORDS else config.MAX_MOVE_SECONDS)
    if seconds > ceiling:
        return ceiling, (f"capped at the {ceiling}s limit for "
                         f"{'reverse' if direction in REVERSE_WORDS else 'a single move'}")
    return seconds, None


def check_forward_clearance() -> tuple[int | None, str, str | None]:
    """(clearance_mm, which sensor, blocking reason or None).

    A clearance of None — no sensor could see — is NOT treated as clear. It is
    treated as unknown, and unknown does not get to drive forward. Defaulting
    a blind robot to "go" is the bug this whole file exists to prevent.
    """
    try:
        distance, source = hardware.ranges().forward_clearance_mm()
    except Exception as exc:                # noqa: BLE001
        return None, "sensor error", f"cannot read the range sensors ({exc})"

    if distance is None:
        return None, source, ("no range sensor can see ahead — refusing to "
                              "drive forward blind")
    if distance < config.SAFETY_STOP_MM:
        return distance, source, (f"something is {distance} mm ahead, inside "
                                  f"the {config.SAFETY_STOP_MM} mm stop "
                                  f"distance ({source})")
    return distance, source, None


def guarded_move(direction: str, seconds: float,
                 speed_pwm: float | None = None) -> MoveResult:
    """Run one movement under the interrupt. The only way tools may move.

    Blocking; the registry runs it in a worker thread so that stop() can be
    answered — and hardware.ABORT set — while this is still running.
    """
    direction = direction.strip().lower()
    is_forward = direction in FORWARD_WORDS

    # LOOK BEFORE YOU MOVE, enforced here rather than asked for in the prompt.
    # This sits above the clearance check on purpose: clearance answers "is
    # there something there right now", the gate answers "has anyone actually
    # looked" -- and a robot driving on a minute-old picture is unsafe even
    # when that picture said clear. See mcp_server/embodiment.py.
    blind = embodiment.check(direction)
    if blind:
        log.warning("refused %s: no fresh observation", direction)
        return MoveResult(direction, seconds, 0.0, True, blind, None)

    requested, cap_note = clamp_duration(direction, seconds)
    notes = [cap_note] if cap_note else []

    if requested <= 0:
        return MoveResult(direction, seconds, 0.0, False,
                          "; ".join(notes) or "nothing to do", None)

    clearance: int | None = None
    if is_forward:
        clearance, source, blocked = check_forward_clearance()
        if blocked:
            log.warning("refused forward: %s", blocked)
            _show_alarm()
            return MoveResult(direction, seconds, 0.0, True, blocked, clearance)
        if clearance is not None and clearance < config.SAFETY_CAUTION_MM:
            notes.append(f"close quarters ({clearance} mm), moving slowly")

    drive = hardware.motors()               # raises if the board is absent

    # Slow down inside the caution band. Applied here rather than trusting the
    # caller, so it holds for every path into this function.
    pwm = hardware.speed_pwm("turn" if direction in ("left", "right") else "move")
    if speed_pwm is not None:
        pwm = speed_pwm
    if (is_forward and clearance is not None
            and clearance < config.SAFETY_CAUTION_MM):
        pwm *= config.CAUTION_SPEED_FACTOR

    sign = -1 if direction in REVERSE_WORDS else 1
    if direction in ("left", "right"):
        left, right = ((pwm, -pwm) if direction == "right" else (-pwm, pwm))
    elif is_forward or direction in REVERSE_WORDS:
        left = right = pwm * sign
    else:
        raise ValueError(f"unknown direction {direction!r} — use forward, "
                         f"backward, left, right or stop")

    hardware.ABORT.clear()
    hardware.MOVING.set()               # the reflex loop watches this
    started = time.monotonic()
    deadline = started + requested
    stopped_early = False
    reason = ""

    try:
        drive.drive(left, right)
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            # stop() sets this. Checked first so it always wins a race with
            # the clearance check.
            if hardware.ABORT.is_set():
                stopped_early = True
                reason = "stop was called"
                break

            if is_forward:
                clearance, source, blocked = check_forward_clearance()
                if blocked:
                    stopped_early = True
                    reason = blocked
                    _show_alarm()
                    break

            # The Arduino stops on its own after config.MOTOR_WATCHDOG_S of
            # silence, so the loop has to keep talking to it. A bare newline
            # pets the watchdog without changing the command.
            try:
                drive.keepalive()
            except Exception as exc:        # noqa: BLE001
                stopped_early = True
                reason = f"lost the motor link ({exc})"
                break

            hardware.ABORT.wait(min(config.SAFETY_POLL_S,
                                    max(0.0, deadline - time.monotonic())))
    finally:
        hardware.MOVING.clear()
        # Unconditional. Every early return, every exception, every Ctrl-C.
        try:
            drive.stop()
        except Exception as exc:            # noqa: BLE001
            log.error("STOP FAILED after %s: %s", direction, exc)

    actual = time.monotonic() - started
    # The wheels turned, so whatever Friday last saw now describes a place it
    # is no longer in. Ages the observation; the next move will need a new one.
    if actual > 0:
        embodiment.note_moved()
    if notes:
        reason = "; ".join(notes + ([reason] if reason else []))
    return MoveResult(direction, seconds, actual, stopped_early, reason,
                      clearance)


def _show_alarm(kind: str = "scared") -> None:
    """Put the alarm on Friday's face when a move is refused or cut short.

    Routed through the face director at SAFETY priority, so it lands even
    mid-sentence and cannot be painted over by whatever the brain was feeling.
    Harsh should be able to see that Friday stopped for something without
    waiting for it to finish a sentence about the weather.

    Never allowed to affect the move: the face is decoration and the stop is
    the point, so any failure here is swallowed.
    """
    try:
        import face_director
        face_director.get().safety(kind)
    except Exception as exc:                # noqa: BLE001
        log.debug("could not show the alarm face: %s", exc)


def emergency_stop() -> str:
    """Stop now. Called by the stop tool, and on shutdown.

    Sets ABORT before touching the serial link, so an in-flight move bails out
    within config.SAFETY_POLL_S even if the link itself is slow or wedged.
    """
    hardware.ABORT.set()
    drive = hardware.motors_if_present()
    if drive is None:
        return ("stop: no motor board connected, so nothing was moving "
                "(the abort flag is set regardless)")
    try:
        drive.stop()
        return "stopped"
    except Exception as exc:                # noqa: BLE001
        return (f"stop command FAILED: {exc}. The Arduino watchdog will cut "
                f"the motors within {config.MOTOR_WATCHDOG_S}s of silence.")
