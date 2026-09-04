"""Movement tools. Everything here goes through safety.guarded_move().

No tool in this file talks to the motors directly, and none of them can
disable the safety interrupt — there is deliberately no argument for it.

ON DISTANCE AND ANGLE, read this before believing a number. Friday's TT gear
motors have NO ENCODERS. Nothing counts wheel rotations, so nothing measures
how far the robot actually went. move_distance and turn_degrees convert a
distance into a DURATION using rates measured by hand and stored in
state/calibration.json, then drive for that long. Every result therefore
carries its error margin, and the brain is expected to say "about" out loud.
"""

from __future__ import annotations

import logging

import config
from mcp_server import embodiment, hardware, safety
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.movement")

DIRECTIONS = ("forward", "backward", "left", "right", "stop")


def _margin(centre: float, unit: str) -> str:
    """'about 30 cm (24-36 cm)' — the honest form of a timed estimate."""
    slack = centre * config.ODOMETRY_ERROR_PERCENT / 100.0
    return (f"about {centre:g} {unit} "
            f"(realistically {centre - slack:g}-{centre + slack:g} {unit}, "
            f"±{config.ODOMETRY_ERROR_PERCENT}% — timed, not measured)")


@tool
def move(direction: str, duration_seconds: float = 1.0) -> str:
    """Drive Friday in a direction for a number of seconds.

    YOU MUST CALL observe() FIRST. Forward and backward moves are refused
    outright unless Friday has looked at the world within the last few
    seconds, and moving ages that look — so a second step needs a second
    observe(). This is enforced by the server, not by convention. Forward moves
    are checked against the range sensors before starting and continuously
    while running: if something comes within the stop distance, the motors cut
    out immediately and this reports how long it actually ran. Believe that
    report — say the move was cut short if it was.

    Args:
        direction: forward, backward, left, right, or stop. Left and right
            pivot in place rather than steering through an arc.
        duration_seconds: how long to drive. Capped by the server.
    """
    direction = (direction or "").strip().lower()
    if direction not in DIRECTIONS:
        return (f"'{direction}' is not a direction. Use one of: "
                f"{', '.join(DIRECTIONS)}.")
    if direction == "stop":
        return safety.emergency_stop()
    blind = embodiment.check(direction)          # see move_distance
    if blind:
        return blind
    return safety.guarded_move(direction, duration_seconds).as_text()


@tool
def stop() -> str:
    """Stop the motors immediately. Always available, never queued.

    Takes effect at once even if a move is already running — it interrupts it
    rather than waiting for it to finish. Use it the moment anything seems
    wrong; there is no cost to calling it unnecessarily.

    STOPS AN AUTONOMOUS RUN TOO. That run is a SEPARATE PROCESS holding the
    motor port under a hardware lease (see mcp_server.hardware), so
    emergency_stop() here reaches motors this server does not currently own and
    would do nothing at all. "Stop" has to mean stop whatever is driving her,
    and there is no version of this where the word has to be qualified.
    """
    said = safety.emergency_stop()
    try:
        from mcp_server.tools import autonomy
        if autonomy._running():
            # The unit's own teardown halts the motors and puts her services
            # back; this returns the lease so the wheels are ours again.
            autonomy.stop_exploration()
            return said + " I have also stopped exploring."
    except Exception as exc:                        # noqa: BLE001
        # NEVER let this path raise. A failure to stop the run must not turn
        # into a failure to stop the motors, which is the one thing this tool
        # exists to guarantee.
        log.error("could not stop the autonomous run: %s: %s",
                  type(exc).__name__, exc)
        return said + " I could NOT stop the exploration run — use the kill switch."
    return said


@tool
def move_distance(cm: float) -> str:
    """Drive forward (or backward, if cm is negative) roughly this far.

    APPROXIMATE. The motors have no encoders, so this drives for a calculated
    length of time rather than measuring distance. Expect an error of around
    ±20%, more on carpet or a low battery. Say "about" when you report it.
    Refuses rather than guessing if the robot has not been calibrated yet.

    Args:
        cm: centimetres. Negative means backward.
    """
    rate = config.load_calibration()["forward_cm_per_s"]
    if rate <= 0:
        return ("cannot do distances yet — forward_cm_per_s is still 0, "
                "meaning nobody has measured how fast Friday drives. Use "
                "move(direction, duration_seconds) instead, or calibrate in "
                "the dashboard by driving a measured distance for a known "
                "time. Refusing to guess.")
    cm, clip_note = embodiment.clip_increment(
        cm, config.MAX_INCREMENT_CM, "cm")
    direction = "backward" if cm < 0 else "forward"
    # Asked here as well as inside guarded_move so that a refusal reads as a
    # refusal. Routed through the mover it comes back wrapped in "STOPPED after
    # 0.0s ... got about 0 cm", which buries the one sentence that says what to
    # do next. The gate in safety.py is still the backstop that cannot be
    # bypassed; this is only about the wording the model reads.
    blind = embodiment.check(direction)
    if blind:
        return blind
    seconds = abs(cm) / rate
    result = safety.guarded_move(direction, seconds)
    if result.stopped_early:
        travelled = result.actual_s * rate
        text = (f"{result.as_text()}. Got {_margin(round(travelled, 1), 'cm')} "
                f"of the {abs(cm):g} cm asked for.")
    else:
        text = f"{result.as_text()}. Travelled {_margin(abs(cm), 'cm')}."
    return f"{text} ({clip_note})" if clip_note else text


@tool
def turn_degrees(degrees: float) -> str:
    """Pivot Friday in place by roughly this many degrees.

    APPROXIMATE, for the same reason as move_distance: this is a timed pivot,
    not a measured one. Expect around ±20%. Refuses rather than guessing if
    the turn rate has not been measured.

    Args:
        degrees: negative turns left, positive turns right.
    """
    rate = config.load_calibration()["turn_deg_per_s"]
    if rate <= 0:
        return ("cannot do angles yet — turn_deg_per_s is still 0, meaning "
                "nobody has measured how fast Friday pivots. Use "
                "move('left'|'right', duration_seconds) instead, or time a "
                "full 360 and set it in the dashboard. Refusing to guess.")
    degrees, clip_note = embodiment.clip_increment(
        degrees, config.MAX_INCREMENT_DEGREES, "degrees")
    direction = "left" if degrees < 0 else "right"
    blind = embodiment.check(direction)          # see move_distance
    if blind:
        return blind
    seconds = abs(degrees) / rate
    result = safety.guarded_move(direction, seconds)
    if result.stopped_early:
        turned = result.actual_s * rate
        text = (f"{result.as_text()}. Turned {_margin(round(turned), 'degrees')} "
                f"of the {abs(degrees):g} asked for.")
    else:
        text = f"{result.as_text()}. Turned {_margin(abs(degrees), 'degrees')}."
    return f"{text} ({clip_note})" if clip_note else text


@tool
def set_speed(percent: int) -> str:
    """Set how fast Friday drives, as a percentage of its safe maximum.

    100% is not full motor power — it is the top of a deliberately
    conservative limit set in the robot's config. Lower it in tight spaces.

    Args:
        percent: 1 to 100.
    """
    try:
        wanted = int(percent)
    except (TypeError, ValueError):
        return f"speed must be a whole number of percent, got {percent!r}"
    actual = hardware.set_speed_percent(wanted)
    if actual != wanted:
        return (f"speed set to {actual}% — {wanted}% was outside the allowed "
                f"1-{config.MAX_SPEED_PERCENT}% range")
    return f"speed set to {actual}%"
