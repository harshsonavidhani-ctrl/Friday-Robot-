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

import config
from mcp_server import hardware, safety
from mcp_server.registry import tool

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

    Call scan_environment first if you have not looked recently. Forward moves
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
    return safety.guarded_move(direction, duration_seconds).as_text()


@tool
def stop() -> str:
    """Stop the motors immediately. Always available, never queued.

    Takes effect at once even if a move is already running — it interrupts it
    rather than waiting for it to finish. Use it the moment anything seems
    wrong; there is no cost to calling it unnecessarily.
    """
    return safety.emergency_stop()


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
    direction = "backward" if cm < 0 else "forward"
    seconds = abs(cm) / rate
    result = safety.guarded_move(direction, seconds)
    if result.stopped_early:
        travelled = result.actual_s * rate
        return (f"{result.as_text()}. Got {_margin(round(travelled, 1), 'cm')} "
                f"of the {abs(cm):g} cm asked for.")
    return f"{result.as_text()}. Travelled {_margin(abs(cm), 'cm')}."


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
    direction = "left" if degrees < 0 else "right"
    seconds = abs(degrees) / rate
    result = safety.guarded_move(direction, seconds)
    if result.stopped_early:
        turned = result.actual_s * rate
        return (f"{result.as_text()}. Turned {_margin(round(turned), 'degrees')} "
                f"of the {abs(degrees):g} asked for.")
    return f"{result.as_text()}. Turned {_margin(abs(degrees), 'degrees')}."


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
