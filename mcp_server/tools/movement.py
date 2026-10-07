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
import time

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
    measured = _gyro_turn(degrees)
    if measured is not None:
        return measured
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
    # No gyro: still never below the calibrated turn_speed -- a pivot at
    # speed 25% (42 duty) only buzzes after its kick (2026-10-07).
    floor = float(config.load_calibration().get("turn_speed", 0) or 0)
    pwm = max(hardware.speed_pwm("turn"), floor) if floor > 0 else None
    result = safety.guarded_move(direction, seconds, speed_pwm=pwm)
    if result.stopped_early:
        turned = result.actual_s * rate
        text = (f"{result.as_text()}. Turned {_margin(round(turned), 'degrees')} "
                f"of the {abs(degrees):g} asked for.")
    else:
        text = f"{result.as_text()}. Turned {_margin(abs(degrees), 'degrees')}."
    return f"{text} ({clip_note})" if clip_note else text


# --- a MEASURED turn, when the motor board has its gyro (2026-10-07) ---------------
GYRO_TURN_LEAD_S = 0.08         # stop this early at the current rate: the coast after "S"
GYRO_TURN_MAX_LEAD_DEG = 8.0
GYRO_TURN_POLL_S = 0.03
GYRO_TURN_SETTLE_S = 0.3
GYRO_TURN_RATE_GUESS = 75.0     # deg/s, only to size the time budget (x2 + 1 s)
# THE POWER, careful of the gears (Harsh, 2026-10-07: "just be careful of the
# motors gears noise and gear tear"). Measured that day: at speed 25% a turn
# ran at 170 x 0.25 = 42 duty -- the kick moved her ~10 deg, then she sat
# BUZZING (energised, going nowhere: the worst thing for the gears) for 3 s.
#  * start no lower than the calibrated turn_speed (170): below it a pivot only buzzes
#  * raise only while the gyro says she is NOT turning: +STEP every STEP_S
#  * never above MOTOR_MAX_DUTY (motor.Motors clamps it anyway)
#  * once turning, hold; ease back a step if faster than needed
#  * at the cap and still not turning for GIVE_UP_S: stop, say stuck -- no grinding
GYRO_TURN_STALL_DPS = 10.0      # slower than this = not turning
GYRO_TURN_MOVING_DPS = 25.0     # faster than this = turning: hold the power
GYRO_TURN_FAST_DPS = 120.0      # faster than this: ease back (noise, overshoot)
GYRO_TURN_STALL_S = 0.4         # judge only after the kick has settled
GYRO_TURN_STEP = 10.0           # duty per step
GYRO_TURN_STEP_S = 0.3
GYRO_TURN_GIVE_UP_S = 0.8       # at the cap, not turning, this long: stuck
GYRO_TURN_READY_WAIT_S = 4.0     # the gyro still measuring its bias: wait this long


def _gyro_turn(degrees: float) -> str | None:
    """turn_degrees, closed on the gyro: turn until the board says the angle
    is done, then report the angle it MEASURED. None = no gyro (or not ready),
    and the old timed pivot runs instead. Still one guarded_move: every check,
    the duration cap and the final stop are as for any turn."""
    try:
        drive = hardware.motors()
    except Exception:                                   # noqa: BLE001
        return None
    reader = getattr(drive, "imu", None)
    start = reader() if reader is not None else None
    # Still measuring its bias (every friday-mcp start resets the board, and
    # that takes ~1 s of stillness): wait for it rather than fall back to a
    # timed pivot that cannot tell whether it turned.
    waited = time.monotonic()
    while (start and start.get("status") == "cal"
           and time.monotonic() - waited < GYRO_TURN_READY_WAIT_S):
        time.sleep(0.2)
        start = reader()
    if not start or not start.get("ok"):
        return None
    degrees, clip_note = embodiment.clip_increment(
        degrees, config.MAX_INCREMENT_DEGREES, "degrees")
    direction = "left" if degrees < 0 else "right"    # the tool's sign: - = left
    target = abs(float(degrees))
    if target < 1.0:
        return "nothing to turn (under 1 degree)"
    blind = embodiment.check(direction)
    if blind:
        return blind
    yaw0 = start["yaw_deg"]                             # + = left (motor.parse_imu)
    sense = 1.0 if direction == "left" else -1.0
    last = {"turned": 0.0}
    trace: list = []                                    # what the gyro said, for the log
    cal = config.load_calibration()
    cap = 255.0 * float(getattr(drive, "max_duty", config.MOTOR_MAX_DUTY))
    pwm0 = min(max(hardware.speed_pwm("turn"), float(cal.get("turn_speed", 0) or 0)), cap)
    power = {"pwm": pwm0, "t0": time.monotonic(), "step_at": 0.0, "capped_at": None,
             "rate": 0.0, "stuck": False, "peak": pwm0}

    def done() -> str | None:
        now = reader()
        trace.append(None if not now else (round(now["yaw_deg"] - yaw0, 1),
                                           round(now.get("rate_dps", 0.0)),
                                           now.get("status"), now.get("errors"),
                                           int(power["pwm"])))
        if not now or not now.get("ok"):
            return None                                 # keep to the time budget
        power["rate"] = abs(now.get("rate_dps", 0.0))
        turned = sense * (now["yaw_deg"] - yaw0)
        last["turned"] = turned
        lead = min(power["rate"] * GYRO_TURN_LEAD_S, GYRO_TURN_MAX_LEAD_DEG)
        if turned >= target - lead:
            return f"the gyro measured {turned:.0f} of {target:.0f} degrees"
        if power["stuck"]:
            return (f"stuck: not turning even at the power limit, so stopping "
                    f"rather than grinding the gears ({turned:.0f} of {target:.0f} degrees)")
        return None

    def throttle() -> float | None:
        now = time.monotonic()
        if now - power["t0"] < GYRO_TURN_STALL_S or power["stuck"]:
            return None                                 # the kick is still settling
        rate_now = power["rate"]
        if rate_now >= GYRO_TURN_FAST_DPS and power["pwm"] - GYRO_TURN_STEP >= pwm0:
            power["pwm"] -= GYRO_TURN_STEP              # faster than needed: ease back
            power["step_at"] = now
            return power["pwm"]
        if rate_now >= GYRO_TURN_MOVING_DPS:
            power["capped_at"] = None
            return None                                 # turning: hold
        if rate_now < GYRO_TURN_STALL_DPS:
            if power["pwm"] >= cap - 0.5:
                if power["capped_at"] is None:
                    power["capped_at"] = now
                elif now - power["capped_at"] >= GYRO_TURN_GIVE_UP_S:
                    power["stuck"] = True
                return None
            if now - power["step_at"] >= GYRO_TURN_STEP_S:
                power["pwm"] = min(power["pwm"] + GYRO_TURN_STEP, cap)
                power["peak"] = max(power["peak"], power["pwm"])
                power["step_at"] = now
                return power["pwm"]
        return None

    rate = cal["turn_deg_per_s"]
    budget = target / (rate if rate > 0 else GYRO_TURN_RATE_GUESS) * 2.0 + 1.0
    result = safety.guarded_move(direction, budget, speed_pwm=pwm0, done=done,
                                 poll_s=GYRO_TURN_POLL_S, throttle=throttle)
    time.sleep(GYRO_TURN_SETTLE_S)                      # she coasts a little after S
    end = reader()
    turned = sense * (end["yaw_deg"] - yaw0) if end and end.get("ok") else last["turned"]
    shown = int(round(turned))                          # never "-0"
    log.info("gyro turn %s %.0f: measured %.1f in %.1fs; power %.0f -> peak %.0f "
             "(cap %.0f)%s; start %s; end %s; %d readings, %d missing, max |rate| %s; "
             "trace %s", direction, target, turned, result.actual_s, pwm0,
             power["peak"], cap, ", STUCK" if power["stuck"] else "", start, end,
             len(trace),
             sum(1 for x in trace if x is None),
             max((abs(x[1]) for x in trace if x), default=None), trace[:40])
    if power["stuck"]:
        text = (f"I couldn't turn: even at my power limit I only turned {shown} of "
                f"{target:.0f} degrees (measured by the gyro), so I stopped rather "
                f"than grind the gears. Something may be blocking my wheels, or my "
                f"battery is low.")
    elif result.stopped_early:
        text = (f"{result.as_text()}. Turned {shown} of the {target:.0f} "
                f"degrees asked for (measured by the gyro).")
    elif turned < target * 0.8:
        text = (f"{result.as_text()}. Only turned {shown} of {target:.0f} "
                f"degrees (measured by the gyro) -- the wheels may be slipping "
                f"or something is in the way.")
    else:
        text = f"Turned {direction} {shown} degrees (measured by the gyro)."
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
