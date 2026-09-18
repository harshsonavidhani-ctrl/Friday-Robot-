"""physical_curiosity.py — File 08 step 9: in the quiet, Friday goes and looks.

    physical_curiosity.start()      from main.py, behind the wake word
    physical_curiosity.heard()      a wake fired -- stop, he is talking
    physical_curiosity.turn_ended() the quiet starts again

⛔ OFF BY DEFAULT (config.PHYSICAL_CURIOSITY). Harsh, in the spec: "Make this
toggleable with a simple on/off flag, default off, so it can be tested in a
controlled, open area before leaving it on unsupervised." scripts/test_smoke.py
fails if that line is ever committed as True.

WIRE-UP ONLY, AND THE WIRING IS THE POINT. This module decides exactly one
thing: whether now is a good moment to wander. The driving is explore_room()
in the MCP server, called the way anyone else calls it -- so every move still
goes through embodiment.check (has she actually looked?), safety.guarded_move
(the clearance interrupt, the duration cap, the 100 ms re-check while moving),
and slam/mission.py's did-the-turn-actually-happen test. There is no
stop-distance logic in this file and there must never be any: a second opinion
about clearance is how a robot ends up with two of them and obeys the wrong
one. See mcp_server/tools/explore.py for what it is trusting.

⚠ SHE MOVES WITH NOBODY ASKING. That is the whole feature and it is the risk,
so the gates are deliberately mean -- every one of them, or the thread just
sleeps:
  * the flag is on
  * daylight (PHYSICAL_CURIOSITY_HOURS) -- not a dark room at 3 am
  * quiet for PHYSICAL_CURIOSITY_AFTER_S, where quiet means no turn has ended
  * the rate limiter says yes (30 min apart, 2 an hour, 12 a day)
  * nothing else is already driving (an RL exploration run, say)
  * the Pi is not under-voltage or throttled right now
and a short leash when it does go: PHYSICAL_CURIOSITY_BUDGET_M metres,
PHYSICAL_CURIOSITY_BUDGET_S seconds, whichever runs out first.

IT STOPS THE INSTANT HE SPEAKS. heard() is called from main.py the moment the
wake word fires, before the capture -- so the wheels are already stopping
while she is still listening to the question. A robot that keeps driving
through "Friday, stop" is the one thing this must never be.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import threading
import time

import config
import rate_limit

log = logging.getLogger("friday.curiosity")

NAME = "physical_curiosity"

_lock = threading.Lock()
_last_activity = time.monotonic()
_thread: threading.Thread | None = None
_loop: "asyncio.AbstractEventLoop | None" = None
_session = None                 # the live MCP session, only while out
_out = False                    # is an outing running right now?
_last_result = ""


def turn_ended() -> None:
    """A turn finished: the quiet starts again from here. Never raises."""
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


def heard() -> None:
    """The wake word fired. Stop driving, now. Never raises.

    Called from main.py's loop before the capture. Safe to call when she is
    not out -- it is a no-op then, and it is called on every single wake.
    """
    turn_ended()
    with _lock:
        loop, session, out = _loop, _session, _out
    if not (out and loop and session):
        return
    log.info("he is talking — stopping the outing")
    try:
        asyncio.run_coroutine_threadsafe(_stop(session), loop)
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not ask her to stop: %s", exc)


async def _stop(session) -> None:
    """stop() through the same session the outing is using.

    stop() is ungated in the server and interrupts a move already running;
    explore_room checks hardware.ABORT between every move, so this ends the
    run rather than just the current leg.
    """
    try:
        await session.call_tool("stop", {})
    except Exception as exc:                    # noqa: BLE001
        log.warning("stop() did not go through: %s", exc)


def quiet_for() -> float:
    with _lock:
        return time.monotonic() - _last_activity


def _daylight(now: "datetime.datetime | None" = None) -> bool:
    start, end = getattr(config, "PHYSICAL_CURIOSITY_HOURS", (9, 21))
    hour = (now or datetime.datetime.now()).hour
    return start <= hour < end


def _already_driving() -> bool:
    """Is an exploration run already going? Then this is not her turn.

    Reads the same state/explore_mode.json the dashboard and main.py read. A
    file that cannot be read is treated as "something might be running": the
    safe answer when the question is whether to start a second driver.
    """
    try:
        status = json.loads(
            (config.ROOT / "state" / "explore_mode.json").read_text())
        return str(status.get("state", "")).lower() == "running"
    except FileNotFoundError:
        return False
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the explore state (%s) — assuming busy", exc)
        return True


async def _power_is_fine(session) -> bool:
    """No driving while the supply cannot keep up.

    THERE IS NO BATTERY GAUGE (config.BATTERY_SENSOR is None), so there is no
    percentage to test and pretending otherwise would be a guess. What is real
    is the SoC's own under-voltage and throttling flags, which is what
    get_battery reports -- and a robot that browns out mid-turn is one that
    stops believing its own odometry.
    """
    try:
        result = await session.call_tool("get_battery", {})
        text = "".join(getattr(item, "text", "") for item in result.content)
        power = json.loads(text).get("power", {})
        bad = [k for k in ("under_voltage_now", "throttled_now")
               if power.get(k)]
        if bad:
            log.info("not going out: %s", ", ".join(bad))
            return False
        return True
    except Exception as exc:                    # noqa: BLE001
        # Unreadable power is not permission. Staying put costs nothing.
        log.info("not going out: could not read the power state (%s)", exc)
        return False


def _why_not_now() -> str | None:
    """The first gate that says no, or None when she may go."""
    if not getattr(config, "PHYSICAL_CURIOSITY", False):
        return "switched off"
    if not _daylight():
        start, end = config.PHYSICAL_CURIOSITY_HOURS
        return f"outside {start:02d}:00-{end:02d}:00"
    quiet = quiet_for()
    after = float(getattr(config, "PHYSICAL_CURIOSITY_AFTER_S", 600.0))
    if quiet < after:
        return f"only quiet for {quiet:.0f}s of {after:.0f}s"
    ok, why = rate_limit.allow(NAME)
    if not ok:
        return why
    if _already_driving():
        return "something is already driving"
    return None


async def _outing() -> str:
    """One short wander, through the MCP server. Never raises."""
    global _session, _out, _last_result
    import brain

    budget_m = float(getattr(config, "PHYSICAL_CURIOSITY_BUDGET_M", 1.5))
    budget_s = float(getattr(config, "PHYSICAL_CURIOSITY_BUDGET_S", 45.0))
    async with brain.mcp_session() as session:
        if not await _power_is_fine(session):
            return "power"
        # The clock and the counter start HERE, not when the gates passed:
        # a run that never opened a session has not used its allowance.
        rate_limit.used(NAME)
        with _lock:
            _session, _out = session, True
        log.info("going for a look — %.1f m, %.0f s", budget_m, budget_s)
        try:
            result = await session.call_tool(
                "explore_room", {"budget_m": budget_m,
                                 "budget_seconds": budget_s})
            text = "".join(getattr(item, "text", "") for item in result.content)
        finally:
            with _lock:
                _session, _out = None, False
        _last_result = text[:400]
        log.info("back: %s", _last_result)
        return text


def _run() -> None:
    global _loop, _session, _out
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with _lock:
        _loop = loop
    said_off = False
    tick = float(getattr(config, "PHYSICAL_CURIOSITY_TICK_S", 20.0))
    while True:
        time.sleep(tick)
        try:
            why = _why_not_now()
            if why == "switched off":
                if not said_off:
                    log.info("physical curiosity is off "
                             "(config.PHYSICAL_CURIOSITY)")
                    said_off = True
                continue
            said_off = False
            if why:
                log.debug("not going out: %s", why)
                continue
            loop.run_until_complete(_outing())
        except Exception as exc:                # noqa: BLE001
            # A failed outing is a quiet evening, never a crashed robot.
            log.warning("the outing failed (%s: %s)", type(exc).__name__, exc)
            with _lock:
                _session, _out = None, False
            time.sleep(tick)
        finally:
            turn_ended()        # the quiet restarts after she gets back


def start() -> None:
    """Start the curiosity thread. Safe to call twice. Never raises."""
    global _thread
    with _lock:
        if _thread is not None:
            return
        _thread = threading.Thread(target=_run, name="physical-curiosity",
                                   daemon=True)
    log.info("physical curiosity %s: after %.0fs quiet, %.1f m / %.0f s per "
             "outing, %s",
             "ON" if getattr(config, "PHYSICAL_CURIOSITY", False) else "off",
             float(getattr(config, "PHYSICAL_CURIOSITY_AFTER_S", 600.0)),
             float(getattr(config, "PHYSICAL_CURIOSITY_BUDGET_M", 1.5)),
             float(getattr(config, "PHYSICAL_CURIOSITY_BUDGET_S", 45.0)),
             rate_limit.limits(NAME) or "no rate limit")
    _thread.start()


def status() -> dict:
    """What it is doing, for the dashboard and dev_status. Never raises."""
    with _lock:
        out, last = _out, _last_result
    return {
        "enabled": bool(getattr(config, "PHYSICAL_CURIOSITY", False)),
        "out_now": out,
        "quiet_for_s": round(quiet_for(), 1),
        "why_not_now": _why_not_now(),
        "today": rate_limit.count(NAME),
        "last_result": last,
    }


def main() -> int:
    """Why she is or is not about to wander, right now."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for key, value in status().items():
        print(f"{key:14s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
