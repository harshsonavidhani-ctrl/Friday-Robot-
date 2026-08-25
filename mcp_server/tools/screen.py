"""Screen tools: what Friday puts on the 7" panel instead of her face.

The apps run inside the face process, so every one of these is a file write —
they cannot block, and they work even while the face is restarting. See
app_control.py for the channel and apps/ for the views.
"""

from __future__ import annotations

import time

import app_control
from mcp_server.registry import tool


@tool
def set_alarm(time: str, label: str = "") -> str:
    """Set an alarm that rings at a given time of day.

    Use for anything tied to a CLOCK TIME — "wake me at 7", "remind me at half
    past 8". For a length of time from now ("in ten minutes") use set_timer
    instead.

    It rings out loud through Friday's speaker and shows an alarm on the
    screen until it is dismissed, and it survives a restart or a power cut, so
    it is a real promise. Say the confirmed time back to Harsh — the time this
    returns is the one Friday actually stored, which is what he needs to hear
    if he was ambiguous.

    Args:
        time: how it was said — "8 AM", "8:30 pm", "19:45", "tomorrow at 7".
            A time that has already passed today is taken as tomorrow.
        label: what it is for, in a couple of words ("school", "the oven").
            Shown on screen when it rings.
    """
    alarm = app_control.add_alarm(time, label)
    tail = f" for {alarm['label']}" if alarm["label"] else ""
    return f"alarm set for {app_control.say_time(alarm['at'])}{tail}"


@tool
def set_timer(duration: str, label: str = "") -> str:
    """Start a countdown and show it on the screen.

    Use for a LENGTH of time from now — "ten minutes", "an hour and a half",
    "90 seconds". For a time of day use set_alarm.

    The countdown goes up on the panel straight away and rings when it
    reaches zero, whatever else is showing by then.

    Args:
        duration: how long, as it was said. A bare number means minutes.
        label: what it is for ("pasta", "tea"), shown under the countdown.
    """
    timer = app_control.add_timer(duration, label)
    app_control.launch("timer", timer_id=timer["id"])
    tail = f" for {timer['label']}" if timer["label"] else ""
    return f"timer set for {app_control.say_duration(timer['seconds'])}{tail}"


@tool
def show_clock() -> str:
    """Put the clock on the screen — "show me the clock", "what time is it".

    Answer out loud as well; the returned time is what is on the panel. The
    clock stays up until something else needs the screen.
    """
    return app_control.launch("clock")


@tool
def show_weather(place: str, condition: str, temp_c: float | None = None,
                 high_c: float | None = None, low_c: float | None = None,
                 wind_kph: float | None = None,
                 humidity: float | None = None) -> str:
    """Show a weather card on the panel.

    FRIDAY HAS NO WEATHER SERVICE OF HER OWN. Look the weather up first with
    web search, then call this with what you found — the card draws whatever
    you pass. Do not guess numbers; leave one out rather than invent it.

    Args:
        place: the town or city the numbers are for.
        condition: in plain words — "clear", "partly cloudy", "rain",
            "thunderstorm", "snow", "fog". The icon is picked from this.
        temp_c: current temperature in Celsius.
        high_c: today's high, Celsius.
        low_c: today's low, Celsius.
        wind_kph: wind speed in km/h.
        humidity: relative humidity, percent.
    """
    return app_control.set_weather({
        "place": place, "condition": condition, "temp_c": temp_c,
        "high_c": high_c, "low_c": low_c, "wind_kph": wind_kph,
        "humidity": humidity,
    })


@tool
def list_alarms() -> str:
    """Every alarm and timer still ahead, soonest first.

    Use before setting another one when Harsh asks something like "do I have
    an alarm?", and to find the id when he wants one cancelled.
    """
    items = app_control.pending()
    if not items:
        return "nothing is set"
    lines = []
    for item in items:
        label = f" — {item['label']}" if item.get("label") else ""
        when = (app_control.say_time(item["at"]) if item["kind"] == "alarm"
                else app_control.say_duration(max(0.0, item["at"] - time.time()))
                + " left")
        lines.append(f"{item['id']}: {item['kind']} {when}{label}")
    return "; ".join(lines)


@tool
def cancel_alarm(which: str) -> str:
    """Cancel one alarm or timer, or all of them.

    Args:
        which: the id from list_alarms ("alarm1", "timer2"), or "all".
    """
    return app_control.cancel(which)


@tool
def dismiss_screen() -> str:
    """Stop whatever is on the panel and go back to Friday's face.

    This is what "stop", "dismiss" and "that's enough" mean while an alarm is
    ringing or an app is up. Silences the ring too.
    """
    running = app_control.running()
    app_control.dismiss()
    return f"{running} dismissed" if running else "the face was already up"


@tool
def get_screen() -> str:
    """What is on the 7" panel right now — an app's name, or the face."""
    running = app_control.running()
    return f"{running} is on the screen" if running else "Friday's face"
