"""File 16's tools: reminders, the to-do list, routines, the calendar, the briefing.

Every one of these is also reachable by voice without the brain (offline.py
reflex intents call the same tools), and the brain can call them for any
phrasing the reflexes do not know. None of them moves the robot.

Answers are written to be SAID: no brackets, no JSON -- offline.speakable
strips parentheses and the brain reads these back as they are.
"""

from __future__ import annotations

import time

import config
import reminders
from mcp_server.registry import tool


@tool
def add_reminder(text: str, when: str = "") -> str:
    """Remind Harsh about something at a time, or add it to his to-do list.

    Use when he says "remind me to X at/in/on Y". With no `when`, it goes on
    the to-do list instead (no time, never announced on its own). She SAYS a
    reminder once, at its time, when nobody is talking.

    Args:
        text: what to remind him of, e.g. "call mom".
        when: a spoken time -- "5 pm", "tomorrow morning", "in 20 minutes",
            "on monday at 10". Empty for a to-do.
    """
    due = None
    if when.strip():
        try:                    # the voice path hands over an exact time
            from datetime import datetime
            due = datetime.fromisoformat(when.strip()).timestamp()
        except ValueError:
            due = reminders.spoken_time(when)
    if when.strip() and due is None:
        return (f"I couldn't tell what time '{when}' means. Try something "
                f"like 5 pm, tomorrow morning, or in 20 minutes.")
    try:
        item = reminders.add(text, due_at=due, source="brain")
    except ValueError as exc:
        return str(exc)
    if due is None:
        return f"Added {item['text']} to your to-do list."
    return f"Okay, I'll remind you to {item['text']} at {reminders.say_when(due)}."


@tool
def add_todo(text: str) -> str:
    """Put something on Harsh's to-do list (no time -- it is read when he asks,
    and in the daily briefing, never pushed on its own).

    Args:
        text: the task, e.g. "buy milk".
    """
    try:
        item = reminders.add(text, source="brain")
    except ValueError as exc:
        return str(exc)
    return f"Added {item['text']} to your to-do list."


@tool
def list_reminders(which: str = "all") -> str:
    """What is on Harsh's reminders and to-do list.

    Args:
        which: "all" (default), "today" (reminders due today), "todo" (the
            to-do list only) or "reminders" (timed reminders only).
    """
    which = (which or "all").strip().lower()
    if which == "today":
        return reminders.today_line()
    if which in ("todo", "to-do", "todos"):
        return reminders.todo_line()
    if which == "reminders":
        return reminders.reminders_line()
    return f"{reminders.reminders_line()} {reminders.todo_line()}"


@tool
def reminders_today() -> str:
    """Reminders due today, as one sentence (the good-morning routine uses it)."""
    return reminders.today_line()


@tool
def complete_reminder(which: str) -> str:
    """Mark a reminder or to-do item done ("I bought the milk", "tick off the
    dentist").

    Args:
        which: the item's words ("milk") or its number.
    """
    item = reminders.complete(which)
    if item is None:
        return f"I couldn't find {which} on your list."
    return f"Done — {item['text']} is ticked off."


@tool
def run_routine(name: str) -> str:
    """Run one of Harsh's predefined routines by name ("good morning",
    "movie time"). Say the result out loud as it is.

    Args:
        name: the routine's name.
    """
    import routines
    said, _ = routines.run(name.strip().lower())
    return said


@tool
def list_routines() -> str:
    """The routines she knows and what each one does."""
    import routines
    parts = [f"{r['name']}: {', '.join(r['steps'])}" for r in routines.describe()]
    return ("I know these routines — " + "; ".join(parts) + ".") if parts \
        else "I have no routines set up."


@tool
def quiet_mode(on: bool = True) -> str:
    """Quiet mode: she stops speaking up on her own (reminders wait, no
    remarks) but still answers when asked. The movie-time routine uses it.

    Args:
        on: True to go quiet, False to talk again.
    """
    import mode
    mode.set_quiet(bool(on), why="routine or brain")
    return ("Quiet mode on. I'll still answer if you ask."
            if on else "Quiet mode off.")


@tool
def calendar_today() -> str:
    """Today's calendar, as one sentence: her own calendar, plus Google's if
    it has been connected."""
    import day_calendar
    now = time.time()
    ahead = [e for e in day_calendar.merged_day(now) if e["end_ts"] > now]
    return (day_calendar.say(ahead, "On your calendar today: ")
            or "Nothing else on your calendar today.")


@tool
def calendar_upcoming(days: float = 1.0) -> str:
    """Upcoming calendar events.

    Args:
        days: how far ahead to look, in days (default 1; 7 for the week).
    """
    import day_calendar
    events = day_calendar.merged_upcoming(max(0.1, min(float(days), 31.0)))
    return (day_calendar.say(events, "Coming up: ")
            or "Nothing on your calendar in that time.")


@tool
def calendar_add_event(title: str, when: str, minutes: float = 60.0) -> str:
    """Put an event on Harsh's calendar (her own, on the robot).

    Args:
        title: what the event is, e.g. "Dentist".
        when: a spoken time -- "tomorrow at 10", "on friday at 3 pm" -- or
            an exact "YYYY-MM-DD HH:MM".
        minutes: how long it lasts (default 60).
    """
    import day_calendar
    from datetime import datetime
    start = None
    if when.strip():
        try:
            start = datetime.fromisoformat(when.strip()).timestamp()
        except ValueError:
            start = reminders.spoken_time(when)
    if start is None:
        return ("I need a time for that — say something like add dentist to "
                "my calendar tomorrow at 10.")
    try:
        event = day_calendar.add(title, start, float(minutes), source="voice")
    except ValueError as exc:
        return str(exc)
    return (f"Added {event['title']} to your calendar at "
            f"{reminders.say_when(event['start_ts'])}.")


@tool
def calendar_remove(which: str) -> str:
    """Take an upcoming event off the calendar.

    Args:
        which: the event's name ("dentist") or number.
    """
    import day_calendar
    found = day_calendar.remove(which)
    if found is None:
        return f"I couldn't find {which} on your calendar."
    return f"Removed {found['title']} from your calendar."


@tool
def daily_briefing() -> str:
    """Harsh's briefing for today: weather, first calendar event, today's
    reminders, how many to-dos. Say it as it is. Counts as today's briefing,
    so she will not offer it again on her own."""
    import briefing
    text = briefing.compose()
    briefing.mark_done(text)
    return text
