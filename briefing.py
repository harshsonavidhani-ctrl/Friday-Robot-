"""briefing.py — File 16 step 5: the daily briefing, said once.

    briefing.compose()          the words: weather, first event, reminders, to-dos
    briefing.wanted(now, heard) is now the moment to offer it?
    briefing.mark_done(text)    said -- not again today

WHEN. Once a day, inside config.BRIEFING_WINDOW (morning), and only once
somebody is here: he spoke to her, or the scene watcher saw a person, within
BRIEFING_PRESENCE_S. A briefing to an empty room is noise. Then the SAME gate
every unprompted sentence goes through (initiative.decide, via ReminderWatch in
reminders.py): not while he talks, not over her own voice, not in quiet mode,
and only after INITIATIVE_IDLE_S of quiet -- so it follows his first exchange
of the morning rather than interrupting it. "Friday, give me my briefing"
says it on demand too (the daily_briefing tool), and that counts as today's.

WHAT. Each part is skipped rather than guessed when it cannot be had: no
internet means no weather sentence, no Google token means no calendar
sentence. Never raises.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import config

log = logging.getLogger("friday.briefing")


def _today(now: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now
                                  ).strftime("%Y-%m-%d")


def _state() -> dict:
    try:
        data = json.loads(Path(config.BRIEFING_FILE).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def done_today(now: float | None = None) -> bool:
    return _state().get("date") == _today(now)


def mark_done(text: str = "", now: float | None = None) -> None:
    now = time.time() if now is None else now
    try:
        path = Path(config.BRIEFING_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({"date": _today(now), "at": now,
                                   "said": text[:600]}))
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not record today's briefing: %s", exc)


def _person_seen_within(seconds: float, now: float) -> bool:
    try:
        scene = json.loads(Path(config.SCENE_JSON).read_text())
    except (OSError, ValueError):
        return False
    if now - float(scene.get("taken_at") or 0) > seconds:
        return False
    return any(o.get("label") == "person" and float(o.get("score") or 0) >= 0.6
               for o in scene.get("objects") or [])


def wanted(now: float | None = None, last_heard: float = 0.0) -> bool:
    """Morning window, not said yet today, and somebody is here."""
    now = time.time() if now is None else now
    if not getattr(config, "BRIEFING", False) or done_today(now):
        return False
    start, end = config.BRIEFING_WINDOW
    if not start <= datetime.fromtimestamp(now).hour < end:
        return False
    presence = float(config.BRIEFING_PRESENCE_S)
    return (now - last_heard <= presence
            or _person_seen_within(presence, now))


def _weather() -> str:
    try:
        import weather_source
        data, _ = weather_source.current()
    except Exception as exc:                        # noqa: BLE001
        log.info("briefing without weather: %s", exc)
        return ""
    if not data or data.get("temp_c") is None:
        return ""
    condition = str(data.get("condition") or "").lower()
    high = data.get("high_c")
    line = f"It's {round(data['temp_c'])} degrees"
    if condition:
        line += f" and {condition}"
    if high is not None:
        line += f", with a high of {round(high)}"
    return line + "."


def _calendar(now: float) -> str:
    try:
        import day_calendar
        events = day_calendar.merged_day(now)
    except Exception as exc:                        # noqa: BLE001
        log.info("briefing without the calendar: %s", exc)
        return ""
    ahead = [e for e in events if e.get("start_ts", 0) >= now - 600]
    if not ahead:
        return ""
    first = ahead[0]
    import reminders
    when = ("all day" if first.get("all_day")
            else f"at {reminders.say_when(first['start_ts'], now)}")
    more = len(ahead) - 1
    return (f"Your first event is {first['title']} {when}"
            + (f", and {more} more after that." if more else "."))


def compose(now: float | None = None) -> str:
    """The briefing's words. Never empty, never raises."""
    now = time.time() if now is None else now
    parts = ["Here's your day."]
    for piece in (_weather, lambda: _calendar(now)):
        try:
            text = piece()
        except Exception as exc:                    # noqa: BLE001
            log.info("briefing part skipped: %s", exc)
            text = ""
        if text:
            parts.append(text)
    try:
        import reminders
        today = reminders.today(now)
        if today:
            parts.append(reminders.today_line(now))
        todo = reminders.todos()
        if todo:
            parts.append(f"And {len(todo)} thing{'s' if len(todo) != 1 else ''}"
                         f" on your to-do list.")
        if not today and not todo:
            parts.append("No reminders today.")
    except Exception as exc:                        # noqa: BLE001
        log.info("briefing without reminders: %s", exc)
    return " ".join(parts)
