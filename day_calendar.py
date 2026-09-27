"""day_calendar.py — a simple calendar that lives on the robot (file 16, step 4).

    day_calendar.add("Dentist", start_ts, minutes=60)
    day_calendar.on_day(ts)          events that day, soonest first
    day_calendar.parse_request("add dentist to my calendar tomorrow at 10")

Harsh, 2026-09-27: "the calendar one add a simple calendar okay" -- so the
calendar is HERS: a `calendar_events` table in memory.db (kind='calendar'),
no Google account, no OAuth, no internet. gcal.py stays for later; when it is
connected, calendar_today/upcoming merge its events in, and nothing else
depends on it.

Named day_calendar, not calendar: a module called calendar would shadow the
standard library's, which the time code in this repo imports.

An event is not a reminder, but she mentions it CALENDAR_HEADS_UP_MIN before it
starts ("In 10 minutes: dentist.") through the same watch and gate reminders
use (reminders.ReminderWatch), once.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any

import config

log = logging.getLogger("friday.calendar")

KIND = "calendar"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_events (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    start_at    REAL NOT NULL,
    minutes     REAL NOT NULL DEFAULT 60,
    kind        TEXT NOT NULL DEFAULT 'calendar',
    source      TEXT,
    created_at  REAL NOT NULL,
    announced_at REAL
);
CREATE INDEX IF NOT EXISTS calendar_start ON calendar_events(start_at);
"""

_ready = False


def _db():
    global _ready
    import memory
    connection = memory.db()
    if not _ready:
        with memory._lock:
            connection.executescript(SCHEMA)
            connection.commit()
        _ready = True
    return connection


def _row(r) -> dict[str, Any]:
    return {"id": r["id"], "title": r["title"], "start_ts": r["start_at"],
            "end_ts": r["start_at"] + r["minutes"] * 60, "minutes": r["minutes"],
            "all_day": False, "kind": r["kind"], "announced_at": r["announced_at"]}


def add(title: str, start: float, minutes: float = 60.0,
        source: str = "voice") -> dict:
    import memory
    title = re.sub(r"\s+", " ", (title or "")).strip(" .,!?")
    if not title:
        raise ValueError("an event needs a name")
    with memory._lock:
        cursor = _db().execute(
            "INSERT INTO calendar_events (title, start_at, minutes, kind, "
            "source, created_at) VALUES (?,?,?,?,?,?)",
            (title, float(start), float(minutes), KIND, source, time.time()))
        _db().commit()
    log.info("calendar #%d: %r at %s", cursor.lastrowid, title,
             time.strftime("%a %H:%M", time.localtime(start)))
    return _row(_db().execute("SELECT * FROM calendar_events WHERE id=?",
                              (cursor.lastrowid,)).fetchone())


def between(start: float, end: float) -> list[dict]:
    rows = _db().execute(
        "SELECT * FROM calendar_events WHERE start_at < ? AND "
        "start_at + minutes * 60 > ? ORDER BY start_at", (end, start)).fetchall()
    return [_row(r) for r in rows]


def on_day(ts: float | None = None) -> list[dict]:
    ts = time.time() if ts is None else ts
    day = datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0,
                                             microsecond=0)
    return between(day.timestamp(), (day + timedelta(days=1)).timestamp())


def upcoming(days: float = 7.0, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    return between(now, now + days * 86400)


def remove(which: str) -> dict | None:
    """Delete the best-matching FUTURE event by title (or id)."""
    import memory
    import reminders
    which = (which or "").strip()
    items = upcoming(366)
    number = re.fullmatch(r"#?\s*(\d+)", which)
    if number:
        found = next((e for e in items if e["id"] == int(number.group(1))), None)
    else:
        scored = sorted(((reminders._similar(which, e["title"]), e)
                         for e in items), key=lambda p: -p[0])
        found = scored[0][1] if scored and scored[0][0] >= 0.5 else None
    if found is None:
        return None
    with memory._lock:
        _db().execute("DELETE FROM calendar_events WHERE id=?", (found["id"],))
        _db().commit()
    return found


def heads_up_due(now: float | None = None) -> list[dict]:
    """Events starting within CALENDAR_HEADS_UP_MIN that were not mentioned."""
    now = time.time() if now is None else now
    ahead = float(getattr(config, "CALENDAR_HEADS_UP_MIN", 10)) * 60
    rows = _db().execute(
        "SELECT * FROM calendar_events WHERE announced_at IS NULL AND "
        "start_at > ? AND start_at <= ? ORDER BY start_at",
        (now - 120, now + ahead)).fetchall()
    return [_row(r) for r in rows]


def mark_announced(ids: list[int]) -> None:
    import memory
    with memory._lock:
        for item in ids:
            _db().execute("UPDATE calendar_events SET announced_at=? WHERE id=?",
                          (time.time(), int(item)))
        _db().commit()


# --- merged with Google, when (and only when) it is connected -----------------

def merged_day(ts: float | None = None) -> list[dict]:
    events = on_day(ts)
    try:
        import gcal
        if gcal.connected():
            events += gcal.today_events(ts)
    except Exception as exc:                        # noqa: BLE001
        log.info("google calendar skipped: %s", exc)
    return sorted(events, key=lambda e: e["start_ts"])


def merged_upcoming(days: float = 1.0, now: float | None = None) -> list[dict]:
    events = upcoming(days, now)
    try:
        import gcal
        if gcal.connected():
            events += gcal.upcoming(days, now)
    except Exception as exc:                        # noqa: BLE001
        log.info("google calendar skipped: %s", exc)
    return sorted(events, key=lambda e: e["start_ts"])


def say(events: list[dict], lead: str) -> str:
    import reminders
    if not events:
        return ""
    words = [e["title"] + ("" if e.get("all_day")
                           else f" at {reminders.say_when(e['start_ts'])}")
             for e in events[:int(config.REMINDER_LIST_MAX)]]
    if len(events) > len(words):
        words.append(f"{len(events) - len(words)} more")
    return lead + (words[0] if len(words) == 1
                   else ", ".join(words[:-1]) + " and " + words[-1]) + "."


# --- what he said -------------------------------------------------------------

_CAL = r"(?:my\s+|the\s+)?(?:calendar|schedule|diary)"
_ADD = re.compile(
    rf"\b(?:add|put|schedule|book|write)\s+(?P<a>.+?)\s+(?:to|on|in|into)\s+"
    rf"{_CAL}\b\s*(?P<wa>.*)$"
    rf"|\b(?:add|put)\s+(?:to|on|in)\s+{_CAL}[:,]?\s+(?P<b>.+)$",
    re.IGNORECASE)
_ASK = re.compile(
    rf"\b(?:what(?:'s| is)?|anything|read|tell me|check|show)\b.{{0,20}}\b"
    rf"(?:on|in)\s+{_CAL}\b(?P<rest>.*)$", re.IGNORECASE)
_REMOVE = re.compile(
    rf"\b(?:remove|delete|cancel|take)\s+(?P<a>.+?)\s+(?:from|off)\s+{_CAL}\b",
    re.IGNORECASE)


def _split_event(text: str, now: datetime | None):
    """"dentist tomorrow at 10" -> ("dentist", ts). The leftmost all-time tail."""
    import reminders
    for marker in reminders._MARKERS.finditer(text):
        tail, title = text[marker.start():], text[:marker.start()]
        if title.strip() and reminders._is_timeish(tail):
            when = reminders.spoken_time(tail, now)
            if when is not None:
                return title.strip(" ,."), when
    return text.strip(" ,."), None


def parse_request(heard: str, now: datetime | None = None) -> dict | None:
    """{"action": "add", "title", "start"} | {"action": "ask", "days"} |
    {"action": "remove", "which"} | None."""
    heard = (heard or "").strip()
    match = _REMOVE.search(heard)
    if match:
        return {"action": "remove", "which": match.group("a").strip(" ,.")}
    match = _ADD.search(heard)
    if match:
        if match.group("a"):
            title = match.group("a")
            when_text = (match.group("wa") or "").strip(" .,!?")
            start = None
            if when_text:
                import reminders
                start = reminders.spoken_time(when_text, now)
            if start is None:
                title, start = _split_event(title, now)
        else:
            title, start = _split_event(match.group("b"), now)
        return {"action": "add", "title": title.strip(" ,."), "start": start}
    match = _ASK.search(heard)
    if match:
        rest = (match.group("rest") or "").lower()
        days = (7 if "week" in rest else 1)
        tomorrow = "tomorrow" in rest
        return {"action": "ask", "days": days, "tomorrow": tomorrow}
    return None
