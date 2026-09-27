"""reminders.py — File 16 steps 1-2: reminders and the to-do list.

    reminders.add("call mom", due_at=epoch)     # a reminder
    reminders.add("buy milk")                   # a to-do: no due time
    reminders.open_items()                      # everything not done
    reminders.complete("milk")                  # best match, marked done
    reminders.parse_request("remind me to call mom at 5 pm")

ONE TABLE, ONE TYPE. Both live in memory.db's `reminders` table with
kind='reminder' -- the type value the file-17 spec reserves for this. A
reminder has a due_at; a to-do does not. Structured columns rather than a row
in `facts`, because facts are de-duplicated on write (memory.remember updates
a near-duplicate in place) and two reminders to "call mom" on different days
must stay two.

NOT A TIMER, NOT AN ALARM. app_control's timers and alarms RING, from the face
process, at a second. A reminder is SAID, once, in words, through the same
may-she-speak-now gate every other unprompted sentence uses (initiative.py):
never over him, never over herself, never in quiet mode. ReminderWatch below is
a wake source like scene_bridge and narration, so what it says lands between
turns. To-do items are never pushed: they are read when he asks, and in the
daily briefing (briefing.py).

TIME WORDS. app_control.parse_when is what alarms use and it is left alone,
but it reads "7:30 in the morning" as 19:30, "half past six" as midnight and
cannot read "noon" -- so reminders parse their own times (spoken_time below)
and reuse only its number words and durations.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Iterator

import config
from wake_word import WakeSource

log = logging.getLogger("friday.reminders")

KIND = "reminder"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    due_at      REAL,
    done        INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'reminder',
    source      TEXT,
    created_at  REAL NOT NULL,
    said_at     REAL,
    done_at     REAL
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders(done, due_at);
"""

_ready = False


def _db():
    """memory.db's connection, with this table on it. Never a second file."""
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
    return {"id": r["id"], "text": r["text"], "due_at": r["due_at"],
            "done": bool(r["done"]), "kind": r["kind"],
            "said_at": r["said_at"], "created_at": r["created_at"]}


# --- the data -----------------------------------------------------------------

def add(text: str, due_at: float | None = None, source: str = "voice") -> dict:
    """Store one reminder (with due_at) or to-do (without). Returns the row."""
    import memory
    text = re.sub(r"\s+", " ", (text or "")).strip(" .,!?")
    if not text:
        raise ValueError("a reminder needs something to remind you of")
    now = time.time()
    with memory._lock:
        cursor = _db().execute(
            "INSERT INTO reminders (text, due_at, kind, source, created_at) "
            "VALUES (?,?,?,?,?)", (text, due_at, KIND, source, now))
        _db().commit()
    log.info("%s #%d: %r%s", "reminder" if due_at else "to-do",
             cursor.lastrowid, text,
             f" at {time.strftime('%a %H:%M', time.localtime(due_at))}"
             if due_at else "")
    return {"id": cursor.lastrowid, "text": text, "due_at": due_at,
            "done": False, "kind": KIND, "said_at": None, "created_at": now}


def open_items() -> list[dict]:
    """Everything not done: reminders soonest first, then to-dos oldest first."""
    rows = _db().execute(
        "SELECT * FROM reminders WHERE done=0 AND kind=? "
        "ORDER BY due_at IS NULL, due_at, id", (KIND,)).fetchall()
    return [_row(r) for r in rows]


def todos() -> list[dict]:
    return [r for r in open_items() if r["due_at"] is None]


def upcoming() -> list[dict]:
    return [r for r in open_items() if r["due_at"] is not None]


def today(now: float | None = None) -> list[dict]:
    """Reminders due between now's midnight and the next, not done."""
    now = time.time() if now is None else now
    start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0,
                                                microsecond=0)
    end = (start + timedelta(days=1)).timestamp()
    return [r for r in upcoming() if start.timestamp() <= r["due_at"] < end]


def due_unsaid(now: float | None = None) -> list[dict]:
    """Reminders whose time has come and that she has not said yet."""
    now = time.time() if now is None else now
    return [r for r in upcoming() if r["due_at"] <= now and not r["said_at"]]


def mark_said(ids: list[int]) -> None:
    """Said out loud: a reminder that has been delivered is finished."""
    import memory
    now = time.time()
    with memory._lock:
        for item in ids:
            _db().execute("UPDATE reminders SET said_at=?, done=1, done_at=? "
                          "WHERE id=?", (now, now, int(item)))
        _db().commit()


def _similar(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9
    words_a, words_b = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    overlap = len(words_a & words_b) / max(1, len(words_a | words_b))
    return max(overlap, SequenceMatcher(None, a, b).ratio())


def find(which: str) -> dict | None:
    """An open item by id ("3", "#3") or by the best text match (>= 0.5)."""
    which = (which or "").strip()
    items = open_items()
    number = re.fullmatch(r"#?\s*(\d+)", which)
    if number:
        return next((r for r in items if r["id"] == int(number.group(1))), None)
    scored = sorted(((_similar(which, r["text"]), r) for r in items),
                    key=lambda pair: -pair[0])
    return scored[0][1] if scored and scored[0][0] >= 0.5 else None


def complete(which: str) -> dict | None:
    """Mark the best-matching open item done. None if nothing matched."""
    import memory
    item = find(which)
    if item is None:
        return None
    with memory._lock:
        _db().execute("UPDATE reminders SET done=1, done_at=? WHERE id=?",
                      (time.time(), item["id"]))
        _db().commit()
    log.info("done #%d: %r", item["id"], item["text"])
    return item


def delete(item_id: int) -> bool:
    import memory
    with memory._lock:
        cursor = _db().execute("DELETE FROM reminders WHERE id=?",
                               (int(item_id),))
        _db().commit()
    return cursor.rowcount > 0


# --- saying them --------------------------------------------------------------

def say_when(epoch: float, now: float | None = None) -> str:
    """"5:00 PM", "9:00 AM tomorrow", "10:30 AM on Tuesday"."""
    now = time.time() if now is None else now
    when = datetime.fromtimestamp(epoch)
    stamp = when.strftime("%H:%M" if getattr(config, "APP_CLOCK_24H", False)
                          else "%-I:%M %p")
    days = (when.date() - datetime.fromtimestamp(now).date()).days
    if days == 0:
        return stamp
    if days == 1:
        return f"{stamp} tomorrow"
    if 1 < days < 7:
        return f"{stamp} on {when.strftime('%A')}"
    return f"{stamp} on {when.strftime('%B %-d')}"


def _listing(items: list[dict], with_time: bool) -> str:
    cap = int(config.REMINDER_LIST_MAX)
    words = [(f"{r['text']} at {say_when(r['due_at'])}" if with_time
              and r["due_at"] else r["text"]) for r in items[:cap]]
    if len(items) > cap:
        words.append(f"{len(items) - cap} more")
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def today_line(now: float | None = None) -> str:
    items = today(now)
    if not items:
        return "No reminders for today."
    return (f"{'One reminder' if len(items) == 1 else f'{len(items)} reminders'}"
            f" today: {_listing(items, with_time=True)}.")


def todo_line() -> str:
    items = todos()
    if not items:
        return "Your to-do list is empty."
    return (f"{'One thing' if len(items) == 1 else f'{len(items)} things'} on "
            f"your to-do list: {_listing(items, with_time=False)}.")


def reminders_line() -> str:
    items = upcoming()
    if not items:
        return "You have no reminders set."
    return (f"{'One reminder' if len(items) == 1 else f'{len(items)} reminders'}"
            f": {_listing(items, with_time=True)}.")


# --- spoken times --------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
_TIMEISH = set("""at in on by after before around about tomorrow today tonight
this next morning afternoon evening night noon midnight am pm a.m. p.m. o'clock
oclock half quarter past to minute minutes min mins hour hours hr hrs second
seconds day days week and a an the halfpast quarterpast quarterto""".split()) \
    | set(_WEEKDAYS)


def _fractions(text: str) -> str:
    """"half past six" -> "halfpast six" BEFORE number words are converted:
    app_control._numbers turns "half" into 0.5 and "quarter" into 0.25, which
    reads "half past six" as "0.5 past 6"."""
    text = re.sub(r"\bhalf past\b", "halfpast", text)
    text = re.sub(r"\bquarter past\b", "quarterpast", text)
    return re.sub(r"\bquarter to\b", "quarterto", text)


def _is_timeish(text: str) -> bool:
    """Every word is a time word or a number -- "at the car at 5" is not."""
    import app_control
    words = re.findall(r"\d+(?:[:.]\d+)?|[a-z.']+", app_control._numbers(
        _fractions(text.lower())))
    return bool(words) and all(w in _TIMEISH
                               or re.fullmatch(r"\d+([:.]\d+)?", w)
                               for w in words)


def spoken_time(text: str, now: datetime | None = None) -> float | None:
    """A spoken time for a reminder -> epoch, or None if it is not one.

    "5 pm", "at 7:30 in the morning", "half past six", "quarter to 9",
    "noon", "tomorrow", "tomorrow morning", "tonight", "on monday at 10",
    "next friday", "in 20 minutes", "in an hour". A time already gone today,
    with no day said, rolls to tomorrow; an hour with no am/pm is the SOONEST
    one ahead ("at 7" said at 9 pm is 7 am).
    """
    import app_control
    now = now or datetime.now()
    raw = re.sub(r"\s+", " ", (text or "").lower().strip(" .,!?"))
    if not raw:
        return None
    relative = re.fullmatch(r"(?:in|after)\s+(.+)", raw)
    if relative:
        try:
            return (now + timedelta(
                seconds=app_control.parse_duration(relative.group(1)))
                    ).timestamp()
        except app_control.ScheduleError:
            return None

    clean = app_control._numbers(_fractions(raw))
    clean = re.sub(r"\bnoon\b|\bmidday\b", "12:00 pm", clean)
    clean = re.sub(r"\bmidnight\b", "12:00 am", clean)
    clean = re.sub(r"\bhalfpast (\d{1,2})\b", r"\1:30", clean)
    clean = re.sub(r"\bquarterpast (\d{1,2})\b", r"\1:15", clean)
    clean = re.sub(r"\bquarterto (\d{1,2})\b",
                   lambda m: f"{(int(m.group(1)) - 1) or 12}:45", clean)

    # THE DAY
    day_offset: int | None = None
    if re.search(r"\b(tomorrow|tmrw)\b", clean):
        day_offset = 1
    elif re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b",
                   clean):
        day_offset = 0
    else:
        # "on friday" / "next friday" / "this friday" -- NEVER a bare "friday":
        # it is her name, and "remind me friday ..." is him talking to her.
        match = re.search(r"\b(on|next|this)\s+(" + "|".join(_WEEKDAYS)
                          + r")\b", clean)
        if match:
            want = _WEEKDAYS.index(match.group(2))
            ahead = (want - now.weekday()) % 7
            if ahead == 0 or match.group(1) == "next" and ahead < 7:
                ahead = ahead or 7
            day_offset = ahead

    # THE PART OF DAY
    part = None
    for name in ("morning", "afternoon", "evening", "tonight", "night"):
        if re.search(rf"\b{name}\b", clean):
            part = name
            break

    # THE CLOCK
    match = re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
                      r"(?=\s|$)", clean)
    hour = minute = None
    meridiem = ""
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        meridiem = (match.group(3) or "").replace(".", "")
        if hour > 23 or minute > 59:
            return None
    if hour is None:
        if day_offset is None and part is None:
            return None                     # no day, no time: not a time
        hour, minute = (int(config.REMINDER_PART_OF_DAY.get(
            part, config.REMINDER_DEFAULT_HOUR)), 0)
        meridiem = "set"
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif not meridiem and hour < 12 and part in ("afternoon", "evening",
                                                 "tonight", "night"):
        hour += 12
    elif not meridiem and hour == 12 and part == "morning":
        hour = 0
    elif not meridiem and not part and day_offset is not None \
            and 1 <= hour <= 6:
        # "tomorrow at 5" is five in the afternoon; nobody books 5 am by
        # leaving it out. 7-11 stay morning ("monday at 9").
        hour += 12

    base = now.replace(second=0, microsecond=0)

    def at(h: int, d: int) -> datetime:
        return base.replace(hour=h % 24, minute=minute) + timedelta(days=d)

    if day_offset is not None:
        return at(hour, day_offset).timestamp()
    if meridiem or part or hour >= 12:
        when = at(hour, 0)
        return (when if when > now else when + timedelta(days=1)).timestamp()
    options = [at(h, d) for h in (hour, hour + 12) for d in (0, 1)]
    return min(w for w in options if w > now).timestamp()


# --- what he said -------------------------------------------------------------

_REMIND = re.compile(r"\bremind me\b\s*(.*)$", re.IGNORECASE)
_MARKERS = re.compile(r"\b(at|in|on|by|after|tomorrow|tonight|today|this|next|"
                      r"noon|midnight)\b", re.IGNORECASE)
_TASK_LEAD = re.compile(r"^(?:to|that|about|of)\s+", re.IGNORECASE)
_TODO_ADD = re.compile(
    r"\b(?:add|put|write)\s+(?P<task>.+?)\s+(?:to|on|in)\s+(?:my|the)\s+"
    r"(?:to-?\s?do|todo)(?:\s+list)?\b"
    r"|\b(?:add|new)\s+(?:a\s+)?(?:to-?\s?do|todo)(?:\s+item)?[:,]?\s+"
    r"(?:to\s+)?(?P<task2>.+)$", re.IGNORECASE)
_TODO_ASK = re.compile(
    r"\b(?:what(?:'s| is| are)?|read|tell me|list|show|check)\b.{0,20}\b"
    r"(?:to-?\s?do|todo)\b|\bmy\s+(?:to-?\s?do|todo)(?:\s+list)?\s*\??$",
    re.IGNORECASE)
_REMINDERS_ASK = re.compile(
    r"\b(?:what(?:'s| is| are)?|read|tell me|list|show|check|any)\b.{0,20}"
    r"\breminders?\b", re.IGNORECASE)
_DONE = re.compile(
    r"\b(?:mark|tick off|cross off|check off)\s+(?P<a>.+?)\s*(?:as\s+)?"
    r"(?:done|complete|completed|finished|off)\b"
    r"|\b(?:remove|delete|take|cross|tick)\s+(?P<b>.+?)\s+(?:off|from)\s+"
    r"(?:my|the)\s+(?:to-?\s?do|todo|reminders?|list)(?:\s+list)?\b"
    r"|\b(?:tick|cross|check)\s+off\s+(?P<c>.+)$",
    re.IGNORECASE)


def _clean_task(text: str) -> str:
    text = _TASK_LEAD.sub("", (text or "").strip(" .,!?"))
    text = re.sub(r"^(?:please\s+)|(?:\s+please)$", "", text, flags=re.I)
    return text.strip(" .,!?")


def parse_request(heard: str, now: datetime | None = None) -> dict | None:
    """What a sentence asks of the reminder list, or None.

    {"action": "add", "text", "due_at"}  -- due_at None means a to-do
    {"action": "list", "which": "todo"|"reminders"}
    {"action": "done", "which": text}
    """
    heard = (heard or "").strip()
    if not heard:
        return None

    match = _DONE.search(heard)
    if match:
        which = _clean_task(match.group("a") or match.group("b")
                            or match.group("c") or "")
        which = re.sub(r"^(?:the\s+)?(?:reminder|to-?\s?do)\s+(?:to\s+|about\s+)?",
                       "", which, flags=re.I)
        if which:
            return {"action": "done", "which": which}

    match = _TODO_ADD.search(heard)
    if match:
        task = _clean_task(match.group("task") or match.group("task2") or "")
        if task:
            return {"action": "add", "text": task, "due_at": None}

    match = _REMIND.search(heard)
    if match:
        rest = match.group(1).strip(" .,!?")
        # "remind me at 5 to call mom": the time first.
        lead = re.match(r"(.+?)\s+(?:to|that|about)\s+(.+)$", rest, re.I)
        if lead and _is_timeish(lead.group(1)):
            due = spoken_time(lead.group(1), now)
            if due is not None:
                return {"action": "add", "text": _clean_task(lead.group(2)),
                        "due_at": due}
        # "remind me to look at the car at 5 tomorrow": the LEFTMOST split
        # whose whole tail is time words, so "at the car" stays in the task
        # and "tomorrow" is not dropped from the time.
        for marker in _MARKERS.finditer(rest):
            tail = rest[marker.start():]
            task = rest[:marker.start()]
            if not task.strip() or not _is_timeish(tail):
                continue
            due = spoken_time(tail, now)
            if due is not None:
                return {"action": "add", "text": _clean_task(task),
                        "due_at": due}
        task = _clean_task(rest)
        if task and not re.fullmatch(r"(?:in|at|on)\b.*", task, re.I):
            return {"action": "add", "text": task, "due_at": None}
        return None

    if _TODO_ASK.search(heard):
        return {"action": "list", "which": "todo"}
    if _REMINDERS_ASK.search(heard):
        return {"action": "list", "which": "reminders"}
    return None


# --- the watch: due reminders, and the daily briefing -------------------------

NAME = "reminders"


class ReminderWatch(WakeSource):
    """Yields `json` {"kind": "reminder"|"briefing", "say", "ids", "at"}.

    A wake source like scene_bridge and narration: what it yields is spoken
    from main.py's loop BETWEEN turns, never on top of one. It asks
    initiative.decide() before offering anything, and keeps a due reminder
    PENDING while he is talking, she is speaking, quiet mode is on or she has
    been told to stop listening.
    """

    name = NAME

    def __init__(self) -> None:
        self._closed = threading.Event()
        self._paused = False
        self._asleep_until = 0.0
        self._last_heard = 0.0
        self._inflight: dict[int, float] = {}     # id -> when offered
        self._briefing_inflight = 0.0
        self._landmark_inflight = 0.0
        self._research_inflight: dict[str, float] = {}
        self._last_why = ""

    # -- told by main.py -------------------------------------------------------
    def heard_harsh(self) -> None:
        self._last_heard = time.time()

    def delivered(self, item: dict) -> None:
        if item.get("kind") == "reminder":
            mark_said(item.get("ids") or [])
        elif item.get("kind") == "calendar":
            import day_calendar
            day_calendar.mark_announced(item.get("ids") or [])
        elif item.get("kind") == "research":
            import laptop_research
            laptop_research.mark_announced(str(item.get("job", "")))
        elif item.get("kind") == "landmark":
            import environments
            for i in item.get("ids") or []:
                environments.mark_asked(int(i))
        elif item.get("kind") == "briefing":
            import briefing
            briefing.mark_done(item.get("say", ""))

    def missed(self, item: dict) -> None:
        sign = -1 if item.get("kind") == "calendar" else 1
        for i in item.get("ids") or []:
            self._inflight.pop(sign * int(i), None)
        if item.get("kind") == "briefing":
            self._briefing_inflight = 0.0
        if item.get("kind") == "landmark":
            self._landmark_inflight = 0.0
        if item.get("kind") == "research":
            self._research_inflight.pop(str(item.get("job", "")), None)

    # -- WakeSource ------------------------------------------------------------
    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self, seconds: float) -> float:
        self._asleep_until = time.monotonic() + seconds
        return seconds

    def wake_up(self) -> None:
        self._asleep_until = 0.0

    def sleeping_for(self) -> float:
        return max(0.0, self._asleep_until - time.monotonic())

    def close(self) -> None:
        self._closed.set()

    def listen(self) -> Iterator[str]:
        log.info("reminder watch ready (poll %.0fs; briefing %s)",
                 float(config.REMINDER_POLL_S),
                 "on" if getattr(config, "BRIEFING", False) else "off")
        while not self._closed.wait(float(config.REMINDER_POLL_S)):
            try:
                item = self.step()
            except Exception as exc:                # noqa: BLE001
                log.warning("reminder step failed: %s: %s",
                            type(exc).__name__, exc)
                continue
            if item is not None:
                yield json.dumps(item)

    def _why_not(self, kind: str, idle_for: float | None = None) -> str | None:
        if self._paused:
            return "mid-turn"
        if self.sleeping_for() > 0:
            return "told to stop listening"
        try:
            import initiative
            channel, why = initiative.decide(kind, have_something=True,
                                             idle_for=idle_for)
        except Exception as exc:                    # noqa: BLE001
            return f"gate unavailable ({exc})"
        return None if channel == initiative.VOICE else why

    def _note(self, why: str) -> None:
        if why != self._last_why:
            log.info("reminders waiting: %s", why)
            self._last_why = why

    def step(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        # Offered but not confirmed for 2 minutes: assume it was lost.
        self._inflight = {i: t for i, t in self._inflight.items()
                          if now - t < 120}
        due = [r for r in due_unsaid(now) if r["id"] not in self._inflight]
        if due:
            why = self._why_not("reminder")
            if why:
                self._note(why)
                return None
            for r in due:
                self._inflight[r["id"]] = now
            return {"kind": "reminder", "at": now,
                    "ids": [r["id"] for r in due],
                    "say": self._reminder_words(due, now)}

        try:
            import day_calendar
            soon = [e for e in day_calendar.heads_up_due(now)
                    if -e["id"] not in self._inflight]
        except Exception as exc:                    # noqa: BLE001
            log.debug("calendar heads-up skipped: %s", exc)
            soon = []
        if soon:
            why = self._why_not("calendar")
            if why:
                self._note(f"calendar — {why}")
                return None
            for e in soon:
                self._inflight[-e["id"]] = now       # negative: not a reminder id
            words = " ".join(
                f"In {max(1, round((e['start_ts'] - now) / 60))} minutes: "
                f"{e['title']}." for e in soon)
            return {"kind": "calendar", "at": now,
                    "ids": [e["id"] for e in soon], "say": words}

        # FILE 19 STEP 4, the voice half: a new area waits for a name AND he
        # is mid-conversation. The dashboard queue is the reliable half; this
        # only asks when someone is plainly here to answer.
        try:
            import environments
            unasked = [p for p in environments.pending() if not p["asked_at"]]
        except Exception as exc:                    # noqa: BLE001
            log.debug("landmark ask skipped: %s", exc)
            unasked = []
        if (unasked and now - self._last_heard
                <= float(getattr(config, "LANDMARK_ASK_WINDOW_S", 90.0))
                and now - self._landmark_inflight > 120):
            why = self._why_not("landmark")
            if not why:
                area = unasked[-1]
                self._landmark_inflight = now
                return {"kind": "landmark", "at": now, "ids": [area["id"]],
                        "say": (f"I found a new area of about "
                                f"{max(1, round(area['area_m2']))} square "
                                f"metres. What should I call it? Say: call "
                                f"it, and a name.")}

        # FILE 20: research the laptop finished (or her fallback) -- said once.
        self._research_inflight = {k: t for k, t in
                                   self._research_inflight.items()
                                   if now - t < 120}
        try:
            import laptop_research
            ended = [j for j in laptop_research.finished()
                     if j["id"] not in self._research_inflight]
        except Exception as exc:                    # noqa: BLE001
            log.debug("research check skipped: %s", exc)
            ended = []
        if ended:
            why = self._why_not("research")
            if why:
                self._note(f"research — {why}")
                return None
            job = ended[0]
            self._research_inflight[job["id"]] = now
            return {"kind": "research", "at": now, "ids": [],
                    "job": job["id"], "say": research_words(job)}

        if getattr(config, "BRIEFING", False) and \
                now - self._briefing_inflight > 120:
            import briefing
            if briefing.wanted(now, self._last_heard):
                why = self._why_not("daily briefing",
                                    idle_for=now - self._last_heard)
                if why:
                    self._note(f"briefing — {why}")
                    return None
                text = briefing.compose(now)
                self._briefing_inflight = now
                return {"kind": "briefing", "at": now, "ids": [],
                        "say": text}
        return None

    @staticmethod
    def _reminder_words(due: list[dict], now: float) -> str:
        lines = []
        for r in due:
            late = now - float(r["due_at"])
            if late > 300:
                lines.append(f"Earlier, at {say_when(r['due_at'], now)}, you "
                             f"asked me to remind you: {r['text']}.")
            else:
                lines.append(f"Reminder: {r['text']}.")
        return " ".join(lines)


def research_words(job: dict) -> str:
    """What she says when notes for the laptop are done or failed (file 20)."""
    query = job.get("query") or "that"
    summary = (job.get("summary") or "").strip()
    if job.get("status") == "done":
        name = [p for p in re.split(r"[\\/]", str(job.get("path") or ""))
                if p][-1:] or ["a new file"]
        words = (f"Your notes on {query} are open on your laptop, saved in "
                 f"Friday Notes as {name[0]}.")
        return f"{words} In short: {summary}" if summary else words
    if job.get("stage") == "save":
        return (f"I researched {query}, but my laptop went away before I "
                f"could save the notes.")
    return f"I couldn't research {query} just now, so there are no notes."


def parse_trigger(trigger: str) -> dict | None:
    prefix = f"{NAME}: "
    if not trigger.startswith(prefix):
        return None
    try:
        item = json.loads(trigger[len(prefix):])
    except ValueError:
        return None
    return item if isinstance(item, dict) and item.get("say") else None
