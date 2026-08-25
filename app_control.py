"""app_control.py — the doorway to Friday's screen apps.

The apps themselves live in apps/ and run inside the face process, because
that process owns the only pygame window on the panel. Nothing else may draw
there. So, exactly like face_control.py, this module is a file writer: it
writes what should be on screen and what is scheduled, and the face picks it
up on its next poll. No sockets, no server, no importing pygame.

    import app_control
    app_control.launch("clock")                  # show a view
    app_control.dismiss()                        # a spoken or touched "stop"
    app_control.add_alarm("8 AM", "school")      # rings even after a reboot
    app_control.add_timer("10 minutes", "pasta")
    app_control.set_weather({...})               # the brain supplies the data

    python3 app_control.py clock                 # same from the shell
    python3 app_control.py alarm "8 am" school
    python3 app_control.py --list

TWO CHANNELS, FOR TWO DIFFERENT LIFETIMES.

  COMMAND   state/app_command.json — "put this on the screen now". A single
            latest-wins message with a token, read once. Fire and forget.
  SCHEDULE  state/schedule.json    — alarms and timers, which are promises.
            They outlive the process that made them, the face, and the Pi's
            power. The face is what watches the clock against this file.

The face writes state/app_status.json back, so a tool can answer "what is on
the screen right now" without guessing.

WHY THE FACE OWNS THE CLOCK, NOT THE BRAIN. An alarm set at 11pm has to ring
at 7am whether or not there is internet, whether or not the MCP server has
been restarted since, and whether or not anyone has spoken to Friday in the
meantime. The face process is the one thing that is always up in front of
Harsh, so it is the one that keeps the promise.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import config

COMMAND_FILE = config.APP_COMMAND_FILE
STATUS_FILE = config.APP_STATUS_FILE
SCHEDULE_FILE = config.APP_SCHEDULE_FILE
STATE_DIR = COMMAND_FILE.parent

# The views apps/ registers. Kept here as well so a tool can validate a name
# without importing pygame -- apps/ imports pygame, this module must not.
APP_NAMES: tuple[str, ...] = ("clock", "timer", "alarm", "weather")


class ScheduleError(ValueError):
    """A time Friday could not make sense of. The message is spoken back."""


# =====================================================================
# atomic files
# =====================================================================

def _write_atomic(path: Path, payload: dict, prefix: str) -> None:
    """Temp file in the SAME directory, then rename -- the rename is atomic on
    one filesystem, so a reader never sees half a JSON object."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


# =====================================================================
# the command channel
# =====================================================================

def _command(action: str, **payload) -> dict:
    message = {"action": action, "token": time.monotonic_ns(), **payload}
    _write_atomic(COMMAND_FILE, message, ".appcmd.")
    return message


def launch(name: str, **kwargs) -> str:
    """Put an app on the screen. Returns what to say back."""
    name = (name or "").strip().lower()
    if name not in APP_NAMES:
        raise ScheduleError(f"there is no {name!r} app. Friday has: "
                            f"{', '.join(APP_NAMES)}")
    _command("launch", app=name, kwargs=kwargs)
    return f"{name} is on the screen"


def dismiss() -> str:
    """Stop whatever is on the screen and give it back to the face."""
    _command("dismiss")
    return "screen dismissed"


def show_face() -> str:
    """Same as dismiss, said the other way round."""
    _command("dismiss")
    return "back to the face"


def status() -> dict:
    """What the face says is on the screen. Stale by at most one frame."""
    return _read_json(STATUS_FILE, {"app": None, "since": 0.0})


def running() -> str | None:
    app = status().get("app")
    return app or None


# =====================================================================
# the schedule
# =====================================================================

_EMPTY = {"version": 1, "alarms": [], "timers": []}


def read_schedule() -> dict:
    data = _read_json(SCHEDULE_FILE, _EMPTY)
    for key in ("alarms", "timers"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def write_schedule(schedule: dict) -> None:
    schedule["version"] = 1
    _write_atomic(SCHEDULE_FILE, schedule, ".sched.")


def _new_id(kind: str, schedule: dict) -> str:
    used = {item.get("id") for item in schedule.get(kind + "s", [])}
    n = 1
    while f"{kind}{n}" in used:
        n += 1
    return f"{kind}{n}"


def add_alarm(when: str, label: str = "") -> dict:
    """Parse a spoken time and store an alarm. Raises ScheduleError if the
    time cannot be understood -- the message is meant to be said out loud."""
    at = parse_when(when)
    schedule = read_schedule()
    alarm = {
        "id": _new_id("alarm", schedule),
        "at": at,
        "label": _clean_label(label),
        "fired": False,
        "created": time.time(),
    }
    schedule["alarms"].append(alarm)
    schedule["alarms"].sort(key=lambda a: a.get("at", 0))
    write_schedule(schedule)
    return alarm


def add_timer(duration: str | float, label: str = "") -> dict:
    """Store a countdown. Unlike an alarm this also puts itself on the screen,
    because a timer with nothing showing is just a surprise later."""
    seconds = (float(duration) if isinstance(duration, (int, float))
               else parse_duration(str(duration)))
    if seconds <= 0:
        raise ScheduleError("a timer has to be longer than zero")
    schedule = read_schedule()
    timer = {
        "id": _new_id("timer", schedule),
        "at": time.time() + seconds,
        "seconds": seconds,
        "label": _clean_label(label),
        "fired": False,
        "created": time.time(),
    }
    schedule["timers"].append(timer)
    schedule["timers"].sort(key=lambda t: t.get("at", 0))
    write_schedule(schedule)
    return timer


def cancel(item_id: str) -> str:
    """Cancel one alarm or timer by id, or 'all'."""
    item_id = (item_id or "").strip().lower()
    schedule = read_schedule()
    if item_id in ("all", "everything"):
        count = len(schedule["alarms"]) + len(schedule["timers"])
        schedule["alarms"], schedule["timers"] = [], []
        write_schedule(schedule)
        return f"cancelled {count}" if count else "there was nothing set"
    for kind in ("alarms", "timers"):
        keep = [i for i in schedule[kind] if i.get("id") != item_id]
        if len(keep) != len(schedule[kind]):
            schedule[kind] = keep
            write_schedule(schedule)
            return f"{item_id} cancelled"
    return f"there is no {item_id}"


def pending() -> list[dict]:
    """Alarms and timers still ahead, soonest first, each tagged with kind."""
    schedule = read_schedule()
    out = []
    for kind in ("alarm", "timer"):
        for item in schedule[kind + "s"]:
            if item.get("fired"):
                continue
            out.append({**item, "kind": kind})
    out.sort(key=lambda i: i.get("at", 0))
    return out


def mark_fired(item_id: str) -> None:
    """The face calls this once it has actually rung. Read-modify-write, so
    an alarm added while one is ringing is not lost."""
    schedule = read_schedule()
    for kind in ("alarms", "timers"):
        for item in schedule[kind]:
            if item.get("id") == item_id:
                item["fired"] = True
                item["fired_at"] = time.time()
    # A fired timer is spent; a fired alarm is kept until the next one is set
    # so "did it go off?" can still be answered.
    schedule["timers"] = [t for t in schedule["timers"] if not t.get("fired")]
    schedule["alarms"] = [a for a in schedule["alarms"]
                          if not a.get("fired")
                          or time.time() - a.get("fired_at", 0) < 3600]
    write_schedule(schedule)


def due(now: float | None = None,
        grace: float | None = None) -> list[dict]:
    """Everything that should be ringing right now.

    Anything due while the Pi was off still rings if it is inside the grace
    window; older than that and it is dropped rather than going off at
    breakfast for something that was wanted at midnight."""
    now = time.time() if now is None else now
    grace = config.APP_ALARM_GRACE_S if grace is None else grace
    ready, stale = [], []
    for item in pending():
        at = item.get("at", 0)
        if at > now:
            continue
        (ready if now - at <= grace else stale).append(item)
    for item in stale:
        mark_fired(item["id"])
    return ready


# =====================================================================
# weather
# =====================================================================

def set_weather(data: dict, hold: float | None = None) -> str:
    """Show a weather card. Friday has no weather API key and no sensor for
    the sky -- the brain looks it up (it has web search) and hands the numbers
    here, which is also why this takes a dict rather than a place name."""
    payload = {k: data.get(k) for k in
               ("place", "condition", "temp_c", "feels_c", "high_c", "low_c",
                "wind_kph", "humidity", "summary")}
    return launch("weather", weather=payload,
                  hold=hold if hold is not None else config.APP_WEATHER_HOLD_S)


# =====================================================================
# parsing what people actually say
# =====================================================================

_DURATION_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
}

_WORD_NUMBERS = {
    "half": 0.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fortyfive": 45, "sixty": 60, "ninety": 90,
}


def _clean_label(label: str) -> str:
    label = re.sub(r"\s+", " ", (label or "")).strip()
    return label[:config.APP_LABEL_MAX_CHARS]


def _numbers(text: str) -> str:
    """Turn spoken numbers into digits so one regex handles both."""
    for word, value in _WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b",
                      str(int(value)) if value == int(value) else str(value),
                      text)
    return text


def parse_duration(text: str) -> float:
    """"10 minutes", "1h30m", "90 seconds", "an hour and a half", "10".

    A bare number means MINUTES, because "set a timer for ten" always does.
    """
    raw = (text or "").strip().lower()
    if not raw:
        raise ScheduleError("how long?")
    clean = raw.replace("an ", "1 ").replace("a ", "1 ")
    # "and a half" is half of whatever came before it, which the unit regex
    # cannot see -- so resolve it here, while the unit is still in the string.
    clean = re.sub(r"1?\s*half\s*(1|an)?\s*hour", "30 minutes", clean)
    clean = re.sub(r"hour[s]?\s*(and\s*)?1?\s*half", "90 minutes", clean)
    clean = re.sub(r"minute[s]?\s*(and\s*)?1?\s*half", "90 seconds", clean)
    clean = _numbers(clean).replace("and", " ")

    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", clean):
        if unit in _DURATION_UNITS:
            total += float(value) * _DURATION_UNITS[unit]
    if total:
        return total

    bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", clean)
    if bare:
        return float(bare.group(1)) * 60.0
    raise ScheduleError(f"I could not tell how long {text!r} is")


def parse_when(text: str, now: datetime | None = None) -> float:
    """A spoken time -> an epoch second.

    "8 AM", "8:30pm", "19:45", "tomorrow at 7", "in 10 minutes", "8" (the
    next 8 o'clock either way). A time that has already gone today rolls to
    tomorrow -- an alarm is always ahead of you.
    """
    raw = (text or "").strip().lower()
    if not raw:
        raise ScheduleError("what time?")
    now = now or datetime.now()

    if raw.startswith("in ") or raw.startswith("after "):
        return (now + timedelta(
            seconds=parse_duration(raw.split(" ", 1)[1]))).timestamp()

    clean = _numbers(raw)
    tomorrow = "tomorrow" in clean or "tmrw" in clean
    tonight = "tonight" in clean or "evening" in clean
    clean = re.sub(r"\b(tomorrow|tmrw|tonight|today|at|o'clock|oclock|"
                   r"in the|morning|evening|sharp)\b", " ", clean)

    match = re.search(r"(\d{1,2})[:.\s]?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?",
                      clean)
    if not match:
        raise ScheduleError(f"I could not tell what time {text!r} is")

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").replace(".", "")

    if minute > 59:
        raise ScheduleError(f"{minute} is not a minute")
    if meridiem.startswith("p") and hour < 12:
        hour += 12
    elif meridiem.startswith("a") and hour == 12:
        hour = 0
    if hour > 23:
        raise ScheduleError(f"{hour} is not an hour")

    def at(h: int, day_offset: int = 0):
        return (now.replace(hour=h % 24, minute=minute, second=0,
                            microsecond=0) + timedelta(days=day_offset))

    if meridiem or tomorrow or hour > 12:
        when = at(hour, 1 if tomorrow else 0)
        if when <= now:
            when += timedelta(days=1)
        return when.timestamp()

    # No am/pm and an hour that could be either: take whichever comes round
    # SOONEST. "wake me at 7" said at 9pm means 7am, ten hours away -- not
    # 7pm, twenty-two hours away, and not a time in the past.
    if tonight and hour < 12:
        hour += 12
        return (at(hour) if at(hour) > now else at(hour, 1)).timestamp()
    options = [at(h, d) for h in (hour, (hour + 12) % 24) for d in (0, 1)]
    return min(w for w in options if w > now).timestamp()


# =====================================================================
# saying times back
# =====================================================================

def say_time(epoch: float) -> str:
    """"8:00 AM", or "8:00 AM tomorrow" when that is what it means."""
    when = datetime.fromtimestamp(epoch)
    fmt = "%H:%M" if config.APP_CLOCK_24H else "%-I:%M %p"
    stamp = when.strftime(fmt)
    days = (when.date() - datetime.now().date()).days
    if days == 1:
        return f"{stamp} tomorrow"
    if days > 1:
        return f"{stamp} on {when.strftime('%A')}"
    return stamp


def say_duration(seconds: float) -> str:
    """"10 minutes", "1 hour 30 minutes", "45 seconds"."""
    seconds = int(round(max(0.0, seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if secs and not hours:
        parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
    return " ".join(parts) or "no time at all"


# =====================================================================
# CLI
# =====================================================================

_USAGE = """usage:
  app_control.py clock|weather            show a view
  app_control.py timer <duration> [label]
  app_control.py alarm <time> [label]
  app_control.py --list                   what is scheduled
  app_control.py --cancel <id|all>
  app_control.py --status                 what is on the screen
  app_control.py --dismiss
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    head, rest = argv[0], argv[1:]
    try:
        if head in ("--list", "-l"):
            items = pending()
            if not items:
                print("nothing scheduled")
            for item in items:
                print(f"{item['id']:<8} {item['kind']:<6} "
                      f"{say_time(item['at']):<16} {item.get('label', '')}")
        elif head == "--cancel":
            print(cancel(rest[0] if rest else ""))
        elif head == "--status":
            print(json.dumps(status()))
        elif head in ("--dismiss", "stop", "dismiss"):
            print(dismiss())
        elif head == "timer":
            timer = add_timer(rest[0], " ".join(rest[1:]))
            launch("timer")
            print(f"timer for {say_duration(timer['seconds'])}")
        elif head == "alarm":
            alarm = add_alarm(rest[0], " ".join(rest[1:]))
            print(f"alarm for {say_time(alarm['at'])}")
        else:
            print(launch(head))
    except IndexError:
        print(_USAGE, file=sys.stderr)
        return 2
    except ScheduleError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
