"""gcal.py — File 16 step 4: Google Calendar, read and write.

    gcal.connected()                     is there a token?
    gcal.today_events() / upcoming(days) [{title, start_ts, end_ts, all_day}]
    gcal.add_event("Dentist", start_ts, minutes=60)

PLAIN REST, NO NEW PACKAGES. The Calendar API is two HTTPS calls and an OAuth
refresh; httpx is already in the venv (the Anthropic client uses it), so there
is no google-api-python-client to install and keep current on the Pi.

THE TOKEN IS MADE ONCE, ON A LAPTOP. Google's consent screen needs a browser,
and the device flow does not allow calendar scopes. scripts/gcal_setup.py runs
on any computer with Python 3 and a browser (standard library only), and
writes google_token.json, which is copied to secrets/ here. ⚠ secrets/ is
gitignored; nothing here ever writes a credential anywhere else, and nothing
logs one.

Every call fails soft: no token, no internet or a Google error gives [] or an
honest sentence, never an exception into a voice turn.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.gcal")

TOKEN_URI = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/calendar/v3"
SCOPE = "https://www.googleapis.com/auth/calendar.events"

_cache: dict[tuple, tuple[float, list]] = {}


class CalendarError(RuntimeError):
    pass


def _token_file() -> Path:
    return Path(config.GCAL_TOKEN_FILE)


def connected() -> bool:
    try:
        token = json.loads(_token_file().read_text())
    except (OSError, ValueError):
        return False
    return bool(token.get("refresh_token") and token.get("client_id"))


def status() -> str:
    if connected():
        return "connected"
    return ("not connected — run scripts/gcal_setup.py on a computer with a "
            "browser and copy google_token.json into secrets/")


def _access_token() -> str:
    """A live access token, refreshed (and saved) when it is near expiry."""
    import httpx
    try:
        token = json.loads(_token_file().read_text())
    except (OSError, ValueError) as exc:
        raise CalendarError("the calendar is not connected") from exc
    if token.get("access_token") and float(token.get("expires_at") or 0) \
            > time.time() + 60:
        return token["access_token"]
    try:
        reply = httpx.post(token.get("token_uri") or TOKEN_URI, data={
            "client_id": token["client_id"],
            "client_secret": token.get("client_secret", ""),
            "refresh_token": token["refresh_token"],
            "grant_type": "refresh_token"}, timeout=config.GCAL_TIMEOUT_S)
    except Exception as exc:                            # noqa: BLE001
        raise CalendarError(f"could not reach Google ({type(exc).__name__})") \
            from exc
    if reply.status_code != 200:
        # The body names the error ("invalid_grant" = the token was revoked);
        # it never contains the secret, but log only the error code anyway.
        code = (reply.json() if reply.headers.get("content-type", "")
                .startswith("application/json") else {}).get("error", "")
        raise CalendarError(f"Google refused the token ({reply.status_code} "
                            f"{code})")
    fresh = reply.json()
    token["access_token"] = fresh["access_token"]
    token["expires_at"] = time.time() + float(fresh.get("expires_in", 3600))
    tmp = _token_file().with_name(".google_token.json.tmp")
    tmp.write_text(json.dumps(token))
    tmp.chmod(0o600)
    tmp.replace(_token_file())
    return token["access_token"]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).astimezone().isoformat()


def _parse_event(item: dict) -> dict[str, Any]:
    start, end = item.get("start") or {}, item.get("end") or {}
    all_day = "date" in start and "dateTime" not in start

    def ts(part: dict) -> float:
        if "dateTime" in part:
            return datetime.fromisoformat(
                part["dateTime"].replace("Z", "+00:00")).timestamp()
        if "date" in part:
            return datetime.fromisoformat(part["date"]).timestamp()
        return 0.0

    return {"id": item.get("id"), "title": item.get("summary") or "(no title)",
            "start_ts": ts(start), "end_ts": ts(end), "all_day": all_day,
            "where": item.get("location") or ""}


def events(start: float, end: float, use_cache: bool = True) -> list[dict]:
    """Events overlapping [start, end), soonest first. Raises CalendarError."""
    import httpx
    key = (round(start / 60), round(end / 60))
    hit = _cache.get(key)
    if use_cache and hit and time.time() - hit[0] < config.GCAL_CACHE_S:
        return hit[1]
    headers = {"Authorization": f"Bearer {_access_token()}"}
    try:
        reply = httpx.get(
            f"{API}/calendars/{config.GCAL_CALENDAR_ID}/events",
            params={"timeMin": _iso(start), "timeMax": _iso(end),
                    "singleEvents": "true", "orderBy": "startTime",
                    "maxResults": "25"},
            headers=headers, timeout=config.GCAL_TIMEOUT_S)
    except Exception as exc:                            # noqa: BLE001
        raise CalendarError(f"could not reach Google ({type(exc).__name__})") \
            from exc
    if reply.status_code != 200:
        raise CalendarError(f"Google Calendar said {reply.status_code}")
    found = [_parse_event(i) for i in reply.json().get("items") or []]
    _cache[key] = (time.time(), found)
    return found


def today_events(now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0,
                                                microsecond=0)
    return events(start.timestamp(), (start + timedelta(days=1)).timestamp())


def upcoming(days: float = 1.0, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    return events(now, now + days * 86400)


def add_event(title: str, start: float, minutes: float = 60.0) -> dict:
    """Create an event on the calendar. Raises CalendarError."""
    import httpx
    title = (title or "").strip()
    if not title:
        raise CalendarError("an event needs a name")
    body = {"summary": title,
            "start": {"dateTime": _iso(start)},
            "end": {"dateTime": _iso(start + minutes * 60)}}
    try:
        reply = httpx.post(
            f"{API}/calendars/{config.GCAL_CALENDAR_ID}/events", json=body,
            headers={"Authorization": f"Bearer {_access_token()}"},
            timeout=config.GCAL_TIMEOUT_S)
    except Exception as exc:                            # noqa: BLE001
        raise CalendarError(f"could not reach Google ({type(exc).__name__})") \
            from exc
    if reply.status_code not in (200, 201):
        raise CalendarError(f"Google Calendar said {reply.status_code}")
    _cache.clear()
    event = _parse_event(reply.json())
    # Step 4: calendar-derived notes go to memory with type "calendar".
    try:
        import memory
        import reminders
        memory.remember(f"Calendar: {title} at "
                        f"{reminders.say_when(event['start_ts'])} "
                        f"({datetime.fromtimestamp(event['start_ts']):%Y-%m-%d})",
                        kind="calendar", source="gcal")
    except Exception as exc:                            # noqa: BLE001
        log.warning("event made, but not noted in memory: %s", exc)
    log.info("calendar event added: %r at %s", title, _iso(start))
    return event
