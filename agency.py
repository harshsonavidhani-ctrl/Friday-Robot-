"""agency.py — Friday acts on her own: she notices, thinks, and does ONE thing.

    watch = agency.AgencyWatch()     # a wake source in main.py's AnyOf
    agency.situation()               # what she knows right now, as a dict
    agency.decide(sit)               # {"action", "text", "emotion", "why"}

Harsh, 2026-09-27: "make the brain more powerful ... thinking and controlling
the Friday body on its own, like humans do according to the needs and
environment". So between conversations she looks at her situation -- the time,
who is here and where, what is in view, what is coming up, how long since he
spoke, what she already did -- and one small Haiku call chooses ONE action:

    nothing          the usual answer, and a good one
    say              one short sentence to whoever is here (greet him back,
                     mention the event in 20 minutes, suggest rest at 1 am)
    feel             show an emotion on the face (bored, curious, happy...)
    show             the clock or the weather on the panel
    look_around      a slow look left and right            } MOTION: only when
    turn_to_person   face the person at her side           } AGENCY_MOTION is
    explore_briefly  a short guarded wander (explore_room)  } True (ships False)

WHAT KEEPS IT A COMPANION AND NOT A NUISANCE
  * Speech goes through initiative.decide() -- never while he talks, never over
    her own voice, never in quiet mode -- and is SPOKEN from main.py's loop,
    between turns, like narration and reminders.
  * She only thinks when the situation has CHANGED (a fingerprint of it) and at
    least AGENCY_MIN_GAP_S since the last time, or once per AGENCY_MAX_GAP_S
    anyway; RATE_LIMITS["agency"] caps the calls (~$0.001 each) and the daily
    spend cap covers them. Nobody here = no "say".
  * She is told what she did in the last hour, and never repeats herself.

WHAT KEEPS IT SAFE
  * Every movement is an MCP tool call on her own session (brain.mcp_session),
    so it keeps embodiment.check, guarded_move, the clearance interrupt, the
    duration cap and stop(): nothing here touches a motor. Short pivots only
    (at most AGENCY_TURN_MAX_S), a fresh scan_environment first, and never
    while a run holds the robot, a turn is in progress or she was told to stop
    listening. explore_briefly is the same explore_room the idle drive uses.
  * AGENCY_MOTION ships False, exactly like PHYSICAL_CURIOSITY: whether she
    moves on her own is Harsh's call after a supervised test.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import config
from wake_word import WakeSource

log = logging.getLogger("friday.agency")

NAME = "agency"
SPEAK = ("say",)
FACE = ("feel",)
SCREEN = ("show",)
MOTION = ("look_around", "turn_to_person", "explore_briefly")
ACTIONS = ("nothing",) + SPEAK + FACE + SCREEN + MOTION

# The content emotions the face arbiter accepts from anyone but the pipeline.
FEELINGS = ("happy", "curious", "bored", "excited", "love", "proud", "wink",
            "surprised", "sad", "confused")


# --- what she knows -----------------------------------------------------------

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def situation(now: float | None = None, last_heard: float = 0.0,
              recent: list[dict] | None = None) -> dict[str, Any]:
    """Everything she can know right now without spending anything."""
    now = time.time() if now is None else now
    local = datetime.fromtimestamp(now)
    sit: dict[str, Any] = {
        "time": local.strftime("%A %H:%M"),
        "hour": local.hour,
        "since_he_spoke_min": (round((now - last_heard) / 60)
                               if last_heard else None),
    }
    scene = _read_json(config.SCENE_JSON)
    age = now - float(scene.get("taken_at") or 0)
    objects = [o for o in scene.get("objects") or []
               if float(o.get("score") or 0) >= 0.6] if age < 60 else []
    people = [o for o in objects if o.get("label") == "person"]
    sit["camera_fresh"] = age < 60
    sit["people_in_view"] = [{"side": p.get("side"),
                              "bearing_deg": p.get("bearing_deg")}
                             for p in people]
    sit["things_in_view"] = sorted({o["label"] for o in objects
                                    if o.get("label") != "person"})
    try:
        import reminders
        soon = [r for r in reminders.upcoming()
                if 0 <= r["due_at"] - now <= 3 * 3600]
        sit["reminders_next_3h"] = [
            {"text": r["text"], "in_min": round((r["due_at"] - now) / 60)}
            for r in soon[:3]]
        sit["todo_count"] = len(reminders.todos())
    except Exception:                               # noqa: BLE001
        pass
    try:
        import day_calendar
        sit["calendar_next_3h"] = [
            {"title": e["title"], "in_min": round((e["start_ts"] - now) / 60)}
            for e in day_calendar.upcoming(3 / 24, now)[:3]]
    except Exception:                               # noqa: BLE001
        pass
    try:
        import mode
        sit["quiet_mode"] = mode.quiet()
    except Exception:                               # noqa: BLE001
        sit["quiet_mode"] = False
    explore = _read_json(config.ROOT / "state" / "explore_mode.json")
    sit["exploring"] = str(explore.get("state", "")).lower() == "running"
    sit["did_recently"] = [
        {"min_ago": round((now - r["at"]) / 60), "action": r["action"],
         "text": r.get("text", "")} for r in (recent or [])
        if now - r["at"] < 3600][-6:]
    return sit


def fingerprint(sit: dict) -> str:
    """What must CHANGE for her to think again: people, things, what is
    coming up, the hour, and whether he has spoken lately."""
    key = {
        "people": [p.get("side") for p in sit.get("people_in_view") or []],
        "things": sit.get("things_in_view"),
        "reminders": [r["text"] for r in sit.get("reminders_next_3h") or []
                      if r["in_min"] <= 30],
        "calendar": [e["title"] for e in sit.get("calendar_next_3h") or []
                     if e["in_min"] <= 30],
        "hour": sit.get("hour"),
        "he_spoke_lately": (sit.get("since_he_spoke_min") is not None
                            and sit["since_he_spoke_min"] <= 30),
        "quiet": sit.get("quiet_mode"),
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()


def someone_here(sit: dict) -> bool:
    spoke = sit.get("since_he_spoke_min")
    return bool(sit.get("people_in_view")) or (
        spoke is not None
        and spoke * 60 <= float(config.AGENCY_AUDIENCE_S))


# --- the decision --------------------------------------------------------------

def _prompt(sit: dict) -> tuple[str, str]:
    motion = bool(getattr(config, "AGENCY_MOTION", False))
    menu = [
        '"nothing" — the usual, and right whenever nothing calls for more',
        '"say" + "text" — ONE short sentence (under 20 words) to whoever is '
        'here: welcome him back after a long gap, mention something coming up '
        'in the next half hour, suggest rest very late at night, react to a '
        'person arriving. Only if someone is here.',
        '"feel" + "emotion" — show it on your face: ' + ", ".join(FEELINGS),
        '"show" + "text": "clock" or "weather" — put it on your screen',
    ]
    if motion:
        menu += [
            '"look_around" — a slow look left and right, when alone and bored',
            '"turn_to_person" — face a person who is at your left or right',
            '"explore_briefly" — a short careful wander, when alone and idle '
            'for a long time in the daytime',
        ]
    system = (
        "You are Friday's own initiative: the part of a small home robot that "
        "decides, between conversations, whether to do something by herself — "
        "the way a thoughtful companion in the room would. Harsh built you and "
        "lives with you.\n\n"
        "Pick EXACTLY ONE action:\n- " + "\n- ".join(menu) + "\n\n"
        "Rules: most of the time choose \"nothing\" — silence is not a failure. "
        "Never repeat something in did_recently. Never speak when nobody is "
        "here. Be warm and brief, never chatty or needy, never narrate your "
        "own actions. Night (after 23:00) means quiet unless something is "
        "urgent.\n"
        'Reply with ONLY a JSON object: {"action": ..., "text": ..., '
        '"emotion": ..., "why": "<one short reason>"}')
    return system, json.dumps(sit, ensure_ascii=False)


def parse_decision(raw: str) -> dict[str, Any]:
    """The model's JSON, checked. Anything malformed or disallowed -> nothing."""
    try:
        found = re.search(r"\{.*\}", raw or "", re.S)
        data = json.loads(found.group(0)) if found else {}
    except (ValueError, AttributeError):
        data = {}
    action = str(data.get("action") or "nothing").strip().lower()
    decision = {"action": action if action in ACTIONS else "nothing",
                "text": str(data.get("text") or "").strip()[:200],
                "emotion": str(data.get("emotion") or "").strip().lower(),
                "why": str(data.get("why") or "").strip()[:160]}
    if decision["action"] == "feel" and decision["emotion"] not in FEELINGS:
        decision["action"] = "nothing"
    if decision["action"] == "say" and not decision["text"]:
        decision["action"] = "nothing"
    if decision["action"] == "show" and decision["text"] not in ("clock",
                                                                "weather"):
        decision["action"] = "nothing"
    return decision


def allowed(decision: dict, sit: dict) -> str | None:
    """Why this decision may NOT be carried out, or None. Code, not prompt."""
    action = decision["action"]
    if action in MOTION:
        if not getattr(config, "AGENCY_MOTION", False):
            return "motion is switched off (AGENCY_MOTION)"
        if sit.get("exploring"):
            return "a run already has the robot"
        start, end = config.AGENCY_MOTION_HOURS
        if not start <= int(sit.get("hour", 0)) < end:
            return "not during the day"
        if action == "turn_to_person" and not any(
                p.get("side") in ("left", "right")
                for p in sit.get("people_in_view") or []):
            return "nobody at her side to turn to"
    if action in SPEAK:
        if not someone_here(sit):
            return "nobody here to hear it"
        if sit.get("quiet_mode"):
            return "quiet mode"
        text = decision["text"].lower()
        if any(text == r.get("text", "").lower()
               for r in sit.get("did_recently") or []):
            return "she said that already"
    return None


def decide(sit: dict) -> tuple[dict | None, str]:
    """One Haiku call. (decision, why) -- None when it could not be asked."""
    try:
        import brain
        import rate_limit
        import scene_describe
        key = scene_describe._api_key()
        if not key:
            return None, "no API key"
        if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return None, "over the daily spend cap"
        ok, why = rate_limit.allow(NAME)
        if not ok:
            return None, f"rate limited — {why}"
        import anthropic
        system, user = _prompt(sit)
        reply = anthropic.Anthropic(api_key=key).messages.create(
            model=config.AGENCY_MODEL, max_tokens=150,
            system=system, messages=[{"role": "user", "content": user}])
        rate_limit.used(NAME)
        brain._record_spend(config.AGENCY_MODEL, reply.usage,
                            brain._cost_usd(config.AGENCY_MODEL, reply.usage))
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:100]}"
    text = "".join(b.text for b in reply.content
                   if getattr(b, "type", "") == "text")
    return parse_decision(text), ""


# --- the body -------------------------------------------------------------------

async def _move(session, direction: str, seconds: float) -> str:
    result = await session.call_tool("move", {
        "direction": direction,
        "duration_seconds": round(min(seconds, config.AGENCY_TURN_MAX_S), 2)})
    return "".join(getattr(i, "text", "") for i in result.content)


async def _act_motion(action: str, sit: dict) -> str:
    """ONE guarded motion, on her own MCP session. Never raises past here."""
    import brain
    async with brain.mcp_session() as session:
        scan = await session.call_tool("scan_environment", {})
        report = json.loads("".join(getattr(i, "text", "")
                                    for i in scan.content) or "{}")
        safe = report.get("safe_directions") or {}
        if action == "turn_to_person":
            person = next(p for p in sit["people_in_view"]
                          if p.get("side") in ("left", "right"))
            seconds = abs(float(person.get("bearing_deg") or 30)) \
                / float(config.AGENCY_TURN_DEG_PER_S)
            return await _move(session, person["side"], max(0.2, seconds))
        if action == "look_around":
            out = [await _move(session, "left", config.AGENCY_LOOK_S)]
            await asyncio.sleep(1.5)
            out.append(await _move(session, "right", 2 * config.AGENCY_LOOK_S))
            await asyncio.sleep(1.5)
            out.append(await _move(session, "left", config.AGENCY_LOOK_S))
            return " | ".join(out)
        if action == "explore_briefly":
            if safe.get("forward") is not True:
                return "not wandering: forward is not clear"
            result = await session.call_tool("explore_room", {
                "budget_m": config.AGENCY_EXPLORE_M,
                "budget_seconds": config.AGENCY_EXPLORE_S})
            return "".join(getattr(i, "text", "") for i in result.content)
    return "nothing done"


def _act_screen(what: str) -> str:
    import app_control
    if what == "weather":
        return app_control.set_weather()
    return app_control.launch("clock")


# --- the watch ------------------------------------------------------------------

class AgencyWatch(WakeSource):
    """Thinks on its own thread; yields `json` for what main.py must do
    (speak, or set the face). Screen and motion it does itself, never
    mid-turn."""

    name = NAME

    def __init__(self) -> None:
        self._closed = threading.Event()
        self._paused = False
        self._asleep_until = 0.0
        self._last_heard = 0.0
        self._last_think = 0.0
        self._last_print = ""
        self._last_why = ""
        self.recent: collections.deque = collections.deque(maxlen=20)
        self._loop = asyncio.new_event_loop()

    # told by main.py
    def heard_harsh(self) -> None:
        self._last_heard = time.time()

    # WakeSource
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
        log.info("agency ready (think every %.0f-%.0f s when idle; motion %s)",
                 float(config.AGENCY_MIN_GAP_S), float(config.AGENCY_MAX_GAP_S),
                 "ON" if getattr(config, "AGENCY_MOTION", False) else "off")
        asyncio.set_event_loop(self._loop)
        while not self._closed.wait(float(config.AGENCY_POLL_S)):
            try:
                item = self.step()
            except Exception as exc:                # noqa: BLE001
                log.warning("agency step failed: %s: %s",
                            type(exc).__name__, exc)
                continue
            if item is not None:
                yield json.dumps(item)

    def _note(self, why: str) -> None:
        if why != self._last_why:
            log.info("agency: %s", why)
            self._last_why = why

    def _gate(self, now: float) -> str | None:
        if not getattr(config, "AGENCY", False):
            return "switched off"
        if self._paused:
            return "mid-turn"
        if self.sleeping_for() > 0:
            return "told to stop listening"
        if now - self._last_heard < float(config.AGENCY_IDLE_S):
            return "he spoke moments ago"
        try:
            import initiative
            if initiative.he_is_talking():
                return "he is talking"
            if initiative.she_is_talking():
                return "she is speaking"
        except Exception:                           # noqa: BLE001
            pass
        return None

    def remember(self, action: str, text: str = "", why: str = "") -> None:
        self.recent.append({"at": time.time(), "action": action,
                            "text": text, "why": why})
        try:
            path = Path(config.AGENCY_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"recent": list(self.recent)[-10:]}))
        except OSError:
            pass

    def step(self, now: float | None = None,
             sit: dict | None = None) -> dict | None:
        now = time.time() if now is None else now
        why = self._gate(now)
        if why:
            self._note(why)
            return None
        sit = sit or situation(now, self._last_heard, list(self.recent))
        mark = fingerprint(sit)
        since = now - self._last_think
        if since < float(config.AGENCY_MIN_GAP_S):
            return None
        if mark == self._last_print and since < float(config.AGENCY_MAX_GAP_S):
            return None                             # nothing new to think about
        self._last_think, self._last_print = now, mark

        decision, why = decide(sit)
        if decision is None:
            self._note(f"could not think ({why})")
            return None
        refused = allowed(decision, sit)
        if refused:
            log.info("agency chose %s — not done: %s", decision["action"],
                     refused)
            return None
        action = decision["action"]
        log.info("agency: %s %s (%s)", action,
                 decision.get("text") or decision.get("emotion") or "",
                 decision.get("why"))
        if action == "nothing":
            return None
        if action in SPEAK:
            try:
                import initiative
                channel, gate = initiative.decide("initiative", True)
            except Exception as exc:                # noqa: BLE001
                channel, gate = None, str(exc)
            if channel != "voice":
                log.info("agency: not saying it — %s", gate)
                return None
            self.remember("say", decision["text"], decision["why"])
            return {"kind": "say", "at": now, "say": decision["text"],
                    "why": decision["why"]}
        if action in FACE:
            self.remember("feel", decision["emotion"], decision["why"])
            return {"kind": "feel", "at": now, "emotion": decision["emotion"],
                    "why": decision["why"]}
        if action in SCREEN:
            result = _act_screen(decision["text"])
            self.remember("show", decision["text"], decision["why"])
            log.info("agency showed %s: %s", decision["text"], str(result)[:80])
            return None
        if action in MOTION:
            if self._paused:
                return None
            try:
                result = self._loop.run_until_complete(
                    _act_motion(action, sit))
            except Exception as exc:                # noqa: BLE001
                result = f"failed: {type(exc).__name__}: {exc}"
            self.remember(action, "", decision["why"])
            log.info("agency %s -> %s", action, str(result)[:160])
        return None


def parse_trigger(trigger: str) -> dict | None:
    prefix = f"{NAME}: "
    if not trigger.startswith(prefix):
        return None
    try:
        item = json.loads(trigger[len(prefix):])
    except ValueError:
        return None
    return item if isinstance(item, dict) and item.get("kind") else None
