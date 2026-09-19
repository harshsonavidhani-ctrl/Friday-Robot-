"""situation.py — what Friday already knows, without asking a tool for it.

    block = situation.block()      # "" when there is nothing fresh to say

WHY THIS EXISTS. Every tool call is a second round trip to the model: measured
on this Pi at 2.6 s to the first token with the tool list attached, so "are you
exploring?" answered through `exploration_status` costs about seven seconds and
two requests to say "no". The answer was sitting in a JSON file on disk the
whole time.

So the cheap, local, already-written state rides along with the question the
way memory does -- after the cached system+tools prefix, never inside it, so
the cache survives. She answers about herself in ONE round instead of two, and
every OTHER answer is better for knowing where she is and what she was doing.

⚠ STALENESS IS THE ONLY REAL RISK HERE, and it is worse than saying nothing.
`state/rl_autonomy.json` is written only while a run is driving; tonight it is
seven hours old and still describes a live-looking robot with a wall 0.35 m to
its left. Handing that to the model as the present tense would have her state,
confidently and out loud, something that stopped being true this afternoon.
So every field carries a maximum age, nothing is emitted past it, and anything
borderline is emitted WITH its age attached rather than silently. A block that
says nothing is correct; a block that is quietly wrong is not.

Nothing here may raise. A missing file, a half-written one, a JSON error -- all
of it degrades to "no block", because this is an enhancement to a turn and must
never be the reason a turn fails.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.situation")

# How old each source may be before it stops being the present tense. These
# are deliberately different: a mood goes stale in minutes, the weather does
# not, and a driving run's sensor frame is meaningless a minute later.
MOOD_MAX_AGE_S = 600.0          # 10 min
RUN_MAX_AGE_S = 60.0            # a live run writes every tick
EXPLORE_MAX_AGE_S = 6 * 3600.0  # "what did you do today" is still a fair ask
WEATHER_MAX_AGE_S = 3 * 3600.0
# Long enough that a bare "thank you" could still be a reply to her. Past it,
# a fragment with no addressee is the room talking, not a conversation.
SPOKE_RECENT_S = 120.0


def _age(path: Path) -> float | None:
    """Seconds since this file was written, or None if it isn't there."""
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _read_json(path: Path) -> Any | None:
    """Parsed JSON, or None. A torn half-written file is not an error here."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _said_age(seconds: float) -> str:
    """An age the way a person says one, for a line she may read aloud."""
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    hours = seconds / 3600
    if hours < 2:
        return "about an hour ago"
    return f"about {int(round(hours))} hours ago"


def _clock() -> str:
    now = time.localtime()
    return time.strftime("%A %-d %B, %-I:%M %p", now).replace("AM", "am").replace("PM", "pm")


def _mood() -> str | None:
    path = config.ROOT / "state" / "emotion.txt"
    age = _age(path)
    if age is None or age > MOOD_MAX_AGE_S:
        return None
    try:
        mood = path.read_text().strip()
    except OSError:
        return None
    return f"your face is showing {mood}" if mood else None


def _last_spoke() -> str | None:
    """How long since her own voice was last in the room.

    THE ADDRESSED TEST NEEDS THIS AND HAD NO WAY TO GET IT. "Thank you." and
    "Yes, I do." are a reply when she has just finished speaking and a
    television when she has not, and the model cannot tell which from the
    history: a turn she ignored is never stored, so the last exchange in there
    may be hours old while the clock says nothing about it. Measured
    2026-09-19: she answered "Thank you.", "Yes, I do." and "come on." from a
    video, each one minutes after she had last said a word to anybody.

    `state/speaking.json` is written by tts.speak() wherever it runs -- the
    exploration narrator and the RL scripts included -- so it is the one
    honest record of when she last made a sound.
    """
    data = _read_json(config.SPEAKING_FILE)
    if not isinstance(data, dict):
        return None
    try:
        since = time.time() - float(data.get("until") or data.get("at"))
    except (TypeError, ValueError):
        return None
    if since < 0:
        return None                 # she is still speaking; not a fact yet
    if since <= SPOKE_RECENT_S:
        return f"you last spoke {int(since)} seconds ago"
    return f"you last spoke {_said_age(since)}"


def _exploring() -> str | None:
    """Whether a run is going, just went, or failed -- and how long ago.

    The distinction that matters: `state` is what the run REPORTED, and the
    file's age is how long ago it reported it. A run that says "running" in a
    file last touched seven hours ago did not finish cleanly; it was killed,
    and she should say that rather than claim to be driving.
    """
    path = config.ROOT / "state" / "explore_mode.json"
    age = _age(path)
    data = _read_json(path)
    if age is None or data is None or age > EXPLORE_MAX_AGE_S:
        return None
    state = str(data.get("state", "")).lower()
    mode = str(data.get("mode", "a run"))
    if state == "running" and age <= RUN_MAX_AGE_S:
        return f"you are exploring right now ({mode})"
    if state == "running":
        return (f"an exploration ({mode}) was last heard from "
                f"{_said_age(age)} and is not reporting — it was probably "
                f"stopped; you are NOT driving now")
    return (f"you are not exploring. Your last run ({mode}) {state} "
            f"{_said_age(age)}")


def _around_me() -> str | None:
    """The last sensor frame, only while it is still the present tense."""
    path = config.ROOT / "state" / "rl_autonomy.json"
    age = _age(path)
    if age is None or age > RUN_MAX_AGE_S:
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    lidar = (data.get("SENSOR_STATE") or {}).get("lidar") or {}
    parts = []
    for name in ("front", "left", "right", "rear"):
        value = lidar.get(name)
        if isinstance(value, (int, float)):
            parts.append(f"{name} {value:.2f} m")
    if not parts:
        return None
    return "clearance around you: " + ", ".join(parts)


def _weather() -> str | None:
    path = config.ROOT / "state" / "weather.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    fetched = data.get("fetched_at")
    if not isinstance(fetched, (int, float)):
        return None
    if time.time() - fetched > WEATHER_MAX_AGE_S:
        return None
    w = data.get("weather") or {}
    place, condition, temp = w.get("place"), w.get("condition"), w.get("temp_c")
    if not (place and condition and temp is not None):
        return None
    return f"outside in {place} it is {condition.lower()}, {temp:.0f} degrees"


def _scene() -> str | None:
    """What the scene watcher's camera named a few seconds ago (scene.py)."""
    data = _read_json(Path(getattr(config, "SCENE_JSON",
                                   config.ROOT / "state" / "scene.json")))
    if not isinstance(data, dict):
        return None
    taken = data.get("taken_at")
    if not isinstance(taken, (int, float)):
        return None
    age = time.time() - taken
    if age < -5 or age > getattr(config, "SCENE_MAX_AGE_S", 30.0):
        return None
    objects = [o for o in (data.get("objects") or []) if o.get("label")]
    if not objects:
        return (f"your camera, {age:.0f} s ago, saw nothing its detector could "
                f"name")
    named = ", ".join(f"a {o['label']} {o.get('side', '')}".rstrip()
                      for o in objects[:4])
    # Labels, not a description: SSD-MobileNet calls a doorway a refrigerator
    # sometimes. Enough to know what is around her, not to describe details.
    return (f"your camera, {age:.0f} s ago, detected {named} (detector labels "
            f"can be wrong; look before describing details)")


def _studied() -> str | None:
    """Something she looked up while idle and has not mentioned yet.

    File 08 step 8: this is how "I read something about X earlier" happens --
    it is in front of her for exactly one turn (idle_study.mark_offered).
    """
    try:
        import idle_study
        note = idle_study.pending()
    except Exception:                           # noqa: BLE001
        return None
    if not note:
        return None
    topic = f" about {note['topic']}" if note.get("topic") else ""
    return (f"while you were idle {_said_age(note['age_s'])} you looked "
            f"something up{topic} and learned: {note['fact']} — worth "
            f"mentioning once if it fits, in your own words")


def block(eyes: bool = True) -> str:
    """One short paragraph of what Friday already knows, or "".

    Never raises. Kept SHORT on purpose: this goes on every single turn, so a
    long block is paid for on every single turn, and a model handed six lines
    of context it did not need learns to skim the section.

    ⚠ eyes=False LEAVES OUT WHAT THE CAMERA SAW, and the fast path passes it.
    MEASURED LIVE 2026-09-17 23:09: the detector had named a "cat" (it was a
    suitcase), the line went to the fast path -- which has no camera and no
    tools -- and she said out loud "Let me look at what's there right now.
    There's a cat ahead, about a metre away, looks calm. I'm stopping here so I
    don't..." She had not looked, there was no cat, and the distance and the
    stopping were invented. A brain that cannot look must not be handed
    something that reads like what it can see.
    """
    try:
        lines = [f"It is {_clock()}."]
        parts = (_mood(), _last_spoke(), _exploring(), _around_me(),
                 _scene() if eyes else None, _weather(), _studied())
        for fact in parts:
            if fact:
                lines.append(fact[0].upper() + fact[1:] + ".")
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not read the situation, answering without it: %s",
                    exc)
        return ""
    if len(lines) == 1 and not getattr(config, "SITUATION_ALWAYS_CLOCK", True):
        return ""
    return ("Right now, without having to look anything up:\n"
            + "\n".join(f"- {line}" for line in lines))


def main() -> int:
    """See exactly what the brain is being handed this second."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    text = block()
    print(text or "(nothing fresh enough to say)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
