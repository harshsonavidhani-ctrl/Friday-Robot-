"""Friday plays with the laptop on her own, and misses it when it is off (2026-10-20).

Harsh: "she is curious to search anything by herself, she opens the laptop
like anything she wants to play ... its her toy ... sometimes when her laptop
is off she says i miss my laptop, turn on the laptop". Offline: no model and
no API credit -- the choices are made here, from a list of things she is
curious about, what she has seen, and the games the laptop listed.

THE GATES, all of them, every time (config.LAPTOP_PLAY_* / LAPTOP_MISS_*):
  * he is around -- he spoke within LAPTOP_PLAY_PRESENCE_S, or a person was
    in front of her camera that recently;
  * inside LAPTOP_PLAY_HOURS (never at night);
  * PLAY: the laptop is on AND nobody has touched its keyboard or mouse for
    LAPTOP_PLAY_IDLE_S (helper 5's /idle -- with an older helper she cannot
    tell, so she never plays); at most once per LAPTOP_PLAY_GAP_S;
  * MISS: the laptop has not said hello for LAPTOP_MISS_AFTER_S; at most once
    per LAPTOP_MISS_GAP_S;
  * and reminders.ReminderWatch asks initiative.decide() before offering it, so
    not while he talks, she speaks, in quiet mode or mid-turn.

It only SPEAKS through ReminderWatch (between turns, never over one) and the
action runs after she has said what she is doing, on its own thread.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("friday.laptop_play")

STATE_FILE = Path(getattr(config, "ROOT", Path("."))) / "state" / "laptop_play.json"
HELLO_FRESH_S = 120.0          # the helper says hello every 30 s

CURIOUS = [
    "how do robots see", "how does lidar work", "mars rover videos",
    "cute baby animals", "how are rainbows made", "deep sea creatures",
    "how do batteries work", "the biggest robots in the world",
    "funny cat videos", "how do planes fly", "satisfying videos",
    "how is chocolate made", "black holes explained", "origami robot",
    "how do magnets work", "amazing science experiments",
]
MISS_LINES = [
    "I miss my laptop. Can you turn it on?",
    "My laptop has been asleep for ages. Can you turn it on? I want to play.",
    "Is my laptop off? I miss it. Turn it on when you can.",
]


def _state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save(st: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_name(f".{STATE_FILE.name}.tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(STATE_FILE)
    except OSError as exc:
        log.warning("could not save laptop play state: %s", exc)


def mark(kind: str, now: float | None = None) -> None:
    st = _state()
    st[f"last_{kind}"] = time.time() if now is None else now
    _save(st)


def _in_hours(now: float) -> bool:
    start, end = getattr(config, "LAPTOP_PLAY_HOURS", (9, 22))
    return start <= time.localtime(now).tm_hour < end


def _person_seen_within(seconds: float, now: float) -> bool:
    try:
        scene = json.loads(Path(config.ROOT / "state" / "scene.json").read_text())
    except (OSError, ValueError, AttributeError):
        return False
    at = float(scene.get("taken_at") or scene.get("at") or 0.0)
    labels = " ".join(str(d.get("label", "") if isinstance(d, dict) else d)
                      for d in scene.get("objects") or [])
    return now - at <= seconds and "person" in labels


def _present(now: float, last_heard: float) -> bool:
    window = float(getattr(config, "LAPTOP_PLAY_PRESENCE_S", 1200.0))
    return (last_heard and now - last_heard <= window) or _person_seen_within(window, now)


def laptop_on(now: float) -> bool:
    import laptop_research as lr
    seen = lr.seen() or {}
    return bool(seen.get("at")) and now - float(seen["at"]) <= HELLO_FRESH_S


def _seen_things(now: float) -> list[str]:
    """Things she saw recently on her camera (the scene watcher), for curiosity."""
    try:
        scene = json.loads(Path(config.ROOT / "state" / "scene.json").read_text())
    except (OSError, ValueError, AttributeError):
        return []
    if now - float(scene.get("taken_at") or scene.get("at") or 0.0) > 3600:
        return []
    out = []
    for d in scene.get("objects") or []:
        label = str(d.get("label", "") if isinstance(d, dict) else d).strip()
        if label and label != "person" and label not in out:
            out.append(label)
    return out


def choose(now: float, rng: random.Random = random) -> dict:
    """What she does with the laptop this time: an item for ReminderWatch."""
    import laptop_research as lr
    games = [a["name"] for a in lr.known_apps() if lr.is_game(a)]
    roll = rng.random()
    if games and roll < 0.30:
        pick = rng.choice(games)
        return {"kind": "laptop_play", "action": {"do": "launch", "name": pick},
                "say": f"I'm bored, so I'm opening {pick} on the laptop. Want to play with me?"}
    if roll < 0.45:
        return {"kind": "laptop_play",
                "action": {"do": "search", "query": config.LAPTOP_DEFAULT_MUSIC,
                           "site": "youtube", "play": True},
                "say": "I feel like some music. I'm putting a song on the laptop."}
    seen = _seen_things(now)
    if seen and rng.random() < 0.5:
        topic = f"fun facts about {rng.choice(seen)}s"
    else:
        topic = rng.choice(CURIOUS)
    return {"kind": "laptop_play",
            "action": {"do": "search", "query": topic, "site": "youtube", "play": False},
            "say": f"I'm curious about {topic}. I'm looking it up on the laptop!"}


def next_item(now: float, last_heard: float, rng: random.Random = random) -> dict | None:
    """The laptop thing she would say now, or None. Never raises."""
    play_on = bool(getattr(config, "LAPTOP_PLAY", False))
    miss_on = bool(getattr(config, "LAPTOP_MISS", False))
    if not (play_on or miss_on):
        return None
    try:
        import laptop_research as lr
        if not lr.configured():
            return None
        st = _state()
        on = laptop_on(now)
        if on or "last_seen_on" not in st:
            st["last_seen_on"] = now          # first run counts as "just seen"
            _save(st)
        if not _in_hours(now) or not _present(now, last_heard):
            return None
        if on and play_on and now - float(st.get("last_play", 0.0)) >= float(
                config.LAPTOP_PLAY_GAP_S):
            info = lr.idle()                  # None with an old helper: no play
            if info and float(info.get("idle_s") or 0.0) >= float(config.LAPTOP_PLAY_IDLE_S):
                return choose(now, rng)
            return None
        if (not on and miss_on
                and now - float(st.get("last_seen_on", now)) >= float(config.LAPTOP_MISS_AFTER_S)
                and now - float(st.get("last_miss", 0.0)) >= float(config.LAPTOP_MISS_GAP_S)):
            return {"kind": "laptop_miss", "say": rng.choice(MISS_LINES)}
    except Exception as exc:                    # noqa: BLE001
        log.debug("laptop play skipped: %s: %s", type(exc).__name__, exc)
    return None


def perform(item: dict) -> None:
    """After she has said it: do it, on its own thread (the laptop can take
    seconds to answer, and this is called from the wake loop)."""
    kind = item.get("kind")
    mark("play" if kind == "laptop_play" else "miss")
    action = item.get("action")
    if kind != "laptop_play" or not action:
        return

    def work() -> None:
        import laptop_research as lr
        import rate_limit
        try:
            if action.get("do") == "launch":
                ok, why = rate_limit.allow("laptop_launch")
                if ok:
                    rate_limit.used("laptop_launch")
                    done, why = lr.launch(str(action["name"]))
            else:
                ok, why = rate_limit.allow("laptop_open")
                if ok:
                    rate_limit.used("laptop_open")
                    done, why = lr.open_search(str(action["query"]), action.get("site", "web"),
                                               bool(action.get("play")))
            log.info("laptop play %s: %s", action, why or "done")
        except Exception as exc:                # noqa: BLE001
            log.warning("laptop play failed: %s: %s", type(exc).__name__, exc)

    threading.Thread(target=work, name="laptop-play", daemon=True).start()
