"""offline.py — the handful of things Friday can still do with no internet.

The brain lives behind the Anthropic API, so no network means no thinking. That
must not mean a robot that sits there saying nothing: the commands that matter
most in the room — drive, stop, turn, what time is it — are all things a dozen
lines of matching can handle without a model.

    intent = offline.understand("aage jao")
    if intent:                              # matched a known command
        intent.tool, intent.args            # -> "move", {...}
        intent.say                          # what to say while doing it

WHAT THIS IS NOT. It is not a fallback brain and it must never pretend to be
one. If it does not recognise something it returns None, main.py says so
plainly, and Friday keeps listening. Guessing would be worse than admitting the
network is down.

STOP IS ALWAYS CHECKED FIRST, online or off. If Harsh says stop, no round trip
to a model should stand between him and the motors halting.

HINDI AND ENGLISH BOTH. Harsh mixes them mid-sentence, so both are matched
against the same intent — "aage jao" and "go forward" are the same command.
"""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field

import config

log = logging.getLogger("friday.offline")


@dataclass
class Intent:
    """One recognised command.

    `steps` carries a SEQUENCE of tool calls, because a directional move is
    never one call: the MCP server refuses move() unless observe() ran within
    the last few seconds, and moving ages that look -- so driving somewhere is
    observe, move, and for "go left" a pivot in between. That rule is enforced
    in the server, not here, and it is the reason Friday cannot be talked into
    driving blind. `tool`/`args` remain for the single-call commands.
    """
    name: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    say: str = ""
    steps: list = field(default_factory=list)

    def plan(self) -> list:
        """Every tool call this intent makes, in order."""
        if self.steps:
            return list(self.steps)
        return [(self.tool, dict(self.args))] if self.tool else []


# Matched in order, so the specific ones come before the general ones. Each
# pattern is searched, not fullmatch'd, because whisper adds punctuation and
# the wake word is often still on the front of the transcript.
_RULES: list[tuple[str, str, str | None, dict, str]] = [
    # name        pattern                                    tool          args                                    say
    ("stop",      r"\b(stop|halt|ruk|ruko|रुक|रुको|थम)\b",   "stop",       {},                                     "Stopping."),
    # English "turn around" only. Romanised "peeche mudo" still resolves to
    # backward -- the go/turn verb matching above the table wins over this row
    # -- and reversing when asked to turn around is a mild wrong answer, not a
    # dangerous one, so it is left rather than fixed with a special case.
    ("around",    r"\b(turn around|turn round|about turn|spin around|ghoom|peeche mud\w*|पीछे मुड़|घूम)\b", "turn_degrees", {"degrees": 180},                      "Turning around."),
    ("backward",  r"\b(back|backward|reverse|peeche|पीछे)\b", "move",      {"direction": "backward",
                                                                            "duration_seconds": 1.0},              "Going back."),
    ("forward",   r"\b(forward|ahead|aage|आगे|चल)\b",         "move",      {"direction": "forward",
                                                                            "duration_seconds": 1.0},              "Going forward."),
    ("left",      r"\b(left|baaye|baayen|बाय|बाएं|बायें)\b",  "turn_degrees", {"degrees": -90},                    "Turning left."),
    ("right",     r"\b(right|daaye|daayen|दाय|दाएं|दायें)\b", "turn_degrees", {"degrees": 90},                     "Turning right."),
    ("time",      r"\b(time|clock|samay|baje|समय|बजे)\b",     None,        {},                                     ""),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE | re.UNICODE),
              tool, args, say)
             for name, pattern, tool, args, say in _RULES]


# --- directions, and how far ------------------------------------------------
# Separate from _RULES because a direction is not a fixed command: it carries
# an amount ("forward two seconds", "back a bit") and, for left and right, a
# choice between pivoting on the spot and actually going that way.
_DIRECTIONS = [
    ("forward",  r"\b(forward|ahead|straight|aage|आगे)\b"),
    ("backward", r"\b(back|backward|backwards|reverse|peeche|पीछे)\b"),
    ("left",     r"\b(left|baaye|baayen|बाय|बाएं|बायें)\b"),
    ("right",    r"\b(right|daaye|daayen|दाय|दाएं|दायें)\b"),
]
_DIRECTION_RE = [(n, re.compile(p, re.IGNORECASE | re.UNICODE))
                 for n, p in _DIRECTIONS]

# "go left" means travel that way; "turn left" means pivot and stop. Friday is
# differential drive, so travelling left is a pivot THEN a drive -- she cannot
# strafe, and pretending otherwise would just make her ignore half the words.
_GO_RE = re.compile(r"\b(go|move|drive|walk|come|chal|chalo|jao|चल|चलो|जाओ)\b",
                    re.IGNORECASE | re.UNICODE)
_TURN_RE = re.compile(r"\b(turn|rotate|spin|face|pivot|mudo|मुड़|मुडो)\b",
                      re.IGNORECASE | re.UNICODE)

# ORDER MATTERS: the specific words are tried first, and the bare articles
# last. With "a" checked first, "half a meter" matched the "a" and became a
# whole metre -- twice the distance asked for, which on a robot is the kind of
# rounding that ends in a wall.
_WORD_NUMBERS = [("half", 0.5), ("quarter", 0.25), ("one", 1), ("two", 2),
                 ("three", 3), ("four", 4), ("five", 5), ("six", 6),
                 ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
                 ("a", 1), ("an", 1)]


def _amount(text: str) -> tuple[float | None, str]:
    """How far, and in what unit. (None, "") when nothing was said."""
    number = None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        number = float(match.group(1))
    else:
        for word, value in _WORD_NUMBERS:
            if re.search(rf"\b{word}\b", text):
                number = value
                break
    if re.search(r"\b(cm|centimet|inch)\b", text):
        return (number or 30.0), "cm"
    if re.search(r"\b(met(er|re)s?|m)\b", text):
        return (number or 1.0) * 100.0, "cm"
    if re.search(r"\b(degrees?|deg)\b", text) and number:
        return number, "deg"
    if re.search(r"\b(seconds?|sec|s)\b", text) and number:
        return number, "s"
    if re.search(r"\b(a little|a bit|slightly|thoda|थोड़ा)\b", text):
        return 0.6, "s"
    if re.search(r"\b(a lot|far|long way|bahut|बहुत)\b", text):
        return 3.0, "s"
    return (number, "s") if number else (None, "")


def _direction_intent(heard: str) -> Intent | None:
    """A movement command, as the sequence of tool calls that carries it out."""
    found = next((n for n, r in _DIRECTION_RE if r.search(heard)), None)
    if found is None:
        return None
    going = bool(_GO_RE.search(heard))
    turning = bool(_TURN_RE.search(heard))
    size, unit = _amount(heard)

    # Bare "left" with no verb keeps the old meaning: pivot and stop.
    if found in ("left", "right") and not going:
        degrees = size if unit == "deg" and size else 90.0
        degrees = -degrees if found == "left" else degrees
        return Intent(name=found, tool="turn_degrees",
                      args={"degrees": degrees},
                      say=f"Turning {found}.")

    steps: list = []
    say = ""
    if found in ("left", "right"):
        # Pivot first, then drive: travelling left IS turning and going.
        degrees = -90.0 if found == "left" else 90.0
        steps.append(("turn_degrees", {"degrees": degrees}))
        say = f"Going {found}."
        drive = "forward"
    else:
        drive = found
        say = "Going forward." if found == "forward" else "Going back."

    # observe() before every move, because the server requires it and moving
    # ages it. This is the safety interrupt's precondition, not a formality.
    steps.append(("observe", {}))
    if unit == "cm":
        distance = size if drive == "forward" else -abs(size)
        steps.append(("move_distance", {"cm": distance}))
    else:
        seconds = size if unit == "s" and size else 1.0
        seconds = max(0.2, min(seconds, 5.0))     # MAX_MOVE_SECONDS in config
        steps.append(("move", {"direction": drive,
                               "duration_seconds": seconds}))
    return Intent(name=f"go_{found}", say=say, steps=steps)


def understand(text: str) -> Intent | None:
    """The command in this text, or None if it is not one of the few we know."""
    if not text or not text.strip():
        return None
    heard = text.strip().lower()

    # Stop always wins, and is checked before anything that could move.
    if _COMPILED[0][1].search(heard):
        return Intent(name="stop", tool="stop", args={}, say="Stopping.")

    for name, pattern, tool, args, say in _COMPILED:
        if name in ("stop", "time") and pattern.search(heard):
            if name == "time":
                return Intent(name="time", say=_time_phrase())
            return Intent(name=name, tool=tool, args=dict(args), say=say)

    moving = _direction_intent(heard)
    if moving is not None:
        return moving

    for name, pattern, tool, args, say in _COMPILED:
        if pattern.search(heard):
            return Intent(name=name, tool=tool, args=dict(args), say=say)
    return None


def is_stop(text: str) -> bool:
    """Just the stop check, for the always-on path."""
    intent = understand(text)
    return intent is not None and intent.name == "stop"


def _time_phrase() -> str:
    now = time.localtime()
    hour = now.tm_hour % 12 or 12
    suffix = "in the morning" if now.tm_hour < 12 else (
        "in the afternoon" if now.tm_hour < 18 else "in the evening")
    if now.tm_min == 0:
        return f"It's {hour} o'clock {suffix}."
    return f"It's {hour}:{now.tm_min:02d} {suffix}."


# --- is there internet? --------------------------------------------------

_last_check = 0.0
_last_result = True
_lock = threading.Lock()


def online(force: bool = False) -> bool:
    """Can Friday reach the API? Cached, because this is asked every turn.

    A TCP connect to the API host, not a ping and not an HTTP request: ICMP is
    frequently filtered and an HTTP request costs a round trip and possibly
    money. Failing closed (assuming offline) would be worse than the occasional
    wasted attempt, so anything unexpected counts as online and the real call
    is allowed to produce the real error.
    """
    global _last_check, _last_result
    with _lock:
        now = time.monotonic()
        if not force and now - _last_check < config.ONLINE_CHECK_TTL_S:
            return _last_result
        _last_check = now
        try:
            with socket.create_connection((config.ONLINE_CHECK_HOST,
                                           config.ONLINE_CHECK_PORT),
                                          timeout=config.ONLINE_CHECK_TIMEOUT_S):
                _last_result = True
        except OSError:
            _last_result = False
        except Exception:                       # noqa: BLE001
            _last_result = True
        return _last_result


def main() -> int:
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        intent = understand(text)
        print(f"{text!r} -> {intent}")
        return 0
    print(f"online: {online(force=True)}")
    for probe in ("aage jao", "stop", "turn left", "what time is it",
                  "tell me about the moon"):
        print(f"  {probe!r:28} -> {understand(probe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
