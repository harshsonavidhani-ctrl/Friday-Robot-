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
    """One recognised command."""
    name: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    say: str = ""


# Matched in order, so the specific ones come before the general ones. Each
# pattern is searched, not fullmatch'd, because whisper adds punctuation and
# the wake word is often still on the front of the transcript.
_RULES: list[tuple[str, str, str | None, dict, str]] = [
    # name        pattern                                    tool          args                                    say
    ("stop",      r"\b(stop|halt|ruk|ruko|रुक|रुको|थम)\b",   "stop",       {},                                     "Stopping."),
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


def understand(text: str) -> Intent | None:
    """The command in this text, or None if it is not one of the few we know."""
    if not text or not text.strip():
        return None
    heard = text.strip().lower()

    for name, pattern, tool, args, say in _COMPILED:
        if not pattern.search(heard):
            continue
        if name == "time":
            return Intent(name="time", say=_time_phrase())
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
