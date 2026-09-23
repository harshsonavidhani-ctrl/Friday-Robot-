"""nav_voice.py — file 14 step 10: what navigation says out loud.

    voice = nav_voice.Voice()               # the real speaker (Piper)
    voice.say("blocked")                    # "Path blocked. Replanning my route."
    voice.say("recovered", dest="the kitchen")

FIXED LINES, NO MODEL. Harsh, 2026-09-23: "no ai will be used, all behaviour
in offline model". Every sentence navigation speaks is in LINES below, filled
in with nothing but a destination name, and spoken by Piper on this Pi. There
is no brain turn behind any of it, so she says the same thing online, offline,
and with the API key gone.

THE STYLE IS THE EXPLORATION'S, not invented here. `rl/friday_exploration.py`
(read only, never imported) already speaks while it drives: short, declarative,
present tense — "Emergency stop. Something is directly in front of me.",
"I am unable to make progress. Initiating recovery.", "Recovery successful.
Continuing exploration." A robot that sounds like one system while exploring
and another while navigating sounds like two robots, so these match it
sentence for sentence where the situations match.

⚠ IT NEVER BLOCKS THE CALLER. `tts.speak()` returns only once the words are
out, and the caller is an async drive inside the MCP server — the same event
loop that has to answer `stop()`. So lines go onto a queue and one daemon
thread speaks them in order. A drive never waits for a sentence to finish, and
a stop is never queued behind one.

⚠ SOUND GOES THROUGH `tts.speak()` AND NOTHING ELSE (CLAUDE.md PART 4). It
writes `state/speaking.json`, which is how the wake detector knows not to hear
her own voice as a command. A second route to the speaker would reopen that.

⚠ SILENT BY DEFAULT IN THE ENGINE. `NavController` gets `SILENT` unless its
caller hands it a real voice, so the test suites — which build hundreds of
controllers — never make a sound on the robot. The MCP tools pass `VOICE`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

log = logging.getLogger("friday.nav_voice")

# Every sentence navigation says. {dest} is the only substitution.
LINES: dict[str, str] = {
    # Setting off, and the reasons not to.
    "setting_off":     "Heading to {dest}.",
    "busy":            "I am already on my way to {dest}.",
    "lost":            "I cannot work out where I am on my map, so I am not "
                       "moving.",
    # While driving. Blocked and stuck are different problems and must never
    # share a sentence: one is something in the way, the other is her wheels.
    "blocked":         "Path blocked. Replanning my route.",
    "clear":           "Clear. Continuing to {dest}.",
    "stuck":           "I am unable to make progress. Initiating recovery.",
    "recovered":       "Recovery successful. Continuing to {dest}.",
    "recovery_failed": "Recovery did not free me. Planning a different route.",
    # How it ended.
    "arrived":         "I have arrived at {dest}.",
    "arrived_home":    "I am home.",
    "arrived_start":   "I am back where I started.",
    "no_route":        "I cannot find a route to {dest}. Stopping here.",
    "failsafe":        "I have stopped and I need help. Say try again when I "
                       "am clear.",
    # Exploration, and coming back from it.
    "frontiers_done":  "Nothing more to explore here. Heading back early.",
    "resuming":        "Resuming exploration.",
    # The way out of FAILSAFE (step 17).
    "try_again":       "Trying again. Finding myself on my map first.",
    "nothing_waiting": "There is nothing waiting to try again.",
}

# Per-line minimum gap. A block that clears and returns three times in ten
# seconds is one announcement, not three; the exploration's announcer throttles
# the same lines the same way. Lines not listed are never throttled — an
# arrival is said every time it happens.
COOLDOWN_S: dict[str, float] = {
    "blocked": 8.0, "clear": 8.0, "stuck": 8.0, "recovered": 4.0,
    "recovery_failed": 8.0,
}


def line(key: str, dest: str = "") -> str:
    """The sentence for `key`, with the destination filled in."""
    text = LINES[key]
    return text.format(dest=dest or "my destination")


def arrival_key(dest: str) -> str:
    """Which arrival sentence fits. "I have arrived at home" is not English."""
    if dest == "home":
        return "arrived_home"
    if dest == "where I set off from":
        return "arrived_start"
    return "arrived"


def _tts_speak(text: str) -> None:
    import tts                                   # the real speaker, lazily
    tts.speak(text)


class Voice:
    """Say fixed navigation lines, throttled, without blocking the caller.

    `speak` is the function that makes the sound — `tts.speak` for real, a list
    append in a test. `background=False` speaks inline, which tests use so the
    order of `said` is the order of events.
    """

    def __init__(self, speak: Callable[[str], None] | None = None,
                 background: bool = True) -> None:
        self._speak = speak or _tts_speak
        self._background = background
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self.said: list[str] = []            # everything actually said, in order

    def say(self, key: str, dest: str = "", now: float | None = None) -> str:
        """Say the line for `key`. Returns the text, or "" if throttled."""
        now = time.monotonic() if now is None else now
        with self._lock:
            gap = COOLDOWN_S.get(key, 0.0)
            if gap and now - self._last.get(key, -1e9) < gap:
                return ""
            self._last[key] = now
            text = line(key, dest)
            self.said.append(text)
        log.info("nav says: %s", text)
        if self._background:
            self._ensure_worker()
            self._queue.put(text)
        else:
            self._say_now(text)
        return text

    def say_text(self, text: str) -> str:
        """Say a sentence built by code rather than taken from LINES.

        For the few answers whose content is data -- "I do not know anywhere
        called the garage. I know: the kitchen, the desk." -- which no fixed
        line can hold. Still no model: the caller composes it from strings.
        Never throttled; it is only used for direct answers to a command.
        """
        text = (text or "").strip()
        if not text:
            return ""
        with self._lock:
            self.said.append(text)
        log.info("nav says: %s", text)
        if self._background:
            self._ensure_worker()
            self._queue.put(text)
        else:
            self._say_now(text)
        return text

    def _say_now(self, text: str) -> None:
        try:
            self._speak(text)
        except Exception as exc:                 # noqa: BLE001
            # Losing her voice must never end a drive. The line is still in
            # `said` and in the log.
            log.warning("could not speak %r: %s", text, exc)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="nav-voice")
            self._worker.start()

    def _run(self) -> None:
        while True:
            self._say_now(self._queue.get())


class _Silent(Voice):
    """Records what it would have said, and says nothing."""

    def __init__(self) -> None:
        super().__init__(speak=lambda _text: None, background=False)

    def say(self, key: str, dest: str = "", now: float | None = None) -> str:
        text = super().say(key, dest, now)
        # A shared silent voice must not grow forever across a long test run.
        del self.said[:-50]
        return text

    def say_text(self, text: str) -> str:
        out = super().say_text(text)
        del self.said[:-50]
        return out


# The engine's default: says nothing. See the module docstring.
SILENT = _Silent()

# The one real voice, shared by every navigation tool in the MCP server so
# that the cooldowns apply across journeys, not per controller.
VOICE = Voice()
