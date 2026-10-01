"""The mission's ears (File 21 step 11): a question asked in friday-mcp,
answered from friday-brain's microphone.

The mission runs inside friday-mcp; the microphone has ONE owner, main.py in
friday-brain (CLAUDE.md PART 5: "nothing under mcp_server/ opens it"). So a
question crosses the two processes like this:

    friday-mcp (the mission)                      friday-brain (main.py)
    ------------------------                      ----------------------
    she says the question (nav_voice, tts.speak)
    Asker.listen_once(10 s)
      writes asking.json {id, kind, listen_s} --> MissionWatch (a wake source)
      waits                                         sees a NEW id -> one trigger
                                                  _mission_turn: waits for her
                                                    voice to end, captures ONE
                                                    utterance (no wake word),
                                                    transcribes it
      Asker.answer(id, text) <------------------- MCP tool mission_answer
      removes asking.json, returns the words

Harsh, 2026-09-29: the wake word stays off while she drives; she listens
without it only while HELD for a question. MissionWatch is therefore never
hushed by the run (its sleep() is a no-op): the run's hush is exactly when
the mission asks. It does honour pause(), so a question never lands in the
middle of one of his turns -- it waits, and lapses with its own deadline.

can_listen() is True only while MissionWatch's heartbeat is fresh: with
friday-brain down, a question would be asked to nobody, so it is not asked.

THE CONVERSATION CLOCK lives here too: `mark_heard()` is what main.py calls at
the end of every turn, and adapters.TalkAdapter reads it for curiosity's "no
conversation for 5 min" (step 10).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("friday.exploration_mission.voice")

ASK_NAME = "asking.json"
ALIVE_NAME = "watch_alive.json"
HEARD_NAME = "heard.json"
HOLD_NAME = "hold.json"
ALIVE_EVERY_S = 5.0
ALIVE_MAX_AGE_S = 30.0
# Her question is still being spoken when listen_once starts, and whisper on
# this Pi takes 3-17 s to transcribe what he says -- the answer can arrive
# well after listen_s. The run stays held the whole time.
ANSWER_GRACE_S = 45.0
TRIGGER_PREFIX = "mission: "


def state_dir() -> Path:
    import config
    return Path(config.ROOT) / "state" / "exploration_mission"


def _read(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


# --- friday-brain's side --------------------------------------------------------

def mark_heard(at: float | None = None, folder: Path | None = None) -> None:
    """A turn just ended (main.py). Never raises."""
    try:
        _write((folder or state_dir()) / HEARD_NAME,
               {"at": time.time() if at is None else float(at)})
    except OSError as exc:
        log.debug("heard mark not written: %s", exc)


def held(folder: Path | None = None, now: float | None = None) -> bool:
    """Is the mission holding the run still (a question, a pause)? The run's
    own hold file, read the way rl/way_back.py reads it: fresh = held."""
    hold = _read((folder or state_dir()) / HOLD_NAME)
    until = hold.get("until")
    return isinstance(until, (int, float)) and until > (time.time() if now is None
                                                        else now)


def parse_trigger(trigger: str) -> dict | None:
    """'mission: <id> <kind> <listen_s>' -> {"id", "kind", "listen_s"}."""
    if not trigger.startswith(TRIGGER_PREFIX):
        return None
    parts = trigger[len(TRIGGER_PREFIX):].split()
    if len(parts) != 3:
        return None
    try:
        return {"id": parts[0], "kind": parts[1], "listen_s": float(parts[2])}
    except ValueError:
        return None


class MissionWatch:
    """A wake source (duck-typed like reminders.ReminderWatch): one trigger
    per NEW question the mission asks."""

    name = "mission"
    stream = None

    def __init__(self, folder: Path | None = None, poll_s: float = 0.25):
        self.folder = folder
        self.poll_s = poll_s
        self._closed = threading.Event()
        self._paused = False
        self._seen: str | None = None
        self._t_alive = 0.0

    def _dir(self) -> Path:
        return self.folder or state_dir()

    def beat(self) -> None:
        try:
            _write(self._dir() / ALIVE_NAME, {"at": time.time(), "pid": os.getpid()})
        except OSError as exc:
            log.debug("watch heartbeat not written: %s", exc)

    def step(self) -> str | None:
        """One poll: the trigger to yield, or None."""
        now = time.time()
        if now - self._t_alive >= ALIVE_EVERY_S:
            self._t_alive = now
            self.beat()
        if self._paused:
            return None
        ask = _read(self._dir() / ASK_NAME)
        ask_id = str(ask.get("id") or "")
        if not ask_id or ask_id == self._seen:
            return None
        if float(ask.get("until") or 0.0) <= now:
            self._seen = ask_id                 # lapsed while he was talking
            return None
        self._seen = ask_id
        kind = "".join(c for c in str(ask.get("kind") or "question") if c.isalnum())
        return f"{ask_id} {kind or 'question'} {float(ask.get('listen_s') or 10.0):g}"

    # -- WakeSource -------------------------------------------------------------
    def listen(self):
        log.info("mission watch ready (%s)", self._dir() / ASK_NAME)
        self.beat()
        while not self._closed.wait(self.poll_s):
            try:
                trig = self.step()
            except Exception as exc:                    # noqa: BLE001
                log.warning("mission watch poll failed: %s", exc)
                trig = None
            if trig:
                yield trig

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self, seconds: float) -> float:
        return 0.0          # never hushed: the run's hush is when she asks

    def wake_up(self) -> None:
        pass

    def sleeping_for(self) -> float:
        return 0.0

    def close(self) -> None:
        self._closed.set()
        try:
            (self._dir() / ALIVE_NAME).unlink()
        except OSError:
            pass


# --- friday-mcp's side ----------------------------------------------------------

class Asker:
    """Asks through MissionWatch and waits for mission_answer. One question
    at a time (the mission never asks two at once)."""

    def __init__(self, folder: Path | None = None, grace_s: float = ANSWER_GRACE_S,
                 wall=time.time):
        self.folder = folder
        self.grace_s = grace_s
        self.wall = wall
        self._lock = threading.Lock()
        self._waiting: dict[str, queue.Queue] = {}

    def _dir(self) -> Path:
        return self.folder or state_dir()

    def can_listen(self) -> bool:
        alive = _read(self._dir() / ALIVE_NAME).get("at")
        return (isinstance(alive, (int, float))
                and 0.0 <= self.wall() - float(alive) <= ALIVE_MAX_AGE_S)

    def listen_once(self, timeout_s: float, kind: str = "question") -> str | None:
        ask_id = uuid.uuid4().hex[:12]
        answers: queue.Queue = queue.Queue()
        with self._lock:
            self._waiting[ask_id] = answers
        path = self._dir() / ASK_NAME
        try:
            _write(path, {"id": ask_id, "kind": kind, "listen_s": float(timeout_s),
                          "until": self.wall() + float(timeout_s) + self.grace_s})
            try:
                text = answers.get(timeout=float(timeout_s) + self.grace_s)
            except queue.Empty:
                log.info("mission question %s: no answer came back", ask_id)
                text = None
        finally:
            with self._lock:
                self._waiting.pop(ask_id, None)
            if _read(path).get("id") == ask_id:
                try:
                    path.unlink()
                except OSError:
                    pass
        text = (text or "").strip()
        return text or None

    def answer(self, ask_id: str, text: str | None) -> bool:
        """The words for question `ask_id` (from mission_answer). False when
        nobody is waiting for that question any more."""
        with self._lock:
            answers = self._waiting.get(str(ask_id))
        if answers is None:
            return False
        answers.put(text or "")
        return True
