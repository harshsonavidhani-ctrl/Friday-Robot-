"""Non-blocking, event-driven voice feedback for Friday.

Design constraints, in priority order:

  1. **It must never slow down navigation.** The control loop runs at 10 Hz with a
     100 ms budget on a Pi 4B that is already sharing four cores with a LiDAR thread,
     a ToF reflex thread at 30 Hz, and two camera grabbers. Speech synthesis takes
     tens to hundreds of milliseconds and `espeak` blocks until the phrase finishes.
     So `say()` only ever appends to a bounded queue and returns; a daemon thread does
     the synthesis. If the queue is full the utterance is DROPPED, never blocked on --
     a robot that stops driving because it is busy talking is a broken robot.

  2. **It must only say true things.** Every phrase in this file is emitted by the
     exploration supervisor from a state transition it actually observed. There is no
     scripted narration, no "I think I see...", and nothing derived from a model that
     does not exist on this robot. See `friday_exploration.py` for the mapping from
     state to phrase, and the honesty note on cameras there.

  3. **It must not chatter.** A naive announcer would say "obstacle ahead" ten times a
     second. Every event carries a cooldown, and a global minimum gap separates any
     two utterances. High-priority events (recovery, safety stops) may pre-empt the
     queue; routine ones are dropped rather than queued behind stale news.

Backends are probed at construction, in order of preference on a Pi:

    espeak-ng / espeak   apt install espeak-ng      (recommended: tiny, instant)
    pyttsx3              pip install pyttsx3        (wraps espeak on Linux anyway)
    say                  macOS, for laptop testing
    log-only             always available; prints at INFO

The log-only fallback is not a failure mode. It means the event stream is still
correct and visible in the journal, and installing espeak later turns it audible with
no code change.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("friday.voice")

_FRIDAY_TTS: object | None = None
_FRIDAY_TTS_TRIED = False


def _load_friday_tts():
    """Friday's OWN voice (Piper via `friday/tts.py`), or None if unavailable.

    Preferred over espeak because it is the voice she already speaks with
    everywhere else -- the shaped Piper model from config.TTS_VOICE, through the
    USB soundcard, and `tts.speak()` animates the face's mouth while it plays.
    Using espeak here would give the same robot two different voices.

    It also avoids `apt install espeak-ng`, which CLAUDE.md Part 1 forbids: a
    system package install is exactly the class of change that killed this Pi
    once. Nothing here leaves the project folder.

    `tts` lives at the repo root next to config.py and this file is one level
    down in rl/, so the parent has to be on sys.path. Importing it is safe --
    the module is only imports, constants and locks, with no device grabbed at
    import time. Cached because it pulls in config.py.
    """
    global _FRIDAY_TTS, _FRIDAY_TTS_TRIED
    if _FRIDAY_TTS_TRIED:
        return _FRIDAY_TTS
    _FRIDAY_TTS_TRIED = True
    try:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        import tts as friday_tts
        voice_model = Path(friday_tts.config.TTS_VOICE)
        if not voice_model.exists():
            LOGGER.warning("Friday TTS present but the Piper voice is missing (%s); "
                           "falling back.", voice_model)
            _FRIDAY_TTS = None
        else:
            _FRIDAY_TTS = friday_tts
    except Exception as exc:                           # pragma: no cover
        LOGGER.debug("Friday TTS unavailable (%s)", exc)
        _FRIDAY_TTS = None
    return _FRIDAY_TTS

# Priorities. Higher pre-empts lower.
P_ROUTINE = 0        # "I found an unexplored area..."
P_NOTABLE = 1        # route changes, area exhausted
P_URGENT = 2         # stuck, recovery, safety intervention

# SHE IS DEAF WHILE SHE TALKS (2026-10-19, Harsh: "she doesn't listen, Friday
# stop"). The wake word mutes itself while any of her lines plays, ~5 s per
# line here. With only min_gap_s between lines, the 17:55-17:57 run said
# "Emergency stop. Something is directly in front of me." every ~7 s and the
# wake word was muted 125 of 167 s (75%). So after each line she LISTENS for
# listen_s, and a line that would land inside that window is dropped, not
# queued (queued, it would be said late, about a stop that is over). These
# keys say something only once a run and are never dropped by it.
ALWAYS_SAY = frozenset({"boot", "done", "halt", "complete",
                        "way_back", "way_back_done"})


@dataclass(order=True)
class _Utterance:
    # Negated priority first so the PriorityQueue pops the most important item.
    _sort_neg_priority: int
    _sort_seq: int
    text: str = field(compare=False)
    key: str = field(compare=False)


class VoiceAnnouncer:
    """Queued, rate-limited, non-blocking speech.

    Thread-safe: `say()` may be called from the control loop, a sensor thread, or the
    supervisor without further locking.
    """

    def __init__(self, *, enabled: bool = True, backend: str = "auto",
                 min_gap_s: float = 1.2, default_cooldown_s: float = 8.0,
                 max_queue: int = 4, voice: str | None = None,
                 rate_wpm: int = 165, listen_s: float = 0.0) -> None:
        self.enabled = bool(enabled)
        # FILE 21 (2026-10-02): a predicate the run sets to its mission hold.
        # While it is true every line is DROPPED: the mission is asking Harsh a
        # question and listening for the answer, and the run's narration ("I
        # have now mapped 64 square metres") landed in the answer twice. A held
        # run is standing still, so no safety line is lost.
        self.muted = None
        self.dropped_muted = 0
        self.min_gap_s = float(min_gap_s)
        self.listen_s = float(listen_s)
        self.dropped_listen = 0
        self._speaking = False
        self.default_cooldown_s = float(default_cooldown_s)
        self.voice = voice
        self.rate_wpm = int(rate_wpm)

        self._q: queue.PriorityQueue[_Utterance] = queue.PriorityQueue(maxsize=max_queue)
        self._seq = 0
        self._lock = threading.Lock()
        self._last_said: dict[str, float] = {}
        self._last_utterance_t = 0.0
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

        self.spoken = 0
        self.dropped_cooldown = 0
        self.dropped_full = 0

        self.backend = self._pick_backend(backend) if self.enabled else "disabled"
        self._engine = None
        if self.backend == "pyttsx3":
            self._engine = self._init_pyttsx3()
            if self._engine is None:
                self.backend = "log"

        if self.backend == "friday":
            # The Piper model takes ~10 s to load and it loads on FIRST USE, which
            # would otherwise land on the first real announcement -- measured at
            # 20.2 s for utterance one versus 5.5 s for the next. preload() exists
            # for exactly this and returns immediately, warming the voice on its own
            # thread while the robot is still starting its sensors.
            try:
                _load_friday_tts().preload()
            except Exception as exc:                   # never block start-up on voice
                LOGGER.debug("Piper preload skipped (%s)", exc)

        self._thread = threading.Thread(target=self._worker, name="voice", daemon=True)
        self._thread.start()
        LOGGER.info("Voice feedback: backend=%s enabled=%s", self.backend, self.enabled)

    # ------------------------------------------------------------------ backends ---
    @staticmethod
    def _pick_backend(requested: str) -> str:
        if requested != "auto":
            return requested
        # Friday's own Piper voice first: same voice as the rest of the robot, and
        # it needs no system package. Only fall through to espeak if it is missing.
        if _load_friday_tts() is not None:
            return "friday"
        for exe in ("espeak-ng", "espeak"):
            if shutil.which(exe):
                return exe
        if shutil.which("say"):                       # macOS, laptop testing
            return "say"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return "log"

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self.rate_wpm)
            if self.voice:
                eng.setProperty("voice", self.voice)
            return eng
        except Exception as exc:                       # pragma: no cover
            LOGGER.warning("pyttsx3 unavailable (%s) -- falling back to log-only.", exc)
            return None

    # ---------------------------------------------------------------------- API ----
    def _listening(self, key: str) -> bool:
        """True while this line would cut into the window after her last line in
        which she listens (see ALWAYS_SAY). Never for the once-a-run lines."""
        if self.listen_s <= 0.0 or key in ALWAYS_SAY:
            return False
        return (self._speaking
                or time.monotonic() - self._last_utterance_t < self.listen_s)

    def say(self, key: str, text: str, *, priority: int = P_ROUTINE,
            cooldown_s: float | None = None) -> bool:
        """Queue `text` for speech. Returns True if accepted. NEVER blocks.

        `key` identifies the event type and is what the cooldown applies to, so two
        different phrasings of "obstacle ahead" still share one cooldown as long as
        they share a key.
        """
        if not self.enabled or self._stop.is_set():
            return False
        try:
            if self.muted is not None and self.muted():
                self.dropped_muted += 1
                return False
        except Exception:                                   # noqa: BLE001
            pass                    # a broken predicate must never silence her

        if self._listening(key):
            self.dropped_listen += 1
            return False

        now = time.monotonic()
        cd = self.default_cooldown_s if cooldown_s is None else float(cooldown_s)
        with self._lock:
            if now - self._last_said.get(key, -1e9) < cd:
                self.dropped_cooldown += 1
                return False
            self._last_said[key] = now
            self._seq += 1
            item = _Utterance(-int(priority), self._seq, text, key)

        try:
            self._q.put_nowait(item)
        except queue.Full:
            # Full queue means we are already behind. Drop routine news; for an urgent
            # event, evict the oldest item to make room -- "I am stuck" outranks a
            # stale "I found an unexplored area".
            if priority >= P_URGENT:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(item)
                    return True
                except (queue.Empty, queue.Full):
                    pass
            self.dropped_full += 1
            return False
        return True

    def close(self, drain_timeout_s: float = 2.0) -> None:
        """Stop the worker. Waits briefly so a final phrase is not cut mid-word."""
        deadline = time.monotonic() + drain_timeout_s
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
        # A Piper utterance runs ~5.5 s and cannot be cut short the way an espeak
        # subprocess can -- there is no self._proc to terminate, because tts.speak()
        # owns the device itself. Joining for 1 s left the worker mid-playback while
        # the interpreter tore down onnxruntime underneath it, which surfaced as
        # "terminate called without an active exception" and exit code 134 on a run
        # that had otherwise finished cleanly. Wait long enough for the phrase to
        # land, but stay bounded so shutdown can never hang.
        self._thread.join(timeout=12.0 if self.backend == "friday" else 1.0)

    # ------------------------------------------------------------------- worker ----
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            gap = self.min_gap_s - (time.monotonic() - self._last_utterance_t)
            if gap > 0:
                # Sleep in slices so `close()` stays responsive.
                end = time.monotonic() + gap
                while time.monotonic() < end and not self._stop.is_set():
                    time.sleep(min(0.05, end - time.monotonic()))
            if self._stop.is_set():
                break
            # ...and AGAIN as it is spoken, not only as it was queued: lines
            # queued while she explored were still being said up to ~36 s into
            # the mission's question (4 queued x ~9 s each), and one landed in
            # Harsh's answer ("I will investigate it" -> "the one investigating").
            # 2026-10-04 16:31:37-16:32:04.
            try:
                if self.muted is not None and self.muted():
                    self.dropped_muted += 1
                    continue
            except Exception:                               # noqa: BLE001
                pass                # a broken predicate must never silence her
            # Two lines queued in the same tick both passed say(); the second
            # would start the moment the first ends.
            if self._listening(item.key):
                self.dropped_listen += 1
                continue

            LOGGER.info("SAY [%s] %s", item.key, item.text)
            self._speaking = True
            try:
                self._speak(item.text)
            except Exception as exc:                   # never kill the thread
                LOGGER.warning("Speech backend failed (%s); continuing silently.", exc)
            finally:
                self._speaking = False
            self._last_utterance_t = time.monotonic()
            self.spoken += 1

    def _speak(self, text: str) -> None:
        b = self.backend
        if b == "friday":
            # BLOCKING, by design and safely: this runs on the announcer's worker
            # thread, never the control loop. tts.speak() holds its own lock so
            # overlapping utterances serialise instead of fighting for the device.
            friday_tts = _load_friday_tts()
            if friday_tts is None:                     # went away after start-up
                return
            friday_tts.speak(text)
        elif b in ("espeak-ng", "espeak"):
            cmd = [b, "-s", str(self.rate_wpm)]
            if self.voice:
                cmd += ["-v", self.voice]
            cmd.append(text)
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            self._proc.wait(timeout=10)
        elif b == "say":
            self._proc = subprocess.Popen(["say", text], stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            self._proc.wait(timeout=10)
        elif b == "pyttsx3" and self._engine is not None:
            self._engine.say(text)
            self._engine.runAndWait()
        # "log" backend: the LOGGER.info above is the whole output.

    # -------------------------------------------------------------------- report ---
    def report(self) -> None:
        LOGGER.info("voice: backend=%s spoken=%d dropped(cooldown=%d full=%d "
                    "listening=%d)", self.backend, self.spoken,
                    self.dropped_cooldown, self.dropped_full, self.dropped_listen)


def build_announcer(args) -> VoiceAnnouncer:
    """Construct from parsed CLI args, honouring FRIDAY_VOICE=0 in the environment."""
    env_off = os.environ.get("FRIDAY_VOICE", "").strip() in ("0", "off", "false")
    return VoiceAnnouncer(
        enabled=(not getattr(args, "no_voice", False)) and not env_off,
        backend=getattr(args, "voice_backend", "auto"),
        min_gap_s=getattr(args, "voice_min_gap", 1.2),
        rate_wpm=getattr(args, "voice_rate", 165),
        listen_s=getattr(args, "voice_listen_gap", 15.0),
    )
