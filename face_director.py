"""face_director.py — when Friday shows what, and who gets to decide.

face_control.py writes an emotion. friday_face.py draws it and tweens between
them. Neither decides WHEN, and that is what this is: the arbiter that turns a
pile of "set the face to X" calls from different parts of the system into one
face that behaves like a person's.

    director = face_director.get()
    director.pipeline("listening")     # layer 1 — what Friday is DOING
    director.emotion("happy")          # layer 2 — what the brain FEELS
    director.safety("scared")          # layer 1 — overrides everything
    director.rest()                    # nothing to show; let it settle

TWO LAYERS, AND A STRICT PRIORITY BETWEEN THEM.

  SAFETY   scared, dizzy          — an obstacle or a failure. Instant, and it
                                    wins mid-sentence. If Friday is about to
                                    hit something, a cheerful face is a lie.
  PIPELINE listening, thinking,   — what Friday is doing right now. Truthful
           speaking, success,       feedback, and the thing Harsh reads to
           confused                 know he has been heard.
  EMOTION  happy, love, proud,    — what the brain chose from what was said.
           excited, curious,        The colour, not the signal.
           surprised, sad, wink,
           angry
  IDLE     neutral, bored,        — nothing is happening. Drifts downward the
           sleepy, sleeping         longer that stays true.

A request never interrupts a higher layer. That is the whole design: the LLM
cannot cheerfully smile over an obstacle warning, because emotion() simply does
not outrank safety().

DEBOUNCE. Every expression is held for at least MIN_DWELL_S before anything of
the same or lower priority may replace it. Without that, a fast turn --
listening, thinking, happy, speaking, all inside two seconds -- reads as a
flicker rather than a sequence. Safety is exempt: it is never made to wait.

DECAY. An emotion is a moment, not a state. Each one carries a hold time, and
when it expires the face eases back to neutral rather than staying stuck on the
last thing that happened. Keep going long enough with nothing at all and it
drifts through bored to sleepy to sleeping, which is what makes an idle robot
look alive rather than switched off.

THE FACE IS DECORATION. Every call here is best-effort and swallows its errors:
a face process that is down, or a bad emotion name, must never take down a
conversation.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import config

log = logging.getLogger("friday.face")

# Priority bands. Higher wins.
#
# ⚠ EMOTION SITS ABOVE PIPELINE, WHICH LOOKS WRONG UNTIL YOU WATCH IT. With
# emotion below, a felt reaction could never appear: "thinking" or "speaking"
# is almost always up when the brain picks one, so happy and love were silently
# swallowed and the face just ticked through its states like a status light.
# The natural chain Harsh asked for -- listening → thinking → slight happy →
# speaking -- needs the happy beat to land ON TOP of the pipeline state.
#
# Pipeline transitions still win, because they are pushed with force=True: a
# genuine change of what Friday is DOING always shows, so an emotion is a beat
# between states rather than something that blocks them. Only safety outranks
# everything, and nothing is ever forced past it.
SAFETY = 100
EMOTION = 60
PIPELINE = 50
IDLE = 10

# Which names belong to which band, so a caller cannot smuggle "scared" in
# through emotion() and get safety priority for it.
SAFETY_FACES = {"scared", "dizzy"}
PIPELINE_FACES = {"listening", "thinking", "speaking", "success", "confused",
                  "sleepy", "sleeping"}
EMOTION_FACES = {"happy", "love", "proud", "excited", "curious", "surprised",
                 "sad", "wink", "angry", "neutral", "bored"}


def _emotion_stamp() -> tuple[int, int] | None:
    """(mtime_ns, size) of the emotion file, or None if it is not there.

    face_control writes it with a rename, so the mtime moves on every write
    even when the name is unchanged -- which is what makes a re-assertion by
    another process count as activity.
    """
    try:
        import face_control
        st = os.stat(face_control.EMOTION_FILE)
    except (OSError, ImportError):
        return None
    return (st.st_mtime_ns, st.st_size)


class FaceDirector:
    """Arbitrates every request for Friday's face."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.RLock()
        self._current = ""
        self._priority = 0
        self._since = 0.0            # when the current face went up
        self._until = 0.0            # when it expires (0 = holds until replaced)
        self._last_activity = time.monotonic()
        self._idle_stage = 0
        self._pending: tuple[str, int, float] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # The stamp of the last write WE made, so a write by anyone else is
        # recognisable. See _adopt_foreign_write().
        self._own_stamp: tuple[int, int] | None = None

    # -- lifecycle --
    def start(self) -> "FaceDirector":
        """Begin the tick that handles decay and idle drift."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._tick, name="face-director",
                                        daemon=True)
        self._thread.start()
        self._apply(config.FACE_DEFAULT_EMOTION, IDLE, hold=0.0)
        return self

    def close(self) -> None:
        self._stop.set()

    # -- the three ways in --
    def safety(self, name: str = "scared") -> None:
        """An obstacle, a cliff, or a failure. Wins over everything.

        Not debounced and not deferred. Whatever the face was doing, this
        replaces it now -- including mid-sentence, which is the point.
        """
        if name not in SAFETY_FACES:
            name = "scared"
        self._request(name, SAFETY, config.FACE_SAFETY_HOLD_S, force=True)

    def pipeline(self, name: str) -> None:
        """What Friday is doing: listening, thinking, speaking…"""
        if name not in PIPELINE_FACES:
            log.debug("not a pipeline face: %s", name)
            return
        # Forced past a lingering emotion, but NEVER past a live safety face:
        # a real change in what Friday is doing should always show, while an
        # obstacle warning must not be cleared by the next status update.
        # No expiry -- rest() or the next pipeline call ends them.
        self._request(name, PIPELINE, hold=0.0,
                      force=not self._safety_live())

    def emotion(self, name: str) -> None:
        """What the brain chose to feel about what was said."""
        if name in SAFETY_FACES:
            # Deliberate: the brain asking for "scared" is an expression, not a
            # sensor reading. It must not gain safety priority, or a model
            # could hold the face hostage.
            self._request(name, EMOTION, config.FACE_EMOTION_HOLD_S)
            return
        if name not in EMOTION_FACES:
            log.debug("not an emotion face: %s", name)
            return
        self._request(name, EMOTION, config.FACE_EMOTION_HOLD_S)

    def rest(self) -> None:
        """The activity is over. Let the face settle back."""
        with self._lock:
            self._last_activity = time.monotonic()
            self._idle_stage = 0
            if self._priority <= PIPELINE:
                self._apply(config.FACE_DEFAULT_EMOTION, IDLE, hold=0.0)

    # -- arbitration --
    def _request(self, name: str, priority: int, hold: float,
                 force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._last_activity = now
            self._idle_stage = 0

            # Safety is never queued, never debounced, never outranked.
            if priority >= SAFETY:
                self._apply(name, priority, hold)
                return

            if not force:
                # A live higher-priority face is never interrupted.
                if priority < self._priority and not self._expired(now):
                    log.debug("%s (%d) held back by %s (%d)", name, priority,
                              self._current, self._priority)
                    return

            # Debounce applies even to forced pipeline changes: force is about
            # outranking a lingering emotion, not about being allowed to
            # flicker. Only safety, handled above, skips this.
            #
            # TWO CORRECTIONS, both from Harsh watching the real screen and
            # saying the order was wrong.
            #
            # 1. IDLE NEVER DEBOUNCES A REAL STATE. The drift thread flips the
            #    face to bored/sleepy/sleeping on its own schedule, and if a
            #    wake landed within FACE_MIN_DWELL_S of one of those, the
            #    "listening" face was thrown away and the next thing on screen
            #    was "thinking". Friday looked like she skipped listening
            #    entirely. Idle animation must never outrank what she is
            #    actually doing.
            #
            # 2. A DEBOUNCED CHANGE IS DEFERRED, NOT DISCARDED. This used to
            #    `return` and lose the request outright, so any state that
            #    arrived too soon after another simply never appeared. It is
            #    now remembered and applied by _step() the moment the dwell is
            #    up -- late by a fraction of a second, rather than missing.
            if (now - self._since < config.FACE_MIN_DWELL_S
                    and name != self._current
                    and priority <= self._priority):
                log.debug("%s deferred behind %s", name, self._current)
                self._pending = (name, priority, hold)
                return
            self._pending = None
            self._apply(name, priority, hold)

    def _expired(self, now: float) -> bool:
        return self._until != 0.0 and now >= self._until

    def _safety_live(self) -> bool:
        """Is a safety face currently up and still within its hold?"""
        with self._lock:
            return (self._priority >= SAFETY
                    and not self._expired(time.monotonic()))

    def _apply(self, name: str, priority: int, hold: float) -> None:
        now = time.monotonic()
        if name == self._current:
            # Re-asserting the same face refreshes its hold without
            # restarting the tween, which would look like a stutter.
            self._priority = max(self._priority, priority)
            self._until = now + hold if hold else 0.0
            return
        try:
            import face_control
            face_control.set_emotion(name)
        except Exception as exc:                # noqa: BLE001
            log.debug("could not set the face to %s: %s", name, exc)
            return
        log.debug("face %s -> %s (p%d)", self._current or "-", name, priority)
        self._current = name
        self._priority = priority
        self._since = now
        self._until = now + hold if hold else 0.0
        self._own_stamp = _emotion_stamp()

    # -- other writers --
    def _adopt_foreign_write(self, now: float) -> bool:
        """Notice an emotion someone else wrote, and treat it as activity.

        THE DIRECTOR IS NOT THE ONLY WRITER. face_control.py from the command
        line, the dashboard, and friday_expression.FaceExpression during an
        autonomous run all write state/emotion.txt directly -- and until this
        existed the director could not see any of them. Two things went wrong:

          * The idle clock kept running. Harsh, 2026-09-03: "while tinkering
            between the different facial expressions the screen goes blank."
            It was the drift reaching stage 3 and writing "sleeping" over what
            he had just set -- and "sleeping" is closed eyes on a black panel,
            which is a blank screen. During a six-minute exploration run it
            landed at the five-minute mark, every time.

          * `self._current` went stale, so `_apply()` would see the name it
            THOUGHT was up, match, and return without writing -- refusing to
            restore its own face.

        Adopting at IDLE priority is deliberate: someone else's face is shown
        and its clock restarts, but the voice pipeline still outranks it and
        the drift resumes normally once the writing actually stops.
        """
        stamp = _emotion_stamp()
        if stamp is None or stamp == self._own_stamp:
            return False
        self._own_stamp = stamp
        try:
            import face_control
            name = face_control.get_emotion()
        except Exception as exc:                # noqa: BLE001
            log.debug("could not read the face back: %s", exc)
            return False
        self._last_activity = now
        self._idle_stage = 0
        if name == self._current:
            return False
        log.debug("face adopted %s -> %s (written elsewhere)",
                  self._current or "-", name)
        self._current = name
        self._priority = IDLE
        self._since = now
        self._until = 0.0
        return True

    # -- decay and idle drift --
    def _tick(self) -> None:
        while not self._stop.wait(config.FACE_TICK_S):
            try:
                self._step()
            except Exception as exc:            # noqa: BLE001
                log.debug("face tick: %s", exc)

    def _step(self) -> None:
        now = time.monotonic()
        with self._lock:
            # Before anything else: has someone else moved the face?
            self._adopt_foreign_write(now)
            # A face that was deferred behind the dwell gets its turn as soon
            # as the dwell is over, so nothing requested is ever simply lost.
            if (self._pending is not None
                    and now - self._since >= config.FACE_MIN_DWELL_S):
                name, priority, hold = self._pending
                self._pending = None
                self._apply(name, priority, hold)
                return
            # An expired emotion eases back to neutral.
            if self._expired(now):
                self._apply(config.FACE_DEFAULT_EMOTION, IDLE, hold=0.0)
                return
            # Nothing at all for a while: drift down through the idle stages.
            if self._priority > IDLE:
                return
            quiet = now - self._last_activity
            stage = 0
            if quiet >= config.FACE_SLEEPING_AFTER_S:
                stage = 3
            elif quiet >= config.FACE_SLEEPY_AFTER_S:
                stage = 2
            elif quiet >= config.FACE_BORED_AFTER_S:
                stage = 1
            if stage != self._idle_stage:
                self._idle_stage = stage
                self._apply(("neutral", "bored", "sleepy", "sleeping")[stage],
                            IDLE, hold=0.0)

    # -- introspection --
    def state(self) -> dict:
        with self._lock:
            return {"face": self._current, "priority": self._priority,
                    "idle_stage": self._idle_stage,
                    "held_for": round(time.monotonic() - self._since, 1)}


# One director per process. main.py drives it; the MCP set_face tool routes
# through it so the brain's choices are arbitrated rather than applied blindly.
_director: FaceDirector | None = None
_make_lock = threading.Lock()


def get() -> FaceDirector:
    global _director
    if _director is not None:
        return _director
    with _make_lock:
        if _director is None:
            _director = FaceDirector(enabled=config.FACE_ENABLED).start()
        return _director


def main() -> int:
    """Walk through a full interaction, so the sequence can be watched."""
    import argparse
    ap = argparse.ArgumentParser(description="Demonstrate the face director.")
    ap.add_argument("--idle", action="store_true",
                    help="sit still and let it drift to sleep")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)-7s %(name)s | %(message)s")

    director = get()
    if args.idle:
        print("idling — watch it go neutral -> bored -> sleepy -> sleeping")
        try:
            while True:
                time.sleep(2)
                print(f"  {director.state()}")
        except KeyboardInterrupt:
            return 0

    script = [
        ("resting", lambda: director.rest(), 2.0),
        ("Harsh talks", lambda: director.pipeline("listening"), 2.0),
        ("processing", lambda: director.pipeline("thinking"), 2.0),
        ("warm reply", lambda: director.emotion("happy"), 1.0),
        ("speaking", lambda: director.pipeline("speaking"), 2.5),
        ("he says thanks", lambda: director.emotion("love"), 2.5),
        ("OBSTACLE", lambda: director.safety("scared"), 3.0),
        ("settling", lambda: director.rest(), 3.0),
    ]
    for label, action, pause in script:
        action()
        print(f"  {label:16} -> {director.state()}")
        time.sleep(pause)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
