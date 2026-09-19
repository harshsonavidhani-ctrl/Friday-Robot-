"""Friday's orchestrator — the thin end-to-end voice loop.

    wake -> listen -> think -> speak -> (act + react)

    ./venv/bin/python main.py                 the loop
    ./venv/bin/python main.py --check         what is ready, then exit
    ./venv/bin/python main.py --once          one turn, then exit
    ./venv/bin/python main.py --say "hello"   skip the mic, type the question

This file is the SPINE and nothing else. Every real capability lives in its own
module, and every physical action goes through the friday-robot MCP server —
main.py never touches a motor, a camera or a sensor. If you find hardware code
in here, it is in the wrong file.

THE FACE IS DRIVEN FROM HERE because it is the one thing that reflects the
*pipeline's* state rather than any single module's: listening while Harsh
talks, thinking while the brain works, speaking while Piper plays, neutral when
idle. No module knows enough on its own to set those.

THE MODELS LOAD IN THE BACKGROUND AT START-UP. Piper costs ~10 s and
faster-whisper costs more; paying either in front of Harsh's first sentence is
the difference between a robot that answers and one that looks broken. Both are
kicked off before the first wake and are warm by the time they are needed.

NOTHING HERE IS ALLOWED TO HARD-CRASH. A turn that fails says so out loud and
the loop continues — a companion robot that dies on one bad transcript is worse
than one that occasionally says "I didn't catch that".
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import sys
import threading
import time

import config
import stages

log = logging.getLogger("friday")

# THE WAKE SOURCE, so an intent can reach it. "Friday, go to sleep" has to
# pause the detector, and the detector is owned by run(). Passing it down
# through handle() and _run_offline() would thread an argument through four
# signatures for one command; this is the spine's own state and it lives here.
# None whenever there is no detector at all (--say, --once with typed input).
WAKE = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / "friday.log"),
        ],
    )
    # These two are chatty at INFO and drown out the conversation.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    return logging.getLogger("friday")


class Face:
    """Friday's expression, as the pipeline sees it.

    A thin wrapper over face_director, which does the actual arbitration
    between what the pipeline is doing, what the brain feels, and what the
    safety system needs to show. Wrapped rather than called directly so that a
    face process that is down — or a bad emotion name — can never take down a
    conversation. The face is decoration; the conversation is the point.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _director(self):
        import face_director
        return face_director.get()

    def set(self, emotion: str) -> None:
        """A pipeline state: listening, thinking, speaking, success…"""
        if not self.enabled:
            return
        try:
            self._director().pipeline(emotion)
        except Exception as exc:            # noqa: BLE001
            log.debug("could not set the face to %s: %s", emotion, exc)

    def feel(self, emotion: str) -> None:
        """A felt reaction, which the director may hold back."""
        if not self.enabled:
            return
        try:
            self._director().emotion(emotion)
        except Exception as exc:            # noqa: BLE001
            log.debug("could not show %s: %s", emotion, exc)

    def alarm(self, kind: str = "dizzy") -> None:
        """Something went wrong. Outranks everything, briefly."""
        if not self.enabled:
            return
        try:
            self._director().safety(kind)
        except Exception as exc:            # noqa: BLE001
            log.debug("could not show the alarm: %s", exc)

    def rest(self) -> None:
        """Nothing is happening; let the face settle and drift."""
        if not self.enabled:
            return
        try:
            self._director().rest()
        except Exception as exc:            # noqa: BLE001
            log.debug("could not rest the face: %s", exc)


class Voice:
    """Speech, off the main thread.

    Sentences are queued and spoken in order by one worker, so the brain can
    keep streaming while Friday is still saying the previous sentence. Speaking
    inline would stall the stream for the length of the audio.
    """

    def __init__(self, face: Face) -> None:
        self.face = face
        self._queue: "asyncio.Queue | None" = None
        self._thread: threading.Thread | None = None
        self._work: list[str] = []
        self._lock = threading.Condition()
        self._stop = False

    def start(self) -> "Voice":
        self._thread = threading.Thread(target=self._run, name="voice",
                                        daemon=True)
        self._thread.start()
        return self

    def say(self, sentence: str) -> None:
        with self._lock:
            self._work.append(sentence)
            self._lock.notify()

    def wait_until_quiet(self, timeout: float = 60.0) -> None:
        """Block until everything queued has actually been spoken."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._work and not self._speaking:
                    return
            time.sleep(0.05)

    _speaking = False

    def _run(self) -> None:
        import tts
        while not self._stop:
            with self._lock:
                while not self._work and not self._stop:
                    self._lock.wait(0.2)
                if self._stop:
                    return
                sentence = self._work.pop(0)
            self._speaking = True
            # THE SPEAKING FACE IS TTS'S JOB, NOT THIS ONE'S. Setting it here
            # too is what left Friday stuck talking: tts._mouth_moving saves
            # the CURRENT emotion so it can put it back afterwards, and if
            # this line has already set "speaking" then that is what it saves
            # and that is what it restores. The face came back to "speaking"
            # at the end of every sentence and stayed there.
            # tts owns it because tts is the only thing that knows when the
            # audio actually starts and stops.
            try:
                tts.speak(sentence)
            except Exception as exc:        # noqa: BLE001
                log.error("could not speak %r: %s", sentence[:40], exc)
            finally:
                self._speaking = False

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()


def preload_models() -> None:
    """Warm Piper and faster-whisper before the first wake."""
    try:
        import tts
        tts.preload()
    except Exception as exc:                # noqa: BLE001
        log.warning("TTS preload failed: %s", exc)
    try:
        import stt
        stt.preload()
    except Exception as exc:                # noqa: BLE001
        log.warning("STT preload failed: %s", exc)


async def check() -> int:
    """Report what is ready without starting anything."""
    print(f"Friday — {config.ROOT}\n")

    unverified = config.missing_hardware_settings()
    print(f"config          {'all verified' if not unverified else 'unverified: ' + ', '.join(unverified)}")
    print(f"brain           {config.MODEL_MAIN}, key "
          f"{'present' if config.ANTHROPIC_API_KEY else 'NOT SET — the brain cannot run'}")
    print(f"wake source     {config.WAKE_SOURCE}")
    print(f"speech          silent={config.DEBUG_SILENT_SPEECH}, "
          f"speed={config.TTS_SPEED}x")

    try:
        import brain
        async with brain.mcp_session() as session:
            tools = await session.list_tools()
            print(f"MCP             {len(tools.tools)} tools at "
                  f"{config.MCP_HTTP_URL}")
    except Exception as exc:                # noqa: BLE001
        print(f"MCP             UNREACHABLE at {config.MCP_HTTP_URL} — "
              f"{type(exc).__name__}. Is friday-mcp.service running?")

    try:
        import brain
        print(f"spend today     ${brain.spent_today():.4f} of "
              f"${config.DAILY_SPEND_CAP_USD:.2f}")
    except Exception:                       # noqa: BLE001
        pass
    return 0


# The expressions the AI may choose after an answer. Professional on purpose:
# no love, wink, angry, excited or bored (personality.md).
_FACE_CHOICES = ("neutral", "happy", "curious", "proud", "surprised", "sad",
                 "confused", "success")
_face_client = None


def _pick_face(question: str, reply: str) -> "str | None":
    """One small-model call: the expression that fits this exchange.

    Runs on its own thread while Friday speaks, so it costs no latency before
    her voice. Returns None on ANY doubt -- no key, spend cap reached, a slow
    or failed call, a word outside _FACE_CHOICES -- and the caller shows the
    old "success" face instead. Counted against the daily spend cap.
    """
    global _face_client
    if not getattr(config, "FACE_AI_DRIVEN", True):
        return None
    try:
        import brain as brain_module
        if brain_module.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return None
        if _face_client is None:
            import anthropic
            _face_client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY, timeout=6.0, max_retries=0)
        model = getattr(config, "FACE_AI_MODEL", config.MODEL_FAST)
        message = _face_client.messages.create(
            model=model, max_tokens=5,
            system=("You choose the facial expression a friendly assistant "
                    "robot shows right after it answers. Reply with exactly one "
                    "word from: " + ", ".join(_FACE_CHOICES) + ". success = a "
                    "task done or a fact given; confused = it could not "
                    "understand or do it; neutral when nothing else fits."),
            messages=[{"role": "user", "content":
                       f"He said: {question[:300]}\nRobot answered: {reply[:400]}"}])
        cost = brain_module._cost_usd(model, message.usage)
        brain_module._record_spend(model, message.usage, cost)
        word = "".join(b.text for b in message.content
                       if getattr(b, "type", None) == "text")
        word = word.strip().strip(".!").lower()
        return word if word in _FACE_CHOICES else None
    except Exception as exc:                    # noqa: BLE001
        log.debug("AI face choice skipped: %s", exc)
        return None


def _ignored(sentence: str) -> bool:
    """True when the brain said the words were not meant for her.

    ⚠ NOT JUST THE BARE SENTINEL ANY MORE. This is checked per SENTENCE as the
    answer streams, so a refusal the model wrote out in its own words was
    spoken one sentence at a time before anything could stop it -- measured
    2026-09-19 23:39. brain.is_ignore knows all three shapes; there is one
    definition of this and both paths use it.
    """
    import brain as brain_module
    return brain_module.is_ignore(sentence)


async def one_turn(brain_obj, session, voice: Voice, face: Face,
                   text: str, turn: "stages.Turn | None" = None) -> str:
    """Think about one thing and say the answer."""
    import brain as brain_module

    turn = turn or stages.Turn("typed")
    face.set("thinking")
    try:
        reply = await turn.run(
            "THINKING",
            brain_obj.ask(
                text,
                # IGNORE is the brain declining speech that was not said to
                # her -- a video, the room, a garbled fragment. Never spoken.
                on_sentence=lambda s: (None if _ignored(s) else voice.say(s)),
                on_tool=lambda name, args: turn.stage("TOOL", f"{name} {args}"),
                session=session,
            ),
            timeout=config.STAGE_BRAIN_TIMEOUT_S,
        )
    except brain_module.SpendCapReached as exc:
        log.warning("spend cap: %s", exc)
        voice.say("I've hit my spending limit for today.")
        face.feel("sad")
        voice.wait_until_quiet()
        turn.done("ABANDONED (spend cap)")
        return ""
    except stages.StageTimeout as exc:
        # SAY SOMETHING. A brain that overran used to leave Friday silent and
        # apparently ignoring the command, which is indistinguishable from
        # never having heard it.
        face.alarm("dizzy")
        voice.say("That one took me too long. Ask me again?")
        voice.wait_until_quiet()
        turn.done(f"ABANDONED ({exc})")
        return ""
    except Exception as exc:                # noqa: BLE001
        # Every stage failure lands here, flashes dizzy, and the loop carries
        # on listening. A companion that dies on one bad turn is worse than
        # one that occasionally admits it lost the thread.
        log.error("the brain failed: %s: %s", type(exc).__name__, exc)
        face.alarm("dizzy")
        voice.say("Something went wrong in my head just then.")
        voice.wait_until_quiet()
        turn.done(f"FAILED ({type(exc).__name__}: {exc})")
        return ""

    if _ignored(reply):
        # Heard, judged not meant for her, and not a word said. Logged as its
        # own outcome so "she ignored me" can be told apart from "she missed".
        turn.note(f"not addressed to her — silent "
                  f"({getattr(brain_obj, 'last_model', '?')}): {text[:60]!r}")
        face.rest()
        turn.done("IGNORED (not said to her)")
        return ""

    if _ignored(reply):
        # Heard, judged not meant for her, and left alone -- no voice, no
        # success face. Logged as its own outcome so it is never mistaken for
        # a turn that failed.
        turn.done(f"IGNORED (not said to her, "
                  f"{getattr(brain_obj, 'last_model', '?')}): {text[:50]!r}")
        face.rest()
        return ""

    # WHICH BRAIN ANSWERED, in the turn log. The fast path is invisible
    # otherwise, and "did that go to Haiku or Sonnet" is the first question
    # anyone asks of a turn that felt quick or felt slow.
    turn.note(f"answered by {getattr(brain_obj, 'last_model', '?')}"
              f"{' (fast path)' if getattr(brain_obj, 'last_fast', False) else ''}"
              f", ${getattr(brain_obj, 'last_cost_usd', 0.0):.4f}")
    # THE REPLY OBJECT (file 08, step 1): the face and anything worth keeping
    # came back in the same response as the words. See reply_contract.py.
    response = getattr(brain_obj, "last_response", None) or {}
    # Already only a content-driven name, or None: reply_contract refuses the
    # pipeline-owned five (listening, speaking, thinking, sleepy, sleeping).
    brain_face = response.get("emotion")
    if response.get("memory_note"):
        _keep_note(response["memory_note"], turn)
    # HE TOLD HER SHE GOT IT WRONG (file 09 step 1). Before spoke() below, so
    # the correction is filed against the reply he was reacting to, not this
    # one. Handles both ways in: the reply object's `correction`, and the
    # phrase tripwire for when the brain did not notice.
    _keep_correction(text, response.get("correction"), turn)
    # SHE ANSWERED WITHOUT REALLY KNOWING (file 09 step 3). Logged, never
    # spoken: the reply above already went out without hedging.
    _keep_uncertainty(response.get("uncertain"), text, reply, turn)

    # THE FACE IS CHOSEN BY THE AI, on a thread, while she speaks. Harsh: "face
    # also ai driven". A thread, not a task: run()'s wake loop is a blocking
    # generator, so an asyncio task would starve (see brain.warm_in_background).
    # Only when the reply object did not already carry a face -- then it is a
    # second Haiku call that is not needed.
    chosen: list = [brain_face]
    picker = threading.Thread(
        target=lambda: chosen.__setitem__(0, _pick_face(text, reply)),
        name="face-pick", daemon=True)
    if reply.strip() and brain_face is None:
        picker.start()

    turn.stage("SPEAKING", stages.quote(reply, 120))
    voice.wait_until_quiet()
    if reply.strip():
        if brain_face is None:
            picker.join(timeout=getattr(config, "FACE_AI_WAIT_S", 1.5))
        mood = chosen[0]
        # No choice in time: "success", as before.
        _show_face(face, mood or "success")
        if mood:
            turn.note(f"face from the reply object: {mood}" if brain_face
                      else f"face chosen by the AI: {mood}")
    _remember_what_she_said(reply)
    turn.done()
    return reply


def _keep_correction(said: str, note: str | None,
                     turn: "stages.Turn") -> None:
    """File 09 step 1: keep a correction, under its own kind. Never raises."""
    try:
        import corrections
        stored = corrections.capture(said, note=note)
        if stored:
            turn.note(f"learned preference: {stored['text'][:80]!r}")
    except Exception as exc:                    # noqa: BLE001
        log.debug("correction capture skipped: %s", exc)


def _keep_uncertainty(topic: str | None, asked: str, said: str,
                      turn: "stages.Turn") -> None:
    """File 09 step 3: log a gap, silently. Never raises."""
    if not topic:
        return
    try:
        import uncertainty
        if uncertainty.log_gap(topic, said=said, asked=asked):
            turn.note(f"uncertain about: {topic[:60]!r}")
    except Exception as exc:                    # noqa: BLE001
        log.debug("uncertainty logging skipped: %s", exc)


def _remember_what_she_said(reply: str) -> None:
    """What a correction on the NEXT turn will be about. Never raises."""
    try:
        import corrections
        corrections.spoke(reply)
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not record the reply: %s", exc)


def _idle_turn_ended(spoken: str = "") -> None:
    """A turn ended: the quiet starts again, and a note she was given in front
    of that turn has now been offered. Never raises."""
    try:
        import idle_study
        idle_study.turn_ended()
        if spoken.strip():
            idle_study.mark_offered()
    except Exception as exc:                    # noqa: BLE001
        log.debug("idle curiosity bookkeeping skipped: %s", exc)
    try:
        import physical_curiosity
        physical_curiosity.turn_ended()
    except Exception as exc:                    # noqa: BLE001
        log.debug("physical curiosity bookkeeping skipped: %s", exc)
    # THE OFFLINE PATH GETS THE TRIPWIRE TOO. one_turn() handles a turn the
    # brain answered; a turn answered locally -- no internet, over the cap, a
    # reflex command -- never goes through it, and "aise mat bolo" is exactly
    # the sentence he says when she is being stupid without a brain.
    # corrections.capture drops the second offer of the same sentence.
    try:
        import corrections
        corrections.capture(spoken)
    except Exception as exc:                    # noqa: BLE001
        log.debug("correction tripwire skipped: %s", exc)


def _physical_curiosity() -> None:
    """Start step 9's thread. Never raises, and does nothing when it is off."""
    try:
        import physical_curiosity
        physical_curiosity.start()
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not start the physical curiosity (%s: %s)",
                    type(exc).__name__, exc)


def _curiosity_heard() -> None:
    """He is talking: stop any outing. Never raises, called on every wake."""
    try:
        import physical_curiosity
        physical_curiosity.heard()
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not stop the outing: %s", exc)


def _show_face(face: "Face", mood: str) -> None:
    """The face a turn ends on, into the band it belongs to."""
    if mood in ("success", "confused"):
        face.set(mood)                         # pipeline band
    else:
        face.feel(mood)                        # emotion band: outlives rest()


async def _scene_turn(brains: "_BrainLoader", voice: "Voice", face: "Face",
                      source, trigger: str) -> None:
    """Something new came into view: one short remark, or nothing.

    File 08 step 7 -- the gates live in scene_bridge.py; by the time a trigger
    gets here it has passed them all. Run from the wake loop like a voice turn,
    so it can never overlap one, and with the detector paused while she speaks
    so her own voice cannot wake her. Never raises.
    """
    import offline
    import scene_bridge

    seen = scene_bridge.parse_trigger(trigger)
    if seen is None:
        return
    turn = stages.Turn(f"scene: {seen['label']}")
    # Not while the brain is still loading (get() would hold the wake loop) and
    # not offline: a remark is never worth waiting for.
    if brains._thread.is_alive() or not offline.online():
        turn.done("SKIPPED (no brain yet, or offline)")
        return
    brain_obj = await brains.get()
    if brain_obj is None:
        turn.done("SKIPPED (no brain)")
        return

    source.pause()
    try:
        face.feel("curious")                   # she has noticed, before a word
        reply = await turn.run(
            "THINKING",
            brain_obj.remark(
                scene_bridge.event_text(seen),
                seen=scene_bridge.frame_blocks(),
                on_sentence=lambda s: (None if _ignored(s) else voice.say(s))),
            timeout=config.STAGE_BRAIN_TIMEOUT_S)
        cost = getattr(brain_obj, "last_cost_usd", 0.0)
        if not reply:
            turn.done("ABANDONED (could not ask the brain)")
            return
        if _ignored(reply):
            turn.done(f"QUIET (not worth a remark, ${cost:.4f})")
            return
        turn.note(f"remark by {getattr(brain_obj, 'last_model', '?')}, "
                  f"${cost:.4f}")
        turn.stage("SPEAKING", stages.quote(reply, 120))
        voice.wait_until_quiet()
        response = getattr(brain_obj, "last_response", None) or {}
        mood = response.get("emotion") or "curious"
        _show_face(face, mood)
        turn.note(f"face from the reply object: {mood}")
        if response.get("memory_note"):
            _keep_note(response["memory_note"], turn)
        turn.done()
    except Exception as exc:                    # noqa: BLE001
        log.warning("scene remark failed: %s: %s", type(exc).__name__, exc)
        turn.done(f"FAILED ({type(exc).__name__})")
    finally:
        face.rest()
        source.resume()
        turn.listening()


def _keep_note(note: str, turn: "stages.Turn") -> None:
    """Save the reply object's memory_note, on a thread. Never raises.

    A thread because memory.remember checks for a near-duplicate against the
    recall index, which rebuilds after every write -- not something to hold
    the end of a turn for. Near-duplicates UPDATE the existing fact, so a note
    that repeats something she already knows (or that she also passed to the
    remember tool) does not become a second copy.
    """
    turn.note(f"memory note: {note[:80]!r}")

    def save() -> None:
        try:
            import memory
            stored = memory.remember(
                note, source=getattr(config, "BRAIN_NOTE_SOURCE", "brain-note"))
            log.info("memory note %s: %r",
                     "updated" if stored["updated"] else "kept", stored["text"])
        except Exception as exc:                # noqa: BLE001
            log.warning("could not keep the memory note (%s: %s): %r",
                        type(exc).__name__, exc, note[:80])

    threading.Thread(target=save, name="memory-note", daemon=True).start()


def _idle_curiosity(brains: "_BrainLoader") -> None:
    """Start the idle look-up thread, once the brain exists. Never raises."""
    def go() -> None:
        try:
            import idle_study
            brain_obj = None
            while brain_obj is None:
                brains._thread.join()
                brain_obj = brains._brain
                if brain_obj is None:
                    return              # no key: nothing to be curious with
            idle_study.start(lambda: list(brain_obj.messages))
        except Exception as exc:                # noqa: BLE001
            log.warning("could not start the idle curiosity (%s: %s)",
                        type(exc).__name__, exc)

    threading.Thread(target=go, name="idle-study-start", daemon=True).start()


def _warm_memory() -> None:
    """Build memory's recall index in the background. Never raises."""
    started = time.monotonic()
    try:
        import memory
        memory.warm()
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not warm the memory index (%s: %s) — the first "
                    "question will pay for it instead", type(exc).__name__, exc)
        return
    log.info("memory index warm in %.1fs", time.monotonic() - started)


class _BrainLoader:
    """`import brain` and Brain(), on a thread, behind the wake word.

    anthropic alone is ~5.6 s of import on this Pi, and hearing "hey Friday"
    needs none of it. get() waits for it; anything it raised is raised there,
    exactly where it used to be raised at start-up.
    """

    def __init__(self) -> None:
        self._brain = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._load, name="brain-load",
                                        daemon=True)

    def start(self) -> "_BrainLoader":
        self._thread.start()
        return self

    def _load(self) -> None:
        try:
            import brain as brain_module
            # A MISSING API KEY MUST NOT BE FATAL. This runs under systemd
            # with Restart=always, so exiting here would crash-loop every few
            # seconds and look, from the outside, exactly like a broken robot.
            # It is also the normal state before the key is set. Friday comes
            # up deaf to reasoning but still awake: the wake word works, and
            # the offline keyword handler still drives, stops, turns and tells
            # the time.
            try:
                self._brain = brain_module.Brain()
            except brain_module.BrainError as exc:
                log.warning("no brain (%s) — running on the offline handler "
                            "only. Set ANTHROPIC_API_KEY and restart to "
                            "enable it.", exc)
                return
            # WARM THE CONNECTION WHILE SHE IS STILL BOOTING, not when Harsh
            # is stood there waiting. A cold DNS lookup costs 5.04 s on this Pi
            # and the first question of the evening was paying all of it.
            # ⚠ A THREAD, NOT create_task(). The wake loop in run() is a
            # BLOCKING generator inside an async function, so the moment it
            # starts the event loop stops running anything else and a
            # scheduled coroutine is starved forever. As a bare create_task
            # this logged "warm in 21.59s" on one start and nothing at all on
            # the next two. Same pattern, same reason, as tts.preload() and
            # stt.preload().
            if getattr(config, "BRAIN_WARM_ON_START", True):
                self._brain.warm_in_background()
            # THE MEMORY INDEX TOO, on its own thread so get() is not held by
            # it. Every question is looked up in memory first, and building
            # that index cold is 6.4 s. openwakeword's import used to load
            # sklearn at start-up by accident; since the wake word stopped
            # doing that (2026-09-15), the first question after a restart
            # was paying for all of it before the request was even sent.
            if getattr(config, "MEMORY_IN_CONTEXT", True):
                threading.Thread(target=_warm_memory, name="memory-warm",
                                 daemon=True).start()
        except BaseException as exc:            # noqa: BLE001 - raised in get()
            self._error = exc

    async def get(self):
        await asyncio.to_thread(self._thread.join)
        if self._error is not None:
            raise self._error
        return self._brain


# HOW LONG A TURN WAITS FOR friday-mcp. At boot it refuses connections for
# ~29 s while it brings the hardware and Piper up (2026-09-15: unit started at
# 15.1 s, serving at 44.2 s), and this process used to connect once, crash into
# the refusal, and pay a whole restart before the wake word was back.
MCP_CONNECT_WAIT_S = 60.0
MCP_CONNECT_RETRY_S = 1.0


async def _mcp_session(held: "contextlib.AsyncExitStack", wait_s: float):
    """A friday-mcp session kept open on `held`, retried for up to wait_s."""
    import brain as brain_module

    deadline = time.monotonic() + wait_s
    while True:
        try:
            session = await held.enter_async_context(
                brain_module.mcp_session())
            break
        except Exception:                       # noqa: BLE001
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(MCP_CONNECT_RETRY_S)
    listed = await session.list_tools()
    log.info("connected to %s — %d tools", config.MCP_SERVER_NAME,
             len(listed.tools))
    return session


async def run(once: bool = False, typed: str | None = None) -> int:
    import contextlib

    import stt
    import wake_word

    face = Face()
    face.set(config.FACE_DEFAULT_EMOTION)

    # THE WAKE WORD COMES UP BEFORE ANYTHING ELSE IS LOADED. Harsh, 2026-09-15:
    # "wake word also just starts on boot within 4 seconds". It was 37 s after
    # this process started and 88 s after power-on, because it queued behind
    # `import brain`, Piper and faster-whisper loading beside it, and the MCP
    # connection -- which friday-mcp refuses for ~29 s at boot, and which
    # crashed this process on that boot. Hearing "hey Friday" needs none of
    # them, so the mic and the detector start first and the rest loads behind.
    stream = source = scene_watch = None
    if typed is None:
        # ONE microphone stream, shared by the wake detector and the capture.
        # Opened here so both get the same one -- see stt.AudioStream for why
        # a second reader on this USB card is not an option.
        stream = stt.AudioStream().start()
        source = wake_word.make(stream=stream)
        # SOMETHING NEW IN VIEW is one more wake source (file 08 step 7): it
        # comes through this same loop, so a remark waits for Harsh's turn and
        # never talks over it. Added FLAT into the AnyOf, not nested -- nesting
        # would turn "openwakeword: 0.52" into "any: openwakeword: 0.52" in
        # every turn log.
        if getattr(config, "SCENE_BRIDGE", False):
            import scene_bridge
            scene_watch = scene_bridge.NewObjectWatch()
            inner = (source.sources if isinstance(source, wake_word.AnyOf)
                     else (source,))
            source = wake_word.AnyOf(*inner, scene_watch)
        global WAKE
        WAKE = source
        log.info("wake source: %s. Ctrl-C to stop.", source.name)
        _explore_hush(source)

    try:
        voice = Voice(face).start()
        preload_models()
        brains = _BrainLoader().start()

        # The session is opened on this stack, in this task, so it is closed
        # by the same task that opened it -- the anyio task group inside it
        # requires that.
        async with contextlib.AsyncExitStack() as held:
            if typed is not None:
                session = await _mcp_session(held, MCP_CONNECT_WAIT_S)
                await handle(await brains.get(), session, voice, face, typed)
                voice.close()
                return 0

            # CONNECTED ON THE FIRST TURN, not here. Waiting here for
            # friday-mcp would park the wake loop behind it again, which is
            # the whole delay this ordering removes.
            session = None
            _explore_targeter(voice)
            # SHE LOOKS THINGS UP IN THE QUIET (file 08 step 8). A thread with
            # its own client, deliberately NOT part of this loop: a search
            # takes 3-15 s and the loop must stay free to hear him.
            _idle_curiosity(brains)
            # SHE GOES AND LOOKS FOR HERSELF (file 08 step 9), if and only if
            # config.PHYSICAL_CURIOSITY is on -- it ships off. Its own thread
            # for the same reason as step 8, and it drives through the MCP
            # server's explore_room, never past it.
            _physical_curiosity()
            face.rest()

            for trigger in source.listen():
                if scene_watch is not None and trigger.startswith("scene: "):
                    await _scene_turn(brains, voice, face, source, trigger)
                    if once:
                        break
                    continue
                turn = stages.Turn(trigger)
                # Acknowledge instantly: the face and the beep both land
                # before any model runs, so Harsh knows he was heard rather
                # than wondering whether to repeat himself.
                face.set("listening")
                # WHEELS FIRST. If step 9 has her out looking around, she
                # stops here -- before the capture, while he is still
                # talking -- rather than driving through his question.
                _curiosity_heard()
                if hasattr(source, "beep"):
                    source.beep()

                # THE DETECTOR STAYS PAUSED FOR THE WHOLE TURN — through the
                # thinking and the speaking too, not just the capture. It used
                # to resume the moment the transcript came back, which left it
                # scoring Friday's OWN voice while she answered. A self-wake
                # there burns the cooldown, so the next real "hey Friday"
                # arrives mid-cooldown and is dropped: the command that looks
                # broken is the one AFTER the one that worked.
                source.pause()
                spoken = ""
                try:
                    audio = await turn.call("CAPTURING", stream.capture,
                                            timeout=config.MAX_UTTERANCE_S
                                            + config.LISTEN_START_TIMEOUT_S
                                            + 5.0)
                    said_at = time.monotonic()   # he stopped talking here
                    face.set("thinking")
                    spoken = await turn.call(
                        "TRANSCRIBING", stt.transcribe, audio,
                        timeout=config.STAGE_STT_TIMEOUT_S)
                    turn.stage("TRANSCRIBED", stages.quote(spoken))

                    # NOTHING CAME BACK? Listen once more before giving up.
                    # The capture can close on the wake word's own tail before
                    # Harsh has started talking -- especially on a borderline
                    # score, which fires earlier in the phrase and leaves more
                    # of it in the buffer. Abandoning there looks exactly like
                    # being ignored, which is the complaint this whole day of
                    # work is about.
                    if not spoken.strip() and config.EMPTY_TRANSCRIPT_RETRY:
                        turn.stage("RE-LISTENING", "nothing in the first take")
                        # No handover on the retry: the wake word is long
                        # past, so there is nothing worth reaching back for,
                        # and reaching back is what put it in the capture the
                        # first time.
                        audio = await turn.call(
                            "CAPTURING",
                            functools.partial(
                                stream.capture,
                                start_timeout=config.RETRY_LISTEN_TIMEOUT_S,
                                handover_ms=0, deadtime_ms=0),
                            timeout=config.MAX_UTTERANCE_S
                            + config.RETRY_LISTEN_TIMEOUT_S + 5.0)
                        said_at = time.monotonic()
                        spoken = await turn.call(
                            "TRANSCRIBING", stt.transcribe, audio,
                            timeout=config.STAGE_STT_TIMEOUT_S)
                        turn.stage("TRANSCRIBED", stages.quote(spoken))
                except stt.Silence as exc:
                    turn.stage("NOTHING SAID", str(exc))
                    turn.done("ABANDONED")
                except stages.StageTimeout as exc:
                    face.alarm("dizzy")
                    voice.say("Sorry, I lost that one.")
                    voice.wait_until_quiet()
                    turn.done(f"ABANDONED ({exc})")
                except Exception as exc:    # noqa: BLE001
                    log.error("could not listen: %s: %s",
                              type(exc).__name__, exc)
                    face.alarm("dizzy")
                    turn.done("FAILED (capture)")
                else:
                    if spoken.strip():
                        brain_obj = await brains.get()
                        if session is None:
                            # At boot friday-mcp may still be coming up, so
                            # the first turn waits for it. The age check below
                            # still counts from said_at, so a move asked for
                            # before it answered is refused as stale, never
                            # driven late.
                            try:
                                session = await _mcp_session(
                                    held, MCP_CONNECT_WAIT_S)
                            except Exception as exc:    # noqa: BLE001
                                log.error("friday-mcp did not answer within "
                                          "%.0fs: %s: %s", MCP_CONNECT_WAIT_S,
                                          type(exc).__name__, exc)
                        if session is None:
                            face.alarm("dizzy")
                            voice.say("Sorry, I lost that one.")
                            voice.wait_until_quiet()
                            turn.done("FAILED (friday-mcp not answering)")
                        else:
                            # Every tool this turn calls goes through the age
                            # check -- reflex or brain -- see COMMAND_MAX_AGE_S.
                            await handle(brain_obj,
                                         _FreshOrders(session, said_at),
                                         voice, face, spoken, turn)
                    else:
                        # Logged as its own outcome. "Heard you, understood
                        # nothing" and "never heard you" are different faults
                        # with different fixes, and they used to look the same.
                        #
                        # set(), not feel(): "confused" is a PIPELINE face, and
                        # face_director.emotion() drops any name that is not in
                        # EMOTION_FACES. Sent through feel() this never once
                        # reached the screen.
                        face.set("confused")
                        turn.done("ABANDONED (empty transcript)")
                finally:
                    if scene_watch is not None and spoken.strip():
                        scene_watch.heard_harsh()   # he is here; give him a moment
                    _idle_turn_ended(spoken)
                    # RE-ARM, ALWAYS. Every path above lands here, including
                    # the timeouts and the crashes, because a turn that ends
                    # without handing the microphone back is exactly the bug
                    # this whole trace exists to make visible.
                    face.rest()
                    source.resume()
                    turn.listening()

                if once:
                    break
    finally:
        # Closed on EVERY exit, including one before the loop was reached:
        # the mic's pw-cat is a child process, and one left behind would be a
        # second reader on the card when systemd starts the next instance.
        if source is not None:
            source.close()
        if stream is not None:
            stream.close()

    voice.close()
    return 0


async def handle(brain_obj, session, voice: Voice, face: Face,
                 spoken: str, turn: "stages.Turn | None" = None) -> str:
    """One utterance: reflexes first, then the brain if it can be reached.

    STOP IS CHECKED BEFORE ANYTHING ELSE, online or off. Nothing should stand
    between Harsh saying stop and the motors halting -- least of all a round
    trip to a language model.

    Falls back to the local keyword handler when there is no internet, so a
    dropped connection costs Friday its conversation but not its ability to be
    told what to do.
    """
    import offline

    turn = turn or stages.Turn("typed")

    if offline.is_stop(spoken):
        turn.note("STOP heard — going straight to the motors")
        await _run_offline(offline.Intent(name="stop", tool="stop",
                                          say="Stopping."),
                           session, voice, face, turn)
        return "stopped"

    # No key is the same situation as no internet, from the robot's point of
    # view: there is no reasoning available, so fall through to the reflexes
    # rather than pretending and failing.
    if brain_obj is None or not offline.online():
        intent = offline.understand(spoken)
        if intent is not None:
            turn.note(f"offline — handling {intent.name!r} locally")
            await _run_offline(intent, session, voice, face, turn)
            return intent.say
        turn.note(f"offline, and {spoken[:40]!r} is not a local command")
        face.feel("sad")
        voice.say("I can't reach my brain right now. Say what can you do "
                  "and I'll tell you what still works.")
        voice.wait_until_quiet()
        turn.done("ABANDONED (offline, no local match)")
        return ""

    # THE REFLEXES RUN EVEN WHEN THERE IS A BRAIN, and that is the point of
    # strict mode. "Turn left" through a model is a pivot that arrives after a
    # round trip -- measured on this Pi at anywhere from 3 to 20 s -- by which
    # time Harsh has said it again and the robot turns twice. An order whose
    # meaning does not depend on the sentence around it should not wait for a
    # sentence to be understood. Questions still go to the brain: strict mode
    # hands back anything outside offline._REFLEX_INTENTS.
    intent = offline.understand(spoken, strict=True)
    if intent is not None:
        turn.note(f"reflex — {intent.name!r} locally, no round trip")
        await _run_offline(intent, session, voice, face, turn)
        return intent.say

    return await one_turn(brain_obj, session, voice, face, spoken, turn)


class _FreshOrders:
    """The MCP session, for one turn, refusing to MOVE on an order gone stale.

    Harsh: "it should not execute any previously given command after 5
    seconds". Transcription has taken 7-17 s on this Pi, so a spoken "turn
    left" could reach the motors long after he had given up on it. The age is
    measured from the end of his speech to the moment the tool is called, and
    only for config.COMMAND_FRESH_TOOLS -- stop() is never refused.

    Everything else is the real session, untouched.
    """

    def __init__(self, session, said_at: float) -> None:
        self._session = session
        self._said_at = said_at
        self._acted_at: float | None = None

    def __getattr__(self, name):
        return getattr(self._session, name)

    async def call_tool(self, name: str, args=None):
        # JUDGED ONCE, WHEN SHE STARTS ACTING -- the first tool of the turn.
        # Checked per tool it refused every move live (06:35, three in a row):
        # the safety layer's observe() runs first and takes ~3.8 s, so "drive
        # backwards" understood 3.5 s after he spoke reached move() at 7.3 s.
        # A plan begun in time runs to its end; one begun late never moves.
        if self._acted_at is None:
            self._acted_at = time.monotonic()
        age = self._acted_at - self._said_at
        limit = getattr(config, "COMMAND_MAX_AGE_S", 0.0)
        if (limit and age > limit
                and name in getattr(config, "COMMAND_FRESH_TOOLS", ())):
            log.warning("REFUSED %s: the order is %.1fs old (limit %gs)",
                        name, age, limit)
            from types import SimpleNamespace
            text = (f"REFUSED — stale order: {name} was asked for {age:.0f} s "
                    f"ago, past the {limit:g} s limit, so the robot did not "
                    f"move. Ask Harsh to say it again.")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                isError=True)
        return await self._session.call_tool(name, args)


# Tools that stop things. Their success text says "stopped", which is the same
# word the safety interrupt uses when it refuses a move, so they are named
# rather than pattern-matched.
_HALTING_TOOLS = {"stop", "stop_exploration"}


def _stop_reason(text: str) -> str:
    """Turn the safety interrupt's own words into one spoken line.

    The MCP result already says exactly why it stopped -- "something is 219 mm
    ahead, inside the 250 mm stop distance (ToF) (clearance 219 mm)" -- and
    Friday used to answer all of that with "I had to stop.", which tells Harsh
    nothing he could act on. He cannot see the ToF reading; the whole point of
    a voice is to hand him what only the robot knows.

    Kept to ONE short natural line, per the narration rules: distance in the
    unit a person would actually say, no sensor jargon, no numbers read out to
    the millimetre.
    """
    import re
    match = re.search(r"clearance\s+(\d+(?:\.\d+)?)\s*mm", text)
    if not match:
        # Refused for a reason with no distance in it -- an observe() that has
        # gone stale, a tool that said no. Say that plainly rather than
        # inventing a measurement.
        if "stale order" in text.lower():
            return "That took me too long, so I didn't move. Say it again?"
        if "refused" in text.lower():
            return "I stopped — the robot refused that one."
        return "Something's in the way, so I stopped."
    mm = float(match.group(1))
    if mm < 100:
        how_far = f"only about {mm / 10:.0f} centimetres"
    elif mm < 1000:
        how_far = f"about {mm / 10:.0f} centimetres"
    else:
        how_far = f"about {mm / 1000:.1f} metres"
    where = "ahead" if "ahead" in text else "in the way"
    return f"Something's {where}, {how_far} off. I've stopped."


# --- the wake word goes quiet while she is out driving -------------------
#
# Harsh, 2026-09-08: "while exploring mode you can stop the wake word file
# while exploring".
#
# WHAT IT COSTS TO LEAVE IT ON, measured during the runs of the same evening:
# pi_deploy speaks "Friday is starting autonomous exploration" through the
# speaker, the detector scores Friday's OWN voice and fires -- 0.99 at
# 18:41:40, right as she announced herself -- and every one of those false
# wakes runs faster-whisper at ~x5 realtime on a Pi that is already missing
# its 100 ms control deadline. Turn 37 took 96 s and transcribed "". The
# detector itself is cheap; the TURN behind it is not, and during a run every
# one of them is a false alarm.
#
# ⚠ AND IT COSTS THE VOICE STOP. "Friday, stop" cannot reach her while the
# detector is asleep -- the stop button in the dashboard and the run's own
# --duration are the remaining ways to end a run early. That is the trade
# being made here, deliberately.
EXPLORE_HUSH_POLL_S = 4.0
# Re-armed in SLICES rather than set once for the run's length, and that is
# the whole safety of it: if a run is SIGKILLed, explore_mode.json is left
# saying "running" forever, and a single long sleep would leave Friday deaf
# until someone noticed. A slice lapses on its own, so the worst a stale file
# can cost is one slice.
EXPLORE_HUSH_SLICE_S = 20.0


def _exploring() -> bool:
    """Is an autonomous run out right now? Cheap enough to ask every 4 s.

    TWO SIGNALS, because neither covers every run on its own. The hardware
    lease is held only by a run started through the MCP tool (the dashboard's
    button, and "Friday, explore"); explore_mode.json is written by
    rl/modes/explore.sh for EVERY run including one started by hand, on every
    exit path including a kill. Either one being true means she is out.
    """
    try:
        from mcp_server import hardware
        if hardware.lease_active() > 0.0:
            return True
    except Exception:                       # noqa: BLE001 - never break the loop
        pass
    try:
        status = json.loads(
            (config.ROOT / "state" / "explore_mode.json").read_text())
        return status.get("state") == "running"
    except Exception:                       # noqa: BLE001
        return False


def _explore_hush(source) -> None:
    """Keep the wake detector asleep for as long as she is out driving."""
    if not hasattr(source, "sleep"):
        return

    def watch() -> None:
        hushed = False
        while True:
            try:
                out = _exploring()
                if out:
                    # Only re-arm as the current slice runs down, so this does
                    # not write "asleep for 20s" to the log every 4 seconds.
                    if source.sleeping_for() < EXPLORE_HUSH_SLICE_S / 2:
                        source.sleep(EXPLORE_HUSH_SLICE_S)
                    if not hushed:
                        log.info("exploring — wake word off until she parks")
                        hushed = True
                elif hushed:
                    # Do not call wake_up() on a detector this never hushed:
                    # "Friday, go to sleep" uses the same gate, and clearing it
                    # here would cancel a sleep Harsh actually asked for.
                    source.wake_up()
                    log.info("she has parked — wake word back on")
                    hushed = False
            except Exception as exc:        # noqa: BLE001
                log.error("explore hush check failed: %s", exc)
            time.sleep(EXPLORE_HUSH_POLL_S)

    threading.Thread(target=watch, name="friday-explore-hush",
                     daemon=True).start()


# --- she picks something to drive at, while she is out -------------------
#
# Harsh, 2026-09-08: "during run the face plus brain will be working because
# brain will help in targetting an object and move towards it", and the target
# is chosen "autonomously, during the run" -- no human in the loop, which it
# could not be anyway now that the wake word is off while she drives.
#
# THE BRAIN CANNOT LOOK THROUGH THE CAMERA MID-RUN. The run holds both /dev/video
# nodes and a V4L2 device does not survive two readers, so `look()` loses to
# friday_localnav_deploy.py for the physical device and `observe`,
# `scan_environment` and `say_what_you_see` all raise HardwareLeased. What the
# brain gets instead is the run's OWN detector, published as text in
# state/rl_autonomy.json -- label, score and a frame fraction. No image, and no
# distance, because the detector has none to give.
#
# COST. Every call goes through brain_router, which owns the routing (DRIVE ->
# Haiku), the DAILY_SPEND_CAP_USD ceiling and the cost log. There is deliberately
# no second API path here. An empty room costs nothing at all: without a sighting
# no call is made, and with no ANTHROPIC_API_KEY -- which is the state of this Pi
# while it is being built -- decide() returns `skipped` and the run behaves
# exactly as it does today.
TARGET_PROMPT = (
    "You are steering a small exploring robot. These are the only things its "
    "camera can name right now. `where` is the object's horizontal position as "
    "a fraction of the frame: +1 is hard left, -1 is hard right, 0 is straight "
    "ahead. `area` is the fraction of frame it fills -- bigger is nearer, very "
    "roughly. There is no distance measurement.\n\n"
    "Reply with ONE label from the list, exactly as written, if it is worth "
    "driving over to look at. Reply with NONE if nothing is. Prefer something "
    "she has not just been to, and prefer the interesting over the ordinary. "
    "One word, nothing else."
)


def _target_prompt(seen: list) -> str:
    lines = [f"- {row.get('label')} (score {float(row.get('score', 0)):.2f}, "
             f"where {float(row.get('where', 0)):+.2f}, "
             f"area {float(row.get('area', 0)):.3f})" for row in seen]
    return TARGET_PROMPT + "\n\nIn view:\n" + "\n".join(lines)


def _explore_targeter(voice: Voice) -> None:
    """While she is out: watch what she sees, name one thing, drive at it."""
    import rl_control
    from brain_router import DRIVE, BrainRouter

    router = BrainRouter()

    def watch() -> None:
        last_call = 0.0
        current = None
        while True:
            try:
                if not _exploring():
                    if current is not None:
                        # Do not leave a standing order behind for the next run
                        # to pick up: it would steer her at something chosen for
                        # a room she may no longer be in.
                        rl_control.clear()
                        current = None
                    time.sleep(config.TARGET_DECIDE_EVERY_S)
                    continue

                seen = rl_control.seen()
                now = time.monotonic()
                if not seen or now - last_call < config.TARGET_DECIDE_EVERY_S:
                    time.sleep(2.0)
                    continue
                last_call = now

                if not router.available():
                    # No key: say so once per run rather than every 20 s, and
                    # stop asking. This is the normal state while building.
                    if current != "__nokey__":
                        log.info("targeting is idle — no ANTHROPIC_API_KEY, so "
                                 "she explores on her frontier planner")
                        current = "__nokey__"
                    time.sleep(config.TARGET_DECIDE_EVERY_S)
                    continue

                decision = asyncio.run(
                    router.decide(DRIVE, _target_prompt(seen), max_tokens=12))
                if not decision.answered:
                    log.info("no target this time: %s", decision.skipped)
                    continue

                answer = (decision.text or "").strip().strip(".").lower()
                labels = {str(r.get("label", "")).lower() for r in seen}
                if answer in ("none", "") or answer not in labels:
                    # An answer that is not one of the labels offered is not a
                    # target. Driving at a hallucinated one is how she ends up
                    # aiming at something that was never in the room.
                    continue
                if answer != current:
                    rl_control.target(answer)
                    current = answer
                    log.info("driving at %r (%s, $%.4f)",
                             answer, decision.model, decision.usd)
                    voice.say(f"I can see a {answer}. Going for a look.")
            except Exception as exc:            # noqa: BLE001
                log.error("targeting check failed: %s", exc)
                time.sleep(config.TARGET_DECIDE_EVERY_S)

    threading.Thread(target=watch, name="friday-explore-target",
                     daemon=True).start()


def _run_local(intent, face: Face, turn: "stages.Turn") -> None:
    """A command that needs no tool: stop listening, or say what you can do."""
    if intent.local == "sleep":
        minutes = float(intent.args.get("minutes",
                                        config.SLEEP_DEFAULT_MINUTES))
        if WAKE is None:
            turn.note("no wake detector to put to sleep")
            turn.done("DONE (nothing to sleep)")
            return
        granted = WAKE.sleep(minutes * 60.0)
        # THE FACE IS THE ONLY SIGN SHE IS ASLEEP RATHER THAN BROKEN. A
        # detector that has stopped answering and a detector that was told to
        # stop answering look identical from the sofa, and this project has
        # already lost an evening to that exact ambiguity in the log.
        face.set("sleeping")
        turn.note(f"detector asleep for {granted / 60:.0f} min")
        turn.done("DONE (asleep)")
        return
    turn.done("DONE")


async def _run_offline(intent, session, voice: Voice, face: Face,
                       turn: "stages.Turn | None" = None) -> None:
    """Carry out a locally-understood command, still through the MCP server.

    Through MCP even with no internet, because that is where the safety
    interrupt lives. An offline path that drove the motors directly would be
    the one way to bypass it.
    """
    import memory
    import offline

    turn = turn or stages.Turn("offline")
    if intent.say:
        voice.say(intent.say)
    outcome = "worked"
    answer = ""

    # SOMETHING ONLY THE SPINE CAN DO. offline.py names it; this performs it.
    # Said BEFORE it takes effect -- a detector put to sleep before the
    # sentence finishes playing is one that never confirms it heard the order.
    # Only for intents with NO tool calls; the ones that have both run their
    # tools first and their local part after (see the end of this function).
    if intent.local and not intent.plan():
        voice.wait_until_quiet()
        _run_local(intent, face, turn)
        return
    for tool, args in intent.plan():
        try:
            # The same ceiling the brain's tool calls get. The offline path is
            # the one that runs when everything else is already degraded, so
            # it is the last place that can afford to hang.
            result = await turn.run(
                "TOOL", session.call_tool(tool, args),
                timeout=config.STAGE_TOOL_TIMEOUT_S, detail=tool)
            # NOT EVERY RESULT IS TEXT. observe() comes back as ImageContent,
            # which has no .text at all -- reading it blindly raised
            # AttributeError and failed the whole command at its FIRST step,
            # which for a directional move is the observe() the server insists
            # on. Caught by the test that was meant to prove observe() worked.
            text = " ".join(getattr(part, "text", "") or ""
                            for part in (result.content or []))
            kinds = ", ".join(type(part).__name__
                              for part in (result.content or [])) or "nothing"
            turn.note(f"{tool} -> {text[:120] or kinds}")
            if text.strip():
                answer = text          # the LAST step's answer is the answer
            # CASE-INSENSITIVE, and that is not a nicety. The server says
            # "REFUSED — ..." in capitals (embodiment.py) while this tested
            # for lowercase "refused", so a REFUSAL never aborted the
            # sequence -- only a STOPPED did. Asked to "go left" with a stale
            # observation, Friday had her pivot refused and then drove
            # FORWARD anyway, because the next step ran regardless. Wrong
            # direction, under the safety interrupt's nose.
            low = text.lower()
            # "STOPPED" IS SUCCESS FOR A TOOL WHOSE JOB IS STOPPING.
            # stop_exploration() answers "Stopped exploring." and stop()
            # answers with the safety layer's own word, so this test read both
            # of them as the safety interrupt refusing -- abandoning the rest
            # of the sequence and announcing an obstacle that was never there
            # ("Something's in the way, so I stopped"). The check is for a
            # move that was REFUSED, so it only applies to tools that move.
            if tool not in _HALTING_TOOLS and (
                    "stopped" in low or "refused" in low):
                outcome = "partial"
                # THE SAFETY INTERRUPT SPOKE. Abandon the rest of the sequence
                # rather than driving the next leg of a plan the robot has
                # already said it cannot carry out.
                turn.note("refused or stopped — abandoning the rest")
                voice.say(_stop_reason(text))
                break
        except stages.StageTimeout as exc:
            voice.say("The robot didn't answer that one.")
            face.alarm("dizzy")
            outcome = "failed"
            turn.note(str(exc))
            break
        except Exception as exc:            # noqa: BLE001
            log.error("offline command failed: %s", exc)
            face.alarm("dizzy")
            outcome = "failed"
            break
    # SAY THE ANSWER. A question whose answer only reaches the log is a
    # question Friday did not answer -- "what's the battery" ran get_battery,
    # noted 400 characters of JSON and then said nothing at all, which from
    # the sofa is indistinguishable from not having understood.
    if intent.speak_result and outcome != "failed":
        spoken = offline.speakable(intent.name, answer)
        if spoken:
            voice.say(spoken)
        else:
            voice.say("I got nothing back for that one.")

    # THE NARRATOR ATTACHES AFTER THE RUN IS ALREADY GOING, and that ordering
    # is the whole fix. Starting the run from inside the spawned process cost
    # 30 SECONDS before the wheels moved -- 10.4 s of python start-up and
    # imports, 19.2 s to open a second MCP session -- when this function
    # already holds a session and start_exploration takes well under one.
    if intent.local == "explore" and outcome == "worked":
        import subprocess
        import exploration_mode
        minutes = float(intent.args.get("minutes", 20.0))
        # THE NARRATOR STANDS DOWN FOR A QUIET RUN. --quiet-services stops
        # friday-mcp, which is the only way this process can speak or read a
        # sensor, so attaching would produce a mute narrator whose teardown
        # calls stop_exploration() into a server that is gone. The run talks
        # for itself in that mode.
        if exploration_mode.mode_runs_quiet():
            turn.note("quiet run — pi_deploy narrates, not exploration_mode")
            voice.wait_until_quiet()
            turn.done("DONE (exploring, quiet)")
            return
        try:
            subprocess.Popen(
                [str(config.ROOT / "venv/bin/python"),
                 str(config.ROOT / "exploration_mode.py"),
                 "--minutes", f"{minutes:g}", "--attach"],
                cwd=str(config.ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            turn.note(f"narrator attached for {minutes:g} min")
        except Exception as exc:                # noqa: BLE001
            # The run is already driving. A narrator that failed to start is a
            # quiet run, not a failed one.
            log.error("could not attach exploration_mode: %s", exc)

    try:
        memory.log_event(f"offline: {intent.name}", task=intent.name,
                         outcome=outcome)
    except Exception:                       # noqa: BLE001
        pass
    voice.wait_until_quiet()
    turn.done("DONE" if outcome == "worked" else f"DONE ({outcome})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Friday.")
    ap.add_argument("--check", action="store_true",
                    help="report what is ready, then exit")
    ap.add_argument("--once", action="store_true", help="one turn, then exit")
    ap.add_argument("--say", metavar="TEXT",
                    help="skip the microphone and ask this directly")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    try:
        if args.check:
            return asyncio.run(check())
        return asyncio.run(run(once=args.once, typed=args.say))
    except KeyboardInterrupt:
        print()
        log.info("stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
