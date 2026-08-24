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
import logging
import sys
import threading
import time

import config

log = logging.getLogger("friday")


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
            self.face.set("speaking")
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


async def one_turn(brain_obj, session, voice: Voice, face: Face,
                   text: str) -> str:
    """Think about one thing and say the answer."""
    import brain as brain_module

    face.set("thinking")
    log.info("asking: %r", text)
    try:
        reply = await brain_obj.ask(
            text,
            on_sentence=voice.say,
            on_tool=lambda name, args: log.info("  tool %s %s", name, args),
            session=session,
        )
    except brain_module.SpendCapReached as exc:
        log.warning("spend cap: %s", exc)
        voice.say("I've hit my spending limit for today.")
        face.feel("sad")
        voice.wait_until_quiet()
        face.rest()
        return ""
    except Exception as exc:                # noqa: BLE001
        # Every stage failure lands here, flashes dizzy, and the loop carries
        # on listening. A companion that dies on one bad turn is worse than
        # one that occasionally admits it lost the thread.
        log.error("the brain failed: %s: %s", type(exc).__name__, exc)
        face.alarm("dizzy")
        voice.say("Something went wrong in my head just then.")
        voice.wait_until_quiet()
        face.rest()
        return ""

    voice.wait_until_quiet()
    if reply.strip():
        face.set("success")
    face.rest()
    return reply


async def run(once: bool = False, typed: str | None = None) -> int:
    import brain as brain_module
    import stt
    import wake_word

    face = Face()
    face.set(config.FACE_DEFAULT_EMOTION)
    voice = Voice(face).start()
    preload_models()

    # A MISSING API KEY MUST NOT BE FATAL. This runs under systemd with
    # Restart=always, so exiting here would crash-loop every few seconds and
    # look, from the outside, exactly like a broken robot. It is also the
    # normal state before the key is set. Friday comes up deaf to reasoning
    # but still awake: the wake word works, and the offline keyword handler
    # still drives, stops, turns and tells the time.
    brain_obj = None
    try:
        brain_obj = brain_module.Brain()
    except brain_module.BrainError as exc:
        log.warning("no brain (%s) — running on the offline handler only. "
                    "Set ANTHROPIC_API_KEY and restart to enable it.", exc)

    async with brain_module.mcp_session() as session:
        listed = await session.list_tools()
        log.info("connected to %s — %d tools", config.MCP_SERVER_NAME,
                 len(listed.tools))

        if typed is not None:
            await one_turn(brain_obj, session, voice, face, typed)
            voice.close()
            return 0

        # ONE microphone stream, shared by the wake detector and the capture.
        # Opened here so both get the same one -- see stt.AudioStream for why
        # a second reader on this USB card is not an option.
        stream = stt.AudioStream().start()
        source = wake_word.make(stream=stream)
        log.info("wake source: %s. Ctrl-C to stop.", source.name)
        face.rest()

        try:
            for trigger in source.listen():
                log.info("woken (%s)", trigger)
                # Acknowledge instantly: the face and the beep both land
                # before any model runs, so Harsh knows he was heard rather
                # than wondering whether to repeat himself.
                face.set("listening")
                if hasattr(source, "beep"):
                    source.beep()

                spoken = ""
                try:
                    # The detector is deaf while Friday listens and answers,
                    # so her own voice and the beep cannot wake her again.
                    source.pause()
                    audio = await asyncio.to_thread(stream.capture)
                    face.set("thinking")
                    spoken = await asyncio.to_thread(stt.transcribe, audio)
                except stt.Silence:
                    log.info("nothing said")
                    face.rest()
                    continue
                except Exception as exc:    # noqa: BLE001
                    log.error("could not listen: %s", exc)
                    face.alarm("dizzy")
                    face.rest()
                    continue
                finally:
                    source.resume()

                if not spoken.strip():
                    log.info("empty transcript")
                    face.feel("confused")
                    face.rest()
                    continue

                await handle(brain_obj, session, voice, face, spoken)
                if once:
                    break
        finally:
            source.close()
            stream.close()

    voice.close()
    return 0


async def handle(brain_obj, session, voice: Voice, face: Face,
                 spoken: str) -> str:
    """One utterance: reflexes first, then the brain if it can be reached.

    STOP IS CHECKED BEFORE ANYTHING ELSE, online or off. Nothing should stand
    between Harsh saying stop and the motors halting -- least of all a round
    trip to a language model.

    Falls back to the local keyword handler when there is no internet, so a
    dropped connection costs Friday its conversation but not its ability to be
    told what to do.
    """
    import offline

    if offline.is_stop(spoken):
        log.info("STOP heard — going straight to the motors")
        await _run_offline(offline.Intent(name="stop", tool="stop",
                                          say="Stopping."),
                           session, voice, face)
        return "stopped"

    # No key is the same situation as no internet, from the robot's point of
    # view: there is no reasoning available, so fall through to the reflexes
    # rather than pretending and failing.
    if brain_obj is None or not offline.online():
        intent = offline.understand(spoken)
        if intent is not None:
            log.info("offline — handling %r locally", intent.name)
            await _run_offline(intent, session, voice, face)
            return intent.say
        log.info("offline and %r is not a local command", spoken[:40])
        face.feel("sad")
        voice.say("I can't reach my brain right now. I can still drive, "
                  "stop, turn and tell you the time.")
        voice.wait_until_quiet()
        face.rest()
        return ""

    return await one_turn(brain_obj, session, voice, face, spoken)


async def _run_offline(intent, session, voice: Voice, face: Face) -> None:
    """Carry out a locally-understood command, still through the MCP server.

    Through MCP even with no internet, because that is where the safety
    interrupt lives. An offline path that drove the motors directly would be
    the one way to bypass it.
    """
    import memory

    if intent.say:
        voice.say(intent.say)
    outcome = "worked"
    if intent.tool:
        try:
            result = await session.call_tool(intent.tool, intent.args)
            text = result.content[0].text if result.content else ""
            log.info("  %s -> %s", intent.tool, text[:120])
            if "STOPPED" in text or "refused" in text:
                outcome = "partial"
        except Exception as exc:            # noqa: BLE001
            log.error("offline command failed: %s", exc)
            face.alarm("dizzy")
            outcome = "failed"
    try:
        memory.log_event(f"offline: {intent.name}", task=intent.name,
                         outcome=outcome)
    except Exception:                       # noqa: BLE001
        pass
    voice.wait_until_quiet()
    face.rest()


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
