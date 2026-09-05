"""wake_word.py — the trigger that starts a turn.

One interface, several implementations, so main.py does not care which one it
got:

    for trigger in wake_word.listen():      # yields once per wake
        run_a_turn()

  * OpenWakeWord — a small classifier scoring a CONTINUOUS 16 kHz stream. The
    real one.
  * EnterKey — press Enter. Needs no model and no microphone, which makes it
    the way in when the audio path is being worked on.

WHY THE OLD ONE IS GONE. Step 1 shipped a stopgap that recorded a window,
transcribed it, and looked for "friday" in the text. It was DEAF WHILE IT
TRANSCRIBED — several seconds inside whisper per pass, during which anything
said was simply not heard — which is exactly the record->transcribe->check loop
CLAUDE.md warns against. It has been deleted rather than kept as a fallback,
because a fallback that silently drops wakes is worse than no fallback: it
makes Friday look broken and it hides the real path's failures.

HOW THIS ONE WORKS. openWakeWord turns 80 ms of audio into a score in ~17 ms on
this Pi (~21% of one core, measured with Friday's other services running), so
every single frame is scored and nothing is ever missed. Detection runs on its
own thread against a subscription to the shared microphone stream, so it keeps
scoring while main.py is busy thinking or speaking.

THREE THINGS STOP IT FIRING WHEN IT SHOULD NOT:
  * a cooldown, so one "hey Friday" cannot register two or three times as the
    score stays high across neighbouring frames;
  * a pause while Friday is speaking, so she cannot wake herself on her own
    voice coming back through the mic;
  * openWakeWord's own Silero VAD gate, so fans, music and doors never even
    reach the classifier.
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterator

import config

log = logging.getLogger("friday.wake")

# openWakeWord is trained on 80 ms chunks at 16 kHz. The shared stream hands
# out 30 ms frames (what webrtcvad needs), so frames are accumulated up to a
# chunk before scoring.
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2


class WakeSource:
    """Something that yields once per wake."""

    name = "wake"
    stream = None                       # an stt.AudioStream, when it has one

    def listen(self) -> Iterator[str]:
        raise NotImplementedError

    def pause(self) -> None:
        """Stop reacting to audio — Friday is talking."""

    def resume(self) -> None:
        """Start reacting again."""

    def sleep(self, seconds: float) -> float:
        """Ignore the wake word for a while. Returns the seconds granted."""
        return 0.0

    def wake_up(self) -> None:
        """Cancel a sleep early."""

    def sleeping_for(self) -> float:
        """Seconds of sleep left, 0 when awake."""
        return 0.0

    def close(self) -> None:
        pass


class EnterKey(WakeSource):
    """Press Enter to wake Friday. The reliable one."""

    name = "enter-key"

    def __init__(self, prompt: str = "\n[press Enter to talk to Friday] ") -> None:
        self.prompt = prompt

    def listen(self) -> Iterator[str]:
        while True:
            try:
                if not sys.stdin.readline():
                    return              # stdin closed (running as a service)
            except (KeyboardInterrupt, EOFError, ValueError):
                return
            yield "enter"


class OpenWakeWord(WakeSource):
    """Score a continuous microphone stream and fire on the wake word."""

    name = "openwakeword"

    def __init__(self, stream=None, threshold: float | None = None,
                 model_path: Path | None = None) -> None:
        self.threshold = (config.WAKE_THRESHOLD if threshold is None
                          else threshold)
        self.model_path = Path(model_path or config.WAKE_WORD_MODEL)
        self.stream = stream
        self._owns_stream = stream is None
        self._model = None
        self._sub = None
        self._fired: "list[str]" = []
        self._ready = threading.Condition()
        self._paused = threading.Event()
        self._settle_until = 0.0
        # DO NOT DISTURB, and deliberately NOT the same flag as _paused.
        # _paused is owned by the turn -- main.py sets it while Friday talks
        # and clears it when she stops -- so a sleep stored there would be
        # wiped by the next resume(), which is the turn that put her to sleep
        # in the first place. Separate gate, and resume() leaves it alone.
        self._sleep_until = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._beep: Path | None = None
        self.last_score = 0.0
        # THE COOLDOWN IS INSTANCE STATE, not a local in the detector loop,
        # because resume() has to be able to clear it. A cooldown that outlives
        # its turn is indistinguishable from a wake word that has stopped
        # working -- Friday sits there scoring frames and discarding every one
        # of them, and from across the room that is simply "she isn't hearing
        # me any more".
        self._cooldown_until = 0.0
        self._peak_since_report = 0.0
        # CONSECUTIVE CHUNKS CURRENTLY OVER THE THRESHOLD. See WAKE_PATIENCE:
        # a false fire is usually one frame, a real wake holds for about five.
        self._above = 0
        self.last_rms = 0.0
        self._peak_rms_since_report = 0.0
        # THE AUDIO THAT CAUSED THE LAST FIRE. A score in the log says a fire
        # happened; it cannot say what was actually said, and that is the only
        # thing that separates "he said hey Friday" from "he said see you
        # Friday" -- the two failures that need opposite fixes. Kept as a ring
        # of scored chunks so a fire can write out the phrase behind it, since
        # openWakeWord fires at the END and the audio is all in the past.
        self._recent: "collections.deque[bytes]" = collections.deque(
            maxlen=max(1, int(config.WAKE_FIRE_SECONDS * config.MIC_SAMPLE_RATE
                              / CHUNK_SAMPLES)))
        self.fires_saved = 0

    # -- the model --
    def model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"wake word model not found: {self.model_path}")
        from openwakeword.model import Model

        started = time.monotonic()
        self._model = Model(
            wakeword_model_paths=[str(self.model_path)],
            vad_threshold=config.WAKE_VAD_THRESHOLD,
        )
        log.info("openWakeWord ready in %.1fs (%s, threshold %.2f)",
                 time.monotonic() - started, self.model_path.name,
                 self.threshold)
        return self._model

    def preload(self) -> None:
        try:
            self.model()
        except Exception as exc:                # noqa: BLE001
            log.warning("wake model preload failed: %s", exc)

    # -- speaking gate --
    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        # Drop whatever arrived while Friday was talking, and clear the
        # classifier's internal buffer — otherwise the first scores after
        # resuming are computed over her own voice.
        if self._sub is not None:
            self._sub.clear()
        self._reset_model()
        # CLEAR THE COOLDOWN. Its only job is to stop ONE utterance scoring
        # high across neighbouring frames and firing twice. Once the turn it
        # belongs to is over that job is done, and carrying it into the next
        # turn can only ever suppress a real wake. Resuming IS the next turn.
        self._cooldown_until = 0.0
        # And the consecutive-frame run, for the same reason: it belongs to
        # the utterance that just ended. Carried into the next turn it would
        # let one leftover frame plus one new one fire a wake nobody said.
        self._above = 0
        # SETTLE BEFORE SCORING AGAIN. clear() drops what has ARRIVED, but the
        # tail of Friday's own reply is still in flight through pipewire and
        # still in the air, and it lands a few frames later. Measured: turn 6
        # woke at 1.000 exactly 0.5 s after turn 5 said LISTENING, which is
        # faster than a person can say "hey Friday". This is NOT the cooldown
        # being reinstated -- that is cleared above and stays cleared, because
        # carrying it forward suppresses real wakes. This is only the handful
        # of frames that were already on their way.
        self._settle_until = time.monotonic() + config.WAKE_RESUME_SETTLE_S
        self._paused.clear()

    def sleep(self, seconds: float) -> float:
        seconds = max(0.0, min(seconds, config.SLEEP_MAX_MINUTES * 60.0))
        self._sleep_until = time.monotonic() + seconds
        log.info("wake detector asleep for %.0fs", seconds)
        return seconds

    def wake_up(self) -> None:
        if self._sleep_until:
            log.info("wake detector awake again")
        self._sleep_until = 0.0
        self._above = 0
        if self._sub is not None:
            self._sub.clear()
        self._reset_model()

    def sleeping_for(self) -> float:
        return max(0.0, self._sleep_until - time.monotonic())

    def _reset_model(self) -> None:
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:                   # noqa: BLE001
                pass

    # -- the detector thread --
    def _detect(self) -> None:
        buffer = bytearray()
        import numpy as np

        # A STALLED DETECTOR IS SILENT, WHICH IS THE WORST WAY TO FAIL.
        # Twice on 2026-08-28 the loop stopped firing after a capture that
        # ended in Silence: the thread was alive, the mic was delivering
        # audio (-26 dBFS measured), and a wake phrase played through
        # Friday's OWN speaker got no reaction at all. Nothing was logged,
        # because scores are only logged above WAKE_LOG_SCORE_ABOVE and a
        # detector receiving no frames produces no scores. So the two states
        # that look identical from the log -- "heard nothing" and "was handed
        # nothing" -- are separated here, the same distinction last_rms
        # already makes for a single frame.
        last_frame = time.monotonic()
        last_beat = last_frame
        scored = 0
        warned = False

        for frame in self._sub.frames():
            if self._stop.is_set():
                return
            now_f = time.monotonic()
            if now_f - last_frame > config.WAKE_STALL_WARN_S and not warned:
                log.warning(
                    "wake detector got NO audio for %.0fs (paused=%s, "
                    "dropped=%d) — it was starved, not deaf",
                    now_f - last_frame, self._paused.is_set(),
                    getattr(self._sub, "dropped", -1))
                warned = True
            elif now_f - last_frame <= config.WAKE_STALL_WARN_S:
                warned = False
            last_frame = now_f

            # HEARTBEAT. The stall warning above proves audio is ARRIVING; it
            # cannot say whether the detector is looking at it. A detector
            # left paused and a detector scoring everything below
            # WAKE_LOG_SCORE_ABOVE are both perfectly silent in the log, and
            # they need opposite fixes. So every WAKE_HEARTBEAT_S of not
            # firing, say which one it is, out loud, with the numbers.
            if now_f - last_beat > config.WAKE_HEARTBEAT_S:
                asleep = max(0.0, self._sleep_until - now_f)
                log.info("wake alive: paused=%s  asleep=%.0fs  scored=%d "
                         "chunks  last_score=%.3f  last_rms=%.0f  dropped=%d",
                         self._paused.is_set(), asleep, scored,
                         self.last_score, self.last_rms,
                         getattr(self._sub, "dropped", -1))
                last_beat = now_f
                scored = 0

            # A GAP IN THE AUDIO BREAKS THE RUN. Frames either side of a
            # pause are not consecutive in any sense a wake word cares about.
            if self._paused.is_set():
                buffer.clear()
                self._above = 0
                continue
            if now_f < self._sleep_until:
                buffer.clear()
                self._above = 0
                continue
            if now_f < self._settle_until:
                buffer.clear()
                self._above = 0
                continue

            buffer.extend(frame)
            while len(buffer) >= CHUNK_BYTES:
                chunk = bytes(buffer[:CHUNK_BYTES])
                del buffer[:CHUNK_BYTES]

                now = time.monotonic()
                scored += 1
                self._recent.append(chunk)
                try:
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    scores = self.model().predict(samples)
                except Exception as exc:        # noqa: BLE001
                    # A prediction that blows up must not end the thread —
                    # that would leave Friday permanently deaf.
                    log.error("wake scoring failed: %s", exc)
                    time.sleep(0.2)
                    continue

                score = float(max(scores.values()) if scores else 0.0)
                self.last_score = score
                self._peak_since_report = max(self._peak_since_report, score)
                # HOW LOUD THE AUDIO ACTUALLY WAS. A low score has two
                # completely different causes and the score alone cannot
                # separate them: the model did not recognise what it heard, or
                # nothing reached it at all. The second is a dead microphone,
                # a muted card or a dropped stream, and no threshold anywhere
                # will ever fix it.
                self.last_rms = float(np.sqrt(
                    (samples.astype(np.float32) ** 2).mean()))
                self._peak_rms_since_report = max(
                    self._peak_rms_since_report, self.last_rms)

                # SCORING CONTINUES THROUGH THE COOLDOWN, and only the FIRING
                # is suppressed. Skipping predict() during the cooldown, as
                # this used to, left the classifier's internal buffer stale
                # and hid exactly the frames worth seeing -- the ones that
                # were loud enough to fire and were thrown away.
                cooling = now < self._cooldown_until
                if score >= config.WAKE_LOG_SCORE_ABOVE:
                    note = ""
                    if score >= self.threshold:
                        note = "  <== over threshold"
                        if not confirmed:
                            note += (f", frame {self._above} of "
                                     f"{config.WAKE_PATIENCE} — not confirmed")
                        if cooling:
                            note += (f", SUPPRESSED with "
                                     f"{self._cooldown_until - now:.1f}s "
                                     f"of cooldown left")
                    log.info("wake score %.3f%s", score, note)

                # THE SHAPE, NOT JUST THE HEIGHT. The score alone cannot
                # tell a conversation from a wake word -- both reach 1.00 --
                # but a conversation reaches it for ONE frame and a wake word
                # holds it. Counted here rather than through openWakeWord's
                # own `patience` argument, because that zeroes the returned
                # score and would blind the logging above, which is the only
                # view into why it fired.
                self._above = self._above + 1 if score >= self.threshold else 0
                confirmed = self._above >= max(1, config.WAKE_PATIENCE)

                if confirmed and not cooling:
                    self._above = 0
                    self._cooldown_until = now + config.WAKE_COOLDOWN_S
                    self._reset_model()
                    audio = b"".join(self._recent)
                    buffer.clear()
                    log.info("wake word detected (%.2f)", score)
                    with self._ready:
                        self._fired.append(f"{score:.2f}")
                        self._ready.notify()
                    # AFTER the notify, never before it. The turn starts on
                    # that notify and a disk write in front of it is latency
                    # Harsh hears as Friday being slow to answer.
                    self._save_fire(audio, score)

    # -- hard negatives --
    def _save_fire(self, audio: bytes, score: float) -> None:
        """Write the audio behind a fire, so it can be judged and trained on.

        EVERY fire, not just the wrong ones. Which ones were wrong is a
        question about what Harsh actually said, and the detector has no way
        to know -- labelling that here would only record the model's own
        opinion back to itself. `--review-fires` asks a person.

        Never allowed to fail a wake: the pile is for later, the turn is now.
        """
        if not config.WAKE_SAVE_FIRES or not audio:
            return
        try:
            directory = Path(config.WAKE_FIRE_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = directory / f"fire-{stamp}-{score:.2f}.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(config.MIC_SAMPLE_RATE)
                out.writeframes(audio)
            self.fires_saved += 1
            self._trim_fires(directory)
        except Exception as exc:                # noqa: BLE001
            log.debug("could not save the wake fire: %s", exc)

    @staticmethod
    def _trim_fires(directory: Path) -> None:
        """Keep the newest WAKE_FIRE_KEEP clips. This runs on an SD card."""
        clips = sorted(directory.glob("fire-*.wav"))
        for stale in clips[:-config.WAKE_FIRE_KEEP]:
            try:
                stale.unlink()
            except OSError:
                pass

    # -- the beep --
    def _beep_file(self) -> Path | None:
        """A short tone, rendered once and reused."""
        if self._beep is not None:
            return self._beep
        if not config.WAKE_BEEP:
            return None
        try:
            import numpy as np

            rate = config.MIC_SAMPLE_RATE
            count = int(rate * config.WAKE_BEEP_MS / 1000)
            t = np.arange(count) / rate
            tone = np.sin(2 * np.pi * config.WAKE_BEEP_HZ * t)
            # Fade both ends, or the discontinuity clicks louder than the tone.
            fade = max(1, count // 8)
            envelope = np.ones(count)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            samples = (tone * envelope * 0.35 * 32767).astype(np.int16)

            handle, name = tempfile.mkstemp(suffix=".wav", prefix="friday-beep-")
            os.close(handle)
            path = Path(name)
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(samples.tobytes())
            self._beep = path
            return path
        except Exception as exc:                # noqa: BLE001
            log.debug("could not build the wake beep: %s", exc)
            return None

    def beep(self) -> None:
        path = self._beep_file()
        if path is None:
            return
        try:
            environment = dict(os.environ)
            environment.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)
            subprocess.Popen(
                ["pw-play", f"--target={config.SPEAKER_DEVICE}", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=environment)
        except Exception as exc:                # noqa: BLE001
            log.debug("could not play the wake beep: %s", exc)

    # -- lifecycle --
    def start(self) -> "OpenWakeWord":
        import stt

        self.model()                            # fail loudly before the thread
        if self.stream is None:
            self.stream = stt.AudioStream().start()
        self._sub = self.stream.subscribe(seconds=config.MIC_WAKE_BUFFER_S,
                                          name="wake")
        self._thread = threading.Thread(target=self._run, name="wake-detect",
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        """The detector thread, restarted if it ever falls over."""
        while not self._stop.is_set():
            try:
                self._detect()
            except Exception as exc:            # noqa: BLE001
                log.error("wake detector died (%s) — restarting", exc)
            if self._stop.wait(0.5):
                return

    def listen(self) -> Iterator[str]:
        if self._thread is None:
            self.start()
        while not self._stop.is_set():
            with self._ready:
                while not self._fired and not self._stop.is_set():
                    self._ready.wait(0.5)
                if self._stop.is_set():
                    return
                score = self._fired.pop(0)
                # Anything queued behind this is the same utterance scoring
                # high on neighbouring frames. Drop it.
                self._fired.clear()
            yield score

    def close(self) -> None:
        self._stop.set()
        with self._ready:
            self._ready.notify_all()
        if self._owns_stream and self.stream is not None:
            self.stream.close()
        if self._beep is not None:
            try:
                self._beep.unlink(missing_ok=True)
            except OSError:
                pass


class AnyOf(WakeSource):
    """Several sources at once — the first to fire wakes Friday."""

    name = "any"

    def __init__(self, *sources: WakeSource) -> None:
        self.sources = sources
        for source in sources:
            if source.stream is not None:
                self.stream = source.stream

    def listen(self) -> Iterator[str]:
        import queue

        fired: queue.Queue = queue.Queue()
        stop = threading.Event()

        def pump(source: WakeSource) -> None:
            try:
                for trigger in source.listen():
                    if stop.is_set():
                        return
                    fired.put(f"{source.name}: {trigger}")
            except Exception as exc:            # noqa: BLE001
                log.error("wake source %s died: %s", source.name, exc)

        for source in self.sources:
            threading.Thread(target=pump, args=(source,),
                             name=f"wake-{source.name}", daemon=True).start()
        try:
            while True:
                yield fired.get()
        finally:
            stop.set()

    def pause(self) -> None:
        for source in self.sources:
            source.pause()

    def resume(self) -> None:
        for source in self.sources:
            source.resume()

    def sleep(self, seconds: float) -> float:
        return max((source.sleep(seconds) for source in self.sources),
                   default=0.0)

    def wake_up(self) -> None:
        for source in self.sources:
            source.wake_up()

    def sleeping_for(self) -> float:
        return max((source.sleeping_for() for source in self.sources),
                   default=0.0)

    def beep(self) -> None:
        for source in self.sources:
            if hasattr(source, "beep"):
                source.beep()

    def close(self) -> None:
        for source in self.sources:
            source.close()


def make(kind: str | None = None, stream=None) -> WakeSource:
    """The wake source named in config, or the one asked for."""
    kind = (kind or config.WAKE_SOURCE).strip().lower()
    if kind == "enter":
        return EnterKey()
    if kind == "voice":
        return OpenWakeWord(stream=stream).start()
    if kind == "both":
        return AnyOf(OpenWakeWord(stream=stream).start(), EnterKey())
    raise ValueError(f"unknown wake source {kind!r} — use enter, voice or both")


def listen(kind: str | None = None) -> Iterator[str]:
    """Convenience: yield once per wake, using the configured source."""
    yield from make(kind).listen()


def tune() -> int:
    """Print the live score so a threshold can be picked against the real room.

    Say the wake word a few times, then say normal sentences that are NOT the
    wake word. A good threshold sits above everything in the second group and
    below everything in the first.
    """
    import numpy as np
    import stt

    detector = OpenWakeWord()
    detector.model()
    stream = stt.AudioStream().start()
    sub = stream.subscribe(seconds=2.0, name="tune")

    print(f"model: {detector.model_path.name}")
    print(f"current threshold: {detector.threshold}")
    print("say the wake word, then talk normally. Ctrl-C to stop.\n")

    peak = 0.0
    buffer = bytearray()
    try:
        for frame in sub.frames():
            buffer.extend(frame)
            while len(buffer) >= CHUNK_BYTES:
                chunk = bytes(buffer[:CHUNK_BYTES])
                del buffer[:CHUNK_BYTES]
                samples = np.frombuffer(chunk, dtype=np.int16)
                scores = detector.model().predict(samples)
                score = max(scores.values()) if scores else 0.0
                peak = max(peak, score)
                bar = "#" * int(score * 50)
                flag = "  <== WOULD FIRE" if score >= detector.threshold else ""
                if score > 0.02:
                    print(f"{score:5.3f} {bar:<50}{flag}")
    except KeyboardInterrupt:
        print(f"\npeak score seen: {peak:.3f}")
    finally:
        stream.close()
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Wait for a wake and report it.")
    ap.add_argument("--source", default=None, choices=["enter", "voice", "both"])
    ap.add_argument("--tune", action="store_true",
                    help="print live scores to pick a threshold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    if args.tune:
        return tune()

    source = make(args.source)
    print(f"wake source: {source.name}. Ctrl-C to stop.")
    started = time.monotonic()
    count = 0
    try:
        for trigger in source.listen():
            count += 1
            print(f"  [{count}] WOKEN — {trigger}  "
                  f"(t+{time.monotonic() - started:.1f}s)")
            if hasattr(source, "beep"):
                source.beep()
    except KeyboardInterrupt:
        print(f"\n{count} wakes in {time.monotonic() - started:.0f}s")
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
