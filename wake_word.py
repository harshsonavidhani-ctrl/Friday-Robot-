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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._beep: Path | None = None
        self.last_score = 0.0

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
        self._paused.clear()

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
        cooldown_until = 0.0
        import numpy as np

        for frame in self._sub.frames():
            if self._stop.is_set():
                return
            if self._paused.is_set():
                buffer.clear()
                continue

            buffer.extend(frame)
            while len(buffer) >= CHUNK_BYTES:
                chunk = bytes(buffer[:CHUNK_BYTES])
                del buffer[:CHUNK_BYTES]

                now = time.monotonic()
                if now < cooldown_until:
                    continue
                try:
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    scores = self.model().predict(samples)
                except Exception as exc:        # noqa: BLE001
                    # A prediction that blows up must not end the thread —
                    # that would leave Friday permanently deaf.
                    log.error("wake scoring failed: %s", exc)
                    time.sleep(0.2)
                    continue

                score = max(scores.values()) if scores else 0.0
                self.last_score = float(score)
                if score >= self.threshold:
                    cooldown_until = now + config.WAKE_COOLDOWN_S
                    self._reset_model()
                    buffer.clear()
                    log.info("wake word detected (%.2f)", score)
                    with self._ready:
                        self._fired.append(f"{score:.2f}")
                        self._ready.notify()

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
        self._sub = self.stream.subscribe(seconds=2.0, name="wake")
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
