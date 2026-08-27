"""stt.py — speech to text, locally, with VAD endpointing.

faster-whisper transcribes (config.STT_MODEL / STT_COMPUTE_TYPE); webrtcvad
decides when the utterance ended. NO FIXED RECORD WINDOW — Friday must not cut
Harsh off mid-sentence, and must not sit recording silence either.

    with Microphone() as mic:
        audio = mic.listen()        # blocks until he stops talking
    text = transcribe(audio)

HOW THE ENDPOINTING WORKS. Audio arrives as 30 ms frames. Friday waits for
config.VAD_START_FRAMES voiced frames in a row before it believes speech has
started (one voiced frame is usually a door or a cough), then keeps recording
until config.VAD_SILENCE_MS of quiet. A rolling pre-roll buffer is prepended so
the first syllable — which happens *before* the VAD is convinced — is not lost.

CAPTURE GOES THROUGH PIPEWIRE, for the same reason playback does: pipewire holds
the USB card open exclusively while the desktop runs, so arecord on the raw ALSA
device fails with "Device or resource busy". See config.MIC_DEVICE.

THE MODEL IS LOADED ONCE. Same lesson as Piper — loading per utterance would
add seconds to every single thing Harsh says.
"""

from __future__ import annotations

import collections
import logging
import os
import shutil
import subprocess
import threading
import time
import wave

import config

log = logging.getLogger("friday.stt")

FRAME_MS = 30                       # webrtcvad accepts 10, 20 or 30 only
BYTES_PER_SAMPLE = 2


class STTError(RuntimeError):
    pass


class Silence(STTError):
    """Nobody said anything. A normal outcome, not a failure."""


# --- the model -----------------------------------------------------------

_model = None
_model_lock = threading.Lock()


def model():
    """The faster-whisper model, loaded once and kept."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError(
                f"faster-whisper is not installed in the venv ({exc})") from exc
        log.info("loading faster-whisper %s (%s)…",
                 config.STT_MODEL, config.STT_COMPUTE_TYPE)
        started = time.monotonic()
        _model = WhisperModel(config.STT_MODEL,
                              device="cpu",
                              compute_type=config.STT_COMPUTE_TYPE,
                              cpu_threads=config.STT_CPU_THREADS,
                              download_root=str(config.MODEL_DIR / "whisper"))
        log.info("faster-whisper ready in %.1fs", time.monotonic() - started)
        return _model


def preload() -> threading.Thread:
    """Load the model in the background, before the first thing Harsh says."""
    def load() -> None:
        try:
            model()
        except STTError as exc:
            log.warning("STT preload failed: %s", exc)

    thread = threading.Thread(target=load, name="stt-preload", daemon=True)
    thread.start()
    return thread


def ready() -> bool:
    return _model is not None


# --- the microphone ------------------------------------------------------

class Microphone:
    """Raw 16 kHz mono frames from the USB mic, through pipewire."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or config.MIC_DEVICE
        self.frame_bytes = int(config.MIC_SAMPLE_RATE * FRAME_MS / 1000
                               ) * BYTES_PER_SAMPLE
        self._process: subprocess.Popen | None = None

    def open(self) -> "Microphone":
        if not shutil.which("pw-cat"):
            raise STTError("pw-cat is not installed — cannot reach the mic")
        environment = dict(os.environ)
        environment.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)
        self._process = subprocess.Popen(
            ["pw-cat", "--record", "--raw", f"--target={self.device}",
             "--rate", str(config.MIC_SAMPLE_RATE), "--channels", "1",
             "--format", "s16", "--latency", "20ms", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment)
        return self

    def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self) -> "Microphone":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def frames(self):
        """Yield fixed-size frames forever."""
        if self._process is None or self._process.stdout is None:
            raise STTError("microphone is not open")
        stream = self._process.stdout
        while True:
            chunk = stream.read(self.frame_bytes)
            if not chunk or len(chunk) < self.frame_bytes:
                return
            yield chunk

    def listen(self, max_seconds: float | None = None,
               start_timeout: float | None = None) -> bytes:
        """Record one utterance, ending on silence. Raises Silence if none.

        Returns raw 16-bit PCM at config.MIC_SAMPLE_RATE.
        """
        return endpoint(self.frames(), max_seconds=max_seconds,
                        start_timeout=start_timeout)


# --- endpointing ---------------------------------------------------------

def endpoint(frames, max_seconds: float | None = None,
             start_timeout: float | None = None,
             preroll_frames: list[bytes] | None = None,
             deadtime_ms: int = 0) -> bytes:
    """Consume 30 ms frames until the speaker stops. Raises Silence if none.

    Split out of Microphone.listen so the SAME endpointing runs whether the
    frames come from a private pw-cat or from a subscription to the shared
    AudioStream. The wake path needs the latter; there must not be two
    different notions of "he stopped talking".

    preroll_frames is audio captured BEFORE this call — the tail end of the
    shared stream's backlog at the moment the wake fired. It is prepended when
    speech starts, so "hey friday move forward" said in one breath does not
    arrive as "ove forward".
    """
    try:
        import webrtcvad
    except ImportError as exc:
        raise STTError(
            f"webrtcvad is not installed in the venv ({exc})") from exc

    max_seconds = max_seconds or config.MAX_UTTERANCE_S
    start_timeout = start_timeout or config.LISTEN_START_TIMEOUT_S
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

    # THE LEVEL GATE. webrtcvad judges spectral shape and knows nothing about
    # loudness, so this room's steady fan reads as speech in 45.6% of frames
    # and the 700 ms of silence never arrives -- see VAD_SPEECH_MARGIN_DB.
    # A frame counts as speech only if it is ALSO clearly above the noise
    # floor, and the floor is estimated from frames webrtcvad calls non-speech
    # so that talking cannot drag it up and gate Harsh out mid-sentence.
    import numpy as _np
    margin = float(getattr(config, "VAD_SPEECH_MARGIN_DB", 0.0) or 0.0)

    # MINIMUM-STATISTICS NOISE FLOOR. Tracked from the very first frame, so
    # there is no bootstrap window where the gate has to trust webrtcvad
    # blindly -- that version started on the fan and closed 1.2 s later,
    # clipping the speech it was supposed to be waiting for. The floor falls
    # instantly to any quieter frame and creeps up slowly, so a burst of
    # speech cannot drag it with it, and a room that genuinely gets louder is
    # still followed within a few seconds.
    floor_db: list[float] = []          # one-element cell, closed over
    strong: list[int] = [0]             # frames since the last full-margin one

    def _level_db(buf: bytes) -> float:
        samples = _np.frombuffer(buf, dtype=_np.int16).astype(_np.float32)
        if samples.size == 0:
            return -120.0
        return 20.0 * _np.log10(max(float(_np.sqrt((samples ** 2).mean())),
                                    1.0) / 32768.0)

    def _is_speech(buf: bytes, sustaining: bool = False) -> bool:
        """webrtcvad AND clearly above the noise floor, with hysteresis.

        A high bar to START (the fan must not open a capture) and a lower one
        to SUSTAIN it. Without the second bar a single margin has to be both,
        and it cannot: 10 dB rejected the room but cut 12% off the end of a
        real phrase, because the quiet consonants at the end of a word fall
        below a threshold the room never reaches.
        """
        raw = vad.is_speech(buf, config.MIC_SAMPLE_RATE)
        if margin <= 0:
            return raw
        level = _level_db(buf)
        if not floor_db:
            floor_db.append(level)
        elif level < floor_db[0]:
            floor_db[0] = level                      # fall instantly
        else:
            floor_db[0] += config.VAD_FLOOR_RISE_DB  # creep up slowly
        if not raw:
            return False
        if level >= floor_db[0] + margin:
            strong[0] = 0                    # a full-margin frame, right now
            return True
        # BELOW THE FULL BAR. Forgivable only just after a strong frame -- the
        # quiet consonants ending a word -- and only for a bounded run of
        # them. Without the bound, room noise sitting a few dB over the floor
        # sustained a capture that a noise burst had opened, and it ran to
        # MAX_UTTERANCE_S exactly as before the gate existed: 8.8 s of
        # nothing, transcribed as ''. Counted in FRAMES, not seconds, so the
        # behaviour is identical when replayed offline from a file.
        strong[0] += 1
        if not sustaining or strong[0] > config.VAD_SUSTAIN_FRAMES:
            return False
        return level >= floor_db[0] + (margin - config.VAD_SUSTAIN_DROP_DB)

    silence_frames = max(1, int(config.VAD_SILENCE_MS / FRAME_MS))
    preroll = collections.deque(
        maxlen=max(1, int(config.VAD_PREROLL_MS / FRAME_MS)))
    if preroll_frames:
        preroll.extend(preroll_frames)

    # THE WAKE WORD'S OWN TAIL LIVES IN HERE. openWakeWord fires at the end of
    # the phrase, so speech in the first deadtime_ms is what Friday just heard
    # herself being called, not what she is being asked. Refusing to start on
    # it is what stops the capture closing before Harsh has spoken.
    deadtime_frames = int(deadtime_ms / FRAME_MS) if deadtime_ms else 0
    seen = 0

    voiced: list[bytes] = []
    preroll_len = 0
    started = False
    quiet_run = 0
    voiced_run = 0
    began = time.monotonic()

    for frame in frames:
        speech = _is_speech(frame, sustaining=started)
        seen += 1
        if seen <= deadtime_frames:
            # Still inside the wake word. Keep it out of the pre-roll too, or
            # it is prepended the moment real speech does start.
            continue

        if not started:
            preroll.append(frame)
            voiced_run = voiced_run + 1 if speech else 0
            if voiced_run >= config.VAD_START_FRAMES:
                started = True
                # The pre-roll carries the first syllable, spoken before
                # the VAD was convinced. Without it "forward" arrives as
                # "orward" and the transcript is wrong in a way that is
                # very hard to debug from the text alone.
                voiced.extend(preroll)
                # Remembered so the length test below can measure what was
                # SAID, not what was carried in ahead of it -- the pre-roll is
                # up to 300 ms on its own and would mask every short blip.
                preroll_len = len(voiced)
                log.debug("speech started")
            elif time.monotonic() - began > start_timeout:
                raise Silence(
                    f"nothing said within {start_timeout:.0f}s")
            continue

        voiced.append(frame)
        quiet_run = 0 if speech else quiet_run + 1
        if quiet_run >= silence_frames:
            # TOO SHORT TO BE A COMMAND? Then it was the wake word's own tail
            # arriving through the pre-roll, not Harsh. Drop it and keep
            # listening, rather than returning a capture of "hey Friday" and
            # calling the turn done -- which is what made a command spoken
            # after a natural pause vanish entirely.
            spoken_ms = (len(voiced) - preroll_len - quiet_run) * FRAME_MS
            if spoken_ms < config.VAD_MIN_UTTERANCE_MS:
                log.debug("discarding a %.0f ms blip (the wake word's tail) "
                          "and listening on", spoken_ms)
                voiced.clear()
                preroll.clear()
                preroll_len = 0
                started = False
                quiet_run = 0
                voiced_run = 0
                continue
            break
        if time.monotonic() - began > max_seconds:
            log.info("hit the %.0fs utterance limit", max_seconds)
            break

    if not voiced:
        raise Silence("no speech captured")
    return b"".join(voiced)


# --- one microphone, many listeners --------------------------------------

class Subscription:
    """A consumer's private view of the shared microphone.

    Backed by a bounded deque, so a slow consumer drops its OLDEST audio
    instead of stalling the reader thread. That trade is deliberate: if the
    wake detector ever falls behind, the newest audio is the audio that
    matters, and blocking the reader would stall every other consumer too.
    """

    def __init__(self, seconds: float, name: str = "sub") -> None:
        self.name = name
        self._frames: collections.deque[bytes] = collections.deque(
            maxlen=max(1, int(seconds * 1000 / FRAME_MS)))
        self._ready = threading.Condition()
        self._closed = False
        # DROPS ARE COUNTED because they are otherwise invisible. A full deque
        # discards its oldest frame silently, and for the wake detector that
        # means audio torn out of the middle of a phrase -- which scores like
        # a phrase that was never said rather than like a near miss. On a Pi
        # already running at load 9 this is a real possibility, not a
        # theoretical one, so it is measured instead of assumed.
        self.dropped = 0

    def offer(self, frame: bytes) -> None:
        with self._ready:
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1
            self._frames.append(frame)
            self._ready.notify()

    def frames(self):
        """Yield frames as they arrive, forever, until the stream closes."""
        while True:
            with self._ready:
                while not self._frames and not self._closed:
                    self._ready.wait(0.5)
                if self._closed and not self._frames:
                    return
                yield self._frames.popleft()

    def drain(self) -> list[bytes]:
        """Take everything buffered right now, leaving the queue empty."""
        with self._ready:
            buffered = list(self._frames)
            self._frames.clear()
            return buffered

    def clear(self) -> None:
        with self._ready:
            self._frames.clear()

    def close(self) -> None:
        with self._ready:
            self._closed = True
            self._ready.notify_all()


class AudioStream:
    """One pw-cat process reading the mic, fanned out to every consumer.

    WHY THIS EXISTS. openWakeWord has to hear the mic continuously, and the
    VAD capture has to start the instant a wake fires. Two pw-cat processes on
    one USB card is the single-access trap the ToF, the LiDAR and the cameras
    have each already sprung on this project. And re-opening the device at wake
    time would spend a chunk of the latency budget at the worst possible
    moment — right after Harsh has said "hey Friday" and is waiting.

    So exactly ONE process reads the card, and consumers subscribe:

        stream = AudioStream().start()
        detector_sub = stream.subscribe(seconds=2.0, name="wake")
        audio = stream.capture()        # VAD-endpointed, on the same stream

    THE READER RESTARTS ITSELF. A USB mic that is unplugged, or a pipewire
    node that goes away, must not end Friday's ability to hear for the rest of
    the session — it retries with a backoff and carries on.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or config.MIC_DEVICE
        self._mic: Microphone | None = None
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._capture: Subscription | None = None
        self._watchdog: threading.Thread | None = None
        self._last_frame = time.monotonic()
        self.restarts = 0
        self.stalls = 0

    # -- lifecycle --
    def start(self) -> "AudioStream":
        if self._thread is not None:
            return self
        self._capture = self.subscribe(
            seconds=config.MIC_CAPTURE_BUFFER_S, name="capture")
        self._thread = threading.Thread(target=self._pump, name="mic-stream",
                                        daemon=True)
        self._thread.start()
        # A WEDGED pw-cat IS SILENT, NOT BROKEN, AND _pump ONLY RESTARTS ON AN
        # EXCEPTION. Measured 2026-08-28 while Friday had gone deaf after
        # answering: pw-cat alive, 0 bytes read in 5 s, 0 seconds of CPU in
        # 4m16s. It never raises, so the read below it blocks forever, every
        # subscription starves, and the wake detector blocks inside frames()
        # -- which is why it logged nothing at all, not even a low score.
        # This turns "no audio is arriving" into the exception the restart
        # path is already written to handle.
        self._watchdog = threading.Thread(target=self._watch, name="mic-watch",
                                          daemon=True)
        self._watchdog.start()
        return self

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            quiet = time.monotonic() - self._last_frame
            if quiet < config.MIC_STALL_RESTART_S:
                continue
            mic = self._mic
            if mic is None:
                continue
            log.error("no audio for %.0fs — the microphone stream is wedged, "
                      "closing it so the pump restarts it", quiet)
            self._last_frame = time.monotonic()   # one kill per stall
            self.stalls += 1
            try:
                mic.close()
            except Exception as exc:              # noqa: BLE001
                log.error("could not close the wedged stream: %s", exc)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for sub in self._subs:
                sub.close()
        if self._mic is not None:
            self._mic.close()
            self._mic = None

    def __enter__(self) -> "AudioStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- subscribers --
    def subscribe(self, seconds: float = 2.0, name: str = "sub") -> Subscription:
        sub = Subscription(seconds, name)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        sub.close()

    # -- the reader --
    def _pump(self) -> None:
        backoff = config.MIC_RESTART_BACKOFF_S
        while not self._stop.is_set():
            try:
                self._mic = Microphone(self.device).open()
                log.info("microphone stream open on %s", self.device)
                backoff = config.MIC_RESTART_BACKOFF_S
                for frame in self._mic.frames():
                    if self._stop.is_set():
                        return
                    with self._lock:
                        subs = list(self._subs)
                    self._last_frame = time.monotonic()
                    for sub in subs:
                        sub.offer(frame)
                # frames() returning means the process ended — fall through
                # to the restart path rather than silently going deaf.
                raise STTError("pw-cat ended")
            except Exception as exc:                # noqa: BLE001
                if self._stop.is_set():
                    return
                self.restarts += 1
                log.error("microphone stream died (%s) — restarting in %.1fs",
                          exc, backoff)
            finally:
                if self._mic is not None:
                    self._mic.close()
                    self._mic = None
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, config.MIC_RESTART_BACKOFF_MAX_S)

    # -- capture --
    def capture(self, max_seconds: float | None = None,
                start_timeout: float | None = None,
                handover_ms: int | None = None,
                deadtime_ms: int | None = None) -> bytes:
        """VAD-endpoint one utterance from the shared stream.

        The capture subscription exists from start-up and is never drained
        between turns, so at the moment a wake fires it already holds the last
        few seconds of audio. The tail of that backlog is handed to the
        endpointer as pre-roll; the rest is discarded (it is the wake word and
        whatever came before it).
        """
        if self._capture is None:
            raise STTError("audio stream is not started")
        handover_ms = (config.WAKE_HANDOVER_MS if handover_ms is None
                       else handover_ms)
        deadtime_ms = (config.WAKE_DEADTIME_MS if deadtime_ms is None
                       else deadtime_ms)
        keep = max(0, int(handover_ms / FRAME_MS))
        backlog = self._capture.drain()
        return endpoint(self._capture.frames(),
                        max_seconds=max_seconds,
                        start_timeout=start_timeout,
                        preroll_frames=backlog[-keep:] if keep else None,
                        deadtime_ms=deadtime_ms)


# --- transcription -------------------------------------------------------

# ONE TRANSCRIPTION AT A TIME, ACROSS THE WHOLE PROCESS.
#
# asyncio.wait_for CANNOT KILL A THREAD. When the orchestrator's STT stage
# times out, the whisper call underneath it keeps running to completion,
# holding its cpu_threads the entire time. The next turn then starts a SECOND
# transcription alongside the abandoned one, which is slower, so it times out
# too, and leaves a third. Measured on this Pi as exactly that spiral: turn 1
# finished in 22 s, turn 2 timed out at 30 s, turn 3 timed out having taken
# even longer -- and from the sofa every one of them is "I said hey Friday and
# she did nothing".
#
# This lock does not make the abandoned work stop; nothing can. What it does
# is stop it being MULTIPLIED, so a slow Pi degrades instead of collapsing.
_transcribe_lock = threading.Lock()


def transcribe(audio: bytes, language: str | None = None) -> str:
    """Raw PCM -> text. Returns "" when nothing intelligible was said."""
    if not audio:
        return ""
    import numpy as np

    # faster-whisper wants float32 in [-1, 1]; the mic gives int16.
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    started = time.monotonic()
    waited = 0.0
    if not _transcribe_lock.acquire(timeout=config.STT_QUEUE_TIMEOUT_S):
        log.error("another transcription is still running after %.0fs — "
                  "skipping this one rather than piling on",
                  config.STT_QUEUE_TIMEOUT_S)
        return ""
    waited = time.monotonic() - started
    try:
        segments, info = model().transcribe(
            samples,
            language=language or config.STT_LANGUAGE,
            beam_size=config.STT_BEAM_SIZE,
            vad_filter=False,           # webrtcvad already endpointed this
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        _transcribe_lock.release()
    # The ratio is logged because it is the number that decides whether the
    # stage ceiling is sane. x3 is this Pi idle; x9 is it fighting the face.
    heard_s = len(samples) / config.MIC_SAMPLE_RATE
    took = time.monotonic() - started
    log.info("heard (%s, %.2f) in %.1fs for %.1fs of audio (x%.1f realtime%s): %r",
             info.language, info.language_probability, took, heard_s,
             took / max(heard_s, 0.01),
             f", {waited:.1f}s queued" if waited > 0.5 else "", text)
    return text


def save_wav(audio: bytes, path) -> None:
    """For debugging a transcript that came out wrong."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(BYTES_PER_SAMPLE)
        out.setframerate(config.MIC_SAMPLE_RATE)
        out.writeframes(audio)


def listen_and_transcribe() -> str:
    with Microphone() as mic:
        audio = mic.listen()
    return transcribe(audio)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Listen, then print what you said.")
    ap.add_argument("--save", help="also write the audio to this WAV")
    ap.add_argument("--loop", action="store_true", help="keep listening")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    model()                             # load before prompting, not after
    while True:
        print("\nspeak now…")
        try:
            with Microphone() as mic:
                audio = mic.listen()
        except Silence as exc:
            print(f"  ({exc})")
            if args.loop:
                continue
            return 0
        except STTError as exc:
            print(f"error: {exc}")
            return 1
        if args.save:
            save_wav(audio, args.save)
        started = time.monotonic()
        text = transcribe(audio)
        print(f"  {len(audio) / 2 / config.MIC_SAMPLE_RATE:.1f}s audio, "
              f"transcribed in {time.monotonic() - started:.1f}s")
        print(f"  >> {text!r}")
        if not args.loop:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
