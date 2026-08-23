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
        try:
            import webrtcvad
        except ImportError as exc:
            raise STTError(
                f"webrtcvad is not installed in the venv ({exc})") from exc

        max_seconds = max_seconds or config.MAX_UTTERANCE_S
        start_timeout = start_timeout or config.LISTEN_START_TIMEOUT_S
        vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        silence_frames = max(1, int(config.VAD_SILENCE_MS / FRAME_MS))
        preroll = collections.deque(
            maxlen=max(1, int(config.VAD_PREROLL_MS / FRAME_MS)))

        voiced: list[bytes] = []
        started = False
        quiet_run = 0
        voiced_run = 0
        began = time.monotonic()

        for frame in self.frames():
            speech = vad.is_speech(frame, config.MIC_SAMPLE_RATE)

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
                    log.debug("speech started")
                elif time.monotonic() - began > start_timeout:
                    raise Silence(
                        f"nothing said within {start_timeout:.0f}s")
                continue

            voiced.append(frame)
            quiet_run = 0 if speech else quiet_run + 1
            if quiet_run >= silence_frames:
                break
            if time.monotonic() - began > max_seconds:
                log.info("hit the %.0fs utterance limit", max_seconds)
                break

        if not voiced:
            raise Silence("no speech captured")
        return b"".join(voiced)


# --- transcription -------------------------------------------------------

def transcribe(audio: bytes, language: str | None = None) -> str:
    """Raw PCM -> text. Returns "" when nothing intelligible was said."""
    if not audio:
        return ""
    import numpy as np

    # faster-whisper wants float32 in [-1, 1]; the mic gives int16.
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model().transcribe(
        samples,
        language=language or config.STT_LANGUAGE,
        beam_size=config.STT_BEAM_SIZE,
        vad_filter=False,               # webrtcvad already endpointed this
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    log.info("heard (%s, %.2f): %r", info.language, info.language_probability,
             text)
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
