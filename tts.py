"""tts.py — Friday's voice: Piper through the USB soundcard.

Speaks sentence by sentence, starting on the FIRST one rather than waiting for
the whole reply to be synthesised. On a Pi that is the difference between a
reply that feels instant and one that feels broken.

DEBUG SILENCE. config.DEBUG_SILENT_SPEECH suppresses every word at the lowest
level in this file, not at the call sites. That placement is deliberate: a
caller that forgets to check the flag still cannot make the robot talk, so
testing tool-use flows never means being talked over.

PLAYBACK GOES THROUGH PIPEWIRE. pipewire holds the USB card open exclusively
while the desktop is running, so aplay on the raw ALSA device fails with
"Device or resource busy". config.SPEAKER_DEVICE is a pipewire node name; the
ALSA fallback is only for a headless boot with no audio server.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import config

log = logging.getLogger("friday.tts")

# Sentence-ish split. Deliberately crude: the goal is to start speaking sooner,
# not to be a linguist, and an over-eager split costs a short pause while an
# under-eager one costs seconds of silence.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

_speak_lock = threading.Lock()      # one voice, one utterance at a time
_engine: "Engine | None" = None


class TTSError(RuntimeError):
    pass


class Engine:
    """A way of turning text into a WAV file."""

    name = "none"

    def available(self) -> bool:
        raise NotImplementedError

    def synthesise(self, text: str, path: Path) -> None:
        raise NotImplementedError


class PiperEngine(Engine):
    """Piper — the target voice. Natural, local, and fast enough on a Pi."""

    name = "piper"

    def __init__(self) -> None:
        self.binary = shutil.which("piper") or shutil.which(
            "piper-tts") or str(Path(config.ROOT) / "venv" / "bin" / "piper")
        self.voice = config.TTS_VOICE

    def available(self) -> bool:
        return bool(self.binary and Path(self.binary).exists()
                    and self.voice and Path(str(self.voice)).exists())

    def synthesise(self, text: str, path: Path) -> None:
        subprocess.run(
            [self.binary, "--model", str(self.voice), "--output_file", str(path)],
            input=text.encode(), check=True, capture_output=True,
            timeout=config.TTS_TIMEOUT_S)


class EspeakEngine(Engine):
    """espeak-ng — robotic, but it is a voice when Piper has no model yet.

    Kept as a fallback rather than the target: it is intelligible, which is
    enough to test the pipeline end to end before a Piper voice is downloaded.
    """

    name = "espeak-ng"

    def __init__(self) -> None:
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")

    def available(self) -> bool:
        return bool(self.binary)

    def synthesise(self, text: str, path: Path) -> None:
        subprocess.run(
            [self.binary, "-w", str(path), "-s", str(config.TTS_ESPEAK_WPM), text],
            check=True, capture_output=True, timeout=config.TTS_TIMEOUT_S)


def engine() -> Engine:
    """The best voice available, chosen once and remembered."""
    global _engine
    if _engine is not None:
        return _engine
    for candidate in (PiperEngine(), EspeakEngine()):
        if candidate.available():
            _engine = candidate
            log.info("TTS engine: %s", candidate.name)
            return _engine
    raise TTSError(
        "no TTS engine available. Piper needs config.TTS_VOICE pointing at a "
        "downloaded .onnx voice; espeak-ng is not installed either.")


def engine_name() -> str:
    """Which voice would be used, without raising if there is none."""
    try:
        return engine().name
    except TTSError as exc:
        return f"none ({exc})"


def play(path: Path) -> None:
    """Play a WAV through the USB soundcard.

    pipewire first, ALSA second. Neither is allowed to hang the caller — a
    wedged audio device must not be able to freeze the robot.
    """
    environment = dict(os.environ)
    environment.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)

    if shutil.which("pw-play"):
        result = subprocess.run(
            ["pw-play", f"--target={config.SPEAKER_DEVICE}", str(path)],
            capture_output=True, env=environment, timeout=config.TTS_TIMEOUT_S)
        if result.returncode == 0:
            return
        log.warning("pw-play failed (%s), falling back to ALSA: %s",
                    result.returncode,
                    result.stderr.decode(errors="replace").strip()[:200])

    if shutil.which("aplay"):
        subprocess.run(["aplay", "-q", "-D", config.SPEAKER_DEVICE_ALSA,
                        str(path)],
                       check=True, capture_output=True,
                       timeout=config.TTS_TIMEOUT_S)
        return
    raise TTSError("no way to play audio — neither pw-play nor aplay exists")


def sentences(text: str) -> list[str]:
    """Split into speakable chunks, so the first one can start immediately."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def speak(text: str) -> str:
    """Say something out loud. Returns what actually happened.

    Blocking — it returns when Friday has finished speaking. Silenced entirely
    when config.DEBUG_SILENT_SPEECH is on.
    """
    text = (text or "").strip()
    if not text:
        return "nothing to say"

    if config.DEBUG_SILENT_SPEECH:
        log.info("[silent] would say: %s", text)
        return f"[debug silence — not spoken] {text}"

    voice = engine()
    spoken = 0
    with _speak_lock:
        for chunk in sentences(text):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = Path(tmp.name)
            try:
                voice.synthesise(chunk, path)
                play(path)
                spoken += 1
            finally:
                try:
                    path.unlink()
                except OSError:
                    pass
    return f"said {spoken} sentence{'s' if spoken != 1 else ''} via {voice.name}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Speak text through Friday.")
    ap.add_argument("text", nargs="*", default=["Friday online"])
    ap.add_argument("--engine", action="store_true", help="report and exit")
    args = ap.parse_args()

    if args.engine:
        print(engine_name())
        return 0
    try:
        print(speak(" ".join(args.text)))
    except TTSError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
