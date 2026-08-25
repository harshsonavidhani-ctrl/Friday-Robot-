"""apps/ring.py — the sound an alarm makes.

Deliberately not a klaxon. Two rising notes, a pause, and again: enough to
wake someone, gentle enough to live with in the same room. The phrase is
synthesised once into state/ring.wav and replayed from there.

    ring.start()      # loops until told otherwise
    ring.stop()

Played with pw-play through config.SPEAKER_DEVICE, the same path as the wake
beep and Piper -- ALSA's plughw cannot be opened while the desktop holds the
card, so pipewire is the only way to the speaker on this Pi.

The playing happens in a daemon thread and every failure is swallowed: a
missing speaker must cost Friday her ring, never her face.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import subprocess
import threading
import wave

import config

log = logging.getLogger("friday.apps.ring")

_RATE = 44100
_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()


def _envelope(i: int, total: int) -> float:
    """A soft attack and a long decay, so the note sounds struck rather than
    switched on. A square-edged sine clicks, and a click is what makes an
    alarm feel like a fault."""
    attack = int(total * 0.06) or 1
    if i < attack:
        return i / attack
    return max(0.0, 1.0 - (i - attack) / (total - attack)) ** 1.6


def _recipe() -> str:
    """Everything about how the phrase sounds, as one string."""
    return json.dumps([list(config.APP_RING_NOTES_HZ), config.APP_RING_NOTE_MS,
                       config.APP_RING_GAP_MS, config.APP_RING_VOLUME])


def build(path=None) -> "object":
    """Write the phrase once, and again whenever it is retuned.

    The recipe is kept beside the WAV: without it, turning the volume up in
    config changed nothing at all, because the old file was still sitting
    there and still being played."""
    path = path or config.APP_RING_FILE
    stamp = path.with_suffix(".json")
    recipe = _recipe()
    if path.exists() and path.stat().st_size > 44:
        try:
            if stamp.read_text() == recipe:
                return path
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)

    note_len = int(_RATE * config.APP_RING_NOTE_MS / 1000)
    frames = bytearray()
    for freq in config.APP_RING_NOTES_HZ:
        for i in range(note_len):
            value = (math.sin(2 * math.pi * freq * i / _RATE)
                     * _envelope(i, note_len) * config.APP_RING_VOLUME)
            frames += struct.pack("<h", int(value * 32767))
    frames += b"\x00\x00" * int(_RATE * config.APP_RING_GAP_MS / 1000)

    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(_RATE)
        out.writeframes(bytes(frames))
    try:
        stamp.write_text(recipe)
    except OSError:
        pass
    log.info("built %s", path)
    return path


def _loop() -> None:
    try:
        path = build()
    except Exception as exc:                    # noqa: BLE001
        log.error("could not build the ring: %s", exc)
        return
    while not _stop.is_set():
        try:
            proc = subprocess.Popen(
                ["pw-play", f"--target={config.SPEAKER_DEVICE}", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, ValueError) as exc:
            log.error("pw-play would not start: %s", exc)
            return
        while proc.poll() is None:
            if _stop.wait(0.1):
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return


def start() -> None:
    """Begin ringing. Calling it twice does not ring twice."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="friday-ring",
                                   daemon=True)
        _thread.start()


def stop() -> None:
    global _thread
    with _lock:
        _stop.set()
        thread, _thread = _thread, None
    if thread is not None:
        thread.join(timeout=2.0)


def is_ringing() -> bool:
    return _thread is not None and _thread.is_alive()
