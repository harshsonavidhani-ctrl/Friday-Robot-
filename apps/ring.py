"""apps/ring.py — the sound an alarm makes.

A short struck-bell arpeggio, a pause, and again: enough to wake someone,
gentle enough to live with in the same room. The phrase is synthesised once
into state/ring.wav and replayed from there.

    ring.start()      # loops until told otherwise
    ring.stop()

WHY IT IS SYNTHESISED THE WAY IT IS. The first version was two pure sine
tones, and it sounded like a microwave finishing. A sine has no partials, so
there is nothing in it for the ear to identify as an instrument -- it reads as
"a beep". What makes a phone ringtone read as a ringtone is that it is a
STRUCK sound: a fundamental plus a stack of partials, each decaying at its own
rate, the high ones dying first. That decay profile carries almost all of the
character. So this is additive synthesis, per-partial, over a real arpeggio,
with each note ringing on into the next instead of being switched off.

Played with pw-play through config.SPEAKER_DEVICE, the same path as the wake
beep and Piper -- ALSA's plughw cannot be opened while the desktop holds the
card, so pipewire is the only way to the speaker on this Pi. It is rendered at
config.APP_RING_RATE, which IS the sink's rate, so pipewire has nothing to
resample. That matters more than it sounds like it should: this Pi runs at
load average 4+ and realtime resampling was part of what made Friday's audio
buzz.

The playing happens in a daemon thread and every failure is swallowed: a
missing speaker must cost Friday her ring, never her face.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import wave

import numpy as np

import config

log = logging.getLogger("friday.apps.ring")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()

# Semitones above C within an octave. Sharps only -- flats are not used in the
# phrase and accepting both would just be somewhere else for a typo to hide.
_SEMITONE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
             "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def note_hz(name: str) -> float:
    """"C6" -> 1046.5. A rest ("-", "", None) is 0.0.

    Scientific pitch notation, A4 = 440 Hz. Written out rather than kept as a
    table of frequencies because a phrase is far easier to retune when it
    reads as notes.
    """
    name = (name or "").strip()
    if not name or name == "-":
        return 0.0
    letter = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    try:
        semitone = _SEMITONE[letter.upper()]
        octave = int(name[len(letter):])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"not a note name: {name!r}") from exc
    # MIDI 69 is A4; C4 is MIDI 60, so octave n starts at 12 * (n + 1).
    midi = 12 * (octave + 1) + semitone
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _strike(freq: float, rate: int, length: int) -> np.ndarray:
    """One struck note: the partial stack of config.APP_RING_PARTIALS, each
    with its own amplitude and its own decay rate.

    Partials above the fundamental are found with a decaying exponential, not
    switched off with the note, which is why the sound has a tail. Partials
    that would land above Nyquist are dropped rather than allowed to alias
    back down as a whistle -- C7's fifth partial is 14 kHz, so this is not
    hypothetical at the top of the phrase.
    """
    t = np.arange(length, dtype=np.float64) / rate
    out = np.zeros(length, dtype=np.float64)
    nyquist = rate / 2.0
    for ratio, amp, decay_mult in config.APP_RING_PARTIALS:
        partial_hz = freq * ratio
        if partial_hz >= nyquist:
            continue
        tau = config.APP_RING_DECAY_S / decay_mult
        out += amp * np.exp(-t / tau) * np.sin(2 * np.pi * partial_hz * t)

    # The attack. A note that starts at full amplitude on sample zero is a
    # step, and a step is a click -- which is the one thing that makes an
    # alarm feel like a fault rather than a sound.
    attack = max(1, int(rate * config.APP_RING_STRIKE_MS / 1000))
    if attack < length:
        out[:attack] *= np.linspace(0.0, 1.0, attack)
    return out


def _recipe() -> str:
    """Everything about how the phrase sounds, as one string.

    Kept beside the WAV. Without it, retuning anything here changed nothing at
    all, because the old file was still sitting there and still being played.
    Every value the synthesis reads has to appear in this list.
    """
    return json.dumps([
        list(config.APP_RING_PHRASE), config.APP_RING_NOTE_MS,
        config.APP_RING_DECAY_S, config.APP_RING_GAP_MS,
        [list(p) for p in config.APP_RING_PARTIALS],
        config.APP_RING_STRIKE_MS, config.APP_RING_SPREAD_HZ,
        config.APP_RING_VOLUME, config.APP_RING_CEILING,
        config.APP_RING_RATE,
    ], sort_keys=True)


def build(path=None):
    """Write the phrase once, and again whenever it is retuned."""
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

    rate = int(config.APP_RING_RATE)
    step = int(rate * config.APP_RING_NOTE_MS / 1000)

    # The buffer has to outlast the last strike, or the final note is cut off
    # mid-decay and the phrase ends on a click. Five time constants is
    # inaudible-quiet (-43 dB) and cheap.
    tail = int(rate * config.APP_RING_DECAY_S * 5)
    gap = int(rate * config.APP_RING_GAP_MS / 1000)
    length = step * len(config.APP_RING_PHRASE) + tail

    # Two voices, detuned by a fraction of a hertz and panned apart. The slow
    # beating between them is heard as width and a little life, and it stops
    # the ring sounding like it is coming out of a box.
    left = np.zeros(length, dtype=np.float64)
    right = np.zeros(length, dtype=np.float64)
    spread = float(config.APP_RING_SPREAD_HZ)

    for i, name in enumerate(config.APP_RING_PHRASE):
        freq = note_hz(name)
        if freq <= 0.0:                       # a rest still takes its beat
            continue
        start = i * step
        room = length - start
        left[start:] += _strike(freq, rate, room)
        right[start:] += _strike(freq + spread, rate, room)

    phrase = np.concatenate([np.stack([left, right], axis=1),
                             np.zeros((gap, 2), dtype=np.float64)])

    # Peak-normalise to APP_RING_VOLUME (capped at APP_RING_CEILING). Notes
    # overlap by design, so their partials sum -- normalising is what lets the
    # phrase be retuned or made longer without anyone having to work out
    # whether it now clips.
    peak = float(np.max(np.abs(phrase)))
    target = min(float(config.APP_RING_VOLUME), float(config.APP_RING_CEILING))
    if peak > 0.0:
        phrase *= target / peak

    pcm = np.clip(phrase, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    try:
        stamp.write_text(recipe)
    except OSError:
        pass
    log.info("built %s (%.2fs, %d Hz, peak %.2f)",
             path, len(pcm) / rate, rate, target)
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
                ["pw-play", f"--target={config.SPEAKER_DEVICE}",
                 # A big buffer, for the same reason speech gets one: an
                 # underrun on this Pi is an audible burst, and an alarm has
                 # no latency requirement whatsoever.
                 f"--latency={config.AUDIO_PLAY_LATENCY}", str(path)],
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
