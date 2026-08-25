#!/usr/bin/env python3
"""audio_doctor.py — find out what is actually buzzing.

Speech is a terrible test signal: it moves in pitch and level constantly, so
"it buzzes" could mean clipping, aliasing, underrun, an overdriven analog
stage, or a loose driver, and you cannot tell which by ear. A pure tone can
only be one thing at a time, so this plays synthetic signals that isolate one
cause each and asks what you heard.

    ./venv/bin/python scripts/audio_doctor.py            # the whole workup
    ./venv/bin/python scripts/audio_doctor.py --test level
    ./venv/bin/python scripts/audio_doctor.py --hw-gain 20

THE TESTS

  level   One 440 Hz tone at -30, -20, -12, -6 and -1 dBFS. Every one is a
          mathematically perfect sine. If the quiet ones are clean and the
          loud ones buzz, nothing in software is at fault -- the analog stage
          or the driver is being overdriven. If ALL of them buzz equally, it
          is not level-dependent and points at the transport instead.

  sweep   20 Hz to 8 kHz, slow, at a safe -12 dBFS. Buzzing that appears only
          in a narrow band is mechanical: a panel or the driver's own surround
          resonating in the chassis. Note the pitch where it starts.

  gap     Tone, silence, tone. Underruns and device wake-up glitches live at
          the edges; a click exactly at the transitions is a buffer problem,
          not a distortion problem.

  rate    The same tone rendered at 22050 and at 48000 Hz. If only one buzzes,
          the resampler is implicated and nothing else is.

Nothing here touches the audio server's configuration. --hw-gain sets the USB
card's own playback level, which is a runtime mixer level this project already
asserts at startup in tts.set_output_volume().
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

import config                                               # noqa: E402

OUT = ROOT / "state" / "audio_doctor"


def write_wav(samples: np.ndarray, rate: int, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


def play(path: Path, latency: str | None = None) -> None:
    cmd = ["pw-play", f"--target={config.SPEAKER_DEVICE}",
           f"--latency={latency or config.AUDIO_PLAY_LATENCY}", str(path)]
    subprocess.run(cmd, capture_output=True, timeout=60)


def tone(freq: float, seconds: float, dbfs: float, rate: int) -> np.ndarray:
    """A pure sine with a short fade, so the onset itself cannot click."""
    t = np.arange(int(rate * seconds)) / rate
    amp = 10.0 ** (dbfs / 20.0)
    wave_ = np.sin(2 * math.pi * freq * t) * amp
    fade = int(rate * 0.02)
    if fade * 2 < len(wave_):
        ramp = np.linspace(0.0, 1.0, fade)
        wave_[:fade] *= ramp
        wave_[-fade:] *= ramp[::-1]
    return wave_.astype(np.float32)


def set_hw_gain(value: int) -> None:
    """Set the USB card's own playback level (0-30 on this GeneralPlus card)."""
    subprocess.run(["amixer", "-c", config.SPEAKER_ALSA_CARD, "sset",
                    "Speaker", str(value)], capture_output=True)
    print(f"  USB card hardware gain set to {value}/30")


def test_level(rate: int) -> None:
    print("\nLEVEL — one perfect 440 Hz sine, getting louder.")
    print("  Listen for the level at which it stops being clean.")
    for db in (-30, -20, -12, -6, -1):
        path = write_wav(tone(440, 1.5, db, rate), rate, f"level_{abs(db)}.wav")
        print(f"    {db:+3d} dBFS …")
        play(path)


def test_sweep(rate: int) -> None:
    print("\nSWEEP — 20 Hz to 8 kHz at a safe -12 dBFS.")
    print("  Buzzing in one narrow band is mechanical resonance, not signal.")
    seconds = 8.0
    t = np.arange(int(rate * seconds)) / rate
    # Exponential sweep: even time per octave, which is how the ear hears it.
    f0, f1 = 20.0, 8000.0
    phase = 2 * math.pi * f0 * seconds / math.log(f1 / f0) * (
        np.exp(t / seconds * math.log(f1 / f0)) - 1.0)
    samples = (np.sin(phase) * 10.0 ** (-12.0 / 20.0)).astype(np.float32)
    play(write_wav(samples, rate, "sweep.wav"))


def test_gap(rate: int) -> None:
    print("\nGAP — tone, silence, tone. A click AT THE EDGES means underrun.")
    parts = [tone(440, 1.0, -12, rate), np.zeros(int(rate * 0.5), np.float32),
             tone(440, 1.0, -12, rate)]
    play(write_wav(np.concatenate(parts), rate, "gap.wav"))


def test_rate() -> None:
    print("\nRATE — the same tone at 22050 Hz, then at 48000 Hz.")
    print("  If only the 22050 one buzzes, the resampler is the cause.")
    for rate in (22050, 48000):
        print(f"    {rate} Hz …")
        play(write_wav(tone(440, 2.0, -12, rate), rate, f"rate_{rate}.wav"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=("level", "sweep", "gap", "rate", "all"),
                    default="all")
    ap.add_argument("--hw-gain", type=int,
                    help="set the USB card's playback level (0-30) first")
    ap.add_argument("--rate", type=int, default=48000)
    args = ap.parse_args()

    print(f"speaker: {config.SPEAKER_DEVICE}")
    print(f"latency: {config.AUDIO_PLAY_LATENCY}")
    if args.hw_gain is not None:
        set_hw_gain(args.hw_gain)

    if args.test in ("level", "all"):
        test_level(args.rate)
    if args.test in ("sweep", "all"):
        test_sweep(args.rate)
    if args.test in ("gap", "all"):
        test_gap(args.rate)
    if args.test in ("rate", "all"):
        test_rate()

    print("\nWhat did you hear?")
    print("  - only the LOUD tones buzz      -> analog stage / driver overdriven")
    print("  - ALL tones buzz the same       -> transport or the card itself")
    print("  - only one PITCH buzzes         -> mechanical resonance in the chassis")
    print("  - clicks only at the GAP edges  -> buffer underrun")
    print("  - only 22050 buzzes             -> resampling")
    print("  - none of them buzz             -> it is the speech pipeline, not the audio path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
