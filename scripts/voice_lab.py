#!/usr/bin/env python3
"""voice_lab.py — render the same line under different tone settings and A/B them.

Timbre is not something that can be settled by reading code or by looking at a
spectrum. It has to be heard, in the room, through Friday's own speaker. So
this renders one sentence under several named presets, measures each one, and
plays them back with a spoken label so Harsh can just say which he wants.

    ./venv/bin/python scripts/voice_lab.py               # render, measure, play
    ./venv/bin/python scripts/voice_lab.py --no-play     # measure only
    ./venv/bin/python scripts/voice_lab.py --only cute   # one preset
    ./venv/bin/python scripts/voice_lab.py --text "..."  # a different line

Nothing here writes to config.py. Picking a winner is a separate, deliberate
edit -- this tool only ever produces sound and numbers.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

import config                                               # noqa: E402
import tts                                                  # noqa: E402

# Each preset is (pitch, body_dB, air_dB, gain_dB) with a note on the intent.
#
# The current voice is the "heavy" preset: it was tuned down in three steps
# (0.95 -> 0.91 -> 0.88) when the complaint was that Friday sounded thin. That
# is the setting Harsh now hears as manly, and the +3 dB of 320 Hz body is the
# prime suspect for the speaker buzzing -- a small driver cannot move that much
# low end and distorts mechanically when asked to.
PRESETS: dict[str, tuple[float, float, float, float, str]] = {
    "current": (0.88, 3.0, -1.0, 4.0,
                "what Friday sounds like now — pitched down, chest boosted"),
    "neutral": (1.00, 0.0, 0.0, 4.0,
                "hfc_female exactly as trained — no pitch shift at all"),
    "warm":    (1.00, 1.0, 1.0, 3.5,
                "natural pitch, a little body for the speaker, a little air"),
    "cute":    (1.03, 0.0, 2.0, 3.5,
                "slightly up, bright and light — the cute end"),
    "cutest":  (1.07, -1.0, 2.5, 3.0,
                "clearly higher and brighter; check it is not chipmunky"),
}


def measure(samples: np.ndarray, rate: int) -> str:
    """Peak, RMS, clipping and low-end energy — the numbers that predict buzz."""
    peak = float(np.max(np.abs(samples))) or 1e-9
    rms = float(np.sqrt(np.mean(samples ** 2))) or 1e-9
    clipped = float(np.mean(np.abs(samples) >= 0.995) * 100.0)

    # Energy below 400 Hz, as a share of the total. A small speaker cannot
    # reproduce this band; a high share is what shows up as buzzing or
    # farting rather than as warmth.
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    total = float(np.sum(spectrum)) or 1e-9
    low = float(np.sum(spectrum[freqs < 400])) / total * 100.0

    return (f"peak {20 * np.log10(peak):6.1f} dBFS   "
            f"RMS {20 * np.log10(rms):6.1f} dBFS   "
            f"clipped {clipped:4.1f}%   <400Hz {low:4.1f}%")


def render(name: str, text: str, out_dir: Path) -> Path:
    pitch, body, air, gain, _ = PRESETS[name]
    # Override the module-level knobs the shaping stages read. Restored by the
    # caller; this process is throwaway anyway.
    config.TTS_PITCH, config.TTS_BODY_DB = pitch, body
    config.TTS_AIR_DB, config.TTS_GAIN_DB = air, gain

    path = tts._render(text)                    # synthesise + shape in one go
    final = out_dir / f"voice_{name}.wav"
    final.write_bytes(path.read_bytes())
    path.unlink(missing_ok=True)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hi Harsh, I'm Friday. How do I sound?")
    ap.add_argument("--only", help="render just this preset")
    ap.add_argument("--no-play", action="store_true", help="measure only")
    args = ap.parse_args()

    names = [args.only] if args.only else list(PRESETS)
    for name in names:
        if name not in PRESETS:
            print(f"unknown preset {name!r}. Have: {', '.join(PRESETS)}")
            return 2

    out_dir = ROOT / "state" / "voice_lab"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading the voice ({config.TTS_VOICE.name})…")
    tts.voice()
    if not args.no_play:
        tts.set_output_volume()

    print(f'\nline: "{args.text}"\n')
    for name in names:
        path = render(name, args.text, out_dir)
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()),
                                 dtype=np.int16).astype(np.float32) / 32768.0
        pitch, body, air, gain, note = PRESETS[name]
        print(f"{name:8}  pitch {pitch:.2f}  body {body:+.1f}  air {air:+.1f}  "
              f"gain {gain:+.1f}")
        print(f"          {note}")
        print(f"          {measure(data, rate)}")
        print(f"          {path}")
        if not args.no_play:
            tts.play(path)
        print()

    if not args.no_play:
        print("Which one? Say the name and I will set it in config.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
