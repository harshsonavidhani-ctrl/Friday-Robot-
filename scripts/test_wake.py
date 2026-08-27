#!/usr/bin/env python3
"""Prove the wake word actually fires — acoustically, through the real room.

    ./venv/bin/python scripts/test_wake.py            # 10 tries, speaker->mic
    ./venv/bin/python scripts/test_wake.py --live     # Harsh says it himself
    ./venv/bin/python scripts/test_wake.py -n 5

WHY SPEAKER-TO-MIC. "Say it 10 times from across the room" is the test that
matters, and it needs a person. This is the part that does NOT: Piper says the
wake phrase out loud, the microphone hears it through the air, and the whole
detection path -- stream, chunking, VAD gate, classifier, threshold, cooldown
-- is exercised for real rather than by feeding the model a file. It cannot
tell you how Friday responds to HARSH'S voice across a room, and it is not
meant to. It tells you the path is alive, which is the half that used to be
guessed at.

WHAT A FAILURE MEANS. If --live catches it and this does not, the path is fine
and the problem is the room, the distance or the voice. If neither catches it,
the problem is in here.

The phrase spoken is whatever WAKE_WORD_MODEL is trained on, NOT necessarily
"hey Friday" -- see config.WAKE_WORD_MODEL, which is still openWakeWord's
bundled hey_jarvis while friday.onnx does not exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                            # noqa: E402
import stt                                               # noqa: E402
import wake_word                                         # noqa: E402

log = logging.getLogger("test.wake")


def loudness(detector) -> str:
    """How loud the last window of microphone audio actually was, in dBFS."""
    try:
        import numpy as np
        rms = detector._peak_rms_since_report
        dropped = getattr(detector._sub, "dropped", 0)
        note = f", {dropped} frames dropped" if dropped else ""
        return f"{20 * np.log10(max(rms, 1) / 32768):.0f} dBFS{note}"
    except Exception as exc:                              # noqa: BLE001
        return f"unknown ({type(exc).__name__})"


def phrase_for_model() -> str:
    """The phrase the CURRENT model listens for."""
    name = Path(config.WAKE_WORD_MODEL).name.lower()
    for key, spoken in (("jarvis", "hey jarvis"), ("mycroft", "hey mycroft"),
                        ("alexa", "alexa"), ("marvin", "hey marvin"),
                        ("friday", "hey Friday")):
        if key in name:
            return spoken
    return "hey Friday"


def run(tries: int, live: bool, settle: float, window: float,
        threshold: float | None = None) -> int:
    detector = wake_word.OpenWakeWord(threshold=threshold)
    detector.model()
    stream = stt.AudioStream().start()
    detector.stream = stream
    detector.start()

    spoken = phrase_for_model()
    print(f"\nmodel      {Path(config.WAKE_WORD_MODEL).name}")
    print(f"phrase     {spoken!r}")
    print(f"threshold  {detector.threshold}")
    print(f"cooldown   {config.WAKE_COOLDOWN_S}s")
    if live:
        print(f"\nSay {spoken!r} when prompted. {tries} tries.\n")
    else:
        print(f"\nPiper will say {spoken!r} {tries} times through the "
              f"speaker.\n")

    caught = 0
    peaks: list[float] = []
    time.sleep(settle)

    for attempt in range(1, tries + 1):
        # Drain anything queued from the previous attempt so each try is
        # judged on its own.
        with detector._ready:
            detector._fired.clear()
        detector._peak_since_report = 0.0
        detector._peak_rms_since_report = 0.0

        if live:
            input(f"  [{attempt}/{tries}] press Enter, then say it: ")
        else:
            import tts
            tts.speak(spoken)

        deadline = time.monotonic() + window
        hit = False
        while time.monotonic() < deadline:
            with detector._ready:
                if detector._fired:
                    detector._fired.clear()
                    hit = True
                    break
            time.sleep(0.05)

        peak = detector._peak_since_report
        peaks.append(peak)
        caught += hit
        # THE MICROPHONE LEVEL IS REPORTED TOO, because a miss has two very
        # different causes and the score alone cannot tell them apart:
        # a low score with audio present means the model did not recognise it;
        # a low score with NO audio present means the sound never arrived, and
        # no threshold anywhere will ever fix that.
        level = loudness(detector)
        print(f"  [{attempt}/{tries}] {'CAUGHT' if hit else 'MISSED'}"
              f"   peak score {peak:.3f}   mic {level}")

        # THE COOLDOWN IS CLEARED BETWEEN TRIES, exactly as a finished turn
        # clears it. A run that catches the first and misses the rest is the
        # signature of a cooldown that never cleared.
        detector.resume()
        time.sleep(0.4)

    stream.close()
    detector.close()

    print(f"\n{caught}/{tries} caught")
    if peaks:
        misses = [p for p, _ in zip(peaks, range(len(peaks)))]
        print(f"peak scores: min {min(misses):.3f}  "
              f"max {max(misses):.3f}  threshold {detector.threshold}")
    if caught == tries:
        print("PASS")
        return 0
    print("FAIL — see the peak scores above. A peak just under the threshold "
          "means lower WAKE_THRESHOLD; a peak near zero means the phrase "
          "never reached the classifier.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--tries", type=int, default=10)
    ap.add_argument("--live", action="store_true",
                    help="you say it, instead of the speaker")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=6.0,
                    help="seconds to wait for a detection after each try")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override config.WAKE_THRESHOLD for this run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    return run(args.tries, args.live, args.settle, args.window,
               args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
