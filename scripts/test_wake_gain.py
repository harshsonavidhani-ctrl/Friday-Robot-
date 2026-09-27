#!/usr/bin/env python3
"""The fixed wake-scoring gain (config.WAKE_SCORE_GAIN). No microphone.

    ./venv/bin/python scripts/test_wake_gain.py

The live detector and the verifier's trainer must boost audio by the SAME
number, or the verifier is fitted on features it will never see. These cases
pin that, and that the boost cannot wrap int16 into noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                           # noqa: E402
import wake_word                                        # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


chunk = np.array([0, 100, -100, 12000, -12000, 20000, -20000, 32767, -32768],
                 np.int16)
gain = float(config.WAKE_SCORE_GAIN)
out = wake_word.score_gain(chunk)
check("the output is still int16 (what openWakeWord is fed)",
      out.dtype == np.int16 and out.shape == chunk.shape)
check(f"quiet samples are boosted x{gain:g}",
      out[1] == min(32767, int(100 * gain)) and out[2] == max(-32768, int(-100 * gain)))
check("loud samples CLIP rather than wrap around",
      out[5] == 32767 and out[6] == -32768 and out[7] == 32767 and out[8] == -32768,
      str(out))

saved = config.WAKE_SCORE_GAIN
config.WAKE_SCORE_GAIN = 1.0
check("gain 1.0 hands back the very same array (off means off)",
      wake_word.score_gain(chunk) is chunk)
config.WAKE_SCORE_GAIN = saved

import train_verifier as tv                             # noqa: E402
check("the verifier trainer boosts by the same number",
      tv.GAIN == float(config.WAKE_SCORE_GAIN))
check("...and keeps its feature cache per gain",
      (tv.CACHE.name == "verifier_features.npz") == (tv.GAIN == 1.0),
      tv.CACHE.name)
stream = chunk.copy()
trained = np.clip(stream.astype(np.float32) * tv.GAIN, -32768, 32767).astype(np.int16)
check("trainer's formula == detector's formula, sample for sample",
      np.array_equal(trained, wake_word.score_gain(chunk)))

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
