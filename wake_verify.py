"""wake_verify.py — a second opinion on every wake, before Friday answers it.

    ./venv/bin/python wake_verify.py clip.wav [clip.wav ...]   # would these wake her?

WHY THIS EXISTS. openWakeWord cannot tell a conversation from "hey Friday".
Over 415 voice turns (2026-09-08) 86% of fires were followed by nobody
speaking to her, and the false fires scored HIGHER than the real ones --
median 0.98 against 0.83 -- so no WAKE_THRESHOLD and no WAKE_PATIENCE can
separate them. Its classifier never heard a real conversation in training and
learned "a voice in this room", which is exactly what a conversation is.

HOW. The features under it are Google's general speech embedding and they do
carry the words; only the classifier on top was taught badly. So this is a
second classifier on the SAME 16x96 features, trained by
scripts/train_verifier.py on exactly the audio that fooled the first one. It
reads the features openWakeWord has already computed at the moment it fires,
so it costs one dot product and adds no latency before the beep. It is asked
ONLY on a fire, so it can only ever reject: a wake openWakeWord misses stays
missed.

IF IT CANNOT RUN, FRIDAY STILL WAKES. A missing or broken model accepts every
fire and says so loudly in the log: a robot that false-fires is annoying, a
robot that never wakes is broken.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from pathlib import Path

import config

log = logging.getLogger("friday.wake")


class Verifier:
    """Is this fire really "hey Friday"? Judged on openWakeWord's own features."""

    def __init__(self, path: Path | None = None,
                 threshold: float | None = None) -> None:
        self.path = Path(path or config.WAKE_VERIFIER_MODEL)
        self.threshold = (config.WAKE_VERIFY_THRESHOLD if threshold is None
                          else threshold)
        self._model: dict | None = None
        self._failed = ""
        self._lock = threading.Lock()

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import numpy as np
                with np.load(self.path) as data:
                    model = {k: np.asarray(data[k], dtype=np.float32)
                             for k in ("mean", "scale", "W0", "b0")}
                if model["mean"].shape != (16 * 96,):
                    raise ValueError(f"expects 16x96 features, file has "
                                     f"{model['mean'].shape}")
                self._model = model
                log.info("wake verifier ready (%s, threshold %.2f)",
                         self.path.name, self.threshold)
            except Exception as exc:            # noqa: BLE001
                self._failed = f"{type(exc).__name__}: {exc}"
                log.error("wake verifier UNAVAILABLE (%s) -- every fire will "
                          "be accepted unchecked, which is the old "
                          "false-fire behaviour", self._failed)
        return self._model is not None

    def probability(self, features) -> float | None:
        """0-1 that these features are "hey Friday"; None if it cannot say."""
        if not self.load():
            return None
        import numpy as np
        m = self._model
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        z = float(((x - m["mean"]) / m["scale"]) @ m["W0"][:, 0] + m["b0"][0])
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def accepts(self, features) -> tuple[bool, float | None]:
        try:
            p = self.probability(features)
        except Exception as exc:                # noqa: BLE001
            log.error("wake verifier failed (%s: %s) -- accepting this fire "
                      "unchecked", type(exc).__name__, exc)
            return True, None
        if p is None:
            return True, None
        return p >= self.threshold, p


def main() -> int:
    """Stream clips through openWakeWord as the live detector does, then verify."""
    import argparse
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    import train_verifier as tv

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")

    verifier = Verifier(threshold=args.threshold)
    tv._init(str(config.ROOT / "assets/models/wake/ambient.wav"))
    woke = 0
    for path in args.clips:
        scores, feats = tv.stream_clip(str(path))
        rows = {k: n for n, k in enumerate(feats)}
        stack = np.array(list(feats.values())) if feats else np.zeros((0, 1536))
        above, asked, verdict = 0, [], "no fire"
        for k, s in enumerate(scores):
            above = above + 1 if s >= config.WAKE_THRESHOLD else 0
            if above >= max(1, config.WAKE_PATIENCE):
                above = 0
                ok, p = verifier.accepts(stack[rows[k]])
                asked.append(p)
                if ok:
                    verdict = "WAKES"
                    break
                verdict = "rejected"
        woke += verdict == "WAKES"
        shown = ", ".join("-" if p is None else f"{p:.2f}" for p in asked)
        print(f"{verdict:9s} peak {scores.max() if len(scores) else 0:.2f}  "
              f"verifier [{shown}]  {path.name}")
    print(f"\n{woke}/{len(args.clips)} would wake her")
    return 0


if __name__ == "__main__":
    sys.exit(main())
