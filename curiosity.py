"""curiosity.py — step 9 §7: notice when the view changes, and look at it.

    cur = Curiosity()
    verdict = cur.consider(frame)          # frame: PIL image or HxW array
    if verdict.novel:
        turn(verdict.turn_degrees)         # ...through the guarded tools

WHAT "NOTABLY DIFFERENT FROM RECENT FRAMES" HAS TO MEAN. Not "different from
the last frame" -- a robot that is turning makes every frame different from the
last one, so that fires constantly while driving and never means anything. It
is different from ALL of the last few frames: the score is the MINIMUM distance
to anything in the ring, so a view she has just seen (including the one she is
turning back towards) is not novel, and something genuinely new is.

WHY THE FRAMES ARE NORMALISED FIRST. The up camera runs auto-exposure -- ruled
out as tunable by measurement, see [[friday-cameras-and-object-detection]] --
so the whole frame's brightness moves on its own as she turns towards or away
from a window. Comparing raw pixels makes that gain change look like the most
interesting thing in the room. Each frame is reduced to a small grid and then
standardised (subtract its mean, divide by its spread), which throws away
global brightness and keeps the pattern.

⚠ THERE IS NO CAMERA FIELD-OF-VIEW CONSTANT HERE, AND ONE MUST NOT BE INVENTED.
The same trap as the ToF's: a plausible-looking derived angle would be wrong in
a way nothing downstream could detect. `turn_degrees` is instead the horizontal
position of the change expressed as a FRACTION of the frame (-1 at the left
edge, +1 at the right) multiplied by a configured swing,
CURIOSITY_LOOK_MAX_DEG. That is a behaviour knob -- "how far to turn when the
change is at the very edge" -- and it is honest about being one. If someone
measures this camera's FOV later, that is the moment to make it geometric.

IT CANNOT OVERRIDE ANYTHING. This module returns a SUGGESTION and touches no
hardware. The caller turns through the same guarded movement tools as
everything else, so the clearance interrupt, the duration cap and stop() all
apply exactly as they already do. A curious robot is still a robot that stops.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger("friday.curiosity")


@dataclass
class Verdict:
    novel: bool
    score: float                  # 0 = identical to something recent
    turn_degrees: float           # + right, - left; 0 when not novel
    where: float                  # -1..+1, horizontal position of the change
    reason: str
    remembered: int = 0


def _reduce(frame) -> np.ndarray:
    """A frame -> a small standardised grid. Robust to gain, not to content."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim == 3:                       # colour -> luma
        arr = arr[..., :3].mean(axis=2)
    h, w = config.CURIOSITY_GRID
    # Block-mean downsample by slicing to a multiple, then reshaping. No
    # dependency on cv2/PIL here so this stays testable with plain arrays.
    ys = np.linspace(0, arr.shape[0], h + 1).astype(int)
    xs = np.linspace(0, arr.shape[1], w + 1).astype(int)
    small = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            block = arr[ys[i]:max(ys[i + 1], ys[i] + 1),
                        xs[j]:max(xs[j + 1], xs[j] + 1)]
            small[i, j] = block.mean() if block.size else 0.0
    spread = small.std()
    if spread < config.CURIOSITY_MIN_CONTRAST:
        # A blank frame -- a lens cap, a dark room, a dead camera. Standardising
        # it would amplify sensor noise into a pattern and make NOTHING look
        # like SOMETHING, which is the one failure that would have her spinning
        # at a wall all night. The guard was 1e-6 and that was FAR too lenient:
        # a frame of pure sensor noise has a tiny but non-zero spread, divides
        # by it, and comes out looking like a scene. Measured: near-flat noise
        # scored 0.80 against a 0.55 threshold -- it would have fired.
        return np.zeros_like(small)
    return (small - small.mean()) / spread


class Curiosity:
    """Remembers what the last few frames looked like and notices a new one."""

    def __init__(self, memory: int | None = None,
                 threshold: float | None = None,
                 cooldown_s: float | None = None) -> None:
        self.memory = memory or config.CURIOSITY_MEMORY
        self.threshold = (config.CURIOSITY_THRESHOLD if threshold is None
                          else threshold)
        self.cooldown_s = (config.CURIOSITY_COOLDOWN_S if cooldown_s is None
                           else cooldown_s)
        self._recent: list[np.ndarray] = []
        # None, NOT 0.0. With 0.0 the cooldown is measured from the epoch of
        # whatever clock the caller passes, so a freshly built Curiosity is
        # silently mute for its whole cooldown -- caught by the tests, which
        # pass a clock starting at 0. It has not fired yet; that is not the
        # same as having fired at time zero.
        self._last_fired: float | None = None
        self.looks = 0

    def consider(self, frame, now: float | None = None) -> Verdict:
        """Score a frame, remember it, and say whether it is worth a look."""
        now = time.monotonic() if now is None else now
        small = _reduce(frame)

        if not self._recent:
            self._remember(small)
            return Verdict(False, 0.0, 0.0, 0.0,
                           "the first frame has nothing to be different from")

        # Distance to the CLOSEST remembered frame: novel means unlike
        # everything recent, not merely unlike the last one.
        diffs = [np.abs(small - old) for old in self._recent]
        scores = [float(d.mean()) for d in diffs]
        best = int(np.argmin(scores))
        score = scores[best]
        remembered = len(self._recent)
        self._remember(small)

        if score < self.threshold:
            return Verdict(False, score, 0.0, 0.0,
                           "close enough to something I have just seen",
                           remembered)

        since = math.inf if self._last_fired is None else now - self._last_fired
        if since < self.cooldown_s:
            # Not novel ENOUGH to interrupt again this soon. Without this she
            # would look, see something new because she has turned, and look
            # again -- curiosity as an infinite loop.
            return Verdict(False, score, 0.0, 0.0,
                           f"still settling from the last look ({since:.1f}s "
                           f"of {self.cooldown_s:.0f}s)", remembered)

        # WHERE it changed: the column with the most change, as a fraction
        # either side of centre.
        per_column = diffs[best].mean(axis=0)
        columns = len(per_column)
        centre = (columns - 1) / 2.0
        peak = int(np.argmax(per_column))
        where = (peak - centre) / centre if centre > 0 else 0.0
        turn = float(np.clip(where, -1.0, 1.0)) * config.CURIOSITY_LOOK_MAX_DEG

        self._last_fired = now
        self.looks += 1
        return Verdict(True, score, round(turn, 1), round(float(where), 3),
                       "something here is not like anything I have just seen",
                       remembered)

    def _remember(self, small: np.ndarray) -> None:
        self._recent.append(small)
        if len(self._recent) > self.memory:
            self._recent.pop(0)
