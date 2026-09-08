"""brain_target.py — the brain names something to drive at, mid-run.

    from rl import brain_target
    target = brain_target.BrainTarget()          # reads state/rl_command.json
    target.update(detector.snapshot(), now)      # ~10 Hz, cheap
    planner = brain_target.TargetedPlanner(planner, target)   # swap it in

Harsh, 2026-09-08: *"during run the face plus brain will be working because
brain will help in targetting an object and move towards it"*.

⚠ THIS FILE IMPORTS NOTHING FROM THE CONTROL LOOP, ON PURPOSE. Every rule in it
is pure and testable with a fake planner and a list of fake detections --
scripts/test_brain_target.py does exactly that, with no robot and no run. The
run is not a place where you can debug a sign error: a wrong sign here means she
drives AWAY from the thing she announced, and each attempt costs a hardware run.

WHAT IT DOES NOT DO
  * It does not choose the target. The brain does, over MCP/Haiku, and writes
    the answer to state/rl_command.json. This file only carries it.
  * It does not drive. It answers `bearing_error()` and the existing control
    loop decides what that is worth -- the frontier heading bias is capped at
    0.55 * omega_max, and the governor, escape, recovery, rear gate and the ToF
    reflex all still sit downstream and still win.
  * It does not measure distance. There is none to measure: the detector gives
    a label, a score, a box and nothing else (rl/friday_vision.py:106-118).

⚠ THE HORIZONTAL POSITION IS A DIRECTION, NOT A GEOMETRY, and that is a decision
Harsh took explicitly. rl/friday_vision.py carries a CAM_HFOV_DEG = 52.0, but it
is a SPEC figure for the C270 that nobody has measured on this camera, and its
own comment (friday_vision.py:70-72) says it is "used ONLY to turn a box centre
into a bearing for narration ... Nothing steers on it." curiosity.py:23-30 makes
the general case: a plausible-looking derived angle "would be wrong in a way
nothing downstream could detect".

So the box centre is used as a FRACTION OF THE FRAME times config.TARGET_SWING_DEG
-- the same honest shape curiosity already uses. Because the nudge is recomputed
on every detection (~1 Hz) the loop closes: if she under-turns, the next frame
still shows the object off-centre and she turns again. An error in the swing
changes how SHARPLY she turns, never which way.

SIGN CONVENTION, WRITTEN ONCE: **positive is LEFT**, matching `omega`,
`ExplorationPlanner.bearing_error()` and `Detection.bearing_deg`. An object left
of frame centre gives a positive `where` and a positive nudge.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

try:                                        # the run has config on sys.path
    import config
    _ROOT = config.ROOT
    _SWING_DEG = config.TARGET_SWING_DEG
    _TTL_S = config.TARGET_TTL_S
    _STALE_S = config.TARGET_SIGHTING_STALE_S
    _MIN_SCORE = config.TARGET_MIN_SCORE
    POLL_TICKS = config.TARGET_COMMAND_POLL_TICKS
except Exception:                           # noqa: BLE001 - importable standalone
    _ROOT = Path(__file__).resolve().parent.parent
    _SWING_DEG, _TTL_S, _STALE_S, _MIN_SCORE = 25.0, 90.0, 3.0, 0.55
    POLL_TICKS = 10

COMMAND_FILE = _ROOT / "state" / "rl_command.json"


# =====================================================================
# pure rules
# =====================================================================

def where_of(box: Any) -> float | None:
    """Box centre as a fraction of the frame: -1 hard right … +1 hard left.

    POSITIVE IS LEFT, matching omega. `box` is the detector's normalised
    (xmin, ymin, xmax, ymax); anything malformed returns None rather than a
    confident zero, because a zero here means "dead ahead" and would quietly
    steer her straight at whatever is in front.
    """
    try:
        xmin, _ymin, xmax, _ymax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= xmin <= 1.0 and 0.0 <= xmax <= 1.0) or xmax < xmin:
        return None
    centre = (xmin + xmax) / 2.0
    return -(centre - 0.5) * 2.0


def nudge_rad(where: float, swing_deg: float = _SWING_DEG) -> float:
    """Turn this much toward something at `where`. A behaviour knob, not optics."""
    return math.radians(swing_deg) * max(-1.0, min(1.0, float(where)))


def seen_from(detections: Iterable[Any]) -> list[dict]:
    """What the brain is shown. Plain data — no live objects, no invented fields.

    Deliberately NOT `bearing_deg`: publishing a number in degrees invites the
    reader to treat it as geometry, which is the one thing this must not become.
    `where` is a fraction and says so. There is no distance field because there
    is no distance; `area` is the detector's own crude "how near" and is passed
    through with its meaning unchanged.
    """
    out: list[dict] = []
    for det in detections or ():
        where = where_of(getattr(det, "box", None))
        if where is None:
            continue
        out.append({
            "label": str(getattr(det, "label", "?")),
            "score": round(float(getattr(det, "score", 0.0)), 3),
            "where": round(where, 3),
            "area": round(float(getattr(det, "area", 0.0)), 4),
        })
    return out


def pick(detections: Iterable[Any], label: str,
         min_score: float = _MIN_SCORE) -> Any | None:
    """The best current sighting of `label`, or None.

    Best = highest score, not largest box: a big low-confidence blob is how she
    ends up driving at a shadow.
    """
    want = (label or "").strip().lower()
    if not want:
        return None
    best, best_score = None, min_score
    for det in detections or ():
        if str(getattr(det, "label", "")).strip().lower() != want:
            continue
        score = float(getattr(det, "score", 0.0))
        if score >= best_score:
            best, best_score = det, score
    return best


def read_command(path: Path = COMMAND_FILE) -> dict | None:
    """The standing order, or None. NEVER raises.

    Same shape as app_control's channel: a latest-wins JSON message carrying a
    monotonic `token`. A missing, half-written or garbage file is simply "no
    order" -- this is consulted from the control loop, and a telemetry file must
    never be able to stop the robot.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    label = str(data.get("label", "")).strip()
    if not label:
        return None
    try:
        token = int(data.get("token", 0))
    except (TypeError, ValueError):
        return None
    try:
        ttl = float(data.get("ttl_s", _TTL_S))
    except (TypeError, ValueError):
        ttl = _TTL_S
    return {"label": label, "token": token, "ttl_s": max(0.0, ttl)}


# =====================================================================
# the live target
# =====================================================================

class BrainTarget:
    """The one thing the brain has asked her to approach, if any.

    Holds its OWN clock. The frontier planner abandons a target at 45 s
    (explorer.TARGET_MAX_AGE_S) and the supervisor at 25 s, and neither of them
    knows the brain meant this one -- so a brain target that took a while to
    reach would be silently dropped and counted as abandoned. It expires on
    config.TARGET_TTL_S instead, and on nothing else.
    """

    def __init__(self, command_file: Path = COMMAND_FILE,
                 swing_deg: float = _SWING_DEG,
                 ttl_s: float = _TTL_S,
                 stale_s: float = _STALE_S,
                 min_score: float = _MIN_SCORE,
                 clock: Any = time.monotonic) -> None:
        # THE CLOCK IS INJECTABLE BECAUSE THE TIMING IS THE SUBTLE PART. Going
        # quiet on a stale sighting and expiring on its own TTL are the two
        # rules that decide whether she steers at something that is no longer
        # there, and testing them against the wall clock means either sleeping
        # through them or passing `now` to some calls and not others -- which is
        # exactly the mismatch that made the first version of the wrapper test
        # fail while the code was right.
        self._clock = clock
        self.command_file = Path(command_file)
        self.swing_deg = float(swing_deg)
        self.ttl_s = float(ttl_s)
        self.stale_s = float(stale_s)
        self.min_score = float(min_score)

        self.label: str | None = None
        self.token: int = 0
        self.expires_at: float = 0.0
        self._nudge: float | None = None
        self._seen_at: float = 0.0
        self.sightings = 0
        self.orders = 0

    # -- the command channel ------------------------------------------------
    def poll(self, now: float | None = None,
             command: dict | None = None) -> bool:
        """Adopt a new order if there is one. Returns True if the target changed.

        A repeated read of the SAME token is a no-op, so polling every tick is
        free and re-reading never restarts the clock. That is the house rule for
        this kind of channel: latest wins, acted on once per unseen token
        (app_control.py:101, face_control.py:29-32).
        """
        now = self._clock() if now is None else now
        msg = read_command(self.command_file) if command is None else command
        if msg is None:
            return False
        if msg["token"] == self.token and self.label == msg["label"]:
            return False
        self.label = msg["label"]
        self.token = int(msg["token"])
        self.expires_at = now + float(msg.get("ttl_s", self.ttl_s))
        self._nudge = None
        self._seen_at = 0.0
        self.orders += 1
        return True

    def clear(self) -> None:
        self.label = None
        self._nudge = None
        self._seen_at = 0.0
        self.expires_at = 0.0

    # -- the sighting -------------------------------------------------------
    def update(self, detections: Iterable[Any],
               now: float | None = None) -> float | None:
        """Re-aim on the current frame. Returns the nudge, or None.

        THIS IS THE CLOSED LOOP. It runs on every tick the detector has anything
        new, so an under-turn is corrected by the next sighting rather than
        accumulating -- which is what makes an unmeasured field of view safe to
        act on at all.
        """
        now = self._clock() if now is None else now
        if self.label is None:
            return None
        if now >= self.expires_at:
            self.clear()
            return None
        det = pick(detections, self.label, self.min_score)
        if det is not None:
            where = where_of(getattr(det, "box", None))
            if where is not None:
                self._nudge = nudge_rad(where, self.swing_deg)
                self._seen_at = now
                self.sightings += 1
        return self.nudge(now)

    def nudge(self, now: float | None = None) -> float | None:
        """The heading nudge to apply right now, or None to leave her alone.

        Goes quiet as soon as the sighting is stale. Steering at where something
        USED to be is worse than not steering: the frontier planner would have
        had her doing something useful with those ticks.
        """
        now = self._clock() if now is None else now
        if self.label is None or self._nudge is None:
            return None
        if now >= self.expires_at:
            return None
        if now - self._seen_at > self.stale_s:
            return None
        return self._nudge

    def active(self, now: float | None = None) -> bool:
        return self.nudge(now) is not None

    def state(self, now: float | None = None) -> dict:
        """For the published state, so "why is she turning" is answerable."""
        now = self._clock() if now is None else now
        nudge = self.nudge(now)
        return {
            "label": self.label,
            "token": self.token,
            "steering": nudge is not None,
            "nudge_rad": None if nudge is None else round(nudge, 4),
            "sighting_age_s": (None if not self._seen_at
                               else round(now - self._seen_at, 2)),
            "expires_in_s": (round(max(0.0, self.expires_at - now), 1)
                             if self.label else None),
            "sightings": self.sightings,
            "orders": self.orders,
        }


# =====================================================================
# the planner wrapper
# =====================================================================

class TargetedPlanner:
    """The frontier planner, answering the brain's target when it has one.

    A TRANSPARENT PROXY RATHER THAN A SUBCLASS, so the real planner keeps
    running its own thread and choosing frontiers underneath -- the moment the
    brain's target goes quiet she carries on exploring from where she is,
    with no gap and nothing to restart.

    The control loop touches a planner in exactly two places
    (rl/pi_deploy.py:3647 and :3656) and the recovery controller reaches in for
    `_lock`, `target`, `blacklist` and `targets_abandoned` (:3861-3868). Every
    one of those is forwarded, INCLUDING WRITES -- `planner.target = None` and
    `planner.targets_abandoned += 1` must land on the real planner, or the two
    objects silently disagree about what she is doing.

    The brain's target is deliberately NOT stored in `.target`, so when recovery
    blacklists whatever it finds there it destroys a frontier and never the
    thing Harsh asked her to look at.
    """

    __slots__ = ("_inner", "_brain")

    def __init__(self, inner: Any, brain: BrainTarget) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_brain", brain)

    # -- the two methods the control loop calls -----------------------------
    def bearing_error(self, pose: Any) -> float | None:
        """The brain's nudge if it has a live sighting, else the frontier's.

        Returned in the same units the caller already expects: a signed,
        robot-relative angle, positive to the left. The loop feeds this both to
        steering_bias() -- capped at 0.55 * omega_max -- and to the DWA as
        `desired_bearing`, so one override reaches both without touching either.
        """
        nudge = self._brain.nudge()
        if nudge is not None:
            return nudge
        return self._inner.bearing_error(pose)

    def note_progress(self, pose: Any) -> None:
        """Retire a REACHED FRONTIER, never the brain's target.

        Held back while the brain is steering because arrival is measured
        against the frontier's coordinates, which have nothing to do with where
        the object is. Letting it run would retire a frontier she is not driving
        to and count it as arrived -- inflating the one number in the summary
        that is meant to be evidence that driving happened.
        """
        if self._brain.active():
            return
        self._inner.note_progress(pose)

    # -- everything else is the real planner --------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def __repr__(self) -> str:
        return f"<TargetedPlanner {self._brain.label!r} over {self._inner!r}>"
