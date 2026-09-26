"""The way back: where the run started, and driving there when it ends.

Harsh, 2026-09-26: "it should be actively exploring same old behaviour just adding
the feature where it knows where it started like should return to starting position
immediately after exploration", then "joint both of them in the old exploration
model code", and "all the things it speaks and backing off, I am stalled, all the
things -- just don't change the behaviour, just add".

So nothing about exploring changes. The way back is the SAME control loop carrying
on after the timer: the policy, the local navigator, the governor, the ToF reflex,
the recovery controller (backing off, "I am stalled") and the narrator all run
exactly as they did while exploring. The one difference is the planner's target:
the start instead of a frontier (`ExplorationPlanner.go_home`).

WHY A SECOND POSE
-----------------
The run's own pose (`LocalExplorationMap.pose`) is dead reckoning from what the
wheels were told -- no encoders, no IMU. That is what the frontier channel was
trained on and it stays exactly as it is. It is not good enough to find a spot
again: the RL map built on it claimed 24 x 24 m for a 7.5 x 8.1 m flat.

`StartTracker` keeps a SECOND, scan-matched pose on its own thread: a `slam.mapper
.Mapper` that starts AT THE START (0, 0, 0) with a fresh map, and is fed the run's
own 72-beam sweeps with the dead-reckoned motion since the last one as its prior.
It steers nothing while she explores. At the end it says where the start is in the
RUN's frame -- `home_rl()` -- which is what the planner and the navigator already
steer in, so none of them needs to know a second frame exists.

It is a fresh map, not the saved one, on purpose: the start spot is the one place
it is certain of, and this room has look-alikes on the saved map (4/4 sweeps
refused from the hostel spot, 2026-09-25).
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_FRIDAY_ROOT = Path(__file__).resolve().parent.parent
if str(_FRIDAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRIDAY_ROOT))

# Close enough to call it the start. The robot is ~0.44 m across; the nav engine's
# own arrival tolerance for a round trip is the same order.
HOME_REACHED_M = 0.25
# How long the way back may take before she stops and says how far off she is.
RETURN_BUDGET_S = 120.0
# Frontier exhaustion must hold this long (and the run be this old) before it ends
# the exploring early -- the planner reports "exhausted" for a cycle or two while
# the map is still arriving.
EXHAUSTED_HOLD_S = 8.0
EXHAUSTED_MIN_RUN_S = 20.0
# While going home the start pulls harder than a frontier did: a frontier only has
# to be SEEN, the start has to be reached.
HOME_ALIGN_TOL_RAD = math.radians(12.0)
HOME_MAX_BIAS_FRACTION = 0.9
# Tracker rate. A fold of 72 beams on a small fresh map is cheap, and smaller
# motion between folds keeps each match inside the matcher's window.
TRACK_HZ = 4.0
# The first sweep is folded this many times so its walls count as walls at once.
SEED_REPEATS = 4
# Every Nth point of a sweep is matched: scoring is linear in points, and the Pi
# runs this beside a 10 Hz control loop.
POINT_STRIDE = 2
# How loosely the motion prior holds the match. slam's Mapper uses 4 cm / 3 deg
# because it trusts its own last pose; this one must be free to take back error
# that ACCUMULATED over many folds, and at 4 cm / 3 deg it never did: measured,
# a 5 cm / 5 deg offset stayed exactly 5 cm / 5 deg through four matches.
MATCH_STEP_DEG = 2.0
PRIOR_FLOOR_M = 0.15
PRIOR_FLOOR_DEG = 10.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _compose(a, b) -> tuple[float, float, float]:
    """Pose a, then b expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + b[0] * c - b[1] * s, a[1] + b[0] * s + b[1] * c,
            _wrap(a[2] + b[2]))


def _inverse(p) -> tuple[float, float, float]:
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-(p[0] * c + p[1] * s), p[0] * s - p[1] * c, _wrap(-p[2]))


def scan_to_dict(scan_m: np.ndarray, max_range: float,
                 blind: np.ndarray | None = None) -> dict[float, int]:
    """72 bins (bin i at i*360/n deg, counter-clockwise from forward) -> {deg: mm}.

    Max-range bins are "no return", not a wall 8 m away, and masked (chassis) bins
    are not the room: both are left out.
    """
    n = len(scan_m)
    keep = np.ones(n, bool)
    if blind is not None and len(blind):
        keep[np.asarray(blind, int) % n] = False
    out: dict[float, int] = {}
    for i in np.nonzero(keep)[0]:
        r = float(scan_m[i])
        if 0.05 < r < max_range - 1e-3:
            out[i * 360.0 / n] = int(r * 1000.0)
    return out


class _Folder:
    """The fold itself: where the robot is relative to the start. Pure state.

    Runs in the tracker's own PROCESS on the robot (see StartTracker), and
    in-process in the tests.
    """

    def __init__(self, max_range: float) -> None:
        self.max_range = float(max_range)
        self.matcher: "_StartMatcher | None" = None
        self.last_rl: tuple[float, float, float] | None = None

    def fold(self, scan_m, rl: tuple, blind, start_rl) -> dict | None:
        # A full-resolution sweep ({deg: mm}, robot frame) when the LiDAR thread
        # publishes one; the 72 bins otherwise.
        scan = (scan_m if isinstance(scan_m, dict)
                else scan_to_dict(scan_m, self.max_range, blind))
        if len(scan) < 12:
            return None
        t0 = time.perf_counter()
        if self.matcher is None:
            # The start: the run's own first pose, whatever frame the RL map put
            # it in, is (0, 0, 0) here.
            self.matcher = _StartMatcher(scan)
            rl0 = start_rl or rl
            if rl != rl0:
                # The run moved before the first fold: count that motion.
                self.matcher.pose = _compose(_inverse(rl0), rl)
            reason = "matched"
        else:
            d = _compose(_inverse(self.last_rl), rl)
            moved, turned = math.copysign(math.hypot(d[0], d[1]), d[0]), d[2]
            reason = self.matcher.update(scan, moved_m=moved,
                                         turned_rad=turned).get("reason", "")
        self.last_rl = rl
        return {"pose": tuple(float(v) for v in self.matcher.pose),
                "reason": reason, "ms": (time.perf_counter() - t0) * 1000.0}


class StartTracker(threading.Thread):
    """A scan-matched pose relative to where the run started. Steers nothing.

    ⚠ THE MATCHING RUNS IN ITS OWN PROCESS. The first version matched on a
    thread inside the run, and the first real run with it (2026-09-26 22:44)
    missed the 100 ms control deadline 262 times against 35 the run before,
    sensor reads went 10 -> 31 ms a tick, and she collided twice: every fold
    held the GIL the control loop and the ToF reflex needed. Now a child
    process (niced) does the matching on another core, and this thread only
    hands it sweeps and waits on a pipe, which holds no lock at all.
    """

    def __init__(self, max_range: float, hz: float = TRACK_HZ) -> None:
        super().__init__(name="way-back", daemon=True)
        self.max_range = float(max_range)
        self.period = 1.0 / hz
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._first: tuple | None = None
        self._pending: tuple | None = None
        self._folder = _Folder(max_range)            # in-process: tests only
        self._proc = None
        self.start_rl: tuple[float, float, float] | None = None
        # The latest fix: the tracker's pose (start frame) and the run's
        # dead-reckoned pose at the same instant.
        self._fix_t: tuple[float, float, float] | None = None
        self._fix_rl: tuple[float, float, float] | None = None
        self.updates = 0
        self.matched = 0
        self.last_reason = ""
        self.last_ms = 0.0
        self.error = ""

    # -------------------------------------------------------------- feed ------
    def submit(self, scan_m: np.ndarray, rl_pose, blind=None) -> None:
        """From the control loop. Copies and returns at once."""
        rl = tuple(float(v) for v in rl_pose)
        job = (dict(scan_m) if isinstance(scan_m, dict)
               else np.asarray(scan_m, np.float32).copy(), rl, blind)
        with self._lock:
            if self.start_rl is None:
                self.start_rl = rl
                # Kept apart: the start's own sweep must be the first fold even
                # if the tracker process is still importing when she sets off.
                self._first = job
            else:
                self._pending = job

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=2.0)
            except Exception:                             # noqa: BLE001
                proc.kill()

    def _next_job(self):
        with self._lock:
            if self._first is not None:
                job, self._first = self._first, None
            else:
                job, self._pending = self._pending, None
        return job

    def run(self) -> None:
        import pickle
        import subprocess
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--serve",
                 str(self.max_range)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as exc:                          # noqa: BLE001
            # Never takes the run down: without fixes, home_rl() is the
            # dead-reckoned start.
            self.error = f"tracker process did not start: {exc}"
            return
        while not self._stop_evt.is_set():
            job = self._next_job()
            if job is not None:
                try:
                    pickle.dump((*job, self.start_rl), self._proc.stdin, protocol=4)
                    self._proc.stdin.flush()
                    out = pickle.load(self._proc.stdout)
                except Exception as exc:                  # noqa: BLE001
                    if not self._stop_evt.is_set():
                        self.error = f"tracker process: {type(exc).__name__}: {exc}"
                    return
                if isinstance(out, dict) and "error" in out:
                    self.error = out["error"]
                elif out:
                    self._apply(out, job[1])
            self._stop_evt.wait(self.period)

    def _fold(self, scan_m, rl: tuple, blind) -> None:
        """In-process fold, for the tests."""
        out = self._folder.fold(scan_m, rl, blind, self.start_rl)
        if out:
            self._apply(out, rl)

    @property
    def mapper(self):
        return self._folder.matcher

    def _apply(self, out: dict, rl: tuple) -> None:
        self.last_ms = float(out["ms"])
        self.updates += 1
        self.last_reason = out["reason"]
        if self.last_reason.startswith("matched"):
            self.matched += 1
        with self._lock:
            self._fix_t = tuple(out["pose"])
            self._fix_rl = rl

    # ------------------------------------------------------------- answers ----
    def home_rl(self) -> tuple[float, float, float] | None:
        """The start pose expressed in the RUN's frame, as of the latest fix.

        start (tracker frame) = origin. The robot was at T (tracker) and R (run)
        at the same instant, so the start in the run's frame is R . T^-1.
        """
        with self._lock:
            ft, fr, s0 = self._fix_t, self._fix_rl, self.start_rl
        if ft is None or fr is None:
            return s0
        return _compose(fr, _inverse(ft))

    def snapshot(self) -> dict[str, Any]:
        return {"updates": self.updates, "matched": self.matched,
                "reason": self.last_reason, "fold_ms": round(self.last_ms, 1),
                "error": self.error, "home_rl": self.home_rl()}


def _serve(max_range: float) -> int:
    """The tracker process: pickled jobs on stdin, pickled fixes on stdout."""
    import os
    import pickle
    os.nice(10)                     # the control loop always wins the CPU
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                   # a stray print must not corrupt the pipe
    inp = sys.stdin.buffer
    folder = _Folder(max_range)
    while True:
        try:
            scan, rl, blind, start_rl = pickle.load(inp)
        except (EOFError, OSError):
            return 0
        try:
            res = folder.fold(scan, rl, blind, start_rl)
        except Exception as exc:                          # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        pickle.dump(res, out, protocol=4)
        out.flush()


def _match(field, local: np.ndarray, centre, window_m: float, window_deg: float,
           prior_m: float, prior_deg: float):
    """slam.core.match_pose's coarse search, without its sub-cell polish.

    The polish is ~340 Python-level scorings per match, and it was most of a
    75-160 ms match on the Pi -- run four times a second beside a control loop
    that already misses its 100 ms budget. Getting back to within 25 cm does
    not need sub-cell accuracy, and a cell is 5 cm. Same prior weighting and
    same raw-score sharpness as match_pose, finer angle steps (2 deg, not 3).
    """
    from slam.core import transform
    px, py, pth = centre
    half = max(int(round(window_m / field.cell)), 1)
    steps = np.arange(-half, half + 1) * field.cell
    prior_xy = np.exp(-0.5 * (steps[:, None] ** 2 + steps[None, :] ** 2)
                      / prior_m ** 2)
    sigma_th = math.radians(prior_deg)
    best, best_w, best_raw = centre, -1.0, 0.0
    spread = []
    for dth in np.radians(np.arange(-window_deg, window_deg + 1e-9, MATCH_STEP_DEG)):
        scores = field.score_shifts(transform(local, px, py, pth + dth), half)
        spread.append(scores)
        weighted = scores * prior_xy * math.exp(-0.5 * dth ** 2 / sigma_th ** 2)
        i, j = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
        if weighted[i, j] > best_w:
            best_w, best_raw = float(weighted[i, j]), float(scores[i, j])
            best = (px + (i - half) * field.cell, py + (j - half) * field.cell,
                    _wrap(pth + dth))
    everything = np.concatenate([b.ravel() for b in spread])
    median = float(np.median(everything))
    sharp = float(everything.max() / median) if median > 1e-9 else float("inf")
    return best, best_raw, sharp


class _StartMatcher:
    """Where she is relative to the start, matched against the START SPOT's own view.

    NOT a SLAM map that keeps growing. Tried first (a slam.mapper.Mapper from a
    fresh map, fed while driving): on a simulated two-minute run on the real
    room with this drivetrain's slips, it was 0.7-2.0 m out at the end -- the
    map it matched against was built from the same drifting poses, so it
    agreed with its own drift. The first sweep is taken standing at the
    start, before any error exists, so it is a reference that cannot drift:
    every later sweep is matched against it, and where it cannot be (a part of
    the room the start could not see) the dead-reckoned motion carries the
    pose until it can.
    """

    def __init__(self, first: dict) -> None:
        from slam.core import LikelihoodField, points_from_scan
        from slam.grid import OccupancyGrid
        self.grid = OccupancyGrid()
        local = points_from_scan(first, forward_deg=0.0)
        for _ in range(SEED_REPEATS):
            self.grid.integrate(local, (0.0, 0.0), occupied=local)
        self.field = LikelihoodField.from_grid(self.grid)
        # ⚠ EVERY BEAM IN THE DENOMINATOR (nav_follower.motion_fraction's
        # lesson). With the known mask, a pose that points the sweep at floor
        # the start could not see is judged on the few beams left and scores
        # ABOVE 1.0 -- measured here: 1.72 while the pose walked 25 deg off.
        self.field.known = None
        self.pose = (0.0, 0.0, 0.0)

    def update(self, scan: dict, moved_m: float, turned_rad: float) -> dict:
        import config
        from slam.core import points_from_scan
        x, y, th = self.pose
        mid = th + turned_rad / 2.0
        dead = (x + moved_m * math.cos(mid), y + moved_m * math.sin(mid),
                _wrap(th + turned_rad))
        local = points_from_scan(scan, forward_deg=0.0)[::POINT_STRIDE]
        if len(local) < config.SLAM_MATCH_MIN_BEAMS // POINT_STRIDE:
            self.pose = dead
            return {"reason": "no beams"}
        # ~22% of this drivetrain's turns execute as stops (2026-09 drivetrain
        # measurements) and dead reckoning counts them as turned, so the search
        # is centred between "all of it" and "none of it", wide enough for both.
        half_m, half_t = moved_m / 2.0, turned_rad / 2.0
        mid = th + half_t / 2.0
        centre = (x + half_m * math.cos(mid), y + half_m * math.sin(mid),
                  _wrap(th + half_t))
        turn_deg = abs(math.degrees(half_t))
        pose, score, sharp = _match(
            self.field, local, centre,
            window_m=max(config.SLAM_POSE_WINDOW_M, abs(half_m) + 0.08),
            window_deg=max(config.SLAM_POSE_WINDOW_DEG, turn_deg + 6.0),
            prior_m=max(abs(half_m) * 1.2, PRIOR_FLOOR_M),
            prior_deg=max(turn_deg * 1.2, PRIOR_FLOOR_DEG))
        if score < config.SLAM_MIN_SCORE:
            self.pose = dead
            return {"reason": "score too low", "score": score}
        if sharp < config.SLAM_POSE_MIN_SHARPNESS:
            self.pose = dead
            return {"reason": "geometry ambiguous", "score": score}
        self.pose = tuple(float(v) for v in pose)
        return {"reason": "matched", "score": score}


class HomeTarget:
    """Looks enough like a `slam.frontier.Frontier` for the planner and telemetry."""

    def __init__(self, xy) -> None:
        self.approach = (float(xy[0]), float(xy[1]))
        self.gain = 0
        self.distance = 0.0
        self.utility = 0.0

    def key(self):
        return ("home",)


class WayBack:
    """The run's end: explore until the time is up, then drive to the start.

    Owned by the control loop. `phase` is "exploring", "returning" or "done".
    """

    def __init__(self, tracker: StartTracker, budget_s: float = RETURN_BUDGET_S,
                 reached_m: float = HOME_REACHED_M) -> None:
        self.tracker = tracker
        self.budget_s = float(budget_s)
        self.reached_m = float(reached_m)
        self.phase = "exploring"
        self.why = ""
        self.started_at: float | None = None
        self.outcome = ""
        self.distance_m: float | None = None
        self._exhausted_since: float | None = None

    def exhausted_long_enough(self, exhausted: bool, run_age_s: float,
                              now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not exhausted:
            self._exhausted_since = None
            return False
        if self._exhausted_since is None:
            self._exhausted_since = now
        return (run_age_s >= EXHAUSTED_MIN_RUN_S
                and now - self._exhausted_since >= EXHAUSTED_HOLD_S)

    def begin(self, why: str, now: float | None = None) -> None:
        self.phase = "returning"
        self.why = why
        self.started_at = time.monotonic() if now is None else now

    def home(self):
        return self.tracker.home_rl()

    def distance(self, rl_pose) -> float | None:
        h = self.home()
        if h is None:
            return None
        return math.hypot(h[0] - float(rl_pose[0]), h[1] - float(rl_pose[1]))

    def check(self, rl_pose, now: float | None = None) -> str | None:
        """While returning: "arrived", "out of time", or None to keep driving."""
        now = time.monotonic() if now is None else now
        d = self.distance(rl_pose)
        self.distance_m = d
        if d is not None and d <= self.reached_m:
            return self._end("arrived")
        if self.started_at is not None and now - self.started_at >= self.budget_s:
            return self._end("out of time")
        return None

    def _end(self, outcome: str) -> str:
        self.phase = "done"
        self.outcome = outcome
        return outcome

    def snapshot(self) -> dict[str, Any]:
        return {"phase": self.phase, "why": self.why, "outcome": self.outcome,
                "distance_m": (None if self.distance_m is None
                               else round(self.distance_m, 2)),
                "tracker": self.tracker.snapshot()}


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--serve":
    sys.exit(_serve(float(sys.argv[2])))
