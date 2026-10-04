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

import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("way_back")
# Every this often the tracker says how far the run's dead reckoning and the
# start's view disagree about where the start is (home_rl against start_rl).
# 2026-10-04 the start's heading in the run frame slid 3.02 -> 1.29 rad over
# one 2-minute run and nothing recorded WHEN; this line is that record.
DRIFT_LOG_S = 5.0

_FRIDAY_ROOT = Path(__file__).resolve().parent.parent
if str(_FRIDAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRIDAY_ROOT))

# Close enough to call it the start. The robot is ~0.44 m across; the nav engine's
# own arrival tolerance for a round trip is the same order.
HOME_REACHED_M = 0.25
# How long the way back may take before she stops and says how far off she is.
# Harsh, 2026-09-28: "she should try try till she reaches the starting position".
# 120 s gave up while she was still working at it; the undo (rl/unstick.py) and
# her asking to be moved now handle being stuck on the way. Ten minutes is the
# battery's limit, not a planning one -- the Pi has lost power mid-run twice.
RETURN_BUDGET_S = 600.0
# Frontier exhaustion must hold this long (and the run be this old) before it ends
# the exploring early -- the planner reports "exhausted" for a cycle or two while
# the map is still arriving.
EXHAUSTED_HOLD_S = 8.0
EXHAUSTED_MIN_RUN_S = 20.0
# THE WAY BACK'S OWN COMMAND. The first real runs steered home with the
# exploring's heading nudge alone, and the policy -- which has no idea where the
# start is -- kept the speed: on 2026-09-26 23:12 it reversed on 34% of the way
# back and drove forward on 18%, the heading error never fell below ~45 deg, and
# she ended 1.05 m out after 120 s. So on the way back only, this replaces the
# POLICY's request (turn to face the start, then drive at it). Everything after
# it is unchanged and still has the last word: the arc planner, the governor,
# escape turns, the recovery controller (backing off, "I am stalled"), the rear
# gate, the ToF reflex and the rate limiter.
HOME_PIVOT_DEG = 50.0          # further off than this: turn in place first
HOME_CRUISE_MPS = 0.30         # = nav_speed_authority 0.6 x v_max, the exploring cruise
HOME_NEAR_M = 0.6
HOME_NEAR_MPS = 0.20
HOME_TURN_GAIN = 1.5           # rad/s of turn per rad of heading error
HOME_OMEGA_MAX = 1.2


def homing_command(err_rad: float | None, distance_m: float | None,
                   omega_max: float, v_floor: float = 0.0
                   ) -> tuple[float, float, str] | None:
    """(v, omega, why) toward the start, or None when there is no bearing."""
    if err_rad is None:
        return None
    side = "left" if err_rad > 0 else "right"
    if abs(err_rad) > math.radians(HOME_PIVOT_DEG):
        return (0.0, math.copysign(omega_max, err_rad),
                f"way back: turning to face the start, "
                f"{math.degrees(abs(err_rad)):.0f} deg {side}")
    v = HOME_CRUISE_MPS
    if distance_m is not None and distance_m < HOME_NEAR_M:
        v = max(HOME_NEAR_MPS, v_floor)
    omega = max(-HOME_OMEGA_MAX, min(HOME_OMEGA_MAX, HOME_TURN_GAIN * err_rad))
    return (v, omega, f"way back: driving at the start, "
                      f"{math.degrees(abs(err_rad)):.0f} deg {side}"
                      + ("" if distance_m is None else f", {distance_m:.2f} m"))


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
        self._drift_t = 0.0
        self._drift_n = (0, 0)              # (updates, matched) at the last line

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
        now = time.monotonic()
        if now - self._drift_t >= DRIFT_LOG_S:
            self._log_drift(now)

    def _log_drift(self, now: float) -> None:
        home, s0 = self.home_rl(), self.start_rl
        if home is None or s0 is None:
            return
        u0, m0 = self._drift_n
        x, y, th = self._fix_t
        LOGGER.info("tracker: start in the run frame moved %.2f m / %+.0f deg "
                    "(dead reckoning vs the start's view); here (%.2f, %.2f) %+.0f deg; "
                    "%d/%d matched since the last line; %s; fold %.0f ms",
                    math.hypot(home[0] - s0[0], home[1] - s0[1]),
                    math.degrees(_wrap(home[2] - s0[2])), x, y, math.degrees(th),
                    self.matched - m0, self.updates - u0, self.last_reason,
                    self.last_ms)
        self._drift_t, self._drift_n = now, (self.updates, self.matched)

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

    def to_rl(self, xy) -> tuple[float, float] | None:
        """FILE 21 U1: a point of the START frame (the mission's map) in the
        RUN's frame, by the same transform as home_rl(): home_rl() is (0, 0)."""
        base = self.home_rl()
        if base is None:
            return None
        p = _compose(base, (float(xy[0]), float(xy[1]), 0.0))
        return (p[0], p[1])

    def snapshot(self) -> dict[str, Any]:
        return {"updates": self.updates, "matched": self.matched,
                "reason": self.last_reason, "fold_ms": round(self.last_ms, 1),
                "error": self.error, "home_rl": self.home_rl()}


# --- FILE 21: the exploration mission's seam (Harsh approved, 2026-09-29) --------
# Two small additions, and neither changes how she explores:
#   * the tracker process PUBLISHES its latest fix + sweep (<= 1 Hz) for the
#     mission (exploration_mission/adapters.py reads it): the mission needs a
#     scan-matched pose and the sweeps to map areas, and only this process has them;
#   * `Hold`: while the mission's hold file is fresh, pi_deploy's tick holds the
#     motors at zero and decides nothing (a question, "go to X", a pause).
MISSION_DIR = _FRIDAY_ROOT / "state" / "exploration_mission"
PUBLISH_PERIOD_S = 1.0
# A hold further ahead than this is refused, however it was written: a wall clock
# stepping backwards must not turn a 3 s hold into an hour of standing still.
HOLD_MAX_AHEAD_S = 30.0
HOLD_POLL_S = 0.2
# "RETURN NOW" (Harsh approved, 2026-09-30): the mission writes return.json when
# exploring should end early ("Friday, go back", its time cap, nothing left to
# explore) and the run starts ITS OWN way back -- the same one the timer starts.
# Honoured only while fresh, and only if written after this run began, so a
# request left over from an earlier mission can never turn a new run round.
RETURN_FRESH_S = 60.0
# THE MISSION'S DRIVE ORDER (FILE 21 U1, Harsh 2026-10-03: "make exactly as
# file ... dont change the exploration behaviour"). Spec 3.2/4.11: the mission
# drives through File 14 (frontier-assist, go-to, the way home), but while the
# run is out it owns the LiDAR and the motors, so File 14 cannot. Instead the
# mission plans with File 14's planner on the map she is building and hands the
# HELD run the path, in the START frame, inside the hold file:
#     {"until": ..., "why": ..., "drive": {"id": "...", "kind": "path",
#                                          "path": [[x, y], ...], "tol": 0.3}}
#     kind "turn" (dir +1/-1): turn in place  -- rung 3's look round
#     kind "back": straight back              -- rung 4
# The run steers at it with the way back's own command (homing_command) and
# every stage below the policy still has the last word: the arc planner, the
# governor, the recovery controller, unstick, the rear gate, the ToF reflex and
# the rate limiter. Only while the hold is FRESH: the mission stopping or dying
# lapses it, and a lapsed hold is exploring again, exactly as before. No drive
# order = the plain hold = stand still, as before. Exploring never sees any of
# this: nothing here runs unless the mission wrote it.
DRIVE_KINDS = ("path", "turn", "back")
DRIVE_MAX_POINTS = 200
DRIVE_TOL_M = (0.10, 1.0)        # what a path's tolerance may be set to
# A PERSON'S PAUSE (FILE 21 U3, spec 4.8: "Friday stop" -> PAUSED, "continue"
# -> on). Written by friday-brain the moment it hears "stop" during a mission,
# cleared only by the mission (continue / go back / its end). Unlike the hold
# it NEVER LAPSES: a mission that died while she was paused must leave her
# standing, never set her off. Ignored if written before this run began (a new
# run is not frozen by an old pause). Paused past PAUSE_MAX_S, the run ENDS.
PAUSE_NAME = "pause.json"
PAUSE_MAX_S = 1800.0
DRIVE_FINAL_TOL_M = 0.10         # the last point: she closes in while the 0.35 m / 2 s arrival dwell runs


def _publish(scan, rl, blind, start_rl, res: dict, max_range: float,
             path: Path | None = None) -> bool:
    """Write the latest fix for the mission. NEVER RAISES.

    Called in the TRACKER PROCESS after the fix has already gone back to the run,
    so the control loop never waits on this write. Not fsynced: it is replaced
    every second and a power cut costing the last one loses nothing.
    """
    import json
    import os
    try:
        sweep = scan if isinstance(scan, dict) else scan_to_dict(scan, max_range, blind)
        payload = {"v": 1, "t": time.time(),
                   "pose": [float(v) for v in res["pose"]],
                   "reason": str(res.get("reason", "")),
                   "rl": [float(v) for v in rl],
                   "start_rl": None if start_rl is None else [float(v) for v in start_rl],
                   "fold_ms": round(float(res.get("ms", 0.0)), 1),
                   "scan": {str(int(round(float(k)))): int(v) for k, v in sweep.items()}}
        target = path or MISSION_DIR / "live.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, target)
        return True
    except Exception:                                     # noqa: BLE001
        return False


class Hold(threading.Thread):
    """The mission asking the run to stand still. Polled here, read in O(1).

    The mission writes {"until": <wall time>, "why": ...} and keeps refreshing it
    while it wants her still; `active()` is True only while that deadline is in
    the future (and no more than HOLD_MAX_AHEAD_S away). Nothing refreshes it =
    it lapses by itself, so a crashed mission cannot leave her frozen. No file =
    inactive, which is every run that has nothing to do with a mission.
    """

    def __init__(self, path: Path | None = None, period_s: float = HOLD_POLL_S):
        super().__init__(name="mission-hold", daemon=True)
        self.path = path or MISSION_DIR / "hold.json"
        self.period_s = float(period_s)
        self._until = 0.0                                 # time.monotonic() deadline
        self.why = ""
        self.return_path = self.path.with_name("return.json")
        self._born = time.time()                          # requests older than this: ignored
        self._return = False
        self.drive: dict | None = None                    # FILE 21 U1: the drive order
        self.pause_path = self.path.with_name(PAUSE_NAME)  # FILE 21 U3
        self._paused_since: float | None = None

    def poll(self) -> None:
        import json
        try:
            data = json.loads(self.path.read_text())
            ahead = float(data["until"]) - time.time()
        except Exception:                                 # noqa: BLE001
            self._until = 0.0
            self.drive = None
            return
        if 0.0 < ahead <= HOLD_MAX_AHEAD_S:
            self.why = str(data.get("why", ""))
            self.drive = drive_order(data.get("drive"))
            self._until = time.monotonic() + ahead
        else:
            self._until = 0.0
            self.drive = None

    def poll_return(self) -> None:
        import json
        try:
            at = float(json.loads(self.return_path.read_text())["at"])
        except Exception:                                 # noqa: BLE001
            self._return = False
            return
        self._return = (at >= self._born
                        and 0.0 <= time.time() - at <= RETURN_FRESH_S)

    def poll_pause(self) -> None:
        import json
        try:
            at = float(json.loads(self.pause_path.read_text())["at"])
            paused = at >= self._born - 1.0
        except FileNotFoundError:
            paused = False
        except Exception:                                 # noqa: BLE001
            # Unreadable mid-write: keep what we had -- a pause is never
            # dropped on a torn read.
            paused = self._paused_since is not None
        if paused and self._paused_since is None:
            self._paused_since = time.monotonic()
        elif not paused:
            self._paused_since = None

    def run(self) -> None:
        while True:
            self.poll()
            self.poll_return()
            self.poll_pause()
            time.sleep(self.period_s)

    def paused(self) -> bool:
        """FILE 21 U3: a person said stop during the mission."""
        return self._paused_since is not None

    def paused_too_long(self) -> bool:
        return (self._paused_since is not None
                and time.monotonic() - self._paused_since > PAUSE_MAX_S)

    def active(self) -> bool:
        return self.paused() or time.monotonic() < self._until

    def return_requested(self) -> bool:
        return self._return

    def driving(self) -> dict | None:
        """FILE 21 U1: the mission's drive order while the hold is fresh --
        never while a person has paused her (U3)."""
        if self.paused():
            return None
        return self.drive if time.monotonic() < self._until else None

    def snapshot(self) -> dict[str, Any]:
        snap = {"active": self.active(), "why": self.why if self.active() else ""}
        if self.driving() is not None:
            snap["drive"] = self.drive.get("kind")
        if self.paused():
            snap["paused"] = True
        return snap


def drive_order(raw) -> dict | None:
    """A drive order from the hold file, checked; None if absent or malformed
    (a malformed order is the plain hold: she stands still)."""
    if not isinstance(raw, dict) or raw.get("kind") not in DRIVE_KINDS:
        return None
    try:
        order = {"id": str(raw.get("id", ""))[:64], "kind": raw["kind"]}
        if raw["kind"] == "path":
            pts = [(float(p[0]), float(p[1])) for p in raw["path"]][:DRIVE_MAX_POINTS]
            if not pts or not all(math.isfinite(v) for p in pts for v in p):
                return None
            lo, hi = DRIVE_TOL_M
            order["path"] = pts
            order["tol"] = min(max(float(raw.get("tol", 0.3)), lo), hi)
        elif raw["kind"] == "turn":
            order["dir"] = 1.0 if float(raw.get("dir", 1.0)) >= 0.0 else -1.0
        return order
    except Exception:                                     # noqa: BLE001
        # Anything malformed is no order. This runs on the hold's polling
        # thread: an exception escaping here would end that thread.
        return None


class MissionDrive:
    """FILE 21 U1: one drive order -> the request that replaces the policy's,
    exactly as the way back's homing_command replaces it. Owned by the control
    loop; cheap (a few points transformed per tick)."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.i = 0                      # the path point she is heading for
        self.ticks = 0
        # The bearing error to the point she is heading for (rad), or None: the
        # run's local navigator steers by it while an order is driven (pi_deploy
        # tick 1c -> 1d), so it dodges obstacles TOWARD the path instead of
        # toward the exploring's old frontier. 2026-10-04 robot test: the two
        # fought tick by tick and the way home stalled three times.
        self.last_err: float | None = None
        # Once a turn in place toward the point begins, it keeps its direction
        # until she faces the point: with the point straight behind her the
        # bearing wraps between +180 and -180 deg and the turn flipped every tick.
        self.turn_sign = 0

    def command(self, order: dict, tracker: "StartTracker | None", rl_pose,
                omega_max: float, v_floor: float, v_back: float
                ) -> tuple[float, float, str]:
        if order.get("id") != self.id:
            self.id, self.i, self.turn_sign = order.get("id"), 0, 0
        self.ticks += 1
        self.last_err = None
        kind = order["kind"]
        if kind == "turn":
            return (0.0, order["dir"] * omega_max, "mission: turning to look round")
        if kind == "back":
            return (-max(abs(v_back), v_floor), 0.0, "mission: backing up")
        if tracker is None:
            return (0.0, 0.0, "mission path: no start tracker, standing still")
        pts = [tracker.to_rl(p) for p in order["path"]]
        if any(p is None for p in pts):
            return (0.0, 0.0, "mission path: no start fix yet, standing still")
        x, y, th = (float(v) for v in rl_pose)
        last = len(pts) - 1
        self.i = min(self.i, last)
        while self.i < last and math.hypot(pts[self.i][0] - x,
                                           pts[self.i][1] - y) <= order["tol"]:
            self.i += 1
        gx, gy = pts[self.i]
        d = math.hypot(gx - x, gy - y)
        if self.i == last and d <= DRIVE_FINAL_TOL_M:
            return (0.0, 0.0, f"mission path: at its end, {d:.2f} m")
        err = _wrap(math.atan2(gy - y, gx - x) - th)
        if abs(err) > math.radians(HOME_PIVOT_DEG):
            if self.turn_sign == 0:
                self.turn_sign = 1 if err > 0 else -1
            err = math.copysign(abs(err), self.turn_sign)   # same way round
        else:
            self.turn_sign = 0
        self.last_err = err
        cmd = homing_command(err, d if self.i == last else None, omega_max, v_floor)
        v, omega, why = cmd
        why = why.replace("way back", f"mission path {self.i + 1}/{last + 1}")
        return (v, omega, why.replace("the start", "the next point"))


def _serve(max_range: float) -> int:
    """The tracker process: pickled jobs on stdin, pickled fixes on stdout."""
    import os
    import pickle
    os.nice(10)                     # the control loop always wins the CPU
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                   # a stray print must not corrupt the pipe
    inp = sys.stdin.buffer
    folder = _Folder(max_range)
    last_pub = 0.0                  # FILE 21
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
        # FILE 21: after the reply, so the run never waits on it.
        if res and "pose" in res and time.monotonic() - last_pub >= PUBLISH_PERIOD_S:
            last_pub = time.monotonic()
            _publish(scan, rl, blind, start_rl, res, max_range)


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
