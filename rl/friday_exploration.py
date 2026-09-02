"""Autonomous exploration supervisor -- the layer above the learned local navigator.

    PERCEIVE -> UNDERSTAND -> MAP/REMEMBER -> SELECT TARGET -> NAVIGATE
             -> OBSERVE RESULT -> CONTINUE

This module owns everything except NAVIGATE. The learned ACT policy owns NAVIGATE and
nothing else: it is handed a target in the robot frame and it drives to it. That split
is deliberate and load-bearing --

  * exploration strategy stays inspectable and tunable without retraining;
  * the policy stays a small, fast, bounded-input network that fits a Pi 4B;
  * the safety chain below it is unchanged and still has final authority.

WHAT MAKES THIS EXPLORATION RATHER THAN WANDERING
-------------------------------------------------
Four things, all of which need memory the reactive policy does not have:

  MEMORY      `LocalExplorationMap.known` accumulates every cell any beam has passed
              through. Nothing is explored twice by accident.
  FRONTIERS   A frontier is a known-free cell touching an unknown cell -- literally
              the boundary of the robot's knowledge. Driving to one is guaranteed to
              reveal new space, which is what separates exploration from patrolling.
  ATTEMPT     Targets that were tried and failed are blacklisted with a radius and a
  MEMORY      timeout, so the robot does not re-select the doorway it just failed to
              get through, forever. This is the single most common failure of naive
              frontier exploration.
  PROGRESS    Coverage (new cells per second) and displacement are both watched. When
  MONITORING  they stall, the supervisor says so and enters an explicit recovery
              state rather than grinding against a wall until the battery dies.

HONESTY RULE FOR VOICE
----------------------
Every announcement below is emitted from a state transition this class actually
observed, computed from real sensor and map data. In particular:

    Friday has NO object recognition. Confirmed in the codebase: `pi_deploy.py`
    documents its camera layer as "a calibrated HEURISTIC (edge density, looming,
    dark-region ratio). It is not object detection and it does not know what it is
    looking at." `VisualHazardMonitor.evaluate` returns a scalar hazard score and a
    stop flag -- no class, no label, no bounding box.

So the robot may honestly announce "the cameras report a possible floor hazard", which
is exactly what that score means, and must NOT announce "I see a chair". If a real
detector is added later, `describe_detection()` below is the single place to extend;
it returns None today because there is nothing truthful to say.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from friday_voice import P_NOTABLE, P_ROUTINE, P_URGENT, VoiceAnnouncer

LOGGER = logging.getLogger("friday.explore")


# ======================================================================================
# States
# ======================================================================================
class Phase:
    BOOTSTRAP = "bootstrap"     # spin up sensors, take a first look around
    SELECT = "select"           # choose an exploration target
    NAVIGATE = "navigate"       # drive to it (learned policy does the driving)
    OBSERVE = "observe"         # arrived: fold in what we learned, decide what next
    RECOVER = "recover"         # no progress -- deliberate escape behaviour
    COMPLETE = "complete"       # no reachable frontier left


@dataclass
class ExplorationConfig:
    # ---- target selection --------------------------------------------------------
    min_target_dist: float = 0.80      # [m] closer than this is not worth driving to
    max_target_dist: float = 5.00      # [m] beyond LiDAR range the goal is a guess
    reach_radius: float = 0.45         # [m] "arrived"
    forward_bias: float = 0.60         # prefer targets ahead; pivots scrub 4 wheels
    replan_period_s: float = 3.0       # re-select even while en route
    min_frontier_cells: int = 6        # ignore single-cell speckle frontiers

    # ---- attempt memory ----------------------------------------------------------
    blacklist_radius: float = 0.70     # [m] around a failed target
    blacklist_ttl_s: float = 45.0      # forget failures eventually; the map changes
    max_attempts_per_target: int = 2

    # ---- progress monitoring -----------------------------------------------------
    stuck_window_s: float = 6.0        # look-back for the progress test
    stuck_min_travel: float = 0.25     # [m] displacement in that window
    stuck_min_new_cells: int = 12      # OR this much new map -- either counts
    target_timeout_s: float = 25.0     # give up on one target and pick another

    # ---- recovery ----------------------------------------------------------------
    recover_duration_s: float = 4.0
    recover_omega: float = 1.45        # [rad/s] must clear the pivot stiction floor

    # ---- completion --------------------------------------------------------------
    complete_min_runtime_s: float = 10.0    # an empty map at t=0 is not "explored"
    complete_min_area_m2: float = 4.0

    # ---- reporting ---------------------------------------------------------------
    coverage_report_step_m2: float = 12.0   # announce each N m^2 of new ground
    obstacle_ahead_dist: float = 0.85       # [m] forward sector min -> "obstacle ahead"
    obstacle_ahead_halfangle: float = 0.45  # [rad] what counts as "ahead"


@dataclass
class _Blacklisted:
    x: float
    y: float
    t: float
    attempts: int = 1


@dataclass
class ExplorationState:
    """Everything the supervisor knows. Exposed so telemetry and tests can read it."""
    phase: str = Phase.BOOTSTRAP
    target: np.ndarray | None = None
    target_started_t: float = 0.0
    targets_selected: int = 0
    targets_reached: int = 0
    targets_abandoned: int = 0
    recoveries: int = 0
    recoveries_successful: int = 0
    explored_m2: float = 0.0
    frontier_cells: int = 0
    last_reason: str = ""
    blacklist: list[_Blacklisted] = field(default_factory=list)


# ======================================================================================
# Supervisor
# ======================================================================================
class ExplorationSupervisor:
    """Drives the exploration state machine and emits grounded voice events.

    Call `update(exploration_map, scan, row)` once per control tick, BEFORE building
    the policy observation. It returns the target the local navigator should pursue,
    in world coordinates, or None when exploration is complete.
    """

    def __init__(self, robot_cfg, cfg: ExplorationConfig | None = None,
                 voice: VoiceAnnouncer | None = None) -> None:
        self.rcfg = robot_cfg
        self.cfg = cfg or ExplorationConfig()
        self.voice = voice
        self.st = ExplorationState()

        self._t0 = time.monotonic()
        self._ticks_since_plan = 10 ** 9
        self._recover_until = 0.0
        self._recover_sign = 1.0
        self._hist: list[tuple[float, float, float, int]] = []   # (t, x, y, cells)
        self._last_coverage_report = 0.0
        self._prev_target: np.ndarray | None = None
        self._was_stuck = False

    # ------------------------------------------------------------------ main tick --
    def update(self, m, scan: np.ndarray, row: dict | None = None) -> np.ndarray | None:
        """One supervisor step. `m` is a `pi_deploy.LocalExplorationMap`."""
        now = time.monotonic()
        px, py, th = float(m.pose[0]), float(m.pose[1]), float(m.pose[2])
        self.st.explored_m2 = float(m.explored_m2)

        self._prune_blacklist(now)
        self._hist.append((now, px, py, int(m.explored_cells)))
        cutoff = now - max(self.cfg.stuck_window_s, 1.0) * 2.0
        while len(self._hist) > 2 and self._hist[0][0] < cutoff:
            self._hist.pop(0)

        # ---- PERCEIVE / UNDERSTAND: what do the live sensors say right now? -------
        self._announce_perception(scan, row)

        # ---- RECOVER holds the wheel until its timer expires ----------------------
        if self.st.phase == Phase.RECOVER:
            if now < self._recover_until:
                return self._recovery_target(px, py, th)
            # Recovery over. Did it work?
            if not self._is_stuck(now):
                self.st.recoveries_successful += 1
                self._say("recovery_ok", "Recovery successful. Continuing exploration.",
                          priority=P_URGENT, cooldown_s=6.0)
                self.st.phase = Phase.SELECT
            else:
                # Still stuck: blacklist the target that led here and try elsewhere.
                if self.st.target is not None:
                    self._blacklist(self.st.target, now)
                self.st.phase = Phase.SELECT
                self._say("recovery_retry",
                          "Still unable to make progress. Choosing a different area.",
                          priority=P_URGENT, cooldown_s=10.0)
            self._ticks_since_plan = 10 ** 9

        # ---- OBSERVE RESULT: stuck / timed out / arrived? --------------------------
        if self.st.phase == Phase.NAVIGATE and self.st.target is not None:
            d = math.hypot(self.st.target[0] - px, self.st.target[1] - py)
            elapsed = now - self.st.target_started_t

            if d <= self.cfg.reach_radius:
                self.st.targets_reached += 1
                self.st.phase = Phase.OBSERVE
                self._say("arrived", "Target area reached. Updating my map.",
                          priority=P_ROUTINE, cooldown_s=10.0)

            elif self._is_stuck(now):
                self.st.phase = Phase.RECOVER
                self.st.recoveries += 1
                self._recover_until = now + self.cfg.recover_duration_s
                self._recover_sign = self._openest_side(scan)
                self._say("stuck", "I am unable to make progress. Initiating recovery.",
                          priority=P_URGENT, cooldown_s=8.0)
                self._hist.clear()
                return self._recovery_target(px, py, th)

            elif elapsed > self.cfg.target_timeout_s:
                self.st.targets_abandoned += 1
                self._blacklist(self.st.target, now)
                self.st.phase = Phase.SELECT
                self._say("timeout",
                          "That area is taking too long to reach. Selecting a new target.",
                          priority=P_NOTABLE, cooldown_s=12.0)

        # ---- SELECT EXPLORATION TARGET --------------------------------------------
        self._ticks_since_plan += 1
        stale = self._ticks_since_plan >= max(
            int(self.cfg.replan_period_s * self.rcfg.control_hz), 1)
        if self.st.phase in (Phase.BOOTSTRAP, Phase.SELECT, Phase.OBSERVE) or (
                self.st.phase == Phase.NAVIGATE and stale):
            target, n_frontier = self._select_target(m, px, py, th, now)
            self.st.frontier_cells = n_frontier

            if target is None:
                if n_frontier == 0 and self._may_declare_complete(now):
                    if self.st.phase != Phase.COMPLETE:
                        self.st.phase = Phase.COMPLETE
                        self._say("complete",
                                  "I have explored all reachable areas from here.",
                                  priority=P_NOTABLE, cooldown_s=30.0)
                elif n_frontier > 0:
                    # Frontiers exist but all are blacklisted or out of range.
                    self._say("explored_here",
                              "This area has already been explored. "
                              "Selecting a new direction.",
                              priority=P_NOTABLE, cooldown_s=15.0)
                # Keep the previous target if we still have one; otherwise the caller
                # falls back to a forward heading and the safety chain still governs.
                return self.st.target
            self._commit_target(target, px, py, now)

        self._report_coverage()
        return self.st.target

    # -------------------------------------------------------------- target choice --
    def _commit_target(self, target: np.ndarray, px: float, py: float,
                       now: float) -> None:
        prev = self.st.target
        changed = prev is None or float(np.hypot(*(target - prev))) > 0.5
        self.st.target = target
        self.st.target_started_t = now
        self._ticks_since_plan = 0
        if changed:
            self.st.targets_selected += 1
            d = math.hypot(target[0] - px, target[1] - py)
            if self.st.phase in (Phase.BOOTSTRAP, Phase.SELECT, Phase.OBSERVE):
                n = max(int(round(d)), 1)
                self._say("new_frontier",
                          "I found an unexplored area {} {} to the {}. "
                          "I will investigate it.".format(
                              n, "metre" if n == 1 else "metres",
                              self._compass(target, px, py)),
                          priority=P_ROUTINE, cooldown_s=9.0)
        self.st.phase = Phase.NAVIGATE

    def _select_target(self, m, px: float, py: float, th: float,
                       now: float) -> tuple[np.ndarray | None, int]:
        """Nearest useful frontier, biased forward, excluding blacklisted areas.

        Returns (target_or_None, total_frontier_cells). The second value distinguishes
        "nothing left to explore" from "everything left is somewhere I already failed",
        which are different situations that deserve different announcements.
        """
        res = self.rcfg.resolution
        known = m.known.astype(bool)
        occ = m.occ.astype(bool)
        free = known & ~occ
        unknown = ~known

        nbr = np.zeros_like(unknown)
        nbr[1:, :] |= unknown[:-1, :]
        nbr[:-1, :] |= unknown[1:, :]
        nbr[:, 1:] |= unknown[:, :-1]
        nbr[:, :-1] |= unknown[:, 1:]
        frontier = free & nbr

        # A frontier cell the footprint cannot occupy is not a place we can drive to.
        pad = max(int(round(self.rcfg.robot_radius / res)), 1)
        frontier &= ~_dilate(occ, pad)

        rr, cc = np.nonzero(frontier)
        n_frontier = int(rr.size)
        if n_frontier < self.cfg.min_frontier_cells:
            return None, n_frontier

        fx = (cc + 0.5) * res
        fy = (rr + 0.5) * res
        d = np.hypot(fx - px, fy - py)

        ok = (d >= self.cfg.min_target_dist) & (d <= self.cfg.max_target_dist)
        if not ok.any():
            ok = d <= self.cfg.max_target_dist
            if not ok.any():
                return None, n_frontier

        # Attempt memory: drop anything near a target we already failed to reach.
        for b in self.st.blacklist:
            ok &= np.hypot(fx - b.x, fy - b.y) > self.cfg.blacklist_radius
        if not ok.any():
            return None, n_frontier

        fx, fy, d = fx[ok], fy[ok], d[ok]

        bearing = np.arctan2(fy - py, fx - px) - th
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        ahead = 0.5 * (np.cos(bearing) + 1.0)          # 1 straight on, 0 behind
        cost = d * (1.0 - self.cfg.forward_bias * 0.5 * ahead)

        k = int(np.argmin(cost))
        return np.array([fx[k], fy[k]], np.float32), n_frontier

    def _may_declare_complete(self, now: float) -> bool:
        """Guard against announcing "fully explored" from a momentarily empty map.

        On the first ticks the known-space grid is empty, so there are no known-free
        cells and therefore no frontiers -- which looks identical to "everything has
        been explored". The same shape appears transiently if the LiDAR drops out.
        Requiring both a warm-up period and a plausible amount of mapped ground makes
        the distinction, and the cost of being wrong is only a delayed announcement.
        """
        return (now - self._t0) > self.cfg.complete_min_runtime_s and \
            self.st.explored_m2 >= self.cfg.complete_min_area_m2

    # ------------------------------------------------------------ attempt memory ---
    def _blacklist(self, target: np.ndarray, now: float) -> None:
        for b in self.st.blacklist:
            if math.hypot(b.x - target[0], b.y - target[1]) < self.cfg.blacklist_radius:
                b.attempts += 1
                b.t = now
                return
        self.st.blacklist.append(_Blacklisted(float(target[0]), float(target[1]), now))

    def _prune_blacklist(self, now: float) -> None:
        ttl = self.cfg.blacklist_ttl_s
        self.st.blacklist = [
            b for b in self.st.blacklist
            if (now - b.t) < ttl * max(b.attempts, 1)
        ]

    # ------------------------------------------------------- progress monitoring ---
    def _is_stuck(self, now: float) -> bool:
        """No meaningful displacement AND no meaningful new map, over the window.

        Both tests matter. A robot pivoting in place to map a room is productive and
        must not be called stuck; a robot sliding along a wall it has already mapped is
        moving but achieving nothing. Only the conjunction is a real stall.
        """
        w = self.cfg.stuck_window_s
        old = None
        for t, x, y, cells in self._hist:
            if now - t <= w:
                break
            old = (t, x, y, cells)
        if old is None or not self._hist:
            return False
        _t, x0, y0, c0 = old
        _t1, x1, y1, c1 = self._hist[-1]
        travelled = math.hypot(x1 - x0, y1 - y0)
        new_cells = c1 - c0
        return (travelled < self.cfg.stuck_min_travel
                and new_cells < self.cfg.stuck_min_new_cells)

    def _recovery_target(self, px: float, py: float, th: float) -> np.ndarray:
        """A virtual goal off to one side, so the learned policy pivots to clear it.

        Recovery is expressed as a TARGET rather than a direct motor command on
        purpose: the learned policy and the whole safety chain stay in the loop, so a
        recovery pivot is still governed, still reflex-vetoed, still rate-limited. A
        recovery that bypassed the safety chain would be the one manoeuvre on this
        robot with unrestricted motor authority, which is exactly what we were told
        not to build.
        """
        heading = th + self._recover_sign * 2.2       # ~126 deg off the nose
        return np.array([px + 1.2 * math.cos(heading),
                         py + 1.2 * math.sin(heading)], np.float32)

    def _openest_side(self, scan: np.ndarray) -> float:
        """+1 to turn left, -1 to turn right, whichever the LiDAR says is more open."""
        if scan is None or scan.size < 8:
            return 1.0
        n = scan.size
        left = float(np.mean(scan[n // 8: 3 * n // 8]))
        right = float(np.mean(scan[5 * n // 8: 7 * n // 8]))
        return 1.0 if left >= right else -1.0

    # ------------------------------------------------------- perception reporting --
    def _announce_perception(self, scan: np.ndarray, row: dict | None) -> None:
        """Speak only what the live sensors and the safety chain actually reported."""
        c = self.cfg

        # LiDAR: is something close, in the forward sector specifically?
        if scan is not None and scan.size:
            n = scan.size
            half = max(int(c.obstacle_ahead_halfangle / (2 * math.pi) * n), 1)
            fwd = np.concatenate([scan[-half:], scan[:half + 1]])
            fwd_min = float(np.min(fwd))
            if fwd_min < c.obstacle_ahead_dist and self.st.phase == Phase.NAVIGATE:
                self._say("obstacle_ahead",
                          "I detected an obstacle {:.1f} metres ahead. "
                          "I am changing my route.".format(fwd_min),
                          priority=P_NOTABLE, cooldown_s=10.0)

        if not row:
            return

        gate = row.get("gate", "ok")
        if gate == "REFLEX_STOP":
            self._say("reflex", "Emergency stop. Something is directly in front of me.",
                      priority=P_URGENT, cooldown_s=6.0)
        elif gate == "visual_stop":
            # Honest wording: the camera layer is a hazard heuristic, not a detector.
            self._say("visual_stop",
                      "My cameras report a possible floor hazard. I am stopping.",
                      priority=P_URGENT, cooldown_s=10.0)
        elif gate == "visual_slow":
            self._say("visual_slow",
                      "Reduced visibility ahead. I am slowing down.",
                      priority=P_ROUTINE, cooldown_s=20.0)
        elif gate in ("stale_lidar", "stale_tof"):
            self._say("sensor_stale",
                      "A sensor has stopped responding. Holding position for safety.",
                      priority=P_URGENT, cooldown_s=8.0)

        desc = self.describe_detection(row)
        if desc:
            self._say("detection", desc, priority=P_NOTABLE, cooldown_s=8.0)

    def describe_detection(self, row: dict) -> str | None:
        """Extension point for a REAL object detector. Returns None today.

        Friday has no object recognition. `pi_deploy.VisualHazardMonitor` produces a
        scalar hazard score from edge density, looming and dark-region ratio, and its
        own docstring says it "is not object detection and it does not know what it is
        looking at". Announcing object names from that score would be fabrication, so
        this returns None and the hazard score is reported as a hazard score above.

        If a detector is added (a quantised MobileNet-SSD or YOLO-nano through the
        same ONNX Runtime session pool would fit the Pi's remaining budget), publish
        its labels into `row["detections"]` and describe them here -- and only labels
        that the detector actually emitted, with its confidence.
        """
        return None

    # -------------------------------------------------------------------- helpers --
    def _report_coverage(self) -> None:
        step = self.cfg.coverage_report_step_m2
        if self.st.explored_m2 - self._last_coverage_report >= step:
            self._last_coverage_report = self.st.explored_m2
            self._say("coverage",
                      "I have now mapped {:.0f} square metres.".format(
                          self.st.explored_m2),
                      priority=P_ROUTINE, cooldown_s=25.0)

    @staticmethod
    def _compass(target: np.ndarray, px: float, py: float) -> str:
        """Coarse direction in the MAP frame -- unambiguous over a spoken channel."""
        ang = math.degrees(math.atan2(float(target[1]) - py, float(target[0]) - px))
        names = ["east", "north east", "north", "north west",
                 "west", "south west", "south", "south east"]
        return names[int(round(ang / 45.0)) % 8]

    def _say(self, key: str, text: str, *, priority: int, cooldown_s: float) -> None:
        self.st.last_reason = key
        if self.voice is not None:
            self.voice.say(key, text, priority=priority, cooldown_s=cooldown_s)
        else:
            LOGGER.info("EVENT [%s] %s", key, text)

    # --------------------------------------------------------------------- report --
    def report(self) -> None:
        st = self.st
        LOGGER.info("exploration: phase=%s explored=%.1f m^2 frontier_cells=%d",
                    st.phase, st.explored_m2, st.frontier_cells)
        LOGGER.info("targets: selected=%d reached=%d abandoned=%d  blacklisted=%d",
                    st.targets_selected, st.targets_reached, st.targets_abandoned,
                    len(st.blacklist))
        LOGGER.info("recoveries: %d attempted, %d successful",
                    st.recoveries, st.recoveries_successful)


# ======================================================================================
def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r cells via summed-area table (no SciPy on the Pi)."""
    if r <= 0:
        return mask
    m = mask.astype(np.uint8)
    pad = np.pad(m, r, mode="constant")
    cs = pad.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    k = 2 * r + 1
    h, w = m.shape
    total = (cs[k:k + h, k:k + w] - cs[:h, k:k + w] - cs[k:k + h, :w] + cs[:h, :w])
    return total > 0
