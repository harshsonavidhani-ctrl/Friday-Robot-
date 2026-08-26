"""Where Friday is, and what is around her: pose tracking over the map.

`core` matches two sweeps, `grid` remembers, `transient` decides what is
furniture. This is the loop that runs them: take a sweep and an odometry
guess, work out where the robot actually is, and fold what it saw into the
room.

WHY MATCH AGAINST THE MAP AND NOT THE PREVIOUS SWEEP. Matching each sweep to
the one before it is dead reckoning with extra steps -- every match inherits
the error of the one before, so the estimate drifts smoothly and confidently
into nonsense. The map has been averaged over every sweep that built it, so
matching against it corrects back towards a consensus instead of accumulating.
It is also, for free, most of what loop closure has to do: coming back into a
mapped room re-localises against the room.

WHY ODOMETRY IS STILL THE PRIOR. There are no wheel encoders. move_distance
and turn_degrees are TIMED conversions from state/calibration.json worth about
+/-20%, which is useless as an answer and excellent as a starting guess: it
tells the matcher which 20 cm box to search, and searching a box instead of a
room is what makes the match affordable at all.

WHEN THE MATCH IS NOT BELIEVED, ODOMETRY IS KEPT AND SAID SO. A bad correction
is worse than no correction: it moves the robot on the map, so the next sweep
is folded in at the wrong place and corrupts the room permanently. Every
update returns why it did what it did, and `pose_trusted` says whether anyone
should be acting on the number.
"""

from __future__ import annotations

import math
import time

import numpy as np

import config
from slam.core import LikelihoodField, match_pose, points_from_scan, transform
from slam.grid import OccupancyGrid
from slam.transient import TransientLayer


class Mapper:
    """Pose tracking, the permanent map, and the transient layer as one loop."""

    def __init__(self, grid: OccupancyGrid | None = None,
                 pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 forward_deg: float | None = None) -> None:
        self.grid = grid if grid is not None else OccupancyGrid()
        self.transient = TransientLayer(cell_m=self.grid.cell)
        self.pose = tuple(float(v) for v in pose)
        self.forward_deg = (config.LIDAR_FORWARD_DEG if forward_deg is None
                            else forward_deg)
        self.home = tuple(self.pose)
        self.updates = 0
        self.pose_trusted = True
        self.travelled_m = 0.0
        # pose plus the sweep taken there, for loop closure and rebuilds
        self.keyframes: list[dict] = []
        self._field: LikelihoodField | None = None
        self._field_at = -1

    # --- the loop --------------------------------------------------------

    def update(self, scan: dict[int, int], moved_m: float = 0.0,
               turned_rad: float = 0.0, now: float | None = None) -> dict:
        """One sweep. Returns what happened, in enough detail to narrate.

        moved_m and turned_rad are what odometry believes has happened since
        the last update, in the robot's own frame.
        """
        now = time.time() if now is None else now
        local = points_from_scan(scan, forward_deg=self.forward_deg)
        prior = self._predict(moved_m, turned_rad)

        pose, score, sharpness, reason = self._locate(local, prior)
        self.pose = pose
        self.pose_trusted = reason == "matched" or self.updates == 0
        self.travelled_m += abs(moved_m)

        world = transform(local, pose[0], pose[1], pose[2])
        confirmed = self.transient.observe(world, now=now)
        # Anything the map already calls a wall keeps its evidence topped up
        # without re-earning it -- otherwise a cell that decays just below the
        # threshold has to serve its three sweeps again to come back.
        known = world[self.grid.classify(world) == 1] if len(world) else world
        marks = (np.vstack([confirmed, known])
                 if len(confirmed) and len(known)
                 else (confirmed if len(confirmed) else known))
        self.grid.integrate(world, (pose[0], pose[1]), occupied=marks)

        self._maybe_keyframe(local, pose)
        self.updates += 1
        return {
            "pose": pose,
            "score": round(float(score), 3),
            "sharpness": round(float(sharpness), 2),
            "reason": reason,
            "trusted": self.pose_trusted,
            "beams": int(len(local)),
            "mapped": int(len(marks)) if len(marks) else 0,
            "unconfirmed": int(len(self.transient.unconfirmed_points(now))),
            "travelled_m": round(self.travelled_m, 2),
            "keyframes": len(self.keyframes),
        }

    def _predict(self, moved_m: float,
                 turned_rad: float) -> tuple[float, float, float]:
        """Odometry prior. Translation is applied along the MIDPOINT heading.

        Turning and driving happen together, so using the heading from before
        the turn puts the robot off to one side of where an arc actually ends.
        The midpoint is the cheap correction and it is worth having: at 92
        deg/s a half-second update can turn 46 degrees.
        """
        x, y, theta = self.pose
        mid = theta + turned_rad / 2.0
        return (x + moved_m * math.cos(mid),
                y + moved_m * math.sin(mid),
                (theta + turned_rad + math.pi) % (2 * math.pi) - math.pi)

    def _locate(self, local: np.ndarray, prior: tuple[float, float, float]
                ) -> tuple[tuple[float, float, float], float, float, str]:
        """Correct the prior against the map, or explain why it was not."""
        if len(local) == 0:
            return prior, 0.0, 1.0, "no beams"
        field = self._map_field()
        if field is None:
            return prior, 0.0, 1.0, "map too thin"
        pose, score, sharpness = match_pose(field, local, prior)
        if score < config.SLAM_MIN_SCORE:
            return prior, score, sharpness, "score too low"
        if sharpness < config.SLAM_POSE_MIN_SHARPNESS:
            return prior, score, sharpness, "geometry ambiguous"
        return pose, score, sharpness, "matched"

    def _map_field(self) -> LikelihoodField | None:
        """A likelihood field over the map's occupied cells, cached.

        Rebuilding blurs the whole grid, so it is not done every sweep. The
        map does not change fast enough for that to matter -- a wall that
        appeared four sweeps ago is still in the same place.
        """
        occupied = np.argwhere(self.grid.occupied_mask())
        if len(occupied) < config.SLAM_MATCH_MIN_CELLS:
            return None
        stale = self.updates - self._field_at >= config.SLAM_FIELD_REBUILD_SCANS
        if self._field is None or stale:
            centres = (occupied + 0.5) * self.grid.cell + self.grid.origin
            self._field = LikelihoodField(centres)
            self._field_at = self.updates
        return self._field

    # --- keyframes -------------------------------------------------------

    def _maybe_keyframe(self, local: np.ndarray,
                        pose: tuple[float, float, float]) -> None:
        if self.keyframes:
            last = self.keyframes[-1]["pose"]
            moved = math.hypot(pose[0] - last[0], pose[1] - last[1])
            turned = abs((pose[2] - last[2] + math.pi) % (2 * math.pi) - math.pi)
            if (moved < config.SLAM_KEYFRAME_M and
                    turned < math.radians(config.SLAM_KEYFRAME_DEG)):
                return
        self.keyframes.append({"pose": tuple(pose), "points": local.copy(),
                               "at": time.time()})
        if len(self.keyframes) > config.SLAM_KEYFRAME_MAX:
            # Drop from the middle, never the ends: the start anchors the map
            # and the recent ones are what loop closure matches against.
            del self.keyframes[len(self.keyframes) // 2]

    # --- what the rest of the robot asks --------------------------------

    def obstacle_points(self, now: float | None = None) -> np.ndarray:
        """Everything to avoid: the mapped room AND whatever is there today.

        Two sources on purpose. The map knows the walls the sweep cannot see
        past; the transient layer knows the person who walked in thirty
        seconds ago and is not furniture. Driving needs both.
        """
        occupied = np.argwhere(self.grid.occupied_mask())
        walls = ((occupied + 0.5) * self.grid.cell + self.grid.origin
                 if len(occupied) else np.empty((0, 2)))
        recent = self.transient.blocking_points(now)
        if len(walls) and len(recent):
            return np.vstack([walls, recent])
        return walls if len(walls) else recent

    def distance_home(self) -> float:
        return math.hypot(self.pose[0] - self.home[0],
                          self.pose[1] - self.home[1])

    def status(self) -> dict:
        return {"pose": [round(v, 3) for v in self.pose[:2]],
                "heading_deg": round(math.degrees(self.pose[2]), 1),
                "trusted": self.pose_trusted,
                "travelled_m": round(self.travelled_m, 2),
                "home_m": round(self.distance_home(), 2),
                "updates": self.updates,
                "keyframes": len(self.keyframes),
                "map": self.grid.stats(),
                "transient": self.transient.stats()}
