"""A scripted RL run in a synthetic house, for the mission state-machine tests.

    sim = SimRun(clock, mm.three_rooms(), PATH)
    a = ad.fake_adapters(clock=clock, run=sim, lidar=sim.lidar(), slam=...)
    ...
    clock.advance(0.1); sim.step(0.1); mgr.tick()

It stands in for everything the mission reads from the real run:

  * live(): the start tracker's fix -- pose in the START frame and a robot-frame
    sweep, ray-cast on the true map, once a second;
  * status(): reason_code ("mission_hold" while held, "REFLEX_STOP" for an
    e-stop), executed speed, the way-back phase / outcome / distance;
  * it HONOURS the mission's three levers exactly as rl/ does: a hold stops it
    (and it publishes nothing while held), a return request starts its way
    back, and its own timer or the end of its path ("explored") does too.

`matched_within_m`: the real tracker only matches while the start room is in
view. None = an ideal tracker (every fix matched); a number = fixes further
than that from the start are published as "score too low".
"""

from __future__ import annotations

import math

from exploration_mission import adapters as ad
from exploration_mission import mission_map as mmap
import synth

SPEED = 0.30            # m/s, the run's cruise
PUBLISH_S = 1.0


class SimRun(ad.FakeRun):
    def __init__(self, clock, world, path, matched_within_m=None, start_delay_s=2.0,
                 fail_return_at_m=None):
        super().__init__(clock)
        self.world = world                      # make_maps.Map (truth)
        self.occ = world.grid.occupied()
        self.path = [tuple(p) for p in path]
        self.start_pose = (self.path[0][0], self.path[0][1], 0.0)
        self.truth = self.start_pose
        self.matched_within_m = matched_within_m
        self.start_delay_s = start_delay_s
        self.phase = "idle"                     # idle|exploring|returning|done
        self.why = ""
        self.outcome = ""
        self.estop_on = False
        self.publishing = True
        self.refuse_start = False
        self.stale_status_age = None            # seconds, to fake an old status file
        self._leg = 1
        self._t_started = None
        self._t_pub = -1e9
        self._duration = None
        self._back: list = []
        self.published = 0
        self.return_asked = False
        self.freeze_until = -1e9                # stands still (but runs) until then
        # Its own way back gives up ("timeout") this far from the start -- the
        # wedged-in-clutter ending the real robot has had.
        self.fail_return_at_m = fail_return_at_m
        self._drive_id, self._drive_i = None, 0
        self.drive_ticks = 0                    # steps spent on a mission drive order

    # --- what the mission reads ---
    def lidar(self):
        """friday-mcp's own LiDAR, before the run takes it: sweeps at the start."""
        sim = self

        class _L:
            calls = 0

            def sweep(self):
                _L.calls += 1
                return None if sim.is_running else sim._sweep()
        return _L()

    def _sweep(self) -> dict:
        g = self.world.grid
        return synth.grid_ray_cast(self.occ, g.origin, g.resolution, self.truth)

    def running(self):
        return self.is_running

    def start(self, minutes):
        self.orders.append(("start", minutes))
        if self.refuse_start:
            return {"started": False, "why": "systemctl refused"}
        self.is_running = True
        self._duration = float(minutes) * 60.0
        self._t_started = self.clock.monotonic()
        # The run's frame begins WHERE SHE STANDS, heading and all -- never a
        # teleport back to the path's first point.
        self.start_pose = self.truth
        self.phase, self._leg = "exploring", 1
        self.outcome, self.why = "", ""
        # rl/way_back.py ignores a return request written before the run began.
        self.return_asked = False
        return {"started": True, "minutes": minutes}

    def clear_return(self):
        self.return_asked = False

    def stop(self):
        self.orders.append(("stop",))
        self.is_running = False
        return "Stopped exploring."

    def request_return(self, mission_id=None):
        self.orders.append(("return", mission_id))
        self.return_asked = True

    def live(self, max_age_s=None):
        return super().live(max_age_s)

    def status(self):
        if self.phase == "idle":
            return None
        held = self.held() is not None
        code = "mission_hold" if held else ("REFLEX_STOP" if self.estop_on else "ok")
        moving = not held and not self.estop_on and self.phase in ("exploring",
                                                                   "returning")
        d = math.hypot(self.truth[0] - self.start_pose[0],
                       self.truth[1] - self.start_pose[1])
        out = {"tick": self.published, "reason_code": code,
               "final_v": SPEED if moving else 0.0, "final_omega": 0.0,
               "frontiers": 0, "exhausted": self.phase != "exploring",
               "planner_error": "", "explored_m2_rl": 0.0,
               "way_back_phase": self.phase, "way_back_outcome": self.outcome,
               "way_back_distance_m": round(d, 2)}
        if self.stale_status_age is not None:
            out["age_s"] = self.stale_status_age
        return out

    # --- the world moving on ---
    def _toward(self, target, dt) -> bool:
        x, y, th = self.truth
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        step = min(SPEED * dt, dist)
        self.truth = (x + dx / dist * step, y + dy / dist * step, math.atan2(dy, dx))
        return step >= dist - 1e-9

    def _begin_back(self, why: str) -> None:
        self.phase, self.why = "returning", why
        self._back = list(reversed(self.path[:self._leg]))

    def _drive(self, order: dict, dt: float) -> None:
        """rl/way_back.py MissionDrive, in the sim: a held run with a drive
        order steers at the mission's path (start frame) / turns / backs up."""
        self.drive_ticks += 1
        if order.get("id") != self._drive_id:
            self._drive_id, self._drive_i = order.get("id"), 0
        x, y, th = self.truth
        if (self.fail_return_at_m is not None and order["kind"] != "turn"
                and math.hypot(x - self.start_pose[0], y - self.start_pose[1])
                <= self.fail_return_at_m):
            return                              # wedged: no way past from here
        if order["kind"] == "turn":
            self.truth = (x, y, math.atan2(math.sin(th + 1.3 * dt), math.cos(th + 1.3 * dt)))
            return
        if order["kind"] == "back":
            self.truth = (x - math.cos(th) * 0.15 * dt, y - math.sin(th) * 0.15 * dt, th)
            return
        pts = [mmap.compose(self.start_pose, (p[0], p[1], 0.0))[:2] for p in order["path"]]
        last = len(pts) - 1
        while (self._drive_i < last and math.hypot(pts[self._drive_i][0] - x,
                                                   pts[self._drive_i][1] - y)
               <= order.get("tol", 0.3)):
            self._drive_i += 1
        gx, gy = pts[self._drive_i]
        if self._drive_i == last and math.hypot(gx - x, gy - y) <= 0.10:
            return                              # rl/way_back.py DRIVE_FINAL_TOL_M
        self._toward((gx, gy), dt)

    def step(self, dt: float) -> None:
        if not self.is_running or self.phase in ("idle", "done"):
            return
        now = self.clock.monotonic()
        if now - self._t_started < self.start_delay_s:
            return                                  # still loading its detector
        order = (self.held() or {}).get("drive")
        if order and not self.estop_on and now >= self.freeze_until:
            self._drive(order, dt)
        if self.held() is None and not self.estop_on and now >= self.freeze_until:
            if self.phase == "exploring":
                if self.return_asked:
                    self._begin_back("mission")
                elif now - self._t_started >= self._duration:
                    self._begin_back("time")
                elif self._leg >= len(self.path):
                    self._begin_back("explored")
                elif self._toward(self.path[self._leg], dt):
                    self._leg += 1
            if self.phase == "returning":
                home = math.hypot(self.truth[0] - self.start_pose[0],
                                  self.truth[1] - self.start_pose[1])
                if (self.fail_return_at_m is not None and self._back
                        and len(self._back) <= 1 and home <= self.fail_return_at_m):
                    self.phase, self.outcome = "done", "timeout"
                    self.is_running = False
                elif not self._back:
                    self.phase, self.outcome = "done", "arrived"
                    self.is_running = False
                elif self._toward(self._back[0], dt):
                    self._back.pop(0)
        # An e-stopped run still ticks and publishes; a HELD one returns before
        # its tracker is fed, so it publishes nothing (rl/pi_deploy.py).
        # ...but a held run DRIVING an order ticks in full, and publishes.
        if ((self.held() is None or order) and self.publishing and self.is_running
                and now - self._t_pub >= PUBLISH_S):
            self._t_pub = now
            pose = mmap.compose(mmap.inverse(self.start_pose), self.truth)
            far = (self.matched_within_m is not None
                   and math.hypot(pose[0], pose[1]) > self.matched_within_m)
            self.live_fix = {"v": 1, "t": self.clock.wall(), "pose": list(pose),
                             "reason": "score too low" if far else "matched",
                             "rl": list(self.truth), "scan": self._sweep()}
            self.published += 1


# A tour of make_maps.three_rooms(): round room A, through the door, round B,
# through the door, round C. Rooms are x 1-4.5 | 4.5-8 | 8-11.5, doors at y = 3.
THREE_ROOM_TOUR = [
    (2.5, 3.0), (2.0, 2.0), (3.6, 2.0), (3.6, 4.0), (2.0, 4.0), (3.4, 3.0),
    (5.4, 3.0), (5.4, 2.0), (7.0, 2.0), (7.0, 4.0), (5.4, 4.0), (6.2, 3.0),
    (8.9, 3.0), (8.9, 2.0), (10.6, 2.0), (10.6, 4.0), (8.9, 4.0), (9.8, 3.0),
]
ROOM_CENTRES = {"A": (2.75, 3.0), "B": (6.25, 3.0), "C": (9.75, 3.0)}

# A tour of make_maps.uneven_rooms(): A is x 1-4.5, B 4.5-7, C 7-11.5; the A-B
# door is at y = 2 and the B-C door at y = 4.
UNEVEN_TOUR = [
    (2.5, 3.0), (2.0, 4.0), (3.6, 4.0), (3.6, 2.0), (2.2, 2.0), (3.4, 2.0),
    (5.4, 2.0), (6.1, 2.0), (6.1, 4.0), (5.4, 4.0), (5.75, 3.0), (5.75, 4.0),
    (7.9, 4.0), (10.6, 4.0), (10.6, 2.0), (8.0, 2.0), (9.2, 3.0),
]
UNEVEN_CENTRES = {"A": (2.75, 3.0), "B": (5.75, 3.0), "C": (9.25, 3.0)}
