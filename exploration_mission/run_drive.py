"""File 14's way of driving, through the held RL run (File 21 U1: Harsh,
2026-10-03, "make exactly as file ... dont change the exploration behaviour").

THE FILE'S PATTERN (3.2, 4.11): the RL policy explores; for the way home (and
frontier-assist and "go to X", U2) "the File 14 engine drives". On this robot
the run owns the LiDAR and the motors while it is out, so File 14's follower
cannot have the wheels -- and parking the run to free them threw away the one
scan-matched pose she had, after which every attempt had to find her again on
a two-minute map (test #3: 0.71 vs 0.70, refused every time).

So the run is HELD, not parked, and this engine does what File 14 does around
it:

    File 14                         here
    -------                         ----
    path_planner (despeckle,        the SAME path_planner.plan(), on the
      inflate, A*, smooth)            mission's own map (start frame)
    RuntimeObstacles + clear()      `obstacles` -- a disc where she stalled,
                                      passed to the planner as extra_blocked;
                                      rung 2 clears it, never the map
    BLOCKED_REPLANNING              re-plan every REPLAN_S from where she is
    the follower                    the held run steering at the path
                                      (rl/way_back.py MissionDrive): the way
                                      back's own command, with the arc planner,
                                      governor, recovery, unstick, rear gate and
                                      ToF reflex all still in charge
    pose from the scan matcher      the run's start tracker (live.json): it
                                      never stops tracking, so she is never
                                      "found again"
    stall -> fail                   no progress_min_m of progress in
                                      progress_timeout_s -> the attempt fails

The ladder (rth_ladder.Ladder) climbs its rungs through this exactly as it
would through File 14: attempt / busy / outcome / spin_relocalise / back_up /
rear_clear_m / clear_runtime_obstacles / distance_to / pose / stop.

WHILE ENGAGED THIS ENGINE OWNS THE HOLD FILE: it writes a drive order while an
action runs and the plain hold (stand still) between actions, refreshed every
hold_refresh_s. `disengage()` releases it. If the mission dies the hold lapses
and the run explores again -- the run never drives a path nobody is watching.
`step(now)` is called from the mission's tick; planning runs as a job.
"""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np

from exploration_mission import rth_ladder

log = logging.getLogger("friday.exploration_mission.run_drive")

LIVE_FRESH_S = 3.0          # a fix older than this is not "where she is now"
LOST_S = 15.0               # driving, and no fix for this long: the attempt fails
REPLAN_S = 5.0              # BLOCKED_REPLANNING: re-plan this often while driving
PATH_TOL_M = 0.30           # a path point counts as passed within this
OBSTACLE_AHEAD_M = 0.30     # a stall marks the path this far ahead of her
OBSTACLE_RADIUS_M = 0.15
SPIN_MAX_S = 30.0           # a full turn takes ~5 s at 75 deg/s; this is the cap
BACK_MAX_S = 8.0
REAR_SECTOR_DEG = 30.0      # +-30 deg behind her (spec 4.11 rung 4)
RUNNING_CHECK_S = 5.0       # running() asks systemd: not every tick
_ids = itertools.count(1)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def remaining_m(pose_xy, path, i: int) -> float:
    """Distance left along `path` from pose, heading for point i."""
    x, y = pose_xy
    if not path:
        return 0.0
    i = min(i, len(path) - 1)
    d = math.hypot(path[i][0] - x, path[i][1] - y)
    for a, b in zip(path[i:], path[i + 1:]):
        d += math.hypot(b[0] - a[0], b[1] - a[1])
    return d


def plan_path(og, start_xy, goal_xy, inflate_m: float, obstacles=()) -> list:
    """File 14's planner on the mission's map. [] = no route (never a guess)."""
    import path_planner
    blocked = None
    if obstacles:
        ring = [(ox + OBSTACLE_RADIUS_M * math.cos(t), oy + OBSTACLE_RADIUS_M * math.sin(t))
                for ox, oy in obstacles for t in np.linspace(0, 2 * math.pi, 12,
                                                              endpoint=False)]
        blocked = np.asarray(list(obstacles) + ring, dtype=np.float64)
    prep = path_planner.preprocess(og, extra_blocked=blocked, inflate_m=inflate_m)
    ix, iy = prep.grid.cells_of(np.array([start_xy, goal_xy], dtype=np.float64))
    cells, _ = path_planner.plan(prep, (int(ix[0]), int(iy[0])),
                                 (int(ix[1]), int(iy[1])))
    if not cells:
        return []
    pts = [tuple(map(float, p)) for p in path_planner.cells_to_world(cells, prep.grid)]
    # The last point is the goal itself, not the centre of its cell.
    pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    return pts[1:] if len(pts) > 1 else pts


class RunDrive:
    """The ladder's engine (and U2's), driving through the held run."""

    def __init__(self, run, map_og, cfg, clock, runner, inflate_m: float):
        self.run, self.map_og, self.cfg, self.clock = run, map_og, cfg, clock
        self.runner = runner
        self.inflate_m = float(inflate_m)
        r = cfg["return"]
        self.arrive = (r.arrive_tolerance_m, r.arrive_hysteresis_m, r.arrive_dwell_s)
        self.progress_min_m = float(cfg.explore.progress_min_m)
        self.progress_timeout_s = float(cfg.explore.progress_timeout_s)
        self.hold_ttl_s = float(cfg.run.hold_ttl_s)
        self.hold_refresh_s = float(cfg.run.hold_refresh_s)
        self.obstacles: list[tuple[float, float]] = []
        self.engaged = False
        self.action: dict | None = None
        self._outcome: str | None = None
        self._pose: tuple[float, float, float] | None = None
        self._fix_t: float | None = None
        self._t_fix_mono = -1e9
        self._scan: dict = {}
        self._t_hold = -1e9
        self._order = None
        self._running = (False, -1e9)
        self.log: list[dict] = []           # one row per finished action

    # ---- engaging --------------------------------------------------------------------
    def engage(self, now: float, pose=None) -> None:
        """Take the hold: from now on she stands still unless driving."""
        self.engaged = True
        if pose is not None and self._pose is None:
            self._pose = tuple(float(v) for v in pose)
        self._write_hold(now, force=True)

    def disengage(self, release: bool = True) -> None:
        """Let go. release=True: the run explores again at once. release=False:
        she is left on the PLAIN hold (never on a drive order) for whoever
        holds her next -- a question, a pause -- so there is no gap in which
        the RL could set off for a tick."""
        if not release and self.engaged:
            self.action = None
            self._write_hold(self.clock.monotonic(), force=True)
        self.engaged = False
        self.action = None
        if release:
            try:
                self.run.release()
            except Exception as exc:                    # noqa: BLE001
                log.warning("could not release the run: %s", exc)

    def hold_now(self, now: float) -> None:
        """Write the hold (or the order) now, whatever the refresh clock says."""
        self._write_hold(now, force=True)

    # ---- the ladder's engine interface ----------------------------------------------
    def available(self) -> bool:
        ok, t = self._running
        now = self.clock.monotonic()
        if now - t >= RUNNING_CHECK_S:
            try:
                ok = bool(self.run.running())
            except Exception:                           # noqa: BLE001
                ok = False
            self._running = (ok, now)
        return ok and self._pose is not None

    def attempt(self, goal_xy, waypoints=None, arrive=None) -> bool:
        """One drive to goal_xy. `arrive` = (tolerance, hysteresis, dwell):
        the way home's (0.35, 0.10, 2 s) unless given -- frontier-assist
        arrives at assist_arrive_m with no dwell, a go-to at 0.35 with none."""
        if not self.engaged or self._pose is None:
            return False
        now = self.clock.monotonic()
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        a = {"kind": "attempt", "id": f"a{next(_ids)}", "goal": goal, "t0": now,
             "path": None, "i": 0, "job": None, "t_plan": now,
             "best": None, "t_best": now, "t_driving_fix": now,
             "arrival": rth_ladder.ArrivalCheck(*(arrive or self.arrive)),
             "breadcrumbs": bool(waypoints)}
        self._start(a)
        if waypoints:
            # Rung 5: the trail is the path (it was driven); no A* round it.
            a["path"] = [tuple(map(float, w)) for w in waypoints] + [goal]
        else:
            a["job"] = self._plan_job(goal)
        return True

    def busy(self) -> bool:
        return self.action is not None

    def outcome(self) -> str | None:
        return self._outcome

    def clear_runtime_obstacles(self) -> None:
        if self.obstacles:
            log.info("runtime obstacles cleared (%d)", len(self.obstacles))
        self.obstacles.clear()

    def spin_relocalise(self) -> bool:
        """Rung 3: one slow full turn. The tracker matches all the way round,
        so the turn IS the relocalisation."""
        if not self.engaged or self._pose is None:
            return False
        self._start({"kind": "turn", "id": f"t{next(_ids)}", "t0": self.clock.monotonic(),
                     "turned": 0.0, "last_th": self._pose[2]})
        return True

    def back_up(self, metres: float) -> bool:
        if not self.engaged or self._pose is None:
            return False
        self._start({"kind": "back", "id": f"b{next(_ids)}", "t0": self.clock.monotonic(),
                     "metres": float(metres), "from": self._pose[:2]})
        return True

    def rear_clear_m(self) -> float | None:
        """Nearest return within +-30 deg of straight behind, metres. None when
        the sector shows nothing at all: no measurement is not 'clear'."""
        rear = [mm for deg, mm in self._scan.items()
                if abs(_wrap(math.radians(float(deg)) - math.pi))
                <= math.radians(REAR_SECTOR_DEG) and mm]
        if len(rear) < 3:
            return None
        return min(rear) / 1000.0

    def distance_to(self, goal_xy) -> float | None:
        if self._pose is None:
            return None
        return math.hypot(self._pose[0] - goal_xy[0], self._pose[1] - goal_xy[1])

    def pose(self):
        return self._pose

    def stop(self) -> None:
        if self.action is not None:
            self._finish("STOPPED", self.clock.monotonic())

    # ---- inner workings --------------------------------------------------------------
    def _start(self, action: dict) -> None:
        if self.action is not None:
            self._finish("SUPERSEDED", action["t0"])
        self.action, self._outcome = action, None
        log.info("drive: %s %s", action["kind"], action["id"])

    def _finish(self, outcome: str, now: float) -> None:
        a = self.action
        self.action, self._outcome = None, outcome
        row = {"kind": a["kind"], "id": a["id"], "outcome": outcome,
               "s": round(now - a["t0"], 1)}
        if a["kind"] == "attempt":
            row["goal"] = [round(v, 2) for v in a["goal"]]
            d = self.distance_to(a["goal"])
            row["distance_m"] = None if d is None else round(d, 2)
        self.log.append(row)
        log.info("drive: %s", row)
        self._write_hold(now, force=True)          # stand still at once

    def _plan_job(self, goal):
        og = self.map_og()
        start = self._pose[:2]
        obstacles = list(self.obstacles)
        inflate = self.inflate_m

        def work():
            if og is None:
                return []
            return plan_path(og, start, goal, inflate, obstacles)
        return self.runner.start(work, "mission-plan")

    def _take_live(self, now: float) -> bool:
        """The run's latest fix. True if it is a NEW one."""
        try:
            live = self.run.live()
        except Exception:                               # noqa: BLE001
            live = None
        if not live or live.get("t") == self._fix_t:
            return False
        if live.get("age_s", 0.0) > LIVE_FRESH_S:
            return False
        self._fix_t = live.get("t")
        self._t_fix_mono = now
        self._pose = tuple(float(v) for v in live["pose"])
        self._scan = dict(live.get("scan") or {})
        return True

    def _write_hold(self, now: float, force: bool = False) -> None:
        if not self.engaged:
            return
        order = self._drive_order()
        if not force and order == self._order and now - self._t_hold < self.hold_refresh_s:
            return
        self._order, self._t_hold = order, now
        why = f"mission-{order['kind']}" if order else "mission-return"
        try:
            self.run.hold(why, self.hold_ttl_s, drive=order)
        except TypeError:                                # an adapter without drives
            self.run.hold(why, self.hold_ttl_s)

    def _drive_order(self) -> dict | None:
        a = self.action
        if a is None:
            return None
        if a["kind"] == "attempt":
            if not a["path"]:
                return None                              # still planning: stand still
            return {"id": a["id"], "kind": "path", "tol": PATH_TOL_M,
                    "path": [[round(x, 3), round(y, 3)] for x, y in a["path"]]}
        if a["kind"] == "turn":
            return {"id": a["id"], "kind": "turn", "dir": 1}
        return {"id": a["id"], "kind": "back"}

    def step(self, now: float) -> None:
        """Called every mission tick while engaged. Never blocks."""
        if not self.engaged:
            return
        new = self._take_live(now)
        a = self.action
        if a is not None:
            getattr(self, "_step_" + a["kind"])(a, now, new)
        self._write_hold(now)

    def _step_attempt(self, a: dict, now: float, new: bool) -> None:
        job = a["job"]
        if job is not None and job.done:
            a["job"] = None
            path = job.result if job.error is None else []
            if path:
                if a["path"] is None or a["path"] != path:
                    a["path"], a["i"] = path, 0
                    a["id"] = f"{a['id'].split('.')[0]}.{next(_ids)}"
            elif a["path"] is None:
                log.info("drive: no route on the map to %s", a["goal"])
                self._finish("NO_ROUTE", now)
                return
            # a re-plan that found nothing keeps the path she is on
        if a["path"] is None:
            return
        if new:
            a["t_driving_fix"] = now
            # which point she is heading for, as the run counts them
            p = self._pose[:2]
            while (a["i"] < len(a["path"]) - 1
                   and math.hypot(a["path"][a["i"]][0] - p[0],
                                  a["path"][a["i"]][1] - p[1]) <= PATH_TOL_M):
                a["i"] += 1
            rem = remaining_m(p, a["path"], a["i"])
            if a["best"] is None or rem < a["best"] - self.progress_min_m:
                a["best"], a["t_best"] = rem, now
            if a["arrival"].update(self.distance_to(a["goal"]), now):
                self._finish("ARRIVED", now)
                return
        if now - a["t_best"] > self.progress_timeout_s:
            self._mark_stall(a)
            self._finish("STALLED", now)
            return
        if now - a["t_driving_fix"] > LOST_S:
            self._finish("LOST", now)
            return
        if (not a["breadcrumbs"] and a["job"] is None
                and now - a["t_plan"] >= REPLAN_S):
            a["t_plan"] = now
            a["job"] = self._plan_job(a["goal"])

    def _mark_stall(self, a: dict) -> None:
        """A runtime obstacle where the path was blocked (File 14's
        RuntimeObstacles): the next attempt plans round it until rung 2 clears it."""
        if self._pose is None or not a["path"]:
            return
        x, y = self._pose[:2]
        tx, ty = a["path"][min(a["i"], len(a["path"]) - 1)]
        d = math.hypot(tx - x, ty - y)
        if d < 1e-6:
            return
        k = min(OBSTACLE_AHEAD_M, d) / d
        self.obstacles.append((x + (tx - x) * k, y + (ty - y) * k))
        log.info("runtime obstacle at (%.2f, %.2f)", *self.obstacles[-1])

    def _step_turn(self, a: dict, now: float, new: bool) -> None:
        if new:
            th = self._pose[2]
            a["turned"] += abs(_wrap(th - a["last_th"]))
            a["last_th"] = th
        if a["turned"] >= 2.0 * math.pi or now - a["t0"] > SPIN_MAX_S:
            self._finish("DONE", now)

    def _step_back(self, a: dict, now: float, new: bool) -> None:
        moved = math.hypot(self._pose[0] - a["from"][0], self._pose[1] - a["from"][1])
        if moved >= a["metres"] or now - a["t0"] > BACK_MAX_S:
            self._finish("DONE", now)

    def snapshot(self) -> dict:
        a = self.action
        return {"engaged": self.engaged,
                "action": None if a is None else {"kind": a["kind"], "id": a["id"]},
                "outcome": self._outcome, "obstacles": len(self.obstacles),
                "pose": None if self._pose is None else [round(v, 2) for v in self._pose],
                "log": self.log[-10:]}
