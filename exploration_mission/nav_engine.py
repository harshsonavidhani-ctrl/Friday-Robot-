"""The way home on the map she built (File 21, 2026-10-02: Harsh, "map path
first").

The first real mission came home the RL run's way -- aim at the start, dodge
locally -- and was pinned 1 m out behind furniture. Now, when exploring ends,
the mission PARKS the run and File 14's engine drives her home on the
MISSION'S OWN MAP (the one built from the run's sweeps during that trip):
A* round what she has seen, one guarded move per tick, re-planning round
anything new, real back-ups (MCP move() at the calibrated speed). The return
ladder (rth_ladder.py) climbs its rungs through this engine; HELP comes last.

    the mission's thread                  friday-mcp's event loop
    --------------------                  -----------------------
    MissionNav.attempt(goal)  --MCP-->    mission_engine("attempt")
                                            HOST.attempt(): wait for the run to
                                            end and give the hardware back,
                                            find her on the mission map (File
                                            14's search, 1.5 m round her last
                                            tracked pose), Mapper on a COPY of
                                            it, NavController.return_to(...)
    MissionNav.busy()/outcome()  <------  HOST's state (same process)

The mission's map is never written: File 14 folds sweeps into a copy.
Everything that moves is File 14's GuardedRobot -> movement.move() ->
safety.guarded_move(); a "stop" ends it like any other journey (it runs as an
ordinary nav_tools journey, so map_in_use and "Friday, stop" see it).

FRAMES. The mission's map is in the START frame (the start is (0, 0, 0)).
It was built from the run's robot-frame sweeps (forward 0); File 14's Mapper
reads friday-mcp's raw sweeps and turns them by LIDAR_FORWARD_DEG -- the same
robot frame. Her first fix on it must be SURE (File 14's restricted search):
a guess would drive her home to the wrong place, so an unsure fix refuses and
the ladder moves on (spin + look again, then HELP).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import threading
import time

log = logging.getLogger("friday.exploration_mission.engine")

# Where to look for her: round the last pose the run's tracker was SURE of.
# 1.5 m only -- File 14's restricted search there was never wrong on the real
# map (0 of 6); 2026-10-02 16:19 a 3 m fallback put her 2 m from where she
# was, on a look-alike, and every attempt after drove at the wrong place.
FIND_RADIUS_M = 1.5
LOOK_TURNS = 4                   # the look-again rung: up to a full turn
RUN_GONE_WAIT_S = 60.0
SETTLE_S = 4.0                   # the LiDAR's first sweeps after a run are partial
SPIN_DEG = 90.0
REAR_SECTOR_DEG = 30.0           # +-30 deg behind her (spec 4.11 rung 4)


class _QuietVoice:
    """File 14's own lines, minus the ones the MISSION says in its own words
    (setting off, arriving, giving up): two voices for one robot contradict
    each other ("say try again" vs "say Friday, come back")."""

    SKIP = {"setting_off", "arrived", "arrived_start", "arrived_home", "failsafe",
            "no_route", "cannot_tell", "lost", "busy"}

    def __init__(self, inner):
        self.inner = inner

    def say(self, key, *a, **k):
        if key in self.SKIP:
            log.info("(the mission speaks for this) %s", key)
            return ""
        return self.inner.say(key, *a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class EngineHost:
    """Lives in friday-mcp. One action at a time; its state is read by the
    mission's thread directly (same process)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.busy = False
        self.outcome: str | None = None
        self.why = ""
        self.action = ""
        self.mapper = None
        self.controller = None
        self.robot = None
        self.og_id = None
        self.reports: list[dict] = []

    # -- what the mission reads --------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            pose = tuple(self.mapper.pose) if self.mapper is not None else None
            return {"busy": self.busy, "outcome": self.outcome, "why": self.why,
                    "action": self.action,
                    "pose": None if pose is None else [round(v, 3) for v in pose],
                    "reports": list(self.reports[-5:])}

    def pose(self):
        return None if self.mapper is None else tuple(float(v) for v in self.mapper.pose)

    def distance_to(self, goal) -> float | None:
        p = self.pose()
        return None if p is None else math.hypot(p[0] - goal[0], p[1] - goal[1])

    def clear_runtime_obstacles(self) -> None:
        if self.controller is not None and hasattr(self.controller, "obstacles"):
            import nav_state_machine as nsm
            self.controller.obstacles = nsm.RuntimeObstacles()

    def reset(self) -> None:
        """A new return: forget the last one's tracker (the map may differ)."""
        with self._lock:
            self.mapper = self.controller = self.robot = None
            self.og_id, self.outcome, self.why, self.reports = None, None, "", []

    # -- starting an action (on friday-mcp's loop) -----------------------------
    def request(self, action: str, args: dict) -> dict:
        with self._lock:
            if self.busy:
                return {"started": False, "why": f"busy ({self.action})"}
            self.busy, self.outcome, self.why, self.action = True, None, "", action
        work = {"attempt": self._attempt, "back_up": self._back_up,
                "spin": self._spin}.get(action)
        if work is None:
            with self._lock:
                self.busy = False
            return {"started": False, "why": f"unknown action {action!r}"}
        from mcp_server.tools import nav_tools
        nav_tools._start_journey(f"mission {action}", self._wrap(work, args))
        return {"started": True}

    async def _wrap(self, work, args: dict):
        out, why = "FAILED", ""
        try:
            out, why = await work(**args)
        except asyncio.CancelledError:
            out, why = "STOPPED", "cancelled"
            raise
        except Exception as exc:                        # noqa: BLE001
            log.exception("mission engine %s failed", self.action)
            out, why = "FAILED", f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self.busy, self.outcome, self.why = False, out, why
            log.info("mission engine %s -> %s %s", self.action, out, why)
        return out

    # -- getting ready: the run gone, the hardware back, her place on the map ------
    async def _ready(self) -> str:
        """'' when the tracker is ready, else why not."""
        from exploration_mission import service
        from mcp_server.tools import autonomy, nav_tools
        mgr = service.get()
        if mgr is None:
            return "no mission"
        og = mgr.analysis.map.og
        if og is None or not getattr(og, "scans", 0):
            return "the mission has no map yet"
        if self.mapper is not None and self.og_id == id(og):
            return ""
        t0 = time.monotonic()
        while autonomy._running() and time.monotonic() - t0 < RUN_GONE_WAIT_S:
            await asyncio.sleep(1.0)
        if autonomy._running():
            return "the exploration run is still out"
        nav_tools._reclaim_if_run_gone()
        await asyncio.sleep(SETTLE_S)
        import nav_follower
        import nav_state_machine as nsm
        import nav_voice
        from mcp_server import hardware
        from slam.mapper import Mapper
        near = tuple(getattr(mgr, "last_matched_pose", None) or mgr.pose)
        fix = await self._find(mgr, og, near)
        if fix is None:
            return "not sure where I am on the map I built"
        log.info("found on the mission map at (%.2f, %.2f) %.0f deg (tracked "
                 "(%.2f, %.2f))", fix[0], fix[1], math.degrees(fix[2]), near[0], near[1])
        with self._lock:
            self.robot = self.robot or nav_follower.GuardedRobot()
            self.controller = nsm.NavController(robot=self.robot,
                                                voice=_QuietVoice(nav_voice.VOICE))
            self.mapper = Mapper(grid=copy.deepcopy(og), pose=tuple(fix))
            self.og_id = id(og)
        return ""

    async def _find(self, mgr, og, near):
        scan = await asyncio.to_thread(self._sweep, mgr)
        if not scan:
            return None
        return await asyncio.to_thread(mgr.a.slam.localise_on, og, scan,
                                       tuple(near)[:2], FIND_RADIUS_M)

    def _arm(self) -> None:
        """Every action starts armed, as File 14 arms every journey
        (nav_tools._controller). File 14's own give-up (FAILSAFE) calls stop,
        and that flag would otherwise end every later rung in under a second
        ("stop was called", 2026-10-02 16:20). A PERSON's stop never reaches
        here: the mission leaves RETURN_TO_START first."""
        import nav_follower
        from mcp_server import hardware
        if self.robot is None:
            self.robot = nav_follower.GuardedRobot()
        hardware.ABORT.clear()
        self.robot.arm()

    @staticmethod
    def _sweep(mgr):
        sweeps = []
        for _ in range(3):
            s = mgr.a.lidar.sweep()
            if s:
                sweeps.append(s)
            time.sleep(0.15)
        return sweeps[-1] if sweeps else None

    # -- the actions --------------------------------------------------------------
    async def _attempt(self, goal, waypoints=None):
        why = await self._ready()
        if why:
            return "REFUSED", why
        import nav_state_machine as nsm
        import nav_voice
        self._arm()
        # A fresh controller per attempt: File 14 counts stuck-recoveries per
        # controller, and the ladder's next rung is a new try, not the old one's
        # leftovers (the same reset "try again" gives).
        self.controller = nsm.NavController(robot=self.robot,
                                            voice=_QuietVoice(nav_voice.VOICE))
        legs = [tuple(w) for w in (waypoints or [])] + [tuple(goal)]
        for i, xy in enumerate(legs):
            last = i == len(legs) - 1
            report = await self.controller.return_to(
                xy, self.mapper, label="where I started" if last else "my trail")
            self.reports.append({"leg": i, "arrived": bool(report.arrived),
                                 "why": report.why, "replans": report.replans})
            if getattr(report.state, "name", "") == "FAILSAFE" or not report.arrived:
                # The mission's ladder decides what next; File 14's own FAILSAFE
                # record would make a later "try again" redo this drive.
                nsm.clear_failsafe()
                return "FAILED", report.why or "did not arrive"
        return "ARRIVED", ""

    async def _back_up(self, m: float = 0.25):
        why = await self._ready()
        if why:
            return "REFUSED", why
        import config
        self._arm()
        speed = float(getattr(config, "MISSION_BACKUP_SPEED_M_S", 0.45))
        seconds = min(max(float(m), 0.05) / speed, float(config.MAX_REVERSE_SECONDS))
        said = await self._move("backward", seconds)
        if "REFUSED" in said.upper():
            return "REFUSED", said[:120]
        await self._relocate(back_m=float(m))
        return "DONE", said[:120]

    async def _move(self, direction: str, seconds: float) -> str:
        """ONE guarded move, looking first -- exactly as File 14's follower
        does (nav_follower.Follower._execute): embodiment.check refuses any
        move, turns included, with no fresh observation. 2026-10-02 22:45
        every look-again turn came back "Friday has not looked at the room
        yet. Call observe() first" and she never turned."""
        await self.robot.observe()
        return str(await self.robot.drive_seconds(direction, seconds))

    async def _spin(self):
        """Look again (rung 3): a quarter turn at a time, up to a full turn,
        finding her after each -- on the tracker's map when she has one, else
        from scratch round the last sure pose. Every turn a guarded move."""
        import config
        from exploration_mission import service
        mgr = service.get()
        why = await self._ready()
        if why and "not sure" not in why:
            return "REFUSED", why
        self._arm()
        deg_s = float(getattr(config, "AGENCY_TURN_DEG_PER_S", 75.0))
        said = ""
        for _ in range(LOOK_TURNS):
            said = await self._move("left", SPIN_DEG / deg_s)
            if "REFUSED" in said.upper():
                return "REFUSED", said[:120]
            if self.mapper is not None:
                if await self._relocate(turn_deg=SPIN_DEG):
                    return "DONE", str(said)[:120]
            elif mgr is not None and not await self._ready():
                return "DONE", "found on the map I built"
        return "FAILED", f"not found after a full turn ({str(said)[:80]})"

    async def _relocate(self, back_m: float = 0.0, turn_deg: float = 0.0) -> bool:
        """After a manoeuvre File 14's follower did not command: find her again
        on the same map near where it should have put her."""
        from exploration_mission import service
        mgr = service.get()
        if mgr is None or self.mapper is None:
            return False
        x, y, th = self.mapper.pose
        guess = (x - back_m * math.cos(th), y - back_m * math.sin(th),
                 th + math.radians(turn_deg))
        scan = await asyncio.to_thread(self._sweep, mgr)
        fix = None if not scan else await asyncio.to_thread(
            mgr.a.slam.localise_on, self.mapper.grid, scan, guess[:2], 1.0)
        if fix is None:
            log.info("after the manoeuvre: not sure where I am (kept the last pose)")
            return False
        self.mapper.pose = tuple(float(v) for v in fix)
        return True


HOST = EngineHost()


# -- the mission's side: the ladder's engine ---------------------------------------

def _call_self(action: str, **args) -> dict:
    """Start an engine action on friday-mcp's own event loop, through its own
    MCP server (the home_dock pattern): the journey keeps every guard. A
    LOOPBACK call to this same process's server -- never the brain, never
    the internet (config.MCP_HTTP_URL is friday-mcp's local address)."""
    async def ask() -> str:
        import config
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        async with streamable_http_client(config.MCP_HTTP_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                res = await session.call_tool(
                    "mission_engine", {"action": action, "args": json.dumps(args)})
                return "".join(getattr(p, "text", "") or ""
                               for p in (res.content or []))
    try:
        return json.loads(asyncio.run(ask()))
    except Exception as exc:                            # noqa: BLE001
        log.warning("could not ask the engine to %s: %s: %s", action,
                    type(exc).__name__, exc)
        return {"started": False, "why": f"{type(exc).__name__}"}


class MissionNav:
    """adapters.NavAdapter's contract (rth_ladder.Ladder's `engine`), backed by
    File 14 on the mission's map. Only inside friday-mcp (service.build)."""

    def __init__(self, lidar=None, host: EngineHost | None = None, call=None):
        self.lidar = lidar
        self.host = host or HOST
        self._call = call or _call_self

    # the NavAdapter parts the mission uses outside the ladder
    def available_during_run(self) -> bool:
        return False

    def go(self, target: dict) -> bool:
        return False            # "go to X" with no mission: File 14's nav_go owns it

    def journey_running(self) -> str:
        from mcp_server.tools import nav_tools
        return str(nav_tools._journey_running() or "")

    # the ladder's engine
    def available(self) -> bool:
        return True

    def new_return(self) -> None:
        self.host.reset()

    def attempt(self, goal_xy, waypoints=None) -> bool:
        res = self._call("attempt", goal=list(goal_xy),
                         waypoints=[list(w) for w in (waypoints or [])])
        if not res.get("started"):
            log.info("engine attempt not started: %s", res.get("why"))
        return bool(res.get("started"))

    def busy(self) -> bool:
        return bool(self.host.busy)

    def outcome(self) -> str | None:
        return self.host.outcome

    def clear_runtime_obstacles(self) -> None:
        self.host.clear_runtime_obstacles()

    def spin_relocalise(self) -> bool:
        return bool(self._call("spin").get("started"))

    def back_up(self, metres: float) -> bool:
        return bool(self._call("back_up", m=float(metres)).get("started"))

    def rear_clear_m(self) -> float | None:
        """The nearest return within +-REAR_SECTOR_DEG behind her (robot frame,
        0 = ahead, so behind = 180)."""
        if self.lidar is None:
            return None
        try:
            scan = self.lidar.sweep()
        except Exception:                               # noqa: BLE001
            scan = None
        if not scan:
            return None
        near = [mm for b, mm in scan.items()
                if mm and abs(((float(b) - 180.0) + 180.0) % 360.0 - 180.0) <= REAR_SECTOR_DEG]
        return min(near) / 1000.0 if near else None

    def distance_to(self, goal_xy) -> float | None:
        return self.host.distance_to(goal_xy)

    def pose(self):
        return self.host.pose()

    def stop(self) -> None:
        try:
            from mcp_server.tools import movement
            movement.stop()
        except Exception as exc:                        # noqa: BLE001
            log.warning("engine stop failed: %s", exc)
