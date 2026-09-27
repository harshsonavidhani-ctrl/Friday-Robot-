"""environment_watch.py — File 19's eyes on the files exploration already writes.

Runs on ONE daemon thread inside the MCP server (started next to the scene
watcher). It reads; it never drives, never decides where to go, and never
holds anything the control loop needs:

  * once at start: the boot check (environments.boot_check) -- one LiDAR sweep,
    no movement, "same place?" -- and says so out loud only if it is NOT.
  * the navigation map (config.SLAM_MAP_FILE) changing: file 14's AreaWatch on
    the new grid -> a pending landmark; new despeckled obstacles inside a named
    place -> that place's obstacle log.
  * state/explore_mode.json going from running to finished: a run summary.
  * new lines in state/nav_log.jsonl arriving home: a run summary.

Everything is polled every ENV_WATCH_S from files, so a slow step here can
only delay a summary, never an exploration tick (those run in rl/'s own
process) or a tool call.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import config

log = logging.getLogger("friday.environment_watch")

_thread: threading.Thread | None = None
_stop = threading.Event()

# Where the explore run leaves its status and its map (rl/ writes these; this
# only reads them). A module value so the tests can point it at a temp dir.
STATE = config.ROOT / "state"


def start() -> None:
    global _thread
    if not getattr(config, "ENVIRONMENTS", True):
        log.info("environment watch off (ENVIRONMENTS = False)")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="friday-environments",
                               daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()


def _iso(text: str | None) -> float | None:
    try:
        return datetime.fromisoformat(text).timestamp() if text else None
    except ValueError:
        return None


class Watch:
    """The state between polls. Separate from the thread so it can be tested."""

    def __init__(self) -> None:
        import landmarks
        self.area = landmarks.AreaWatch(float(config.LANDMARK_MIN_AREA_M2))
        self.map_mtime = 0.0
        self.grid = None
        self.env_id = None
        self.explore_state = None
        self.nav_offset = None

    # -- the navigation map ------------------------------------------------------
    def map_changed(self) -> list[dict]:
        import environments
        from slam.grid import OccupancyGrid
        env = environments.active()
        if env and env["env_id"] != self.env_id:
            # A different environment: its map is not "new area" of the last.
            import landmarks
            self.area = landmarks.AreaWatch(float(config.LANDMARK_MIN_AREA_M2))
            self.grid, self.map_mtime, self.env_id = None, 0.0, env["env_id"]
        path = Path(config.SLAM_MAP_FILE)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        if mtime == self.map_mtime:
            return []
        self.map_mtime = mtime
        grid = OccupancyGrid.load(path)
        found = []
        discovery = self.area.update(grid)
        if discovery is not None:
            x, y = discovery.centroid
            import landmarks
            if landmarks.inside_known_place((x, y, 0.0)) is None:
                found.append(environments.add_pending(
                    x, y, discovery.area_m2, discovery.radius_m))
        if self.grid is not None:
            environments.log_obstacles(
                environments.obstacles_near_places(self.grid, grid))
        self.grid = grid
        return found

    # -- the explore run ---------------------------------------------------------
    def explore_finished(self) -> dict | None:
        try:
            status = json.loads((STATE / "explore_mode.json").read_text())
        except (OSError, ValueError):
            return None
        state = str(status.get("state") or "")
        before, self.explore_state = self.explore_state, state
        if before != "running" or state == "running":
            return None                      # only the running -> done edge
        import environments
        import run_summary
        started = _iso(status.get("started")) or time.time()
        ended = _iso(status.get("ended")) or time.time()
        after = environments.known_area_m2(STATE / "rl_map.npz")
        prev = environments.known_area_m2(STATE / "rl_map.prev.npz")
        gained = (after - prev) if after is not None and prev is not None else None
        came_back = None
        tail = ""
        try:
            tail = Path(status.get("log") or "").read_text(errors="replace")[-4000:]
            came_back = "back where I started" in tail
        except OSError:
            pass
        summary = run_summary.finish(run_summary.build(
            "explore", started, ended, gained, came_back=came_back))
        # EXPLORING ENDS AT HOME (home_dock.py): a run that ended on its own
        # hands over to the navigation engine to finish the way home.
        try:
            import home_dock
            go, why = home_dock.wants_home_after(status, tail)
            log.info("after the run: %s", why)
            if go:
                home_dock.request_home_after_explore()
        except Exception as exc:                    # noqa: BLE001
            log.warning("home after exploring skipped: %s: %s",
                        type(exc).__name__, exc)
        return summary

    # -- file 14's navigation home ---------------------------------------------
    def nav_arrivals(self) -> list[dict]:
        import nav_state_machine as nsm             # file 14's own log path
        path = Path(nsm.NAV_LOG)
        try:
            size = path.stat().st_size
        except OSError:
            return []
        if self.nav_offset is None or size < self.nav_offset:
            self.nav_offset = size           # start at the end: no old runs
            return []
        with path.open() as handle:
            handle.seek(self.nav_offset)
            lines = handle.readlines()
            self.nav_offset = handle.tell()
        import run_summary
        done = []
        for raw in lines:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            dest = str(row.get("destination") or "")
            if row.get("arrived") and dest in config.RUN_SUMMARY_HOME_NAMES:
                ended = float(row.get("at") or time.time())
                done.append(run_summary.finish(run_summary.build(
                    "return", ended - float(row.get("seconds") or 0), ended,
                    came_back=True, destination=dest,
                    driven_m=row.get("driven_m"))))
        return done


def _boot(watch: Watch) -> None:
    """Step 2, once per server start, only when the LiDAR has a sweep."""
    import environments
    environments.ensure()
    watch.env_id = (environments.active() or {}).get("env_id")
    scan = None
    deadline = time.time() + float(config.ENV_BOOT_WAIT_S)
    while scan is None and time.time() < deadline and not _stop.is_set():
        try:
            from mcp_server import hardware
            if hardware.lease_active() > 0:
                environments._write_status({"state": "unknown",
                                            "why": "a run holds the robot"})
                return
            from mcp_server.tools.mapping import _sweep
            scan = _sweep() or None
        except Exception as exc:                    # noqa: BLE001
            log.debug("no sweep yet: %s", exc)
            _stop.wait(3.0)
    result = environments.boot_check(scan)
    # HOME AT BOOT (home_dock.py): no home in this environment yet and a sure
    # fix (or a brand-new map) -> this spot is home. An existing one is kept.
    try:
        import home_dock
        home_dock.save_boot_home(result)
    except Exception as exc:                        # noqa: BLE001
        log.warning("home at boot skipped: %s: %s", type(exc).__name__, exc)
    if result.get("state") == "no_match" and result.get("ask"):
        try:
            import nav_voice
            nav_voice.VOICE.say_text(result["ask"])
        except Exception as exc:                    # noqa: BLE001
            log.info("could not say the boot question: %s", exc)


def _run() -> None:
    watch = Watch()
    try:
        _boot(watch)
    except Exception as exc:                        # noqa: BLE001
        log.warning("boot check failed: %s: %s", type(exc).__name__, exc)
    while not _stop.wait(float(config.ENV_WATCH_S)):
        for step in (watch.map_changed, watch.explore_finished,
                     watch.nav_arrivals):
            try:
                step()
            except Exception as exc:                # noqa: BLE001
                log.warning("environment watch: %s failed (%s: %s)",
                            step.__name__, type(exc).__name__, exc)
