"""The ONLY module in exploration_mission/ that touches existing Friday code.

File 21 (spec 3.4). Everything the mission needs from the rest of Friday --
the RL run, the SLAM map, File 14, the face, her voice, the detector, memory,
environments -- comes through one object here, so that:

  * the rest of the package is pure logic that tests drive with FakeAdapters;
  * the offline rule (hard rule 6) has one place to hold: nothing here imports
    the file 08 brain, any cloud client library, or anything that reaches a
    network, and every Friday import is LAZY (inside the method that needs
    it), so importing this package pulls in nothing of Friday's at all;
  * motion has one place to go through: the RL run (start/stop/hold) and
    File 14's engine. Nothing here opens a motor or a serial port.

THE SEAM WITH THE RL RUN (approved by Harsh 2026-09-29, flagged in rl/):
  state/exploration_mission/live.json  written by rl/way_back.py's tracker
      process, <= 1 Hz: the start tracker's pose (start frame), whether that
      sweep matched, the run's dead-reckoned pose, and the sweep itself.
  state/exploration_mission/hold.json  written HERE: while its "until" is in
      the future the run holds its motors at zero and decides nothing. It
      lapses by itself, so a crashed mission cannot leave her frozen.

    python -m exploration_mission.adapters --probe   # prints, never moves,
                                                     # never writes
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from exploration_mission import grid_utils
from exploration_mission.durable import atomic_write_json  # noqa: F401 (re-export)

log = logging.getLogger("friday.exploration_mission.adapters")

LIVE_NAME = "live.json"
HOLD_NAME = "hold.json"
RETURN_NAME = "return.json"
PAUSE_NAME = "pause.json"           # rl/way_back.py PAUSE_NAME (U3)
STATE_NAME = "mission_state.json"   # voice_link.STATE_NAME
# rl/telemetry.py REASON_CODES: the gates that mean "stopped for safety".
ESTOP_GATES = frozenset({"REFLEX_STOP", "visual_stop", "stale_lidar", "stale_tof"})
# The run refuses a hold further in the future than this, however it was
# written (see rl/way_back.py Hold) -- a wall clock stepping backwards must not
# turn a 3 s hold into an hour of standing still. Kept in step with it here.
HOLD_MAX_AHEAD_S = 30.0


def _friday_root() -> Path:
    import config
    return Path(config.ROOT)


def mission_state_dir() -> Path:
    return _friday_root() / "state" / "exploration_mission"


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


from exploration_mission.home_anchor import scan_to_bins  # noqa: E402,F401


# =============================================================================
# the real adapters
# =============================================================================

class Clock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class RunAdapter:
    """The RL run: its published pose/sweep, its status, start/stop, hold."""

    def __init__(self, state_dir: Path | None = None,
                 autonomy_file: Path | None = None, clock: Clock | None = None):
        self.state_dir = Path(state_dir) if state_dir else mission_state_dir()
        self.autonomy_file = (Path(autonomy_file) if autonomy_file else
                              _friday_root() / "state" / "rl_autonomy.json")
        self.clock = clock or Clock()

    @property
    def live_file(self) -> Path:
        return self.state_dir / LIVE_NAME

    @property
    def hold_file(self) -> Path:
        return self.state_dir / HOLD_NAME

    def live(self, max_age_s: float | None = None) -> dict | None:
        """The tracker's latest published fix, with `age_s`, or None."""
        data = _read_json(self.live_file)
        if not isinstance(data, dict) or "pose" not in data:
            return None
        age = self.clock.wall() - float(data.get("t", 0.0))
        if max_age_s is not None and age > max_age_s:
            return None
        data["age_s"] = age
        scan = data.get("scan") or {}
        data["scan"] = {int(k): int(v) for k, v in scan.items()}
        return data

    def status(self) -> dict | None:
        """The parts of rl_autonomy.json the mission reads (all read-only)."""
        data = _read_json(self.autonomy_file)
        if not isinstance(data, dict):
            return None
        target = data.get("EXPLORATION_TARGET") or {}
        mstate = data.get("MAP_STATE") or {}
        wb = data.get("way_back") or {}
        try:
            age = self.clock.wall() - self.autonomy_file.stat().st_mtime
        except OSError:
            age = None
        final = data.get("FINAL_EXECUTED_ACTION") or {}
        return {"tick": data.get("TICK"), "age_s": age,
                # "mission_hold" while the run is holding for us (its gate).
                "reason_code": data.get("REASON_CODE"),
                "final_v": final.get("v"), "final_omega": final.get("omega"),
                "frontiers": target.get("frontiers"),
                "exhausted": target.get("exhausted"),
                "planner_error": target.get("error") or "",
                "explored_m2_rl": mstate.get("explored_m2"),
                "gate": (data.get("SAFETY_STATUS") or {}).get("detail"),
                "way_back_phase": wb.get("phase"),
                "way_back_outcome": wb.get("outcome"),
                "way_back_distance_m": wb.get("distance_m"),
                # U4: the run's own detector, when it is running (rl/pi_deploy).
                "objects": [(o.get("label"), o.get("score", 0.0))
                            for o in (data.get("OBJECTS") or []) if isinstance(o, dict)],
                "objects_frames": data.get("OBJECTS_FRAMES") or 0}

    def running(self) -> bool:
        from mcp_server.tools import autonomy
        return bool(autonomy._running())

    def ended_how(self, since_wall: float | None = None) -> str:
        """How the last run ended, from rl/modes/explore.sh's own record
        (state/explore_mode.json): "self" (it finished: exit 0, its timer or
        nothing left), "person" (SIGINT / a stop / a crash -- no drive back),
        or "pending" (the record still says running: its exit trap has not
        written yet). A record older than this mission is "person".

        Why: the run's way back can go returning -> arrived -> done in ONE
        tick and exit without publishing it (2026-10-02 15:43, she was 0.09 m
        from the start when the timer fired), so rl_autonomy.json still said
        "exploring" and the mission took the end for a person's stop."""
        import config
        from exploration_mission import area_store
        data = _read_json(Path(config.ROOT) / "state" / "explore_mode.json")
        if not isinstance(data, dict):
            return "person"
        state = str(data.get("state") or "")
        if state == "running":
            return "pending"
        if since_wall is not None:
            try:
                if area_store.iso_to_wall(data.get("ended") or "") < since_wall - 5.0:
                    return "person"
            except (TypeError, ValueError):
                return "person"
        if state.startswith("completed") and data.get("exit_code") in (0, 134):
            return "self"
        return "person"

    def start(self, minutes: float) -> dict:
        """MOTION: starts the RL run (friday-explore@N) through the MCP tool
        module's own function -- lease first, then the unit."""
        from mcp_server.tools import autonomy
        return json.loads(autonomy.start_exploration(minutes=minutes))

    def stop(self) -> str:
        from mcp_server.tools import autonomy
        return autonomy.stop_exploration()

    def hold(self, why: str, ttl_s: float, drive: dict | None = None) -> None:
        """Ask the run to stand still for `ttl_s` (refreshed by the caller) --
        or, with `drive` (File 21 U1, run_drive.py), to drive that order: a
        path in the start frame, a turn, a back-up. rl/way_back.py drive_order."""
        now = self.clock.wall()
        data = {"until": now + min(float(ttl_s), HOLD_MAX_AHEAD_S),
                "why": str(why), "t": now, "pid": os.getpid()}
        if drive:
            data["drive"] = drive
        atomic_write_json(self.hold_file, data)

    def release(self) -> None:
        try:
            self.hold_file.unlink()
        except FileNotFoundError:
            pass

    # -- U3: a person's pause (written by friday-brain, cleared here) --
    def pause(self, by: str = "mission") -> None:
        atomic_write_json(self.state_dir / PAUSE_NAME,
                          {"at": self.clock.wall(), "by": str(by), "pid": os.getpid()})

    def unpause(self) -> None:
        try:
            (self.state_dir / PAUSE_NAME).unlink()
        except FileNotFoundError:
            pass

    def paused(self) -> bool:
        return (self.state_dir / PAUSE_NAME).exists()

    def publish_state(self, state: str, mission_id: str | None) -> None:
        """For friday-brain: is a mission's run out? (voice_link.mission_state)"""
        try:
            atomic_write_json(self.state_dir / STATE_NAME,
                              {"state": state, "mission": mission_id,
                               "t": self.clock.wall()})
        except OSError as exc:
            log.warning("mission state not published: %s", exc)

    def held(self) -> dict | None:
        data = _read_json(self.hold_file)
        if not isinstance(data, dict):
            return None
        until = float(data.get("until", 0.0))
        now = self.clock.wall()
        if until <= now or until - now > HOLD_MAX_AHEAD_S:
            return None
        return data

    # The run announces its own return and arrival ("Exploration completed.
    # Returning to where I started." / "I am back where I started."), so the
    # mission must not say those two lines again.
    announces_return = True

    @property
    def return_file(self) -> Path:
        return self.state_dir / RETURN_NAME

    def request_return(self, mission_id: str | None = None) -> None:
        """Ask the run to start ITS OWN way back now (rl/way_back.py reads
        this; honoured only while fresh and only if written after the run
        began)."""
        atomic_write_json(self.return_file, {"at": self.clock.wall(),
                                             "mission": mission_id,
                                             "pid": os.getpid()})

    def clear_return(self) -> None:
        try:
            self.return_file.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def estop(status: dict | None) -> bool:
        """Is the run's OWN safety layer holding her right now (ToF reflex, a
        floor hazard, a stale sensor)? Read from its published gate."""
        return bool(status) and status.get("reason_code") in ESTOP_GATES


class SlamAdapter:
    """The saved SLAM map (File 14's frame). Read, never written."""

    def __init__(self, map_file: Path | None = None,
                 session_file: Path | None = None):
        self._map_file = Path(map_file) if map_file else None
        self._session_file = Path(session_file) if session_file else None

    @property
    def map_file(self) -> Path:
        if self._map_file is not None:
            return self._map_file
        import config
        return Path(config.SLAM_MAP_FILE)

    def load(self):
        """The slam.grid.OccupancyGrid, or None when there is no map yet."""
        from slam.grid import OccupancyGrid
        try:
            return OccupancyGrid.load(self.map_file)
        except FileNotFoundError:
            return None

    def grid(self) -> grid_utils.Grid | None:
        g = self.load()
        return None if g is None else grid_utils.from_occupancy_grid(g)

    def planner_grid(self, og=None) -> grid_utils.Grid | None:
        """The map AS FILE 14's PLANNER SEES IT: isolated 1-2 cell wall specks
        cleared first (path_planner.despeckle -- each would otherwise inflate
        into a 60 cm disc of phantom wall), then normalised. This is the grid
        to hand frontiers/segmentation, so "reachable" there is reachable for
        the planner. `og` may be any slam OccupancyGrid (the mission's own)."""
        og = og if og is not None else self.load()
        if og is None:
            return None
        import path_planner
        clean = path_planner.despeckle(og)[0]
        return grid_utils.from_occupancy_grid(clean)

    def new_grid(self):
        """An empty slam OccupancyGrid for the mission's own map."""
        from slam.grid import OccupancyGrid
        return OccupancyGrid()

    def localise(self, scan_robot: dict) -> tuple[float, float, float] | None:
        """Where a robot-frame sweep was taken on the SAVED map, if -- and only
        if -- global_localise is sure (it refuses when two spots look alike).
        None = not placed; the mission then keeps its areas in its own frame."""
        og = self.load()
        if og is None or not scan_robot:
            return None
        import global_localise
        scan = {int(round(float(k))) % 360: int(v) for k, v in scan_robot.items()}
        try:
            fix = global_localise.localise(og, scan, forward_deg=0.0)
        except Exception:                             # noqa: BLE001
            return None
        # Fix.ok is a property; the pose is None whenever it refused.
        return tuple(float(v) for v in fix.pose) if fix.pose is not None else None

    def fit_onto_saved(self, og_mission):
        """Where a mission's START is on the saved map, by fitting the whole
        mission map onto it (map_growth.fit: File 14's global search on a
        room's worth of walls, then its matcher's polish). Far more distinctive
        than one sweep, which is why it can succeed where the start-of-mission
        fix refused. Reads the saved map, never writes it. Sure, or None."""
        saved = self.load()
        if saved is None or og_mission is None:
            return None
        import map_growth
        try:
            fix = map_growth.fit(saved, og_mission)
        except Exception as exc:                      # noqa: BLE001
            log.warning("whole-map fit failed: %s: %s", type(exc).__name__, exc)
            return None
        if fix.pose is None:
            log.info("whole-map fit refused: %s (best %.2f, runner-up %.2f)",
                     fix.why or "not sure", fix.score, fix.runner_up_score)
            return None
        return tuple(float(v) for v in fix.pose)

    def load_grid(self, path):
        """A saved slam OccupancyGrid (the mission's own map), or None."""
        from slam.grid import OccupancyGrid
        try:
            return OccupancyGrid.load(path)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def localise_on(self, og, scan_robot: dict, near_xy=None,
                    radius_m: float | None = None):
        """Where a robot-frame sweep was taken on `og`, by File 14's global
        search -- optionally only within `radius_m` of `near_xy`, which is what
        separates two rooms that look alike. Sure, or None: it never guesses."""
        if og is None or not scan_robot:
            return None
        import global_localise
        scan = {int(round(float(k))) % 360: int(v) for k, v in scan_robot.items()}
        allowed = None
        if near_xy is not None and radius_m is not None:
            nx, ny = og.shape
            xs = og.origin[0] + (np.arange(nx) + 0.5) * og.cell
            ys = og.origin[1] + (np.arange(ny) + 0.5) * og.cell
            allowed = (np.hypot(xs[:, None] - float(near_xy[0]),
                                ys[None, :] - float(near_xy[1])) <= float(radius_m))
        try:
            fix = global_localise.localise(og, scan, forward_deg=0.0,
                                           restrict_to=allowed)
        except Exception:                             # noqa: BLE001
            return None
        # Fix.ok is a property; the pose is None whenever it refused.
        return tuple(float(v) for v in fix.pose) if fix.pose is not None else None

    def inflate_m(self) -> float:
        """File 14's own inflation (slam/plan.py::passable_mask)."""
        import config
        return float(config.NAV_INFLATE_M)

    def body_radius_m(self) -> float:
        import config
        return float(config.NAV_ROBOT_RADIUS_M)

    def map_id(self, read_only: bool = False) -> str | None:
        """File 14's map_session_id. `read_only` reads the session file and
        never rotates it (the probe must not write)."""
        if read_only:
            path = (self._session_file or
                    _friday_root() / "state" / "map_session.json")
            data = _read_json(path)
            return None if not isinstance(data, dict) else data.get("id")
        import landmarks
        return landmarks.map_session_id(self.load())


class FaceAdapter:
    def names(self) -> tuple[str, ...]:
        import face_control
        return tuple(face_control.EMOTIONS)

    def set(self, name: str) -> str:
        """Through the state file; face_director adopts outside writes at its
        IDLE band, so a live safety face is never overridden."""
        import face_control
        return face_control.set_emotion(name)


class VoiceAdapter:
    """Speaking through nav_voice's queue -> tts.speak() (CLAUDE.md PART 4).
    Listening lives in friday-brain (the mic has one owner): with an `asker`
    (voice_link.Asker, given by service.build inside friday-mcp) a question is
    answered through MissionWatch there; without one there are no ears."""

    def __init__(self, voice=None, asker=None):
        self._voice = voice
        self.asker = asker

    def _v(self):
        if self._voice is None:
            import nav_voice
            self._voice = nav_voice.VOICE
        return self._voice

    def say(self, text: str) -> str:
        return self._v().say_text(text)

    def listen_once(self, timeout_s: float) -> str | None:
        if self.asker is None:
            raise NotImplementedError("no ears: the mission's voice link runs "
                                      "only inside friday-mcp")
        self.wait_until_quiet()
        return self.asker.listen_once(timeout_s)

    def wait_until_quiet(self, max_s: float = 30.0) -> None:
        """Her question must be SAID, and out of the room, before anyone listens
        for the answer. The line is queued (nav_voice speaks on its own thread),
        so right after say() nothing is playing yet and the mark still reads
        quiet: 2026-10-02 16:18 the capture opened at once and her own re-ask
        ("Sorry, I didn't catch that") was taken as his answer."""
        v = self._v()
        q = getattr(v, "_queue", None)
        t0 = time.monotonic()
        time.sleep(0.3)                     # the worker picks the line up
        while time.monotonic() - t0 < max_s:
            pending = q is not None and q.qsize() > 0
            try:
                import tts                                  # noqa: PLC0415
                speaking = tts.quiet_at() > time.time()
            except Exception:                               # noqa: BLE001
                speaking = False
            if not pending and not speaking:
                return
            time.sleep(0.1)
        log.warning("her voice did not go quiet in %.0f s; listening anyway", max_s)

    def can_listen(self) -> bool:
        """Can a question be answered right now? Only while friday-brain's
        MissionWatch is alive: curiosity's ask mode checks this BEFORE it says
        a word, so she never asks a question nobody can hear the answer to."""
        return self.asker is not None and self.asker.can_listen()


def _heard_file() -> Path:
    import config
    return Path(config.ROOT) / "state" / "exploration_mission" / "heard.json"


class TalkAdapter:
    """When anyone last talked, readable from friday-mcp (curiosity's "no
    conversation for 5 min", spec 4.15).

    His side: every "he spoke" clock lives inside friday-brain's wake sources,
    so the voice hook (step 11) calls `mark_heard()` there on every turn. Her
    side: tts.speak() already writes state/speaking.json from whatever process
    speaks. No heard.json = UNKNOWN (None), and unknown is never quiet."""

    def __init__(self, heard_file: Path | None = None,
                 speaking_file: Path | None = None, clock: Clock | None = None):
        self._heard = heard_file
        self._speaking = speaking_file
        self.clock = clock or Clock()

    def heard_file(self) -> Path:
        return self._heard or _heard_file()

    def _speaking_file(self) -> Path:
        if self._speaking is None:
            import config
            self._speaking = Path(config.ROOT) / "state" / "speaking.json"
        return self._speaking

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            data = json.loads(Path(path).read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def mark_heard(self, at: float | None = None) -> None:
        """friday-brain, at the end of every turn (step 11). Never raises."""
        try:
            path = self.heard_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(".heard.json.tmp")
            tmp.write_text(json.dumps({"at": self.clock.wall() if at is None
                                       else float(at)}))
            os.replace(tmp, path)
        except OSError as exc:
            log.debug("heard mark not written: %s", exc)

    def quiet_s(self) -> float | None:
        """Seconds since anyone talked, or None when nobody has ever marked
        a turn."""
        heard = self._read(self.heard_file()).get("at")
        if not isinstance(heard, (int, float)):
            return None
        last = float(heard)
        voice = self._read(self._speaking_file())
        for key in ("until", "at"):
            v = voice.get(key)
            if isinstance(v, (int, float)):
                last = max(last, float(v))
                break
        return max(0.0, self.clock.wall() - last)

    def quiet_mode(self) -> bool:
        try:
            import mode                                 # noqa: PLC0415
            return bool(mode.quiet())
        except Exception:                               # noqa: BLE001
            return False


class DetectorAdapter:
    """scene.py's detector, read from state/scene.json. Never loaded here."""

    def __init__(self, scene_file: Path | None = None, clock: Clock | None = None):
        self._scene_file = Path(scene_file) if scene_file else None
        self.clock = clock or Clock()

    @property
    def scene_file(self) -> Path:
        if self._scene_file is not None:
            return self._scene_file
        import config
        return Path(getattr(config, "SCENE_JSON",
                            _friday_root() / "state" / "scene.json"))

    def status(self) -> dict:
        import config
        period = float(getattr(config, "SCENE_PERIOD_S", 5.0))
        try:
            age = self.clock.wall() - self.scene_file.stat().st_mtime
        except OSError:
            return {"running": False, "age_s": None, "labels": []}
        data = _read_json(self.scene_file) or {}
        objects = data.get("objects") or []
        labels = sorted({str(o.get("label")) for o in objects
                         if isinstance(o, dict) and o.get("label")})
        return {"running": age <= 3.0 * period, "age_s": round(age, 1),
                "labels": labels}

    def sightings(self) -> tuple:
        """U4 (spec 4.7 point 9): (frame key, [(label, conf), ...]) for the
        detector's latest picture while it is RUNNING, else (None, []).
        The key changes with each new picture, so one frame counts once."""
        import config
        period = float(getattr(config, "SCENE_PERIOD_S", 5.0))
        try:
            mtime = self.scene_file.stat().st_mtime
        except OSError:
            return None, []
        if self.clock.wall() - mtime > 3.0 * period:
            return None, []
        data = _read_json(self.scene_file) or {}
        seen = [(str(o.get("label")), float(o.get("score", o.get("conf", 0.0)) or 0.0))
                for o in (data.get("objects") or [])
                if isinstance(o, dict) and o.get("label")]
        return ("scene", data.get("t", mtime)), seen


class MemoryAdapter:
    def backend(self) -> dict:
        import config
        path = Path(config.MEMORY_DB)
        mode = None
        if path.exists():
            try:
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
                try:
                    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
                finally:
                    con.close()
            except sqlite3.Error:
                mode = "unreadable"
        return {"kind": "sqlite", "path": str(path), "exists": path.exists(),
                "journal_mode": mode}


class EnvAdapter:
    """File 19's active environment id; "home" when none exists (spec 1)."""

    def env_id(self) -> str:
        import config
        path = Path(config.MEMORY_DB)
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
            try:
                row = con.execute(
                    "SELECT env_id FROM environments WHERE active=1").fetchone()
            finally:
                con.close()
        except sqlite3.Error:
            row = None
        return str(row[0]) if row and row[0] else "home"


class LidarAdapter:
    """Robot-frame sweeps (0 = ahead, CCW) for the start signature.

    Two sources, never both: while an RL run holds the hardware lease, the
    run's published sweep (already robot frame); otherwise friday-mcp's own
    LiDAR through mcp_server/tools/mapping.py's `_sweep` (the File 19 boot
    check's path), turned from raw bearings by config.LIDAR_FORWARD_DEG.

    ⚠ `in_server` MUST be True only inside friday-mcp. Anywhere else,
    hardware.ranges() would open a SECOND reader on the LiDAR's serial port
    that friday-mcp already holds; the default refuses instead.
    """

    def __init__(self, run: RunAdapter | None = None, in_server: bool = False,
                 sweep_fn: Callable[[], dict] | None = None,
                 forward_deg: float | None = None, live_max_age_s: float = 1.5):
        self.run = run
        self.in_server = in_server
        self._sweep_fn = sweep_fn
        self._forward = forward_deg
        self.live_max_age_s = live_max_age_s

    def _forward_deg(self) -> float:
        if self._forward is None:
            import config
            self._forward = float(config.LIDAR_FORWARD_DEG)
        return self._forward

    def _leased(self) -> bool:
        from mcp_server import hardware
        return hardware.lease_active() > 0.0

    def sweep(self) -> dict[float, int] | None:
        """One robot-frame sweep, or None when no source can give one."""
        from exploration_mission.home_anchor import raw_to_robot
        if self._sweep_fn is not None:                   # tests
            raw = self._sweep_fn()
            return None if raw is None else raw_to_robot(raw, self._forward_deg())
        if self.run is not None:
            fix = self.run.live(self.live_max_age_s)
            if fix is not None:
                return {float(k): int(v) for k, v in fix["scan"].items()}
        if not self.in_server or self._leased():
            return None
        from mcp_server.tools.mapping import _sweep
        raw = _sweep() or None
        return None if raw is None else raw_to_robot(raw, self._forward_deg())


class PowerAdapter:
    """No battery gauge is wired (config.BATTERY_SENSOR is None), so there is
    no power_low signal. None = unknown, and the mission never invents one."""

    def low(self) -> bool | None:
        return None


class PlacesAdapter:
    """File 14's landmarks: slam/places.py's `places` table, which always
    belongs to the ACTIVE environment (File 19). Called, never changed.

    Exact names only. `places.find` falls back to a unique SUBSTRING, which is
    right for "go to the kitch" and wrong for deciding whether two names are
    the same place, so identity here never goes through it.
    """

    def __init__(self, bundle: Callable[[str], None] | None = None):
        self._bundle = bundle

    def all(self) -> list[dict]:
        from slam import places
        return places.all()

    def find_exact(self, name: str) -> dict | None:
        want = " ".join(str(name).strip().lower().split())
        for row in self.all():
            if row["name"] == want:
                return row
        return None

    def upsert(self, name: str, x: float, y: float, radius_m: float,
               note: str) -> dict:
        from slam import places
        row = places.remember_here(name, (float(x), float(y), 0.0), kind="area",
                                   note=note, radius_m=float(radius_m))
        self._refresh(f"area saved: {row['name']}")
        return row

    def remove(self, name: str) -> bool:
        from slam import places
        gone = places.forget(name)
        if gone:
            self._refresh(f"area forgotten: {name}")
        return gone

    def stamp(self, map_id: str | None, text: str) -> str:
        """The note File 14 reads its map session from ([map:<id>] ...)."""
        import landmarks
        return landmarks.stamp_note(map_id or "unknown", text)

    def _refresh(self, reason: str) -> None:
        """Keep File 14's navigation bundle current, as remember_place does.
        Never raises: losing the bundle copy must not lose the area."""
        try:
            if self._bundle is not None:
                self._bundle(reason)
            else:
                import nav_bundle
                nav_bundle.save_quietly(reason)
        except Exception:                             # noqa: BLE001
            pass


class NavAdapter:
    """File 14's engine, as far as the mission may use it in step 7.

    ⚠ NOT DURING A RUN. The RL run holds the hardware lease (LiDAR + motors),
    so File 14 cannot drive while it is out, held or not. A go-to in the
    middle of exploring is therefore refused ("I can't do that right now")
    rather than faked; the spec's pause -> TARGET_NAV -> resume needs the run
    to give up the wheels, which it cannot do without losing its start frame.
    The return ladder (step 8) drives with File 14 only once the run has ended.
    """

    def available_during_run(self) -> bool:
        return False

    def go(self, target: dict) -> bool:
        """Start a journey to a saved place. Wired in step 11 with the voice
        hook (it needs friday-mcp's event loop); until then: not available."""
        return False

    def busy(self) -> bool:
        return False

    def outcome(self) -> str | None:
        return None

    def journey_running(self) -> str:
        """A File 14 journey (go to, go home, come back) has the robot: its
        name, else "". The same question map_in_use asks; read-only."""
        from mcp_server.tools import nav_tools          # noqa: PLC0415
        return str(nav_tools._journey_running() or "")

    # --- the return ladder's engine (rth_ladder.Ladder) ---
    # File 14's NavController lives on friday-mcp's event loop; the mission is
    # given that loop when it is started inside friday-mcp (the launcher line,
    # step 9). Until then there is no engine, and the ladder says so: it goes
    # straight to HELP instead of pretending to try.
    def available(self) -> bool:
        return False

    def attempt(self, goal_xy, waypoints=None) -> bool:
        return False

    def clear_runtime_obstacles(self) -> None:
        pass

    def spin_relocalise(self) -> bool:
        return False

    def back_up(self, metres: float) -> bool:
        return False

    def rear_clear_m(self) -> float | None:
        return None

    def distance_to(self, goal_xy) -> float | None:
        return None

    def pose(self):
        return None

    def stop(self) -> None:
        pass


class Adapters:
    """Everything the mission may touch, in one bundle."""

    def __init__(self, *, clock=None, run=None, slam=None, face=None,
                 voice=None, detector=None, memory=None, env=None, power=None,
                 places=None, lidar=None, nav=None, talk=None):
        self.clock = clock or Clock()
        self.run = run or RunAdapter(clock=self.clock)
        self.slam = slam or SlamAdapter()
        self.face = face or FaceAdapter()
        self.voice = voice or VoiceAdapter()
        self.detector = detector or DetectorAdapter(clock=self.clock)
        self.memory = memory or MemoryAdapter()
        self.env = env or EnvAdapter()
        self.power = power or PowerAdapter()
        self.places = places or PlacesAdapter()
        self.lidar = lidar or LidarAdapter(run=self.run)
        self.nav = nav or NavAdapter()
        self.talk = talk or TalkAdapter(clock=self.clock)


# =============================================================================
# fakes -- scripted, in-memory, for tests and sim
# =============================================================================

class FakePlaces:
    """File 14's places table as a dict (name -> row)."""

    def __init__(self, rows: list[dict] | None = None):
        self.rows: dict[str, dict] = {}
        for r in rows or []:
            self.rows[r["name"]] = dict(r)
        self.refreshes = 0

    def all(self):
        return [dict(r) for r in self.rows.values()]

    def find_exact(self, name):
        row = self.rows.get(" ".join(str(name).strip().lower().split()))
        return None if row is None else dict(row)

    def upsert(self, name, x, y, radius_m, note):
        name = " ".join(name.strip().lower().split())
        self.rows[name] = {"name": name, "x": float(x), "y": float(y),
                           "radius_m": float(radius_m), "kind": "area",
                           "note": note}
        self.refreshes += 1
        return dict(self.rows[name])

    def remove(self, name):
        gone = self.rows.pop(" ".join(name.strip().lower().split()), None)
        if gone is not None:
            self.refreshes += 1
        return gone is not None

    def stamp(self, map_id, text):
        return f"[map:{map_id or 'unknown'}] {text}".strip()

class FakeClock(Clock):
    def __init__(self, start: float = 1_000_000.0):
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class FakeRun:
    """A scripted RL run: set `.live_fix`/`.status_data`; records orders."""

    end_how = "person"          # what ended_how() says once it is gone

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.live_fix: dict | None = None
        self.status_data: dict | None = None
        self.is_running = False
        self.orders: list[tuple] = []
        self._hold: dict | None = None

    def live(self, max_age_s=None):
        if self.live_fix is None:
            return None
        age = self.clock.wall() - float(self.live_fix.get("t", self.clock.wall()))
        if max_age_s is not None and age > max_age_s:
            return None
        return dict(self.live_fix, age_s=age)

    def status(self):
        return None if self.status_data is None else dict(self.status_data)

    def running(self):
        return self.is_running

    def ended_how(self, since_wall=None):
        return "pending" if self.is_running else self.end_how

    def start(self, minutes):
        self.orders.append(("start", minutes))
        self.is_running = True
        self._paused = False             # a new run ignores an older pause (U3)
        return {"started": True, "minutes": minutes}

    def stop(self):
        self.orders.append(("stop",))
        self.is_running = False
        return "Stopped exploring."

    def hold(self, why, ttl_s, drive=None):
        self.orders.append(("hold", why))
        self._hold = {"until": self.clock.wall() + min(ttl_s, HOLD_MAX_AHEAD_S),
                      "why": why}
        if drive:
            self._hold["drive"] = dict(drive)

    def release(self):
        self.orders.append(("release",))
        self._hold = None

    def held(self):
        if self._paused:                      # a person's pause: held, no order
            return {"until": float("inf"), "why": "paused"}
        if self._hold and self._hold["until"] > self.clock.wall():
            return dict(self._hold)
        return None

    _paused = False
    states_published: list = []

    def pause(self, by="mission"):
        self.orders.append(("pause", by))
        self._paused = True

    def unpause(self):
        if self._paused:
            self.orders.append(("unpause",))
        self._paused = False

    def paused(self):
        return self._paused

    def publish_state(self, state, mission_id):
        self.states_published = self.states_published + [state]

    announces_return = False

    def request_return(self, mission_id=None):
        self.orders.append(("return", mission_id))

    def clear_return(self):
        pass

    estop = staticmethod(RunAdapter.estop)


class FakeLidar:
    """Robot-frame sweeps from a list (the last one repeats); None = no LiDAR."""

    def __init__(self, sweeps=None):
        self.sweeps = list(sweeps or [])
        self.calls = 0

    def sweep(self):
        self.calls += 1
        if not self.sweeps:
            return None
        return dict(self.sweeps[min(self.calls - 1, len(self.sweeps) - 1)])


class FakeNav:
    """File 14 for the tests.

    A go-to journey ends when the test says (`finish(outcome)`). The return
    ladder's attempts are SCRIPTED: `script` is the outcome of each attempt in
    turn ("FAILED", "ARRIVED", "REFUSED" = attempt() returns False); past the
    end of the script every attempt fails. Each attempt or manoeuvre takes
    `attempt_s` / `maneuver_s` of the fake clock. `calls` records the order.
    """

    def __init__(self, during_run: bool = True, clock=None, script=None,
                 engine: bool = True, attempt_s: float = 4.0,
                 maneuver_s: float = 2.0, rear_m: float | None = 1.0,
                 position=(1.5, 0.0, 0.0)):
        self.during_run = during_run
        self.journeys: list[dict] = []
        self._busy = False
        self._outcome: str | None = None
        self.clock = clock
        self.script = list(script or [])
        self.engine = engine
        self.attempt_s, self.maneuver_s, self.rear_m = attempt_s, maneuver_s, rear_m
        self.position = tuple(position)
        self.calls: list[tuple] = []
        self._until = 0.0
        self._pending: str | None = None
        self.journey = ""                   # a File 14 journey the mission did not start

    def journey_running(self):
        return self.journey

    # --- the ladder's engine ---
    def available(self):
        return self.engine

    def _start(self, seconds: float, outcome: str | None) -> None:
        self._until = self.clock.monotonic() + seconds
        self._pending, self._busy, self._outcome = outcome, True, None

    def attempt(self, goal_xy, waypoints=None):
        result = self.script.pop(0) if self.script else "FAILED"
        self.calls.append(("attempt", tuple(round(float(v), 2) for v in goal_xy),
                           None if waypoints is None else len(waypoints), result))
        if result == "REFUSED":
            return False
        self._start(self.attempt_s, result)
        return True

    def clear_runtime_obstacles(self):
        self.calls.append(("clear",))

    def spin_relocalise(self):
        self.calls.append(("spin",))
        self._start(self.maneuver_s, "SPUN")
        return True

    def back_up(self, metres):
        self.calls.append(("backup", metres))
        self._start(self.maneuver_s, "BACKED")
        return True

    def rear_clear_m(self):
        return self.rear_m

    def distance_to(self, goal_xy):
        return math.hypot(self.position[0] - goal_xy[0], self.position[1] - goal_xy[1])

    def pose(self):
        return self.position

    def stop(self):
        self.calls.append(("stop",))
        self._busy = False

    def available_during_run(self):
        return self.during_run

    def go(self, target):
        self.journeys.append(dict(target))
        self._busy, self._outcome = True, None
        return True

    def busy(self):
        if self._busy and self._pending is not None and self.clock is not None \
                and self.clock.monotonic() >= self._until:
            self._busy, self._outcome, self._pending = False, self._pending, None
            if self._outcome == "ARRIVED":
                self.position = (0.05, 0.0, 0.0)
        return self._busy

    def outcome(self):
        return self._outcome

    def finish(self, outcome: str = "ARRIVED"):
        self._busy, self._outcome, self._pending = False, outcome, None


class FakeSlam:
    def __init__(self, grid: grid_utils.Grid | None = None,
                 map_id: str = "map-fake", inflate_m: float = 0.30):
        self._grid, self._map_id, self._inflate = grid, map_id, inflate_m

    def load(self):
        return None

    def grid(self):
        return self._grid

    def planner_grid(self, og=None):
        return self._grid

    def inflate_m(self):
        return self._inflate

    def map_id(self, read_only=False):
        return self._map_id


class FakeFace:
    NAMES = ("neutral", "thinking", "happy", "sad", "angry", "surprised",
             "listening", "speaking", "confused", "dizzy", "success", "excited",
             "wink", "love", "curious", "proud", "bored", "sleepy", "sleeping",
             "scared")

    def __init__(self):
        self.shown: list[str] = []

    def names(self):
        return self.NAMES

    def set(self, name):
        if name not in self.NAMES:
            raise ValueError(name)
        self.shown.append(name)
        return name


class FakeVoice:
    """`replies` is consumed by listen_once; None in the list = silence."""

    def __init__(self, replies: list[str | None] | None = None):
        self.said: list[str] = []
        self.replies = list(replies or [])
        self.listened: list[float] = []

    def say(self, text):
        self.said.append(text)
        return text

    def listen_once(self, timeout_s):
        self.listened.append(timeout_s)
        return self.replies.pop(0) if self.replies else None

    def can_listen(self):
        return getattr(self, "listening", True)


class FakeTalk:
    """`quiet` = seconds since anyone talked (None = unknown)."""

    def __init__(self, quiet=None, quiet_mode=False):
        self.quiet = quiet
        self.mode_quiet = quiet_mode

    def quiet_s(self):
        return self.quiet

    def quiet_mode(self):
        return self.mode_quiet


class FakeDetector:
    def __init__(self, running=False, labels=()):
        self._s = {"running": running, "age_s": 0.0 if running else None,
                   "labels": list(labels)}

    def status(self):
        return dict(self._s)

    frames: list = []                    # [(label, conf), ...] per picture

    def sightings(self):
        if not self._s["running"] or not self.frames:
            return None, []
        self._n = getattr(self, "_n", 0) + 1
        return ("fake", self._n), list(self.frames[(self._n - 1) % len(self.frames)])


class FakeMemory:
    def backend(self):
        return {"kind": "sqlite", "path": ":memory:", "exists": True,
                "journal_mode": "memory"}


class FakeEnv:
    def __init__(self, env_id="home"):
        self._id = env_id

    def env_id(self):
        return self._id


class FakePower:
    def __init__(self, low=None):
        self._low = low

    def low(self):
        return self._low


def fake_adapters(**kw) -> Adapters:
    clock = kw.pop("clock", None) or FakeClock()
    return Adapters(clock=clock,
                    run=kw.pop("run", None) or FakeRun(clock),
                    slam=kw.pop("slam", None) or FakeSlam(),
                    face=kw.pop("face", None) or FakeFace(),
                    voice=kw.pop("voice", None) or FakeVoice(),
                    detector=kw.pop("detector", None) or FakeDetector(),
                    memory=kw.pop("memory", None) or FakeMemory(),
                    env=kw.pop("env", None) or FakeEnv(),
                    power=kw.pop("power", None) or FakePower(),
                    places=kw.pop("places", None) or FakePlaces(),
                    lidar=kw.pop("lidar", None) or FakeLidar(),
                    nav=kw.pop("nav", None) or FakeNav(),
                    talk=kw.pop("talk", None) or FakeTalk())


# =============================================================================
# the probe
# =============================================================================

def probe(out: Callable[[str], None] = print) -> dict:
    """What the mission would see right now. NEVER MOVES, NEVER WRITES:
    no hold, no start/stop, no map-session rotation, read-only sqlite."""
    a = Adapters()
    report: dict[str, Any] = {}

    def line(key: str, value: Any) -> None:
        report[key] = value
        out(f"{key:<18} {value}")

    live = a.run.live()
    if live is None:
        line("pose", "no live fix (no run since the publisher, or no run now)")
        line("scan", "none published")
    else:
        line("pose", {"start_frame": [round(v, 3) for v in live["pose"]],
                      "reason": live.get("reason"),
                      "age_s": round(live["age_s"], 1)})
        line("scan", f"{len(live['scan'])} bearings "
                     f"({int(np.isfinite(scan_to_bins(live['scan'])).sum())}/360 "
                     f"one-degree bins with a return)")
    st = a.run.status()
    line("rl_run", "no rl_autonomy.json" if st is None else
         {k: st[k] for k in ("tick", "frontiers", "exhausted", "way_back_phase")}
         | {"age_s": None if st["age_s"] is None else round(st["age_s"], 1)})
    line("hold", a.run.held() or "none")

    g = a.slam.grid()
    if g is None:
        line("map", f"no map at {a.slam.map_file}")
    else:
        line("map", {"file": str(a.slam.map_file), "shape": list(g.shape),
                     "resolution_m": g.resolution,
                     "origin": [round(v, 3) for v in g.origin],
                     "explored_m2": round(g.explored_m2(), 2),
                     "axis_order": "[ix, iy] = [x, y] (slam/grid.py)"})
    line("map_id", a.slam.map_id(read_only=True))
    line("inflate_m", a.slam.inflate_m())

    try:
        import nav_state_machine as nsm
        ok = all(hasattr(nsm.NavController, n)
                 for n in ("return_to", "_drive_to", "pause_exploration"))
        line("file14_engine", "importable; NavController.return_to/_drive_to "
             "present (dry check, nothing constructed or driven)" if ok
             else "NavController is missing an expected method")
    except Exception as exc:                          # noqa: BLE001
        line("file14_engine", f"import failed: {type(exc).__name__}: {exc}")

    line("emotions", ", ".join(a.face.names()))
    line("detector", a.detector.status())
    line("memory", a.memory.backend())
    line("env_id", a.env.env_id())
    line("power_low", "unavailable (no battery gauge)")
    quiet = a.talk.quiet_s()
    line("conversation", {"quiet_for_s": None if quiet is None else round(quiet),
                          "quiet_mode": a.talk.quiet_mode(),
                          "can_listen": a.voice.can_listen(),
                          "note": "no heard.json until the voice hook marks "
                                  "turns (step 11)" if quiet is None else ""})
    return report


if __name__ == "__main__":
    import sys
    if "--probe" not in sys.argv[1:]:
        print("usage: python -m exploration_mission.adapters --probe")
        raise SystemExit(2)
    probe()
