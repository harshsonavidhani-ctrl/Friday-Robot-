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
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from exploration_mission import grid_utils
from exploration_mission.durable import atomic_write_json  # noqa: F401 (re-export)

LIVE_NAME = "live.json"
HOLD_NAME = "hold.json"
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
                "way_back_distance_m": wb.get("distance_m")}

    def running(self) -> bool:
        from mcp_server.tools import autonomy
        return bool(autonomy._running())

    def start(self, minutes: float) -> dict:
        """MOTION: starts the RL run (friday-explore@N) through the MCP tool
        module's own function -- lease first, then the unit."""
        from mcp_server.tools import autonomy
        return json.loads(autonomy.start_exploration(minutes=minutes))

    def stop(self) -> str:
        from mcp_server.tools import autonomy
        return autonomy.stop_exploration()

    def hold(self, why: str, ttl_s: float) -> None:
        """Ask the run to stand still for `ttl_s` (refreshed by the caller)."""
        now = self.clock.wall()
        atomic_write_json(self.hold_file, {
            "until": now + min(float(ttl_s), HOLD_MAX_AHEAD_S),
            "why": str(why), "t": now, "pid": os.getpid()})

    def release(self) -> None:
        try:
            self.hold_file.unlink()
        except FileNotFoundError:
            pass

    def held(self) -> dict | None:
        data = _read_json(self.hold_file)
        if not isinstance(data, dict):
            return None
        until = float(data.get("until", 0.0))
        now = self.clock.wall()
        if until <= now or until - now > HOLD_MAX_AHEAD_S:
            return None
        return data


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

    def inflate_m(self) -> float:
        """File 14's own inflation (slam/plan.py::passable_mask)."""
        import config
        return float(config.NAV_INFLATE_M)

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
    """Speaking only, through nav_voice's queue -> tts.speak() (CLAUDE.md
    PART 4). Listening lives in friday-brain (the mic has one owner) and
    arrives in Step 11 as MissionWatch; until then `listen_once` says so."""

    def __init__(self, voice=None):
        self._voice = voice

    def _v(self):
        if self._voice is None:
            import nav_voice
            self._voice = nav_voice.VOICE
        return self._voice

    def say(self, text: str) -> str:
        return self._v().say_text(text)

    def listen_once(self, timeout_s: float) -> str | None:
        raise NotImplementedError("listening arrives with the voice hook "
                                  "(File 21 step 11)")


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


class Adapters:
    """Everything the mission may touch, in one bundle."""

    def __init__(self, *, clock=None, run=None, slam=None, face=None,
                 voice=None, detector=None, memory=None, env=None, power=None,
                 places=None):
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


class FakeRun:
    """A scripted RL run: set `.live_fix`/`.status_data`; records orders."""

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

    def start(self, minutes):
        self.orders.append(("start", minutes))
        self.is_running = True
        return {"started": True, "minutes": minutes}

    def stop(self):
        self.orders.append(("stop",))
        self.is_running = False
        return "Stopped exploring."

    def hold(self, why, ttl_s):
        self.orders.append(("hold", why))
        self._hold = {"until": self.clock.wall() + min(ttl_s, HOLD_MAX_AHEAD_S),
                      "why": why}

    def release(self):
        self.orders.append(("release",))
        self._hold = None

    def held(self):
        if self._hold and self._hold["until"] > self.clock.wall():
            return dict(self._hold)
        return None


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


class FakeDetector:
    def __init__(self, running=False, labels=()):
        self._s = {"running": running, "age_s": 0.0 if running else None,
                   "labels": list(labels)}

    def status(self):
        return dict(self._s)


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
                    places=kw.pop("places", None) or FakePlaces())


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
    return report


if __name__ == "__main__":
    import sys
    if "--probe" not in sys.argv[1:]:
        print("usage: python -m exploration_mission.adapters --probe")
        raise SystemExit(2)
    probe()
