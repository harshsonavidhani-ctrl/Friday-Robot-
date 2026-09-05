"""exploration_mode.py — a timed, narrated exploration run.

    ./venv/bin/python exploration_mode.py --say "Friday, explore for 5 minutes"
    ./venv/bin/python exploration_mode.py --minutes 5
    ./venv/bin/python exploration_mode.py --minutes 2 --standalone
    ./venv/bin/python exploration_mode.py --say "explore for 3 minutes" --dry-run

WHY THIS IS ITS OWN FILE. The path planner, the obstacle avoidance and the
SLAM in rl/ and slam/ are verified and are not to be touched. This file adds a
LAYER ON TOP of them and changes nothing underneath: it starts the existing
autonomous run through the existing MCP tool, watches the map that run is
already building, says what it sees, and writes telemetry for the dashboard.
Delete this file and the robot explores exactly as it did before.

THE LEASE IS WHY THE FEATURES COME FROM THE MAP AND NOT FROM THE LiDAR.
An autonomous run takes a hardware lease on the range sensors for its whole
duration, so lidar_sweep() answers

    FAILED — HardwareLeased: the range sensors are in use by an autonomous run

for every second of the run. Polling the LiDAR during a run is therefore not a
thing that can work, and taking the sensors back would fight the very run this
is narrating. So the features are read out of the OCCUPANCY GRID the run is
already writing to state/rl_map.npz -- which needs no lease, costs no CPU on
the sensor path, and is strictly better for this job anyway: a corridor is two
parallel walls, which is a shape in a map and only an inference from one sweep.

`--standalone` is the other way round: no run is started, and the LiDAR is read
directly. That is for testing the narration, and for describing a room while
Friday stands still.

THIS FILE IS WHAT MAKES THE DURATION REAL, and that is not a design choice.
start_exploration(minutes) uses the number ONLY to size the hardware lease:

    run_s = DEFAULT_RUN_S if minutes is None else max(30.0, minutes * 60.0)
    hardware.lease_out(run_s + LEASE_MARGIN_S, holder="friday-explore")

The run itself is rl/modes/explore.sh, and its drive time is fixed at
`--duration 1200`. So the tool answers `"minutes": 2` and then drives for
twenty -- measured, 2026-09-06. Changing that script is out of scope here and
it is verified code, so the timer is enforced from OUTSIDE instead: when the
clock runs out this process calls stop_exploration(), which is the same call
"Friday, stop" makes and is the run's normal teardown.

Which means THIS PROCESS DYING MUST ALSO STOP THE RUN. If it simply exits, the
robot keeps driving for the full twenty minutes with nothing left watching it.
SIGINT and SIGTERM are therefore caught and turned into a clean shutdown, so
Ctrl-C, `kill`, and systemd stopping the unit all end the drive. SIGKILL still
cannot be caught -- do not use `kill -9` on this, use Ctrl-C or say "stop".

WHAT IT CAN AND CANNOT SEE. Walls, corridors, openings, turns, dead ends and
obstacles are all real geometry and come out of the grid. Objects come from the
camera detector. MIRRORS ARE A HEURISTIC AND ARE LABELLED AS ONE: a mirror is
invisible to LiDAR by definition -- the beam reflects away and the surface
reads as open space or as nothing at all -- so what this reports is a HOLE in
an otherwise closed boundary, at low confidence, described out loud as "might
be glass or a mirror". It is not a mirror detector and does not pretend to be.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import signal
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config

log = logging.getLogger("friday.explore_mode")

SCHEMA_VERSION = 1

# --- where the telemetry lands ---------------------------------------------
# live.json is overwritten every poll and is what a dashboard polls or watches;
# feed.jsonl only ever grows and is the record of the run; session-*.json is
# the whole thing in one file when it finishes.
TELEMETRY_DIR = config.ROOT / "state" / "exploration"
LIVE_FILE = TELEMETRY_DIR / "live.json"
FEED_FILE = TELEMETRY_DIR / "feed.jsonl"

# --- how often, and how loud -----------------------------------------------
POLL_S = 3.0                 # how often the map is re-read
SPEAK_MIN_GAP_S = 8.0        # a robot narrating every three seconds is noise
MAX_SPOKEN_PER_POLL = 1

# --- what counts as what, in metres ----------------------------------------
RAY_MAX_M = 4.0              # beyond this the grid is mostly unknown anyway
RAY_STEP_BEARINGS = 24       # every 15 degrees, matching lidar_sweep's sectors
NEAR_WALL_M = 1.20           # a surface this close is "beside me"
FLAT_TOLERANCE_M = 0.35      # ranges varying less than this read as one wall
CORRIDOR_OPEN_M = 2.00       # ...with this much room ahead, it is a corridor
OPENING_JUMP_M = 1.00        # a bearing this much longer than its neighbours
DEAD_END_M = 0.80
OBSTACLE_M = 1.50
OBSTACLE_STEP_M = 0.50
OCCUPIED_LOGODDS = 0.0       # > 0 is occupied, < 0 free, == 0 never seen


# --- the spoken request -----------------------------------------------------
# "Friday, explore for five minutes". The duration is pulled out and parsed
# rather than matched against a list, so "two", "2", "ten", "half an hour" and
# "90 seconds" all work -- app_control.parse_duration already does that job for
# timers and there is no reason for a second, worse copy of it here.
_EXPLORE_RE = re.compile(
    r"\b(?:go\s+)?(?:and\s+)?explor(?:e|ing)\b(?:\s+(?:the\s+)?"
    r"(?:room|house|place|area|around))?"
    r"(?:\s+(?:for|during)\b)?\s*(?P<amount>.+)?", re.IGNORECASE)

_BARE_EXPLORE_RE = re.compile(r"\b(explore|exploring|exploration)\b",
                              re.IGNORECASE)

# "STOP EXPLORING" IS NOT A REQUEST TO EXPLORE. It contains every word one
# does, and without this guard it parsed as "explore for the default twenty
# minutes" -- the exact opposite of the order, given to a robot with wheels.
_NOT_A_START_RE = re.compile(
    r"\b(stop|end|cancel|abort|quit|halt|don'?t|do not|no more|band)\b"
    r"[^.]{0,20}\b(explor\w*)\b", re.IGNORECASE)


def parse_request(text: str) -> float | None:
    """Minutes asked for, or None if this is not an explore command.

    Returns the DEFAULT when someone just says "explore" -- that is still an
    explore command, it simply did not name a duration. None means the sentence
    was about something else entirely.
    """
    if not text or not _BARE_EXPLORE_RE.search(text):
        return None
    if _NOT_A_START_RE.search(text):
        return None

    import app_control

    match = _EXPLORE_RE.search(text)
    amount = (match.group("amount") or "").strip() if match else ""
    # Trim the politeness and any trailing clause; parse_duration wants the
    # quantity and nothing else.
    amount = re.sub(r"\b(please|thanks|thank you|ok(ay)?|and then.*)\b", " ",
                    amount, flags=re.IGNORECASE)
    amount = amount.strip(" .,!?;:-")
    if not amount:
        return float(DEFAULT_MINUTES)
    try:
        seconds = app_control.parse_duration(amount)
    except Exception:                           # noqa: BLE001
        return float(DEFAULT_MINUTES)
    minutes = seconds / 60.0
    return max(MIN_MINUTES, min(minutes, MAX_MINUTES))


DEFAULT_MINUTES = 20.0
MIN_MINUTES = 0.5
MAX_MINUTES = 120.0


# --- one thing Friday noticed ----------------------------------------------
@dataclass
class Feature:
    """A structural feature or object, with where it is.

    Carries THREE coordinate frames on purpose. `local` is what the sentence
    means ("on my left"), `global` is where it is in the map and is the only
    one that survives the robot moving, and `grid` is the cell a dashboard
    draws it in. Storing only one of them means the other two get recomputed
    later by something that does not have the pose it was measured at.
    """
    id: str
    kind: str                       # wall | corridor | opening | turn | ...
    label: str                      # what gets said out loud
    ts: float
    bearing_deg: float              # relative to Friday's heading, + is left
    distance_mm: int
    local_x_m: float
    local_y_m: float
    global_x_m: float
    global_y_m: float
    grid_col: int
    grid_row: int
    cell_m: float
    confidence: float
    source: str                     # "map" | "lidar" | "camera"
    spoken: bool = False

    def key(self) -> tuple:
        """Same feature seen again? Rounded to half a metre, which is about
        how precisely a thing can be placed from a 5 cm grid and a pose that
        drifts."""
        return (self.kind, round(self.global_x_m * 2) / 2,
                round(self.global_y_m * 2) / 2)


# --- reading the map the run is building ------------------------------------
def mode_runs_quiet() -> bool:
    """Does the saved mode stop friday-mcp for the run?

    IF IT DOES, THIS NARRATOR MUST STAND DOWN. Every word it says and every
    sensor it reads goes through the MCP server, and --quiet-services stops
    that server for the whole run -- so attaching would give a narrator with
    no voice, no tools, and a teardown that calls stop_exploration() into a
    socket that is not there. The run already talks by itself: MODE_VOICE=1 is
    pi_deploy's own voice and does not touch MCP, which is where "Recovery
    successful. Continuing exploration." comes from.

    Read from the mode file rather than assumed, so turning quiet off in
    confident_explore.env brings the narrator back with no code change.
    """
    mode = config.ROOT / "rl" / "modes" / "confident_explore.env"
    try:
        for line in mode.read_text().splitlines():
            line = line.strip()
            if line.startswith("MODE_QUIET_SERVICES"):
                return line.split("=", 1)[1].strip().strip('"\'') == "1"
    except OSError:
        pass
    return False


def read_pose() -> dict | None:
    """Where the run thinks it is. Read-only; the run owns this file."""
    for path in (config.ROOT / "state" / "rl_map.pose.json",):
        try:
            pose = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if {"x", "y", "theta"} <= set(pose):
            return pose
    return None


def read_grid() -> tuple | None:
    """The occupancy grid, its origin and its cell size. None if unreadable."""
    import numpy as np
    for name in ("rl_map.npz", "map.npz"):
        path = config.ROOT / "state" / name
        if not path.exists():
            continue
        try:
            data = np.load(path)
            return (data["logodds"], tuple(data["origin"]),
                    float(data["cell"]), int(data.get("scans", 0)))
        except Exception:                       # noqa: BLE001
            continue
    return None


def _cast(grid, origin, cell: float, x: float, y: float,
          bearing_rad: float) -> tuple[float, str]:
    """Walk one ray out through the grid. Returns (metres, what it hit).

    This is a ray cast against a map, not a sensor reading, and the difference
    matters when reading the result: it reports what Friday has ALREADY seen
    and mapped in that direction, which is why it works while the LiDAR is
    leased and why it says "unknown" rather than "open" for ground never
    covered.
    """
    steps = max(1, int(RAY_MAX_M / cell))
    rows, cols = grid.shape
    for i in range(1, steps + 1):
        distance = i * cell
        col = int((x + distance * math.cos(bearing_rad) - origin[0]) / cell)
        row = int((y + distance * math.sin(bearing_rad) - origin[1]) / cell)
        if not (0 <= col < cols and 0 <= row < rows):
            return distance, "edge"
        value = float(grid[row, col])
        if value > OCCUPIED_LOGODDS:
            return distance, "wall"
        if value == 0.0:
            return distance, "unknown"
    return RAY_MAX_M, "open"


def virtual_scan(grid, origin, cell: float, pose: dict) -> list[tuple]:
    """A sweep taken from the map instead of the sensor.

    Returns [(bearing_deg_relative, metres, hit)] every 15 degrees, in the same
    shape lidar_sweep() reports so the feature rules below do not care which
    of the two produced them.
    """
    out = []
    for index in range(RAY_STEP_BEARINGS):
        relative = index * (360.0 / RAY_STEP_BEARINGS)
        relative = relative - 360.0 if relative > 180.0 else relative
        bearing = pose["theta"] + math.radians(relative)
        distance, hit = _cast(grid, origin, cell, pose["x"], pose["y"], bearing)
        out.append((relative, distance, hit))
    out.sort(key=lambda row: row[0])
    return out


# --- turning a sweep into things worth saying -------------------------------
def _side(bearing_deg: float) -> str:
    """The word a person would use for this bearing."""
    b = bearing_deg
    if -22.5 <= b <= 22.5:
        return "ahead"
    if 22.5 < b <= 67.5:
        return "ahead and to the left"
    if 67.5 < b <= 112.5:
        return "on the left"
    if 112.5 < b <= 157.5:
        return "behind me on the left"
    if -67.5 <= b < -22.5:
        return "ahead and to the right"
    if -112.5 <= b < -67.5:
        return "on the right"
    if -157.5 <= b < -112.5:
        return "behind me on the right"
    return "behind me"


def _at(bearing_deg: float, metres: float) -> str:
    """"about a metre and a half on the left" -- distance the way it is said."""
    where = _side(bearing_deg)
    if metres < 0.6:
        return f"right up against me {where}"
    if metres < 1.0:
        return f"about {metres * 100:.0f} centimetres {where}"
    return f"about {metres:.1f} metres {where}"


def _lookup(scan: list[tuple], bearing_deg: float) -> tuple[float, str]:
    """The scan entry nearest a bearing."""
    best = min(scan, key=lambda row: abs(row[0] - bearing_deg))
    return best[1], best[2]


def detect_features(scan: list[tuple], pose: dict, cell: float,
                    origin: tuple, source: str = "map") -> list[Feature]:
    """Every structural feature in one sweep.

    Ordered most-informative first, because only the top one or two get said
    out loud -- a robot that reads out nine features per poll is a robot
    nobody listens to.
    """
    found: list[Feature] = []
    if not scan:
        return found

    def make(kind: str, label: str, bearing: float, metres: float,
             confidence: float) -> Feature:
        angle = pose["theta"] + math.radians(bearing)
        gx = pose["x"] + metres * math.cos(angle)
        gy = pose["y"] + metres * math.sin(angle)
        return Feature(
            id=uuid.uuid4().hex[:12], kind=kind, label=label, ts=time.time(),
            bearing_deg=round(bearing, 1), distance_mm=int(metres * 1000),
            local_x_m=round(metres * math.cos(math.radians(bearing)), 3),
            local_y_m=round(metres * math.sin(math.radians(bearing)), 3),
            global_x_m=round(gx, 3), global_y_m=round(gy, 3),
            grid_col=int((gx - origin[0]) / cell),
            grid_row=int((gy - origin[1]) / cell),
            cell_m=cell, confidence=round(confidence, 2), source=source)

    forward, forward_hit = _lookup(scan, 0.0)
    left, left_hit = _lookup(scan, 90.0)
    right, right_hit = _lookup(scan, -90.0)

    # -- a corridor is two parallel walls with somewhere to go ---------------
    # Checked FIRST because it is the only feature that describes the room
    # rather than a piece of it, and because the two walls that make it would
    # otherwise be reported separately as two unrelated walls.
    if (left_hit == "wall" and right_hit == "wall"
            and left <= NEAR_WALL_M and right <= NEAR_WALL_M
            and forward >= CORRIDOR_OPEN_M):
        width = left + right
        found.append(make(
            "corridor",
            f"I'm in a corridor, about {width:.1f} metres wide, "
            f"running {forward:.1f} metres ahead",
            0.0, forward, 0.85))

    # -- a dead end ----------------------------------------------------------
    # UNKNOWN IS NOT A WALL, and conflating them makes every unexplored
    # direction a dead end. A ray stops at the first cell Friday has never
    # seen, so in a half-mapped room the three forward rays all terminate
    # within a metre and this rule fired on open floor she simply had not
    # driven yet -- "This looks like a dead end", said into an empty room.
    # A dead end is three rays that hit something SOLID.
    ahead_left, ahead_left_hit = _lookup(scan, 45.0)
    ahead_right, ahead_right_hit = _lookup(scan, -45.0)
    if (max(forward, ahead_left, ahead_right) <= DEAD_END_M
            and forward_hit == "wall" and ahead_left_hit == "wall"
            and ahead_right_hit == "wall"):
        found.append(make("dead_end", "This looks like a dead end.",
                          0.0, forward, 0.8))

    # -- openings and turns --------------------------------------------------
    # A bearing that runs much further than both its neighbours is a gap in a
    # surface: a doorway, a side passage, or the turn at the end of a corridor.
    for index, (bearing, metres, hit) in enumerate(scan):
        before = scan[index - 1][1]
        after = scan[(index + 1) % len(scan)][1]
        if metres - max(before, after) < OPENING_JUMP_M:
            continue
        if hit == "unknown" and metres < 1.0:
            continue                    # unmapped ground, not a doorway
        if 45.0 <= abs(bearing) <= 135.0:
            kind = "turn"
            label = (f"There's a turn {_side(bearing)}, "
                     f"{metres:.1f} metres of it")
        elif abs(bearing) < 45.0:
            kind = "opening"
            label = f"An opening {_side(bearing)}, going {metres:.1f} metres"
        else:
            continue
        found.append(make(kind, label, bearing, metres, 0.7))

    # -- flat walls ----------------------------------------------------------
    # Three or more adjacent bearings hitting at nearly the same range is a
    # surface. One bearing hitting is a thing, and is handled below.
    run: list[tuple] = []
    for entry in scan + [(999.0, 0.0, "end")]:
        bearing, metres, hit = entry
        if hit == "wall" and metres <= NEAR_WALL_M and (
                not run or abs(metres - run[-1][1]) <= FLAT_TOLERANCE_M):
            run.append(entry)
            continue
        if len(run) >= 3:
            middle = run[len(run) // 2]
            spread = run[-1][0] - run[0][0]
            found.append(make(
                "wall", f"There's a wall {_at(middle[0], middle[1])}",
                middle[0], middle[1], min(0.9, 0.5 + spread / 120.0)))
        run = [entry] if hit == "wall" and metres <= NEAR_WALL_M else []

    # -- obstacles -----------------------------------------------------------
    # One or two bearings much CLOSER than their neighbours: a chair leg, a
    # box, a person's foot. The opposite test to the opening above.
    for index, (bearing, metres, hit) in enumerate(scan):
        if hit != "wall" or metres > OBSTACLE_M:
            continue
        before = scan[index - 1][1]
        after = scan[(index + 1) % len(scan)][1]
        if min(before, after) - metres >= OBSTACLE_STEP_M:
            found.append(make(
                "obstacle", f"Something's {_at(bearing, metres)}",
                bearing, metres, 0.6))

    # -- a hole in a closed boundary: MAYBE glass, maybe nothing -------------
    # A mirror cannot be detected by LiDAR -- the beam reflects away and the
    # surface reads as open or as no return at all. So this reports the
    # SYMPTOM and says so out loud: a bearing that runs to the edge of the map
    # while both its neighbours hit a wall close by. A doorway does that too,
    # which is why the confidence is low and the sentence is hedged.
    for index, (bearing, metres, hit) in enumerate(scan):
        if hit not in ("edge", "open") or metres < RAY_MAX_M - 0.01:
            continue
        before_m, before_hit = scan[index - 1][1], scan[index - 1][2]
        after_m, after_hit = (scan[(index + 1) % len(scan)][1],
                              scan[(index + 1) % len(scan)][2])
        if (before_hit == "wall" and after_hit == "wall"
                and max(before_m, after_m) <= 2.0):
            found.append(make(
                "reflective_suspect",
                f"There's a gap {_side(bearing)} that the laser goes straight "
                f"through — it might be glass or a mirror",
                bearing, min(before_m, after_m), 0.35))

    found.sort(key=lambda f: -f.confidence)
    return found


# --- telemetry --------------------------------------------------------------
class Telemetry:
    """Everything the run produced, in the three shapes a consumer wants.

    THREE FILES, ONE TRUTH. A dashboard that polls wants the latest state in
    one small file; a dashboard that streams wants an append-only feed it can
    tail from wherever it got to; a person debugging tomorrow wants the whole
    session in one place. Deriving any of the three from either of the others
    later means writing a parser, so all three are written now.

    The live file is written atomically -- tmp then rename -- because a
    dashboard polling at 1 Hz WILL eventually read a file mid-write, and half
    a JSON document is a crash in the app rather than a stale number.
    """

    def __init__(self, session_id: str, requested_minutes: float) -> None:
        self.session_id = session_id
        self.requested_minutes = requested_minutes
        self.started = time.time()
        self.state = "starting"
        self.features: list[Feature] = []
        self.events: list[dict] = []
        self.pose: dict | None = None
        self.map: dict = {}
        self.stopped_because = ""
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        self.session_file = (TELEMETRY_DIR /
                             f"session-{time.strftime('%Y%m%d-%H%M%S')}.json")
        # The dashboard mirror. Optional on purpose: no dashboard running must
        # never be a reason a run does not happen.
        try:
            import dashboard_log
            self._dash = dashboard_log.Turn(model="exploration_mode")
            self._dash.session_init([{"name": "friday-robot"}])
        except Exception as exc:                # noqa: BLE001
            log.debug("no dashboard to talk to: %s", exc)
            self._dash = None

    # -- recording -------------------------------------------------------
    def note(self, level: str, text: str) -> None:
        self.events.append({"ts": time.time(), "level": level, "text": text})
        log.info("%s", text)

    def add(self, feature: Feature) -> None:
        self.features.append(feature)
        # `event`, not `kind`: a Feature already HAS a kind ("wall"), and
        # spreading it over the row type made every feature row claim to be a
        # row of type "wall". A reader could not then tell a feature from a
        # speech line without guessing, which is the one thing a stream format
        # must never ask of a reader.
        self._append_feed({"event": "feature", **asdict(feature)})
        if self._dash is not None:
            try:
                self._dash.tool_result("exploration_mode.feature",
                                       asdict(feature))
            except Exception:                   # noqa: BLE001
                pass

    def said(self, sentence: str) -> None:
        self.note("speech", sentence)
        self._append_feed({"event": "speech", "ts": time.time(),
                           "text": sentence})
        if self._dash is not None:
            try:
                self._dash.text(sentence)
            except Exception:                   # noqa: BLE001
                pass

    # -- the payload -----------------------------------------------------
    def payload(self) -> dict:
        """The structured export. This is the dashboard's contract."""
        now = time.time()
        elapsed = now - self.started
        total = self.requested_minutes * 60.0
        counts: dict[str, int] = {}
        for feature in self.features:
            counts[feature.kind] = counts.get(feature.kind, 0) + 1
        return {
            "schema": SCHEMA_VERSION,
            "session_id": self.session_id,
            "state": self.state,
            "requested_minutes": round(self.requested_minutes, 2),
            "started_at": self.started,
            "updated_at": now,
            "elapsed_s": round(elapsed, 1),
            "remaining_s": round(max(0.0, total - elapsed), 1),
            "progress": round(min(1.0, elapsed / total), 3) if total else 0.0,
            "stopped_because": self.stopped_because,
            "pose": self.pose,
            "map": self.map,
            "counts": counts,
            "feature_count": len(self.features),
            "features": [asdict(f) for f in self.features],
            "events": self.events[-200:],
        }

    # -- writing ---------------------------------------------------------
    def flush(self) -> None:
        self._write_atomic(LIVE_FILE, self.payload())

    def finish(self, state: str, why: str = "") -> None:
        self.state = state
        self.stopped_because = why
        payload = self.payload()
        self._write_atomic(LIVE_FILE, payload)
        self._write_atomic(self.session_file, payload)
        self._append_feed({"event": "session_end", "ts": time.time(),
                           "state": state, "why": why})
        if self._dash is not None:
            try:
                self._dash.tool_result("exploration_mode.summary", payload)
                self._dash.done(f"Exploration {state}. "
                                f"{len(self.features)} features logged.")
                import dashboard_log
                dashboard_log.flush(timeout=3.0)
            except Exception:                   # noqa: BLE001
                pass

    @staticmethod
    def _write_atomic(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)
        except OSError as exc:
            log.debug("could not write %s: %s", path.name, exc)

    def _append_feed(self, row: dict) -> None:
        try:
            with FEED_FILE.open("a") as out:
                out.write(json.dumps(row) + "\n")
        except OSError as exc:
            log.debug("could not append to the feed: %s", exc)


# --- the run ----------------------------------------------------------------
class ExplorationMode:
    """Start the existing autonomous run, then watch it and talk about it.

    NOTHING HERE DRIVES THE ROBOT. start_exploration() and stop_exploration()
    are the existing MCP tools and the run behind them is untouched -- this
    class never calls move, turn or any planner. If it crashed halfway the run
    would carry on exactly as it does today, which is the test of whether a
    layer is really a layer.
    """

    def __init__(self, minutes: float, standalone: bool = False,
                 speak: bool = True, attach: bool = False) -> None:
        self.minutes = max(MIN_MINUTES, min(minutes, MAX_MINUTES))
        self.standalone = standalone
        # ATTACH TO A RUN ALREADY GOING, rather than starting one.
        # Starting it from here cost 30 SECONDS before the wheels moved --
        # 10.4 s of python start-up and imports, 19.2 s to open a fresh MCP
        # session -- while the voice loop already HAS a session open and can
        # call start_exploration in under a second. So the spine starts the
        # run and this process catches up and narrates it. The clock is still
        # enforced here, because rl/modes/explore.sh is hardcoded to 1200 s.
        self.attach = attach
        self.speak = speak
        self.telemetry = Telemetry(uuid.uuid4().hex[:8], self.minutes)
        self._seen: set[tuple] = set()
        self._last_spoke = 0.0
        self._runlog = RunLog()

    # -- talking ---------------------------------------------------------
    async def _say(self, session, sentence: str) -> None:
        """Out loud, through the MCP speak tool -- the one voice Friday has."""
        self.telemetry.said(sentence)
        if not self.speak:
            return
        try:
            await asyncio.wait_for(
                session.call_tool("speak", {"text": sentence}),
                timeout=config.STAGE_TOOL_TIMEOUT_S)
        except Exception as exc:                # noqa: BLE001
            # Never fatal. A run that stops because the speaker is busy is
            # worse than a run nobody hears.
            log.debug("could not speak: %s", exc)

    async def _call(self, session, tool: str, args: dict | None = None):
        """One MCP call, returning its text, never raising."""
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool, args or {}),
                timeout=config.STAGE_TOOL_TIMEOUT_S)
            return " ".join(getattr(part, "text", "") or ""
                            for part in (result.content or []))
        except Exception as exc:                # noqa: BLE001
            log.debug("%s failed: %s", tool, exc)
            return ""

    # -- one look around -------------------------------------------------
    async def _poll(self, session) -> list[Feature]:
        """Read the world once and return what is NEW in it."""
        pose = read_pose()
        grid = read_grid()
        source = "map"

        # THE RUN'S OWN LOG FIRST, whenever there is a run. It is the only
        # thing that is live: the map and the pose file are both written at
        # the END of a run, so during one they describe the previous run.
        if not self.standalone:
            cell = grid[2] if grid else config.SLAM_CELL_M
            origin = grid[1] if grid else (0.0, 0.0)
            ticks = self._runlog.read()
            if ticks:
                latest = ticks[-1]
                self.telemetry.pose = {
                    "x_m": round(latest["x"], 3), "y_m": round(latest["y"], 3),
                    "theta_deg": round(math.degrees(latest["theta"]), 1),
                    "grid_col": int((latest["x"] - origin[0]) / cell),
                    "grid_row": int((latest["y"] - origin[1]) / cell),
                    "cell_m": cell, "source": "run_log dead reckoning",
                    "tick": latest["t"]}
                self.telemetry.map.setdefault("cell_m", cell)
                return self._new_only(
                    features_from_ticks(ticks, cell, origin))

        if self.standalone or pose is None or grid is None:
            # No run to watch, or no map yet: ask the sensor directly. That
            # only works when nothing holds the lease.
            features = await self._poll_lidar(session)
            if features or grid is None or pose is None:
                return features
            # THE FALLBACK HAS TO ACTUALLY FALL BACK. This used to log
            # "reading the map instead" and then return nothing at all, which
            # is a worse failure than the one it was reporting: a run that
            # says it is narrating and never says another word. A lease is a
            # reason to read the map, so read it.

        logodds, origin, cell, scans = grid
        self.telemetry.pose = {"x_m": round(pose["x"], 3),
                               "y_m": round(pose["y"], 3),
                               "theta_deg": round(math.degrees(pose["theta"]), 1),
                               "grid_col": int((pose["x"] - origin[0]) / cell),
                               "grid_row": int((pose["y"] - origin[1]) / cell),
                               "cell_m": cell}
        import numpy as np
        self.telemetry.map = {
            "cell_m": cell, "origin_m": [float(origin[0]), float(origin[1])],
            "size_cells": list(logodds.shape), "scans": scans,
            "occupied_cells": int((logodds > OCCUPIED_LOGODDS).sum()),
            "free_cells": int((logodds < 0).sum()),
            "floor_m2": round(float((logodds < 0).sum()) * cell * cell, 1)}

        scan = virtual_scan(logodds, origin, cell, pose)
        return self._new_only(detect_features(scan, pose, cell, origin, source))

    async def _poll_lidar(self, session) -> list[Feature]:
        """Standalone: one real sweep, same rules applied to it."""
        raw = await self._call(session, "lidar_sweep", {"full": True})
        if not raw or not raw.lstrip().startswith("{"):
            if "HardwareLeased" in raw:
                self.telemetry.note("info", "sensors are leased by the run — "
                                            "reading the map instead")
            return []
        try:
            data = json.loads(raw)
        except ValueError:
            return []
        sectors = data.get("sectors_mm") or {}
        if not sectors:
            return []
        # lidar_sweep reports absolute bearings; forward is LIDAR_FORWARD_DEG.
        # Converted here rather than anywhere else, because every consumer of
        # these features thinks in "ahead / left / right".
        forward = float(getattr(config, "LIDAR_FORWARD_DEG", 0.0) or 0.0)
        scan = []
        for key, mm in sectors.items():
            try:
                absolute, metres = float(key), float(mm) / 1000.0
            except (TypeError, ValueError):
                continue
            relative = (absolute - forward + 180.0) % 360.0 - 180.0
            scan.append((relative, metres, "wall" if metres < RAY_MAX_M
                         else "open"))
        scan.sort(key=lambda row: row[0])
        pose = read_pose() or {"x": 0.0, "y": 0.0, "theta": 0.0}
        grid = read_grid()
        cell = grid[2] if grid else config.SLAM_CELL_M
        origin = grid[1] if grid else (0.0, 0.0)
        return self._new_only(detect_features(scan, pose, cell, origin,
                                              source="lidar"))

    def _new_only(self, features: list[Feature]) -> list[Feature]:
        """Drop anything already reported. A wall does not become news again
        because three seconds passed."""
        fresh = []
        for feature in features:
            key = feature.key()
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(feature)
        return fresh

    # -- objects, from the camera ----------------------------------------
    async def _look(self, session) -> None:
        """What the camera sees. speak=False -- this decides when to talk."""
        seen = await self._call(session, "say_what_you_see",
                                {"camera": "up", "speak": False})
        seen = seen.strip()
        if not seen or seen.startswith("FAILED"):
            return
        key = ("object", round(time.time() / 60), 0)
        if key in self._seen:
            return
        self._seen.add(key)
        pose = read_pose() or {"x": 0.0, "y": 0.0, "theta": 0.0}
        grid = read_grid()
        cell = grid[2] if grid else config.SLAM_CELL_M
        origin = grid[1] if grid else (0.0, 0.0)
        feature = Feature(
            id=uuid.uuid4().hex[:12], kind="objects", label=seen[:300],
            ts=time.time(), bearing_deg=0.0, distance_mm=0,
            local_x_m=0.0, local_y_m=0.0,
            global_x_m=round(pose["x"], 3), global_y_m=round(pose["y"], 3),
            grid_col=int((pose["x"] - origin[0]) / cell),
            grid_row=int((pose["y"] - origin[1]) / cell),
            cell_m=cell, confidence=0.5, source="camera")
        self.telemetry.add(feature)
        await self._say(session, seen[:300])
        feature.spoken = True

    # -- the loop --------------------------------------------------------
    async def run(self) -> dict:
        import brain

        async with brain.mcp_session() as session:
            started_run = False
            if self.attach:
                # The run is already going. Tail its log from the end so the
                # first seconds of it are not replayed as news.
                started_run = True
                for _ in range(20):
                    if self._runlog.open():
                        self.telemetry.note(
                            "info", f"attached to {self._runlog.path.name}")
                        break
                    await asyncio.sleep(0.5)
            elif not self.standalone:
                reply = await self._call(session, "start_exploration",
                                         {"minutes": self.minutes})
                started_run = '"started": true' in reply.lower()
                if not started_run:
                    self.telemetry.note("warn", f"the run did not start: "
                                                f"{reply[:200]}")
                    # KEEP GOING ANYWAY. Narrating a robot that is standing
                    # still is a smaller failure than refusing to narrate.
                else:
                    self.telemetry.note("info",
                                        f"run started for {self.minutes:g} min")
                    # Give the run a moment to create its log, then tail it
                    # from the end so old runs are not replayed as news.
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        if self._runlog.open():
                            self.telemetry.note(
                                "info", f"tailing {self._runlog.path.name}")
                            break

            self.telemetry.state = "running"
            await self._say(session,
                            f"Exploring for {self.minutes:g} minutes.")

            deadline = time.time() + self.minutes * 60.0
            why = "finished the time asked for"
            last_look = 0.0

            # A SIGNAL MUST END THE DRIVE, NOT JUST THIS PROCESS. Without
            # this, killing the narrator leaves the run driving for its full
            # hardcoded 1200 s with nothing watching -- which is exactly what
            # happened the first time this was tested.
            stopping = asyncio.Event()
            loop = asyncio.get_running_loop()
            for signame in ("SIGINT", "SIGTERM"):
                try:
                    loop.add_signal_handler(
                        getattr(signal, signame), stopping.set)
                except (NotImplementedError, AttributeError, ValueError):
                    pass

            try:
                while time.time() < deadline and not stopping.is_set():
                    features = await self._poll(session)
                    spoken = 0
                    for feature in features:
                        self.telemetry.add(feature)
                        if (spoken < MAX_SPOKEN_PER_POLL and self.speak
                                and time.time() - self._last_spoke
                                >= SPEAK_MIN_GAP_S):
                            await self._say(session, feature.label)
                            feature.spoken = True
                            self._last_spoke = time.time()
                            spoken += 1

                    # The camera is slower and noisier than the map, so it is
                    # asked less often.
                    if time.time() - last_look > 30.0:
                        last_look = time.time()
                        await self._look(session)

                    self.telemetry.flush()

                    if started_run:
                        status = await self._call(session,
                                                  "exploration_status")
                        if '"exploring": false' in status.lower().replace(
                                " ", "").replace('"exploring":false',
                                                 '"exploring": false'):
                            why = "the run ended on its own"
                            break
                    try:
                        await asyncio.wait_for(stopping.wait(), timeout=POLL_S)
                    except asyncio.TimeoutError:
                        pass
                if stopping.is_set():
                    why = "asked to stop"
            except asyncio.CancelledError:
                why = "cancelled"
                raise
            except KeyboardInterrupt:
                why = "stopped by hand"
            finally:
                if started_run:
                    await self._call(session, "stop_exploration")
                self.telemetry.finish(
                    "finished" if why.startswith("finished") else "stopped",
                    why)

            summary = self._summary()
            await self._say(session, summary)   # _say already records it
            return self.telemetry.payload()

    def _summary(self) -> str:
        counts: dict[str, int] = {}
        for feature in self.telemetry.features:
            counts[feature.kind] = counts.get(feature.kind, 0) + 1
        if not counts:
            return "I'm done exploring. I didn't find anything new."
        said = {"wall": ("wall", "walls"),
                "corridor": ("corridor", "corridors"),
                "turn": ("turn", "turns"),
                "opening": ("opening", "openings"),
                "obstacle": ("obstacle", "obstacles"),
                "dead_end": ("dead end", "dead ends"),
                "objects": ("thing I looked at", "things I looked at"),
                "reflective_suspect": ("gap that might be glass",
                                       "gaps that might be glass")}
        bits = [f"{n} {said.get(k, (k, k))[0 if n == 1 else 1]}"
                for k, n in sorted(counts.items(), key=lambda kv: -kv[1])[:4]]
        return "I'm done exploring. I found " + ", ".join(bits) + "."


# --- entry points -----------------------------------------------------------
async def explore_for(minutes: float, standalone: bool = False,
                      speak: bool = True, attach: bool = False) -> dict:
    """Run one timed, narrated exploration. The one function to call."""
    return await ExplorationMode(minutes, standalone, speak, attach).run()


async def handle_command(text: str, standalone: bool = False) -> dict | None:
    """Run this if it is an explore command; return None if it is not.

    This is the wiring point. Anything that has a transcript -- main.py,
    offline.py, a dashboard button, a test -- can hand it here without
    knowing anything about durations, maps or telemetry.
    """
    minutes = parse_request(text)
    if minutes is None:
        return None
    return await explore_for(minutes, standalone=standalone)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=float, help="how long to explore")
    ap.add_argument("--say", metavar="TEXT",
                    help='a spoken command, e.g. "explore for 5 minutes"')
    ap.add_argument("--standalone", action="store_true",
                    help="do not start a run; read the LiDAR directly")
    ap.add_argument("--attach", action="store_true",
                    help="a run is ALREADY going — narrate it rather than "
                         "starting another. This is how main.py uses it, so "
                         "the wheels move in under a second instead of 30")
    ap.add_argument("--quiet", action="store_true", help="log, do not speak")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report, touch no hardware")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")

    minutes = args.minutes
    if args.say:
        minutes = parse_request(args.say)
        if minutes is None:
            print(f"{args.say!r} is not an explore command.")
            return 1
    if minutes is None:
        ap.error("give --minutes or --say")

    if args.dry_run:
        print(f"would explore for {minutes:g} minutes "
              f"({'standalone' if args.standalone else 'watching the run'})")
        pose, grid = read_pose(), read_grid()
        print(f"pose: {pose}")
        if grid is not None:
            logodds, origin, cell, scans = grid
            print(f"map:  {logodds.shape} cells @ {cell} m, origin {origin}, "
                  f"{scans} scans")
            if pose:
                scan = virtual_scan(logodds, origin, cell, pose)
                found = detect_features(scan, pose, cell, origin)
                print(f"\n{len(found)} feature(s) from the map right now:")
                for f in found:
                    print(f"  {f.confidence:.2f}  {f.kind:20} "
                          f"({f.global_x_m:6.2f},{f.global_y_m:6.2f}) "
                          f"cell({f.grid_col},{f.grid_row})  {f.label}")
        return 0

    payload = asyncio.run(explore_for(minutes, args.standalone,
                                      speak=not args.quiet,
                                      attach=args.attach))
    print(f"\n{payload['feature_count']} features -> {LIVE_FILE}")
    print(json.dumps(payload["counts"], indent=2))
    return 0


# --- the run's own live telemetry -------------------------------------------
# WHY THIS EXISTS, AFTER THE MAP READER ABOVE. state/rl_map.npz and
# rl_map.pose.json are written when a run FINISHES, not while it goes -- both
# were 23 hours stale during a live run, measured 2026-09-06. So a narrator
# reading the map during a run describes where Friday was yesterday, and after
# the first poll dedups itself into total silence. That is exactly what the
# first version did: 97 seconds of driving, zero features, "I didn't find
# anything new."
#
# What IS live is the run's own log, appended at ~10 Hz:
#
#   00:14:42 INFO pi_deploy: t=    5 | lidar 0.22 | tof 1.15 | vis 0.00 |
#            pol(+0.50,-0.58) -> out(+0.00,-0.58) | governor     | 576.2 ms
#
# Reading a file the run already writes costs the run nothing and changes
# nothing about it. Tailing it is the only way to narrate a run live without
# reaching into the run.
_TICK_RE = re.compile(
    r"(?P<clock>\d\d:\d\d:\d\d).*?pi_deploy:\s*t=\s*(?P<t>\d+)\s*\|"
    r"\s*lidar\s+(?P<lidar>[-\d.]+)\s*\|"
    r"\s*tof\s+(?P<tof>[-\d.]+)\s*\|"
    r"\s*vis\s+(?P<vis>[-\d.]+)\s*\|"
    r"\s*pol\((?P<pv>[-+\d.]+),(?P<pw>[-+\d.]+)\)\s*->\s*"
    r"out\((?P<ov>[-+\d.]+),(?P<ow>[-+\d.]+)\)"
    r"(?:\s*\|\s*(?P<state>[A-Za-z_]+))?")

TURN_W = 0.45                # rad/s sustained before it counts as a turn
TURN_TICKS = 4               # ...for this many ticks (~0.4 s at 10 Hz)
CLEAR_RUN_M = 2.0            # lidar this open ahead is "a clear run"
TICK_HZ = 10.0


def parse_tick(line: str) -> dict | None:
    """One line of the run's log, or None if it is not a tick."""
    match = _TICK_RE.search(line)
    if not match:
        return None
    g = match.groupdict()
    try:
        return {"clock": g["clock"], "t": int(g["t"]),
                "lidar_m": float(g["lidar"]), "tof_m": float(g["tof"]),
                "vis": float(g["vis"]),
                "v": float(g["ov"]), "w": float(g["ow"]),
                "wanted_v": float(g["pv"]), "wanted_w": float(g["pw"]),
                "state": (g["state"] or "").strip()}
    except (TypeError, ValueError):
        return None


class RunLog:
    """Tail the log of the run that is happening now.

    Dead-reckons a pose from the commanded velocities as it goes, because the
    log has no pose in it and a feature without a position is not telemetry.
    That estimate DRIFTS -- it is wheel commands, not odometry, with no
    encoders behind it -- so anything derived from it is marked
    source="run_log" and carries a lower confidence than the same feature read
    off the map. Saying where something is approximately is useful; pretending
    it is surveyed is not.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self._handle = None
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.ticks = 0

    def _newest(self) -> Path | None:
        try:
            status = json.loads(
                (config.ROOT / "state" / "explore_mode.json").read_text())
            candidate = Path(status.get("log", ""))
            if candidate.exists():
                return candidate
        except (OSError, ValueError):
            pass
        logs = sorted((config.ROOT / "state" / "runs").glob("run-*.log"))
        return logs[-1] if logs else None

    def open(self, from_start: bool = False) -> bool:
        path = self._newest()
        if path is None:
            return False
        if self._handle is not None and self.path == path:
            return True
        self.close()
        try:
            self._handle = path.open("r", errors="replace")
        except OSError:
            return False
        self.path = path
        if not from_start:
            self._handle.seek(0, 2)         # only what happens from now on
        return True

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
        self._handle = None

    def read(self) -> list[dict]:
        """Every tick appended since the last call."""
        if self._handle is None and not self.open():
            return []
        out = []
        for line in self._handle.readlines():
            tick = parse_tick(line)
            if tick is None:
                continue
            # Dead reckoning, one tick of it.
            dt = 1.0 / TICK_HZ
            self.theta += tick["w"] * dt
            self.x += tick["v"] * math.cos(self.theta) * dt
            self.y += tick["v"] * math.sin(self.theta) * dt
            tick["x"] = self.x
            tick["y"] = self.y
            tick["theta"] = self.theta
            self.ticks += 1
            out.append(tick)
        return out


def features_from_ticks(ticks: list[dict], cell: float,
                        origin: tuple) -> list[Feature]:
    """What the run's own telemetry says Friday just met.

    Deliberately conservative: a robot narrating every tick is a robot nobody
    can listen to, so a feature needs a RUN of ticks agreeing, not one.
    """
    found: list[Feature] = []
    if not ticks:
        return found

    def make(kind: str, label: str, tick: dict, bearing: float,
             metres: float, confidence: float) -> Feature:
        angle = tick["theta"] + math.radians(bearing)
        gx = tick["x"] + metres * math.cos(angle)
        gy = tick["y"] + metres * math.sin(angle)
        return Feature(
            id=uuid.uuid4().hex[:12], kind=kind, label=label, ts=time.time(),
            bearing_deg=round(bearing, 1), distance_mm=int(metres * 1000),
            local_x_m=round(metres * math.cos(math.radians(bearing)), 3),
            local_y_m=round(metres * math.sin(math.radians(bearing)), 3),
            global_x_m=round(gx, 3), global_y_m=round(gy, 3),
            grid_col=int((gx - origin[0]) / cell),
            grid_row=int((gy - origin[1]) / cell),
            cell_m=cell, confidence=round(confidence, 2), source="run_log")

    # -- something stopped her -------------------------------------------
    stops = [t for t in ticks if t["state"] == "REFLEX_STOP"]
    if len(stops) >= 3:
        nearest = min(stops, key=lambda t: min(t["lidar_m"], t["tof_m"]))
        metres = min(nearest["lidar_m"], nearest["tof_m"])
        which = "tof" if nearest["tof_m"] <= nearest["lidar_m"] else "lidar"
        found.append(make(
            "obstacle",
            f"Something's blocking me, {_at(0.0, max(metres, 0.05))}",
            nearest, 0.0, max(metres, 0.05),
            0.8 if which == "tof" else 0.7))

    # -- a turn, held long enough to be one -------------------------------
    run_len, run_sign, best = 0, 0, None
    for tick in ticks:
        sign = 1 if tick["w"] > TURN_W else (-1 if tick["w"] < -TURN_W else 0)
        if sign != 0 and sign == run_sign:
            run_len += 1
        else:
            run_sign, run_len = sign, 1 if sign else 0
        if run_len >= TURN_TICKS and (best is None or run_len > best[0]):
            best = (run_len, sign, tick)
    if best is not None:
        _, sign, tick = best
        way = "left" if sign > 0 else "right"
        found.append(make("turn", f"Turning {way}.", tick,
                          90.0 * sign, max(tick["lidar_m"], 0.1), 0.75))

    # -- a clear run ahead ------------------------------------------------
    open_ticks = [t for t in ticks
                  if t["lidar_m"] >= CLEAR_RUN_M and t["v"] > 0.05]
    if len(open_ticks) >= 5:
        furthest = max(open_ticks, key=lambda t: t["lidar_m"])
        found.append(make(
            "corridor",
            f"Clear run ahead, about {furthest['lidar_m']:.1f} metres of it",
            furthest, 0.0, furthest["lidar_m"], 0.65))

    # -- states worth saying ----------------------------------------------
    spoken_states = {
        "rear_blocked": "There's something behind me too.",
        "recovery": "I got stuck, backing out of it.",
        "tof_slow": "Slowing down, something's close.",
        "reverse_lifted": "I'm backing up.",
    }
    for state, sentence in spoken_states.items():
        hits = [t for t in ticks if t["state"] == state]
        if len(hits) >= 2:
            tick = hits[-1]
            found.append(make(state, sentence, tick, 0.0,
                              max(tick["lidar_m"], 0.1), 0.6))

    found.sort(key=lambda f: -f.confidence)
    return found


if __name__ == "__main__":
    raise SystemExit(main())
