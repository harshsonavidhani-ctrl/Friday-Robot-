#!/usr/bin/env python3
"""llm_cognitive_exploration.py — Claude Sonnet 5 as the Tier-2 cognitive planner.

    export ANTHROPIC_API_KEY=sk-ant-...
    ./venv/bin/python llm_cognitive_exploration.py              # 5 s cadence
    ./venv/bin/python llm_cognitive_exploration.py --on-stuck   # only when stuck
    ./venv/bin/python llm_cognitive_exploration.py --once --dry-run   # no API call

WHAT THIS IS. A STANDALONE bridge. It imports nothing from the LeRobot stack,
the exploration code, the SLAM map or the local controllers, and it writes to
none of them: every sensor read in here is a MOCK with a single, obvious seam
(see "THE FIVE SEAMS" below) for wiring to the real robot later. Run it and it
prints one JSON payload per tick on stdout; whatever consumes that payload is
the only place the two systems touch.

TIER 2, NOT TIER 1 — READ THIS BEFORE WIRING IT TO MOTORS.
    Tier 1 (local, on-robot, milliseconds) owns collision avoidance, cliff
    detection, the emergency stop and every other reflex. It must be able to
    ignore or override anything that arrives from here.
    Tier 2 (this file, seconds, over the network) is ADVISORY: it answers
    "where should I go next and why", not "is it safe to move right now".
A planner that reaches the robot over the internet, at a 5 s cadence, behind a
rate limit, is structurally incapable of being a safety layer. Treat the
`target_heading_offset` below as a suggestion the local controller is free to
refuse.

THE FIVE SEAMS (replace these, change nothing else):
    capture_front_vision()  -> two base64 JPEG frames
    read_lidar_metrics()    -> 8 sector distances + odometry
    get_robot_state()       -> the local controller's state machine
    read_battery()          -> battery percentage
    emit_payload()          -> hand the decision to the local controller

MODEL. `claude-sonnet-5` — the exact API id. There is no "anthropic/" prefix
and no date suffix; those are rejected. On Sonnet 5 thinking is configured as
`{"type": "adaptive"}` and `budget_tokens` is REMOVED (sending it is a 400).
Depth is controlled by `output_config.effort` instead, set to "low" here
because a navigation decision every five seconds is a latency budget, not a
reasoning marathon.

COST. Sonnet 5 is $2/MTok in, $10/MTok out (2026-06 rates, update PRICING
below). Two 640x480 JPEGs are roughly 1.5k tokens EACH, so vision dominates
the bill: at a 5 s cadence this is on the order of a few dollars an hour.
`--on-stuck` exists for that reason, and DAILY_USD_CAP stops the loop dead
rather than surprising anyone with a bill.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("friday.cognitive")

# --- configuration -----------------------------------------------------------

MODEL = "claude-sonnet-5"
API_TIMEOUT_S = 20.0           # a planner that answers late is worse than none
TICK_SECONDS = 5.0
MAX_TOKENS = 4096              # the schema is tiny; adaptive thinking needs room
EFFORT = "low"                 # "low" | "medium" | "high" | "xhigh" | "max"

# Battery. Exploration is the most expensive thing this robot does, and a flat
# battery mid-room is a rescue, not a fault. These are TIER-2 advisory numbers:
# the local controller should have its own, lower, hard cutoff.
BATTERY_RETURN_PCT = 25.0      # stop exploring, head home
BATTERY_CRITICAL_PCT = 12.0    # stop moving; do not even spend an API call
BATTERY_LOW_TICK_SECONDS = 15.0  # think less often when power is short

# Cost guard. See the module docstring: vision dominates.
PRICING_USD_PER_MTOK = {"input": 2.00, "output": 10.00}
DAILY_USD_CAP = 2.00

MEMORY_DECISIONS = 5           # "last 5 decisions", as specified
MEMORY_LANDMARKS = 12

# Runtime state published by the exploration run -- READ ONLY, and output
# rather than source, which is what keeps this file zero-touch.
HERE = Path(__file__).resolve().parent
AUTONOMY_STATE = HERE / "state" / "rl_autonomy.json"
POSE_STATE = HERE / "state" / "rl_map.pose.json"
DECISION_LOG = HERE / "logs" / "cognitive_decisions.jsonl"
STALE_AFTER_S = 5.0    # the run republishes about twice a second

# VISION COMES FROM THE DASHBOARD, NEVER FROM /dev/video*. The cameras are
# single-reader on this robot and an exploration run takes them -- opening the
# device here would fight whatever is driving. The dashboard already arbitrates
# who holds them, so one frame is pulled from its MJPEG stream and the
# connection is closed again. Measured 2026-09-12: 4.9 s for the first frame
# (camera spin-up), then 0.1-0.2 s while it stays warm.
DASHBOARD_URL = "http://127.0.0.1:8080"
CAMERA_TIMEOUT_S = 9.0
# Downscaled before sending: a 640x480 frame is ~1.5k tokens and vision
# dominates the bill. 512 px wide keeps the semantics (door, stairs, clutter)
# that this planner actually uses.
VISION_MAX_WIDTH = 512
VISION_JPEG_QUALITY = 70
# THE RUN PUBLISHES NO LIVE POSE. rl_map.pose.json is written when the map is
# saved, so after a run that aborts in teardown it can be days old -- measured
# at 3.2 days on 2026-09-12. Old coordinates presented as current are worse
# than none: they silently freeze the displacement check in SpatialMemory.
POSE_STALE_AFTER_S = 60.0

# The strict output contract. additionalProperties=False plus required on every
# field means the response either matches this exactly or the API errors --
# there is no half-parsed payload for the controller to misread.
ACTION_MODES = [
    "FRONTIER_CONTINUE",          # keep pushing into open space
    "ESCAPE_DEADLOCK",            # wedged or looping; break out
    "180_TURN_BACKTRACK",         # dead end; go back the way we came
    "AVOID_SEMANTIC_OBSTACLE",    # something the LiDAR cannot see (stairs, glass)
    "RETURN_TO_BASE_LOW_BATTERY",  # power, not geometry, is now the constraint
    "HOLD_POSITION",              # ambiguous scene; let Tier 1 settle
]

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_understanding": {
            "type": "string",
            "description": "One or two sentences fusing what the cameras show "
                           "with what the LiDAR measures.",
        },
        "deadlock_analysis": {
            "type": "boolean",
            "description": "True if the robot is looping, wedged, or "
                           "re-exploring ground it has already covered.",
        },
        "action_mode": {"type": "string", "enum": ACTION_MODES},
        "target_heading_offset": {
            "type": "number",
            "description": "Degrees to turn, relative to current heading. "
                           "Negative is left, positive is right, -180 to 180.",
        },
    },
    "required": ["scene_understanding", "deadlock_analysis",
                 "action_mode", "target_heading_offset"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the cognitive planner for a small differential-drive indoor robot that \
is exploring an unknown building. You are its deliberate, slow-thinking layer: \
you run every few seconds, and a local reflex controller runs every few \
milliseconds beneath you.

WHAT THE LOCAL CONTROLLER ALREADY HANDLES, so never spend a decision on it:
collision avoidance, cliff and stair-edge stops, the emergency stop, wheel slip, \
and any obstacle closer than its own safety margin. It can and will override \
you. Your heading offset is a suggestion.

WHAT ONLY YOU CAN DO — this is the entire reason you are in the loop:
1. SEMANTIC BARRIERS the LiDAR cannot measure. A 2D scan sees a doorway and an \
open lift shaft as equally passable, and reads a glass wall, a descending \
staircase, a dark rug and a drop-off as free space. The cameras are the only \
sensor that can tell those apart. When vision and LiDAR disagree, vision wins \
on semantics and LiDAR wins on distance.
2. FRONTIER CHOICE. Open space is not equally worth exploring. A corridor or an \
open doorway leads to new rooms; the middle of a room you have already crossed \
does not. Prefer frontiers that plausibly lead somewhere new.
3. DEADLOCK RECOGNITION. You are given the last decisions and the landmarks \
already seen. If the robot is oscillating between two headings, revisiting the \
same landmark, or has not moved while believing it is exploring, say so in \
deadlock_analysis and pick a decisive escape — a large turn, or a backtrack — \
rather than another small correction in the same direction.

HOW TO READ THE INPUT. Sector distances are metres from the robot's current \
heading: N straight ahead, E 90 degrees right, W 90 degrees left, S directly \
behind. YOU MAY BE GIVEN FEWER THAN EIGHT SECTORS -- this robot measures four, \
and the diagonals are absent rather than zero. Never assume an unlisted sector \
is clear. `run_reported_state` is the exploration run's own account of itself: \
which frontier it chose, what it abandoned, why it is recovering, and what its \
drivetrain can actually execute. Trust it over your own reading of the \
geometry, and use `what_the_drivetrain_can_execute` to keep your heading inside \
what the wheels can do. Odometry is metres and degrees in the map frame. \
Anything under about 0.35 m is effectively a wall.

BATTERY. When the battery is low the mission changes: stop seeking frontiers \
and choose RETURN_TO_BASE_LOW_BATTERY with a heading that moves back toward the \
origin of the odometry frame. Exploring until flat strands the robot.

HOW TO CHOOSE A HEADING. target_heading_offset is degrees relative to where the \
robot faces now: negative left, positive right, 0 straight on. Prefer the \
largest clear sector consistent with your reasoning. Do not propose a heading \
into a sector under 0.5 m. Small corrections (under 15 degrees) are for staying \
in a corridor; use 90 or 180 when the situation calls for a real change.

Be decisive and brief. scene_understanding is for a human reading a log later, \
so write what you actually saw and measured, not a restatement of these rules."""


# --- mocked sensor ingestion -------------------------------------------------

@dataclass
class VisionFrames:
    """Base64 JPEG frames from the two USB cameras."""

    up_b64: str | None = None      # forward camera, above the screen
    down_b64: str | None = None     # floor camera, above the bumper
    media_type: str = "image/jpeg"
    captured_at: float = field(default_factory=time.time)
    status: dict[str, str] = field(default_factory=dict)

    def frames(self) -> list[tuple[str, str]]:
        """(label, base64) for whichever cameras actually produced a frame."""
        out = []
        if self.up_b64:
            out.append(("FORWARD camera (above the screen)", self.up_b64))
        if self.down_b64:
            out.append(("FLOOR camera (above the bumper)", self.down_b64))
        return out


def _one_frame(which: str, timeout_s: float) -> tuple[bytes | None, str]:
    """One JPEG from the dashboard's MJPEG stream, then hang up.

    Reads until the first complete SOI..EOI and closes, so the camera is held
    for as little time as possible. Any failure is returned as a REASON, not
    raised: a camera that is busy driving the robot is the normal case, not an
    error, and the planner must still think without it.
    """
    import urllib.request

    started = time.monotonic()
    url = f"{DASHBOARD_URL}/cam/{which}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            buffer = b""
            while len(buffer) < 600_000 and time.monotonic() - started < timeout_s:
                chunk = response.read(8192)
                if not chunk:
                    break
                buffer += chunk
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end > 0:
                    return buffer[start:end + 2], f"ok ({time.monotonic() - started:.1f}s)"
            return None, "no complete frame before the timeout"
    except Exception as exc:                # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"


def _shrink(jpeg: bytes) -> bytes:
    """Downscale to VISION_MAX_WIDTH. Vision dominates the token bill."""
    try:
        import io
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg))
        if image.width <= VISION_MAX_WIDTH:
            return jpeg
        height = round(image.height * VISION_MAX_WIDTH / image.width)
        image = image.convert("RGB").resize((VISION_MAX_WIDTH, height),
                                            Image.LANCZOS)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=VISION_JPEG_QUALITY)
        return out.getvalue()
    except Exception as exc:                # noqa: BLE001
        log.debug("could not shrink the frame (%s) — sending it as it is", exc)
        return jpeg


def capture_front_vision(enabled: bool = True,
                         timeout_s: float = CAMERA_TIMEOUT_S) -> VisionFrames:
    """SEAM 1 (WIRED) -- one frame from each USB camera, base64 encoded.

    Through the dashboard, NEVER /dev/video*: see DASHBOARD_URL. A camera that
    is unavailable -- busy, unplugged, or missing its by-id symlink, which is
    how the floor camera fails on this robot -- simply contributes no frame and
    a reason. Zero frames is a supported outcome; the planner then reasons from
    the LiDAR sectors and the run's own state, and is told vision was missing.
    """
    frames = VisionFrames()
    if not enabled:
        frames.status = {"up": "disabled (--no-vision)",
                         "down": "disabled (--no-vision)"}
        return frames
    for which in ("up", "down"):
        jpeg, why = _one_frame(which, timeout_s)
        if jpeg:
            small = _shrink(jpeg)
            encoded = base64.standard_b64encode(small).decode("ascii")
            setattr(frames, f"{which}_b64", encoded)
            why = f"{why}, {len(jpeg) // 1024} kB -> {len(small) // 1024} kB"
        frames.status[which] = why
    frames.captured_at = time.time()
    return frames


@dataclass
class LidarMetrics:
    """One 8-sector range slice plus where the robot thinks it is."""

    sectors_m: dict[str, float]
    odom_x_m: float | None
    odom_y_m: float | None
    heading_deg: float | None
    linear_velocity_mps: float
    source: str = "mock"
    pose_age_s: float | None = None
    tof_nearest_m: float | None = None

    def tightest(self) -> tuple[str, float]:
        name = min(self.sectors_m, key=self.sectors_m.get)
        return name, self.sectors_m[name]

    def known_pose(self) -> bool:
        return self.odom_x_m is not None and self.heading_deg is not None

    def distance_from_origin_m(self) -> float | None:
        if not self.known_pose():
            return None
        return math.hypot(self.odom_x_m, self.odom_y_m)


def _read_json(path: Path) -> tuple[dict | None, float]:
    """Load a runtime JSON file and say how old it is. Never raises."""
    try:
        age = time.time() - path.stat().st_mtime
        return json.loads(path.read_text()), age
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("cannot read %s (%s)", path.name, type(exc).__name__)
        return None, float("inf")


def read_autonomy() -> tuple[dict | None, float]:
    """The exploration run's own published state, and its age in seconds.

    ZERO-TOUCH BY DESIGN. This is the runtime state file the run writes for the
    dashboard and the voice layer to read -- output, not source. Nothing here
    imports the exploration stack, opens a serial port, or touches the map.
    That matters for more than tidiness: the LiDAR allows exactly ONE reader
    and friday-mcp already holds it, so reading the device directly would break
    whatever run is in progress.
    """
    return _read_json(AUTONOMY_STATE)


def read_lidar_metrics(autonomy: dict | None = None,
                       age_s: float = float("inf")) -> LidarMetrics:
    """SEAM 2 (WIRED) -- sector ranges and odometry from the live run.

    The run publishes FOUR sectors, not eight: front, left, right, rear. The
    diagonals are not measured and are NOT invented here -- a fabricated NE
    reading is a wall the planner will happily drive into. Whatever is real
    goes in; the model is told which sectors it has.

    Falls back to the mock only when there is no state file at all, and says so
    in `source` either way.
    """
    if autonomy is None or age_s > STALE_AFTER_S:
        base = {"N": 2.4, "NE": 1.8, "E": 0.9, "SE": 1.2, "S": 3.1, "SW": 2.2,
                "W": 1.1, "NW": 1.6}
        jitter = {k: round(max(0.12, v + random.uniform(-0.25, 0.25)), 2)
                  for k, v in base.items()}
        return LidarMetrics(
            sectors_m=jitter, odom_x_m=round(random.uniform(-4.0, 4.0), 2),
            odom_y_m=round(random.uniform(-4.0, 4.0), 2),
            heading_deg=round(random.uniform(0.0, 359.9), 1),
            linear_velocity_mps=round(random.choice([0.0, 0.0, 0.12, 0.18]), 2),
            source=("mock (no run state)" if autonomy is None
                    else f"mock (run state {age_s:.0f}s stale)"))

    lidar = (autonomy.get("SENSOR_STATE") or {}).get("lidar") or {}
    sectors = {name: round(float(lidar[key]), 2)
               for name, key in (("N", "front"), ("E", "right"),
                                 ("W", "left"), ("S", "rear"))
               if isinstance(lidar.get(key), (int, float))}
    action = autonomy.get("FINAL_EXECUTED_ACTION") or {}
    pose, pose_age = _read_json(POSE_STATE)
    fresh = pose is not None and pose_age <= POSE_STALE_AFTER_S
    pose = pose or {}
    return LidarMetrics(
        sectors_m=sectors or {"N": 0.0},
        odom_x_m=round(float(pose.get("x", 0.0)), 2) if fresh else None,
        odom_y_m=round(float(pose.get("y", 0.0)), 2) if fresh else None,
        heading_deg=(round(math.degrees(float(pose.get("theta", 0.0))) % 360.0, 1)
                     if fresh else None),
        linear_velocity_mps=round(float(action.get("v", 0.0)), 3),
        source=f"live run tick {autonomy.get('TICK')} ({age_s:.1f}s old)",
        pose_age_s=round(pose_age, 1),
        tof_nearest_m=((autonomy.get("SENSOR_STATE") or {}).get("tof") or {}).get("nearest"),
    )


def get_robot_state(autonomy: dict | None = None,
                    age_s: float = float("inf")) -> str:
    """SEAM 3 (WIRED) -- the run's own verdict, not a guess from the geometry.

    Derived in priority order, because these overlap: a reflex stop while
    recovering is still a reflex stop. The recovery triggers are the valuable
    part -- `oscillation` and `no_progress` are the run telling you it is
    deadlocked, which is exactly what the planner is here to break.
    """
    if autonomy is None:
        return "IDLE"
    if age_s > STALE_AFTER_S:
        return "IDLE"                       # a finished run is not a stuck one
    recovery = (autonomy.get("RECOVERY_STATUS") or {}).get("active") or {}
    trigger = recovery.get("trigger")
    reason = autonomy.get("REASON_CODE") or ""
    velocity = float((autonomy.get("FINAL_EXECUTED_ACTION") or {}).get("v", 0.0))
    if trigger in ("no_progress", "oscillation", "retreading", "boxed_in"):
        return "STUCK_VELOCITY_0" if abs(velocity) < 0.01 else "RECOVERING"
    if reason == "REFLEX_STOP" or (autonomy.get("SAFETY_STATUS") or {}).get("overridden"):
        return "OBSTACLE_PROXIMITY"
    if abs(velocity) < 0.01:
        return "STUCK_VELOCITY_0"
    return "EXPLORING"


def read_battery(manual_pct: float | None = None) -> tuple[float | None, str]:
    """SEAM 4 (WIRED, AND THE ANSWER IS "UNKNOWN") -- (percent, how we know).

    THERE IS NO FUEL GAUGE ON THIS ROBOT. config.py says so in as many words:
    "No battery fuel gauge is wired on this robot. get_status() reports battery
    as unknown rather than inventing a percentage." So this returns None, and
    every battery rule downstream is skipped rather than run on a number nobody
    measured. Inventing one would be worse than having none: it would strand
    the robot confidently.

    Two real signals do exist:
      * --battery-pct, for a human reading the powerbank's own display;
      * the Pi's undervoltage/throttle flags, which are about supply health,
        not charge -- and given this Pi reset four times on 2026-09-12 with
        rsts 0x1000 (power), they are worth reporting.
    """
    if manual_pct is not None:
        return manual_pct, "--battery-pct (human-entered)"
    flags = ""
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3)
        raw = out.stdout.strip().split("=")[-1]
        bits = int(raw, 16) if raw.startswith("0x") else 0
        if bits & 0x1:
            flags = "UNDERVOLTAGE NOW"
        elif bits & 0x10000:
            flags = "undervoltage happened since boot"
        else:
            flags = f"supply ok ({raw})"
    except Exception as exc:                # noqa: BLE001
        flags = f"throttle flags unavailable ({type(exc).__name__})"
    return None, f"no fuel gauge wired; {flags}"


# --- semantic spatial memory -------------------------------------------------

class SpatialMemory:
    """Short-term memory: the last few decisions, and landmarks already seen.

    WHY THIS EXISTS. Each API call is stateless, so without memory the planner
    re-derives the same conclusion from the same-looking corridor forever: it
    turns right at the same junction, arrives at the same dead end, turns right
    again. Handing it its own recent decisions is what makes "I have been here
    before" possible at all, and it is cheap -- a few hundred tokens.

    Deliberately SHORT-term. This is not a map and must not pretend to be one;
    the SLAM stack owns the map. Five decisions is enough to see an oscillation
    and little enough that stale beliefs age out on their own.
    """

    def __init__(self) -> None:
        self.decisions: deque[dict[str, Any]] = deque(maxlen=MEMORY_DECISIONS)
        self.landmarks: deque[dict[str, Any]] = deque(maxlen=MEMORY_LANDMARKS)

    def remember(self, decision: dict[str, Any], lidar: LidarMetrics) -> None:
        self.decisions.append({
            "at": round(time.time(), 1),
            "action_mode": decision.get("action_mode"),
            "heading_offset": decision.get("target_heading_offset"),
            "deadlock": decision.get("deadlock_analysis"),
            "odom": [lidar.odom_x_m, lidar.odom_y_m],
            "scene": (decision.get("scene_understanding") or "")[:160],
        })
        scene = (decision.get("scene_understanding") or "").strip()
        if scene:
            self.landmarks.append({
                "seen_at": round(time.time(), 1),
                "odom": [lidar.odom_x_m, lidar.odom_y_m],
                "description": scene[:120],
            })

    def looping(self) -> bool:
        """A local, cheap deadlock check that does not need the model.

        Two signatures, both measured from the decisions themselves: the same
        action mode repeated with no net movement, and headings that cancel
        out (+90 then -90 then +90). Used to TELL the model it is looping, not
        to replace its judgement -- and to escalate effort when it happens.
        """
        if len(self.decisions) < 3:
            return False
        recent = list(self.decisions)[-3:]
        modes = {d["action_mode"] for d in recent}
        # Only usable when the pose is real. With no pose the displacement is
        # always zero, which would report every repeated mode as a deadlock.
        odoms = [d["odom"] for d in recent]
        if all(o[0] is not None for o in odoms):
            moved = math.hypot(odoms[-1][0] - odoms[0][0],
                               odoms[-1][1] - odoms[0][1])
            if len(modes) == 1 and moved < 0.25:
                return True
        offsets = [d["heading_offset"] or 0.0 for d in recent]
        return abs(sum(offsets)) < 10.0 and all(abs(o) > 30.0 for o in offsets)

    def as_prompt_block(self) -> dict[str, Any]:
        return {
            "recent_decisions_oldest_first": list(self.decisions),
            "landmarks_already_seen": list(self.landmarks),
            "local_loop_detector": self.looping(),
        }


# --- the planner -------------------------------------------------------------

class SpendGuard:
    """Rough running cost, and a hard daily stop. See CLAUDE.md cost safety."""

    def __init__(self, cap_usd: float = DAILY_USD_CAP) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.day = time.strftime("%Y-%m-%d")

    def record(self, usage: Any) -> float:
        if time.strftime("%Y-%m-%d") != self.day:      # new day, new budget
            self.day, self.spent_usd, self.calls = time.strftime("%Y-%m-%d"), 0.0, 0
        cost = (getattr(usage, "input_tokens", 0) / 1e6 * PRICING_USD_PER_MTOK["input"]
                + getattr(usage, "output_tokens", 0) / 1e6 * PRICING_USD_PER_MTOK["output"])
        self.spent_usd += cost
        self.calls += 1
        return cost

    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd


def _client():
    """The async client. Raises with a readable message when there is no key."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:                          # noqa: BLE001
        raise RuntimeError(
            "the anthropic SDK is not installed in this venv "
            "(pip install anthropic)") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or run --dry-run to "
            "exercise the pipeline without the API.")
    return AsyncAnthropic()                             # reads the env var itself


def fallback_decision(reason: str) -> dict[str, Any]:
    """What the robot does when the planner cannot answer.

    HOLD_POSITION and a zero turn, always. The alternative -- keeping the last
    heading, or guessing -- means an unreachable API turns into a robot driving
    on a stale belief. Tier 1 keeps it safe either way; this keeps it honest.
    """
    return {
        "scene_understanding": f"No cognitive decision available: {reason}.",
        "deadlock_analysis": False,
        "action_mode": "HOLD_POSITION",
        "target_heading_offset": 0.0,
        "degraded": True,
    }


async def query_sonnet5_planner(client, vision: VisionFrames, lidar: LidarMetrics,
                                state: str, battery_pct: float | None,
                                memory: SpatialMemory, spend: SpendGuard,
                                context: dict[str, Any] | None = None,
                                battery_note: str = "") -> dict[str, Any]:
    """One multimodal decision from Claude Sonnet 5, or a safe fallback.

    The two camera frames go in as image blocks and everything measurable goes
    in as one JSON text block -- numbers as numbers, not prose, so the model is
    never parsing English to find a distance.
    """
    if spend.exhausted():
        return fallback_decision(
            f"daily spend cap reached (${spend.spent_usd:.2f} of ${spend.cap_usd:.2f})")

    telemetry = {
        "robot_state_from_local_controller": state,
        "battery_percent": battery_pct,
        "battery_provenance": battery_note,
        "lidar_source": lidar.source,
        "camera_status": vision.status,
        "tof_nearest_m": lidar.tof_nearest_m,
        "run_reported_state": context or {},
        "battery_policy": {
            "return_to_base_below_pct": BATTERY_RETURN_PCT,
            "critical_below_pct": BATTERY_CRITICAL_PCT,
        },
        "lidar_sectors_m": lidar.sectors_m,
        "tightest_sector": dict(zip(("sector", "metres"), lidar.tightest())),
        "odometry": ({
            "x_m": lidar.odom_x_m, "y_m": lidar.odom_y_m,
            "heading_deg": lidar.heading_deg,
            "linear_velocity_mps": lidar.linear_velocity_mps,
            "distance_from_origin_m": round(lidar.distance_from_origin_m(), 2),
        } if lidar.known_pose() else {
            "position": "UNKNOWN -- this run publishes no live pose and the "
                        f"saved one is {lidar.pose_age_s}s old. Do not reason "
                        "about where you are; reason from the sectors.",
            "linear_velocity_mps": lidar.linear_velocity_mps,
        }),
        "short_term_spatial_memory": memory.as_prompt_block(),
    }

    # However many cameras actually produced a frame -- often one, sometimes
    # none. Each is labelled, because "the floor camera sees carpet" and "the
    # forward camera sees carpet" mean completely different things.
    content: list[dict[str, Any]] = []
    for label, encoded in vision.frames():
        content.append({"type": "text", "text": f"{label}:"})
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": vision.media_type,
                                   "data": encoded}})
    if not content:
        content.append({"type": "text", "text":
                        "NO CAMERA FRAMES this tick — "
                        + json.dumps(vision.status)
                        + ". Decide from the sectors and the run state alone, "
                          "and say in scene_understanding that you were blind."})
    content.append({"type": "text",
                    "text": "Telemetry:\n" + json.dumps(telemetry, indent=2)})

    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                # Stable prefix first so it can cache; note the minimum cacheable
                # prefix is model-dependent, so a short system prompt may simply
                # not cache -- check usage.cache_read_input_tokens before
                # claiming it does.
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                # Sonnet 5: adaptive is the only on-mode, and budget_tokens is
                # removed (sending it is a 400). Depth comes from effort.
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "max" if memory.looping() else EFFORT,
                    "format": {"type": "json_schema", "schema": DECISION_SCHEMA},
                },
            ),
            timeout=API_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return fallback_decision(f"planner did not answer within {API_TIMEOUT_S:.0f}s")
    except Exception as exc:                            # noqa: BLE001
        return fallback_decision(_describe_api_error(exc))

    cost = spend.record(response.usage)
    log.info("sonnet5 %s in=%d out=%d ~$%.4f (day $%.3f)", response.stop_reason,
             response.usage.input_tokens, response.usage.output_tokens,
             cost, spend.spent_usd)

    # A refusal is a 200 with no usable content -- check before reading it.
    if response.stop_reason == "refusal":
        return fallback_decision("the model declined this request")
    try:
        text = next(b.text for b in response.content if b.type == "text")
        decision = json.loads(text)
    except (StopIteration, json.JSONDecodeError, AttributeError) as exc:
        return fallback_decision(f"unparseable response ({type(exc).__name__})")

    decision["target_heading_offset"] = _clamp_heading(
        decision.get("target_heading_offset", 0.0))
    return decision


def _describe_api_error(exc: Exception) -> str:
    """Most specific first: retryable and non-retryable must not look alike."""
    try:
        import anthropic
    except ImportError:
        return f"{type(exc).__name__}: {exc}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"model {MODEL} not found for this key"
    if isinstance(exc, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY rejected"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited — back off and keep driving on Tier 1"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API status {exc.status_code} (retryable)" if exc.status_code >= 500 \
            else f"API rejected the request: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "network unreachable — the robot is offline"
    return f"{type(exc).__name__}: {exc}"


def _clamp_heading(value: Any) -> float:
    """Keep the turn inside -180..180 whatever the model sent."""
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        return 0.0
    degrees = math.fmod(degrees, 360.0)
    if degrees > 180.0:
        degrees -= 360.0
    elif degrees < -180.0:
        degrees += 360.0
    return round(degrees, 1)


def battery_override(battery_pct: float | None,
                     lidar: LidarMetrics) -> dict[str, Any] | None:
    """Power is a local hard rule, not something to ask a language model about.

    Below the critical line the robot stops; below the return line it goes home.
    Both are decided here, before any API call, because a flat battery halfway
    through a request is exactly when the network is least useful.
    """
    if battery_pct is None:
        return None            # no gauge on this robot; nothing to rule on
    if battery_pct <= BATTERY_CRITICAL_PCT:
        return {
            "scene_understanding": f"Battery critical at {battery_pct:.0f}%. "
                                   f"Holding position to preserve charge.",
            "deadlock_analysis": False,
            "action_mode": "HOLD_POSITION",
            "target_heading_offset": 0.0,
            "battery_override": True,
        }
    if battery_pct <= BATTERY_RETURN_PCT:
        if not lidar.known_pose():
            return {
                "scene_understanding": (
                    f"Battery {battery_pct:.0f}% is below the return threshold, but "
                    f"there is no current pose, so the bearing home is unknown."),
                "deadlock_analysis": False,
                "action_mode": "HOLD_POSITION",
                "target_heading_offset": 0.0,
                "battery_override": True,
            }
        # Turn toward the odometry origin: bearing to (0,0) minus current heading.
        bearing = math.degrees(math.atan2(-lidar.odom_y_m, -lidar.odom_x_m))
        return {
            "scene_understanding": (
                f"Battery {battery_pct:.0f}% is below the {BATTERY_RETURN_PCT:.0f}% "
                f"return threshold; {lidar.distance_from_origin_m():.1f} m from base."),
            "deadlock_analysis": False,
            "action_mode": "RETURN_TO_BASE_LOW_BATTERY",
            "target_heading_offset": _clamp_heading(bearing - lidar.heading_deg),
            "battery_override": True,
        }
    return None


def exploration_context(autonomy: dict | None) -> dict[str, Any]:
    """What the run itself already knows, forwarded to the planner.

    THIS IS THE POINT OF THE WHOLE BRIDGE. Local geometry alone cannot explain
    the measured failure: a 10-minute run added 0.5 m2 to a 321 m2 map with a
    revisit ratio of 0.64, while the planner reported 474 frontiers and the
    navigator handed back most ticks. The run already knows which targets it
    abandoned and why it is recovering; a planner that cannot see that is
    guessing at the one question worth asking.
    """
    if not autonomy:
        return {}
    target = autonomy.get("EXPLORATION_TARGET") or {}
    recovery = autonomy.get("RECOVERY_STATUS") or {}
    map_state = autonomy.get("MAP_STATE") or {}
    capability = autonomy.get("CAPABILITY_CHECK") or {}
    return {
        "mode": autonomy.get("CURRENT_MODE"),
        "tick": autonomy.get("TICK"),
        "why_last_action": autonomy.get("REASON"),
        "reason_code": autonomy.get("REASON_CODE"),
        "next_objective": autonomy.get("NEXT_OBJECTIVE"),
        "frontier_planner": {k: target.get(k) for k in
                             ("has_target", "target_distance", "target_age_s",
                              "frontiers", "exhausted", "chosen", "arrived",
                              "abandoned", "blacklisted")},
        "recovery": {"state": recovery.get("state"),
                     "active": recovery.get("active"),
                     "triggers": recovery.get("triggers")},
        "coverage": {k: map_state.get(k) for k in
                     ("explored_m2", "visited_m2", "revisited_m2",
                      "revisit_ratio")},
        "what_the_drivetrain_can_execute": capability.get("action_space"),
        "safety": autonomy.get("SAFETY_STATUS"),
        "run_totals": {k: (autonomy.get("RUN_TOTALS") or {}).get(k) for k in
                       ("ticks", "reflex_stops", "escape_turns", "nav_handoffs",
                        "recovery_ticks", "reverse_escapes")},
    }


# --- the loop ----------------------------------------------------------------

def emit_payload(payload: dict[str, Any], log_path: Path | None = None) -> None:
    """SEAM 5 — hand the decision to the local controller.

    stdout by design: a pipe is the smallest possible coupling, and it keeps
    this file free of any import from the robot's own stack. Swap for a socket,
    an MCP call or a file drop; keep it one function.
    """
    line = json.dumps(payload, separators=(",", ":"))
    print(line, flush=True)
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            log.warning("could not write %s (%s)", log_path, exc)


async def tick(client, memory: SpatialMemory, spend: SpendGuard,
               dry_run: bool, state: str | None = None,
               autonomy: dict | None = None, age_s: float = float("inf"),
               manual_battery: float | None = None,
               log_path: Path | None = None,
               vision_enabled: bool = True) -> dict[str, Any]:
    """One full cycle: sense, decide, remember, log.

    The run state is read ONCE by the caller and passed in. Reading it again in
    here was a real bug: --on-stuck gated on one sample and the payload then
    reported a second, independent one.
    """
    started = time.monotonic()
    vision = capture_front_vision(vision_enabled)
    lidar = read_lidar_metrics(autonomy, age_s)
    state = get_robot_state(autonomy, age_s) if state is None else state
    battery_pct, battery_note = read_battery(manual_battery)
    context = exploration_context(autonomy)

    decision = battery_override(battery_pct, lidar)
    if decision is None:
        if dry_run:
            decision = fallback_decision("--dry-run, no API call made")
        else:
            decision = await query_sonnet5_planner(
                client, vision, lidar, state, battery_pct, memory, spend,
                context, battery_note)

    memory.remember(decision, lidar)
    payload = {
        "schema": "friday.cognitive.v1",
        "at": round(time.time(), 2),
        "tier": 2,
        "advisory": True,
        # NOTHING STEERS ON THIS. No consumer is wired: this process observes a
        # run and writes down what it would have advised. Wiring it to motion is
        # a separate, deliberate step -- and the brain-target seam is still the
        # only cognitive->motion link on this robot, itself never proven.
        "steering": "not_wired (log only)",
        "robot_state": state,
        "battery_percent": battery_pct,
        "sources": {
            "lidar_odometry": lidar.source,
            "pose_age_s": lidar.pose_age_s,
            "robot_state": ("live run" if autonomy and age_s <= STALE_AFTER_S
                            else f"no live run (state {age_s:.0f}s old)"),
            "battery": battery_note,
            "cameras": vision.status,
        },
        "run_tick": (autonomy or {}).get("TICK"),
        "decision": decision,
        "planner_latency_s": round(time.monotonic() - started, 2),
        "spend_usd_today": round(spend.spent_usd, 4),
    }
    emit_payload(payload, log_path)
    return payload


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=TICK_SECONDS,
                    help="seconds between decisions (default %(default)s)")
    ap.add_argument("--on-stuck", action="store_true",
                    help="only call the planner when the local controller says "
                         "STUCK_VELOCITY_0 or OBSTACLE_PROXIMITY (far cheaper)")
    ap.add_argument("--once", action="store_true", help="one decision, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the whole pipeline without calling the API")
    ap.add_argument("--battery-pct", type=float, default=None,
                    help="battery percentage by hand -- there is NO fuel gauge "
                         "on this robot, so battery rules stay off without it")
    ap.add_argument("--log", type=Path, default=DECISION_LOG,
                    help="append every decision here as JSONL (default %(default)s)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-vision", action="store_true",
                    help="skip the cameras (they are the token bill, and an "
                         "exploration run owns them while it drives)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s", stream=sys.stderr)

    client = None
    if not args.dry_run:
        try:
            client = _client()
        except RuntimeError as exc:
            log.error("%s", exc)
            return 2

    memory, spend = SpatialMemory(), SpendGuard()
    log.info("cognitive planner up: model=%s interval=%.1fs mode=%s cap=$%.2f",
             MODEL, args.interval, "on-stuck" if args.on_stuck else "periodic",
             spend.cap_usd)
    try:
        while True:
            # ONE read per tick, used by both the gate and the decision.
            autonomy, age_s = read_autonomy()
            state_now = get_robot_state(autonomy, age_s)
            if args.on_stuck and state_now in ("EXPLORING", "IDLE"):
                await asyncio.sleep(args.interval)
                continue
            try:
                payload = await tick(client, memory, spend, args.dry_run,
                                     state_now, autonomy, age_s,
                                     args.battery_pct,
                                     None if args.no_log else args.log,
                                     not args.no_vision)
            except Exception as exc:                    # noqa: BLE001
                # One bad tick must never end the loop: a planner that dies is a
                # robot that silently stops thinking while still driving.
                log.exception("tick failed (%s) — continuing", type(exc).__name__)
                await asyncio.sleep(args.interval)
                continue
            if args.once:
                return 0
            if spend.exhausted():
                log.error("daily cap $%.2f reached — stopping the planner; "
                          "the robot keeps running on Tier 1", spend.cap_usd)
                return 3
            low = payload["battery_percent"] <= BATTERY_RETURN_PCT
            await asyncio.sleep(BATTERY_LOW_TICK_SECONDS if low else args.interval)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("stopped after %d decisions, ~$%.3f", spend.calls, spend.spent_usd)
        return 0
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
