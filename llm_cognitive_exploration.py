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

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

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

HOW TO READ THE INPUT. Sector distances are metres, clockwise from the robot's \
current heading: N is straight ahead, E is 90 degrees right, W is 90 degrees \
left, S is directly behind. Odometry is metres and degrees in the map frame. \
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

    up_b64: str            # forward camera, above the screen
    down_b64: str          # floor camera, above the bumper
    media_type: str = "image/jpeg"
    captured_at: float = field(default_factory=time.time)


def _mock_jpeg(label: str) -> str:
    """A valid 1x1 JPEG, so the payload shape is real even while the data is not.

    Deliberately not random bytes: the wire format, the base64 encoding and the
    content-block shape are exactly what the real frames will use, so swapping
    in a camera later changes one function and nothing downstream.
    """
    jpeg_1x1 = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffc20011"
        "08000100010101110000ffc40014000100000000000000000000"
        "000000000009ffda0008010100000000013fffd9")
    log.debug("mock frame for %s (%d bytes)", label, len(jpeg_1x1))
    return base64.standard_b64encode(jpeg_1x1).decode("ascii")


def capture_front_vision() -> VisionFrames:
    """SEAM 1 — grab one frame from each USB camera, base64 encoded.

    Real implementation: read the two /dev/video nodes, take ONE frame each,
    encode as JPEG at around 640x480, and base64 it. Two things matter when you
    wire this up:
      * resolution is the bill. Each 640x480 frame is roughly 1.5k tokens; going
        to 1280x960 quadruples that for detail this planner does not use.
      * a stale frame is worse than no frame. Stamp captured_at and refuse to
        plan on anything older than a tick.
    """
    return VisionFrames(up_b64=_mock_jpeg("up"), down_b64=_mock_jpeg("down"))


@dataclass
class LidarMetrics:
    """One 8-sector range slice plus where the robot thinks it is."""

    sectors_m: dict[str, float]
    odom_x_m: float
    odom_y_m: float
    heading_deg: float
    linear_velocity_mps: float

    def tightest(self) -> tuple[str, float]:
        name = min(self.sectors_m, key=self.sectors_m.get)
        return name, self.sectors_m[name]

    def distance_from_origin_m(self) -> float:
        return math.hypot(self.odom_x_m, self.odom_y_m)


def read_lidar_metrics() -> LidarMetrics:
    """SEAM 2 — 8 sector minima from the LiDAR, plus odometry.

    Real implementation: take the 360-degree scan, collapse it into these eight
    45-degree sectors using the MINIMUM range in each (a mean hides the table
    leg that stops the robot), and mask the sectors the chassis occludes.
    """
    base = {"N": 2.4, "NE": 1.8, "E": 0.9, "SE": 1.2, "S": 3.1, "SW": 2.2,
            "W": 1.1, "NW": 1.6}
    jitter = {k: round(max(0.12, v + random.uniform(-0.25, 0.25)), 2)
              for k, v in base.items()}
    return LidarMetrics(
        sectors_m=jitter,
        odom_x_m=round(random.uniform(-4.0, 4.0), 2),
        odom_y_m=round(random.uniform(-4.0, 4.0), 2),
        heading_deg=round(random.uniform(0.0, 359.9), 1),
        linear_velocity_mps=round(random.choice([0.0, 0.0, 0.12, 0.18]), 2),
    )


def get_robot_state() -> str:
    """SEAM 3 — the local controller's own state machine label.

    EXPLORING | STUCK_VELOCITY_0 | OBSTACLE_PROXIMITY | RECOVERING | IDLE.
    This is the single most valuable input here: the local controller knows it
    is stuck long before the geometry looks stuck from outside.
    """
    return random.choice(
        ["EXPLORING", "EXPLORING", "EXPLORING", "STUCK_VELOCITY_0",
         "OBSTACLE_PROXIMITY"])


def read_battery() -> float:
    """SEAM 4 — battery percentage, 0-100."""
    return round(random.uniform(18.0, 95.0), 1)


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
        moved = math.hypot(recent[-1]["odom"][0] - recent[0]["odom"][0],
                           recent[-1]["odom"][1] - recent[0]["odom"][1])
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
                                state: str, battery_pct: float,
                                memory: SpatialMemory,
                                spend: SpendGuard) -> dict[str, Any]:
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
        "battery_policy": {
            "return_to_base_below_pct": BATTERY_RETURN_PCT,
            "critical_below_pct": BATTERY_CRITICAL_PCT,
        },
        "lidar_sectors_m": lidar.sectors_m,
        "tightest_sector": dict(zip(("sector", "metres"), lidar.tightest())),
        "odometry": {
            "x_m": lidar.odom_x_m, "y_m": lidar.odom_y_m,
            "heading_deg": lidar.heading_deg,
            "linear_velocity_mps": lidar.linear_velocity_mps,
            "distance_from_origin_m": round(lidar.distance_from_origin_m(), 2),
        },
        "short_term_spatial_memory": memory.as_prompt_block(),
    }

    content = [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": vision.media_type,
                                     "data": vision.up_b64}},
        {"type": "text", "text": "Above: FORWARD camera. Below: FLOOR camera."},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": vision.media_type,
                                     "data": vision.down_b64}},
        {"type": "text", "text": "Telemetry:\n" + json.dumps(telemetry, indent=2)},
    ]

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


def battery_override(battery_pct: float, lidar: LidarMetrics) -> dict[str, Any] | None:
    """Power is a local hard rule, not something to ask a language model about.

    Below the critical line the robot stops; below the return line it goes home.
    Both are decided here, before any API call, because a flat battery halfway
    through a request is exactly when the network is least useful.
    """
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


# --- the loop ----------------------------------------------------------------

def emit_payload(payload: dict[str, Any]) -> None:
    """SEAM 5 — hand the decision to the local controller.

    stdout by design: a pipe is the smallest possible coupling, and it keeps
    this file free of any import from the robot's own stack. Swap for a socket,
    an MCP call or a file drop; keep it one function.
    """
    print(json.dumps(payload, separators=(",", ":")), flush=True)


async def tick(client, memory: SpatialMemory, spend: SpendGuard,
               dry_run: bool, state: str | None = None) -> dict[str, Any]:
    """One full cycle: sense, decide, remember, emit.

    `state` is passed in by the caller so the robot's state machine is read
    EXACTLY ONCE per tick. Reading it again in here was a real bug: --on-stuck
    gated on one sample and the payload then reported a second, independent
    one, so decisions could be logged against a state nobody had checked.
    """
    started = time.monotonic()
    vision = capture_front_vision()
    lidar = read_lidar_metrics()
    state = get_robot_state() if state is None else state
    battery = read_battery()

    decision = battery_override(battery, lidar)
    if decision is None:
        if dry_run:
            decision = fallback_decision("--dry-run, no API call made")
        else:
            decision = await query_sonnet5_planner(
                client, vision, lidar, state, battery, memory, spend)

    memory.remember(decision, lidar)
    payload = {
        "schema": "friday.cognitive.v1",
        "at": round(time.time(), 2),
        "tier": 2,
        "advisory": True,            # Tier 1 may override; see the docstring
        "robot_state": state,
        "battery_percent": battery,
        "decision": decision,
        "planner_latency_s": round(time.monotonic() - started, 2),
        "spend_usd_today": round(spend.spent_usd, 4),
    }
    emit_payload(payload)
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
            state_now = get_robot_state()
            if args.on_stuck and state_now == "EXPLORING":
                await asyncio.sleep(args.interval)
                continue
            try:
                payload = await tick(client, memory, spend, args.dry_run, state_now)
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
