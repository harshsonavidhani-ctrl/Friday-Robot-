"""Decision provenance for the RL navigation stack: what decided each tick, and why.

WHY THIS EXISTS
---------------
The first autonomous run produced a black box whose `v_policy` column was a lie on
11.3% of ticks. `ControlLoop.tick` overwrote `v_pol`/`omega_pol` in the no-reverse
fallback BEFORE recording them, so 34 ticks whose heading came from `_open_side()` --
a hand-written rule -- were logged under the policy's name, at exactly `+-omega_max`.
The single question the log existed to answer ("is my trained policy driving this
robot?") was the one question it could not answer.

So the rule this module enforces is:

    THE RAW POLICY OUTPUT IS CAPTURED ONCE, IMMEDIATELY AFTER INFERENCE,
    AND IS NEVER WRITTEN TO AGAIN.

Every later stage appends a new entry instead of mutating the old one. The chain is
then a provenance record rather than a single mutable pair of floats, and "who chose
this heading" is answerable by reading it rather than by reasoning about the order of
assignments in a 160-line function.

NO FABRICATED VALUES
--------------------
Every number in the rendered blocks is measured or is explicitly marked absent.
Where a subsystem is not contributing -- an uncalibrated visual gate, a camera that
never opened, a ToF that has not reported -- the block says so in those words rather
than printing a plausible-looking default. A status display that invents a clearance
is worse than no status display, because it is believed.

The `reason` lines are generated from the record's own fields (the sector clearances,
the threshold that fired, the stage that changed the command) and never from a
template chosen by vibe. If a reason cannot be grounded in a field, it is not emitted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Every stage of the chain, in the order tick() applies them. Ordering matters: the
# rendered chain is read top-to-bottom as "what happened to the command", and a stage
# appearing out of order would misrepresent which one had the last word.
STAGE_ORDER = (
    "policy",           # raw ONNX output -- the only stage that CREATES a command
    "no_reverse",       # control adaptation: unexecutable reverse -> pivot
    "planner",          # frontier target: bounded HEADING bias, never speed
    "navigator",        # local nav: the best DRIVABLE arc, capped by the policy's v
    "recovery",         # stuck: a deliberate, escalating, bounded escape
    "governor",         # LiDAR braking; may only reduce
    "escape",           # control adaptation: boxed in -> decisive pivot
    "reverse_floor",    # control adaptation: sub-stiction reverse -> slowest executable
    "visual",           # camera hazard; advisory, may only reduce
    "tof",              # ToF reflex; hard veto
    "limiter",          # slew limit; may only reduce rate of change
)

# ======================================================================================
# ACTION SOURCES -- the five things that can decide what the wheels do
# ======================================================================================
# Every executed action traces to exactly one of these. They are separate because they
# fail differently, and lumping any two together destroys the ability to diagnose:
#
#   RL_POLICY            the trained network. The only source that CREATES a command.
#   EXPLORATION_PLANNER  a decision about WHERE TO GO, from the persistent map.
#                        Bounded heading bias; it can never change speed.
#   SAFETY_OVERRIDE      a veto or a brake. Can only ever REDUCE.
#   RECOVERY_CONTROLLER  the robot is stuck and a deliberate escape is being driven.
#   CAPABILITY_ADAPTER   the command is physically unexecutable by this drivetrain and
#                        was translated into one that is. NOT a safety event and NOT a
#                        decision -- the intent is preserved, the means are changed.
#   MOTION_SMOOTHING     shapes how fast a command is approached, never what it is.
#
# The distinction between RECOVERY and CAPABILITY matters and was previously lost: both
# used to be "RULE". `escape` fires because the ROBOT is stuck; `no_reverse` fires
# because the COMMAND cannot be executed. A robot reporting "recovery" while merely
# translating a reverse tells you to go look for a trap that is not there.
#   LOCAL_NAVIGATOR      a choice between COMMANDS THAT ARE ALL ALLOWED -- which of the
#                        drivable arcs actually goes somewhere. It is not a veto and not
#                        a brake, and filing it under SAFETY_OVERRIDE (which is what an
#                        unregistered stage name defaults to) would have inflated the
#                        very number it was built to reduce.
SRC_POLICY = "RL_POLICY"
SRC_PLANNER = "EXPLORATION_PLANNER"
SRC_NAVIGATOR = "LOCAL_NAVIGATOR"
SRC_SAFETY = "SAFETY_OVERRIDE"
SRC_RECOVERY = "RECOVERY_CONTROLLER"
SRC_CAPABILITY = "CAPABILITY_ADAPTER"
SRC_SMOOTHING = "MOTION_SMOOTHING"

POLICY_STAGES = frozenset({"policy"})
PLANNER_STAGES = frozenset({"planner"})
NAVIGATOR_STAGES = frozenset({"navigator"})
# `_open_side()` -- "turn toward whichever side has more open space" -- is called from
# BOTH of these, and that is why its architectural role was ambiguous. It is not itself
# a controller; it is a SHARED PRIMITIVE, a direction chooser that reads the live scan.
# The two callers use it for different reasons and are classified by the reason.
RECOVERY_STAGES = frozenset({"escape", "recovery"})   # the robot is stuck
# Both are the same class of adaptation -- "the drivetrain cannot execute what was
# asked" -- and differ only in which way they resolve it. `no_reverse` substitutes a
# pivot when NO reverse is executable; `reverse_floor` raises a too-gentle reverse to
# the slowest one that is, wherever in the chain it was authored.
CAPABILITY_STAGES = frozenset({"no_reverse", "reverse_floor"})
RULE_STAGES = RECOVERY_STAGES | CAPABILITY_STAGES   # kept: "not policy, not safety"
SAFETY_STAGES = frozenset({"governor", "visual", "tof"})
# The slew limiter is NOT a safety override and must not be counted as one. It changes
# how fast the command is approached, never what the command is -- a policy asking for
# a right arc still gets a right arc, reached over two ticks instead of one. It also
# fires on nearly every tick, because any acceleration from standstill trips it. Filed
# under SAFETY it would report "Safety Override: YES" almost always, which trains the
# reader to ignore the field, which is worse than not showing it. It gets its own line.
SMOOTHING_STAGES = frozenset({"limiter"})


@dataclass
class Stage:
    """One transformation of the command, with the measurement that caused it."""

    name: str
    v: float
    omega: float
    changed: bool = False
    detail: str = ""          # the measured quantity and threshold that fired

    @property
    def kind(self) -> str:
        if self.name in POLICY_STAGES:
            return SRC_POLICY
        if self.name in PLANNER_STAGES:
            return SRC_PLANNER
        if self.name in NAVIGATOR_STAGES:
            return SRC_NAVIGATOR
        if self.name in RECOVERY_STAGES:
            return SRC_RECOVERY
        if self.name in CAPABILITY_STAGES:
            return SRC_CAPABILITY
        if self.name in SMOOTHING_STAGES:
            return SRC_SMOOTHING
        return SRC_SAFETY


@dataclass
class DecisionRecord:
    """Everything known about one control tick. Built by tick(), rendered here.

    Fields default to None rather than to 0.0 wherever "not measured" and "measured
    as zero" are different facts. A ToF that has not reported is not a ToF reading
    0.00 m, and rendering it as such would make a dead sensor look like a clear path.
    """

    tick: int = 0
    t_wall: float = field(default_factory=time.time)

    # ---- observation, as fed to the network ----
    obs_dim: int = 0
    lidar_front: float | None = None
    lidar_left: float | None = None
    lidar_right: float | None = None
    lidar_rear: float | None = None
    lidar_min: float | None = None
    lidar_age: float | None = None
    frontier_near: float | None = None      # closest unknown cell along any ray
    frontier_live: bool = False             # does the channel carry real variation?
    tof_min: float | None = None
    tof_age: float | None = None
    visual_score: float | None = None
    visual_enabled: bool = False
    cams_reporting: int = 0
    # Per-role camera state and the up camera's measured context. `context` holds
    # brightness/change/occlusion ONLY -- there is no object detection behind it and
    # no field here names what is in view.
    cam_state: dict | None = None
    context: dict | None = None

    # ---- policy ----
    model_name: str = ""
    model_loaded: bool = False
    infer_ms: float | None = None
    v_raw: float | None = None              # NEVER overwritten after inference
    omega_raw: float | None = None

    # ---- chain ----
    stages: list[Stage] = field(default_factory=list)

    # ---- actuation ----
    final_v: float = 0.0
    final_omega: float = 0.0
    wire: str = ""                          # the literal line sent to the Arduino
    duties: tuple[int, int] | None = None
    executed_v: float | None = None
    executed_omega: float | None = None

    # ---- bookkeeping ----
    gate: str = "ok"
    # Planner context for this tick: bearing error, applied bias, and why. None means
    # the planner is off, which is a different fact from "the planner had nothing to
    # say" and is rendered differently.
    planner: dict | None = None
    # Local navigator: the situation it assessed, whether it was holding a committed
    # direction, and the clearance of the arc it chose. None when it declined the tick
    # (nothing forward was drivable) or is switched off -- which, as with `planner`
    # above, is a different fact from "it ran and found nothing".
    navigator: dict | None = None
    # Result of asking "can the drivetrain execute what the policy asked for?", run
    # every tick so "checked and fine" is distinguishable from "never checked".
    capability: dict | None = None
    explored_m2: float | None = None
    # Persistent world model (world.WorldMap). None => mapping is off, which is a
    # different fact from "mapped nothing" and is rendered differently.
    world: dict | None = None
    world_loaded: bool = False
    pose: tuple[float, float, float] | None = None
    loop_ms: float | None = None

    # ------------------------------------------------------------------ helpers ----
    def add(self, name: str, v: float, omega: float, detail: str = "") -> None:
        """Append a stage, marking it changed only if it actually moved the command.

        Compared against the PREVIOUS stage, not against the raw policy: a stage that
        passes through a command another stage already reduced has not itself changed
        anything, and marking it as though it had would spread the credit for one
        intervention across every stage downstream of it.
        """
        prev = self.stages[-1] if self.stages else None
        changed = prev is None or abs(v - prev.v) > 1e-9 or abs(omega - prev.omega) > 1e-9
        self.stages.append(Stage(name, v, omega, changed and prev is not None, detail))

    @property
    def deciding_stage(self) -> Stage | None:
        """The last stage that changed WHAT the command is.

        Smoothing is skipped: the limiter shapes the approach to a command without
        choosing it, so attributing the decision to it would report SAFETY on almost
        every tick and bury the ticks where safety genuinely took over.
        """
        for s in reversed(self.stages):
            if s.changed and s.name not in SMOOTHING_STAGES:
                return s
        return self.stages[0] if self.stages else None

    @property
    def rate_limited(self) -> bool:
        return any(s.changed for s in self.stages if s.name in SMOOTHING_STAGES)

    @property
    def authority(self) -> str:
        """ACTION_SOURCE: the one thing that decided what the wheels do this tick."""
        s = self.deciding_stage
        return s.kind if s is not None else "NONE"

    @property
    def action_source(self) -> str:
        """Alias for `authority`, under the name the architecture uses."""
        return self.authority

    def provenance(self) -> dict[str, Any]:
        """The full per-action record: raw policy in, executed action out, and why.

        This is the contract Phase 4 asks for, in one place, so a caller never has to
        reassemble it from stages and risk getting the attribution wrong.
        """
        safety = [s for s in self.stages if s.changed and s.name in SAFETY_STAGES]
        return {
            "RAW_POLICY_ACTION": None if self.v_raw is None
            else {"v": self.v_raw, "omega": self.omega_raw},
            "PLANNER_CONTEXT": self.planner,
            "SAFETY_DECISION": {
                "overridden": bool(safety),
                "stages": [s.name for s in safety],
                "detail": "; ".join(s.detail for s in safety if s.detail),
            },
            "FINAL_EXECUTED_ACTION": {
                "v": self.final_v, "omega": self.final_omega,
                "name": action_name(self.final_v, self.final_omega),
                "wire": self.wire, "duties": self.duties,
            },
            "ACTION_SOURCE": self.action_source,
            "CAPABILITY_CHECK": self.capability,
            "MODIFICATION_REASON": (self.deciding_stage.detail
                                    if self.deciding_stage is not None else ""),
        }

    @property
    def policy_survived(self) -> bool:
        """Did the network's command reach the wheels unmodified (bar slew limiting)?"""
        return not any(s.changed for s in self.stages if s.name != "limiter")


# ======================================================================================
# Sector clearances -- shared so the status block and the reasons cannot disagree
# ======================================================================================
def sector(scan: np.ndarray, centre_frac: float, half_frac: float = 1 / 24) -> float:
    """Minimum range over a wedge of the scan, centred on a fraction of a full turn.

    Beam 0 is forward and indices run counter-clockwise (see
    `LocalExplorationMap._beam_angles`), so 0.0 = ahead, 0.25 = left, 0.5 = behind,
    0.75 = right. Fractions rather than indices so this is correct for any `n_beams`.

    MINIMUM here, deliberately, and unlike `_open_side`'s mean. These numbers are read
    as "how much room is there", where one close return is the whole answer. `_open_side`
    asks a different question -- "which way is more open overall" -- and a minimum
    answers that one badly, which is why the two differ.
    """
    n = len(scan)
    c = int(round(centre_frac * n)) % n
    h = max(1, int(round(half_frac * n)))
    idx = (np.arange(c - h, c + h + 1) % n)
    return float(np.min(scan[idx]))


def clearances(scan: np.ndarray) -> dict[str, float]:
    return {
        "front": sector(scan, 0.0),
        "left": sector(scan, 0.25),
        "rear": sector(scan, 0.5),
        "right": sector(scan, 0.75),
    }


# ======================================================================================
# Rendering
# ======================================================================================
def _m(x: float | None, unit: str = " m", nd: int = 2) -> str:
    """A measurement, or an explicit statement that there isn't one."""
    return "not reported" if x is None else f"{x:.{nd}f}{unit}"


def policy_status(rec: DecisionRecord) -> str:
    """The RL POLICY STATUS block: sensor -> policy -> action -> safety -> motors.

    This is the block that answers "is my trained model actually controlling Friday".
    It shows the raw network output beside the final wheel command, so a run in which
    the policy is being overruled looks obviously different from one in which it is
    driving -- rather than both looking like "the robot moved".
    """
    L = []
    add = L.append
    add("=" * 70)
    add("RL POLICY STATUS".center(70))
    add("=" * 70)
    add(f"ONNX Model Loaded:  {'YES  ' + rec.model_name if rec.model_loaded else 'NO'}")
    add(f"Inference Running:  {'YES' if rec.infer_ms is not None else 'NO'}"
        + (f"   ({rec.infer_ms:.3f} ms this tick)" if rec.infer_ms is not None else ""))
    add("")
    add(f"Observation Input:  {rec.obs_dim} values")
    add(f"  LiDAR  front {_m(rec.lidar_front)} | left {_m(rec.lidar_left)} | "
        f"right {_m(rec.lidar_right)} | rear {_m(rec.lidar_rear)}")
    if rec.frontier_near is not None:
        add(f"  Frontier  nearest unknown {_m(rec.frontier_near)}   "
            f"channel {'LIVE' if rec.frontier_live else 'FLAT (no unknown space in range)'}")
    add("")
    if rec.v_raw is None:
        add("Raw Policy Output:  NONE -- inference did not run this tick")
    else:
        add(f"Raw Policy Output:  v = {rec.v_raw:+.4f} m/s    omega = {rec.omega_raw:+.4f} rad/s")
        add("                    (straight from the network, before any safety stage)")
    add("")
    add("Chain:")
    for s in rec.stages:
        mark = "CHANGED" if s.changed else "pass   "
        add(f"  {s.name:<11} {mark}  v={s.v:+.3f}  omega={s.omega:+.3f}"
            + (f"   [{s.kind}] {s.detail}" if s.detail else f"   [{s.kind}]"))
    add("")
    d = rec.deciding_stage
    add(f"Selected Action:    {action_name(rec.final_v, rec.final_omega)}")
    add(f"Decided By:         {rec.authority}"
        + (f"  (last stage to change the command: {d.name})" if d is not None else ""))
    add("")
    over = [s for s in rec.stages if s.changed and s.name in SAFETY_STAGES]
    add(f"Safety Override:    {'YES -- ' + ', '.join(s.name for s in over) if over else 'NO'}")
    add(f"Rate Limited:       {'YES (acceleration shaped; heading unchanged)' if rec.rate_limited else 'NO'}")
    rec_st = [s for s in rec.stages if s.changed and s.name in RECOVERY_STAGES]
    cap_st = [s for s in rec.stages if s.changed and s.name in CAPABILITY_STAGES]
    if rec_st:
        add(f"Recovery:           ACTIVE -- {', '.join(s.name for s in rec_st)} "
            f"(the robot was stuck; this heading is NOT the policy's)")
    if cap_st:
        add(f"Capability Adapt:   YES -- {', '.join(s.name for s in cap_st)} "
            f"(the policy's command is unexecutable by this drivetrain; "
            f"intent kept, means changed)")
    pl_st = [s for s in rec.stages if s.changed and s.name in PLANNER_STAGES]
    if pl_st:
        add("Planner Steering:   YES -- heading biased toward the exploration target "
            "(speed untouched)")
    add("")
    add(f"Motor Command:      {rec.wire or '(nothing sent)'}")
    if rec.duties is not None:
        add(f"  LEFT duty {rec.duties[0]:+4d}   RIGHT duty {rec.duties[1]:+4d}")
    if rec.executed_v is not None:
        add(f"  executed twist    v={rec.executed_v:+.3f} m/s  omega={rec.executed_omega:+.3f} rad/s")
    add("")
    add(f"Final Executed:     v={rec.final_v:+.3f} m/s  omega={rec.final_omega:+.3f} rad/s")
    add("=" * 70)
    return "\n".join(L)


def action_name(v: float, omega: float, turn_eps: float = 0.15) -> str:
    """A human label for a continuous twist. The numbers stay beside it everywhere
    it is printed, so this is a reading aid and never the source of truth."""
    moving = abs(v) > 1e-3
    turning = abs(omega) > turn_eps
    if not moving and not turning:
        return "STOP"
    if not moving and turning:
        return "PIVOT_LEFT" if omega > 0 else "PIVOT_RIGHT"
    if moving and not turning:
        return "FORWARD" if v > 0 else "REVERSE"
    return ("FORWARD_ARC_LEFT" if omega > 0 else "FORWARD_ARC_RIGHT") if v > 0 else \
           ("REVERSE_ARC_LEFT" if omega > 0 else "REVERSE_ARC_RIGHT")


def reason(rec: DecisionRecord, cfg: Any) -> str:
    """One sentence saying why this action, grounded ONLY in this record's fields.

    Built from the stage that actually changed the command and the measurement that
    caused it. If no stage changed anything, the reason is the policy's own, and says
    so rather than inventing a rationale the network cannot be known to hold.
    """
    d = rec.deciding_stage
    if d is None:
        return "no decision recorded this tick"

    if not d.changed and d.name == "policy":
        # The honest limit of explainability here: a policy net does not expose a
        # reason. What CAN be stated is the observation it acted on and the action it
        # chose -- so state exactly that, and no more.
        bits = []
        if rec.lidar_front is not None:
            bits.append(f"{rec.lidar_front:.2f} m ahead")
        if rec.lidar_left is not None and rec.lidar_right is not None:
            side = "left" if rec.lidar_left > rec.lidar_right else "right"
            bits.append(f"more room {side} ({max(rec.lidar_left, rec.lidar_right):.2f} m)")
        seen = ", ".join(bits) if bits else "the current scan"
        return (f"the navigation policy chose {action_name(rec.final_v, rec.final_omega)} "
                f"from {seen}; no safety stage modified it")

    if d.name == "recovery":
        return d.detail or "recovery manoeuvre in progress"
    if d.name == "planner":
        p = rec.planner or {}
        e = p.get("bearing_error")
        if e is None:
            return "steering toward the exploration target"
        return (f"steering toward the exploration target: it is "
                f"{math.degrees(abs(e)):.0f} deg {'left' if e > 0 else 'right'} of the "
                f"nose, so a {abs(p.get('bias', 0.0)):.2f} rad/s bias was added to the "
                f"policy's turn")
    if d.name == "no_reverse":
        return (f"the policy asked to reverse at {rec.v_raw:+.2f} m/s, below this "
                f"drivetrain's {getattr(cfg, '_floor_mps', float('nan')):.2f} m/s floor, so it "
                f"would execute as a stop; turning toward the more open side instead")
    if d.name == "escape":
        return (f"forward has been blocked for {getattr(rec, '_stuck_ticks', '?')} ticks with "
                f"{_m(rec.lidar_min)} clearance, and this drivetrain cannot reverse, so "
                f"pivoting decisively toward open space")
    if d.name == "governor":
        return (f"forward clearance {_m(rec.lidar_min)} is inside the "
                f"{cfg.safe_distance:.2f} m braking bubble, so speed was cut to "
                f"{d.v:+.2f} m/s")
    if d.name == "tof":
        if rec.gate == "REFLEX_STOP":
            return (f"the bumper ToF reports {_m(rec.tof_min)}, inside the "
                    f"{cfg.tof_stop_distance:.2f} m stop distance -- emergency stop")
        return (f"the bumper ToF reports {_m(rec.tof_min)}, inside the "
                f"{cfg.tof_slow_distance:.2f} m slow distance, so speed was capped")
    if d.name == "visual":
        return (f"the floor camera hazard score is {rec.visual_score:.2f} against a "
                f"{cfg.visual_hazard_threshold:.2f} threshold")
    if d.name == "limiter":
        return (f"the command was slew-limited to stay physically achievable "
                f"({d.v:+.2f} m/s, {d.omega:+.2f} rad/s)")
    return f"{d.name} changed the command"


# ======================================================================================
# Capability checking
# ======================================================================================
def capability_check(v: float, omega: float, floor_mps: float, v_min: float,
                     v_max: float, omega_max: float) -> dict[str, Any]:
    """Can this drivetrain actually execute the action the policy just asked for?

    Evaluated on EVERY tick, not only when it fails, so the telemetry can say "checked,
    executable" rather than leaving the reader to infer it from the absence of a
    complaint. Absence of evidence and evidence of absence look identical in a log.

    The finding this exists to make unmissable: on this robot `v_min` is -0.15 m/s and
    the stiction floor is 0.179 m/s, so EVERY reverse the policy is capable of asking
    for is below the slowest speed the wheels can turn. Not most -- all of them. The
    trained action space and the hardware do not overlap on that axis at all, and the
    honest fix is retraining with the measured floor, not patching the runtime.

    Returns a record, never a decision. The caller decides what to do with it, and what
    it did is recorded separately.
    """
    reasons = []
    executable = True

    if v < 0.0:
        if floor_mps > 0.0 and abs(v) < floor_mps:
            executable = False
            reasons.append(
                f"reverse {v:+.3f} m/s is below the {floor_mps:.3f} m/s stiction floor: "
                f"the wheels cannot turn this slowly and it would execute as a full stop")
        if abs(v_min) < floor_mps:
            reasons.append(
                f"NOTE: v_min {v_min:+.3f} m/s is itself below the floor, so NO reverse "
                f"in the trained action space is executable on this drivetrain")
    elif 0.0 < v < (floor_mps or 0.0):
        # Not "inexecutable" -- creep pulsing handles it -- but worth recording,
        # because it is why executed != commanded on a healthy forward tick.
        reasons.append(
            f"forward {v:.3f} m/s is below the {floor_mps:.3f} m/s floor and will be "
            f"creep-pulsed to average out")

    return {
        "requested": {"v": v, "omega": omega},
        "executable": executable,
        "floor_mps": floor_mps,
        "action_space": {"v_min": v_min, "v_max": v_max, "omega_max": omega_max},
        "findings": reasons,
    }


# ======================================================================================
# The autonomy state interface
# ======================================================================================
# REASON CODES. The gate string was already carrying this information; naming the set
# makes it an interface rather than an implementation detail that happens to be
# printed. Each maps to a condition that is MEASURED, never inferred.
REASON_CODES = {
    "ok": "policy command executed as given",
    "planner": "heading biased toward the exploration target",
    "governor": "speed reduced for LiDAR clearance",
    "tof_slow": "speed capped by the bumper ToF",
    "visual_slow": "speed capped by the floor camera",
    "visual_stop": "stopped by a sustained floor hazard",
    "REFLEX_STOP": "emergency stop: obstacle inside the ToF stop distance",
    "no_reverse_turn": "reverse is unexecutable on this drivetrain; turned instead",
    "escape_turn": "boxed in with no reverse available; pivoted to open space",
    "recovery": "no progress detected; running a recovery manoeuvre",
    "stale_lidar": "LiDAR went stale; stopped rather than assume clear",
    "stale_tof": "ToF went stale; stopped rather than assume clear",
}


def autonomy_state(rec: DecisionRecord, cfg: Any, mode: str,
                   planner: dict | None = None,
                   recovery: dict | None = None,
                   objective: str = "",
                   stats: dict[str, int] | None = None) -> dict[str, Any]:
    """The complete autonomy state, as data rather than as a printed page.

    Every value here is measured or explicitly absent (None). Nothing is estimated to
    fill a field, and nothing is named that the code cannot establish -- which is why
    there is no "objects detected" and no "room" key: neither is implemented, and a
    plausible-looking guess in a state interface is worse than a gap, because whatever
    reads it cannot tell the difference.

    This is the machine-readable twin of `autonomy_status()`. The rendered block is for
    a person watching a run; this is for anything else -- the dashboard, a log, a voice
    layer deciding whether an event is worth speaking.
    """
    return {
        "CURRENT_MODE": mode,
        "TICK": rec.tick,
        "SENSOR_STATE": {
            "lidar": {"front": rec.lidar_front, "left": rec.lidar_left,
                      "right": rec.lidar_right, "rear": rec.lidar_rear,
                      "min": rec.lidar_min, "age_s": rec.lidar_age},
            "tof": {"nearest": rec.tof_min, "age_s": rec.tof_age},
            "cameras": rec.cam_state,
            "floor_hazard": {"score": rec.visual_score, "gate_active": rec.visual_enabled},
            "up_context": rec.context,
        },
        "MAP_STATE": rec.world,
        "EXPLORATION_TARGET": planner,
        "RAW_RL_POLICY_OUTPUT": (None if rec.v_raw is None
                                 else {"v": rec.v_raw, "omega": rec.omega_raw,
                                       "inference_ms": rec.infer_ms,
                                       "model": rec.model_name}),
        "PLANNER_CONTRIBUTION": rec.planner,
        "CAPABILITY_CHECK": rec.capability,
        "SAFETY_STATUS": rec.provenance()["SAFETY_DECISION"],
        "RECOVERY_STATUS": recovery,
        "ACTION_SOURCE": rec.action_source,
        "FINAL_EXECUTED_ACTION": rec.provenance()["FINAL_EXECUTED_ACTION"],
        "REASON_CODE": rec.gate,
        "REASON": reason(rec, cfg),
        "NEXT_OBJECTIVE": objective,
        "RUN_TOTALS": dict(stats or {}),
    }


def autonomy_status(rec: DecisionRecord, cfg: Any, mode: str,
                    stats: dict[str, int] | None = None,
                    objective: str = "") -> str:
    """The LIVE AUTONOMY STATUS block. Absent measurements are named, never defaulted."""
    L = []
    add = L.append
    add("=" * 70)
    add("FRIDAY AUTONOMOUS STATUS".center(70))
    add("=" * 70)
    add(f"MODE:            {mode}      tick {rec.tick}"
        + (f"   loop {rec.loop_ms:.1f} ms" if rec.loop_ms is not None else ""))
    add("")
    add(f"RL POLICY:       loaded {'YES' if rec.model_loaded else 'NO'}   "
        f"inference {'ACTIVE' if rec.infer_ms is not None else 'NOT RUNNING'}"
        + (f"   {rec.infer_ms:.3f} ms" if rec.infer_ms is not None else ""))
    add("")
    add("LIDAR:")
    add(f"  front {_m(rec.lidar_front)}   left {_m(rec.lidar_left)}   "
        f"right {_m(rec.lidar_right)}   rear {_m(rec.lidar_rear)}")
    add(f"  closest return {_m(rec.lidar_min)}"
        + (f"   (age {rec.lidar_age * 1000:.0f} ms)" if rec.lidar_age is not None else ""))
    add("")
    add("TOF / DEPTH:")
    add(f"  nearest obstacle {_m(rec.tof_min)}"
        + (f"   (age {rec.tof_age * 1000:.0f} ms)" if rec.tof_age is not None else ""))
    add("")
    add("CAMERAS:")
    if rec.cam_state:
        for role in ("up", "down"):
            st = rec.cam_state.get(role)
            if st is None:
                continue
            where = st.get("device") or "NOT ON THE BUS"
            add(f"  {role:<5} {where:<14} frames {st.get('updates', 0):<6}"
                f" reconnects {st.get('reconnects', 0)}")
    elif rec.cams_reporting == 0:
        add("  NO FRAMES -- neither camera contributed to this tick")
    add(f"  floor hazard (DOWN cam only): {_m(rec.visual_score, '', 2)}   "
        f"gate {'ACTIVE' if rec.visual_enabled else 'DISABLED (uncalibrated -- advisory)'}")
    if rec.context:
        c = rec.context
        # Measured statistics, not perception. Nothing here says what is in view.
        add(f"  up-cam context: brightness {_m(c.get('brightness'), '', 0)}   "
            f"change {_m(c.get('change'), '', 1)}   "
            f"occluded {_m(c.get('occluded'), '', 2)}"
            f"   (measured only -- no object detection)")
    add("")
    add("MAP:")
    if rec.explored_m2 is None:
        add("  not tracked")
    else:
        add(f"  observation map: {rec.explored_m2:.1f} m^2 swept this run   "
            f"frontier {'present' if rec.frontier_live else 'none in range'}")
    if rec.world is None:
        add("  persistent map: DISABLED")
    else:
        w = rec.world
        add(f"  persistent map: {w['explored_m2']:6.1f} m^2 known "
            f"({w['free_m2']:.1f} free)   {w['unknown']} cells unknown")
        add(f"                  {w['visited_m2']:6.1f} m^2 physically driven   "
            f"{w['sweeps']} sweeps this run / {w.get('sweeps_total', 0)} ever"
            + (f"  ({w['dropped']} superseded)" if w.get("dropped") else ""))
        # Revisit ratio is the number a coverage figure cannot show: a robot driving
        # the same 2 m of corridor forever reports a healthy constant visited_m2.
        rr = w.get("revisit_ratio", 0.0)
        add(f"                  {w.get('revisited_m2', 0.0):6.1f} m^2 re-entered   "
            f"revisit ratio {rr:.2f}"
            + ("   <- re-treading ground" if rr > 0.5 else ""))
        add(f"                  {'restored from disk' if rec.world_loaded else 'fresh this run'}"
            f"   bounds {w['bounds'][0]:.1f},{w['bounds'][1]:.1f} .. "
            f"{w['bounds'][2]:.1f},{w['bounds'][3]:.1f} m")
    if rec.pose is not None:
        add(f"  dead-reckoned pose  x={rec.pose[0]:.2f} y={rec.pose[1]:.2f} "
            f"theta={np.degrees(rec.pose[2]):+.0f} deg")
    add("")
    if rec.v_raw is not None:
        add(f"POLICY OUTPUT:   v={rec.v_raw:+.4f} m/s   omega={rec.omega_raw:+.4f} rad/s")
    else:
        add("POLICY OUTPUT:   none this tick")
    add(f"SELECTED ACTION: {action_name(rec.final_v, rec.final_omega)}")
    add(f"DECIDED BY:      {rec.authority}")
    if rec.planner is None:
        add("EXPLORATION:     planner disabled")
    else:
        e = rec.planner.get("bearing_error")
        if e is None:
            add(f"EXPLORATION:     no target -- {rec.planner.get('reason', '')}")
        else:
            add(f"EXPLORATION:     target {math.degrees(abs(e)):.0f} deg "
                f"{'left' if e > 0 else 'right'}   bias "
                f"{rec.planner.get('bias', 0.0):+.2f} rad/s"
                f"   ({rec.planner.get('reason', '')})")
    over = [s.name for s in rec.stages if s.changed and s.name in SAFETY_STAGES]
    add(f"SAFETY STATUS:   {'OVERRIDE -- ' + ', '.join(over) if over else 'SAFE (no override)'}")
    add(f"ACTION SOURCE:   {rec.action_source}")
    add(f"REASON CODE:     {rec.gate}")
    add(f"MOTOR COMMAND:   {rec.wire or '(nothing sent)'}")
    add("")
    add(f"REASON:          {reason(rec, cfg)}")
    if objective:
        add(f"NEXT OBJECTIVE:  {objective}")
    if stats:
        add("")
        add(f"RUN TOTALS:      " + "  ".join(f"{k}={v}" for k, v in stats.items() if v))
    add("=" * 70)
    return "\n".join(L)
