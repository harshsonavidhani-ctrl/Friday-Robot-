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
    "governor",         # LiDAR braking; may only reduce
    "escape",           # control adaptation: boxed in -> decisive pivot
    "visual",           # camera hazard; advisory, may only reduce
    "tof",              # ToF reflex; hard veto
    "limiter",          # slew limit; may only reduce rate of change
)

# Which stages are the ROBOT'S OWN DECISION versus a hand-written rule. This is the
# distinction the whole module exists to make legible, so it is data, not prose.
POLICY_STAGES = frozenset({"policy"})
# The exploration planner is its own authority, deliberately not lumped in with the
# hand-written rules. It is a DECISION about where to go, made from the persistent map
# -- not a hardware workaround like `no_reverse`, and not a safety veto. Collapsing the
# three into "not the policy" is what made the first run's log unreadable.
PLANNER_STAGES = frozenset({"planner"})
RULE_STAGES = frozenset({"no_reverse", "escape"})
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
            return "POLICY"
        if self.name in PLANNER_STAGES:
            return "PLANNER"
        if self.name in RULE_STAGES:
            return "RULE"
        if self.name in SMOOTHING_STAGES:
            return "SMOOTHING"
        return "SAFETY"


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
        """POLICY / RULE / SAFETY -- who chose what the wheels are doing this tick."""
        s = self.deciding_stage
        return s.kind if s is not None else "NONE"

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
    rules = [s for s in rec.stages if s.changed and s.name in RULE_STAGES]
    if rules:
        add(f"Rule Substitution:  YES -- {', '.join(s.name for s in rules)} "
            f"(hand-written, NOT the policy)")
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
    if rec.cams_reporting == 0:
        add("  NO FRAMES -- neither camera contributed to this tick")
    else:
        add(f"  {rec.cams_reporting} reporting   hazard score "
            f"{_m(rec.visual_score, '', 2)}   "
            f"gate {'ACTIVE' if rec.visual_enabled else 'DISABLED (uncalibrated -- advisory only)'}")
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
