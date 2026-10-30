"""POLICY_SOURCE = LEROBOT_EXPERIMENTAL -- Pi-side adapter for the ACT local-nav policy.

Copy this file and the exported .onnx next to `pi_deploy.py` on the Raspberry Pi.
It does NOT modify `pi_deploy.py`; it imports from it and subclasses `ControlLoop`,
overriding only the two stages that differ:

    _build_observation()   77-dim goal-conditioned vector instead of the 146-dim
                           LiDAR + frontier-ray vector the old PPO policy used
    self.policy.infer()    ACT chunk queue instead of a per-tick forward pass

Everything downstream is inherited untouched:

    policy -> LidarGovernor -> VisualGate -> ToFReflex -> RateLimiter -> motors
                                                 |
                       Arduino firmware watchdog (independent, in hardware)

The learned policy therefore never has unrestricted motor control. It proposes a
(v, omega); four independent stages may only ever REDUCE that authority, and the
firmware watchdog stops the motors regardless if this process dies.

Architecture, as specified:

    PERCEIVE                     <- pi_deploy sensor threads (LiDAR / ToF / cameras)
              |
    UNDERSTAND + MAP / REMEMBER  <- LocalExplorationMap (occupancy + known-space)
              |
    SELECT EXPLORATION TARGET    <- ExplorationSupervisor (classical, NOT learned):
              |                     frontier detection, attempt memory, progress
              |                     monitoring, recovery
    TRAINED LEARNED NAVIGATION POLICY  <- ACT, this file's LeRobotLocalNavPolicy
              |
    SAFETY SUPERVISOR            <- inherited from pi_deploy, unchanged
              |
    MOTOR CONTROLLER
              |
    OBSERVE RESULT -> CONTINUE   <- fed back into the supervisor each tick, which is
                                    what closes the exploration loop and what makes
                                    this exploration rather than obstacle avoidance

Voice feedback runs beside all of this on its own thread (`friday_voice.py`), driven
purely by supervisor state transitions and real sensor readings. It can never delay a
control tick, and it never claims perception the robot does not have -- Friday has no
object recogniser, so it reports hazards and geometry, not object names.

Usage:
    python3 friday_localnav_deploy.py --simulate --duration 60
    python3 friday_localnav_deploy.py --model friday_localnav_act.onnx \\
        --motor friday --i-am-watching --motor-max-duty 0.5 --no-cameras --duration 30
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

# The working runtime. Imported, never edited.
import pi_deploy
from pi_deploy import ControlLoop, RobotConfig

from friday_exploration import ExplorationConfig, ExplorationSupervisor, Phase
from friday_voice import build_announcer

POLICY_SOURCE = "LEROBOT_EXPERIMENTAL"

LOGGER = logging.getLogger("friday.localnav")
HERE = Path(__file__).resolve().parent


# ======================================================================================
# 1. Training contract -- must match friday_localnav/config.py
# ======================================================================================
class LocalNavSpec:
    """The observation/action contract baked into the exported ONNX graph.

    These values are duplicated from `friday_localnav.config` rather than imported,
    because the Pi does not have that package (or LeRobot, or PyTorch) installed.
    `verify_against_robot_config` checks the subset that is mechanically checkable.
    """

    N_BEAMS = 72
    LIDAR_MAX_RANGE = 5.00
    GOAL_DIST_NORM = 4.00
    OBS_DIM = N_BEAMS + 5
    ACTION_DIM = 2
    CHUNK_SIZE = 16
    N_ACTION_STEPS = 8

    V_MAX = 0.50
    V_MIN = -0.15
    OMEGA_MAX = 1.60

    @classmethod
    def to_si(cls, a0: float, a1: float) -> tuple[float, float]:
        a0 = min(max(float(a0), -1.0), 1.0)
        a1 = min(max(float(a1), -1.0), 1.0)
        v = a0 * (cls.V_MAX - cls.V_MIN) / 2.0 + (cls.V_MAX + cls.V_MIN) / 2.0
        return v, a1 * cls.OMEGA_MAX


def verify_against_robot_config(cfg: RobotConfig) -> None:
    """Fail loudly on any mismatch between the robot config and the training spec.

    A mismatch here does not raise on its own -- it silently feeds the network
    out-of-distribution input, which HANDOFF.md correctly identifies as the easiest
    way to put this robot into a wall. So it is checked explicitly, at startup,
    before anything can move.
    """
    problems = []
    for name, got, want in (
        ("n_beams", cfg.n_beams, LocalNavSpec.N_BEAMS),
        ("lidar_max_range", cfg.lidar_max_range, LocalNavSpec.LIDAR_MAX_RANGE),
        ("v_max", cfg.v_max, LocalNavSpec.V_MAX),
        ("v_min", cfg.v_min, LocalNavSpec.V_MIN),
        ("omega_max", cfg.omega_max, LocalNavSpec.OMEGA_MAX),
    ):
        if abs(float(got) - float(want)) > 1e-9:
            problems.append(f"  {name}: RobotConfig={got}  training spec={want}")
    if problems:
        raise ValueError(
            "RobotConfig does not match the LEROBOT_EXPERIMENTAL training spec:\n"
            + "\n".join(problems)
            + "\nRefusing to run: the policy would receive out-of-distribution input."
        )
    LOGGER.info("Config verified against %s training spec.", POLICY_SOURCE)


# ======================================================================================
# 2. High-level exploration lives in `friday_exploration.ExplorationSupervisor`.
#
# It was a bare nearest-frontier selector in this file; it is now a full state machine
# (select -> navigate -> observe -> recover) with attempt memory and progress
# monitoring, and it lives in its own module so there is exactly ONE implementation of
# frontier detection rather than two copies drifting apart.
# ======================================================================================

# ======================================================================================
# 3. The learned local-navigation policy (ONNX + action-chunk queue)
# ======================================================================================
class LeRobotLocalNavPolicy:
    """ACT policy in ONNX Runtime, exposing `pi_deploy`'s `.infer(obs) -> (v, omega)`.

    ACT predicts CHUNK_SIZE future actions per forward pass and we execute
    N_ACTION_STEPS of them before re-planning. Two consequences, both wanted:

      * a chunk is a committed short plan, so the controller cannot dither between
        left and right on adjacent ticks -- this is the anti-oscillation property;
      * inference cost is amortised over N_ACTION_STEPS ticks, which is what makes a
        transformer affordable on a Cortex-A72 inside a 100 ms budget.

    Returns SI units, matching what `ControlLoop.tick` expects from `.infer`.
    """

    def __init__(self, model_path: Path, cfg: RobotConfig, threads: int = 2) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:                     # pragma: no cover
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime") from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads              # leave cores for sensor threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(model_path), so,
                                         providers=["CPUExecutionProvider"])

        inp = self.sess.get_inputs()[0]
        out = self.sess.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name

        declared = inp.shape[-1]
        if isinstance(declared, int) and declared != LocalNavSpec.OBS_DIM:
            raise ValueError(
                f"Observation width mismatch: model expects {declared}, this adapter "
                f"produces {LocalNavSpec.OBS_DIM}. Wrong .onnx file for this adapter?"
            )
        if len(out.shape) != 3 or (isinstance(out.shape[-1], int)
                                   and out.shape[-1] != LocalNavSpec.ACTION_DIM):
            raise ValueError(
                f"Expected an action-chunk output [B, T, 2], got {out.shape}. "
                f"This adapter requires a chunked ACT export."
            )

        self._buf = np.zeros((1, LocalNavSpec.OBS_DIM), np.float32)
        self._queue: list[np.ndarray] = []
        self.infer_calls = 0
        self.ticks = 0
        # pi_deploy's telemetry reads these two off whatever object is installed as
        # `.policy`. PolicyRunner defines them; this adapter must too, or tick()
        # AttributeErrors on the first pass through the black box.
        self.model_name = model_path.name
        self.last_infer_ms: float = 0.0
        LOGGER.info("%s policy loaded: %s (obs=%d, chunk=%d, exec=%d, threads=%d)",
                    POLICY_SOURCE, model_path.name, LocalNavSpec.OBS_DIM,
                    LocalNavSpec.CHUNK_SIZE, LocalNavSpec.N_ACTION_STEPS, threads)

    def reset(self) -> None:
        self._queue = []

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """77-dim observation -> np.array([v, omega]) in SI units."""
        self.ticks += 1
        if not self._queue:
            self._buf[0, :] = obs
            _t0 = time.perf_counter()
            chunk = self.sess.run([self.output_name],
                                  {self.input_name: self._buf})[0][0]
            self.last_infer_ms = (time.perf_counter() - _t0) * 1000.0
            n = min(LocalNavSpec.N_ACTION_STEPS, chunk.shape[0])
            self._queue = [chunk[i] for i in range(n)]
            self.infer_calls += 1
        else:
            # Served from the committed chunk: this tick ran no forward pass, so its
            # true inference cost is zero. Reporting it as such is what makes the
            # amortisation visible in the tick-budget telemetry rather than hiding it.
            self.last_infer_ms = 0.0
        a = self._queue.pop(0)
        v, omega = LocalNavSpec.to_si(a[0], a[1])
        return np.array([v, omega], np.float32)


# ======================================================================================
# 4. Control loop -- safety chain inherited unchanged
# ======================================================================================
class LeRobotExperimentalControlLoop(ControlLoop):
    """`pi_deploy.ControlLoop` with a goal-conditioned observation.

    Only `_build_observation` is overridden. `tick()` -- and with it the whole
    governor / visual / reflex / rate-limiter chain, the staleness fail-closed logic
    and the black box -- is inherited verbatim. That is the point: the experimental
    policy changes what is proposed, never what is permitted.
    """

    def __init__(self, *a, **kw) -> None:
        supervisor = kw.pop("supervisor", None)
        voice = kw.pop("voice", None)
        super().__init__(*a, **kw)
        self.voice = voice
        self.supervisor = supervisor or ExplorationSupervisor(self.cfg, voice=voice)
        self._obs_ln = np.zeros(LocalNavSpec.OBS_DIM, np.float32)
        self._last_row: dict | None = None
        self.stats.setdefault("no_target_ticks", 0)

    # ---- OBSERVE RESULT -> CONTINUE ----------------------------------------------
    def tick(self) -> dict:
        """Inherited tick, plus the feedback edge that closes the exploration loop.

        `super().tick()` is called unmodified, so the entire safety chain -- staleness
        fail-closed, governor, visual gate, ToF reflex, rate limiter -- runs exactly as
        it does for the production policy. We only READ the row it returns, so that the
        supervisor's next decision (and any announcement) reflects what the safety
        chain actually did, not what the policy asked for.
        """
        row = super().tick()
        self._last_row = row
        return row

    def _build_observation(self, scan: np.ndarray) -> np.ndarray:
        c = self.cfg

        # 1. MAP / REMEMBER. `update_and_frontier` integrates the scan into the
        #    occupancy and known-space grids, which is what gives the supervisor its
        #    memory of where Friday has already been. Its frontier-ray return value is
        #    unused here -- that channel belonged to the old 146-dim observation.
        self.map.update_and_frontier(scan)

        # 2. UNDERSTAND -> SELECT TARGET. Classical, not learned: frontier detection,
        #    attempt memory, progress monitoring, recovery, and the voice events that
        #    report all of it. Fed the previous tick's safety-chain result so it can
        #    react to what actually happened.
        target = self.supervisor.update(self.map, scan, self._last_row)

        # 3. Express it in the robot frame.
        pose = self.map.pose
        px, py, th = float(pose[0]), float(pose[1]), float(pose[2])
        if target is None:
            # No frontier available (fully explored, or the map is degenerate). Aim
            # at a point straight ahead so the policy keeps making forward progress
            # rather than receiving a garbage goal. The safety chain still governs
            # everything downstream.
            self.stats["no_target_ticks"] += 1
            gx, gy = px + 1.5 * math.cos(th), py + 1.5 * math.sin(th)
        else:
            gx, gy = float(target[0]), float(target[1])

        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - th
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))

        # 4. Assemble, in exactly the ObsSpec order used in training.
        n = LocalNavSpec.N_BEAMS
        b = self._obs_ln
        b[:n] = np.clip(scan / LocalNavSpec.LIDAR_MAX_RANGE, 0.0, 1.0)
        b[n + 0] = min(dist / LocalNavSpec.GOAL_DIST_NORM, 1.0)
        b[n + 1] = math.cos(bearing)
        b[n + 2] = math.sin(bearing)
        b[n + 3] = self.last_action[0]      # set by tick() from the POLICY output
        b[n + 4] = self.last_action[1]
        return b

    def report(self) -> None:
        super().report()
        LOGGER.info("policy source: %s", POLICY_SOURCE)
        self.supervisor.report()
        LOGGER.info("no_target_ticks=%d", self.stats.get("no_target_ticks", 0))
        if self.voice is not None:
            self.voice.report()
        pol = self.policy
        if isinstance(pol, LeRobotLocalNavPolicy) and pol.ticks:
            LOGGER.info("ACT chunks: %d forward passes over %d ticks (%.2f ticks/pass)",
                        pol.infer_calls, pol.ticks, pol.ticks / max(pol.infer_calls, 1))


# ======================================================================================
# 5. CLI
# ======================================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Reuses pi_deploy's parser so every existing flag keeps working."""
    ap = pi_deploy.build_arg_parser()
    ap.description = f"Friday -- {POLICY_SOURCE} local-navigation policy"
    ap.add_argument("--localnav-model", type=str,
                    default=str(HERE / "friday_localnav_act.onnx"),
                    help="ACT local-navigation ONNX model")
    ap.add_argument("--localnav-threads", type=int, default=2)
    ap.add_argument("--frontier-replan-s", type=float, default=3.0)

    ap.add_argument("--no-voice", action="store_true",
                    help="disable spoken feedback (events are still logged)")
    ap.add_argument("--voice-backend", type=str, default="auto",
                    choices=("auto", "espeak-ng", "espeak", "say", "pyttsx3", "log"))
    ap.add_argument("--voice-min-gap", type=float, default=1.2,
                    help="[s] minimum gap between two spoken phrases")
    ap.add_argument("--voice-listen-gap", type=float, default=15.0,
                    help="[s] after each phrase she stays quiet this long so the "
                         "wake word can hear Harsh (0 = off); see ALWAYS_SAY")
    ap.add_argument("--voice-rate", type=int, default=165, help="words per minute")
    return ap


def build_motors(cfg: RobotConfig, args) -> "pi_deploy.MotorDriver":
    """Construct the motor driver exactly as `pi_deploy.main` does.

    `pi_deploy` builds its driver inline inside `main()`, so there is no function to
    call. This mirrors that block rather than approximating it, because the arguments
    it passes are all safety-relevant: `max_duty` caps PWM, `watchdog_s` arms the
    firmware watchdog that stops the motors if this process dies, and `min_duty` /
    `creep_cutoff` / `turn_min_duty` implement the stiction handling that HANDOFF.md
    warns must not be bypassed.

    An earlier version of this file guarded with `hasattr(pi_deploy, "build_motors")`
    and fell back to `NullMotorDriver`. Since no such function exists, that fallback
    ALWAYS fired: `--motor friday` was accepted and silently ignored, and the robot
    would never have moved. Failing safe is right; failing safe while reporting
    success is not.
    """
    # ONE construction site, in pi_deploy. This file used to build the driver itself
    # and had already fallen behind main()'s version: turn_kick_duty,
    # turn_kick_cooldown_s and pulse_creep_duty were never passed, so the pivot
    # break-away kick and the creep pulse silently ran on defaults in every real
    # exploration run -- which all go through THIS file, not through main().
    return pi_deploy.build_motors(cfg, args)



def _flags_this_main_ignores(args, parser) -> tuple[list[str], bool]:
    """(flags set to a non-default that this module never names, args-handed-off).

    Deliberately source-text based: it asks whether this file mentions the flag AT ALL,
    which is exactly the failure mode being guarded against -- a flag the parser
    accepts and this entry point never reads. Anything subtler would need the wiring to
    be declared somewhere, and a declaration that can drift is the thing that drifted.

    Two forms count as a mention, because both are used here: `args.foo`, and the name
    as a bare string for `getattr(args, "foo", ...)`. Missing the second reported
    --blackbox-seconds as ignored when line 436 reads it.

    The second return value says whether `args` is ever handed to a callee whole (as
    `build_announcer(args)` is). When it is, this check CANNOT see inside, so the list
    is "worth checking" rather than "definitely dead" -- and it says so rather than
    asserting something it has not established.
    """
    try:
        src = Path(__file__).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], False
    handoff = re.search(r"\w\(\s*args\s*[,)]", src) is not None
    out = []
    for action in parser._actions:
        dest = action.dest
        if dest in ("help", "duration"):
            continue
        value = getattr(args, dest, None)
        if value == action.default or value is None:
            continue                       # left alone: silence about it is harmless
        if re.search(rf"\bargs\.{re.escape(dest)}\b", src):
            continue
        if re.search(rf"""['"]{re.escape(dest)}['"]""", src):
            continue                       # getattr(args, "dest", ...) and friends
        out.append("--" + dest.replace("_", "-"))
    return sorted(out), handoff

def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.info("POLICY_SOURCE = %s", POLICY_SOURCE)

    cfg = RobotConfig()
    verify_against_robot_config(cfg)

    # These four come from the CLI and RobotConfig's defaults are deliberately
    # neutral, so they are not optional -- pi_deploy.main() sets all four and this
    # main() is a parallel entry point that has to do the same. Omitting them was a
    # real, measured bug, and neither symptom looked like a missing assignment:
    #
    #   * self_mask_deg defaulted to () -- NO chassis mask. The robot's own body sat
    #     0.21 m from the LiDAR and, because the governor takes min() over every
    #     beam, it pinned the scan minimum on every tick. Forward speed was clamped
    #     to zero with 1.34 m of clear floor ahead, and it read as "the policy will
    #     not drive" rather than as a config gap. pi_deploy warns about precisely
    #     this -- the warning just lives in the main() we do not call.
    #   * lidar_offset_deg defaulted to 0.0, discarding the MEASURED -51.8 deg
    #     mounting rotation. Beam 0 then points 51.8 deg off the nose, so the whole
    #     72-beam observation handed to the policy is rotated. The policy is not
    #     wrong in that situation; it is being asked to navigate a rotated world.
    cfg.cam_enabled = not args.no_cameras
    cfg.tof_enabled = not args.no_tof
    cfg.self_mask_deg = pi_deploy.parse_self_mask(args.lidar_self_mask)
    cfg.lidar_offset_deg = args.lidar_offset_deg
    # Same class of gap as the two above, and it cost a run: these flags are parsed by
    # pi_deploy's parser but APPLIED in pi_deploy's main(), which this file does not
    # call -- so anything added there is silently inert here. A 180 s run asked for
    # --blackbox-seconds 200 and got the 30 s default, which is a 316-row ring buffer,
    # so 150 s of the run was overwritten before it was ever written out.
    cfg.blackbox_seconds = max(1.0, float(getattr(args, "blackbox_seconds", 30.0)))
    cfg.lidar_min_valid_range = max(0.0, float(getattr(args, "lidar_min_range", 0.0)))
    if cfg.self_mask_deg:
        LOGGER.info("LiDAR chassis self-mask: %s (raw bearings, pre-offset); "
                    "mounting offset %.1f deg",
                    ", ".join(f"{lo:g}-{hi:g}" for lo, hi in cfg.self_mask_deg),
                    cfg.lidar_offset_deg)
    else:
        LOGGER.warning("No LiDAR self-mask. If any beam sees the robot's own body it "
                       "will pin the scan minimum and forward motion stays forbidden.")
    if not cfg.tof_enabled:
        LOGGER.warning("ToF DISABLED -- emergency reflex braking is OFF. Bench use only.")

    slots, threads, sim = pi_deploy.build_sensors(cfg, args)
    for t in threads:
        t.start()

    policy = LeRobotLocalNavPolicy(Path(args.localnav_model), cfg,
                                   threads=args.localnav_threads)
    motors = build_motors(cfg, args)

    voice = build_announcer(args)
    supervisor = ExplorationSupervisor(
        cfg,
        ExplorationConfig(replan_period_s=args.frontier_replan_s),
        voice=voice,
    )

    loop = LeRobotExperimentalControlLoop(cfg, policy, motors, slots, sim=sim,
                                          supervisor=supervisor, voice=voice)

    # Same omission as the self-mask above: pi_deploy.main() wires this from the CLI
    # and this parallel main() did not, so --no-reverse-fallback was accepted and
    # silently ignored. It matters more than a normally-ignored flag would, because
    # the fallback is what turns every reverse request into a pivot: measured over one
    # 316-tick run, the policy asked to reverse 117 times and NOT ONE executed, while
    # the recovery controller produced 84 pivots and zero translation. A robot that
    # cannot back out of a doorway can only spin there, which is exactly what it did.
    loop.reverse_fallback = not args.no_reverse_fallback
    # ...and the same omission again for the local navigator, wired here AT THE SAME
    # TIME as the flag was added rather than discovered missing on hardware later. This
    # is the fourth flag in this list; the pattern is the bug, not any one of them.
    # `run_localnav.sh` drives THIS main(), so a flag that only pi_deploy.main() reads
    # is a flag that does nothing on the robot.
    loop.navigator_enabled = not args.no_navigator
    cfg.nav_speed_authority = float(getattr(args, "navigator_speed_authority", 0.0))
    # The ninth: the reverse floor measured on the floor (2026-10-10).
    cfg.reverse_floor_mps = max(0.0, float(getattr(args, "reverse_floor", 0.0)))
    # The tenth and eleventh (2026-10-20): the 09-04 speed, and the strict reverse.
    cfg.max_forward_speed = max(0.0, float(getattr(args, "max_forward_speed", 0.0)))
    cfg.strict_reverse = bool(getattr(args, "strict_reverse", False))
    cfg.bumper_escape = bool(getattr(args, "bumper_escape", False))
    cfg.live_map = bool(getattr(args, "live_map", False))
    if cfg.bumper_escape:
        import escape as _escape
        loop.bumper_escape = _escape.EscapeManeuver(cfg)
        LOGGER.info("Bumper escape ON: one committed back-then-turn at a ToF block")
    # The seventh and eighth: the way back (2026-09-26) and undoing the last move
    # (2026-09-28). Without these lines --no-way-back / --no-unstick did nothing on
    # the robot, because run_localnav.sh drives THIS main().
    loop.way_back_enabled = not getattr(args, "no_way_back", False)
    loop.unstick_enabled = not getattr(args, "no_unstick", False)
    loop.unstick.keep_trying = bool(getattr(args, "unstick_keep_trying", False))
    loop.turn_commit_s = max(0.0, float(getattr(args, "turn_commit", 0.0)))
    # --map-file and --state-file were the fifth and sixth. --map-file being ignored
    # meant a diagnostic run pointed at a scratch map OVERWROTE the real one.
    if getattr(args, "map_file", None):
        loop.world.map_file = Path(args.map_file)
    if getattr(args, "state_file", None):
        loop.state_file = (None if args.state_file.lower() == "none"
                           else Path(args.state_file))

    # STOP FINDING THESE ONE AT A TIME, ON HARDWARE. Six flags have now been accepted
    # by the parser and silently ignored by this main(), and each cost a run to notice.
    # The parser lives in pi_deploy; the wiring lives here; nothing connected them.
    #
    # So: read this module's own source for the `args.<name>` it actually mentions, and
    # warn about any flag the user set away from its default that does not appear. It
    # cannot catch a flag that is read and then dropped, but it catches the whole
    # not-wired-at-all class, which is the one that has actually happened -- and it
    # costs one regex over one file at startup.
    _unwired, _handoff = _flags_this_main_ignores(args, parser)
    if _unwired:
        LOGGER.warning("CHECK THESE FLAGS: %s. You set them away from their defaults, "
                       "and this entry point never names them%s. Six flags have already "
                       "been accepted by the parser and silently ignored here, each "
                       "costing a hardware run to notice.", ", ".join(_unwired),
                       " (though `args` is also passed to callees whole, which this "
                       "check cannot see inside)" if _handoff else "")
    voice.say("boot", "Friday is starting autonomous exploration.",
              priority=1, cooldown_s=0.0)
    try:
        loop.run(duration=getattr(args, "duration", None))
    finally:
        wb = getattr(loop, "way_back", None)
        if wb is not None and wb.phase == "done":
            pass        # she has just said where she ended up; nothing to add
        elif supervisor.st.phase == Phase.COMPLETE:
            voice.say("done", "Exploration complete. Shutting down.",
                      priority=2, cooldown_s=0.0)
        else:
            voice.say("halt", "Exploration stopped.", priority=2, cooldown_s=0.0)
        voice.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
