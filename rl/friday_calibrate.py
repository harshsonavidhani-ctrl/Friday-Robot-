"""Guided hardware bring-up and calibration for the LEROBOT_EXPERIMENTAL policy.

Run this ON THE PI, before the first autonomous run. It walks the checks in dependency
order -- each one's result is only meaningful if the previous ones passed -- and writes
what it measures to `friday_calibration.json`, which the deploy script then loads.

    STAGE 1  imports and model            no hardware needed
    STAGE 2  observation contract         no hardware needed
    STAGE 3  LiDAR health and alignment   LiDAR only, wheels dead
    STAGE 4  ToF health                   ToF only, wheels dead
    STAGE 5  cameras (advisory)           cameras only, wheels dead
    STAGE 6  timing under real sensors    all sensors, wheels dead
    STAGE 7  motor stiction floor         WHEELS TURN -- robot on a stand
    STAGE 8  wheel speed calibration      WHEELS TURN -- robot on the floor

Stages 7 and 8 require `--i-am-watching` and are skipped by default, because they spin
the wheels. Everything before them is safe to run at any time.

WHY STAGE 8 MATTERS MOST
------------------------
`RobotConfig.wheel_speed_max` is flagged UNVERIFIED in HANDOFF.md (open blocker B3),
and the entire training simulation is calibrated against it -- the stiction model, the
speed randomisation band, the step budget. Everything the policy learned about how
fast it can go rests on a number nobody has measured. Stage 8 measures it in about
thirty seconds and replaces the guess with a fact.

Usage:
    python3 friday_calibrate.py                          # stages 1-6, safe
    python3 friday_calibrate.py --motor friday --i-am-watching --stiction
    python3 friday_calibrate.py --motor friday --i-am-watching --speed 120
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("friday.calibrate")
HERE = Path(__file__).resolve().parent
OUT = HERE / "friday_calibration.json"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.measured: dict = {}

    def add(self, stage: str, status: str, detail: str = "") -> None:
        self.rows.append((stage, status, detail))
        mark = {PASS: "[PASS]", FAIL: "[FAIL]", WARN: "[WARN]", SKIP: "[SKIP]"}[status]
        print(f"  {mark} {stage}" + (f"  --  {detail}" if detail else ""))

    def failed(self) -> bool:
        return any(s == FAIL for _n, s, _d in self.rows)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({
            "measured": self.measured,
            "checks": [{"stage": n, "status": s, "detail": d}
                       for n, s, d in self.rows],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        print(f"\ncalibration -> {path}")


# ======================================================================================
def stage1_model(rep: Report, model_path: Path, threads: int):
    print("\nSTAGE 1  imports and model")
    try:
        import onnxruntime as ort
    except ImportError:
        rep.add("onnxruntime installed", FAIL, "pip3 install onnxruntime")
        return None
    rep.add("onnxruntime installed", PASS, f"version {ort.__version__}")

    if not model_path.exists():
        rep.add("model file present", FAIL, f"not found: {model_path}")
        return None
    rep.add("model file present", PASS, f"{model_path.stat().st_size / 1024:.0f} KB")

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    try:
        sess = ort.InferenceSession(str(model_path), so,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:
        rep.add("model loads", FAIL, str(exc)[:120])
        return None

    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    rep.add("model loads", PASS, f"in {inp.shape} -> out {out.shape}")
    rep.measured["obs_dim"] = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else None

    # Latency on THIS machine, which is the Pi when run here.
    obs = np.zeros((1, int(inp.shape[-1])), np.float32)
    for _ in range(5):
        sess.run(None, {inp.name: obs})
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        sess.run(None, {inp.name: obs})
    ms = (time.perf_counter() - t0) / n * 1000.0
    rep.measured["inference_ms"] = round(ms, 2)
    budget = 100.0
    rep.add("inference latency", PASS if ms < budget * 0.5 else WARN,
            f"{ms:.1f} ms per forward pass (budget {budget:.0f} ms/tick)")
    return sess


def stage2_contract(rep: Report, sess, cfg):
    print("\nSTAGE 2  observation contract")
    try:
        from friday_localnav_deploy import LocalNavSpec, verify_against_robot_config
    except ImportError as exc:
        rep.add("deploy module imports", FAIL, str(exc)[:120])
        return
    rep.add("deploy module imports", PASS)

    try:
        verify_against_robot_config(cfg)
        rep.add("RobotConfig matches training spec", PASS)
    except ValueError as exc:
        rep.add("RobotConfig matches training spec", FAIL, str(exc)[:200])
        return

    if sess is not None:
        declared = sess.get_inputs()[0].shape[-1]
        if isinstance(declared, int) and declared != LocalNavSpec.OBS_DIM:
            rep.add("model width matches adapter", FAIL,
                    f"model wants {declared}, adapter builds {LocalNavSpec.OBS_DIM}")
        else:
            rep.add("model width matches adapter", PASS,
                    f"{LocalNavSpec.OBS_DIM} channels")


def stage3_lidar(rep: Report, slots, cfg, seconds: float):
    print(f"\nSTAGE 3  LiDAR health and alignment ({seconds:.0f} s)")
    t_end = time.monotonic() + seconds
    scans, ages = [], []
    while time.monotonic() < t_end:
        s, age = slots["lidar"].read()
        if s is not None:
            scans.append(s.copy())
            ages.append(age)
        time.sleep(0.05)

    if not scans:
        rep.add("LiDAR publishing", FAIL,
                "no scans at all -- see HANDOFF.md blocker B1")
        return
    arr = np.array(scans)
    rep.add("LiDAR publishing", PASS, f"{len(scans)} scans, {arr.shape[1]} beams")

    if arr.shape[1] != cfg.n_beams:
        rep.add("beam count", FAIL,
                f"got {arr.shape[1]}, policy needs {cfg.n_beams}")
    else:
        rep.add("beam count", PASS, str(cfg.n_beams))

    stale = float(np.mean(np.array(ages) > cfg.lidar_deadline))
    rep.add("LiDAR freshness", PASS if stale < 0.02 else FAIL,
            f"{stale:.1%} of reads older than {cfg.lidar_deadline:.2f} s")

    # Dead beams: a beam pinned at max range for the whole window is not measuring.
    pinned = float(np.mean(np.all(arr >= cfg.lidar_max_range - 1e-3, axis=0)))
    rep.add("no dead beams", PASS if pinned < 0.10 else WARN,
            f"{pinned:.1%} of beams never returned a hit")

    rep.measured["lidar_mean_min_range"] = round(float(np.mean(arr.min(axis=1))), 3)
    print("\n    Alignment check -- beam 0 must point along the robot's nose.")
    print("    Place a box ~1 m DIRECTLY IN FRONT and confirm the minimum sits near")
    print("    index 0. If it sits elsewhere, pass --lidar-offset-deg to pi_deploy.")
    last = arr[-1]
    k = int(np.argmin(last))
    deg = k / len(last) * 360.0
    rep.measured["nearest_beam_index"] = k
    rep.measured["nearest_beam_deg"] = round(deg, 1)
    rep.add("alignment (informational)", WARN,
            f"nearest return at beam {k}/{len(last)} = {deg:.0f} deg, "
            f"{float(last[k]):.2f} m -- verify this points where you put the box")


def stage4_tof(rep: Report, slots, cfg, seconds: float):
    print(f"\nSTAGE 4  ToF health ({seconds:.0f} s)")
    t_end = time.monotonic() + seconds
    frames = []
    while time.monotonic() < t_end:
        d, age = slots["tof"].read()
        if d is not None and age <= cfg.tof_deadline:
            frames.append(np.asarray(d).copy())
        time.sleep(0.05)

    if not frames:
        rep.add("ToF publishing", FAIL,
                "no frames -- I2C off, or wrong library (HANDOFF.md blocker B2). "
                "The ToF reflex is the hard safety veto; do not drive without it.")
        return
    arr = np.array(frames)
    rep.add("ToF publishing", PASS, f"{len(frames)} frames of {arr.shape[1:]} ")
    rep.measured["tof_rate_hz"] = round(len(frames) / seconds, 1)
    rep.add("ToF rate", PASS if len(frames) / seconds > 8 else WARN,
            f"{len(frames) / seconds:.1f} Hz (device caps ~15 Hz in 8x8)")
    rep.measured["tof_median_min_m"] = round(float(np.median(arr.min(axis=(1, 2)))), 3)


def stage5_cameras(rep: Report, slots, cfg, seconds: float):
    print(f"\nSTAGE 5  cameras (advisory layer)")
    got = {}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for key in ("cam0", "cam1"):
            if key in slots:
                f, age = slots[key].read()
                if f is not None and age <= cfg.cam_deadline:
                    got[key] = got.get(key, 0) + 1
        time.sleep(0.05)
    for key in ("cam0", "cam1"):
        if got.get(key):
            rep.add(f"{key} publishing", PASS, f"{got[key]} frames")
        else:
            rep.add(f"{key} publishing", WARN,
                    "no frames -- the visual gate fails OPEN, so this degrades "
                    "gracefully. Find real indices with v4l2-ctl --list-devices.")


def stage6_timing(rep: Report, slots, cfg, sess, seconds: float):
    print(f"\nSTAGE 6  end-to-end tick timing ({seconds:.0f} s, wheels dead)")
    if sess is None:
        rep.add("tick timing", SKIP, "no model session")
        return
    inp = sess.get_inputs()[0]
    obs = np.zeros((1, int(inp.shape[-1])), np.float32)

    times = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        scan, _age = slots["lidar"].read()
        if scan is not None:
            obs[0, : min(len(scan), obs.shape[1])] = \
                np.clip(scan[: obs.shape[1]] / cfg.lidar_max_range, 0.0, 1.0)
        sess.run(None, {inp.name: obs})
        times.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(max(0.0, cfg.dt - (time.perf_counter() - t0)))

    t = np.array(times)
    rep.measured["tick_ms_mean"] = round(float(t.mean()), 2)
    rep.measured["tick_ms_p95"] = round(float(np.percentile(t, 95)), 2)
    rep.measured["tick_ms_max"] = round(float(t.max()), 2)
    budget = cfg.dt * 1000.0
    rep.add("tick within budget", PASS if t.max() < budget else FAIL,
            f"mean {t.mean():.1f} | p95 {np.percentile(t, 95):.1f} | "
            f"max {t.max():.1f} ms (budget {budget:.0f})")


def stage7_stiction(rep: Report, cfg, args):
    print("\nSTAGE 7  motor stiction floor  --  WHEELS WILL TURN")
    import pi_deploy
    drv = pi_deploy.FridayMotorDriver(
        cfg, pi_deploy.resolve_serial_port("motor", args.motor_port),
        max_duty=args.motor_max_duty, watchdog_s=args.motor_watchdog,
        invert_left=args.motor_invert_left, invert_right=args.motor_invert_right,
        min_duty=0, creep_cutoff=0.0, turn_min_duty=None)
    try:
        pi_deploy.motor_duty_sweep(cfg, drv)
        rep.add("stiction sweep", PASS,
                "note the lowest duty giving STEADY rotation -> --motor-min-duty")
    finally:
        drv.emergency_stop()
        drv.close() if hasattr(drv, "close") else None


def stage8_speed(rep: Report, cfg, args):
    print("\nSTAGE 8  wheel speed calibration  --  WHEELS WILL TURN, ON THE FLOOR")
    print("    This measures the number the whole training simulation rests on.")
    import pi_deploy
    drv = pi_deploy.FridayMotorDriver(
        cfg, pi_deploy.resolve_serial_port("motor", args.motor_port),
        max_duty=args.motor_max_duty, watchdog_s=args.motor_watchdog,
        invert_left=args.motor_invert_left, invert_right=args.motor_invert_right,
        min_duty=0, creep_cutoff=0.0, turn_min_duty=None)
    try:
        pi_deploy.motor_speed_calibration(cfg, drv, args.speed, args.calibrate_seconds)
        print("\n    Measure the distance travelled, then compute:")
        print(f"        speed_at_duty_{args.speed} = distance_m / {args.calibrate_seconds:.1f}")
        print("        breakaway_frac = speed_at_breakaway_duty / speed_at_duty_255")
        print("    Put breakaway_frac into friday_localnav/config.py DrivetrainSpec")
        print("    and RETRAIN. Until then the drivetrain model is an estimate.")
        rep.add("speed calibration run", PASS, "record the distance by hand")
    finally:
        drv.emergency_stop()


# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Friday hardware bring-up + calibration")
    ap.add_argument("--model", type=str, default=str(HERE / "friday_localnav_act.onnx"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic sensors, for checking this script off-robot")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--no-tof", action="store_true")
    ap.add_argument("--stiction", action="store_true", help="STAGE 7: wheels turn")
    ap.add_argument("--speed", type=int, default=None, help="STAGE 8: duty to drive at")
    ap.add_argument("--calibrate-seconds", type=float, default=3.0)
    ap.add_argument("--motor", choices=["null", "friday"], default="null")
    ap.add_argument("--motor-port", type=str, default="auto")
    # 0.75 == friday/config.py MOTOR_MAX_DUTY == pi_deploy's default: the approved
    # ceiling for this L298N. This used to default to 0.5, which is not a neutral
    # choice -- 0.5 x 255 = 127, and the MEASURED pivot break-away on this robot is
    # 190. A stiction sweep left on the old default therefore stopped ramping before
    # the duty at which the wheels actually turn, and reported a floor that does not
    # exist. Three files disagreeing about one hardware ceiling is how a throttle
    # becomes invisible. Raising it ABOVE 0.75 still needs Harsh's explicit approval.
    ap.add_argument("--motor-max-duty", type=float, default=0.75)
    ap.add_argument("--motor-watchdog", type=float, default=0.5)
    ap.add_argument("--motor-invert-left", action="store_true")
    ap.add_argument("--motor-invert-right", action="store_true")
    ap.add_argument("--i-am-watching", action="store_true")
    ap.add_argument("--lidar-port", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    print("=" * 78)
    print("Friday bring-up and calibration -- LEROBOT_EXPERIMENTAL")
    print("=" * 78)

    rep = Report()
    sess = stage1_model(rep, Path(args.model), args.threads)

    try:
        import pi_deploy
    except ImportError:
        rep.add("pi_deploy importable", FAIL, "run this next to pi_deploy.py")
        rep.write(OUT)
        return 2

    cfg = pi_deploy.RobotConfig()
    stage2_contract(rep, sess, cfg)

    # build_sensors() reads pi_deploy's full CLI namespace, which is wider than
    # this script's -- it wants --lidar-baud, --tof-port and friends that bring-up
    # has no reason to expose. Seed every field from pi_deploy's own parser
    # defaults, then overlay the options this script does expose, so adding a flag
    # there can never AttributeError here again.
    sensor_args = pi_deploy.build_arg_parser().parse_args([])
    for _k, _v in vars(args).items():
        setattr(sensor_args, _k, _v)

    slots, threads, _sim = pi_deploy.build_sensors(cfg, sensor_args)
    for t in threads:
        t.start()
    time.sleep(1.5)                      # let the sensor threads reach steady state

    try:
        stage3_lidar(rep, slots, cfg, args.seconds)
        if args.no_tof:
            rep.add("ToF", SKIP, "--no-tof: the hard safety veto is DISABLED")
        else:
            stage4_tof(rep, slots, cfg, args.seconds)
        if args.no_cameras:
            rep.add("cameras", SKIP, "--no-cameras")
        else:
            stage5_cameras(rep, slots, cfg, min(args.seconds, 4.0))
        stage6_timing(rep, slots, cfg, sess, args.seconds)

        wheels = args.stiction or args.speed is not None
        if wheels and not (args.motor == "friday" and args.i_am_watching):
            rep.add("motor stages", SKIP,
                    "need --motor friday --i-am-watching, robot on a stand")
        elif args.stiction:
            stage7_stiction(rep, cfg, args)
        if args.speed is not None and args.motor == "friday" and args.i_am_watching:
            stage8_speed(rep, cfg, args)
        elif args.speed is None:
            rep.add("STAGE 8 wheel speed (blocker B3)", SKIP,
                    "run with --speed 120 --motor friday --i-am-watching. "
                    "Until measured, the drivetrain model is an estimate.")
    finally:
        for t in threads:
            if hasattr(t, "stop"):
                t.stop()

    print("\n" + "=" * 78)
    n_pass = sum(1 for _n, s, _d in rep.rows if s == PASS)
    n_fail = sum(1 for _n, s, _d in rep.rows if s == FAIL)
    n_warn = sum(1 for _n, s, _d in rep.rows if s == WARN)
    print(f"{n_pass} passed, {n_warn} warnings, {n_fail} failed")
    print("=" * 78)
    rep.write(OUT)

    if rep.failed():
        print("\nDO NOT run autonomously until the failures above are resolved.")
        return 1
    print("\nSafe to proceed to:  python3 friday_localnav_deploy.py --simulate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
