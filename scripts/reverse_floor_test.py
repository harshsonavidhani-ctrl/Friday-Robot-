#!/usr/bin/env python3
"""Bug #1 (2026-10-02 diagnosis): how fast must she reverse to actually move ON
THE FLOOR? Run WITH HARSH WATCHING.

    ./venv/bin/python scripts/reverse_floor_test.py --dry          # the plan, no motion
    ./venv/bin/python scripts/reverse_floor_test.py --i-am-watching

Why: every escape reverse of the RL run goes out at v_min = -0.15 m/s, chosen
because the friction floor measured 0.149 m/s -- ON A STAND. On the floor the
first real mission's 25 escape reverses moved her NOTHING (bumper ToF 8 cm ->
8 cm every time), and the 2026-09-29 pulses moved 0-0.6 cm at -0.15.

⚠ 2026-10-02: the run logs say the 09-29 floor measurement was taken WITH the
"quieter gears" drive (a58b9e3, kick only after a real standstill), and
reverses freed her 38-58% of the time before that change, ~20% after. So
--drive old (the default now, as the mode runs since Harsh restored it) and
--drive quiet measure the two side by side.

What it does, nothing else:
  1. With friday-mcp still up: checks there is something flat 15-60 cm in front
     (the ruler) and that the rear is clear (>= 0.6 m) -- and asks Harsh.
  2. Stops friday-mcp (Friday's own unit) so the motor and ToF ports are free.
  3. Builds the motors EXACTLY as a run does (rl/pi_deploy.build_motors with the
     parser's defaults, the same kick and floor) and the run's own ToF reader.
  4. For v = -0.15, -0.20, ... -0.45 m/s: ONE pulse of 1.0 s straight back, stop,
     settle, read the ToF to the ruler before and after. Stops early once two
     speeds in a row really move her (>= 3 cm), or at any rear/ToF trouble.
  5. Stops the motors and ALWAYS restarts friday-mcp (try/finally).
Prints a table and writes it to state/reverse_floor_test.json. Changes nothing
in rl/: the fix is a separate, reviewed edit once the numbers are known.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))

SPEEDS = (-0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45)
PULSE_S = 1.0
TICK_S = 0.1
SETTLE_S = 1.0
MOVED_M = 0.03                      # "really moved"
RULER_MIN_M, RULER_MAX_M = 0.15, 0.60
REAR_MIN_M = 0.60


def mcp_check() -> dict:
    """Front ruler + rear clearance through friday-mcp (still running)."""
    import asyncio

    import config
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def ask():
        async with streamable_http_client(config.MCP_HTTP_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as s:
                await s.initialize()
                r = await s.call_tool("lidar_sweep", {"full": True})
                text = next((getattr(b, "text", "") for b in r.content
                             if getattr(b, "text", "")), "{}")
                return json.loads(text)
    data = asyncio.run(ask())
    raw = {int(b): int(mm) for b, mm in (data.get("bearings_mm") or {}).items()}
    fwd = float(config.LIDAR_FORWARD_DEG)
    def sector(centre, half):
        out = [mm for b, mm in raw.items()
               if abs(((b - fwd - centre) + 180) % 360 - 180) <= half and mm]
        return min(out) / 1000.0 if out else None
    return {"front_m": sector(0, 10), "rear_m": sector(180, 30)}


def tof_ruler(slot) -> float | None:
    """The ruler ahead: the 10th percentile of the central 4x4 ToF cells."""
    vals = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5 and len(vals) < 5:
        value, age = slot.read()
        if value is not None and age < 0.5:
            c = np.asarray(value, dtype=float)[2:6, 2:6].ravel()
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(float(np.percentile(c, 10)))
        time.sleep(0.12)
    return float(np.median(vals)) if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--i-am-watching", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan, move nothing")
    ap.add_argument("--drive", choices=("old", "quiet"), default="old",
                    help="old = the drive the mode uses now (kick every start); "
                         "quiet = a58b9e3's quieter gears")
    args = ap.parse_args(argv)
    print(__doc__.split("What it does")[0].strip()[:600], "\n")
    print(f"Drive: {args.drive}")
    print("Plan:", ", ".join(f"{v:+.2f}" for v in SPEEDS), f"m/s, {PULSE_S:g} s each, "
          f"straight back; stop once two speeds move >= {MOVED_M * 100:.0f} cm.")
    if args.dry:
        print("--dry: nothing moved, nothing stopped.")
        return 0
    if not args.i_am_watching:
        print("Refusing: pass --i-am-watching (Harsh must watch the robot).")
        return 2

    pre = mcp_check()
    print(f"Before: LiDAR front {pre['front_m']} m, rear {pre['rear_m']} m")
    if pre["rear_m"] is not None and pre["rear_m"] < REAR_MIN_M:
        print(f"Refusing: the rear is only {pre['rear_m']:.2f} m clear (need {REAR_MIN_M}).")
        return 2
    if input("Is there a FLAT box/wall 15-60 cm in front, and nothing behind? "
             "Type yes: ").strip().lower() != "yes":
        print("Not confirmed. Nothing moved.")
        return 2

    import pi_deploy as pd
    rows: list[dict] = []
    stopped_mcp = False
    motors = tof_thread = None
    try:
        subprocess.run(["systemctl", "--user", "stop", "friday-mcp"], check=True)
        stopped_mcp = True
        time.sleep(2.0)
        cfg = pd.RobotConfig()
        flags = ["--motor", "friday", "--i-am-watching"]
        if args.drive == "old":
            flags += ["--motor-kick-rest", "0", "--motor-pulse-burst", "1",
                      "--no-motor-plug-guard"]
        pargs = pd.build_arg_parser().parse_args(flags)
        slot = pd.SensorSlot("tof", cfg.tof_deadline, critical=True)
        tof_thread = pd.DFRobotToFThread(slot, cfg, pd.resolve_serial_port("tof", pargs.tof_port),
                                         pargs.tof_baud)
        tof_thread.start()
        time.sleep(2.0)
        motors = pd.build_motors(cfg, pargs)
        moved_in_a_row = 0
        for v in SPEEDS:
            before = tof_ruler(slot)
            if before is None or not (RULER_MIN_M <= before <= RULER_MAX_M + 0.2):
                print(f"  {v:+.2f}: the ruler reads {before} m -- out of range, stopping.")
                break
            t0 = time.monotonic()
            while time.monotonic() - t0 < PULSE_S:
                motors.drive(v, 0.0)
                time.sleep(TICK_S)
            motors.drive(0.0, 0.0)
            time.sleep(SETTLE_S)
            after = tof_ruler(slot)
            moved = None if after is None else after - before
            rows.append({"v": v, "before_m": before, "after_m": after, "moved_m": moved})
            print(f"  {v:+.2f} m/s for {PULSE_S:g} s: ruler {before * 100:.1f} -> "
                  f"{'?' if after is None else f'{after * 100:.1f}'} cm  "
                  f"moved {'?' if moved is None else f'{moved * 100:+.1f}'} cm")
            moved_in_a_row = moved_in_a_row + 1 if (moved or 0) >= MOVED_M else 0
            if moved_in_a_row >= 2:
                break
            if after is not None and after > RULER_MAX_M + 0.3:
                print("  the ruler is getting far; stopping.")
                break
    finally:
        try:
            if motors is not None:
                motors.drive(0.0, 0.0)
                motors.close()
        except Exception as exc:                        # noqa: BLE001
            print(f"motor stop/close: {exc}")
        try:
            if tof_thread is not None:
                tof_thread.stop()
        except Exception:                               # noqa: BLE001
            pass
        if stopped_mcp:
            subprocess.run(["systemctl", "--user", "start", "friday-mcp"])
            print("friday-mcp restarted:", subprocess.run(
                ["systemctl", "--user", "is-active", "friday-mcp"],
                capture_output=True, text=True).stdout.strip())
    out = ROOT / "state" / f"reverse_floor_test_{args.drive}.json"
    out.write_text(json.dumps({"at": time.time(), "drive": args.drive,
                               "pulse_s": PULSE_S, "rows": rows},
                              indent=1))
    first = next((r["v"] for r in rows if (r["moved_m"] or 0) >= MOVED_M), None)
    print(f"\nSlowest reverse that really moved her: {first} m/s -> {out}")
    print("Now tell Claude what you SAW (did she move each time, and how far).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
