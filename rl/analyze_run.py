#!/usr/bin/env python3
"""Read a black box and say who actually drove the robot.

    python3 analyze_run.py                       # newest blackbox-*.csv
    python3 analyze_run.py baseline_v1_blackbox.csv
    python3 analyze_run.py a.csv b.csv           # compare two runs

WHY THIS IS SEPARATE FROM pi_deploy.py
--------------------------------------
It needs no hardware, no venv and no LiDAR, so the analysis can be re-run and argued
with long after the robot has been put away -- and, more to the point, on the baseline
recording of the first autonomous run, which cannot be repeated.

TWO SCHEMAS
-----------
Runs before the provenance fix have 10 columns and no `v_raw`. In those files
`v_policy` is NOT the network's output: `ControlLoop.tick` overwrote it in the
no-reverse fallback before recording, so every tick whose heading came from the
`_open_side()` rule is logged under the policy's name at exactly +-omega_max. This
script detects that schema and reports what it can, refusing to report what it cannot
-- an inferred number presented beside measured ones would be indistinguishable from
them.
"""

from __future__ import annotations

import csv
import glob
import math
import statistics as st
import sys
from collections import Counter

# Gates whose heading came from a hand-written rule rather than from the network.
# Kept in sync with telemetry.RULE_STAGES by name.
RULE_GATES = {"no_reverse_turn", "escape_turn"}
SAFETY_GATES = {"governor", "tof_slow", "visual_slow", "visual_stop", "REFLEX_STOP",
                "stale_lidar", "stale_tof"}
DT = 0.1  # control period; matches RobotConfig.control_hz = 10


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows, ("v_raw" in rows[0])


def fnum(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r[key])
        except (ValueError, KeyError, TypeError):
            continue
        if not math.isnan(v):
            out.append(v)
    return out


def analyse(path):
    rows, has_raw = load(path)
    n = len(rows)
    t = fnum(rows, "t")
    span = (t[-1] - t[0]) if len(t) > 1 else 0.0

    print("=" * 74)
    print(f"RUN ANALYSIS  {path}")
    print("=" * 74)
    print(f"{n} ticks over {span:.1f} s"
          f"   schema: {'v_raw present (provenance-correct)' if has_raw else 'LEGACY 10-column'}")
    if not has_raw:
        print()
        print("  !! This file predates the provenance fix. Its `v_policy` column is the")
        print("     POST-SUBSTITUTION working value, so on no_reverse_turn ticks it shows")
        print("     the rule's choice under the policy's name. Rule-vs-policy attribution")
        print("     below is derived from the GATE column, which is reliable; the raw")
        print("     network output for those ticks was never recorded and is unrecoverable.")

    # ---- who decided ------------------------------------------------------------
    gates = Counter(r["gate"] for r in rows)
    print("\n-- GATE DISTRIBUTION (which stage had the last word) --")
    for g, c in gates.most_common():
        tag = "RULE  " if g in RULE_GATES else ("SAFETY" if g in SAFETY_GATES else "POLICY")
        print(f"   {g:<18} {c:5d}  {100 * c / n:5.1f}%   [{tag}]")

    if has_raw:
        auth = Counter(r.get("authority", "?") for r in rows)
        print("\n-- AUTHORITY (measured per tick, not inferred from the gate) --")
        for a, c in auth.most_common():
            print(f"   {a:<18} {c:5d}  {100 * c / n:5.1f}%")
        subs = sum(1 for r in rows
                   if abs(float(r["v_raw"]) - float(r["v_policy"])) > 1e-9
                   or abs(float(r["omega_raw"]) - float(r["omega_policy"])) > 1e-9)
        print(f"\n   ticks where a rule REPLACED the network's answer: {subs} "
              f"({100 * subs / n:.1f}%)")

    # ---- motion attribution -----------------------------------------------------
    v_out = fnum(rows, "v_out")
    w_out = fnum(rows, "omega_out")
    dist = sum(abs(v) for v in v_out) * DT
    rot = sum(abs(w) for w in w_out) * DT
    rule_rot = sum(abs(float(r["omega_out"])) for r in rows if r["gate"] in RULE_GATES) * DT
    print("\n-- MOTION (integrated from the commanded twist) --")
    print(f"   distance commanded : {dist:6.2f} m")
    print(f"   rotation commanded : {math.degrees(rot):6.0f} deg "
          f"({math.degrees(rot) / 360:.2f} full turns)")
    if rot > 1e-9:
        print(f"   rotation from RULE ticks: {math.degrees(rule_rot):.0f} deg "
              f"({100 * rule_rot / rot:.0f}% of all turning)")
    if dist > 1e-6:
        print(f"   turning per metre  : {math.degrees(rot) / dist:6.0f} deg/m"
              f"   <- high means it turned more than it travelled")

    # ---- how much did safety take away ------------------------------------------
    src = "v_raw" if has_raw else "v_policy"
    pol_v = fnum(rows, src)
    if len(pol_v) == len(v_out):
        cut = sum(1 for a, b in zip(pol_v, v_out) if b < a - 1e-6)
        thru = sum(1 for a, b in zip(pol_v, v_out) if abs(a - b) <= 1e-6)
        print("\n-- SAFETY CHAIN --")
        print(f"   forward speed passed through unchanged : {thru:5d}  ({100 * thru / n:5.1f}%)")
        print(f"   forward speed reduced downstream       : {cut:5d}  ({100 * cut / n:5.1f}%)")

    # ---- policy behaviour -------------------------------------------------------
    pol_w = fnum(rows, "omega_raw" if has_raw else "omega_policy")
    print(f"\n-- POLICY OUTPUT ({'raw network' if has_raw else 'post-substitution -- see warning'}) --")
    if pol_v:
        print(f"   v     {min(pol_v):+.3f} .. {max(pol_v):+.3f}   mean {st.mean(pol_v):+.3f}"
              f"   distinct {len(set(pol_v))}")
        rev = sum(1 for x in pol_v if x < 0)
        print(f"   reverse requested on {rev} ticks ({100 * rev / n:.1f}%)"
              f"{'  <- this drivetrain cannot execute reverse' if rev else ''}")
    if pol_w:
        print(f"   omega {min(pol_w):+.3f} .. {max(pol_w):+.3f}   mean {st.mean(pol_w):+.3f}"
              f"   distinct {len(set(pol_w))}")
        sat = sum(1 for x in pol_w if abs(abs(x) - 1.6) < 1e-9)
        if sat:
            print(f"   omega at exactly +-1.600 on {sat} ticks"
                  + ("" if has_raw else "  <- the rule's signature, not the policy's"))

    # ---- sensors ----------------------------------------------------------------
    lm = fnum(rows, "lidar_min")
    tm = fnum(rows, "tof_min")
    vs = fnum(rows, "visual")
    print("\n-- SENSORS --")
    if lm:
        print(f"   lidar_min {min(lm):.2f} .. {max(lm):.2f} m   distinct {len(set(lm))}"
              + ("   !! PINNED -- LiDAR was not updating" if len(set(lm)) < 3 else ""))
        print(f"   closest approach to anything: {min(lm):.2f} m")
    if tm:
        print(f"   tof_min   {min(tm):.2f} .. {max(tm):.2f} m   distinct {len(set(tm))}")
    if vs:
        if len(set(vs)) == 1 and vs[0] == 0.0:
            print("   visual    0.00 on every tick  !! NEITHER CAMERA CONTRIBUTED")
        else:
            print(f"   visual    {min(vs):.2f} .. {max(vs):.2f}   distinct {len(set(vs))}")

    # ---- health -----------------------------------------------------------------
    ms = fnum(rows, "loop_ms")
    if ms:
        over = sum(1 for x in ms if x > DT * 1000)
        print("\n-- TIMING --")
        print(f"   loop ms mean {st.mean(ms):.2f}  p95 "
              f"{sorted(ms)[int(0.95 * len(ms))]:.2f}  max {max(ms):.2f}"
              f"   (budget {DT * 1000:.0f})")
        print(f"   overruns: {over}")

    # ---- the §18 headline metrics ----------------------------------------------
    print("\n-- BASELINE METRICS --")
    stuck = gates.get("escape_turn", 0)
    print(f"   distance travelled     {dist:.2f} m")
    print(f"   min obstacle distance  {min(lm):.2f} m" if lm else "   min obstacle distance  n/a")
    print(f"   safety interventions   {sum(gates.get(g, 0) for g in SAFETY_GATES)}")
    print(f"   rule substitutions     {sum(gates.get(g, 0) for g in RULE_GATES)}")
    print(f"   stuck/escape events    {stuck}")
    print(f"   emergency stops        {gates.get('REFLEX_STOP', 0)}")
    print(f"   stale-sensor stops     {gates.get('stale_lidar', 0) + gates.get('stale_tof', 0)}")
    if dist > 1e-6:
        print(f"   turn efficiency        {math.degrees(rot) / dist:.0f} deg/m "
              f"(lower is more direct)")
    print("=" * 74)
    return {"path": path, "n": n, "dist": dist, "rot_deg": math.degrees(rot),
            "rule_rot_pct": (100 * rule_rot / rot) if rot > 1e-9 else 0.0,
            "min_lidar": min(lm) if lm else float("nan"),
            "safety": sum(gates.get(g, 0) for g in SAFETY_GATES)}


def main(argv):
    paths = argv[1:] or sorted(glob.glob("blackbox-*.csv"))[-1:]
    if not paths:
        raise SystemExit("no blackbox-*.csv found; pass a path")
    out = [analyse(p) for p in paths]
    if len(out) > 1:
        print("\nCOMPARISON")
        print(f"{'run':<34}{'ticks':>7}{'dist m':>9}{'rot deg':>9}"
              f"{'rule rot %':>12}{'min m':>8}{'safety':>8}")
        for r in out:
            print(f"{r['path'][-33:]:<34}{r['n']:>7}{r['dist']:>9.2f}{r['rot_deg']:>9.0f}"
                  f"{r['rule_rot_pct']:>12.0f}{r['min_lidar']:>8.2f}{r['safety']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
