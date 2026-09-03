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

# Gates whose heading came from something other than the policy. Split by ROLE, not
# lumped as "rules": `escape_turn` means the ROBOT was stuck (recovery), while
# `no_reverse_turn` means the COMMAND was unexecutable (capability adaptation). They
# fail differently and a run dominated by one says something different from the other.
RECOVERY_GATES = {"escape_turn"}
CAPABILITY_GATES = {"no_reverse_turn"}
PLANNER_GATES = {"planner"}
RULE_GATES = RECOVERY_GATES | CAPABILITY_GATES
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
        tag = ("RECOVERY  " if g in RECOVERY_GATES else
               "CAPABILITY" if g in CAPABILITY_GATES else
               "PLANNER   " if g in PLANNER_GATES else
               "SAFETY    " if g in SAFETY_GATES else "RL_POLICY ")
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

    # ---- Phase 10: control-source shares, coverage rate, and health -------------
    if has_raw and "infer_ms" in rows[0]:
        print("\n-- CONTROL SOURCE SHARE --")
        auth = Counter(r["authority"] for r in rows)
        for a in ("RL_POLICY", "EXPLORATION_PLANNER", "LOCAL_NAVIGATOR",
                  "SAFETY_OVERRIDE", "RECOVERY_CONTROLLER", "CAPABILITY_ADAPTER",
                  "MOTION_SMOOTHING"):
            c = auth.get(a, 0)
            print(f"   {a:<22}{c:5d}  {100 * c / n:5.1f}%")

        cov = fnum(rows, "explored_m2")
        if cov and max(cov) > 0:
            gained = max(cov) - min(cov)
            print("\n-- EXPLORATION --")
            print(f"   coverage {min(cov):.1f} -> {max(cov):.1f} m^2  (+{gained:.1f})")
            if span > 0:
                print(f"   rate {gained / span * 60:.1f} m^2/min")
            if dist > 1e-6:
                # The efficiency number that matters: area gained per metre driven.
                # Distance alone rewards a robot that drives in circles.
                print(f"   efficiency {gained / dist:.1f} m^2 per metre driven")

        lat = fnum(rows, "infer_ms")
        if lat:
            print("\n-- POLICY LATENCY --")
            print(f"   inference mean {st.mean(lat):.3f} ms  p95 "
                  f"{sorted(lat)[int(0.95 * len(lat))]:.3f}  max {max(lat):.3f}")

        recs = [r.get("recovery", "") for r in rows]
        episodes = sum(1 for a, b in zip(recs, recs[1:])
                       if a != "RECOVERING" and b == "RECOVERING")
        in_rec = sum(1 for x in recs if x == "RECOVERING")
        print("\n-- RECOVERY --")
        print(f"   episodes {episodes}   ticks spent recovering {in_rec} "
              f"({100 * in_rec / n:.1f}%)")

        pb = [abs(float(r["plan_bias"])) for r in rows
              if r.get("plan_bias") not in ("", None)]
        fired = [x for x in pb if x > 1e-9]
        if pb:
            print("\n-- PLANNER --")
            print(f"   biased {len(fired)}/{len(pb)} ticks ({100 * len(fired) / len(pb):.0f}%)")
            if fired:
                print(f"   |bias| median {st.median(fired):.3f} rad/s  max {max(fired):.3f}")

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
    cov = fnum(rows, "explored_m2") if "explored_m2" in rows[0] else []
    lat = fnum(rows, "infer_ms") if "infer_ms" in rows[0] else []
    auth = Counter(r.get("authority", "") for r in rows)
    return {"path": path, "n": n, "dist": dist, "rot_deg": math.degrees(rot),
            "rule_rot_pct": (100 * rule_rot / rot) if rot > 1e-9 else 0.0,
            "min_lidar": min(lm) if lm else float("nan"),
            "safety": sum(gates.get(g, 0) for g in SAFETY_GATES),
            "deg_per_m": (math.degrees(rot) / dist) if dist > 1e-6 else float("nan"),
            "coverage": (max(cov) - min(cov)) if cov else float("nan"),
            "infer_ms": st.mean(lat) if lat else float("nan"),
            # NaN, not 0.0, when the column is absent. A legacy file has no
            # `authority` column, and reporting "RL_POLICY 0.0%" for it would state
            # that the policy never drove -- which is the opposite of what that run
            # actually showed. Unknown and zero must not render the same.
            "policy_pct": (100 * auth.get("RL_POLICY", 0) / n
                           if (has_raw and n) else float("nan")),
            "planner_pct": (100 * auth.get("EXPLORATION_PLANNER", 0) / n
                            if (has_raw and n) else float("nan")),
            "safety_pct": (100 * auth.get("SAFETY_OVERRIDE", 0) / n
                           if (has_raw and n) else float("nan"))}


def main(argv):
    paths = argv[1:] or sorted(glob.glob("blackbox-*.csv"))[-1:]
    if not paths:
        raise SystemExit("no blackbox-*.csv found; pass a path")
    out = [analyse(p) for p in paths]
    if len(out) > 1:
        print("\n" + "=" * 74)
        print("COMPARISON  (BASELINE vs V2)")
        print("=" * 74)
        rows_ = [
            ("ticks", "{:.0f}", "n"),
            ("distance m", "{:.2f}", "dist"),
            ("rotation deg", "{:.0f}", "rot_deg"),
            ("deg per metre", "{:.0f}", "deg_per_m"),
            ("non-policy rot %", "{:.0f}", "rule_rot_pct"),
            ("coverage m^2", "{:.1f}", "coverage"),
            ("min clearance m", "{:.2f}", "min_lidar"),
            ("safety events", "{:.0f}", "safety"),
            ("RL_POLICY %", "{:.1f}", "policy_pct"),
            ("PLANNER %", "{:.1f}", "planner_pct"),
            ("SAFETY %", "{:.1f}", "safety_pct"),
            ("inference ms", "{:.3f}", "infer_ms"),
        ]
        head = f"{'metric':<20}" + "".join(f"{r['path'][-16:]:>18}" for r in out)
        print(head)
        print("-" * len(head))
        for label, fmt, key in rows_:
            cells = ""
            for r in out:
                v = r.get(key, float("nan"))
                cells += f"{(fmt.format(v) if v == v else 'n/a'):>18}"
            print(f"{label:<20}{cells}")
        print("\nNote: 'n/a' means the column is absent from that file, NOT zero.")
        print("COMPARE LIKE WITH LIKE. A simulated run and a hardware run are not")
        print("comparable on coverage or clearance: the simulator is an open box and")
        print("the room is not. Use this to compare hardware against hardware.")
        print("Legacy 10-column black boxes predate the per-source and coverage")
        print("columns, so those rows cannot be filled for them and are not guessed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
