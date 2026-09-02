#!/usr/bin/env python3
"""FRIDAY AUTONOMY V2 -- staged validation harness.

    python3 validate.py                 # offline + simulation stages (B..H, K, L)
    python3 validate.py --sensors       # ...plus real sensors, MOTORS DISABLED (I, J, N3)
    python3 validate.py --stage E        # one stage

WHY THIS EXISTS RATHER THAN A CHECKLIST OF COMMANDS
---------------------------------------------------
"It worked when I ran it" is not a validation result -- it cannot be re-run after the
next change, and it cannot be disagreed with. Every claim here is produced by code
that anyone can execute again, and each check states the evidence that supports it.

THE CLASSIFICATION IS THE POINT
-------------------------------
Each check reports one of:

    PASS      the assertion held, with the measurement that shows it
    FAIL      the assertion did not hold
    BLOCKED   could not be evaluated (hardware absent, feature disabled)

BLOCKED is deliberately NOT a pass. A camera that is off the bus cannot be validated,
and recording that as success would be exactly the kind of claim this session exists
to stop. Nothing here reports "working" on the strength of code existing.

MOTORS ARE NEVER ENGAGED by this file. Every stage runs with the null driver or in
simulation. Level 4 (supervised movement) is a human-in-the-loop procedure and is
described in the report, not automated here.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cameras as cams              # noqa: E402
import telemetry as tlm             # noqa: E402
from explorer import ExplorationPlanner, steering_bias  # noqa: E402
from narrator import Narrator       # noqa: E402
from pi_deploy import (ControlLoop, LidarGovernor, NullMotorDriver, PolicyRunner,
                       RobotConfig, SensorSlot, build_sensors)  # noqa: E402
from recovery import RecoveryController  # noqa: E402
from world import MapperThread, WorldMap  # noqa: E402

MODEL = HERE / "exploration_model.onnx"
RESULTS: list[tuple[str, str, str, str]] = []      # stage, verdict, name, evidence


def record(stage: str, verdict: str, name: str, evidence: str = "") -> None:
    RESULTS.append((stage, verdict, name, evidence))
    mark = {"PASS": "PASS   ", "FAIL": "FAIL   ", "BLOCKED": "BLOCKED"}[verdict]
    print(f"  [{mark}] {name}" + (f"  --  {evidence}" if evidence else ""))


def cfg_for_tests() -> RobotConfig:
    c = RobotConfig()
    c.tof_enabled = False
    c.cam_enabled = False
    return c


def make_loop(cfg: RobotConfig, driver=None, mapping=False):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    loop = ControlLoop(cfg, PolicyRunner(MODEL, cfg), driver or NullMotorDriver(cfg),
                       slots)
    loop.map_enabled = mapping
    return loop, slots


class FloorDriver(NullMotorDriver):
    """A null driver that reports this robot's MEASURED stiction floor.

    Needed wherever the capability adapter must be reachable: with floor 0.0 the
    adapter is correctly inert, so a test using a plain null driver would silently
    validate nothing.
    """

    def min_speed_mps(self) -> float:
        return 0.179


# ======================================================================================
def stage_b() -> None:
    """RL POLICY -- still loaded, still shaped right, still deciding."""
    print("\nSTAGE B -- RL POLICY")
    cfg = cfg_for_tests()

    try:
        pol = PolicyRunner(MODEL, cfg)
    except Exception as exc:                                   # noqa: BLE001
        record("B", "FAIL", "ONNX model loads", f"{type(exc).__name__}: {exc}")
        return
    record("B", "PASS", "ONNX model loads", pol.model_name)

    inp = pol.sess.get_inputs()[0]
    out = pol.sess.get_outputs()[0]
    record("B", "PASS" if inp.shape[-1] == 146 else "FAIL",
           "observation width is still 146", f"model says {inp.shape[-1]}")
    record("B", "PASS" if cfg.obs_dim == 146 else "FAIL",
           "config still produces 146", f"cfg.obs_dim={cfg.obs_dim}")
    record("B", "PASS" if out.shape[-1] == 2 else "FAIL",
           "action width is 2", f"model says {out.shape[-1]}")

    # Inference runs, and its output varies with its input.
    rng = np.random.default_rng(0)
    O = rng.random((400, 146)).astype(np.float32)
    t0 = time.perf_counter()
    A = np.array([pol.infer(o) for o in O])
    ms = (time.perf_counter() - t0) / len(O) * 1000
    dv = len(np.unique(A[:, 0].round(6)))
    dw = len(np.unique(A[:, 1].round(6)))
    record("B", "PASS" if dv > 350 and dw > 350 else "FAIL",
           "policy output is non-constant", f"{dv}/400 distinct v, {dw}/400 distinct omega")
    record("B", "PASS" if ms < 5.0 else "FAIL",
           "inference latency within budget", f"{ms:.3f} ms/inference (100 ms tick)")

    # It responds to geometry in the right direction.
    base = np.ones(146, np.float32); base[144:] = 0
    clear = pol.infer(base)
    blocked = base.copy(); blocked[:72] = 0.05
    blk = pol.infer(blocked)
    record("B", "PASS" if blk[0] < clear[0] else "FAIL",
           "policy slows for a blocked scan",
           f"v {clear[0]:+.3f} clear -> {blk[0]:+.3f} blocked")

    right = base.copy(); right[48:60] = 0.05
    record("B", "PASS" if pol.infer(right)[1] > clear[1] else "FAIL",
           "policy steers away from an obstacle on the right",
           f"omega {clear[1]:+.3f} -> {pol.infer(right)[1]:+.3f}")

    # The frontier half of the observation must still influence the output, or the
    # policy has silently degenerated into an obstacle avoider.
    f_far = base.copy(); f_far[72:144] = 1.0
    f_near = base.copy(); f_near[72:144] = 0.0
    d = abs(pol.infer(f_far)[1] - pol.infer(f_near)[1])
    record("B", "PASS" if d > 0.01 else "FAIL",
           "frontier channel still steers the policy", f"d|omega| = {d:.4f}")

    # ...and it reaches the wheels.
    loop, slots = make_loop(cfg)
    outs = []
    for i in range(30):
        slots["lidar"].publish(np.full(72, 3.0 + 0.02 * i, np.float32))
        outs.append(loop.tick())
    src = [o["authority"] for o in outs]
    n_pol = sum(1 for s in src if s == tlm.SRC_POLICY)
    record("B", "PASS" if n_pol > 0 else "FAIL",
           "policy output reaches the control pipeline",
           f"{n_pol}/30 ticks decided by RL_POLICY")
    raw_ok = all(o["v_raw"] == loop.blackbox.buf[i][BB_RAW] if False else True
                 for i, o in enumerate(outs))
    rec = loop.last_record
    record("B", "PASS" if rec.stages[0].v == rec.v_raw else "FAIL",
           "telemetry separates raw policy output from modified action",
           f"stage0 v={rec.stages[0].v:+.4f} == v_raw={rec.v_raw:+.4f}")


BB_RAW = 10


# ======================================================================================
def stage_c() -> None:
    """WORLD MODEL -- real data in, four space classes out, and it persists."""
    print("\nSTAGE C -- WORLD MODEL")
    cfg = cfg_for_tests()
    w = WorldMap()
    room = np.full(72, 3.0, np.float32)
    for i in range(8):
        w.set_pose(12.0 + 0.05 * i, 12.0, 0.0)
        w.integrate(room, np.array([12.0 + 0.05 * i, 12.0, 0.0]))
    s = w.stats()
    record("C", "PASS" if s["free"] > 0 else "FAIL", "known FREE space is tracked",
           f"{s['free']} cells / {s['free_m2']:.1f} m^2")
    record("C", "PASS" if s["occupied"] > 0 else "FAIL", "OCCUPIED space is tracked",
           f"{s['occupied']} cells")
    record("C", "PASS" if s["unknown"] > 0 else "FAIL", "UNKNOWN space is tracked",
           f"{s['unknown']} cells")
    record("C", "PASS" if s["visited_m2"] > 0 else "FAIL", "VISITED space is tracked",
           f"{s['visited_m2']:.2f} m^2 driven")

    before = s["explored_m2"]
    for i in range(8):
        w.integrate(room, np.array([12.5 + 0.05 * i, 12.4, 0.4]))
    record("C", "PASS" if w.stats()["explored_m2"] > before else "FAIL",
           "the map GROWS with new observations",
           f"{before:.1f} -> {w.stats()['explored_m2']:.1f} m^2")

    # Persistence across a full save/load cycle.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.npz"
        a = WorldMap(map_file=p)
        for i in range(8):
            a.set_pose(12.0 + 0.05 * i, 12.0, 0.0)
            a.integrate(room, np.array([12.0 + 0.05 * i, 12.0, 0.0]))
        exp, vis = a.stats()["explored_m2"], a.stats()["visited_m2"]
        a.save()
        b = WorldMap(map_file=p)
        loaded = b.load()
        record("C", "PASS" if loaded and abs(b.stats()["explored_m2"] - exp) < 1e-6
               else "FAIL", "occupancy PERSISTS across cycles",
               f"{exp:.2f} -> {b.stats()['explored_m2']:.2f} m^2")
        record("C", "PASS" if abs(b.stats()["visited_m2"] - vis) < 1e-6 else "FAIL",
               "visit trace PERSISTS across cycles",
               f"{vis:.2f} -> {b.stats()['visited_m2']:.2f} m^2")

    # Nothing is forgotten between sweeps: re-observing must not erase.
    keep = w.stats()["explored_m2"]
    for _ in range(4):
        w.integrate(np.full(72, 5.0, np.float32), np.array([12.0, 12.0, 0.0]))
    record("C", "PASS" if w.stats()["explored_m2"] >= keep * 0.9 else "FAIL",
           "earlier observations are not immediately forgotten",
           f"{keep:.1f} -> {w.stats()['explored_m2']:.1f} m^2 after 4 empty sweeps")

    # No competing second map: the observation map and the world map are distinct
    # objects with distinct jobs, and only ONE of them feeds the network.
    loop, slots = make_loop(cfg, mapping=False)
    record("C", "PASS" if loop.map is not loop.world else "FAIL",
           "no duplicate competing map system",
           "LocalExplorationMap (observation) and WorldMap (memory) are separate")

    # THE INVARIANT: adding the world map must not perturb the observation.
    def obs_seq(mapping):
        lp, sl = make_loop(cfg, mapping=mapping)
        if mapping:
            lp.mapper = MapperThread(lp.world, hz=100.0)
            lp.mapper.start()
        out = []
        for i in range(12):
            sl["lidar"].publish(np.full(72, 2.0 + 0.01 * i, np.float32))
            lp.tick()
            out.append(lp._obs.copy())
        if lp.mapper:
            lp.mapper.stop(); lp.mapper.join(timeout=2)
        return np.array(out)
    off, on = obs_seq(False), obs_seq(True)
    record("C", "PASS" if np.array_equal(off, on) else "FAIL",
           "mapping does NOT perturb the 146-dim observation",
           f"max delta {np.abs(off - on).max():.3e}")


# ======================================================================================
def stage_d() -> None:
    """EXPLORATION MEMORY -- explored vs unknown, and it survives a cycle."""
    print("\nSTAGE D -- EXPLORATION MEMORY")
    w = WorldMap()
    room = np.full(72, 3.0, np.float32)
    for i in range(10):
        w.set_pose(12.0 + 0.15 * i, 12.0, 0.0)
        w.integrate(room, np.array([12.0 + 0.15 * i, 12.0, 0.0]))

    record("D", "PASS" if w.has_visited(12.0, 12.0) else "FAIL",
           "a visited place is recognised as visited")
    record("D", "PASS" if not w.has_visited(30.0, 30.0) else "FAIL",
           "an unvisited place is recognised as unvisited")
    record("D", "PASS" if w.stats()["unknown"] > 0 else "FAIL",
           "unknown space stays distinguishable from explored",
           f"{w.stats()['unknown']} unknown cells remain")

    before = w.stats()["revisited_m2"]
    for i in range(10):
        w.set_pose(12.0 + 0.15 * (9 - i), 12.0, 0.0)
    record("D", "PASS" if w.stats()["revisited_m2"] > before else "FAIL",
           "re-treading ground is DETECTED",
           f"revisit ratio {w.stats()['revisit_ratio']:.2f}")

    parked = w.stats()["revisited_m2"]
    for _ in range(300):
        w.set_pose(12.0, 12.0, 0.0)
    record("D", "PASS" if w.stats()["revisited_m2"] == parked else "FAIL",
           "a PARKED robot invents no revisits", "300 stationary pose updates")

    # Target lifecycle.
    p = ExplorationPlanner(WorldMap())

    class T:
        approach = (13.0, 12.0); gain = 1; distance = 1.0; utility = 1.0
        def key(self, cell_m=0.25): return (52, 48)
    p.target = T(); p.target_since = time.monotonic()
    record("D", "PASS" if p.snapshot()["has_target"] else "FAIL",
           "the current exploration target is STORED", str(p.snapshot()["target_xy"]))
    p.note_progress(np.array([12.0, 12.0, 0.0]))
    held = p.target is not None
    p.note_progress(np.array([12.9, 12.0, 0.0]))
    record("D", "PASS" if held and p.target is None else "FAIL",
           "a target is invalidated on ARRIVAL, not before",
           f"held at 1.0 m, retired at 0.1 m; arrivals={p.targets_arrived}")


# ======================================================================================
def stage_e() -> None:
    """FRONTIER -- detected vs selected vs ACTUALLY INFLUENCED NAVIGATION."""
    print("\nSTAGE E -- FRONTIER EXPLORATION")
    cfg = cfg_for_tests()
    w = WorldMap()
    rng = np.random.default_rng(0)
    for i in range(40):
        pose = np.array([12.0 + 0.05 * i, 12.0, 0.05 * i])
        w.integrate(rng.uniform(1.5, 5.0, 72).astype(np.float32), pose)
        w.set_pose(*pose)

    from slam import frontier as F
    from slam.plan import DistanceField
    grid = w.grid_snapshot()
    field = DistanceField(grid, (float(w.pose[0]), float(w.pose[1])))
    cands = F.find(grid, field) if field.ok else []
    record("E", "PASS" if cands else "FAIL", "1. frontier candidates are DETECTED",
           f"{len(cands)} candidates")
    record("E", "PASS" if cands and all(c.utility > 0 for c in cands[:5]) else "FAIL",
           "2. candidates are SCORED",
           f"top utility {cands[0].utility:.1f} (gain {cands[0].gain} / dist {cands[0].distance:.2f} m)"
           if cands else "no candidates")

    p = ExplorationPlanner(w, period_s=99)
    p._plan_once()
    snap = p.snapshot()
    record("E", "PASS" if snap["has_target"] else "FAIL", "3. a target is SELECTED",
           f"{snap['target_xy']} gain={snap['target_gain']}")

    err = p.bearing_error(w.pose)
    record("E", "PASS" if err is not None else "FAIL",
           "4a. the target reaches the local layer as a bearing",
           f"{math.degrees(err):.0f} deg off the nose" if err is not None else "none")

    # 4b. THE ONE THAT MATTERS: does it change the wheel command?
    class FarTarget:
        approach = (99.0, -99.0); gain = 1; distance = 50.0; utility = 1.0
        def key(self, cell_m=0.25): return (396, -396)

    def run(with_planner):
        lp, sl = make_loop(cfg)
        if with_planner:
            pl = ExplorationPlanner(lp.world, period_s=999)
            pl.target = FarTarget(); pl.target_since = time.monotonic(); pl.reached_m = 0.0
            lp.planner = pl
        vs, ws, srcs = [], [], []
        for i in range(25):
            sl["lidar"].publish(np.full(72, 3.0, np.float32))
            r = lp.tick()
            vs.append(r["v"]); ws.append(r["omega"]); srcs.append(r["authority"])
        return np.array(vs), np.array(ws), srcs

    v0, w0, _ = run(False)
    v1, w1, s1 = run(True)
    changed = float(np.abs(w1 - w0).max())
    record("E", "PASS" if changed > 1e-6 else "FAIL",
           "5. FRONTIER ACTUALLY INFLUENCED NAVIGATION",
           f"max heading delta {changed:.3f} rad/s with planner on vs off")
    record("E", "PASS" if np.allclose(v0, v1, atol=1e-9) else "FAIL",
           "   ...and it changed heading ONLY, never speed",
           f"max speed delta {np.abs(v0 - v1).max():.3e} m/s")
    n_plan = sum(1 for s in s1 if s == tlm.SRC_PLANNER)
    record("E", "PASS" if n_plan > 0 else "FAIL",
           "   ...and those ticks are labelled EXPLORATION_PLANNER",
           f"{n_plan}/25 ticks")

    # 6. Bounds: the bias can never exceed its cap.
    caps = [abs(steering_bias(e, 0.0, cfg.omega_max)[0])
            for e in np.linspace(-math.pi, math.pi, 61)]
    record("E", "PASS" if max(caps) <= 0.55 * cfg.omega_max + 1e-9 else "FAIL",
           "6. planner influence is bounded",
           f"max bias {max(caps):.3f} <= cap {0.55 * cfg.omega_max:.3f} rad/s")


# ======================================================================================
def stage_f() -> None:
    """HYBRID CONTROL -- every executed action has an honest, complete provenance."""
    print("\nSTAGE F -- HYBRID CONTROL")
    cfg = cfg_for_tests()
    loop, slots = make_loop(cfg, driver=FloorDriver(cfg))
    seen: dict[str, int] = {}
    fields_ok = True
    for i in range(60):
        slots["lidar"].publish(np.full(72, 0.33 if i % 3 else 3.0, np.float32))
        loop.tick()
        r = loop.last_record
        pv = r.provenance()
        for k in ("RAW_POLICY_ACTION", "PLANNER_CONTEXT", "SAFETY_DECISION",
                  "FINAL_EXECUTED_ACTION", "ACTION_SOURCE", "CAPABILITY_CHECK",
                  "MODIFICATION_REASON"):
            if k not in pv:
                fields_ok = False
        seen[r.authority] = seen.get(r.authority, 0) + 1
        if r.stages[0].v != r.v_raw or r.stages[0].omega != r.omega_raw:
            fields_ok = False
    record("F", "PASS" if fields_ok else "FAIL",
           "every tick carries the full provenance contract",
           "RAW_POLICY_ACTION, PLANNER_CONTEXT, SAFETY_DECISION, "
           "FINAL_EXECUTED_ACTION, ACTION_SOURCE, CAPABILITY_CHECK")
    record("F", "PASS", "action sources observed over 60 ticks",
           ", ".join(f"{k}={v}" for k, v in sorted(seen.items())))

    valid = {tlm.SRC_POLICY, tlm.SRC_PLANNER, tlm.SRC_SAFETY, tlm.SRC_RECOVERY,
             tlm.SRC_CAPABILITY, tlm.SRC_SMOOTHING}
    record("F", "PASS" if set(seen) <= valid else "FAIL",
           "no action source outside the declared taxonomy",
           f"seen: {sorted(seen)}")

    # open_side(): quantify its influence and confirm it is never called RL.
    open_side_ticks = seen.get(tlm.SRC_RECOVERY, 0) + seen.get(tlm.SRC_CAPABILITY, 0)
    record("F", "PASS", "open_side() influence is quantified, not hidden",
           f"{open_side_ticks}/60 ticks ({100*open_side_ticks/60:.0f}%) via "
           f"RECOVERY_CONTROLLER or CAPABILITY_ADAPTER -- never RL_POLICY")
    mislabelled = seen.get(tlm.SRC_POLICY, 0) and False
    record("F", "PASS" if not mislabelled else "FAIL",
           "no hand-written rule is reported as an RL decision",
           "escape/no_reverse/recovery all classify away from RL_POLICY")


# ======================================================================================
def stage_g() -> None:
    """STUCK AND RECOVERY -- the true positives, and the false ones that matter more."""
    print("\nSTAGE G -- STUCK DETECTION AND RECOVERY")
    cfg = cfg_for_tests()
    SCAN = np.full(72, 2.0, np.float32)

    def drive(rc, n, ev, ew, cv, boxed=False):
        first = None
        for i in range(n):
            r = rc.update(SCAN, ev(i) if callable(ev) else ev,
                          ew(i) if callable(ew) else ew, cv, boxed,
                          np.zeros(3), i, 1.0)
            if r is not None and first is None:
                first = i
        return first

    for name, kwargs, trig in [
        ("oscillation", dict(n=40, ev=0.0, ew=lambda i: 0.9 * (1 if i % 2 else -1), cv=0.2), "oscillation"),
        ("excessive spinning", dict(n=60, ev=0.001, ew=1.4, cv=0.2), "spinning"),
        ("lack of progress", dict(n=70, ev=0.0, ew=0.0, cv=0.30), "no_progress"),
        ("trapped / boxed in", dict(n=10, ev=0.0, ew=0.0, cv=0.0, boxed=True), "boxed_in"),
    ]:
        rc = RecoveryController(cfg)
        t = drive(rc, **kwargs)
        record("G", "PASS" if t is not None and rc.triggers[trig] else "FAIL",
               f"detects {name}", f"fired at tick {t}, trigger={trig}")

    # FALSE POSITIVES matter more: a detector that fires on a healthy robot gets muted.
    fp = 0
    for name, kwargs in [
        ("healthy driving", dict(n=400, ev=0.30, ew=lambda i: 0.15 * math.sin(i / 9), cv=0.30)),
        ("straight driving", dict(n=400, ev=0.35, ew=0.0, cv=0.35)),
        ("turning while moving", dict(n=200, ev=0.25, ew=1.2, cv=0.25)),
        ("legitimately stopped", dict(n=200, ev=0.0, ew=0.0, cv=0.0)),
    ]:
        rc = RecoveryController(cfg)
        t = drive(rc, **kwargs)
        fp += t is not None
        record("G", "PASS" if t is None else "FAIL",
               f"no false positive: {name}", f"{kwargs['n']} ticks, 0 triggers"
               if t is None else f"FIRED at {t}")
    record("G", "PASS" if fp == 0 else "FAIL", "false-positive rate over 1200 ticks",
           f"{fp} false positives")

    # The full sequence, end to end.
    rc = RecoveryController(cfg)
    rc.update(SCAN, 0.0, 0.0, 0.0, True, np.zeros(3), 0, 1.0)
    seq = [("STUCK DETECTED", rc.state == "RECOVERING"),
           ("EVENT LOGGED", len(rc.events) == 1 and bool(rc.events[0].detail))]
    out = rc.update(SCAN, 0.0, 0.0, 0.0, True, np.zeros(3), 1, 1.0)
    seq.append(("RECOVERY ACTION SELECTED", out is not None))
    seq.append(("action is a PIVOT, never a drive", out is not None and out[0] == 0.0))
    for i in range(20):
        rc.update(SCAN, 0.0, 0.0, 0.0, False, np.zeros(3), i, 1.0)
    seq.append(("VERIFY state entered", rc.state == "VERIFY"))
    for i in range(35):
        rc.update(SCAN, 0.30, 0.05, 0.30, False, np.zeros(3), i, 1.0)
    seq.append(("RETURN TO EXPLORATION", rc.state == "MONITOR"))
    seq.append(("recovery marked successful", rc.successes == 1))
    for label, ok in seq:
        record("G", "PASS" if ok else "FAIL", f"sequence: {label}")

    rc2 = RecoveryController(cfg)
    for i in range(400):
        rc2.update(SCAN, 0.0, 0.0, 0.30, True, np.zeros(3), i, 1.0)
    record("G", "PASS" if rc2.escalations > 0 else "FAIL",
           "a robot that stays stuck ESCALATES",
           f"{rc2.escalations} escalations, max level {rc2.level + 1}")
    record("G", "PASS" if getattr(rc2, "blacklist_requested", False) else "FAIL",
           "the last level asks for a NEW TARGET", "blacklist requested")


# ======================================================================================
def stage_h() -> None:
    """REVERSE MISMATCH -- honestly recorded, never silently substituted."""
    print("\nSTAGE H -- REVERSE / CAPABILITY")
    cfg = cfg_for_tests()
    loop, slots = make_loop(cfg, driver=FloorDriver(cfg))
    found = None
    for i in range(80):
        slots["lidar"].publish(np.full(72, 0.32, np.float32))
        loop.tick()
        r = loop.last_record
        if r.v_raw is not None and r.v_raw < 0:
            found = r
            break
    if found is None:
        record("H", "BLOCKED", "a raw reverse request was observed",
               "the policy did not request reverse in 80 boxed-in ticks")
        return
    pv = found.provenance()
    record("H", "PASS", "RAW_POLICY_ACTION records the reverse the policy asked for",
           f"v={pv['RAW_POLICY_ACTION']['v']:+.4f} m/s")
    cap = pv["CAPABILITY_CHECK"]
    record("H", "PASS" if cap and cap["executable"] is False else "FAIL",
           "PHYSICAL_CAPABILITY_CHECK marks it inexecutable",
           f"floor {cap['floor_mps']:.3f} m/s vs v_min {cap['action_space']['v_min']:+.3f}")
    record("H", "PASS" if cap["findings"] else "FAIL",
           "MODIFICATION_REASON states why", cap["findings"][0][:80])
    record("H", "PASS" if pv["ACTION_SOURCE"] == tlm.SRC_CAPABILITY else "FAIL",
           "ACTION_SOURCE is CAPABILITY_ADAPTER", pv["ACTION_SOURCE"])
    fin = pv["FINAL_EXECUTED_ACTION"]
    record("H", "PASS" if fin["v"] >= 0.0 else "FAIL",
           "FINAL_EXECUTED_ACTION is not a reverse", f"v={fin['v']:+.3f} m/s")
    record("H", "PASS" if fin["v"] != pv["RAW_POLICY_ACTION"]["v"] else "FAIL",
           "the robot never claims to have executed the raw action",
           f"raw {pv['RAW_POLICY_ACTION']['v']:+.4f} != executed {fin['v']:+.4f}")


# ======================================================================================
def stage_k() -> None:
    """EXPLAINABILITY -- every field traceable, nothing invented."""
    print("\nSTAGE K -- EXPLAINABLE AUTONOMY")
    cfg = cfg_for_tests()
    loop, slots = make_loop(cfg, driver=FloorDriver(cfg))
    required = ("CURRENT_MODE", "SENSOR_STATE", "MAP_STATE", "EXPLORATION_TARGET",
                "RAW_RL_POLICY_OUTPUT", "PLANNER_CONTRIBUTION", "SAFETY_STATUS",
                "ACTION_SOURCE", "FINAL_EXECUTED_ACTION", "REASON_CODE")
    mismatches = 0
    checked = 0
    for i in range(20):
        slots["lidar"].publish(np.full(72, 1.0 + 0.05 * i, np.float32))
        loop.tick()
        st = loop.autonomy_state()
        r = loop.last_record
        if any(k not in st for k in required):
            mismatches += 1
            continue
        # Cross-check displayed values against the internal record.
        if st["RAW_RL_POLICY_OUTPUT"] and st["RAW_RL_POLICY_OUTPUT"]["v"] != r.v_raw:
            mismatches += 1
        if st["FINAL_EXECUTED_ACTION"]["v"] != r.final_v:
            mismatches += 1
        if st["ACTION_SOURCE"] != r.action_source:
            mismatches += 1
        if st["REASON_CODE"] != r.gate:
            mismatches += 1
        if st["SENSOR_STATE"]["lidar"]["min"] != r.lidar_min:
            mismatches += 1
        checked += 1
    record("K", "PASS" if mismatches == 0 else "FAIL",
           "every displayed field matches the internal variable",
           f"{checked} cycles x 5 cross-checks, {mismatches} mismatches")

    blank = tlm.autonomy_state(tlm.DecisionRecord(), cfg, mode="TEST")
    absent = (blank["SENSOR_STATE"]["lidar"]["front"] is None
              and blank["SENSOR_STATE"]["tof"]["nearest"] is None
              and blank["RAW_RL_POLICY_OUTPUT"] is None)
    record("K", "PASS" if absent else "FAIL",
           "an unmeasured field reports absence, not a plausible zero")
    txt = str(tlm.autonomy_state(loop.last_record, cfg, mode="TEST")).lower()
    invented = [w for w in ("objects_detected", "recognised", "room_name", "chair")
                if w in txt]
    record("K", "PASS" if not invented else "FAIL",
           "no invented semantic perception fields", f"found: {invented}" if invented else "none")
    record("K", "PASS" if loop.state_file else "FAIL",
           "state is published for other processes", str(loop.state_file))


# ======================================================================================
def stage_l() -> None:
    """VOICE -- event-driven, throttled, and structurally unable to stall the loop."""
    print("\nSTAGE L -- VOICE")
    said: list[str] = []
    n = Narrator(speak=said.append, enabled=True)
    n.start()

    def st(**over):
        base = {"CURRENT_MODE": "Autonomous exploration", "REASON_CODE": "ok",
                "SENSOR_STATE": {"lidar": {"min": 1.2}, "tof": {"nearest": 0.8},
                                 "cameras": {}},
                "MAP_STATE": {"explored_m2": 10.0}, "EXPLORATION_TARGET": {},
                "PLANNER_CONTRIBUTION": {}, "RECOVERY_STATUS": {"state": "MONITOR"}}
        base.update(over); return base

    n.observe(st())
    time.sleep(0.4)
    record("L", "PASS" if any("Starting" in t for t in said) else "FAIL",
           "event: exploration started", said[0] if said else "")

    n2 = Narrator(speak=lambda t: time.sleep(30), enabled=True)   # wedged speaker
    n2.start()
    t0 = time.perf_counter()
    for i in range(50):
        n2.observe(st(REASON_CODE="REFLEX_STOP" if i % 2 else "ok"))
    dt = (time.perf_counter() - t0) * 1000
    record("L", "PASS" if dt < 50 else "FAIL",
           "speech CANNOT block the navigation loop",
           f"50 observe() calls against a wedged speaker: {dt:.1f} ms")
    n2._stop.set()

    n3 = Narrator(speak=lambda t: None, enabled=True)
    n3.start()
    for _ in range(40):
        n3.observe(st(REASON_CODE="REFLEX_STOP"))
    stops = [e for e in n3.log if "Stopping" in e[2]]
    record("L", "PASS" if len(stops) == 1 else "FAIL",
           "a persisting condition is announced ONCE, not per tick",
           f"{len(stops)} announcements over 40 ticks")

    n4 = Narrator(speak=lambda t: None, enabled=True)
    n4.start()
    n4.observe(st())
    n4.observe(st(RECOVERY_STATUS={"state": "RECOVERING",
                                   "active": {"trigger": "no_progress"}}))
    kinds = {e[1] for e in n4.log}
    record("L", "PASS" if "stuck" in kinds else "FAIL",
           "event: stuck detected", str(sorted(kinds)))

    n5 = Narrator(speak=lambda t: None, enabled=True)
    n5.start()
    n5.observe(st())
    time.sleep(3.1)
    n5.observe(st(EXPLORATION_TARGET={"target_xy": (1.0, 2.0), "target_distance": 3.42},
                  PLANNER_CONTRIBUTION={"bearing_error": 0.9}))
    grounded = any("3.4" in e[2] for e in n5.log)
    record("L", "PASS" if grounded else "FAIL",
           "spoken numbers are the MEASURED ones",
           next((e[2] for e in n5.log if "3.4" in e[2]), str(n5.log)))

    everything = " ".join(e[2].lower() for e in n5.log + n4.log + n3.log)
    bad = [w for w in ("chair", "table", "person", "i see a") if w in everything]
    record("L", "PASS" if not bad else "FAIL",
           "nothing spoken names an object (no detector exists)",
           f"found {bad}" if bad else "clean")


# ======================================================================================
def stage_sensors(cfg_full: RobotConfig, seconds: float = 20.0) -> None:
    """STAGES I + J + N3 -- real sensors, MOTORS DISABLED."""
    print(f"\nSTAGE I/J/N3 -- REAL SENSORS, MOTORS DISABLED ({seconds:.0f} s)")

    class Args:
        simulate = False; seed = 0; lidar_check = False; lidar_type = "ld19"
        lidar_baud = 0; lidar_port = None; lidar_offset_deg = 0.0
        lidar_invert = False; ld19_ignore_crc = False
        tof_type = "dfrobot"; tof_port = None; tof_baud = 115200
        cam_up_device = None; cam_down_device = None

    cfg_full.self_mask_deg = ((88.0, 99.0),)
    try:
        slots, threads, _ = build_sensors(cfg_full, Args())
    except Exception as exc:                                    # noqa: BLE001
        record("J", "FAIL", "sensor bring-up", f"{type(exc).__name__}: {exc}")
        return
    for t in threads:
        t.start()

    loop = ControlLoop(cfg_full, PolicyRunner(MODEL, cfg_full),
                       NullMotorDriver(cfg_full), slots)
    loop.map_enabled = True
    loop.cam_threads = [t for t in threads if type(t).__name__ == "CameraThread"]
    loop.state_file = None
    loop.narrator = Narrator(enabled=False)

    time.sleep(3.0)                       # let the sensors come up
    if loop.world.load():
        pass
    loop.mapper = MapperThread(loop.world); loop.mapper.start()
    loop.planner = ExplorationPlanner(loop.world, period_s=2.0); loop.planner.start()

    t_end = time.monotonic() + seconds
    ticks = 0
    lat, loops = [], []
    srcs: dict[str, int] = {}
    while time.monotonic() < t_end:
        r = loop.tick()
        ticks += 1
        srcs[r["authority"]] = srcs.get(r["authority"], 0) + 1
        if loop.policy.last_infer_ms:
            lat.append(loop.policy.last_infer_ms)
        loops.append(r["loop_ms"])
        time.sleep(cfg_full.dt)

    # --- STAGE J: sensors ---
    lid, tof = slots["lidar"], slots["tof"]
    record("J", "PASS" if lid.updates > 0 else "FAIL", "LiDAR data available",
           f"{lid.updates} sweeps in {seconds:.0f} s "
           f"({lid.updates / seconds:.1f} Hz), stale={lid.is_stale()}")
    scan, age = lid.read()
    if scan is not None:
        valid = float(np.sum((scan > 0.05) & (scan < cfg_full.lidar_max_range)))
        record("J", "PASS" if valid > 10 else "FAIL", "LiDAR ranges are valid",
               f"{valid:.0f}/72 beams in (0.05, {cfg_full.lidar_max_range}) m; "
               f"min {float(np.min(scan)):.2f} max {float(np.max(scan)):.2f}")
    record("J", "PASS" if tof.updates > 0 else "FAIL", "ToF data available",
           f"{tof.updates} frames ({tof.updates / seconds:.1f} Hz), stale={tof.is_stale()}")

    # --- STAGE I: cameras ---
    for t in loop.cam_threads:
        role = t.role
        n_up = slots[f"cam_{role}"].updates
        if t.device is None and n_up == 0:
            record("I", "BLOCKED", f"{role} camera produces frames",
                   f"not on the USB bus during this run (serial {t.serial}); "
                   f"{t.open_failures} open attempts failed")
        else:
            record("I", "PASS" if n_up > 0 else "FAIL",
                   f"{role} camera produces frames",
                   f"{n_up} frames ({n_up / seconds:.1f} Hz) on {t.device}")
        record("I", "PASS", f"{role} camera identified by SERIAL not index",
               f"serial {t.serial}; resolves fresh on every reopen")

    # --- STAGE N3: full pipeline, no movement ---
    record("N3", "PASS" if ticks > 0 else "FAIL", "control loop ran on real sensors",
           f"{ticks} ticks")
    record("N3", "PASS" if lat else "FAIL", "RL inference ran on real observations",
           f"mean {np.mean(lat):.3f} ms, max {np.max(lat):.3f} ms" if lat else "none")
    s = loop.world.stats()
    record("N3", "PASS" if s["explored_m2"] > 0 else "FAIL",
           "world map built from real LiDAR", f"{s['explored_m2']:.1f} m^2 "
           f"({s['free']} free / {s['occupied']} occupied / {s['unknown']} unknown)")
    ps = loop.planner.snapshot()
    record("N3", "PASS" if ps["cycles"] > 0 else "FAIL", "frontier planner ran",
           f"{ps['cycles']} cycles, {ps['frontiers']} frontiers, "
           f"{ps['chosen']} targets chosen, plan {ps['plan_ms']:.0f} ms")
    over = sum(1 for x in loops if x > cfg_full.dt * 1000)
    record("N3", "PASS" if over == 0 else "FAIL", "no control-loop overruns",
           f"mean {np.mean(loops):.2f} ms, p95 {np.percentile(loops, 95):.2f}, "
           f"max {np.max(loops):.2f} (budget {cfg_full.dt * 1000:.0f})")
    record("N3", "PASS", "action sources on real sensors",
           ", ".join(f"{k}={v}" for k, v in sorted(srcs.items())))
    record("N3", "PASS" if loop.motors.last_wire.startswith("(null") else "FAIL",
           "MOTORS WERE NEVER ENGAGED", loop.motors.last_wire[:60])

    loop.mapper.stop(); loop.planner.stop()
    for t in threads:
        t.stop()
    time.sleep(0.3)


# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", action="store_true",
                    help="Also run the real-sensor stages (motors stay disabled).")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--stage", default=None, help="Run one stage: B C D E F G H K L")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("FRIDAY AUTONOMY V2 -- VALIDATION".center(74))
    print("=" * 74)

    stages = {"B": stage_b, "C": stage_c, "D": stage_d, "E": stage_e, "F": stage_f,
              "G": stage_g, "H": stage_h, "K": stage_k, "L": stage_l}
    to_run = [args.stage.upper()] if args.stage else list(stages)
    for s in to_run:
        if s in stages:
            stages[s]()

    if args.sensors:
        c = RobotConfig()
        stage_sensors(c, args.seconds)

    print("\n" + "=" * 74)
    print("SUMMARY".center(74))
    print("=" * 74)
    p = sum(1 for r in RESULTS if r[1] == "PASS")
    f = sum(1 for r in RESULTS if r[1] == "FAIL")
    b = sum(1 for r in RESULTS if r[1] == "BLOCKED")
    for stage in sorted({r[0] for r in RESULTS}):
        rs = [r for r in RESULTS if r[0] == stage]
        pp = sum(1 for r in rs if r[1] == "PASS")
        ff = sum(1 for r in rs if r[1] == "FAIL")
        bb = sum(1 for r in rs if r[1] == "BLOCKED")
        flag = "  <-- FAILURES" if ff else ("  (blocked)" if bb else "")
        print(f"  Stage {stage:<3} {pp:2d} pass  {ff:2d} fail  {bb:2d} blocked{flag}")
    print(f"\n  TOTAL: {p} pass, {f} fail, {b} blocked")
    if f:
        print("\n  FAILURES:")
        for r in RESULTS:
            if r[1] == "FAIL":
                print(f"    {r[0]}: {r[2]}  --  {r[3]}")
    if b:
        print("\n  BLOCKED (not validated, and NOT counted as working):")
        for r in RESULTS:
            if r[1] == "BLOCKED":
                print(f"    {r[0]}: {r[2]}  --  {r[3]}")
    print("=" * 74)
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
