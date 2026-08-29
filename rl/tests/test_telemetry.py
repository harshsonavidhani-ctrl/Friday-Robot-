"""The raw policy output must survive every stage that follows it.

The first autonomous run's black box could not answer "did my trained policy drive
this robot", because `ControlLoop.tick` overwrote `v_pol`/`omega_pol` in the
no-reverse fallback BEFORE recording them. 34 of 300 ticks whose heading came from
the `_open_side()` rule were logged under the policy's name at exactly +-omega_max.

Every assertion here exists to stop that specific class of defect coming back: a
number that is presented as the network's answer while being something else's.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import telemetry as tlm
from pi_deploy import (BlackBox, ControlLoop, HERE, NullMotorDriver, PolicyRunner,
                       RobotConfig, SensorSlot)

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class NoReverseDriver(NullMotorDriver):
    """A stiction floor, so the no-reverse substitution is reachable."""

    def min_speed_mps(self) -> float:
        return 0.179


def make_loop(driver=None):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg)
    loop = ControlLoop(cfg, policy, driver or NullMotorDriver(cfg), slots)
    return loop, slots


def boxed_scan():
    """Everything close: forward forbidden, and the policy wants out."""
    return np.full(cfg.n_beams, 0.33, np.float32)


def open_scan():
    return np.full(cfg.n_beams, 4.5, np.float32)


print("--- telemetry / decision provenance ---")

# 1. THE CENTRAL GUARANTEE. Drive into a box until a rule substitutes, and confirm
#    the recorded raw output is still the network's, not the rule's.
loop, slots = make_loop(driver=NoReverseDriver(cfg))
subs = 0
for _ in range(40):
    slots["lidar"].publish(boxed_scan())
    loop.tick()
    r = loop.last_record
    if r is not None and any(s.changed for s in r.stages if s.name in tlm.RULE_STAGES):
        subs += 1
        # The raw pair must equal what the policy stage recorded -- the first entry
        # in the chain -- and must NOT have been replaced by the rule's +-omega_max.
        pol = r.stages[0]
        if abs(pol.v - r.v_raw) > 1e-9 or abs(pol.omega - r.omega_raw) > 1e-9:
            check("raw output survived a rule substitution", False,
                  f"stage0=({pol.v:+.3f},{pol.omega:+.3f}) raw=({r.v_raw:+.3f},{r.omega_raw:+.3f})")
            break
else:
    check("raw output survived every rule substitution", True, f"{subs} substitutions seen")
check("a rule substitution actually occurred (otherwise the test proves nothing)",
      subs > 0, f"{subs} ticks")

# 2. When a rule substitutes, authority must say RULE -- not POLICY.
rule_ticks = []
loop, slots = make_loop(driver=NoReverseDriver(cfg))
for _ in range(40):
    slots["lidar"].publish(boxed_scan())
    loop.tick()
    if loop.last_record is not None:
        rule_ticks.append(loop.last_record.authority)
check("an unexecutable-reverse tick reports CAPABILITY_ADAPTER, never RL_POLICY",
      tlm.SRC_CAPABILITY in rule_ticks,
      f"authorities seen: {sorted(set(rule_ticks))}")

# 3. The slew limiter must NOT count as a safety override. It fires on almost every
#    acceleration; counted as an override it would report YES nearly always and the
#    field would stop meaning anything.
rec = tlm.DecisionRecord(v_raw=0.4, omega_raw=0.1)
rec.add("policy", 0.4, 0.1)
rec.add("governor", 0.4, 0.1)
rec.add("limiter", 0.08, 0.1)          # speed shaped, heading untouched
check("limiter is not a safety override", rec.authority == tlm.SRC_POLICY, rec.authority)
check("limiter is reported separately", rec.rate_limited is True)
check("limiter kind is MOTION_SMOOTHING", rec.stages[-1].kind == tlm.SRC_SMOOTHING)

# 4. A genuine safety cut MUST be reported as one.
rec2 = tlm.DecisionRecord(v_raw=0.4, omega_raw=0.1)
rec2.add("policy", 0.4, 0.1)
rec2.add("governor", 0.05, 0.1, "clearance 0.40 m")
rec2.add("limiter", 0.05, 0.1)
check("a governor cut reports SAFETY_OVERRIDE", rec2.authority == tlm.SRC_SAFETY,
      rec2.authority)

# 5. NO FABRICATED VALUES. An unmeasured field must render as absence, never as 0.00.
blank = tlm.DecisionRecord()
txt = tlm.autonomy_status(blank, cfg, mode="TEST")
check("an unreported ToF renders as absence, not 0.00 m",
      "not reported" in txt and "nearest obstacle 0.00" not in txt)
check("zero cameras is stated explicitly", "NO FRAMES" in txt)

# 6. The black box schema must carry the raw columns, and at the SAME precision as
#    v_policy -- at 3 dp against 4, every tick reads as a substitution.
check("blackbox schema has v_raw/omega_raw",
      "v_raw" in BlackBox.FIELDS and "omega_raw" in BlackBox.FIELDS)
check("blackbox schema has authority", "authority" in BlackBox.FIELDS)

loop, slots = make_loop()
for _ in range(5):
    slots["lidar"].publish(open_scan())
    loop.tick()
rows = list(loop.blackbox.buf)
i_pol, i_raw = BlackBox.FIELDS.index("v_policy"), BlackBox.FIELDS.index("v_raw")
same_precision = all(
    abs(r[i_pol] - r[i_raw]) < 1e-9 for r in rows
)   # open space: no substitution, so the two columns must agree EXACTLY
check("v_policy and v_raw agree exactly when no rule fires (matched precision)",
      same_precision, f"{len(rows)} rows")

# 7. Sector clearances must be orientation-correct: beam 0 is forward.
scan = np.full(cfg.n_beams, 5.0, np.float32)
scan[0] = 0.5                                    # something directly ahead
cl = tlm.clearances(scan)
check("an obstacle at beam 0 reads as FRONT", cl["front"] < 1.0, f"front={cl['front']:.2f}")
check("...and not as left/right/rear",
      min(cl["left"], cl["right"], cl["rear"]) > 4.0)
scan2 = np.full(cfg.n_beams, 5.0, np.float32)
scan2[cfg.n_beams // 4] = 0.5                    # 90 deg CCW = left
check("an obstacle at n/4 reads as LEFT", tlm.clearances(scan2)["left"] < 1.0)

# 8. Action naming must not call a pivot "forward".
check("pivot is named a pivot", tlm.action_name(0.0, 1.2) == "PIVOT_LEFT")
check("stop is named a stop", tlm.action_name(0.0, 0.0) == "STOP")
check("an arc is named an arc", tlm.action_name(0.3, -0.9) == "FORWARD_ARC_RIGHT")

# 9. ACTION SOURCES must be five DISTINCT roles. Recovery and capability adaptation
#    used to share the label "RULE", and they fail differently: `escape` fires because
#    the ROBOT is stuck, `no_reverse` because the COMMAND is unexecutable. A robot
#    reporting "recovery" while merely translating a reverse sends you hunting for a
#    trap that is not there.
srcs = {tlm.SRC_POLICY, tlm.SRC_PLANNER, tlm.SRC_SAFETY, tlm.SRC_RECOVERY,
        tlm.SRC_CAPABILITY, tlm.SRC_SMOOTHING}
check("the six action sources are all distinct", len(srcs) == 6, str(sorted(srcs)))
check("escape is RECOVERY_CONTROLLER",
      tlm.Stage("escape", 0, 0).kind == tlm.SRC_RECOVERY)
check("no_reverse is CAPABILITY_ADAPTER, not recovery",
      tlm.Stage("no_reverse", 0, 0).kind == tlm.SRC_CAPABILITY)
check("planner is EXPLORATION_PLANNER",
      tlm.Stage("planner", 0, 0).kind == tlm.SRC_PLANNER)
check("governor is SAFETY_OVERRIDE",
      tlm.Stage("governor", 0, 0).kind == tlm.SRC_SAFETY)

# 10. THE PROVENANCE CONTRACT. Every executed action must carry the full record.
rec3 = tlm.DecisionRecord(v_raw=0.40, omega_raw=-0.20)
rec3.planner = {"bearing_error": 0.9, "bias": 0.3, "reason": "target is 52 deg left"}
rec3.add("policy", 0.40, -0.20, "network output")
rec3.add("planner", 0.40, 0.10, "target is 52 deg left")
rec3.add("governor", 0.12, 0.10, "clearance 0.44 m")
rec3.add("limiter", 0.08, 0.10, "slew limited")
rec3.final_v, rec3.final_omega = 0.08, 0.10
rec3.wire = "M100 60"
p = rec3.provenance()
for key in ("RAW_POLICY_ACTION", "PLANNER_CONTEXT", "SAFETY_DECISION",
            "FINAL_EXECUTED_ACTION", "ACTION_SOURCE", "MODIFICATION_REASON"):
    check(f"provenance carries {key}", key in p)
check("RAW_POLICY_ACTION is the NETWORK's, not the last stage's",
      p["RAW_POLICY_ACTION"]["v"] == 0.40 and p["RAW_POLICY_ACTION"]["omega"] == -0.20,
      str(p["RAW_POLICY_ACTION"]))
check("SAFETY_DECISION records the override", p["SAFETY_DECISION"]["overridden"] is True)
check("...and names the stage", "governor" in p["SAFETY_DECISION"]["stages"])
check("FINAL_EXECUTED_ACTION carries the literal wire command",
      p["FINAL_EXECUTED_ACTION"]["wire"] == "M100 60")
check("ACTION_SOURCE is the LAST non-smoothing change (the governor)",
      p["ACTION_SOURCE"] == tlm.SRC_SAFETY, p["ACTION_SOURCE"])
check("MODIFICATION_REASON is the measurement that caused it",
      "clearance" in p["MODIFICATION_REASON"], p["MODIFICATION_REASON"])

# With no safety cut, the planner must own the action.
rec4 = tlm.DecisionRecord(v_raw=0.4, omega_raw=-0.2)
rec4.add("policy", 0.4, -0.2)
rec4.add("planner", 0.4, 0.1, "target left")
rec4.add("limiter", 0.4, 0.1)
check("an unvetoed planner tick is EXPLORATION_PLANNER",
      rec4.provenance()["ACTION_SOURCE"] == tlm.SRC_PLANNER)
check("...and reports no safety override",
      rec4.provenance()["SAFETY_DECISION"]["overridden"] is False)

# 11. THE AUTONOMY STATE INTERFACE. Every field Phase 7 requires must be present, and
#     an unmeasured one must be None rather than a plausible number.
st = tlm.autonomy_state(rec3, cfg, mode="TEST",
                        planner={"has_target": True, "target_xy": (1.0, 2.0)},
                        recovery={"state": "MONITOR"}, objective="obj",
                        stats={"ticks": 7})
for key in ("CURRENT_MODE", "SENSOR_STATE", "MAP_STATE", "EXPLORATION_TARGET",
            "RAW_RL_POLICY_OUTPUT", "PLANNER_CONTRIBUTION", "SAFETY_STATUS",
            "ACTION_SOURCE", "FINAL_EXECUTED_ACTION", "REASON_CODE"):
    check(f"autonomy state exposes {key}", key in st)
check("RAW_RL_POLICY_OUTPUT is the network's own numbers",
      st["RAW_RL_POLICY_OUTPUT"]["v"] == 0.40)
check("...and carries the model name and inference time",
      "model" in st["RAW_RL_POLICY_OUTPUT"] and "inference_ms" in st["RAW_RL_POLICY_OUTPUT"])
check("REASON_CODE is a code, and REASON is prose", st["REASON_CODE"] == rec3.gate
      and isinstance(st["REASON"], str))

# An empty record must yield absence, not zeros. A state interface that reports 0.00 m
# for a sensor that has never spoken is worse than one that reports nothing, because
# whatever reads it cannot tell the two apart.
empty = tlm.autonomy_state(tlm.DecisionRecord(), cfg, mode="TEST")
check("an unmeasured LiDAR reports None, not 0.0",
      empty["SENSOR_STATE"]["lidar"]["front"] is None)
check("an unmeasured ToF reports None, not 0.0",
      empty["SENSOR_STATE"]["tof"]["nearest"] is None)
check("no policy output reports None, not a zero action",
      empty["RAW_RL_POLICY_OUTPUT"] is None)
check("the state interface invents no semantic perception fields",
      not any(k in str(st).lower() for k in ("objects_detected", "room_name",
                                             "recognised")))
check("every reason code has a description",
      set(tlm.REASON_CODES) >= {"ok", "planner", "governor", "REFLEX_STOP",
                                "recovery", "no_reverse_turn", "escape_turn"})

# 12. CAPABILITY CHECKING (Phase 9). The action space and this drivetrain do not
#     overlap on the reverse axis AT ALL: v_min is -0.15 m/s and the stiction floor is
#     0.179, so every reverse the policy can ask for is below the slowest speed the
#     wheels turn. The check must say so, and must run on every tick -- "checked and
#     executable" and "never checked" look identical in a log otherwise.
cap_bad = tlm.capability_check(-0.07, 0.3, 0.179, cfg.v_min, cfg.v_max, cfg.omega_max)
check("a sub-floor reverse is reported INEXECUTABLE", cap_bad["executable"] is False)
check("...with the measured floor in the finding",
      any("0.179" in f for f in cap_bad["findings"]))
check("...and the fact that NO reverse is executable at all",
      any("NO reverse" in f for f in cap_bad["findings"]))
check("the check records what was requested",
      cap_bad["requested"]["v"] == -0.07)
check("...and the action space it was judged against",
      cap_bad["action_space"]["v_min"] == cfg.v_min)

cap_ok = tlm.capability_check(0.35, 0.2, 0.179, cfg.v_min, cfg.v_max, cfg.omega_max)
check("a normal forward command is executable", cap_ok["executable"] is True)
check("...with nothing to report", cap_ok["findings"] == [])

cap_creep = tlm.capability_check(0.08, 0.0, 0.179, cfg.v_min, cfg.v_max, cfg.omega_max)
check("a sub-floor FORWARD is executable (creep pulsing handles it)",
      cap_creep["executable"] is True)
check("...but is still reported, since it makes executed != commanded",
      any("creep" in f for f in cap_creep["findings"]))

# A driver with no floor (the null driver, or hardware that can reverse) must not be
# told it cannot reverse.
cap_free = tlm.capability_check(-0.10, 0.0, 0.0, cfg.v_min, cfg.v_max, cfg.omega_max)
check("a drivetrain with no floor CAN reverse", cap_free["executable"] is True)

# 13. The provenance contract must carry the capability check (Phase 9's requirement).
rec5 = tlm.DecisionRecord(v_raw=-0.07, omega_raw=0.2)
rec5.capability = cap_bad
rec5.add("policy", -0.07, 0.2)
rec5.add("no_reverse", 0.0, 1.6, "reverse not executable")
p5 = rec5.provenance()
check("provenance carries CAPABILITY_CHECK", "CAPABILITY_CHECK" in p5)
check("...and RAW_POLICY_ACTION still shows the REVERSE the policy asked for",
      p5["RAW_POLICY_ACTION"]["v"] == -0.07)
check("...while ACTION_SOURCE says the adapter acted",
      p5["ACTION_SOURCE"] == tlm.SRC_CAPABILITY)
check("the autonomy state exposes CAPABILITY_CHECK too",
      "CAPABILITY_CHECK" in tlm.autonomy_state(rec5, cfg, mode="TEST"))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
