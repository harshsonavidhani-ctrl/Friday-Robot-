#!/usr/bin/env python3
"""File 21, the fixes after the first real mission (2026-10-02).

    ./venv/bin/python scripts/test_file21_fixes.py

What went wrong on the robot, as whole missions in the simulated house:
  1. runs stopped friday-brain, so nothing could hear a room's name
     (rl/modes/confident_explore.env now quiets the face only);
  2. the 2-minute cap ended exploring with frontiers left in the room she was
     in -> no question, the room saved unnamed (the final ask now includes the
     room she is standing in, complete or not, saved as incomplete);
  3. HELP_WAIT could only be left by restarting friday-mcp ("Friday, stop" and
     a new "explore" now end it).
Network blocked; no robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
import config                                           # noqa: E402
config.EXPLORATION_MISSION["label"]["ask_rooms"] = True   # these tests are of the question itself
netguard.block()

from exploration_mission import intents                 # noqa: E402
from exploration_mission import service                 # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


ASK = "I've finished exploring this area. What should I call it?"
ROOM = mm.closed_room()
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


def booted(w: World) -> World:
    w.mgr.boot()
    w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
    return w


# =============================================================================
print("--- 1. runs keep friday-brain up (it hears the answers) ---")
env = (ROOT / "rl" / "modes" / "confident_explore.env").read_text()
quiet = [ln for ln in env.splitlines() if ln.startswith("MODE_QUIET_UNITS=")]
check("confident_explore.env quiets the face only, not friday-brain",
      quiet == ['MODE_QUIET_UNITS="friday-face.service"'], str(quiet))

# =============================================================================
print("--- 2. the time cap ends exploring: she still asks about her room ---")
# Three rooms, a 30 s cap: the run's own timer starts its way back while she
# is still in room A with frontiers left -- last night's ending.
w = booted(World(["kitchen"], world=mm.three_rooms(),
                 cfg={"explore": {"mission_cap_min": 0.5}}))
w.mgr.start()
w.run()
m = w.mission()
asked = w.voice.said.count(ASK)
areas = w.areas()
check("the run's timer ended it (time_cap)", m["stop_reason"] == "time_cap",
      str(m["stop_reason"]))
check("...and she asked about the room she was in before going home",
      asked == 1 and any(a["name"] == "kitchen" for a in areas),
      f"asked {asked}, areas {[(a['name'], a['complete']) for a in areas]}")
from types import SimpleNamespace                       # noqa: E402
from exploration_mission import segmentation as seg     # noqa: E402
lp = seg.LabelParams()
room = SimpleNamespace(key=7, area_m2=9.0, complete=False, area_id=None)
dw = SimpleNamespace(inside_s={7: 3.0}, exited_at={}, current=7)
check("the rule: an INCOMPLETE room she is standing in qualifies at the end "
      "(final ask) ...", seg.why_not(room, dw, 100.0, lp, final=True) == [],
      str(seg.why_not(room, dw, 100.0, lp, final=True)))
check("...but never mid-run (frontiers left = keep exploring it)",
      "not complete" in seg.why_not(room, dw, 100.0, lp, final=False))
other = SimpleNamespace(key=8, area_m2=9.0, complete=False, area_id=None)
check("...and an incomplete room she is NOT in is still not asked at the end",
      "not complete" in seg.why_not(other, dw, 100.0, lp, final=True))
fake = SimpleNamespace(segments=[room])
choice = seg.choose_label_target(fake, dw, 100.0, None, True, lp, ignore_cooldown=True)
check("...and it is recorded as incomplete, not passed off as complete",
      choice is not None and choice.complete is False, str(choice))
check("...the run was HELD for the question (it stood still while she asked)",
      any(o[0] == "hold" and "asking" in str(o) for o in w.sim.orders),
      str([o for o in w.sim.orders if o[0] == "hold"][:3]))
check("...and she still got home", w.mgr.state == "IDLE"
      and m["status"] in ("returned", "arrived"), f"{w.mgr.state} {m['status']}")
said = " ".join(w.voice.said)
check("the summary names the room", "kitchen" in said.split("Exploration complete")[-1]
      or "kitchen" in said, said[-300:])

# =============================================================================
print("--- 3. HELP_WAIT has ways out ---")


def help_world() -> World:
    """A one-room mission whose way back gives up 1.1 m out -> HELP_WAIT.
    (run_first: the shortest way into HELP_WAIT; these are its ways OUT.)"""
    w = booted(World(["study"], world=ROOM, path=ROOM_PATH, fail_return_at_m=1.5,
                     cfg={"return": {"mode": "run_first"}}))
    w.mgr.start()
    w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=400)
    return w


w = help_world()
check("the way back gave up: HELP_WAIT", w.mgr.state == "HELP_WAIT", w.mgr.state)
service._manager, service._router = w.mgr, None
r = service.voice("stop")
w.step()
check("'Friday, stop' in HELP_WAIT -> the mission closes, IDLE",
      r["handled"] and w.mgr.state == "IDLE" and w.mgr.mission is None,
      f"{r} {w.mgr.state}")
check("...the mission stays saved as help_wait (nothing lost)",
      w.mission()["status"] == "help_wait")

w = help_world()
service._manager, service._router = w.mgr, None
w.sim.truth = (3.0, 2.5, 0.0)                           # someone carried her back
n_before = len(w.store.missions("home"))
r = service.voice("explore")
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=60)
check("'explore' in HELP_WAIT -> a NEW mission starts (no restart needed)",
      r["handled"] and len(w.store.missions("home")) == n_before + 1
      and w.mgr.state == "EXPLORE", f"{r} {w.mgr.state}")
w.run()

w = help_world()
service._manager, service._router = w.mgr, None
r = service.start_mission(2)
w.run(lambda x: x.mgr.state in ("MISSION_INIT", "EXPLORE"), max_s=10)
check("the dashboard's Explore button in HELP_WAIT -> a new mission",
      r["ok"] and w.mgr.state in ("MISSION_INIT", "EXPLORE"), f"{r} {w.mgr.state}")

ro = intents.Router(type("A", (), {"__getattr__": lambda s, n: (lambda *a, **k: True)})(),
                    resolve_area=lambda n: None)
check("'explore' is still refused while she is actually exploring",
      ro.handle("explore", intents.Context(state="EXPLORE")) is True)

# =============================================================================
print("--- 4. the run finishes on its own and says nothing (2026-10-02 15:43) ---")
from exploration_mission import adapters as ad          # noqa: E402


def silent_end(how: str, nav=True) -> World:
    """Her start reached by chance as the timer fires: the run's way back goes
    returning -> done in one tick and it exits without publishing it."""
    # Three rooms: she is still mid-way through them when the run vanishes,
    # so nothing but the run's end can end exploring.
    w = booted(World(["kitchen"], world=mm.three_rooms()))
    if nav:
        w.a.nav = ad.FakeNav(during_run=False, clock=w.clock, script=["ARRIVED"])
        w.mgr.a.nav = w.a.nav
    w.mgr.start()
    w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 25, max_s=60)
    assert w.mgr.state == "EXPLORE", w.mgr.state
    w.sim.is_running, w.sim.publishing = False, False   # gone, status still "exploring"
    w.sim.end_how = how
    return w


w = silent_end("self")
w.run(max_s=300)
m = w.mission()
said = " ".join(w.voice.said)
check("finished on its own -> NOT 'aborted': the last question, the summary, home",
      m["status"] in ("returned", "arrived") and ASK in said
      and "Exploration complete" in said and "reached where I started" in said,
      f"{m['status']} | {said[-300:]}")
check("...the room was named ('kitchen'), the stop reason is not a person's",
      any(a["name"] == "kitchen" for a in w.areas())
      and m["stop_reason"] in ("time_cap", "no_frontiers"), str(m["stop_reason"]))
check("...the question was asked with no run to hold (motors already still)",
      not any(e["kind"] == "ask_skipped" for e in m["events"]),
      str([e["kind"] for e in m["events"]][-8:]))

w = silent_end("person")
w.run(max_s=60)
m = w.mission()
check("a person's stop (dashboard Stop / SIGINT) is still 'aborted', no drive back",
      m["status"] == "aborted" and not w.a.nav.calls if hasattr(w.a.nav, "calls") else True,
      m["status"])

w = silent_end("pending")
for _ in range(150):                                    # 15 s: record not written yet
    w.step()
check("its exit record not written yet -> she waits, decides nothing",
      w.mgr.state == "EXPLORE" and w.mission()["status"] == "running", w.mgr.state)
w.sim.end_how = "self"
w.run(max_s=300)
check("...then it says 'completed' -> exploring is over, she gets home",
      w.mission()["status"] in ("returned", "arrived"), w.mission()["status"])

print("--- the run's record, read for real ---")
import json, tempfile, config as _cfg                   # noqa: E401,E402
tmp = Path(tempfile.mkdtemp(prefix="friday-fix4-"))
(tmp / "state").mkdir()
real_root = _cfg.ROOT
_cfg.ROOT = tmp
run = ad.RunAdapter(state_dir=tmp / "state", clock=ad.Clock())
def rec(state, code, ended="2026-10-02T15:43:39+05:30"):
    (tmp / "state" / "explore_mode.json").write_text(json.dumps(
        {"state": state, "exit_code": code, "ended": ended}))
since = __import__("datetime").datetime.fromisoformat("2026-10-02T15:40:41+05:30").timestamp()
rec("completed", 0)
check("explore_mode 'completed' exit 0 -> self", run.ended_how(since) == "self")
rec("failed", 130)
check("'failed' exit 130 (SIGINT, last night's dashboard stop) -> person",
      run.ended_how(since) == "person")
rec("running", None)
check("'running' -> pending", run.ended_how(since) == "pending")
rec("completed", 0, ended="2026-10-01T10:00:00+05:30")
check("a record older than the mission -> person (never trust a stale one)",
      run.ended_how(since) == "person")
_cfg.ROOT = real_root

# =============================================================================
print("--- 5. the active exploration back (Harsh, 2026-10-02) ---")
env = (ROOT / "rl" / "modes" / "confident_explore.env").read_text()
# 16:1x Harsh chose the old drive + unstick off; 23:22 (test #5) unstick off
# brought back the triple lock, so unstick is ON again in keep-trying form.
# 2026-10-04 ("noise more than movement ... it should be smooth"): the QUIET
# drive is back -- the old drive was only chosen on the weak 31 kHz firmware.
check("mode file: QUIET motor drive + steady turns, unstick ON in keep-trying form",
      'MODE_QUIET_MOTORS="1"' in env and 'MODE_STEADY_TURNS="1"' in env
      and 'MODE_UNSTICK="1"' in env
      and 'MODE_UNSTICK_KEEP_TRYING="1"' in env)
import subprocess                                        # noqa: E402
out = subprocess.run(["bash", "-c", f'set -a; . "{ROOT}/rl/modes/confident_explore.env"; '
                      'ARGS=(); if [ "${MODE_QUIET_MOTORS:-1}" = "0" ]; then '
                      'ARGS+=(--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard); fi; '
                      '[ "${MODE_UNSTICK:-1}" = "0" ] && ARGS+=(--no-unstick); '
                      '[ "${MODE_UNSTICK_KEEP_TRYING:-0}" = "1" ] && ARGS+=(--unstick-keep-trying); '
                      '[ "${MODE_STEADY_TURNS:-0}" = "1" ] && ARGS+=(--motor-steady-pivots); '
                      '[ -n "${MODE_START_DEBOUNCE:-}" ] && ARGS+=(--motor-start-debounce "$MODE_START_DEBOUNCE"); '
                      '[ -n "${MODE_TURN_COMMIT:-}" ] && ARGS+=(--turn-commit "$MODE_TURN_COMMIT"); '
                      'echo "${ARGS[*]}"'], capture_output=True, text=True).stdout.strip()
sh = (ROOT / "rl" / "modes" / "explore.sh").read_text()
check("explore.sh keeps the old drive as a switch, and passes the quiet drive "
      "(the run's defaults) and keep-trying unstick to the run",
      "--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard" in sh
      and "--unstick-keep-trying" in sh and "--motor-steady-pivots" in sh
      and "--motor-start-debounce" in sh and "--turn-commit" in sh
      and out == "--unstick-keep-trying --motor-steady-pivots --motor-start-debounce 2 "
      "--turn-commit 1.0", out)
sys.path.insert(0, str(ROOT / "rl"))
import friday_localnav_deploy as L                       # noqa: E402
a = L.build_arg_parser().parse_args(["--motor", "friday", "--i-am-watching"] + out.split())
check("...and the run's own parser takes them (unstick on + keep trying; kick only "
      "after 0.3 s at rest, 3-tick pulses, coast before reversing)",
      not a.no_unstick and a.unstick_keep_trying
      and a.motor_kick_rest == 0.3 and a.motor_pulse_burst == 3
      and a.motor_plug_guard is True and a.motor_steady_pivots is True
      and a.motor_start_debounce == 2 and a.turn_commit == 1.0)

# =============================================================================
print("--- 6. her own voice in an answer capture is not his answer ---")
import asyncio, time as _time, types                     # noqa: E401,E402
import config as _c2                                      # noqa: E402
import main as _main                                      # noqa: E402
from exploration_mission import voice_link as _vl         # noqa: E402
_tmp6 = Path(tempfile.mkdtemp(prefix="friday-fix6-"))
_c2.SPEAKING_FILE = _tmp6 / "speaking.json"
_c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
check("no speech since the capture opened -> not her voice",
      _main._own_voice_since(_time.time() - 1) is False)


class _Stream:
    def capture(self, **kw):                              # the run says a line mid-capture
        _c2.SPEAKING_FILE.write_text(json.dumps({"state": "speaking", "at": _time.time()}))
        return b"audio"


class _Turn:
    def __init__(self, *a): self.notes = []
    async def call(self, name, fn, *a, timeout, detail=""): return fn(*a)
    def stage(self, *a): pass
    def note(self, text, dash=True): self.notes.append(text)
    def answered_by(self, *a): pass
    def done(self, *a): pass
    def listening(self): pass


got6 = []
class _Sess:
    async def call_tool(self, name, args): got6.append(args["text"]); return None
class _Link:
    async def get(self, w): return _Sess()
class _Src:
    def pause(self): pass
    def resume(self): pass
class _Face:
    def set(self, *a): pass
    def rest(self): pass
fake_stt = types.ModuleType("stt"); fake_stt.Silence = type("Silence", (Exception,), {})
fake_stt.transcribe = lambda audio, **kw: "I found an unexplored area 3 metres to the west"
_real_stt = sys.modules.get("stt"); sys.modules["stt"] = fake_stt
import initiative as _ini                                 # noqa: E402
_ini.she_is_talking = lambda: False
_real_turn = _main.stages.Turn; _main.stages.Turn = _Turn
_main._mission_heard = lambda s: None
asyncio.run(_main._mission_turn(None, _Face(), _Src(), _Stream(),
                                _vl.TRIGGER_PREFIX + "abc label 10", _Link()))
_main.stages.Turn = _real_turn
if _real_stt is not None:
    sys.modules["stt"] = _real_stt
check("the run narrated during the capture -> the answer sent back is EMPTY "
      "(silence: she asks again), not her own words", got6 == [""], str(got6))

# =============================================================================
print("--- 7. the run is quiet while the mission asks (2026-10-02 23:00) ---")
sys.path.insert(0, str(ROOT / "rl"))
import friday_voice as _fv                                # noqa: E402
_va = _fv.VoiceAnnouncer(enabled=True, backend="log")
_held = [True]
_va.muted = lambda: _held[0]
check("held: the run's narration is dropped ('I have now mapped 64 square "
      "metres' twice landed in his answer)",
      _va.say("coverage", "I have now mapped 64 square metres.") is False
      and _va.dropped_muted == 1)
_held[0] = False
check("...released: she talks again", _va.say("coverage", "I have now mapped 12.") is True)
_va.muted = lambda: 1 / 0
check("...a broken mute never silences her", _va.say("boot", "Starting.") is True)
_va.close() if hasattr(_va, "close") else None
# 2026-10-04 16:31:37-16:32:04: lines QUEUED while she explored were still said
# ~30 s into the question -- the mute was only checked as a line was queued.
_vb = _fv.VoiceAnnouncer(enabled=True, backend="log", min_gap_s=0.5)
_held_b = [False]
_vb.muted = lambda: _held_b[0]
_ok = [_vb.say(k, f"line {k}.") for k in ("obstacle_ahead", "new_frontier", "coverage")]
_held_b[0] = True                               # the mission starts asking
_time.sleep(2.0)
check("lines queued BEFORE the hold are dropped as they come up, not said over "
      "his answer", all(_ok) and _vb.spoken <= 1 and _vb.dropped_muted >= 2,
      f"spoken {_vb.spoken}, dropped {_vb.dropped_muted}")
_vb.close() if hasattr(_vb, "close") else None
# 2026-10-19 17:55-17:57: a line every ~7 s muted the wake word 75% of the run,
# and "Friday, stop" never got through. After a line she listens for listen_s.
_vc = _fv.VoiceAnnouncer(enabled=True, backend="log", min_gap_s=0.0, listen_s=1.0)
_v1 = _vc.say("reflex", "Emergency stop.")
_time.sleep(0.3)
_v2 = _vc.say("obstacle_ahead", "Obstacle ahead.")
check("a line inside the listening window after the last one is dropped",
      _v1 and not _v2 and _vc.dropped_listen == 1,
      f"{_v1} {_v2} dropped {_vc.dropped_listen}")
check("...a once-a-run line is still said ('Exploration stopped.')",
      _vc.say("halt", "Exploration stopped.") is True)
_time.sleep(1.4)
check("...and after the window she talks again",
      _vc.say("obstacle_ahead", "Obstacle ahead.") is True)
_vc.close()
_vd = _fv.VoiceAnnouncer(enabled=True, backend="log", min_gap_s=0.0, listen_s=1.0)
_both = [_vd.say("reflex", "Emergency stop."), _vd.say("stuck", "I am stuck.")]
_time.sleep(0.6)
check("two lines queued in the same tick: the second is dropped as it comes up",
      all(_both) and _vd.spoken == 1 and _vd.dropped_listen == 1,
      f"spoken {_vd.spoken}, dropped {_vd.dropped_listen}")
_vd.close()
check("the run's default listening window is 15 s",
      _fv.build_announcer(type("A", (), {"no_voice": True})()).listen_s == 15.0)
pd_src = (ROOT / "rl" / "pi_deploy.py").read_text()
check("the run wires its voice's mute to the mission hold",
      "self.voice.muted = self.hold.active" in pd_src)
# the mission says the way back itself when it asked at the end
from exploration_mission import mission_manager as _MM  # noqa: E402
_real_choose = _MM.segmentation.choose_label_target
# Only the LAST question (at the cap) is allowed here, so the run's own
# "Returning to where I started." falls inside the hold.
_MM.segmentation.choose_label_target = (
    lambda *a, **k: _real_choose(*a, **k) if k.get("ignore_cooldown") else None)
w7 = booted(World(["kitchen"], world=mm.three_rooms(),
                  cfg={"explore": {"mission_cap_min": 0.5}, "return": {"mode": "run_first"}}))
w7.a.run.announces_return = True
w7.mgr.start()
w7.run()
_MM.segmentation.choose_label_target = _real_choose
said7 = w7.voice.said
check("(the question really came at the end)", w7.mgr._end_asked is True
      and any(e["kind"] == "state" and e.get("to") == "ASK_LABEL"
              for e in w7.mission()["events"]))
check("asked at the end + the run's own line muted -> she says 'Going back' herself",
      "Going back to where I started." in said7 and ASK in said7, str(said7[-5:]))
w8 = booted(World(["kitchen"], world=mm.three_rooms(),
                  cfg={"explore": {"mission_cap_min": 0.5}, "return": {"mode": "run_first"}}))
w8.a.run.announces_return = True
w8.mgr.start()
w8.run()
check("...but NOT when the room was asked about earlier (no hold at the end: the "
      "run says its own line, no echo)", "Going back to where I started." not in w8.voice.said,
      str(w8.voice.said[-4:]))

# =============================================================================
print("--- 8. blocked: she keeps trying, never 'please move me' (2026-10-02 23:22) ---")
import unstick as _un                                     # noqa: E402
def _lock(u, now, n=_un.LOCK_TICKS):
    out = None
    for i in range(n):
        out = u.update(frozen=True, tof_min=0.11, rear_min=0.80, pose=(0.5, -0.3, 3.0),
                       now=now + i * 0.1)
    return out
u = _un.Unstick(0.22, -0.15, 1.6, 0.12, keep_trying=True)
for _ in range(20):                                       # she drove in at 0.2 m/s
    u.record(0.2, 0.0, 0.1)
first = _lock(u, 100.0)
check("a lock -> the undo reverses along the way she came", first is not None
      and first[0] < 0, str(first))
u._end_undo(102.0, freed=False)
for _ in range(20): u.record(0.2, 0.0, 0.1)
_lock(u, 103.0)
u._end_undo(105.0, freed=False)
_lock(u, 106.0)                                           # MAX_TRIES spent: no history
said8 = [k for k, _ in u.events]
check("tries spent -> NOT 'please move me': she rests and says she is trying "
      "another way", "unstuck_help" not in said8 and "unstuck_retry" in said8
      and u.state == "resting", f"{said8} {u.state}")
for _ in range(20): u.record(0.0, 0.8, 0.1)               # whatever moved her meanwhile
again8 = _lock(u, 106.0 + _un.REST_S + 0.5, n=_un.LOCK_TICKS + 1)
check(f"...after {_un.REST_S:g} s she tries again by herself (no person needed)",
      again8 is not None and u.state == "undoing", f"{again8} {u.state}")
old_u = _un.Unstick(0.22, -0.15, 1.6, 0.12)
_lock(old_u, 0.0)
check("without keep_trying the old behaviour is unchanged (asks, waits)",
      old_u.state == "asking" and ("unstuck_help" in [k for k, _ in old_u.events]))
envm = (ROOT / "rl" / "modes" / "confident_explore.env").read_text()
check("the mode runs unstick ON in keep-trying form",
      'MODE_UNSTICK="1"' in envm and 'MODE_UNSTICK_KEEP_TRYING="1"' in envm)

print("--- 9. the last question is asked even during an e-stop (23:21) ---")
_MM.segmentation.choose_label_target = (
    lambda *a, **k: _real_choose(*a, **k) if k.get("ignore_cooldown") else None)
w9 = booted(World(["kitchen"], world=mm.three_rooms(),
                  cfg={"explore": {"mission_cap_min": 0.5}, "return": {"mode": "run_first"}}))
w9.mgr.start()
w9.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 20, max_s=60)
w9.sim.estop_on = True                                    # pressed against something
w9.run(lambda x: x.mgr.state in ("ASK_LABEL", "SUMMARIZE", "RETURN_TO_START"), max_s=60)
w9.sim.estop_on = False
w9.run()
_MM.segmentation.choose_label_target = _real_choose
check("time cap while reflex-stopped -> she still asks about the room",
      ASK in w9.voice.said and any(a["name"] == "kitchen" for a in w9.areas()),
      str(w9.voice.said[-5:]))

# =============================================================================
print("--- 10. the answer capture waits until her question has really been played ---")
import nav_voice as _nv                                   # noqa: E402
import threading as _th                                   # noqa: E402
played = []
slow = _nv.Voice(speak=lambda text: (_time.sleep(1.2), played.append(text)))
va = ad.VoiceAdapter()
va._voice = slow
slow.say_text("I've finished exploring this area. What should I call it?")
t0 = _time.monotonic()
va.wait_until_quiet(max_s=5.0)
waited = _time.monotonic() - t0
check("while Piper still synthesizes (queue already empty, no speaking mark yet) "
      "she is NOT quiet: the capture waits for the line to finish",
      played and waited >= 1.1, f"waited {waited:.2f} s, played {played}")
check("...and the voice says when it is busy", slow.busy is False)

print("--- 11. one voice at a time across processes ---")
import tts as _tts                                        # noqa: E402
import tempfile as _tempfile                              # noqa: E402
lock = Path(_tempfile.mkdtemp(prefix="friday-turn-")) / "speaking.lock"
holder = subprocess.Popen([sys.executable, "-c",
    "import sys, time; sys.path.insert(0, %r); import tts\n"
    "with tts._voice_turn(path=%r, max_wait_s=1):\n"
    "    print('held', flush=True); time.sleep(1.5)" % (str(ROOT), str(lock))],
    stdout=subprocess.PIPE, text=True)
assert holder.stdout.readline().strip() == "held"
t0 = _time.monotonic()
with _tts._voice_turn(path=lock, max_wait_s=10):
    got = _time.monotonic() - t0
holder.wait(5)
check("another process is speaking: this one waits its turn (no two lines at once)",
      0.8 <= got <= 3.0, f"waited {got:.2f} s")
holder = subprocess.Popen([sys.executable, "-c",
    "import sys, time; sys.path.insert(0, %r); import tts\n"
    "with tts._voice_turn(path=%r, max_wait_s=1):\n"
    "    print('held', flush=True); time.sleep(5)" % (str(ROOT), str(lock))],
    stdout=subprocess.PIPE, text=True)
assert holder.stdout.readline().strip() == "held"
t0 = _time.monotonic()
with _tts._voice_turn(path=lock, max_wait_s=0.5):
    got = _time.monotonic() - t0
check("...but a turn held too long never silences her: she speaks after the wait",
      0.4 <= got <= 1.5, f"{got:.2f} s")
holder.kill()
holder.wait(5)
t0 = _time.monotonic()
with _tts._voice_turn(path=lock, max_wait_s=5):
    got = _time.monotonic() - t0
check("...and a process that dies holding it frees it at once (the kernel drops it)",
      got < 0.3, f"{got:.2f} s")
_src = (ROOT / "tts.py").read_text()
check("tts.speak takes the turn before it marks her speaking",
      "with _speak_lock, _voice_turn(), _FromFirstSound() as heard," in _src)
check("...and marks her speaking at her FIRST SOUND (heard.start() before play), "
      "not while the voice loads and synthesises",
      "heard.start()\n                mouth.start()\n                play(item)" in _src)
_c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
with _tts._FromFirstSound() as _h:
    _unmarked = json.loads(_c2.SPEAKING_FILE.read_text())["state"]
    _t0 = _time.monotonic()
    _h.start()
    _lead = _time.monotonic() - _t0
    _marked = json.loads(_c2.SPEAKING_FILE.read_text())["state"]
    _h.start()
_after = json.loads(_c2.SPEAKING_FILE.read_text())
check("_FromFirstSound: no mark until the first sound; marked then, early enough "
      "for the detector's poll; quiet + drain after",
      _unmarked == "quiet" and _marked == "speaking" and _after["state"] == "quiet"
      and _after["until"] > _time.time() - 1
      and _lead >= _c2.WAKE_SPEECH_POLL_S - _c2.AUDIO_START_LAG_S, f"{_lead:.2f} s")
_c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
with _tts._FromFirstSound():
    pass
check("...a speak() that never played (synthesis failed) marks nothing",
      json.loads(_c2.SPEAKING_FILE.read_text())["at"] < _time.time() - 30)

# =============================================================================
print("--- 12. a capture ends where her own voice starts (2026-10-04 16:22) ---")
import wave as _wave                                      # noqa: E402
import importlib.util as _ilu                             # noqa: E402
# The REAL stt.py: an earlier section left a stand-in under the name "stt".
_spec = _ilu.spec_from_file_location("stt_real_fix12", ROOT / "stt.py")
_stt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stt)
_clips = sorted((ROOT / "state" / "wake_fires").glob("fire-*.wav"))
if not _clips:
    print("        (no saved voice clips; skipped)")
else:
    with _wave.open(str(max(_clips, key=lambda c: c.stat().st_size))) as _w:
        _pcm = _w.readframes(_w.getnframes())
    _fb = 16000 * 2 * _stt.FRAME_MS // 1000
    _voice = [_pcm[i:i + _fb] for i in range(0, len(_pcm) - _fb + 1, _fb)]
    _silence = [b"\0" * _fb] * 40

    def _feed(frames, mark_at=None, mark=None):
        for i, f in enumerate(frames):
            if i == mark_at:
                _time.sleep(_stt.OWN_VOICE_GRACE_S + 0.1)
                _c2.SPEAKING_FILE.write_text(json.dumps(
                    {"state": "speaking", "at": _time.time()}))
            yield f

    _c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
    _whole = _stt.endpoint(_feed(_voice * 3 + _silence), max_seconds=30)
    _c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
    _cut_at = len(_voice) + 10                  # well into his second "sentence"
    _cut = _stt.endpoint(_feed(_voice * 3 + _silence, mark_at=_cut_at), max_seconds=30)
    # (The capture counts from where his speech STARTS -- frame ~74 in this
    # clip -- so a mark at frame 108 leaves ~40 frames of him.)
    check("her voice starts mid-capture (another process): the capture ends there, "
          "keeping what he said before it",
          20 * _fb <= len(_cut) < len(_whole) - 10 * _fb,
          f"cut {len(_cut) // _fb} frames, whole {len(_whole) // _fb}, mark at {_cut_at}")
    _c2.SPEAKING_FILE.write_text(json.dumps({"state": "speaking",
                                             "at": _time.time() - 1.0}))
    _old = _stt.endpoint(_feed(_voice * 3 + _silence), max_seconds=30)
    check("...a voice already playing when the capture opened does not cut it "
          "(only one that STARTS after)", len(_old) == len(_whole),
          f"{len(_old) // _fb} vs {len(_whole) // _fb}")
    _c2.SPEAKING_FILE.write_text(json.dumps({"state": "quiet", "at": _time.time() - 60}))
    try:
        _stt.endpoint(_feed(_silence * 3 + _voice, mark_at=20), max_seconds=30,
                      start_timeout=20)
        _sil = False
    except _stt.Silence:
        _sil = True
    check("...her voice before he has said anything: nothing captured (Silence)", _sil)
    check("the wake beep is pw-play, never tts.speak (it cannot cut a capture)",
          "_mark(" not in (ROOT / "wake_word.py").read_text())

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
