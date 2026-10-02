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
    """A one-room mission whose way back gives up 1.1 m out -> HELP_WAIT."""
    w = booted(World(["study"], world=ROOM, path=ROOM_PATH, fail_return_at_m=1.5))
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
check("MODE_QUIET_MOTORS=0 and MODE_UNSTICK=0 in the mode file",
      'MODE_QUIET_MOTORS="0"' in env and 'MODE_UNSTICK="0"' in env)
import subprocess                                        # noqa: E402
out = subprocess.run(["bash", "-c", f'set -a; . "{ROOT}/rl/modes/confident_explore.env"; '
                      'ARGS=(); if [ "${MODE_QUIET_MOTORS:-1}" = "0" ]; then '
                      'ARGS+=(--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard); fi; '
                      '[ "${MODE_UNSTICK:-1}" = "0" ] && ARGS+=(--no-unstick); echo "${ARGS[*]}"'],
                     capture_output=True, text=True).stdout.strip()
sh = (ROOT / "rl" / "modes" / "explore.sh").read_text()
check("explore.sh passes the old drive and the old blocked behaviour to the run",
      "--motor-kick-rest 0 --motor-pulse-burst 1 --no-motor-plug-guard" in sh
      and "--no-unstick" in sh and out == "--motor-kick-rest 0 --motor-pulse-burst 1 "
      "--no-motor-plug-guard --no-unstick", out)
sys.path.insert(0, str(ROOT / "rl"))
import friday_localnav_deploy as L                       # noqa: E402
a = L.build_arg_parser().parse_args(["--motor", "friday", "--i-am-watching"] + out.split())
check("...and the run's own parser takes them (unstick off, kick every start, "
      "1-tick pulses, no coast)", a.no_unstick and a.motor_kick_rest == 0.0
      and a.motor_pulse_burst == 1 and a.motor_plug_guard is False)

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
fake_stt.transcribe = lambda audio: "I found an unexplored area 3 metres to the west"
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

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
