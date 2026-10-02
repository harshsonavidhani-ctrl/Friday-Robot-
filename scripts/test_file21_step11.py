#!/usr/bin/env python3
"""File 21 step 11: the voice hook, the mission's ears, the dashboard panel.

    ./venv/bin/python scripts/test_file21_step11.py

Network blocked; no robot; no sound. The spec's checks: non-mission speech
takes exactly the old path (this week's real transcripts, before vs after) ·
mission commands never reach the brain (the brain stub asserts it) ·
dashboard endpoints return JSON · naming -> landmark sync · no http(s):// in
the dashboard's static files. Plus the voice link between the two processes,
the stale-order rule, and the tools the brain must never see.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()
import leakcheck                                        # noqa: E402
LEAK_BEFORE = leakcheck.snapshot()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import intents                 # noqa: E402
from exploration_mission import service                 # noqa: E402
from exploration_mission import speech                  # noqa: E402
from exploration_mission import voice_link as vl        # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s11-"))
vl.state_dir = lambda: TMP / "state"                    # never the real state/
ROOM = mm.closed_room()
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


def plug(w: World, asker: vl.Asker | None = None) -> None:
    """Make `w`'s manager the one the service (and so the MCP tools) sees."""
    service._manager = w.mgr
    service._router = None
    service._asker = asker


# =============================================================================
print("--- the voice link: a question in friday-mcp, answered in friday-brain ---")

d = TMP / "link"
asker = vl.Asker(folder=d, grace_s=2.0)
watch = vl.MissionWatch(folder=d)
check("no MissionWatch heartbeat -> can_listen False (nobody to hear)",
      asker.can_listen() is False)
watch.beat()
check("a fresh heartbeat -> can_listen True", asker.can_listen() is True)
old = vl._read(d / vl.ALIVE_NAME)
(d / vl.ALIVE_NAME).write_text(json.dumps({"at": time.time() - 120}))
check("a heartbeat 2 min old -> False (friday-brain is down)", asker.can_listen() is False)
(d / vl.ALIVE_NAME).write_text(json.dumps(old))

got: list = []
t = threading.Thread(target=lambda: got.append(asker.listen_once(10.0, "label")))
t.start()
trig = None
for _ in range(100):
    trig = watch.step()
    if trig:
        break
    time.sleep(0.02)
ask = vl.parse_trigger(vl.TRIGGER_PREFIX + (trig or ""))
check("MissionWatch yields ONE trigger for the new question, with its kind and "
      "listen_s", ask is not None and ask["kind"] == "label" and ask["listen_s"] == 10.0,
      str(trig))
check("...and not again for the same question", watch.step() is None)
check("mission_answer for the right id is taken", asker.answer(ask["id"], " kitchen "))
t.join(3.0)
check("listen_once returns the words (trimmed) and removes asking.json",
      got == ["kitchen"] and not (d / vl.ASK_NAME).exists(), str(got))
check("an answer for a question nobody waits for any more is refused",
      asker.answer(ask["id"], "late") is False)

got.clear()
t = threading.Thread(target=lambda: got.append(asker.listen_once(0.2)))
t.start()
t.join(5.0)
check("no answer within listen_s + grace -> None (silence), file removed",
      got == [None] and not (d / vl.ASK_NAME).exists(), str(got))

t = threading.Thread(target=lambda: got.append(asker.listen_once(5.0)))
got.clear()
watch.pause()
t.start()
time.sleep(0.2)
check("paused (he is mid-turn): the question waits, no trigger", watch.step() is None)
watch.resume()
trig2 = watch.step()
check("...resumed: it is offered then", trig2 is not None)
asker.answer(vl.parse_trigger(vl.TRIGGER_PREFIX + trig2)["id"], "")
t.join(3.0)
check("an empty answer is silence (None)", got == [None], str(got))
check("MissionWatch is never hushed by a run: sleep() is a no-op",
      watch.sleep(20.0) == 0.0 and watch.sleeping_for() == 0.0)
(d / vl.ASK_NAME).write_text(json.dumps({"id": "old1", "kind": "label",
                                         "listen_s": 10, "until": time.time() - 1}))
check("a question whose deadline passed is skipped", watch.step() is None)
check("parse_trigger refuses junk", vl.parse_trigger("mission: a b") is None
      and vl.parse_trigger("reminders: x") is None)
vl.mark_heard(at=123.0, folder=d)
check("mark_heard writes the conversation clock TalkAdapter reads",
      ad.TalkAdapter(heard_file=d / vl.HEARD_NAME,
                     speaking_file=d / "none.json",
                     clock=ad.FakeClock(start=223.0)).quiet_s() == 100.0)
(d / vl.HOLD_NAME).write_text(json.dumps({"until": time.time() + 3, "why": "asking"}))
check("held(): a fresh hold file = she is held still", vl.held(folder=d) is True)
(d / vl.HOLD_NAME).write_text(json.dumps({"until": time.time() - 1, "why": "asking"}))
check("...a stale one is not", vl.held(folder=d) is False)
check("the real VoiceAdapter with no asker has no ears",
      ad.VoiceAdapter().can_listen() is False)

# =============================================================================
print("--- a whole mission whose question is answered through the link ---")

w = World([], world=ROOM, path=ROOM_PATH)
link_dir = TMP / "w-link"
link = vl.Asker(folder=link_dir, grace_s=3.0)
brain_side = vl.MissionWatch(folder=link_dir)
brain_side.beat()
w.voice.listen_once = link.listen_once                   # her ears = the link
w.voice.can_listen = link.can_listen
heard_kinds: list = []
stop_brain = threading.Event()


def friday_brain() -> None:
    """main._mission_turn's part: see the question, answer it."""
    while not stop_brain.is_set():
        trig = brain_side.step()
        if trig:
            q = vl.parse_trigger(vl.TRIGGER_PREFIX + trig)
            heard_kinds.append(q["kind"])
            link.answer(q["id"], "kitchen")
        time.sleep(0.01)


threading.Thread(target=friday_brain, daemon=True).start()
plug(w, link)
w.mgr.start()
w.run()
stop_brain.set()
check("the room's question went through the link and 'kitchen' named it",
      [a["name"] for a in w.areas()] == ["kitchen"] and heard_kinds,
      f"{w.areas()} {heard_kinds}")

# =============================================================================
print("--- service.voice: the router in front of the brain ---")

w = World([], world=ROOM, path=ROOM_PATH)
w.mgr.boot()
w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
plug(w)
r = service.voice("what's the weather like")
check("ordinary speech is not the mission's", r["handled"] is False)
r = service.voice("Friday, explore the house", age_s=2.0)
w.step()
check("'explore the house' is the mission's: MISSION_INIT",
      r["handled"] and w.mgr.state in ("MISSION_INIT", "EXPLORE"), str(r))
w.run()
w2 = World([], world=ROOM, path=ROOM_PATH)
w2.mgr.boot()
w2.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
plug(w2)
said = len(w2.voice.said)
r = service.voice("explore karo", age_s=config.COMMAND_MAX_AGE_S + 4)
w2.step()
check("a STALE 'explore karo' (older than COMMAND_MAX_AGE_S) is handled but "
      "refused: no mission, she says why",
      r["handled"] and r["refused"] == "start" and w2.mgr.state == "IDLE"
      and speech.render("stale_order")[0] in w2.voice.said[said:], str(r))
w2.mgr.start()
w2.run(lambda x: x.mgr.state == "EXPLORE", max_s=60)
r = service.voice("ruko", age_s=60.0)
w2.step()
check("...but a stop is NEVER refused for age: 'ruko' 60 s old still pauses",
      r["handled"] and r["refused"] is None and w2.mgr.state == "PAUSED",
      f"{r} {w2.mgr.state}")
r = service.voice("this is home")
check("'this is home' (his permanent home, File 10) is not the mission's",
      r["handled"] is False)
service._manager = None
check("switch off (no manager): every transcript is 'not handled'",
      service.voice("explore")["handled"] is False
      and service.status()["running"] is False
      and service.start_mission(5)["ok"] is False)

print("--- a run that dies while she is PAUSED ends the mission ---")
w2.sim.is_running = False                               # "Friday, stop" / dashboard Stop
w2.run(lambda x: x.mgr.state == "IDLE", max_s=30)
m = w2.mission()
check("PAUSED on a dead run -> saved as aborted (a person's stop), IDLE, no drive back",
      w2.mgr.state == "IDLE" and m["status"] == "aborted"
      and m["stop_reason"] == "voice_stop" and not w2.orders("return"),
      f"{w2.mgr.state} {m['status']} {m['stop_reason']}")

# =============================================================================
print("--- this week's real transcripts: what the mission would claim ---")

texts = []
log_file = ROOT / "logs" / "friday.log"
if log_file.exists():
    for line in log_file.read_bytes().decode("utf-8", "replace").splitlines():
        m_ = re.search(r'TRANSCRIBED\s+"(.*)"', line)
        if m_ and m_.group(1).strip():
            texts.append(m_.group(1))


class Rec:
    """Records what the router would do; drives nothing."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.calls.append(name)
            return True
        return call


import offline                                          # noqa: E402
idle = intents.Context(state="IDLE")
claimed: dict[str, list] = {}
for text in texts:
    ro = intents.Router(Rec(), resolve_area=lambda n: None,
                        resolve_place=lambda n: None)
    if ro.handle(text, idle):
        kind = intents.classify(text).name
        claimed.setdefault(kind, []).append(text)
ok_kinds = {"START", "LIST", "STATUS", "SET_HOME"}
n_claimed = sum(len(v) for v in claimed.values())
check(f"{len(texts)} real transcripts: only mission phrases are claimed with no "
      f"mission running ({n_claimed}: {sorted((k, len(v)) for k, v in claimed.items())})",
      len(texts) > 100 and set(claimed) <= ok_kinds,
      str({k: v[:5] for k, v in claimed.items()}))
old_paths = {}
for text in claimed.get("START", []):
    it = offline.understand(text, strict=True)
    old_paths[it.name if it else "brain"] = old_paths.get(it.name if it else "brain", 0) + 1
check(f"...the claimed 'explore' orders went to offline's explore before "
      f"(Harsh: voice explore = the mission): {old_paths}",
      set(old_paths) <= {"explore", "brain"})
for text in claimed.get("STATUS", []) + claimed.get("LIST", []):
    check(f"...{text!r} is a mission question, not the robot's status",
          intents.normalize(text) != "status")
check("a bare 'status' is never claimed with no mission (offline.py's status)",
      all(intents.normalize(t) != "status" for v in claimed.values() for t in v))

# =============================================================================
print("--- main.handle: the hook, the old path, the brain never asked ---")

import main                                             # noqa: E402


class FakeTurn:
    trigger = "voice"

    def __init__(self):
        self.notes, self.outcome = [], None

    def heard(self, *a, **k): pass
    def note(self, text, dash=True): self.notes.append(text)
    def answered_by(self, *a, **k): pass
    def done(self, outcome="DONE"): self.outcome = outcome
    def stage(self, *a, **k): pass
    def spoke(self, *a, **k): pass


class FakeVoiceOut:
    def __init__(self): self.said = []
    def say(self, text): self.said.append(text)
    def wait_until_quiet(self): pass


class FakeFace:
    def feel(self, *a): pass
    def set(self, *a): pass
    def alarm(self, *a): pass
    def rest(self): pass


class FakeBrain:
    def __init__(self): self.asked = []


class FakeSession:
    """The MCP session: mission_voice is the real tool function; anything
    else is recorded and answered blandly."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args=None):
        self.calls.append(name)
        if name == "mission_voice":
            from mcp_server.tools import mission_tools
            text = mission_tools.mission_voice(**(args or {}))
        else:
            text = "ok"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               isError=False)


downstream: list = []


async def fake_offline(intent, session, voice, face, turn):
    downstream.append(("reflex", intent.name))


async def fake_one_turn(brain_obj, session, voice, face, spoken, turn):
    downstream.append(("brain", spoken))
    brain_obj.asked.append(spoken)
    return "brain answer"


main._run_offline = fake_offline
main.one_turn = fake_one_turn
offline.online = lambda: True


def run_handle(text, session=None, said_at=None):
    downstream.clear()
    s = session or FakeSession()
    if said_at is not None:
        s = main._FreshOrders(s, said_at)
    brain = FakeBrain()
    out = asyncio.run(main.handle(brain, s, FakeVoiceOut(), FakeFace(), text,
                                  FakeTurn()))
    return out, list(downstream), brain, s


sample = [t for t in texts if not any(t in v for v in claimed.values())][:300]
config.EXPLORATION_MISSION_SERVICE = False
before = [run_handle(t)[1] for t in sample]
config.EXPLORATION_MISSION_SERVICE = True
w3 = World([], world=ROOM, path=ROOM_PATH)
w3.mgr.boot()
w3.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
plug(w3)
after = [run_handle(t)[1] for t in sample]
diff = [(t, b, a) for t, b, a in zip(sample, before, after) if b != a]
check(f"{len(sample)} real non-mission transcripts take EXACTLY the old path with "
      f"the switch on (reflex / brain, same calls)", not diff and len(sample) > 50,
      str(diff[:5]))

for text in ("explore the house", "explore karo", "ghoom ke dekho",
             "how much have you explored", "what have you saved"):
    w4 = World([], world=ROOM, path=ROOM_PATH)
    w4.mgr.boot()
    w4.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
    plug(w4)
    out, calls, brain, _s = run_handle(text)
    check(f"{text!r}: handled by the mission, the brain is never asked",
          out == "mission" and not brain.asked and not calls, f"{out} {calls}")
out, calls, brain, s = run_handle("stop")
check("'stop' still goes straight to the motors FIRST (offline stop); only "
      "then is the mission told (2026-10-02 fix: a stop must reach a mission "
      "waiting for help)", calls == [("reflex", "stop")]
      and s.calls == ["mission_voice"], f"{calls} {s.calls}")

w5 = World([], world=ROOM, path=ROOM_PATH)
w5.mgr.boot()
w5.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
plug(w5)
said_at = time.monotonic() - (config.COMMAND_MAX_AGE_S + 5)
out, calls, brain, fresh = run_handle("explore", said_at=said_at)
w5.step()
check("a stale spoken 'explore' (through _FreshOrders) is refused by the mission",
      out == "mission" and w5.mgr.state == "IDLE"
      and speech.render("stale_order")[0] in w5.voice.said, f"{out} {w5.mgr.state}")
check("...and the hook did NOT start _FreshOrders' clock (later moves keep their "
      "own age check)", fresh._acted_at is None)

config.EXPLORATION_MISSION_SERVICE = False
out, calls, brain, s = run_handle("explore")
check("switch OFF: 'explore' takes today's path (offline explore), no MCP call "
      "to the mission", calls == [("reflex", "explore")] and "mission_voice" not in s.calls,
      str(calls))


class BrokenSession(FakeSession):
    async def call_tool(self, name, args=None):
        if name == "mission_voice":
            raise RuntimeError("friday-mcp restarted")
        return await super().call_tool(name, args)


expected = {}
for text in ("what time is it", "tell me a story about robots"):
    expected[text] = run_handle(text)[1]                # switch off: today's path
config.EXPLORATION_MISSION_SERVICE = True
for text in ("what time is it", "tell me a story about robots"):
    out, calls, brain, s = run_handle(text, session=BrokenSession())
    check(f"a mission hook that FAILS is 'not ours': {text!r} carries on exactly "
          f"as today ({expected[text]})", calls == expected[text], str(calls))
config.EXPLORATION_MISSION_SERVICE = False

print("--- the wake-word hush while she is held ---")
hold = TMP / "state" / vl.HOLD_NAME
hold.parent.mkdir(parents=True, exist_ok=True)
hold.write_text(json.dumps({"until": time.time() + 3, "why": "asking"}))
check("switch off: never 'held' (the hush is exactly as before)",
      main._mission_held() is False)
config.EXPLORATION_MISSION_SERVICE = True
check("switch on + a fresh hold: held -> the wake word may be on",
      main._mission_held() is True)
hold.write_text(json.dumps({"until": time.time() - 1}))
check("...a lapsed hold: driving again, hushed", main._mission_held() is False)
main._mission_heard("explore")
check("a turn writes the conversation clock (switch on)",
      (TMP / "state" / vl.HEARD_NAME).exists())
config.EXPLORATION_MISSION_SERVICE = False

# =============================================================================
print("--- main._mission_turn: one answer, no wake word ---")

import types                                            # noqa: E402
fake_stt = types.ModuleType("stt")


class Silence(Exception):
    pass


fake_stt.Silence = Silence
fake_stt.transcribe = lambda audio: "the kitchen" if audio else ""
real_stt = sys.modules.get("stt")
sys.modules["stt"] = fake_stt
import initiative                                       # noqa: E402
initiative.she_is_talking = lambda: False


class FakeStream:
    def __init__(self): self.kw = None
    def capture(self, **kw):
        self.kw = kw
        return b"audio"


class FakeSource:
    def __init__(self): self.log = []
    def pause(self): self.log.append("pause")
    def resume(self): self.log.append("resume")


class AnswerSession:
    def __init__(self, asker_): self.asker, self.calls = asker_, []

    async def call_tool(self, name, args=None):
        self.calls.append((name, args))
        from mcp_server.tools import mission_tools
        return SimpleNamespace(content=[SimpleNamespace(
            text=mission_tools.mission_answer(**args))])


class FakeLink:
    def __init__(self, s): self.s = s
    async def get(self, wait): return self.s


link_dir2 = TMP / "turn"
asker2 = vl.Asker(folder=link_dir2, grace_s=5.0)
service._asker = asker2
got = []
t = threading.Thread(target=lambda: got.append(asker2.listen_once(10.0, "curiosity")))
t.start()
w6 = vl.MissionWatch(folder=link_dir2)
trig = None
for _ in range(100):
    trig = w6.step()
    if trig:
        break
    time.sleep(0.02)


class StubTurn(FakeTurn):
    """A real stages.Turn posts to the dashboard's log: not from a test."""

    def __init__(self, trigger=""):
        super().__init__()

    async def call(self, name, fn, *args, timeout, detail=""):
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout)

    def listening(self):
        pass


stream, src = FakeStream(), FakeSource()
sess = AnswerSession(asker2)
real_turn = main.stages.Turn
main.stages.Turn = StubTurn
asyncio.run(main._mission_turn(FakeVoiceOut(), FakeFace(), src, stream,
                               vl.TRIGGER_PREFIX + trig, FakeLink(sess)))
main.stages.Turn = real_turn
t.join(3.0)
check("_mission_turn: captures with the question's listen_s and no handover",
      stream.kw == {"start_timeout": 10.0, "handover_ms": 0, "deadtime_ms": 0},
      str(stream.kw))
check("...hands the words back via mission_answer, and the mission gets them",
      sess.calls and sess.calls[0][0] == "mission_answer" and got == ["the kitchen"],
      f"{sess.calls} {got}")
check("...the other wake sources are paused for it, then resumed",
      src.log == ["pause", "resume"], str(src.log))
if real_stt is not None:
    sys.modules["stt"] = real_stt
else:
    del sys.modules["stt"]

# =============================================================================
print("--- the brain never sees the plumbing ---")

import brain                                            # noqa: E402
tools = [SimpleNamespace(name=n, description=n, inputSchema={"type": "object"})
         for n in ("move", "mission_voice", "mission_answer", "mission_status",
                   "mission_start", "mission_name_area", "stop")]
names = [x["name"] for x in brain._to_anthropic_tools(tools)]
check("brain._to_anthropic_tools drops the five mission_* tools, keeps the rest in "
      "order", names == ["move", "stop"], str(names))
from mcp_server import registry                         # noqa: E402
import mcp_server.tools.mission_tools as mt             # noqa: E402
check("the five tools exist in friday-mcp's registry module",
      all(callable(getattr(mt, n)) for n in config.BRAIN_HIDDEN_TOOLS))

# =============================================================================
print("--- the dashboard: JSON, naming only, the Explore button ---")

from dashboard import server as dash                    # noqa: E402

w7 = World(["save it as kitchen"], world=ROOM, path=ROOM_PATH)
w7.mgr.boot()
w7.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
w7.mgr.start()
w7.run()
plug(w7)
spare = w7.store.add_area("home", "map-test", polygon=[[0, 0], [2, 0], [2, 2], [0, 2]],
                          center=(1.0, 1.0), area_m2=4.0, frame="map")


async def fake_call(tool, arguments=None, timeout=None):
    fn = getattr(mt, tool, None)
    if fn is None:
        raise AssertionError(f"unexpected tool {tool}")
    return fn(**(arguments or {}))

dash.robot.call = fake_call


class FakeCams:
    def __init__(self): self.stopped = 0
    async def stop_all(self): self.stopped += 1


class Req:
    def __init__(self, body=None, app=None):
        self._body, self.app = body, app or {"cams": FakeCams()}

    async def json(self):
        return self._body


def call(handler, body=None, app=None):
    try:
        resp = asyncio.run(handler(Req(body, app)))
        return resp.status, json.loads(resp.text)
    except Exception as exc:                            # noqa: BLE001
        return getattr(exc, "status", type(exc).__name__), getattr(exc, "text", str(exc))


st, data = call(dash.api_mission_get)
check("GET /api/mission returns JSON: state, areas, missions, home",
      st == 200 and data["running"] and data["state"] == "IDLE"
      and any(a["name"] == "kitchen" for a in data["areas"]) and data["missions"],
      str(data)[:300])
check("...drawn in ONE frame; the map-frame area is counted, not drawn",
      data["frame"] != "map" and data["areas_elsewhere"] == 1
      and all(a["name"] != spare["name"] for a in data["areas"]), str(data)[:200])
check("...the unnamed area is in the naming queue",
      any(q["id"] == spare["id"] for q in data["naming_queue"]))
st, body = call(dash.api_mission_post, {"area_id": spare["id"], "name": "study",
                                        "action": "drive"})
check("POST with anything besides {area_id, name} -> 400 (naming only)", st == 400)
st, body = call(dash.api_mission_post, {"area_id": spare["id"], "name": "  "})
check("an empty name -> 400", st == 400)
st, body = call(dash.api_mission_post, {"area_id": spare["id"], "name": "kitchen"})
check("a name already used elsewhere -> 400 with a suggestion", st == 400
      and "kitchen" in str(body))
st, body = call(dash.api_mission_post, {"area_id": spare["id"], "name": "Study"})
check("a valid name -> 200, the area is named", st == 200
      and w7.store.get(spare["id"])["name"] == "study", str(body))
check("...and synced to File 14's landmarks (a map-frame area)",
      w7.a.places.find_exact("study") is not None, str(w7.a.places.rows.keys()))
service._manager = None
st, data = call(dash.api_mission_get)
check("switch off: GET /api/mission still answers JSON (running: false)",
      st == 200 and data["running"] is False)
plug(w7)

config.EXPLORATION_MISSION_SERVICE = True
app = {"cams": FakeCams()}
st, body = call(dash.api_explore_post, {"action": "start", "minutes": 5}, app)
w7.step()
check("Explore button, switch on: a MISSION starts (mission_start), cameras "
      "handed back", st == 200 and body.get("mission") is True
      and w7.mgr.state in ("MISSION_INIT", "EXPLORE") and app["cams"].stopped == 1,
      f"{st} {body} {w7.mgr.state}")
st, body = call(dash.api_explore_post, {"action": "start", "minutes": 5}, app)
check("...pressed again mid-mission: refused (400), nothing doubled", st == 400)
config.EXPLORATION_MISSION_SERVICE = False
w7.run()

static = ROOT / "dashboard" / "static"
offenders = [f.name for f in sorted(static.glob("*"))
             if f.suffix in (".html", ".js", ".css")
             and re.search(r"https?://", f.read_text(errors="replace"))]
check("no http:// or https:// in the dashboard's templates / static files",
      offenders == [], str(offenders))
routes = (ROOT / "dashboard" / "server.py").read_text()
check("POST /api/mission is registered once", routes.count('"/api/mission"') == 2
      and routes.count('add_post("/api/mission"') == 1)

# =============================================================================
print("--- nothing else changed ---")
diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--",
                       "rl", "slam", "exploration_mode.py", "mcp_server/tools/explore.py",
                       "offline.py", "mcp_server/safety.py",
                       ":(exclude)rl/modes/*.env"],      # mode config, his OK
                      capture_output=True, text=True).stdout.split()
check("rl/, slam/, exploration_mode.py, explore.py, offline.py, safety.py untouched",
      diff == [], str(diff))
num = subprocess.run(["git", "-C", str(ROOT), "diff", "--numstat", "HEAD", "--",
                      "main.py", "brain.py"], capture_output=True, text=True).stdout
removed = {ln.split()[2]: int(ln.split()[1]) for ln in num.splitlines() if ln.strip()}
check("main.py and brain.py: additions only, plus two changed lines in main.py "
      "(the hush condition, one more '= None')",
      removed.get("main.py", 0) <= 2 and removed.get("brain.py", 0) == 0, str(removed))
check("the switch is set in config.py, one line (Harsh turned it on 2026-10-02)",
      len(re.findall(r"^EXPLORATION_MISSION_SERVICE = (True|False)$",
                     (ROOT / "config.py").read_text(), re.M)) == 1)
check("nothing here wrote to the real state folder",
      not leakcheck.leaked(LEAK_BEFORE), str(leakcheck.leaked(LEAK_BEFORE)))

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
