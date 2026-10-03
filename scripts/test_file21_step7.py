#!/usr/bin/env python3
"""File 21 step 7: the mission state machine.

    ./venv/bin/python scripts/test_file21_step7.py

Network blocked; no robot, no microphone, no speaker. The RL run is a scripted
stand-in (tests/exploration_mission/simworld.py) that drives a synthetic house,
publishes ray-cast sweeps once a second and honours hold / return exactly as
rl/ does. Everything downstream is REAL: the slam occupancy grid the sweeps
are folded into, File 14's despeckle, the frontier and room code, the area
store (temp database), the dialog and the speech lines. The clock is fake, so
two minutes of mission take a few seconds.
"""

from __future__ import annotations

import math
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import area_store as st        # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import intents                 # noqa: E402
from exploration_mission import mission_manager as M    # noqa: E402
from exploration_mission import mission_map as mmap     # noqa: E402
import make_maps as mm                                  # noqa: E402
import simworld as sw                                   # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s7-"))
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


from harness import DeferredRunner, World                # noqa: E402


def in_state(name):
    return lambda w: w.mgr.state == name


ROOM = mm.closed_room()                  # one 4 x 3 m room, robot at (3.0, 2.5)
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


# =============================================================================
print("--- the happy path: 3 rooms, 3 questions, summary, return (spec step 7) ---")

w = World(["save it as hall", "isko kitchen bolo", "ye bedroom hai"])
w.mgr.start()
end = w.run()
check("the mission runs to the end and is back in IDLE", end == "IDLE", str(w.states))
check("it passed through MISSION_INIT, EXPLORE, ASK_LABEL x3, RETURN_TO_START",
      w.seen()[:2] == ["MISSION_INIT", "EXPLORE"] and w.seen().count("ASK_LABEL") == 3
      and "RETURN_TO_START" in w.seen(), str(w.states))
areas = w.areas()
check("three named areas: hall, kitchen, bedroom",
      sorted(a["name"] for a in areas) == ["bedroom", "hall", "kitchen"]
      and all(a["status"] == "named" for a in areas))
start = w.sim.start_pose
by_name = {a["name"]: (a["center_x"] + start[0], a["center_y"] + start[1]) for a in areas}
rooms = {n: min(sw.ROOM_CENTRES, key=lambda r: math.hypot(
    c[0] - sw.ROOM_CENTRES[r][0], c[1] - sw.ROOM_CENTRES[r][1])) for n, c in by_name.items()}
check("each name landed on its own room, in the order she toured them",
      rooms == {"hall": "A", "kitchen": "B", "bedroom": "C"}, str(rooms))
check("each is about the room's real floor (11-13.5 m2)",
      all(10.0 <= a["area_m2"] <= 13.5 for a in areas),
      str([round(a["area_m2"], 1) for a in areas]))
said = w.voice.said
check("she said so at the start",
      said[0] == "Starting exploration. I've marked this spot as my starting point.")
check("three questions, three confirmations",
      said.count("I've finished exploring this area. What should I call it?") == 3
      and [s for s in said if s.startswith("Saved as")]
      == ["Saved as hall.", "Saved as kitchen.", "Saved as bedroom."], str(said))
summary = [s for s in said if s.startswith("Exploration complete")]
check("the summary names what she saved, new areas in order",
      len(summary) == 1 and summary[0].endswith(
          "and I saved 3 new areas: hall, kitchen and bedroom."), str(summary))
m2 = int(summary[0].split("about ")[1].split(" ")[0]) if summary else 0
check("...and the map it quotes is about the three rooms' floor (33-40 m2)",
      33 <= m2 <= 40, str(m2))
check("then the return and the arrival, in that order",
      said[-2:] == ["Going back to where I started.",
                    "I've reached where I started. Exploration saved."], str(said[-3:]))
mrow = w.mission()
check("the mission row: returned, no_frontiers, 3 new areas, distance, home error",
      mrow["status"] == "returned" and mrow["stop_reason"] == "no_frontiers"
      and len(mrow["areas_new"]) == 3 and mrow["distance_m"] > 15
      and mrow["home_error_m"] is not None and mrow["ended_at"], str(
          {k: mrow[k] for k in ("status", "stop_reason", "distance_m", "home_error_m")}))
check("she really is back at the start (the run's own way back)",
      math.hypot(w.sim.truth[0] - start[0], w.sim.truth[1] - start[1]) < 0.35)
check("the run was started once and held once per question",
      len(w.orders("start")) == 1 and len(w.orders("hold")) >= 3)
check("with no saved map to place the start on, areas stay in the mission's own "
      "frame and are NOT File 14 landmarks",
      all(a["frame"].startswith("mission:") for a in areas) and not w.a.places.all())
check("the full trail and the final checkpoint are on disk",
      (w.store.env_dir("home") / "trails" / f"{mrow['id']}.json").exists()
      and w.store.load_checkpoint("home")[0]["state"] == "IDLE")
ev = [e["kind"] for e in mrow["events"]]
check("the event log has the states, the dialogs, the saves and the summary",
      {"start", "state", "dialog", "area_saved", "summary", "mission_saved"} <= set(ev),
      str(sorted(set(ev))))
p95 = sorted(w.explore_ms)[int(len(w.explore_ms) * 0.95)]
print(f"        tick while exploring: median {statistics.median(w.explore_ms):.2f} ms, "
      f"p95 {p95:.2f} ms, worst {max(w.explore_ms):.2f} ms over "
      f"{len(w.explore_ms)} ticks")
check("tick p95 under 20 ms (spec)", p95 < 20.0)
res = w.mgr.analysis.latest
check("frontier and segmentation work was timed and stayed in budget",
      res and res["frontier_ms"] < 150.0)


# =============================================================================
print("--- saved before the first motion (spec 4.9) ---")

w = World(["kitchen"], world=ROOM, path=ROOM_PATH)
at_start: dict = {}
real_start = w.sim.start


def spy_start(minutes):
    at_start["mission"] = w.store.missions("home")
    at_start["anchor"] = w.store.home_anchor("home")
    at_start["checkpoint"] = w.store.load_checkpoint("home")[0]
    at_start["said"] = list(w.voice.said)
    return real_start(minutes)


w.sim.start = spy_start
w.mgr.start()
w.run(in_state("EXPLORE"))
check("when the run is started, the mission row already exists (running)",
      len(at_start["mission"]) == 1 and at_start["mission"][0]["status"] == "running")
sig = at_start["anchor"]["signature"]
check("...and the home anchor with its 360-bin signature (set_by auto)",
      at_start["anchor"]["set_by"] == "auto" and len(sig) == 360
      and sum(1 for v in sig if not math.isnan(v)) > 300)
check("...and the checkpoint", at_start["checkpoint"]["mission_id"]
      == at_start["mission"][0]["id"])
check("...and she has said she is starting", at_start["said"]
      == ["Starting exploration. I've marked this spot as my starting point."])
check("the run is asked for the mission cap's minutes",
      w.orders("start") == [("start", 25.0)])
end = w.run()
check("a one-room mission: asked at completion even with little dwell, then back",
      end == "IDLE" and [a["name"] for a in w.areas()] == ["kitchen"]
      and w.mission()["status"] == "returned", str(w.states) + str(w.voice.said))


# =============================================================================
print("--- stop -> PAUSED -> continue ---")

w = World(["save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 12)
w.mgr.halt()
held_at_once = w.sim.held() is not None
w.step()
pose0 = w.sim.truth
check("'stop': the hold is written in the call itself, before the next tick",
      held_at_once)
check("...PAUSED, and she says the fixed line", w.mgr.state == "PAUSED"
      and w.voice.said[-1] == "Stopping. Say continue when you're ready.")
for _ in range(300):                                    # 30 s of standing still
    w.step()
check("30 s later she has not moved and is still held (the hold is refreshed)",
      w.sim.truth == pose0 and w.sim.held() is not None and w.mgr.state == "PAUSED")
w.mgr.resume()
w.step()
check("'continue': EXPLORE again, released, 'Continuing exploration.'",
      w.mgr.state == "EXPLORE" and w.sim.held() is None
      and w.voice.said[-1] == "Continuing exploration.")
w.run(lambda x: x.t > 50)
check("...and she drives on", w.sim.truth != pose0)
check("a checkpoint was written for the pause",
      any(s == "PAUSED" for _, s in w.states))
end = w.run()
check("the mission still finishes normally afterwards", end == "IDLE"
      and len(w.areas()) == 3)


# =============================================================================
print("--- a question in progress: the hold never lapses ---")

runner = DeferredRunner()
w = World(["save it as hall"], runner=runner)
w.mgr.start()
w.step()                                                # the start job is created...
runner.finish_all()                                     # ...and its sweeps come in
w.run(in_state("ASK_LABEL"))
pose_q = w.sim.truth
lapsed = 0
for _ in range(250):                                    # a 25 s conversation
    w.step()
    lapsed += int(w.sim.held() is None)
check("during a 25 s question she is held at every single tick",
      lapsed == 0 and w.mgr.state == "ASK_LABEL", f"{lapsed} ticks unheld")
check("...and has not moved a millimetre", w.sim.truth == pose_q)
check("...and the dialog only began once the motors were confirmed stopped",
      w.mgr._stage == "dialog" and len(runner.waiting) == 1)
w.sim.estop_on = True                                   # her reflex fires mid-question
for _ in range(20):
    w.step()
check("an e-stop during the question changes nothing: still asking, still held",
      w.mgr.state == "ASK_LABEL" and w.sim.held() is not None)
runner.finish_all()                                     # he answers
w.step()
check("the answer is saved the moment it is given",
      [a["name"] for a in w.areas()] == ["hall"] and w.mgr.state == "EXPLORE")
for _ in range(400):
    w.step()
check("while the e-stop lasts no new question is started",
      w.mgr.state == "EXPLORE" and w.mgr.estop and len(w.areas()) == 1)
w.sim.estop_on = False


# =============================================================================
print("--- 'Friday stop' in the middle of a question ---")

w = World(["Friday stop", "save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(in_state("PAUSED"))
cp = w.store.load_checkpoint("home")[0]
check("stopped mid-question: PAUSED, nothing saved", w.mgr.state == "PAUSED"
      and w.areas() == [])
check("...the label stays pending, in the checkpoint too",
      w.mgr.pending_label is not None and cp["state"] == "PAUSED"
      and cp["pending_label"] and cp["pending_label"]["area_m2"] > 5)
w.mgr.resume()
w.run(lambda x: len(x.areas()) == 1, max_s=30)
check("'continue': the same question is asked again, and this time saved",
      [a["name"] for a in w.areas()] == ["hall"]
      and w.voice.said.count("I've finished exploring this area. "
                             "What should I call it?") == 2, str(w.voice.said))
check("...and exploring carries on", w.run(lambda x: x.mgr.state == "EXPLORE", 5)
      == "EXPLORE")


# =============================================================================
print("--- how exploring ends ---")

w = World(["hall", "kitchen", "bedroom"], cfg={"explore": {"mission_cap_min": 0.6}})
w.mgr.start()
end = w.run()
said = " | ".join(w.voice.said)
check("the mission's time cap: stop_reason time_cap and the run is told to return",
      end == "IDLE" and w.mission()["stop_reason"] == "time_cap"
      and (len(w.orders("return")) >= 1 or w.sim.why == "time"), str(w.sim.orders[-6:]))
check("...the summary opens with 'That's enough exploring for now.'",
      "That's enough exploring for now. Exploration complete." in said, said)
check("...and she still comes back to the start", w.mission()["status"] == "returned")

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start(minutes=0.5)                                # "explore for half a minute"
end = w.run()
check("'explore for N minutes': the run is started for N, its own timer ends it",
      w.orders("start") == [("start", 0.5)] and w.sim.why == "time"
      and w.mission()["stop_reason"] == "time_cap" and end == "IDLE",
      f"{w.orders('start')} {w.sim.why} {w.mission()['stop_reason']}")
w = World(["hall", "kitchen"])
w.mgr.start(minutes=500)
w.run(in_state("EXPLORE"))
check("asking for more than the cap is capped (25 min)",
      w.orders("start") == [("start", 25.0)])

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(in_state("RETURN_TO_START"), max_s=20)
# U1 (2026-10-03, "exactly as file"): the way home is the mission's own --
# File 14's planner on her map, the HELD run driving it -- not the run's.
check("'Friday, go back': summary, then the mission drives her home itself "
      "(the run is held with a path, never told to return)",
      w.mgr.state == "RETURN_TO_START" and not w.orders("return")
      and (w.sim.held() or {}).get("drive") is not None
      and w.mgr.stop_reason == "voice_stop", str(w.sim.orders[-5:]))
end = w.run()
check("...she arrives and the mission is saved as voice_stop / returned",
      end == "IDLE" and w.mission()["stop_reason"] == "voice_stop"
      and w.mission()["status"] == "returned" and w.sim.drive_ticks > 0
      and ("stop",) in w.sim.orders, f"{w.mission()['status']} {w.sim.orders[-4:]}")


# =============================================================================
print("--- when it goes wrong ---")

w = World(["hall"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
w.sim.publishing = False                                # the tracker goes silent
w.run(in_state("FAILSAFE"), max_s=30)
check("no published fix for 10 s with the run still out -> FAILSAFE",
      w.mgr.state == "FAILSAFE")
check("...the run is stopped, she says she needs help, the mission is saved",
      ("stop",) in w.sim.orders and w.voice.said[-1] == "I've stopped and I need help."
      and w.mission()["status"] == "failsafe" and w.mission()["stop_reason"] == "failsafe")
for _ in range(100):
    w.step()
check("...and she stays there: no autonomy", w.mgr.state == "FAILSAFE"
      and len(w.orders("start")) == 1)
w.mgr.halt()
w.step()
check("...until a person acknowledges it with 'stop': back to IDLE",
      w.mgr.state == "IDLE")

# 2026-10-02 (after the first real mission): a person's new "explore" is a
# new order, and a new order ends FAILSAFE (File 14's rule too) -- the old
# mission stays saved. Before, it was refused and only a restart got out.
w = World(["hall"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
w.sim.publishing = False
w.run(in_state("FAILSAFE"), max_s=30)
w.sim.publishing = True
w.mgr.start()
w.step()
check("...a new 'explore' in FAILSAFE closes it and starts a NEW mission",
      w.mgr.state in ("MISSION_INIT", "EXPLORE") and len(w.store.missions("home")) == 2
      and w.store.missions("home")[1]["status"] == "failsafe",
      f"{w.mgr.state} {[m['status'] for m in w.store.missions('home')]}")

w = World(["hall"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
w.sim.is_running = False                                # the dashboard's Stop
w.run(max_s=30)
check("the run is stopped from the dashboard: the mission ends as aborted, "
      "with no drive back", w.mgr.state == "IDLE" and w.mission()["status"] == "aborted"
      and not w.orders("return"), str(w.states))

w = World([], sweeps=False)
w.mgr.start()
w.run(max_s=5)
check("no LiDAR sweeps at the start -> 'Please place me at my home spot.', "
      "no mission, no run", w.mgr.state == "IDLE" and w.store.missions("home") == []
      and not w.orders("start") and w.voice.said == [
          "I'm not sure where I am. Please place me at my home spot."], str(w.voice.said))
w = World([])
w.sim.refuse_start = True
w.mgr.start()
w.run(max_s=5)
check("the run refuses to start -> the mission is closed as aborted",
      w.mgr.state == "IDLE" and w.mission()["status"] == "aborted")

w = World(["hall", "kitchen", "bedroom"])
w.sim.stale_status_age = 86_000.0                       # yesterday's status file
w.mgr.start()
w.run(lambda x: x.t > 40)
check("a day-old status file saying 'returning' is NOT believed: no summary, no "
      "return", "RETURN_TO_START" not in w.seen() and w.mgr.stop_reason is None
      and w.mgr.state in ("EXPLORE", "ASK_LABEL"), str(w.states))
check("...and with no fresh status she cannot confirm the motors are stopped, so "
      "she does not ask (never a question while she may be rolling)",
      w.areas() == [] and "What should I call it?" not in " ".join(w.voice.said)
      and any(e["kind"] == "ask_skipped" for e in w.mission()["events"]))


# =============================================================================
print("--- go-to in the middle of exploring ---")

w = World(["hall"], nav=ad.FakeNav(during_run=False))
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
w.mgr.go_to("kitchen", {"name": "kitchen", "x": 1.0, "y": 1.0})
w.step()
check("with the real run holding the wheels, File 14 cannot drive: refused, "
      "exploring carries on", w.mgr.state == "EXPLORE"
      and w.voice.said[-1] == "I can't do that right now.")
nav = ad.FakeNav(during_run=True)
w = World(["hall", "kitchen", "bedroom"], nav=nav)
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
trail_before, crumbs_before = w.mgr.trail, len(w.mgr.trail)
w.mgr.go_to("kitchen", {"name": "kitchen", "x": 1.0, "y": 1.0})
w.step()
check("where File 14 CAN take the wheels: TARGET_NAV, exploration held",
      w.mgr.state == "TARGET_NAV" and w.sim.held() is not None
      and nav.journeys == [{"name": "kitchen", "x": 1.0, "y": 1.0}])
for _ in range(100):
    w.step()
check("...held for as long as the journey lasts", w.sim.held() is not None
      and w.mgr.state == "TARGET_NAV")
nav.finish("ARRIVED")
w.step()
check("...arrived: released, EXPLORE again with the same trail and blacklist "
      "(the snapshot)", w.mgr.state == "EXPLORE" and w.sim.held() is None
      and w.mgr.trail is trail_before and len(w.mgr.trail) >= crumbs_before)


# =============================================================================
print("--- frames: the home spot's frame, never a one-sweep map fix ---")

import simworld as _sw                                  # noqa: E402

w = World(["save it as hall", "isko kitchen bolo", "ye bedroom hai", "study"],
          world=mm.uneven_rooms(), path=_sw.UNEVEN_TOUR)
localise_calls = []
w.a.slam.localise = lambda scan: localise_calls.append(1) or (10.0, 5.0, 0.0)
w.mgr.start()
w.run()
check("the first mission: its areas are saved in its OWN frame (no saved map "
      "here to fit onto), named, not landmarks",
      len(w.areas()) == 3 and all(a["frame"].startswith("mission:") for a in w.areas())
      and not w.a.places.all())
check("...and the start is NEVER placed on the saved map from one sweep (File 14's "
      "whole-map search was sure and wrong at 3 of 6 real spots)", localise_calls == [])
anchor1 = w.store.home_anchor("home")
check("...its start became the home anchor, in its frame",
      anchor1["frame"] == w.areas()[0]["frame"] and anchor1["x"] == 0.0)
n = len(w.voice.said)
end = w.run(lambda x: False, max_s=0.1)
w.mgr.start()                                           # again, from where she stopped
end = w.run()
said2 = w.voice.said[n:]
ev = [e for e in w.mission()["events"] if e["kind"] == "start"][0]
check("a second mission from the same spot recognises it: the home anchor's frame "
      "is adopted (she arrived facing another way; the signature measured the turn)",
      ev["at_home"] and ev["frame"] == anchor1["frame"], str(ev))
check("...every room the first mission named is recognised: no question at all",
      "What should I call it?" not in " ".join(said2)
      and any("I didn't find any new areas" in x for x in said2), str(said2))
check("...and nothing was saved twice", len(w.areas()) == 3)


# =============================================================================
print("--- the open question: a tracker that only matches near the start ---")

w = World(["hall", "kitchen", "bedroom"], matched_within_m=2.2)
w.mgr.start()
w.run()
near = [a["name"] for a in w.areas() if a["status"] == "named"]
stats = w.mgr.analysis.map.stats() if w.mgr.analysis.map else {}
print(f"        policy 'matched', tracker matching only within 2.2 m of the start: "
      f"named {near}; map folded {stats.get('used')}/{stats.get('seen')} sweeps")
check("policy 'matched': a room never seen from near the start (C) is not mapped "
      "and never asked about -- the known limit, to be measured on a real recording",
      "bedroom" not in near and len(near) < 3 and stats.get("used", 0) < stats.get("seen", 0),
      str(near))
w = World(["hall", "kitchen", "bedroom"], matched_within_m=2.2, policy="all")
w.mgr.start()
w.run()
check("policy 'all' (every sweep, trusting the pose): all three rooms",
      sorted(a["name"] for a in w.areas()) == ["bedroom", "hall", "kitchen"])


# =============================================================================
print("--- 'save this as X' out of the blue, stagnation, the small commands ---")

w = World(["kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.mgr.analysis.latest is not None
      and x.mgr.analysis.latest["seg"].at(*x.mgr.pose[:2]) is not None and x.t > 8)
w.mgr.label_now("den", "Den")
w.run(lambda x: len(x.areas()) == 1, max_s=10)
a0 = w.areas()[0] if w.areas() else {}
check("'save this as Den' while exploring: the room she is in, saved at once, "
      "no question asked", a0.get("name") == "den" and a0.get("display_name") == "Den"
      and "What should I call it?" not in " ".join(w.voice.said)
      and w.voice.said[-1] == "Saved as Den.", str(w.voice.said))
check("...marked complete=False: part of it was still unexplored",
      a0.get("complete") is False)
w.run(lambda x: x.mgr.state == "EXPLORE", 5)
check("...and exploring carries on", w.mgr.state == "EXPLORE")

door = mm.open_door()
w = World([], world=door, path=[(2.0, 2.5), (2.2, 2.5)])
w.sim.freeze_until = 1e18                               # the run stands still
w.mgr.start()
w.run(lambda x: x.t > 150)                              # the 2nd 60 s window
ev = [e for e in w.mission()["events"] if e["kind"] == "stagnation"]
check("no new floor for 60 s with a reachable frontier left: a stagnation event",
      len(ev) >= 1 and ev[0]["reachable_frontiers"] >= 1, str(ev))
check("...and the mission commands NOTHING for it (the run's own planner drives)",
      not w.orders("hold", "return", "stop") and w.mgr.state == "EXPLORE")

w = World(["save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run()
hall = w.store.by_name("home", "hall")
check("rename through the manager: renamed and said",
      w.mgr.rename(hall, "Big Hall") and w.store.by_name("home", "big hall")
      and w.voice.said[-1] == "Renamed hall to Big Hall.")
check("...a name that is taken is refused, nothing said",
      w.mgr.rename(w.store.by_name("home", "big hall"), "kitchen") is False)
w.mgr.say_list()
check("'list areas' reads what is saved", w.voice.said[-1].startswith("I know 3 areas: "))
w.mgr.say_status()
check("'status' quotes the map and the count",
      "square meters, and I've saved 3 areas so far." in w.voice.said[-1])
w.mgr.forget(w.store.by_name("home", "kitchen"))
check("forget removes it (forgotten.jsonl first)",
      w.store.by_name("home", "kitchen") is None
      and (w.store.env_dir("home") / "forgotten.jsonl").exists())


# =============================================================================
print("--- the router driving the manager ---")

w = World(["save it as hall", "kitchen", "bedroom"])
router = intents.Router(w.mgr, lambda n: w.store.by_name("home", n),
                        w.a.places.find_exact, monotonic=w.clock.monotonic)
check("'Friday, explore karo' starts a mission",
      router.handle("Friday, explore karo", w.mgr.context()) and w.step() == "MISSION_INIT")
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 10)
check("the router sees the mission as active and moving",
      w.mgr.context().mission_active and w.mgr.context().moving)
check("'how are you' is still not ours", router.handle("how are you", w.mgr.context())
      is False)
router.handle("Friday stop", w.mgr.context())
w.step()
check("'Friday stop' -> PAUSED", w.mgr.state == "PAUSED")
router.handle("chalo", w.mgr.context())
w.step()
check("'chalo' -> EXPLORE", w.mgr.state == "EXPLORE")
router.handle("wapas aao", w.mgr.context())
end = w.run()
check("'wapas aao' -> summary, return, arrived", end == "IDLE"
      and w.mission()["status"] == "returned"
      and w.mission()["stop_reason"] == "voice_stop", str(w.states[-4:]))
# =============================================================================
print("--- the real threads: the 10 Hz loop, a job thread, the analysis worker ---")

import threading                                        # noqa: E402
import time as _time                                    # noqa: E402

rclock = ad.Clock()                                     # the real clock this time
rsim = sw.SimRun(rclock, ROOM, ROOM_PATH)
rslam = ad.SlamAdapter(map_file=TMP / "rt" / "none.npz", session_file=TMP / "rt" / "s.json")
rslam.map_id = lambda read_only=False: "map-test"
rslam.localise = lambda scan: None
ra = ad.fake_adapters(clock=rclock, run=rsim, voice=ad.FakeVoice(["kitchen"]),
                      slam=rslam, lidar=rsim.lidar(),
                      nav=ad.FakeNav(during_run=False, clock=rclock, engine=False))
rstore = st.AreaStore(db_path=TMP / "rt" / "m.db", data_dir=TMP / "rt" / "data",
                      places=ra.places)
rstore.ensure_schema()
rmgr = M.MissionManager(ra, rstore, mcfg.load())        # ThreadRunner + worker thread
stop = threading.Event()
loop_thread = threading.Thread(target=rmgr.run_forever, args=(stop,), daemon=True)
loop_thread.start()


def _world() -> None:
    while not stop.is_set():
        rsim.step(0.05)
        _time.sleep(0.05)


threading.Thread(target=_world, daemon=True).start()
rmgr.start()
deadline = _time.time() + 60
while _time.time() < deadline and not (rmgr.state == "EXPLORE"
                                       and rmgr.analysis.latest is not None):
    _time.sleep(0.2)
reached = (rmgr.state, rmgr.analysis.latest is not None)
_time.sleep(3.0)
rmgr.halt()
_time.sleep(0.6)
paused = rmgr.state
stop.set()
rmgr.analysis.stop_thread()
loop_thread.join(5)
rt = sorted(rmgr.tick_ms)
rp95 = rt[int(len(rt) * 0.95)] if rt else 1e9
print(f"        real threads: {len(rt)} ticks, median {statistics.median(rt):.2f} ms, "
      f"p95 {rp95:.2f} ms, worst {rt[-1]:.1f} ms; analysis "
      f"{(rmgr.analysis.latest or {}).get('frontier_ms', 0):.0f} ms per look")
check("on real threads the mission starts, the run comes up, she is exploring",
      reached == ("EXPLORE", True), str(reached))
check("...the capture ran on a job thread and the map on the worker thread "
      "(the loop never waited for them)", rp95 < 20.0 and len(rt) > 50)
check("...a 'stop' from another thread pauses her", paused == "PAUSED"
      and rsim.held() is not None)
check("...the loop thread exits cleanly", not loop_thread.is_alive())


# =============================================================================
print("--- the rl/ side of 'return now' (rl/way_back.py Hold) ---")

import json                                             # noqa: E402
import time                                             # noqa: E402
sys.path.insert(0, str(ROOT / "rl"))
import way_back as wb                                   # noqa: E402

seam = TMP / "seam"
seam.mkdir()
h = wb.Hold(path=seam / "hold.json")
h.poll_return()
check("no return.json -> no return requested", not h.return_requested())
real_run = ad.RunAdapter(state_dir=seam)
real_run.request_return("m1")
h.poll_return()
check("the mission's request_return() is seen by the run", h.return_requested()
      and json.loads((seam / "return.json").read_text())["mission"] == "m1")
(seam / "return.json").write_text(json.dumps({"at": time.time() - 120}))
h.poll_return()
check("a request older than 60 s is ignored", not h.return_requested())
real_run.request_return("m0")
h2 = wb.Hold(path=seam / "hold.json")                   # a run that starts AFTER it
h2._born = time.time() + 1.0
h2.poll_return()
check("a request written BEFORE this run began never turns it round",
      not h2.return_requested())
(seam / "return.json").write_text("{not json")
h.poll_return()
check("garbage is no request", not h.return_requested())
real_run.clear_return()
real_run.clear_return()
check("clear_return() removes it, twice is harmless",
      not (seam / "return.json").exists())
check("a hold and a return request are separate files, separate meanings",
      h.path.name == "hold.json" and h.return_path.name == "return.json"
      and not h.active())
# rl/modes/*.env is mode CONFIGURATION, changed on purpose with Harsh's OK
# (2026-10-02: runs keep friday-brain up); the code in rl/ stays additive.
changed = __import__("subprocess").run(
    ["git", "diff", "-U0", "HEAD", "--", "rl", ":(exclude)rl/modes/*"],
    capture_output=True, text=True, cwd=str(ROOT)).stdout
removed = [ln for ln in changed.splitlines()
           if ln.startswith("-") and not ln.startswith("---")]
check("rl/: lines only added by this step, none removed or rewritten", not removed,
      "\n".join(removed))
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
