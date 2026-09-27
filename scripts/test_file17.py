#!/usr/bin/env python3
"""File 17 (memory governance), her own calendar, and her own initiative.

    ./venv/bin/python scripts/test_file17.py

No mic, no motors, no money: memory.db, the pending-request file, the
reflection file, the agency file and the mode file are temp copies; the
model calls are fakes. Nothing here can move the robot -- the one motion test
proves motion is REFUSED while AGENCY_MOTION is off.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file17-"))
config.MEMORY_DB = TMP / "memory.db"
config.MEMORY_PENDING_FILE = TMP / "memory_pending.json"
config.MODE_FILE = TMP / "mode.json"
config.LISTENING_FILE = TMP / "listening.json"
config.SPEAKING_FILE = TMP / "speaking.json"
config.SCENE_JSON = TMP / "scene.json"
config.AGENCY_FILE = TMP / "agency.json"
config.GCAL_TOKEN_FILE = TMP / "no_token.json"
config.RATE_LIMIT_FILE = TMP / "rate_limits.json"
config.ANTHROPIC_API_KEY = ""
config.DESCRIBE_KEY_FILE = TMP / "no.env"

import agency                                           # noqa: E402
import day_calendar                                     # noqa: E402
import memory                                           # noqa: E402
import memory_governance as mg                          # noqa: E402
import offline                                          # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


# ---------------------------------------------------------------------------
print("\nher own calendar")
NOW = datetime(2026, 9, 27, 17, 55)
got = day_calendar.parse_request("add dentist to my calendar tomorrow at 10", NOW)
check("'add dentist to my calendar tomorrow at 10'",
      got["title"] == "dentist"
      and datetime.fromtimestamp(got["start"]).strftime("%a %H:%M") == "Mon 10:00",
      str(got))
got = day_calendar.parse_request("add to my calendar lunch with Rahul tomorrow at 1", NOW)
check("'add to my calendar lunch with Rahul tomorrow at 1'",
      got["title"] == "lunch with Rahul"
      and datetime.fromtimestamp(got["start"]).strftime("%H:%M") == "13:00")
check("'what's on my calendar tomorrow' asks about tomorrow",
      day_calendar.parse_request("what's on my calendar tomorrow")["tomorrow"])
intent = offline.understand("Friday, add dentist to my calendar tomorrow at 10",
                            strict=True)
check("voice -> calendar_add_event, no brain",
      intent.tool == "calendar_add_event" and intent.args["title"] == "dentist")
check("'remove dentist from my calendar' -> calendar_remove",
      offline.understand("remove dentist from my calendar",
                         strict=True).tool == "calendar_remove")
now = time.time()
e1 = day_calendar.add("Standup", now + 5 * 60)
e2 = day_calendar.add("Dinner", now + 3 * 3600)
row = memory.db().execute("SELECT kind FROM calendar_events WHERE id=?",
                          (e1["id"],)).fetchone()
check("stored in memory.db with kind 'calendar'", row["kind"] == "calendar")
check("upcoming lists both, soonest first",
      [e["title"] for e in day_calendar.upcoming(1)] == ["Standup", "Dinner"])
check("the heads-up is due for the one starting in 5 minutes",
      [e["title"] for e in day_calendar.heads_up_due()] == ["Standup"])
day_calendar.mark_announced([e1["id"]])
check("...and only once", day_calendar.heads_up_due() == [])
import mcp_server.tools.reminders_tools as rt           # noqa: E402
said = rt.calendar_add_event("Haircut", "tomorrow at 6 pm")
check("the tool adds with no Google", said.startswith("Added Haircut"), said)
check("the tool asks for a time rather than guessing",
      "I need a time" in rt.calendar_add_event("Haircut", ""))
check("remove by name", day_calendar.remove("dinner")["title"] == "Dinner")

# ---------------------------------------------------------------------------
print("\nstep 1: the type convention is the one in use")
for kind in ("fact", "learned_preference", "exploration_sighting", "calendar",
             "reminder"):
    check(f"'{kind}' is documented", kind in mg.TYPES)
check("the same constants the writers use",
      config.LEARNED_PREFERENCE_KIND == "learned_preference"
      and day_calendar.KIND == "calendar")

# ---------------------------------------------------------------------------
print("\nstep 2: explicit remember, and never-forget")
for said, name, fact in (
        ("Friday, remember that my sister is called Priya",
         "remember_explicit", "Harsh's sister is called Priya"),
        ("always remember that mom's birthday is 12 March",
         "remember_forever", "mom's birthday is 12 March"),
        ("never forget that I am allergic to peanuts",
         "remember_forever", "Harsh is allergic to peanuts")):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} -> {name}", intent is not None and intent.name == name
          and intent.args["fact"] == fact, str(intent))
check("'do you remember when...' is a question, not a save",
      offline.understand("do you remember when we went out", strict=True) is None)
plain = mg.remember("Harsh's sister is called Priya")
kept = mg.remember("Harsh is allergic to peanuts", protected=True)
check("a plain remember is saved, not protected",
      memory.all_facts()[0].text and not mg.is_protected(plain["id"]))
check("an emphatic one is protected", mg.is_protected(kept["id"]))
again = mg.remember("Harsh is allergic to peanuts")
check("re-stating it plainly does NOT unprotect it",
      again["id"] == kept["id"] and mg.is_protected(kept["id"]))

# ---------------------------------------------------------------------------
print("\nstep 3: forget asks first, every time")
mg._haiku_pick = lambda description, rows: None     # no model: scores decide
mg.remember("The wifi password is on the fridge")
mg.remember("Harsh's meeting with Arjun is on Monday")
mg.remember("Harsh's meeting with the bank is on Tuesday")
before = len(memory.all_facts())
q = mg.start_forget("the wifi password")
check("one clear match: names it and ASKS", "wifi password" in q
      and "Should I delete it" in q, q)
check("...and deleted nothing yet", len(memory.all_facts()) == before)
check("'no' keeps it", mg.answer("no") == "Okay, I'll keep it."
      and len(memory.all_facts()) == before)
mg.start_forget("the wifi password")
done = mg.answer("yes")
check("'yes' deletes exactly that one", done.startswith("Done")
      and len(memory.all_facts()) == before - 1
      and not any("wifi" in f.text for f in memory.all_facts()), done)
q = mg.start_forget("my meeting")
check("several matches: lists them and asks which", "Which one" in q
      and "Arjun" in q and "bank" in q, q)
q = mg.answer("the second one")
check("choosing one asks yes/no again", "Should I delete it" in q, q)
mg.answer("yes")
left = [f.text for f in memory.all_facts() if "meeting" in f.text]
check("...then deletes only that one", len(left) == 1, str(left))
q = mg.start_forget("my cat's vet appointment")
check("no match: says so, touches nothing", q.startswith("I don't have")
      and mg.pending() is None, q)
pend = offline.understand("yes", strict=True)
check("a bare 'yes' with nothing pending is not a delete",
      pend is None or pend.name != "forget_answer")
mg.start_forget("allergic to peanuts")
check("a protected one can still be forgotten by him — with a warning",
      "never to forget" in (mg.pending() and mg.start_forget("allergic to peanuts")))
check("'yes please turn left' is NOT a yes",
      offline.understand("yes please turn left", strict=True) is None
      or offline.understand("yes please turn left", strict=True).name
      != "forget_answer")
mg.answer("no")

print("\nthe one delete function")
fid = mg.remember("Temporary note for the test")["id"]
check("no request, no delete", not mg.confirm_delete("anything", fid)
      and mg.entry(fid) is not None)
req = mg.request_delete(fid)
check("the dashboard's first click deletes nothing",
      req["token"] and mg.entry(fid) is not None)
check("a wrong token deletes nothing", not mg.confirm_delete("wrong", fid)
      and mg.entry(fid) is not None)
other = mg.remember("Another note")["id"]
check("the right token for another row deletes nothing",
      not mg.confirm_delete(req["token"], other) and mg.entry(other) is not None)
req = mg.request_delete(fid)
check("the right token for this row deletes it",
      mg.confirm_delete(req["token"], fid) and mg.entry(fid) is None)

# ---------------------------------------------------------------------------
print("\nstep 4: reflection never flags a protected memory")
config.LEARNED_PREFERENCE_KIND = "learned_preference"
p1 = memory.remember("Keep answers short", kind="learned_preference")
p2 = memory.remember("Speak in Hindi", kind="learned_preference")
mg.set_protected(p2["id"], True)
import reflection                                       # noqa: E402
before = (ROOT / "reflection.py").read_text()
check("the guard is in reflection.py",
      "memory_governance.is_protected(fact_id)" in before)
import subprocess                                       # noqa: E402
diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD~0", "--",
                       "reflection.py"], capture_output=True, text=True).stdout
added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
removed = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
check("the only change to reflection.py is the guard (vs last commit)",
      not removed and len(added) <= 4, "\n".join(added + removed))

# ---------------------------------------------------------------------------
print("\nher own initiative")
sit = agency.situation(time.time(), last_heard=0.0, recent=[])
check("alone: nobody to talk to", not agency.someone_here(sit))
check("a 'say' with nobody here is refused",
      agency.allowed({"action": "say", "text": "Hello!"}, sit) is not None)
check("motion is refused while AGENCY_MOTION is off",
      agency.allowed({"action": "look_around"}, sit)
      == "motion is switched off (AGENCY_MOTION)")
config.SCENE_JSON.write_text(json.dumps({
    "taken_at": time.time(),
    "objects": [{"label": "person", "score": 0.9, "side": "left",
                 "bearing_deg": -40}]}))
sit = agency.situation(time.time(), last_heard=0.0, recent=[])
check("a person in view: someone is here", agency.someone_here(sit))
check("a 'say' to a person in view is allowed",
      agency.allowed({"action": "say", "text": "Hi there!"}, sit) is None)
config.AGENCY_MOTION = True
config.AGENCY_MOTION_HOURS = (0, 24)
check("with motion on, turning to a person at her side is allowed",
      agency.allowed({"action": "turn_to_person"}, sit) is None)
config.AGENCY_MOTION = False
check("the model's nonsense becomes 'nothing'",
      agency.parse_decision("I think I'll dance")["action"] == "nothing"
      and agency.parse_decision('{"action":"fly"}')["action"] == "nothing"
      and agency.parse_decision('{"action":"feel","emotion":"sleeping"}')
      ["action"] == "nothing")
check("a good decision parses",
      agency.parse_decision('{"action":"say","text":"Welcome back!","why":"he '
                            'returned"}')["text"] == "Welcome back!")
recent = [{"at": time.time() - 60, "action": "say", "text": "Welcome back!"}]
sit2 = agency.situation(time.time(), 0.0, recent)
check("she never repeats what she just said",
      agency.allowed({"action": "say", "text": "Welcome back!"}, sit2)
      == "she said that already")

calls = []
agency.decide = lambda s: (calls.append(s) or
                           ({"action": "say", "text": "Hi Harsh!",
                             "emotion": "", "why": "he is here"}, ""))
watch = agency.AgencyWatch()
watch._last_heard = time.time() - 600
item = watch.step()
check("the watch thinks and hands main.py a line to say",
      item is not None and item["kind"] == "say" and len(calls) == 1, str(item))
check("...and does not think again straight away (min gap)",
      watch.step() is None and len(calls) == 1)
watch._last_think = time.time() - config.AGENCY_MIN_GAP_S - 1
check("nothing changed since: still no new thought",
      watch.step() is None and len(calls) == 1)
watch.pause()
watch._last_think, watch._last_print = 0.0, ""
check("never mid-turn", watch.step() is None and len(calls) == 1)
watch.resume()
watch._last_heard = time.time()
check("never right after he spoke", watch.step() is None and len(calls) == 1)
config.AGENCY = False
watch._last_heard = time.time() - 600
check("AGENCY = False: nothing at all", watch.step() is None and len(calls) == 1)
config.AGENCY = True
check("AGENCY_MOTION ships False", "AGENCY_MOTION = False" in
      (ROOT / "config.py").read_text())

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
