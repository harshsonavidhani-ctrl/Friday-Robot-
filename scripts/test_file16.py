#!/usr/bin/env python3
"""File 16: reminders, to-do, routines, calendar, briefing. No mic, no money.

    ./venv/bin/python scripts/test_file16.py

Every file it touches is a temp copy: memory.db, the briefing record, the
listening/speaking marks and the mode file are all redirected before any
module opens them, and Google, the weather and the tools a routine calls are
fakes. The real reminders list is never read or written.
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

TMP = Path(tempfile.mkdtemp(prefix="friday-file16-"))
config.MEMORY_DB = TMP / "memory.db"
config.BRIEFING_FILE = TMP / "briefing.json"
config.LISTENING_FILE = TMP / "listening.json"
config.SPEAKING_FILE = TMP / "speaking.json"
config.MODE_FILE = TMP / "mode.json"
config.SCENE_JSON = TMP / "scene.json"
config.WAKE_HEARTBEAT_FILE = TMP / "wake_alive.json"
config.GCAL_TOKEN_FILE = TMP / "secrets" / "google_token.json"

import briefing                                         # noqa: E402
import gcal                                             # noqa: E402
import memory                                           # noqa: E402
import offline                                          # noqa: E402
import reminders                                        # noqa: E402
import routines                                         # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


NOW = datetime(2026, 9, 27, 17, 55)                     # a Sunday evening
at = lambda text: reminders.parse_request(text, NOW)    # noqa: E731


def hm(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%a %H:%M")


# ---------------------------------------------------------------------------
print("\nwhat he says (step 2)")
for said, text, when in (
        ("remind me to call mom at 7 pm", "call mom", "Sun 19:00"),
        ("remind me to call mom at 7", "call mom", "Sun 19:00"),
        ("remind me to take my medicine at 9 in the morning", "take my medicine", "Mon 09:00"),
        ("remind me to look at the car at 5 tomorrow", "look at the car", "Mon 17:00"),
        ("remind me at 8:30 to check the oven", "check the oven", "Sun 20:30"),
        ("remind me in 20 minutes to check the rice", "check the rice", "Sun 18:15"),
        ("remind me to eat in half an hour", "eat", "Sun 18:25"),
        ("remind me to call dad at half past six", "call dad", "Sun 18:30"),
        ("remind me to leave at quarter to 9 tonight", "leave", "Sun 20:45"),
        ("remind me to pay rent on monday", "pay rent", "Mon 09:00"),
        ("remind me to water plants tomorrow morning", "water plants", "Mon 09:00"),
        ("remind me to stop at the shop at noon", "stop at the shop", "Mon 12:00")):
    got = at(said)
    ok = (got and got["action"] == "add" and got["text"] == text
          and got["due_at"] and hm(got["due_at"]) == when)
    check(f"{said!r} -> {text!r} {when}", ok, str(got))
for said, text in (("remind me to buy milk", "buy milk"),
                   ("add eggs to my to-do list", "eggs"),
                   ("put call the bank on my todo list", "call the bank")):
    got = at(said)
    check(f"{said!r} -> to-do {text!r}",
          got == {"action": "add", "text": text, "due_at": None}, str(got))
for said, which in (("mark buy milk as done", "buy milk"),
                    ("remove eggs from my to-do list", "eggs"),
                    ("tick off call the bank", "call the bank")):
    got = at(said)
    check(f"{said!r} -> done {which!r}",
          got == {"action": "done", "which": which}, str(got))
check("'what's on my to-do list' -> list todo",
      at("what's on my to-do list") == {"action": "list", "which": "todo"})
check("'what are my reminders' -> list reminders",
      at("what are my reminders") == {"action": "list", "which": "reminders"})

print("\nthe old commands are untouched")
for said, name in (("remind me in ten minutes", "timer"),
                   ("set a timer for 5 minutes", "timer"),
                   ("wake me up at 6", "alarm"),
                   ("stop", "stop"),
                   ("be quiet", "sleep")):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} is still {name}", getattr(intent, "name", None) == name,
          str(intent))
check("'remind me to stop at the shop at 6' is NOT a motor stop",
      not offline.is_stop("remind me to stop at the shop at 6"))

print("\nvoice -> tools, no brain (all reflexes, all answered out loud)")
for said, name, tool in (
        ("Hey Friday, remind me to call mom at 7 pm", "reminder_add", "add_reminder"),
        ("add eggs to my to-do list", "todo_add", "add_todo"),
        ("what's on my to-do list", "reminder_list", "list_reminders"),
        ("mark eggs as done", "reminder_done", "complete_reminder"),
        ("good morning", "routine", "run_routine"),
        ("Friday, movie time", "routine", "run_routine"),
        ("give me my daily briefing", "briefing", "daily_briefing")):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} -> {tool}",
          intent is not None and intent.name == name and intent.tool == tool
          and intent.speak_result, str(intent))
check("a video's 'good morning everyone, welcome back...' is no routine",
      offline.understand("good morning everyone welcome back to the channel "
                         "today we", strict=True) is None)
q = offline.understand("quiet mode", strict=True)
check("'quiet mode' switches quiet on", q is not None and q.name == "quiet"
      and q.args == {"on": True})
q = offline.understand("you can talk again", strict=True)
check("'you can talk again' switches it off",
      q is not None and q.args == {"on": False})

# ---------------------------------------------------------------------------
print("\nthe data (step 1): one table, kind='reminder'")
now = time.time()
r1 = reminders.add("call mom", due_at=now - 30)          # due already
r2 = reminders.add("dentist", due_at=now + 3600)
t1 = reminders.add("buy milk")
row = memory.db().execute("SELECT kind, due_at FROM reminders WHERE id=?",
                          (t1["id"],)).fetchone()
check("stored in memory.db with kind 'reminder'", row["kind"] == "reminder")
check("a to-do has no due time", row["due_at"] is None)
check("two reminders with the same words stay two",
      reminders.add("dentist", due_at=now + 7200)["id"] != r2["id"])
check("open_items: reminders first, soonest first, then to-dos",
      [r["text"] for r in reminders.open_items()][:3] == ["call mom", "dentist",
                                                          "dentist"])
check("due_unsaid finds only the one whose time has come",
      [r["id"] for r in reminders.due_unsaid()] == [r1["id"]])
check("find by words", reminders.find("milk")["id"] == t1["id"])
check("find by number", reminders.find(f"#{r2['id']}")["id"] == r2["id"])
check("nothing close is no match", reminders.find("elephant") is None)

# ---------------------------------------------------------------------------
print("\ndelivery (step 2): through the may-she-speak gate, between turns")
watch = reminders.ReminderWatch()
config.BRIEFING = False
item = watch.step()
check("a due reminder is offered", item is not None
      and item["kind"] == "reminder" and item["ids"] == [r1["id"]]
      and "call mom" in item["say"], str(item))
check("...and not offered twice while it waits in the queue",
      watch.step() is None)
watch.missed(item)
check("handed back (he started talking): offered again",
      (watch.step() or {}).get("ids") == [r1["id"]])
watch.delivered(item)
check("said -> done, never again", reminders.due_unsaid() == []
      and reminders.find(f"#{r1['id']}") is None)

r3 = reminders.add("take pills", due_at=time.time() - 5)
config.LISTENING_FILE.write_text(json.dumps({"listening": True,
                                             "at": time.time()}))
check("held while he is talking", watch.step() is None)
config.LISTENING_FILE.unlink()
import mode                                             # noqa: E402
mode.set_quiet(True, why="test")
check("held in quiet mode", watch.step() is None)
mode.set_quiet(False, why="test")
watch.pause()
check("held mid-turn", watch.step() is None)
watch.resume()
late = watch.step()
check("then said", late is not None and late["ids"] == [r3["id"]])
watch.delivered(late)
check("a to-do is never pushed on its own",
      all(i is None for i in (watch.step(), watch.step())))
old = reminders.add("feed the cat", due_at=time.time() - 3600)
words = watch.step()["say"]
check("a missed one says when it was for", words.startswith("Earlier, at"),
      words)

# ---------------------------------------------------------------------------
print("\nroutines (step 3)")
check("'good morning' names the routine",
      routines.match("Hey Friday, good morning!") == "good morning")
check("so does 'start my day'", routines.match("start my day") == "good morning")
calls = []


def fake_tool(name):
    table = {"weather_now": lambda: "24 degrees in Pune, clear — it's on the screen",
             "reminders_today": lambda: "No reminders for today.",
             "calendar_today": lambda: "Your calendar isn't connected yet.",
             "dismiss_screen": lambda: "back to the face",
             "quiet_mode": lambda on=True: "Quiet mode on."}
    fn = table.get(name)
    return (lambda **kw: (calls.append(name), fn(**kw))[1]) if fn else None


routines._tool = fake_tool
said, report = routines.run("good morning")
check("good morning runs its chain IN ORDER",
      calls == ["weather_now", "reminders_today", "calendar_today"], str(calls))
check("...and says each part", said.startswith("Good morning!")
      and "24 degrees" in said and "No reminders" in said, said)
calls.clear()
said, _ = routines.run("movie time")
check("movie time runs its steps silently after its own line",
      calls == ["dismiss_screen", "quiet_mode"] and "back to the face" not in said,
      said)
config.ROUTINES = {**config.ROUTINES, "test": {
    "phrases": ("test routine",),
    "steps": ({"tool": "no_such_tool"}, {"tool": "weather_now"})}}
said, report = routines.run("test")
check("a new routine is a config entry; a missing step is skipped, not fatal",
      report[0]["ok"] is False and report[1]["ok"] and "24 degrees" in said)
check("an unknown routine says which ones exist",
      "good morning" in routines.run("dance party")[0])

# ---------------------------------------------------------------------------
print("\ncalendar (step 4) — Google faked")
check("no token: not connected", not gcal.connected())
import mcp_server.tools.reminders_tools as tools        # noqa: E402
check("her own calendar answers with no Google at all",
      tools.calendar_today() == "Nothing else on your calendar today.")
config.GCAL_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
config.GCAL_TOKEN_FILE.write_text(json.dumps({
    "client_id": "id", "client_secret": "s", "refresh_token": "r",
    "access_token": "a", "expires_at": time.time() + 3000}))
check("with a token: connected", gcal.connected())


class Reply:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._body


start = datetime.now().replace(hour=23, minute=0, second=0, microsecond=0)
import httpx                                            # noqa: E402
httpx.get = lambda *a, **k: Reply(200, {"items": [
    {"id": "e1", "summary": "Standup",
     "start": {"dateTime": start.astimezone().isoformat()},
     "end": {"dateTime": (start + timedelta(minutes=30)).astimezone().isoformat()}},
    {"id": "e2", "summary": "Holiday", "start": {"date": start.date().isoformat()},
     "end": {"date": (start.date() + timedelta(days=1)).isoformat()}}]})
events = gcal.today_events()
check("events are read and parsed", [e["title"] for e in events]
      == ["Standup", "Holiday"] and events[1]["all_day"], str(events))
posted = {}


def fake_post(url, **kw):
    posted.update(kw.get("json") or {})
    return Reply(200, {"id": "e3", "summary": kw["json"]["summary"],
                       "start": kw["json"]["start"], "end": kw["json"]["end"]})


httpx.post = fake_post
event = gcal.add_event("Dentist", time.time() + 86400)
check("a Google event is created", posted.get("summary") == "Dentist"
      and event["title"] == "Dentist", str(event))
noted = memory.facts_of_kind("calendar")
check("...and noted in memory with type 'calendar'",
      noted and noted[0].text.startswith("Calendar: Dentist"), str(noted))

# ---------------------------------------------------------------------------
print("\nthe daily briefing (step 5)")
config.BRIEFING = True
import weather_source                                   # noqa: E402
weather_source.current = lambda *a, **k: ({"temp_c": 24.4, "condition": "Clear",
                                           "high_c": 31}, "fake")
gcal._cache.clear()
morning = datetime.now().replace(hour=8, minute=0).timestamp()
reminders.add("call the plumber", due_at=morning + 2 * 3600)
text = briefing.compose(morning)
check("combines weather, calendar, reminders and to-dos",
      "24 degrees" in text and "Standup" in text and "plumber" in text
      and "to-do list" in text, text)
check("not wanted before anyone is here",
      not briefing.wanted(morning, last_heard=0.0))
check("wanted in the morning once he has spoken",
      briefing.wanted(morning, last_heard=morning - 60))
evening = datetime.now().replace(hour=19, minute=0).timestamp()
check("not in the evening", not briefing.wanted(evening, evening - 60))

# Through the watch, the way it happens live: offered once he has been here
# and quiet INITIATIVE_IDLE_S, then never again that day.
for r in reminders.due_unsaid():
    reminders.mark_said([r["id"]])
w2 = reminders.ReminderWatch()
real_wanted = briefing.wanted
briefing.wanted = lambda now=None, heard=0.0: real_wanted(morning, morning - 60)
w2._last_heard = time.time() - 5
check("the watch holds it while he has only just spoken", w2.step() is None)
w2._last_heard = time.time() - config.INITIATIVE_IDLE_S - 5
offer = w2.step()
check("...then offers the briefing", offer is not None
      and offer["kind"] == "briefing" and "24 degrees" in offer["say"],
      str(offer))
w2.delivered(offer)
briefing.wanted = real_wanted
check("once a day: not again after it was said", briefing.done_today()
      and not briefing.wanted(time.time(), last_heard=time.time() - 60))

# ---------------------------------------------------------------------------
print("\nsafety (step 7)")
ignored = (ROOT / ".gitignore").read_text()
check("secrets/ is gitignored", "\nsecrets/" in ignored)
check("no credential in config.py", "client_secret" not in
      (ROOT / "config.py").read_text())

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
