#!/usr/bin/env python3
"""File 21 step 6: the label dialog, the offline intent router, the speech lines.

    ./venv/bin/python scripts/test_file21_step6.py

Network blocked; no microphone, no speaker, no motors. The voice is a scripted
fake, the actions are a recording fake, the area store is a real one on a temp
database with a fake File 14 places table.
"""

from __future__ import annotations

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
from exploration_mission import intents as it           # noqa: E402
from exploration_mission import label_dialog as ld      # noqa: E402
from exploration_mission import speech                  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s6-"))
_results: list[bool] = []
_n = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def fresh_store(places=None) -> st.AreaStore:
    _n[0] += 1
    s = st.AreaStore(db_path=TMP / f"s{_n[0]}.db", data_dir=TMP / f"s{_n[0]}",
                     places=places if places is not None else ad.FakePlaces())
    s.ensure_schema()
    return s


SQUARE = [[0, 0], [4, 0], [4, 3], [0, 3]]
FAR = [[10, 10], [13, 10], [13, 13], [10, 13]]


def seg(polygon=SQUARE, **kw) -> dict:
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    return dict({"env_id": "home", "map_id": "map-1", "polygon": polygon,
                 "center": (cx, cy), "area_m2": 12.0}, **kw)


# =============================================================================
print("--- normalisation (spec 4.7) ---")

for text, want in (
        ("Friday, save it as Kitchen.", "save it as kitchen"),
        ("Hey Friday!  Save-it as   kitchen", "save it as kitchen"),
        ("OK Friday, isko guest_room bolo", "isko guest room bolo"),
        ("umm, name this area living room please", "name this area living room"),
        ("save it as area three", "save it as area 3"),
        ("Call it Harsh’s room", "call it harsh's room"),
        ("acha okay uh hmm", ""), ("twenty", "20"), ("", ""),
        ("hi friday", ""), ("friday friday stop", "friday stop")):
    check(f"{text!r} -> {want!r}", ld.normalize(text) == want, ld.normalize(text))
check("capitals are kept for the display name",
      ld.normalize_cased("Friday, call it Harsh's Room please") == "call it Harsh's Room")


# =============================================================================
print("--- the parser table (spec 4.7, all 26 rows) ---")

TABLE = [
    (1, "Friday, save it as kitchen.", None, "NAME kitchen"),
    (2, "save this as the kitchen", None, "NAME kitchen"),
    (3, "Call it Harsh's room", None, "NAME harsh's room"),
    (4, "name this area living room please", None, "NAME living room"),
    (5, "this is the bathroom", None, "NAME bathroom"),
    (6, "Kitchen.", None, "NAME kitchen"),
    (7, "hey friday save it as study room", None, "NAME study room"),
    (8, "isko kitchen naam se save karo", None, "NAME kitchen"),
    (9, "ise bedroom save kar do", None, "NAME bedroom"),
    (10, "ye drawing room hai", None, "NAME drawing room"),
    (11, "iska naam pooja ghar rakho", None, "NAME pooja ghar"),
    (12, "Friday isko guest room bolo", None, "NAME guest room"),
    (13, "yes", "kitchen", "ACCEPT kitchen"),
    (14, "haan sahi hai", "kitchen", "ACCEPT kitchen"),
    (15, "no, it's the dining room", "kitchen", "NAME dining room"),
    (16, "skip", None, "SKIP"),
    (17, "baad mein", None, "SKIP"),
    (18, "", None, "UNCLEAR"),
    (19, "umm what", None, "UNCLEAR"),
    (20, "Friday stop", None, "STOP"),
    (21, "save it as stop", None, "UNCLEAR"),
    (22, "save it as area three", None, "NAME area 3"),
    (23, "save it as the big room near the main door", None, "UNCLEAR"),
    # 24 is the duplicate: the parse is a NAME; the dialog asks "kitchen 2?"
    (24, "save it as kitchen", None, "NAME kitchen"),
    (25, "wapas chalo", None, "FINISH"),
    (26, "chalo", None, "UNCLEAR"),
]
for row, text, suggestion, want in TABLE:
    got = ld.parse_label(text, suggestion)
    check(f"#{row:<2d} {text!r}{' (suggestion ' + suggestion + ')' if suggestion else ''}"
          f" -> {want}", str(got) == want, f"{got} ({got.why})")
check("'yes, save it as kitchen' -> NAME kitchen (the name wins over the yes)",
      str(ld.parse_label("yes, save it as kitchen", "bedroom")) == "NAME kitchen")
check("'haan sahi hai' with NO suggestion -> UNCLEAR",
      ld.parse_label("haan sahi hai").kind == "UNCLEAR")
check("the display name is as he said it",
      ld.parse_label("Call it Harsh's room").display == "Harsh's room")
check("#19 is refused because 'what' is reserved",
      "reserved" in ld.parse_label("umm what").why)
check("#23 is refused for being over 4 words",
      "4 words" in ld.parse_label("save it as the big room near the main door").why)

print("--- more of the grammar ---")
for text, want in (
        ("is room ko study naam se save kardo", "NAME study"),
        ("is area ko store room ke naam se save kar do", "NAME store room"),
        ("ise kitchen call kar do", "NAME kitchen"),
        ("isse balcony save kr do", "NAME balcony"),
        ("is ka naam hall hai", "NAME hall"),
        ("ise pooja room kaho", "NAME pooja room"),
        ("yeh bathroom he", "NAME bathroom"),
        ("mark this as the lab", "NAME lab"),
        ("label this room to workshop", "NAME workshop"),
        ("that's my room", "NAME room"),
        ("it is a store room", "NAME store room"),
        ("haan isko kitchen bolo", "NAME kitchen"),
        ("nahi ye bedroom hai", "NAME bedroom"),
        ("the kitchen", "NAME kitchen"),
        ("guest bed room", "NAME guest bed room"),
        ("big guest bed room", "UNCLEAR"),              # 4 bare words
        ("pata nahi", "SKIP"), ("i don't know", "SKIP"), ("later", "SKIP"),
        ("not now friday", "SKIP"), ("rehne do", "SKIP"), ("chhodo", "SKIP"),
        ("ruko", "STOP"), ("bas", "STOP"), ("ruk jao", "STOP"),
        ("stop exploring", "FINISH"), ("bas karo", "FINISH"), ("go back", "FINISH"),
        ("come back", "FINISH"), ("that's enough", "FINISH"),
        ("go home", "UNCLEAR"), ("continue", "UNCLEAR"), ("no", "UNCLEAR"),
        ("yes", "UNCLEAR"), ("save it as home", "UNCLEAR"),
        ("save it as home office", "NAME home office"),
        ("what should i call it", "UNCLEAR")):
    got = ld.parse_label(text)
    check(f"{text!r} -> {want}", str(got) == want, f"{got} ({got.why})")
print("--- what his own turn log says is NOT a name ---")
for text in ("Thank you.", "Thanks for watching!", "Bye.", "bye bye", "I love you",
             "\u6211\u554a", "\u7121\u6cd5\u8f2a", "Move forward", "drive backwards",
             "What's the time?", "what's my name", "Open YouTube", "That's it.",
             "Yeah, it's good.", "it's fine", "done", "haha"):
    got = ld.parse_label(text)
    check(f"{text!r} is not saved as an area name", got.kind == "UNCLEAR",
          f"{got} ({got.why})")
check("...but the same words ORDERED as a name are one: 'save it as play room'",
      str(ld.parse_label("save it as play room")) == "NAME play room")
check("...and a real bare name still is: 'Guest room'",
      str(ld.parse_label("Guest room")) == "NAME guest room")
check("a statement is a name only as an ANSWER: name_slot(declarative=False)",
      ld.name_slot("this is the bathroom") is not None
      and ld.name_slot("this is the bathroom", declarative=False) is None
      and ld.name_slot("ye drawing room hai", declarative=False) is None
      and ld.name_slot("isko guest room bolo", declarative=False) is not None)
check("yes_no: haan / theek hai / nahi / kitchen",
      (ld.yes_no("haan"), ld.yes_no("theek hai"), ld.yes_no("nahi"),
       ld.yes_no("kitchen")) == (True, True, False, None))

print("--- object hints (4.7 point 9) ---")
check("a fridge seen twice at 0.7 -> 'kitchen'",
      ld.suggest([("refrigerator", 0.7), ("refrigerator", 0.8)]) == "kitchen")
check("seen once is not enough", ld.suggest([("refrigerator", 0.9)]) is None)
check("low confidence does not count",
      ld.suggest([("bed", 0.5), ("bed", 0.55), ("bed", 0.4)]) is None)
check("a label with no room says nothing",
      ld.suggest([("chair", 0.9), ("chair", 0.9), ("person", 0.9)]) is None)
check("more evidence wins",
      ld.suggest([("sofa", 0.9)] * 2 + [("tv", 0.9)] * 2 + [("bed", 0.9)] * 2)
      == "living room")
check("a tie is no suggestion at all",
      ld.suggest([("bed", 0.9)] * 2 + [("toilet", 0.9)] * 2) is None)


# =============================================================================
print("--- the dialog (spec 4.7 flow) ---")


def dialog(replies, store=None, **kw):
    a = ad.fake_adapters(voice=ad.FakeVoice(replies))
    store = store or fresh_store()
    d = ld.LabelDialog(a, store, **kw)
    return a, store, d


a, s, d = dialog(["Friday, save it as kitchen."])
r = d.run(seg())
check("answered at once: saved under the name",
      r.outcome == "saved" and r.area["name"] == "kitchen"
      and s.by_name("home", "kitchen") is not None)
check("...she asked the fixed question, then confirmed",
      a.voice.said == ["I've finished exploring this area. What should I call it?",
                       "Saved as kitchen."], str(a.voice.said))
check("...curious while asking, happy once saved",
      a.face.shown == ["curious", "happy"], str(a.face.shown))
check("...listened once, for listen_s", a.voice.listened == [10.0])
check("...the named area is a File 14 landmark at once",
      s.places.find_exact("kitchen") is not None)
check("...and the question + answer are in the transcript for the log",
      [e.get("heard") for e in r.transcript if "heard" in e]
      == ["Friday, save it as kitchen."])

a, s, d = dialog([None, None])
r = d.run(seg())
check("silence -> one re-ask -> silence -> saved as 'area 1', unnamed",
      r.outcome == "unnamed" and r.area["name"] == "area 1"
      and r.area["status"] == "unnamed" and len(a.voice.listened) == 2)
check("...the re-ask and the unnamed line are the fixed ones",
      a.voice.said[1:] == ["Sorry, I didn't catch that. What should I call this area?",
                           "I'll save it as area 1 for now. You can rename it later."],
      str(a.voice.said))
check("...and it waits in the naming queue",
      [x["name"] for x in s.naming_queue("home")] == ["area 1"])
a, s, d = dialog(["umm what", "ye drawing room hai"])
r = d.run(seg())
check("unclear, then a name on the re-ask -> saved",
      r.outcome == "saved" and r.area["name"] == "drawing room")
a, s, d = dialog(["chalo", "chalo", "kitchen"])
r = d.run(seg())
check("only ONE re-ask: a third answer is never waited for",
      r.outcome == "unnamed" and len(a.voice.listened) == 2)
a, s, d = dialog(["skip"])
r = d.run(seg())
check("skip -> unnamed + queued, no re-ask (nothing is thrown away)",
      r.outcome == "unnamed" and len(a.voice.listened) == 1
      and s.naming_queue("home"))
a, s, d = dialog([None, None], reask_once=False)
r = d.run(seg())
check("reask_once off: one listen only", len(a.voice.listened) == 1)

base = fresh_store()
base.add_area("home", "map-1", polygon=FAR, center=(11.5, 11.5), area_m2=9,
              name="kitchen")
a, s, d = dialog(["save it as kitchen", "yes"], store=base)
r = d.run(seg())
check("#24 the name is taken elsewhere -> 'I already have a kitchen. Should I "
      "call this kitchen 2?'",
      a.voice.said[1] == "I already have a kitchen. Should I call this kitchen 2?",
      str(a.voice.said))
check("...yes -> saved as kitchen 2", r.outcome == "saved"
      and r.area["name"] == "kitchen 2" and a.face.shown[1] == "thinking")
a, s, d = dialog(["save it as kitchen", "call it pantry"], store=base)
r = d.run(seg())
check("...a new name instead -> that name", r.outcome == "saved"
      and r.area["name"] == "pantry")
a, s, d = dialog(["save it as kitchen", "save it as pantry"], store=base)
r = d.run(seg())
check("...a new name that is ALSO taken -> unnamed (one loop at most)",
      r.outcome == "unnamed" and r.area["status"] == "unnamed")
a, s, d = dialog(["save it as kitchen", "no"], store=base)
r = d.run(seg())
check("...anything else -> unnamed", r.outcome == "unnamed")

a, s, d = dialog(["haan sahi hai"])
r = d.run(seg(), suggestion="kitchen")
check("a suggestion is offered in the question",
      a.voice.said[0] == "I've finished exploring this area. It looks like a "
                         "kitchen. What should I call it?")
check("...and saved only after a yes", r.outcome == "saved"
      and r.area["name"] == "kitchen")
a, s, d = dialog(["no, it's the dining room"])
r = d.run(seg(), suggestion="kitchen")
check("...'no, it's the dining room' saves the dining room instead",
      r.area["name"] == "dining room")
a, s, d = dialog([None, None])
r = d.run(seg(), suggestion="kitchen")
check("...no answer: the suggestion is NOT saved as the name",
      r.outcome == "unnamed" and r.area["name"] == "area 1")

a, s, d = dialog(["Friday stop"])
r = d.run(seg())
check("'Friday stop' mid-dialog: stopped, nothing saved (the label stays pending)",
      r.outcome == "stopped" and r.area is None and s.list_areas("home") == [])
a, s, d = dialog(["wapas chalo"])
r = d.run(seg())
check("#25 'wapas chalo' mid-dialog: finish, and the area is saved unnamed + queued",
      r.outcome == "finished" and r.area["status"] == "unnamed"
      and len(a.voice.said) == 1)
a, s, d = dialog([None, "bas karo"])
r = d.run(seg())
check("...on the re-ask too", r.outcome == "finished")

harsh_places = ad.FakePlaces([{"name": "desk", "x": 1.0, "y": 1.0, "radius_m": 0.75,
                               "kind": "place", "note": "[map:map-1] by hand"}])
a, s, d = dialog(["this is the desk"], store=fresh_store(harsh_places))
r = d.run(seg())
check("a place Harsh saved himself inside this area = the same place: linked, "
      "no question", r.outcome == "saved" and r.area["hints"].get("linked_place")
      == "desk" and len(a.voice.listened) == 1)


class DeafVoice(ad.FakeVoice):
    def listen_once(self, timeout_s):
        self.listened.append(timeout_s)
        raise OSError("microphone gone")


deaf = ad.fake_adapters(voice=DeafVoice())
r = ld.LabelDialog(deaf, fresh_store()).run(seg())
check("a microphone failure is silence, not a crash: saved unnamed",
      r.outcome == "unnamed" and r.area["name"] == "area 1")
a, s, d = dialog(["save it as lobby"])
r = d.run(seg(complete=False, mission_id=None, hints={"objects": ["sofa"]}))
check("complete=False and hints pass through to the saved area",
      r.area["complete"] is False and r.area["hints"] == {"objects": ["sofa"]})


# =============================================================================
print("--- intents: every example in the spec's table (4.8) ---")

EXAMPLES = {
    "HALT": ["Friday stop", "stop", "ruko", "ruk jao", "bas"],
    "START": ["explore the house", "start exploring", "go explore", "explore karo",
              "ghar explore karo", "ghoom ke dekho", "Friday, explore"],
    "CONTINUE": ["continue", "carry on", "resume", "keep exploring",
                 "continue karo", "chalo", "aage badho"],
    "FINISH": ["stop exploring", "finish exploring", "that's enough", "go back",
               "come back", "wapas aao", "wapas chalo", "bas karo"],
    "LABEL_NOW": ["save this as kitchen", "isko kitchen bolo",
                  "iska naam pooja ghar rakho", "call it Harsh's room",
                  "ise bedroom save kar do", "name this area living room"],
    "GO_TO": ["go to kitchen", "go to the kitchen", "kitchen pe jao",
              "kitchen mein jao", "kitchen chalo"],
    "RENAME": ["rename kitchen to pantry", "change kitchen to pantry",
               "kitchen ka naam pantry kar do"],
    "FORGET": ["forget kitchen", "delete kitchen", "kitchen bhool jao"],
    "LIST": ["what have you saved", "list areas", "which rooms do you know",
             "kya kya save kiya"],
    "STATUS": ["status", "how much have you explored", "kitna explore hua"],
    # The file's three (4.8). U4, Harsh 2026-10-03 "exactly as file": "this is
    # home" / "yahan home hai" set BOTH homes -- friday-brain runs File 10's
    # permanent home first, then the mission's start (main.py).
    "SET_HOME": ["set home here", "mark the start here", "set the start here",
                 "this is home", "yahan home hai"],
}
for name, texts in EXAMPLES.items():
    got = [(t, (it.classify(t) or it.Intent("-")).name) for t in texts]
    check(f"{name}: {len(texts)} examples", all(g == name for _, g in got),
          str([x for x in got if x[1] != name]))
for text in ("how are you", "what's the weather", "go back to sleep",
             "what time is it", "tell me a joke", "continue the story please",
             "remind me to call mom", "", "friday"):
    check(f"normal talk is no intent: {text!r}", it.classify(text) is None,
          str(it.classify(text)))
check("'stop exploring' is FINISH, not HALT",
      it.classify("stop exploring").name == "FINISH")
check("'please stop the music' only matches the loose HALT (last in the order)",
      it.classify("please stop the music").args.get("loose") is True)
check("'chalo' is CONTINUE but 'kitchen chalo' is GO_TO kitchen",
      it.classify("chalo").name == "CONTINUE"
      and it.classify("kitchen chalo").args == {"name": "kitchen"})
check("'go to the kitchen and say hello' -> GO_TO kitchen (no post-arrival action)",
      it.classify("go to the kitchen and say hello").args == {"name": "kitchen"})
check("'kitchen pe jao aur light on karo' -> kitchen",
      [c for c in it.candidates("go to kitchen aur light on karo")
       if c.name == "GO_TO"][0].args == {"name": "kitchen"})
check("'explore for five minutes' carries the minutes",
      it.classify("explore for five minutes").args == {"minutes": 5})
check("'rename kitchen to Big Pantry' keeps his capitals for the new name",
      it.classify("rename kitchen to Big Pantry").args
      == {"name": "kitchen", "new": "Big Pantry"})
check("out of the blue, 'this is the bathroom' / \"it's good\" / \"that's it\" "
      "label nothing (only an ORDER does)",
      all("LABEL_NOW" not in [c.name for c in it.candidates(t)]
          for t in ("this is the bathroom", "it's good", "That's it.",
                    "ye drawing room hai", "yeah it's the kitchen")))
check("'this is home' / 'yahan home hai' are SET_HOME with the saved map (File 10's "
      "home is saved first, by main), and never an area called home",
      [(i.name, i.args) for i in it.candidates("this is home")]
      == [("SET_HOME", {"with_place": True})]
      and [(i.name, i.args) for i in it.candidates("yahan home hai")]
      == [("SET_HOME", {"with_place": True})])


# =============================================================================
print("--- the router: who claims what, in which state ---")


class Actions:
    def __init__(self, rename_ok=True):
        self.calls: list[tuple] = []
        self.rename_ok = rename_ok

    def __getattr__(self, name):
        def call(*args, **kw):
            self.calls.append((name, *args) + ((kw,) if kw else ()))
            return self.rename_ok if name == "rename" else None
        return call


KITCHEN = {"id": "k1", "name": "kitchen", "display_name": "Kitchen",
           "center_x": 2.0, "center_y": 1.5, "radius_m": 1.9}
AREAS = {"kitchen": KITCHEN}
PLACES = {"the desk": {"name": "the desk", "x": 5.0, "y": 5.0, "radius_m": 0.75}}
clock = ad.FakeClock()


def router(**kw):
    acts = Actions(**kw)
    return acts, it.Router(acts, AREAS.get, PLACES.get, monotonic=clock.monotonic)


def ctx(state="IDLE", **kw) -> it.Context:
    return it.Context(state=state, **kw)


MISSION_STATES = sorted(it.STATES - it.IDLE_STATES)
quiet = []
for state in sorted(it.STATES):
    for text in ("how are you", "what's the weather", "go back to sleep",
                 "tell me a joke"):
        acts, ro = router()
        if ro.handle(text, ctx(state)) or acts.calls:
            quiet.append((state, text))
check("normal talk returns False in EVERY state and touches nothing", not quiet,
      str(quiet))
acts, ro = router()
check("an empty transcript is not ours", ro.handle("", ctx("EXPLORE")) is False)

halted = []
for state in MISSION_STATES:
    acts, ro = router()
    if ro.handle("Friday stop", ctx(state)) and acts.calls == [("halt",)]:
        halted.append(state)
check("'Friday stop' is HALT in every mission state", halted == MISSION_STATES,
      str(set(MISSION_STATES) - set(halted)))
acts, ro = router()
check("...with no mission and nothing moving it is NOT ours (the existing stop "
      "handles it)", ro.handle("stop", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
check("...but moving with no mission (a go-to) it is",
      ro.handle("ruko", ctx("IDLE", moving=True)) and acts.calls == [("halt",)])
acts, ro = router()
check("'please stop the music' during a mission still halts (loose HALT)",
      ro.handle("please stop the music", ctx("EXPLORE")) and acts.calls == [("halt",)])

for state, want in (("PAUSED", [("resume",)]), ("EXPLORE", [("say", "cant_now")]),
                    ("RETURN_TO_START", [("say", "cant_now")])):
    acts, ro = router()
    ok = ro.handle("continue", ctx(state))
    check(f"'continue' in {state} -> {want[0][-1]}", ok and acts.calls == want,
          str(acts.calls))
acts, ro = router()
check("'continue' with an interrupted mission waiting -> resume",
      ro.handle("chalo", ctx("IDLE", interrupted=True)) and acts.calls == [("resume",)])
acts, ro = router()
check("'continue' with no mission at all is NOT ours (the brain's word)",
      ro.handle("continue", ctx("IDLE")) is False and not acts.calls)

for state in ("EXPLORE", "PAUSED", "HELP_WAIT"):
    acts, ro = router()
    check(f"'wapas aao' in {state} -> finish",
          ro.handle("wapas aao", ctx(state)) and acts.calls == [("finish",)])
acts, ro = router()
check("'go back' while already returning -> can't",
      ro.handle("go back", ctx("RETURN_TO_START")) and acts.calls == [("say", "cant_now")])
acts, ro = router()
check("'come back' with no mission is NOT ours (File 14's own reflex)",
      ro.handle("come back", ctx("IDLE")) is False and not acts.calls)

acts, ro = router()
check("'save this as kitchen' while exploring -> label the current area",
      ro.handle("Friday, save this as Kitchen", ctx("EXPLORE"))
      and acts.calls == [("label_now", "kitchen", "Kitchen")], str(acts.calls))
acts, ro = router()
check("'isko pooja ghar bolo' while paused -> label",
      ro.handle("isko pooja ghar bolo", ctx("PAUSED"))
      and acts.calls == [("label_now", "pooja ghar", "pooja ghar")])
acts, ro = router()
check("'this is the kitchen' with no mission is NOT ours (File 14's remember_place)",
      ro.handle("this is the kitchen", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
check("\"That's it.\" while exploring labels nothing and is NOT ours",
      ro.handle("That's it.", ctx("EXPLORE")) is False and not acts.calls)
acts, ro = router()
check("'save this as kitchen' while returning -> can't",
      ro.handle("save this as kitchen", ctx("RETURN_TO_START"))
      and acts.calls == [("say", "cant_now")])
for state, want in (("IDLE", ("set_home", {"with_place": False})),
                    ("PAUSED", ("set_home", {"with_place": False})),
                    ("EXPLORE", ("say", "cant_now"))):
    acts, ro = router()
    check(f"'mark the start here' in {state} -> {want[-1]}",
          ro.handle("mark the start here", ctx(state)) and acts.calls == [want],
          str(acts.calls))
for state, want in (("IDLE", ("set_home", {"with_place": True})),
                    ("PAUSED", ("set_home", {"with_place": True})),
                    ("EXPLORE", ("say", "cant_now"))):
    acts, ro = router()
    check(f"'this is home' in {state} -> {want[0] if want[0] != 'say' else 'cant_now'} "
          "(spec 4.8: SET_HOME only in IDLE / PAUSED)",
          ro.handle("this is home", ctx(state)) and acts.calls == [want],
          str(acts.calls))

acts, ro = router()
check("'go to the kitchen' with NO mission is NOT ours: File 14's nav_go drives it",
      ro.handle("Friday, go to the kitchen", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
ok = ro.handle("Friday, go to the kitchen", ctx("PAUSED"))
check("'go to the kitchen' in a mission -> go_to with the saved area's centre",
      ok and acts.calls[0][:2] == ("go_to", "kitchen")
      and (acts.calls[0][2]["x"], acts.calls[0][2]["y"], acts.calls[0][2]["kind"])
      == (2.0, 1.5, "area"), str(acts.calls))
acts, ro = router()
check("'kitchen chalo' mid-explore -> go_to kitchen",
      ro.handle("kitchen chalo", ctx("EXPLORE")) and acts.calls[0][:2] == ("go_to", "kitchen"))
acts, ro = router()
ok = ro.handle("desk pe jao", ctx("PAUSED"))
check("a File 14 landmark ('the desk') resolves too, exact names only",
      ok and acts.calls[0][:2] == ("go_to", "the desk")
      and acts.calls[0][2]["kind"] == "landmark", str(acts.calls))
acts, ro = router()
check("'go to the kitchen and say hello' -> kitchen, the rest ignored",
      ro.handle("go to the kitchen and say hello", ctx("EXPLORE"))
      and acts.calls[0][:2] == ("go_to", "kitchen"))
acts, ro = router()
check("'go to sleep' with no mission is NOT ours",
      ro.handle("go to sleep", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
check("'go to the kitch' is not fuzzy-matched", ro.handle("go to the kitch",
                                                         ctx("IDLE")) is False)
acts, ro = router()
check("an unknown place DURING a mission -> 'I don't know where X is yet.'",
      ro.handle("go to the garage", ctx("EXPLORE"))
      and acts.calls == [("say", "unknown_place", {"name": "garage"})], str(acts.calls))
acts, ro = router()
check("'go to the kitchen' while returning -> can't",
      ro.handle("go to the kitchen", ctx("RETURN_TO_START"))
      and acts.calls == [("say", "cant_now")])
acts, ro = router()
check("'go to the kitchen' while not localized and idle is NOT ours either",
      ro.handle("go to the kitchen", ctx("IDLE", localized=False)) is False)
acts, ro = router()
check("a bare 'status' with no mission is NOT ours (offline.py's robot status)",
      ro.handle("status", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
check("...but in a mission 'status' is the mission's",
      ro.handle("status", ctx("EXPLORE")) and acts.calls == [("say_status",)],
      str(acts.calls))
acts, ro = router()
check("'how much have you explored' is the mission's even when idle",
      ro.handle("how much have you explored", ctx("IDLE"))
      and acts.calls == [("say_status",)], str(acts.calls))

acts, ro = router()
check("'forget it' is NOT ours (nothing called 'it')",
      ro.handle("forget it", ctx("IDLE")) is False and not acts.calls)
acts, ro = router()
ok1 = ro.handle("forget the kitchen", ctx("IDLE"))
check("'forget the kitchen' -> 'Should I forget Kitchen?', nothing forgotten yet",
      ok1 and acts.calls == [("say", "forget_ask", {"name": "Kitchen"})], str(acts.calls))
ok2 = ro.handle("haan", ctx("IDLE"))
check("...'haan' -> forgotten, 'Forgot Kitchen.'",
      ok2 and acts.calls[1:] == [("forget", KITCHEN), ("say", "forgot", {"name": "Kitchen"})],
      str(acts.calls))
acts, ro = router()
ro.handle("kitchen bhool jao", ctx("IDLE"))
check("...'nahi' -> nothing forgotten", ro.handle("nahi", ctx("IDLE"))
      and len(acts.calls) == 1)
acts, ro = router()
ro.handle("delete kitchen", ctx("IDLE"))
ok = ro.handle("list areas", ctx("IDLE"))
check("...neither yes nor no: the question is dropped and the words are routed "
      "normally", ok and acts.calls[1:] == [("say_list",)] and ro.pending is None)
acts, ro = router()
ro.handle("forget kitchen", ctx("IDLE"))
clock.advance(31.0)
check("...a 'yes' 31 s later is not a yes to it any more",
      ro.handle("yes", ctx("IDLE")) is False and len(acts.calls) == 1)
acts, ro = router()
check("forget while moving -> can't", ro.handle("forget kitchen", ctx("EXPLORE", moving=True))
      and acts.calls == [("say", "cant_now")])

acts, ro = router()
check("'rename kitchen to Pantry' -> rename(area, 'Pantry')",
      ro.handle("rename kitchen to Pantry", ctx("IDLE"))
      and acts.calls == [("rename", KITCHEN, "Pantry")], str(acts.calls))
acts, ro = router()
check("'kitchen ka naam rasoi kar do' -> rename",
      ro.handle("kitchen ka naam rasoi kar do", ctx("PAUSED"))
      and acts.calls == [("rename", KITCHEN, "rasoi")])
acts, ro = router()
check("renaming something she does not know is NOT ours",
      ro.handle("rename the garage to gym", ctx("IDLE")) is False)
acts, ro = router(rename_ok=False)
check("a rename the store refuses (taken / not a name) -> can't",
      ro.handle("rename kitchen to stop", ctx("IDLE"))
      and acts.calls[-1] == ("say", "cant_now"))

listed = [s for s in sorted(it.STATES)
          if (lambda a_r: a_r[1].handle("list areas", ctx(s)) and a_r[0].calls
              == [("say_list",)])(router())]
check("'list areas' answers in any state", listed == sorted(it.STATES))
statused = [s for s in sorted(it.STATES)
            if (lambda a_r: a_r[1].handle("kitna explore hua", ctx(s)) and a_r[0].calls
                == [("say_status",)])(router())]
check("'kitna explore hua' answers in any state", statused == sorted(it.STATES))

acts, ro = router()
check("'explore karo' idle + localized -> start",
      ro.handle("Friday, explore karo", ctx("IDLE")) and acts.calls == [("start", None, False)])
acts, ro = router()
check("'explore for 5 minutes' -> start(5)",
      ro.handle("explore for five minutes", ctx("IDLE")) and acts.calls == [("start", 5, False)])
acts, ro = router()
check("not localized -> 'Please place me at my home spot.'",
      ro.handle("explore the house", ctx("IDLE", localized=False))
      and acts.calls == [("say", "cant_start")])
acts, ro = router()
check("still relocalising at boot -> refused, never a run without a mission",
      ro.handle("explore", ctx("RELOCALIZE")) and acts.calls == [("say", "cant_start")])
acts, ro = router()
check("'explore' mid-mission -> can't",
      ro.handle("explore", ctx("EXPLORE")) and acts.calls == [("say", "cant_now")])
acts, ro = router()
ro.handle("explore", ctx("IDLE", interrupted=True))
check("an interrupted mission is waiting -> 'Continue the last exploration or "
      "start a new one?'", acts.calls == [("say", "continue_or_new")])
check("...'continue' -> resume", ro.handle("continue", ctx("IDLE", interrupted=True))
      and acts.calls[-1] == ("resume",))
acts, ro = router()
ro.handle("explore for 10 minutes", ctx("IDLE", interrupted=True))
check("...'start a new one' -> a fresh mission, minutes kept",
      ro.handle("start a new one", ctx("IDLE", interrupted=True))
      and acts.calls[-1] == ("start", 10, True), str(acts.calls))

acts, ro = router()
check("a dialog is open: 'stop' still halts",
      ro.handle("stop", ctx("ASK_LABEL", dialog="label")) and acts.calls == [("halt",)])
acts, ro = router()
check("...'wapas chalo' still finishes",
      ro.handle("wapas chalo", ctx("ASK_LABEL", dialog="label")) and acts.calls == [("finish",)])
acts, ro = router()
check("...every other utterance goes to the dialog, not the intents",
      ro.handle("list areas", ctx("ASK_LABEL", dialog="label"))
      and acts.calls == [("dialog_answer", "list areas")])


# =============================================================================
print("--- his real turn log through the router (read only) ---")

import collections                                      # noqa: E402
import re                                               # noqa: E402

LOG = ROOT / "logs" / "friday.log"
real = []
if LOG.exists():
    rx = re.compile(r'TRANSCRIBED "(.*)"\s+\(')
    for line in open(LOG, "rb"):
        m = rx.search(line.decode("utf-8", "replace"))
        if m and m.group(1).strip():
            real.append(m.group(1))
if len(real) < 100:
    print(f"        (only {len(real)} transcripts in the log; skipped)")
else:
    # A DELIBERATE label command is not an accident, whatever whisper made of
    # the name: 2026-10-04 his "save this as home" came out "Save this as full."
    deliberate = re.compile(r"^\W*(?:friday\W+)?(?:save|call|name|label|mark)\s+"
                            r"(?:this|it|here)\b", re.I)

    def claimed(state):
        out = collections.Counter()
        for t in real:
            if state != "IDLE" and deliberate.match(t):
                continue
            acts_, ro_ = router()
            if ro_.handle(t, ctx(state)):
                out[acts_.calls[0][0] if acts_.calls else "?"] += 1
        return out
    idle, exploring = claimed("IDLE"), claimed("EXPLORE")
    kinds = collections.Counter(ld.parse_label(t).kind for t in real)
    rate = kinds["NAME"] / len(real)
    print(f"        {len(real)} real transcripts. IDLE claims {dict(idle)}; "
          f"EXPLORE claims {dict(exploring)}; as dialog answers {dict(kinds)} "
          f"({100 * rate:.1f}% would be saved as a name)")
    check("with no mission, the only things claimed are his 'explore' commands",
          set(idle) <= {"start"}, str(dict(idle)))
    check("during a mission, nothing he ever said would label an area by accident",
          exploring.get("label_now", 0) == 0, str(dict(exploring)))
    check("as answers to her question, under 10% of random speech becomes a name",
          rate < 0.10, f"{100 * rate:.1f}%")


# =============================================================================
print("--- speech: fixed templates (spec 4.13) ---")

SAMPLE = {"suggestion": "kitchen", "name": "Kitchen", "d": 1.234, "old": "kitchen",
          "new": "Pantry", "m2": 23.6}
bad = []
for key, (template, mood) in speech.LINES.items():
    text, m = speech.render(key, **SAMPLE)
    if "{" in text or "}" in text or not text.endswith((".", "?")) \
            or mood not in speech.MOOD_TO_FACE:
        bad.append(key)
check(f"all {len(speech.LINES)} lines render, end in . or ?, and have a known mood",
      not bad, str(bad))
check("every mood maps onto one of the 20 faces that exist",
      all(f in ad.FakeFace.NAMES for f in speech.MOOD_TO_FACE.values()))
check("worried -> scared, determined -> thinking, an unknown mood -> neutral",
      (speech.face_for("worried"), speech.face_for("determined"),
       speech.face_for("smug")) == ("scared", "thinking", "neutral"))
EXACT = {
    "mission_start": "Starting exploration. I've marked this spot as my starting point.",
    "ask_label": "I've finished exploring this area. What should I call it?",
    "paused": "Stopping. Say continue when you're ready.",
    "returning": "Going back to where I started.",
    "arrived": "I've reached where I started. Exploration saved.",
    "help_wait": "I'll wait here. Say Friday, come back, to try again.",
    "cant_now": "I can't do that right now.",
    "failsafe": "I've stopped and I need help.",
    "map_reset": "My map was reset, so I need to re-explore my saved areas.",
}
check("the spec's wording, word for word",
      all(speech.render(k)[0] == v for k, v in EXACT.items()),
      str([k for k, v in EXACT.items() if speech.render(k)[0] != v]))
check("help: distance to one decimal, 'meters' as a word",
      speech.render("help", d=1.234)[0] == "I'm stuck about 1.2 meters from where "
                                           "I started. Please help me or clear the way.")
check("join: a / a and b / a, b and c",
      (speech.join_names(["a"]), speech.join_names(["a", "b"]),
       speech.join_names(["a", "b", "c"])) == ("a", "a and b", "a, b and c"))
ten = [f"room {i}" for i in range(1, 11)]
check("more than 8 names -> the first 8 and '2 more'",
      speech.join_names(ten) == ", ".join(ten[:8]) + " and 2 more")
for n, names in ((0, []), (1, ["Kitchen"]), (3, ["Kitchen", "Hall", "Harsh's room"]),
                 (10, ten)):
    text, mood = speech.summary(23.6, names, total=max(n, 4))
    ok = text.startswith("Exploration complete. My map now covers about 24 square meters")
    if n == 0:
        ok = ok and text.endswith("I didn't find any new areas.") and mood == "neutral"
    else:
        ok = ok and f"I saved {n} new area{'' if n == 1 else 's'}: " in text \
            and mood == "proud" and ("In total I know" in text) == (max(n, 4) > n)
    lt, _ = speech.list_line(names)
    ok = ok and (lt == "I haven't saved any areas yet." if n == 0
                 else lt.startswith(f"I know {n} area{'' if n == 1 else 's'}: "))
    stt, _ = speech.status_line(23.6, n)
    ok = ok and f"I've saved {n} area{'' if n == 1 else 's'} so far." in stt
    ok = ok and f"I saved {n} area{'' if n == 1 else 's'}." in speech.interrupted_line(n)[0]
    check(f"summary / list / status / interrupted for {n} areas", ok, text)
check("the summary for 3, in full",
      speech.summary(23.6, ["Kitchen", "Hall", "Harsh's room"], 5)[0]
      == "Exploration complete. My map now covers about 24 square meters, and I saved "
         "3 new areas: Kitchen, Hall and Harsh's room. In total I know 5 areas.")
check("the time-cap prefix and the not-entered line join it",
      speech.summary(10, ["Hall"], 1, time_cap=True, not_entered=2)[0]
      == "That's enough exploring for now. Exploration complete. My map now covers "
         "about 10 square meters, and I saved 1 new area: Hall. I also found 2 areas "
         "I didn't enter. You can name them on the dashboard.")
fa = ad.fake_adapters()
speech.say(fa, "help", d=2.0)
check("say(): the face (worried -> scared), then the voice",
      fa.face.shown == ["scared"] and fa.voice.said == [speech.render("help", d=2.0)[0]])


class Broken:
    def names(self):
        raise OSError("face process gone")

    def set(self, name):
        raise OSError("face process gone")


fb = ad.fake_adapters(face=Broken())
check("a dead face never silences her", speech.say(fb, "arrived")
      == "I've reached where I started. Exploration saved." and fb.voice.said)
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
