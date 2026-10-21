#!/usr/bin/env python3
"""laptop_games.py (2026-10-20): she watches him play and reacts; she plays a
SLOW game herself only on his word. Laptop, vision model and voice all faked:
no network, no money, nothing clicked.

    ./venv/bin/python scripts/test_laptop_games.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402

_TMP = Path(tempfile.mkdtemp())
config.LAPTOP_SEEN_FILE = _TMP / "laptop_seen.json"
config.LAPTOP_APPS_FILE = _TMP / "laptop_apps.json"
config.RATE_LIMIT_FILE = _TMP / "rate_limits.json"
config.LAPTOP_GAME_STATE_FILE = _TMP / "laptop_game.json"
config.GAME_PLAY_PAUSE_S = 0.01
config.LAPTOP_GAME_WATCH = True

import brain                                             # noqa: E402
import laptop_games as lg                                # noqa: E402
import laptop_play                                       # noqa: E402
import laptop_research as lr                             # noqa: E402
import scene_describe                                    # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


lr._save_apps([{"name": "Solitaire & Casual Games", "via": "store"},
               {"name": "Grand Theft Auto V Enhanced", "via": "desktop"},
               {"name": "WhatsApp", "via": "store"}])
IDLE = {"v": {"idle_s": 3.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}}
lr.idle = lambda: IDLE["v"]
SHOTS, CLICKS = [], []
SCREEN = {"ok": True}


def fake_screen(game):
    SHOTS.append(game)
    return (b"\x89PNG fake", "") if SCREEN["ok"] else (None, "that game is not the window in front")


lr.screen = fake_screen
lr.click = lambda game, x, y: (CLICKS.append((game, round(x, 2), round(y, 2))) or (True, ""))
STEPS, LAUNCHED = [], []
VERSION = {"v": 7}
lr.helper_version = lambda: VERSION["v"]
KEPT: list = []


def fake_step(game, keys, ms, before=None, after=None, shot=True, keep=0):
    if keys:
        STEPS.append((game, tuple(keys), ms, before, after))
        KEPT.append(keep)
    return (b"\x89PNG step" if shot else b""), ""


lr.step = fake_step


def fake_launch(name):
    LAUNCHED.append(name)
    IDLE["v"] = {"idle_s": 1.0, "app": "x.exe", "title": name}
    return True, ""


lr.launch = fake_launch
config.GAME_OPEN_WAIT_S = 3.0
laptop_play.laptop_on = lambda now: True
REPLIES: list = []
CALLS: list = []


def fake_call(limit, png, system, text, max_tokens, media_type="image/jpeg"):
    CALLS.append((limit, media_type, system[:40]))
    return REPLIES.pop(0) if REPLIES else (None, "nothing scripted")


scene_describe._call = fake_call
brain._CREDIT_OUT_AT[0] = 0.0
SAID: list[str] = []
say = SAID.append

print("which game is in front")
apps = lr.known_apps()
check("Solitaire's window -> 'Solitaire & Casual Games'",
      lg.game_in_front(IDLE["v"], apps) == "Solitaire & Casual Games")
check("WhatsApp in front -> no game (never)",
      lg.game_in_front({"title": "WhatsApp", "idle_s": 1}, apps) is None)
check("GTA's window -> GTA", lg.game_in_front({"title": "Grand Theft Auto V", "idle_s": 1}, apps)
      == "Grand Theft Auto V Enhanced")

print("PLAY: only on his word, ANY game")
IDLE["v"] = {"idle_s": 3.0, "app": "gta5.exe", "title": "Grand Theft Auto V"}
said = lg.play_start(say=say)
check("GTA in front, helper 7 -> 'copy the new helper', nothing started",
      "too old" in said and not lg.playing(), said)
IDLE["v"] = {"idle_s": 3.0, "app": "chrome.exe", "title": "YouTube - Google Chrome"}
said = lg.play_start("whatsapp", say=say)
check("'you play WhatsApp' -> not a game, nothing opened",
      "can't find" in said and not LAUNCHED and not lg.playing(), said)
said = lg.play_start(say=say)
check("no game named, none in front -> 'open it first'",
      said.startswith("Open the game first") and not lg.playing(), said)
brain._CREDIT_OUT_AT[0] = time.monotonic()
said = lg.play_start(say=say)
check("out of API credit -> says so, does nothing", "credits have run out" in said, said)
brain._CREDIT_OUT_AT[0] = 0.0

IDLE["v"] = {"idle_s": 3.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
REPLIES[:] = [('{"x": 0.31, "y": 0.62, "why": "move the red 7"}', ""),
              ("not json at all", ""),
              ('{"x": 0.55, "y": 0.40, "why": "draw a card"}', ""),
              ('{"done": true, "why": "no more moves"}', "")]
said = lg.play_start(say=say)
check("Solitaire in front, his word -> she starts and says so",
      said.startswith("Okay, my turn! I'm playing Solitaire"), said)
for _ in range(100):
    if not lg.playing():
        break
    time.sleep(0.02)
check("each move: a picture, a look, a click inside the game",
      CLICKS == [("Solitaire & Casual Games", 0.31, 0.62), ("Solitaire & Casual Games", 0.55, 0.4)],
      str(CLICKS))
check("...looks are PNG, through the paid-look limit 'game_play'",
      all(c[0] == "game_play" and c[1] == "image/png" for c in CALLS), str(CALLS))
check("...a garbled answer is skipped, not clicked", len(CLICKS) == 2)
check("...and she says when she's done", SAID and "done playing" in SAID[-1], str(SAID))

print("stopping her")
REPLIES[:] = [('{"x": 0.5, "y": 0.5, "why": "x"}', "")] * 200
config.GAME_PLAY_PAUSE_S = 0.2
CLICKS.clear()
lg.play_start(say=say)
time.sleep(0.1)
check("'stop' (the stop tool's hook) ends her game", lg.stop_playing() is True)
for _ in range(100):
    if not lg.playing():
        break
    time.sleep(0.02)
check("...at once, and she hands it back", not lg.playing() and SAID[-1] == "Okay, your turn.",
      str(SAID[-1:]))
check("stop when she isn't playing -> False (nothing to say)", lg.stop_playing() is False)
config.GAME_PLAY_PAUSE_S = 0.01
SCREEN["ok"] = False
CLICKS.clear()
lg.play_start(say=say)
for _ in range(100):
    if not lg.playing():
        break
    time.sleep(0.02)
check("he switches to another window -> the laptop refuses the picture, she stops, no click",
      CLICKS == [] and "isn't in front" in SAID[-1], str(SAID[-1:]))
SCREEN["ok"] = True
src = (ROOT / "mcp_server/tools/movement.py").read_text()
check("the stop tool calls stop_playing AFTER the motors",
      src.index("safety.emergency_stop()") < src.index("laptop_games.stop_playing()"))

print("a fast game, with the keyboard (helper 8)")
VERSION["v"] = 8
SAID.clear()
CALLS.clear()
REPLIES[:] = [('{"keys": ["up"], "ms": 900, "keep": true, "why": "go"}', ""),
              ('{"keys": ["up", "left"], "ms": 400, "why": "bend"}', ""),
              ('{"keys": ["alt", "f4"], "ms": 100}', ""),
              ('{"x": 0.5, "y": 0.8, "why": "continue button"}', ""),
              ('{"done": true, "why": "race over"}', "")]
IDLE["v"] = {"idle_s": 3.0, "app": "gta5.exe", "title": "Grand Theft Auto V"}
said = lg.play_start(say=say)
check("GTA in front, helper 8 -> she plays it", said.startswith("Okay, my turn! I'm playing Grand"),
      said)
for _ in range(200):
    if not lg.playing():
        break
    time.sleep(0.02)
check("...each step holds the keys she chose, for as long as she chose",
      [x[1:3] for x in STEPS] == [(("up",), 900), (("up", "left"), 400)], str(STEPS))
check("...'keep' keeps the keys down while she thinks (GAME_KEEP_MS), only when she asks",
      KEPT[:2] == [config.GAME_KEEP_MS, 0], str(KEPT))
check("...keys the laptop would refuse are never sent (alt+f4)",
      all("f4" not in x[1] for x in STEPS))
check("...a menu button is still a click", CLICKS[-1:] == [("Grand Theft Auto V Enhanced", 0.5, 0.8)],
      str(CLICKS[-1:]))
check("...told it is a keyboard game", any("keyboard" in c[2] or "keys" in c[2] for c in CALLS)
      or "keyboard" in lg.KEYS_SYSTEM, str(CALLS[:1]))
check("...and she says when it is over", SAID and "race over" in SAID[-1], str(SAID[-1:]))
st = lg.state()
check("...and the dashboard's file says what happened (not running, moves, why)",
      st.get("running") is False and st.get("moves") == 4 and st.get("fast") is True
      and st.get("why") == "race over", str(st))

STEPS.clear()
config.GAME_PAUSE_KEYS = {"grand": "escape"}
REPLIES[:] = [('{"keys": ["up"], "ms": 500}', ""), ('{"keys": ["right"], "ms": 300}', ""),
              ('{"done": true, "why": "ok"}', "")]
lg.play_start(say=say)
for _ in range(200):
    if not lg.playing():
        break
    time.sleep(0.02)
check("a pause key: the first step pauses after, the next un-pauses before and pauses after",
      [(x[3], x[4]) for x in STEPS] == [(None, "escape"), ("escape", "escape")], str(STEPS))
config.GAME_PAUSE_KEYS = {}

print("a fast 3D game with a driver on the laptop (helper 10)")
AUTO: list = []
lr.autopilot = lambda game, mode, **kw: (AUTO.append((mode, kw.get("bias"), kw.get("speed")))
                                         or ({"running": mode != "off"}, ""))
VERSION["v"] = 10
STEPS.clear()
CALLS.clear()
REPLIES[:] = [('{"drive": {"bias": -0.5, "speed": 0.8}, "why": "left at the junction"}', ""),
              ('{"keys": ["s"], "ms": 800, "why": "back out of the wall"}', ""),
              ('{"drive": {"bias": 0, "speed": 1}}', ""),
              ('{"done": true, "why": "parked"}', "")]
IDLE["v"] = {"idle_s": 3.0, "app": "gta5.exe", "title": "Grand Theft Auto V"}
lg.play_start(say=say)
for _ in range(300):
    if not lg.playing():
        break
    time.sleep(0.02)
check("GTA + helper 10: the laptop's driver starts first, in road mode",
      AUTO and AUTO[0][0] == "road", str(AUTO))
check("...her looks become directions to it (bias, speed)",
      ("road", -0.5, 0.8) in AUTO and ("road", 0.0, 1.0) in AUTO, str(AUTO))
check("...she is told it drives for her", any("AUTOPILOT" in c[2] or "autopilot" in c[2].lower()
                                             for c in CALLS) or "AUTOPILOT" in lg.DRIVE_SYSTEM)
check("...a manual move is still a step", [x[1] for x in STEPS] == [("s",)], str(STEPS))
check("...and when she is done the driver is switched off", AUTO[-1][0] == "off", str(AUTO))
check("autopilot_for: GTA drives on the road, Hill Climb cruises, Solitaire has none",
      lg.autopilot_for("Grand Theft Auto V Enhanced")["mode"] == "road"
      and lg.autopilot_for("Hill Climb Racing")["mode"] == "cruise"
      and lg.autopilot_for("Solitaire & Casual Games") is None)
check("'drive' parses, clamped",
      lg.parse_move('{"drive": {"bias": -3, "speed": 2}}', keys_ok=True)
      == {"drive": True, "bias": -1.0, "speed": 1.0, "why": ""})
VERSION["v"] = 8

print("a game he names that is not open is opened first")
LAUNCHED.clear()
STEPS.clear()
IDLE["v"] = {"idle_s": 3.0, "app": "chrome.exe", "title": "YouTube - Google Chrome"}
REPLIES[:] = [('{"keys": ["up"], "ms": 300}', ""), ('{"done": true, "why": "ok"}', "")]
said = lg.play_start("gta", say=say)
check("'you play GTA' with Chrome in front -> opens GTA and says so",
      LAUNCHED == ["Grand Theft Auto V Enhanced"] and said.startswith("Opening Grand"), said)
for _ in range(300):
    if not lg.playing():
        break
    time.sleep(0.02)
check("...then plays it once it is in front", STEPS and STEPS[0][0] == "Grand Theft Auto V Enhanced",
      str(STEPS))
VERSION["v"] = 7
LAUNCHED.clear()
IDLE["v"] = {"idle_s": 3.0, "app": "chrome.exe", "title": "YouTube - Google Chrome"}
said = lg.play_start("gta", say=say)
check("...but not with an old helper (it could not press the keys)",
      not LAUNCHED and "too old" in said, said)
VERSION["v"] = 8

print("parse_move")
check("keys", lg.parse_move('{"keys": ["UP", "left"], "ms": 600, "why": "w"}', keys_ok=True)
      == {"keys": ["up", "left"], "ms": 600, "keep": False, "why": "w"})
check("keys only in a keyboard game", lg.parse_move('{"keys": ["up"], "ms": 600}') is None)
check("a key the laptop does not have -> refused",
      lg.parse_move('{"keys": ["ctrl", "w"], "ms": 600}', keys_ok=True) is None)
check("ms held to 100..GAME_STEP_MAX_MS",
      lg.parse_move('{"keys": ["up"], "ms": 9000}', keys_ok=True)["ms"] == config.GAME_STEP_MAX_MS
      and lg.parse_move('{"keys": ["up"], "ms": 5}', keys_ok=True)["ms"] == 100)
check("a move", lg.parse_move('sure: {"x": 0.2, "y": 0.9, "why": "w"}') == {"x": 0.2, "y": 0.9, "why": "w"})
check("off the image -> refused", lg.parse_move('{"x": 1.4, "y": 0.2}') is None)
check("done", lg.parse_move('{"done": true, "why": "won"}') == {"done": True, "why": "won"})
check("the title bar (close/minimise, top 6%) is never clicked",
      lg.parse_move('{"x": 0.976, "y": 0.02, "why": "close"}') is None
      and lg.parse_move('{"x": 0.5, "y": 0.05}') is None
      and lg.parse_move('{"x": 0.5, "y": 0.07}') is not None)
check("nonsense", lg.parse_move("click the king") is None)

print("WATCH: she reacts while he plays")
lg._state["last_watch"] = 0.0
IDLE["v"] = {"idle_s": 2.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
REPLIES[:] = [("Ooh, that ace is free! Grab it!", "")]
CALLS.clear()
item = lg.watch_item()
check("he is playing Solitaire -> one look, one reaction",
      item is not None and item["kind"] == "game_watch" and "ace" in item["say"], str(item))
check("...through 'game_watch' (its own limit), as a PNG",
      CALLS and CALLS[-1][0] == "game_watch" and CALLS[-1][1] == "image/png", str(CALLS))
check("...and not again within the gap", lg.watch_item() is None)
lg._state["last_watch"] = 0.0
REPLIES[:] = [("SKIP", "")]
check("nothing worth saying -> SKIP -> silence", lg.watch_item() is None)
lg._state["last_watch"] = 0.0
IDLE["v"] = {"idle_s": 0.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
REPLIES[:] = [("Nice move!", "")]
check("touched 0 s ago counts as playing (0 is not 'never')", lg.watch_item() is not None)
lg._state["last_watch"] = 0.0
IDLE["v"] = {"idle_s": 600.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
n = len(CALLS)
check("game open but untouched 10 min -> he isn't playing -> no look (no money)",
      lg.watch_item() is None and len(CALLS) == n)
IDLE["v"] = {"idle_s": 2.0, "app": "whatsapp.exe", "title": "WhatsApp"}
lg._state["last_watch"] = 0.0
check("WhatsApp in front -> never looks", lg.watch_item() is None and len(CALLS) == n)
config.LAPTOP_GAME_WATCH = False
IDLE["v"] = {"idle_s": 2.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
check("LAPTOP_GAME_WATCH False -> never", lg.watch_item() is None and len(CALLS) == n)
config.LAPTOP_GAME_WATCH = True
brain._CREDIT_OUT_AT[0] = time.monotonic()
lg._state["last_watch"] = 0.0
item = lg.watch_item()
check("out of API credit -> no paid look, an OFFLINE cheer instead",
      item is not None and item.get("offline") and len(CALLS) == n, str(item))
check("...not again within the cheer gap (150 s)", lg.watch_item() is None)
brain._CREDIT_OUT_AT[0] = 0.0

print("offline cheering")
import random                                             # noqa: E402
lg._state.update(last_game="", recent=[])
first = lg.cheer_line("Asphalt 8 - Car Racing Game", random.Random(1))
check("the first line for a new game is a hello with its short name",
      "Asphalt 8" in first and "Car Racing" not in first, first)
lines = [lg.cheer_line("Asphalt 8 - Car Racing Game", random.Random(i)) for i in range(8)]
check("then racing lines (or general ones), never a line from the last 5 again",
      all(lines[i] not in lines[max(0, i - 4):i] for i in range(len(lines))), str(lines))
check("kinds: GTA, racing, horror, shooter, cards, building, flying",
      [lg.cheer_kind(g) for g in ("Grand Theft Auto V Enhanced", "Asphalt 8 - Car Racing Game",
                                   "Evil Nun The Broken Mask", "Free Fire MAX Fire Kickoff",
                                   "Solitaire & Casual Games", "Minecraft for Windows", "DCS World")]
      == ["gta", "racing", "horror", "shooter", "cards", "building", "flying"])
lr._save_apps(lr.known_apps() + [{"name": "Asphalt 8 - Car Racing Game", "via": "store"}])
apps = lr.known_apps()
check("helper 5 (no window title): 'Asphalt8.exe' is Asphalt",
      lg.game_in_front({"idle_s": 1, "app": "Asphalt8.exe"}, apps) == "Asphalt 8 - Car Racing Game")
check("...'GTA5.exe' is GTA (nickname)",
      lg.game_in_front({"idle_s": 1, "app": "GTA5.exe"}, apps) == "Grand Theft Auto V Enhanced")
check("...'WhatsApp.exe' is not a game", lg.game_in_front({"idle_s": 1, "app": "WhatsApp.exe"}, apps) is None)
SCREEN["ok"] = False
IDLE["v"] = {"idle_s": 2.0, "app": "solitaire.exe", "title": "Solitaire & Casual Games"}
lg._state["last_watch"] = 0.0
n = len(CALLS)
item = lg.watch_item()
check("no picture (an older helper / another window) -> a cheer, no paid look",
      item is not None and item.get("offline") and len(CALLS) == n, str(item))
SCREEN["ok"] = True
config.LAPTOP_GAME_WATCH_ONLINE = False
lg._state["last_watch"] = 0.0
item = lg.watch_item()
check("LAPTOP_GAME_WATCH_ONLINE False -> always cheers, never pays",
      item is not None and item.get("offline") and len(CALLS) == n, str(item))
config.LAPTOP_GAME_WATCH_ONLINE = True

print("her words")
import offline                                            # noqa: E402
for phrase, args in (("Friday, you play the game", {"action": "start", "game": ""}),
                     ("you play solitaire", {"action": "start", "game": "solitaire"}),
                     ("play it yourself", {"action": "start", "game": ""}),
                     ("stop playing", {"action": "stop"}), ("my turn", {"action": "stop"})):
    got = offline.understand(phrase, strict=True)
    check(f"{phrase!r} -> laptop_game_play {args}", got is not None and got.tool == "laptop_game_play"
          and got.args == args, str(got and (got.tool, got.args)))
check("'stop' is still the ROBOT stop (which also ends a game)",
      offline.understand("Friday stop").name == "stop")
check("'play minecraft' still opens/plays it for HIM",
      offline.understand("play minecraft", strict=True).tool == "laptop_play")

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
