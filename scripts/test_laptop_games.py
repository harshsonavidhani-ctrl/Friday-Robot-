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

print("PLAY: only on his word, only slow games")
IDLE["v"] = {"idle_s": 3.0, "app": "gta5.exe", "title": "Grand Theft Auto V"}
said = lg.play_start(say=say)
check("GTA in front -> 'too fast for me', nothing started", "too fast" in said and not lg.playing(), said)
IDLE["v"] = {"idle_s": 3.0, "app": "chrome.exe", "title": "YouTube - Google Chrome"}
said = lg.play_start("solitaire", say=say)
check("no game in front -> 'open it first'", said.startswith("Open Solitaire") and not lg.playing(), said)
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

print("parse_move")
check("a move", lg.parse_move('sure: {"x": 0.2, "y": 0.9, "why": "w"}') == {"x": 0.2, "y": 0.9, "why": "w"})
check("off the image -> refused", lg.parse_move('{"x": 1.4, "y": 0.2}') is None)
check("done", lg.parse_move('{"done": true, "why": "won"}') == {"done": True, "why": "won"})
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
check("out of API credit -> no look at all", lg.watch_item() is None and len(CALLS) == n)
brain._CREDIT_OUT_AT[0] = 0.0

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
