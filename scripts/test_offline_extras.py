#!/usr/bin/env python3
"""Offline commands that need no API credit (offline_extras.py, 2026-10-20),
and the out-of-credit fallback in brain.py. No network, no money, no motors.

    ./venv/bin/python scripts/test_offline_extras.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import brain                                            # noqa: E402
import offline                                          # noqa: E402
import offline_extras as ox                             # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


def said(text: str, strict: bool = True):
    i = offline.understand(text, strict=strict)
    return None if i is None else (i.name, i.say)


print("the date")
check("struct_time -> a sentence",
      ox.date_phrase(time.strptime("2026-10-21", "%Y-%m-%d"))
      == "It's Wednesday, the 21st of October 2026.")
check("11th, not 11st", "the 11th of" in ox.date_phrase(time.strptime("2026-11-11", "%Y-%m-%d")))
for q in ("what's the date today", "what is the date", "what day is it",
          "Friday, what day is today?", "aaj kya date hai", "today's date"):
    got = said(q)
    check(f"{q!r} -> date", got is not None and got[0] == "date", str(got))

print("the calculator")
for q, want in (("what is 25 times 4", "It's 100."),
                ("what's 15 percent of 200", "15 percent of 200 is 30."),
                ("how much is 100 divided by 8", "It's 12.5."),
                ("what is 2 plus 3 times 4", "It's 14."),
                ("what is the square root of 81", "The square root of 81 is 9."),
                ("what is twenty five plus seventeen", "It's 42."),
                ("what is 12 squared", "12 squared is 144."),
                ("what is 2 to the power of 10", "It's 1,024."),
                ("calculate 7 minus 10", "It's -3."),
                ("what is 5 into 6", "It's 30."),
                ("what is 5 divided by 0", "You can't divide by zero."),
                ("what is 1 over 3", "It's 0.33.")):
    got = said(q)
    check(f"{q!r} -> {want!r}", got == ("calc", want), str(got))
check("15 percent of 200 no longer sets her SPEED", said("what's 15 percent of 200")[0] != "speed")
check("'speed 40 percent' is still the speed", offline.understand("speed 40 percent").name == "speed")
for q in ("what is the time", "what's the weather", "what is your name",
          "what is the capital of France", "what is 5", "what is love"):
    got = said(q)
    check(f"{q!r} is not a sum", got is None or got[0] != "calc", str(got))
check("a huge power is refused, not computed", ox.calculate("what is 9 to the power of 99999")
      == "That number is too big for me.")

print("jokes, coin, dice, numbers")
rng = random.Random(3)
seen = [ox.joke(rng) for _ in range(30)]
check("a joke never repeats twice in a row", all(a != b for a, b in zip(seen, seen[1:])))
check("'tell me a joke' -> joke", said("tell me a joke")[0] == "joke")
check("'koi joke sunao' -> joke", said("koi joke sunao")[0] == "joke")
check("coin -> Heads/Tails", said("flip a coin")[1] in ("Heads.", "Tails."))
check("dice -> 1..6", said("roll a dice")[1] in {f"You rolled a {n}." for n in range(1, 7)})
nums = {int(ox.understand("pick a number between 1 and 3", True, rng)[1].rstrip("."))
        for _ in range(40)}
check("pick a number between 1 and 3 -> only 1..3, all of them", nums == {1, 2, 3}, str(nums))

print("small talk is OFFLINE only (the brain answers it when it can)")
for q, want in (("how are you", "talk"), ("thank you", "talk"), ("what's your name", "talk"),
                ("hello", "talk")):
    check(f"{q!r} with a brain -> left to the brain", said(q, strict=True) is None,
          str(said(q, strict=True)))
    got = said(q, strict=False)
    check(f"{q!r} offline -> a line", got is not None and got[0] == want, str(got))

print("nothing old was taken")
for q, name in (("stop", "stop"), ("Friday stop", "stop"), ("stop it", "stop"),
                ("stop that", "stop"), ("stop now", "stop"), ("what's the time", "time"),
                ("go forward", "go_forward"), ("what can you do", "help"),
                ("play lofi music on YouTube", "laptop_search")):
    i = offline.understand(q, strict=True)
    check(f"{q!r} is still {name}", i is not None and i.name == name, str(i and i.name))
for q in ("which rooms do you know", "what places have you saved", "where can you go"):
    i = offline.understand(q, strict=True)
    check(f"{q!r} -> the place list, said in one sentence",
          i is not None and i.tool == "list_places" and i.args == {"spoken": True}
          and i.speak_result, str(i and (i.tool, i.args)))
check("'go to the bedroom' is still a drive, not the list",
      offline.understand("go to the bedroom", strict=True).name == "nav_go")
print("what whisper makes of 'play a song' and 'open WhatsApp' (2026-10-20 22:3x)")
for q in ("Blair Assong", "Blear song.", "Bleed during that song.", "Play Asong"):
    i = offline.understand(q, strict=True)
    check(f"{q!r} -> plays a song", i is not None and i.tool == "laptop_search"
          and i.args.get("play") is True, str(i and (i.tool, i.args)))
for q in ("I like that song", "sing me a song", "what song is this"):
    i = offline.understand(q, strict=True)
    check(f"{q!r} -> NOT a song on the laptop", i is None or i.tool != "laptop_search",
          str(i and (i.tool, i.args)))
for q in ("play a better song, change it.", "play another song", "change the song"):
    i = offline.understand(q, strict=True)
    check(f"{q!r} -> the next track, not a search for those words",
          i is not None and i.args == {"action": "next"}, str(i and (i.tool, i.args)))
import laptop_research                                   # noqa: E402
apps = [{"name": "WhatsApp", "via": "store"}, {"name": "Excel", "via": "desktop"},
        {"name": "Steam", "via": "desktop"}]
for q in ("Watch", "words", "watch app", "what's app"):
    got, _ = laptop_research.match_app(q, apps)
    check(f"open {q!r} -> WhatsApp", got is not None and got["name"] == "WhatsApp", str(got))
got, _ = laptop_research.match_app("Steam", apps)
check("...and a real name still wins (Steam is Steam)", got and got["name"] == "Steam")
print("recording his driving (helper 17)")
for q in ("record my driving", "start recording", "record me driving", "watch how I drive"):
    i = offline.understand(q, strict=True)
    check(f"{q!r} -> starts recording", i is not None and i.tool == "laptop_record"
          and i.args == {"action": "start"}, str(i and (i.tool, i.args)))
i = offline.understand("stop recording", strict=True)
check("'stop recording' stays a STOP (and the stop tool ends a recording)",
      i is not None and i.name == "stop", str(i and i.name))
mv = (ROOT / "mcp_server/tools/movement.py").read_text()
check("...the stop tool ends a recording AFTER the motors",
      mv.index("safety.emergency_stop()") < mv.index("research_tools.recording_on()"))
check("help mentions the laptop and sums",
      "laptop" in offline.help_text() and "sums" in offline.help_text())

print("out of API credit")
brain._CREDIT_OUT_AT[0] = 0.0
check("not out at start", not brain.credits_out())
check("an unrelated error is not a credit error",
      not brain.note_api_error(RuntimeError("timeout")) and not brain.credits_out())
err = Exception("Error code: 400 - {'type': 'error', 'error': {'type': "
                "'invalid_request_error', 'message': 'Your credit balance is too low "
                "to access the Anthropic API.'}}")
check("the credit 400 is recognised", brain.note_api_error(err) and brain.credits_out())
brain._CREDIT_OUT_AT[0] = time.monotonic() - brain.CREDIT_RETRY_S - 1
check("...and forgotten after CREDIT_RETRY_S (he may have topped up)", not brain.credits_out())
brain._CREDIT_OUT_AT[0] = 0.0
main_src = (ROOT / "main.py").read_text()
check("main routes a turn to _no_credit_turn while credits are out",
      "brain_module.credits_out()" in main_src and "async def _no_credit_turn" in main_src)
check("a brain call that fails on credit tries the offline commands",
      "if brain_module.note_api_error(exc):" in main_src)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
