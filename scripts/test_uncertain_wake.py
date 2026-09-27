#!/usr/bin/env python3
"""An uncertain wake is heard in silence. No microphone, no money.

    ./venv/bin/python scripts/test_uncertain_wake.py

Harsh, 2026-09-27: "it should never fire randomly only by hey friday". The
verifier cannot separate his real commands from video lines by score (see
config.WAKE_CONFIDENT_VERIFY for the table), so a wake below the confident
level shows nothing -- no beep, no face -- unless what follows is for her.
These cases pin the two decisions: which wakes are uncertain, and which words
after one still get an answer. The brain is a fake that records the question.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-wake-"))
config.SPEAKING_FILE = TMP / "speaking.json"            # "she spoke" is ours to set

import main                                             # noqa: E402
import offline                                          # noqa: E402

offline.online = lambda force=False: True              # never touch the network

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


class FakeTurn:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str, dash: bool = True) -> None:
        self.notes.append(text)


class FakeBrain:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def addressed_strictly(self, text: str):
        self.asked.append(text)
        return self.answer


def meant(spoken: str, brain) -> bool:
    return asyncio.run(main._meant_for_her(brain, spoken, FakeTurn()))


print("\nwhich wakes are uncertain")
config.WAKE_CONFIDENT_VERIFY = 0.85
for trigger, want in (("openwakeword: 0.94 v0.70", True),
                      ("openwakeword: 1.00 v0.60", True),
                      ("openwakeword: 0.94 v0.85", False),
                      ("openwakeword: 0.52 v0.97", False),
                      ("openwakeword: 0.94", False),        # verifier off
                      ("enter-key: pressed", False),
                      ("", False)):
    check(f"{trigger!r} -> {'gated' if want else 'certain'}",
          main._uncertain_wake(trigger) is want)

print("\nwhich wakes BEEP (he waits for it) -- separate from the answer gate")
config.WAKE_ACK_VERIFY = 0.5
for trigger, want in (("openwakeword: 0.94 v0.79", True),   # his, live 17:23
                      ("openwakeword: 0.95 v0.58", True),
                      ("openwakeword: 1.00 v0.50", True),
                      ("openwakeword: 0.55 v0.47", False),  # below the ack line
                      ("openwakeword: 0.94 v0.90", True),   # certain
                      ("openwakeword: 0.94", True),         # verifier off
                      ("enter-key: pressed", True)):
    check(f"{trigger!r} -> {'beep' if want else 'silent'}",
          main._acknowledged_wake(trigger) is want)
check("a beeping uncertain wake is still answer-gated",
      main._acknowledged_wake("openwakeword: 0.9 v0.6")
      and main._uncertain_wake("openwakeword: 0.9 v0.6"))
config.WAKE_UNCERTAIN_SILENT = False
check("switched off: every wake beeps again",
      main._uncertain_wake("openwakeword: 0.94 v0.61") is False)
config.WAKE_UNCERTAIN_SILENT = True

print("\nafter an uncertain wake: answered without asking anyone")
config.SPEAKING_FILE.unlink(missing_ok=True)
brain = FakeBrain(False)
for said in ("stop", "move forward", "Go explore for 2 minutes",
             "what is the time?", "keep describing what you see"):
    check(f"{said!r} is a known order", meant(said, brain))
check("...and none of them cost a model call", brain.asked == [])

print("\nafter an uncertain wake: the strict check decides the rest")
brain = FakeBrain(False)
for said in ("and video on it.", "It is better today than the last one.",
             "Thank you very much.", "I'm going to take a shower."):
    check(f"{said!r} with a NO -> silent", not meant(said, brain))
check("each was asked once", len(brain.asked) == 4, str(brain.asked))
check("a YES answers", meant("What do you think of this jacket?",
                             FakeBrain(True)))
check("no answer from the check is silence", not meant(
    "What do you think of this jacket?", FakeBrain(None)))
check("no brain at all is silence", not meant(
    "What do you think of this jacket?", None))


class Boom(FakeBrain):
    async def addressed_strictly(self, text):
        raise RuntimeError("API down")


check("a failing check is silence, not a crash",
      not meant("What do you think of this jacket?", Boom(None)))

print("\nmid-conversation with her, a question is his")
config.SPEAKING_FILE.write_text(json.dumps(
    {"state": "quiet", "at": time.time(), "until": time.time()}))
brain = FakeBrain(False)
check("a question seconds after she spoke answers without the check",
      meant("what do you like?", brain) and brain.asked == [], str(brain.asked))
config.SPEAKING_FILE.unlink()

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
