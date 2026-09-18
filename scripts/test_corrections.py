#!/usr/bin/env python3
"""File 09 step 1: correction capture. Scratch database, no brain, no robot.

    ./venv/bin/python scripts/test_corrections.py

⚠ RUNS ON ITS OWN DATABASE. config.MEMORY_DB is pointed at a temp file BEFORE
memory is imported, so nothing here touches state/memory.db -- which is the
live one, is in WAL mode, and holds things Harsh actually told her.

The cases that matter are the ones that must NOT store anything. Everything
saved here is read back later as a standing instruction about how to behave,
so a false positive teaches Friday a preference nobody ever expressed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

_scratch = Path(tempfile.mkdtemp(prefix="friday-corrections-")) / "memory.db"
config.MEMORY_DB = _scratch

import corrections                                      # noqa: E402
import memory                                           # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def main() -> int:
    print("DETECTION — what counts as a correction")
    for said in ("nahi yeh galat hai",
                 "nahi ye galat tha",
                 "aise mat bolo",
                 "aisa mat bolo Friday",
                 "that's not right",
                 "that is wrong",
                 "you're wrong about that",
                 "no that's not what I said",
                 "don't call me Sujoo",
                 "stop saying that",
                 "that's not correct",
                 "wrong answer"):
        check(f"correction: {said!r}",
              corrections.looks_like_correction(said) is not None)

    print("\nDETECTION — what must NOT count")
    for said in ("what's wrong with the wifi",
                 "is that right?",
                 "no problem at all",
                 "tell me if anything goes wrong",
                 "what did I say wrong yesterday about the map",
                 "right, let's go",
                 "say hello to everyone"):
        found = corrections.looks_like_correction(said)
        check(f"not a correction: {said!r}", found is None,
              f"matched {found!r} — this would teach her a preference "
              f"nobody expressed")

    print("\nSTORING")
    corrections.spoke("Sure thing, Sujoo.")
    stored = corrections.capture("nahi yeh galat hai")
    check("a correction with a reply behind it is kept", bool(stored))
    check("...under its own kind, not as an ordinary fact",
          bool(stored) and stored["kind"] == config.LEARNED_PREFERENCE_KIND,
          str(stored))
    check("...and it names what she said",
          bool(stored) and "Sujoo" in stored["text"], str(stored))

    check("the same sentence offered twice is stored once",
          corrections.capture("nahi yeh galat hai") is None)

    corrections.spoke("Okay.")
    brainy = corrections.capture(
        "call me Harsh",
        note="Harsh does not want to be called Sujoo; his name is Harsh")
    check("the brain's own correction is kept without a phrase match",
          bool(brainy))
    check("...and the brain's wording is what is stored",
          bool(brainy) and brainy["text"].startswith("Harsh does not want"),
          str(brainy))

    print("\nSTORING — what must NOT be stored")
    corrections._last_reply = ""
    corrections._last_said = ""
    check("a correction with nothing behind it stores nothing",
          corrections.capture("that's not right") is None,
          "a bare complaint with no reply to attach it to is not a preference")
    corrections._last_said = ""
    corrections.spoke("It is 4 pm.")
    check("an ordinary sentence stores nothing",
          corrections.capture("what's wrong with the wifi") is None)

    print("\nQUERYING — separate from ordinary memory")
    memory.remember("Harsh's sister is called Riya", source="brain-note")
    prefs = corrections.preferences()
    check("both corrections come back", len(prefs) == 2, str(prefs))
    check("the ordinary note does not",
          all("Riya" not in row["text"] for row in prefs))
    check("each carries the date it was learned",
          all(len(row.get("date", "")) == 10 for row in prefs), str(prefs))
    check("ordinary recall still finds the ordinary note",
          any("Riya" in f.text for f in memory.recall("sister")))

    print("\nTHE KIND MUST SURVIVE A NEAR-DUPLICATE UPDATE")
    before = corrections.preferences()[0]
    memory.remember(before["text"], source="brain-note")      # kind defaults
    after = [row for row in corrections.preferences()
             if row["id"] == before["id"]]
    check("a plain note landing on a preference does not declassify it",
          bool(after),
          "the correction dropped out of the learned_preference query")
    reasserted = memory.remember("Harsh prefers short answers in the morning",
                                 kind=config.LEARNED_PREFERENCE_KIND,
                                 source="correction")
    check("an explicit kind is still applied on update or insert",
          reasserted["kind"] == config.LEARNED_PREFERENCE_KIND)

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    print(f"(scratch database: {_scratch})")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
