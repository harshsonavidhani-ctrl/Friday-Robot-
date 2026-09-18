#!/usr/bin/env python3
"""File 09 steps 2-3: preferences reach the prompt, gaps stay silent.

    ./venv/bin/python scripts/test_preferences.py

Scratch database (config.MEMORY_DB is redirected before memory is imported),
no brain, no robot. The point of step 2 is that being corrected changes
BEHAVIOUR, so what is tested is that the correction actually reaches the text
the model is given — and as an instruction, not as trivia.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

config.MEMORY_DB = Path(tempfile.mkdtemp(prefix="friday-prefs-")) / "memory.db"

import memory                                           # noqa: E402
import uncertainty                                      # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def main() -> int:
    print("STEP 2 — the correction reaches the prompt")
    memory.remember("Harsh does not want to be called Sujoo; his name is Harsh",
                    kind=config.LEARNED_PREFERENCE_KIND, source="correction")
    memory.remember("Harsh's sister is called Riya", source="brain-note")

    block = memory.context_block("what is my name")
    check("a preference is pulled into the context", "Sujoo" in block, block)
    check("...in its own section, worded as an instruction",
          "told you to behave" in block, block)
    check("...and not as something she merely remembers",
          block.index("told you to behave") < block.index("What you remember")
          if "What you remember" in block else True, block)
    check("an ordinary fact still comes through",
          "Riya" in memory.context_block("tell me about my sister"))

    # THE CASE THAT MATTERS: a standing instruction has to arrive on a turn
    # that has nothing to do with it. Relevance scoring alone does not do
    # that -- "what is my name" scores 0.00 against this correction.
    check("a correction reaches an UNRELATED turn too",
          "Sujoo" in memory.context_block("is it going to rain tomorrow"),
          "a preference that waits to be mentioned never changes behaviour")

    prefs = memory.recall("Sujoo", kinds=(config.LEARNED_PREFERENCE_KIND,))
    check("kind-filtered recall returns only preferences",
          prefs and all(f.kind == config.LEARNED_PREFERENCE_KIND
                        for f in prefs), str([f.kind for f in prefs]))
    plain = memory.recall("name", exclude_kinds=(config.LEARNED_PREFERENCE_KIND,))
    check("the ordinary list excludes them, so nothing is said twice",
          all(f.kind != config.LEARNED_PREFERENCE_KIND for f in plain),
          str([f.kind for f in plain]))

    print("\nSTEP 2 — bounded on purpose")
    for i in range(10):
        memory.remember(f"Harsh prefers option number {i} when naming things",
                        kind=config.LEARNED_PREFERENCE_KIND,
                        source="correction")
    pulled = memory.recall("naming things",
                           limit=config.PREFERENCE_CONTEXT_MAX,
                           kinds=(config.LEARNED_PREFERENCE_KIND,))
    check(f"at most {config.PREFERENCE_CONTEXT_MAX} reach the prompt",
          len(pulled) <= config.PREFERENCE_CONTEXT_MAX, str(len(pulled)))

    config.PREFERENCE_IN_CONTEXT = False
    check("and the whole injection has an off switch",
          "told you to behave" not in memory.context_block("what is my name"))
    config.PREFERENCE_IN_CONTEXT = True

    print("\nSTEP 3 — a gap is logged, and stays out of what she 'knows'")
    uncertainty.log_gap("how loop closure works in her map",
                        asked="how does your map work", said="Roughly, it…")
    check("the gap is logged", len(uncertainty.gaps()) == 1)
    check("it is NOT stored as a fact she knows",
          not any("loop closure" in f.text for f in memory.all_facts()),
          "a guess filed as a fact becomes something she is confident about")
    check("it never surfaces as 'how similar things went before'",
          not any("loop closure" in e["text"]
                  for e in memory.recall_experience("loop closure")),
          "recall_experience must stay outcome-only")
    check("it does not reach the turn context either",
          "loop closure" not in memory.context_block("how does your map work"))
    check("the idle look-up can find it",
          uncertainty.next_topic() == "how loop closure works in her map")

    check("an empty topic logs nothing", uncertainty.log_gap("") is None)
    config.UNCERTAINTY_LOGGING = False
    check("and it has an off switch", uncertainty.log_gap("something") is None)
    config.UNCERTAINTY_LOGGING = True

    old = uncertainty.next_topic(within_s=0.0)
    check("a stale gap is not chased by the idle look-up", old is None)

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
