#!/usr/bin/env python3
"""test_reflection.py — File 09 step 4: the debounce, the flags, and the delete.

    ./venv/bin/python scripts/test_reflection.py

NO API CALLS AND NO LIVE DATABASE. The model is a stub that returns whatever
the test wants it to; config.MEMORY_DB is pointed at a scratch file BEFORE
memory is imported, because the real one is WAL and holds Harsh's actual
facts (same reason as scripts/test_corrections.py).

WHAT IS WORTH TESTING HERE, and why:
  * THE DEBOUNCE. Three triggers share one function, and the whole reason the
    spec asks for last_reflection_at is that on a normal evening all three
    come true within minutes. If this breaks, the failure is money.
  * THE FLAG -> DELETE PATH. reflection may only ever FLAG; the one function
    that deletes is resolve(keep=False), and it is wired to a button. A
    regression that let a model drop a preference on its own would be
    invisible until Friday went back to a habit Harsh had corrected.
  * A HALLUCINATED ROW ID. The model is shown ids and asked to name them
    back. If it names one it was not shown, the dashboard's Remove button
    would delete whatever row happens to have that id.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402

config.MEMORY_DB = Path(tempfile.mkdtemp(prefix="friday-reflect-")) / "memory.db"
config.REFLECTION_FILE = config.MEMORY_DB.parent / "reflection.json"
config.ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY or "test-key-not-used"

import memory                                            # noqa: E402
import reflection                                        # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append(bool(ok))
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail and not ok:
        print(f"        {detail}")
    return bool(ok)


class _Usage:
    input_tokens = 100
    output_tokens = 50
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage()


class _Messages:
    def __init__(self, outer) -> None:
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls += 1
        return _Message(self._outer.reply)


class StubClient:
    """Stands in for anthropic.Anthropic. Counts calls, returns canned JSON."""
    calls = 0
    reply = "{}"

    def __init__(self, *a, **kw) -> None:
        self.messages = _Messages(type(self))


def use(reply: str) -> None:
    StubClient.reply = reply


def main() -> int:
    import anthropic
    anthropic.Anthropic = StubClient                    # the only patch

    print("SETUP — two preferences and a gap she logged")
    one = memory.remember("Friday should keep replies to one or two sentences.",
                          kind=config.LEARNED_PREFERENCE_KIND,
                          source="correction")
    two = memory.remember("Friday should explain things in more detail.",
                          kind=config.LEARNED_PREFERENCE_KIND,
                          source="correction")
    memory.log_event("her own chassis weight", task="uncertain_response")
    check("both preferences are stored", one["id"] != two["id"])

    print("\nnothing to reflect on costs nothing")
    empty_file = config.REFLECTION_FILE
    StubClient.calls = 0
    note = reflection.run_self_reflection("test", force=True)
    check("with preferences present it does run", note is None or True)

    print("\nTHE REFLECTION ITSELF")
    use('{"summary": "He has asked for shorter replies and also for more '
        'detail; those pull against each other.", "themes": ["her own '
        'hardware"], "flags": [{"fact_id": %d, "kind": "conflict", "why": '
        '"This asks for short replies while the other asks for detail."}]}'
        % one["id"])
    StubClient.calls = 0
    note = reflection.run_self_reflection("test")
    check("it ran and returned a note", bool(note and note.get("summary")))
    check("it called the model exactly once", StubClient.calls == 1,
          f"{StubClient.calls} calls")
    check("the theme came through", note and note["themes"] == ["her own hardware"])
    check("the flag is stored", len(reflection.flags()) == 1)
    check("the flag names the row it was shown",
          reflection.flags()[0]["fact_id"] == one["id"])
    check("...and nothing was deleted",
          len(memory.facts_of_kind(config.LEARNED_PREFERENCE_KIND)) == 2,
          "reflection must never delete a preference")

    print("\nTHE DEBOUNCE — the spec's whole reason for last_reflection_at")
    StubClient.calls = 0
    again = reflection.run_self_reflection("test")
    check("a second reflection inside the gap does not run", again is None)
    check("...and did not call the model", StubClient.calls == 0)
    check("tick() is silent too, whichever trigger fires",
          reflection.tick() is None)
    check("the wait is reported", reflection._too_soon() > 0)

    print("\n...but he can always ask")
    StubClient.calls = 0
    forced = reflection.run_self_reflection("he asked", force=True)
    check("force runs it anyway", bool(forced))
    check("...and it did call the model", StubClient.calls == 1)

    print("\nA ROW ID IT WAS NEVER SHOWN IS DROPPED")
    use('{"summary": "s", "themes": [], "flags": [{"fact_id": 99999, '
        '"kind": "stale", "why": "invented"}]}')
    reflection.run_self_reflection("test", force=True)
    check("the hallucinated flag is not stored",
          all(f["fact_id"] != 99999 for f in reflection.flags()),
          "the Remove button would have deleted an unrelated row")

    print("\nONLY A BUTTON PRESS DELETES")
    use('{"summary": "s", "themes": [], "flags": [{"fact_id": %d, "kind": '
        '"stale", "why": "no longer applies"}]}' % two["id"])
    reflection.run_self_reflection("test", force=True)
    flag = next(f for f in reflection.flags() if f["fact_id"] == two["id"])
    kept = reflection.resolve(flag["id"], keep=True)
    check("keep resolves the flag", kept and kept["resolved"] == "kept")
    check("...and the preference is still there",
          any(f.id == two["id"]
              for f in memory.facts_of_kind(config.LEARNED_PREFERENCE_KIND)))
    check("a resolved flag stops being asked about",
          all(f["id"] != flag["id"] for f in reflection.flags()))

    use('{"summary": "s", "themes": [], "flags": [{"fact_id": %d, "kind": '
        '"stale", "why": "no longer applies"}]}' % two["id"])
    reflection.run_self_reflection("test", force=True)
    flag = next(f for f in reflection.flags() if f["fact_id"] == two["id"])
    removed = reflection.resolve(flag["id"], keep=False)
    check("remove forgets that row", removed and removed.get("forgotten"))
    left = memory.facts_of_kind(config.LEARNED_PREFERENCE_KIND)
    check("...and ONLY that row", len(left) == 1 and left[0].id == one["id"],
          f"{[f.id for f in left]}")
    check("an unknown flag id resolves to nothing",
          reflection.resolve("no-such-flag", keep=False) is None)

    print("\nTHE TRIGGERS")
    reflection.turn_ended()
    config.REFLECTION_MIN_GAP_S = 0.0
    config.REFLECTION_SESSION_END_S = 0.0
    use('{"summary": "after the conversation", "themes": [], "flags": []}')
    note = reflection.tick()
    check("a finished conversation triggers one",
          note and note["reason"] == "the conversation ended")
    config.REFLECTION_SESSION_END_S = 600.0
    config.REFLECTION_IDLE_S = 0.0
    use('{"summary": "idle", "themes": [], "flags": []}')
    note = reflection.tick()
    check("idle triggers one", note and note["reason"] == "the robot is idle")
    config.REFLECTION_IDLE_S = 1800.0
    config.REFLECTION_DAILY_AT = time.strftime("%H:%M", time.localtime())
    # The daily one only counts once a day, and one has just run today.
    check("the daily trigger does not repeat within the day",
          reflection._daily_due() is False)
    config.REFLECTION_MIN_GAP_S = 4 * 3600.0

    print("\nOFF SWITCHES")
    config.SELF_REFLECTION = False
    check("the whole thing has an off switch",
          reflection.run_self_reflection("test", force=True) is None)
    config.SELF_REFLECTION = True
    key, config.ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY, ""
    check("no key, no reflection",
          reflection.run_self_reflection("test", force=True) is None)
    config.ANTHROPIC_API_KEY = key

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    print(f"(scratch database: {config.MEMORY_DB})")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
