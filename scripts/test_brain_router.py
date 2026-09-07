#!/usr/bin/env python3
"""Test the Haiku/Sonnet split. No API key, no network, no cost.

    scripts/test_brain_router.py

Everything in brain_router except `_call` is pure, so all of the POLICY is
testable offline: which brain gets which decision, when Haiku escalates, what
the spend cap does, and whether the log actually shows the split. `_call` is
stubbed here -- it is the only part that needs a key, and it is deliberately
the smallest part.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
import brain_router as br                               # noqa: E402
from brain_router import BrainRouter, DRIVE, PLAN, TAG, STUCK   # noqa: E402


class Usage:
    def __init__(self, i=100, o=20):
        self.input_tokens, self.output_tokens = i, o
        self.cache_read_input_tokens = self.cache_creation_input_tokens = 0


def router(tmp: Path, replies: dict[str, str], key="test-key", cap=2.0):
    """A router whose only fake is the API call itself."""
    r = BrainRouter(cap_usd=cap, cost_log=tmp / "spend.jsonl")
    r.available = lambda: bool(key)

    async def fake_call(model, prompt, system, max_tokens):
        return replies.get(model, "ok"), Usage()
    r._call = fake_call
    return r


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


async def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp())

    print("\n1. the routing table is the policy")
    ok &= check("driving goes to Haiku",
                BrainRouter.route(DRIVE) == config.MODEL_FAST,
                BrainRouter.route(DRIVE))
    for kind in (PLAN, TAG, STUCK):
        ok &= check(f"{kind} goes to Sonnet",
                    BrainRouter.route(kind) == config.MODEL_MAIN)
    ok &= check("an unknown kind goes to the CAPABLE model, not the cheap one",
                BrainRouter.route("nonsense") == config.MODEL_MAIN)

    print("\n2. Haiku answering is NOT escalated")
    r = router(tmp, {config.MODEL_FAST: "turn left"})
    d = await r.decide(DRIVE, "which way?")
    ok &= check("answered by Haiku", d.model == config.MODEL_FAST, d.model)
    ok &= check("not escalated", not d.escalated)
    ok &= check("cost recorded", d.usd > 0, f"${d.usd:.6f}")

    print("\n3. Haiku saying UNSURE wakes Sonnet, exactly once")
    r = router(tmp, {config.MODEL_FAST: "UNSURE",
                     config.MODEL_MAIN: "go through the doorway"})
    d = await r.decide(DRIVE, "which way?")
    ok &= check("escalated", d.escalated)
    ok &= check("final brain is Sonnet", d.model == config.MODEL_MAIN, d.model)
    ok &= check("remembers who gave up",
                d.escalated_from == config.MODEL_FAST)
    ok &= check("answer is Sonnet's", d.text == "go through the doorway")
    ok &= check("billed for BOTH calls", d.input_tokens == 200,
                f"{d.input_tokens} input tokens")

    print("\n4. 'unsure' mentioned inside a real answer is NOT an escalation")
    ok &= check("bare UNSURE escalates", BrainRouter.is_unsure("UNSURE"))
    ok &= check("with punctuation/space", BrainRouter.is_unsure("  unsure. "))
    ok &= check("a real answer that says the word does not",
                not BrainRouter.is_unsure("I am unsure but turn left"))
    ok &= check("empty reply counts as unsure", BrainRouter.is_unsure(None))

    print("\n5. the spend cap stops the spending")
    r = router(tmp, {config.MODEL_FAST: "left"}, cap=0.0001)
    (tmp / "spend.jsonl").write_text(json.dumps(
        {"date": __import__("time").strftime("%Y-%m-%d"), "usd": 5.0}) + "\n")
    d = await r.decide(DRIVE, "which way?")
    ok &= check("refused", d.skipped and "cap" in d.skipped, d.skipped)
    ok &= check("nothing was spent", d.usd == 0.0)

    print("\n6. no API key: routed and logged, never pretended")
    r = router(tmp, {}, key="")
    d = await r.decide(PLAN, "plan a run")
    ok &= check("still routed", d.model == config.MODEL_MAIN)
    ok &= check("says why", d.skipped and "ANTHROPIC_API_KEY" in d.skipped)
    ok &= check("not counted as answered", not d.answered)

    print("\n7. the cost log shows the split")
    tmp2 = Path(tempfile.mkdtemp())
    r = router(tmp2, {config.MODEL_FAST: "left", config.MODEL_MAIN: "ok"})
    await r.decide(DRIVE, "a")
    await r.decide(DRIVE, "b")
    await r.decide(PLAN, "c")
    r2 = router(tmp2, {config.MODEL_FAST: "UNSURE", config.MODEL_MAIN: "ok"})
    r2.decisions = r.decisions
    await r2.decide(DRIVE, "d")
    split = r2.split()
    print(f"     {split}")
    ok &= check("four decisions", split["decisions"] == 4)
    ok &= check("Haiku did the driving",
                split["by_brain"].get(config.MODEL_FAST) == 2)
    ok &= check("Sonnet did planning + the escalation",
                split["by_brain"].get(config.MODEL_MAIN) == 2)
    ok &= check("one escalation", split["escalations"] == 1)

    rows = [json.loads(l) for l in (tmp2 / "spend.jsonl").read_text().splitlines()]
    brains = [r_["brain"] for r_ in rows]
    ok &= check("every decision is on disk with its brain", len(rows) == 4,
                str(brains))
    ok &= check("the log distinguishes the two brains",
                len(set(brains)) == 2, str(set(brains)))

    print(f"\n{'ALL PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
