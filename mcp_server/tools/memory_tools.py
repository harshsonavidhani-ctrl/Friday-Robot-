"""Memory tools: what Friday keeps between conversations and between boots.

These are the only way the brain writes to memory, and that is deliberate.
Nothing is remembered automatically — Friday decides what is worth keeping,
the same way a person does. An automatic transcript dump would fill recall
with noise and crowd out the handful of things that actually matter.

See memory.py for the storage and the retrieval.
"""

from __future__ import annotations

import json

import memory
from mcp_server.registry import tool


@tool
def remember(fact: str) -> str:
    """Keep something worth remembering after this conversation ends.

    Use this for things that will still be true later and that you would be
    worse off not knowing: where rooms and objects are, how Harsh likes you to
    behave, names, preferences, standing instructions.

    Do NOT use it for the passing content of a conversation, for anything you
    can look up or sense again in a moment, or for what you just did — that is
    log_event's job.

    Saying the same thing twice updates the existing memory rather than adding
    a second copy, so re-stating a fact to correct it works as expected.

    Args:
        fact: one clear sentence, in the third person. "Harsh's room is
            upstairs on the left" — not "he told me his room is upstairs".
    """
    stored = memory.remember(fact, source="brain")
    verb = "updated" if stored["updated"] else "remembered"
    return f"{verb}: {stored['text']}"


@tool
def recall(query: str) -> str:
    """Look up what you already know about something.

    Relevant memories are added to your context automatically before each
    reply, so reach for this only when you need to check something specific —
    when Harsh asks what you remember, or when you are about to act and want
    to be sure.

    Returns nothing at all when nothing is relevant. That is a real answer:
    say you do not know rather than guessing.

    Args:
        query: what you are trying to remember about.
    """
    found = memory.recall(query)
    if not found:
        return (f"nothing remembered about {query!r}. Say you do not know "
                f"rather than inventing something.")
    lines = [f"- {fact.text}" for fact in found]
    return "\n".join(lines)


@tool
def log_event(event: str, task: str = "", outcome: str = "") -> str:
    """Record something you did and how it turned out, so you can learn.

    Worth logging after anything that could have gone otherwise: a move, a
    turn, a look, a task Harsh gave you. Past outcomes for a similar task are
    put in front of you automatically next time, so an honest "failed" here is
    what stops the same mistake being repeated.

    Record failures as carefully as successes. A log of only successes teaches
    nothing.

    Args:
        event: what happened, in one sentence, including the circumstances
            that mattered — "turning left near the sofa stalled the wheels"
            is useful, "turn failed" is not.
        task: a short label to group similar attempts: move, turn, look,
            speak, find. Leave empty if none fits.
        outcome: "worked", "failed" or "partial". Leave empty only if the
            result genuinely is not known yet.
    """
    stored = memory.log_event(event, task=task or None,
                              outcome=outcome or None)
    tail = f" ({stored['outcome']})" if stored["outcome"] else ""
    return f"logged{tail}: {stored['text']}"


@tool
def memory_status() -> str:
    """How much Friday is currently remembering, and the most recent facts.

    Useful when Harsh asks what you remember in general, rather than about
    one particular thing.
    """
    stats = memory.stats()
    facts = memory.all_facts(limit=12)
    report = {
        "counts": {k: stats[k] for k in ("facts", "events", "summaries")},
        "recent_facts": [f.text for f in facts],
        "recent_activity": memory.recent_events(limit=5),
    }
    return json.dumps(report, indent=2)
