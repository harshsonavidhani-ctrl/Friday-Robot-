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


@tool
def what_have_you_learned() -> str:
    """What she has learned lately — her own reflection, read back.

    File 09 step 4. Answers "what have you learned?", "what did you learn
    today?", "have you noticed anything about yourself?". It READS the last
    reflection; it does not think a new one. Nothing here is a fresh thought,
    so say it as what it is: something she noticed earlier.

    Empty when she has not reflected yet — say that plainly rather than
    inventing something she noticed.
    """
    import reflection                                     # noqa: PLC0415
    note = reflection.last()
    if not note:
        return ("nothing yet — she has not reflected since the last restart, "
                "or there was nothing to reflect on")
    report = {"when": note.get("date"), "summary": note.get("summary"),
              "themes": note.get("themes") or [],
              "waiting_for_harsh": [
                  {"instruction": f["text"], "why": f["why"], "kind": f["kind"]}
                  for f in reflection.flags()]}
    return json.dumps(report, indent=2)


@tool
def reflect_now() -> str:
    """Look back over what she has learned, now, and save it.

    Use ONLY when he asks her to think about herself -- "reflect", "think
    about what you've learned", "review your instructions". It runs on its
    own three times a day; this is the on-demand path, it costs a call, and
    it is rate limited.

    Returns the summary. It is written down either way: the dashboard shows
    it, and what_have_you_learned reads it back later.
    """
    import rate_limit                                     # noqa: PLC0415
    import reflection                                     # noqa: PLC0415
    allowed, why = rate_limit.allow("reflection")
    if not allowed:
        return f"not now: {why}"
    note = reflection.run_self_reflection("he asked", force=True)
    if note is None:
        return ("nothing to reflect on yet — no corrections and no gaps have "
                "been logged")
    rate_limit.used("reflection")
    flagged = reflection.flags()
    tail = (f" {len(flagged)} instruction(s) need a decision from him."
            if flagged else "")
    return f"{note['summary']}{tail}"


@tool
def how_much_have_you_grown() -> str:
    """Her own growth numbers — "how much have you learned?", "what level are you?"

    File 09 step 6. Every number is a count of rows that already existed: the
    corrections she kept, what she remembers, jobs that worked or failed,
    reflections done, gaps admitted and gaps closed. The level is a rough
    milestone over those counts, NOT a score she earns for talking.

    Say the summary line; the rest is there if he asks for detail. Never
    inflate it — the numbers are the point.
    """
    import growth                                        # noqa: PLC0415
    data = growth.report()
    return json.dumps({"summary": data["summary"], "level": data["level"],
                       "counts": data["counts"], "next": data["next"]},
                      indent=2)


@tool
def search_notes(question: str) -> str:
    """Look in HIS notes — the PDFs and pages he has given her — for an answer.

    File 12a. Use this whenever he asks something that sounds like it comes
    from his own material: a subject he is studying, a definition, "what did
    my notes say about X". It searches the study corpus, not the web and not
    her own memory, and returns the passages themselves with where they came
    from.

    ⚠ ANSWER FROM WHAT IT RETURNS, and say the source. If it returns nothing,
    say the notes do not cover it rather than answering from general
    knowledge — that is the whole point of asking his notes.

    Args:
        question: what he asked, in his own words.
    """
    import study                                          # noqa: PLC0415
    found = study.search(question)
    if not found:
        return ("nothing in his notes covers that — say so, and offer to "
                "look it up instead")
    return json.dumps({"passages": [
        {"from": hit["title"], "page": hit["page"], "text": hit["text"]}
        for hit in found]}, indent=2)


@tool
def study_status() -> str:
    """What is in the study corpus, and what is due for review right now.

    File 12a. Answers "how many cards do I have", "anything to review?",
    "what have I loaded". The due count is the one he acts on.
    """
    import study                                          # noqa: PLC0415
    stats = study.stats()
    due = [{"id": c["id"], "question": c["question"], "topic": c["topic"]}
           for c in study.due_cards(limit=5)]
    return json.dumps({"stats": stats, "due_now": due}, indent=2)
