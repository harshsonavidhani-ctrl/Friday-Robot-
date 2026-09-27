"""File 20's tools: research done properly on Harsh's laptop, or notes saved there.

delegate_to_laptop returns AT ONCE (a research run outlasts the 25 s tool
limit); the result is said between turns when it arrives. See
laptop_research.py for the flow and the fallback.
"""

from __future__ import annotations

import laptop_research as lr
from mcp_server.registry import tool


@tool
def delegate_to_laptop(query: str, mode: str = "answer") -> str:
    """Research something PROPERLY on Harsh's laptop (deep, multi-source web
    research), or write notes about it into a file there.

    Use this for multi-part questions, "explain thoroughly", broad or
    open-ended topics, comparisons, anything where a quick search would be
    shallow -- and ALWAYS when he says "properly research", "research this in
    depth" or "notes on ...". For a quick fact, use your own web search.

    Answer mode returns at once; the laptop works for a minute or two and you
    will be told the result later -- just tell him you're on it. If the laptop
    is not available, this answers directly from a quick search instead: say
    that the laptop wasn't available, then give the answer.

    Args:
        query: the question or topic, fully stated (he will not be asked again).
        mode: "answer" (say what was found) or "notes" (save a notes file on
            the laptop and say where).
    """
    query = (query or "").strip()
    mode = (mode or "answer").strip().lower()
    if not query:
        return "What should I research?"
    if mode not in ("answer", "notes"):
        mode = "answer"
    import rate_limit
    ok, why = rate_limit.allow("laptop_research")
    if not ok:
        return f"I've sent a lot of research to the laptop already — {why}."
    up, why = lr.available() if lr.configured() else (False, "not set up")
    if not up:
        if mode == "notes":
            return ("My laptop isn't available right now, so I can't save "
                    "notes there.")
        answer, fail = lr.fallback_answer(query)
        if not answer:
            return (f"My laptop isn't available, and I couldn't look it up "
                    f"myself either ({fail}).")
        return f"My laptop isn't available, so here's the direct answer. {answer}"
    rate_limit.used("laptop_research")
    lr.start(query, mode)
    if mode == "notes":
        return (f"I'm researching {query} on your laptop and I'll save notes "
                f"there — I'll tell you where when they're ready.")
    return (f"I'm researching {query} properly on your laptop — I'll tell you "
            f"what I find in a minute or two.")


@tool
def research_detail(query: str = "") -> str:
    """The fuller detail of research the laptop did, when Harsh asks for more
    ("tell me more about that", "what else did you find out about X").

    Args:
        query: words from the topic; empty for the most recent research.
    """
    job = lr.latest(query)
    if job is None:
        return "I haven't researched anything like that yet."
    detail = (job.get("detail") or job.get("summary") or "").strip()
    where = f" The notes are on your laptop at {job['path']}." \
        if job.get("path") else ""
    return (f"On {job['query']}: {detail[:1500]}{where}")
