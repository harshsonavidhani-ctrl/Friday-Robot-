"""File 20's tools: Friday uses Harsh's laptop -- a search on his screen, or notes.

The laptop ONLY: when it cannot be reached she says so and does nothing else.
laptop_notes returns AT ONCE (research outlasts the 25 s tool limit); the
result is said between turns when it is saved. See laptop_research.py.
"""

from __future__ import annotations

import laptop_research as lr
from mcp_server.registry import tool

NOT_AVAILABLE = "My laptop isn't available right now."


def _up() -> bool:
    return lr.configured() and lr.available()[0]


@tool
def laptop_search(query: str, site: str = "web", play: bool = False) -> str:
    """Search on Harsh's LAPTOP screen: a Google search, a YouTube search, or
    play the top YouTube video. Use whenever he asks to search, google or look
    something up on his laptop, or to search or play something on YouTube.
    It only opens the search in his browser -- you do not see the results, so
    do not describe them.

    Args:
        query: the words to search for (not a URL).
        site: "web" (Google) or "youtube".
        play: YouTube only -- true to play the top video instead of listing.
    """
    query = (query or "").strip()
    site = (site or "web").strip().lower()
    if not query:
        return "What should I search for?"
    if site not in ("web", "youtube"):
        site = "web"
    import rate_limit
    ok, why = rate_limit.allow("laptop_open")
    if not ok:
        return f"I've opened a lot on your laptop already — {why}."
    if not _up():
        return NOT_AVAILABLE
    rate_limit.used("laptop_open")
    opened, why = lr.open_search(query, site, bool(play) and site == "youtube")
    if not opened:
        return f"I couldn't open that on your laptop ({why})."
    if site == "youtube":
        return (f"Playing {query} on YouTube on your laptop." if play
                else f"Searching YouTube for {query} on your laptop.")
    return f"Searching for {query} on your laptop."


@tool
def laptop_notes(topic: str) -> str:
    """Research a topic on the web and put notes about it on Harsh's LAPTOP,
    opened on his screen. Use for "make notes on X", "notes about X",
    "properly research X", "research X in depth". Returns at once; the notes
    take a minute and you will be told when they are saved -- just tell him
    you're on it.

    Args:
        topic: the topic, fully stated (he will not be asked again).
    """
    topic = (topic or "").strip()
    if not topic:
        return "What should I make notes on?"
    import rate_limit
    ok, why = rate_limit.allow("laptop_notes")
    if not ok:
        return f"I've made a lot of notes already — {why}."
    if not _up():
        return f"{NOT_AVAILABLE} I can't put notes there."
    rate_limit.used("laptop_notes")
    lr.start_notes(topic)
    return (f"I'm researching {topic} and I'll put the notes on your laptop — "
            f"I'll tell you when they're open.")


@tool
def research_detail(query: str = "") -> str:
    """What the notes Friday made say, when Harsh asks for more ("tell me more
    about that", "what did you find out about X").

    Args:
        query: words from the topic; empty for the most recent notes.
    """
    job = lr.latest(query)
    if job is None:
        return "I haven't made notes on anything like that yet."
    detail = (job.get("detail") or job.get("summary") or "").strip()
    return (f"On {job['query']}: {detail[:1500]} The notes are on your laptop "
            f"at {job['path']}.")
