"""uncertainty.py — File 09 step 3: what Friday did not really know.

    uncertainty.log(topic, said=...)     keep a gap, silently
    uncertainty.gaps()                   read them back
    uncertainty.next_topic()             the one worth looking up

SILENT ON PURPOSE. The spec: "it doesn't need to say 'I'm not sure' out loud,
just log it." A companion that hedges every uncertain answer is exhausting, and
the hedging is not what makes her better -- going and finding out is. So the
reply stays normal and the gap goes here.

WHERE IT GOES, and why not into facts. These are rows in the EVENTS table with
task="uncertain_response" and NO outcome, which matters twice over:
  * the facts table is read back to her as things she KNOWS. A guess filed
    there becomes a fact she is confident about next week -- the exact failure
    that had her describing a cat she had never seen (see situation.block).
  * recall_experience() only returns events WITH an outcome, so a gap can
    never surface as "how similar things went before". A gap is not a lesson.

WHAT READS IT BACK:
  * idle_study (file 08 step 8) prefers a logged gap over guessing a topic
    from the conversation -- it now has a list of things she actually did not
    know, which is better than anything it could infer.
  * the reflection step of this file, to spot the pattern in them.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import config

log = logging.getLogger("friday.uncertainty")

TASK = "uncertain_response"

# ⚠ THE MODEL FORGETS TO FILL THE FIELD, and the whole step is that field.
# Measured 2026-09-20: "roughly how much does your chassis weigh" came back
# "I don't have a spec sheet in my head for my own weight ... I'd have to ask
# him." -- a flat admission, exactly what this step exists to catch -- with
# `uncertain` null. The contract now spells that case out, and this is the
# backstop for when it is forgotten anyway: her OWN words decide, and the
# topic is the question she was asked.
_ADMISSION_RE = re.compile(
    r"\b(?:i (?:don'?t|do not) (?:know|have|remember)"
    r"|i'?m not (?:sure|certain)|i am not (?:sure|certain)"
    r"|no idea|not sure (?:what|how|why|where|when|which)"
    r"|can'?t (?:remember|recall|tell you)"
    r"|(?:i'?d|i would) have to (?:ask|look|check|find out)"
    r"|i (?:have|'ve) never (?:measured|weighed|counted|checked)"
    r"|nobody (?:has )?told me)\b", re.IGNORECASE)

# What is left of a question once the asking is stripped off the front, so the
# stored topic reads like a subject ("how much does your chassis weigh") and
# not like a transcript ("hey friday, roughly how much does your chassis
# weigh?"). idle_study searches on it later, so it has to stand alone.
_ASKING_RE = re.compile(
    r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?friday[\s,]*)?"
    r"(?:so|and|well|um|uh|just|please|roughly|approximately|about)?[\s,]*"
    r"(?:can you|could you|do you|did you|would you|tell me|"
    r"i wanted to know|i want to know)?[\s,]*", re.IGNORECASE)


def admitted_not_knowing(said: str | None, asked: str | None) -> str | None:
    """The topic of an answer that says, in her own words, that she does not
    know it -- or None. Never raises.

    Only an ADMISSION counts. A confident wrong answer is not visible from
    here and is not this function's job; the model's own `uncertain` field is
    the only thing that can catch those.
    """
    if not said or not asked:
        return None
    try:
        if not _ADMISSION_RE.search(said):
            return None
        topic = _ASKING_RE.sub("", asked.strip()).strip(" ?.!,")
        return topic or None
    except Exception:                           # noqa: BLE001
        return None


def log_gap(topic: str, said: str | None = None,
            asked: str | None = None) -> dict | None:
    """Record that she answered without really knowing. Never raises."""
    if not getattr(config, "UNCERTAINTY_LOGGING", True):
        return None
    topic = (topic or "").strip()
    if not topic:
        return None
    limit = int(getattr(config, "UNCERTAINTY_MAX_CHARS", 160))
    if len(topic) > limit:
        topic = topic[:limit].rsplit(" ", 1)[0] + "…"
    try:
        import memory
        meta: dict[str, Any] = {}
        if asked:
            meta["asked"] = asked[:200]
        if said:
            meta["said"] = said[:200]
        row = memory.log_event(topic, task=TASK, meta=meta or None)
        log.info("uncertain about %r", topic)
        return row
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not log the uncertainty (%s: %s)",
                    type(exc).__name__, exc)
        return None


def gaps(limit: int = 50) -> list[dict]:
    """What she has been unsure about, newest first. Never raises."""
    try:
        import memory
        return memory.events_of_task(TASK, limit)
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not read the uncertainties: %s", exc)
        return []


def next_topic(within_s: float | None = None) -> str | None:
    """The most recent gap worth looking up, or None.

    Recent ones only: a question from three weeks ago is not what he is about
    to ask about again, and the idle look-up has a budget (8 a day) that is
    better spent on what came up today.
    """
    window = float(within_s if within_s is not None
                   else getattr(config, "UNCERTAINTY_FRESH_S", 86400.0))
    now = time.time()
    for row in gaps(limit=20):
        if now - float(row.get("created_at") or 0.0) <= window:
            return row["text"]
    return None


def main() -> int:
    """What Friday does not know yet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    rows = gaps()
    if not rows:
        print("no gaps logged")
        return 0
    for row in rows:
        print(f"{row['age']:>12}  {row['text']}")
    print(f"\nnext to look up: {next_topic() or '(nothing recent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
