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
import time
from typing import Any

import config

log = logging.getLogger("friday.uncertainty")

TASK = "uncertain_response"


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
