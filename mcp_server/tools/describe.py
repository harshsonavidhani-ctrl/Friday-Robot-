"""describe_scene — File 15 step 1: what is in view, or the answer to one question about it.

    describe_scene()                          a plain description
    describe_scene(focus="how many people")   the answer to that, not a tour

NOT SPOKEN HERE (step 2). say_what_you_see speaks for itself; this returns
words to the brain, which says them in Friday's own voice, picks the face from
the emotion table and keeps a memory_note when something is worth keeping.
A tool that talked would bypass reply_contract and the face arbiter.

THE FRAME, cheapest first:
  1. the scene watcher's frame (state/scene_latest.jpg), if it is under
     DESCRIBE_FRESH_S old -- no camera opened at all;
  2. otherwise vision.look("up"), which captures under the same lock as
     look() and observe(). Never a second handle on the camera.

COOLDOWN (step 7). RATE_LIMITS["describe_scene"] keeps calls apart; inside
the gap the previous answer is handed back, marked with its age, rather than
refused -- "what do you see" twice in a breath deserves the same answer, not
an error.
"""

from __future__ import annotations

import io
import logging
import time

import config
import scene_describe
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.describe")

# (time.time(), focus, text) of the last answer, for the cooldown.
_last: tuple[float, str, str] | None = None


def _frame() -> tuple[bytes | None, str, list]:
    """(jpeg, where it came from, detector objects). jpeg None = no picture."""
    cached = scene_describe.fresh_frame()
    if cached is not None:
        jpeg, age, objects = cached
        return jpeg, f"scene watcher's frame, {age:.0f}s old", objects
    try:
        import vision                                   # noqa: PLC0415
        image, note = vision.look("up")
    except Exception as exc:                            # noqa: BLE001
        return None, f"no camera ({type(exc).__name__}: {exc})", []
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=config.CAPTURE_QUALITY)
    return buffer.getvalue(), f"live capture ({note})", []


def _sensors() -> str:
    """The range-sensor line say_what_you_see already builds. "" if none."""
    try:
        from mcp_server.tools.narrate import _sensor_summary  # noqa: PLC0415
        summary, _ = _sensor_summary()
        return summary
    except Exception as exc:                            # noqa: BLE001
        log.debug("no range sensors for the description: %s", exc)
        return ""


@tool
def describe_scene(focus: str = "") -> str:
    """Describe what Friday's camera sees right now, or answer one question about it.

    Use this when Harsh asks about what is in front of you and you have NOT
    already been given a fresh picture this turn: "what do you see", and
    especially targeted questions -- "what's written on that", "how many
    people are here", "what colour is the cup". Pass his question as `focus`
    so the answer targets it instead of describing the whole room.

    The result is for YOU, not for reading out: say it in your own words, pick
    your emotion from what was seen, and put anything genuinely notable (a new
    thing in the room, someone arriving) in memory_note.

    Continuous narration ("keep describing what you see") is switched on only
    by Harsh's own words or the dashboard button -- you cannot turn it on.

    Args:
        focus: his question about the view, in his words. Empty for a
            general description.
    """
    global _last
    focus = (focus or "").strip()[:200]
    now = time.time()

    jpeg, where, objects = _frame()
    if jpeg is None:
        return (f"I cannot see anything right now — {where}. Say so; do not "
                f"describe the room from memory.")

    text, why = scene_describe.describe(jpeg, focus=focus or None,
                                        sensors=_sensors(), objects=objects)
    if text is None:
        # Inside the cooldown, the last answer is still the truth about the
        # room a moment ago -- hand it back with its age rather than nothing.
        if (_last is not None and why.startswith("rate limited")
                and now - _last[0] <= float(config.DESCRIBE_REUSE_S)
                and _last[1] == focus):
            return (f"{_last[2]}\n\n[the same look as {now - _last[0]:.0f}s "
                    f"ago — {why}]")
        return (f"I could not describe the view ({why}). Say that plainly; "
                f"do not guess what is there.")

    _last = (now, focus, text)
    cost = scene_describe.last_cost_usd
    log.info("describe_scene(%r): %d chars from the %s, $%.4f",
             focus, len(text), where, cost)
    return f"{text}\n\n[{where}; ${cost:.4f}]"
