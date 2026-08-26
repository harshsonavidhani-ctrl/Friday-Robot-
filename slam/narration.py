"""narration.py — deciding what is worth saying out loud while exploring.

    narrator = Narrator()
    line = narrator.consider(decision=d, closure=c, gained_m2=g, now=t)
    if line:
        tts.speak(line)

WHY THIS IS NOT JUST `tts.speak(decision.say)`. The explorer produces a line
every cycle, and a cycle is a couple of seconds. Speaking all of them means
"unexplored space straight ahead, about two metres away" nine times in a row
while she drives towards it, which is not narration, it is a stuck record --
and worse, it talks over the things that ARE worth hearing.

So this file is the editor rather than the writer. It holds three rules:

  * DO NOT REPEAT YOURSELF. The same sentence, or a sentence that differs only
    in the distance, is not new information. Suppressed until it changes in a
    way a listener would notice.
  * SOME THINGS INTERRUPT. Finding a loop, finishing, giving up, walking into
    a room she has not been in -- these are events, not status, and they are
    allowed past the rate limit because a person waiting to hear "I am done"
    should not wait for a gap in the travelogue.
  * OTHERWISE, WAIT. A floor of NARRATION_MIN_GAP_S between routine lines. The
    number is in config, because how talkative Friday should be is a taste
    decision and taste decisions do not belong in code.

Nothing here speaks. It returns a string or None and the caller decides -- so
it is testable without a speaker, and the same decisions can be logged to the
dashboard instead of, or as well as, being said.
"""

from __future__ import annotations

import config


def _rounded(text: str) -> str:
    """A sentence with its distance blurred to the nearest half metre.

    "about 2.3 metres away" and "about 2.1 metres away" are the same news.
    Comparing on this rather than the raw string is what stops a slow approach
    to one frontier being narrated as a dozen separate discoveries.
    """
    out, changed = [], False
    for word in text.split():
        try:
            value = float(word)
        except ValueError:
            out.append(word)
            continue
        out.append(f"{round(value * 2) / 2:.1f}")
        changed = True
    return " ".join(out) if changed else text


class Narrator:
    """Turns a stream of exploration events into occasional sentences."""

    def __init__(self, min_gap_s: float | None = None) -> None:
        self.min_gap_s = (config.NARRATION_MIN_GAP_S if min_gap_s is None
                          else min_gap_s)
        self.last_said_at: float | None = None
        self.last_routine: str = ""
        self.spoken: list[str] = []
        self.suppressed = 0

    # --- the events that jump the queue ---------------------------------

    def _event(self, decision, closure, gained_m2: float) -> str | None:
        if closure:
            # The map has just been rewritten under her. Worth saying even
            # mid-sentence: it is the one moment she can tell she was wrong.
            return closure.get("say")
        if decision is not None and decision.done:
            return decision.say or decision.reason
        if gained_m2 >= config.NARRATION_ROOM_M2:
            return (f"that opened up, about {gained_m2:.0f} square metres "
                    f"I could not see before")
        return None

    def consider(self, decision=None, say: str | None = None,
                 closure: dict | None = None, gained_m2: float = 0.0,
                 now: float = 0.0) -> str | None:
        """The next thing worth saying, or None to stay quiet.

        `say` is for callers that plan their own route and have a line but no
        Decision object; it takes precedence over decision.say.
        """
        event = self._event(decision, closure, gained_m2)
        if event:
            self.last_said_at = now
            # An event is allowed to repeat a routine line, so the routine
            # memory is cleared rather than left to suppress it later.
            self.last_routine = ""
            self.spoken.append(event)
            return event

        line = say if say is not None else (
            getattr(decision, "say", "") if decision is not None else "")
        if not line:
            return None
        if _rounded(line) == self.last_routine:
            self.suppressed += 1
            return None
        if (self.last_said_at is not None
                and now - self.last_said_at < self.min_gap_s):
            self.suppressed += 1
            return None

        self.last_routine = _rounded(line)
        self.last_said_at = now
        self.spoken.append(line)
        return line
