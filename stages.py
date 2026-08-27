"""stages.py — one line per stage transition, and a deadline on every stage.

WHY THIS EXISTS. Friday's worst failure was not a crash. It was a turn that
went quiet: the wake word fired, the face changed to listening, and then
nothing happened, with no way to tell whether the capture grabbed no audio,
whether whisper returned an empty string, whether the brain threw, or whether
a tool call was still hanging. Every one of those looks identical from across
the room -- "she heard me and did nothing" -- and each has a different fix.

So every stage now announces itself:

    WAKE (0.83) -> CAPTURING -> TRANSCRIBED "move forward" -> THINKING
    -> TOOL move_forward -> TOOL move_forward ok (0.4s) -> SPEAKING -> DONE
    -> LISTENING

and every stage has a ceiling. A stage that overruns raises StageTimeout,
which the orchestrator turns into a short spoken apology and a clean re-arm.
NOTHING IS ALLOWED TO HANG SILENTLY. A Friday that says "that took too long"
is diagnosable; a Friday that stares at you is not.

The trace is logged at INFO, not DEBUG, deliberately: this is the log Harsh
reads when a command does not happen, so it has to be there in production
without anyone having thought to turn it on first.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

log = logging.getLogger("friday.turn")

T = TypeVar("T")


class StageTimeout(Exception):
    """A stage ran past its ceiling and was abandoned."""

    def __init__(self, stage: str, seconds: float) -> None:
        self.stage = stage
        self.seconds = seconds
        super().__init__(f"{stage} exceeded {seconds:g}s")


class Turn:
    """The trace for one turn, from wake to re-arm.

    Not a context manager on purpose. A turn does not have one nesting shape:
    it may end at TRANSCRIBED (nothing was said), at THINKING (no brain), or
    after several TOOL rounds, and forcing that into `with` blocks buried the
    interesting part in indentation.
    """

    __slots__ = ("id", "began", "stage_began", "stage_name", "_counter")

    _sequence = 0

    def __init__(self, trigger: str = "") -> None:
        Turn._sequence += 1
        self.id = Turn._sequence
        self.began = time.monotonic()
        self.stage_began = self.began
        self.stage_name = "WAKE"
        self._counter = 0
        self._say("WAKE" + (f" ({trigger})" if trigger else ""))

    # -- the trace ---------------------------------------------------------
    def _say(self, text: str) -> None:
        log.info("[turn %d] %s", self.id, text)

    def stage(self, name: str, detail: str = "") -> None:
        """Enter a stage. Reports how long the previous one took."""
        now = time.monotonic()
        took = now - self.stage_began
        self.stage_began = now
        self.stage_name = name
        line = f"{name}"
        if detail:
            line += f" {detail}"
        self._say(f"{line}   (+{took:.1f}s, {now - self.began:.1f}s in)")

    def note(self, text: str) -> None:
        """Something worth seeing that is not a stage change."""
        self._say(f"  {text}")

    def done(self, outcome: str = "DONE") -> None:
        self._say(f"{outcome}   (turn took {time.monotonic() - self.began:.1f}s)")

    def listening(self) -> None:
        """The re-arm. Its own line because its ABSENCE is the bug.

        A turn whose trace ends without this one did not hand the microphone
        back, and the command that appears to fail is the NEXT one.
        """
        self._say("LISTENING")

    # -- deadlines ---------------------------------------------------------
    async def run(self, name: str, awaitable: Awaitable[T],
                  timeout: float, detail: str = "") -> T:
        """Enter a stage and run it, or abandon it at the ceiling."""
        self.stage(name, detail)
        try:
            return await asyncio.wait_for(awaitable, timeout)
        except asyncio.TimeoutError:
            self._say(f"  {name} TIMED OUT after {timeout:g}s — abandoning")
            raise StageTimeout(name, timeout) from None

    async def call(self, name: str, fn: Callable[..., T], *args: Any,
                   timeout: float, detail: str = "") -> T:
        """The same, for a blocking function that has to go to a thread."""
        return await self.run(name, asyncio.to_thread(fn, *args),
                              timeout, detail)


def quote(text: str, limit: int = 200) -> str:
    """A transcript as it should appear in the log — always, even when empty.

    An empty transcript is the single most useful line in this whole file: it
    separates "the microphone heard nothing" from "the brain ignored me",
    which are the two halves of the same complaint.
    """
    text = (text or "").strip()
    if not text:
        return '"" (EMPTY — nothing was transcribed)'
    if len(text) > limit:
        return f'"{text[:limit]}…" ({len(text)} chars)'
    return f'"{text}"'
