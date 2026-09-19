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


class _Dash:
    """The dashboard side of a turn, or nothing at all.

    A thin adapter rather than dashboard_log.Turn directly, because that
    class is built around one long agent run (it owns the model name and the
    cost arithmetic) while a voice turn may be answered locally, by the fast
    path or by the full brain, and only knows which at the end.
    """

    def __init__(self) -> None:
        import dashboard_log                     # noqa: PLC0415
        self._log = dashboard_log

    def heard(self, text: str, source: str = "voice") -> None:
        self._log.emit({"type": "heard", "text": text, "source": source})

    def thinking(self, text: str) -> None:
        self._log.emit({"type": "thinking", "text": text})

    def tool_call(self, tool: str, args: dict) -> None:
        self._log.emit({"type": "tool_call", "tool": tool, "args": args})

    def tool_result(self, tool: str, result: Any) -> None:
        self._log.emit({"type": "tool_result", "tool": tool,
                        "result": result, "refs": []})

    def done_event(self, text: str, model: str, usd: float,
                   elapsed: float, turns: int) -> None:
        self._log.emit({"type": "done", "text": text, "model": model,
                        "cost_usd": round(float(usd or 0.0), 5),
                        "elapsed_s": round(elapsed, 1), "turns": turns})


def _dashboard(turn: "Turn") -> "_Dash | None":
    """One adapter per turn, or None when the feature is off or unavailable."""
    try:
        import config                            # noqa: PLC0415
        if not getattr(config, "DASHBOARD_TURNS", True):
            return None
        return _Dash()
    except Exception as exc:                     # noqa: BLE001
        log.debug("no dashboard for this turn: %s", exc)
        return None


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

    __slots__ = ("id", "began", "stage_began", "stage_name", "_counter",
                 "_dash", "_said", "_tools", "_model", "_usd",
                 "trigger")

    _sequence = 0

    def __init__(self, trigger: str = "") -> None:
        Turn._sequence += 1
        self.id = Turn._sequence
        self.began = time.monotonic()
        self.stage_began = self.began
        self.stage_name = "WAKE"
        self._counter = 0
        self.trigger = trigger
        self._said = ""
        self._tools = 0
        self._model = ""
        self._usd = 0.0
        self._dash = _dashboard(self)
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

    def note(self, text: str, dash: bool = True) -> None:
        """Something worth seeing that is not a stage change.

        dash=False keeps it out of the dashboard when a card already carries
        the same thing -- a tool's answer is a tool_result card, and posting
        the note as well showed every offline tool twice.
        """
        self._say(f"  {text}")
        if not dash:
            return
        # A note is a DECISION she made without speaking -- "reflex ... no
        # round trip", "not addressed to her — silent", "IGNORE overridden".
        # On the dashboard those are the most interesting cards of all,
        # because they explain a turn where she did nothing.
        self._to_dash("thinking", text=text)

    # -- the same trace, on the dashboard ---------------------------------
    # ⚠ EVERY ONE OF THESE IS FIRE AND FORGET. dashboard_log posts on its own
    # thread and drops events when the queue is full; nothing here may raise,
    # block, or slow a turn down, because this is the robot's hot path and
    # the dashboard is a convenience.

    def _to_dash(self, kind: str, **fields: Any) -> None:
        if self._dash is None:
            return
        try:
            getattr(self._dash, kind)(**fields)
        except Exception as exc:                # noqa: BLE001
            log.debug("dashboard event dropped (%s): %s", kind, exc)

    def heard(self, text: str, source: str = "voice") -> None:
        """What was said to her, as the transcriber heard it."""
        self._to_dash("heard", text=text, source=source)

    def tool_call(self, tool: str, args: Any = None) -> None:
        self._tools += 1
        self._to_dash("tool_call", tool=tool,
                      args=args if isinstance(args, dict) else {"args": args})

    def tool_result(self, tool: str, result: Any) -> None:
        self._to_dash("tool_result", tool=tool, result=result)

    def spoke(self, text: str) -> None:
        """What she said out loud. Kept for the `done` card at the end."""
        self._said = text or self._said

    def answered_by(self, model: str, usd: float = 0.0) -> None:
        """Which brain answered, and what the turn cost. Shown on `done`."""
        self._model = model or self._model
        self._usd = float(usd or 0.0)

    def finished(self, outcome: str, model: str = "", usd: float = 0.0) -> None:
        """The closing card: what she said, on which brain, for how much."""
        self._to_dash("done_event", text=self._said or outcome, model=model,
                      usd=usd, elapsed=time.monotonic() - self.began,
                      turns=self._tools)

    def done(self, outcome: str = "DONE") -> None:
        self._say(f"{outcome}   (turn took {time.monotonic() - self.began:.1f}s)")
        self.finished(outcome, self._model, self._usd)

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
