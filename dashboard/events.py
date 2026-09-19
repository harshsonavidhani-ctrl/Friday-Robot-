"""The event bus behind the live log.

Anything that wants to appear in the dashboard emits an event here; every
connected browser is pushed it over SSE. Events are also kept in a bounded
ring buffer so a browser that connects late still sees recent history
instead of an empty screen.

The taxonomy is fixed -- session_init, thinking, tool_call, tool_result,
text, done -- because the UI colours and groups by it. Adding a new type
means adding a card style to the front end too.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from collections import deque
from typing import Any

import logging

import config

# The card types the front end knows how to render. Anything else is
# rejected at ingest rather than silently showing up as a blank card.
log = logging.getLogger("friday.dashboard.events")

EVENT_TYPES = (
    "heard",          # what was said TO her, as the transcriber heard it
    "session_init",   # model + connected MCP servers, at the start of a turn
    "thinking",       # the model's reasoning text
    "tool_call",      # a tool invocation and its arguments
    "tool_result",    # what the tool returned
    "text",           # Friday's spoken/written reply (markdown)
    "done",           # final reply + cost / elapsed / turn count
)


class EventBus:
    """Fan-out from one producer to many browsers, plus recent history.

    THE HISTORY IS ON DISK AS WELL AS IN MEMORY. Harsh, 2026-09-20: "the logs
    which comes should remain and not get lost after robot is off or
    disconnected". It used to live only in this deque, so the log he was
    reading vanished whenever the DASHBOARD restarted -- and tonight that was
    a dozen times. The robot going away never mattered (nothing posts, the
    cards stay); restarting this service did.

    One JSON object per line, appended as it fans out, and the tail read back
    at start-up. Bounded by rewriting the file down to the last
    DASHBOARD_HISTORY events once it passes DASHBOARD_EVENT_MAX_BYTES, so it
    cannot grow forever on an SD card that has already been full once.
    """

    def __init__(self, history: int = config.DASHBOARD_HISTORY) -> None:
        self._history: deque[dict] = deque(maxlen=history)
        self._subscribers: set[asyncio.Queue] = set()
        self._seq = 0
        self._file = Path(getattr(config, "DASHBOARD_EVENT_FILE",
                                  config.ROOT / "state" / "dashboard_events.jsonl"))
        self._load()

    # --- the disk half -----------------------------------------------

    def _load(self) -> None:
        """Read the tail back. Never raises: a lost log is not a dead server."""
        try:
            if not self._file.exists():
                return
            lines = self._file.read_text(errors="replace").splitlines()
        except OSError as exc:
            log.warning("could not read the event log: %s", exc)
            return
        kept = 0
        for line in lines[-self._history.maxlen:]:
            try:
                event = json.loads(line)
            except ValueError:
                continue            # a torn last line must not lose the rest
            if isinstance(event, dict) and event.get("type") in EVENT_TYPES:
                self._history.append(event)
                self._seq = max(self._seq, int(event.get("seq") or 0))
                kept += 1
        if kept:
            log.info("event log: %d card%s restored from disk",
                     kept, "" if kept == 1 else "s")

    def _persist(self, stored: dict) -> None:
        """Append one event, and keep the file bounded. Never raises."""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with self._file.open("a") as handle:
                handle.write(json.dumps(stored) + "\n")
            limit = int(getattr(config, "DASHBOARD_EVENT_MAX_BYTES",
                                2_000_000))
            if limit and self._file.stat().st_size > limit:
                # Rewritten from the deque, which is already the last N.
                self._file.write_text("".join(
                    json.dumps(e) + "\n" for e in self._history))
                log.info("event log trimmed to %d cards", len(self._history))
        except OSError as exc:
            log.warning("could not write the event log: %s", exc)

    # --- producing ---------------------------------------------------

    def emit(self, event: dict) -> dict:
        """Record and fan out one event. Returns the stored form.

        Never raises on a slow or dead subscriber -- a browser that has
        stopped reading must not be able to block the robot.
        """
        kind = event.get("type")
        if kind not in EVENT_TYPES:
            raise ValueError(
                f"unknown event type {kind!r}. Valid: {', '.join(EVENT_TYPES)}"
            )
        self._seq += 1
        stored = dict(event)
        stored["seq"] = self._seq
        stored.setdefault("ts", time.time())
        self._history.append(stored)
        self._persist(stored)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(stored)
            except asyncio.QueueFull:
                # Drop for this subscriber only; it will resync on reload.
                pass
        return stored

    def clear(self) -> None:
        """Both halves. Clearing is something a person asked for."""
        self._history.clear()
        try:
            if self._file.exists():
                self._file.unlink()
        except OSError as exc:
            log.warning("could not clear the event log: %s", exc)

    # --- consuming ---------------------------------------------------

    def history(self) -> list[dict]:
        return list(self._history)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def sse_format(event: dict) -> bytes:
    """One Server-Sent Event frame."""
    return f"id: {event.get('seq', 0)}\ndata: {json.dumps(event)}\n\n".encode()


# =====================================================================
# Turn tracking -- what fills in the DONE card's footer
# =====================================================================

# Published per-million-token prices, so the footer shows a real number
# rather than a placeholder. Update if Anthropic's pricing changes.
# Anthropic first-party rates, USD per million tokens. Checked 2026-09-08
# against the current model table; the previous values here were guesses that
# were wrong in both directions -- sonnet-5 was listed at 3/15 (it is 2/10) and
# opus-5 at 15/75 (it is 5/25), so every figure the dashboard has ever shown was
# overstated, opus by 3x.
#
# ⚠ THE KEY IS THE EXACT MODEL ID AND DATE SUFFIXES ARE NOT VALID. The old
# "claude-haiku-4-5-20251001" is rejected by the API, and because
# turn_cost_usd returns 0.0 for an unknown key it would ALSO have silently
# logged every Haiku call as costing nothing -- a cost log that reads $0.00 is
# indistinguishable from one that is working. Unknown models are now loud.
import logging as _logging
_LOG = _logging.getLogger("friday.cost")

PRICING_USD_PER_MTOK = {
    "claude-opus-5":     {"input": 5.00, "output": 25.00},
    "claude-sonnet-5":   {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00},
}


def turn_cost_usd(model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0,
                  cache_write_tokens: int = 0) -> float:
    """Dollar cost of one turn from its real token usage.

    Cached reads bill at 10% of the input rate and cache writes at 125%,
    which is why they are counted separately rather than folded into
    input_tokens.
    """
    price = PRICING_USD_PER_MTOK.get(model)
    if price is None:
        # NOT silently zero. A model missing from the table is a typo or a new
        # model, and both must be visible: returning 0.0 made a broken cost log
        # look like a cheap day.
        _LOG.warning("no price for %r -- its cost is being logged as 0.00. "
                     "Add it to PRICING_USD_PER_MTOK.", model)
        return 0.0
    per_tok_in = price["input"] / 1_000_000
    per_tok_out = price["output"] / 1_000_000
    return (input_tokens * per_tok_in
            + output_tokens * per_tok_out
            + cache_read_tokens * per_tok_in * 0.10
            + cache_write_tokens * per_tok_in * 1.25)


class Turn:
    """Accumulates one turn's usage so the DONE card can report it.

        turn = Turn(bus, model=config.MODEL_MAIN)
        turn.session_init(servers=[...])
        turn.thinking("...")
        turn.tool_call("mcp__friday-robot__move", {"direction": "forward"})
        turn.tool_result("mcp__friday-robot__move", "OK")
        turn.done("On my way.", input_tokens=..., output_tokens=...)
    """

    def __init__(self, bus: EventBus, model: str) -> None:
        self.bus = bus
        self.model = model
        self.started = time.monotonic()
        self.tool_calls = 0

    def session_init(self, servers: list[dict[str, Any]] | None = None) -> None:
        self.bus.emit({
            "type": "session_init",
            "model": self.model,
            "servers": servers or [],
        })

    def heard(self, text: str, source: str = "voice") -> None:
        """What was said TO her — the one thing the stream never showed.

        The dashboard has carried her side since it was written (thinking,
        tool calls, replies) and none of his, because only exploration_mode
        ever posted to it. A conversation with one side missing reads like a
        robot talking to itself.
        """
        self.bus.emit({"type": "heard", "text": text, "source": source})

    def thinking(self, text: str) -> None:
        self.bus.emit({"type": "thinking", "text": text})

    def tool_call(self, tool: str, args: dict | None = None) -> None:
        self.tool_calls += 1
        self.bus.emit({"type": "tool_call", "tool": tool, "args": args or {}})

    def tool_result(self, tool: str, result: Any,
                    refs: list[str] | None = None) -> None:
        self.bus.emit({"type": "tool_result", "tool": tool,
                       "result": result, "refs": refs or []})

    def text(self, markdown: str) -> None:
        self.bus.emit({"type": "text", "text": markdown})

    def done(self, markdown: str, input_tokens: int = 0,
             output_tokens: int = 0, cache_read_tokens: int = 0,
             cache_write_tokens: int = 0, turns: int | None = None) -> None:
        self.bus.emit({
            "type": "done",
            "text": markdown,
            "model": self.model,
            "cost_usd": turn_cost_usd(self.model, input_tokens, output_tokens,
                                      cache_read_tokens, cache_write_tokens),
            "elapsed_s": time.monotonic() - self.started,
            "turns": self.tool_calls if turns is None else turns,
        })
