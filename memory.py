"""memory.py — what Friday remembers, between turns and between boots.

Two different things, deliberately kept apart:

  * conversation memory — the last few turns, so "and what about there?"
    resolves. Short, in RAM, dropped on restart.
  * long-term memory — facts worth keeping ("the kitchen is through the left
    door"). On disk, small, and written only when the brain explicitly decides
    something is worth keeping. Never a dump of every transcript.

The MCP server's decision log (mcp_server/decision_log.py) is a separate
record: it logs what Friday DID. This module is what Friday KNOWS.

Not built yet — a later step.
"""

from __future__ import annotations


def recent_turns(limit: int = 6) -> list[dict]:
    raise NotImplementedError("memory.recent_turns() — not built yet")


def remember(fact: str) -> None:
    raise NotImplementedError("memory.remember() — not built yet")
