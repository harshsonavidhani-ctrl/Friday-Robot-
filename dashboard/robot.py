"""robot.py — the dashboard's link to Friday's body.

The dashboard does NOT open the motor serial port. It calls the friday-robot
MCP server, exactly as the brain does, for two reasons that both matter:

  * The Arduino is a single serial link and the MCP server already holds it.
    A second opener would fight it for replies and reset the board.
  * Manual control then gets the SAME safety interrupt as the brain — the
    clearance check, the duration cap, the automatic stop. A "manual" mode
    that bypassed those would be the one way to drive Friday into a wall from
    a phone, which is precisely what the interrupt exists to prevent.

A fresh session per call, on purpose. Button presses are not a hot path, one
extra local round trip is imperceptible, and it means there is no long-lived
session to go stale, reconnect, or leak when the MCP server restarts.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger("friday.dashboard.robot")


class RobotUnavailable(RuntimeError):
    """The MCP server is not reachable. Reported, never raised at a user."""


async def call(tool: str, arguments: dict | None = None,
               timeout: float | None = None) -> str:
    """Call one friday-robot tool and return its text result.

    Never raises for a tool-level failure — the MCP server already turns those
    into readable strings. Raises RobotUnavailable only if the server itself
    cannot be reached, which is a different problem with a different fix.
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:                      # noqa: BLE001
        raise RobotUnavailable(f"mcp is not installed in the venv ({exc})")

    timeout = config.DASHBOARD_ROBOT_TIMEOUT_S if timeout is None else timeout
    try:
        async with streamable_http_client(config.MCP_HTTP_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments or {})
    except Exception as exc:                        # noqa: BLE001
        log.warning("MCP call %s failed: %s", tool, exc)
        raise RobotUnavailable(
            f"cannot reach Friday's MCP server at {config.MCP_HTTP_URL} "
            f"({type(exc).__name__}). Is friday-mcp.service running?") from exc

    parts = [item.text for item in result.content
             if getattr(item, "type", None) == "text"]
    return "\n".join(parts).strip() or "(no output)"


async def available() -> bool:
    """Is the MCP server answering? For the status dot, not for gating."""
    try:
        await call("get_face", timeout=3.0)
        return True
    except RobotUnavailable:
        return False
