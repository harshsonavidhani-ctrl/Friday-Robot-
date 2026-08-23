"""brain.py — Claude, the MCP tool loop, and nothing else.

Takes a transcript, returns spoken text. Everything physical happens through
the friday-robot MCP server; this module never imports motor.py, vision.py or
sensors.py directly. That rule is what keeps the safety interrupt in the MCP
server un-bypassable — see mcp_server/safety.py.

  * model         config.MODEL_MAIN (Sonnet 5), config.MODEL_FAST for reflexes
  * system prompt  personality.md
  * caching        on, for the system prompt and the tool definitions
  * spend          logged per call, capped at config.DAILY_SPEND_CAP_USD

Not built yet — Step 1 fills this in.
"""

from __future__ import annotations
