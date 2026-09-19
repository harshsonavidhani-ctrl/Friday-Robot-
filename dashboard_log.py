"""Client side of the dashboard's live log.

Friday's brain does not run inside the dashboard process, so it cannot push
to the EventBus directly -- it POSTs to /api/events instead. This module is
the thin helper that does that, with the same call shape as
`dashboard.events.Turn` so brain code reads identically either way:

    from dashboard_log import Turn

    turn = Turn(model=config.MODEL_MAIN)
    turn.session_init(servers=[{"name": "friday-robot", "status": "connected"}])
    turn.thinking("The user wants me to move forward. Checking the path first.")
    turn.tool_call("mcp__friday-robot__is_path_clear", {})
    turn.tool_result("mcp__friday-robot__is_path_clear", {"clear": True})
    turn.done("Moving forward now.", input_tokens=1240, output_tokens=180)

Design rules, because this sits in the robot's hot path:

* Never raise. If the dashboard is down, not yet started, or slow, logging
  must not take the robot down with it. Every failure is swallowed after
  one warning.
* Never block for long. Posts go out on a background thread with a short
  timeout, so a stalled HTTP call cannot stall a motor command.
* No third-party imports -- urllib only, so the brain has no extra dep.
"""

from __future__ import annotations

import json
import queue
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import config
from dashboard.events import turn_cost_usd

_TIMEOUT_S = 2.0
_QUEUE_MAX = 256


class _Sender:
    """One background thread posting events in order, dropping under load."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._warned = False
        self._thread = threading.Thread(
            target=self._run, name="dashboard-log", daemon=True)
        self._thread.start()
        # Self-signed cert on the LAN: we are talking to our own Pi over
        # loopback, so verification buys nothing and would just fail.
        self._ctx = ssl.create_default_context()
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE

    def send(self, event: dict) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            pass          # the log is not worth back-pressuring the robot

    def flush(self, timeout: float = 3.0) -> None:
        """Wait for queued events to drain. For use before shutdown."""
        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.02)

    def _run(self) -> None:
        while True:
            event = self._q.get()
            try:
                body = json.dumps(event).encode()
                req = urllib.request.Request(
                    self.url, data=body, method="POST",
                    headers={"Content-Type": "application/json"})
                kwargs = {"timeout": _TIMEOUT_S}
                if self.url.startswith("https:"):
                    kwargs["context"] = self._ctx
                with urllib.request.urlopen(req, **kwargs):
                    pass
                self._warned = False
            except Exception as exc:            # noqa: BLE001 - never propagate
                if not self._warned:
                    print(f"[dashboard_log] dashboard unreachable ({exc}); "
                          f"log events are being dropped", file=sys.stderr)
                    self._warned = True


_sender: _Sender | None = None
_sender_lock = threading.Lock()


def _default_url() -> str:
    # Loopback plain HTTP: the brain and the dashboard are the same machine,
    # and this keeps TLS out of the robot's hot path entirely.
    return f"http://127.0.0.1:{config.DASHBOARD_HTTP_PORT}/api/events"


def sender(url: str | None = None) -> _Sender:
    global _sender
    with _sender_lock:
        if _sender is None:
            _sender = _Sender(url or _default_url())
        return _sender


def emit(event: dict) -> None:
    """Post one raw event. Prefer Turn below."""
    sender().send(event)


def flush(timeout: float = 3.0) -> None:
    if _sender is not None:
        _sender.flush(timeout)


class Turn:
    """Mirror of dashboard.events.Turn that posts over HTTP.

    Deliberately the same method names and arguments, so brain code can be
    pointed at either one without changing a line.
    """

    def __init__(self, model: str = "claude-sonnet-5",
                 url: str | None = None) -> None:
        self.model = model
        self.started = time.monotonic()
        self.tool_calls = 0
        self._sender = sender(url)

    def session_init(self, servers: list[dict[str, Any]] | None = None) -> None:
        self._sender.send({"type": "session_init", "model": self.model,
                           "servers": servers or []})

    def heard(self, text: str, source: str = "voice") -> None:
        emit({"type": "heard", "text": text, "source": source})

    def thinking(self, text: str) -> None:
        self._sender.send({"type": "thinking", "text": text})

    def tool_call(self, tool: str, args: dict | None = None) -> None:
        self.tool_calls += 1
        self._sender.send({"type": "tool_call", "tool": tool,
                           "args": args or {}})

    def tool_result(self, tool: str, result: Any,
                    refs: list[str] | None = None) -> None:
        self._sender.send({"type": "tool_result", "tool": tool,
                           "result": result, "refs": refs or []})

    def text(self, markdown: str) -> None:
        self._sender.send({"type": "text", "text": markdown})

    def done(self, markdown: str, input_tokens: int = 0,
             output_tokens: int = 0, cache_read_tokens: int = 0,
             cache_write_tokens: int = 0, turns: int | None = None) -> None:
        self._sender.send({
            "type": "done",
            "text": markdown,
            "model": self.model,
            "cost_usd": turn_cost_usd(self.model, input_tokens, output_tokens,
                                      cache_read_tokens, cache_write_tokens),
            "elapsed_s": time.monotonic() - self.started,
            "turns": self.tool_calls if turns is None else turns,
        })


if __name__ == "__main__":
    # Smoke test: python dashboard_log.py  -- posts one complete turn.
    t = Turn()
    t.session_init(servers=[{"name": "friday-robot", "status": "connected"}])
    t.thinking("Checking the path before moving.")
    t.tool_call("mcp__friday-robot__is_path_clear", {})
    t.tool_result("mcp__friday-robot__is_path_clear", {"clear": True})
    t.done("Path is clear, moving forward.",
           input_tokens=1200, output_tokens=300)
    flush()
    print("posted one turn to", _default_url())
