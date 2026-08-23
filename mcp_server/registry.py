"""registry.py — how a capability becomes a tool the brain can call.

ADDING A TOOL IS ONE DECORATED FUNCTION. Drop a file in mcp_server/tools/,
write this, and you are done — no registration list to update, no server file
to edit, nothing to import anywhere:

    from mcp_server.registry import tool

    @tool
    def wave_arm(times: int = 2) -> str:
        \"\"\"Wave Friday's arm. Use when greeting someone.

        Args:
            times: how many waves, 1-5.
        \"\"\"
        return f"waved {times} times"

Every module in mcp_server/tools/ is imported automatically at start-up, so
the file existing is the registration.

WHAT THE DECORATOR ADDS, so no tool has to remember it:

  * Failure containment. Any exception becomes a readable string the brain can
    act on. A tool NEVER takes the server down — a broken camera must not stop
    the motors from working, and vice versa.
  * The decision log. Name, arguments, result, duration and timestamp of every
    call, written whether it succeeded or failed.
  * Blocking work off the event loop. Hardware calls are blocking and some take
    seconds. They run in a worker thread so the server can still answer — which
    is what makes stop() able to interrupt a move that is already running.

THE DOCSTRING IS THE PROMPT. The model chooses tools by reading the first line
and the Args section, so write them for the model, not for a maintainer. Say
when to use it, not just what it does.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import pkgutil
import time
from dataclasses import dataclass
from typing import Any, Callable

import anyio

from mcp_server import decision_log

log = logging.getLogger("friday.mcp")

# Populated by @tool at import time; drained by build_server().
REGISTRY: list[tuple[Callable[..., Any], str, str]] = []


@dataclass
class ToolResult:
    """For tools whose useful output is not text — look(), mainly.

    `content` goes to the model (an mcp Image, say); `log_text` is the human
    sentence written to the decision log, because "<Image object>" is useless
    at 1am and "up cam ok; down cam not on the bus" is not.
    """
    content: Any
    log_text: str


def _describe(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    return doc.strip() or f"{func.__name__} (no description)"


def _format_error(exc: BaseException) -> str:
    """An error the model can actually act on.

    The type name is included because "no Arduino found on USB" and
    "Permission denied" call for completely different next moves, and a bare
    message often hides which one it is.
    """
    name = type(exc).__name__
    text = str(exc).strip() or "(no detail)"
    return f"{name}: {text}"


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None):
    """Register a function as an MCP tool. Use bare or with a name override."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        description = _describe(fn)
        signature = inspect.signature(fn)
        is_async = inspect.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def invoke(*args: Any, **kwargs: Any) -> str:
            bound: dict[str, Any] = {}
            try:
                call = signature.bind(*args, **kwargs)
                call.apply_defaults()
                bound = dict(call.arguments)
            except TypeError:
                bound = {"args": args, "kwargs": kwargs}

            started = time.monotonic()
            try:
                if is_async:
                    result = await fn(*args, **kwargs)
                else:
                    # to_thread keeps the event loop free, which is what lets
                    # a stop() call be answered while a move is still running.
                    result = await anyio.to_thread.run_sync(
                        functools.partial(fn, *args, **kwargs))
                if isinstance(result, ToolResult):
                    payload, text = result.content, result.log_text
                elif isinstance(result, str):
                    payload = text = result
                else:
                    payload = text = str(result)
                decision_log.record(
                    tool_name, bound, text, True, None,
                    int((time.monotonic() - started) * 1000))
                return payload
            except Exception as exc:        # noqa: BLE001 - containment IS the job
                message = _format_error(exc)
                log.warning("tool %s failed: %s", tool_name, message)
                decision_log.record(
                    tool_name, bound, None, False, message,
                    int((time.monotonic() - started) * 1000))
                return f"FAILED — {message}"

        # MCP builds the argument schema from the signature. functools.wraps
        # alone leaves the wrapper looking like (*args, **kwargs), so the real
        # signature is put back explicitly.
        #
        # The return annotation is deliberately dropped rather than declared.
        # Every tool can return EITHER its normal result or an error string,
        # and the server registers these with structured_output=False so the
        # SDK renders whatever comes back as content instead of validating it
        # against a schema that a failure would violate.
        invoke.__signature__ = signature.replace(
            return_annotation=inspect.Signature.empty)
        invoke.__annotations__ = {
            k: v for k, v in getattr(fn, "__annotations__", {}).items()
            if k != "return"}

        REGISTRY.append((invoke, tool_name, description))
        return fn                            # the plain function stays callable

    return wrap(func) if func is not None else wrap


def load_tool_modules() -> list[str]:
    """Import every module in mcp_server/tools/. Importing is registering.

    A module that fails to import is reported and SKIPPED, not fatal: one
    broken tool file must not cost Friday every other capability.
    """
    from mcp_server import tools as tools_pkg

    loaded: list[str] = []
    for info in pkgutil.iter_modules(tools_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{tools_pkg.__name__}.{info.name}")
            loaded.append(info.name)
        except Exception as exc:            # noqa: BLE001
            log.error("tool module %r failed to import: %s",
                      info.name, _format_error(exc))
    return loaded
