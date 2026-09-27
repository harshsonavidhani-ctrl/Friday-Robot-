"""routines.py — File 16 step 3: predefined routines, one phrase, a fixed chain.

    routines.match("good morning")   -> "good morning"
    routines.run("good morning")     -> "Good morning! 24 degrees ... No reminders ..."

A routine is config.ROUTINES[name]: trigger phrases, an optional first line,
and a list of STEPS. A step is the NAME of a tool the MCP server already has
(weather_now, dismiss_screen, reminders_today, ...) plus its arguments, so
adding a routine is adding a dict entry -- no code, no new tool. Voice-authored
routines ("Friday, when I say X do Y") are future scope; this is the structure
they would write into.

It runs INSIDE the MCP server (run_routine), because that is where the tools
are: each step calls the tool's plain function -- registry.tool returns it
undecorated -- so a step is exactly what the brain would have got. A step that
fails or does not exist is skipped and logged; the rest of the routine still
runs. Nothing here moves the robot: routines are tools-only by design, and a
movement tool in a routine would still go through the MCP safety layer.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from typing import Any, Callable

import config

log = logging.getLogger("friday.routines")


def names() -> list[str]:
    return list(getattr(config, "ROUTINES", {}))


def describe() -> list[dict[str, Any]]:
    """For the dashboard and the brain: name, phrases, step tool names."""
    return [{"name": name,
             "phrases": list(spec.get("phrases") or ()),
             "steps": [step.get("tool") for step in spec.get("steps") or ()]}
            for name, spec in getattr(config, "ROUTINES", {}).items()]


def match(heard: str) -> str | None:
    """The routine a short sentence names, or None.

    Short on purpose (config.ROUTINE_MAX_WORDS, without "hey Friday"): "good
    morning" is the ask; "good morning everyone, welcome back to the channel"
    is a video.
    """
    text = re.sub(r"^\W*(?:hey|hi|ok|okay)?\W*friday\W*", "",
                  (heard or "").lower()).strip(" .,!?")
    if not text or len(text.split()) > int(config.ROUTINE_MAX_WORDS):
        return None
    for name, spec in getattr(config, "ROUTINES", {}).items():
        for phrase in (name, *(spec.get("phrases") or ())):
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                return name
    return None


def _tool(name: str) -> Callable[..., Any] | None:
    """The plain function of an MCP tool, by name, from mcp_server/tools/."""
    import mcp_server.tools as package
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"{package.__name__}.{info.name}")
        except Exception:                           # noqa: BLE001
            continue
        found = getattr(module, name, None)
        if callable(found):
            return found
    return None


def _spoken(result: Any) -> str:
    """A tool's answer as a sentence: brackets and trailing notes dropped."""
    text = result if isinstance(result, str) else str(result or "")
    text = text.split("\n\n")[0]
    text = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", text)
    return text.strip()


def run(name: str) -> tuple[str, list[dict]]:
    """Run a routine's steps in order. -> (what to say, per-step report)."""
    spec = getattr(config, "ROUTINES", {}).get(name)
    if spec is None:
        return (f"I don't have a routine called {name}. I know "
                f"{', '.join(names()) or 'none yet'}.", [])
    said = [spec["say_first"]] if spec.get("say_first") else []
    report = []
    for step in spec.get("steps") or ():
        tool, args = step.get("tool"), dict(step.get("args") or {})
        fn = _tool(tool)
        if fn is None:
            log.warning("routine %r: no tool called %r — skipped", name, tool)
            report.append({"tool": tool, "ok": False, "result": "no such tool"})
            continue
        try:
            result = fn(**args)
        except Exception as exc:                    # noqa: BLE001
            log.warning("routine %r: %s failed (%s: %s) — skipped", name,
                        tool, type(exc).__name__, exc)
            report.append({"tool": tool, "ok": False, "result": str(exc)})
            continue
        report.append({"tool": tool, "ok": True, "result": str(result)[:300]})
        if step.get("speak", True):
            text = _spoken(result)
            if text:
                said.append(text if text.endswith((".", "!", "?"))
                            else text + ".")
    log.info("routine %r: %s", name,
             ", ".join(f"{r['tool']}={'ok' if r['ok'] else 'SKIPPED'}"
                       for r in report))
    return " ".join(said) or "Done.", report
