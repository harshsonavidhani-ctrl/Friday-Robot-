"""mode.py — File 09 step 8: companion by default, professional when it matters.

    mode.current()                  the sticky mode, from disk
    mode.set_mode("professional")   the explicit toggle, until turned off
    mode.for_turn(text)             what THIS turn should sound like, and why
    mode.instructions(name)         the paragraph that rides on the turn
    mode.restrain(emotion, name)    the face, toned down where it must be

Harsh's spec: "companion (default — warm, casual, human-like...) or
professional (crisp, precise, minimal filler, confident — for demos,
technical explanations, or formal requests)", with two ways in — an explicit
command that sticks until it is turned off, and automatic detection "without
needing the explicit command every time".

WHY THE MODE IS NOT IN THE SYSTEM PROMPT. personality.md is the cached prefix
of every request; changing it per turn would throw the prompt cache away on
every turn and cost more than the feature is worth. The mode instead rides on
the question itself, the same seam memory, the situation block and the
reflection note already use (brain._outgoing).

THE STICKY MODE IS ON DISK, not in memory: "toggles it until turned off" has
to survive a restart, and the brain restarts far more often than a demo
lasts. state/mode.json, one small object, and a missing or torn file simply
means companion.

⚠ AUTOMATIC DETECTION IS PHRASING, NOT IDENTITY. The spec also asks for
"someone Friday hasn't talked to before asking it to explain itself". She
cannot tell who is speaking -- face recognition is a separate module and is
not built -- so the honest proxy is used instead: an explain-yourself
question arriving when she has NOT been in conversation (nobody has heard her
voice for MODE_STRANGER_QUIET_S) is treated as a stranger at a demo. It is a
proxy and it is written down as one; when face recognition lands, that is the
signal to replace it with.

AUTOMATIC IS FOR ONE EXCHANGE ONLY. The spec is explicit -- "default to
professional tone for that exchange" -- so an automatic switch never touches
the stored mode. Only the spoken command does.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.mode")

COMPANION = "companion"
PROFESSIONAL = "professional"
MODES = (COMPANION, PROFESSIONAL)


def _file() -> Path:
    return Path(getattr(config, "MODE_FILE", config.ROOT / "state" / "mode.json"))


# --- the sticky mode --------------------------------------------------------

def current() -> str:
    """The mode he last set, or the default. Never raises."""
    default = getattr(config, "MODE_DEFAULT", COMPANION)
    try:
        data = json.loads(_file().read_text())
        name = str(data.get("mode", "")).strip().lower()
        return name if name in MODES else default
    except (OSError, ValueError, TypeError, AttributeError):
        return default


def state() -> dict[str, Any]:
    """The whole stored record — mode, when, and why. For the dashboard."""
    try:
        data = json.loads(_file().read_text())
        if isinstance(data, dict) and data.get("mode") in MODES:
            return data
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return {"mode": getattr(config, "MODE_DEFAULT", COMPANION),
            "since": 0.0, "why": "never set"}


def quiet() -> bool:
    """File 12 step 2: has he asked her to keep it to the screen?

    Stored beside the companion/professional mode because it is the same
    kind of thing — a standing choice about HOW she answers, set by voice or
    from the dashboard, that has to survive a restart.
    """
    return bool(state().get("quiet"))


def set_quiet(on: bool, why: str = "he asked") -> bool:
    """Turn quiet mode on or off. Returns what is now in force."""
    data = state()
    data["quiet"] = bool(on)
    data["quiet_why"] = why
    data["quiet_since"] = time.time()
    try:
        path = _file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        log.info("quiet mode -> %s (%s)", "on" if on else "off", why)
    except OSError as exc:
        log.warning("could not save quiet mode: %s", exc)
    return bool(on)


def set_mode(name: str, why: str = "he asked") -> str:
    """Set the sticky mode. Returns the mode actually in force."""
    name = (name or "").strip().lower()
    if name not in MODES:
        return current()
    try:
        path = _file()
        path.parent.mkdir(parents=True, exist_ok=True)
        # KEEP WHAT IS ALREADY THERE. quiet mode lives in this file too, and
        # writing a fresh object would turn it off every time he switched
        # between companion and professional.
        data = state()
        data.update({"mode": name, "since": time.time(), "why": why})
        path.write_text(json.dumps(data))
        log.info("mode -> %s (%s)", name, why)
    except OSError as exc:
        log.warning("could not save the mode: %s", exc)
    return name


# --- the explicit command ---------------------------------------------------

# Said out loud. Kept here rather than in offline.py's table because the two
# directions and the words for them belong with the mode they set.
_PROFESSIONAL_ON = re.compile(
    r"\b(professional|demo|presentation|formal|business)\s*mode\b"
    r"|\bmode\s*(professional|demo|formal)\b"
    r"|\b(go|switch|turn it|be)\s+professional\b"
    r"|\bdemo\s+mode\s+on\b", re.IGNORECASE)
_PROFESSIONAL_OFF = re.compile(
    r"\b(companion|casual|normal|friendly|relax(ed)?|personal)\s*mode\b"
    r"|\bmode\s*(companion|casual|normal)\b"
    r"|\b(demo|professional|formal)\s+mode\s+off\b"
    r"|\b(stop|end|exit|leave)\s+(the\s+)?(demo|professional|formal)\b"
    r"|\bbe\s+(casual|normal|yourself)\b", re.IGNORECASE)


_QUIET_ON = re.compile(
    r"\b(quiet mode|be quiet|stay quiet|don'?t speak|do not speak|"
    r"no talking|silent mode|shush|mute yourself)\b", re.IGNORECASE)
_QUIET_OFF = re.compile(
    r"\b(quiet mode off|you can (talk|speak)( again)?|speak again|"
    r"talk again|unmute|stop being quiet|out loud again)\b", re.IGNORECASE)


def quiet_asked_for(text: str) -> bool | None:
    """True/False when this sentence is about quiet mode, else None.

    OFF FIRST, for the same reason as the demo mode above: "quiet mode off"
    contains "quiet mode".
    """
    text = text or ""
    try:
        if _QUIET_OFF.search(text):
            return False
        if _QUIET_ON.search(text):
            return True
    except Exception:                           # noqa: BLE001
        return None
    return None


def asked_for(text: str) -> str | None:
    """Which mode this sentence is asking for, or None. Never raises.

    OFF IS CHECKED FIRST: "demo mode off" contains "demo mode", and with the
    on-pattern first every attempt to leave a demo would start another one.
    """
    text = text or ""
    try:
        if _PROFESSIONAL_OFF.search(text):
            return COMPANION
        if _PROFESSIONAL_ON.search(text):
            return PROFESSIONAL
    except Exception:                           # noqa: BLE001
        return None
    return None


# --- the automatic one ------------------------------------------------------

# Formal or technical phrasing. Whole phrases, not single words: "how does
# your obstacle avoidance work" is a demo question, "how does that work" over
# breakfast is not, and the difference is the noun.
_FORMAL = re.compile(
    r"\b(could you (please )?(explain|describe|demonstrate|walk)"
    r"|would you (please )?(explain|describe|demonstrate)"
    r"|please (explain|describe|demonstrate|elaborate)"
    r"|walk (me|us) through|talk (me|us) through"
    r"|explain (how|what|why|the|your)"
    r"|(what|which) (sensors?|hardware|model|models|components?|"
    r"architecture|framework|algorithm|library|libraries)"
    r"|how (do|does) (you|your) \w+ (work|operate|function)"
    r"|technical (details?|specification|overview|explanation)"
    r"|specifications?\b|architecture\b|latency\b|throughput\b"
    r"|for the record|on the record|for our (guests?|visitors?))\b",
    re.IGNORECASE)

# "Explain yourself" — what a stranger asks a robot they have just met.
_EXPLAIN_YOURSELF = re.compile(
    r"\b(what (are|is) (you|this|that)\b"
    r"|who (are|made|built) you"
    r"|what (can|do) you do"
    r"|what'?s (this|that) robot"
    r"|how (do|does) (you|it) work"
    r"|tell (me|us) about (yourself|you|this robot)"
    r"|introduce yourself)\b", re.IGNORECASE)


def _in_conversation(seconds: float) -> bool:
    """Has her own voice been in the room lately? Never raises."""
    try:
        import brain                            # noqa: PLC0415
        return brain.spoke_within(seconds)
    except Exception:                           # noqa: BLE001
        return False


def automatic(text: str) -> tuple[bool, str]:
    """Should THIS exchange be professional anyway? (yes/no, why)."""
    if not getattr(config, "MODE_AUTO", True):
        return False, ""
    text = text or ""
    if _FORMAL.search(text):
        return True, "formal or technical phrasing"
    if _EXPLAIN_YOURSELF.search(text):
        quiet = float(getattr(config, "MODE_STRANGER_QUIET_S", 600.0))
        if not _in_conversation(quiet):
            # ⚠ A PROXY FOR A STRANGER, NOT A STRANGER. See the header.
            return True, "asked to explain herself, out of the blue"
    return False, ""


def for_turn(text: str) -> tuple[str, str]:
    """The mode for this one turn, and the reason. Never raises.

    The sticky mode wins when it is already professional -- he turned it on
    and it stays on until he turns it off. Otherwise the automatic test may
    lift THIS exchange into professional without touching what is stored.
    """
    sticky = current()
    if sticky == PROFESSIONAL:
        return PROFESSIONAL, "professional mode is on"
    auto, why = automatic(text)
    return (PROFESSIONAL, why) if auto else (COMPANION, "")


# --- what the model is told -------------------------------------------------

_PROFESSIONAL_VOICE = (
    "⚠ ANSWER THIS TURN IN PROFESSIONAL MODE. Crisp, precise, confident, "
    "minimal filler: no \"umm\", no \"you know\", no Hinglish, no pet names, "
    "no jokes you were not asked for, and nothing that sounds like a friend "
    "on the sofa. Lead with the answer, give the number or the name when you "
    "have one, and stop. Warmth is not rudeness -- be courteous -- but this "
    "is a demonstration or a technical question and it should sound like one. "
    "Keep your face restrained: no `love`, no `wink`; `neutral`, `success`, "
    "`proud` and `curious` are the ones that fit. Set the reply object's "
    "`mode` to \"professional\"."
)


def instructions(name: str) -> str:
    """The paragraph that rides on this turn's question, or ""."""
    return _PROFESSIONAL_VOICE if name == PROFESSIONAL else ""


# --- what the face may show -------------------------------------------------

# The spec: "'love' and 'wink' should essentially not fire in this mode".
# ⚠ AND NOTHING ELSE IS SUPPRESSED, deliberately. `dizzy` is how a failed
# tool looks from across the room and `confused` is how a misheard command
# looks; a mode that hid those would make her harder to read exactly when
# somebody is watching her work.
RESTRAINED = {"love": "neutral", "wink": "neutral"}
PROFESSIONAL_FIT = ("neutral", "success", "proud", "curious")


def restrain(emotion: str | None, name: str) -> str | None:
    """The face this mode allows. Companion mode changes nothing."""
    if not emotion or name != PROFESSIONAL:
        return emotion
    swapped = RESTRAINED.get(emotion.strip().lower())
    if swapped:
        log.info("professional mode: %s -> %s", emotion, swapped)
    return swapped or emotion


def main() -> int:
    """Read it, or set it: ./venv/bin/python mode.py [companion|professional]"""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if len(sys.argv) > 1:
        print(set_mode(sys.argv[1], why="the command line"))
        return 0
    data = state()
    when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(data["since"]))
            if data.get("since") else "never")
    print(f"{data['mode']}  (set {when}: {data.get('why', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
