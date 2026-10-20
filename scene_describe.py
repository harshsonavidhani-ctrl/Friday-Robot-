"""scene_describe.py — File 15: the one Haiku vision call, and the frame it sees.

    jpeg, where = scene_describe.fresh_frame()          # the watcher's, or None
    text, why = scene_describe.describe(jpeg, focus="how many people")
    text, why = scene_describe.narrate(jpeg, previous="I can see a desk.",
                                       hint="new: chair (left)")
    if text == scene_describe.NO_CHANGE: ...stay quiet

TWO CALLERS, ONE CALL. describe_scene() (mcp_server/tools/describe.py, asked
for) and continuous narration (narration.py, Tier 2) both come here, so there
is one place that sends a picture to a model for File 15, one place that
checks the day's money, and one place that records what it cost.

SEPARATE FROM THE BRAIN'S TURN ON PURPOSE (spec step 1): a picture is the
most expensive thing in a request, so it rides only on the call that needs
it, never on every turn's history.

THE FRAME. The scene watcher (scene.py, inside the MCP server, which owns the
cameras) already writes the up camera's latest frame to state/scene_latest.jpg
every SCENE_PERIOD_S. Reading that file is free and opens nothing. Only when
it is stale does describe_scene take a picture itself, and then through
vision.look(), under the same capture lock as look() and observe() -- so no
second cv2.VideoCapture is ever opened on a camera.

NEVER RAISES. Every failure is (None, why): over the cap, rate limited, no
key, no internet. The caller decides what silence means.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path

import config

log = logging.getLogger("friday.scene_describe")

NO_CHANGE = "NO_CHANGE"
_NO_CHANGE_RE = re.compile(r"\bNO[_ ]?CHANGE\b", re.IGNORECASE)

# What the last call cost, for the caller's log line and the dashboard card.
last_cost_usd = 0.0
last_model = ""


def fresh_frame(max_age_s: float | None = None) -> tuple[bytes, float, list] | None:
    """The scene watcher's latest frame, if young enough: (jpeg, age_s, objects).

    None when there is no frame, the file is unreadable, or it is older than
    max_age_s -- the dashboard streaming the up camera, or an exploration run
    holding the robot, both pause the watcher and leave the file to age.
    """
    max_age_s = float(config.DESCRIBE_FRESH_S if max_age_s is None
                      else max_age_s)
    try:
        meta = json.loads(Path(config.SCENE_JSON).read_text())
        age = time.time() - float(meta["taken_at"])
        # The image is written BEFORE the JSON that dates it (scene._tick), so
        # the picture read here is never older than `age` says.
        jpeg = Path(config.SCENE_IMAGE).read_bytes()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if age > max_age_s or age < -5:
        return None
    return jpeg, age, list(meta.get("objects") or [])


def _labels(objects: list) -> str:
    named = [f"{o.get('label')} ({o.get('side')})" if o.get("side")
             else str(o.get("label"))
             for o in objects if o.get("label")]
    return ", ".join(named) or "nothing"


def _api_key() -> str | None:
    """The key, from the environment or, failing that, from .env directly.

    ⚠ friday-mcp's unit deliberately has NO EnvironmentFile: that server runs
    dev_task, which hands its whole environment to Claude Code, and a key
    there would bill Harsh's API instead of his subscription (CLAUDE.md,
    cost safety). So describe_scene reads the one line it needs, for its own
    request, and never puts it in os.environ where a child could inherit it.
    """
    if config.ANTHROPIC_API_KEY:
        return config.ANTHROPIC_API_KEY
    try:
        for line in Path(config.DESCRIBE_KEY_FILE).read_text().splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip().removeprefix("export ").strip() == "ANTHROPIC_API_KEY":
                return value.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _call(limit_name: str, jpeg: bytes, system: str, text: str,
          max_tokens: int, media_type: str = "image/jpeg") -> tuple[str | None, str]:
    """One vision request. (text, "") or (None, why). Never raises.
    `media_type`: the laptop's game pictures are PNG (laptop_games.py)."""
    global last_cost_usd, last_model
    last_cost_usd, last_model = 0.0, ""
    key = _api_key()
    if not key:
        return None, "no API key"
    try:
        import brain                                    # noqa: PLC0415
        if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return None, "over the daily spend cap"
    except Exception as exc:                            # noqa: BLE001
        log.debug("could not read today's spend: %s", exc)
        brain = None
    import rate_limit                                   # noqa: PLC0415
    allowed, why = rate_limit.allow(limit_name)
    if not allowed:
        return None, f"rate limited — {why}"

    model = config.DESCRIBE_MODEL
    try:
        import anthropic                                # noqa: PLC0415
        client = anthropic.Anthropic(api_key=key)
        reply = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            system=system,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": text},
            ]}],
        )
    except Exception as exc:                            # noqa: BLE001
        log.warning("vision call failed (%s: %s)", type(exc).__name__, exc)
        try:
            import brain as _brain                      # noqa: PLC0415
            if _brain.note_api_error(exc):
                return None, "no API credit"
        except Exception:                               # noqa: BLE001
            pass
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"

    # Counted as used only once it has actually run -- a call that failed to
    # start has not spent its budget (rate_limit's own rule).
    rate_limit.used(limit_name)
    try:
        if brain is not None:
            last_cost_usd = brain._cost_usd(model, reply.usage)
            brain._record_spend(model, reply.usage, last_cost_usd)
    except Exception:                                   # noqa: BLE001
        log.warning("could not record the cost of a vision call")
    last_model = model
    said = "".join(block.text for block in reply.content
                   if getattr(block, "type", "") == "text").strip()
    return (said or None), ("" if said else "the model said nothing")


# --- on demand (describe_scene) ---------------------------------------------

def describe(jpeg: bytes, focus: str | None = None, sensors: str = "",
             objects: list | None = None) -> tuple[str | None, str]:
    """What is in this picture -- or the answer to `focus` about it.

    The words go back to the BRAIN as a tool result, not to the speaker (spec
    step 2): the brain says them in her own voice with her own face. So this
    asks for plain facts, not for a performance.
    """
    words = int(config.DESCRIBE_MAX_WORDS)
    system = (
        "You are the eyes of Friday, a small home robot. You get one photo "
        "from her forward camera, which sits above her screen and tilts "
        "slightly up. Report what is really in it, in the first person "
        f"('I can see ...'), in at most {words} words and two sentences. Say "
        "roughly where things are (left, right, ahead). Use the range-sensor "
        "line for distances -- a photo cannot judge them -- and give none if "
        "there is no sensor line. If the picture is too dark, blurred or "
        "blocked to tell, say exactly that. Never guess at what is not "
        "visible. No preamble, no 'the image shows'.")
    lines = []
    if focus and focus.strip():
        lines.append(
            f"Answer THIS question about the view, directly and first: "
            f"{focus.strip()!r}. Mention other things only if they help "
            f"answer it. If the picture cannot answer it (text too small to "
            f"read, the thing is out of view), say what is missing.")
    else:
        lines.append("Describe the view.")
    if sensors:
        lines.append(f"Range sensors: {sensors}.")
    if objects:
        lines.append(f"An object detector named: {_labels(objects)} -- it is "
                     f"often wrong, so trust the picture.")
    return _call("describe_scene", jpeg, system, "\n".join(lines),
                 config.DESCRIBE_MAX_TOKENS)


# --- continuous (narration Tier 2) ------------------------------------------

def narrate(jpeg: bytes, previous: str | None = None,
            hint: str = "") -> tuple[str | None, str]:
    """ONE call that both compares and speaks (spec step 4).

    With `previous` -- what she said last time -- the answer is NO_CHANGE or
    one sentence about only what is new. Without it (the first look after he
    turned narration on) it is a short description of the whole view. The
    words are spoken AS THEY ARE, so they are asked for in her voice.
    """
    system = (
        "You are Friday, a small home robot, telling the person with you what "
        "your forward camera sees. They asked you to keep describing it as it "
        "changes. Your reply is spoken aloud exactly as written: first "
        "person, warm and plain, no preamble, no lists, no numbers for "
        "distance (the camera cannot judge them), never 'the image'.")
    if previous:
        text = (
            f"What you said last time: \"{previous}\"\n"
            f"A cheap change detector flagged: {hint or 'something changed'} "
            f"(it can be wrong).\n"
            f"Compare this new view with what you said. If nothing "
            f"meaningfully different is visible -- the same things, a shift in "
            f"light or colour, camera noise, a small wobble of the view -- "
            f"reply with exactly {NO_CHANGE} and nothing else. Otherwise reply "
            f"with ONE short sentence, at most {config.NARRATION_MAX_WORDS} "
            f"words, about ONLY what is new or has changed. Do not repeat "
            f"anything you already said.")
    else:
        text = (
            f"This is your first look since they asked. Describe the view in "
            f"one or two short sentences, at most "
            f"{config.NARRATION_FIRST_MAX_WORDS} words.")
    said, why = _call("vision_narrate", jpeg, system, text,
                      config.NARRATION_MAX_TOKENS)
    if said is None:
        return None, why
    # "NO_CHANGE." / "NO_CHANGE - the room is the same" / "no change": all
    # the same answer. Only a PREVIOUS makes it one -- the first look must
    # always say something.
    if _NO_CHANGE_RE.search(said):
        if previous:
            return NO_CHANGE, ""
        return None, "answered NO_CHANGE to a first look"   # never spoken
    return said.strip().strip('"'), ""
