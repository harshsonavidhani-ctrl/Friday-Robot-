"""Friday says out loud what she can see.

    say_what_you_see()            look, describe it, speak it
    say_what_you_see(speak=False) same words, not spoken

One tool, because "what do you see" is one thing a person asks for, and
making them call look() and then speak() is making them do the wiring.

HOW THE WORDS ARE MADE. The camera image goes to Haiku -- the cheap model,
per CLAUDE.md -- along with a one-line summary of what the range sensors say,
because a picture cannot tell you that the thing in it is 40 cm away and the
sensors can. The two together are what makes the sentence useful rather than
decorative: "your desk is right in front of me, about half a metre" is worth
saying, "I see a desk" is not.

IT STILL WORKS WITH NO INTERNET AND NO KEY. If the API is unreachable, over
the daily spend cap, or the cameras are unplugged, it falls back to describing
the room purely from the range sensors -- shorter and blunter, but true, and
spoken the same way. Friday going quiet because the WiFi dropped would be
worse than Friday saying something plain.

Speaking happens through tts.speak, which is the same path as every other
thing she says, so it queues behind whatever she is already saying rather than
talking over herself.
"""

from __future__ import annotations

import io
import logging

import config
import tts
import vision
from mcp_server import embodiment, hardware
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.narrate")

# Two sentences, out loud. Long enough to be worth hearing, short enough that
# nobody is standing there waiting for her to finish.
MAX_WORDS = 45


def _distance_phrase(mm: int | None) -> str:
    """Millimetres as something a person would actually say."""
    if mm is None:
        return "an unknown distance"
    if mm >= config.TOF_NO_RETURN_MM:
        return "further than I can measure"
    if mm < 300:
        return f"{mm} millimetres"
    return f"{mm / 1000:.1f} metres".replace("0.", "0.")


def _sensor_summary() -> tuple[str, dict]:
    """One line of what the range sensors can see, and the raw clearances.

    Directions that nothing can measure are said to be unmeasured rather than
    quietly dropped -- the same rule the rest of the perception tools follow.
    An unknown direction is not an open one.
    """
    ranges = hardware.ranges()
    clearance = ranges.clearance_mm()
    parts = []
    for key, said in (("forward", "ahead"), ("left", "to my left"),
                      ("right", "to my right"), ("backward", "behind me")):
        mm = clearance.get(key)
        if mm is None:
            continue                      # unmeasured: say nothing, not "open"
        parts.append(f"{said} {_distance_phrase(mm)}")
    if not parts:
        summary = "the range sensors cannot see in any direction right now"
    else:
        summary = "nearest surface " + ", ".join(parts)
    return summary, clearance


def _offline_sentence(summary: str, clearance: dict) -> str:
    """What to say when there is no model to ask. Blunt, but true."""
    forward = clearance.get("forward")
    if forward is None:
        return ("I can't see what's ahead of me at the moment, so I'd rather "
                "not guess.")
    if forward < config.SAFETY_STOP_MM:
        return (f"Something is right in front of me, about "
                f"{_distance_phrase(forward)} away. I'm not going anywhere.")
    return (f"I can't describe what I'm looking at without my connection, "
            f"but the way ahead is clear for about {_distance_phrase(forward)}.")


def _describe_with_haiku(image, summary: str) -> str | None:
    """Ask the cheap model for two sentences. None if it cannot be asked.

    Everything here is deliberately small: one image, one short instruction,
    a low token ceiling. The cost is recorded in the same spend log the brain
    writes, so `dev_status` counts it and the daily cap covers it.
    """
    if not config.ANTHROPIC_API_KEY:
        return None
    import brain                                        # noqa: PLC0415
    if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
        log.warning("over the daily spend cap; describing from sensors only")
        return None
    # AND A SPEED LIMIT, NOT JUST A CEILING (file 08 step 10). The cap above is
    # the end of the day's money; this is what stops one loop reaching it in a
    # minute. "Let me look again" is a reasonable-sounding thought for a model
    # to have twice a second, and nothing here used to disagree. Falling back
    # to the sensor sentence is not a failure -- _offline_sentence still
    # describes the room, from real clearances, for nothing.
    import rate_limit                                   # noqa: PLC0415
    allowed, why = rate_limit.allow("vision_describe")
    if not allowed:
        log.info("describing from sensors only — %s", why)
        return None

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=config.CAPTURE_QUALITY)

    import anthropic                                    # noqa: PLC0415
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        reply = client.messages.create(
            model=config.MODEL_FAST,
            max_tokens=160,
            system=(
                "You are Friday, a small robot, describing what you can see "
                "to the person in front of you. Speak in the first person, "
                f"in at most {MAX_WORDS} words and at most two sentences. "
                "Say what is actually in the picture and roughly where it is "
                "relative to you. Use the sensor line for distances -- the "
                "camera cannot judge those. Be plain and specific; no "
                "preamble, no 'I see an image of'. This will be read aloud, "
                "so write it the way it should sound."),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": __import__("base64").b64encode(
                        buffer.getvalue()).decode()}},
                {"type": "text", "text": f"Range sensors: {summary}."},
            ]}],
        )
    except Exception as exc:                            # noqa: BLE001
        # Any API failure is a reason to fall back, never a reason to fail:
        # this tool's job is that Friday says something.
        log.warning("could not describe the view (%s: %s)",
                    type(exc).__name__, exc)
        return None

    try:
        brain._record_spend(config.MODEL_FAST, reply.usage,
                            brain._cost_usd(config.MODEL_FAST, reply.usage))
        rate_limit.used("vision_describe")
    except Exception:                                   # noqa: BLE001
        log.warning("could not record the cost of the description")

    text = "".join(block.text for block in reply.content
                   if getattr(block, "type", "") == "text").strip()
    return text or None


@tool
def say_what_you_see(camera: str = "both", speak: bool = True) -> str:
    """Look, describe what is there, and say it out loud.

    Use this when Harsh asks what you can see, what is in front of you, or
    what the room looks like. It takes a picture, reads the range sensors for
    real distances, and speaks one or two sentences.

    It never stays silent: with no internet, no key, or no camera it
    describes the room from the range sensors instead.

    Args:
        camera: "both" (default), "up" for the room, "down" for the floor.
        speak: False to return the words without saying them.
    """
    summary, clearance = _sensor_summary()

    image = None
    camera_note = ""
    try:
        image, camera_note = vision.look(camera)
    except Exception as exc:                            # noqa: BLE001
        camera_note = f"no camera ({type(exc).__name__}: {exc})"
        log.warning("cameras unavailable: %s", camera_note)

    sentence = None
    if image is not None:
        sentence = _describe_with_haiku(image, summary)
    if not sentence:
        sentence = _offline_sentence(summary, clearance)

    # Looking counts as looking. This is a real picture at a known moment
    # plus real clearances, which is exactly what the movement gate wants --
    # so asking Friday what she sees also earns her the right to move.
    embodiment.record(clearance, source="say_what_you_see",
                      saw_image=image is not None)

    if speak:
        try:
            tts.speak(sentence)
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not speak the description: %s", exc)
            return f"{sentence}\n\n(could not speak it: {exc})"

    return f"{sentence}\n\n[{camera_note}; {summary}]"
