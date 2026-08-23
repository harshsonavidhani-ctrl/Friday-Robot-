"""Expression tools: Friday's face and Friday's voice.

The face runs as its own process (friday-face.service) watching a state file,
so setting an emotion is a file write — it cannot block, and it works even if
the face process is restarting.
"""

from __future__ import annotations

import config
import face_control
import tts
from mcp_server.registry import tool


@tool
def set_face(emotion: str) -> str:
    """Change the expression on Friday's screen.

    The face is how Harsh reads Friday before a word is spoken, so keep it
    honest: listening while he talks, thinking while you work, speaking while
    you talk, neutral when idle. Use the expressive ones when they are true,
    not as decoration.

    Args:
        emotion: one of neutral, thinking, happy, sad, angry, surprised,
            listening, speaking, confused, dizzy, success, excited, wink,
            love, curious, proud, bored, sleepy, sleeping, scared. Also
            accepts "yawn", which plays once and returns to the current mood.
    """
    try:
        name = face_control.set_emotion(emotion)
    except ValueError as exc:
        return str(exc)
    return f"face set to {name}"


@tool
def get_face() -> str:
    """What expression Friday is showing right now."""
    return face_control.get_emotion()


@tool
def speak(text: str) -> str:
    """Say something out loud through Friday's speaker.

    Blocks until Friday has finished speaking, so the reply after this lands
    once the words are actually out. Keep it to one or two sentences — this is
    heard, not read.

    Args:
        text: what to say.
    """
    result = tts.speak(text)
    if config.DEBUG_SILENT_SPEECH:
        return (f"{result}  (debug silence is ON in config — Friday is not "
                f"audible right now)")
    return result
