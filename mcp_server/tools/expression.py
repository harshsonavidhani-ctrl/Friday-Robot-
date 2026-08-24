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
    """Show what you feel about what is being said.

    The face is how Harsh reads you before a word is spoken. Use it when there
    is something real to show, and let it pass when there is not — a face that
    reacts to everything reads as noise, not as feeling.

    YOU DO NOT NEED TO SET listening, thinking OR speaking. Those follow what
    you are doing and are handled for you; setting them by hand only fights
    the pipeline. Choose from the FEELINGS:

        happy      a greeting, or good news
        love       thanks, or affection
        proud      you did something well
        excited    something genuinely exciting
        curious    an interesting question
        surprised  something unexpected
        sad        bad news
        wink       you are joking
        angry      only if Harsh is being very rude, and keep it mild
        neutral    nothing in particular to show

    Match the strength to the moment: mild for small things, strong for big
    ones. The expression fades back to neutral on its own after a few seconds,
    so there is no need to reset it.

    Safety expressions always win. If Friday is about to hit something, the
    warning face stays up whatever you ask for here — asking for "scared"
    yourself is treated as a feeling, not as a sensor reading.

    Args:
        emotion: one of the feelings above. Also accepts "yawn", which plays
            once and returns to the current mood.
    """
    name = (emotion or "").strip().lower()
    if name == face_control.YAWN_NAME:
        face_control.trigger_yawn()
        return "yawned"
    if name not in face_control.EMOTIONS:
        return (f"unknown expression {name!r}. Valid: "
                f"{', '.join(face_control.EMOTIONS)}")

    # Through the director, not straight to the file: it arbitrates against
    # what the pipeline and the safety system are showing, so a cheerful face
    # can never paint over an obstacle warning.
    import face_director
    face_director.get().emotion(name)
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
