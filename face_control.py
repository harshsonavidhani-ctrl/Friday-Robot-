"""face_control.py — tell Friday's face what to feel.

The face is a separate process. It watches a single text file; anything that
can write a file can drive it. No sockets, no server, no imports of pygame.

    import face_control
    face_control.set_emotion("happy")
    face_control.get_emotion()          -> "happy"

    python3 face_control.py happy       set it
    python3 face_control.py             print the current emotion
    python3 face_control.py --list      print every valid name
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
EMOTION_FILE = STATE_DIR / "emotion.txt"

DEFAULT_EMOTION = "neutral"

# The canonical list. friday_face.py imports this and refuses to start unless
# it has geometry for every name here, so the two can never drift apart.
EMOTIONS: tuple[str, ...] = (
    # core
    "neutral", "happy", "sad", "angry", "surprised",
    # functional — what Friday is doing
    "listening", "thinking", "speaking", "confused", "dizzy", "success",
    # personality
    "excited", "wink", "love", "curious", "proud", "bored",
    # low energy
    "sleepy", "sleeping", "scared",
)


def set_emotion(name: str) -> str:
    """Write the emotion atomically. Returns the name that was written."""
    name = name.strip().lower()
    if name not in EMOTIONS:
        raise ValueError(
            f"unknown emotion {name!r}. Valid: {', '.join(EMOTIONS)}"
        )

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Temp file in the SAME directory so the rename stays on one filesystem
    # and is therefore atomic. The face can never read a half-written name.
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=".emotion.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(name + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, EMOTION_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return name


def get_emotion() -> str:
    """Current emotion, or the default if the file is missing or garbage."""
    try:
        name = EMOTION_FILE.read_text().strip().lower()
    except (OSError, UnicodeDecodeError):
        return DEFAULT_EMOTION
    return name if name in EMOTIONS else DEFAULT_EMOTION


def main(argv: list[str]) -> int:
    if not argv:
        print(get_emotion())
        return 0
    if argv[0] in ("--list", "-l"):
        print("\n".join(EMOTIONS))
        return 0
    try:
        print(set_emotion(argv[0]))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
