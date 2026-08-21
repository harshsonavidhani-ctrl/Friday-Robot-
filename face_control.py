"""face_control.py — tell Friday's face what to feel.

The face is a separate process. It watches a single text file; anything that
can write a file can drive it. No sockets, no server, no imports of pygame.

    import face_control
    face_control.set_emotion("happy")
    face_control.get_emotion()          -> "happy"
    face_control.trigger_yawn()

    python3 face_control.py happy       set it
    python3 face_control.py             print the current emotion
    python3 face_control.py --list      print every valid name
    python3 face_control.py --yawn      play the yawn overlay once
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
EMOTION_FILE = STATE_DIR / "emotion.txt"
# The yawn is a transient overlay, not an emotion, so it gets its own file.
# Each trigger writes a fresh token; the face plays the animation once per
# token it hasn't seen, then returns to whatever emotion was already active.
YAWN_FILE = STATE_DIR / "yawn.txt"

DEFAULT_EMOTION = "neutral"

# The canonical list. friday_face.py imports this and refuses to start unless
# it has geometry for every name here, so the two can never drift apart.
EMOTIONS: tuple[str, ...] = (
    # neutral, then thinking right after -- this order is the demo/check order
    "neutral", "thinking",
    # core
    "happy", "sad", "angry", "surprised",
    # functional — what Friday is doing
    "listening", "speaking", "confused", "dizzy", "success",
    # personality
    "excited", "wink", "love", "curious", "proud", "bored",
    # low energy
    "sleepy", "sleeping", "scared",
)


def _write_atomic(path: Path, text: str, prefix: str) -> None:
    """Temp file in the SAME directory, then rename. The rename is atomic
    on one filesystem, so a reader never sees a half-written file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def set_emotion(name: str) -> str:
    """Write the emotion atomically. Returns the name that was written."""
    name = name.strip().lower()
    if name not in EMOTIONS:
        raise ValueError(
            f"unknown emotion {name!r}. Valid: {', '.join(EMOTIONS)}"
        )
    _write_atomic(EMOTION_FILE, name + "\n", ".emotion.")
    return name


def trigger_yawn() -> None:
    """Ask the face to play the yawn overlay once, on top of whatever
    emotion is currently showing, then return to it. Fire-and-forget."""
    _write_atomic(YAWN_FILE, f"{time.monotonic_ns()}\n", ".yawn.")


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
    if argv[0] in ("--yawn", "-y"):
        trigger_yawn()
        return 0
    try:
        print(set_emotion(argv[0]))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
