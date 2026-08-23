"""wake_word.py — the trigger that starts a turn.

One interface, two implementations, so Step 2 can swap the real one in
without touching main.py:

    for _ in wake_word.listen():      # yields once per detection
        run_a_turn()

Step 1 ships the stand-in (press Enter, or the word "friday" in a short
transcript). Step 2 replaces it with openWakeWord running a custom-trained
friday.onnx continuously on the mic.

The distinction matters: a record -> transcribe -> check loop is deaf for as
long as it is transcribing, so it drops roughly every other wake. That is why
config points at a real wake-word model rather than reusing the STT path.

Not built yet — Step 1 fills this in.
"""

from __future__ import annotations

from typing import Iterator


def listen() -> Iterator[str]:
    """Yield once per wake, forever. The yielded string names the trigger."""
    raise NotImplementedError("wake_word.listen() — Step 1")
