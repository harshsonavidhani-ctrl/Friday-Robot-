"""tts.py — Friday's voice: Piper through the USB soundcard.

Speaks sentence by sentence, starting on the FIRST one rather than waiting
for the whole reply — on a Pi that is the difference between a reply that
feels instant and one that feels broken.

Honours config.DEBUG_SILENT_SPEECH: when that is on, every speak() is logged
and swallowed, so the robot can be tested without talking constantly.

Not built yet — Step 1 fills this in.
"""

from __future__ import annotations
