"""stt.py — speech to text, locally, with VAD endpointing.

faster-whisper (config.STT_MODEL / STT_COMPUTE_TYPE) transcribes; a VAD
decides when the utterance ended. No fixed record window — Friday must not
cut Harsh off mid-sentence, and must not sit recording silence either.

Not built yet — Step 1 fills this in.
"""

from __future__ import annotations
