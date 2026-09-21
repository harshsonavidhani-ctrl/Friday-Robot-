"""initiative.py — File 12 step 2: may she speak up right now, and how?

    channel, why = initiative.decide("scene remark", have_something=True)
    if channel == "voice":  ...say it
    elif channel == "face": ...show it instead
    else:                   ...stay out of the way

Harsh's spec, in three rules:
  * "Never let Friday speak or move unsolicited while the VAD/wake-word
    pipeline shows the user is actively speaking."
  * "Proactive engagement only fires when BOTH are true: an idle timeout has
    passed, AND there is something worth surfacing... Never initiate just
    because idle time passed with nothing to say."
  * "Choose the response channel based on context: default to voice + face;
    if ambient noise is high or an explicit quiet mode is set, prefer a
    face/display change only, without speaking."

ONE GATE, NOT A RULE PER CALLER. Three things in this robot already act
without being asked — the scene remark, the idle look-up and the idle drive
— and each grew its own set of conditions (cooldowns, audience checks, quiet
windows). Those stay: they answer "is this worth saying at all", which is
their own business. This answers the one question they were all answering
separately and differently: "is now a moment when she may interrupt, and out
loud or not."

WHAT IT READS, all of it a file some other process already writes:
  * state/listening.json  — the spine marks the moment a capture starts and
    clears it when the turn ends. THE POINT OF THE WHOLE STEP: a remark
    landing on top of him mid-sentence is the rudest thing she does.
  * state/speaking.json   — her own voice (tts.speak). Speaking over herself
    is the second rudest.
  * state/wake_alive.json — the detector's heartbeat, for how loud the room
    is right now.
  * state/mode.json       — quiet mode, set by voice or from the dashboard.

Never raises. A gate that throws is a robot that goes silent for good.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import config

log = logging.getLogger("friday.initiative")

VOICE = "voice"
FACE = "face"


def _read(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


# --- is he talking? ---------------------------------------------------------

def listening_file() -> Path:
    return Path(getattr(config, "LISTENING_FILE",
                        config.ROOT / "state" / "listening.json"))


def mark_listening(on: bool, detail: str = "") -> None:
    """The spine says a capture has started (or ended). Never raises."""
    try:
        path = listening_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"listening": bool(on), "at": time.time(),
                                    "detail": detail[:80]}))
    except OSError as exc:
        log.debug("could not mark listening: %s", exc)


def he_is_talking() -> bool:
    """Is a capture running right now?

    ⚠ WITH A STALENESS LIMIT. A turn that died between the mark and the
    clear would otherwise silence every proactive path forever, and "she
    never speaks up again after one crash" is exactly the kind of failure
    this project keeps finding weeks later. LISTENING_STALE_S is the longest
    a capture can honestly take.
    """
    mark = _read(listening_file())
    if not mark.get("listening"):
        return False
    age = time.time() - float(mark.get("at") or 0)
    return 0 <= age <= float(getattr(config, "LISTENING_STALE_S", 60.0))


def she_is_talking() -> bool:
    """Is her own voice in the room? Never raises."""
    try:
        import brain                                     # noqa: PLC0415
        return brain.spoke_within(
            float(getattr(config, "INITIATIVE_AFTER_HER_VOICE_S", 3.0)))
    except Exception:                                    # noqa: BLE001
        return False


# --- how loud is it, and does he want quiet? --------------------------------

def room_rms() -> float | None:
    """The detector's last loudness reading, or None if it is not fresh."""
    beat = _read(Path(getattr(config, "WAKE_HEARTBEAT_FILE",
                              config.ROOT / "state" / "wake_alive.json")))
    if not beat:
        return None
    age = time.time() - float(beat.get("at") or 0)
    if age > float(getattr(config, "WAKE_HEARTBEAT_STALE_S", 120.0)):
        return None                 # the detector is not running; do not guess
    value = beat.get("peak_rms") or beat.get("rms")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_loud() -> bool:
    """Is the room loud enough that speaking up would be shouting into it?"""
    rms = room_rms()
    if rms is None:
        return False                # unknown is not loud
    return rms >= float(getattr(config, "INITIATIVE_LOUD_RMS", 6000.0))


def quiet_mode() -> bool:
    """Has he asked her to keep it to the screen? Never raises."""
    try:
        import mode                                      # noqa: PLC0415
        return bool(mode.quiet())
    except Exception:                                    # noqa: BLE001
        return False


# --- the decision -----------------------------------------------------------

def decide(kind: str, have_something: bool,
           idle_for: float | None = None) -> tuple[str | None, str]:
    """May she start something now, and on which channel?

    Returns (VOICE | FACE | None, why). The `why` is for the log — every
    refusal here is a thing she did NOT do, which is invisible otherwise.

    `have_something` is the caller's own answer to "is this worth saying" —
    a curiosity finding, a due card, a reflection note. False means silence,
    and this is the rule the spec is most explicit about: "Never initiate
    just because idle time passed with nothing to say."
    """
    if not getattr(config, "INITIATIVE", True):
        return None, "proactive engagement is switched off"
    if not have_something:
        return None, "nothing worth surfacing"
    if he_is_talking():
        return None, "he is talking"
    if she_is_talking():
        return None, "she is still speaking"
    wait = float(getattr(config, "INITIATIVE_IDLE_S", 45.0))
    if idle_for is not None and idle_for < wait:
        return None, f"only {idle_for:.0f}s idle, wants {wait:.0f}s"
    if quiet_mode():
        return FACE, "quiet mode: the face only"
    if is_loud():
        return FACE, f"the room is loud ({room_rms():.0f} rms): the face only"
    return VOICE, kind


def main() -> int:
    """What she would be allowed to do this second, and why."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for kind, something in (("with something to say", True),
                            ("with nothing to say", False)):
        channel, why = decide(kind, something)
        print(f"  {str(channel):6s}  {why}")
    print(f"\n  he is talking : {he_is_talking()}")
    print(f"  she is talking: {she_is_talking()}")
    print(f"  room rms      : {room_rms()}")
    print(f"  quiet mode    : {quiet_mode()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
