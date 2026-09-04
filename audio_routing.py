"""audio_routing.py — find Friday's speaker and microphone as they are RIGHT NOW.

WHY THIS EXISTS. `config.SPEAKER_DEVICE` and `config.MIC_DEVICE` are pipewire
node names, and a node name encodes the CARD PROFILE, not just the card:

    alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo   <- config
    alsa_output.usb-GeneralPlus_USB_Audio_Device-00.pro-output-0    <- reality

Same speaker. Different profile, so a different node name, so every hard-coded
target misses. And on this Pi that is not an edge case: the USB devices drop
off the bus and re-enumerate on their own (see the USB notes in CLAUDE.md and
`friday-usb-drops-devices`), and the profile does not always come back the same
way.

WHAT THAT COST, MEASURED 2026-09-04 just before midnight
=======================================================
Harsh: *"shes mute"*. `wpctl status` showed the USB card present as a DEVICE
with no sink and no source node at all, the default sink fallen back to HDMI,
and:

    friday.tts | playback failed: Command '['pw-play',
      '--target=alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo',
      ...]' timed out after 20 seconds

pw-play does not fail fast on a target that does not exist -- it waits, and
Friday is mute for twenty seconds per sentence with an error nobody is reading.

WORSE, ON THE INPUT SIDE, BECAUSE IT LOOKED LIKE IT WAS WORKING. pw-cat with a
missing `--target` silently falls back to the DEFAULT source, so the wake word
had been listening through the WEBCAM's mono microphone rather than Friday's
USB shotgun. Audio arrived, the detector scored, the heartbeat was healthy --
and every score was computed on the wrong microphone. A hard-coded name that
misses is not a loud failure here; it is a quiet one.

THE RULE. Ask what nodes exist, match on the part of the name that identifies
the CARD, and let the profile suffix be whatever it is today. If nothing
matches, return None so the caller omits `--target` entirely and gets the
system default -- degraded, but audible, and it says so.
"""

from __future__ import annotations

import json
import logging
import subprocess

log = logging.getLogger("friday.audio")

# How long to wait for pw-dump. It is a local socket query; if it has not
# answered in this long, pipewire itself is in trouble and the caller is better
# off with the default device than blocked here.
DUMP_TIMEOUT_S = 4.0

# The part of a node name that identifies the CARD rather than the profile.
# Everything before the profile suffix is stable across re-enumeration; the
# suffix (.analog-stereo, .pro-output-0, .mono-fallback) is not.
def card_key(node_name: str) -> str:
    """`alsa_output.usb-Foo_Bar-00.analog-stereo` -> `usb-Foo_Bar-00`."""
    if not node_name:
        return ""
    body = node_name.split(".", 1)[-1]          # drop alsa_output/alsa_input
    return body.rsplit(".", 1)[0] if "." in body else body


def _nodes() -> list[tuple[str, str]]:
    """[(media.class, node.name)] for every audio node pipewire currently has."""
    try:
        raw = subprocess.run(["pw-dump"], capture_output=True, text=True,
                             timeout=DUMP_TIMEOUT_S).stdout
        dump = json.loads(raw)
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not ask pipewire what exists (%s) — falling back to "
                    "the default device", exc)
        return []
    out = []
    for obj in dump:
        props = ((obj.get("info") or {}).get("props") or {})
        media = props.get("media.class", "")
        name = props.get("node.name", "")
        if name and media.startswith("Audio/"):
            out.append((media, name))
    return out


def _resolve(preferred: str, media_class: str, what: str) -> str | None:
    """The live node name for `preferred`, or None to mean "use the default"."""
    nodes = [n for cls, n in _nodes() if cls == media_class]
    if not nodes:
        return None
    if preferred in nodes:
        return preferred
    key = card_key(preferred)
    if key:
        for name in nodes:
            if card_key(name) == key:
                # The card is there under a different profile. Say so once,
                # loudly enough to be found later, because silently using a
                # different node is how the webcam mic went unnoticed.
                log.warning("%s %r is not present; using %r on the same card "
                            "(the profile changed)", what, preferred, name)
                return name
    log.error("%s %r is NOT PRESENT and no node on that card is either. Using "
              "the system default, which may be the wrong device entirely. "
              "Present: %s", what, preferred, ", ".join(nodes) or "nothing")
    return None


# A "pro-audio" node is NOT a usable speaker, however much it looks like one.
# The pro profile exposes raw per-channel ports with no mixer and no resampler,
# and pw-play to one does not fail -- it HANGS, measured at the full 25 s
# timeout with a correctly formatted 48 kHz stereo file. Treating it as a sink
# is worse than finding none at all, because finding none sends tts.play()
# straight to ALSA, which works.
#
# This matters at BOOT specifically: which profile wireplumber picks for this
# card is not stable across reboots, and one of the choices it makes is
# pro-audio. Excluding it here is what makes her voice behave the same way on
# every boot instead of depending on that.
PRO_AUDIO_MARKER = ".pro-"


def speaker(preferred: str) -> str | None:
    node = _resolve(preferred, "Audio/Sink", "speaker")
    if node and PRO_AUDIO_MARKER in node:
        log.warning("ignoring %r: a pro-audio node cannot be played to, and "
                    "pw-play hangs on it rather than failing. Using ALSA.", node)
        return None
    return node


def microphone(preferred: str) -> str | None:
    return _resolve(preferred, "Audio/Source", "microphone")


def target_args(node: str | None) -> list[str]:
    """`--target=NODE`, or nothing at all when we want the default.

    Passing no target is what makes the fallback WORK: pw-play and pw-cat both
    use the default device when none is given, and both hang or silently pick
    something else when given one that does not exist.
    """
    return [f"--target={node}"] if node else []
