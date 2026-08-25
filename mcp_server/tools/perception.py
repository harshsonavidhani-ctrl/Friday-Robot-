"""Perception tools: the cameras and the range sensors.

scan_environment is the one the brain should reach for before moving. It
returns STRUCTURED data — booleans and numbers — rather than prose, because
"the way ahead looks fairly clear" is not something a decision can be made
from, and because a model is much less likely to talk itself past
`"forward": false` than past a sentence.

Unknown is a first-class answer here. Where a sensor cannot see, these tools
say so instead of substituting a large number. See sensors.py.
"""

from __future__ import annotations

import io
import json

from mcp.server.mcpserver import Image

import config
import vision
from mcp_server import embodiment, hardware
from mcp_server.registry import ToolResult, tool


def _safe(clearance: int | None) -> bool | None:
    """Is this direction passable? None means nothing could see it."""
    if clearance is None:
        return None
    return clearance >= config.SAFETY_STOP_MM


@tool
def scan_environment() -> str:
    """Check what is around Friday. CALL THIS BEFORE ANY MOVE.

    Fuses the 360° LiDAR with the forward 8x8 depth sensor and returns JSON:

      safe_directions   forward/backward/left/right — true, false, or null.
                        NULL MEANS NOT MEASURABLE, not "probably fine". Do not
                        drive a direction that reports null.
      forward_blocked_by  "clear", or a short reason.
      clearance_mm      nearest obstacle per direction, in millimetres; null
                        where no sensor covers that direction.
      sensors           which sensors actually answered.

    If the LiDAR is missing or its bearings are uncalibrated, the sideways and
    backward directions come back null while forward still works — the depth
    sensor on the bumper covers forward on its own.
    """
    ranges = hardware.ranges()
    clearance = ranges.clearance_mm()
    forward = clearance.get("forward")

    if forward is None:
        blocked_by = "unknown — no sensor can see ahead"
    elif forward < config.SAFETY_STOP_MM:
        blocked_by = f"obstacle {forward} mm ahead"
    elif forward < config.SAFETY_CAUTION_MM:
        blocked_by = f"clear but tight — {forward} mm ahead"
    else:
        blocked_by = "clear"

    report = {
        "safe_directions": {
            "forward": _safe(forward),
            "backward": _safe(clearance.get("backward")),
            "left": _safe(clearance.get("left")),
            "right": _safe(clearance.get("right")),
        },
        "forward_blocked_by": blocked_by,
        "clearance_mm": {
            key: clearance.get(key) for key in (
                "forward", "forward_left", "forward_right", "left", "right",
                "backward", "backward_left", "backward_right", "bumper",
                "nearest_any")
        },
        "stop_threshold_mm": config.SAFETY_STOP_MM,
        "sensors": ranges.status(),
    }
    # Counts as having looked -- numbers without a picture, but a real reading
    # of the world at a known moment, which is what the movement gate needs.
    embodiment.record(clearance, source="scan_environment", saw_image=False)
    return json.dumps(report, indent=2)


@tool
def get_distance() -> str:
    """How far away is the nearest thing directly in front of Friday.

    A quick single number in centimetres, for when you only need "is there
    something there". Use scan_environment before actually moving.
    """
    distance, source = hardware.ranges().forward_clearance_mm()
    if distance is None:
        return (f"cannot tell — {source}. Treat forward as unknown, "
                f"not as clear.")
    return f"{distance / 10:.0f} cm ahead ({distance} mm, from the {source})"


@tool
def look(camera: str = "both") -> str:
    """Look through Friday's cameras and see the result.

    By default returns ONE image with both views stacked: the up camera on
    top (the room, at eye level) and the down camera underneath (the floor
    immediately ahead). One image, the whole picture — what is around Friday
    and what is underfoot.

    If a camera is unplugged its half is replaced by a "no signal" panel, so
    a missing view is visible rather than silently absent.

    Args:
        camera: "both" (default), "up" for the room only, or "down" for the
            floor only.
    """
    image, note = vision.look(camera)
    # Encoded straight from memory rather than saved: capture_image() is the
    # tool that writes to disk, and doing both here would double the shutter
    # time for no gain.
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=config.CAPTURE_QUALITY)
    return ToolResult(Image(data=buffer.getvalue(), format="jpeg"),
                      f"look({camera}): {note}")


@tool
def observe(camera: str = "both") -> str:
    """Look and measure at the same moment. USE THIS BEFORE YOU MOVE.

    One atomic observation: the camera frame AND the range readings, taken
    together. That matters more than it sounds. look() and scan_environment()
    called separately can be seconds apart, and in those seconds the robot or
    the world can move — so the picture and the numbers end up describing
    different moments, and a plan built on both is built on a contradiction.

    HOW TO USE IT, and this is the whole point:

      1. observe() — see the scene and the distances together.
      2. Say what you actually see, and what you intend to do about it.
         Name the obstacle, not just the number: "there's a chair leg about
         30 cm ahead and slightly left" beats "forward clearance 310 mm".
      3. Act in ONE SMALL INCREMENT. A short move, then stop. Not the whole
         distance in one command — you cannot correct a move that has already
         finished, and a small error repeated is recoverable while a large one
         is a collision.
      4. observe() AGAIN and check the world changed the way you predicted.
         If it did not, say so and rethink rather than repeating the move.
      5. If you are unsure — the view is ambiguous, a direction reads null,
         the numbers and the picture disagree — ASK HARSH. Do not guess.
         A question costs a moment; a wrong guess costs the robot.

    A direction reported as null means NOT MEASURABLE, never "probably fine".
    Do not drive a direction that reports null.

    Args:
        camera: "both" (default, room + floor), "up" for the room, "down" for
            the floor immediately ahead — use "down" when checking for a drop
            or something low that the top-mounted LiDAR cannot see.
    """
    ranges = hardware.ranges()
    clearance = ranges.clearance_mm()
    forward = clearance.get("forward")

    reading = {
        "clearance_mm": {key: clearance.get(key) for key in (
            "forward", "forward_left", "forward_right", "left", "right",
            "backward", "bumper", "nearest_any")},
        "safe_to_move": {
            "forward": _safe(forward),
            "backward": _safe(clearance.get("backward")),
            "left": _safe(clearance.get("left")),
            "right": _safe(clearance.get("right")),
        },
        "stop_threshold_mm": config.SAFETY_STOP_MM,
        "sensors": ranges.status(),
    }

    # Recorded before the camera is touched: if the camera is dead the range
    # readings are still a genuine observation, and losing a lens must not
    # also cost Friday the right to move carefully.
    embodiment.record(clearance, source="observe", saw_image=False)

    note_parts = [json.dumps(reading, indent=2)]
    if forward is None:
        note_parts.append(
            "\nFORWARD IS UNMEASURABLE right now. Treat it as blocked, not as "
            "clear. Say so and ask rather than driving into what you cannot "
            "see.")
    elif forward < config.SAFETY_STOP_MM:
        note_parts.append(
            f"\nFORWARD IS BLOCKED at {forward} mm — inside the "
            f"{config.SAFETY_STOP_MM} mm stop distance. The server will refuse "
            f"a forward move. Look at the image and say WHAT is in the way.")

    try:
        image, camera_note = vision.look(camera)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=config.CAPTURE_QUALITY)
        # Re-recorded now the frame is actually in hand: the observation
        # that gates movement should say it SAW, not merely measured.
        embodiment.record(clearance, source="observe", saw_image=True)
        note_parts.append(f"\ncameras: {camera_note}")
        note_parts.append(
            f"\n\nThis observation is good for about "
            f"{config.OBSERVATION_TTL_S:.0f}s, less once you have moved. "
            f"Move ONE small step, then observe() again to check the world "
            f"changed the way you expected.")
        return ToolResult(Image(data=buffer.getvalue(), format="jpeg"),
                          "".join(note_parts))
    except Exception as exc:                    # noqa: BLE001
        # The ranges are the safety-critical half and they already succeeded.
        # A dead camera must not cost Friday the numbers too -- it should
        # degrade to "measuring but not seeing", and SAY that, rather than
        # failing the whole observation.
        note_parts.append(
            f"\n⚠ NO CAMERA IMAGE this time ({type(exc).__name__}: {exc}). "
            f"You are measuring but not seeing. Be correspondingly more "
            f"cautious, and say that you cannot see.")
        return "".join(note_parts)


@tool
def capture_image(camera: str = "both") -> str:
    """Save a photo from Friday's cameras to disk and return the file path.

    The same stacked dual view as look(), but written to a file — use this
    when the image needs to be kept or shown later rather than just looked at
    now. Old captures are pruned automatically.

    Args:
        camera: "both" (default), "up", or "down".
    """
    path, note = vision.capture_to_file(camera)
    return f"saved {path} — {note}"
