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
from mcp_server import hardware
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
