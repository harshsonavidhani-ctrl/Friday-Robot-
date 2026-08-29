"""Finding Friday's two cameras reliably, and giving each the job it can actually do.

WHY NUMERIC INDICES WERE ALWAYS GOING TO FAIL
---------------------------------------------
`pi_deploy.py` opened `cam_indices = (0, 1)`. Measured on this Pi:

    /dev/video0   UVC 046d:0825       Video Capture      <- the UP camera
    /dev/video1   UVC 046d:0825       Metadata Capture   <- the SAME camera's metadata
    /dev/video2   Generic HD camera   Video Capture      <- the DOWN camera
    /dev/video3   Generic HD camera   Metadata Capture

So "camera 1" was the up camera's metadata node, which cannot deliver a frame and
does not even open, and the down camera was never touched at all. Confirmed on a live
run: `cam0 updates=87, cam1 updates=0 stale=True`.

Modern UVC drivers expose a metadata node per camera, so EVERY camera on this machine
occupies two consecutive video numbers. Index arithmetic cannot survive that, and it
cannot survive a device enumerating in a different order after a reboot either.

Friday's own `config.py` already solved this properly -- stable `/dev/v4l/by-id/`
paths keyed on USB serial -- and the RL stack duplicated the problem instead of
reusing the answer. This module resolves cameras the same way, and then VERIFIES the
node reports Video Capture before handing it back, because a by-id path can point at
a metadata node just as easily as an index can.

ROLES, AND THE DEFECT THAT SHARING THEM CAUSED
----------------------------------------------
`VisualHazardMonitor` scores a bottom-centre band of the frame -- "the floor the robot
is about to drive onto" -- and took `max()` across both cameras, worst wins. That is
correct for the DOWN camera, which is mounted above the bumper and sees floor. It is
meaningless for the UP camera, which is above the screen and sees the room: the bottom
of its frame is whatever furniture happens to be in the middle distance, and scoring
it as floor hazard lets a sofa brake the robot.

So the roles are separated here:

    DOWN  ->  floor hazard. May slow the robot (advisory, capped, fail-open).
    UP    ->  environmental context. MEASURED and REPORTED, never acts on motion.

NO FAKE PERCEPTION
------------------
The up camera reports three measured image statistics -- mean brightness, frame-to-
frame change energy, and how much of the view is very close/occluded. That is all.
There is no object detection here and nothing that names what it is looking at,
because none is implemented. A field called `objects` that contained guesses would be
worse than no field at all.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BY_ID = Path("/dev/v4l/by-id")

# From friday/config.py, which identifies these by USB serial rather than by position.
UP_SERIAL = "9B5B4130"            # Logitech C270, 046d:0825 -- above the screen
DOWN_SERIAL = "20181212000000"    # Generic HD camera, 349c:2317 -- above the bumper


@dataclass
class CameraDevice:
    role: str                     # "up" or "down"
    path: str                     # /dev/videoN, resolved from a stable by-id link
    by_id: str                    # the stable name it was found under
    serial: str


def _is_capture_node(dev: str) -> bool:
    """True only if this node reports Video Capture, not Metadata Capture.

    Checked rather than assumed. The whole bug this module exists to fix was a node
    that looked like a camera by every naming convention and could not produce a
    frame. `v4l2-ctl` missing is treated as "cannot verify" -> False, so a machine
    without it falls back to the explicit --cam-*-device arguments rather than
    silently opening a metadata node again.
    """
    try:
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--info"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    caps = out.split("Device Caps", 1)
    return len(caps) > 1 and "Video Capture" in caps[1]


def resolve(role_serials: dict[str, str] | None = None) -> dict[str, CameraDevice]:
    """Map role -> device, by USB serial, verifying each is a real capture node.

    Returns only what it actually found. A missing camera is an absent key, never a
    guessed index: opening "probably the other one" is how the down camera went
    unused for the whole of this stack's life.
    """
    wanted = role_serials or {"up": UP_SERIAL, "down": DOWN_SERIAL}
    found: dict[str, CameraDevice] = {}
    if not BY_ID.is_dir():
        return found

    for link in sorted(BY_ID.iterdir()):
        name = link.name
        for role, serial in wanted.items():
            if role in found or serial not in name:
                continue
            try:
                dev = str(link.resolve())
            except OSError:
                continue
            if not _is_capture_node(dev):
                continue          # almost certainly the metadata node; keep looking
            found[role] = CameraDevice(role=role, path=dev, by_id=name, serial=serial)
    return found


def describe(found: dict[str, CameraDevice]) -> str:
    if not found:
        return "no cameras resolved"
    return "  ".join(f"{r}={d.path}" for r, d in sorted(found.items()))


class ContextMonitor:
    """Measured image statistics from the UP camera. Reports; never steers.

    Three numbers, all directly measured, none interpreted:

      brightness  mean grey level of the frame, 0-255. Says whether the room is lit.
      change      mean absolute frame-to-frame difference. Non-zero means the scene
                  is changing -- which is the robot moving, something else moving, or
                  both. It cannot distinguish those and does not claim to.
      occluded    fraction of the frame that is both very dark and very flat, the
                  signature of something pressed up against the lens.

    This is NOT object detection and nothing here names what is in view. The reason
    the field list is this short is that this is genuinely everything the current code
    can establish.
    """

    def __init__(self) -> None:
        self._prev = None
        self.brightness = None
        self.change = None
        self.occluded = None
        self.frames = 0

    def update(self, frame) -> dict[str, float | None]:
        import numpy as np
        if frame is None:
            return self.snapshot()
        f = frame.astype("float32")
        self.brightness = float(f.mean())
        if self._prev is not None and self._prev.shape == f.shape:
            self.change = float(np.abs(f - self._prev).mean())
        self._prev = f
        dark = f < 40.0
        self.occluded = float(dark.mean())
        self.frames += 1
        return self.snapshot()

    def snapshot(self) -> dict[str, float | None]:
        return {"brightness": self.brightness, "change": self.change,
                "occluded": self.occluded, "frames": self.frames}
