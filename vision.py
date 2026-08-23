"""vision.py — the two cameras, and the stitched dual view.

The up cam sees the horizon; the down cam sees the floor directly ahead.
look() stacks them into ONE image (up on top, down underneath) so a single
frame tells the brain both what is in the room and what is underfoot.

Camera identity is resolved on EVERY call, never cached: the second camera
drops off this Pi's USB bus and comes back with a different USB descriptor,
which renames its /dev/v4l/by-id path. A path captured once goes stale.
See config.CAM_SECOND.

A missing camera is normal, not an error — the stitch falls back to whichever
camera answered and says so.

Not built yet — Step 0.5 fills this in.
"""

from __future__ import annotations
